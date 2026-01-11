"""
AC4C Unbalanced Dataset Training Script

This script implements:
1. AC4C unbalanced dataset training (full train set for training, full test set for testing)
2. Weight loading support (specify checkpoint path or train from scratch)
3. Few-shot benchmark evaluation
4. Comprehensive metrics and logging
5. Tensorboard visualization
"""

import os
import random
import json
import numpy as np
import torch
import argparse
from datetime import datetime

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.main_model import RNA_ClassQuery_Model
from dataset.ac4c import AC4CDataset
from utils import (
    setup_logging, setup_tensorboard, log_metrics_to_tensorboard,
    get_smoothed_pos_weights,
    save_checkpoint, load_config,
    print_evaluation_results, print_few_shot_results,
    train_epoch, test_epoch,
    get_all_predictions, run_few_shot_benchmark_ac4c,
    MOD_NAMES
)


def main(config_path='json/ac4c_unbalan.json', checkpoint_path=None):
    """
    Main training function

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to checkpoint file to load (optional, trains from scratch if None)
    """
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Setup logging
    logger = setup_logging(Config.log_dir, Config.experiment_name + '_ac4c_unbalanced')

    # Setup tensorboard
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_ac4c_unbalanced')

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("AC4C Unbalanced Dataset Multi-label Classification Training")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
    else:
        logger.info("Training from scratch (no checkpoint specified)")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Data directory for unbalanced AC4C
    ac4c_data_dir = 'npy/ac4c_processed/unbalanced_ac4c'

    # Load train dataset
    logger.info(f"\nLoading AC4C unbalanced TRAIN dataset from {ac4c_data_dir}...")
    train_dataset = AC4CDataset(
        mode='train',
        data_dir=ac4c_data_dir,
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )

    # Precompute structures if needed
    train_cache_stats = train_dataset.get_cache_stats()
    if not train_cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("Train dataset batch cache not found, precomputing structures...")
        logger.info("="*60)
        train_dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\nTrain dataset batch cache exists: {train_cache_stats['batch_cache']['path']}")
        logger.info(f"  File size: {train_cache_stats['batch_cache']['size_mb']:.2f} MB")

    logger.info(f"Train dataset loaded: {len(train_dataset)} samples")

    # Load test dataset
    logger.info(f"\nLoading AC4C unbalanced TEST dataset from {ac4c_data_dir}...")
    test_dataset = AC4CDataset(
        mode='test',
        data_dir=ac4c_data_dir,
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )

    # Precompute structures if needed
    test_cache_stats = test_dataset.get_cache_stats()
    if not test_cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("Test dataset batch cache not found, precomputing structures...")
        logger.info("="*60)
        test_dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\nTest dataset batch cache exists: {test_cache_stats['batch_cache']['path']}")
        logger.info(f"  File size: {test_cache_stats['batch_cache']['size_mb']:.2f} MB")

    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

    # Get train indices for pos_weight calculation
    train_indices = list(range(len(train_dataset)))

    # Calculate smoothed class weights from train dataset
    pos_weight = get_smoothed_pos_weights(
        train_dataset, train_indices, num_classes=Config.num_classes, logger=logger
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train batch size: {Config.batch_size}, Train batches: {len(train_loader)}")
    logger.info(f"  Test batch size: {Config.batch_size}, Test batches: {len(test_loader)}")

    # Create model
    logger.info(f"\nCreating model...")
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=Config.num_classes,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=getattr(Config, 'use_hierarchical', False),
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Load checkpoint if specified
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    elif checkpoint_path:
        logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")
        logger.warning("Training from scratch instead...")

    # Loss function with smoothed pos_weight
    pos_weight = pos_weight.to(Config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.num_epochs,
        eta_min=1e-6
    )

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Optimizer: AdamW (lr={Config.learning_rate}, wd={Config.weight_decay})")
    logger.info(f"  Scheduler: CosineAnnealing (T_max={Config.num_epochs})")
    logger.info(f"  Loss: BCEWithLogitsLoss with smoothed pos_weight")
    logger.info(f"  Epochs: {Config.num_epochs}")
    logger.info(f"  Batch size: {Config.batch_size}")

    # Log hyperparameters to tensorboard
    tb_writer.add_hparams(
        {
            'learning_rate': Config.learning_rate,
            'batch_size': Config.batch_size,
            'weight_decay': Config.weight_decay,
            'cnn_hidden_dim': Config.cnn_hidden_dim,
            'gcn_hidden_dim': Config.gcn_hidden_dim,
            'num_epochs': Config.num_epochs,
            'dataset': 'ac4c_unbalanced'
        },
        {}
    )

    # Training loop
    best_macro_f1 = 0.0
    best_epoch = 0

    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}")

    for epoch in range(start_epoch, Config.num_epochs + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Epoch {epoch}/{Config.num_epochs}")
        logger.info(f"{'#'*60}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=Config.use_hierarchical,
            use_amp=Config.use_amp and torch.cuda.is_available()
        )

        # Log train loss to tensorboard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Evaluate based on test_interval
        should_eval = (epoch % Config.test_interval == 0)
        current_macro_f1 = 0.0
        metrics_unbalance = None

        if should_eval:
            # Evaluate
            logger.info(f"\nEvaluating...")
            test_loss = test_epoch(
                model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=Config.use_hierarchical,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log test loss to tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Get predictions
            logger.info(f"\n  Getting predictions for test data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)

            # Import AC4C evaluation function
            from utils.metrics import evaluate_ac4c
            metrics_unbalance = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                             dataset_mode='unbalanced', y_4prob=y_4prob)
            metrics_balanceb = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                           dataset_mode='balanced', y_4prob=y_4prob)

            # Print results
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch} - AC4C Unbalanced Dataset Test Results")
            logger.info(f"{'='*60}")
            logger.info(f"Unbalance Mode - Macro F1: {metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_unbalance.get('group_ac4c_micro_f1', 0.0):.4f}")
            logger.info(f"BalanceB Mode - Macro F1: {metrics_balanceb.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_balanceb.get('group_ac4c_micro_f1', 0.0):.4f}")

            # Log metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_ac4c_unbalanced_unbalance', epoch, key_prefix='group_ac4c_')
            log_metrics_to_tensorboard(tb_writer, metrics_balanceb, 'test_ac4c_unbalanced_balanceb', epoch, key_prefix='group_ac4c_')

            # Run Few-Shot Benchmark
            logger.info(f"\n>>> Running AC4C Unbalanced Few-Shot Benchmark <<<")
            # Use empty datasets for balanced (not used for unbalanced training)
            from dataset.ac4c import AC4CDataset
            empty_ac4c_train = AC4CDataset(mode='train', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir)
            empty_ac4c_test = AC4CDataset(mode='test', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir)

            run_few_shot_benchmark_ac4c(
                model=model,
                ac4c_balanced_train=empty_ac4c_train,  # Not used
                ac4c_balanced_test=empty_ac4c_test,    # Not used
                ac4c_unbalanced_train=train_dataset,
                ac4c_unbalanced_test=test_dataset,
                device=Config.device,
                shots=['full'],
                epoch=epoch,
                logger=logger,
                tb_writer=tb_writer,
                config=Config
            )

            # Check for best model
            current_macro_f1 = metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0)
            logger.info(f"Current Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")

        # Save checkpoint every epoch
        if Config.save_every_epoch:
            epoch_checkpoint_path = os.path.join(Config.checkpoint_dir, f'ac4c_unbalanced_epoch_{epoch:03d}.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance if should_eval else None,
                          epoch_checkpoint_path, logger, config_dict)

        # Update and save best model
        if should_eval and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            # Save best model checkpoint
            checkpoint_path = os.path.join(Config.checkpoint_dir, 'ac4c_unbalanced_best_model.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, checkpoint_path, logger, config_dict)

            # Log to tensorboard
            tb_writer.add_scalar('test/best_macro_f1', best_macro_f1, epoch)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model saved to: {os.path.join(Config.checkpoint_dir, 'ac4c_unbalanced_best_model.pt')}")

    # Close tensorboard writer
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on AC4C unbalanced dataset')
    parser.add_argument('--config', type=str, default='json/ac4c_unbalan.json',
                       help='Path to configuration file (default: json/ac4c_unbalan.json)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file to load (optional, trains from scratch if not specified)')

    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
