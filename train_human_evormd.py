"""
RNA Multi-label Classification Training Script (EvoRMD Architecture)

Ported from train_human_multirm.py, replacing model_v3 (BiLSTM + BahdanauAttention)
with EvoRMD-style architecture (Conv1dEmbedder + TrainableAttention + MulticlassClassifier).

This script implements:
1. Multi-label disjoint data split (7:3 train/test)
2. Smoothed class weighting for imbalanced learning
3. Dual evaluation modes (Unbalance & BalanceB)
4. Comprehensive metrics (F1, Acc, Precision, Recall, AUC, AUPRC, MCC, Sensitivity, Specificity)
5. Logging to file and console
6. Tensorboard visualization
7. TQDM progress monitoring
"""

import os
import random
import json
import numpy as np
import torch
import copy
from datetime import datetime

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.evormd_human import EvoRMDForHuman
from dataset.human import Mer100Dataset
from utils import (
    setup_logging, setup_tensorboard, log_metrics_to_tensorboard,
    multi_label_disjoint_split, get_smoothed_pos_weights,
    evaluate_unbalance, evaluate_balanceb, evaluate_with_optimal_threshold,
    evaluate_4class_with_optimal_threshold,
    evaluate_group_balanceb,
    save_checkpoint, MOD_NAMES,
    DynamicBalancedBatchSampler,
    load_config, print_evaluation_results, train_epoch, test_epoch,
    GROUP_TO_CLASS_INDICES,
    get_all_predictions, get_all_predictions_and_attention,
    calculate_topk_recall, print_topk_table,
    calculate_comprehensive_localization_metrics, print_comprehensive_table
)


# ============================================================================
# Main Training Loop
# ============================================================================

def main(config_path='json/human_evormd.json'):
    """
    Main training function for RNA multi-label classification with EvoRMD architecture.

    Uses Conv1dEmbedder + TrainableAttention + MulticlassClassifier instead of
    BiLSTM + BahdanauAttention (model_v3).

    Args:
        config_path (str): Path to the JSON configuration file.
    """

    # Load configuration from JSON file
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Set EvoRMD-specific config fields (not parsed by generic load_config)
    model_cfg = config_dict.get('model', {})
    Config.d_fm = model_cfg.get('d_fm', 640)
    Config.mlp_depth = model_cfg.get('mlp_depth', 2)
    Config.conv_kernel_size = model_cfg.get('conv_kernel_size', 7)
    Config.conv_dropout = model_cfg.get('conv_dropout', 0.1)

    # Setup logging to both file and console
    logger = setup_logging(Config.log_dir, Config.experiment_name)

    # Setup tensorboard for visualization
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name)

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("RNA Multi-label Classification Training (EvoRMD)")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")

    # Set random seeds for reproducibility across all libraries
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load RNA sequence dataset with human m6A modification data
    logger.info(f"\nLoading dataset from {Config.data.human_data_dir}...")
    dataset = Mer100Dataset(
        mode='train',
        data_dir=Config.data.human_data_dir,
        cache_dir=Config.data.cache_dir,
        use_human3=True,
        use_cache=True
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Precompute secondary structures if batch cache doesn't exist
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("批量缓存不存在，开始预计算所有二级结构...")
        logger.info("="*60)
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\n批量缓存已存在: {cache_stats['batch_cache']['path']}")
        logger.info(f"  文件大小: {cache_stats['batch_cache']['size_mb']:.2f} MB")
        if cache_stats['batch_cache'].get('loaded_in_memory', False):
            logger.info(f"  状态: 已加载到内存")
        else:
            logger.info(f"  状态: 未加载到内存")

    # Split dataset into train and test sets with disjoint labels
    train_indices, test_indices = multi_label_disjoint_split(
        dataset,
        train_ratio=Config.train_ratio,
        random_seed=Config.random_seed,
        logger=logger
    )

    # Calculate smoothed positive weights for imbalanced classes
    pos_weight_unbalanced = get_smoothed_pos_weights(
        dataset, train_indices, num_classes=Config.num_classes, logger=logger
    )

    # Create subsets for train and test datasets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Use dynamic sampler for training to handle extreme class imbalance
    if Config.use_dynamic_sampler:
        train_batch_sampler = DynamicBalancedBatchSampler(
            dataset=dataset,
            train_indices=train_indices,
            batch_size=Config.batch_size,
            num_classes=Config.num_classes,
            balance_ratio=Config.balance_ratio,
            total_epochs=Config.num_epochs,
            random_seed=Config.random_seed
        )
    else:
        from utils import MultilabelBalancedBatchSampler
        train_batch_sampler = MultilabelBalancedBatchSampler(
            dataset=dataset,
            train_indices=train_indices,
            batch_size=Config.batch_size,
            num_classes=Config.num_classes,
            random_seed=Config.random_seed
        )

    # Create data loaders for training and testing
    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_batch_sampler,
        num_workers=16,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train batch size: {Config.batch_size}, Train batches (balanced): {len(train_loader)}")
    logger.info(f"  Test batch size: {Config.batch_size}, Test batches: {len(test_loader)}")

    # Initialize EvoRMD-style model: Conv1dEmbedder + TrainableAttention + MulticlassClassifier
    logger.info(f"\nCreating EvoRMD model...")
    model = EvoRMDForHuman(
        num_task=Config.num_classes,
        d_fm=Config.d_fm,
        mlp_depth=Config.mlp_depth,
        conv_kernel_size=Config.conv_kernel_size,
        conv_dropout=Config.conv_dropout,
        use_hierarchical=Config.use_hierarchical,
    ).to(Config.device)

    # Count and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  Architecture: Conv1dEmbedder({4}->{Config.d_fm}) + TrainableAttention + MLP(depth={Config.mlp_depth})")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Initialize loss function with class-balanced weights
    pos_weight = pos_weight_unbalanced.to(Config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize optimizer and learning rate scheduler
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

    # Log hyperparameters to Tensorboard
    tb_writer.add_hparams(
        {
            'learning_rate': Config.learning_rate,
            'batch_size': Config.batch_size,
            'weight_decay': Config.weight_decay,
            'd_fm': Config.d_fm,
            'mlp_depth': Config.mlp_depth,
            'num_epochs': Config.num_epochs
        },
        {}
    )

    # Initialize tracking variables for best model
    best_macro_f1 = 0.0
    best_epoch = 0

    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}")

    # Main training loop over all epochs
    for epoch in range(1, Config.num_epochs + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Epoch {epoch}/{Config.num_epochs}")
        logger.info(f"{'#'*60}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Update dynamic sampler mode based on current epoch
        if isinstance(train_batch_sampler, DynamicBalancedBatchSampler):
            train_batch_sampler.set_epoch(epoch)
            logger.info(f"Sampler mode: {train_batch_sampler.get_mode_info()}")

            # Update pos_weight based on sampler mode
            new_pos_weight = pos_weight_unbalanced.to(Config.device)
            if train_batch_sampler.use_balanced_mode:
                logger.info(f"Class weights: BALANCED mode (smoothed class weights)")
            else:
                logger.info(f"Class weights: FULL_COVERAGE mode (smoothed class weights)")

            criterion = nn.BCEWithLogitsLoss(pos_weight=new_pos_weight, reduction='mean')

        # Execute one training epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=Config.use_hierarchical,
            use_amp=Config.use_amp and torch.cuda.is_available(),
            use_attention_supervision=getattr(Config, 'use_attention_supervision', False),
            attention_lambda=getattr(Config, 'attention_lambda', 1.0)
        )

        # Log training metrics to Tensorboard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Check if evaluation should be performed this epoch
        should_eval = (epoch % Config.test_interval == 0)
        metrics_unbalance = None
        current_macro_f1 = 0.0

        if should_eval:
            # Run evaluation on test set
            logger.info(f"\nEvaluating...")
            test_loss = test_epoch(
                model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=Config.use_hierarchical,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Evaluation Phase - Get predictions and compute metrics
            logger.info(f"\n{'='*60} Epoch {epoch} Testing {'='*60}")

            logger.info(f"\n  Getting predictions for human data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)

            metrics_unbalance = evaluate_unbalance(y_true, y_prob, Config.device, y_4class, Config.random_seed, y_4prob)
            metrics_balanceb = evaluate_balanceb(y_true, y_prob, y_4class, Config.device, Config.random_seed, y_4prob)
            metrics_group_balanceb = evaluate_group_balanceb(y_true, y_prob, y_4class, Config.random_seed)
            metrics_opt = evaluate_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)
            metrics_4class = evaluate_4class_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)

            # Compute attention-based localization metrics if attention supervision is enabled
            topk_results = {}
            comprehensive_results = {}
            if Config.use_attention_supervision:
                logger.info(f"\n  Computing Top-K site recall...")
                y_true, y_prob, y_4class, y_4prob, attn_weights, y_site = get_all_predictions_and_attention(
                    model, test_loader, Config.device, Config.use_hierarchical
                )
                if attn_weights is not None and y_site is not None:
                    topk_results = calculate_topk_recall(attn_weights, y_site, k_list=[1, 3, 5, 7, 10, 20, 50])
                    logger.info(f"  Computing comprehensive localization metrics...")
                    comprehensive_results = calculate_comprehensive_localization_metrics(
                        attn_weights, y_site, k_list=[1, 3, 5, 7, 10]
                    )
                else:
                    logger.info(f"  Attention weights or site labels not available, skipping Top-K evaluation.")

            # Print formatted evaluation results
            print_evaluation_results(
                metrics_unbalance=metrics_unbalance,
                metrics_balanceb=metrics_balanceb,
                epoch=epoch,
                logger=logger,
                metrics_opt=metrics_opt,
                metrics_group_balanceb=metrics_group_balanceb,
                plant_metrics_unbalance=None,
                plant_metrics_balanceb=None,
                metrics_4class=metrics_4class,
                plant_metrics_4class=None,
                plant_metrics_opt=None
            )

            if topk_results:
                print_topk_table(topk_results, k_list=[1, 3, 5, 7, 10, 20, 50], logger=logger)

            if comprehensive_results:
                print_comprehensive_table(comprehensive_results, k_list=[1, 3, 5, 7, 10], logger=logger)

            # Log test metrics to Tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_unbalance', epoch)
            log_metrics_to_tensorboard(tb_writer, metrics_balanceb, 'test_balanceb', epoch)

            # Track best model based on macro F1 score
            current_macro_f1 = metrics_unbalance['group_macro_f1']
            logger.info(f"Current Unbalance Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")

        # Save checkpoint for each epoch if enabled
        if Config.save_every_epoch and metrics_unbalance is not None:
            epoch_checkpoint_path = os.path.join(Config.checkpoint_dir, f'epoch_{epoch:03d}.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, epoch_checkpoint_path, logger, config_dict)

        # Update and save best model checkpoint if performance improved
        if metrics_unbalance is not None and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, checkpoint_path, logger, config_dict)

            tb_writer.add_scalar('test/best_macro_f1', best_macro_f1, epoch)

    # Print training summary and close resources
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model saved to: {os.path.join(Config.checkpoint_dir, 'best_model.pt')}")

    tb_writer.close()


if __name__ == "__main__":
    main(config_path='json/human_evormd.json')
