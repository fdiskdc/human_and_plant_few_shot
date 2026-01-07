"""
RNA Multi-label Classification Training Script

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.main_model import RNA_ClassQuery_Model
from dataset.human import Mer100Dataset
from dataset.plant import PlantDataset
from utils import (
    setup_logging, setup_tensorboard, log_metrics_to_tensorboard,
    multi_label_disjoint_split, get_smoothed_pos_weights,
    evaluate_unbalance, evaluate_balanceb, evaluate_with_optimal_threshold,
    evaluate_4class_with_optimal_threshold,
    evaluate_plant_unbalance, evaluate_plant_balanceb,
    evaluate_group_balanceb,
    save_checkpoint, MOD_NAMES, MultilabelBalancedBatchSampler,
    DynamicBalancedBatchSampler,
    load_config, print_evaluation_results, print_few_shot_results, train_epoch, test_epoch,
    GROUP_TO_INDEX, INDEX_TO_NUCLEOTIDE, GROUP_TO_CLASS_INDICES,
    get_center_nucleotide, get_all_predictions
)


# ============================================================================
# Few-Shot Benchmark Function
# ============================================================================

def run_few_shot_benchmark(model, plant_dataset, device, shots=[0, 1, 3, 5, 7, 10], 
                          epoch=0, logger=None, tb_writer=None, config=None):
    """
    Runs Plant Few-Shot Benchmark.
    Ensures model is restored to pre-adaptation state after execution.
    
    Args:
        model: The neural network model
        plant_dataset: PlantDataset instance
        device: Device to run on (cuda/cpu)
        shots: List of shot counts to evaluate
        epoch: Current epoch number
        logger: Logger instance
        tb_writer: Tensorboard writer
        config: Configuration object
        
    Returns:
        dict: Results for all shots
    """
    logger.info(f"\n>>> Starting Plant Few-Shot Benchmark (Shots: {shots}) <<<")
    
    # 1. Save Human model state (Deep Copy is crucial)
    original_state_dict = copy.deepcopy(model.state_dict())
    all_results = {}
    
    for k in shots:
        logger.info(f"\n--- Running {k}-Shot Adaptation ---")
        
        # 2. Reset model to Human state before EVERY shot experiment
        model.load_state_dict(original_state_dict)
        
        # 3. Get Data Split (Fixed 90% Test, Sample k from 10% Pool)
        support_indices, test_indices = plant_dataset.get_few_shot_split(
            k_shots=k, valid_classes=[5, 8, 9], test_ratio=0.9, seed=config.random_seed
        )
        
        # Create Test Loader (Fixed)
        test_subset = Subset(plant_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        # 4. Fine-tuning (Only if k > 0)
        if k > 0:
            support_subset = Subset(plant_dataset, support_indices)
            ft_batch_size = min(32, len(support_indices)) if len(support_indices) > 0 else 1
            support_loader = DataLoader(support_subset, batch_size=ft_batch_size, shuffle=True)
            
            # New Optimizer for Fine-tuning (Do not affect global optimizer)
            ft_optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            ft_criterion = nn.BCEWithLogitsLoss()
            
            model.train()
            ft_epochs = 10  # Short adaptation
            
            for ft_ep in range(ft_epochs):
                for batch in support_loader:
                    batch = batch.to(device)
                    if not isinstance(batch.y, torch.Tensor):
                        batch.y = torch.tensor(batch.y, dtype=torch.float32)
                    batch.y = batch.y.to(device)
                    
                    ft_optimizer.zero_grad()
                    if config.use_hierarchical:
                        logits_12, _ = model(batch.x, batch.edge_index, batch.batch)
                        loss = ft_criterion(logits_12, batch.y)
                    else:
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = ft_criterion(logits, batch.y)
                    
                    loss.backward()
                    ft_optimizer.step()

        # 5. Evaluation on Fixed Test Set
        y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, device, config.use_hierarchical)
        
        metrics_unbalance = evaluate_plant_unbalance(y_true, y_prob, device, y_4class, config.random_seed, y_4prob)
        metrics_balanceb = evaluate_plant_balanceb(y_true, y_prob, y_4class, device, config.random_seed, y_4prob)
        
        all_results[k] = {"unbalance": metrics_unbalance, "balanceb": metrics_balanceb}

    # 6. Print Summary
    print_few_shot_results(all_results, epoch, logger)
    
    # 7. Restore Model State (Crucial)
    model.load_state_dict(original_state_dict)
    logger.info(">>> Few-Shot Benchmark Finished. Model parameters restored. <<<")
    
    return all_results


# ============================================================================
# Main Training Loop
# ============================================================================

def main(config_path='model.json'):
    """Main training function"""
    # Load configuration from JSON
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Setup logging
    logger = setup_logging(Config.log_dir, Config.experiment_name)

    # Setup tensorboard
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name)

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("RNA Multi-label Classification Training")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load dataset
    logger.info(f"\nLoading dataset from {Config.data_dir}...")
    dataset = Mer100Dataset(mode='train', data_dir=Config.data_dir, use_human3=True, use_cache=True)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # 预计算所有二级结构（如果批量缓存不存在）
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("批量缓存不存在，开始预计算所有二级结构...")
        logger.info("="*60)
        # 使用多进程预计算，num_workers=None 表示自动使用所有CPU核心
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\n批量缓存已存在: {cache_stats['batch_cache']['path']}")
        logger.info(f"  文件大小: {cache_stats['batch_cache']['size_mb']:.2f} MB")
        if cache_stats['batch_cache'].get('loaded_in_memory', False):
            logger.info(f"  状态: 已加载到内存")
        else:
            logger.info(f"  状态: 未加载到内存")

    # Multi-label disjoint split
    train_indices, test_indices = multi_label_disjoint_split(
        dataset,
        train_ratio=Config.train_ratio,
        random_seed=Config.random_seed,
        logger=logger
    )

    # Calculate smoothed class weights (used for both BALANCED and FULL_COVERAGE modes)
    pos_weight_unbalanced = get_smoothed_pos_weights(
        dataset, train_indices, num_classes=Config.num_classes, logger=logger
    )

    # Create dataloaders
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Use dynamic sampler for training to handle extreme class imbalance
    # Early epochs: balanced mode (rare class determines batches)
    # Late epochs: full coverage mode (all samples get sampled)
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
        train_batch_sampler = MultilabelBalancedBatchSampler(
            dataset=dataset,
            train_indices=train_indices,
            batch_size=Config.batch_size,
            num_classes=Config.num_classes,
            random_seed=Config.random_seed
        )

    # Important: When using batch_sampler, set shuffle=False and don't specify batch_size
    # The batch_sampler handles both shuffling and batching
    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_batch_sampler,  # Use our custom batch sampler
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train batch size: {Config.batch_size}, Train batches (balanced): {len(train_loader)}")
    logger.info(f"  Test batch size: {Config.batch_size}, Test batches: {len(test_loader)}")

    # Load plant dataset for evaluation
    logger.info(f"\nLoading plant dataset for evaluation...")
    plant_dataset = PlantDataset(plant_dir="npy/plant", use_cache=True, preload_cache=True)

    # 预计算plant所有二级结构（如果批量缓存不存在）
    plant_cache_stats = plant_dataset.get_cache_stats()
    if not plant_cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("Plant批量缓存不存在，开始预计算所有二级结构...")
        logger.info("="*60)
        # 使用多进程预计算，num_workers=None 表示自动使用所有CPU核心
        plant_dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\nPlant批量缓存已存在: {plant_cache_stats['batch_cache']['path']}")
        logger.info(f"  文件大小: {plant_cache_stats['batch_cache']['size_mb']:.2f} MB")
        if plant_cache_stats['batch_cache'].get('loaded_in_memory', False):
            logger.info(f"  状态: 已加载到内存")
        else:
            logger.info(f"  状态: 未加载到内存")

    plant_test_loader = DataLoader(
        plant_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    logger.info(f"Plant dataset loaded: {len(plant_dataset)} samples")
    logger.info(f"  Plant test batches: {len(plant_test_loader)}")

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

    # Loss function with dynamic positive weights
    # Initially set to unbalanced weights, will be updated per epoch based on sampler mode
    pos_weight = pos_weight_unbalanced.to(Config.device)
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
            'num_epochs': Config.num_epochs
        },
        {}
    )

    # Training loop
    best_macro_f1 = 0.0
    best_epoch = 0

    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}")

    for epoch in range(1, Config.num_epochs + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Epoch {epoch}/{Config.num_epochs}")
        logger.info(f"{'#'*60}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Update sampler mode for dynamic sampler
        if isinstance(train_batch_sampler, DynamicBalancedBatchSampler):
            train_batch_sampler.set_epoch(epoch)
            logger.info(f"Sampler mode: {train_batch_sampler.get_mode_info()}")

            # Update pos_weight based on sampler mode
            # Both BALANCED and FULL_COVERAGE modes use smoothed class weights
            new_pos_weight = pos_weight_unbalanced.to(Config.device)
            if train_batch_sampler.use_balanced_mode:
                logger.info(f"Class weights: BALANCED mode (smoothed class weights)")
            else:
                logger.info(f"Class weights: FULL_COVERAGE mode (smoothed class weights)")

            # Update the criterion with new pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=new_pos_weight, reduction='mean')

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=Config.use_hierarchical
        )

        # Log train loss to tensorboard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Evaluate based on test_interval
        should_eval = (epoch % Config.test_interval == 0)
        metrics_unbalance = None
        current_macro_f1 = 0.0

        if should_eval:
            # Evaluate
            logger.info(f"\nEvaluating...")
            test_loss = test_epoch(
                model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=Config.use_hierarchical
            )

            # Log test loss to tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Get evaluation metrics for human data
            # OPTIMIZATION: Run inference once and get all predictions
            logger.info(f"\n  Getting predictions for human data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)

            # Pass predictions to evaluation functions (no additional inference needed)
            metrics_unbalance = evaluate_unbalance(y_true, y_prob, Config.device, y_4class, Config.random_seed, y_4prob)
            metrics_balanceb = evaluate_balanceb(y_true, y_prob, y_4class, Config.device, Config.random_seed, y_4prob)
            metrics_group_balanceb = evaluate_group_balanceb(y_true, y_prob, y_4class, Config.random_seed)
            metrics_opt = evaluate_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)
            metrics_4class = evaluate_4class_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)

            # Evaluate on plant data
            logger.info(f"\n  Getting predictions for plant data...")
            y_true_plant, y_prob_plant, y_4class_plant, y_4prob_plant = get_all_predictions(model, plant_test_loader, Config.device, Config.use_hierarchical)

            plant_metrics_unbalance = evaluate_plant_unbalance(y_true_plant, y_prob_plant, Config.device, y_4class_plant, Config.random_seed, y_4prob_plant)
            plant_metrics_balanceb = evaluate_plant_balanceb(y_true_plant, y_prob_plant, y_4class_plant, Config.device, Config.random_seed, y_4prob_plant)
            # For plant data, use 4-class metrics already computed in plant_metrics_unbalance/balanceb
            # These contain the 'group_plant_4class_*' keys needed for the tables
            plant_metrics_4class = plant_metrics_unbalance if plant_metrics_unbalance else plant_metrics_balanceb
            plant_metrics_opt = evaluate_with_optimal_threshold(model, plant_test_loader, Config.device, Config.use_hierarchical)

            # Print and log results (including plant metrics and 4-class metrics)
            print_evaluation_results(
                metrics_unbalance, metrics_balanceb, epoch, logger, metrics_opt=metrics_opt,
                plant_metrics_unbalance=plant_metrics_unbalance,
                plant_metrics_balanceb=plant_metrics_balanceb,
                metrics_4class=metrics_4class,
                plant_metrics_4class=plant_metrics_4class,
                plant_metrics_opt=plant_metrics_opt,
                metrics_group_balanceb=metrics_group_balanceb
            )

            # Log metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_unbalance', epoch)
            log_metrics_to_tensorboard(tb_writer, metrics_balanceb, 'test_balanceb', epoch)

            # Log plant metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, plant_metrics_unbalance, 'plant_unbalance', epoch, key_prefix='group_plant_')
            log_metrics_to_tensorboard(tb_writer, plant_metrics_balanceb, 'plant_balanceb', epoch, key_prefix='group_plant_')

            # Run Plant Few-Shot Benchmark
            run_few_shot_benchmark(
                model=model,
                plant_dataset=plant_dataset,
                device=Config.device,
                shots=[0, 1, 3, 5, 7, 10], 
                epoch=epoch,
                logger=logger,
                tb_writer=tb_writer,
                config=Config
            )

            # Check for best model (based on Unbalance Macro-F1)
            current_macro_f1 = metrics_unbalance['group_macro_f1']
            logger.info(f"Current Unbalance Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")

        # Save checkpoint every epoch (only if we have metrics)
        if Config.save_every_epoch and metrics_unbalance is not None:
            epoch_checkpoint_path = os.path.join(Config.checkpoint_dir, f'epoch_{epoch:03d}.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, epoch_checkpoint_path, logger, config_dict)

        # Update and save best model (only if we evaluated)
        if metrics_unbalance is not None and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            # Save best model checkpoint
            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, checkpoint_path, logger, config_dict)

            # Log to tensorboard
            tb_writer.add_scalar('test/best_macro_f1', best_macro_f1, epoch)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model saved to: {os.path.join(Config.checkpoint_dir, 'best_model.pt')}")

    # Close tensorboard writer
    tb_writer.close()


if __name__ == "__main__":
    main()
