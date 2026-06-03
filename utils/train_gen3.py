"""
utils/train_gen3.py - 3gen数据集训练流水线 / 3rd-Generation Dataset Training Pipeline

3gen (PacBio/ONT) 数据集主训练流水线:数据划分、平衡采样、Unbalance/BalanceB 评估、TensorBoard、小样本基准测试。
3gen (PacBio/ONT) dataset main training pipeline: data split, balanced sampling, dual evaluation, TensorBoard, few-shot.

功能模块 / Modules:
- main: 主训练函数 / Main training function
- 数据划分 (7:3) / Data split (7:3)
- 平衡采样 / Balanced sampling
- Unbalance / BalanceB 评估 / Dual evaluation
- TensorBoard / TensorBoard
- 小样本基准 / Few-shot benchmark
- 命令行参数 / CLI args

输入 / Inputs:
- json/gen3.json: 训练配置 / Training config
- 3gen/seq.npy, 3gen/12loc.npy: 3gen 数据 / 3gen data
- checkpoints/best_model.pt: 预训练模型 (可选) / Optional pretrained model
- 命令行参数 / CLI: --config, --gpu, --seed

输出 / Outputs:
- checkpoints/best_gen3.pt: 最佳模型 / Best model
- logs/gen3_*/train_*.log: 训练日志 / Training logs
- logs/gen3_*/results.json: 评估结果 / Evaluation results
- TensorBoard events: 可视化 / Visualization

数据流 / Data Flow:
1. 加载配置 / Load config
2. 构建 3gen 数据集 / Build 3gen dataset
3. 划分 / Split
4. 训练循环 / Training loop
5. 评估 + 小样本 / Evaluate + few-shot
6. 保存 / Save

相关文件 / Related Files:
- 调用 / Calls: dataset.gen3.Gen3Dataset, model.main_model, utils.{common,metrics,logging}
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python -m utils.train_gen3 --config json/gen3.json --gpu 0

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
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

from model.main_model import RNA_ClassQuery_Model
from dataset.gen3 import Gen3Dataset
# from dataset.plant import PlantDataset
# from dataset.ac4c import AC4CDataset
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
    get_center_nucleotide, get_all_predictions, get_all_predictions_and_attention,
    run_few_shot_benchmark, run_few_shot_benchmark_ac4c,
    calculate_topk_recall, print_topk_table,
    calculate_comprehensive_localization_metrics, print_comprehensive_table
)


# ============================================================================
# Main Training Loop
# ============================================================================

def main(config_path='json/3gen.json'):
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
    logger.info(f"\nLoading dataset from {Config.data.human_data_dir}...")
    dataset = Gen3Dataset(
        mode='train', 
        data_dir='npy/3gen', 
        # cache_dir=Config.data.cache_dir,
        # use_human3=True, 
        # use_cache=True
    )
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

    # # Load plant dataset for evaluation
    # logger.info(f"\nLoading plant dataset for evaluation...")
    # plant_dataset = PlantDataset(
    #     plant_dir=Config.data.plant_data_dir,
    #     cache_dir=Config.data.cache_dir,
    #     use_cache=True, 
    #     preload_cache=True
    # )

    # # 预计算plant所有二级结构（如果批量缓存不存在）
    # plant_cache_stats = plant_dataset.get_cache_stats()
    # if not plant_cache_stats['batch_cache'].get('exists', False):
    #     logger.info("\n" + "="*60)
    #     logger.info("Plant批量缓存不存在，开始预计算所有二级结构...")
    #     logger.info("="*60)
    #     # 使用多进程预计算，num_workers=None 表示自动使用所有CPU核心
    #     plant_dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    # else:
    #     logger.info(f"\nPlant批量缓存已存在: {plant_cache_stats['batch_cache']['path']}")
    #     logger.info(f"  文件大小: {plant_cache_stats['batch_cache']['size_mb']:.2f} MB")
    #     if plant_cache_stats['batch_cache'].get('loaded_in_memory', False):
    #         logger.info(f"  状态: 已加载到内存")
    #     else:
    #         logger.info(f"  状态: 未加载到内存")

    # plant_test_loader = DataLoader(
    #     plant_dataset,
    #     batch_size=Config.batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=True
    # )
    # logger.info(f"Plant dataset loaded: {len(plant_dataset)} samples")
    # logger.info(f"  Plant test batches: {len(plant_test_loader)}")

    # # ========================================================================
    # # Load AC4C datasets for few-shot evaluation
    # # ========================================================================
    # # AC4C 数据集有独立的 train/test 文件夹，需要分别加载
    # # train 数据集用于采样 Support Set，test 数据集用于评估

    # # --- Balanced AC4C ---
    # logger.info(f"\n{'='*60}")
    # logger.info("Loading AC4C Balanced datasets...")
    # logger.info(f"{'='*60}")

    # # 训练集 (用于 Support Set 采样)
    # logger.info(f"\nLoading AC4C balanced TRAIN dataset (for Support Set)...")
    # ac4c_balanced_train = AC4CDataset(
    #     mode='train',
    #     data_dir='npy/ac4c_processed/balanced_ac4c',
    #     cache_dir=Config.data.cache_dir,
    #     use_cache=True,
    #     preload_cache=True
    # )

    # ac4c_balanced_train_cache_stats = ac4c_balanced_train.get_cache_stats()
    # if not ac4c_balanced_train_cache_stats['batch_cache'].get('exists', False):
    #     logger.info("\n" + "="*60)
    #     logger.info("AC4C balanced TRAIN 批量缓存不存在，开始预计算所有二级结构...")
    #     logger.info("="*60)
    #     ac4c_balanced_train.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    # else:
    #     logger.info(f"\nAC4C balanced TRAIN 批量缓存已存在: {ac4c_balanced_train_cache_stats['batch_cache']['path']}")
    #     logger.info(f"  文件大小: {ac4c_balanced_train_cache_stats['batch_cache']['size_mb']:.2f} MB")
    #     if ac4c_balanced_train_cache_stats['batch_cache'].get('loaded_in_memory', False):
    #         logger.info(f"  状态: 已加载到内存")
    #     else:
    #         logger.info(f"  状态: 未加载到内存")
    # logger.info(f"AC4C balanced TRAIN dataset loaded: {len(ac4c_balanced_train)} samples")

    # # 测试集 (用于评估)
    # logger.info(f"\nLoading AC4C balanced TEST dataset (for Evaluation)...")
    # ac4c_balanced_test = AC4CDataset(
    #     mode='test',
    #     data_dir='npy/ac4c_processed/balanced_ac4c',
    #     cache_dir=Config.data.cache_dir,
    #     use_cache=True,
    #     preload_cache=True
    # )

    # ac4c_balanced_test_cache_stats = ac4c_balanced_test.get_cache_stats()
    # if not ac4c_balanced_test_cache_stats['batch_cache'].get('exists', False):
    #     logger.info("\n" + "="*60)
    #     logger.info("AC4C balanced TEST 批量缓存不存在，开始预计算所有二级结构...")
    #     logger.info("="*60)
    #     ac4c_balanced_test.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    # else:
    #     logger.info(f"\nAC4C balanced TEST 批量缓存已存在: {ac4c_balanced_test_cache_stats['batch_cache']['path']}")
    #     logger.info(f"  文件大小: {ac4c_balanced_test_cache_stats['batch_cache']['size_mb']:.2f} MB")
    #     if ac4c_balanced_test_cache_stats['batch_cache'].get('loaded_in_memory', False):
    #         logger.info(f"  状态: 已加载到内存")
    #     else:
    #         logger.info(f"  状态: 未加载到内存")
    # logger.info(f"AC4C balanced TEST dataset loaded: {len(ac4c_balanced_test)} samples")

    # # --- Unbalanced AC4C ---
    # logger.info(f"\n{'='*60}")
    # logger.info("Loading AC4C Unbalanced datasets...")
    # logger.info(f"{'='*60}")

    # # 训练集 (用于 Support Set 采样)
    # logger.info(f"\nLoading AC4C unbalanced TRAIN dataset (for Support Set)...")
    # ac4c_unbalanced_train = AC4CDataset(
    #     mode='train',
    #     data_dir='npy/ac4c_processed/unbalanced_ac4c',
    #     cache_dir=Config.data.cache_dir,
    #     use_cache=True,
    #     preload_cache=True
    # )

    # ac4c_unbalanced_train_cache_stats = ac4c_unbalanced_train.get_cache_stats()
    # if not ac4c_unbalanced_train_cache_stats['batch_cache'].get('exists', False):
    #     logger.info("\n" + "="*60)
    #     logger.info("AC4C unbalanced TRAIN 批量缓存不存在，开始预计算所有二级结构...")
    #     logger.info("="*60)
    #     ac4c_unbalanced_train.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    # else:
    #     logger.info(f"\nAC4C unbalanced TRAIN 批量缓存已存在: {ac4c_unbalanced_train_cache_stats['batch_cache']['path']}")
    #     logger.info(f"  文件大小: {ac4c_unbalanced_train_cache_stats['batch_cache']['size_mb']:.2f} MB")
    #     if ac4c_unbalanced_train_cache_stats['batch_cache'].get('loaded_in_memory', False):
    #         logger.info(f"  状态: 已加载到内存")
    #     else:
    #         logger.info(f"  状态: 未加载到内存")
    # logger.info(f"AC4C unbalanced TRAIN dataset loaded: {len(ac4c_unbalanced_train)} samples")

    # # 测试集 (用于评估)
    # logger.info(f"\nLoading AC4C unbalanced TEST dataset (for Evaluation)...")
    # ac4c_unbalanced_test = AC4CDataset(
    #     mode='test',
    #     data_dir='npy/ac4c_processed/unbalanced_ac4c',
    #     cache_dir=Config.data.cache_dir,
    #     use_cache=True,
    #     preload_cache=True
    # )

    # ac4c_unbalanced_test_cache_stats = ac4c_unbalanced_test.get_cache_stats()
    # if not ac4c_unbalanced_test_cache_stats['batch_cache'].get('exists', False):
    #     logger.info("\n" + "="*60)
    #     logger.info("AC4C unbalanced TEST 批量缓存不存在，开始预计算所有二级结构...")
    #     logger.info("="*60)
    #     ac4c_unbalanced_test.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    # else:
    #     logger.info(f"\nAC4C unbalanced TEST 批量缓存已存在: {ac4c_unbalanced_test_cache_stats['batch_cache']['path']}")
    #     logger.info(f"  文件大小: {ac4c_unbalanced_test_cache_stats['batch_cache']['size_mb']:.2f} MB")
    #     if ac4c_unbalanced_test_cache_stats['batch_cache'].get('loaded_in_memory', False):
    #         logger.info(f"  状态: 已加载到内存")
    #     else:
    #         logger.info(f"  状态: 未加载到内存")
    # logger.info(f"AC4C unbalanced TEST dataset loaded: {len(ac4c_unbalanced_test)} samples")

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
        # print(getattr(Config, 'use_attention_supervision', False))
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=Config.use_hierarchical,
            use_amp=Config.use_amp and torch.cuda.is_available(),
            use_attention_supervision=getattr(Config, 'use_attention_supervision', False),
            attention_lambda=getattr(Config, 'attention_lambda', 1.0)
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
                use_hierarchical=Config.use_hierarchical,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log test loss to tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # ========================================================================
            # Evaluation Phase - Get predictions and compute metrics
            # ========================================================================
            logger.info(f"\n{'='*60} Epoch {epoch} Testing {'='*60}")

            # 1. Get evaluation metrics for human data
            # OPTIMIZATION: Run inference once and get all predictions
            logger.info(f"\n  Getting predictions for human data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)

            # 2. Compute basic classification metrics (keep original logic)
            # Human metrics
            metrics_unbalance = evaluate_unbalance(y_true, y_prob, Config.device, y_4class, Config.random_seed, y_4prob)
            metrics_balanceb = evaluate_balanceb(y_true, y_prob, y_4class, Config.device, Config.random_seed, y_4prob)
            metrics_group_balanceb = evaluate_group_balanceb(y_true, y_prob, y_4class, Config.random_seed)
            metrics_opt = evaluate_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)
            metrics_4class = evaluate_4class_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)

            # 3. Compute Top-K Attention Recall (Train.py unique logic)
            topk_results = {}
            comprehensive_results = {}
            if Config.use_attention_supervision:
                logger.info(f"\n  Computing Top-K site recall...")
                y_true, y_prob, y_4class, y_4prob, attn_weights, y_site = get_all_predictions_and_attention(
                    model, test_loader, Config.device, Config.use_hierarchical
                )
                if attn_weights is not None and y_site is not None:
                    # Original Macro-Average Top-K Recall
                    topk_results = calculate_topk_recall(attn_weights, y_site, k_list=[1, 3, 5, 7, 10, 20, 50])
                    # New Comprehensive Localization Metrics (Global/Micro-Average)
                    logger.info(f"  Computing comprehensive localization metrics...")
                    comprehensive_results = calculate_comprehensive_localization_metrics(
                        attn_weights, y_site, k_list=[1, 3, 5, 7, 10]
                    )
                else:
                    logger.info(f"  Attention weights or site labels not available, skipping Top-K evaluation.")

            # 4. Evaluate on plant data (if available)
            plant_metrics_unbalance = None
            plant_metrics_balanceb = None
            plant_metrics_4class = None
            plant_metrics_opt = None
            # Uncomment below to enable plant evaluation
            # logger.info(f"\n  Getting predictions for plant data...")
            # y_true_plant, y_prob_plant, y_4class_plant, y_4prob_plant = get_all_predictions(model, plant_test_loader, Config.device, Config.use_hierarchical)
            # plant_metrics_unbalance = evaluate_plant_unbalance(y_true_plant, y_prob_plant, Config.device, y_4class_plant, Config.random_seed, y_4prob_plant)
            # plant_metrics_balanceb = evaluate_plant_balanceb(y_true_plant, y_prob_plant, y_4class_plant, Config.device, Config.random_seed, y_4prob_plant)
            # plant_metrics_4class = plant_metrics_unbalance if plant_metrics_unbalance else plant_metrics_balanceb
            # plant_metrics_opt = evaluate_with_optimal_threshold(model, plant_test_loader, Config.device, Config.use_hierarchical)

            # =========================================================
            # 5. Output Tables (Modified: Match train_plant_single2.py style)
            # =========================================================

            # (A) Output all classification performance tables
            print_evaluation_results(
                metrics_unbalance=metrics_unbalance,
                metrics_balanceb=metrics_balanceb,
                epoch=epoch,
                logger=logger,
                metrics_opt=metrics_opt,
                metrics_group_balanceb=metrics_group_balanceb,
                plant_metrics_unbalance=plant_metrics_unbalance,
                plant_metrics_balanceb=plant_metrics_balanceb,
                metrics_4class=metrics_4class,
                plant_metrics_4class=plant_metrics_4class,
                plant_metrics_opt=plant_metrics_opt
            )

            # (B) Output Top-K performance table (Train.py specific)
            if topk_results:
                print_topk_table(topk_results, k_list=[1, 3, 5, 7, 10, 20, 50], logger=logger)

            # (C) Output Comprehensive Localization Metrics table (New)
            if comprehensive_results:
                print_comprehensive_table(comprehensive_results, k_list=[1, 3, 5, 7, 10], logger=logger)

            # Log metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_unbalance', epoch)
            log_metrics_to_tensorboard(tb_writer, metrics_balanceb, 'test_balanceb', epoch)

            # # Log plant metrics to tensorboard
            # log_metrics_to_tensorboard(tb_writer, plant_metrics_unbalance, 'plant_unbalance', epoch, key_prefix='group_plant_')
            # log_metrics_to_tensorboard(tb_writer, plant_metrics_balanceb, 'plant_balanceb', epoch, key_prefix='group_plant_')

            # # Run Plant Few-Shot Benchmark
            # run_few_shot_benchmark(
            #     model=model,
            #     plant_dataset=plant_dataset,
            #     device=Config.device,
            #     shots=[0, 1, 3, 5,25,50,100], 
            #     epoch=epoch,
            #     logger=logger,
            #     tb_writer=tb_writer,
            #     config=Config
            # )

            # # Run AC4C Few-Shot Benchmark (Balanced & Unbalanced)
            # # 分别使用 train 数据集采样 Support Set，test 数据集进行评估
            # # shots 支持：整数（每类采样k个）、浮点数（采样该类别50%）、'full'（采样100%）
            # run_few_shot_benchmark_ac4c(
            #     model=model,
            #     ac4c_balanced_train=ac4c_balanced_train,
            #     ac4c_balanced_test=ac4c_balanced_test,
            #     ac4c_unbalanced_train=ac4c_unbalanced_train,
            #     ac4c_unbalanced_test=ac4c_unbalanced_test,
            #     device=Config.device,
            #     shots=['full'],
            #     epoch=epoch,
            #     logger=logger,
            #     tb_writer=tb_writer,
            #     config=Config
            # )

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
