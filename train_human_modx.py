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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.modx import RNAClassifierWithWord2Vec
from dataset.human import Mer100Dataset
from dataset.plant import PlantDataset
from dataset.ac4c import AC4CDataset
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

def main(config_path='json/human.json'):
    """
    Main training function for RNA multi-label classification model.
    
    This function implements the complete training pipeline including:
    - Data loading and preprocessing
    - Multi-label disjoint data split
    - Model initialization and configuration
    - Training with dynamic class balancing
    - Multi-mode evaluation (Unbalance, BalanceB, Group BalanceB, Optimal Threshold)
    - Checkpoint saving and best model tracking
    - Tensorboard logging and visualization
    
    Args:
        config_path (str): Path to the JSON configuration file. Default is 'json/human.json'.
                          The config file contains hyperparameters, paths, and training settings.
    
    Returns:
        None: The function trains the model, saves checkpoints, and logs results to files.
              Best model is saved based on macro F1 score on the test set.
    """
    
    """
    RNA多标签分类模型的主训练函数。
    
    该函数实现了完整的训练流程，包括：
    - 数据加载和预处理
    - 多标签不相交数据分割
    - 模型初始化和配置
    - 动态类别平衡的训练
    - 多模式评估（Unbalance、BalanceB、Group BalanceB、Optimal Threshold）
    - 检查点保存和最佳模型跟踪
    - Tensorboard日志记录和可视化
    
    参数:
        config_path (str): JSON配置文件的路径。默认为'json/human.json'。
                          配置文件包含超参数、路径和训练设置。
    
    返回:
        None: 该函数训练模型、保存检查点，并将结果记录到文件中。
              基于测试集上的宏观F1分数保存最佳模型。
    """
    # Load configuration from JSON file
    # 加载JSON配置文件
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Setup logging to both file and console
    # 设置日志系统，同时输出到文件和控制台
    logger = setup_logging(Config.log_dir, Config.experiment_name)

    # Check if hierarchical mode is enabled (not supported by RNAClassifierWithWord2Vec)
    # 检查是否启用了分层模式（RNAClassifierWithWord2Vec不支持）
    if Config.use_hierarchical:
        logger.warning("WARNING: use_hierarchical=True is not supported by RNAClassifierWithWord2Vec.")
        logger.warning("Forcing use_hierarchical=False for compatibility.")
        Config.use_hierarchical = False
        config_dict['model']['use_hierarchical'] = False

    # Setup tensorboard for visualization
    # 设置Tensorboard用于可视化
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name)

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("RNA Multi-label Classification Training")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")

    # Set random seeds for reproducibility across all libraries
    # 为所有库设置随机种子以确保结果可复现
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load RNA sequence dataset with human m6A modification data
    # 加载RNA序列数据集，包含人类m6A修饰数据
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
    # 如果批量缓存不存在，预计算所有二级结构
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("批量缓存不存在，开始预计算所有二级结构...")
        logger.info("="*60)
        # Use multiprocessing for precomputation, num_workers=None means use all CPU cores automatically
        # 使用多进程预计算，num_workers=None 表示自动使用所有CPU核心
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\n批量缓存已存在: {cache_stats['batch_cache']['path']}")
        logger.info(f"  文件大小: {cache_stats['batch_cache']['size_mb']:.2f} MB")
        if cache_stats['batch_cache'].get('loaded_in_memory', False):
            logger.info(f"  状态: 已加载到内存")
        else:
            logger.info(f"  状态: 未加载到内存")

    # Split dataset into train and test sets with disjoint labels
    # Ensure no sample appears in both train and test sets
    # 将数据集分割为训练集和测试集，确保标签不相交
    # 确保没有样本同时出现在训练集和测试集中
    train_indices, test_indices = multi_label_disjoint_split(
        dataset,
        train_ratio=Config.train_ratio,
        random_seed=Config.random_seed,
        logger=logger
    )

    # Calculate smoothed positive weights for imbalanced classes
    # Used in loss function to handle class imbalance
    # 计算平滑的正样本权重以处理类别不平衡
    # 用于损失函数中以处理类别不平衡问题
    pos_weight_unbalanced = get_smoothed_pos_weights(
        dataset, train_indices, num_classes=Config.num_classes, logger=logger
    )

    # Create subsets for train and test datasets
    # 为训练集和测试集创建子集
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Use dynamic sampler for training to handle extreme class imbalance
    # Early epochs: balanced mode (rare class determines batches)
    # Late epochs: full coverage mode (all samples get sampled)
    # 使用动态采样器处理极端的类别不平衡问题
    # 早期训练轮次：平衡模式（稀有类别决定批次）
    # 后期训练轮次：全覆盖模式（所有样本都会被采样）
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

    # Create data loaders for training and testing
    # Important: When using batch_sampler, set shuffle=False and don't specify batch_size
    # The batch_sampler handles both shuffling and batching
    # 创建训练和测试的数据加载器
    # 重要提示：使用batch_sampler时，设置shuffle=False且不指定batch_size
    # batch_sampler同时处理洗牌和批处理
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

    # Initialize RNA classification model with BiLSTM and Bahdanau Attention
    # 初始化RNA分类模型，包含BiLSTM和Bahdanau注意力机制
    logger.info(f"\nCreating model...")
    model = RNAClassifierWithWord2Vec(
        input_dim=4,  # One-hot encoding for A, C, G, U
        embedding_dim=Config.cnn_hidden_dim,  # Use cnn_hidden_dim as embedding_dim
        hidden_dim=Config.gcn_hidden_dim,  # Use gcn_hidden_dim as LSTM hidden_dim
        num_layers=Config.gcn_num_layers,  # Use gcn_num_layers as LSTM layers
        output_dim=Config.num_classes,
        dropout=Config.cnn_dropout  # Use cnn_dropout as dropout rate
    ).to(Config.device)

    # Count and log model parameters
    # 计算并记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Initialize loss function with class-balanced weights
    # BCEWithLogitsLoss combines sigmoid and BCE loss for numerical stability
    # pos_weight handles class imbalance by weighting positive examples
    # 初始化带类别平衡权重的损失函数
    # BCEWithLogitsLoss结合了sigmoid和BCE损失以提高数值稳定性
    # pos_weight通过加权正样本处理类别不平衡
    pos_weight = pos_weight_unbalanced.to(Config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize optimizer and learning rate scheduler
    # AdamW: Adam with decoupled weight decay for better generalization
    # CosineAnnealing: Anneals learning rate following a cosine curve
    # 初始化优化器和学习率调度器
    # AdamW：带有解耦权重衰减的Adam优化器，可提高泛化能力
    # CosineAnnealing：按照余弦曲线衰减学习率
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

    # Log hyperparameters to Tensorboard for tracking and comparison
    # 将超参数记录到Tensorboard以便跟踪和比较
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

    # Initialize tracking variables for best model
    # 初始化最佳模型的跟踪变量
    best_macro_f1 = 0.0
    best_epoch = 0

    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}")

    # Main training loop over all epochs
    # 遍历所有训练轮次的主循环
    for epoch in range(1, Config.num_epochs + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Epoch {epoch}/{Config.num_epochs}")
        logger.info(f"{'#'*60}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Update dynamic sampler mode based on current epoch
        # Updates from BALANCED (early) to FULL_COVERAGE (late) automatically
        # 根据当前训练轮次更新动态采样器模式
        # 自动从BALANCED（早期）切换到FULL_COVERAGE（后期）
        if isinstance(train_batch_sampler, DynamicBalancedBatchSampler):
            train_batch_sampler.set_epoch(epoch)
            logger.info(f"Sampler mode: {train_batch_sampler.get_mode_info()}")

            # Update pos_weight based on sampler mode
            # Both BALANCED and FULL_COVERAGE modes use smoothed class weights
            # 根据采样器模式更新pos_weight
            # BALANCED和FULL_COVERAGE模式都使用平滑的类别权重
            new_pos_weight = pos_weight_unbalanced.to(Config.device)
            if train_batch_sampler.use_balanced_mode:
                logger.info(f"Class weights: BALANCED mode (smoothed class weights)")
            else:
                logger.info(f"Class weights: FULL_COVERAGE mode (smoothed class weights)")

            # Update the criterion with new pos_weight
            # 使用新的pos_weight更新损失函数
            criterion = nn.BCEWithLogitsLoss(pos_weight=new_pos_weight, reduction='mean')

        # Execute one training epoch
        # 训练一个完整的轮次
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=Config.use_hierarchical,
            use_amp=Config.use_amp and torch.cuda.is_available(),
            use_attention_supervision=getattr(Config, 'use_attention_supervision', False),
            attention_lambda=getattr(Config, 'attention_lambda', 1.0)
        )

        # Log training metrics to Tensorboard
        # 将训练指标记录到Tensorboard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Check if evaluation should be performed this epoch
        # 检查此轮次是否应该执行评估
        should_eval = (epoch % Config.test_interval == 0)
        metrics_unbalance = None
        current_macro_f1 = 0.0

        if should_eval:
            # Run evaluation on test set
            # 在测试集上运行评估
            logger.info(f"\nEvaluating...")
            test_loss = test_epoch(
                model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=Config.use_hierarchical,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log test loss to Tensorboard
            # 将测试损失记录到Tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # ========================================================================
            # Evaluation Phase - Get predictions and compute metrics
            # ========================================================================
            logger.info(f"\n{'='*60} Epoch {epoch} Testing {'='*60}")

            # 1. Get predictions for all test samples
            # OPTIMIZATION: Run inference once and cache all predictions for multiple metrics
            # 1. 获取所有测试样本的预测结果
            # 优化：运行一次推理并缓存所有预测结果用于多个指标计算
            logger.info(f"\n  Getting predictions for human data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)

            # 2. Compute multiple evaluation metrics with different strategies
            # - Unbalance: Standard threshold (0.5) evaluation
            # - BalanceB: Balanced threshold evaluation
            # - Group BalanceB: Group-aware balanced evaluation
            # - Optimal: Optimal threshold search
            # - 4class: 4-class classification metrics
            # 2. 使用不同策略计算多个评估指标
            # - Unbalance：标准阈值（0.5）评估
            # - BalanceB：平衡阈值评估
            # - Group BalanceB：组感知平衡评估
            # - Optimal：最优阈值搜索
            # - 4class：4类分类指标
            metrics_unbalance = evaluate_unbalance(y_true, y_prob, Config.device, y_4class, Config.random_seed, y_4prob)
            metrics_balanceb = evaluate_balanceb(y_true, y_prob, y_4class, Config.device, Config.random_seed, y_4prob)
            metrics_group_balanceb = evaluate_group_balanceb(y_true, y_prob, y_4class, Config.random_seed)
            metrics_opt = evaluate_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)
            metrics_4class = evaluate_4class_with_optimal_threshold(model, test_loader, Config.device, Config.use_hierarchical)

            # 3. Compute attention-based localization metrics if attention supervision is enabled
            # - Top-K Recall: Macro-average recall of true sites in top-K predictions
            # - Comprehensive Metrics: Global/micro-average localization performance
            # 3. 如果启用了注意力监督，计算基于注意力的定位指标
            # - Top-K Recall：真实位点在Top-K预测中的宏平均召回率
            # - Comprehensive Metrics：全局/微平均定位性能
            topk_results = {}
            comprehensive_results = {}
            if Config.use_attention_supervision:
                logger.info(f"\n  Computing Top-K site recall...")
                y_true, y_prob, y_4class, y_4prob, attn_weights, y_site = get_all_predictions_and_attention(
                    model, test_loader, Config.device, Config.use_hierarchical
                )
                if attn_weights is not None and y_site is not None:
                    # Original Macro-Average Top-K Recall
                    # 原始的宏平均Top-K召回率
                    topk_results = calculate_topk_recall(attn_weights, y_site, k_list=[1, 3, 5, 7, 10, 20, 50])
                    # New Comprehensive Localization Metrics (Global/Micro-Average)
                    # 新的综合定位指标（全局/微平均）
                    logger.info(f"  Computing comprehensive localization metrics...")
                    comprehensive_results = calculate_comprehensive_localization_metrics(
                        attn_weights, y_site, k_list=[1, 3, 5, 7, 10]
                    )
                else:
                    logger.info(f"  Attention weights or site labels not available, skipping Top-K evaluation.")

            # 4. Plant data evaluation (placeholder for cross-species transfer learning)
            # Not implemented in current version
            # 4. 植物数据评估（用于跨物种迁移学习的占位符）
            # 当前版本中未实现
            plant_metrics_unbalance = None
            plant_metrics_balanceb = None
            plant_metrics_4class = None
            plant_metrics_opt = None


            # =========================================================
            # 5. Output Tables (Modified: Match train_plant_single2.py style)
            # =========================================================

            # 5. Print formatted evaluation results
            # (A) Classification performance table
            # 5. 打印格式化的评估结果
            # (A) 分类性能表
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

            # (B) Top-K attention localization performance table
            # (B) Top-K注意力定位性能表
            if topk_results:
                print_topk_table(topk_results, k_list=[1, 3, 5, 7, 10, 20, 50], logger=logger)

            # (C) Comprehensive localization metrics table
            # (C) 综合定位指标表
            if comprehensive_results:
                print_comprehensive_table(comprehensive_results, k_list=[1, 3, 5, 7, 10], logger=logger)

            # Log test metrics to Tensorboard for visualization
            # 将测试指标记录到Tensorboard以便可视化
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_unbalance', epoch)
            log_metrics_to_tensorboard(tb_writer, metrics_balanceb, 'test_balanceb', epoch)

            
            # Track best model based on macro F1 score
            # 基于宏F1分数跟踪最佳模型
            current_macro_f1 = metrics_unbalance['group_macro_f1']
            logger.info(f"Current Unbalance Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")

        # Save checkpoint for each epoch if enabled
        # 如果启用，为每个训练轮次保存检查点
        if Config.save_every_epoch and metrics_unbalance is not None:
            epoch_checkpoint_path = os.path.join(Config.checkpoint_dir, f'epoch_{epoch:03d}.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, epoch_checkpoint_path, logger, config_dict)

        # Update and save best model checkpoint if performance improved
        # 如果性能提升，更新并保存最佳模型检查点
        if metrics_unbalance is not None and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            # Save best model checkpoint
            # 保存最佳模型检查点
            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, metrics_unbalance, checkpoint_path, logger, config_dict)

            # Log best F1 to Tensorboard
            # 将最佳F1分数记录到Tensorboard
            tb_writer.add_scalar('test/best_macro_f1', best_macro_f1, epoch)

    # Print training summary and close resources
    # 打印训练摘要并关闭资源
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Unbalance Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model saved to: {os.path.join(Config.checkpoint_dir, 'best_model.pt')}")

    # Close Tensorboard writer
    # 关闭Tensorboard写入器
    tb_writer.close()


if __name__ == "__main__":
    main(config_path='json/human_modx.json')
