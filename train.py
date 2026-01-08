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
# Data Augmentation Utilities
# ============================================================================

def apply_advanced_augmentation(data, protected_mask=None,
                                mutation_prob=0.01,    # 降低突变率
                                protection_radius=2,   # [关键] 保护半径 +/- 2bp
                                cutout_prob=0.1,       # [新增] 区域遮挡
                                drop_edge_prob=0.15):  # [新增] 图结构丢边
    """
    综合 RNA 增强：带上下文保护的突变 + 区域遮挡 + 结构丢边
    """
    from torch_geometric.utils import dropout_adj

    aug_data = data.clone()
    device = aug_data.x.device
    seq_len = aug_data.x.size(0)

    # ---------------------------------------------------------
    # 0. 预处理保护掩码 (Dilate Mask)
    # ---------------------------------------------------------
    final_protected = None
    if protected_mask is not None:
        # 确保 mask 是 Tensor
        if not isinstance(protected_mask, torch.Tensor):
            mask_tensor = torch.tensor(protected_mask, device=device, dtype=torch.float32)
        else:
            mask_tensor = protected_mask.to(device, dtype=torch.float32)

        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.view(1, 1, -1) # (1, 1, L)

        # 使用 MaxPool1d 进行膨胀 (Dilation)
        # Kernel size = 2*r + 1, Stride = 1, Padding = r
        if protection_radius > 0:
            k_size = 2 * protection_radius + 1
            dilated = torch.nn.functional.max_pool1d(
                mask_tensor, kernel_size=k_size, stride=1, padding=protection_radius
            )
            final_protected = dilated.view(-1) > 0.5 # 回到 (L, ) Boolean
        else:
            final_protected = mask_tensor.view(-1) > 0.5

    # ---------------------------------------------------------
    # 1. 序列突变 (Mutation / Flipping)
    # ---------------------------------------------------------
    if mutation_prob > 0:
        flip_mask = torch.rand(seq_len, device=device) < mutation_prob

        # 应用扩大的保护掩码
        if final_protected is not None:
            flip_mask = flip_mask & (~final_protected)

        num_flips = flip_mask.sum().item()
        if num_flips > 0:
            new_bases = torch.randint(0, 4, (num_flips,), device=device)
            new_one_hot = torch.zeros(num_flips, 4, device=device)
            new_one_hot.scatter_(1, new_bases.unsqueeze(1), 1.0)
            aug_data.x[flip_mask] = new_one_hot

    # ---------------------------------------------------------
    # 2. 区域遮挡 (Cutout / Span Masking) - 比突变更安全
    # ---------------------------------------------------------
    if cutout_prob > 0 and (torch.rand(1).item() < cutout_prob):
        cutout_len = 10
        # 随机选起点
        if seq_len > cutout_len:
            start_idx = torch.randint(0, seq_len - cutout_len, (1,)).item()
            end_idx = start_idx + cutout_len

            # 检查是否覆盖了受保护区域
            is_safe = True
            if final_protected is not None:
                if torch.any(final_protected[start_idx:end_idx]):
                    is_safe = False

            if is_safe:
                aug_data.x[start_idx:end_idx] = 0.0 # 遮挡

    # ---------------------------------------------------------
    # 3. 图结构丢边 (DropEdge) - 增强 GCN 鲁棒性
    # ---------------------------------------------------------
    if drop_edge_prob > 0 and aug_data.edge_index.size(1) > 0:
        aug_data.edge_index, _ = dropout_adj(
            aug_data.edge_index, p=drop_edge_prob, force_undirected=False
        )

    return aug_data


# ============================================================================
# Few-Shot Benchmark Function
# ============================================================================

def run_few_shot_benchmark(model, plant_dataset, device, shots=[0, 1, 3, 5, 7, 10],
                          epoch=0, logger=None, tb_writer=None, config=None):
    """
    Run Few-Shot Learning Benchmark with Bias Initialization Fix

    Key changes:
    1. Use mutual negative sampling (all classes in support set)
    2. Initialize head bias to -2.0 to fix 0-shot positive bias
    3. Full dataset for evaluation
    """
    from torch_geometric.data import Batch as PyGBatch

    logger.info(f"\n>>> Starting Plant Few-Shot Benchmark (Shots: {shots}) <<<")
    logger.info(">>> Strategy: Bias Reset (-2.0) + Mutual Negative Sampling <<<")

    original_state_dict = copy.deepcopy(model.state_dict())
    all_results = {}
    valid_indices = [5, 8, 9]  # Plant valid classes

    # 获取 Dataset 保护标签
    dataset_labels = None
    if hasattr(plant_dataset, 'full_labels'):
        dataset_labels = plant_dataset.full_labels
        logger.info(f"Augmentation: Using full_labels with +/- 2bp context protection.")
    else:
        logger.warning("Augmentation: 'full_labels' not found!")

    for k in shots:
        logger.info(f"\n--- Running {k}-Shot Adaptation ---")
        model.load_state_dict(original_state_dict)

        # ------------------------------------------------------------------
        # 1. 采样 (使用互为负样本策略)
        # ------------------------------------------------------------------
        support_indices = []
        if k > 0:
            y_true = plant_dataset.y_12class
            pos_indices_set = set()

            # 对每个有效类采样 k 个
            for c in valid_indices:
                c_pos_indices = np.where(y_true[:, c] == 1)[0]
                if len(c_pos_indices) >= k:
                    selected = np.random.choice(c_pos_indices, k, replace=False)
                else:
                    selected = c_pos_indices
                pos_indices_set.update(selected.tolist())

            support_indices = list(pos_indices_set)
            np.random.shuffle(support_indices)
            logger.info(f"Support set size: {len(support_indices)} (Valid Classes: {valid_indices})")

        # 准备测试集 (全量)
        test_loader = DataLoader(plant_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # ------------------------------------------------------------------
        # 2. 微调训练 (Fine-tuning)
        # ------------------------------------------------------------------
        if k > 0 and len(support_indices) > 0:
            support_data_list = [plant_dataset[i] for i in support_indices]

            # 提取保护 Mask
            support_masks = []
            if dataset_labels is not None:
                for idx in support_indices:
                    lbl = dataset_labels[idx]
                    mask = (lbl > 0)
                    support_masks.append(mask)

            # 冻结 Backbone, 激活 Head
            trainable_params = []
            head_bias_params = []  # 专门收集 Bias 参数

            for name, param in model.named_parameters():
                if "class_query_head" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    if "bias" in name:
                        head_bias_params.append(param)
                else:
                    param.requires_grad = False

            # [关键修复]：重置 Head 的 Bias
            # 0-shot 时模型倾向于预测全 1 (Sp=0)，我们需要手动把它按下去。
            # 将 valid_indices 对应的 Bias 设为 -2.0 (Sigmoid(-2.0) ≈ 0.12)
            # 这样模型初始状态会倾向于预测 0 (Negative)，从而大幅提升 Precision/Specificity
            with torch.no_grad():
                for bias in head_bias_params:
                    # 确保只修改 valid_indices 的 bias (如果是 12 类的 bias)
                    if bias.shape[0] == 12:
                        for c in valid_indices:
                            bias[c].fill_(-2.0)  # 强行设为负值
                    elif bias.shape[0] == 1:  # 如果是单输出
                        bias.fill_(-2.0)

            logger.info("Initialized Head Bias to -2.0 to fix 0-shot Positive Bias.")

            # 优化器
            ft_optimizer = optim.AdamW(trainable_params, lr=5e-3, weight_decay=0.01)
            ft_criterion = nn.BCEWithLogitsLoss()

            model.eval()  # Freeze BN
            ft_epochs = 50
            aug_factor = 8 if k <= 5 else 4  # 增强倍数

            for ft_ep in range(ft_epochs):
                batch_list = []

                # A. 原始样本
                for data in support_data_list:
                    batch_list.append(data.clone())

                # B. 增强样本
                for _ in range(aug_factor):
                    for idx, data in enumerate(support_data_list):
                        curr_mask = None
                        if len(support_masks) > idx:
                            curr_mask = support_masks[idx]

                        # 使用综合增强函数
                        aug_data = apply_advanced_augmentation(
                            data,
                            protected_mask=curr_mask,
                            mutation_prob=0.01,
                            protection_radius=2,
                            cutout_prob=0.1,
                            drop_edge_prob=0.15
                        )
                        batch_list.append(aug_data)

                if len(batch_list) > 0:
                    batch = PyGBatch.from_data_list(batch_list).to(device)
                    ft_optimizer.zero_grad()

                    if config.use_hierarchical:
                        logits_12, _ = model(batch.x, batch.edge_index, batch.batch)
                    else:
                        logits_12 = model(batch.x, batch.edge_index, batch.batch)

                    # 计算 Loss：此时 Class 5 的正样本会推高 Bias[5]，
                    # 而 Class 8/9 的样本（对于 Class 5 是负样本）会压低 Bias[5]。
                    # 由于 Bias 初始值已经是 -2.0，模型会更容易学到"拒绝"。
                    loss = ft_criterion(logits_12[:, valid_indices], batch.y[:, valid_indices])
                    loss.backward()
                    ft_optimizer.step()

        # ------------------------------------------------------------------
        # 3. 评估
        # ------------------------------------------------------------------
        y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, device, config.use_hierarchical)
        metrics_unbalance = evaluate_plant_unbalance(y_true, y_prob, device, y_4class, config.random_seed, y_4prob)
        metrics_balanceb = evaluate_plant_balanceb(y_true, y_prob, y_4class, device, config.random_seed, y_4prob)
        all_results[k] = {"unbalance": metrics_unbalance, "balanceb": metrics_balanceb}

        macro_f1 = metrics_unbalance.get('group_plant_opt_macro_f1', 0.0)
        logger.info(f"{k}-Shot Result - Macro F1: {macro_f1:.4f}")

    print_few_shot_results(all_results, epoch, logger)
    model.load_state_dict(original_state_dict)
    logger.info(">>> Few-Shot Benchmark Finished. <<<")
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
    logger.info(f"\nLoading dataset from {Config.data.human_data_dir}...")
    dataset = Mer100Dataset(
        mode='train', 
        data_dir=Config.data.human_data_dir, 
        cache_dir=Config.data.cache_dir,
        use_human3=True, 
        use_cache=True
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
    plant_dataset = PlantDataset(
        plant_dir=Config.data.plant_data_dir,
        cache_dir=Config.data.cache_dir,
        use_cache=True, 
        preload_cache=True
    )

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
                shots=[0, 1, 3, 5, 7, 10,25,50,100], 
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
