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
    Run Few-Shot Learning Benchmark with Stability Guarantees
    Strategy: 
    1. Bias-Only Tuning for Low Shots (Prevent Feature Destruction)
    2. Weight Interpolation (Soft Landing to 0-shot baseline)
    """
    from torch_geometric.data import Batch as PyGBatch
    import copy
    import numpy as np

    logger.info(f"\n>>> Starting Plant Few-Shot Benchmark (Shots: {shots}) <<<")

    # 保存原始 0-shot 参数作为锚点 (Anchor)
    original_state_dict = copy.deepcopy(model.state_dict())
    all_results = {}
    valid_indices = [5, 8, 9]

    # 获取增强保护 Mask
    dataset_labels = None
    if hasattr(plant_dataset, 'full_labels'):
        dataset_labels = plant_dataset.full_labels

    for k in shots:
        logger.info(f"\n--- Running {k}-Shot Adaptation ---")

        # [显存优化] 每个shot开始前清理显存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 1. 每次开始前，重置回原始状态 (从 0 开始)
        model.load_state_dict(original_state_dict)

        # ------------------------------------------------------------------
        # A. 采样 (Sampling)
        # ------------------------------------------------------------------
        support_indices = []
        if k > 0:
            y_true = plant_dataset.y_12class
            pos_indices_set = set()
            for c in valid_indices:
                c_pos_indices = np.where(y_true[:, c] == 1)[0]
                if len(c_pos_indices) >= k:
                    selected = np.random.choice(c_pos_indices, k, replace=False)
                else:
                    selected = c_pos_indices
                pos_indices_set.update(selected.tolist())
            support_indices = list(pos_indices_set)
            np.random.shuffle(support_indices)
            logger.info(f"Support set size: {len(support_indices)}")

        # 测试集准备
        test_loader = DataLoader(plant_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # ------------------------------------------------------------------
        # B. 策略配置 (Strategy Config)
        # ------------------------------------------------------------------
        if k > 0 and len(support_indices) > 0:
            support_data_list = [plant_dataset[i] for i in support_indices]
            
            # Mask 提取
            support_masks = []
            if dataset_labels is not None:
                for idx in support_indices:
                    support_masks.append((dataset_labels[idx] > 0))

            # [策略 1]: 动态冻结层 (Dynamic Freezing)
            # Low Shot (<10): 只有 Bias 能动 (Bias-Only Tuning)
            # High Shot (>=10): 整个 Head 能动
            trainable_params = []

            # 标记是否为 Bias-Only 模式
            is_bias_only = (k < 10)

            # Detect model type: RNA_ClassQuery_Model uses "class_query_head", model_v3 uses "NaiveFC"
            is_class_query_model = any("class_query_head" in name for name, _ in model.named_parameters())
            is_model_v3 = any("NaiveFC" in name for name, _ in model.named_parameters())

            for name, param in model.named_parameters():
                # For RNA_ClassQuery_Model: train class_query_head parameters
                # For model_v3: train NaiveFC (output layers) and optionally Attention
                should_train = False

                if is_class_query_model and "class_query_head" in name:
                    should_train = True
                elif is_model_v3:
                    if "NaiveFC" in name:
                        # Always train output FC layers
                        should_train = True
                    elif "Attention" in name and not is_bias_only:
                        # Train attention only in full-head mode
                        should_train = True

                if should_train:
                    if is_bias_only and is_class_query_model:
                        # For class_query_model in bias-only mode: only train bias
                        if "bias" in name:
                            param.requires_grad = True
                            trainable_params.append(param)
                        else:
                            param.requires_grad = False
                    else:
                        # Full-head mode or model_v3: train all selected parameters
                        param.requires_grad = True
                        trainable_params.append(param)
                else:
                    param.requires_grad = False

            mode_str = "Bias-Only" if is_bias_only else "Full-Head"
            logger.info(f"Training Mode: {mode_str} (samples={k}), Model: {'ClassQuery' if is_class_query_model else 'model_v3'})")

            # [策略 2]: 软 Bias 初始化
            # 帮助模型打破 Sp=0 的僵局，但不要像 -2.0 那么激进
            if k <= 5 and is_class_query_model:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "class_query_head" in name and "bias" in name:
                            # 仅针对 Valid Classes 微调初始值
                            if param.dim() == 1 and param.shape[0] == 12:
                                for c in valid_indices:
                                    # -0.5 对应 sigmoid 0.37，比较中性，既不全是1也不全是0
                                    param[c].fill_(-0.5)
            elif k <= 5 and is_model_v3:
                # For model_v3, initialize the final layer bias of NaiveFC for valid classes
                with torch.no_grad():
                    for i in valid_indices:
                        fc_layer = getattr(model, f"NaiveFC{i}")
                        # NaiveFC is a Sequential: Linear -> ReLU -> Dropout -> Linear
                        # Get the last Linear layer's bias
                        last_linear = fc_layer[-1]  # The final Linear layer
                        if hasattr(last_linear, 'bias') and last_linear.bias is not None:
                            last_linear.bias.fill_(-0.5) 

            # 优化器
            ft_optimizer = optim.AdamW(trainable_params, lr=1e-2, weight_decay=0.0) # Bias需要较大的LR
            ft_criterion = nn.BCEWithLogitsLoss()

            model.eval()
            ft_epochs = 30 # 轮数减少，Bias收敛很快

            # [显存优化]: 动态调整增强倍数
            # 低shot: 保持高增强以扩充数据
            # 高shot: 降低增强倍数以控制显存
            if k <= 10:
                aug_factor = 8
            elif k <= 50:
                aug_factor = 4
            else:
                aug_factor = 2

            # ------------------------------------------------------------------
            # C. 训练循环 (显存优化: 分批训练 + 梯度累积)
            # ------------------------------------------------------------------
            # [显存优化] 设置小batch size，通过梯度累积实现大batch训练
            micro_batch_size = 64  # 每个微批次的最大样本数

            for ft_ep in range(ft_epochs):
                batch_list = []
                # 原始 + 增强
                for data in support_data_list:
                    batch_list.append(data.clone())
                for _ in range(aug_factor):
                    for idx, data in enumerate(support_data_list):
                        curr_mask = None
                        if len(support_masks) > idx: curr_mask = support_masks[idx]
                        aug_data = apply_advanced_augmentation(
                            data, protected_mask=curr_mask,
                            mutation_prob=0.01, protection_radius=2,
                            cutout_prob=0.1, drop_edge_prob=0.15
                        )
                        batch_list.append(aug_data)

                # [显存优化] 分批处理所有数据，使用梯度累积
                if len(batch_list) > 0:
                    ft_optimizer.zero_grad()
                    total_loss = 0.0

                    # 将batch_list拆分为多个小batch
                    num_micro_batches = (len(batch_list) + micro_batch_size - 1) // micro_batch_size

                    for i in range(num_micro_batches):
                        start_idx = i * micro_batch_size
                        end_idx = min((i + 1) * micro_batch_size, len(batch_list))
                        micro_batch_list = batch_list[start_idx:end_idx]

                        # 构建当前微批次
                        batch = PyGBatch.from_data_list(micro_batch_list).to(device)

                        if config.use_hierarchical:
                            logits_12, _ = model(batch.x, batch.edge_index, batch.batch)
                        else:
                            logits_12 = model(batch.x, batch.edge_index, batch.batch)

                        loss = ft_criterion(logits_12[:, valid_indices], batch.y[:, valid_indices])
                        # 归一化损失以便梯度累积
                        loss = loss / num_micro_batches
                        loss.backward()
                        total_loss += loss.item()

                        # [显存优化] 清理中间变量
                        del batch, logits_12, loss
                        if i < num_micro_batches - 1:  # 最后一个batch后清理，保留loss用于logging
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    ft_optimizer.step()

            # ------------------------------------------------------------------
            # D. [核心策略 3]: 权重插值 (Weight Interpolation)
            # ------------------------------------------------------------------
            # 训练完后，强行把参数拉回 0-shot 附近
            # alpha 是 "新参数的保留比例"
            # k=1 -> alpha=0.2 (保留 80% 原知识)
            # k=10 -> alpha=0.8 (保留 20% 原知识)
            alpha = min(1.0, 0.1 + 0.1 * k)

            logger.info(f"Weight Interpolation: Mixing {alpha:.2f} Fine-tuned + {1-alpha:.2f} Original")

            current_state_dict = model.state_dict()
            mixed_state_dict = {}

            for key in current_state_dict:
                # 只对我们动过的 Head 参数做插值
                # For RNA_ClassQuery_Model: class_query_head parameters
                # For model_v3: NaiveFC and optionally Attention parameters
                should_interpolate = False
                if is_class_query_model and "class_query_head" in key:
                    should_interpolate = True
                elif is_model_v3:
                    if "NaiveFC" in key:
                        should_interpolate = True
                    elif "Attention" in key and not is_bias_only:
                        # Interpolate attention only if we trained it (full-head mode)
                        should_interpolate = True

                if should_interpolate:
                    w_ft = current_state_dict[key]
                    w_orig = original_state_dict[key]
                    # 插值公式
                    mixed_state_dict[key] = alpha * w_ft + (1 - alpha) * w_orig
                else:
                    mixed_state_dict[key] = current_state_dict[key] # Backbone 没动

            # 加载融合后的参数
            model.load_state_dict(mixed_state_dict)

            # [显存优化] 清理训练过程中的缓存
            del current_state_dict, mixed_state_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ------------------------------------------------------------------
        # E. 评估
        # ------------------------------------------------------------------
        # [显存优化] 评估前清理显存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, device, config.use_hierarchical)
        metrics_unbalance = evaluate_plant_unbalance(y_true, y_prob, device, y_4class, config.random_seed, y_4prob)
        metrics_balanceb = evaluate_plant_balanceb(y_true, y_prob, y_4class, device, config.random_seed, y_4prob)

        # [显存优化] 评估后清理预测结果
        del y_true, y_prob, y_4class, y_4prob
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_results[k] = {"unbalance": metrics_unbalance, "balanceb": metrics_balanceb}
        macro_f1 = metrics_unbalance.get('group_plant_opt_macro_f1', 0.0)
        logger.info(f"{k}-Shot Result - Macro F1: {macro_f1:.4f}")

    print_few_shot_results(all_results, epoch, logger)
    # 最后恢复原始模型
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
            use_hierarchical=Config.use_hierarchical,
            use_amp=Config.use_amp and torch.cuda.is_available()
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
