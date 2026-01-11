"""
Few-Shot Learning Benchmark Functions for RNA Multi-label Classification

This module contains:
- run_few_shot_benchmark: Few-shot benchmark for Plant dataset
- run_few_shot_benchmark_ac4c: Few-shot benchmark for AC4C dataset (balanced and unbalanced)
"""

import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as PyGBatch
import logging

# Import constants from common
from .common import MOD_NAMES, GROUP_TO_CLASS_INDICES

# Import evaluation functions
from .metrics import (
    evaluate_unbalance, evaluate_balanceb, evaluate_ac4c,
    get_all_predictions
)


# =============================================================================
# Refactored Strategy Functions (Shared between both benchmarks)
# =============================================================================

def apply_dynamic_freezing(model, k, full_head_threshold, enable_dynamic_freezing,
                          logger=None):
    """
    [策略 1]: 动态冻结层 (Dynamic Freezing)

    Args:
        model: The model to configure
        k: Current shot count
        full_head_threshold: Threshold for switching from bias-only to full-head
        enable_dynamic_freezing: Whether to enable dynamic freezing
        logger: Logger instance

    Returns:
        tuple: (trainable_params, is_bias_only, is_class_query_model, is_model_v3)
    """
    trainable_params = []
    is_bias_only = enable_dynamic_freezing and (isinstance(k, (int, float)) and k < full_head_threshold)

    # Detect model type
    is_class_query_model = any("class_query_head" in name for name, _ in model.named_parameters())
    is_model_v3 = any("NaiveFC" in name for name, _ in model.named_parameters())

    for name, param in model.named_parameters():
        should_train = False

        if is_class_query_model and "class_query_head" in name:
            should_train = True
        elif is_model_v3:
            if "NaiveFC" in name:
                should_train = True
            elif "Attention" in name and not is_bias_only:
                should_train = True

        if should_train:
            if is_bias_only and is_class_query_model:
                if "bias" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params.append(param)
        else:
            param.requires_grad = False

    mode_str = "Bias-Only" if is_bias_only else "Full-Head"
    if logger:
        logger.info(f"Training Mode: {mode_str} (samples={k}), Model: {'ClassQuery' if is_class_query_model else 'model_v3'}")

    return trainable_params, is_bias_only, is_class_query_model, is_model_v3


def apply_soft_bias_init(model, k, bias_init_threshold, bias_init_value,
                        enable_soft_bias_init, valid_indices, logger=None):
    """
    [策略 2]: 软 Bias 初始化 (Soft Bias Initialization)

    帮助模型打破 Sp=0 的僵局，但不要像 -2.0 那么激进

    Args:
        model: The model to configure
        k: Current shot count
        bias_init_threshold: Threshold for applying bias initialization
        bias_init_value: Value to initialize bias with
        enable_soft_bias_init: Whether to enable soft bias initialization
        valid_indices: List of valid class indices
        logger: Logger instance
    """
    if not (enable_soft_bias_init and isinstance(k, (int, float)) and k <= bias_init_threshold):
        return

    # Detect model type
    is_class_query_model = any("class_query_head" in name for name, _ in model.named_parameters())
    is_model_v3 = any("NaiveFC" in name for name, _ in model.named_parameters())

    if is_class_query_model:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "class_query_head" in name and "bias" in name:
                    if param.dim() == 1 and param.shape[0] == 12:
                        for c in valid_indices:
                            param[c].fill_(bias_init_value)
    elif is_model_v3:
        with torch.no_grad():
            for i in valid_indices:
                fc_layer = getattr(model, f"NaiveFC{i}")
                last_linear = fc_layer[-1]
                if hasattr(last_linear, 'bias') and last_linear.bias is not None:
                    last_linear.bias.fill_(bias_init_value)


def apply_weight_interpolation(model, original_state_dict, k,
                               enable_weight_interpolation,
                               weight_interp_base_alpha, weight_interp_alpha_mult,
                               is_bias_only, is_class_query_model, is_model_v3,
                               logger=None):
    """
    [核心策略 3]: 权重插值 (Weight Interpolation)

    训练完后，强行把参数拉回 0-shot 附近
    alpha 是 "新参数的保留比例"

    Args:
        model: The model to update
        original_state_dict: Original model state dict (0-shot anchor)
        k: Current shot count
        enable_weight_interpolation: Whether to enable weight interpolation
        weight_interp_base_alpha: Base alpha for interpolation
        weight_interp_alpha_mult: Alpha multiplier per shot
        is_bias_only: Whether bias-only mode was used
        is_class_query_model: Whether model is class query type
        is_model_v3: Whether model is v3 type
        logger: Logger instance
    """
    # 根据 shot 类型调整 alpha 值
    if enable_weight_interpolation:
        if k == 'full':
            alpha = 1.0
        elif isinstance(k, float):
            alpha = min(1.0, k * 1.5)
        else:
            alpha = min(1.0, weight_interp_base_alpha + weight_interp_alpha_mult * k)
        if logger:
            logger.info(f"Weight Interpolation: Mixing {alpha:.2f} Fine-tuned + {1-alpha:.2f} Original")
    else:
        alpha = 1.0
        if logger:
            logger.info("Weight Interpolation: DISABLED, using fine-tuned weights only")

    current_state_dict = model.state_dict()
    mixed_state_dict = {}

    for key in current_state_dict:
        should_interpolate = False
        if enable_weight_interpolation:
            if is_class_query_model and "class_query_head" in key:
                should_interpolate = True
            elif is_model_v3:
                if "NaiveFC" in key:
                    should_interpolate = True
                elif "Attention" in key and not is_bias_only:
                    should_interpolate = True

        if should_interpolate:
            w_ft = current_state_dict[key]
            w_orig = original_state_dict[key]
            mixed_state_dict[key] = alpha * w_ft + (1 - alpha) * w_orig
        else:
            mixed_state_dict[key] = current_state_dict[key]

    model.load_state_dict(mixed_state_dict)


def prepare_training_batch(support_data_list, support_masks, aug_factor, k,
                          shuffle=True, logger=None):
    """
    准备训练批次：原始数据 + 增强数据

    Args:
        support_data_list: List of support data samples
        support_masks: List of protection masks for each sample
        aug_factor: Augmentation factor
        k: Current shot count
        shuffle: Whether to shuffle the batch list (non-full mode)
        logger: Logger instance

    Returns:
        list: Prepared batch list with augmented data
    """
    batch_list = []

    # 原始数据
    for data in support_data_list:
        batch_list.append(data.clone())

    # 增强数据
    for _ in range(aug_factor):
        for idx, data in enumerate(support_data_list):
            curr_mask = None
            if len(support_masks) > idx:
                curr_mask = support_masks[idx]
            aug_data = apply_advanced_augmentation(
                data, protected_mask=curr_mask,
                mutation_prob=0.01, protection_radius=5,
                cutout_prob=0.1, drop_edge_prob=0.15
            )
            batch_list.append(aug_data)

    # 打乱数据（非full模式）
    if shuffle and k != 'full':
        np.random.shuffle(batch_list)
        # if logger:
        #     logger.info(f"Shuffled batch_list: {len(batch_list)} samples (original + augmented)")

    return batch_list


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


def run_few_shot_benchmark(model, plant_dataset, device, shots=[0, 1, 3, 5, 7, 10],
                          epoch=0, logger=None, tb_writer=None, config=None):
    """
    Run Few-Shot Learning Benchmark on Plant Dataset with Stability Guarantees
    Strategy:
    1. Bias-Only Tuning for Low Shots (Prevent Feature Destruction)
    2. Weight Interpolation (Soft Landing to 0-shot baseline)

    Args:
        model: The model to evaluate
        plant_dataset: PlantDataset object
        device: Device to run evaluation on
        shots: List of shot counts to evaluate (default [0, 1, 3, 5, 7, 10])
        epoch: Current epoch number
        logger: Logger instance
        tb_writer: Tensorboard writer
        config: Configuration object (must contain few_shot config dict)

    Returns:
        Dictionary of results for each shot count
    """
    # 获取 few_shot 配置
    fs_config = getattr(config, 'few_shot', {}) if config else {}
    enable_dynamic_freezing = fs_config.get('enable_dynamic_freezing', False)
    enable_soft_bias_init = fs_config.get('enable_soft_bias_init', False)
    enable_weight_interpolation = fs_config.get('enable_weight_interpolation', False)
    bias_init_threshold = fs_config.get('bias_init_threshold', 5)
    full_head_threshold = fs_config.get('full_head_threshold', 10)
    weight_interp_base_alpha = fs_config.get('weight_interpolation_base_alpha', 0.1)
    weight_interp_alpha_mult = fs_config.get('weight_interpolation_alpha_multiplier', 0.1)
    bias_init_value = fs_config.get('bias_init_value', -0.5)
    ft_lr = fs_config.get('ft_lr', 5e-5)
    ft_epochs = fs_config.get('ft_epochs', 30)
    aug_factor_low = fs_config.get('aug_factor_low_shot', 8)
    aug_factor_mid = fs_config.get('aug_factor_mid_shot', 4)
    aug_factor_high = fs_config.get('aug_factor_high_shot', 2)

    logger.info(f"\n>>> Starting Plant Few-Shot Benchmark (Shots: {shots}) <<<")
    logger.info(f"Few-Shot Strategies: Dynamic Freezing={enable_dynamic_freezing}, "
                f"Soft Bias Init={enable_soft_bias_init}, Weight Interpolation={enable_weight_interpolation}")

    # 保存原始 0-shot 参数作为锚点 (Anchor)
    original_state_dict = copy.deepcopy(model.state_dict())
    all_results = {}
    valid_indices = [5, 8, 9]  # Plant valid classes: Y, m5C, m6A

    # 获取增强保护 Mask
    dataset_labels = None
    if hasattr(plant_dataset, 'full_labels'):
        dataset_labels = plant_dataset.full_labels

    for k in shots:
        logger.info(f"\n--- Running {k}-Shot Adaptation ---")

        # 1. 每次开始前，重置回原始状态 (从 0 开始)
        model.load_state_dict(original_state_dict)

        # ------------------------------------------------------------------
        # A. 采样 (Sampling)
        # ------------------------------------------------------------------
        support_indices = []
        if (isinstance(k, (int, float)) and k > 0) or (k == 'full'):
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
        test_loader = DataLoader(plant_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

        # ------------------------------------------------------------------
        # B. 策略配置 (Strategy Config)
        # ------------------------------------------------------------------
        if (isinstance(k, (int, float)) and k > 0) or (k == 'full') and len(support_indices) > 0:
            support_data_list = [plant_dataset[i] for i in support_indices]

            # Mask 提取
            support_masks = []
            if dataset_labels is not None:
                for idx in support_indices:
                    support_masks.append((dataset_labels[idx] > 0))

            # [策略 1]: 动态冻结层 (Dynamic Freezing)
            trainable_params, is_bias_only, is_class_query_model, is_model_v3 = apply_dynamic_freezing(
                model, k, full_head_threshold, enable_dynamic_freezing, logger
            )

            # [策略 2]: 软 Bias 初始化
            apply_soft_bias_init(
                model, k, bias_init_threshold, bias_init_value,
                enable_soft_bias_init, valid_indices, logger
            )

            # 优化器
            ft_optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-4)
            ft_criterion = nn.BCEWithLogitsLoss()

            model.eval()

            # [显存优化]: 动态调整增强倍数
            if isinstance(k, (int, float)) and k <= full_head_threshold:
                aug_factor = aug_factor_low
            elif isinstance(k, (int, float)) and k <= 50:
                aug_factor = aug_factor_mid
            else:
                aug_factor = aug_factor_high

            # ------------------------------------------------------------------
            # C. 训练循环 (正常反向传播，无梯度累积)
            # ------------------------------------------------------------------
            for ft_ep in range(ft_epochs):
                # 准备训练批次：原始数据 + 增强数据
                # 非full模式下打乱数据
                shuffle = (k != 'full')
                batch_list = prepare_training_batch(
                    support_data_list, support_masks, aug_factor, k,
                    shuffle=shuffle, logger=logger
                )

                # 正常反向传播，无梯度累积
                if len(batch_list) > 0:
                    batch = PyGBatch.from_data_list(batch_list).to(device)

                    if config.use_hierarchical:
                        logits_12, _ = model(batch.x, batch.edge_index, batch.batch)
                    else:
                        logits_12 = model(batch.x, batch.edge_index, batch.batch)

                    loss = ft_criterion(logits_12[:, valid_indices], batch.y[:, valid_indices])

                    ft_optimizer.zero_grad()
                    loss.backward()
                    ft_optimizer.step()

                    # 清理中间变量
                    del batch, logits_12, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # ------------------------------------------------------------------
            # D. [核心策略 3]: 权重插值 (Weight Interpolation)
            # ------------------------------------------------------------------
            apply_weight_interpolation(
                model, original_state_dict, k,
                enable_weight_interpolation,
                weight_interp_base_alpha, weight_interp_alpha_mult,
                is_bias_only, is_class_query_model, is_model_v3,
                logger
            )

        # ------------------------------------------------------------------
        # E. 评估
        # ------------------------------------------------------------------
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, device, config.use_hierarchical)

        # Use plant-specific evaluation functions
        from .metrics import evaluate_plant_unbalance, evaluate_plant_balanceb
        metrics_unbalance = evaluate_plant_unbalance(y_true, y_prob, device, y_4class, config.random_seed, y_4prob)
        metrics_balanceb = evaluate_plant_balanceb(y_true, y_prob, y_4class, device, config.random_seed, y_4prob)

        # 评估后清理预测结果
        del y_true, y_prob, y_4class, y_4prob
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_results[k] = {"unbalance": metrics_unbalance, "balanceb": metrics_balanceb}
        macro_f1 = metrics_unbalance.get('group_plant_opt_macro_f1', 0.0)
        logger.info(f"{k}-Shot Result - Macro F1: {macro_f1:.4f}")

    # Print results using utility function
    from .logging import print_few_shot_results
    print_few_shot_results(all_results, epoch, logger)

    # 最后恢复原始模型
    model.load_state_dict(original_state_dict)
    logger.info(">>> Few-Shot Benchmark Finished. <<<")
    return all_results


def run_few_shot_benchmark_ac4c(model,
                                ac4c_balanced_train, ac4c_balanced_test,
                                ac4c_unbalanced_train, ac4c_unbalanced_test,
                                device, shots=[0, 1, 3, 5, 7, 10, 0.5, 'full'],
                                epoch=0, logger=None, tb_writer=None, config=None):
    """
    Run Few-Shot Learning Benchmark on AC4C Dataset (Both Balanced and Unbalanced)
    重构版本：修复"结果全为0"的问题，使用 Train/Test 分离的策略

    核心修复：
    1. 使用 train 数据集采样 Support Set（避免数据泄露）
    2. 使用 test 数据集进行固定评估（确保公平比较）
    3. 支持整数 shot（每类k个）、浮点数 shot（50%）、'full' shot（100%）

    Evaluates both:
    - balanced_ac4c: train (support) -> test (query)
    - unbalanced_ac4c: train (support) -> test (query)

    Args:
        model: The model to evaluate
        ac4c_balanced_train: AC4CDataset for balanced training data (support set)
        ac4c_balanced_test: AC4CDataset for balanced test data (query set)
        ac4c_unbalanced_train: AC4CDataset for unbalanced training data (support set)
        ac4c_unbalanced_test: AC4CDataset for unbalanced test data (query set)
        device: Device to run evaluation on
        shots: List of shot counts to evaluate. Supports:
               - int: sample k samples per class (e.g., 1, 3, 5)
               - float: sample k% of samples per class (e.g., 0.5 = 50%)
               - 'full': sample 100% of samples per class
        epoch: Current epoch number
        logger: Logger instance
        tb_writer: Tensorboard writer
        config: Configuration object (must contain few_shot config dict)

    Returns:
        Dictionary containing results for both balanced and unbalanced datasets
        Format: {
            'balanced': {k_shot: {'unbalance': metrics, 'balanceb': metrics}},
            'unbalanced': {k_shot: {'unbalance': metrics, 'balanceb': metrics}}
        }
    """
    # 获取 few_shot 配置
    fs_config = getattr(config, 'few_shot', {}) if config else {}
    enable_dynamic_freezing = fs_config.get('enable_dynamic_freezing', False)
    enable_soft_bias_init = fs_config.get('enable_soft_bias_init', False)
    enable_weight_interpolation = fs_config.get('enable_weight_interpolation', False)
    bias_init_threshold = fs_config.get('bias_init_threshold', 5)
    full_head_threshold = fs_config.get('full_head_threshold', 10)
    weight_interp_base_alpha = fs_config.get('weight_interpolation_base_alpha', 0.1)
    weight_interp_alpha_mult = fs_config.get('weight_interpolation_alpha_multiplier', 0.1)
    bias_init_value = fs_config.get('bias_init_value', -0.5)
    ft_lr = fs_config.get('ft_lr', 5e-5)
    ft_epochs = fs_config.get('ft_epochs', 30)
    aug_factor_low = fs_config.get('aug_factor_low_shot', 8)
    aug_factor_mid = fs_config.get('aug_factor_mid_shot', 4)
    aug_factor_high = fs_config.get('aug_factor_high_shot', 2)

    logger.info(f"\n{'='*80}")
    logger.info(f">>> Starting AC4C Few-Shot Benchmark (Train->Support, Test->Eval) <<<")
    logger.info(f"{'='*80}")
    logger.info(f"Shots: {shots}")
    logger.info(f"Strategy: Sample from TRAIN set, Evaluate on TEST set (Fixed Query Set)")
    logger.info(f"Few-Shot Strategies: Dynamic Freezing={enable_dynamic_freezing}, "
                f"Soft Bias Init={enable_soft_bias_init}, Weight Interpolation={enable_weight_interpolation}")

    # Save all results
    all_results = {
        'balanced': {},
        'unbalanced': {}
    }

    # Helper function: Get valid classes from TRAIN dataset
    def get_valid_classes_from_train(train_dataset, dataset_name, logger):
        """
        从训练集确定哪些类别有数据
        这是关键修复：valid_classes 应该基于训练集，而不是测试集
        """
        y_12class = train_dataset.y_12class
        valid_classes = []
        class_counts = {}
        for c in range(12):
            pos_count = np.sum(y_12class[:, c] == 1)
            if pos_count > 0:
                valid_classes.append(c)
                class_counts[c] = int(pos_count)

        if logger:
            logger.info(f"AC4C {dataset_name} TRAIN - Valid classes: {valid_classes}")
            logger.info(f"  Class distribution: {class_counts}")
        return valid_classes

    # Get valid classes from TRAIN datasets (not test!)
    valid_classes_balanced = get_valid_classes_from_train(ac4c_balanced_train, 'balanced', logger)
    valid_classes_unbalanced = get_valid_classes_from_train(ac4c_unbalanced_train, 'unbalanced', logger)

    # Run evaluation for both datasets
    datasets_to_test = [
        ('balanced', ac4c_balanced_train, ac4c_balanced_test, valid_classes_balanced),
        ('unbalanced', ac4c_unbalanced_train, ac4c_unbalanced_test, valid_classes_unbalanced)
    ]

    for dataset_name, train_dataset, test_dataset, valid_indices in datasets_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f">>> Testing AC4C {dataset_name.upper()} Dataset <<<")
        logger.info(f"{'='*80}")

        # 保存原始 0-shot 参数作为锚点 (Anchor)
        original_state_dict = copy.deepcopy(model.state_dict())

        # 获取增强保护 Mask (AC4C doesn't have full_labels)
        dataset_labels = None
        if hasattr(train_dataset, 'full_labels'):
            dataset_labels = train_dataset.full_labels

        for k in shots:
            logger.info(f"\n--- Running {k}-Shot Adaptation on AC4C {dataset_name} ---")

            # 1. 每次开始前，重置回原始状态 (从 0 开始)
            model.load_state_dict(original_state_dict)

            # ------------------------------------------------------------------
            # A. 采样 (Sampling) - 从 TRAIN 集采样 Support Set
            # ------------------------------------------------------------------
            support_indices = []

            # 跳过 0-shot 的采样
            if k != 0:
                y_true = train_dataset.y_12class
                pos_indices_set = set()

                for c in valid_indices:
                    # 在 TRAIN 集中找到该类别的所有正样本
                    c_pos_indices = np.where(y_true[:, c] == 1)[0]
                    total_available = len(c_pos_indices)

                    if total_available == 0:
                        logger.warning(f"  Warning: Class {c} has no positive samples in TRAIN set, skipping...")
                        continue

                    # 根据 shot 类型确定采样数量
                    if k == 'full':
                        # 采样 100% 的正样本
                        num_samples = total_available
                        shot_desc = f"full (100%, {total_available} samples)"
                    elif isinstance(k, float):
                        # 采样 k% 的正样本
                        num_samples = max(1, int(total_available * k))  # 至少采样 1 个
                        shot_desc = f"{int(k*100)}% ({num_samples}/{total_available} samples)"
                    else:
                        # 整数 shot：每类采样 k 个
                        num_samples = k
                        shot_desc = f"{k}-shot ({num_samples} samples)"

                    # 执行采样
                    if total_available >= num_samples:
                        selected = np.random.choice(c_pos_indices, num_samples, replace=False)
                    else:
                        # 如果可用样本不足，全部采样
                        selected = c_pos_indices
                        logger.warning(f"  Warning: Class {c} only has {total_available} samples, using all instead of {num_samples}")

                    pos_indices_set.update(selected.tolist())

                support_indices = list(pos_indices_set)
                np.random.shuffle(support_indices)
                logger.info(f"Support set sampled from TRAIN: {len(support_indices)} samples")
            else:
                logger.info(f"0-shot: No support set, using pretrained model directly")

            # 测试集准备 - 使用 TEST 集进行固定评估（不在训练集中采样）
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
            logger.info(f"Test set for evaluation: {len(test_dataset)} samples (Fixed Query Set)")

            # ------------------------------------------------------------------
            # B. 策略配置 (Strategy Config)
            # ------------------------------------------------------------------
            if (isinstance(k, (int, float)) and k > 0) or (k == 'full') and len(support_indices) > 0:
                # 从 TRAIN 集采样数据
                support_data_list = [train_dataset[i] for i in support_indices]

                # Mask 提取
                support_masks = []
                if dataset_labels is not None:
                    for idx in support_indices:
                        support_masks.append((dataset_labels[idx] > 0))

                # [策略 1]: 动态冻结层 (Dynamic Freezing)
                trainable_params, is_bias_only, is_class_query_model, is_model_v3 = apply_dynamic_freezing(
                    model, k, full_head_threshold, enable_dynamic_freezing, logger
                )

                # [策略 2]: 软 Bias 初始化
                apply_soft_bias_init(
                    model, k, bias_init_threshold, bias_init_value,
                    enable_soft_bias_init, valid_indices, logger
                )

                # 优化器
                ft_optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-4)
                ft_criterion = nn.BCEWithLogitsLoss()

                model.eval()

                # [显存优化]: 动态调整增强倍数
                if isinstance(k, (int, float)) and k <= full_head_threshold:
                    aug_factor = aug_factor_low
                elif isinstance(k, (int, float)) and k <= 50:
                    aug_factor = aug_factor_mid
                else:
                    aug_factor = aug_factor_high

                # ------------------------------------------------------------------
                # C. 训练循环 (正常反向传播，无梯度累积)
                # ------------------------------------------------------------------
                # Mini-batch 配置
                mini_batch_size = 256

                for ft_ep in range(ft_epochs):
                    # 准备训练批次：原始数据 + 增强数据
                    # 非full模式下打乱数据
                    shuffle = (k != 'full')
                    batch_list = prepare_training_batch(
                        support_data_list, support_masks, aug_factor, k,
                        shuffle=shuffle, logger=logger
                    )

                    # 正常反向传播，分mini-batch处理以减少显存压力
                    if len(batch_list) > 0:
                        # 计算mini-batch数量
                        num_mini_batches = (len(batch_list) + mini_batch_size - 1) // mini_batch_size

                        for mini_batch_idx in range(num_mini_batches):
                            # 获取当前mini-batch
                            start_idx = mini_batch_idx * mini_batch_size
                            end_idx = min(start_idx + mini_batch_size, len(batch_list))
                            mini_batch_list = batch_list[start_idx:end_idx]

                            # 将mini-batch转换为PyG Batch
                            batch = PyGBatch.from_data_list(mini_batch_list).to(device)

                            # 前向传播
                            if config.use_hierarchical:
                                logits_12, _ = model(batch.x, batch.edge_index, batch.batch)
                            else:
                                logits_12 = model(batch.x, batch.edge_index, batch.batch)

                            # 计算损失
                            loss = ft_criterion(logits_12[:, valid_indices], batch.y[:, valid_indices])

                            # 反向传播
                            ft_optimizer.zero_grad()
                            loss.backward()
                            ft_optimizer.step()

                            # 清理中间变量以释放显存
                            # del batch, logits_12, loss

                        # 每个epoch结束后清理显存
                        # torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # ------------------------------------------------------------------
                # D. [核心策略 3]: 权重插值 (Weight Interpolation)
                # ------------------------------------------------------------------
                apply_weight_interpolation(
                    model, original_state_dict, k,
                    enable_weight_interpolation,
                    weight_interp_base_alpha, weight_interp_alpha_mult,
                    is_bias_only, is_class_query_model, is_model_v3,
                    logger
                )

            # ------------------------------------------------------------------
            # E. 评估 - 使用 TEST 集进行评估
            # ------------------------------------------------------------------
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, device, config.use_hierarchical)

            # Use evaluate_ac4c with dataset_mode based on current dataset
            metrics_unbalance = evaluate_ac4c(y_true, y_prob, y_4class, device, config.random_seed,
                                           dataset_mode='unbalanced', y_4prob=y_4prob)
            metrics_balanceb = evaluate_ac4c(y_true, y_prob, y_4class, device, config.random_seed,
                                         dataset_mode='balanced', y_4prob=y_4prob)

            del y_true, y_prob, y_4class, y_4prob
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 处理 shot 的 key 名称（处理非 hashable 类型）
            if isinstance(k, str):
                key_name = k  # 'full'
                shot_display = k.upper()
            elif isinstance(k, float):
                key_name = k  # 0.5
                shot_display = f"{int(k*100)}%"
            else:
                key_name = k  # 整数
                shot_display = f"{k}-shot"

            all_results[dataset_name][key_name] = {"unbalance": metrics_unbalance, "balanceb": metrics_balanceb}

            # 只打印 AC4C 相关的指标（过滤掉 Human/Plant 指标）
            ac4c_macro_f1 = metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0)
            ac4c_micro_f1 = metrics_unbalance.get('group_ac4c_micro_f1', 0.0)
            logger.info(f"AC4C {dataset_name} {shot_display} Result - Macro F1: {ac4c_macro_f1:.4f}, Micro F1: {ac4c_micro_f1:.4f}")

        # 最后恢复原始模型
        model.load_state_dict(original_state_dict)

    # Print results for both datasets - 打印 AC4C 详细指标表格
    if logger is not None:
        logger.info(f"\n{'='*120}")
        logger.info(f">>> AC4C FEW-SHOT BENCHMARK SUMMARY <<<")
        logger.info(f"{'='*120}")

    for dataset_name in ['balanced', 'unbalanced']:
        if logger is not None:
            logger.info(f"\n{'='*120}")
            logger.info(f">>> AC4C {dataset_name.upper()} DATASET RESULTS <<<")
            logger.info(f"{'='*120}")

        results = all_results[dataset_name]
        if not results:
            if logger is not None:
                logger.info(f"No results for {dataset_name} dataset.")
            continue

        # 打印表头 - 详细指标
        if logger is not None:
            logger.info(f"{'Shot':<10} {'Class':<8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8} {'AUPRC':>8} {'Sn':>8} {'Sp':>8} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8}")
            logger.info(f"{'-'*130}")

        # 按照原始 shots 列表的顺序打印结果
        for k in shots:
            # 处理 key 名称
            if isinstance(k, str):
                key_name = k
                shot_display = k.upper()
            elif isinstance(k, float):
                key_name = k
                shot_display = f"{int(k*100)}%"
            else:
                key_name = k
                shot_display = f"{k}-shot"

            if key_name in results and logger is not None:
                metrics = results[key_name]['unbalance']

                # 获取所有有数据的类别
                valid_classes = []
                for c in range(12):
                    if f'group_ac4c_class_{c}_opt_f1' in metrics:
                        valid_classes.append(c)

                # 打印每个类别的详细指标
                for c in valid_classes:
                    f1 = metrics.get(f'group_ac4c_class_{c}_opt_f1', 0.0)
                    prec = metrics.get(f'group_ac4c_class_{c}_opt_precision', 0.0)
                    rec = metrics.get(f'group_ac4c_class_{c}_opt_recall', 0.0)
                    auc = metrics.get(f'group_ac4c_class_{c}_auc', 0.0)
                    auprc = metrics.get(f'group_ac4c_class_{c}_auprc', 0.0)
                    sn = metrics.get(f'group_ac4c_class_{c}_opt_sensitivity', 0.0)
                    sp = metrics.get(f'group_ac4c_class_{c}_opt_specificity', 0.0)
                    tp = metrics.get(f'group_ac4c_class_{c}_opt_tp', 0)
                    tn = metrics.get(f'group_ac4c_class_{c}_opt_tn', 0)
                    fp = metrics.get(f'group_ac4c_class_{c}_opt_fp', 0)
                    fn = metrics.get(f'group_ac4c_class_{c}_opt_fn', 0)

                    if c == valid_classes[0]:
                        # 第一个类别显示 shot
                        logger.info(f"{shot_display:<10} {c:<8} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {auc:>8.4f} {auprc:>8.4f} {sn:>8.4f} {sp:>8.4f} {tp:>8.0f} {tn:>8.0f} {fp:>8.0f} {fn:>8.0f}")
                    else:
                        # 后续类别不显示 shot
                        logger.info(f"{'':<10} {c:<8} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {auc:>8.4f} {auprc:>8.4f} {sn:>8.4f} {sp:>8.4f} {tp:>8.0f} {tn:>8.0f} {fp:>8.0f} {fn:>8.0f}")

                # 打印平均指标
                macro_f1 = metrics.get('group_ac4c_opt_macro_f1', 0.0)
                micro_f1 = metrics.get('group_ac4c_micro_f1', 0.0)
                logger.info(f"{'':<10} {'AVG':<8} {macro_f1:>8.4f} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8}")
                logger.info(f"{'':<10} {'MICRO':<8} {micro_f1:>8.4f}")
                logger.info(f"{'-'*130}")

    if logger is not None:
        logger.info(f"{'='*80}")
        logger.info(">>> AC4C Few-Shot Benchmark Finished. <<<")
        logger.info(f"{'='*80}\n")

    return all_results
