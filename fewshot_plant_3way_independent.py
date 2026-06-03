"""
fewshot_plant_3way_independent.py - 植物3路独立二分类训练 / Plant 3-Way Independent Binary Classification Training

3 个独立二分类任务 (one-vs-rest) 训练,避免类别难度差异 (m5C 简单,Y/m6A 难) 之间的梯度干扰。
为每个目标类 (Y=5, m5C=8, m6A=9) 训练单独的模型:每个模型从 checkpoint 重新加载并针对二分类任务特化。
3 independent binary classification tasks (one-vs-rest), avoiding gradient interference between classes of vastly different difficulties.
Trains separate models for each target class (Y=5, m5C=8, m6A=9): each model is reloaded from scratch and specialized.

功能模块 / Modules:
- 3 个独立模型训练 / 3 independent model training
- 1:1 二元采样 / 1:1 binary sampling
- 动态偏差冻结 / Dynamic bias freezing
- 二元 Focal Loss / Binary Focal Loss
- 一对多评估 / One-vs-Rest evaluation
- main: 主入口 / Main entry point

输入 / Inputs:
- json/plant_3way.json: 训练配置 / Training config
- plant3/seq.npy, plant3/12loc.npy: 植物数据 / Plant data
- 命令行参数 / CLI: --config, --gpu, --seed

输出 / Outputs:
- checkpoints/plant_y.pt, plant_m5c.pt, plant_m6a.pt: 3 个独立模型 / 3 independent models
- logs/plant_3way_*/results.json: 评估结果 / Evaluation results

数据流 / Data Flow:
1. 加载植物数据 / Load plant data
2. 初始化 3 个模型 / Init 3 models
3. 独立二分类训练 / Independent binary training
4. 一对多评估 (AUC, F1, Precision, Recall) / One-vs-Rest evaluation
5. 保存 3 个模型 / Save 3 models

相关文件 / Related Files:
- 调用 / Calls: model.main_model.RNA_ClassQuery_Model, dataset.plant.PlantDataset
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python fewshot_plant_3way_independent.py --config json/plant_3way.json --gpu 0

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import random
import json
import numpy as np
import torch
import argparse
import copy
from datetime import datetime
from prettytable import PrettyTable
import torch.nn.functional as F

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score
)

from model.main_model import RNA_ClassQuery_Model
from dataset.plant_single import PlantSingleDataset
from utils import (
    setup_logging, setup_tensorboard,
    save_checkpoint, load_config,
    get_all_predictions
)
from utils.common import GROUP_TO_CLASS_INDICES, INDEX_TO_GROUP
from utils.logging import print_few_shot_results

# Import few-shot utilities
from utils.few_shot import prepare_training_batch, apply_advanced_augmentation
import threading
import queue


# ============================================================================
# Binary Focal Loss Implementation
# ============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for One-vs-Rest classification.

    Focal Loss reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified examples.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t: probability of the true class
    - gamma: focusing parameter (higher = more focus on hard examples)
    - alpha: class weight (optional)

    Args:
        gamma: Focusing parameter. Default=2.0 (standard), 3.0 (aggressive)
        pos_weight: Weight for positive class (optional)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [N, 1] or [N]
            targets: Ground truth binary labels [N] (0 or 1)

        Returns:
            Focal loss value
        """
        # Ensure inputs are [N, 1]
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Compute binary cross-entropy loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.pos_weight, reduction='none'
        )

        # Compute p_t (probability of true class)
        pt = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, pt, 1 - pt)

        # Apply focal loss modulation: (1 - p_t)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Dynamic Augmentation with Prefetching (Producer-Consumer Pattern)
# ============================================================================

class AugmentationPrefetcher:
    """
    Data augmentation prefetcher using a producer-consumer pattern.

    This class maintains a buffer of augmented batches for future epochs,
    using a background thread to generate augmented data while the GPU trains
    the current epoch. This prevents augmentation from becoming a bottleneck.

    The queue stores lists of Data objects (each item is one full epoch's data).
    """

    def __init__(self, support_data, aug_factor, k_shot, buffer_size=5):
        """
        Initialize the augmentation prefetcher.

        Args:
            support_data: List of support data samples
            aug_factor: Augmentation factor (number of augmented copies per sample)
            k_shot: Current shot count
            buffer_size: Number of epochs to keep ready in the queue
        """
        self.support_data = support_data
        self.aug_factor = aug_factor
        self.k_shot = k_shot
        self.buffer_size = buffer_size

        # Queue stores lists of batches (each item is one full epoch's data)
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.loader_thread = threading.Thread(target=self._worker, daemon=True)
        self.loader_thread.start()

    def _worker(self):
        """
        Background thread worker that continuously fills the queue.

        This runs in a separate thread and generates augmented epochs
        while the main thread is training on the GPU.
        """
        while not self.stop_event.is_set():
            # If queue is full, wait briefly before checking again
            if self.queue.full():
                import time
                time.sleep(0.01)
                continue

            # Generate one epoch's worth of augmented data
            new_batch_list = self._prepare_binary_batch(
                self.support_data,
                self.aug_factor,
                self.k_shot
            )

            try:
                self.queue.put(new_batch_list, timeout=1)
            except queue.Full:
                pass

    def _prepare_binary_batch(self, support_data_list, aug_factor, k):
        """
        Prepare training batch for binary classification.

        Args:
            support_data_list: List of support data samples
            aug_factor: Augmentation factor
            k: Current shot count

        Returns:
            list: Prepared batch list with augmented data
        """
        batch_list = []

        # Original data
        for data in support_data_list:
            batch_list.append(data.clone())

        # Augmented data
        for _ in range(aug_factor):
            for data in support_data_list:
                aug_data = apply_advanced_augmentation(
                    data, protected_mask=None,
                    mutation_prob=0.01, protection_radius=5,
                    cutout_prob=0.1, drop_edge_prob=0.15
                )
                batch_list.append(aug_data)

        # Shuffle data
        if k != 'full':
            np.random.shuffle(batch_list)

        return batch_list

    def get_next_epoch_data(self):
        """
        Get the next epoch's augmented data from the buffer.

        This will block if the queue is empty (augmentation slower than training),
        but this is unlikely for few-shot scenarios.

        Returns:
            list: A list of Data objects for one epoch
        """
        return self.queue.get()

    def cleanup(self):
        """Signal the background thread to stop and wait for it to finish."""
        self.stop_event.set()
        if self.loader_thread.is_alive():
            self.loader_thread.join(timeout=2.0)


# ============================================================================
# Plant Class Configuration
# ============================================================================

PLANT_CLASS_MAPPING = {
    5: {'class_name': 'Y', 'group_idx': 3, 'group_name': 'U'},
    8: {'class_name': 'm5C', 'group_idx': 1, 'group_name': 'C'},
    9: {'class_name': 'm6A', 'group_idx': 0, 'group_name': 'A'}
}

TARGET_CLASSES = [5, 8, 9]


# ============================================================================
# Binary Sampling Functions
# ============================================================================

def sample_binary_support_set(dataset, plant_train_indices, k_shot, target_class):
    """
    Sample support set for binary classification (1:1 balanced).

    Positives: K samples from target_class
    Negatives: K samples from other plant classes (excluding target_class)

    CRITICAL: Do not use Zero/Background samples. This is Plant vs. Plant.

    Args:
        dataset: PlantSingleDataset instance
        plant_train_indices: List of available Plant training indices
        k_shot: Number of shots per class (total samples = 2*k_shot)
        target_class: Target class ID (5, 8, or 9)

    Returns:
        list: Support indices (balanced 1:1, total 2*k_shot samples)
    """
    if k_shot == 0 or k_shot == '0':
        return []

    # Get indices for target class (positives)
    target_positives = dataset.get_plant_indices_by_class(target_class)
    valid_positives = list(set(target_positives) & set(plant_train_indices))

    # Get indices for other classes (negatives)
    other_classes = [c for c in TARGET_CLASSES if c != target_class]
    valid_negatives = []
    for other_c in other_classes:
        other_indices = dataset.get_plant_indices_by_class(other_c)
        valid_negatives.extend(list(set(other_indices) & set(plant_train_indices)))

    # Sample with 1:1 balance
    selected_pos = []
    selected_neg = []

    # Sample k positives from target class
    if len(valid_positives) >= k_shot:
        selected_pos = np.random.choice(valid_positives, k_shot, replace=False).tolist()
    else:
        selected_pos = valid_positives
        print(f"Warning: Only {len(valid_positives)} positive samples available for class {target_class}, requested {k_shot}")

    # Sample k negatives from other classes
    num_needed = len(selected_pos)
    if len(valid_negatives) >= num_needed:
        selected_neg = np.random.choice(valid_negatives, num_needed, replace=False).tolist()
    else:
        selected_neg = np.random.choice(valid_negatives, num_needed, replace=True).tolist()
        print(f"Warning: Only {len(valid_negatives)} negative samples available for class {target_class}, using replacement")

    # Combine and shuffle
    support_indices = selected_pos + selected_neg
    np.random.shuffle(support_indices)

    return support_indices


def prepare_binary_labels(support_data_list, target_class):
    """
    Prepare binary labels for support set.

    Args:
        support_data_list: List of support Data objects
        target_class: Target class ID (5, 8, or 9)

    Returns:
        torch.Tensor: Binary labels (0 or 1) for each sample
    """
    binary_labels = []
    for data in support_data_list:
        # Check if this sample belongs to target class
        if data.y[0, target_class] == 1:
            binary_labels.append(1)  # Positive
        else:
            binary_labels.append(0)  # Negative

    return torch.tensor(binary_labels, dtype=torch.long)


# ============================================================================
# Dynamic Bias Freezing Strategy (Retained from Hybrid)
# ============================================================================

def apply_hybrid_strategy(model, k_shot, logger):
    """
    Apply HYBRID strategy with DYNAMIC bias freezing based on shot count.

    Key Innovation:
    --------------
    DYNAMIC Bias Freezing:
    - If k_shot <= 10: FREEZE bias (preserves pre-trained class priors for low-shot performance)
    - If k_shot > 10: UNFREEZE bias (allows updating priors with sufficient data)

    Rationale:
    ----------
    At low shots (<= 10), the frozen bias preserves class priors from pre-training,
    which is critical for m5C performance.

    At high shots (> 10), we have sufficient data (50+ samples) to safely update
    the class priors. Keeping bias frozen at high shots causes conflict between
    outdated priors and updated weights, leading to performance degradation.

    Args:
        model: The RNA_ClassQuery_Model
        k_shot: Current shot count (determines bias freezing strategy)
        logger: Logger instance

    Returns:
        list: Trainable parameters (weights + optionally bias)
    """
    freeze_bias = k_shot <= 10

    logger.info(f"\n{'='*80}")
    if freeze_bias:
        logger.info("HYBRID STRATEGY - LOW SHOT MODE (k_shot <= 10)")
        logger.info("Action: FREEZE Bias (preserves pre-trained class priors)")
    else:
        logger.info("HYBRID STRATEGY - HIGH SHOT MODE (k_shot > 10)")
        logger.info("Action: UNFREEZE Bias (allows updating class priors with sufficient data)")
    logger.info(f"{'='*80}")

    # Step 1: Freeze EVERYTHING first
    for param in model.parameters():
        param.requires_grad = False

    frozen_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Step 1: Frozen ALL parameters ({frozen_count:,} params)")

    # Step 2: Selectively unfreeze parameters based on mode
    trainable_params = []
    unfrozen_weight_count = 0
    unfrozen_bias_count = 0
    frozen_bias_count = 0

    # Navigate to the classification head
    head = model.class_query_head

    # Always unfreeze WEIGHTS in classification head
    if hasattr(head, 'output_proj'):
        for name, param in head.output_proj.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj.{name} ({param.numel():,} params)")

    if hasattr(head, 'output_proj_12class'):
        for name, param in head.output_proj_12class.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_12class.{name} ({param.numel():,} params)")

    if hasattr(head, 'output_proj_4class'):
        for name, param in head.output_proj_4class.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_4class.{name} ({param.numel():,} params)")

    # Conditionally unfreeze BIAS based on shot count
    if not freeze_bias:
        # High shot mode: Unfreeze bias to update class priors
        if hasattr(head, 'output_proj'):
            for name, param in head.output_proj.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    unfrozen_bias_count += param.numel()
                    logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj.{name} ({param.numel():,} params) [UPDATING PRIORS]")

        if hasattr(head, 'output_proj_12class'):
            for name, param in head.output_proj_12class.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    unfrozen_bias_count += param.numel()
                    logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_12class.{name} ({param.numel():,} params) [UPDATING PRIORS]")

        if hasattr(head, 'output_proj_4class'):
            for name, param in head.output_proj_4class.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    unfrozen_bias_count += param.numel()
                    logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_4class.{name} ({param.numel():,} params) [UPDATING PRIORS]")
    else:
        # Low shot mode: Keep bias frozen
        if hasattr(head, 'output_proj'):
            for name, param in head.output_proj.named_parameters():
                if 'bias' in name:
                    param.requires_grad = False
                    frozen_bias_count += param.numel()
                    logger.info(f"  ✗ FROZEN:  class_query_head.output_proj.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

        if hasattr(head, 'output_proj_12class'):
            for name, param in head.output_proj_12class.named_parameters():
                if 'bias' in name:
                    param.requires_grad = False
                    frozen_bias_count += param.numel()
                    logger.info(f"  ✗ FROZEN:  class_query_head.output_proj_12class.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

        if hasattr(head, 'output_proj_4class'):
            for name, param in head.output_proj_4class.named_parameters():
                if 'bias' in name:
                    param.requires_grad = False
                    frozen_bias_count += param.numel()
                    logger.info(f"  ✗ FROZEN:  class_query_head.output_proj_4class.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

    # Also unfreeze class queries if present
    if hasattr(head, 'class_queries'):
        head.class_queries.requires_grad = True
        trainable_params.append(head.class_queries)
        logger.info(f"  ✓ UNFROZEN: class_query_head.class_queries ({head.class_queries.numel():,} params)")

    if hasattr(head, 'group_queries'):
        head.group_queries.requires_grad = True
        trainable_params.append(head.group_queries)
        logger.info(f"  ✓ UNFROZEN: class_query_head.group_queries ({head.group_queries.numel():,} params)")

    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = total_model_params - trainable_count

    # Distinguish weights and biases among trainable params based on dimensionality
    # This assumes weights are typically 2D or higher, and biases are 1D
    trainable_weights_count = sum(p.numel() for p in trainable_params if p.dim() > 1)
    trainable_biases_count = sum(p.numel() for p in trainable_params if p.dim() == 1)

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY:")
    logger.info(f"  Total Parameters:     {total_model_params:,}")
    if total_model_params > 0: # Avoid division by zero
        logger.info(f"  Frozen Parameters:    {frozen_count:,} ({frozen_count/total_model_params*100:.2f}%)")
        logger.info(f"  Trainable Parameters: {trainable_count:,} ({trainable_count/total_model_params*100:.2f}%)")
    else:
        logger.info(f"  Frozen Parameters:    {frozen_count:,} (0.00%)")
        logger.info(f"  Trainable Parameters: {trainable_count:,} (0.00%)")
    
    logger.info(f"    └─ Trainable Weights: {trainable_weights_count:,}")
    logger.info(f"    └─ Trainable Biases:  {trainable_biases_count:,}")
    logger.info(f"{'='*80}\n")

    return trainable_params


# ============================================================================
# Binary Training Function
# ============================================================================

def train_binary_model(model, support_data_list, target_class, config, device, logger, k_shot):
    """
    Train a single binary classification model for One-vs-Rest task.

    Key Features:
    --------------
    1. DYNAMIC Bias Freezing (based on k_shot)
    2. BINARY Focal Loss
    3. L2-SP Regularization (prevents drift from 0-shot)

    Args:
        model: The RNA model (freshly loaded from checkpoint)
        support_data_list: List of support samples (balanced binary)
        target_class: Target class ID (5, 8, or 9)
        config: Configuration object
        device: Device to train on
        logger: Logger instance
        k_shot: Number of shots per class (total samples = 2*k_shot)

    Returns:
        Trained model
    """
    # Save original state for L2-SP regularization
    original_head_state = copy.deepcopy(model.class_query_head.state_dict())

    # Prepare binary labels
    binary_labels = prepare_binary_labels(support_data_list, target_class)

    # ========================================================================
    # STEP 1: Choose finetune strategy (full model vs. hybrid head-only)
    # ========================================================================
    full_finetune = bool(getattr(config, 'full_finetune', False))
    if full_finetune:
        logger.info(f"\n{'='*80}")
        logger.info("FINETUNE STRATEGY: FULL MODEL (full_finetune=True)")
        logger.info("Action: UNFREEZE ALL parameters and optimize model.parameters()")
        logger.info(f"{'='*80}\n")
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = model.parameters()
    else:
        # Hybrid strategy: freeze all then selectively unfreeze a small head subset
        trainable_params = apply_hybrid_strategy(model, k_shot, logger)

    # ========================================================================
    # STEP 2: Training Hyperparameters
    # ========================================================================
    ft_lr = 1e-4  # Learning rate for weights

    # Dynamic epochs based on shot count
    if k_shot <= 10:
        ft_epochs = 10
    else:
        ft_epochs = 20

    aug_factor = 10

    optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-3)

    # ========================================================================
    # STEP 3: Loss Function - BINARY FOCAL LOSS
    # ========================================================================
    gamma = 2.0
    criterion = BinaryFocalLoss(gamma=gamma, reduction='mean')

    logger.info(f"Binary Training Configuration:")
    logger.info(f"  Target Class: {target_class} ({PLANT_CLASS_MAPPING[target_class]['class_name']})")
    logger.info(f"  Loss: BinaryFocalLoss(gamma={gamma})")
    logger.info(f"  Learning Rate: {ft_lr}")
    logger.info(f"  Epochs: {ft_epochs}")
    logger.info(f"  Augmentation Factor: {aug_factor}")
    logger.info(f"  Positives: {binary_labels.sum().item()}, Negatives: {(binary_labels == 0).sum().item()}")

    # -------------------------------------------------------------------------
    # DYNAMIC AUGMENTATION: Initialize Prefetcher
    # -------------------------------------------------------------------------
    prefetcher = AugmentationPrefetcher(
        support_data_list,
        aug_factor,
        k_shot,
        buffer_size=5
    )

    model.train()
    mini_batch_size = 64

    # ========================================================================
    # STEP 4: Training Loop with L2-SP Regularization
    # ========================================================================
    l2_sp_lambda = 0.01

    logger.info(f"\nStarting training loop...")
    logger.info(f"  L2-SP Regularization: {l2_sp_lambda} (prevents drift from 0-shot)")

    try:
        for epoch in range(ft_epochs):
            # Get fresh augmented data for this epoch
            batch_list = prefetcher.get_next_epoch_data()

            epoch_loss = 0.0
            epoch_focal_loss = 0.0
            epoch_l2sp_loss = 0.0
            batches_processed = 0

            # Calculate mini-batches
            if len(batch_list) > 0:
                num_mini_batches = (len(batch_list) + mini_batch_size - 1) // mini_batch_size
            else:
                num_mini_batches = 0

            for i in range(num_mini_batches):
                batch_slice = batch_list[i*mini_batch_size : (i+1)*mini_batch_size]
                if not batch_slice:
                    continue

                batch = PyGBatch.from_data_list(batch_slice).to(device)
                optimizer.zero_grad()

                # Prepare binary labels for this mini-batch
                batch_binary_labels = prepare_binary_labels(batch_slice, target_class).to(device)

                # Forward pass
                if config.use_hierarchical:
                    # Model returns 3 values when in training mode: (logits_12class, logits_4class, attn_weights_12)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    if isinstance(out, tuple):
                        if len(out) == 3:
                            logits_class, logits_group, _ = out
                        else:
                            logits_class, logits_group = out
                    else:
                        logits_class = out
                    # Use the target class logit for binary classification
                    binary_logits = logits_class[:, target_class].unsqueeze(1)
                else:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    if isinstance(out, tuple):
                        logits_class = out[0]
                    else:
                        logits_class = out
                    binary_logits = logits_class[:, target_class].unsqueeze(1)

                # Binary focal loss
                focal_loss = criterion(binary_logits, batch_binary_labels.float().unsqueeze(1))

                # L2-SP Regularization: Penalize deviation from initial weights
                l2sp_loss = torch.tensor(0.0, device=device)
                current_head_state = model.class_query_head.state_dict()
                for name in current_head_state:
                    if name in original_head_state and current_head_state[name].requires_grad:
                        l2sp_loss += torch.sum((current_head_state[name] - original_head_state[name]) ** 2)

                # Total loss
                loss = focal_loss + l2_sp_lambda * l2sp_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_focal_loss += focal_loss.item()
                epoch_l2sp_loss += l2sp_loss.item()
                batches_processed += 1

            if batches_processed > 0 and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / batches_processed
                avg_focal = epoch_focal_loss / batches_processed
                avg_l2sp = epoch_l2sp_loss / batches_processed
                logger.info(f"Class {target_class} Shot {k_shot} Ep {epoch+1}/{ft_epochs}: "
                          f"Total={avg_loss:.4f}, Focal={avg_focal:.4f}, L2-SP={l2_sp_lambda}*{avg_l2sp:.4f}")

    finally:
        # Ensure thread cleanup
        prefetcher.cleanup()

    logger.info(f"Binary training completed for class {target_class} at {k_shot}-shot")

    return model


# ============================================================================
# Binary Evaluation Functions
# ============================================================================

def compute_binary_metrics(y_true, y_pred, y_prob):
    """
    Calculate detailed binary metrics for One-vs-Rest evaluation.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probabilities (float)

    Returns:
        dict: Binary metrics (F1, Prec, Rec, Acc, AUC, AUPRC, Sn, Sp, TP, TN, FP, FN)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Sensitivity (Recall) & Specificity
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC / AUPRC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    return {
        'F1': f1, 'Prec': prec, 'Rec': rec, 'Acc': acc,
        'AUC': auc, 'AUPRC': auprc, 'Sn': sn, 'Sp': sp,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def _balanced_subsample_indices(pos_indices, neg_indices, seed):
    """Return a balanced list of indices: all positives + min(len(pos), len(neg)) negatives."""
    pos_indices = list(pos_indices)
    neg_indices = list(neg_indices)

    if len(pos_indices) == 0:
        return [], []

    rng = np.random.RandomState(seed)
    if len(neg_indices) <= len(pos_indices):
        sampled_neg = neg_indices
    else:
        sampled_neg = rng.choice(neg_indices, size=len(pos_indices), replace=False).tolist()

    return pos_indices, sampled_neg


def evaluate_binary_task(model, test_loader, device, target_class, use_hierarchical=True):
    """Backward-compatible evaluation (keeps original Plant-only evaluation behavior)."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            logits_class = out[0] if isinstance(out, tuple) else out

            # Get probability for target class using sigmoid
            binary_logits = logits_class[:, target_class]
            prob = torch.sigmoid(binary_logits)

            # Binary prediction (threshold = 0.5)
            pred = (prob >= 0.5).long()

            # True binary label
            label = batch.y[:, target_class]

            # Original behavior: only include Plant-vs-Plant samples (exclude Zero)
            mask = batch.y[:, TARGET_CLASSES].sum(dim=1) > 0

            all_probs.append(prob[mask].cpu())
            all_preds.append(pred[mask].cpu())
            all_labels.append(label[mask].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return compute_binary_metrics(all_labels, all_preds, all_probs)


def evaluate_binary_task_balanced(
    model,
    dataset,
    indices,
    device,
    target_class,
    use_hierarchical=True,
    seed=42,
    batch_size=32,
):
    """Balanced evaluation on a given index pool.

    Note: This function intentionally ignores `use_hierarchical` because the model forward
    already returns (logits, ...) in this codebase; we handle tuple outputs safely.
    """
    # Determine pos/neg within provided indices
    y12 = dataset.y_12class  # (N, 12)
    pos_indices = [i for i in indices if y12[i, target_class] == 1]
    neg_indices = [i for i in indices if y12[i, target_class] == 0]

    pos_indices, sampled_neg = _balanced_subsample_indices(pos_indices, neg_indices, seed=seed)
    balanced_indices = pos_indices + sampled_neg

    # Short-circuit when no positives
    if len(pos_indices) == 0:
        return {
            'F1': 0.0, 'Prec': 0.0, 'Rec': 0.0, 'Acc': 0.0,
            'AUC': 0.5, 'AUPRC': 0.0, 'Sn': 0.0, 'Sp': 0.0,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            '_n_pos': 0, '_n_neg': 0, '_n_total': 0,
        }

    # Use PyG DataLoader to properly collate torch_geometric.data.Data objects
    from torch_geometric.loader import DataLoader as PyGDataLoader

    eval_loader = PyGDataLoader(
        Subset(dataset, balanced_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            logits_class = out[0] if isinstance(out, tuple) else out

            binary_logits = logits_class[:, target_class]
            prob = torch.sigmoid(binary_logits)
            pred = (prob >= 0.5).long()
            label = batch.y[:, target_class]

            all_probs.append(prob.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_binary_metrics(all_labels, all_preds, all_probs)
    metrics['_n_pos'] = int(len(pos_indices))
    metrics['_n_neg'] = int(len(sampled_neg))
    metrics['_n_total'] = int(len(balanced_indices))
    return metrics


def evaluate_all_metrics_once(model, test_loader, device, target_class, precomputed_indices, plant_test_indices, seed=42):
    """
    Evaluate all metrics in a single forward pass over the test set.
    
    This function performs ONE forward pass through the test set, caches all predictions,
    and then computes multiple evaluation metrics from the cached results:
    1. Original evaluation (all plant test samples)
    2. Balanced evaluation with plant negatives
    3. Balanced evaluation with zero negatives
    
    This avoids 3 separate forward passes and DataLoader creations, significantly
    improving evaluation speed.
    
    Args:
        model: The trained model
        test_loader: DataLoader for the test set
        device: Device to run evaluation on
        target_class: Target class ID (5, 8, or 9)
        precomputed_indices: Precomputed index pools
            {
                'plant_pos': List of positive indices (plant test samples of target class),
                'plant_neg': List of negative indices (plant test samples of other classes),
                'zero_neg': List of negative indices (zero test samples)
            }
        plant_test_indices: List of global indices in the test set (needed for index mapping)
        seed: Random seed for balanced sampling
    
    Returns:
        tuple: (metrics_original, metrics_balanced_plant, metrics_balanced_zero)
            Each is a dict with binary metrics (F1, Prec, Rec, Acc, AUC, AUPRC, Sn, Sp, TP, TN, FP, FN)
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    # ========================================================================
    # STEP 1: Single forward pass through the entire test set
    # ========================================================================
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            logits_class = out[0] if isinstance(out, tuple) else out
            
            # Cache all logits and labels
            all_logits.append(logits_class.cpu())
            all_labels.append(batch.y.cpu())
    
    # Merge all batches
    all_logits = torch.cat(all_logits)  # shape: (N, 12)
    all_labels = torch.cat(all_labels)  # shape: (N, 12)
    
    # ========================================================================
    # Create mapping from global indices to local test indices
    # ========================================================================
    # test_loader only contains plant_test_indices, so we need to map
    # global indices (0..242140) to local indices (0..78780)
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(plant_test_indices)}
    
    # ========================================================================
    # STEP 2: Compute original metrics (all plant test samples)
    # ========================================================================
    plant_mask = all_labels[:, TARGET_CLASSES].sum(dim=1) > 0
    logits_target = all_logits[plant_mask, target_class]
    labels_target = all_labels[plant_mask, target_class]
    probs = torch.sigmoid(logits_target)
    preds = (probs >= 0.5).long()
    
    metrics_original = compute_binary_metrics(
        labels_target.numpy(),
        preds.numpy(),
        probs.numpy()
    )
    
    # ========================================================================
    # STEP 3: Compute balanced metrics with plant negatives
    # ========================================================================
    plant_pos = precomputed_indices['plant_pos']
    plant_neg = precomputed_indices['plant_neg']
    
    if len(plant_pos) == 0:
        metrics_balanced_plant = {
            'F1': 0.0, 'Prec': 0.0, 'Rec': 0.0, 'Acc': 0.0,
            'AUC': 0.5, 'AUPRC': 0.0, 'Sn': 0.0, 'Sp': 0.0,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            '_n_pos': 0, '_n_neg': 0, '_n_total': 0,
        }
    else:
        # Balanced sampling: all positives + equal number of negatives
        rng = np.random.RandomState(seed)
        if len(plant_neg) > len(plant_pos):
            sampled_neg = rng.choice(plant_neg, len(plant_pos), replace=False).tolist()
        else:
            sampled_neg = plant_neg
        balanced_plant_indices = plant_pos + sampled_neg
        
        # Convert global indices to local test indices
        local_balanced_plant_indices = [global_to_local[idx] for idx in balanced_plant_indices if idx in global_to_local]
        
        # Extract predictions for balanced indices
        logits_balanced = all_logits[local_balanced_plant_indices, target_class]
        labels_balanced = all_labels[local_balanced_plant_indices, target_class]
        probs_balanced = torch.sigmoid(logits_balanced)
        preds_balanced = (probs_balanced >= 0.5).long()
        
        metrics_balanced_plant = compute_binary_metrics(
            labels_balanced.numpy(),
            preds_balanced.numpy(),
            probs_balanced.numpy()
        )
        metrics_balanced_plant['_n_pos'] = len(plant_pos)
        metrics_balanced_plant['_n_neg'] = len(sampled_neg)
        metrics_balanced_plant['_n_total'] = len(balanced_plant_indices)
    
    # ========================================================================
    # STEP 4: Compute balanced metrics with zero negatives
    # ========================================================================
    zero_neg = precomputed_indices['zero_neg']
    
    if len(plant_pos) == 0:
        metrics_balanced_zero = {
            'F1': 0.0, 'Prec': 0.0, 'Rec': 0.0, 'Acc': 0.0,
            'AUC': 0.5, 'AUPRC': 0.0, 'Sn': 0.0, 'Sp': 0.0,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            '_n_pos': 0, '_n_neg': 0, '_n_total': 0,
        }
    else:
        # Balanced sampling: all positives + equal number of zero negatives
        rng = np.random.RandomState(seed)
        if len(zero_neg) > len(plant_pos):
            sampled_zero = rng.choice(zero_neg, len(plant_pos), replace=False).tolist()
        else:
            sampled_zero = zero_neg
        balanced_zero_indices = plant_pos + sampled_zero
        
        # Convert global indices to local test indices
        local_balanced_zero_indices = [global_to_local[idx] for idx in balanced_zero_indices if idx in global_to_local]
        
        # Extract predictions for balanced indices
        logits_balanced = all_logits[local_balanced_zero_indices, target_class]
        labels_balanced = all_labels[local_balanced_zero_indices, target_class]
        probs_balanced = torch.sigmoid(logits_balanced)
        preds_balanced = (probs_balanced >= 0.5).long()
        
        metrics_balanced_zero = compute_binary_metrics(
            labels_balanced.numpy(),
            preds_balanced.numpy(),
            probs_balanced.numpy()
        )
        metrics_balanced_zero['_n_pos'] = len(plant_pos)
        metrics_balanced_zero['_n_neg'] = len(sampled_zero)
        metrics_balanced_zero['_n_total'] = len(balanced_zero_indices)
    
    return metrics_original, metrics_balanced_plant, metrics_balanced_zero


# ============================================================================
# Results Printing Functions
# ============================================================================

def print_binary_results(all_results_history, logger):
    """
    Print detailed results table for each binary task.

    Args:
        all_results_history: Results dictionary {target_class: {k_shot: metrics}}
        logger: Logger instance
    """
    for target_class in TARGET_CLASSES:
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']

        logger.info(f"\n{'='*100}")
        logger.info(f"Binary Classification Results: Class {target_class} ({class_name}) vs. Rest")
        logger.info(f"{'='*100}")

        table = PrettyTable()
        table.field_names = ["Shot", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
        table.align = "r"

        shot_keys = sorted(all_results_history[target_class].keys(), key=lambda x: int(x) if x!='full' else 999)

        for k in shot_keys:
            m = all_results_history[target_class][k]

            table.add_row([
                k,
                f"{m['F1']:.4f}", f"{m['Prec']:.4f}", f"{m['Rec']:.4f}", f"{m['Acc']:.4f}",
                f"{m['AUC']:.4f}", f"{m['AUPRC']:.4f}",
                f"{m['Sn']:.4f}", f"{m['Sp']:.4f}",
                m['TP'], m['TN'], m['FP'], m['FN']
            ])

        logger.info(f"\n{table}")


# ============================================================================
# Main Training Function
# ============================================================================

def main(config_path='json/plant_single.json', checkpoint_path=None):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name + '_3way_independent')
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_3way_independent')

    logger.info(f"\n{'='*80}")
    logger.info("Plant 3-Way Classification - INDEPENDENT Binary Tasks (One-vs-Rest)")
    logger.info(f"{'='*80}")
    logger.info("Core Strategy:")
    logger.info("  - Train 3 SEPARATE models (one for each class)")
    logger.info("  - Each model specializes in binary classification (Target vs. Rest)")
    logger.info("  - Avoids gradient interference between classes")
    logger.info("  - Binary sampling: 1:1 balance (K positives + K negatives)")
    logger.info(f"{'='*80}\n")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load Plant Dataset
    logger.info(f"Loading Plant Dataset...")
    full_dataset = PlantSingleDataset(
        plant_dir=Config.data.plant_data_dir,
        zero_dir=getattr(Config.data, 'zero_data_dir', 'npy/zero'),
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )

    # Prepare Plant train/test indices for few-shot sampling
    plant_train_indices = []
    plant_test_indices = []

    np.random.seed(Config.random_seed)
    for c in TARGET_CLASSES:
        c_pos_indices = full_dataset.get_plant_indices_by_class(c)
        np.random.shuffle(c_pos_indices)

        split_idx = int(0.1 * len(c_pos_indices))
        plant_train_indices.extend(c_pos_indices[:split_idx])
        plant_test_indices.extend(c_pos_indices[split_idx:])

    plant_train_indices = list(set(plant_train_indices))
    plant_test_indices = list(set(plant_test_indices))
    plant_test_indices = [i for i in plant_test_indices if i not in plant_train_indices]

    # Test set: Plant test only (no Zero samples)
    test_dataset = Subset(full_dataset, plant_test_indices)

    # Use PyG DataLoader for torch_geometric.data.Data batches
    from torch_geometric.loader import DataLoader as PyGDataLoader
    test_loader = PyGDataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Plant Train Pool: {len(plant_train_indices)}, Plant Test: {len(plant_test_indices)}")
    logger.info(f"Test Set: Plant Only (Classes {TARGET_CLASSES})")

    # Load checkpoint for base model
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error("Checkpoint path is required for independent binary training!")
        return

    # Results storage: {target_class: {k_shot: metrics}}
    all_results = {c: {} for c in TARGET_CLASSES}

    # Shot counts to evaluate
    shot_counts = [0, 2, 4, 6, 8, 10,20, 50, 100]

    # ========================================================================
    # PRECOMPUTE EVALUATION INDICES (Optimization: compute once, reuse for all shots)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("PRECOMPUTING EVALUATION INDICES")
    logger.info(f"{'='*80}")
    
    # Cache y_12class property to avoid repeated np.concatenate() calls
    # This is critical for performance: each access to y_12class would otherwise
    # concatenate 11.6 MB of data (242,141 samples x 12 classes x 4 bytes)
    y_12class_cached = full_dataset.y_12class
    
    precomputed_indices = {}
    for target_class in TARGET_CLASSES:
        other_classes = [c for c in TARGET_CLASSES if c != target_class]
        
        # Plant positives: test samples of target class
        plant_pos = [i for i in plant_test_indices if y_12class_cached[i, target_class] == 1]
        
        # Plant negatives: test samples of other plant classes
        plant_neg = [
            i for i in plant_test_indices
            if any(y_12class_cached[i, oc] == 1 for oc in other_classes)
        ]
        
        # Zero negatives: zero test samples
        _, zero_test = full_dataset.get_zero_split(test_ratio=0.2, seed=Config.random_seed)
        zero_neg = list(zero_test)
        
        precomputed_indices[target_class] = {
            'plant_pos': plant_pos,
            'plant_neg': plant_neg,
            'zero_neg': zero_neg
        }
        
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']
        logger.info(f"Class {target_class} ({class_name}): "
                   f"Plant Pos={len(plant_pos)}, Plant Neg={len(plant_neg)}, Zero Neg={len(zero_neg)}")
    
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # INDEPENDENT BINARY TRAINING LOOP
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"{'='*80}")
    logger.info(f"TRAINING 3 INDEPENDENT BINARY MODELS")
    logger.info(f"  Model 1: Class 5 (Y) vs. Rest")
    logger.info(f"  Model 2: Class 8 (m5C) vs. Rest")
    logger.info(f"  Model 3: Class 9 (m6A) vs. Rest")
    logger.info(f"  Strategy: Independent binary tasks with Hybrid training")
    logger.info(f"{'='*80}")
    logger.info(f"{'='*80}\n")

    # Loop over each target class
    for target_class in TARGET_CLASSES:
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']

        logger.info(f"\n{'#'*80}")
        logger.info(f"#{' '*78}#")
        logger.info(f"#  BINARY TASK: Class {target_class} ({class_name}) vs. Rest")
        logger.info(f"#{' '*78}#")
        logger.info(f"{'#'*80}\n")

        # Loop over shot counts
        for k_shot in shot_counts:
            logger.info(f"\n{'='*80}")
            logger.info(f"Class {target_class} ({class_name}) - {k_shot}-Shot Learning")
            logger.info(f"{'='*80}\n")

            # ====================================================================
            # STEP 1: Reload fresh model from checkpoint (CRITICAL!)
            # ====================================================================
            logger.info(f"Reloading fresh model from checkpoint...")
            model = RNA_ClassQuery_Model(
                cnn_hidden_dim=Config.cnn_hidden_dim,
                cnn_kernel_sizes=Config.cnn_kernel_sizes,
                cnn_dropout=Config.cnn_dropout,
                gcn_hidden_dim=Config.gcn_hidden_dim,
                gcn_out_channels=Config.gcn_out_channels,
                gcn_num_layers=Config.gcn_num_layers,
                gcn_dropout=Config.gcn_dropout,
                num_classes=12,
                num_attn_heads=Config.num_attn_heads,
                attn_dropout=Config.attn_dropout,
                use_simple_pooling=Config.use_simple_pooling,
                use_hierarchical=getattr(Config, 'use_hierarchical', True),
                use_layer_norm=Config.use_layer_norm
            ).to(Config.device)

            # Load pre-trained weights
            checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pre-trained weights from {checkpoint_path}")

            if k_shot == 0:
                # Zero-shot: Just evaluate directly
                logger.info(f"Zero-shot: Skipping training, evaluating directly...")
            else:
                # ====================================================================
                # STEP 2: Sample binary support set (1:1 balance)
                # ====================================================================
                np.random.seed(Config.random_seed + k_shot + target_class)
                support_indices = sample_binary_support_set(
                    full_dataset, plant_train_indices, k_shot, target_class
                )

                if len(support_indices) == 0:
                    logger.warning(f"No support samples found at {k_shot}-shot for class {target_class}!")
                    continue

                support_data_list = [full_dataset[i] for i in support_indices]

                # Count positives and negatives
                binary_labels = prepare_binary_labels(support_data_list, target_class)
                num_pos = (binary_labels == 1).sum().item()
                num_neg = (binary_labels == 0).sum().item()

                logger.info(f"Binary Sampling: {len(support_indices)} total samples "
                           f"(Positives: {num_pos}, Negatives: {num_neg})")

                # ====================================================================
                # STEP 3: Train binary model
                # ====================================================================
                logger.info(f"\nTraining binary model for class {target_class}...")
                Config.full_finetune=True
                model = train_binary_model(
                    model, support_data_list, target_class,
                    Config, Config.device, logger, k_shot
                )
                logger.info(f"Completed {k_shot}-shot binary training for class {target_class}")

            # ====================================================================
            # STEP 4: Evaluate binary model (OPTIMIZED: single forward pass for all metrics)
            # ====================================================================
            logger.info(f"Evaluating binary model for class {target_class}...")
            
            # Use seed for reproducibility in balanced sampling
            seed_eval = Config.random_seed + 1000 + int(k_shot) + int(target_class)
            
            # Single forward pass, compute all metrics from cached predictions
            metrics, metrics_bal_plantneg, metrics_bal_zeroneg = evaluate_all_metrics_once(
                model=model,
                test_loader=test_loader,
                device=Config.device,
                target_class=target_class,
                precomputed_indices=precomputed_indices[target_class],
                plant_test_indices=plant_test_indices,
                seed=seed_eval
            )

            # Store results
            all_results[target_class][k_shot] = metrics

            # Store balanced results separately for later pretty tables
            if 'all_results_balanced' not in locals():
                all_results_balanced = {c: {'plant_negative': {}, 'zero_negative': {}} for c in TARGET_CLASSES}
            all_results_balanced[target_class]['plant_negative'][k_shot] = metrics_bal_plantneg
            all_results_balanced[target_class]['zero_negative'][k_shot] = metrics_bal_zeroneg

            # Log results
            logger.info(f"\n{'='*80}")
            logger.info(f"Class {target_class} ({class_name}) - {k_shot}-Shot Results")
            logger.info(f"{'='*80}")
            logger.info(f"F1: {metrics['F1']:.4f}, Prec: {metrics['Prec']:.4f}, Rec: {metrics['Rec']:.4f}, "
                       f"Acc: {metrics['Acc']:.4f}, AUC: {metrics['AUC']:.4f}, AUPRC: {metrics['AUPRC']:.4f}")

            # Log to TensorBoard
            if tb_writer:
                tb_writer.add_scalar(f'binary_{target_class}/F1', metrics['F1'], k_shot)
                tb_writer.add_scalar(f'binary_{target_class}/Prec', metrics['Prec'], k_shot)
                tb_writer.add_scalar(f'binary_{target_class}/Rec', metrics['Rec'], k_shot)
                tb_writer.add_scalar(f'binary_{target_class}/Acc', metrics['Acc'], k_shot)
                tb_writer.add_scalar(f'binary_{target_class}/AUC', metrics['AUC'], k_shot)
                tb_writer.add_scalar(f'binary_{target_class}/AUPRC', metrics['AUPRC'], k_shot)

    # Print final results tables
    print_binary_results(all_results, logger)

    # -------------------------------------------------------------------------
    # Additional tables (NEW): Balanced evaluation with different negative sources
    # -------------------------------------------------------------------------
    if 'all_results_balanced' in locals():
        for target_class in TARGET_CLASSES:
            class_name = PLANT_CLASS_MAPPING[target_class]['class_name']

            logger.info(f"\n{'='*100}")
            logger.info(f"Balanced Evaluation (Plant Negatives): Class {target_class} ({class_name}) vs. Rest")
            logger.info(f"{'='*100}")
            t1 = PrettyTable()
            t1.field_names = ["Shot", "#Pos", "#Neg", "#Total", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
            t1.align = "r"
            shot_keys = sorted(all_results_balanced[target_class]['plant_negative'].keys(), key=lambda x: int(x) if x!='full' else 999)
            for k in shot_keys:
                m = all_results_balanced[target_class]['plant_negative'][k]
                t1.add_row([
                    k,
                    m.get('_n_pos', 0), m.get('_n_neg', 0), m.get('_n_total', 0),
                    f"{m['F1']:.4f}", f"{m['Prec']:.4f}", f"{m['Rec']:.4f}", f"{m['Acc']:.4f}",
                    f"{m['AUC']:.4f}", f"{m['AUPRC']:.4f}", f"{m['Sn']:.4f}", f"{m['Sp']:.4f}",
                    m['TP'], m['TN'], m['FP'], m['FN']
                ])
            logger.info(f"\n{t1}")

            logger.info(f"\n{'='*100}")
            logger.info(f"Balanced Evaluation (Zero Negatives): Class {target_class} ({class_name}) vs. Rest")
            logger.info(f"{'='*100}")
            t2 = PrettyTable()
            t2.field_names = ["Shot", "#Pos", "#Neg", "#Total", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
            t2.align = "r"
            shot_keys = sorted(all_results_balanced[target_class]['zero_negative'].keys(), key=lambda x: int(x) if x!='full' else 999)
            for k in shot_keys:
                m = all_results_balanced[target_class]['zero_negative'][k]
                t2.add_row([
                    k,
                    m.get('_n_pos', 0), m.get('_n_neg', 0), m.get('_n_total', 0),
                    f"{m['F1']:.4f}", f"{m['Prec']:.4f}", f"{m['Rec']:.4f}", f"{m['Acc']:.4f}",
                    f"{m['AUC']:.4f}", f"{m['AUPRC']:.4f}", f"{m['Sn']:.4f}", f"{m['Sp']:.4f}",
                    m['TP'], m['TN'], m['FP'], m['FN']
                ])
            logger.info(f"\n{t2}")

    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    results_path = os.path.join(Config.checkpoint_dir, 'plant_3way_independent_results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Organize results for better JSON structure
    serializable_results = {}
    for target_class in TARGET_CLASSES:
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']
        class_key = f"class_{target_class}_{class_name}_binary"

        serializable_results[class_key] = {}
        for k_shot, metrics in all_results[target_class].items():
            if k_shot == 'full':
                shot_key = 'full'
            else:
                shot_key = f"{k_shot}_shot"
            serializable_results[class_key][shot_key] = convert_to_serializable(metrics)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"Independent Binary results saved to: {results_path}")
    logger.info(f"{'='*80}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Plant 3-way classification using INDEPENDENT binary tasks (One-vs-Rest)"
    )
    parser.add_argument('--config', type=str, default='json/plant_single.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to pre-trained checkpoint',
                       default="logs/old/rna_classification_20260129_164810/checkpoints/epoch_030.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
