"""
Plant vs. Plant Binary Classification Training Script

This script implements "Plant Target Class vs. Other Plant Classes" binary classification:
1. Loop over plant classes [5, 8, 9] - training a specialized BINARY model for each class

NEW APPROACH: Plant Target vs. Plant Others
- Positives: k samples from Plant (specific target class)
- Negatives: k samples from Plant (other Plant classes, i.e., the remaining two classes)
- This is a TRUE binary classification within Plant: Target Modification vs. Other Modifications

2. Strict 1:1 Balanced Sampling:
   - Target Class: k positives from Plant (specific class)
   - Negative Pool: k samples from Plant (other classes, i.e., the remaining two)
   - Example: Class 5 uses k Plant-5 samples + k samples from {Plant-8, Plant-9}

3. Binary Cross-Entropy Loss (BCEWithLogitsLoss):
   - Standard binary classification loss
   - Suitable for balanced 1:1 binary classification

4. Adaptive Threshold Search:
   - After training, searches for optimal threshold on support set
   - Threshold range: [0.1, 0.95] with step 0.05
   - Maximizes F1 score on the support set

5. Model Pruning: Cuts the model to only compute for a specific target class

6. Filtered Output: Results tables show ONLY the current training class

7. Per-class results tracking with plant_fewshot_binary_results.json output

Plant Class Mappings:
- Class 5 (Y) -> Group U (index 3)
- Class 8 (m5C) -> Group C (index 1)
- Class 9 (m6A) -> Group A (index 0)

Dataset Structure:
- Plant: plant/seq.npy, plant/12loc.npy, plant/4loc.npy, plant/1001loc.npy
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

from model.main_model import RNA_ClassQuery_Model
from dataset.plant_single import PlantSingleDataset
from utils import (
    setup_logging, setup_tensorboard,
    save_checkpoint, load_config,
    evaluate_plant_unbalance, evaluate_plant_balanceb,
    get_all_predictions
)
from utils.common import GROUP_TO_CLASS_INDICES, INDEX_TO_GROUP
from utils.logging import print_few_shot_results

# Import few-shot utilities
from utils.few_shot import prepare_training_batch
import threading
import queue


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

    def __init__(self, support_data, support_masks, aug_factor, k_shot, buffer_size=5):
        """
        Initialize the augmentation prefetcher.

        Args:
            support_data: List of support data samples
            support_masks: List of protection masks for each sample
            aug_factor: Augmentation factor (number of augmented copies per sample)
            k_shot: Current shot count
            buffer_size: Number of epochs to keep ready in the queue
        """
        self.support_data = support_data
        self.support_masks = support_masks
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
            # Note: prepare_training_batch creates a list of Data objects
            new_batch_list = prepare_training_batch(
                self.support_data,
                self.support_masks,
                self.aug_factor,
                self.k_shot,
                shuffle=True,
                logger=None
            )

            try:
                self.queue.put(new_batch_list, timeout=1)
            except queue.Full:
                pass

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
# Plant Class to Group Mapping (Critical for hierarchical pruning)
# ============================================================================
PLANT_CLASS_MAPPING = {
    5: {'class_name': 'Y', 'group_idx': 3, 'group_name': 'U'},
    8: {'class_name': 'm5C', 'group_idx': 1, 'group_name': 'C'},
    9: {'class_name': 'm6A', 'group_idx': 0, 'group_name': 'A'}
}


class PrunedModelWrapper(nn.Module):
    """
    Wrapper for the pruned model to interface with existing evaluation functions.
    It pads the pruned 1-dim output back to 12-dim (filling others with very small logits).
    """
    def __init__(self, pruned_model, class_idx, group_idx):
        super().__init__()
        self.model = pruned_model
        self.class_idx = class_idx
        self.group_idx = group_idx
        self.use_hierarchical = pruned_model.use_hierarchical

    def forward(self, x, edge_index, batch=None):
        device = x.device

        # Get pruned output (Batch, 1) or (Batch, 1), (Batch, 1)
        out = self.model(x, edge_index, batch)

        if self.use_hierarchical:
            logits_1, logits_4_1 = out
            batch_size = logits_1.size(0)

            # Pad 12-class logits
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_idx] = logits_1.view(-1)

            # Pad 4-class logits
            padded_logits_4 = torch.full((batch_size, 4), -1e9, device=device)
            padded_logits_4[:, self.group_idx] = logits_4_1.view(-1)

            return padded_logits_12, padded_logits_4
        else:
            logits_1 = out
            batch_size = logits_1.size(0)

            # Pad 12-class logits
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_idx] = logits_1.view(-1)

            return padded_logits_12


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smooth_targets)
        return loss


def compute_f1_score(y_true, y_pred):
    """
    Compute F1 score for binary classification.

    Args:
        y_true: True binary labels (numpy array or tensor)
        y_pred: Predicted binary labels (numpy array or tensor)

    Returns:
        F1 score (float)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def sample_support_set_plant_vs_plant(dataset, plant_train_indices, k, target_class):
    """
    Sample support set for Plant vs. Plant binary classification with strict 1:1 balance.

    This function is used for "Plant Target vs. Plant Others" binary classification.
    It samples exactly k positives from Plant (target class) and k negatives from Plant (other classes).

    Sampling Strategy:
    1. Positives: k samples from Plant Train Indices (specific target class)
    2. Negatives: k samples from Plant Train Indices (other Plant classes)
    3. Result: Support set has 2*k samples (k positives + k negatives)

    Args:
        dataset: PlantSingleDataset instance containing Plant data
        plant_train_indices: List of available Plant training indices
        k: Number of shots (total samples = 2*k for binary)
        target_class: Target class ID (e.g., 5, 8, or 9 for Plant valid classes)

    Returns:
        list: Support indices (balanced 1:1, total 2*k samples)
              Positives from Plant (target class) + Negatives from Plant (other classes)

    Example:
        >>> dataset = PlantSingleDataset(plant_dir='plant')
        >>> plant_indices = dataset.get_plant_indices_by_class(target_class=5)
        >>> support = sample_support_set_plant_vs_plant(dataset, plant_indices, k=5, target_class=5)
        >>> # Returns 10 samples: 5 Plant (class 5) + 5 Plant (classes 8 or 9)
    """
    if k == 0 or k == '0':
        return []

    # 1. Get Positives from Plant (Intersection of Plant Train Indices AND Target Class)
    all_class_positives = dataset.get_plant_indices_by_class(target_class)
    valid_positives = list(set(all_class_positives) & set(plant_train_indices))

    # 2. Get Negatives from Plant (Other Plant Classes) - STRATIFIED SAMPLING
    # Define valid Plant classes for binary classification
    valid_classes = [5, 8, 9]  # Y, m5C, m6A

    # Identify negative classes (valid classes excluding target)
    negative_classes = [c for c in valid_classes if c != target_class]
    num_neg_classes = len(negative_classes)

    # 3. Sample with strict 1:1 balance
    selected_pos = []
    selected_neg = []

    # Sample k positives from Plant (target class)
    if len(valid_positives) >= k:
        selected_pos = np.random.choice(valid_positives, k, replace=False).tolist()
    else:
        # If not enough positives, use all available
        selected_pos = valid_positives
        print(f"Warning: Only {len(valid_positives)} positive samples available for class {target_class}, requested {k}")

    # STRATIFIED NEGATIVE SAMPLING: Distribute negatives evenly across negative classes
    # This ensures balanced representation (e.g., 50:50 instead of 1:99)
    num_needed = len(selected_pos)  # Should equal k for 1:1 balance

    if num_neg_classes > 0:
        # Calculate base count per class and remainder
        base_count = num_needed // num_neg_classes
        remainder = num_needed % num_neg_classes

        for idx, neg_c in enumerate(negative_classes):
            # Calculate count for this specific class
            count = base_count + (1 if idx < remainder else 0)

            # Get indices for this specific negative class ONLY
            c_indices = dataset.get_plant_indices_by_class(neg_c)
            valid_c_indices = list(set(c_indices) & set(plant_train_indices))

            # Sample 'count' items from this class
            if len(valid_c_indices) >= count:
                selected_from_class = np.random.choice(valid_c_indices, count, replace=False).tolist()
            else:
                # If not enough samples in this class, sample with replacement
                selected_from_class = np.random.choice(valid_c_indices, count, replace=True).tolist()
                print(f"Warning: Only {len(valid_c_indices)} samples available for negative class {neg_c}, using replacement for {count} samples")

            selected_neg.extend(selected_from_class)
    else:
        # Fallback: No negative classes found (should not happen in normal cases)
        print(f"Warning: No negative classes found for target class {target_class}")
        selected_neg = []

    # Combine and shuffle to mix positives and negatives
    support_indices = selected_pos + selected_neg
    np.random.shuffle(support_indices)

    return support_indices


def filter_evaluation_data_for_binary_classification(y_true, y_prob, y_4class, y_4prob,
                                                      test_indices, full_dataset, target_class):
    """
    Filter evaluation data to enforce "Target Plant Class vs. Other Plant Classes" binary classification.

    This function filters out ALL Zero samples and non-Plant data, ensuring that:
    - Positives: Target Plant Class samples (e.g., Class 5 samples when training Class 5)
    - Negatives: Other Plant Class samples ONLY (e.g., Class 8 and 9 when training Class 5)

    Args:
        y_true: True labels array (shape: [N, 12] for 12-class)
        y_prob: Probability predictions array (shape: [N, 12])
        y_4class: 4-class true labels (shape: [N, 4]) or None
        y_4prob: 4-class probability predictions (shape: [N, 4]) or None
        test_indices: List of indices in the test set (matches order of y_true/y_prob)
        full_dataset: PlantSingleDataset instance (has num_plant attribute)
        target_class: The target class being trained (e.g., 5, 8, 9)

    Returns:
        Filtered y_true, y_prob, y_4class, y_4prob, and valid_indices
    """
    num_plant = full_dataset.num_plant

    # Identify which samples to keep:
    # 1. KEEP: Plant samples (indices < num_plant) ONLY
    # 2. DISCARD: Zero samples (indices >= num_plant)
    # 3. Among Plant samples:
    #    - Positives: y_true[:, target_class] == 1 (target class)
    #    - Negatives: y_true[:, target_class] == 0 (other plant classes)

    valid_indices = []

    for idx, test_idx in enumerate(test_indices):
        # Check if this is a Plant sample
        is_plant_sample = (test_idx < num_plant)

        if is_plant_sample:
            # Keep all Plant samples (both target class and other classes)
            valid_indices.append(idx)
        # else: This is a Zero sample - DISCARD

    # Apply the filter
    filtered_y_true = y_true[valid_indices]
    filtered_y_prob = y_prob[valid_indices]

    filtered_y_4class = y_4class[valid_indices] if y_4class is not None else None
    filtered_y_4prob = y_4prob[valid_indices] if y_4prob is not None else None

    print(f"  [Filter] Original test set: {len(y_true)} samples")
    print(f"  [Filter] Filtered test set (Plant only): {len(filtered_y_true)} samples")
    print(f"  [Filter] Removed {len(y_true) - len(filtered_y_true)} Zero/background samples")

    return filtered_y_true, filtered_y_prob, filtered_y_4class, filtered_y_4prob, valid_indices


def train_few_shot_single_class_binary(model, support_data_list, target_class, group_idx,
                                  config, device, logger, k_shot):
    """
    Train a pruned model for BINARY classification of a SINGLE target class.
    Uses strict 1:1 balanced sampling with Focal Loss and adaptive threshold search.

    Args:
        model: The pruned RNA model
        support_data_list: List of support samples (balanced 1:1 pos/neg)
        target_class: The target class index (e.g., 5, 8, 9)
        group_idx: The corresponding group index (e.g., 3, 1, 0)
        config: Configuration object
        device: Device to train on
        logger: Logger instance
        k_shot: Number of shots per class (total samples = 2*k_shot)

    Returns:
        Trained model and best threshold found on support set
    """
    # Save original state for potential weight interpolation
    original_head_state = copy.deepcopy(model.class_query_head.state_dict())

    # Determine training strategy based on shot count and class
    # if k_shot <= 10 and target_class != 5:
    if True:
        # logger.info(f"Shot {k_shot}: Low Shot Strategy - Surgical Freezing (Frozen Backbone & MLPs, Train Projections Only)")

        # # Freeze EVERYTHING first
        # for param in model.parameters():
        #     param.requires_grad = False

        # # Only Unfreeze the Output Projections in the Head
        # for name, param in model.class_query_head.named_parameters():
        #     if "output_proj" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # trainable_params = [p for n, p in model.class_query_head.named_parameters() if "output_proj" in n]
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = model.parameters()
        ft_lr = 1e-4
        ft_epochs = 15
        aug_factor = 10
        use_weight_interpolation = False
        interpolation_alpha = 0.5

    else:
        if target_class == 5:
            logger.info(f"Shot {k_shot}: Class 5 Special Strategy - Full Fine-tuning (Correction Mode)")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = model.parameters()
            ft_lr = 1e-5
            ft_epochs = 50
            aug_factor = 8
            use_weight_interpolation = False
        else:
            logger.info(f"Shot {k_shot}: High Shot Strategy - Full Fine-tuning")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = model.parameters()
            ft_lr = 5e-5
            ft_epochs = 50
            aug_factor = 2
            use_weight_interpolation = False

    optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-4)

    # BINARY CROSS-ENTROPY LOSS
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------------------------------------------------------
    # DYNAMIC AUGMENTATION: Initialize Prefetcher (Instead of static batch prep)
    # -------------------------------------------------------------------------
    # The prefetcher will generate augmented data for multiple epochs in advance
    # using a background thread, preventing augmentation from becoming a bottleneck
    prefetcher = AugmentationPrefetcher(
        support_data_list,
        [],  # No masks for binary classification
        aug_factor,
        k_shot,
        buffer_size=5  # Prepare 5 epochs ahead
    )

    model.train()
    mini_batch_size = 64

    # Training loop with try-finally to ensure prefetcher cleanup
    try:
        for epoch in range(ft_epochs):
            # DYNAMIC AUGMENTATION: Get fresh augmented data for this epoch
            batch_list = prefetcher.get_next_epoch_data()

            epoch_loss = 0.0
            batches_processed = 0

            # Calculate mini-batches based on the new list
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

                if config.use_hierarchical:
                    logits_class, logits_group = model(batch.x, batch.edge_index, batch.batch)
                    # Binary label for target class only
                    target_class_label = batch.y[:, target_class].float().view(-1, 1)
                    # Select only the logits for the target class
                    logits_class_target = logits_class[:, target_class].unsqueeze(-1)
                    loss_class = criterion(logits_class_target, target_class_label)

                    if hasattr(batch, 'y_4class'):
                        target_group_label = batch.y_4class[:, group_idx].float().view(-1, 1)

                        # >>>>>>>>> 修改开始 >>>>>>>>>
                        # 提取对应 Group 的 logit (例如 Group U 对应的 logit)
                        logits_group_target = logits_group[:, group_idx].unsqueeze(-1)
                        loss_group = criterion(logits_group_target, target_group_label)
                        # <<<<<<<<< 修改结束 <<<<<<<<<

                        loss = loss_class + loss_group
                    else:
                        loss = loss_class
                else:
                    logits_class = model(batch.x, batch.edge_index, batch.batch)
                    # Binary label for target class only
                    target_class_label = batch.y[:, target_class].float().view(-1, 1)
                    # Select only the logits for the target class
                    logits_class_target = logits_class[:, target_class].unsqueeze(-1)
                    loss = criterion(logits_class_target, target_class_label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches_processed += 1

            if batches_processed > 0 and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / batches_processed
                logger.info(f"Binary Class {target_class} Shot {k_shot} Ep {epoch+1}/{ft_epochs}: BCE Loss {avg_loss:.4f}")
    finally:
        # Ensure thread cleanup
        prefetcher.cleanup()

    # Weight Interpolation for low shots
    # FORCE disable for Class 5 regardless of other settings
    if use_weight_interpolation and target_class != 5:
        logger.info(f"Applying Weight Interpolation (alpha={interpolation_alpha})...")
        with torch.no_grad():
            current_head_state = model.class_query_head.state_dict()
            for name, param in current_head_state.items():
                if name in original_head_state:
                    param.copy_((1 - interpolation_alpha) * param + interpolation_alpha * original_head_state[name])
    elif target_class == 5 and use_weight_interpolation:
        logger.info(f"Class 5 Special Strategy: Overriding Weight Interpolation (disabled)")

    # ========================================================================
    # ADAPTIVE THRESHOLD SEARCH on Support Set
    # ========================================================================
    logger.info(f"Searching for optimal threshold on support set...")

    model.eval()
    best_threshold = 0.5  # Default
    best_f1 = 0.0

    with torch.no_grad():
        # Get predictions on support set
        support_batch = PyGBatch.from_data_list(support_data_list).to(device)

        if config.use_hierarchical:
            logits_class, _ = model(support_batch.x, support_batch.edge_index, support_batch.batch)
            support_logits = logits_class[:, target_class].cpu()
        else:
            logits_class = model(support_batch.x, support_batch.edge_index, support_batch.batch)
            support_logits = logits_class[:, target_class].cpu()

        support_labels = support_batch.y[:, target_class].cpu()
        support_probs = torch.sigmoid(support_logits).numpy()
        support_labels_np = support_labels.numpy()

        # Search over thresholds
        threshold_range = np.arange(0.1, 0.95, 0.05)
        for threshold in threshold_range:
            preds = (support_probs >= threshold).astype(int)
            f1 = compute_f1_score(support_labels_np, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    logger.info(f"Best threshold found: {best_threshold:.2f} (Support Set F1: {best_f1:.4f})")

    return model, best_threshold


def print_single_class_few_shot_results(results_dict, target_class, logger):
    """
    Print few-shot results table for ONLY the target class.
    Filters the full results to show only the current training class.

    Args:
        results_dict: Results dictionary for all shot counts
        target_class: The target class to display (e.g., 5, 8, 9)
        logger: Logger instance
    """
    # Get plant class names
    class_names = {5: 'Y', 8: 'm5C', 9: 'm6A'}

    logger.info(f"\n{'='*120}")
    logger.info(f"BINARY CLASSIFICATION RESULTS - CLASS {target_class} ({class_names.get(target_class, 'Unknown')})")
    logger.info(f"Strategy: Plant Target vs. Plant Others (1:1 Balanced)")
    logger.info(f"{'='*120}")

    # Table 1: Unbalanced Results (only target class)
    logger.info(f"\nTable: Unbalanced Binary Classification - Class {target_class} Only")
    table_unbal = PrettyTable()
    table_unbal.field_names = ["Shot", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
    table_unbal.align = "r"

    for k_shot in sorted(results_dict.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and x != 'full' else 999):
        metrics = results_dict[k_shot].get('unbalance', {})
        f1 = metrics.get(f'group_plant_class_{target_class}_opt_f1', 0.0)
        prec = metrics.get(f'group_plant_class_{target_class}_opt_precision', 0.0)
        rec = metrics.get(f'group_plant_class_{target_class}_opt_recall', 0.0)
        acc = metrics.get(f'group_plant_class_{target_class}_opt_accuracy', 0.0)
        auc = metrics.get(f'group_plant_class_{target_class}_auc', 0.0)
        auprc = metrics.get(f'group_plant_class_{target_class}_auprc', 0.0)
        sn = metrics.get(f'group_plant_class_{target_class}_opt_sensitivity', 0.0)
        sp = metrics.get(f'group_plant_class_{target_class}_opt_specificity', 0.0)
        tp = metrics.get(f'group_plant_class_{target_class}_opt_tp', 0)
        tn = metrics.get(f'group_plant_class_{target_class}_opt_tn', 0)
        fp = metrics.get(f'group_plant_class_{target_class}_opt_fp', 0)
        fn = metrics.get(f'group_plant_class_{target_class}_opt_fn', 0)
        table_unbal.add_row([k_shot, f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{acc:.4f}",
                            f"{auc:.4f}", f"{auprc:.4f}", f"{sn:.4f}", f"{sp:.4f}",
                            tp, tn, fp, fn])

    logger.info(f"\n{table_unbal}")

    # Table 2: Balanced Results (only target class)
    logger.info(f"\nTable: Balanced Binary Classification - Class {target_class} Only")
    table_bal = PrettyTable()
    table_bal.field_names = ["Shot", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
    table_bal.align = "r"

    for k_shot in sorted(results_dict.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and x != 'full' else 999):
        metrics = results_dict[k_shot].get('balanceb', {})
        f1 = metrics.get(f'group_plant_class_{target_class}_opt_f1', 0.0)
        prec = metrics.get(f'group_plant_class_{target_class}_opt_precision', 0.0)
        rec = metrics.get(f'group_plant_class_{target_class}_opt_recall', 0.0)
        acc = metrics.get(f'group_plant_class_{target_class}_opt_accuracy', 0.0)
        auc = metrics.get(f'group_plant_class_{target_class}_auc', 0.0)
        auprc = metrics.get(f'group_plant_class_{target_class}_auprc', 0.0)
        sn = metrics.get(f'group_plant_class_{target_class}_opt_sensitivity', 0.0)
        sp = metrics.get(f'group_plant_class_{target_class}_opt_specificity', 0.0)
        tp = metrics.get(f'group_plant_class_{target_class}_opt_tp', 0)
        tn = metrics.get(f'group_plant_class_{target_class}_opt_tn', 0)
        fp = metrics.get(f'group_plant_class_{target_class}_opt_fp', 0)
        fn = metrics.get(f'group_plant_class_{target_class}_opt_fn', 0)
        table_bal.add_row([k_shot, f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{acc:.4f}",
                          f"{auc:.4f}", f"{auprc:.4f}", f"{sn:.4f}", f"{sp:.4f}",
                          tp, tn, fp, fn])

    logger.info(f"\n{table_bal}")


def print_single_class_results(metrics, target_class, logger, threshold=0.5):
    """
    Print ONLY the table row for the current target class.
    Filters the full metrics to show only the current training class.

    Args:
        metrics: Metrics dictionary from evaluation
        target_class: The target class to display (e.g., 5, 8, 9)
        logger: Logger instance
        threshold: The threshold used for prediction
    """
    # Get plant class names
    class_names = {5: 'Y', 8: 'm5C', 9: 'm6A'}

    f1 = metrics.get(f'group_plant_class_{target_class}_opt_f1', 0.0)
    prec = metrics.get(f'group_plant_class_{target_class}_opt_precision', 0.0)
    rec = metrics.get(f'group_plant_class_{target_class}_opt_recall', 0.0)
    acc = metrics.get(f'group_plant_class_{target_class}_opt_accuracy', 0.0)
    auc = metrics.get(f'group_plant_class_{target_class}_auc', 0.0)
    auprc = metrics.get(f'group_plant_class_{target_class}_auprc', 0.0)
    sn = metrics.get(f'group_plant_class_{target_class}_opt_sensitivity', 0.0)
    sp = metrics.get(f'group_plant_class_{target_class}_opt_specificity', 0.0)
    tp = metrics.get(f'group_plant_class_{target_class}_opt_tp', 0)
    tn = metrics.get(f'group_plant_class_{target_class}_opt_tn', 0)
    fp = metrics.get(f'group_plant_class_{target_class}_opt_fp', 0)
    fn = metrics.get(f'group_plant_class_{target_class}_opt_fn', 0)

    logger.info(f"Class {target_class} ({class_names.get(target_class, 'Unknown')}) | Threshold: {threshold:.2f} | "
                f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f} | "
                f"AUC: {auc:.4f} | AUPRC: {auprc:.4f} | Sn: {sn:.4f} | Sp: {sp:.4f} | "
                f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")


def main(config_path='json/plant.json', checkpoint_path=None):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name + '_plant_vs_plant_binary')
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_plant_vs_plant_binary')

    logger.info(f"\n{'='*60}")
    logger.info("Plant vs. Plant Binary Classification Training")
    logger.info("Strategy: Plant Target Class vs. Other Plant Classes")
    logger.info("         Strict 1:1 Balanced | BCE Loss | Adaptive Threshold")
    logger.info(f"{'='*60}")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load Plant Dataset (No Zero dataset needed for Plant vs Plant)
    logger.info(f"Loading Plant Dataset...")
    full_dataset = PlantSingleDataset(
        plant_dir=Config.data.plant_data_dir,
        zero_dir=getattr(Config.data, 'zero_data_dir', 'npy/zero'),
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )

    # Plant target classes to iterate over
    target_classes = [5, 8, 9]

    # Prepare Plant train/test indices for few-shot sampling
    plant_train_indices = []
    plant_test_indices = []

    np.random.seed(Config.random_seed)
    for c in target_classes:
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
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Plant Train Pool: {len(plant_train_indices)}, Plant Test: {len(plant_test_indices)}")
    logger.info(f"Test Set: Plant Only (No Zero/background)")

    # Load checkpoint for base model
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error("Checkpoint path is required for binary classification training!")
        return

    # Results storage: structure is all_results[target_class][k_shot] = metrics
    all_results = {}

    # shot_counts = [0, 1, 3, 5, 7, 10, 50, 100]
    shot_counts = [0, 2, 4, 6, 8, 10, 50, 100]

    # ========================================================================
    # MAIN LOOP: Iterate over each target class for BINARY classification
    # ========================================================================
    for target_class in target_classes:
        logger.info(f"\n{'='*80}")
        logger.info(f"{'='*80}")
        logger.info(f"TRAINING BINARY MODEL FOR CLASS {target_class} ({PLANT_CLASS_MAPPING[target_class]['class_name']})")
        logger.info(f"  Group: {PLANT_CLASS_MAPPING[target_class]['group_name']} (index {PLANT_CLASS_MAPPING[target_class]['group_idx']})")
        logger.info(f"  Strategy: Plant (Positives) vs. Plant (Other Classes/Negatives)")
        logger.info(f"  Strict 1:1 Balanced | BCE Loss | Adaptive Threshold")
        logger.info(f"{'='*80}")
        logger.info(f"{'='*80}\n")

        class_info = PLANT_CLASS_MAPPING[target_class]
        group_idx = class_info['group_idx']
        class_name = class_info['class_name']

        # Initialize results dict for this class
        all_results[target_class] = {}

        # Create FRESH model for this class
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

        # Save original 12-class state for this class's shots
        original_state_dict = copy.deepcopy(model.state_dict())

        # ====================================================================
        # Loop over shot counts for this class
        # ====================================================================
        for k_shot in shot_counts:
            logger.info(f"\n--- Binary Class {target_class} ({class_name}): Running {k_shot}-Shot Learning ---")

            # Reset model to original 12-class state
            model.load_state_dict(original_state_dict, strict=False)

            # Clear any pruning buffers from previous iterations
            if hasattr(model.class_query_head, 'valid_class_indices'):
                del model.class_query_head.valid_class_indices
            if hasattr(model.class_query_head, 'valid_group_indices'):
                del model.class_query_head.valid_group_indices

            if k_shot == 0:
                # Zero-shot: Just prune and evaluate
                logger.info(f"Zero-shot: Pruning to class {target_class} only")
                model.prune_heads([target_class], [group_idx])
            else:
                # BINARY Balanced Sampling: k_shot positives from Plant + k_shot negatives from Plant (other classes)
                np.random.seed(Config.random_seed + target_class + k_shot)
                support_indices = sample_support_set_plant_vs_plant(
                    full_dataset, plant_train_indices, k_shot, target_class
                )

                if len(support_indices) == 0:
                    logger.warning(f"No support samples found for class {target_class} at {k_shot}-shot!")
                    continue

                support_data_list = [full_dataset[i] for i in support_indices]

                # Count positives and negatives for logging
                # Positives: samples where target_class == 1
                pos_count = sum(1 for i in support_indices if i < full_dataset.num_plant and full_dataset.plant_y12[i, target_class] == 1)
                neg_count = len(support_indices) - pos_count
                logger.info(f"Plant vs. Plant Sampling: {len(support_indices)} samples ({pos_count} Plant pos + {neg_count} Plant neg)")

                # Train with BINARY classification (balanced, Focal Loss)
                model, best_threshold = train_few_shot_single_class_binary(
                    model, support_data_list, target_class, group_idx,
                    Config, Config.device, logger, k_shot
                )
                logger.info(f"Class {target_class} | {k_shot}-Shot | Best Threshold: {best_threshold:.2f}")

                # Prune model after training for evaluation
                logger.info(f"Pruning to class {target_class} only after {k_shot}-shot binary training")
                model.prune_heads([target_class], [group_idx])

            # Wrap model for evaluation
            wrapped_model = PrunedModelWrapper(model, class_idx=target_class, group_idx=group_idx)

            # Evaluate
            wrapped_model.eval()

            logger.info(f"Getting predictions for binary class {target_class}...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(
                wrapped_model, test_loader, Config.device,
                getattr(Config, 'use_hierarchical', True)
            )

            # ====================================================================
            # CRITICAL: Filter evaluation data to enforce "Target Plant vs. Other Plant" only
            # Remove ALL Zero samples (background) - keep only Plant samples
            # ====================================================================
            logger.info(f"[Binary Class {target_class}] Applying Target-vs-OtherPlant filtering to evaluation data...")
            filtered_y_true, filtered_y_prob, filtered_y_4class, filtered_y_4prob, valid_idx = \
                filter_evaluation_data_for_binary_classification(
                    y_true, y_prob, y_4class, y_4prob,
                    plant_test_indices, full_dataset, target_class
                )

            # Run evaluation on FILTERED data (Plant only)
            metrics_unbalance = evaluate_plant_unbalance(
                filtered_y_true, filtered_y_prob, Config.device, filtered_y_4class,
                random_seed=Config.random_seed
            )
            metrics_balanceb = evaluate_plant_balanceb(
                filtered_y_true, filtered_y_prob, filtered_y_4class, Config.device,
                random_seed=Config.random_seed
            )

            # Store results
            all_results[target_class][k_shot] = {
                'unbalance': metrics_unbalance,
                'balanceb': metrics_balanceb
            }

            # Log results (filtered to show only target class)
            logger.info(f"\n{'='*60}")
            logger.info(f"Binary Class {target_class} ({class_name}) - {k_shot}-Shot Results")
            logger.info(f"{'='*60}")
            logger.info(f"Unbalance Macro F1: {metrics_unbalance.get('group_plant_opt_macro_f1', 0.0):.4f}")
            logger.info(f"Unbalance Micro F1: {metrics_unbalance.get('group_plant_micro_f1', 0.0):.4f}")
            logger.info(f"BalanceB Macro F1: {metrics_balanceb.get('group_plant_opt_macro_f1', 0.0):.4f}")
            logger.info(f"BalanceB Micro F1: {metrics_balanceb.get('group_plant_micro_f1', 0.0):.4f}")

            # Log class-specific metrics
            class_f1 = metrics_unbalance.get(f'group_plant_class_{target_class}_opt_f1', 0.0)
            class_prec = metrics_unbalance.get(f'group_plant_class_{target_class}_opt_precision', 0.0)
            class_rec = metrics_unbalance.get(f'group_plant_class_{target_class}_opt_recall', 0.0)
            logger.info(f"Binary Class {target_class} - F1: {class_f1:.4f}, Prec: {class_prec:.4f}, Rec: {class_rec:.4f}")

            if tb_writer:
                tb_writer.add_scalar(f'binary_class_{target_class}/few_shot_macro_f1', metrics_unbalance.get('group_plant_opt_macro_f1', 0.0), k_shot)
                tb_writer.add_scalar(f'binary_class_{target_class}/few_shot_class_f1', class_f1, k_shot)

        # Print filtered results table for this class only
        print_single_class_few_shot_results(all_results[target_class], target_class, logger)

    # =========================================================================
    # Final Results Summary
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("FINAL PLANT vs. PLANT BINARY CLASSIFICATION RESULTS SUMMARY")
    logger.info("Positive: Plant (specific class) | Negative: Plant (other classes)")
    logger.info(f"{'='*80}")

    for target_class in target_classes:
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']
        logger.info(f"\n--- Binary Class {target_class} ({class_name}) Results ---")
        print_single_class_few_shot_results(all_results[target_class], target_class, logger)

    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    results_path = os.path.join(Config.checkpoint_dir, 'plant_fewshot_binary_results.json')

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

    # Reorganize results for better JSON structure
    serializable_results = {}
    for target_class, shot_results in all_results.items():
        class_name = PLANT_CLASS_MAPPING[target_class]['class_name']
        serializable_results[f"plant_vs_plant_class_{target_class}_{class_name}"] = {}
        for k_shot, metrics_dict in shot_results.items():
            # Handle both numeric shots and 'full' key
            if k_shot == 'full':
                shot_key = 'full'
            else:
                shot_key = f"{k_shot}_shot"
            serializable_results[f"plant_vs_plant_class_{target_class}_{class_name}"][shot_key] = convert_to_serializable(metrics_dict)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"Binary classification results saved to: {results_path}")
    logger.info(f"{'='*80}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant vs. Plant binary classification models (Plant pos, Plant neg, 1:1 balanced)")
    parser.add_argument('--config', type=str, default='json/plant_single.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to pre-trained checkpoint', default="logs/rna_classification_20260111_111223/checkpoints/epoch_010.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
