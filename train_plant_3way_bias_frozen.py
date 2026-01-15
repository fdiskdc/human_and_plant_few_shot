"""
Plant 3-Way Multi-class Classification Training Script - Bias-Frozen Linear Probing

This script implements BIAS-FROZEN LINEAR PROBING for Plant 3-Way classification.

Key Innovation:
--------------
Standard fine-tuning fails because training on balanced support sets destroys the
Class Prior (Bias) learned during pre-training on imbalanced data.

SOLUTION: Fine-tune the linear decision boundary (Weights) to adapt to new data,
while STRICTLY PRESERVING the class imbalance prior (Bias) from pre-training.

Strategy:
--------
1. Freeze Backbone: All CNN/GCN parameters frozen
2. Freeze Head Bias: ALL bias terms in classification head frozen
3. Train Head Weights: Only weight matrices in final projection layers unfrozen

This preserves the pre-trained class priors while allowing feature boundary adaptation.

Plant Classes: 5 (Y), 8 (m5C), 9 (m6A)
Label Mapping: Class 5 -> 0, Class 8 -> 1, Class 9 -> 2
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
    roc_auc_score, average_precision_score, confusion_matrix
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
# Plant Class to Label Mapping for 3-Way Classification
# ============================================================================
PLANT_CLASS_MAPPING = {
    5: {'class_name': 'Y', 'group_idx': 3, 'group_name': 'U', '3way_label': 0},
    8: {'class_name': 'm5C', 'group_idx': 1, 'group_name': 'C', '3way_label': 1},
    9: {'class_name': 'm6A', 'group_idx': 0, 'group_name': 'A', '3way_label': 2}
}


class PrunedModelWrapper(nn.Module):
    """
    Wrapper for the pruned 3-way model to interface with existing evaluation functions.
    It pads the pruned 3-dim output back to 12-dim (filling others with very small logits).
    """
    def __init__(self, pruned_model, class_indices, group_indices):
        super().__init__()
        self.model = pruned_model
        self.class_indices = class_indices  # [5, 8, 9]
        self.group_indices = group_indices  # [3, 1, 0]
        self.use_hierarchical = pruned_model.use_hierarchical

    def forward(self, x, edge_index, batch=None):
        device = x.device

        # Get pruned output (Batch, 3) or (Batch, 3), (Batch, 3)
        out = self.model(x, edge_index, batch)

        if self.use_hierarchical:
            logits_1, logits_4_1 = out
            batch_size = logits_1.size(0)

            # Pad 12-class logits
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_indices] = logits_1

            # Pad 4-class logits
            padded_logits_4 = torch.full((batch_size, 4), -1e9, device=device)
            padded_logits_4[:, self.group_indices] = logits_4_1

            return padded_logits_12, padded_logits_4
        else:
            logits_1 = out
            batch_size = logits_1.size(0)

            # Pad 12-class logits
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_indices] = logits_1

            return padded_logits_12


def map_labels_3way(y_onehot, class_to_label):
    """
    Map one-hot encoded labels to 3-way classification labels.

    Args:
        y_onehot: One-hot encoded labels (N, 12)
        class_to_label: Mapping dict {5: 0, 8: 1, 9: 2}

    Returns:
        Tensor of shape (N,) with values 0, 1, or 2
    """
    raw_classes = y_onehot.argmax(dim=1)

    # Map to 0, 1, 2
    target_labels = torch.zeros_like(raw_classes)
    for original_cls, target_lbl in class_to_label.items():
        mask = (raw_classes == original_cls)
        target_labels[mask] = target_lbl

    return target_labels


def compute_3way_metrics(y_true, y_pred):
    """
    Compute accuracy and F1 for 3-way multi-class classification.

    Args:
        y_true: True labels (0, 1, 2)
        y_pred: Predicted labels (0, 1, 2)

    Returns:
        Dictionary containing accuracy, macro_f1, per_class_f1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)

    # Per-class F1 scores
    num_classes = 3
    per_class_f1 = []

    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        per_class_f1.append(f1)

    macro_f1 = np.mean(per_class_f1)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_0_f1': per_class_f1[0],  # Class 5 (Y)
        'class_1_f1': per_class_f1[1],  # Class 8 (m5C)
        'class_2_f1': per_class_f1[2]   # Class 9 (m6A)
    }


def calculate_binary_metrics(y_true, y_pred, y_prob):
    """
    Calculate detailed binary metrics for One-vs-Rest evaluation.
    y_true: 0 or 1
    y_pred: 0 or 1
    y_prob: float (probability of positive class)
    """
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
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }


def sample_support_set_3way(dataset, plant_train_indices, k, target_classes):
    """
    Sample support set for 3-way multi-class classification.

    Samples K shots for EACH of the 3 classes.
    Total samples = 3*K (K from each class).

    Args:
        dataset: PlantSingleDataset instance containing Plant data
        plant_train_indices: List of available Plant training indices
        k: Number of shots per class (total samples = 3*k)
        target_classes: List of target class IDs [5, 8, 9]

    Returns:
        list: Support indices (balanced, total 3*k samples)
    """
    if k == 0 or k == '0':
        return []

    support_indices = []

    for cls_idx in target_classes:
        # Get indices for this specific class
        c_indices = dataset.get_plant_indices_by_class(cls_idx)
        valid_c_indices = list(set(c_indices) & set(plant_train_indices))

        # Sample K items from this class
        if len(valid_c_indices) >= k:
            selected_from_class = np.random.choice(valid_c_indices, k, replace=False).tolist()
        else:
            # If not enough samples, sample with replacement
            selected_from_class = np.random.choice(valid_c_indices, k, replace=True).tolist()
            print(f"Warning: Only {len(valid_c_indices)} samples available for class {cls_idx}, using replacement for {k} samples")

        support_indices.extend(selected_from_class)

    # Shuffle to mix classes
    np.random.shuffle(support_indices)

    return support_indices


def apply_bias_frozen_strategy(model, logger):
    """
    Apply BIAS-FROZEN LINEAR PROBING strategy.

    This is the KEY FIX for preserving class priors from pre-training:

    Strategy:
    ---------
    1. Freeze Backbone: All CNN/GCN parameters frozen
    2. Freeze Head Bias: ALL bias terms in classification head frozen
    3. Train Head Weights: Only weight matrices in final projection layers unfrozen

    Why this works:
    --------------
    - The Bias terms encode the class prior P(y) learned from imbalanced pre-training data
    - The Weight matrices encode the feature boundaries P(x|y)
    - By freezing bias, we preserve the pre-trained class distribution prior
    - By training weights, we adapt the feature boundaries to new few-shot data

    Args:
        model: The RNA_ClassQuery_Model
        logger: Logger instance

    Returns:
        list: Trainable parameters (only head weights)
    """
    logger.info(f"\n{'='*80}")
    logger.info("BIAS-FROZEN LINEAR PROBING STRATEGY")
    logger.info(f"{'='*80}")

    # Step 1: Freeze EVERYTHING first
    for param in model.parameters():
        param.requires_grad = False

    frozen_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Step 1: Frozen ALL parameters ({frozen_count:,} params)")

    # Step 2: Selectively unfreeze ONLY the WEIGHTS in the classification head
    # DO NOT unfreeze any bias parameters

    trainable_params = []
    unfrozen_weight_count = 0
    frozen_bias_count = 0

    # Navigate to the classification head
    head = model.class_query_head

    if hasattr(head, 'output_proj'):
        # Standard ClassQueryHead or ClassQueryHeadPooling
        # output_proj is a Sequential: LayerNorm -> Linear -> (optional layers)
        for name, param in head.output_proj.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj.{name} ({param.numel():,} params)")
            elif 'bias' in name:
                # Keep bias frozen to preserve class priors
                param.requires_grad = False
                frozen_bias_count += param.numel()
                logger.info(f"  ✗ FROZEN:  class_query_head.output_proj.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

    # Handle HierarchicalClassQueryHeadPooling (has two output projections)
    if hasattr(head, 'output_proj_12class'):
        for name, param in head.output_proj_12class.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_12class.{name} ({param.numel():,} params)")
            elif 'bias' in name:
                param.requires_grad = False
                frozen_bias_count += param.numel()
                logger.info(f"  ✗ FROZEN:  class_query_head.output_proj_12class.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

    if hasattr(head, 'output_proj_4class'):
        for name, param in head.output_proj_4class.named_parameters():
            if 'weight' in name:
                param.requires_grad = True
                trainable_params.append(param)
                unfrozen_weight_count += param.numel()
                logger.info(f"  ✓ UNFROZEN: class_query_head.output_proj_4class.{name} ({param.numel():,} params)")
            elif 'bias' in name:
                param.requires_grad = False
                frozen_bias_count += param.numel()
                logger.info(f"  ✗ FROZEN:  class_query_head.output_proj_4class.{name} ({param.numel():,} params) [PRESERVES PRIOR]")

    # Also unfreeze class queries if present (these are not bias/weight in traditional sense)
    if hasattr(head, 'class_queries'):
        head.class_queries.requires_grad = True
        trainable_params.append(head.class_queries)
        logger.info(f"  ✓ UNFROZEN: class_query_head.class_queries ({head.class_queries.numel():,} params)")

    if hasattr(head, 'group_queries'):
        head.group_queries.requires_grad = True
        trainable_params.append(head.group_queries)
        logger.info(f"  ✓ UNFROZEN: class_query_head.group_queries ({head.group_queries.numel():,} params)")

    total_trainable = sum(p.numel() for p in trainable_params)

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY:")
    logger.info(f"  Total Parameters:     {frozen_count + unfrozen_weight_count:,}")
    logger.info(f"  Frozen Parameters:    {frozen_count:,} ({100*frozen_count/(frozen_count+unfrozen_weight_count):.2f}%)")
    logger.info(f"  Trainable Weights:    {unfrozen_weight_count:,} ({100*unfrozen_weight_count/(frozen_count+unfrozen_weight_count):.2f}%)")
    logger.info(f"  Frozen Bias Terms:    {frozen_bias_count:,} [PRESERVING CLASS PRIORS]")
    logger.info(f"{'='*80}\n")

    return trainable_params


def train_few_shot_3way_bias_frozen(model, support_data_list, target_classes, group_indices,
                                   config, device, logger, k_shot):
    """
    Train a single model for 3-WAY multi-class classification using BIAS-FROZEN LINEAR PROBING.

    Key Innovation:
    --------------
    - Freezes ALL bias terms to preserve class priors from pre-training
    - Only trains weight matrices in final projection layers
    - Uses CrossEntropyLoss with label smoothing (NO class weights)
    - Optional L2-SP regularization to prevent drift from pre-trained solution

    Args:
        model: The RNA model (to be pruned to 3 classes)
        support_data_list: List of support samples (from all 3 classes)
        target_classes: The target class indices [5, 8, 9]
        group_indices: Corresponding group indices [3, 1, 0]
        config: Configuration object
        device: Device to train on
        logger: Logger instance
        k_shot: Number of shots per class (total samples = 3*k_shot)

    Returns:
        Trained model
    """
    # Save original state for L2-SP regularization
    original_head_state = copy.deepcopy(model.class_query_head.state_dict())

    # Create label mapping
    class_to_label = {5: 0, 8: 1, 9: 2}

    # ========================================================================
    # STEP 1: Apply BIAS-FROZEN LINEAR PROBING Strategy
    # ========================================================================
    trainable_params = apply_bias_frozen_strategy(model, logger)

    # ========================================================================
    # STEP 2: Training Hyperparameters
    # ========================================================================
    # Moderate LR - we're restricting parameter space so can use higher LR
    ft_lr = 1e-4

    # Dynamic epochs based on shot count
    if k_shot <= 10:
        ft_epochs = 30
    else:
        ft_epochs = 50

    aug_factor = 10

    optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-3)

    # ========================================================================
    # STEP 3: Loss Function - CrossEntropyLoss with Label Smoothing
    # ========================================================================
    # IMPORTANT: NO class_weight argument!
    # We want the model to learn the decision boundary, NOT the class distribution
    # The class distribution (prior) is preserved in the FROZEN bias terms
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    logger.info(f"Training Configuration:")
    logger.info(f"  Loss: CrossEntropyLoss(label_smoothing=0.1)")
    logger.info(f"  Learning Rate: {ft_lr}")
    logger.info(f"  Epochs: {ft_epochs}")
    logger.info(f"  Augmentation Factor: {aug_factor}")
    logger.info(f"  Optimizer: AdamW")

    # -------------------------------------------------------------------------
    # DYNAMIC AUGMENTATION: Initialize Prefetcher
    # -------------------------------------------------------------------------
    prefetcher = AugmentationPrefetcher(
        support_data_list,
        [],  # No masks for 3-way classification
        aug_factor,
        k_shot,
        buffer_size=5
    )

    model.train()
    mini_batch_size = 64

    # ========================================================================
    # STEP 4: Training Loop with L2-SP Regularization
    # ========================================================================
    # L2-SP (L2 regularization with Starting Point)
    # Prevents weights from drifting too far from the working 0-shot solution
    l2_sp_lambda = 0.01  # L2-SP regularization strength

    logger.info(f"\nStarting training loop...")
    logger.info(f"  L2-SP Regularization: {l2_sp_lambda} (prevents drift from 0-shot)")

    try:
        for epoch in range(ft_epochs):
            # Get fresh augmented data for this epoch
            batch_list = prefetcher.get_next_epoch_data()

            epoch_loss = 0.0
            epoch_ce_loss = 0.0
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

                # Label Mapping: Convert (N, 12) to (N,) with values 0,1,2
                target_labels = map_labels_3way(batch.y, class_to_label)

                # Forward pass
                if config.use_hierarchical:
                    logits_class, logits_group = model(batch.x, batch.edge_index, batch.batch)
                    # Use only class-level logits for 3-way classification
                    ce_loss = criterion(logits_class, target_labels)
                else:
                    logits_class = model(batch.x, batch.edge_index, batch.batch)
                    ce_loss = criterion(logits_class, target_labels)

                # L2-SP Regularization: Penalize deviation from initial weights
                l2sp_loss = torch.tensor(0.0, device=device)
                current_head_state = model.class_query_head.state_dict()
                for name in current_head_state:
                    if name in original_head_state and current_head_state[name].requires_grad:
                        # Only penalize trainable parameters
                        l2sp_loss += torch.sum((current_head_state[name] - original_head_state[name]) ** 2)

                # Total loss
                loss = ce_loss + l2_sp_lambda * l2sp_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_l2sp_loss += l2sp_loss.item()
                batches_processed += 1

            if batches_processed > 0 and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / batches_processed
                avg_ce = epoch_ce_loss / batches_processed
                avg_l2sp = epoch_l2sp_loss / batches_processed
                logger.info(f"Shot {k_shot} Ep {epoch+1}/{ft_epochs}: "
                          f"Total={avg_loss:.4f}, CE={avg_ce:.4f}, L2-SP={l2_sp_lambda}*{avg_l2sp:.4f}")

    finally:
        # Ensure thread cleanup
        prefetcher.cleanup()

    logger.info(f"Training completed for {k_shot}-shot")

    return model


def evaluate_3way_detailed(model, test_loader, device, target_classes, use_hierarchical=True):
    """
    Run inference and compute One-vs-Rest metrics for all 3 classes.

    Args:
        model: The pruned 3-way model
        test_loader: DataLoader for test set
        device: Device to evaluate on
        target_classes: List of target class indices [5, 8, 9]
        use_hierarchical: Whether model uses hierarchical classification

    Returns:
        Dictionary containing One-vs-Rest metrics for each class {class_id: {metrics_dict}}
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    # Mappings
    # 0 -> Class 5, 1 -> Class 8, 2 -> Class 9
    class_to_label = {5: 0, 8: 1, 9: 2}

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward (Assuming model outputs 3-class logits)
            if use_hierarchical:
                out = model(batch.x, batch.edge_index, batch.batch)
                if isinstance(out, tuple):
                    logits_class = out[0]
                else:
                    logits_class = out
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
                if isinstance(out, tuple):
                    logits_class = out[0]
                else:
                    logits_class = out

            # Softmax to get probabilities
            probs = torch.softmax(logits_class, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Map labels (Ensure your loader returns 0,1,2 or map them here)
            # For now, assuming batch.y is (N, 12), map it:
            raw = batch.y.argmax(dim=1)
            target_labels = torch.zeros_like(raw)
            for k, v in class_to_label.items():
                target_labels[raw == k] = v

            # Filter out samples that don't belong to our 3 classes
            valid_mask = batch.y[:, target_classes].sum(dim=1) > 0

            all_probs.append(probs[valid_mask].cpu())
            all_preds.append(preds[valid_mask].cpu())
            all_labels.append(target_labels[valid_mask].cpu())

    all_probs = torch.cat(all_probs).numpy() # Shape (N, 3)
    all_preds = torch.cat(all_preds).numpy() # Shape (N,)
    all_labels = torch.cat(all_labels).numpy() # Shape (N,)

    results = {}

    # Loop over classes 0, 1, 2 (corresponding to 5, 8, 9)
    map_idx_to_real = {0: 5, 1: 8, 2: 9}

    for idx in range(3):
        real_cls = map_idx_to_real[idx]

        # Create One-vs-Rest Binary targets
        y_bin_true = (all_labels == idx).astype(int)
        y_bin_pred = (all_preds == idx).astype(int)
        y_bin_prob = all_probs[:, idx]

        metrics = calculate_binary_metrics(y_bin_true, y_bin_pred, y_bin_prob)
        results[real_cls] = metrics

    return results


def print_detailed_3way_results(all_results_history, logger):
    """
    Print separate tables for Class 5, 8, and 9.
    all_results_history structure: { k_shot: { class_id: {metric_dict} } }

    Args:
        all_results_history: Results dictionary for all shot counts
        logger: Logger instance
    """
    class_info = {
        5: "Class 5 (Y)",
        8: "Class 8 (m5C)",
        9: "Class 9 (m6A)"
    }

    shot_keys = sorted(all_results_history.keys(), key=lambda x: int(x) if x!='full' else 999)

    for cls_id in [5, 8, 9]:
        logger.info(f"\n{'='*100}")
        logger.info(f"Detailed Results: {class_info[cls_id]}")
        logger.info(f"{'='*100}")

        table = PrettyTable()
        table.field_names = ["Shot", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
        table.align = "r"

        for k in shot_keys:
            if cls_id not in all_results_history[k]:
                continue

            m = all_results_history[k][cls_id]

            table.add_row([
                k,
                f"{m['F1']:.4f}", f"{m['Prec']:.4f}", f"{m['Rec']:.4f}", f"{m['Acc']:.4f}",
                f"{m['AUC']:.4f}", f"{m['AUPRC']:.4f}",
                f"{m['Sn']:.4f}", f"{m['Sp']:.4f}",
                m['TP'], m['TN'], m['FP'], m['FN']
            ])

        logger.info(f"\n{table}")


def main(config_path='json/plant_single.json', checkpoint_path=None):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name + '_3way_bias_frozen')
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_3way_bias_frozen')

    logger.info(f"\n{'='*80}")
    logger.info("Plant 3-Way Multi-class Classification - BIAS-FROZEN LINEAR PROBING")
    logger.info(f"{'='*80}")
    logger.info("Strategy:")
    logger.info("  - Freeze Backbone (CNN/GCN)")
    logger.info("  - Freeze Head Bias (PRESERVES class priors from pre-training)")
    logger.info("  - Train Head Weights (adapts feature boundaries)")
    logger.info("  - CrossEntropyLoss with label smoothing (NO class weights)")
    logger.info("  - L2-SP regularization (prevents drift from 0-shot)")
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

    # Target classes for 3-way classification
    target_classes = [5, 8, 9]
    group_indices = [3, 1, 0]  # Corresponding group indices

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
    logger.info(f"Test Set: Plant Only (Classes {target_classes})")

    # Load checkpoint for base model
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error("Checkpoint path is required for 3-way classification training!")
        return

    # Results storage
    all_results = {}

    # shot_counts = [0, 1, 3, 5, 7, 10, 50, 100]
    shot_counts = [0, 2, 4, 6, 8, 10, 50, 100]

    # ========================================================================
    # CREATE ONE SINGLE MODEL FOR 3-WAY CLASSIFICATION
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"{'='*80}")
    logger.info(f"TRAINING 3-WAY MODEL FOR CLASSES {target_classes}")
    logger.info(f"  Class 5 (Y) -> Group U (index 3)")
    logger.info(f"  Class 8 (m5C) -> Group C (index 1)")
    logger.info(f"  Class 9 (m6A) -> Group A (index 0)")
    logger.info(f"  Strategy: BIAS-FROZEN LINEAR PROBING")
    logger.info(f"{'='*80}")
    logger.info(f"{'='*80}\n")

    # Create model
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

    # Save original 12-class state
    original_state_dict = copy.deepcopy(model.state_dict())

    # ====================================================================
    # Loop over shot counts for 3-way classification
    # ====================================================================
    for k_shot in shot_counts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running {k_shot}-Shot Learning with BIAS-FROZEN LINEAR PROBING")
        logger.info(f"{'='*80}\n")

        # Reset model to original 12-class state
        model.load_state_dict(original_state_dict, strict=False)

        # Clear any pruning buffers from previous iterations
        if hasattr(model.class_query_head, 'valid_class_indices'):
            del model.class_query_head.valid_class_indices
        if hasattr(model.class_query_head, 'valid_group_indices'):
            del model.class_query_head.valid_group_indices

        if k_shot == 0:
            # Zero-shot: Just prune and evaluate
            logger.info(f"Zero-shot: Pruning to classes {target_classes}")
            model.prune_heads(target_classes, group_indices)
        else:
            # 3-WAY Sampling: k_shot samples from EACH class (total = 3*k_shot)
            np.random.seed(Config.random_seed + k_shot)
            support_indices = sample_support_set_3way(
                full_dataset, plant_train_indices, k_shot, target_classes
            )

            if len(support_indices) == 0:
                logger.warning(f"No support samples found at {k_shot}-shot!")
                continue

            support_data_list = [full_dataset[i] for i in support_indices]

            # Count samples per class for logging
            class_counts = {5: 0, 8: 0, 9: 0}
            for i in support_indices:
                if i < full_dataset.num_plant:
                    for c in target_classes:
                        if full_dataset.plant_y12[i, c] == 1:
                            class_counts[c] += 1

            logger.info(f"3-Way Sampling: {len(support_indices)} total samples "
                        f"(Class 5: {class_counts[5]}, Class 8: {class_counts[8]}, Class 9: {class_counts[9]})")

            # Train with BIAS-FROZEN LINEAR PROBING
            logger.info(f"\nApplying BIAS-FROZEN LINEAR PROBING...")
            model = train_few_shot_3way_bias_frozen(
                model, support_data_list, target_classes, group_indices,
                Config, Config.device, logger, k_shot
            )
            logger.info(f"Completed {k_shot}-shot bias-frozen training")

            # Prune model after training for evaluation
            logger.info(f"Pruning to classes {target_classes} after {k_shot}-shot training")
            model.prune_heads(target_classes, group_indices)

        # Wrap model for evaluation
        wrapped_model = PrunedModelWrapper(model, class_indices=target_classes, group_indices=group_indices)

        # Evaluate with detailed One-vs-Rest metrics
        metrics = evaluate_3way_detailed(
            model, test_loader, Config.device, target_classes,
            getattr(Config, 'use_hierarchical', True)
        )

        # Store results in new structure: {k_shot: {5: metrics, 8: metrics, 9: metrics}}
        all_results[k_shot] = metrics

        # Log results
        logger.info(f"\n{'='*80}")
        logger.info(f"3-Way Classification - {k_shot}-Shot Results (BIAS-FROZEN)")
        logger.info(f"{'='*80}")
        for cls_id in [5, 8, 9]:
            cls_metrics = metrics[cls_id]
            cls_name = PLANT_CLASS_MAPPING[cls_id]['class_name']
            logger.info(f"Class {cls_id} ({cls_name}) | F1: {cls_metrics['F1']:.4f}, "
                       f"Prec: {cls_metrics['Prec']:.4f}, Rec: {cls_metrics['Rec']:.4f}, "
                       f"Acc: {cls_metrics['Acc']:.4f}, AUC: {cls_metrics['AUC']:.4f}, "
                       f"AUPRC: {cls_metrics['AUPRC']:.4f}")

        if tb_writer:
            for cls_id in [5, 8, 9]:
                cls_metrics = metrics[cls_id]
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_F1', cls_metrics['F1'], k_shot)
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_Prec', cls_metrics['Prec'], k_shot)
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_Rec', cls_metrics['Rec'], k_shot)
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_Acc', cls_metrics['Acc'], k_shot)
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_AUC', cls_metrics['AUC'], k_shot)
                tb_writer.add_scalar(f'3way_biasfrozen/class_{cls_id}_AUPRC', cls_metrics['AUPRC'], k_shot)

    # Print final results tables (one per class)
    print_detailed_3way_results(all_results, logger)

    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    results_path = os.path.join(Config.checkpoint_dir, 'plant_3way_bias_frozen_results.json')

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
    for k_shot, metrics in all_results.items():
        if k_shot == 'full':
            shot_key = 'full'
        else:
            shot_key = f"{k_shot}_shot"
        serializable_results[shot_key] = convert_to_serializable(metrics)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"3-Way BIAS-FROZEN results saved to: {results_path}")
    logger.info(f"{'='*80}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Plant 3-way multi-class classification using BIAS-FROZEN LINEAR PROBING"
    )
    parser.add_argument('--config', type=str, default='json/plant_single.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to pre-trained checkpoint',
                       default="logs/rna_classification_20260111_111223/checkpoints/epoch_010.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
