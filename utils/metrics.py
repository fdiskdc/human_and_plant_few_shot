"""
Evaluation metrics and functions for RNA Multi-label Classification

This module contains:
- Metric calculation functions
- Optimal threshold finding
- Evaluation functions (unbalance, balanceb, plant evaluations)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple

# Import constants from common
from utils.common import MOD_NAMES, GROUP_TO_INDEX, INDEX_TO_NUCLEOTIDE, GROUP_TO_CLASS_INDICES, INDEX_TO_GROUP


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximizes F1 score for binary classification.

    Args:
        y_true: Ground truth labels (N,)
        y_prob: Predicted probabilities (N,)

    Returns:
        Optimal threshold value
    """
    # Only search if we have both classes
    if len(np.unique(y_true)) < 2:
        return 0.5

    best_threshold = 0.5
    best_f1 = 0.0

    # Search thresholds from 0.05 to 0.95 with step 0.01
    thresholds = np.arange(0.05, 1.0, 0.01)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def calculate_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification.

    Args:
        y_true: Ground truth labels, shape (N, C)
        y_pred: Binary predictions, shape (N, C)
        y_prob: Predicted probabilities, shape (N, C)

    Returns:
        Dictionary of metrics with 'group_' prefix
    """
    metrics = {}
    N, C = y_true.shape

    # Convert to numpy for sklearn
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    y_prob_np = y_prob.cpu().numpy() if isinstance(y_prob, torch.Tensor) else y_prob

    # Calculate metrics for each average type (macro, micro, weighted)
    for avg in ['macro', 'micro', 'weighted']:
        metrics[f'group_{avg}_f1'] = f1_score(y_true_np, y_pred_np, average=avg, zero_division=0)
        metrics[f'group_{avg}_precision'] = precision_score(y_true_np, y_pred_np, average=avg, zero_division=0)
        metrics[f'group_{avg}_recall'] = recall_score(y_true_np, y_pred_np, average=avg, zero_division=0)

    # Calculate accuracy (average-independent)
    metrics['group_accuracy'] = accuracy_score(y_true_np, y_pred_np)

    # Per-class metrics - with 'group_' prefix
    for c in range(C):
        y_true_c = y_true_np[:, c]
        y_pred_c = y_pred_np[:, c]
        y_prob_c = y_prob_np[:, c]

        # Confusion matrix values
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_class_{c}_tp'] = tp
        metrics[f'group_class_{c}_tn'] = tn
        metrics[f'group_class_{c}_fp'] = fp
        metrics[f'group_class_{c}_fn'] = fn

        # Basic metrics
        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        # Sensitivity (same as recall) and Specificity
        sensitivity = recall
        specificity = tn / (tn + fp + eps)

        # Matthews Correlation Coefficient
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        mcc = numerator / (denominator + eps)

        metrics[f'group_class_{c}_precision'] = precision
        metrics[f'group_class_{c}_recall'] = recall
        metrics[f'group_class_{c}_f1'] = f1
        metrics[f'group_class_{c}_accuracy'] = accuracy
        metrics[f'group_class_{c}_sensitivity'] = sensitivity
        metrics[f'group_class_{c}_specificity'] = specificity
        metrics[f'group_class_{c}_mcc'] = mcc

        # AUC and AUPRC (only if both classes exist)
        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_class_{c}_auc'] = 0.0
                metrics[f'group_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_class_{c}_auc'] = 0.0
            metrics[f'group_class_{c}_auprc'] = 0.0

    return metrics


def evaluate_unbalance(y_true: np.ndarray, y_prob: np.ndarray, device, 
                      y_4class: Optional[np.ndarray] = None,
                      random_seed: int = 42, 
                      y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate using unbalanced dataset: real distribution testing.

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1
    - Negative set: all samples where y_true[:, c] == 0
    - Calculate metrics on this unbalanced dataset

    Args:
        y_true: Ground truth labels (N, num_classes)
        y_prob: Predicted probabilities (N, num_classes)
        device: Device to run evaluation on
        y_4class: 4-class ground truth labels (N, 4)
        random_seed: Random seed for reproducibility
        y_4prob: 4-class predicted probabilities (N, 4)

    Returns:
        Dictionary of metrics for unbalanced evaluation
    """
    np.random.seed(random_seed)
    N, C = y_true.shape
    metrics = {}

    # For each class, calculate metrics on unbalanced dataset
    for c in range(C):
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        # Find optimal threshold
        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_class_{c}_opt_threshold'] = opt_threshold

        # Calculate predictions with optimal threshold
        y_pred_c = (y_prob_c >= opt_threshold).astype(int)

        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true_c, y_pred_c).ravel()

        metrics[f'group_class_{c}_opt_tp'] = tp
        metrics[f'group_class_{c}_opt_tn'] = tn
        metrics[f'group_class_{c}_opt_fp'] = fp
        metrics[f'group_class_{c}_opt_fn'] = fn

        # Calculate metrics with optimal threshold
        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        sensitivity = recall
        specificity = tn / (tn + fp + eps)

        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        mcc = numerator / (denominator + eps)

        metrics[f'group_class_{c}_opt_f1'] = f1
        metrics[f'group_class_{c}_opt_precision'] = precision
        metrics[f'group_class_{c}_opt_recall'] = recall
        metrics[f'group_class_{c}_opt_accuracy'] = accuracy
        metrics[f'group_class_{c}_opt_sensitivity'] = sensitivity
        metrics[f'group_class_{c}_opt_specificity'] = specificity
        metrics[f'group_class_{c}_opt_mcc'] = mcc

        # AUC and AUPRC (threshold-independent)
        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_class_{c}_auc'] = 0.0
                metrics[f'group_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_class_{c}_auc'] = 0.0
            metrics[f'group_class_{c}_auprc'] = 0.0

    # Calculate macro averages
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_class_{c}_opt_recall'])

    metrics['group_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_opt_macro_recall'] = np.mean(all_rec)

    # Add standard macro/micro/weighted metrics (without _opt_ suffix) for logging compatibility
    metrics['group_macro_f1'] = metrics['group_opt_macro_f1']
    metrics['group_macro_precision'] = metrics['group_opt_macro_precision']
    metrics['group_macro_recall'] = metrics['group_opt_macro_recall']
    metrics['group_micro_f1'] = metrics['group_opt_macro_f1']
    metrics['group_micro_precision'] = metrics['group_opt_macro_precision']
    metrics['group_micro_recall'] = metrics['group_opt_macro_recall']
    metrics['group_weighted_f1'] = metrics['group_opt_macro_f1']
    metrics['group_weighted_precision'] = metrics['group_opt_macro_precision']
    metrics['group_weighted_recall'] = metrics['group_opt_macro_recall']

    # 4-class evaluation (if hierarchical)
    if y_4prob is not None and y_4class is not None:
        for c in range(4):
            y_true_c = y_4class[:, c]
            y_prob_c = y_4prob[:, c]

            opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
            metrics[f'group_4class_{c}_opt_threshold'] = opt_threshold

            y_pred_c = (y_prob_c >= opt_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_c, y_pred_c).ravel()

            eps = 1e-10
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
            sensitivity = recall
            specificity = tn / (tn + fp + eps)
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
            mcc = numerator / (denominator + eps)

            metrics[f'group_4class_{c}_opt_f1'] = f1
            metrics[f'group_4class_{c}_opt_precision'] = precision
            metrics[f'group_4class_{c}_opt_recall'] = recall
            metrics[f'group_4class_{c}_opt_accuracy'] = accuracy
            metrics[f'group_4class_{c}_opt_sensitivity'] = sensitivity
            metrics[f'group_4class_{c}_opt_specificity'] = specificity
            metrics[f'group_4class_{c}_opt_mcc'] = mcc
            metrics[f'group_4class_{c}_opt_tp'] = tp
            metrics[f'group_4class_{c}_opt_tn'] = tn
            metrics[f'group_4class_{c}_opt_fp'] = fp
            metrics[f'group_4class_{c}_opt_fn'] = fn

            if len(np.unique(y_true_c)) > 1:
                try:
                    metrics[f'group_4class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                    metrics[f'group_4class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
                except ValueError:
                    metrics[f'group_4class_{c}_auc'] = 0.0
                    metrics[f'group_4class_{c}_auprc'] = 0.0
            else:
                metrics[f'group_4class_{c}_auc'] = 0.0
                metrics[f'group_4class_{c}_auprc'] = 0.0

        all_f1_4, all_prec_4, all_rec_4 = [], [], []
        for c in range(4):
            all_f1_4.append(metrics[f'group_4class_{c}_opt_f1'])
            all_prec_4.append(metrics[f'group_4class_{c}_opt_precision'])
            all_rec_4.append(metrics[f'group_4class_{c}_opt_recall'])

        metrics['group_4class_opt_macro_f1'] = np.mean(all_f1_4)
        metrics['group_4class_opt_macro_precision'] = np.mean(all_prec_4)
        metrics['group_4class_opt_macro_recall'] = np.mean(all_rec_4)

    return metrics


def evaluate_balanceb(y_true: np.ndarray, y_prob: np.ndarray, y_4class: np.ndarray,
                     device, random_seed: int = 42, 
                     y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate using balanced dataset: 1:1 positive:negative ratio.

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1
    - Negative set: randomly sampled from samples where y_true[:, c] == 0
    - Downsampling: min(|P|, |N|)

    Args:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        y_4class: 4-class ground truth labels (N, 4)
        device: Device to run evaluation on
        random_seed: Random seed for sampling
        y_4prob: 4-class predicted probabilities (N, 4)

    Returns:
        Dictionary of metrics for balanced evaluation
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}

    # 12-Class BalanceB Evaluation
    for c in range(C):
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx_all = np.where(y_true[:, c] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            for key in ['precision', 'recall', 'f1', 'accuracy', 'sensitivity',
                       'specificity', 'mcc', 'auc', 'auprc', 'tp', 'tn', 'fp', 'fn']:
                metrics[f'group_class_{c}_{key}'] = 0.0
            for key in ['opt_precision', 'opt_recall', 'opt_f1', 'opt_accuracy', 'opt_sensitivity',
                       'opt_specificity', 'opt_mcc', 'opt_auc', 'opt_auprc', 'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn',
                       'opt_threshold']:
                metrics[f'group_class_{c}_{key}'] = 0.0
            continue

        target_count = min(n_pos, n_neg_all)
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_c = y_true[selected_idx, c]
        y_prob_c = y_prob[selected_idx, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_class_{c}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_class_{c}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_class_{c}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_class_{c}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_class_{c}_opt_sensitivity'] = metrics[f'group_class_{c}_opt_recall']
        metrics[f'group_class_{c}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_class_{c}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_class_{c}_opt_tp'] = tp
        metrics[f'group_class_{c}_opt_tn'] = tn
        metrics[f'group_class_{c}_opt_fp'] = fp
        metrics[f'group_class_{c}_opt_fn'] = fn

        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_class_{c}_auc'] = 0.0
                metrics[f'group_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_class_{c}_auc'] = 0.0
            metrics[f'group_class_{c}_auprc'] = 0.0

    # 4-Class BalanceB Evaluation
    for g in range(4):
        pos_idx = np.where(y_4class[:, g] == 1)[0]
        neg_idx_all = np.where(y_4class[:, g] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            for key in ['precision', 'recall', 'f1', 'accuracy', 'sensitivity',
                       'specificity', 'mcc', 'auc', 'auprc', 'tp', 'tn', 'fp', 'fn',
                       'opt_precision', 'opt_recall', 'opt_f1', 'opt_accuracy', 'opt_sensitivity',
                       'opt_specificity', 'opt_mcc', 'opt_auc', 'opt_auprc', 'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn',
                       'opt_threshold']:
                metrics[f'group_4class_{g}_{key}'] = 0.0
            continue

        target_count = min(n_pos, n_neg_all)
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_g = y_4class[selected_idx, g]
        
        if y_4prob is not None:
            y_prob_g = y_4prob[selected_idx, g]
        else:
            # Use INDEX_TO_GROUP to convert integer index to group name
            group_name = INDEX_TO_GROUP[g]
            class_indices = GROUP_TO_CLASS_INDICES[group_name]
            y_prob_g = y_prob[selected_idx][:, class_indices].max(axis=1)

        opt_threshold = find_optimal_threshold(y_true_g, y_prob_g)
        metrics[f'group_4class_{g}_opt_threshold'] = opt_threshold

        y_pred_g = (y_prob_g >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_g == 1) & (y_pred_g == 1))
        tn = np.sum((y_true_g == 0) & (y_pred_g == 0))
        fp = np.sum((y_true_g == 0) & (y_pred_g == 1))
        fn = np.sum((y_true_g == 1) & (y_pred_g == 0))

        metrics[f'group_4class_{g}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_4class_{g}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_4class_{g}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_4class_{g}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_4class_{g}_opt_sensitivity'] = metrics[f'group_4class_{g}_opt_recall']
        metrics[f'group_4class_{g}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_4class_{g}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_4class_{g}_opt_tp'] = tp
        metrics[f'group_4class_{g}_opt_tn'] = tn
        metrics[f'group_4class_{g}_opt_fp'] = fp
        metrics[f'group_4class_{g}_opt_fn'] = fn

        if len(np.unique(y_true_g)) > 1:
            try:
                metrics[f'group_4class_{g}_auc'] = roc_auc_score(y_true_g, y_prob_g)
                metrics[f'group_4class_{g}_auprc'] = average_precision_score(y_true_g, y_prob_g)
            except ValueError:
                metrics[f'group_4class_{g}_auc'] = 0.0
                metrics[f'group_4class_{g}_auprc'] = 0.0
        else:
            metrics[f'group_4class_{g}_auc'] = 0.0
            metrics[f'group_4class_{g}_auprc'] = 0.0

    # Calculate macro/micro/weighted averages
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_class_{c}_opt_recall'])

    metrics['group_macro_f1'] = np.mean(all_f1)
    metrics['group_macro_precision'] = np.mean(all_prec)
    metrics['group_macro_recall'] = np.mean(all_rec)
    metrics['group_micro_f1'] = metrics['group_macro_f1']
    metrics['group_micro_precision'] = metrics['group_macro_precision']
    metrics['group_micro_recall'] = metrics['group_macro_recall']
    metrics['group_weighted_f1'] = metrics['group_macro_f1']
    metrics['group_weighted_precision'] = metrics['group_macro_precision']
    metrics['group_weighted_recall'] = metrics['group_macro_recall']

    # 4-class macro averages
    all_f1_4, all_prec_4, all_rec_4 = [], [], []
    for g in range(4):
        opt_f1_key = f'group_4class_{g}_opt_f1'
        if opt_f1_key in metrics:
            all_f1_4.append(metrics[opt_f1_key])
            all_prec_4.append(metrics[f'group_4class_{g}_opt_precision'])
            all_rec_4.append(metrics[f'group_4class_{g}_opt_recall'])

    metrics['group_4class_macro_f1'] = np.mean(all_f1_4)
    metrics['group_4class_macro_precision'] = np.mean(all_prec_4)
    metrics['group_4class_macro_recall'] = np.mean(all_rec_4)
    metrics['group_4class_micro_f1'] = metrics['group_4class_macro_f1']
    metrics['group_4class_micro_precision'] = metrics['group_4class_macro_precision']
    metrics['group_4class_micro_recall'] = metrics['group_4class_macro_recall']

    return metrics


def evaluate_plant_unbalance(y_true: np.ndarray, y_prob: np.ndarray, device,
                             y_4class: Optional[np.ndarray] = None,
                             random_seed: int = 42,
                             y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate plant data in Unbalance mode: real distribution testing.
    Only evaluates classes with data: 5 (Y), 8 (m5C), 9 (m6A)

    Args:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        device: Device to run evaluation on
        y_4class: 4-class ground truth labels (N, 4)
        random_seed: Random seed for reproducibility
        y_4prob: 4-class predicted probabilities (N, 4)

    Returns:
        Dictionary of metrics with "group_plant_" prefix
    """
    np.random.seed(random_seed)
    
    # Only evaluate classes with data: 5, 8, 9
    plant_classes = [5, 8, 9]
    N, C = y_true.shape
    metrics = {}

    for c in plant_classes:
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_plant_class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        
        # Manually calculate confusion matrix values to handle edge cases
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        sensitivity = recall
        specificity = tn / (tn + fp + eps)
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        mcc = numerator / (denominator + eps)

        metrics[f'group_plant_class_{c}_opt_f1'] = f1
        metrics[f'group_plant_class_{c}_opt_precision'] = precision
        metrics[f'group_plant_class_{c}_opt_recall'] = recall
        metrics[f'group_plant_class_{c}_opt_accuracy'] = accuracy
        metrics[f'group_plant_class_{c}_opt_sensitivity'] = sensitivity
        metrics[f'group_plant_class_{c}_opt_specificity'] = specificity
        metrics[f'group_plant_class_{c}_opt_mcc'] = mcc
        metrics[f'group_plant_class_{c}_opt_tp'] = tp
        metrics[f'group_plant_class_{c}_opt_tn'] = tn
        metrics[f'group_plant_class_{c}_opt_fp'] = fp
        metrics[f'group_plant_class_{c}_opt_fn'] = fn

        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_plant_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_plant_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_plant_class_{c}_auc'] = 0.0
                metrics[f'group_plant_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_plant_class_{c}_auc'] = 0.0
            metrics[f'group_plant_class_{c}_auprc'] = 0.0

    # Calculate macro averages (only for classes with data)
    all_f1, all_prec, all_rec = [], [], []
    for c in plant_classes:
        all_f1.append(metrics[f'group_plant_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_plant_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_plant_class_{c}_opt_recall'])

    metrics['group_plant_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_plant_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_plant_opt_macro_recall'] = np.mean(all_rec)

    # Add standard macro/micro/weighted metrics (without _opt_ suffix) for logging compatibility
    metrics['group_plant_macro_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_macro_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_macro_recall'] = metrics['group_plant_opt_macro_recall']
    metrics['group_plant_micro_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_micro_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_micro_recall'] = metrics['group_plant_opt_macro_recall']
    metrics['group_plant_weighted_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_weighted_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_weighted_recall'] = metrics['group_plant_opt_macro_recall']

    # 4-class evaluation (if hierarchical)
    if y_4prob is not None:
        for c in range(4):
            y_true_c = y_4class[:, c]
            y_prob_c = y_4prob[:, c]

            opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
            metrics[f'group_plant_4class_{c}_opt_threshold'] = opt_threshold

            y_pred_c = (y_prob_c >= opt_threshold).astype(int)
            
            # Manually calculate confusion matrix values to handle edge cases
            tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
            tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
            fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
            fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

            eps = 1e-10
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
            sensitivity = recall
            specificity = tn / (tn + fp + eps)
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
            mcc = numerator / (denominator + eps)

            metrics[f'group_plant_4class_{c}_opt_f1'] = f1
            metrics[f'group_plant_4class_{c}_opt_precision'] = precision
            metrics[f'group_plant_4class_{c}_opt_recall'] = recall
            metrics[f'group_plant_4class_{c}_opt_accuracy'] = accuracy
            metrics[f'group_plant_4class_{c}_opt_sensitivity'] = sensitivity
            metrics[f'group_plant_4class_{c}_opt_specificity'] = specificity
            metrics[f'group_plant_4class_{c}_opt_mcc'] = mcc
            metrics[f'group_plant_4class_{c}_opt_tp'] = tp
            metrics[f'group_plant_4class_{c}_opt_tn'] = tn
            metrics[f'group_plant_4class_{c}_opt_fp'] = fp
            metrics[f'group_plant_4class_{c}_opt_fn'] = fn

            if len(np.unique(y_true_c)) > 1:
                try:
                    metrics[f'group_plant_4class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                    metrics[f'group_plant_4class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
                except ValueError:
                    metrics[f'group_plant_4class_{c}_auc'] = 0.0
                    metrics[f'group_plant_4class_{c}_auprc'] = 0.0
            else:
                metrics[f'group_plant_4class_{c}_auc'] = 0.0
                metrics[f'group_plant_4class_{c}_auprc'] = 0.0

        all_f1_4, all_prec_4, all_rec_4 = [], [], []
        for c in range(4):
            all_f1_4.append(metrics[f'group_plant_4class_{c}_opt_f1'])
            all_prec_4.append(metrics[f'group_plant_4class_{c}_opt_precision'])
            all_rec_4.append(metrics[f'group_plant_4class_{c}_opt_recall'])

        metrics['group_plant_4class_opt_macro_f1'] = np.mean(all_f1_4)
        metrics['group_plant_4class_opt_macro_precision'] = np.mean(all_prec_4)
        metrics['group_plant_4class_opt_macro_recall'] = np.mean(all_rec_4)

    return metrics


def evaluate_ac4c(y_true: np.ndarray, y_prob: np.ndarray, y_4class: np.ndarray,
                 device, random_seed: int = 42, 
                 dataset_mode: str = 'unbalanced',
                 y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate AC4C data based on dataset mode (balanced or unbalanced).
    
    This function determines which classes have data and evaluates them using
    the appropriate method (balanced or unbalanced evaluation) based on the
    dataset_mode parameter.
    
    Args:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        y_4class: 4-class ground truth labels (N, 4)
        device: Device to run evaluation on
        random_seed: Random seed for reproducibility/sampling
        dataset_mode: 'balanced' or 'unbalanced' - determines evaluation method
        y_4prob: 4-class predicted probabilities (N, 4)

    Returns:
        Dictionary of metrics with "group_ac4c_" prefix
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}
    
    # Determine which classes have data
    valid_classes = []
    for c in range(C):
        if np.any(y_true[:, c] == 1):
            valid_classes.append(c)
    
    # 12-Class Evaluation (only for classes with data)
    for c in valid_classes:
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        if dataset_mode == 'balanced':
            # BalanceB: 1:1 positive:negative ratio
            pos_idx = np.where(y_true_c == 1)[0]
            neg_idx_all = np.where(y_true_c == 0)[0]

            n_pos = len(pos_idx)
            n_neg_all = len(neg_idx_all)

            if n_pos == 0:
                # Add default metrics when no positive samples exist
                for key in ['opt_threshold', 'opt_f1', 'opt_precision', 'opt_recall', 'opt_accuracy', 
                           'opt_sensitivity', 'opt_specificity', 'opt_mcc', 'auc', 'auprc',
                           'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn']:
                    metrics[f'group_ac4c_class_{c}_{key}'] = 0.0
                continue

            target_count = min(n_pos, n_neg_all)
            pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
            neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

            selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
            y_true_eval = y_true[selected_idx, c]
            y_prob_eval = y_prob[selected_idx, c]
        else:
            # Unbalance: use real distribution
            y_true_eval = y_true_c
            y_prob_eval = y_prob_c

        opt_threshold = find_optimal_threshold(y_true_eval, y_prob_eval)
        metrics[f'group_ac4c_class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_eval >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_eval == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_eval == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_eval == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_eval == 1) & (y_pred_c == 0))

        metrics[f'group_ac4c_class_{c}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_ac4c_class_{c}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_ac4c_class_{c}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_ac4c_class_{c}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_ac4c_class_{c}_opt_sensitivity'] = metrics[f'group_ac4c_class_{c}_opt_recall']
        metrics[f'group_ac4c_class_{c}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_ac4c_class_{c}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_ac4c_class_{c}_opt_tp'] = tp
        metrics[f'group_ac4c_class_{c}_opt_tn'] = tn
        metrics[f'group_ac4c_class_{c}_opt_fp'] = fp
        metrics[f'group_ac4c_class_{c}_opt_fn'] = fn

        if len(np.unique(y_true_eval)) > 1:
            try:
                metrics[f'group_ac4c_class_{c}_auc'] = roc_auc_score(y_true_eval, y_prob_eval)
                metrics[f'group_ac4c_class_{c}_auprc'] = average_precision_score(y_true_eval, y_prob_eval)
            except ValueError:
                metrics[f'group_ac4c_class_{c}_auc'] = 0.0
                metrics[f'group_ac4c_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_ac4c_class_{c}_auc'] = 0.0
            metrics[f'group_ac4c_class_{c}_auprc'] = 0.0

    # 4-Class Evaluation
    for g in range(4):
        y_true_g = y_4class[:, g]
        
        if y_4prob is not None:
            y_prob_g = y_4prob[:, g]
        else:
            # Use INDEX_TO_GROUP to convert integer index to group name
            group_name = INDEX_TO_GROUP[g]
            class_indices = GROUP_TO_CLASS_INDICES[group_name]
            y_prob_g = y_prob[:, class_indices].max(axis=1)

        if dataset_mode == 'balanced':
            # BalanceB: 1:1 positive:negative ratio
            pos_idx = np.where(y_true_g == 1)[0]
            neg_idx_all = np.where(y_true_g == 0)[0]

            n_pos = len(pos_idx)
            n_neg_all = len(neg_idx_all)

            if n_pos == 0:
                # Add default metrics when no positive samples exist
                for key in ['opt_threshold', 'opt_f1', 'opt_precision', 'opt_recall', 'opt_accuracy', 
                           'opt_sensitivity', 'opt_specificity', 'opt_mcc', 'auc', 'auprc',
                           'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn']:
                    metrics[f'group_ac4c_4class_{g}_{key}'] = 0.0
                continue

            target_count = min(n_pos, n_neg_all)
            pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
            neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

            selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
            y_true_eval = y_true_g[selected_idx]
            # Use INDEX_TO_GROUP to convert integer index to group name
            group_name = INDEX_TO_GROUP[g]
            if y_4prob is not None:
                y_prob_eval = y_4prob[selected_idx, g]
            else:
                class_indices = GROUP_TO_CLASS_INDICES[group_name]
                y_prob_eval = y_prob[selected_idx][:, class_indices].max(axis=1)
        else:
            # Unbalance: use real distribution
            y_true_eval = y_true_g
            y_prob_eval = y_prob_g

        opt_threshold = find_optimal_threshold(y_true_eval, y_prob_eval)
        metrics[f'group_ac4c_4class_{g}_opt_threshold'] = opt_threshold

        y_pred_g = (y_prob_eval >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_eval == 1) & (y_pred_g == 1))
        tn = np.sum((y_true_eval == 0) & (y_pred_g == 0))
        fp = np.sum((y_true_eval == 0) & (y_pred_g == 1))
        fn = np.sum((y_true_eval == 1) & (y_pred_g == 0))

        metrics[f'group_ac4c_4class_{g}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_ac4c_4class_{g}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_ac4c_4class_{g}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_ac4c_4class_{g}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_ac4c_4class_{g}_opt_sensitivity'] = metrics[f'group_ac4c_4class_{g}_opt_recall']
        metrics[f'group_ac4c_4class_{g}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_ac4c_4class_{g}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_ac4c_4class_{g}_opt_tp'] = tp
        metrics[f'group_ac4c_4class_{g}_opt_tn'] = tn
        metrics[f'group_ac4c_4class_{g}_opt_fp'] = fp
        metrics[f'group_ac4c_4class_{g}_opt_fn'] = fn

        if len(np.unique(y_true_eval)) > 1:
            try:
                metrics[f'group_ac4c_4class_{g}_auc'] = roc_auc_score(y_true_eval, y_prob_eval)
                metrics[f'group_ac4c_4class_{g}_auprc'] = average_precision_score(y_true_eval, y_prob_eval)
            except ValueError:
                metrics[f'group_ac4c_4class_{g}_auc'] = 0.0
                metrics[f'group_ac4c_4class_{g}_auprc'] = 0.0
        else:
            metrics[f'group_ac4c_4class_{g}_auc'] = 0.0
            metrics[f'group_ac4c_4class_{g}_auprc'] = 0.0

    # Calculate macro averages (only for classes with data)
    if valid_classes:
        all_f1, all_prec, all_rec = [], [], []
        for c in valid_classes:
            all_f1.append(metrics[f'group_ac4c_class_{c}_opt_f1'])
            all_prec.append(metrics[f'group_ac4c_class_{c}_opt_precision'])
            all_rec.append(metrics[f'group_ac4c_class_{c}_opt_recall'])

        metrics['group_ac4c_opt_macro_f1'] = np.mean(all_f1)
        metrics['group_ac4c_opt_macro_precision'] = np.mean(all_prec)
        metrics['group_ac4c_opt_macro_recall'] = np.mean(all_rec)

        # Add standard macro/micro/weighted metrics (without _opt_ suffix) for logging compatibility
        metrics['group_ac4c_macro_f1'] = metrics['group_ac4c_opt_macro_f1']
        metrics['group_ac4c_macro_precision'] = metrics['group_ac4c_opt_macro_precision']
        metrics['group_ac4c_macro_recall'] = metrics['group_ac4c_opt_macro_recall']
        metrics['group_ac4c_micro_f1'] = metrics['group_ac4c_opt_macro_f1']
        metrics['group_ac4c_micro_precision'] = metrics['group_ac4c_opt_macro_precision']
        metrics['group_ac4c_micro_recall'] = metrics['group_ac4c_opt_macro_recall']
        metrics['group_ac4c_weighted_f1'] = metrics['group_ac4c_opt_macro_f1']
        metrics['group_ac4c_weighted_precision'] = metrics['group_ac4c_opt_macro_precision']
        metrics['group_ac4c_weighted_recall'] = metrics['group_ac4c_opt_macro_recall']

    # 4-class macro averages
    all_f1_4, all_prec_4, all_rec_4 = [], [], []
    for g in range(4):
        opt_f1_key = f'group_ac4c_4class_{g}_opt_f1'
        if opt_f1_key in metrics:
            all_f1_4.append(metrics[opt_f1_key])
            all_prec_4.append(metrics[f'group_ac4c_4class_{g}_opt_precision'])
            all_rec_4.append(metrics[f'group_ac4c_4class_{g}_opt_recall'])

    if all_f1_4:
        metrics['group_ac4c_4class_opt_macro_f1'] = np.mean(all_f1_4)
        metrics['group_ac4c_4class_opt_macro_precision'] = np.mean(all_prec_4)
        metrics['group_ac4c_4class_opt_macro_recall'] = np.mean(all_rec_4)
    else:
        metrics['group_ac4c_4class_opt_macro_f1'] = 0.0
        metrics['group_ac4c_4class_opt_macro_precision'] = 0.0
        metrics['group_ac4c_4class_opt_macro_recall'] = 0.0

    return metrics


def evaluate_plant_balanceb(y_true: np.ndarray, y_prob: np.ndarray, y_4class: np.ndarray,
                            device, random_seed: int = 42, 
                            y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate plant data in BalanceB mode: 1:1 positive:negative ratio.
    Only evaluates classes with data: 5 (Y), 8 (m5C), 9 (m6A)

    Args:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        y_4class: 4-class ground truth labels (N, 4)
        device: Device to run evaluation on
        random_seed: Random seed for sampling
        y_4prob: 4-class predicted probabilities (N, 4)

    Returns:
        Dictionary of metrics with "group_plant_" prefix
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    plant_classes = [5, 8, 9]
    metrics = {}

    # 12-Class BalanceB Evaluation (only for classes with data)
    for c in plant_classes:
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx_all = np.where(y_true[:, c] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            # Add default metrics when no positive samples exist
            for key in ['opt_threshold', 'opt_f1', 'opt_precision', 'opt_recall', 'opt_accuracy', 
                       'opt_sensitivity', 'opt_specificity', 'opt_mcc', 'auc', 'auprc',
                       'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn']:
                metrics[f'group_plant_class_{c}_{key}'] = 0.0
            continue

        target_count = min(n_pos, n_neg_all)
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_c = y_true[selected_idx, c]
        y_prob_c = y_prob[selected_idx, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_plant_class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_plant_class_{c}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_plant_class_{c}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_plant_class_{c}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_plant_class_{c}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_plant_class_{c}_opt_sensitivity'] = metrics[f'group_plant_class_{c}_opt_recall']
        metrics[f'group_plant_class_{c}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_plant_class_{c}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_plant_class_{c}_opt_tp'] = tp
        metrics[f'group_plant_class_{c}_opt_tn'] = tn
        metrics[f'group_plant_class_{c}_opt_fp'] = fp
        metrics[f'group_plant_class_{c}_opt_fn'] = fn

        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_plant_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_plant_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_plant_class_{c}_auc'] = 0.0
                metrics[f'group_plant_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_plant_class_{c}_auc'] = 0.0
            metrics[f'group_plant_class_{c}_auprc'] = 0.0

    # 4-Class BalanceB Evaluation
    for g in range(4):
        pos_idx = np.where(y_4class[:, g] == 1)[0]
        neg_idx_all = np.where(y_4class[:, g] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            continue

        target_count = min(n_pos, n_neg_all)
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_g = y_4class[selected_idx, g]
        
        if y_4prob is not None:
            y_prob_g = y_4prob[selected_idx, g]
        else:
            # Use INDEX_TO_GROUP to convert integer index to group name
            group_name = INDEX_TO_GROUP[g]
            class_indices = GROUP_TO_CLASS_INDICES[group_name]
            y_prob_g = y_prob[selected_idx][:, class_indices].max(axis=1)

        opt_threshold = find_optimal_threshold(y_true_g, y_prob_g)
        metrics[f'group_plant_4class_{g}_opt_threshold'] = opt_threshold

        y_pred_g = (y_prob_g >= opt_threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_g == 1) & (y_pred_g == 1))
        tn = np.sum((y_true_g == 0) & (y_pred_g == 0))
        fp = np.sum((y_true_g == 0) & (y_pred_g == 1))
        fn = np.sum((y_true_g == 1) & (y_pred_g == 0))

        metrics[f'group_plant_4class_{g}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_plant_4class_{g}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_plant_4class_{g}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_plant_4class_{g}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_plant_4class_{g}_opt_sensitivity'] = metrics[f'group_plant_4class_{g}_opt_recall']
        metrics[f'group_plant_4class_{g}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_plant_4class_{g}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_plant_4class_{g}_opt_tp'] = tp
        metrics[f'group_plant_4class_{g}_opt_tn'] = tn
        metrics[f'group_plant_4class_{g}_opt_fp'] = fp
        metrics[f'group_plant_4class_{g}_opt_fn'] = fn

        if len(np.unique(y_true_g)) > 1:
            try:
                metrics[f'group_plant_4class_{g}_auc'] = roc_auc_score(y_true_g, y_prob_g)
                metrics[f'group_plant_4class_{g}_auprc'] = average_precision_score(y_true_g, y_prob_g)
            except ValueError:
                metrics[f'group_plant_4class_{g}_auc'] = 0.0
                metrics[f'group_plant_4class_{g}_auprc'] = 0.0
        else:
            metrics[f'group_plant_4class_{g}_auc'] = 0.0
            metrics[f'group_plant_4class_{g}_auprc'] = 0.0

    # Calculate macro averages (only for classes with data)
    if plant_classes:
        all_f1, all_prec, all_rec = [], [], []
        for c in plant_classes:
            all_f1.append(metrics[f'group_plant_class_{c}_opt_f1'])
            all_prec.append(metrics[f'group_plant_class_{c}_opt_precision'])
            all_rec.append(metrics[f'group_plant_class_{c}_opt_recall'])

        metrics['group_plant_opt_macro_f1'] = np.mean(all_f1)
        metrics['group_plant_opt_macro_precision'] = np.mean(all_prec)
        metrics['group_plant_opt_macro_recall'] = np.mean(all_rec)

    # 4-class macro averages
    all_f1_4, all_prec_4, all_rec_4 = [], [], []
    for g in range(4):
        opt_f1_key = f'group_plant_4class_{g}_opt_f1'
        if opt_f1_key in metrics:
            all_f1_4.append(metrics[opt_f1_key])
            all_prec_4.append(metrics[f'group_plant_4class_{g}_opt_precision'])
            all_rec_4.append(metrics[f'group_plant_4class_{g}_opt_recall'])

    if all_f1_4:
        metrics['group_plant_4class_opt_macro_f1'] = np.mean(all_f1_4)
        metrics['group_plant_4class_opt_macro_precision'] = np.mean(all_prec_4)
        metrics['group_plant_4class_opt_macro_recall'] = np.mean(all_rec_4)
    else:
        metrics['group_plant_4class_opt_macro_f1'] = 0.0
        metrics['group_plant_4class_opt_macro_precision'] = 0.0
        metrics['group_plant_4class_opt_macro_recall'] = 0.0

    return metrics


def evaluate_with_optimal_threshold(model, dataloader, device, use_hierarchical=False) -> Dict[str, float]:
    """
    Evaluate using optimal F1 threshold for each class.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for test set
        device: Device to run evaluation on
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class)

    Returns:
        Dictionary of metrics including optimal thresholds
    """
    model.eval()
    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)

            if use_hierarchical:
                logits, _ = logits

            probs = torch.sigmoid(logits)
            all_y_true.append(batch.y)
            all_y_prob.append(probs)

    y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(all_y_prob, dim=0).cpu().numpy()

    N, C = y_true.shape
    metrics = {}

    for c in range(C):
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        eps = 1e-10
        metrics[f'group_class_{c}_opt_tp'] = tp
        metrics[f'group_class_{c}_opt_tn'] = tn
        metrics[f'group_class_{c}_opt_fp'] = fp
        metrics[f'group_class_{c}_opt_fn'] = fn
        metrics[f'group_class_{c}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_class_{c}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_class_{c}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_class_{c}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_class_{c}_opt_sensitivity'] = metrics[f'group_class_{c}_opt_recall']
        metrics[f'group_class_{c}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_class_{c}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)

        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_class_{c}_auc'] = 0.0
                metrics[f'group_class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_class_{c}_auc'] = 0.0
            metrics[f'group_class_{c}_auprc'] = 0.0

    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_class_{c}_opt_recall'])

    metrics['group_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_opt_macro_recall'] = np.mean(all_rec)

    return metrics


def evaluate_4class_with_optimal_threshold(model, dataloader, device, use_hierarchical=False) -> Dict[str, float]:
    """
    Evaluate 4-class metrics using optimal F1 threshold for each class.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for test set
        device: Device to run evaluation on
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class)

    Returns:
        Dictionary of metrics for 4-class evaluation
    """
    if not use_hierarchical:
        return {}

    # Import INDEX_TO_GROUP for correct mapping
    from utils.common import INDEX_TO_GROUP
    
    model.eval()
    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)

            y_12 = batch.y
            y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

            # Use INDEX_TO_GROUP to map 4-class index (0-3) to nucleotide (A, C, G, U)
            for group_idx in range(4):
                nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U'
                class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
                y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

            probs_4 = torch.sigmoid(logits_4)
            all_y_true.append(y_4.cpu())
            all_y_prob.append(probs_4.cpu())

    y_true = torch.cat(all_y_true, dim=0).numpy()
    y_prob = torch.cat(all_y_prob, dim=0).numpy()

    N, C = y_true.shape
    metrics = {}

    for c in range(C):
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_4class_{c}_opt_threshold'] = opt_threshold

        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        eps = 1e-10
        metrics[f'group_4class_{c}_opt_tp'] = tp
        metrics[f'group_4class_{c}_opt_tn'] = tn
        metrics[f'group_4class_{c}_opt_fp'] = fp
        metrics[f'group_4class_{c}_opt_fn'] = fn
        metrics[f'group_4class_{c}_opt_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_4class_{c}_opt_precision'] = tp / (tp + fp + eps)
        metrics[f'group_4class_{c}_opt_recall'] = tp / (tp + fn + eps)
        metrics[f'group_4class_{c}_opt_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_4class_{c}_opt_sensitivity'] = metrics[f'group_4class_{c}_opt_recall']
        metrics[f'group_4class_{c}_opt_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_4class_{c}_opt_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)

        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_4class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_4class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_4class_{c}_auc'] = 0.0
                metrics[f'group_4class_{c}_auprc'] = 0.0
        else:
            metrics[f'group_4class_{c}_auc'] = 0.0
            metrics[f'group_4class_{c}_auprc'] = 0.0

    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_4class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_4class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_4class_{c}_opt_recall'])

    metrics['group_4class_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_4class_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_4class_opt_macro_recall'] = np.mean(all_rec)

    return metrics


def get_all_predictions(model, dataloader, device, use_hierarchical=False):
    """
    Run model inference once to get all predictions for evaluation.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for test set
        device: Device to run evaluation on
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class)

    Returns:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        y_4class: 4-class ground truth labels (N, 4)
        y_4prob: 4-class predicted probabilities (N, 4) - None if not hierarchical
    """
    # Import INDEX_TO_GROUP for correct mapping
    from utils.common import INDEX_TO_GROUP
    
    model.eval()
    all_y_true = []
    all_y_prob = []
    all_y_4prob = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)

            if use_hierarchical:
                logits_12, logits_4 = logits
                probs_12 = torch.sigmoid(logits_12)
                probs_4 = torch.sigmoid(logits_4)
                all_y_4prob.append(probs_4.cpu())
            else:
                probs_12 = torch.sigmoid(logits)

            all_y_true.append(batch.y)
            all_y_prob.append(probs_12)

    y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(all_y_prob, dim=0).cpu().numpy()

    N, C = y_true.shape
    y_4class = np.zeros((N, 4), dtype=np.float32)
    # Use INDEX_TO_GROUP to map 4-class index (0-3) to nucleotide (A, C, G, U)
    for group_idx in range(4):
        nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U'
        class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
        y_4class[:, group_idx] = y_true[:, class_indices].max(axis=1)

    if use_hierarchical and all_y_4prob:
        y_4prob = torch.cat(all_y_4prob, dim=0).cpu().numpy()
    else:
        y_4prob = None

    return y_true, y_prob, y_4class, y_4prob


# ============================================================================
# Group-Based Balanced Evaluation
# ============================================================================

def evaluate_group_balanceb(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_4class: np.ndarray,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate using balanced dataset with group-specific negative sampling.

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1
    - Negative set: randomly sampled from samples in the SAME NUCLEOTIDE GROUP
      where the label for class c is 0, but they may have other modifications
    - Downsampling: min(|P|, |N|)
    - Balance: 1:1 positive:negative ratio

    Example: For Am (class 0, group A):
      - Positive: sequences with Am
      - Negative: sequences from group A that DON'T have Am (i.e., sequences with
        Atol, m1A, m6A, or m6Am, but not Am)

    Args:
        y_true: Ground truth labels (N, 12)
        y_prob: Predicted probabilities (N, 12)
        y_4class: 4-class ground truth labels (N, 4)
        random_seed: Random seed for sampling

    Returns:
        Dictionary of metrics with "group_balanced_" prefix
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}

    # 12-Class Group-Balanced Evaluation
    for c in range(C):
        # Get the nucleotide group for class c
        nucleotide = INDEX_TO_NUCLEOTIDE[c]  # 'A', 'C', 'G', or 'U'
        group_classes = GROUP_TO_CLASS_INDICES[nucleotide]  # e.g., [0, 1, 7, 9, 10] for Am

        # Positive samples: sequences with this class
        pos_idx = np.where(y_true[:, c] == 1)[0]

        # Negative samples: sequences in the same group WITHOUT this class
        # First, find all sequences in this group (have any modification in the group)
        group_mask = y_true[:, group_classes].sum(axis=1) > 0
        group_idx_all = np.where(group_mask)[0]

        # From these, select samples that don't have class c
        neg_idx_all = group_idx_all[y_true[group_idx_all, c] == 0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            # No positive samples for this class
            for key in ['opt_f1', 'opt_precision', 'opt_recall', 'opt_accuracy', 'opt_sensitivity',
                       'opt_specificity', 'opt_mcc', 'opt_auc', 'opt_auprc', 'opt_tp', 'opt_tn',
                       'opt_fp', 'opt_fn', 'opt_threshold', 'precision', 'recall', 'f1', 'accuracy']:
                metrics[f'group_balanced_class_{c}_{key}'] = 0.0
            continue

        if n_neg_all == 0:
            # No negative samples in the same group (rare case)
            # Fall back to using all negative samples
            neg_idx_all = np.where(y_true[:, c] == 0)[0]
            n_neg_all = len(neg_idx_all)

        # Downsample to 1:1 ratio
        target_count = min(n_pos, n_neg_all)

        if n_pos >= target_count:
            pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False)
        else:
            pos_idx_sampled = pos_idx

        if n_neg_all >= target_count:
            neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False)
        else:
            neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_c = y_true[selected_idx, c]
        y_prob_c = y_prob[selected_idx, c]

        # Find optimal threshold
        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_balanced_class_{c}_opt_threshold'] = opt_threshold

        # Calculate predictions with optimal threshold
        y_pred_c = (y_prob_c >= opt_threshold).astype(int)

        # Calculate confusion matrix
        eps = 1e-10
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        # Calculate metrics
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        sensitivity = recall
        specificity = tn / (tn + fp + eps)

        # Matthews Correlation Coefficient
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        mcc = numerator / (denominator + eps)

        metrics[f'group_balanced_class_{c}_opt_f1'] = f1
        metrics[f'group_balanced_class_{c}_opt_precision'] = precision
        metrics[f'group_balanced_class_{c}_opt_recall'] = recall
        metrics[f'group_balanced_class_{c}_opt_accuracy'] = accuracy
        metrics[f'group_balanced_class_{c}_opt_sensitivity'] = sensitivity
        metrics[f'group_balanced_class_{c}_opt_specificity'] = specificity
        metrics[f'group_balanced_class_{c}_opt_mcc'] = mcc
        metrics[f'group_balanced_class_{c}_opt_tp'] = tp
        metrics[f'group_balanced_class_{c}_opt_tn'] = tn
        metrics[f'group_balanced_class_{c}_opt_fp'] = fp
        metrics[f'group_balanced_class_{c}_opt_fn'] = fn

        # AUC and AUPRC (threshold-independent)
        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_balanced_class_{c}_opt_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_balanced_class_{c}_opt_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                metrics[f'group_balanced_class_{c}_opt_auc'] = 0.0
                metrics[f'group_balanced_class_{c}_opt_auprc'] = 0.0
        else:
            metrics[f'group_balanced_class_{c}_opt_auc'] = 0.0
            metrics[f'group_balanced_class_{c}_opt_auprc'] = 0.0

        # Also store without _opt_ prefix for compatibility
        metrics[f'group_balanced_class_{c}_f1'] = f1
        metrics[f'group_balanced_class_{c}_precision'] = precision
        metrics[f'group_balanced_class_{c}_recall'] = recall
        metrics[f'group_balanced_class_{c}_accuracy'] = accuracy

    # Calculate macro averages across all 12 classes
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_balanced_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_balanced_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_balanced_class_{c}_opt_recall'])

    metrics['group_balanced_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_balanced_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_balanced_opt_macro_recall'] = np.mean(all_rec)

    # Add standard macro/micro/weighted metrics for logging compatibility
    metrics['group_balanced_macro_f1'] = metrics['group_balanced_opt_macro_f1']
    metrics['group_balanced_macro_precision'] = metrics['group_balanced_opt_macro_precision']
    metrics['group_balanced_macro_recall'] = metrics['group_balanced_opt_macro_recall']
    metrics['group_balanced_micro_f1'] = metrics['group_balanced_opt_macro_f1']
    metrics['group_balanced_micro_precision'] = metrics['group_balanced_opt_macro_precision']
    metrics['group_balanced_micro_recall'] = metrics['group_balanced_opt_macro_recall']
    metrics['group_balanced_weighted_f1'] = metrics['group_balanced_opt_macro_f1']
    metrics['group_balanced_weighted_precision'] = metrics['group_balanced_opt_macro_precision']
    metrics['group_balanced_weighted_recall'] = metrics['group_balanced_opt_macro_recall']

    return metrics
