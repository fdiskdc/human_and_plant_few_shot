"""
Group-based balanced evaluation for RNA Multi-label Classification

This module contains:
- evaluate_group_balanceb: Evaluate with group-specific negative sampling
  For each class c (e.g., Am), negative samples are only drawn from
  other classes in the same nucleotide group (e.g., Atol, m1A, m6A, m6Am for group A)
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, Optional

# Import constants from common
from utils.common import MOD_NAMES, GROUP_TO_CLASS_INDICES, INDEX_TO_NUCLEOTIDE


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximizes F1 score for binary classification.

    Args:
        y_true: Ground truth labels (N,)
        y_prob: Predicted probabilities (N,)

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score

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
