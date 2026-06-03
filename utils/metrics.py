"""
utils/metrics.py - RNA多标签分类评估指标 / Evaluation Metrics for RNA Multi-label Classification

全面的机器学习评估:最优阈值查找、Unbalance / BalanceB / Plant / Group 评估、4 类最优阈值、F1/Acc/Precision/Recall/AUC/MCC/AUPRC。
Comprehensive ML evaluation: optimal threshold finding, unbalance/balanceb/plant/group evaluations, 4-class with optimal
threshold, F1/Acc/Precision/Recall/AUC/MCC/AUPRC.

功能模块 / Modules:
- evaluate_unbalance: Unbalance 模式评估 / Unbalance mode evaluation
- evaluate_balanceb: BalanceB 模式评估 / BalanceB mode evaluation
- evaluate_plant_unbalance: 植物 Unbalance 评估 / Plant unbalance
- evaluate_plant_balanceb: 植物 BalanceB 评估 / Plant balanceb
- evaluate_group_balanceb: 4 组 BalanceB 评估 / 4-group balanceb
- evaluate_4class_with_optimal_threshold: 4 类最优阈值评估 / 4-class with optimal threshold
- evaluate_with_optimal_threshold: 12 类最优阈值评估 / 12-class with optimal threshold
- find_optimal_threshold_per_class: 每类最优阈值 / Per-class optimal threshold
- print_evaluation_results: 结果格式化输出 / Formatted result output
- print_few_shot_results: 小样本结果输出 / Few-shot result output

输入 / Inputs:
- y_true: 真实标签 [N, num_classes] / Ground truth labels
- y_score / y_pred: 预测分数或类别 / Predicted scores or labels
- 各种参数 / Various parameters

输出 / Outputs:
- 评估指标字典 / Evaluation metrics dict
- PrettyTable 输出 / PrettyTable output
- 终端详细报告 / Detailed terminal report

数据流 / Data Flow:
1. 计算 y_score / Compute predictions
2. 找最优阈值 / Find optimal threshold
3. 计算每类指标 / Compute per-class metrics
4. 聚合 + 输出 / Aggregate + output

相关文件 / Related Files:
- 调用 / Calls: sklearn.metrics, numpy, utils.common
- 被调用 / Called by: train_*.py, test_*.py, evaluation scripts

使用示例 / Usage Example:
    from utils import evaluate_unbalance, evaluate_with_optimal_threshold
    metrics = evaluate_unbalance(y_true, y_pred, y_score, MOD_NAMES)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
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
    
    /找到使F1分数最大的二分类阈值

    Args:
        y_true: Ground truth labels (N,) / 真实标签 (N,)
        y_prob: Predicted probabilities (N,) / 预测概率 (N,)

    Returns:
        Optimal threshold value / 最优阈值
    """
    # Only search if we have both classes
    # 仅当存在两个类别时才搜索
    if len(np.unique(y_true)) < 2:
        return 0.5

    best_threshold = 0.5
    best_f1 = 0.0

    # Search thresholds from 0.05 to 0.95 with step 0.01
    # 在0.05到0.95之间搜索阈值，步长为0.01
    thresholds = np.arange(0.05, 1.0, 0.01)

    # Iterate through all thresholds to find the best one
    # 遍历所有阈值，找到使F1分数最大的阈值
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
    
    /计算多标签分类的综合指标

    Args:
        y_true: Ground truth labels, shape (N, C) / 真实标签，形状(N, C)
        y_pred: Binary predictions, shape (N, C) / 二分类预测，形状(N, C)
        y_prob: Predicted probabilities, shape (N, C) / 预测概率，形状(N, C)

    Returns:
        Dictionary of metrics with 'group_' prefix / 带有'group_'前缀的指标字典
    """
    metrics = {}
    N, C = y_true.shape

    # Convert to numpy for sklearn
    # 转换为numpy格式以使用sklearn
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    y_prob_np = y_prob.cpu().numpy() if isinstance(y_prob, torch.Tensor) else y_prob

    # Calculate metrics for each average type (macro, micro, weighted)
    # 计算每种平均类型（宏平均、微平均、加权平均）的指标
    for avg in ['macro', 'micro', 'weighted']:
        metrics[f'group_{avg}_f1'] = f1_score(y_true_np, y_pred_np, average=avg, zero_division=0)
        metrics[f'group_{avg}_precision'] = precision_score(y_true_np, y_pred_np, average=avg, zero_division=0)
        metrics[f'group_{avg}_recall'] = recall_score(y_true_np, y_pred_np, average=avg, zero_division=0)

    # Calculate accuracy (average-independent)
    # 计算准确率（独立于平均方式）
    metrics['group_accuracy'] = accuracy_score(y_true_np, y_pred_np)

    # Per-class metrics - with 'group_' prefix
    # 每个类别的指标 - 带有'group_'前缀
    for c in range(C):
        y_true_c = y_true_np[:, c]
        y_pred_c = y_pred_np[:, c]
        y_prob_c = y_prob_np[:, c]

        # Confusion matrix values
        # 混淆矩阵值：真正例、真负例、假正例、假负例
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_class_{c}_tp'] = tp
        metrics[f'group_class_{c}_tn'] = tn
        metrics[f'group_class_{c}_fp'] = fp
        metrics[f'group_class_{c}_fn'] = fn

        # Basic metrics
        # 基础指标
        eps = 1e-10  # Small epsilon to avoid division by zero / 避免除以零的小常数
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        # Sensitivity (same as recall) and Specificity
        # 灵敏度（同召回率）和特异度
        sensitivity = recall
        specificity = tn / (tn + fp + eps)

        # Matthews Correlation Coefficient
        # Matthews相关系数
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
        # AUC和AUPRC（仅当两个类别都存在时计算）
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
    
    /使用不平衡数据集评估：真实分布测试

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1 / 正样本集：y_true[:, c] == 1的所有样本
    - Negative set: all samples where y_true[:, c] == 0 / 负样本集：y_true[:, c] == 0的所有样本
    - Calculate metrics on this unbalanced dataset / 在此不平衡数据集上计算指标

    Args:
        y_true: Ground truth labels (N, num_classes) / 真实标签 (N, num_classes)
        y_prob: Predicted probabilities (N, num_classes) / 预测概率 (N, num_classes)
        device: Device to run evaluation on / 运行评估的设备
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        random_seed: Random seed for reproducibility / 用于可重复性的随机种子
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics for unbalanced evaluation / 不平衡评估的指标字典
    """
    np.random.seed(random_seed)
    N, C = y_true.shape
    metrics = {}

    # For each class, calculate metrics on unbalanced dataset
    # 对每个类别，在不平衡数据集上计算指标
    for c in range(C):
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        # Find optimal threshold for this class
        # 找到该类别的最优阈值
        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_class_{c}_opt_threshold'] = opt_threshold

        # Calculate predictions with optimal threshold
        # 使用最优阈值计算预测
        y_pred_c = (y_prob_c >= opt_threshold).astype(int)

        # Confusion matrix values - handle edge cases
        # 混淆矩阵值 - 处理边界情况
        try:
            cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge case where only one class is present
                # 处理只存在一个类别的边界情况
                tn, fp, fn, tp = 0, 0, 0, 0
                if len(np.unique(y_true_c)) == 1:
                    if y_true_c[0] == 1:
                        # All positive samples
                        # 全部为正样本
                        tp = np.sum(y_pred_c == 1)
                        fn = np.sum(y_pred_c == 0)
                    else:
                        # All negative samples
                        # 全部为负样本
                        tn = np.sum(y_pred_c == 0)
                        fp = np.sum(y_pred_c == 1)
        except:
            # Fallback to manual calculation
            # 回退到手动计算
            tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
            tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
            fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
            fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_class_{c}_opt_tp'] = tp
        metrics[f'group_class_{c}_opt_tn'] = tn
        metrics[f'group_class_{c}_opt_fp'] = fp
        metrics[f'group_class_{c}_opt_fn'] = fn

        # Calculate metrics with optimal threshold
        # 使用最优阈值计算指标
        eps = 1e-10  # Small epsilon to avoid division by zero / 避免除以零的小常数
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        sensitivity = recall  # Same as recall / 与召回率相同
        specificity = tn / (tn + fp + eps)  # True negative rate / 真负例率

        # Matthews Correlation Coefficient calculation
        # Matthews相关系数计算
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

        # AUC and AUPRC (threshold-independent metrics)
        # AUC和AUPRC（与阈值无关的指标）
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

    # Calculate macro averages across all classes
    # 计算所有类别的宏平均
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_class_{c}_opt_recall'])

    metrics['group_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_opt_macro_recall'] = np.mean(all_rec)

    # Add standard macro/micro/weighted metrics (without _opt_ suffix) for logging compatibility
    # 添加标准的宏平均/微平均/加权平均指标（不带_opt_后缀）以便于日志记录兼容性
    metrics['group_macro_f1'] = metrics['group_opt_macro_f1']
    metrics['group_macro_precision'] = metrics['group_opt_macro_precision']
    metrics['group_macro_recall'] = metrics['group_opt_macro_recall']
    metrics['group_micro_f1'] = metrics['group_opt_macro_f1']
    metrics['group_micro_precision'] = metrics['group_opt_macro_precision']
    metrics['group_micro_recall'] = metrics['group_opt_macro_recall']
    metrics['group_weighted_f1'] = metrics['group_opt_macro_f1']
    metrics['group_weighted_precision'] = metrics['group_opt_macro_precision']
    metrics['group_weighted_recall'] = metrics['group_opt_macro_recall']

    # 4-class evaluation (if hierarchical model is used)
    # 4类评估（如果使用分层模型）
    if y_4prob is not None and y_4class is not None:
        for c in range(4):
            y_true_c = y_4class[:, c]
            y_prob_c = y_4prob[:, c]

            opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
            metrics[f'group_4class_{c}_opt_threshold'] = opt_threshold

            y_pred_c = (y_prob_c >= opt_threshold).astype(int)
            # Confusion matrix values - handle edge cases
            try:
                cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # Handle edge case where only one class is present
                    tn, fp, fn, tp = 0, 0, 0, 0
                    if len(np.unique(y_true_c)) == 1:
                        if y_true_c[0] == 1:
                            # All positive samples
                            tp = np.sum(y_pred_c == 1)
                            fn = np.sum(y_pred_c == 0)
                        else:
                            # All negative samples
                            tn = np.sum(y_pred_c == 0)
                            fp = np.sum(y_pred_c == 1)
            except:
                # Fallback to manual calculation
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
    
    /使用平衡数据集评估：1:1正负样本比例

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1 / 正样本集：y_true[:, c] == 1的所有样本
    - Negative set: randomly sampled from samples where y_true[:, c] == 0 / 负样本集：从y_true[:, c] == 0的样本中随机采样
    - Downsampling: min(|P|, |N|) / 下采样：min(|P|, |N|)

    Args:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        device: Device to run evaluation on / 运行评估的设备
        random_seed: Random seed for sampling / 用于采样的随机种子
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics for balanced evaluation / 平衡评估的指标字典
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}

    # 12-Class BalanceB Evaluation
    # 12类平衡评估
    for c in range(C):
        # Get positive and negative sample indices
        # 获取正样本和负样本索引
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx_all = np.where(y_true[:, c] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        # Handle case with no positive samples
        # 处理没有正样本的情况
        if n_pos == 0:
            for key in ['precision', 'recall', 'f1', 'accuracy', 'sensitivity',
                       'specificity', 'mcc', 'auc', 'auprc', 'tp', 'tn', 'fp', 'fn']:
                metrics[f'group_class_{c}_{key}'] = 0.0
            for key in ['opt_precision', 'opt_recall', 'opt_f1', 'opt_accuracy', 'opt_sensitivity',
                       'opt_specificity', 'opt_mcc', 'opt_auc', 'opt_auprc', 'opt_tp', 'opt_tn', 'opt_fp', 'opt_fn',
                       'opt_threshold']:
                metrics[f'group_class_{c}_{key}'] = 0.0
            continue

        # Downsample to achieve 1:1 balance
        # 下采样以实现1:1平衡
        target_count = min(n_pos, n_neg_all)
        # Sample positive indices (downsample if needed)
        # 采样正样本索引（如需要则下采样）
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        # Sample negative indices (downsample if needed, or upsample with replacement)
        # 采样负样本索引（如需要则下采样，否则使用有放回采样）
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_c = y_true[selected_idx, c]
        y_prob_c = y_prob[selected_idx, c]

        # Find optimal threshold for balanced dataset
        # 为平衡数据集找到最优阈值
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
    # 4类平衡评估
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
        
        # Get 4-class probabilities
        # 获取4类概率
        if y_4prob is not None:
            y_prob_g = y_4prob[selected_idx, g]
        else:
            # Use INDEX_TO_GROUP to convert integer index to group name
            # Use max probability across classes in the same nucleotide group
            # 使用INDEX_TO_GROUP将整数索引转换为组名
            # 使用同一核苷酸组中类别的最大概率
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

    # Calculate macro/micro/weighted averages across all classes
    # 计算所有类别的宏平均/微平均/加权平均
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

    # Calculate 4-class macro averages
    # 计算4类宏平均
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

def evaluate_balanceb_th(y_true: np.ndarray, y_prob: np.ndarray, y_4class: np.ndarray,
                        device, threshold: float = 0.5, random_seed: int = 42,
                        y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate using balanced dataset: 1:1 positive:negative ratio with fixed threshold.
    
    /使用固定阈值进行平衡数据集评估：1:1正负样本比例

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1 / 正样本集：y_true[:, c] == 1的所有样本
    - Negative set: randomly sampled from samples where y_true[:, c] == 0 / 负样本集：从y_true[:, c] == 0的样本中随机采样
    - Downsampling: min(|P|, |N|) / 下采样：min(|P|, |N|)
    - Thresholding: Use the provided threshold (default 0.5) instead of searching for optimal / 使用提供的阈值（默认0.5）而不是搜索最优阈值

    Args:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        device: Device to run evaluation on / 运行评估的设备
        threshold: Fixed threshold for all classes (default 0.5) / 所有类别的固定阈值（默认0.5）
        random_seed: Random seed for sampling / 用于采样的随机种子
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics for balanced evaluation with fixed threshold / 使用固定阈值的平衡评估指标字典
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}

    # 12-Class BalanceB Evaluation with Fixed Threshold
    # 使用固定阈值的12类平衡评估
    for c in range(C):
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx_all = np.where(y_true[:, c] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            for key in ['th_f1', 'th_precision', 'th_recall', 'th_accuracy', 'th_sensitivity',
                       'th_specificity', 'th_mcc', 'th_tp', 'th_tn', 'th_fp', 'th_fn']:
                metrics[f'group_class_{c}_{key}'] = 0.0
            continue

        target_count = min(n_pos, n_neg_all)
        pos_idx_sampled = np.random.choice(pos_idx, size=target_count, replace=False) if n_pos >= target_count else pos_idx
        neg_idx_sampled = np.random.choice(neg_idx_all, size=target_count, replace=False) if n_neg_all >= target_count else np.random.choice(neg_idx_all, size=target_count, replace=True)

        selected_idx = np.concatenate([pos_idx_sampled, neg_idx_sampled])
        y_true_c = y_true[selected_idx, c]
        y_prob_c = y_prob[selected_idx, c]

        # Use fixed threshold instead of searching for optimal
        # 使用固定阈值而不是搜索最优阈值
        y_pred_c = (y_prob_c >= threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        metrics[f'group_class_{c}_th_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_class_{c}_th_precision'] = tp / (tp + fp + eps)
        metrics[f'group_class_{c}_th_recall'] = tp / (tp + fn + eps)
        metrics[f'group_class_{c}_th_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_class_{c}_th_sensitivity'] = metrics[f'group_class_{c}_th_recall']
        metrics[f'group_class_{c}_th_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_class_{c}_th_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_class_{c}_th_tp'] = tp
        metrics[f'group_class_{c}_th_tn'] = tn
        metrics[f'group_class_{c}_th_fp'] = fp
        metrics[f'group_class_{c}_th_fn'] = fn

    # 4-Class BalanceB Evaluation with Fixed Threshold
    # 使用固定阈值的4类平衡评估
    for g in range(4):
        pos_idx = np.where(y_4class[:, g] == 1)[0]
        neg_idx_all = np.where(y_4class[:, g] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            for key in ['th_f1', 'th_precision', 'th_recall', 'th_accuracy', 'th_sensitivity',
                       'th_specificity', 'th_mcc', 'th_tp', 'th_tn', 'th_fp', 'th_fn']:
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

        # Use fixed threshold
        # 使用固定阈值
        y_pred_g = (y_prob_g >= threshold).astype(int)
        eps = 1e-10
        tp = np.sum((y_true_g == 1) & (y_pred_g == 1))
        tn = np.sum((y_true_g == 0) & (y_pred_g == 0))
        fp = np.sum((y_true_g == 0) & (y_pred_g == 1))
        fn = np.sum((y_true_g == 1) & (y_pred_g == 0))

        metrics[f'group_4class_{g}_th_f1'] = 2 * tp / (2 * tp + fp + fn + eps)
        metrics[f'group_4class_{g}_th_precision'] = tp / (tp + fp + eps)
        metrics[f'group_4class_{g}_th_recall'] = tp / (tp + fn + eps)
        metrics[f'group_4class_{g}_th_accuracy'] = (tp + tn) / (tp + tn + fp + fn + eps)
        metrics[f'group_4class_{g}_th_sensitivity'] = metrics[f'group_4class_{g}_th_recall']
        metrics[f'group_4class_{g}_th_specificity'] = tn / (tn + fp + eps)
        metrics[f'group_4class_{g}_th_mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
        metrics[f'group_4class_{g}_th_tp'] = tp
        metrics[f'group_4class_{g}_th_tn'] = tn
        metrics[f'group_4class_{g}_th_fp'] = fp
        metrics[f'group_4class_{g}_th_fn'] = fn

    # Calculate macro/micro/weighted averages
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_th_f1'])
        all_prec.append(metrics[f'group_class_{c}_th_precision'])
        all_rec.append(metrics[f'group_class_{c}_th_recall'])

    metrics['group_th_macro_f1'] = np.mean(all_f1)
    metrics['group_th_macro_precision'] = np.mean(all_prec)
    metrics['group_th_macro_recall'] = np.mean(all_rec)
    metrics['group_th_micro_f1'] = metrics['group_th_macro_f1']
    metrics['group_th_micro_precision'] = metrics['group_th_macro_precision']
    metrics['group_th_micro_recall'] = metrics['group_th_macro_recall']
    metrics['group_th_weighted_f1'] = metrics['group_th_macro_f1']
    metrics['group_th_weighted_precision'] = metrics['group_th_macro_precision']
    metrics['group_th_weighted_recall'] = metrics['group_th_macro_recall']

    # 4-class macro averages
    all_f1_4, all_prec_4, all_rec_4 = [], [], []
    for g in range(4):
        th_f1_key = f'group_4class_{g}_th_f1'
        if th_f1_key in metrics:
            all_f1_4.append(metrics[th_f1_key])
            all_prec_4.append(metrics[f'group_4class_{g}_th_precision'])
            all_rec_4.append(metrics[f'group_4class_{g}_th_recall'])

    metrics['group_4class_th_macro_f1'] = np.mean(all_f1_4)
    metrics['group_4class_th_macro_precision'] = np.mean(all_prec_4)
    metrics['group_4class_th_macro_recall'] = np.mean(all_rec_4)
    metrics['group_4class_th_micro_f1'] = metrics['group_4class_th_macro_f1']
    metrics['group_4class_th_micro_precision'] = metrics['group_4class_th_macro_precision']
    metrics['group_4class_th_micro_recall'] = metrics['group_4class_th_macro_recall']

    return metrics


def evaluate_plant_unbalance(y_true: np.ndarray, y_prob: np.ndarray, device,
                             y_4class: Optional[np.ndarray] = None,
                             random_seed: int = 42,
                             y_4prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate plant data in Unbalance mode: real distribution testing.
    
    /使用不平衡模式评估植物数据：真实分布测试
    Only evaluates classes with data: 5 (Y), 8 (m5C), 9 (m6A) / 仅评估有数据的类别：5 (Y), 8 (m5C), 9 (m6A)

    Args:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        device: Device to run evaluation on / 运行评估的设备
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        random_seed: Random seed for reproducibility / 用于可重复性的随机种子
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics with "group_plant_" prefix / 带有"group_plant_"前缀的指标字典
    """
    np.random.seed(random_seed)
    
    # Only evaluate classes with data: 5, 8, 9
    # 仅评估有数据的类别：5, 8, 9
    plant_classes = [5, 8, 9]
    N, C = y_true.shape
    metrics = {}
    for c in plant_classes:
        # Extract ground truth and probabilities for current class
        # 提取当前类别的真实标签和概率
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        # Find optimal threshold using F1 score maximization
        # 使用F1分数最大化找到最优阈值
        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_plant_class_{c}_opt_threshold'] = opt_threshold

        # Generate binary predictions using optimal threshold
        # 使用最优阈值生成二分类预测
        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        
        # Manually calculate confusion matrix values to handle edge cases
        # 手动计算混淆矩阵值以处理边界情况
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        eps = 1e-10  # Small epsilon to avoid division by zero / 避免除以零的小常数
        # Calculate precision, recall, and F1 score
        # 计算精确率、召回率和F1分数
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
    # 计算宏平均（仅针对有数据的类别）
    all_f1, all_prec, all_rec = [], [], []
    for c in plant_classes:
        all_f1.append(metrics[f'group_plant_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_plant_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_plant_class_{c}_opt_recall'])

    # Compute macro average metrics
    # 计算宏平均指标
    metrics['group_plant_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_plant_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_plant_opt_macro_recall'] = np.mean(all_rec)

    # Add standard macro/micro/weighted metrics (without _opt_ suffix) for logging compatibility
    # 添加标准的宏平均/微平均/加权平均指标（不带_opt_后缀）以便于日志记录兼容性
    metrics['group_plant_macro_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_macro_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_macro_recall'] = metrics['group_plant_opt_macro_recall']
    metrics['group_plant_micro_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_micro_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_micro_recall'] = metrics['group_plant_opt_macro_recall']
    metrics['group_plant_weighted_f1'] = metrics['group_plant_opt_macro_f1']
    metrics['group_plant_weighted_precision'] = metrics['group_plant_opt_macro_precision']
    metrics['group_plant_weighted_recall'] = metrics['group_plant_opt_macro_recall']

    # 4-class evaluation (if hierarchical model is used)
    # 4类评估（如果使用分层模型）
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
    
    /根据数据集模式（平衡或不平衡）评估AC4C数据
    
    This function determines which classes have data and evaluates them using
    appropriate method (balanced or unbalanced evaluation) based on the
    dataset_mode parameter.
    /该函数确定哪些类别有数据，并根据dataset_mode参数使用适当的方法（平衡或不平衡评估）进行评估

    Args:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        device: Device to run evaluation on / 运行评估的设备
        random_seed: Random seed for reproducibility/sampling / 用于可重复性/采样的随机种子
        dataset_mode: 'balanced' or 'unbalanced' - determines evaluation method / 'balanced'或'unbalanced' - 决定评估方法
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics with "group_ac4c_" prefix / 带有"group_ac4c_"前缀的指标字典
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    metrics = {}
    
    # Determine which classes have data
    # 确定哪些类别有数据
    valid_classes = []
    for c in range(C):
        if np.any(y_true[:, c] == 1):
            valid_classes.append(c)
    
    # 12-Class Evaluation (only for classes with data)
    # 12类评估（仅针对有数据的类别）
    for c in valid_classes:
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        if dataset_mode == 'balanced':
            # BalanceB: 1:1 positive:negative ratio / BalanceB: 1:1正负样本比例
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
            # Unbalance: use real distribution / 不平衡：使用真实分布
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
    # 4类评估
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
            # BalanceB: 1:1 positive:negative ratio / BalanceB: 1:1正负样本比例
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
            # Unbalance: use real distribution / 不平衡：使用真实分布
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
    # 计算宏平均（仅针对有数据的类别）
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
    # 4类宏平均
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
    /使用BalanceB模式评估植物数据：1:1正负样本比例
    Only evaluates classes with data: 5 (Y), 8 (m5C), 9 (m6A) / 仅评估有数据的类别：5 (Y), 8 (m5C), 9 (m6A)

    Args:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        device: Device to run evaluation on / 运行评估的设备
        random_seed: Random seed for sampling / 用于采样的随机种子
        y_4prob: 4-class predicted probabilities (N, 4) / 4类预测概率 (N, 4)

    Returns:
        Dictionary of metrics with "group_plant_" prefix / 带有"group_plant_"前缀的指标字典
    """
    np.random.seed(random_seed)
    N_test, C = y_true.shape
    plant_classes = [5, 8, 9]
    metrics = {}

    # 12-Class BalanceB Evaluation (only for classes with data)
    # 12类平衡评估（仅针对有数据的类别）
    for c in plant_classes:
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx_all = np.where(y_true[:, c] == 0)[0]

        n_pos = len(pos_idx)
        n_neg_all = len(neg_idx_all)

        if n_pos == 0:
            # Add default metrics when no positive samples exist
            # 当没有正样本时添加默认指标
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
    # 4类平衡评估
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
    # 计算宏平均（仅针对有数据的类别）
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
    # 4类宏平均
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
    
    /使用最优F1阈值对每个类别进行评估

    Args:
        model: The model to evaluate / 要评估的模型
        dataloader: DataLoader for test set / 测试集的DataLoader
        device: Device to run evaluation on / 运行评估的设备
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class) / 如果为True，模型返回元组（12类logits，4类logits）

    Returns:
        Dictionary of metrics including optimal thresholds / 包含最优阈值的指标字典
    """
    model.eval()
    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        # Iterate through batches to collect predictions
        # 遍历批次以收集预测结果
        for batch in tqdm(dataloader, desc="Computing predictions with optimal threshold", unit="batch"):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)

            if use_hierarchical:
                # Handle model returning (logits_12, logits_4) or (logits_12, logits_4, atten_weight)
                # 处理模型返回（12类logits，4类logits）或（12类logits，4类logits，注意力权重）
                if isinstance(logits, tuple) and len(logits) == 3:
                    logits, _, _ = logits
                else:
                    logits, _ = logits

            probs = torch.sigmoid(logits)
            all_y_true.append(batch.y)
            all_y_prob.append(probs)

    # Concatenate all batches and convert to numpy
    # 连接所有批次并转换为numpy格式
    y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(all_y_prob, dim=0).cpu().numpy()

    N, C = y_true.shape
    metrics = {}

    # Calculate metrics for each class
    # 为每个类别计算指标
    for c in range(C):
        y_true_c = y_true[:, c]
        y_prob_c = y_prob[:, c]

        # Find optimal threshold that maximizes F1 score
        # 找到使F1分数最大的最优阈值
        opt_threshold = find_optimal_threshold(y_true_c, y_prob_c)
        metrics[f'group_class_{c}_opt_threshold'] = opt_threshold

        # Generate binary predictions using optimal threshold
        # 使用最优阈值生成二分类预测
        y_pred_c = (y_prob_c >= opt_threshold).astype(int)
        tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
        tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
        fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
        fn = np.sum((y_true_c == 1) & (y_pred_c == 0))

        eps = 1e-10  # Small epsilon to avoid division by zero / 避免除以零的小常数
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

        # AUC and AUPRC (threshold-independent metrics)
        # AUC和AUPRC（与阈值无关的指标）
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

    # Calculate macro averages across all classes
    # 计算所有类别的宏平均
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_class_{c}_opt_recall'])

    # Compute macro average metrics
    # 计算宏平均指标
    metrics['group_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_opt_macro_recall'] = np.mean(all_rec)

    return metrics


def evaluate_4class_with_optimal_threshold(model, dataloader, device, use_hierarchical=False) -> Dict[str, float]:
    """
    Evaluate 4-class metrics using optimal F1 threshold for each class.
    
    /使用最优F1阈值评估4类指标

    Args:
        model: The model to evaluate / 要评估的模型
        dataloader: DataLoader for test set / 测试集的DataLoader
        device: Device to run evaluation on / 运行评估的设备
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class) / 如果为True，模型返回元组（12类logits，4类logits）

    Returns:
        Dictionary of metrics for 4-class evaluation / 4类评估的指标字典
    """
    if not use_hierarchical:
        return {}

    # Import INDEX_TO_GROUP for correct mapping
    # 导入INDEX_TO_GROUP以进行正确的映射
    from utils.common import INDEX_TO_GROUP
    
    model.eval()
    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        # Iterate through batches to collect predictions
        # 遍历批次以收集预测结果
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Handle model returning (logits_12, logits_4) or (logits_12, logits_4, atten_weight)
            # 处理模型返回（12类logits，4类logits）或（12类logits，4类logits，注意力权重）
            if isinstance(logits, tuple) and len(logits) == 3:
                logits_12, logits_4, _ = logits
            else:
                logits_12, logits_4 = logits

            y_12 = batch.y
            y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

            # Use INDEX_TO_GROUP to map 4-class index (0-3) to nucleotide (A, C, G, U)
            # 使用INDEX_TO_GROUP将4类索引（0-3）映射到核苷酸（A, C, G, U）
            for group_idx in range(4):
                nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U' / 'A', 'C', 'G', 或 'U'
                class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
                y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

            probs_4 = torch.sigmoid(logits_4)
            all_y_true.append(y_4.cpu())
            all_y_prob.append(probs_4.cpu())

    # Concatenate all batches and convert to numpy
    # 连接所有批次并转换为numpy格式
    y_true = torch.cat(all_y_true, dim=0).numpy()
    y_prob = torch.cat(all_y_prob, dim=0).numpy()

    N, C = y_true.shape
    metrics = {}

    # Calculate metrics for each 4-class group
    # 为每个4类组计算指标
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

        eps = 1e-10  # Small epsilon to avoid division by zero / 避免除以零的小常数
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

        # AUC and AUPRC (threshold-independent metrics)
        # AUC和AUPRC（与阈值无关的指标）
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

    # Calculate macro averages across all 4-class groups
    # 计算所有4类组的宏平均
    all_f1, all_prec, all_rec = [], [], []
    for c in range(C):
        all_f1.append(metrics[f'group_4class_{c}_opt_f1'])
        all_prec.append(metrics[f'group_4class_{c}_opt_precision'])
        all_rec.append(metrics[f'group_4class_{c}_opt_recall'])

    # Compute macro average metrics
    # 计算宏平均指标
    metrics['group_4class_opt_macro_f1'] = np.mean(all_f1)
    metrics['group_4class_opt_macro_precision'] = np.mean(all_prec)
    metrics['group_4class_opt_macro_recall'] = np.mean(all_rec)

    return metrics


def get_all_predictions(model, dataloader, device, use_hierarchical=False):
    """
    Run model inference once to get all predictions for evaluation.
    
    /运行一次模型推理以获取所有预测结果用于评估

    Args:
        model: The model to evaluate / 要评估的模型
        dataloader: DataLoader for test set / 测试集的DataLoader
        device: Device to run evaluation on / 运行评估的设备
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class) / 如果为True，模型返回元组（12类logits，4类logits）

    Returns:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        y_4prob: 4-class predicted probabilities (N, 4) - None if not hierarchical / 4类预测概率 (N, 4) - 如果不是分层模型则为None
    """
    # Import INDEX_TO_GROUP for correct mapping
    # 导入INDEX_TO_GROUP以进行正确的映射
    from utils.common import INDEX_TO_GROUP
    
    model.eval()
    all_y_true = []
    all_y_prob = []
    all_y_4prob = []

    with torch.no_grad():
        # Iterate through all batches to collect predictions
        # 遍历所有批次以收集预测结果
        for batch in tqdm(dataloader, desc="Running inference on full dataset", unit="batch"):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)

            if use_hierarchical:
                # Handle model returning (logits_12, logits_4) or (logits_12, logits_4, atten_weight)
                # 处理模型返回（12类logits，4类logits）或（12类logits，4类logits，注意力权重）
                if isinstance(logits, tuple) and len(logits) == 3:
                    logits_12, logits_4, _ = logits
                else:
                    logits_12, logits_4 = logits
                probs_12 = torch.sigmoid(logits_12)
                probs_4 = torch.sigmoid(logits_4)
                all_y_4prob.append(probs_4.cpu())
            else:
                probs_12 = torch.sigmoid(logits)

            all_y_true.append(batch.y)
            all_y_prob.append(probs_12)

    # Concatenate all batches and convert to numpy
    # 连接所有批次并转换为numpy格式
    y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(all_y_prob, dim=0).cpu().numpy()

    N, C = y_true.shape
    y_4class = np.zeros((N, 4), dtype=np.float32)
    # Use INDEX_TO_GROUP to map 4-class index (0-3) to nucleotide (A, C, G, U)
    # 使用INDEX_TO_GROUP将4类索引（0-3）映射到核苷酸（A, C, G, U）
    for group_idx in range(4):
        nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U' / 'A', 'C', 'G', 或 'U'
        class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
        y_4class[:, group_idx] = y_true[:, class_indices].max(axis=1)

    if use_hierarchical and all_y_4prob:
        y_4prob = torch.cat(all_y_4prob, dim=0).cpu().numpy()
    else:
        y_4prob = None

    return y_true, y_prob, y_4class, y_4prob


def get_all_predictions_and_attention(
    model, dataloader, device, use_hierarchical=False
):
    """
    Run model inference once to get all predictions and attention weights for Top-K evaluation.
    
    /运行一次模型推理以获取所有预测结果和注意力权重用于Top-K评估

    Args:
        model: The model to evaluate / 要评估的模型
        dataloader: DataLoader for test set / 测试集的DataLoader
        device: Device to run evaluation on / 运行评估的设备
        use_hierarchical: If True, model returns tuple (logits_12class, logits_4class) / 如果为True，模型返回元组（12类logits，4类logits）

    Returns:
        y_true: Ground truth labels (N, 12) / 真实标签 (N, 12)
        y_prob: Predicted probabilities (N, 12) / 预测概率 (N, 12)
        y_4class: 4-class ground truth labels (N, 4) / 4类真实标签 (N, 4)
        y_4prob: 4-class predicted probabilities (N, 4) - None if not hierarchical / 4类预测概率 (N, 4) - 如果不是分层模型则为None
        attn_weights: Attention weights (N, 12, 1001) - None if model doesn't support it / 注意力权重 (N, 12, 1001) - 如果模型不支持则为None
        y_site: Site-level labels (N, 1001) - None if not available / 站点级别标签 (N, 1001) - 如果不可用则为None
    """
    # Import INDEX_TO_GROUP for correct mapping
    # 导入INDEX_TO_GROUP以进行正确的映射
    from utils.common import INDEX_TO_GROUP

    model.eval()
    all_y_true = []
    all_y_prob = []
    all_y_4prob = []
    all_attn_weights = []
    all_y_site = []

    with torch.no_grad():
        # Process each batch to collect predictions and attention
        # 处理每个批次以收集预测和注意力
        for batch in dataloader:
            batch = batch.to(device)

            # Check if batch has y_site attribute
            # 检查批次是否有y_site属性
            has_y_site = hasattr(batch, 'y_site')

            # Try to get attention weights
            # 尝试获取注意力权重
            try:
                result = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                if isinstance(result, tuple):
                    if use_hierarchical and len(result) == 3:
                        # Hierarchical model with attention: (logits_12, logits_4, attn)
                        # 带注意力的分层模型：（12类logits，4类logits，注意力）
                        logits_12, logits_4, attn = result
                        all_attn_weights.append(attn.cpu())
                        # Store as tuple for unified processing
                        # 存储为元组以便统一处理
                        logits = (logits_12, logits_4)
                    elif len(result) == 2:
                        # Non-hierarchical with attention: (logits, attn)
                        # 带注意力的非分层模型：（logits，注意力）
                        logits, attn = result
                        all_attn_weights.append(attn.cpu())
                    else:
                        logits = result
                        attn = None
                else:
                    logits = result
                    attn = None
            except:
                logits = model(batch.x, batch.edge_index, batch.batch)
                attn = None

            # Process logits based on hierarchical mode
            # 根据分层模式处理logits
            if use_hierarchical:
                # Extract logits for hierarchical mode
                # 提取分层模式的logits
                if isinstance(logits, tuple) and len(logits) >= 2:
                    # Have tuple with at least 2 elements
                    # 有至少2个元素的元组
                    logits_12, logits_4 = logits[0], logits[1]
                elif isinstance(logits, tuple) and len(logits) == 1:
                    # Have tuple with 1 element - use it
                    # 有1个元素的元组 - 使用它
                    logits_12 = logits[0]
                    logits_4 = None
                elif not isinstance(logits, tuple):
                    # Have single tensor
                    # 有单个张量
                    logits_12 = logits
                    logits_4 = None
                else:
                    # Empty tuple or other unexpected case - skip this batch
                    # 空元组或其他意外情况 - 跳过此批次
                    continue
                
                probs_12 = torch.sigmoid(logits_12)
                if logits_4 is not None:
                    probs_4 = torch.sigmoid(logits_4)
                    all_y_4prob.append(probs_4.cpu())
            else:
                # Non-hierarchical mode
                # 非分层模式
                # Extract logits for non-hierarchical mode
                # 提取非分层模式的logits
                if isinstance(logits, tuple) and len(logits) > 0:
                    # Have tuple - take first element
                    # 有元组 - 取第一个元素
                    logits_12 = logits[0]
                elif not isinstance(logits, tuple):
                    # Have single tensor
                    # 有单个张量
                    logits_12 = logits
                else:
                    # Empty tuple or other unexpected case - skip this batch
                    # 空元组或其他意外情况 - 跳过此批次
                    continue
                
                probs_12 = torch.sigmoid(logits_12)

            all_y_true.append(batch.y)
            all_y_prob.append(probs_12)

            if has_y_site:
                all_y_site.append(batch.y_site)

    # Concatenate all batches and convert to numpy
    # 连接所有批次并转换为numpy格式
    y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(all_y_prob, dim=0).cpu().numpy()

    N, C = y_true.shape
    y_4class = np.zeros((N, 4), dtype=np.float32)
    # Use INDEX_TO_GROUP to map 4-class index (0-3) to nucleotide (A, C, G, U)
    # 使用INDEX_TO_GROUP将4类索引（0-3）映射到核苷酸（A, C, G, U）
    for group_idx in range(4):
        nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U' / 'A', 'C', 'G', 或 'U'
        class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
        y_4class[:, group_idx] = y_true[:, class_indices].max(axis=1)

    if use_hierarchical and all_y_4prob:
        y_4prob = torch.cat(all_y_4prob, dim=0).cpu().numpy()
    else:
        y_4prob = None

    if all_attn_weights:
        attn_weights = torch.cat(all_attn_weights, dim=0)
    else:
        attn_weights = None

    if all_y_site:
        y_site = torch.cat(all_y_site, dim=0)
    else:
        y_site = None

    return y_true, y_prob, y_4class, y_4prob, attn_weights, y_site


# ============================================================================
# Group-Based Balanced Evaluation
# ============================================================================

def gen3_evaluate_group_balanceb(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_4class: np.ndarray,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate using balanced dataset: Positive vs Pure Negative samples.

    For each class c:
    - Positive set: all samples where y_true[:, c] == 1
    - Negative set: randomly sampled from samples with NO modifications at all (row sum == 0)
    - Downsampling: min(|P|, |N|)
    - Balance: 1:1 positive:negative ratio

    Example: For Am (class 0):
      - Positive: sequences with Am
      - Negative: sequences with NO modifications (pure negative samples)

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

    # Identify Pure Negative Samples (Rows where sum is 0)
    # y_true shape is (N, 12)
    pure_negative_indices = np.where(y_true.sum(axis=1) == 0)[0]
    n_pure_neg = len(pure_negative_indices)

    # 12-Class Group-Balanced Evaluation
    for c in range(C):
        # Positive samples: sequences with this class
        pos_idx = np.where(y_true[:, c] == 1)[0]

        # Negative samples: Pure negatives (sequences with NO modifications)
        neg_idx_all = pure_negative_indices
        n_neg_all = n_pure_neg

        n_pos = len(pos_idx)

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

        # AUC and AUPRC (threshold-independent metrics)
        # AUC和AUPRC（与阈值无关的指标）
        if len(np.unique(y_true_c)) > 1:
            try:
                metrics[f'group_plant_class_{c}_auc'] = roc_auc_score(y_true_c, y_prob_c)
                metrics[f'group_plant_class_{c}_auprc'] = average_precision_score(y_true_c, y_prob_c)
            except ValueError:
                # Handle cases where AUC calculation fails
                # 处理AUC计算失败的情况
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
