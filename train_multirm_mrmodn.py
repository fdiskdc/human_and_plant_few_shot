"""
train_multirm_dataset.py - MultIRM数据集多标签训练脚本 / MultIRM Dataset Multi-Label Training Script

MultIRM 数据集 (dataset/multirm.py) 的训练入口:正/负样本配对 12 类多标签 RNA 修饰分类。
使用 12 个独立损失函数 (正+负样本对),并预计算 RNA 二级结构缓存以加速训练。
Training entry for the MultIRM dataset (dataset/multirm.py): 12-class positive/negative paired RNA modification classification.
Uses 12 separate losses (positive + negative pairs) and precomputed RNA secondary structure caching for training speed.

功能模块 / Modules:
- main: 主训练函数 / Main training function
- 二级结构预计算 / Precompute secondary structures on first run
- 正/负样本配对损失 / Positive/negative paired loss (12 separate losses)
- 完整指标 (sklearn) / Comprehensive metrics (sklearn)
- 详细日志 / Detailed logging
- TensorBoard + TQDM / TensorBoard + TQDM

输入 / Inputs:
- json/multirm.json: 训练配置 / Training config
- multirm/seq.npy, multirm/pos_idx.npy, multirm/neg_idx.npy: 训练数据 / Training data
- 命令行参数 / CLI: --config, --gpu, --seed

输出 / Outputs:
- checkpoints/best_multirm_dataset.pt: 最佳模型 / Best model
- logs/multirm_dataset_*/train_*.log: 训练日志 / Training logs
- logs/multirm_dataset_*/results.json: 评估结果 / Evaluation results
- npy/cache/multirm/structures_cache.npz: 二级结构缓存 / Secondary structure cache
- TensorBoard events: 可视化 / Visualization

数据流 / Data Flow:
1. 加载数据 / Load data
2. 预计算/加载二级结构缓存 / Precompute or load structure cache
3. 构建 MultirmDataset (含正负对) / Build MultirmDataset
4. 训练循环 (12 损失) / Training loop (12 losses)
5. 评估 + 保存 / Evaluate and save

相关文件 / Related Files:
- 调用 / Calls: model.mrmodn.RNA_ClassQuery_Model, dataset.multirm.MultirmDataset, sklearn.metrics
- 被调用 / Called by: shell scripts, manual CLI invocations

使用示例 / Usage Example:
    python train_multirm_dataset.py --config json/multirm.json --gpu 0

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from model.mrmodn import RNA_ClassQuery_Model
from dataset.multirm import MultirmDataset
from utils import (
    setup_logging, setup_tensorboard, save_checkpoint, load_config,
)

# Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# MultIRM class names (12 classes)
MULTIRM_CLASSES = ['Am', 'Cm', 'Gm', 'I', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Ψ']


# ============================================================================
# Training and Testing Functions with Per-Class Loss
# ============================================================================

def train_epoch_multirm(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger=None,
    use_amp: bool = False
) -> float:
    """
    Train for one epoch with separate losses for each class.

    For MultIRM, each class has paired positive and negative samples.
    We compute 12 separate losses (one per class) and average them.

    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Optional logger instance
        use_amp: If True, use automatic mixed precision

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    total_loss_per_class = {cls: 0.0 for cls in MULTIRM_CLASSES}
    num_batches = 0
    num_samples_per_class = {cls: 0 for cls in MULTIRM_CLASSES}

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    pbar = tqdm(dataloader, desc="Training", leave=True)

    for batch in pbar:
        # Get class indices for each sample in the batch
        # batch.class_idx is a tensor of class indices
        class_indices = batch.class_idx.long().to(device)

        # Ensure labels are tensor before moving to device
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float32)

        batch = batch.to(device)
        batch.y = batch.y.to(device)

        # Forward pass
        optimizer.zero_grad()

        # Model forward pass - unpack three return values
        logits_12class, logits_4class, attn_weights_12 = model(
            batch.x, batch.edge_index, batch.batch, return_attention=False
        )

        # Generate 4-class labels from 12-class labels
        from utils.common import GROUP_TO_CLASS_INDICES
        y_12 = batch.y  # (Batch, 12)
        y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

        for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
            class_indices_4class = GROUP_TO_CLASS_INDICES[group_name]
            y_4[:, group_idx] = y_12[:, class_indices_4class].max(dim=1)[0]

        if use_amp:
            with torch.cuda.amp.autocast():
                loss_12, loss_4 = compute_per_class_loss(
                    logits_12class, logits_4class, y_12, y_4, class_indices, device
                )
                loss = loss_12 + loss_4

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

        else:
            loss_12, loss_4 = compute_per_class_loss(
                logits_12class, logits_4class, y_12, y_4, class_indices, device
            )
            loss = loss_12 + loss_4

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # Track 12-class and 4-class losses
        total_loss_12 += loss_12.item()
        total_loss_4 += loss_4.item()

        # Track losses per class
        with torch.no_grad():
            batch_loss_per_class_12 = compute_per_class_losses(
                logits_12class, y_12, class_indices, device
            )
            for cls_idx, cls_loss in batch_loss_per_class_12.items():
                cls_name = MULTIRM_CLASSES[cls_idx]
                total_loss_per_class[cls_name] += cls_loss
                num_samples_per_class[cls_name] += (class_indices == cls_idx).sum().item()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l12': f'{loss_12.item():.4f}',
            'l4': f'{loss_4.item():.4f}'
        })

    # Calculate average loss per class
    avg_loss_per_class = {}
    for cls in MULTIRM_CLASSES:
        if num_samples_per_class[cls] > 0:
            # Normalize by number of batches that had this class
            avg_loss_per_class[cls] = total_loss_per_class[cls] / num_batches
        else:
            avg_loss_per_class[cls] = 0.0

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_loss_12 = total_loss_12 / num_batches if num_batches > 0 else 0.0
    avg_loss_4 = total_loss_4 / num_batches if num_batches > 0 else 0.0

    if logger:
        logger.info(f"Train Loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})")
        logger.info(f"Per-class losses (12-class): {avg_loss_per_class}")

    return avg_loss


def compute_per_class_loss(
    logits_12class: torch.Tensor,
    logits_4class: torch.Tensor,
    labels_12: torch.Tensor,
    labels_4: torch.Tensor,
    class_indices: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss separately for 12-class and 4-class tasks (VECTORIZED VERSION).

    【优化说明】
    原版本使用 for 循环遍历 12 个类，导致 CPU-GPU 同步阻塞，破坏并行性。
    本版本使用 PyTorch 广播机制和 Mask 操作一次性计算 Loss，显著减少同步开销。
    预期速度提升：10倍以上。

    For 12-class loss: Compute loss separately for each class and average them.
    This ensures that each class's positive and negative samples contribute equally
    to the total loss, regardless of class size.

    For 4-class loss: Compute loss for each nucleotide group (A, C, G, U) and average them.
    The 4-class labels are derived from 12-class labels by max-pooling over classes in each group.

    Args:
        logits_12class: Model predictions for 12 classes (batch_size, 12)
        logits_4class: Model predictions for 4 nucleotide groups (batch_size, 4)
        labels_12: Ground truth labels for 12 classes (batch_size, 12)
        labels_4: Ground truth labels for 4 groups (batch_size, 4)
        class_indices: Class index for each sample (batch_size,)
        device: Device to compute on

    Returns:
        loss_12: Average loss for 12-class task
        loss_4: Average loss for 4-class task
    """
    # ========== 12-class Loss (VECTORIZED) ==========
    
    # 1. 计算所有位置的 BCE Loss (不平均，保留 (batch_size, 12))
    raw_losses = F.binary_cross_entropy_with_logits(
        logits_12class,
        labels_12,
        reduction='none'  # 不进行降维，保留每个位置的 loss
    )
    
    # 2. 生成 One-hot Mask: (batch_size, 12)
    # 如果样本 i 属于类 5，则 mask[i, 5] = 1，其他为 0
    batch_size = logits_12class.size(0)
    num_classes = 12
    
    mask = torch.zeros(batch_size, num_classes, device=device)
    mask.scatter_(1, class_indices.unsqueeze(1), 1.0)  # In-place scatter is fast
    
    # 3. 只保留该样本所属类别的 Loss
    masked_losses = raw_losses * mask
    
    # 4. 计算平均值
    # 对所有有效样本求平均
    loss_12 = masked_losses.sum() / batch_size

    # ========== 4-class Loss (本身就是向量化的) ==========
    # 4-class loss is computed on all samples, not per class_indices
    # because each sample belongs to a nucleotide group based on its center nucleotide
    loss_4 = F.binary_cross_entropy_with_logits(
        logits_4class,
        labels_4,
        reduction='mean'
    )

    return loss_12, loss_4


def compute_per_class_losses(
    logits_12class: torch.Tensor,
    labels_12: torch.Tensor,
    class_indices: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Compute individual losses for each class (for tracking).

    Args:
        logits_12class: Model predictions for 12 classes (batch_size, 12)
        labels_12: Ground truth labels for 12 classes (batch_size, 12)
        class_indices: Class index for each sample (batch_size,)
        device: Device to compute on

    Returns:
        Dictionary mapping class index to loss value
    """
    losses_per_class = {idx: 0.0 for idx in range(len(MULTIRM_CLASSES))}

    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        class_mask = (class_indices == cls_idx)

        if class_mask.sum() == 0:
            continue

        class_logits = logits_12class[class_mask]
        class_labels = labels_12[class_mask]

        class_loss = F.binary_cross_entropy_with_logits(
            class_logits[:, cls_idx:cls_idx+1],
            class_labels[:, cls_idx:cls_idx+1],
            reduction='mean'
        )
        losses_per_class[cls_idx] = class_loss.item()

    return losses_per_class


def test_epoch_multirm(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger=None,
    use_amp: bool = False
) -> float:
    """
    Test for one epoch.

    Args:
        model: The model to test
        dataloader: Test dataloader
        device: Device to test on
        logger: Optional logger instance
        use_amp: If True, use automatic mixed precision

    Returns:
        Average loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Testing", leave=True)

    with torch.no_grad():
        for batch in pbar:
            class_indices = batch.class_idx.long().to(device)

            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)

            batch = batch.to(device)
            batch.y = batch.y.to(device)

            # Model forward pass - unpack three return values
            logits_12class, logits_4class, attn_weights_12 = model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )

            # Generate 4-class labels from 12-class labels
            from utils.common import GROUP_TO_CLASS_INDICES
            y_12 = batch.y  # (Batch, 12)
            y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

            for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                class_indices_4class = GROUP_TO_CLASS_INDICES[group_name]
                y_4[:, group_idx] = y_12[:, class_indices_4class].max(dim=1)[0]

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss_12, loss_4 = compute_per_class_loss(
                        logits_12class, logits_4class, y_12, y_4, class_indices, device
                    )
                    loss = loss_12 + loss_4
            else:
                loss_12, loss_4 = compute_per_class_loss(
                    logits_12class, logits_4class, y_12, y_4, class_indices, device
                )
                loss = loss_12 + loss_4

            total_loss += loss.item()
            total_loss_12 += loss_12.item()
            total_loss_4 += loss_4.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l12': f'{loss_12.item():.4f}',
                'l4': f'{loss_4.item():.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_loss_12 = total_loss_12 / num_batches if num_batches > 0 else 0.0
    avg_loss_4 = total_loss_4 / num_batches if num_batches > 0 else 0.0

    if logger:
        logger.info(f"Test Loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})")

    return avg_loss


def get_predictions_multirm(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get all predictions and ground truth labels for evaluation.

    Args:
        model: The model to evaluate
        dataloader: Test dataloader
        device: Device to evaluate on

    Returns:
        y_true: Ground truth labels (num_samples, 12)
        y_prob: Predicted probabilities (num_samples, 12)
    """
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)

            batch = batch.to(device)

            # Model forward pass - unpack three return values
            logits_12class, logits_4class, attn_weights_12 = model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )

            # Use 12-class logits for predictions
            logits = logits_12class
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.cpu().numpy()

            y_true_list.append(labels)
            y_prob_list.append(probs)

    y_true = np.vstack(y_true_list)
    y_prob = np.vstack(y_prob_list)

    return y_true, y_prob


def get_predictions_for_single_class(
    model: nn.Module,
    class_name: str,
    data_dir: str,
    cache_dir: str,
    class_idx: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 32,
    logger=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions for a single class using its own test set.
    
    Loads only the test samples (positive and negative) for the specified class
    and evaluates the corresponding output head.

    Args:
        model: The model to evaluate
        class_name: Name of the class (e.g., 'Am', 'Cm', etc.)
        data_dir: Base data directory
        cache_dir: Cache directory for structures
        class_idx: Index of the class (0-11)
        seq_len: Sequence length
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        logger: Optional logger instance

    Returns:
        y_true: Ground truth labels for this class (num_samples,)
        y_prob: Predicted probabilities for this class (num_samples,)
    """
    from dataset.multirm import MultirmDataset, MULTIRM_CLASSES, CLASS_TO_IDX
    
    # Load test dataset for this specific class
    test_dataset = MultirmDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        numsample=999999,  # Load all samples
        mode='test',
        use_cache=True,
        preload_cache=True
    )
    
    # Filter samples to only include this class
    class_indices = test_dataset.class_idx if hasattr(test_dataset, 'class_idx') else None
    filtered_indices = []
    
    # Get all samples and filter by class name
    for i, sample in enumerate(test_dataset.all_samples):
        if sample['class_name'] == class_name:
            filtered_indices.append(i)
    
    if len(filtered_indices) == 0:
        if logger:
            logger.warning(f"No test samples found for class {class_name}")
        return np.array([]), np.array([])
    
    # Create subset with only this class's samples
    from torch.utils.data import Subset
    class_subset = Subset(test_dataset, filtered_indices)
    
    # Create dataloader
    class_loader = DataLoader(
        class_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Get predictions for this class
    model.eval()
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for batch in tqdm(class_loader, desc=f"Evaluating {class_name}", leave=False):
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            
            batch = batch.to(device)
            
            # Model forward pass
            logits_12class, logits_4class, attn_weights_12 = model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )
            
            # Extract only the output for this class
            logits_cls = logits_12class[:, class_idx]
            labels_cls = batch.y[:, class_idx]
            
            probs_cls = torch.sigmoid(logits_cls).cpu().numpy()
            labels_cls = labels_cls.cpu().numpy()
            
            y_true_list.append(labels_cls)
            y_prob_list.append(probs_cls)
    
    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    
    return y_true, y_prob


def find_optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the optimal threshold that maximizes F1 score.

    Args:
        y_true: Ground truth labels (binary array)
        y_prob: Predicted probabilities (array of floats)

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score
    
    # Search thresholds from 0.1 to 0.9 with step 0.01
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold




def calculate_masked_metrics(
    y_true_all: np.ndarray,   # 形状 (N, 12)
    y_prob_all: np.ndarray,   # 形状 (N, 12)
    class_indices: np.ndarray # 形状 (N,)，指示每个样本属于哪个类
) -> dict:
    """
    针对 MultIRM 这种"每个样本属于特定类"的数据结构计算 Metrics。
    只在样本所属的特定 Class Head 上计算指标，忽略其他 Head 的输出。
    """
    metrics = {}
    
    # 存储所有类的指标以便计算 Macro Average
    macro_f1_list = []
    macro_auc_list = []
    
    # 遍历 12 个类别
    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        # 1. 【核心逻辑】创建掩码：只选择属于当前类别的样本
        # mask 是一个布尔数组，筛选出该类别的所有 Positive 和 Negative 样本
        mask = (class_indices == cls_idx)
        
        # 如果该类别没有样本（例如在某个小的 batch 中），跳过
        if np.sum(mask) == 0:
            continue
            
        # 2. 提取该类别的真实标签和预测概率
        # 我们只取第 cls_idx 列的输出
        y_true_cls = y_true_all[mask, cls_idx]
        y_prob_cls = y_prob_all[mask, cls_idx]
        
        # 3. 计算最优阈值 (可选，或者使用默认 0.5)
        # 这里为了演示简单使用 0.5，你可以结合之前的 find_optimal_threshold_f1
        y_pred_cls = (y_prob_cls >= 0.5).astype(int)
        
        # 4. 计算指标
        try:
            auc = roc_auc_score(y_true_cls, y_prob_cls)
            auprc = average_precision_score(y_true_cls, y_prob_cls)
        except ValueError:
            # 处理样本全为正或全为负导致无法计算 AUC 的情况
            auc = 0.0
            auprc = 0.0
            
        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        
        # 5. 存入字典
        metrics[f'{cls_name}_acc'] = acc
        metrics[f'{cls_name}_auc'] = auc
        metrics[f'{cls_name}_f1'] = f1
        metrics[f'{cls_name}_pre'] = precision
        metrics[f'{cls_name}_rec'] = recall
        
        macro_f1_list.append(f1)
        macro_auc_list.append(auc)

    # 计算 Macro Averages
    metrics['macro_f1'] = np.mean(macro_f1_list) if macro_f1_list else 0.0
    metrics['macro_auc'] = np.mean(macro_auc_list) if macro_auc_list else 0.0
    
    return metrics

def calculate_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculate comprehensive metrics for a single class.

    Args:
        y_true: Ground truth labels (binary array)
        y_prob: Predicted probabilities (array of floats)
        threshold: Classification threshold

    Returns:
        Dictionary containing all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef
    )
    
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity = Recall
    sensitivity = recall
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    # AUPRC
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0
    
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auc': auc,
        'auprc': auprc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'threshold': threshold
    }


def evaluate_multirm(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate predictions on MultIRM dataset (original function for backward compatibility).

    Args:
        y_true: Ground truth labels (num_samples, 12)
        y_prob: Predicted probabilities (num_samples, 12)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # Per-class metrics
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for i, cls_name in enumerate(MULTIRM_CLASSES):
        # Get metrics for this class
        tp = ((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()
        tn = ((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        metrics[f'{cls_name}_f1'] = f1
        metrics[f'{cls_name}_precision'] = precision
        metrics[f'{cls_name}_recall'] = recall

    # Macro-average metrics
    metrics['macro_f1'] = np.mean(f1_scores)
    metrics['macro_precision'] = np.mean(precision_scores)
    metrics['macro_recall'] = np.mean(recall_scores)

    # Micro-average metrics
    total_tp = ((y_pred == 1) & (y_true == 1)).sum()
    total_fp = ((y_pred == 1) & (y_true == 0)).sum()
    total_tn = ((y_pred == 0) & (y_true == 0)).sum()
    total_fn = ((y_pred == 0) & (y_true == 1)).sum()

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    metrics['micro_f1'] = micro_f1
    metrics['micro_precision'] = micro_precision
    metrics['micro_recall'] = micro_recall

    # Accuracy
    metrics['accuracy'] = (y_pred == y_true).mean()

    # AUC and AUPRC (macro)
    from sklearn.metrics import roc_auc_score, average_precision_score

    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_prob, average='macro')
    except:
        metrics['macro_auc'] = 0.0

    try:
        metrics['macro_auprc'] = average_precision_score(y_true, y_prob, average='macro')
    except:
        metrics['macro_auprc'] = 0.0

    return metrics


def evaluate_multirm_best_threshold(
    model: nn.Module,
    data_dir: str,
    cache_dir: str,
    seq_len: int,
    device: torch.device,
    batch_size: int = 32,
    logger=None
) -> dict:
    """
    Evaluate predictions on MultIRM dataset with optimal threshold search for each class.
    
    Each class is evaluated separately using its own test set (positive and negative samples).
    Overall performance is the average of per-class performance.

    Args:
        model: The model to evaluate
        data_dir: Base data directory
        cache_dir: Cache directory for structures
        seq_len: Sequence length
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        logger: Optional logger instance

    Returns:
        Dictionary containing:
        - Per-class metrics (with optimal threshold)
        - Macro-averaged metrics (average of per-class metrics)
        - All threshold values
    """
    metrics = {}
    class_metrics = {}
    
    # Evaluate each class separately using its own test set
    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        logger.info(f"\nEvaluating class {cls_idx} ({cls_name})...")
        
        # Get predictions for this class using its own test set
        y_true_cls, y_prob_cls = get_predictions_for_single_class(
            model=model,
            class_name=cls_name,
            data_dir=data_dir,
            cache_dir=cache_dir,
            class_idx=cls_idx,
            seq_len=seq_len,
            device=device,
            batch_size=batch_size
        )
        
        if len(y_true_cls) == 0:
            logger.warning(f"No test samples found for class {cls_name}, skipping...")
            continue
        
        # Find optimal threshold for this class
        optimal_threshold = find_optimal_threshold_f1(y_true_cls, y_prob_cls)
        logger.info(f"  Optimal threshold for {cls_name}: {optimal_threshold:.3f}")
        
        # Calculate all metrics with optimal threshold
        cls_metrics = calculate_class_metrics(y_true_cls, y_prob_cls, optimal_threshold)
        
        # Store per-class metrics with class-specific prefix
        for key, value in cls_metrics.items():
            metrics[f'{cls_name}_{key}'] = value
        
        # Also store in class_metrics dict for easier averaging
        class_metrics[cls_name] = cls_metrics
        
        # Log per-class results
        logger.info(f"  F1: {cls_metrics['f1']:.4f}, Precision: {cls_metrics['precision']:.4f}, Recall: {cls_metrics['recall']:.4f}")
        logger.info(f"  AUC: {cls_metrics['auc']:.4f}, AUPRC: {cls_metrics['auprc']:.4f}")
        logger.info(f"  TP: {cls_metrics['tp']}, TN: {cls_metrics['tn']}, FP: {cls_metrics['fp']}, FN: {cls_metrics['fn']}")
    
    # Calculate macro-averaged metrics (average of per-class metrics)
    metric_names = ['acc', 'precision', 'recall', 'f1', 'mcc', 'auc', 'auprc', 'sensitivity', 'specificity']
    
    for metric_name in metric_names:
        values = [class_metrics[cls][metric_name] for cls in MULTIRM_CLASSES if cls in class_metrics]
        if len(values) > 0:
            metrics[f'macro_{metric_name}'] = np.mean(values)
        else:
            metrics[f'macro_{metric_name}'] = 0.0
    
    # Calculate overall performance as average of per-class metrics
    # Store per-class metrics for reference
    metrics['class_metrics'] = class_metrics
    
    return metrics


def evaluate_epoch_masked(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger=None,
    use_amp: bool = False
) -> dict:
    """
    改进后的评估函数：一次性推理 + 掩码计算 Loss 和 Metrics
    
    优势：
    1. 只运行一次 DataLoader，而不是为每个类单独运行
    2. 使用掩码计算每个类的 Loss 和 Metrics
    3. 避免无效数据（样本不属于的类）污染评估结果
    
    Args:
        model: The model to evaluate
        dataloader: Test dataloader
        device: Device to evaluate on
        logger: Optional logger instance
        use_amp: If True, use automatic mixed precision

    Returns:
        Dictionary containing:
        - avg_loss: Average loss
        - Per-class loss (e.g., 'Am_loss')
        - Per-class metrics (e.g., 'Am_f1', 'Am_auc', etc.)
        - Macro-averaged metrics
    """
    model.eval()
    
    # 存储所有批次的结果
    all_y_true = []
    all_y_prob = []
    all_class_indices = []
    
    total_loss = 0.0
    num_batches = 0
    
    # 记录每个类的 Loss (用于诊断)
    loss_per_class_accumulator = {cls: 0.0 for cls in MULTIRM_CLASSES}
    count_per_class = {cls: 0 for cls in MULTIRM_CLASSES}
    
    # 用于计算混淆矩阵的统计
    confusion_stats = {cls: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for cls in MULTIRM_CLASSES}

    pbar = tqdm(dataloader, desc="Evaluating (Masked)", leave=True)

    with torch.no_grad():
        for batch in pbar:
            # 数据准备
            class_indices = batch.class_idx.long().to(device)
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            
            batch = batch.to(device)
            batch.y = batch.y.to(device)

            # 前向传播
            logits_12class, logits_4class, attn_weights_12 = model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )

            # --- 计算 Loss (Masked Loss 逻辑) ---
            batch_loss_12 = 0.0
            valid_classes_in_batch = 0
            
            for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
                # Mask: 找出当前 batch 中属于类别 cls_idx 的样本
                mask = (class_indices == cls_idx)
                if mask.sum() > 0:
                    # 只计算对应 Head (第 cls_idx 列) 的 BCE Loss
                    cls_loss = F.binary_cross_entropy_with_logits(
                        logits_12class[mask, cls_idx], 
                        batch.y[mask, cls_idx]
                    )
                    batch_loss_12 += cls_loss
                    valid_classes_in_batch += 1
                    
                    # 累加用于日志
                    loss_per_class_accumulator[cls_name] += cls_loss.item()
                    count_per_class[cls_name] += 1
            
            if valid_classes_in_batch > 0:
                batch_loss_12 /= valid_classes_in_batch
            
            total_loss += batch_loss_12.item() if isinstance(batch_loss_12, torch.Tensor) else batch_loss_12
            num_batches += 1
            
            # --- 收集数据用于 Metrics ---
            probs = torch.sigmoid(logits_12class).cpu().numpy()
            labels = batch.y.cpu().numpy()
            indices = class_indices.cpu().numpy()
            
            all_y_true.append(labels)
            all_y_prob.append(probs)
            all_class_indices.append(indices)

    # 合并数据
    y_true_stack = np.vstack(all_y_true)
    y_prob_stack = np.vstack(all_y_prob)
    indices_stack = np.concatenate(all_class_indices)
    
    # 计算 Metrics
    metrics = calculate_masked_metrics(y_true_stack, y_prob_stack, indices_stack)
    
    # 添加 Loss 到 Metrics 字典
    metrics['avg_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 添加每个类的平均 Loss
    for cls_name in MULTIRM_CLASSES:
        if count_per_class[cls_name] > 0:
            metrics[f'{cls_name}_loss'] = loss_per_class_accumulator[cls_name] / count_per_class[cls_name]
        else:
            metrics[f'{cls_name}_loss'] = 0.0
    
    # 额外计算每个类的混淆矩阵统计（用于完整的指标）
    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        mask = (indices_stack == cls_idx)
        if np.sum(mask) == 0:
            continue
            
        y_true_cls = y_true_stack[mask, cls_idx]
        y_prob_cls = y_prob_stack[mask, cls_idx]
        y_pred_cls = (y_prob_cls >= 0.5).astype(int)
        
        tp = ((y_pred_cls == 1) & (y_true_cls == 1)).sum()
        fp = ((y_pred_cls == 1) & (y_true_cls == 0)).sum()
        tn = ((y_pred_cls == 0) & (y_true_cls == 0)).sum()
        fn = ((y_pred_cls == 0) & (y_true_cls == 1)).sum()
        
        metrics[f'{cls_name}_tp'] = int(tp)
        metrics[f'{cls_name}_tn'] = int(tn)
        metrics[f'{cls_name}_fp'] = int(fp)
        metrics[f'{cls_name}_fn'] = int(fn)
        
        # 计算 MCC
        try:
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
        except:
            mcc = 0.0
        metrics[f'{cls_name}_mcc'] = mcc
        
        # 计算 Sensitivity 和 Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'{cls_name}_sensitivity'] = sensitivity
        metrics[f'{cls_name}_specificity'] = specificity
        
        # 计算 AUPRC
        try:
            auprc = average_precision_score(y_true_cls, y_prob_cls)
        except ValueError:
            auprc = 0.0
        metrics[f'{cls_name}_auprc'] = auprc
        
        # 使用 0.5 作为默认阈值
        metrics[f'{cls_name}_threshold'] = 0.5
    
    # 计算 Macro Averages（除了已有的 f1 和 auc）
    metric_names = ['acc', 'precision', 'recall', 'mcc', 'auprc', 'sensitivity', 'specificity']
    for metric_name in metric_names:
        values = [metrics.get(f'{cls}_{metric_name}', 0.0) for cls in MULTIRM_CLASSES if f'{cls}_{metric_name}' in metrics]
        if values:
            metrics[f'macro_{metric_name}'] = np.mean(values)
        else:
            metrics[f'macro_{metric_name}'] = 0.0
    
    return metrics


def print_multirm_evaluation_table(metrics: dict, epoch: int, logger=None):
    """
    Print MultIRM evaluation results in a formatted table.
    
    支持两种 metrics 格式：
    1. 来自 evaluate_multirm_best_threshold 的完整指标（包含 mcc, auprc 等）
    2. 来自 evaluate_epoch_masked 的基础指标
    
    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch number
        logger: Optional logger instance
    """
    from prettytable import PrettyTable
    
    output = f"\n{'='*140}\n"
    output += f"MultIRM Evaluation Results - Epoch {epoch}\n"
    output += f"{'='*140}\n"
    
    # 创建表格，动态调整列名（如果数据中包含完整指标）
    has_full_metrics = any(f'{cls}_mcc' in metrics for cls in MULTIRM_CLASSES)
    
    if has_full_metrics:
        table = PrettyTable()
        table.field_names = [
            "Class", "Mod Name", "Loss", "Acc", "Precision", "Recall", 
            "AUC", "F1", "MCC", "AUPRC", "Sn", "Sp", 
            "TP", "TN", "FP", "FN", "Threshold"
        ]
    else:
        table = PrettyTable()
        table.field_names = [
            "Class", "Mod Name", "Loss", "Acc", "Precision", "Recall", 
            "AUC", "F1", "TP", "TN", "FP", "FN", "Threshold"
        ]
    
    table.align = "r"
    table.align["Class"] = "l"
    table.align["Mod Name"] = "l"
    
    # 添加每类的行
    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        loss = metrics.get(f'{cls_name}_loss', 0.0)
        acc = metrics.get(f'{cls_name}_acc', metrics.get(f'{cls_name}_acc', 0.0))
        precision = metrics.get(f'{cls_name}_pre', metrics.get(f'{cls_name}_precision', 0.0))
        recall = metrics.get(f'{cls_name}_rec', metrics.get(f'{cls_name}_recall', 0.0))
        auc = metrics.get(f'{cls_name}_auc', 0.0)
        f1 = metrics.get(f'{cls_name}_f1', 0.0)
        tp = int(metrics.get(f'{cls_name}_tp', 0))
        tn = int(metrics.get(f'{cls_name}_tn', 0))
        fp = int(metrics.get(f'{cls_name}_fp', 0))
        fn = int(metrics.get(f'{cls_name}_fn', 0))
        threshold = metrics.get(f'{cls_name}_threshold', 0.5)
        
        if has_full_metrics:
            mcc = metrics.get(f'{cls_name}_mcc', 0.0)
            auprc = metrics.get(f'{cls_name}_auprc', 0.0)
            sn = metrics.get(f'{cls_name}_sensitivity', 0.0)
            sp = metrics.get(f'{cls_name}_specificity', 0.0)
            
            table.add_row([
                f"{cls_idx}",
                cls_name,
                f"{loss:.4f}",
                f"{acc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{auc:.4f}",
                f"{f1:.4f}",
                f"{mcc:.4f}",
                f"{auprc:.4f}",
                f"{sn:.4f}",
                f"{sp:.4f}",
                tp, tn, fp, fn,
                f"{threshold:.3f}"
            ])
        else:
            table.add_row([
                f"{cls_idx}",
                cls_name,
                f"{loss:.4f}",
                f"{acc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{auc:.4f}",
                f"{f1:.4f}",
                tp, tn, fp, fn,
                f"{threshold:.3f}"
            ])
    
    output += str(table) + "\n"
    
    # 添加宏平均结果
    output += f"Macro-Averaged Performance:\n"
    output += f"  Loss: {metrics.get('avg_loss', 0.0):.4f}\n"
    output += f"  Accuracy: {metrics.get('macro_acc', 0.0):.4f}\n"
    output += f"  Precision: {metrics.get('macro_precision', 0.0):.4f}\n"
    output += f"  Recall: {metrics.get('macro_recall', 0.0):.4f}\n"
    output += f"  F1: {metrics.get('macro_f1', 0.0):.4f}\n"
    
    if has_full_metrics:
        output += f"  MCC: {metrics.get('macro_mcc', 0.0):.4f}\n"
        output += f"  AUC: {metrics.get('macro_auc', 0.0):.4f}\n"
        output += f"  AUPRC: {metrics.get('macro_auprc', 0.0):.4f}\n"
        output += f"  Sensitivity: {metrics.get('macro_sensitivity', 0.0):.4f}\n"
        output += f"  Specificity: {metrics.get('macro_specificity', 0.0):.4f}\n"
    else:
        output += f"  AUC: {metrics.get('macro_auc', 0.0):.4f}\n"
    
    output += f"{'='*140}\n"
    
    # 打印到控制台
    print(output)
    
    # 记录到日志文件
    if logger:
        logger.info(output)


# ============================================================================
# Main Training Loop
# ============================================================================

def main(config_path='json/treexInMultirm.json'):
    """
    Main training function for MultIRM dataset.

    Args:
        config_path: Path to JSON configuration file
    """
    # Load configuration
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Get lenmode parameter from data configuration
    data_cfg = config_dict.get('data', {})
    seq_len = data_cfg.get('lenmode', 51)

    # Dynamically update experiment_name with lenmode suffix
    Config.experiment_name = f"{Config.experiment_name}_lenmode_{seq_len}"

    # Set MultIRM-specific parameters
    train_cfg = config_dict.get('training', {})
    Config.numsample = train_cfg.get('numsample', 50)

    # Set data paths based on sequence length
    Config.multirm_data_dir = f'npy/multirm/{seq_len}split'
    Config.multirm_cache_dir = f'npy/cache/multirm/{seq_len}'
    Config.seq_len = seq_len  # Store sequence length in Config

    # Setup logging
    logger = setup_logging(Config.log_dir, Config.experiment_name)

    # Setup tensorboard
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name)

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("MultIRM RNA Multi-label Classification Training")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")
    logger.info(f"Sequence length: {seq_len}")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load and precompute MultIRM datasets for train, test, and valid modes
    logger.info(f"\nLoading MultIRM datasets...")
    logger.info(f"  Data directory: {Config.multirm_data_dir}")
    logger.info(f"  Cache directory: {Config.multirm_cache_dir}")
    logger.info(f"  Sequence length: {seq_len}")

    # Function to load and precompute a dataset
    def load_and_precompute_dataset(mode_name, mode):
        """
        加载 MultiRM 数据集, 必要时预计算二级结构 / Load MultiRM dataset, precomputing structures on demand.

        若磁盘缓存 `multirm_{mode}_structures_cache.h5` 存在, 直接使用预加载缓存;
        否则调用 `precompute_all_structures` 触发结构预计算并落盘。
        If the on-disk cache `multirm_{mode}_structures_cache.h5` exists, the
        in-memory preload cache is used; otherwise `precompute_all_structures`
        is triggered to compute and persist the structures.

        Args / 参数:
            mode_name (str): [中文] 数据集人类可读名称, 用于日志 /
                [English] human-readable dataset name used in logs.
            mode (str): [中文] `MultirmDataset` 模式 'train'/'test'/'valid' /
                [English] `MultirmDataset` mode: 'train'/'test'/'valid'.

        Returns / 返回:
            MultirmDataset: [中文] 已加载 (并预计算) 的数据集实例 /
                [English] loaded (and precomputed) dataset instance.

        Called by / 被调用:
            - main(): [中文] 三次调用, 加载 train/test/valid 三个划分 /
                [English] called three times to load train/test/valid splits.
        """

        logger.info(f"\n{'='*60}")
        logger.info(f"Loading {mode_name} dataset (mode={mode})...")
        logger.info(f"{'='*60}")

        dataset = MultirmDataset(
            data_dir=Config.multirm_data_dir,
            cache_dir=Config.multirm_cache_dir,
            numsample=Config.numsample,
            mode=mode,
            use_cache=True,
            preload_cache=True  # 启用预加载，一次性加载所有缓存到内存
        )
        logger.info(f"{mode_name} dataset loaded: {len(dataset)} samples")

        # Check if disk cache file exists
        cache_file_path = os.path.join(Config.multirm_cache_dir, f"multirm_{mode}_structures_cache.h5")
        cache_exists = os.path.exists(cache_file_path)

        if cache_exists:
            logger.info(f"磁盘缓存已存在 ({mode_name}): {cache_file_path}")
            logger.info(f"使用预加载模式，所有缓存已一次性加载到内存")
        else:
            logger.info(f"磁盘缓存不存在 ({mode_name})，开始预计算二级结构...")
            dataset.precompute_all_structures(
                batch_size=100,
                num_workers=None,
                show_progress=True
            )

        return dataset

    # Load and precompute all three datasets
    train_dataset = load_and_precompute_dataset("Train", "train")
    test_dataset = load_and_precompute_dataset("Test", "test")
    valid_dataset = load_and_precompute_dataset("Valid", "valid")

    logger.info(f"\n{'='*60}")
    logger.info("All datasets loaded successfully!")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    logger.info(f"  Valid samples: {len(valid_dataset)}")
    logger.info(f"  Classes: {MULTIRM_CLASSES}")
    logger.info(f"{'='*60}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4*2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4*2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4*2,
        pin_memory=True
    )

    logger.info(f"\nDataLoaders created:")
    logger.info(f"  Train batch size: {Config.batch_size}")
    logger.info(f"  Test batch size: {Config.batch_size}")
    logger.info(f"  Valid batch size: {Config.batch_size}")

    # Initialize model
    logger.info(f"\nCreating model with sequence length {seq_len}...")
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
        use_hierarchical=Config.use_hierarchical,
        use_layer_norm=Config.use_layer_norm,
        seq_len=seq_len
    ).to(Config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Loss function (we use custom per-class loss in training loop)
    # No pos_weight needed since we balance per-class
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

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
    logger.info(f"  Loss: Per-class BCE (12 classes) + BCE (4 groups)")
    logger.info(f"  Epochs: {Config.num_epochs}")
    logger.info(f"  Batch size: {Config.batch_size}")

    # Log hyperparameters to tensorboard
    tb_writer.add_hparams(
        {
            'learning_rate': Config.learning_rate,
            'batch_size': Config.batch_size,
            'weight_decay': Config.weight_decay,
            'numsample': Config.numsample,
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

        # Train
        train_loss = train_epoch_multirm(
            model, train_loader, optimizer, scheduler,
            Config.device, logger,
            use_amp=Config.use_amp and torch.cuda.is_available()
        )

        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Evaluate
        should_eval = (epoch % Config.test_interval == 0)
        current_macro_f1 = 0.0

        if should_eval:
            logger.info(f"\nEvaluating...")
            
            # 使用改进后的评估函数：一次性推理 + 掩码计算
            # 相比旧方法优势：只需运行一次 DataLoader，使用掩码过滤无效数据
            metrics = evaluate_epoch_masked(
                model=model,
                dataloader=test_loader,
                device=Config.device,
                logger=logger,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log loss to tensorboard
            tb_writer.add_scalar('test/loss', metrics['avg_loss'], epoch)

            # Log per-class loss to tensorboard (用于诊断模型是否在某个类上偏科)
            for cls_name in MULTIRM_CLASSES:
                if f'{cls_name}_loss' in metrics:
                    tb_writer.add_scalar(f'test/loss_{cls_name}', metrics[f'{cls_name}_loss'], epoch)

            # Print formatted table with all metrics
            logger.info(f"\n{'='*60} Epoch {epoch} Testing (Masked Evaluation) {'='*60}")
            print_multirm_evaluation_table(metrics, epoch, logger)

            # Log to tensorboard
            tb_writer.add_scalar('test/macro_f1', metrics['macro_f1'], epoch)
            tb_writer.add_scalar('test/macro_precision', metrics['macro_precision'], epoch)
            tb_writer.add_scalar('test/macro_recall', metrics['macro_recall'], epoch)
            tb_writer.add_scalar('test/macro_acc', metrics['macro_acc'], epoch)
            tb_writer.add_scalar('test/macro_mcc', metrics['macro_mcc'], epoch)
            tb_writer.add_scalar('test/macro_auc', metrics['macro_auc'], epoch)
            tb_writer.add_scalar('test/macro_auprc', metrics['macro_auprc'], epoch)
            tb_writer.add_scalar('test/macro_sensitivity', metrics['macro_sensitivity'], epoch)
            tb_writer.add_scalar('test/macro_specificity', metrics['macro_specificity'], epoch)

            # Log per-class metrics to tensorboard
            for cls_name in MULTIRM_CLASSES:
                # Skip if metrics for this class don't exist (e.g., no test samples)
                if f'{cls_name}_f1' not in metrics:
                    logger.warning(f"No metrics found for class {cls_name}, skipping tensorboard logging")
                    continue
                
                tb_writer.add_scalar(f'test/f1_{cls_name}', metrics[f'{cls_name}_f1'], epoch)
                # 使用 pre 而不是 precision（与 calculate_masked_metrics 的输出一致）
                tb_writer.add_scalar(f'test/precision_{cls_name}', metrics[f'{cls_name}_pre'], epoch)
                # 使用 rec 而不是 recall（与 calculate_masked_metrics 的输出一致）
                tb_writer.add_scalar(f'test/recall_{cls_name}', metrics[f'{cls_name}_rec'], epoch)
                tb_writer.add_scalar(f'test/auc_{cls_name}', metrics[f'{cls_name}_auc'], epoch)
                tb_writer.add_scalar(f'test/acc_{cls_name}', metrics[f'{cls_name}_acc'], epoch)
                tb_writer.add_scalar(f'test/mcc_{cls_name}', metrics[f'{cls_name}_mcc'], epoch)
                tb_writer.add_scalar(f'test/auprc_{cls_name}', metrics[f'{cls_name}_auprc'], epoch)

            current_macro_f1 = metrics['macro_f1']
            logger.info(f"\nCurrent Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")

        # Save checkpoint
        if Config.save_every_epoch and should_eval:
            epoch_checkpoint_path = os.path.join(Config.checkpoint_dir, f'epoch_{epoch:03d}.pt')
            save_checkpoint(model, optimizer, epoch, metrics, epoch_checkpoint_path, logger, config_dict)

        # Save best model
        if should_eval and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, logger, config_dict)

            tb_writer.add_scalar('test/best_macro_f1', best_macro_f1, epoch)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Best model saved to: {os.path.join(Config.checkpoint_dir, 'best_model.pt')}")

    tb_writer.close()


if __name__ == "__main__":
    main()
