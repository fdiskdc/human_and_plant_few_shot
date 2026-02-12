"""
MultIRM Dataset Training Script

This script implements training for the MultIRM RNA modification dataset:
1. Loads data from dataset/multirm.py with positive/negative sample pairing
2. Precomputes all secondary structures on first run (stored in npy/cache/multirm)
3. Uses 12 separate losses for each class (positive + negative samples)
4. Comprehensive metrics evaluation
5. Logging to file and console
6. Tensorboard visualization
7. TQDM progress monitoring
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

from model.main_model import RNA_ClassQuery_Model
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
    total_loss_per_class = {cls: 0.0 for cls in MULTIRM_CLASSES}
    num_batches = 0
    num_samples_per_class = {cls: 0 for cls in MULTIRM_CLASSES}

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    pbar = tqdm(dataloader, desc="Training", leave=True)

    for batch in pbar:
        # Get class indices for each sample in the batch
        # batch.class_idx is a tensor of class indices
        class_indices = batch.class_idx

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

        # Use 12-class logits for loss computation
        logits = logits_12class

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = compute_per_class_loss(logits, batch.y, class_indices, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

        else:
            loss = compute_per_class_loss(logits, batch.y, class_indices, device)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # Track losses per class
        with torch.no_grad():
            batch_loss_per_class = compute_per_class_losses(logits, batch.y, class_indices, device)
            for cls_idx, cls_loss in batch_loss_per_class.items():
                cls_name = MULTIRM_CLASSES[cls_idx]
                total_loss_per_class[cls_name] += cls_loss
                num_samples_per_class[cls_name] += (class_indices == cls_idx).sum().item()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate average loss per class
    avg_loss_per_class = {}
    for cls in MULTIRM_CLASSES:
        if num_samples_per_class[cls] > 0:
            # Normalize by number of batches that had this class
            avg_loss_per_class[cls] = total_loss_per_class[cls] / num_batches
        else:
            avg_loss_per_class[cls] = 0.0

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if logger:
        logger.info(f"Train Loss: {avg_loss:.4f}")
        logger.info(f"Per-class losses: {avg_loss_per_class}")

    return avg_loss


def compute_per_class_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_indices: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute loss separately for each class and average them.

    This ensures that each class's positive and negative samples
    contribute equally to the total loss, regardless of class size.

    Args:
        logits: Model predictions (batch_size, 12)
        labels: Ground truth labels (batch_size, 12)
        class_indices: Class index for each sample (batch_size,)
        device: Device to compute on

    Returns:
        Average loss across all classes
    """
    losses_per_class = []
    batch_size = logits.size(0)

    for cls_idx, cls_name in enumerate(MULTIRM_CLASSES):
        # Find samples belonging to this class
        class_mask = (class_indices == cls_idx)

        if class_mask.sum() == 0:
            continue

        # Get logits and labels for this class's samples
        class_logits = logits[class_mask]
        class_labels = labels[class_mask]

        # Compute BCE loss for this class
        # Each sample has a 12-dim label vector, but we only care about
        # the loss for samples belonging to this class
        class_loss = F.binary_cross_entropy_with_logits(
            class_logits[:, cls_idx:cls_idx+1],
            class_labels[:, cls_idx:cls_idx+1],
            reduction='mean'
        )
        losses_per_class.append(class_loss)

    if losses_per_class:
        # Average losses across all classes present in this batch
        return torch.stack(losses_per_class).mean()
    else:
        return torch.tensor(0.0, device=device)


def compute_per_class_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_indices: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Compute individual losses for each class (for tracking).

    Args:
        logits: Model predictions (batch_size, 12)
        labels: Ground truth labels (batch_size, 12)
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

        class_logits = logits[class_mask]
        class_labels = labels[class_mask]

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
    num_batches = 0

    pbar = tqdm(dataloader, desc="Testing", leave=True)

    with torch.no_grad():
        for batch in pbar:
            class_indices = batch.class_idx

            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)

            batch = batch.to(device)
            batch.y = batch.y.to(device)

            # Model forward pass - unpack three return values
            logits_12class, logits_4class, attn_weights_12 = model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )

            # Use 12-class logits for loss computation
            logits = logits_12class

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = compute_per_class_loss(logits, batch.y, class_indices, device)
            else:
                loss = compute_per_class_loss(logits, batch.y, class_indices, device)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if logger:
        logger.info(f"Test Loss: {avg_loss:.4f}")

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


def evaluate_multirm(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate predictions on MultIRM dataset.

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
    lenmode = data_cfg.get('lenmode', 51)
    # Calculate sequence length: lenmode * 4 (one-hot encoded)
    seq_len = lenmode * 4

    # Set MultIRM-specific parameters
    # The numsample parameter is used by MultirmDataset to sample positive/negative pairs
    train_cfg = config_dict.get('training', {})
    Config.numsample = train_cfg.get('numsample', 50)

    # Set data paths based on lenmode
    Config.multirm_data_dir = f'npy/multirm/{lenmode}split'
    Config.multirm_cache_dir = f'npy/cache/multirm/{lenmode}'
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
    logger.info(f"lenmode: {lenmode} (sequence length: {seq_len})")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load and precompute MultIRM datasets for train, test, and valid modes
    logger.info(f"\nLoading MultIRM datasets...")
    logger.info(f"  Data directory: {Config.multirm_data_dir}")
    logger.info(f"  Cache directory: {Config.multirm_cache_dir}")
    logger.info(f"  lenmode: {lenmode}")

    # Function to load and precompute a dataset
    def load_and_precompute_dataset(mode_name, mode):
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading {mode_name} dataset (mode={mode})...")
        logger.info(f"{'='*60}")

        dataset = MultirmDataset(
            data_dir=Config.multirm_data_dir,
            cache_dir=Config.multirm_cache_dir,
            numsample=Config.numsample,
            mode=mode,
            use_cache=True,
            preload_cache=False  # 禁用预加载，使用懒加载以提高启动速度
        )
        logger.info(f"{mode_name} dataset loaded: {len(dataset)} samples")

        # Check if disk cache file exists
        cache_file_path = os.path.join(Config.multirm_cache_dir, f"multirm_{mode}_structures_cache.npz")
        cache_exists = os.path.exists(cache_file_path)

        if cache_exists:
            logger.info(f"磁盘缓存已存在 ({mode_name}): {cache_file_path}")
            logger.info(f"使用懒加载模式，缓存将在首次访问时按需加载")
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
    logger.info(f"  Loss: Per-class BCE (12 classes)")
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
            test_loss = test_epoch_multirm(
                model, test_loader, Config.device, logger,
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Get predictions and compute metrics
            logger.info(f"\n{'='*60} Epoch {epoch} Testing {'='*60}")
            y_true, y_prob = get_predictions_multirm(model, test_loader, Config.device)

            metrics = evaluate_multirm(y_true, y_prob, threshold=0.5)

            # Print results
            logger.info(f"\n{'='*60}")
            logger.info("Evaluation Results:")
            logger.info(f"{'='*60}")
            logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
            logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
            logger.info(f"Micro F1: {metrics['micro_f1']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"AUC (Macro): {metrics['macro_auc']:.4f}")
            logger.info(f"AUPRC (Macro): {metrics['macro_auprc']:.4f}")

            # Per-class F1 scores
            logger.info(f"\nPer-class F1 scores:")
            for cls_name in MULTIRM_CLASSES:
                logger.info(f"  {cls_name}: {metrics[f'{cls_name}_f1']:.4f}")

            # Log to tensorboard
            tb_writer.add_scalar('test/macro_f1', metrics['macro_f1'], epoch)
            tb_writer.add_scalar('test/micro_f1', metrics['micro_f1'], epoch)
            tb_writer.add_scalar('test/accuracy', metrics['accuracy'], epoch)
            tb_writer.add_scalar('test/auc', metrics['macro_auc'], epoch)
            tb_writer.add_scalar('test/auprc', metrics['macro_auprc'], epoch)

            for cls_name in MULTIRM_CLASSES:
                tb_writer.add_scalar(f'test/f1_{cls_name}', metrics[f'{cls_name}_f1'], epoch)

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
