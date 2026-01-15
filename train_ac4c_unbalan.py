
"""
AC4C Unbalanced Dataset Training Script (PRUNED MODE)

This script implements:
1. AC4C unbalanced dataset training (Pruned to single class index 6)
2. Model Pruning: Cuts the model to only compute AC4C queries.
3. Custom Training/Test Loops: Handles the 1-dim (model) vs 12-dim (label) mismatch.
"""

import os
import random
import json
import numpy as np
import torch
import argparse
from datetime import datetime
from prettytable import PrettyTable

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.main_model import RNA_ClassQuery_Model
from dataset.ac4c import AC4CDataset
from utils import (
    setup_logging, setup_tensorboard, log_metrics_to_tensorboard,
    get_smoothed_pos_weights,
    save_checkpoint, load_config,
    print_evaluation_results, print_few_shot_results,
    run_few_shot_benchmark_ac4c,
    MOD_NAMES, get_all_predictions
)
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # 将 [0, 1] 标签转换为 [0.05, 0.95] (假设 smoothing=0.1)
        # targets: (Batch,)
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smooth_targets)
        return loss
# Constants for AC4C
AC4C_CLASS_IDX = 6
AC4C_GROUP_IDX = 1  # Group C

class PrunedModelWrapper(nn.Module):
    """
    Wrapper for the pruned model to interface with existing evaluation functions.
    It pads the pruned 1-dim output back to 12-dim (filling others with very small logits).
    """
    def __init__(self, pruned_model, class_idx=AC4C_CLASS_IDX, group_idx=AC4C_GROUP_IDX):
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
            # Use -1e9 for non-active classes so sigmoid(logits) -> 0
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

def train_epoch_pruned(model, dataloader, criterion, optimizer, scheduler, device, logger, use_hierarchical, use_amp):
    """
    Optimized training loop using the pruned model directly (avoiding wrapper overhead).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training (Pruned)", leave=True)
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get target for AC4C only
        target_12 = batch.y[:, AC4C_CLASS_IDX].float()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                if use_hierarchical:
                    logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)
                    logits_12 = logits_12.view(-1)
                    logits_4 = logits_4.view(-1)
                    
                    loss_12 = criterion(logits_12, target_12)
                    target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                    loss = loss_12 + loss_4
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    logits = logits.view(-1)
                    loss = criterion(logits, target_12)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_hierarchical:
                logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)
                logits_12 = logits_12.view(-1)
                logits_4 = logits_4.view(-1)
                
                loss_12 = criterion(logits_12, target_12)
                target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                loss = loss_12 + loss_4
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
                logits = logits.view(-1)
                loss = criterion(logits, target_12)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    return total_loss / num_batches

def test_epoch(model, dataloader, criterion, device, phase, logger, use_hierarchical, use_amp):
    """
    Local test_epoch that handles the Wrapped model correctly.
    Calculates loss only on the relevant class to avoid noise from padded values.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            target_12 = batch.y[:, AC4C_CLASS_IDX].float()
            
            # If using wrapper, we can access the internal pruned model for efficiency/correctness in loss
            if isinstance(model, PrunedModelWrapper):
                internal_model = model.model
            else:
                internal_model = model
                
            if use_hierarchical:
                logits_12, logits_4 = internal_model(batch.x, batch.edge_index, batch.batch)
                logits_12 = logits_12.view(-1)
                logits_4 = logits_4.view(-1)
                
                loss_12 = criterion(logits_12, target_12)
                target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                loss = loss_12 + loss_4
            else:
                logits = internal_model(batch.x, batch.edge_index, batch.batch)
                logits = logits.view(-1)
                loss = criterion(logits, target_12)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches

def main(config_path='json/ac4c_unbalan.json', checkpoint_path=None):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name)
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_ac4c_pruned')

    logger.info(f"\n{'='*60}")
    logger.info("AC4C Unbalanced Dataset Training (PRUNED MODE)")
    logger.info(f"{'='*60}")
    logger.info(f"Target Class: Index {AC4C_CLASS_IDX} (ac4c)")
    
    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Data directory
    ac4c_data_dir = 'npy/ac4c_processed/unbalanced_ac4c'

    # Load datasets
    logger.info(f"Loading AC4C Datasets...")
    train_dataset = AC4CDataset(mode='train', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir, use_cache=True, preload_cache=True)
    test_dataset = AC4CDataset(mode='test', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir, use_cache=True, preload_cache=True)
    
    # Calculate weights based on FULL 12 classes first
    train_indices = list(range(len(train_dataset)))
    pos_weight_full = get_smoothed_pos_weights(
        train_dataset, train_indices, num_classes=12, logger=logger
    )
    
    # Extract only the weight for AC4C
    pos_weight_ac4c = pos_weight_full[AC4C_CLASS_IDX].view(1) # Shape (1,)
    logger.info(f"Pruned Pos Weight for AC4C: {pos_weight_ac4c.item():.4f}")

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
        num_classes=Config.num_classes, # Init with 12
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=getattr(Config, 'use_hierarchical', False),
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    # PRUNE THE MODEL HEADS
    logger.info(f"\n{'!'*60}")
    logger.info(f"PRUNING MODEL HEADS TO AC4C ONLY")
    logger.info(f"{'!'*60}")
    model.prune_heads(
        valid_class_indices=[AC4C_CLASS_IDX], 
        valid_group_indices=[AC4C_GROUP_IDX] if getattr(Config, 'use_hierarchical', False) else None
    )

    # Load checkpoint if exists (Handle strict loading manually if needed)
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=Config.device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("Loaded state dict (strict=False)")
        except Exception as e:
            logger.warning(f"Could not load checkpoint directly: {e}")

    # Wrap the pruned model for evaluation compatibility
    wrapped_model = PrunedModelWrapper(model)

    # Loss function for single class
    pos_weight_ac4c = pos_weight_ac4c.to(Config.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ac4c)
    criterion = LabelSmoothingLoss(smoothing=0.1).to(Config.device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, Config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{Config.num_epochs}")
        
        # Train: Use raw pruned model for efficiency
        train_loss = train_epoch_pruned(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=getattr(Config, 'use_hierarchical', False),
            use_amp=Config.use_amp and torch.cuda.is_available()
        )
        
        tb_writer.add_scalar('train/loss', train_loss, epoch)

        # Verification Logic (Preserved from user request)
        should_eval = (epoch % Config.test_interval == 0)
        if should_eval:
            # Evaluate
            logger.info(f"\nEvaluating...")
            # Note: Passing wrapped_model here to handle interface compatibility
            test_loss = test_epoch(
                wrapped_model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=getattr(Config, 'use_hierarchical', False),
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log test loss to tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Get predictions
            logger.info(f"\n  Getting predictions for test data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(wrapped_model, test_loader, Config.device, getattr(Config, 'use_hierarchical', False))

            # Import AC4C evaluation function
            from utils.metrics import evaluate_ac4c
            metrics_unbalance = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                             dataset_mode='unbalanced', y_4prob=y_4prob)
            metrics_balanced = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                           dataset_mode='balanced', y_4prob=y_4prob)

            # Print results
            def print_ac4c_table(metrics, title, logger):
                table = PrettyTable()
                table.field_names = ["Class", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
                table.align = "r"
                table.align["Class"] = "l"
                
                # Get metrics for AC4C class (index 6)
                c = AC4C_CLASS_IDX
                mod_name = "ac4c"
                
                # Extract metrics
                f1 = metrics.get(f'group_ac4c_class_{c}_opt_f1', 0.0)
                prec = metrics.get(f'group_ac4c_class_{c}_opt_precision', 0.0)
                rec = metrics.get(f'group_ac4c_class_{c}_opt_recall', 0.0)
                acc = metrics.get(f'group_ac4c_class_{c}_opt_accuracy', 0.0)
                auc = metrics.get(f'group_ac4c_class_{c}_auc', 0.0)
                auprc = metrics.get(f'group_ac4c_class_{c}_auprc', 0.0)
                sn = metrics.get(f'group_ac4c_class_{c}_opt_sensitivity', 0.0)
                sp = metrics.get(f'group_ac4c_class_{c}_opt_specificity', 0.0)
                
                tp = int(metrics.get(f'group_ac4c_class_{c}_opt_tp', 0))
                tn = int(metrics.get(f'group_ac4c_class_{c}_opt_tn', 0))
                fp = int(metrics.get(f'group_ac4c_class_{c}_opt_fp', 0))
                fn = int(metrics.get(f'group_ac4c_class_{c}_opt_fn', 0))
                
                # Add row
                table.add_row([
                    f"{c}({mod_name})",
                    f"{f1:.4f}",
                    f"{prec:.4f}",
                    f"{rec:.4f}",
                    f"{acc:.4f}",
                    f"{auc:.4f}",
                    f"{auprc:.4f}",
                    f"{sn:.4f}",
                    f"{sp:.4f}",
                    tp, tn, fp, fn
                ])
                
                # Add macro average row
                macro_f1 = metrics.get(f'group_ac4c_opt_macro_f1', 0.0)
                macro_p = metrics.get(f'group_ac4c_opt_macro_precision', 0.0)
                macro_r = metrics.get(f'group_ac4c_opt_macro_recall', 0.0)
                table.add_row(["Avg", f"{macro_f1:.4f}", f"{macro_p:.4f}", f"{macro_r:.4f}", "-", "-", "-", "-", "-", "-", "-", "-"])
                
                # Print table
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch} - {title}")
                logger.info(f"{'='*80}")
                logger.info(table)
            
            # Print tables
            print_ac4c_table(metrics_unbalance, "AC4C Unbalanced Dataset Test Results", logger)
            print_ac4c_table(metrics_balanced, "AC4C Balanced Dataset Test Results", logger)
            
            # Also print the brief summary as before
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch} - AC4C Test Results Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Unbalance Mode - Macro F1: {metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_unbalance.get('group_ac4c_micro_f1', 0.0):.4f}")
            logger.info(f"Balance Mode - Macro F1: {metrics_balanced.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_balanced.get('group_ac4c_micro_f1', 0.0):.4f}")

            # Log metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_ac4c_unbalanced_unbalance', epoch, key_prefix='group_ac4c_')
            log_metrics_to_tensorboard(tb_writer, metrics_balanced, 'test_ac4c_unbalanced_balance', epoch, key_prefix='group_ac4c_')

            # Run Few-Shot Benchmark
            logger.info(f"\n>>> Running AC4C Unbalanced Few-Shot Benchmark <<<")


            # Check for best model
            current_macro_f1 = metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0)
            logger.info(f"Current Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")
            
            # Check best
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, metrics_unbalance, 
                              os.path.join(Config.checkpoint_dir, 'ac4c_pruned_best.pt'), logger, config_dict)

    logger.info(f"Training Complete. Best F1: {best_macro_f1:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='json/ac4c_unbalan.json')
    parser.add_argument('--checkpoint', type=str, default="logs/rna_classification_20260111_111223/checkpoints/best_model.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
