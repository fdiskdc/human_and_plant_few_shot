#!/usr/bin/env python3
"""
apply_train.py — Shared training module for motif_apply workflow.

Provides `ensure_converged_checkpoint()` which:
  1. Loads a target dataset (gen3 or plant) with a deterministic 90/10 split.
  2. Initializes a model from a fixed human pre-trained checkpoint.
  3. Trains with early stopping (patience on validation macro F1).
  4. Saves best_converged.pt + train_history.tsv.
  5. Reuses existing checkpoint if present (unless --force_retrain).

All training outputs go to ipynb/motif_apply/checkpoints/{dataset_name}/.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.common import (
    MOD_NAMES, GROUP_TO_CLASS_INDICES,
    load_config, get_smoothed_pos_weights,
)
from model.main_model import RNA_ClassQuery_Model

# ---------------------------------------------------------------------------
# Plant valid classes
# ---------------------------------------------------------------------------
PLANT_VALID_CLASSES = [5, 8, 9]   # Y, m5C, m6A
PLANT_VALID_GROUPS = [0, 1, 3]   # A, C, U  (group indices)

# Default config paths
DEFAULT_GEN3_CONFIG = 'json/3gen.json'
DEFAULT_PLANT_CONFIG = 'json/plant.json'

DEFAULT_INIT_CHECKPOINT = 'logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt'


# ============================================================================
# Data split helpers
# ============================================================================

def _split_gen3(dataset, val_ratio=0.1, seed=666):
    """Deterministic 90/10 split for gen3 using multi-label disjoint strategy."""
    from utils.common import multi_label_disjoint_split
    # Use existing split function which returns (train, test) at train_ratio
    train_indices, val_indices = multi_label_disjoint_split(
        dataset, train_ratio=1.0 - val_ratio, random_seed=seed,
        use_cache=True, cache_dir=dataset.CACHE_DIR)
    return train_indices, val_indices


def _split_plant(dataset, val_ratio=0.1, seed=42):
    """Split for plant ensuring each valid class [5,8,9] has positive samples in validation."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    all_labels = dataset.y_12class[:n].copy()  # (N, 12)

    val_set = set()
    for cls in PLANT_VALID_CLASSES:
        pos_indices = np.where(all_labels[:, cls] == 1)[0]
        if len(pos_indices) == 0:
            continue
        n_val = max(1, int(len(pos_indices) * val_ratio))
        selected = rng.choice(pos_indices, size=n_val, replace=False)
        val_set.update(selected.tolist())

    all_indices = set(range(n))
    train_set = all_indices - val_set
    return sorted(train_set), sorted(val_set)


# ============================================================================
# Model builder
# ============================================================================

def _build_model(config_dict, init_checkpoint, device):
    """Instantiate RNA_ClassQuery_Model from config and load init weights."""
    cfg = config_dict
    model_cfg = cfg.get('model', {})

    model = RNA_ClassQuery_Model(
        num_classes=model_cfg.get('num_classes', 12),
        use_hierarchical=model_cfg.get('use_hierarchical', True),
        cnn_hidden_dim=model_cfg.get('cnn_hidden_dim', 64),
        cnn_kernel_sizes=tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7])),
        cnn_dropout=model_cfg.get('cnn_dropout', 0.1),
        gcn_hidden_dim=model_cfg.get('gcn_hidden_dim', 128),
        gcn_out_channels=model_cfg.get('gcn_out_channels', 128),
        gcn_num_layers=model_cfg.get('gcn_num_layers', 3),
        gcn_dropout=model_cfg.get('gcn_dropout', 0.3),
        num_attn_heads=model_cfg.get('num_attn_heads', 8),
        attn_dropout=model_cfg.get('attn_dropout', 0.1),
        use_simple_pooling=model_cfg.get('use_simple_pooling', False),
        use_layer_norm=model_cfg.get('use_layer_norm', True),
    )

    if init_checkpoint and os.path.isfile(init_checkpoint):
        ckpt = torch.load(init_checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"[apply_train] Loaded init checkpoint: {init_checkpoint}")
    elif init_checkpoint:
        print(f"[apply_train] WARNING: init checkpoint not found: {init_checkpoint}")

    model.to(device)
    return model


# ============================================================================
# Evaluation helpers
# ============================================================================

def _compute_macro_f1(logits, targets, threshold=0.5, class_indices=None):
    """Compute macro F1 over specified class indices (default all 12)."""
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()
    if class_indices is not None:
        probs = probs[:, class_indices]
        targets = targets[:, class_indices]

    n_classes = probs.shape[1]
    f1s = []
    for c in range(n_classes):
        pred = (probs[:, c] >= threshold).astype(int)
        tp = ((pred == 1) & (targets[:, c] == 1)).sum()
        fp = ((pred == 1) & (targets[:, c] == 0)).sum()
        fn = ((pred == 0) & (targets[:, c] == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def _eval_epoch(model, dataloader, device, dataset_name):
    """Run one validation epoch, return (avg_loss, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    n_batches = 0

    class_indices = PLANT_VALID_CLASSES if dataset_name == 'plant' else None

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            batch = batch.to(device, non_blocking=True)
            batch.y = batch.y.to(device, non_blocking=True)

            cfg_hier = model.use_hierarchical if hasattr(model, 'use_hierarchical') else True
            result = model(batch.x, batch.edge_index, batch.batch,
                           return_attention=False)
            if cfg_hier:
                logits_12 = result[0]
                logits_4 = result[1]
            else:
                logits_12 = result if not isinstance(result, tuple) else result[0]
                logits_4 = None

            # Select targets/logits based on dataset
            if dataset_name == 'plant':
                y_target = batch.y[:, PLANT_VALID_CLASSES]
                logits_sel = logits_12[:, PLANT_VALID_CLASSES]
            else:
                y_target = batch.y
                logits_sel = logits_12

            loss = F.binary_cross_entropy_with_logits(logits_sel, y_target)
            total_loss += loss.item()
            all_logits.append(logits_12.cpu())
            all_targets.append(batch.y.cpu())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    macro_f1 = _compute_macro_f1(all_logits, all_targets,
                                  class_indices=class_indices)
    return avg_loss, macro_f1


# ============================================================================
# Main training loop
# ============================================================================

def _train_loop(model, train_loader, val_loader, dataset_name, config_dict,
                output_dir, max_epochs, patience, min_delta, device):
    """Train with early stopping on validation macro F1."""
    # Move model to device NOW, after DataLoaders are already created
    model.to(device)
    print(f"[train_loop] model moved to {device}")

    train_cfg = config_dict.get('training', {})
    lr = train_cfg.get('learning_rate', 1e-3)
    wd = train_cfg.get('weight_decay', 1e-4)
    use_hierarchical = config_dict.get('model', {}).get('use_hierarchical', True)
    use_amp = train_cfg.get('use_amp', True)
    use_attn_supervision = False  # Disable attention supervision during fine-tuning

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    history_path = os.path.join(output_dir, 'train_history.tsv')
    with open(history_path, 'w') as f:
        f.write('epoch\ttrain_loss\tval_loss\tval_macro_f1\tlearning_rate\tis_best\n')

    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_dir = output_dir
    best_path = os.path.join(ckpt_dir, 'best_converged.pt')
    last_path = os.path.join(ckpt_dir, 'last.pt')

    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    from tqdm import tqdm
    import sys

    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0
        print(f"[Epoch {epoch:03d}] starting training...", flush=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=True)
        for batch in pbar:
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            batch = batch.to(device, non_blocking=True)
            batch.y = batch.y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    result = model(batch.x, batch.edge_index, batch.batch,
                                   return_attention=False)
                    if use_hierarchical:
                        logits_12, logits_4 = result[0], result[1]
                        loss_12 = F.binary_cross_entropy_with_logits(logits_12, batch.y)

                        y_4 = torch.zeros(logits_12.size(0), 4, device=device)
                        for g_idx, g_name in enumerate(['A', 'C', 'G', 'U']):
                            cls_indices = GROUP_TO_CLASS_INDICES[g_name]
                            y_4[:, g_idx] = batch.y[:, cls_indices].max(dim=1)[0]
                        loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                        loss = loss_12 + loss_4
                    else:
                        logits = result if not isinstance(result, tuple) else result[0]
                        loss = F.binary_cross_entropy_with_logits(logits, batch.y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                result = model(batch.x, batch.edge_index, batch.batch,
                               return_attention=False)
                if use_hierarchical:
                    logits_12, logits_4 = result[0], result[1]
                    loss_12 = F.binary_cross_entropy_with_logits(logits_12, batch.y)

                    y_4 = torch.zeros(logits_12.size(0), 4, device=device)
                    for g_idx, g_name in enumerate(['A', 'C', 'G', 'U']):
                        cls_indices = GROUP_TO_CLASS_INDICES[g_name]
                        y_4[:, g_idx] = batch.y[:, cls_indices].max(dim=1)[0]
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                    loss = loss_12 + loss_4
                else:
                    logits = result if not isinstance(result, tuple) else result[0]
                    loss = F.binary_cross_entropy_with_logits(logits, batch.y)

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        pbar.close()
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        val_loss, val_f1 = _eval_epoch(model, val_loader, device, dataset_name)

        cur_lr = optimizer.param_groups[0]['lr']
        is_best = val_f1 > best_f1 + min_delta
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'dataset_name': dataset_name,
                'best_metric': best_f1,
                'config': config_dict,
            }, best_path)
        else:
            epochs_no_improve += 1

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
              f"{'BEST' if is_best else ''} "
              f"no_improve={epochs_no_improve}/{patience}")

        with open(history_path, 'a') as f:
            f.write(f'{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t'
                    f'{val_f1:.6f}\t{cur_lr:.8f}\t{is_best}\n')

        if epochs_no_improve >= patience:
            print(f"[EarlyStop] No improvement for {patience} epochs. "
                  f"Best F1={best_f1:.4f} at epoch {best_epoch}")
            break

    # Save last
    final_loss = val_f1 if 'val_f1' in dir() else 0.0
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'dataset_name': dataset_name,
        'best_metric': val_f1,
        'config': config_dict,
    }, last_path)

    print(f"[Done] Best F1={best_f1:.4f} at epoch {best_epoch}, saved to {best_path}")
    return best_path, history_path


# ============================================================================
# Public interface
# ============================================================================

def ensure_converged_checkpoint(
    dataset_name,
    config_path,
    init_checkpoint=DEFAULT_INIT_CHECKPOINT,
    output_root=None,
    force_retrain=False,
    max_epochs=None,
    patience=10,
    min_delta=1e-4,
    val_ratio=0.1,
    seed=None,
    batch_size=None,
    device=None,
):
    """Train a model on *dataset_name* until convergence and return checkpoint info.

    Parameters
    ----------
    dataset_name : str
        'gen3' or 'plant'.
    config_path : str
        Path to JSON config (e.g. 'json/3gen.json').
    init_checkpoint : str
        Path to human pre-trained checkpoint for weight initialization.
    output_root : str | None
        Root dir for checkpoints.  Default: ipynb/motif_apply/checkpoints/.
    force_retrain : bool
        If True, retrain even if best_converged.pt exists.
    max_epochs : int | None
        Maximum training epochs.  Default: config's num_epochs.
    patience : int
        Early stopping patience (default 10).
    min_delta : float
        Minimum F1 improvement to reset patience (default 1e-4).
    val_ratio : float
        Fraction for validation (default 0.1).
    seed : int | None
        Random seed.  Default: config's random_seed.
    batch_size : int | None
        Batch size.  Default: config's batch_size.
    device : torch.device | None
        Device.  Default: auto-detect CUDA.

    Returns
    -------
    dict with keys: checkpoint_path, train_indices, val_indices, dataset, config, history_path
    """
    if output_root is None:
        output_root = os.path.join(os.path.dirname(__file__), 'checkpoints')
    ckpt_dir = os.path.join(output_root, dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, 'best_converged.pt')

    # ---- Reuse existing checkpoint ----
    if os.path.isfile(best_path) and not force_retrain:
        print(f"[apply_train] Reusing existing checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        train_indices = ckpt.get('train_indices', [])
        val_indices = ckpt.get('val_indices', [])
        config_dict = ckpt.get('config', {})
        history_path = os.path.join(ckpt_dir, 'train_history.tsv')
        return {
            'checkpoint_path': best_path,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'dataset': None,  # Will be loaded by caller if needed
            'config': config_dict,
            'history_path': history_path,
        }

    # ---- Load config ----
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_cfg = config_dict.get('training', {})
    if seed is None:
        seed = train_cfg.get('random_seed', 666 if dataset_name == 'gen3' else 42)
    if batch_size is None:
        batch_size = train_cfg.get('batch_size', 10)
    if max_epochs is None:
        max_epochs = train_cfg.get('num_epochs', 5)

    # ---- Build dataset ----
    data_cfg = config_dict.get('data', {})
    if dataset_name == 'gen3':
        from dataset.gen3 import Gen3Dataset
        data_dir = os.path.join(_PROJECT_ROOT,
                                data_cfg.get('gen3_data_dir', 'npy/3gen'))
        cache_dir = os.path.join(_PROJECT_ROOT,
                                 data_cfg.get('gen3_cache_dir',
                                              data_cfg.get('cache_dir', 'npy/cache')))
        dataset = Gen3Dataset(mode='train', data_dir=data_dir,
                              cache_dir=cache_dir, use_cache=True,
                              preload_cache=True, skip_attn=True)
        train_indices, val_indices = _split_gen3(dataset, val_ratio, seed)
    elif dataset_name == 'plant':
        from dataset.plant import PlantDataset
        data_dir = os.path.join(_PROJECT_ROOT,
                                data_cfg.get('plant_data_dir', 'npy/plant'))
        cache_dir = os.path.join(_PROJECT_ROOT,
                                 data_cfg.get('plant_cache_dir',
                                              data_cfg.get('cache_dir', 'npy/cache')))
        dataset = PlantDataset(plant_dir=data_dir, cache_dir=cache_dir,
                               use_cache=True, preload_cache=True)
        train_indices, val_indices = _split_plant(dataset, val_ratio, seed)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    print(f"[apply_train] {dataset_name}: {len(train_indices)} train, "
          f"{len(val_indices)} val (total {len(dataset)})")

    # ---- Subset + DataLoader ----
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Use num_workers>0 for parallel data loading; persistent_workers avoids
    # repeated process startup; pin_memory accelerates CPU->GPU transfer.
    _nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=_nw, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=_nw, pin_memory=True,
                            persistent_workers=True)

    # ---- Build model (on CPU first, move to device inside training loop) ----
    cpu_device = torch.device('cpu')
    model = _build_model(config_dict, init_checkpoint, cpu_device)

    # ---- Save config ----
    train_config_path = os.path.join(ckpt_dir, 'train_config.json')
    run_info = {
        'dataset_name': dataset_name,
        'config_path': config_path,
        'init_checkpoint': init_checkpoint,
        'max_epochs': max_epochs,
        'patience': patience,
        'min_delta': min_delta,
        'val_ratio': val_ratio,
        'seed': seed,
        'batch_size': batch_size,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'device': str(device),
    }
    with open(train_config_path, 'w') as f:
        json.dump(run_info, f, indent=2)

    # ---- Train ----
    best_path, history_path = _train_loop(
        model, train_loader, val_loader, dataset_name, config_dict,
        ckpt_dir, max_epochs, patience, min_delta, device)

    # ---- Re-save best with train/val indices ----
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    ckpt['train_indices'] = train_indices
    ckpt['val_indices'] = val_indices
    torch.save(ckpt, best_path)

    return {
        'checkpoint_path': best_path,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'dataset': dataset,
        'config': config_dict,
        'history_path': history_path,
    }


# ============================================================================
# CLI entry point (for standalone testing)
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='Motif apply: converged training')
    p.add_argument('--dataset', required=True, choices=['gen3', 'plant'],
                   help='Target dataset name')
    p.add_argument('--config', default=None,
                   help='Config JSON path (auto-detected from dataset if omitted)')
    p.add_argument('--init_checkpoint', default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument('--output_root', default=None)
    p.add_argument('--force_retrain', action='store_true')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min_delta', type=float, default=1e-4)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.config is None:
        args.config = DEFAULT_GEN3_CONFIG if args.dataset == 'gen3' else DEFAULT_PLANT_CONFIG

    device = torch.device(args.device) if args.device else None

    result = ensure_converged_checkpoint(
        dataset_name=args.dataset,
        config_path=args.config,
        init_checkpoint=args.init_checkpoint,
        output_root=args.output_root,
        force_retrain=args.force_retrain,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"\n[Result] checkpoint: {result['checkpoint_path']}")
    print(f"[Result] train: {len(result['train_indices'])}, "
          f"val: {len(result['val_indices'])}")
    print(f"[Result] history: {result['history_path']}")


if __name__ == '__main__':
    main()
