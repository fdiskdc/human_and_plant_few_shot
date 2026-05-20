"""
Ablation script for HierarchicalClassQueryHeadPooling on human dataset.

Runs 8 ablation configurations and outputs:
  - logs_abla/ablation_results.csv
  - logs_abla/fig1_query_ablation.png
  - logs_abla/fig2_dim_ablation.png
  - logs_abla/fig3_class_heatmap.png

Usage:
    python abla_mohe.py
"""

import os
import sys
import csv
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.abla_model import AblationModel
from cal_flops_mohe import count_model_flops, count_fullattn_params, _count_backbone_flops, count_fullattn_head_flops
from dataset.human import Mer100Dataset
from utils import (
    setup_logging, multi_label_disjoint_split, get_smoothed_pos_weights,
    evaluate_unbalance, get_all_predictions, GROUP_TO_CLASS_INDICES,
    DynamicBalancedBatchSampler, MultilabelBalancedBatchSampler,
)

# ============================================================================
# Monkey-patched train_epoch / test_epoch
# The original functions in utils/common.py unpack (logits_12, logits_4) = model(...)
# but AblationModel.forward always returns 3 values (logits_12, logits_4, attn).
# We rewrite the hierarchical branch to handle 3-value returns consistently.
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, logger,
                use_hierarchical=False, use_amp=False,
                use_attention_supervision=False, attention_lambda=1.0):
    import torch.nn.functional as F
    from utils.common import GROUP_TO_CLASS_INDICES, compute_attention_supervision_loss

    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training", leave=True)
    for batch in pbar:
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float32)
        batch = batch.to(device)
        batch.y = batch.y.to(device)
        optimizer.zero_grad()

        should_return_attention = use_attention_supervision and hasattr(batch, 'y_site')

        def _forward():
            if use_hierarchical:
                out = model(batch.x, batch.edge_index, batch.batch)
                logits_12, logits_4 = out[0], out[1]
                y_12 = batch.y
                y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)
                for g_idx, g_name in enumerate(['A', 'C', 'G', 'U']):
                    y_4[:, g_idx] = y_12[:, GROUP_TO_CLASS_INDICES[g_name]].max(dim=1)[0]
                loss_12 = criterion(logits_12, y_12)
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                loss = loss_12 + loss_4
                if should_return_attention and len(out) == 3:
                    loss_attn = compute_attention_supervision_loss(out[2], batch.y_site)
                    loss = loss + attention_lambda * loss_attn
                return loss
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
                return criterion(logits, batch.y)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = _forward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _forward()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()
    return total_loss / max(num_batches, 1)


def test_epoch(model, dataloader, criterion, device, phase="test", logger=None,
               use_hierarchical=False, use_amp=False):
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    with torch.no_grad():
        for batch in pbar:
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            batch = batch.to(device)
            batch.y = batch.y.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            if use_hierarchical:
                logits_12, logits_4 = out[0], out[1]
                y_12 = batch.y
                y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)
                for g_idx, g_name in enumerate(['A', 'C', 'G', 'U']):
                    y_4[:, g_idx] = y_12[:, GROUP_TO_CLASS_INDICES[g_name]].max(dim=1)[0]
                loss_12 = criterion(logits_12, y_12)
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                loss = loss_12 + loss_4
            else:
                loss = criterion(out, batch.y)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)

# ============================================================================
# Ablation configurations: 4 query_types x 4 group_query_dims = 16 runs
# ============================================================================

QUERY_TYPES = ["1query", "4query", "12query", "fullattn"]
QUERY_DIMS  = [128, 256, 512, 1001]

QUERY_TYPE_DESC = {
    "1query":   "1 global query + 12-way MLP",
    "4query":   "4-group hierarchical (baseline)",
    "12query":  "12 independent queries",
    "fullattn": "12 queries + self-attention",
}

ABLATION_CONFIGS = []
for qt in QUERY_TYPES:
    for qd in QUERY_DIMS:
        ABLATION_CONFIGS.append({
            "name": f"{qt}_d{qd}",
            "query_type": qt,
            "group_query_dim": qd,
            "desc": f"{QUERY_TYPE_DESC[qt]}, dim={qd}",
        })


def run_single_ablation(ablation_cfg, config_dict, dataset, train_indices, test_indices,
                        pos_weight_unbalanced, device, output_dir):
    """Train and evaluate one ablation configuration. Returns a dict of results."""

    cfg_name = ablation_cfg["name"]
    query_type = ablation_cfg["query_type"]
    group_query_dim = ablation_cfg["group_query_dim"]

    print(f"\n{'='*60}")
    print(f"  Ablation: {cfg_name} — {ablation_cfg['desc']}")
    print(f"{'='*60}")

    # Reproducibility
    seed = config_dict["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Samplers
    batch_size = config_dict["training"]["batch_size"]
    num_epochs = config_dict["training"]["num_epochs"]

    train_batch_sampler = DynamicBalancedBatchSampler(
        dataset=dataset, train_indices=train_indices,
        batch_size=batch_size, num_classes=12,
        # balance_ratio=config_dict["training"].get("balance_ratio", 0.9),
        balance_ratio=1.0,
        total_epochs=num_epochs, random_seed=seed
    )

    train_loader = DataLoader(
        train_subset, batch_sampler=train_batch_sampler,
        num_workers=16, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )

    # Model
    model_cfg = config_dict["model"]
    model = AblationModel(
        query_type=query_type,
        group_query_dim=group_query_dim,
        cnn_hidden_dim=model_cfg["cnn_hidden_dim"],
        cnn_kernel_sizes=tuple(model_cfg["cnn_kernel_sizes"]),
        cnn_dropout=model_cfg["cnn_dropout"],
        gcn_hidden_dim=model_cfg["gcn_hidden_dim"],
        gcn_out_channels=model_cfg["gcn_out_channels"],
        gcn_num_layers=model_cfg["gcn_num_layers"],
        gcn_dropout=model_cfg["gcn_dropout"],
        num_classes=model_cfg["num_classes"],
        num_attn_heads=model_cfg["num_attn_heads"],
        attn_dropout=model_cfg["attn_dropout"],
        use_layer_norm=model_cfg["use_layer_norm"],
        seq_len=1001
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # FLOPs calculation
    if query_type == "fullattn":
        flops_params = count_fullattn_params(group_query_dim, model_cfg)
        flops = _count_backbone_flops(model_cfg, seq_len=1001) + \
                count_fullattn_head_flops(group_query_dim, seq_len=1001, num_classes=model_cfg["num_classes"])
    else:
        flops_params = total_params
        flops = count_model_flops(model, seq_len=1001, batch_size=1)
    flops_m = flops / 1e6
    flops_g = flops / 1e9
    print(f"  FLOPs: {flops_m:,.1f}M ({flops_g:.3f}G)")

    # Loss / Optimizer / Scheduler
    pos_weight = pos_weight_unbalanced.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config_dict["training"]["learning_rate"],
        weight_decay=config_dict["training"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Training loop — train all epochs, evaluate once at the end
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        if isinstance(train_batch_sampler, DynamicBalancedBatchSampler):
            train_batch_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, logger=None,
            use_hierarchical=True,
            use_amp=config_dict["training"].get("use_amp", False) and torch.cuda.is_available(),
            use_attention_supervision=False, attention_lambda=0.0
        )
        print(f"  Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}")

    elapsed = time.time() - start_time

    # Single evaluation after all epochs
    y_true, y_prob, y_4class, y_4prob = get_all_predictions(
        model, test_loader, device, use_hierarchical=True
    )
    metrics_final = evaluate_unbalance(y_true, y_prob, device, y_4class, seed, y_4prob)

    auc_values = []
    per_class_auc = {}
    for c in range(12):
        key = f"group_class_{c}_auc"
        val = metrics_final.get(key, 0.0)
        per_class_auc[c] = val
        auc_values.append(val)
    best_auc = np.mean(auc_values) if auc_values else 0.0

    print(f"\n  [{cfg_name}] Avg AUC: {best_auc:.4f} | Time: {elapsed:.1f}s")

    result = {
        "name": cfg_name,
        "query_type": query_type,
        "group_query_dim": group_query_dim,
        "desc": ablation_cfg["desc"],
        "best_avg_auc": best_auc,
        "best_epoch": num_epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops_params": flops_params,
        "flops": flops,
        "flops_m": round(flops_m, 2),
        "flops_g": round(flops_g, 4),
        "train_time_s": elapsed,
    }
    for c in range(12):
        result[f"class_{c}_auc"] = per_class_auc[c]

    return result


def generate_visualizations(csv_path, output_dir):
    """Generate 4 ablation figures from the CSV results (4x4 matrix)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping visualization.")
        return

    # Read CSV
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("  [WARN] No results to visualize.")
        return

    # Build lookup: (query_type, dim) -> row
    lookup = {}
    for r in rows:
        key = (r["query_type"], int(r["group_query_dim"]))
        lookup[key] = r

    query_types = ["1query", "4query", "12query", "fullattn"]
    dims = [128, 256, 512, 1001]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    markers = ['o', 's', '^', 'D']

    # --- Figure 1: Line chart — each line is a query_type, X=dim, Y=avg AUC ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, qt in enumerate(query_types):
        aucs = []
        for d in dims:
            r = lookup.get((qt, d))
            aucs.append(float(r["best_avg_auc"]) if r else 0.0)
        ax.plot(dims, aucs, f'{markers[i]}-', color=colors[i], linewidth=2,
                markersize=8, label=qt)
        for d, auc in zip(dims, aucs):
            ax.annotate(f'{auc:.3f}', (d, auc), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('group_query_dim', fontsize=12)
    ax.set_ylabel('Average AUC', fontsize=12)
    ax.set_title('Ablation: Avg AUC vs group_query_dim per Query Type', fontsize=13)
    ax.set_xscale('log')
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_auc_vs_dim.png'), dpi=150)
    plt.close(fig)
    print("  Saved fig1_auc_vs_dim.png")

    # --- Figure 2: Grouped bar chart — X=dim (4 groups), bars=query_types ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dims))
    bar_width = 0.18
    for i, qt in enumerate(query_types):
        aucs = []
        for d in dims:
            r = lookup.get((qt, d))
            aucs.append(float(r["best_avg_auc"]) if r else 0.0)
        bars = ax.bar(x + i * bar_width, aucs, bar_width, label=qt,
                      color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)
    ax.set_xlabel('group_query_dim', fontsize=12)
    ax.set_ylabel('Average AUC', fontsize=12)
    ax.set_title('Ablation: Query Type x Dim (Grouped Bar)', fontsize=13)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_grouped_bar.png'), dpi=150)
    plt.close(fig)
    print("  Saved fig2_grouped_bar.png")

    # --- Figure 3: Heatmap — 16 configs x 12 classes ---
    from utils.common import MOD_NAMES
    config_names = [r["name"] for r in rows]
    class_cols = [f"class_{c}_auc" for c in range(12)]

    heatmap_data = []
    for r in rows:
        heatmap_data.append([float(r.get(col, 0.0)) for col in class_cols])
    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=1.0)
    ax.set_xticks(range(12))
    ax.set_xticklabels([MOD_NAMES[i] for i in range(12)], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=8)
    ax.set_title('Per-Class AUC across 16 Ablation Configurations', fontsize=13)

    for i in range(len(config_names)):
        for j in range(12):
            val = heatmap_data[i, j]
            color = 'white' if val > 0.75 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='AUC')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_class_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  Saved fig3_class_heatmap.png")

    # --- Figure 4: Heatmap — 4x4 summary (query_type x dim) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = np.zeros((4, 4))
    for i, qt in enumerate(query_types):
        for j, d in enumerate(dims):
            r = lookup.get((qt, d))
            summary[i, j] = float(r["best_avg_auc"]) if r else 0.0

    im = ax.imshow(summary, cmap='RdYlGn', aspect='auto', vmin=summary.min() - 0.02, vmax=summary.max() + 0.02)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(d) for d in dims], fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(query_types, fontsize=10)
    ax.set_xlabel('group_query_dim', fontsize=12)
    ax.set_ylabel('Query Type', fontsize=12)
    ax.set_title('Avg AUC: Query Type x Dim', fontsize=13)

    for i in range(4):
        for j in range(4):
            val = summary[i, j]
            color = 'white' if val > (summary.max() + summary.min()) / 2 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Avg AUC')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_summary_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  Saved fig4_summary_heatmap.png")


# ============================================================================
# Main
# ============================================================================

def main(config_path='json/abla_human.json'):
    print("="*60)
    print("  HierarchicalClassQueryHeadPooling Ablation Study")
    print("="*60)

    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config_dict["training"]["random_seed"]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config_dict["paths"].get("log_dir", "./logs_abla"), f"ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output dir: {output_dir}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset (shared across all ablation runs)
    print("\n  Loading dataset...")
    dataset = Mer100Dataset(
        mode='train',
        data_dir=config_dict["data"]["human_data_dir"],
        cache_dir=config_dict["data"]["cache_dir"],
        use_human3=True,
        use_cache=True
    )
    print(f"  Dataset: {len(dataset)} samples")

    # Precompute structures if needed
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        print("  Precomputing secondary structures...")
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)

    # Split (shared across all runs)
    train_indices, test_indices = multi_label_disjoint_split(
        dataset, train_ratio=config_dict["training"]["train_ratio"],
        random_seed=seed, logger=None
    )
    print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Pos weights
    pos_weight_unbalanced = get_smoothed_pos_weights(
        dataset, train_indices, num_classes=12, logger=None
    )

    # Run all ablation configurations
    all_results = []
    for i, ablation_cfg in enumerate(ABLATION_CONFIGS):
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(ABLATION_CONFIGS)}] Running: {ablation_cfg['name']}")
        print(f"{'#'*60}")

        try:
            result = run_single_ablation(
                ablation_cfg=ablation_cfg,
                config_dict=config_dict,
                dataset=dataset,
                train_indices=train_indices,
                test_indices=test_indices,
                pos_weight_unbalanced=pos_weight_unbalanced,
                device=device,
                output_dir=output_dir
            )
            all_results.append(result)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  [OOM] CUDA OOM for {ablation_cfg['name']} — skipping, filling NaN")
                torch.cuda.empty_cache()
                result = {
                    "name": ablation_cfg["name"],
                    "query_type": ablation_cfg["query_type"],
                    "group_query_dim": ablation_cfg["group_query_dim"],
                    "desc": ablation_cfg["desc"],
                    "best_avg_auc": float('nan'),
                    "best_epoch": 0,
                    "total_params": 0,
                    "trainable_params": 0,
                    "flops_params": 0,
                    "flops": 0,
                    "flops_m": 0.0,
                    "flops_g": 0.0,
                    "train_time_s": 0.0,
                }
                for c in range(12):
                    result[f"class_{c}_auc"] = float('nan')
                all_results.append(result)
            else:
                raise

    # Write CSV
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    fieldnames = [
        "name", "query_type", "group_query_dim", "desc", "best_avg_auc", "best_epoch",
        "total_params", "trainable_params", "flops_params", "flops", "flops_m", "flops_g",
        "train_time_s"
    ] + [f"class_{c}_auc" for c in range(12)]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'='*105}")
    print(f"{'Config':<18} {'Query':<10} {'Dim':>5} {'Avg AUC':>10} {'Epoch':>6} {'Params':>12} {'FLOPs(M)':>12} {'Time(s)':>8}")
    print(f"{'-'*105}")
    for r in all_results:
        print(f"{r['name']:<18} {r['query_type']:<10} {r['group_query_dim']:>5d} "
              f"{r['best_avg_auc']:>10.4f} {r['best_epoch']:>6d} "
              f"{r['total_params']:>12,} {r['flops_m']:>12.1f} {r['train_time_s']:>8.1f}")
    print(f"{'='*105}")

    # Generate visualizations
    print("\n  Generating visualizations...")
    generate_visualizations(csv_path, output_dir)

    print("\n  Ablation study complete!")


if __name__ == "__main__":
    main()
