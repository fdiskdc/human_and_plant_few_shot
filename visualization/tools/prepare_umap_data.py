"""
prepare_umap_data.py - 人类数据集 UMAP 数据准备 (含模型推理) / Prepare UMAP Data for Human Dataset

加载人类数据集,运行模型推理收集 MHA 注意力输出 + softmax 概率,过滤纯样本,分层采样,UMAP 降维,
计算 A/C/G/U 组的 KDE 密度等高线,输出 Web 可视化 JSON 文件。
Loads human dataset, runs model inference to collect MHA attention outputs + softmax probabilities, filters
pure samples, stratified sampling, UMAP reduction, KDE contours, outputs JSON for web visualization.

功能模块 / Modules:
- 模型推理 + 注意力收集 / Model inference + attention collection
- 纯样本过滤 + 分层采样 / Pure sample filtering + stratified sampling
- UMAP 降维 / UMAP dimensionality reduction
- KDE 密度等高线 / KDE density contours
- main: 主入口 / Main entry point

输入 / Inputs:
- json/human.json: 配置 / Config
- checkpoints/best_model.pt: 模型 / Model
- 命令行参数 / CLI: --config, --checkpoint, --output, --n_per_class, --batch_size, --device

输出 / Outputs:
- npy/umap_human_data.json: Web 可视化 JSON / Web visualization JSON
- 包含 UMAP 坐标、密度、组标签 / UMAP coords, density, group labels

数据流 / Data Flow:
1. 加载数据 / Load data
2. 模型推理 + 注意力收集 / Inference + attention collection
3. 过滤纯样本 / Filter pure samples
4. UMAP 降维 / UMAP reduction
5. KDE 等高线 / KDE contours
6. 保存 JSON / Save JSON

相关文件 / Related Files:
- 调用 / Calls: dataset.human_with_seq.Mer100DatasetWithSeq, model.mrmodn, umap
- 被调用 / Called by: web visualization

使用示例 / Usage Example:
    python prepare_umap_data.py --config json/human.json --checkpoint checkpoints/best_model.pt --output npy/umap_human_data.json

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from utils import load_config
from utils.common import MOD_NAMES, INDEX_TO_GROUP, GROUP_TO_CLASS_INDICES
from dataset.human_with_seq import Mer100DatasetWithSeq
from model.mrmodn_collect_atten import RNA_ClassQuery_Model_Collect_Atten

# Label names in original order (0-11)
LABEL_NAMES = ["Am", "Atol", "Cm", "Gm", "Tm", "Y", "ac4C", "m1A", "m5C", "m6A", "m6Am", "m7G"]

# Morandi color palette (consistent with notebook)
MORANDI_COLORS = {
    "Am": "#B9837D", "Atol": "#7A9CC6", "Cm": "#8FA68E",
    "Gm": "#A090B8", "Tm": "#D4B082", "Y": "#B8A08C",
    "ac4C": "#D6A3B8", "m1A": "#7DB5B5", "m5C": "#9BB89C",
    "m6A": "#B89595", "m6Am": "#A3B8C7", "m7G": "#B8A8C5"
}

GROUP_COLORS = {
    "A": "#EE5253", "C": "#2E86DE", "G": "#10AC84", "U": "#FF9F43"
}

# Group to class indices (0-11)
GROUP_MAPPING = {
    "A": [0, 1, 7, 9, 10],   # Am, Atol, m1A, m6A, m6Am
    "C": [2, 6, 8],           # Cm, ac4C, m5C
    "G": [3, 11],             # Gm, m7G
    "U": [4, 5]               # Tm, Y
}


def get_group_for_label(label_idx):
    """Get A/C/G/U group for a label index."""
    return INDEX_TO_GROUP.get(label_idx, 'U')


def compute_density_contours(umap_embeddings, labels, groups=['A', 'C', 'G', 'U']):
    """
    Compute KDE density contours for each nucleotide group.
    Uses scipy gaussian_kde + matplotlib contour to extract polygon vertices.
    """
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    density_contours = {}

    for group in groups:
        group_class_indices = GROUP_MAPPING[group]
        mask = np.isin(labels, group_class_indices)
        points = umap_embeddings[mask]

        if len(points) < 10:
            density_contours[group] = []
            continue

        try:
            kde = gaussian_kde(points.T)

            xmin, xmax = points[:, 0].min() - 1, points[:, 0].max() + 1
            ymin, ymax = points[:, 1].min() - 1, points[:, 1].max() + 1
            xi, yi = np.meshgrid(
                np.linspace(xmin, xmax, 80),
                np.linspace(ymin, ymax, 80)
            )
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

            level = np.percentile(zi.ravel(), 80)

            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contour(xi, yi, zi, levels=[level])
            plt.close(fig)

            polygons = []
            for collection in contour.collections:
                for path in collection.get_paths():
                    vertices = path.to_polygons()
                    if vertices:
                        poly = vertices[0].tolist()
                        if len(poly) >= 3:
                            polygons.append(poly)

            density_contours[group] = [{"level": float(level), "polygons": polygons}]
            plt.close('all')

        except Exception as e:
            print(f"  Warning: Could not compute contours for group {group}: {e}")
            density_contours[group] = []

    return density_contours


def main():
    parser = argparse.ArgumentParser(description='Prepare UMAP data for Human dataset')
    parser.add_argument('--config', type=str, default='json/human.json',
                        help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='npy/umap_human_data.json',
                        help='Output JSON path')
    parser.add_argument('--n-per-class', type=int, default=1000,
                        help='Number of samples per class')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.3,
                        help='UMAP min_dist parameter')
    args = parser.parse_args()

    print("=" * 60)
    print("RGCNFormer UMAP Data Preparation (Human Dataset)")
    print("=" * 60)

    # ============================================================
    # Step 1: Load configuration
    # ============================================================
    print("\n[Step 1/7] Loading configuration...")
    Config, config_dict = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"  Device: {device}")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")

    # ============================================================
    # Step 2: Load dataset
    # ============================================================
    print("\n[Step 2/7] Loading Human dataset...")
    dataset = Mer100DatasetWithSeq(
        mode='train',
        data_dir=Config.data.human_data_dir,
        cache_dir=Config.data.cache_dir,
        use_human3=True,
        use_cache=True,
        preload_cache=True
    )
    print(f"  Dataset: {len(dataset)} total samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"  DataLoader ready: {len(dataloader)} batches")

    # ============================================================
    # Step 3: Load model
    # ============================================================
    print("\n[Step 3/7] Loading model from checkpoint...")
    model = RNA_ClassQuery_Model_Collect_Atten(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=tuple(Config.cnn_kernel_sizes),
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=Config.num_classes,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=False,
        use_hierarchical=True,
        use_layer_norm=Config.use_layer_norm
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"  Warning: Checkpoint not found at {args.checkpoint}, using random weights")

    model.eval()
    print(f"  Model ready (gcn_out_channels={Config.gcn_out_channels})")

    # ============================================================
    # Step 4: Run inference
    # ============================================================
    print("\n[Step 4/7] Running inference to collect features and probabilities...")
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    autocast = torch.cuda.amp.autocast if use_amp else torch.no_grad

    accumulated_attn = []
    accumulated_probs = []
    accumulated_labels = []
    accumulated_seqs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Running inference", unit="batch")
        for batch in pbar:
            batch = batch.to(device)

            with autocast():
                logits_12, logits_4, attn_out_12, attn_out_4 = model(
                    batch.x, batch.edge_index, batch.batch
                )
                probs = F.softmax(logits_12, dim=-1)

            accumulated_attn.append(attn_out_12.detach().cpu().numpy())
            accumulated_probs.append(probs.detach().cpu().numpy())
            accumulated_labels.append(batch.y.squeeze(1).detach().cpu().numpy())

            if hasattr(batch, 'seq_str'):
                if isinstance(batch.seq_str, list):
                    accumulated_seqs.extend(batch.seq_str)
                else:
                    accumulated_seqs.extend(batch.seq_str.tolist())
            else:
                accumulated_seqs.extend([''] * batch.num_graphs)

            pbar.set_postfix({
                'acc': len(accumulated_attn) * args.batch_size,
                'attn_shape': accumulated_attn[-1].shape
            })

    all_attn = np.concatenate(accumulated_attn, axis=0)
    all_probs = np.concatenate(accumulated_probs, axis=0)
    all_labels = np.concatenate(accumulated_labels, axis=0)
    print(f"  Inference complete: {all_attn.shape}, {all_probs.shape}, {all_labels.shape}")

    # ============================================================
    # Step 5: Filter pure samples + stratified sampling
    # ============================================================
    print("\n[Step 5/7] Filtering pure samples (single-label) and stratified sampling...")

    row_sums = np.sum(all_labels, axis=1)
    pure_mask = row_sums == 1
    pure_indices = np.where(pure_mask)[0]
    print(f"  Pure samples (rowSums==1): {len(pure_indices)} / {len(all_labels)}")

    labels_final = np.argmax(all_labels, axis=1)

    sample_indices = []
    label_num_samples = {}
    for class_idx in range(12):
        curr_indices = pure_indices[labels_final[pure_indices] == class_idx]
        if len(curr_indices) == 0:
            continue

        n_take = min(len(curr_indices), args.n_per_class)
        sampled = random.sample(list(curr_indices), n_take)
        sample_indices.extend(sampled)

        label_name = LABEL_NAMES[class_idx]
        label_num_samples[label_name] = n_take
        print(f"  Class {class_idx} ({label_name}): {n_take} samples (from {len(curr_indices)} pure)")

    print(f"  Total sampled: {len(sample_indices)}")

    sampled_attn = all_attn[sample_indices]
    sampled_probs = all_probs[sample_indices]
    sampled_labels = labels_final[sample_indices]
    sampled_seqs = [accumulated_seqs[i] for i in sample_indices]

    # attn_out_12 shape: [N, 12, 128] -> flatten to [N, 1536]
    mat_for_umap = sampled_attn.reshape(len(sample_indices), -1)
    print(f"  UMAP input shape: {mat_for_umap.shape}")

    # ============================================================
    # Step 6: UMAP dimensionality reduction
    # ============================================================
    print("\n[Step 6/7] Running UMAP...")
    try:
        import umap
    except ImportError:
        print("  ERROR: umap-learn not installed. Install with: pip install umap-learn")
        sys.exit(1)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric='cosine',
        n_components=2,
        verbose=True,
        n_jobs=16
    )
    umap_embeddings = reducer.fit_transform(mat_for_umap)
    print(f"  UMAP complete: {umap_embeddings.shape}")

    # ============================================================
    # Step 7: Compute density contours
    # ============================================================
    print("\n[Step 7/7] Computing density contours...")
    density_contours = compute_density_contours(umap_embeddings, sampled_labels)
    print(f"  Contours computed for groups: {list(density_contours.keys())}")

    # ============================================================
    # Build and save output JSON
    # ============================================================
    print("\n[Output] Building JSON...")

    points = []
    for i in range(len(sample_indices)):
        label_idx = sampled_labels[i]
        label_name = LABEL_NAMES[label_idx]
        group = INDEX_TO_GROUP.get(label_idx, 'U')

        point = {
            "u1": float(umap_embeddings[i, 0]),
            "u2": float(umap_embeddings[i, 1]),
            "label": label_name,
            "group": group,
            "seq": sampled_seqs[i][:80] if sampled_seqs[i] else '',
            "probs": [float(p) for p in sampled_probs[i]]
        }
        points.append(point)

    output_data = {
        "points": points,
        "density_contours": density_contours,
        "metadata": {
            "n_per_class": args.n_per_class,
            "total_points": len(points),
            "valid_classes": list(range(12)),
            "color_map": MORANDI_COLORS,
            "group_colors": GROUP_COLORS,
            "label_names": LABEL_NAMES,
            "group_mapping": GROUP_MAPPING,
            "label_num_samples": label_num_samples
        }
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f)

    file_size_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"  Saved to: {args.output} ({file_size_mb:.2f} MB)")
    print(f"  Points: {len(points)}, Contours: {sum(len(v) for v in density_contours.values())}")
    print("\n" + "=" * 60)
    print("UMAP data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()