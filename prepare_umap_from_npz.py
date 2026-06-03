"""
prepare_umap_from_npz.py - 从预收集的 npz 准备 UMAP 数据 (离线) / Prepare UMAP Data from Pre-Collected NPZ

从预收集的 human_atten.npz 加载注意力输出,过滤纯样本 + 正确预测,可选每类数量上限,UMAP 降维,计算 KDE 等高线,输出 JSON。
Loads pre-collected human_atten.npz, filters pure samples + correct predictions, optional per-class cap,
UMAP reduction, KDE contours, outputs JSON.

功能模块 / Modules:
- npz 加载 / npz loading
- 纯样本 + 正确预测过滤 / Pure sample + correct prediction filtering
- UMAP 降维 / UMAP reduction
- KDE 密度等高线 / KDE density contours
- main: 主入口 / Main entry point

输入 / Inputs:
- npy/human_atten.npz: 预收集的注意力 / Pre-collected attention
- 命令行参数 / CLI: --input, --output, --n_per_class, --n_neighbors, --min_dist

输出 / Outputs:
- npy/umap_human_data.json: Web 可视化 JSON / Web visualization JSON
- 包含 UMAP 坐标、密度、组标签 / UMAP coords, density, group labels

数据流 / Data Flow:
1. 加载 npz / Load npz
2. 过滤纯样本 + 正确预测 / Filter pure + correct
3. UMAP 降维 / UMAP reduction
4. KDE 等高线 / KDE contours
5. 保存 JSON / Save JSON

相关文件 / Related Files:
- 调用 / Calls: utils.common, umap
- 被调用 / Called by: web visualization

使用示例 / Usage Example:
    python prepare_umap_from_npz.py --input npy/human_atten.npz --output npy/umap_human_data.json

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
from tqdm import tqdm

from utils.common import MOD_NAMES, INDEX_TO_GROUP, GROUP_TO_CLASS_INDICES

LABEL_NAMES = ["Am", "Atol", "Cm", "Gm", "Tm", "Y", "ac4C", "m1A", "m5C", "m6A", "m6Am", "m7G"]

MORANDI_COLORS = {
    "Am": "#B9837D", "Atol": "#7A9CC6", "Cm": "#8FA68E",
    "Gm": "#A090B8", "Tm": "#D4B082", "Y": "#B8A08C",
    "ac4C": "#D6A3B8", "m1A": "#7DB5B5", "m5C": "#9BB89C",
    "m6A": "#B89595", "m6Am": "#A3B8C7", "m7G": "#B8A8C5"
}

GROUP_COLORS = {
    "A": "#EE5253", "C": "#2E86DE", "G": "#10AC84", "U": "#FF9F43"
}

GROUP_MAPPING = {
    "A": [0, 1, 7, 9, 10],
    "C": [2, 6, 8],
    "G": [3, 11],
    "U": [4, 5]
}


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
            if hasattr(contour, 'allsegs') and contour.allsegs:
                for seg in contour.allsegs[0]:
                    if len(seg) >= 3:
                        polygons.append(seg.tolist())
            else:
                for collection in contour.collections:
                    for path in collection.get_paths():
                        vertices = path.to_polygons()
                        if vertices and len(vertices[0]) >= 3:
                            polygons.append(vertices[0].tolist())

            density_contours[group] = [{"level": float(level), "polygons": polygons}]
            plt.close('all')

        except Exception as e:
            print(f"  Warning: Could not compute contours for group {group}: {e}")
            density_contours[group] = []

    return density_contours


def main(input_path='npy/human_atten.npz', output_path='npy/umap_human_data.json',
         n_per_class=None, n_neighbors=30, min_dist=0.3):
    print("=" * 60)
    print("UMAP Data Preparation from NPZ (Human Dataset)")
    print("=" * 60)

    print(f"\n[Step 1/6] Loading NPZ: {input_path}")
    data = np.load(input_path, allow_pickle=True)

    attn_out_12 = data['attn_out_12']
    label12 = data['label12']
    seqs = data['seqs']
    probs12 = data['probs12']

    print(f"  attn_out_12: {attn_out_12.shape}")
    print(f"  label12: {label12.shape}")
    print(f"  seqs: {seqs.shape}")
    print(f"  probs12: {probs12.shape}")
    n_total = attn_out_12.shape[0]

    print(f"\n[Step 2/6] Filtering pure samples (rowSums==1)...")
    row_sums = np.sum(label12, axis=1)
    pure_mask = row_sums == 1
    pure_indices = np.where(pure_mask)[0]
    print(f"  Pure samples: {len(pure_indices)} / {n_total}")

    labels_final = np.argmax(label12, axis=1)
    predicted_labels = np.argmax(probs12, axis=1)

    print(f"\n[Step 3/6] Filtering correct predictions (argmax(probs)==argmax(label))...")
    correct_mask = predicted_labels == labels_final
    correct_indices = np.where(correct_mask)[0]
    filtered_indices = np.intersect1d(pure_indices, correct_indices)
    print(f"  Correct predictions: {len(correct_indices)} / {n_total}")
    print(f"  Pure + Correct: {len(filtered_indices)} / {n_total}")

    if n_per_class is not None:
        print(f"\n[Step 4/6] Capping at {n_per_class} per class...")
        sample_indices = []
        label_num_samples = {}
        for class_idx in range(12):
            curr_indices = filtered_indices[labels_final[filtered_indices] == class_idx]
            if len(curr_indices) == 0:
                continue
            n_take = min(len(curr_indices), n_per_class)
            sampled = random.sample(list(curr_indices), n_take)
            sample_indices.extend(sampled)
            label_name = LABEL_NAMES[class_idx]
            label_num_samples[label_name] = n_take
            print(f"  Class {class_idx} ({label_name}): {n_take} samples")
    else:
        print(f"\n[Step 4/6] Using all filtered samples (no per-class cap)...")
        sample_indices = filtered_indices.tolist()
        label_num_samples = {}
        for class_idx in range(12):
            n_in_class = int(np.sum(labels_final[filtered_indices] == class_idx))
            if n_in_class > 0:
                label_name = LABEL_NAMES[class_idx]
                label_num_samples[label_name] = n_in_class
                print(f"  Class {class_idx} ({label_name}): {n_in_class} samples")

    print(f"  Total samples: {len(sample_indices)}")

    sampled_attn = attn_out_12[sample_indices]
    sampled_probs = probs12[sample_indices]
    sampled_labels = labels_final[sample_indices]
    sampled_seqs = [str(s)[:80] if s else '' for s in seqs[sample_indices]]

    mat_for_umap = sampled_attn.reshape(len(sample_indices), -1)
    print(f"  UMAP input shape: {mat_for_umap.shape}")

    print(f"\n[Step 5/6] Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    try:
        import umap
    except ImportError:
        print("  ERROR: umap-learn not installed. Install with: pip install umap-learn")
        sys.exit(1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        n_components=2,
        verbose=True,
        n_jobs=16
    )
    umap_embeddings = reducer.fit_transform(mat_for_umap)
    print(f"  UMAP complete: {umap_embeddings.shape}")

    print(f"\n[Step 6/6] Computing density contours...")
    density_contours = compute_density_contours(umap_embeddings, sampled_labels)
    print(f"  Contours computed for groups: {list(density_contours.keys())}")

    print(f"\n[Output] Building JSON...")
    points = []
    for i in range(len(sample_indices)):
        label_idx = sampled_labels[i]
        label_name = LABEL_NAMES[label_idx]
        group = INDEX_TO_GROUP.get(label_idx, 'U')

        points.append({
            "u1": float(umap_embeddings[i, 0]),
            "u2": float(umap_embeddings[i, 1]),
            "label": label_name,
            "group": group,
            "seq": sampled_seqs[i],
            "probs": [float(p) for p in sampled_probs[i]]
        })

    output_data = {
        "points": points,
        "density_contours": density_contours,
        "metadata": {
            "n_per_class": n_per_class,
            "n_total_filtered": len(filtered_indices),
            "total_points": len(points),
            "valid_classes": list(range(12)),
            "color_map": MORANDI_COLORS,
            "group_colors": GROUP_COLORS,
            "label_names": LABEL_NAMES,
            "group_mapping": GROUP_MAPPING,
            "label_num_samples": label_num_samples
        }
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Saved to: {output_path} ({file_size_mb:.2f} MB)")
    print(f"  Points: {len(points)}, Contours: {sum(len(v) for v in density_contours.values())}")
    print("\n" + "=" * 60)
    print("UMAP data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare UMAP data from pre-collected human_atten.npz')
    parser.add_argument('--input', type=str, default='npy/human_atten.npz',
                        help='Path to input NPZ file')
    parser.add_argument('--output', type=str, default='npy/umap_human_data.json',
                        help='Output JSON path')
    parser.add_argument('--n-per-class', type=str, default='None',
                        help='Number of samples per class (None for no limit)')
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.3,
                        help='UMAP min_dist parameter')

    args = parser.parse_args()
    n_per_class = None if args.n_per_class == 'None' else int(args.n_per_class)
    main(
        input_path=args.input,
        output_path=args.output,
        n_per_class=n_per_class,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
