"""
utils/fewshot_analysis_spatial_motif.py - Captum 空间 motif 分析 / Captum Spatial Motif Analysis

使用 Captum Integrated Gradients 进行特征归因,生成 motif logo,可选 KMeans 聚类。
Uses Captum Integrated Gradients for feature attribution, generates motif logos, optional KMeans clustering.

功能模块 / Modules:
- Captum IG 归因 / Captum IG attribution
- Motif logo 生成 / Motif logo generation
- KMeans 聚类 / KMeans clustering
- main 分析函数 / Main analysis functions

输入 / Inputs:
- features, labels: 特征与标签 / Features and labels
- mod_type: 修饰类型 / Modification type
- n_clusters: 聚类数 / Number of clusters

输出 / Outputs:
- motif_logo/seq_*.png: motif logo
- 聚类可视化 / Cluster visualization
- 归因分数 / Attribution scores

数据流 / Data Flow:
1. IG 归因 / IG attribution
2. 提取序列上下文 / Extract sequence context
3. 聚类 (可选) / Cluster (optional)
4. 生成 logo / Generate logo
5. 保存 / Save

相关文件 / Related Files:
- 调用 / Calls: captum, logomaker, sklearn
- 被调用 / Called by: zero_shot_fewshot_analysis.py

使用示例 / Usage Example:
    from utils.fewshot_analysis_spatial_motif import run_spatial_motif_analysis
    run_spatial_motif_analysis(features, labels, mod_type='m6A')

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from dataset.gen3_zero import LABEL_MAPPING, MOD_NAMES, INDEX_TO_NUCLEOTIDE
from utils.fewshot_analysis_constants import (
    MIN_REL_POS, MAX_REL_POS, REL_RANGE, CENTER_IDX, NUC_TO_INDEX, MORANDI_COLORS
)

import matplotlib.pyplot as plt

# Try importing optional dependencies
try:
    import logomaker
    HAS_LOGOMAKER = True
except ImportError:
    HAS_LOGOMAKER = False

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


class ModelWrapper(nn.Module):
    """为 Integrated Gradients 定制的模型包装器 / Wrapper tailored for Integrated Gradients attribution."""
    def __init__(self, model, target_class_idx, edge_index, batch):
        """
        初始化包装器 / Initialize the wrapper.

        Args / 参数:
            model (nn.Module): [中文] 原始 GNN 模型 / [English] original GNN model.
            target_class_idx (int): [中文] 待归因的目标类索引 / [English] target class index
                to attribute.
            edge_index (LongTensor): [中文] 静态图边索引 / [English] static graph edge index.
            batch (LongTensor): [中文] 静态批索引 / [English] static batch index.
        """

        super().__init__()
        self.model = model
        self.target_class_idx = target_class_idx
        self.edge_index = edge_index
        self.batch = batch
        self.model.eval()

    def forward(self, x_flat):
        """
        前向传播, 只输出目标类标量 / Forward pass returning only the target-class logit.

        Captum 等归因工具要求输入展平为 `(N, F)` 并返回标量, 这里把
        `x_flat` reshape 回 `(N, 4)`, 复用固定 `edge_index` / `batch`, 取出
        `target_class_idx` 对应 logit。
        Captum-style attribution requires `(N, F)` flat inputs and scalar
        outputs, so we reshape `x_flat` back to `(N, 4)`, reuse the cached
        `edge_index` / `batch`, and return the logit at `target_class_idx`.

        Args / 参数:
            x_flat (Tensor): [中文] 展平后的节点特征, 形状 (N*4,) /
                [English] flattened node features, shape (N*4,).

        Returns / 返回:
            Tensor: [中文] 目标类 logit, 形状 () / [English] target-class logit, shape ().
        """

        x = x_flat.view(-1, 4)
        if hasattr(self.model, 'use_hierarchical') and self.model.use_hierarchical:
            output = self.model(x, self.edge_index, self.batch, return_attention=True)
            logits_12 = output[0]
        else:
            output = self.model(x, self.edge_index, self.batch)
            if isinstance(output, tuple):
                logits_12 = output[0]
            else:
                logits_12 = output
        return logits_12[0, self.target_class_idx]


def calculate_spatial_attribution(
    model, dataset, target_class_idx, target_indices, device,
    n_steps=20, internal_batch_size=4, batch_size=128
):
    """
    Calculate spatial attribution for specified target samples using Integrated Gradients.
    """
    REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
    original_label_id = REVERSE_LABEL_MAPPING.get(target_class_idx)
    aggregated_importance = np.zeros((REL_RANGE, 4), dtype=np.float32)

    print(f"\nComputing spatial attribution for {len(target_indices)} samples...")

    if len(target_indices) == 0:
        print("Warning: No target indices provided!")
        return aggregated_importance

    processed_count = 0
    class_name = MOD_NAMES.get(target_class_idx, str(target_class_idx))

    for i in range(0, len(target_indices), batch_size):
        batch_idxs = target_indices[i:i+batch_size]

        for idx in tqdm(batch_idxs, desc=f"Processing {class_name} samples", leave=False):
            try:
                data = dataset[idx]
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                y_site = data.y_site
                batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                anchor_indices = torch.where(y_site == original_label_id)[0].tolist()
                if len(anchor_indices) == 0:
                    continue

                model.eval()
                x_flat = x.flatten()
                x_attrib = x_flat.clone().detach().requires_grad_(True)

                sample_wrapper = ModelWrapper(model, target_class_idx, edge_index, batch)
                sample_ig = IntegratedGradients(sample_wrapper)

                try:
                    attributions_flat = sample_ig.attribute(
                        x_attrib.unsqueeze(0),
                        n_steps=n_steps,
                        internal_batch_size=internal_batch_size
                    )
                    attributions = attributions_flat.view(1001, 4)
                except Exception:
                    x_grad = x.clone().detach().requires_grad_(True)
                    fallback_wrapper = ModelWrapper(model, target_class_idx, edge_index, batch)
                    output = fallback_wrapper(x_grad.flatten().unsqueeze(0))
                    output.backward()
                    attributions = x_grad.grad

                sample_attrib_matrix = attributions.cpu().detach().numpy()

                for anchor_pos in anchor_indices:
                    start_rel = MIN_REL_POS
                    end_rel = MAX_REL_POS
                    start_abs = max(0, anchor_pos + start_rel)
                    end_abs = min(1001, anchor_pos + end_rel + 1)

                    rel_idx_start = (start_abs - anchor_pos) - MIN_REL_POS
                    rel_idx_end = (end_abs - anchor_pos) - MIN_REL_POS

                    aggregated_importance[rel_idx_start:rel_idx_end, :] += sample_attrib_matrix[start_abs:end_abs, :]
                    processed_count += 1

            except Exception as e:
                print(f"Warning: Failed to process sample {idx}: {e}")
                continue

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if processed_count > 0:
        aggregated_importance /= processed_count
        print(f"Successfully processed {processed_count} samples.")
    else:
        print("Warning: No samples were successfully processed!")

    return aggregated_importance


def plot_top_k_logo(aggregated_importance, class_idx, class_name, node_num=10, output_dir="spatial_motif"):
    """
    Generate and save a Sequence Logo for the given attribution matrix.
    Uses hard zeroing (no background) approach.
    """
    if not HAS_LOGOMAKER:
        print("Warning: logomaker not installed, skipping spatial motif plot.")
        return ""
    os.makedirs(output_dir, exist_ok=True)

    saliency_matrix = np.abs(aggregated_importance)
    center_idx = CENTER_IDX
    num_positions = saliency_matrix.shape[0]

    # Otsu's method for position filtering
    position_otsu_scores = {}
    IDX_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

    for pos in range(num_positions):
        if pos == center_idx:
            continue

        base_values = saliency_matrix[pos, :].copy()
        base_sum = np.sum(base_values)
        if base_sum < 1e-9:
            continue
        normalized_values = base_values / base_sum

        sorted_indices = np.argsort(normalized_values)[::-1]
        sorted_values = normalized_values[sorted_indices]

        max_between_variance = -1.0
        best_signal_bases = []

        for split_point in [1, 2, 3]:
            signal_values = sorted_values[:split_point]
            background_values = sorted_values[split_point:]

            omega_0 = len(signal_values) / 4.0
            omega_1 = len(background_values) / 4.0
            mu_0 = np.mean(signal_values) if len(signal_values) > 0 else 0.0
            mu_1 = np.mean(background_values) if len(background_values) > 0 else 0.0
            between_variance = omega_0 * omega_1 * (mu_0 - mu_1) ** 2

            if between_variance > max_between_variance:
                max_between_variance = between_variance
                best_signal_bases = [IDX_TO_NUC[sorted_indices[i]] for i in range(split_point)]

        position_otsu_scores[pos] = (max_between_variance, best_signal_bases)

    # Dynamic baseline filtering
    if position_otsu_scores:
        all_variances = np.array([v[0] for v in position_otsu_scores.values()])
        baseline_noise_level = np.median(all_variances)
        valid_positions = [
            pos for pos, (variance, _) in position_otsu_scores.items()
            if variance > baseline_noise_level
        ]
    else:
        valid_positions = []

    # Sort and truncate Top-K
    if valid_positions:
        valid_position_scores = [
            (pos, np.sum(saliency_matrix[pos, :]))
            for pos in valid_positions
        ]
        valid_position_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(node_num, len(valid_position_scores))
        top_indices = np.array([valid_position_scores[i][0] for i in range(top_k)])
    else:
        temp_scores = np.sum(saliency_matrix, axis=1)
        temp_scores[center_idx] = -1.0
        top_indices = np.argsort(temp_scores)[-node_num:]

    # Build logo matrix with hard zeroing
    all_indices = np.append(top_indices, center_idx)
    all_indices = np.sort(all_indices)

    logo_matrix_raw = saliency_matrix[all_indices]
    row_sums = np.sum(logo_matrix_raw, axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0
    logo_matrix_norm = logo_matrix_raw / row_sums

    # Build significant_dict
    significant_dict = {}
    for local_idx, global_pos in enumerate(all_indices):
        if global_pos == center_idx:
            significant_dict[local_idx] = ['A', 'C', 'G', 'U']
            continue
        if global_pos in position_otsu_scores:
            _, sig_bases = position_otsu_scores[global_pos]
            if sig_bases:
                significant_dict[local_idx] = sig_bases

    # Hard zeroing
    anchor_local_idx = np.where(all_indices == center_idx)[0][0]
    for local_idx in range(logo_matrix_norm.shape[0]):
        if local_idx == anchor_local_idx:
            continue
        if local_idx in significant_dict:
            sig_bases = significant_dict[local_idx]
            for nuc, col_idx in NUC_TO_INDEX.items():
                if nuc not in sig_bases:
                    logo_matrix_norm[local_idx, col_idx] = 0.0

    # Position 0 handling
    target_nuc = INDEX_TO_NUCLEOTIDE.get(class_idx, 'N')
    if target_nuc in NUC_TO_INDEX:
        target_col = NUC_TO_INDEX[target_nuc]
        logo_matrix_norm[anchor_local_idx, :] = 0
        logo_matrix_norm[anchor_local_idx, target_col] = 1.0

    logo_df = pd.DataFrame(logo_matrix_norm, columns=['A', 'C', 'G', 'U'])

    # Plotting
    fig, ax = plt.subplots(figsize=(max(10, node_num * 0.8), 6))
    fig.patch.set_facecolor('#eceff2')
    ax.set_facecolor('#eceff2')

    logo = logomaker.Logo(logo_df, ax=ax, color_scheme=MORANDI_COLORS,
                          font_name='DejaVu Sans', center_values=False)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_yticks([])
    ax.set_yticklabels([])
    logo.style_spines(spines=['left'], visible=False)

    ax.set_title(f"{class_name}: Spatial Motif (Top {node_num} Context, Hard Zeroing)",
                 fontsize=30, fontfamily='sans-serif', fontweight='bold', color='#2D3748', pad=15)
    ax.set_ylim(0, 1.05)

    real_rel_positions = [idx - CENTER_IDX for idx in all_indices]
    ax.set_xticks(range(len(all_indices)))
    ax.set_xticklabels(real_rel_positions, rotation=90 if len(str(max(real_rel_positions))) > 3 else 0,
                       fontfamily='sans-serif', fontsize=20, color='#4A5568')

    anchor_plot_idx = int(np.where(all_indices == center_idx)[0][0])
    logo.highlight_position(p=anchor_plot_idx, color='#9E2A2B', alpha=0.6)

    x_labels = ax.get_xticklabels()
    if len(x_labels) > anchor_plot_idx:
        x_labels[anchor_plot_idx].set_fontweight('bold')
        x_labels[anchor_plot_idx].set_color('#2D3748')

    safe_name = class_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'motif_logo_{safe_name}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial motif logo to {output_path}")
    return output_path


def extract_embeddings(model, dataset, indices, device):
    """Extract graph embeddings from the model for clustering."""
    model.eval()
    embeddings_list = []
    idx_list = []

    target_module_path = 'class_query_head.mha_12'
    hook_handle = None
    activation = {}

    def get_hook(name):
        """
        返回一个 closure hook, 记录 module 输出到 `activation[name]` /
        Return a closure that records a module's output to `activation[name]`.

        Args / 参数:
            name (str): [中文] 输出键名 / [English] key for the activation dict.

        Returns / 返回:
            Callable: [中文] PyTorch forward hook / [English] PyTorch forward hook.
        """

        def hook(module, input, output):
            """
            实际 hook: 缓存 `output[0]` (tuple 时) 或 `output`, 并 detach /
            The actual hook: caches `output[0]` (for tuples) or `output`, then detaches.
            """

            if isinstance(output, tuple):
                activation[name] = output[0].detach()
            else:
                activation[name] = output.detach()
        return hook

    target_module = None
    for name, module in model.named_modules():
        if name == target_module_path:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Could not find module '{target_module_path}' in the model.")

    hook_handle = target_module.register_forward_hook(get_hook(target_module_path))

    try:
        with torch.no_grad():
            for idx in tqdm(indices, desc="Extracting embeddings"):
                try:
                    data = dataset[idx]
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                    activation.clear()
                    _ = model(x, edge_index, batch)

                    node_features = activation.get(target_module_path)
                    if node_features is not None:
                        if node_features.dim() == 3:
                            sample_embedding = node_features.flatten().cpu().numpy()
                        else:
                            sample_embedding = node_features.mean(dim=0).cpu().numpy()
                        embeddings_list.append(sample_embedding)
                        idx_list.append(idx)
                except Exception as e:
                    print(f"Warning: Failed to extract embedding for sample {idx}: {e}")
                    continue
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if len(embeddings_list) == 0:
        raise ValueError("No embeddings were successfully extracted!")

    embeddings = np.array(embeddings_list)
    print(f"Successfully extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    return embeddings


def cluster_samples(embeddings, indices, n_clusters=3, pca_components=None, random_state=42):
    """Cluster samples in latent space using PCA + K-Means."""
    print(f"\nClustering {len(embeddings)} samples into {n_clusters} clusters...")

    if pca_components is not None and pca_components < embeddings.shape[1]:
        print(f"Applying PCA: {embeddings.shape[1]} -> {pca_components} dimensions...")
        pca = PCA(n_components=pca_components, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.4f}")
        clustering_input = embeddings_reduced
    else:
        print("Skipping PCA (using original embeddings)...")
        clustering_input = embeddings

    print(f"Running K-Means with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(clustering_input)

    clusters_dict = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = indices[cluster_mask]
        clusters_dict[cluster_id] = cluster_indices
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} samples")

    print(f"\nClustering complete. Inertia: {kmeans.inertia_:.4f}")
    return clusters_dict


def run_spatial_motif_analysis(
    model, dataset, dataset_name, target_classes, output_dir, device,
    n_clusters=3, pca_components=50, node_num=10, n_steps=20,
    internal_batch_size=512, batch_size=512
):
    """
    Run spatial motif analysis for a given dataset.
    
    Args:
        model: The RNA_ClassQuery_Model
        dataset: Dataset to analyze
        dataset_name: Name of the dataset (e.g., 'plant', 'gen3')
        target_classes: List of (class_idx, class_name) tuples
        output_dir: Output directory
        device: torch.device
        n_clusters: Number of clusters
        pca_components: PCA components for dimensionality reduction
        node_num: Number of context positions for motif
        n_steps: Number of steps for Integrated Gradients
        internal_batch_size: Internal batch size for IG
        batch_size: Batch size for processing samples
    """
    if not HAS_LOGOMAKER or not HAS_CAPTUM:
        print(f"Warning: logomaker or captum not installed, skipping spatial motif analysis for {dataset_name}.")
        return

    motif_output_dir = os.path.join(output_dir, f'spatial_motif_{dataset_name}')
    os.makedirs(motif_output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Spatial Motif Analysis for {dataset_name}")
    print(f"{'='*80}")

    for class_idx, class_name in target_classes:
        print(f"\n{'='*80}")
        print(f"Analyzing {class_name} (Class Index: {class_idx})")
        print(f"{'='*80}")

        # Find all positive samples for this class
        if hasattr(dataset, 'y_12class'):
            all_labels = dataset.y_12class
        elif hasattr(dataset, 'y'):
            all_labels = dataset.y
        else:
            print(f"Warning: Dataset does not have y_12class or y attribute, skipping {class_name}.")
            continue

        positive_indices = np.where(all_labels[:, class_idx] == 1)[0]
        total_found = len(positive_indices)

        if total_found == 0:
            print(f"Warning: No samples found for {class_name} in {dataset_name}!")
            continue

        print(f"Found {total_found} positive samples for {class_name}.")

        # Extract embeddings for clustering
        print(f"\nStep 1: Extracting latent embeddings...")
        try:
            embeddings = extract_embeddings(model, dataset, positive_indices, device)
        except ValueError as e:
            print(f"Error during embedding extraction: {e}")
            continue

        # Cluster samples
        print(f"\nStep 2: Clustering samples...")
        try:
            clusters_dict = cluster_samples(
                embeddings, positive_indices,
                n_clusters=n_clusters, pca_components=pca_components, random_state=42
            )
        except Exception as e:
            print(f"Error during clustering: {e}")
            continue

        # Compute attribution and plot logo for each cluster
        print(f"\nStep 3: Computing cluster-specific attributions and generating logos...")
        for cluster_id, cluster_indices in clusters_dict.items():
            print(f"\n{'─'*60}")
            print(f"Cluster {cluster_id}: {len(cluster_indices)} samples")
            print(f"{'─'*60}")

            imp_matrix = calculate_spatial_attribution(
                model, dataset, class_idx, cluster_indices, device,
                n_steps=n_steps, internal_batch_size=internal_batch_size, batch_size=batch_size
            )

            cluster_class_name = f"{class_name}_Cluster_{cluster_id}"
            plot_top_k_logo(
                imp_matrix, class_idx, cluster_class_name,
                node_num=node_num, output_dir=motif_output_dir
            )

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Spatial motif analysis complete for {dataset_name}!")
    print(f"Logos saved to: {motif_output_dir}/")
    print(f"{'='*80}")
