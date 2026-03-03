#!/usr/bin/env python3
"""
SpatialMotif.py - Visualize "Spatial Motifs" with Top-K Sequence Logos

NEW FEATURE: Latent Space Activation Clustering
- Extract embeddings from GCN layer (before classification head)
- Cluster samples in latent space using PCA + K-Means
- Generate separate Sequence Logos for each cluster
This prevents "signal washout" caused by averaging across different topologies.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Clustering imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Try importing optional dependencies
try:
    import logomaker
    HAS_LOGOMAKER = True
except ImportError:
    HAS_LOGOMAKER = False
    print("Warning: logomaker not installed. Please install with: pip install logomaker")

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Error: captum is required. Install with: pip install captum")
    exit(1)

# Import project-specific modules
from model.main_model import RNA_ClassQuery_Model
from utils import load_config

# Import the new Motif Dataset class
from dataset.human_motif import Mer100DatasetMotif, LABEL_MAPPING, INDEX_TO_NUCLEOTIDE

# 12类修饰名称映射 (Copied here for consistency)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}

# ============================================================================
# Configuration & Constants
# ============================================================================

SEQ_LENGTH = 1001
NUCLEOTIDES = ['A', 'C', 'G', 'U']
MIN_REL_POS = -1000
MAX_REL_POS = 1000
REL_RANGE = (MAX_REL_POS - MIN_REL_POS) + 1  # 2001 positions
CENTER_IDX = -MIN_REL_POS
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Morandi Liquid Color Scheme (RGBA with transparency for glass effect)
MORANDI_COLORS = {
    'A': (95/255, 158/255, 160/255, 0.80),   # Cadet Blue - 灰蓝
    'C': (188/255, 143/255, 143/255, 0.75),  # Rosy Brown - 豆沙灰粉
    'G': (143/255, 188/255, 143/255, 0.80),  # Dark Sea Green - 灰绿
    'U': (218/255, 165/255, 32/255, 0.78)    # Golden Rod - 金灰
}

NUC_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


# ============================================================================
# Model Wrapper
# ============================================================================

class ModelWrapper(nn.Module):
    def __init__(self, model: RNA_ClassQuery_Model, target_class_idx: int, edge_index, batch):
        super().__init__()
        self.model = model
        self.target_class_idx = target_class_idx
        self.edge_index = edge_index
        self.batch = batch
        self.model.eval()

    def forward(self, x_flat):
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


# ============================================================================
# Feature Extraction via Hooks (NEW)
# ============================================================================

def extract_embeddings(
    model: RNA_ClassQuery_Model,
    dataset: Mer100DatasetMotif,
    indices: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Extract graph embeddings from the GCN layer (before classification head).

    This function uses PyTorch's register_forward_hook to capture the node_features
    output from the GCN block. This is the latent space representation that the
    classification head uses for making predictions.

    NOTE: If you need to hook to a different layer, modify the target_module_path below.
    Common options:
        - 'gcn_block' (default): Output of GCN layers, after graph convolution
        - 'cnn_block': Output of CNN feature extraction
        - 'class_query_head': Input to the classification head (same as gcn_block output)

    Args:
        model: The RNA_ClassQuery_Model
        dataset: Mer100DatasetMotif dataset
        indices: Sample indices to extract embeddings for
        device: torch.device

    Returns:
        embeddings: np.ndarray of shape [num_samples, embedding_dim]
                   For GCN output, embedding_dim = gcn_out_channels (default 128)
                   We use global mean pooling to aggregate node features to a single vector per sample
    """
    model.eval()

    # Storage for extracted embeddings
    embeddings_list = []
    idx_list = []

    # Hook function to capture GCN output
    # ========================================================================
    # IMPORTANT: Hook target configuration
    # ========================================================================
    # The hook captures the output of the GCN block, which is the node_features
    # that are passed to the classification head. This is the latent space
    # representation of the input RNA sequence after graph convolution.
    #
    # If you want to hook to a different layer, modify target_module_path:
    #   - 'gcn_block': Output of GCN (default) - graph-convolved node features
    #   - 'cnn_block': Output of CNN - local multi-scale features
    #   - 'class_query_head.mha_12': Output of 12-class attention (if use_simple_pooling)
    #   - 'class_query_head.mha_12': Output of 12-class MHA (if use_hierarchical)
    #
    # To inspect the model structure and find the correct layer name, run:
    #   print(model)
    # or
    #   for name, module in model.named_modules():
    #       print(f"{name}: {type(module)}")
    # ========================================================================
    # target_module_path = 'gcn_block'
    target_module_path = 'class_query_head.mha_12'

    # Forward hook container
    hook_handle = None
    activation = {}

    def get_hook(name):
        def hook(module, input, output):
            # MultiheadAttention returns a tuple: (attn_output, attn_weights)
            # We only need the attn_output for clustering
            if isinstance(output, tuple):
                activation[name] = output[0].detach()  # Take the first element (attn_out)
            else:
                activation[name] = output.detach()
        return hook

    # Register the hook
    # We need to traverse the model to find the target module
    target_module = None
    for name, module in model.named_modules():
        if name == target_module_path:
            target_module = module
            break

    if target_module is None:
        raise ValueError(
            f"Could not find module '{target_module_path}' in the model. "
            f"Please check the model structure by printing model.named_modules() "
            f"and update target_module_path in extract_embeddings()."
        )

    hook_handle = target_module.register_forward_hook(get_hook(target_module_path))

    print(f"\n{'='*60}")
    print(f"Extracting embeddings from '{target_module_path}'...")
    print(f"{'='*60}")

    try:
        with torch.no_grad():
            for idx in tqdm(indices, desc="Extracting embeddings"):
                try:
                    data = dataset[idx]
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                    # Clear previous activation
                    activation.clear()

                    # Forward pass (hook will capture gcn_block output)
                    _ = model(x, edge_index, batch)

                    # Get the captured activation
                    node_features = activation.get(target_module_path)

                    if node_features is not None:
                        # Handle different output shapes depending on hook target:
                        # - gcn_block: [num_nodes, hidden_dim] -> aggregate nodes
                        # - mha_12: [batch, num_classes, hidden_dim] -> already has class info
                        if node_features.dim() == 3:  # [Batch, Num_Classes, Hidden_Dim]
                            # Flatten: [Batch, Num_Classes * Hidden_Dim]
                            # This preserves class-specific information for clustering
                            sample_embedding = node_features.flatten().cpu().numpy()
                        else:  # [Num_Nodes, Hidden_Dim]
                            # Aggregate node features to a single vector per sample
                            sample_embedding = node_features.mean(dim=0).cpu().numpy()
                        embeddings_list.append(sample_embedding)
                        idx_list.append(idx)
                    else:
                        print(f"Warning: No activation captured for sample {idx}")

                except Exception as e:
                    print(f"Warning: Failed to extract embedding for sample {idx}: {e}")
                    continue

    finally:
        # Always remove the hook
        if hook_handle is not None:
            hook_handle.remove()

    if len(embeddings_list) == 0:
        raise ValueError("No embeddings were successfully extracted!")

    embeddings = np.array(embeddings_list)
    print(f"Successfully extracted {len(embeddings)} embeddings with shape {embeddings.shape}")

    return embeddings


# ============================================================================
# Clustering Functions (NEW)
# ============================================================================

def cluster_samples(
    embeddings: np.ndarray,
    indices: np.ndarray,
    n_clusters: int = 3,
    pca_components: Optional[int] = None,
    random_state: int = 42
) -> Dict[int, np.ndarray]:
    """
    Cluster samples in latent space using PCA + K-Means.

    This function performs dimensionality reduction (optional but recommended for
    high-dimensional embeddings) followed by K-Means clustering to group samples
    with similar latent representations.

    Args:
        embeddings: np.ndarray of shape [num_samples, embedding_dim]
        indices: Original sample indices corresponding to each embedding
        n_clusters: Number of clusters to create
        pca_components: If specified, reduce dimensions to this value using PCA.
                        If None, skip PCA. Recommended: 20-50 for 128-dim embeddings.
        random_state: Random seed for reproducibility

    Returns:
        clusters_dict: Dictionary mapping cluster_id -> array of sample indices
                      {cluster_id: [sample_idx_1, sample_idx_2, ...]}
    """
    print(f"\n{'='*60}")
    print(f"Clustering {len(embeddings)} samples into {n_clusters} clusters...")
    print(f"{'='*60}")

    # Optional: PCA for dimensionality reduction
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

    # K-Means clustering
    print(f"Running K-Means with n_clusters={n_clusters}...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # Number of centroid initializations
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(clustering_input)

    # Organize samples by cluster
    clusters_dict = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = indices[cluster_mask]
        clusters_dict[cluster_id] = cluster_indices
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} samples")

    # Optional: Print cluster centroids distance (for debugging)
    print(f"\nClustering complete. Inertia: {kmeans.inertia_:.4f}")

    return clusters_dict


# ============================================================================
# Core Attribution Function (REFACTORED)
# ============================================================================

def calculate_spatial_attribution(
    model: RNA_ClassQuery_Model,
    dataset: Mer100DatasetMotif,
    target_class_idx: int,
    target_indices: np.ndarray,
    device: torch.device,
    n_steps: int = 20,
    internal_batch_size: int = 4,
    batch_size: int = 128
) -> np.ndarray:
    """
    Calculate spatial attribution for specified target samples using Integrated Gradients.

    REFACTORED: Now accepts explicit target_indices instead of searching the dataset.
    This allows us to compute attributions cluster-by-cluster.

    Args:
        model: The RNA_ClassQuery_Model
        dataset: Mer100DatasetMotif dataset
        target_class_idx: Target class index (0-11)
        target_indices: Explicit list of sample indices to compute attributions for
        device: torch.device
        n_steps: Number of steps for Integrated Gradients
        internal_batch_size: Internal batch size for IG computation
        batch_size: Batch size for processing samples

    Returns:
        aggregated_importance: Averaged attribution matrix of shape [REL_RANGE, 4]
    """
    original_label_id = REVERSE_LABEL_MAPPING.get(target_class_idx)
    aggregated_importance = np.zeros((REL_RANGE, 4), dtype=np.float32)

    print(f"\nComputing attribution for {len(target_indices)} samples...")

    if len(target_indices) == 0:
        print("Warning: No target indices provided!")
        return aggregated_importance

    processed_count = 0
    class_name = MOD_NAMES[target_class_idx]

    # Batch processing loop
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

        # Explicitly clear GPU cache after each batch to prevent OOM
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if processed_count > 0:
        aggregated_importance /= processed_count
        print(f"Successfully processed {processed_count} samples.")
    else:
        print("Warning: No samples were successfully processed!")

    return aggregated_importance


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_top_k_logo(
    aggregated_importance: np.ndarray,
    class_idx: int,
    class_name: str,
    node_num: int = 10,
    output_dir: str = "motif_logo_clustered"  # Changed default output directory
) -> str:
    """
    Generate and save a Sequence Logo for the given attribution matrix.

    The class_name parameter is now used to generate unique filenames for each cluster.
    Example: class_name = "m6A_Cluster_0" will save to "motif_logo_clustered/motif_logo_m6A_Cluster_0.pdf"
    """
    if not HAS_LOGOMAKER:
        return ""
    os.makedirs(output_dir, exist_ok=True)

    # Strategy: Take Absolute value (Saliency)
    saliency_matrix = np.abs(aggregated_importance)

    # 1. Calculate Importance Score for sorting (Sum of RAW saliency scores)
    importance_scores = np.sum(saliency_matrix, axis=1)

    # 2. Force include Center
    center_idx = CENTER_IDX
    temp_scores = importance_scores.copy()
    temp_scores[center_idx] = -1.0  # Exclude from top-k selection

    # 3. Select Top K context positions
    top_indices = np.argsort(temp_scores)[-node_num:]

    # 4. Combine and Sort Indices
    all_indices = np.append(top_indices, center_idx)
    all_indices = np.sort(all_indices)

    # 5. Extract Sub-Matrix
    logo_matrix_raw = saliency_matrix[all_indices]

    # --- [MODIFICATION] L1 Normalization (Sum to 1) ---
    row_sums = np.sum(logo_matrix_raw, axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0

    logo_matrix_norm = logo_matrix_raw / row_sums

    # --- Position 0 Handling ---
    anchor_local_idx = np.where(all_indices == center_idx)[0][0]

    target_nuc = INDEX_TO_NUCLEOTIDE.get(class_idx, 'N')
    if target_nuc in NUC_TO_INDEX:
        target_col = NUC_TO_INDEX[target_nuc]

        # 1. Zero out all nucleotides at center
        logo_matrix_norm[anchor_local_idx, :] = 0

        # 2. Set the target nucleotide to 1.0 (Max importance)
        logo_matrix_norm[anchor_local_idx, target_col] = 1.0

    logo_df = pd.DataFrame(logo_matrix_norm, columns=['A', 'C', 'G', 'U'])

    # 6. Plotting with Frosted Glass Background
    fig, ax = plt.subplots(figsize=(max(10, node_num * 0.8), 6))

    # Set frosted glass background (light advanced grey)
    fig.patch.set_facecolor('#eceff2')
    ax.set_facecolor('#eceff2')

    # Morandi Colors with rounded liquid font
    logo = logomaker.Logo(logo_df,
                         ax=ax,
                         color_scheme=MORANDI_COLORS,
                         font_name='DejaVu Sans',  # Rounded liquid font
                         center_values=False)

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)

    # Hide y-axis and y-axis label
    ax.set_yticks([])
    ax.set_yticklabels([])
    logo.style_spines(spines=['left'], visible=False)

    # 7. Add 3D Glass Effects (PathEffects) to all glyphs
    highlight = path_effects.Stroke(linewidth=0.8,
                                   foreground=(1, 1, 1, 0.6),
                                   alpha=0.7)

    shadow = path_effects.SimplePatchShadow(offset=(1.5, -1.5),
                                            alpha=0.4,
                                            rho=0.5)

    # Adaptive Soft Consensus Glow Effect (淡黄色发光)
    glow = path_effects.Stroke(linewidth=3.5*2, foreground='#FFD700', alpha=0.8)
    normal = path_effects.Normal()

    # Calculate which positions/letters need highlighting (最小满足集合原则)
    IDX_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    highlight_dict = {}  # 格式: {position_index: ['A', 'G']}

    for i in range(len(all_indices)):
        # 排除中心锚点
        if i == anchor_local_idx:
            continue

        vals = logo_matrix_norm[i, :]
        sorted_idx = np.argsort(vals)[::-1]

        top1_val = vals[sorted_idx[0]]
        top2_val = top1_val + vals[sorted_idx[1]]
        top3_val = top2_val + vals[sorted_idx[2]]

        highlight_chars = []
        # 最小满足集合逻辑 (50%, 85%, 95%)
        if top1_val > 0.50:
            highlight_chars.append(IDX_TO_NUC[sorted_idx[0]])
        elif top2_val > 0.85:
            highlight_chars.append(IDX_TO_NUC[sorted_idx[0]])
            highlight_chars.append(IDX_TO_NUC[sorted_idx[1]])
        elif top3_val > 0.95:
            highlight_chars.append(IDX_TO_NUC[sorted_idx[0]])
            highlight_chars.append(IDX_TO_NUC[sorted_idx[1]])
            highlight_chars.append(IDX_TO_NUC[sorted_idx[2]])

        if highlight_chars:
            highlight_dict[i] = highlight_chars

    # Apply path_effects with conditional glow for highlighted letters
    for glyph in logo.glyph_list:
        if hasattr(glyph, 'patch') and glyph.patch is not None:
            # 检查当前字符是否需要高亮
            if glyph.p in highlight_dict and glyph.c in highlight_dict[glyph.p]:
                # 叠加发光特效：阴影 -> 发光 -> 高光 -> 本体
                glyph.patch.set_path_effects([shadow, glow, highlight, normal])
            else:
                # 维持原有的玻璃特效：阴影 -> 高光 -> 本体
                glyph.patch.set_path_effects([shadow, highlight, normal])

    # 8. Modern Title and Labels
    ax.set_title(f"{class_name}: Spatial Motif (Top {node_num} Context)",
                fontsize=30, fontfamily='sans-serif', fontweight='bold',
                color='#2D3748', pad=15)

    ax.set_ylim(0, 1.05)

    # 9. Customize X-Axis with modern styling
    real_rel_positions = [idx - CENTER_IDX for idx in all_indices]
    ax.set_xticks(range(len(all_indices)))
    ax.set_xticklabels(real_rel_positions, rotation=90 if len(str(max(real_rel_positions))) > 3 else 0,
                      fontfamily='sans-serif', fontsize=20, color='#4A5568')

    # 10. Highlight Anchor with Morandi-style grey
    anchor_plot_idx = int(np.where(all_indices == center_idx)[0][0])
    logo.highlight_position(p=anchor_plot_idx, color='#9E2A2B', alpha=0.6)

    x_labels = ax.get_xticklabels()
    if len(x_labels) > anchor_plot_idx:
        x_labels[anchor_plot_idx].set_fontweight('bold')
        x_labels[anchor_plot_idx].set_color('#2D3748')

    # Save as PDF
    # Use class_name directly for unique cluster-specific filenames
    safe_name = class_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'motif_logo_{safe_name}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved logo to {output_path}")
    return output_path


# ============================================================================
# Main Execution (REFACTORED)
# ============================================================================

def int_or_none(v):
    """Convert string to int or None if value is 'none'"""
    if v.lower() == 'none':
        return None
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{v}' is not an integer or 'none'")


def main():
    parser = argparse.ArgumentParser(
        description='Spatial Motif Analysis with Latent Space Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze m6A with 3 clusters
  python SpatialMotif.py --classes m6A --n_clusters 3

  # Analyze all classes with 5 clusters, PCA to 30 dimensions
  python SpatialMotif.py --n_clusters 5 --pca_components 30

  # Disable PCA, use raw embeddings
  python SpatialMotif.py --classes m6A --n_clusters 3 --pca_components none
        """
    )
    parser.add_argument('--node_num', type=int, default=10, help="Number of context positions")
    parser.add_argument('--classes', nargs='+', default=None, help="Classes (e.g. m6A)")
    parser.add_argument('--config', type=str, default='json/human.json')
    parser.add_argument('--checkpoint', type=str, default='logs/old/rna_classification_20260129_195404/checkpoints/best_model.pt')
    parser.add_argument('--num_samples', type=int_or_none, default=None, help="Number of samples to process (None for all)")
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--internal_batch_size', type=int, default=128*4)
    parser.add_argument('--batch_size', type=int, default=128*4, help="Batch size for processing samples")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16, help="Workers for structure precomputation")

    # NEW: Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=5,
                       help="Number of clusters for latent space clustering (default: 3)")
    parser.add_argument('--pca_components', type=int_or_none, default=50,
                       help="PCA components for dimensionality reduction (None to skip PCA, default: 30)")
    parser.add_argument('--output_dir', type=str, default='motif_logo_clustered',
                       help="Output directory for cluster-specific logos (default: motif_logo_clustered)")

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Config, _ = load_config(args.config)

    # 1. Load the ALL-DATA Dataset
    print(f"Loading FULL Dataset (Mer100DatasetMotif)...")
    dataset = Mer100DatasetMotif(use_human3=True, preload_cache=True)

    # 2. Precompute/Check Secondary Structures
    print(f"\n{'='*60}")
    print(f"Ensuring Secondary Structures are Precomputed...")
    print(f"{'='*60}")
    dataset.precompute_all_structures(batch_size=100, num_workers=args.num_workers, show_progress=True)

    # 3. Load Model
    print(f"\nLoading model from {args.checkpoint}...")
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
        use_layer_norm=Config.use_layer_norm
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Checkpoint not found, using random weights (DEBUG MODE)")

    if args.classes:
        target_classes = [(k, v) for k, v in MOD_NAMES.items() if v in args.classes]
    else:
        target_classes = list(MOD_NAMES.items())

    # 4. Analyze each class with clustering
    for class_idx, class_name in target_classes:
        print(f"\n{'='*80}")
        print(f"Analyzing {class_name} (Class Index: {class_idx})")
        print(f"{'='*80}")

        # Step 1: Find all positive samples for this class
        all_labels = dataset.y_12class
        positive_indices = np.where(all_labels[:, class_idx] == 1)[0]
        total_found = len(positive_indices)

        if total_found == 0:
            print(f"Warning: No samples found for {class_name} in the dataset!")
            continue

        print(f"Found {total_found} positive samples for {class_name}.")

        # Optional: Limit samples if specified
        np.random.seed(42)
        np.random.shuffle(positive_indices)
        if args.num_samples is not None and args.num_samples < total_found:
            positive_indices = positive_indices[:args.num_samples]
            print(f"Using {len(positive_indices)} samples (randomly selected from {total_found})...")

        # Step 2: Extract embeddings for clustering
        print(f"\nStep 1: Extracting latent embeddings...")
        try:
            embeddings = extract_embeddings(model, dataset, positive_indices, device)
        except ValueError as e:
            print(f"Error during embedding extraction: {e}")
            continue

        # Step 3: Cluster samples in latent space
        print(f"\nStep 2: Clustering samples...")
        try:
            clusters_dict = cluster_samples(
                embeddings,
                positive_indices,
                n_clusters=args.n_clusters,
                pca_components=args.pca_components,
                random_state=42
            )
        except Exception as e:
            print(f"Error during clustering: {e}")
            continue

        # Step 4: Compute attribution and plot logo for each cluster
        print(f"\nStep 3: Computing cluster-specific attributions and generating logos...")
        for cluster_id, cluster_indices in clusters_dict.items():
            print(f"\n{'─'*60}")
            print(f"Cluster {cluster_id}: {len(cluster_indices)} samples")
            print(f"{'─'*60}")

            # Compute attribution for this cluster only
            imp_matrix = calculate_spatial_attribution(
                model, dataset, class_idx, cluster_indices, device,
                n_steps=args.n_steps,
                internal_batch_size=args.internal_batch_size,
                batch_size=args.batch_size
            )

            # Generate cluster-specific logo
            cluster_class_name = f"{class_name}_Cluster_{cluster_id}"
            plot_top_k_logo(
                imp_matrix,
                class_idx,
                cluster_class_name,
                node_num=args.node_num,
                output_dir=args.output_dir
            )

        # Clean up GPU memory after each class
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Logos saved to: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
