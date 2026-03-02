#!/usr/bin/env python3
"""
SpatialMotif.py - Visualize "Spatial Motifs" with Top-K Sequence Logos
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

# [CHANGE] Import the new Motif Dataset class
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
# RGB values normalized to 0-1 range for matplotlib
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
# Core Attribution Function
# ============================================================================

def calculate_spatial_attribution(
    model: RNA_ClassQuery_Model,
    dataset: Mer100DatasetMotif,
    target_class_idx: int,
    device: torch.device,
    num_samples: Optional[int] = None, 
    n_steps: int = 20,
    internal_batch_size: int = 4,
    batch_size: int = 128
) -> np.ndarray:
    
    original_label_id = REVERSE_LABEL_MAPPING.get(target_class_idx)
    aggregated_importance = np.zeros((REL_RANGE, 4), dtype=np.float32)

    print(f"\nScanning ENTIRE dataset for {MOD_NAMES[target_class_idx]} samples (Target: All if None)...")
    
    # Check all labels
    all_labels = dataset.y_12class
    potential_indices = np.where(all_labels[:, target_class_idx] == 1)[0]
    
    total_found = len(potential_indices)
    if total_found == 0:
        print(f"Warning: No samples found for {MOD_NAMES[target_class_idx]} in the entire dataset!")
        return aggregated_importance
    
    print(f"Found {total_found} samples in total.")

    # Shuffle indices to ensure representative sampling
    np.random.seed(42)
    np.random.shuffle(potential_indices)

    if num_samples is not None and num_samples < total_found:
        valid_indices = potential_indices[:num_samples]
        print(f"Processing {len(valid_indices)} samples (randomly selected from {total_found})...")
    else:
        valid_indices = potential_indices
        print(f"Processing all {len(valid_indices)} samples...")
    
    print(f"Processing {len(valid_indices)} samples with n_steps={n_steps}, batch_size={batch_size}...")

    processed_count = 0
    class_name = MOD_NAMES[target_class_idx]
    
    # Batch processing loop
    for i in range(0, len(valid_indices), batch_size):
        batch_idxs = valid_indices[i:i+batch_size]
        
        for idx in tqdm(batch_idxs, desc=f"Processing {class_name} samples"):
            try:
                data = dataset[idx]
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                y_site = data.y_site
                batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                anchor_indices = torch.where(y_site == original_label_id)[0].tolist()
                if len(anchor_indices) == 0: continue

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

            except Exception:
                continue
        
        # Explicitly clear GPU cache after each batch to prevent OOM
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if processed_count > 0:
        aggregated_importance /= processed_count

    return aggregated_importance


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_top_k_logo(
    aggregated_importance: np.ndarray, 
    class_idx: int,
    class_name: str,
    node_num: int = 10,
    output_dir: str = "motif_logo"
) -> str:
    if not HAS_LOGOMAKER: return ""
    os.makedirs(output_dir, exist_ok=True)

    # Strategy: Take Absolute value (Saliency)
    saliency_matrix = np.abs(aggregated_importance)

    # 1. Calculate Importance Score for sorting (Sum of RAW saliency scores)
    importance_scores = np.sum(saliency_matrix, axis=1)

    # 2. Force include Center
    center_idx = CENTER_IDX
    temp_scores = importance_scores.copy()
    temp_scores[center_idx] = -1.0 # Exclude from top-k selection
    
    # 3. Select Top K context positions
    top_indices = np.argsort(temp_scores)[-node_num:]
    
    # 4. Combine and Sort Indices
    all_indices = np.append(top_indices, center_idx)
    all_indices = np.sort(all_indices) 
    
    # 5. Extract Sub-Matrix
    logo_matrix_raw = saliency_matrix[all_indices]
    
    # --- [MODIFICATION] L1 Normalization (Sum to 1) ---
    # Divide each row by its sum, so that A+C+G+U = 1.0
    row_sums = np.sum(logo_matrix_raw, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums < 1e-9] = 1.0 
    
    logo_matrix_norm = logo_matrix_raw / row_sums

    # --- Position 0 Handling ---
    # Re-apply anchor logic to the normalized matrix
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
    # Highlight: small upward-left offset bright edge for glass refraction
    highlight = path_effects.Stroke(linewidth=0.8,
                                   foreground=(1, 1, 1, 0.6),
                                   alpha=0.7)
    
    # Drop Shadow: slight downward offset with blur for floating effect
    shadow = path_effects.SimplePatchShadow(offset=(1.5, -1.5),
                                            alpha=0.4,
                                            rho=0.5)
    
    # Apply effects to all glyph polygons
    glass_effect = [shadow, highlight]
    
    # Apply path effects to all glyphs in the logo
    for glyph in logo.glyph_list:
        # Access the glyph's patch (polygon) artist
        if hasattr(glyph, 'patch') and glyph.patch is not None:
            glyph.patch.set_path_effects(glass_effect)
    
    # 8. Modern Title and Labels
    ax.set_title(f"{class_name}: Spatial Motif (Top {node_num} Context)",
                fontsize=30, fontfamily='sans-serif', fontweight='bold',
                color='#2D3748', pad=15)
    
    # Set y-axis limit to exactly 1.0 or slightly higher for clarity
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
        x_labels[anchor_plot_idx].set_color('#2D3748')  # Dark Grey

    # Save as PDF
    output_path = os.path.join(output_dir, f'motif_logo_{class_name}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved logo to {output_path}")
    return output_path


# ============================================================================
# Main Execution
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
    parser = argparse.ArgumentParser(description='Spatial Motif Analysis (Logo Only)')
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
    
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Config, _ = load_config(args.config)
    
    # 1. Load the ALL-DATA Dataset
    print(f"Loading FULL Dataset (Mer100DatasetMotif)...")
    dataset = Mer100DatasetMotif(use_human3=True, preload_cache=True)
    
    # 2. [REQ] Precompute/Check Secondary Structures at the beginning
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

    # 4. Analyze
    for class_idx, class_name in target_classes:
        print(f"\nAnalyzing {class_name}...")
        
        imp_matrix = calculate_spatial_attribution(
            model, dataset, class_idx, device, 
            num_samples=args.num_samples,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            batch_size=args.batch_size
        )
        
        plot_top_k_logo(
            imp_matrix, 
            class_idx,
            class_name, 
            node_num=args.node_num, 
            output_dir="motif_logo"
        )

if __name__ == "__main__":
    main()