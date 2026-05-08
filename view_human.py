"""
Human Dataset Modification Sites Violin Plot Analysis

This script analyzes and visualizes the distribution of modification site counts
for each of the 12 modification types in the human dataset.

Referenced files:
- view3.py: for Morandi color scheme, violin plot style, and helper functions
- dataset/human.py: for MOD_NAMES, LABEL_MAPPING, and data structure conventions
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set DejaVu Sans font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Morandi color palette - muted soft colors (reused from view3.py)
MORANDI_COLORS = {
    'blue': '#8E9AAF',      # Dusty blue
    'pink': '#E8C1C1',      # Dusty pink
    'green': '#A8B5A0',     # Dusty green
    'beige': '#D8CFC4',     # Beige
    'lavender': '#B8A8C4',  # Dusty lavender
    'coral': '#D4A5A5',     # Coral pink
    'sage': '#B5C6B0',      # Sage green
    'rose': '#C8A8A8',      # Rose gray
    'slate': '#9AA4B0',     # Slate gray
    'cream': '#E8DBC4',     # Cream
}

# Morandi color palette list (reused from view3.py)
MORANDI_PALETTE = [
    '#8E9AAF', '#E8C1C1', '#A8B5A0', '#D8CFC4',
    '#B8A8C4', '#D4A5A5', '#B5C6B0', '#C8A8A8',
    '#9AA4B0', '#E8DBC4', '#C4B6A8', '#A8A8C4'
]


def compute_mode(arr):
    """Compute the mode of an array (reused from view3.py)"""
    unique, counts = np.unique(arr, return_counts=True)
    max_idx = np.argmax(counts)
    return unique[max_idx]


# 12 modification types mapping (from dataset/human.py)
# LABEL_MAPPING: mod_index (1-12) -> model index (0-11)
LABEL_MAPPING = {
    1: 0,   # Am
    2: 1,   # Atol
    3: 2,   # Cm
    4: 3,   # Gm
    5: 4,   # Tm
    6: 5,   # Y
    7: 6,   # ac4C
    8: 7,   # m1A
    9: 8,   # m5C
    10: 9,  # m6A
    11: 10, # m6Am
    12: 11  # m7G
}

# 12 modification names (from dataset/human.py)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}


def plot_violin(data_dict, title, xlabel, ylabel, save_path, figsize=(12, 6)):
    """
    Create violin plots using seaborn with Morandi color palette
    (reused from view3.py)

    Args:
        data_dict: Dictionary with category names as keys and data arrays as values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Prepare data
    categories = []
    values = []
    for cat, vals in data_dict.items():
        categories.extend([cat] * len(vals))
        values.extend(vals)

    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'Category': categories,
        'Value': values
    })

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw violin plot
    palette = MORANDI_PALETTE[:len(data_dict)]
    sns.violinplot(
        data=df,
        x='Category',
        y='Value',
        palette=palette,
        inner='box',
        density_norm='width',
        ax=ax,
        linewidth=1.2,
        saturation=0.85
    )

    # Set title and labels
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')

    # Optimize x-axis labels
    plt.xticks(rotation=0, ha='center')

    # Set background color
    ax.set_facecolor('#FAF9F6')
    fig.patch.set_facecolor('#FAF9F6')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8E9AAF')
    ax.spines['bottom'].set_color('#8E9AAF')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#B8A8C4')
    ax.xaxis.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAF9F6')

    # Also save as PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='#FAF9F6')
    plt.close()
    print(f"Violin plot saved to: {save_path}")
    print(f"Violin plot saved to: {pdf_path}")


def analyze_human_modification_sites():
    """
    Analyze modification site counts for each modification type in human dataset.
    
    For each sample, count how many sites belong to each of the 12 modification types.
    Then visualize the distribution using violin plots.
    """
    print("=" * 70)
    print("=== Human Dataset Modification Sites Analysis ===")
    print("=" * 70)

    # Data paths (following dataset/human.py conventions)
    human3_dir = os.path.join('npy', 'human3')
    
    # Check if human3 directory exists, try alternative paths
    if not os.path.exists(human3_dir):
        # Try relative path from parent directory
        alt_paths = [
            os.path.join('..', 'npy', 'human3'),
            '/home/dc/vscode/vscode20260406/rgcnformer_sum/npy/human3',
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                human3_dir = alt_path
                break
        else:
            raise FileNotFoundError(f"Cannot find human3 directory. Tried: {human3_dir} and alternatives.")

    # Load 1001loc.npy (contains 1-12 labels for each site)
    loc_path = os.path.join(human3_dir, '1001loc.npy')
    print(f"\nLoading data from: {loc_path}")
    
    loc_data = np.load(loc_path, mmap_mode='r')
    num_samples = loc_data.shape[0]
    seq_length = loc_data.shape[1]
    
    print(f"Dataset loaded:")
    print(f"  Total samples: {num_samples}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Data type: {loc_data.dtype}")

    # Copy data to memory for analysis
    print("\nCopying data to memory for analysis...")
    full_labels = np.array(loc_data)

    # Initialize count arrays for each modification type
    # Each array has length = num_samples (count per sample)
    mod_counts = {mod_id: np.zeros(num_samples, dtype=np.int32) for mod_id in range(1, 13)}

    # Count occurrences of each modification type per sample
    print("Counting modification sites per sample...")
    for i in range(num_samples):
        sample_labels = full_labels[i]
        for mod_id in range(1, 13):
            mod_counts[mod_id][i] = np.sum(sample_labels == mod_id)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")

    # Convert to model index (0-11) for use with MOD_NAMES
    mod_counts_by_name = {}
    for mod_id, model_idx in LABEL_MAPPING.items():
        mod_name = MOD_NAMES[model_idx]
        mod_counts_by_name[mod_name] = mod_counts[mod_id]

    # Print statistics for each modification type
    print("\n" + "=" * 70)
    print("Modification Site Statistics per Sequence")
    print("=" * 70)
    print(f"{'Mod':<8} {'Samples':>8} {'Mean':>8} {'Median':>8} {'Mode':>6} {'Min':>6} {'Max':>6} {'Std':>8}")
    print("-" * 70)

    stats_dict = {}
    for model_idx in range(12):
        mod_name = MOD_NAMES[model_idx]
        counts = mod_counts_by_name[mod_name]
        
        # Calculate statistics
        mean_val = np.mean(counts)
        median_val = np.median(counts)
        mode_val = compute_mode(counts)
        min_val = np.min(counts)
        max_val = np.max(counts)
        std_val = np.std(counts)
        
        stats_dict[mod_name] = {
            'samples': len(counts),
            'mean': mean_val,
            'median': median_val,
            'mode': mode_val,
            'min': min_val,
            'max': max_val,
            'std': std_val
        }
        
        print(f"{mod_name:<8} {len(counts):>8} {mean_val:>8.2f} {median_val:>8.2f} {mode_val:>6.0f} {min_val:>6.0f} {max_val:>6.0f} {std_val:>8.2f}")

    print("=" * 70)

    # Print detailed statistics for each modification type
    print("\n" + "=" * 70)
    print("Detailed Statistics by Modification Type")
    print("=" * 70)

    for model_idx in range(12):
        mod_name = MOD_NAMES[model_idx]
        counts = mod_counts_by_name[mod_name]
        
        print(f"\n{mod_name}:")
        print(f"  Samples with sites (>0): {np.sum(counts > 0)} / {len(counts)} ({100*np.sum(counts > 0)/len(counts):.2f}%)")
        print(f"  Mean: {np.mean(counts):.2f}")
        print(f"  Median: {np.median(counts):.2f}")
        print(f"  Mode: {compute_mode(counts)}")
        print(f"  Min: {np.min(counts)}, Max: {np.max(counts)}")
        print(f"  Std Dev: {np.std(counts):.2f}")

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Draw violin plot
    print("\n" + "=" * 70)
    print("Generating violin plot...")
    print("=" * 70)

    plot_human_mod_sites_violin(
        mod_counts_by_name,
        save_path='output/violin_human_mod_sites.png'
    )

    print("\nAnalysis completed!")


def plot_human_mod_sites_violin(mod_counts_by_name, save_path='output/violin_human_mod_sites.png'):
    """
    Draw violin plot for modification site counts across all 12 modification types.
    
    Args:
        mod_counts_by_name: Dictionary mapping modification names to count arrays
        save_path: Path to save the figure
    """
    plot_violin(
        data_dict=mod_counts_by_name,
        title='Human Dataset: Modification Site Counts per Sequence',
        xlabel='Modification Type',
        ylabel='Number of Sites in Sequence',
        save_path=save_path,
        figsize=(14, 7)
    )


if __name__ == "__main__":
    analyze_human_modification_sites()
