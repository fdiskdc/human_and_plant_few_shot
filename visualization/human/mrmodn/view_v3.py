"""
view3.py - Morandi配色方案与12类修饰可视化工具 / Morandi Color Palette & 12-class Modification Visualization Utilities

为 12 类 RNA 修饰定义 Morandi 莫兰迪配色方案 (12 种柔和色) 和可视化工具函数。
被 view_human.py、view_human_total.py 等多个可视化脚本复用。
Defines Morandi color palette (12 muted soft colors) and visualization utility functions for 12-class RNA modifications.
Reused by view_human.py, view_human_total.py, and other visualization scripts.

功能模块 / Modules:
- MORANDI_COLORS: 12 种 Morandi 颜色字典 / 12 Morandi colors dict
- MORANDI_PALETTE: 12 种 Morandi 颜色列表 / 12 Morandi colors list
- 字体配置 (DejaVu Sans) / Font config
- 通用绘图工具函数 / Common plotting utilities

输入 / Inputs:
- 12 类修饰数据 (从其他脚本传入) / 12-class modification data (passed from other scripts)

输出 / Outputs:
- 颜色方案供其他脚本导入 / Color scheme imported by other scripts
- 图表样式配置 / Figure style config

数据流 / Data Flow:
1. 定义 Morandi 配色 / Define Morandi colors
2. 设置 matplotlib 字体 / Set matplotlib fonts
3. 导出常量供复用 / Export constants for reuse

相关文件 / Related Files:
- 调用 / Calls: numpy, matplotlib, seaborn
- 被调用 / Called by: view_human.py, view_human_total.py, etc.

使用示例 / Usage Example:
    from view3 import MORANDI_COLORS, MORANDI_PALETTE

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set DejaVu Sans font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Morandi color palette - muted soft colors
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

# Morandi color palette list
MORANDI_PALETTE = [
    '#8E9AAF', '#E8C1C1', '#A8B5A0', '#D8CFC4',
    '#B8A8C4', '#D4A5A5', '#B5C6B0', '#C8A8A8',
    '#9AA4B0', '#E8DBC4', '#C4B6A8', '#A8A8C4'
]


def compute_mode(arr):
    """Compute the mode of an array"""
    unique, counts = np.unique(arr, return_counts=True)
    max_idx = np.argmax(counts)
    return unique[max_idx]


# 12 modification types mapping
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}


def plot_violin(data_dict, title, xlabel, ylabel, save_path, figsize=(10, 6)):
    """
    Create violin plots using seaborn with Morandi color palette

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


def plot_mod_counts_violin(mod_counts, save_path='output/violin_mod_counts.png'):
    """
    Draw violin plot for modification count distribution
    """
    data_dict = {'Modification Count': mod_counts}
    plot_violin(
        data_dict=data_dict,
        title='RNA Modification Count Distribution',
        xlabel='',
        ylabel='Number of Modifications',
        save_path=save_path,
        figsize=(6, 8)
    )


def plot_multi_violin_comparison(pos_counts, neg_counts, save_path='output/violin_comparison.png'):
    """
    Draw violin plot comparing positive and negative samples
    """
    data_dict = {
        'Modified': pos_counts,
        'Unmodified': neg_counts
    }
    plot_violin(
        data_dict=data_dict,
        title='Comparison of Modification Counts',
        xlabel='Sequence Type',
        ylabel='Number of Modifications',
        save_path=save_path,
        figsize=(8, 6)
    )


def analyze_modification_counts():
    """
    Analyze modification count statistics for Gen3ZeroDataset
    """
    print("=" * 60)
    print("=== Gen3ZeroDataset Modification Count Analysis ===")
    print("=" * 60)

    # Data paths
    gen3_dir = os.path.join('npy', '3gen_alignment')
    zero_dir = os.path.join('npy', 'zero')

    # Load 3gen data (positive samples)
    print("\nLoading 3gen data...")
    g3_12 = np.load(os.path.join(gen3_dir, '12loc.npy'), mmap_mode='r')
    g3_12_loaded = np.array(g3_12)
    pos_mask = np.any(g3_12_loaded != 0, axis=1)

    g3_1001 = np.load(os.path.join(gen3_dir, '1001loc.npy'), mmap_mode='r')
    pos_1001 = np.array(g3_1001)[pos_mask]

    # Load zero data (negative samples)
    print("Loading zero data...")
    z_1001 = np.load(os.path.join(zero_dir, 'zero_label1001.npy'), mmap_mode='r')
    num_pos = len(pos_1001)
    num_neg = min(num_pos, len(z_1001))
    neg_1001 = np.array(z_1001[:num_neg])

    # Merge data (simulate Gen3ZeroDataset behavior)
    full_labels = np.concatenate([pos_1001, neg_1001], axis=0)

    print(f"Dataset loaded: {len(full_labels)} samples (positive: {num_pos}, negative: {num_neg})")

    # Only calculate for sequences with modifications (positive samples)
    pos_full_labels = pos_1001
    print(f"Sequences with modifications: {len(pos_full_labels)}")

    # Calculate modification count per sequence (using 1001loc, count non-zero elements per row)
    mod_counts = np.count_nonzero(pos_full_labels, axis=1)
    neg_counts = np.count_nonzero(neg_1001, axis=1)

    print("\n" + "=" * 60)
    print("Modification Count Statistics")
    print("=" * 60)

    # Calculate statistics
    mean_count = np.mean(mod_counts)
    median_count = np.median(mod_counts)
    mode_count = compute_mode(mod_counts)

    # Additional statistics
    min_count = np.min(mod_counts)
    max_count = np.max(mod_counts)
    std_count = np.std(mod_counts)

    print(f"Mean:     {mean_count:.2f}")
    print(f"Median:   {median_count:.2f}")
    print(f"Mode:     {mode_count:.0f}")
    print(f"Min:      {min_count:.0f}")
    print(f"Max:      {max_count:.0f}")
    print(f"Std Dev:  {std_count:.2f}")

    # Print modification count distribution
    print("\n" + "=" * 60)
    print("Modification Count Distribution")
    print("=" * 60)

    unique_counts, counts = np.unique(mod_counts, return_counts=True)
    for count, freq in zip(unique_counts, counts):
        percentage = freq / len(mod_counts) * 100
        print(f"  {count} modifications: {freq:6d} sequences ({percentage:5.2f}%)")

    # Count occurrence of each modification type (based on 1001loc)
    print("\n" + "=" * 60)
    print("Modification Type Occurrence in Modified Sequences")
    print("=" * 60)

    # LABEL_MAPPING: mod_index (1-12) -> model index (0-11)
    LABEL_MAPPING = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
        7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11
    }

    for mod_id, class_idx in LABEL_MAPPING.items():
        mod_name = MOD_NAMES.get(class_idx, f"Class_{class_idx}")
        count = np.sum(pos_full_labels == mod_id)
        percentage = count / np.sum(pos_full_labels > 0) * 100
        print(f"  {class_idx:2d} ({mod_name:6s}): {count:6d} sites ({percentage:5.2f}%)")

    print("\n" + "=" * 60)

    # Draw violin plots
    print("\nDrawing violin plots...")
    os.makedirs('output', exist_ok=True)

    # Single violin plot - modification count distribution
    plot_mod_counts_violin(mod_counts, save_path='output/violin_mod_counts.png')

    # Comparison violin plot - positive vs negative samples
    plot_multi_violin_comparison(mod_counts, neg_counts, save_path='output/violin_comparison.png')

    print("\nViolin plots completed!")


if __name__ == "__main__":
    analyze_modification_counts()
