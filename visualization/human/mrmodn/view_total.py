"""
view_human_total.py - 人类数据集总修饰位点小提琴图 / Human Dataset Total Modification Sites Violin Plot

可视化人类数据集中所有 12 类修饰的位点总数 (求和) 分布。复用 view3.py 的 Morandi 配色。
Visualizes the total distribution of modification site counts (sum of all 12 modification types) in the human dataset.
Reuses Morandi color scheme from view3.py.

功能模块 / Modules:
- 总位点数统计 / Total site count statistics
- 小提琴图绘制 / Violin plot rendering
- Morandi 配色方案 / Morandi color scheme
- main: 主入口 / Main entry point

输入 / Inputs:
- human3/1001loc.npy: 1001 位点级标签 / 1001 site-level labels
- MOD_NAMES, LABEL_MAPPING from dataset.human

输出 / Outputs:
- att_fig/total_violin.png: 总位点小提琴图 / Total site violin plot
- 终端统计输出 / Terminal statistics output

数据流 / Data Flow:
1. 加载位点标签 / Load site labels
2. 计算每序列总位点数 / Compute total sites per sequence
3. 绘制小提琴图 / Draw violin plot
4. 保存为 PNG / Save as PNG

相关文件 / Related Files:
- 调用 / Calls: numpy, seaborn, matplotlib, scipy.stats
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python view_human_total.py

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
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


def plot_single_violin(data_array, title, xlabel, ylabel, save_path, figsize=(8, 8)):
    """
    Create a single violin plot using seaborn with Morandi color palette
    (reused from view3.py style, adapted for single violin)

    Args:
        data_array: 1D array of values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        figsize: Figure size
    """
    import pandas as pd
    
    # Prepare data - single category
    categories = ['Total Modifications'] * len(data_array)
    values = list(data_array)

    # Create DataFrame
    df = pd.DataFrame({
        'Category': categories,
        'Value': values
    })

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw violin plot with single color
    sns.violinplot(
        data=df,
        x='Category',
        y='Value',
        palette=[MORANDI_PALETTE[0]],
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


def analyze_human_total_modification_sites():
    """
    Analyze total modification site counts (sum of all 12 types) for human dataset.
    
    For each sample, count the total number of modification sites across all types.
    Then visualize the distribution using a single violin plot.
    """
    print("=" * 70)
    print("=== Human Dataset Total Modification Sites Analysis ===")
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

    # Calculate total modification count per sample
    # Each non-zero value in 1001loc represents a modification site
    print("Calculating total modification counts per sample...")
    total_counts = np.count_nonzero(full_labels, axis=1)

    # Print statistics
    print("\n" + "=" * 70)
    print("Total Modification Site Statistics per Sequence")
    print("=" * 70)
    print(f"Total samples: {len(total_counts)}")
    print(f"  Mean:     {np.mean(total_counts):.2f}")
    print(f"  Median:   {np.median(total_counts):.2f}")
    print(f"  Mode:     {compute_mode(total_counts):.0f}")
    print(f"  Min:      {np.min(total_counts):.0f}")
    print(f"  Max:      {np.max(total_counts):.0f}")
    print(f"  Std Dev:  {np.std(total_counts):.2f}")

    # Print distribution
    print("\n" + "=" * 70)
    print("Total Modification Count Distribution")
    print("=" * 70)
    
    unique_counts, counts = np.unique(total_counts, return_counts=True)
    for count, freq in zip(unique_counts, counts):
        percentage = freq / len(total_counts) * 100
        print(f"  {count:3d} modifications: {freq:7d} sequences ({percentage:6.2f}%)")
    
    print("=" * 70)

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Draw violin plot
    print("\n" + "=" * 70)
    print("Generating total modification violin plot...")
    print("=" * 70)

    plot_total_mod_sites_violin(
        total_counts,
        save_path='output/violin_human_total_mod_sites.png'
    )

    # Draw bar chart
    print("\n" + "=" * 70)
    print("Generating total modification count bar chart...")
    print("=" * 70)

    plot_total_mod_count_bar(
        total_counts,
        save_path='output/bar_human_total_mod_sites.png'
    )

    # Draw nested donut chart
    print("\n" + "=" * 70)
    print("Generating nested donut chart for total modification distribution...")
    print("=" * 70)

    plot_total_mod_count_nested_donut(
        total_counts,
        save_path='output/donut_human_total_mod_sites.png'
    )

    print("\nAnalysis completed!")


def plot_total_mod_sites_violin(total_counts, save_path='output/violin_human_total_mod_sites.png'):
    """
    Draw violin plot for total modification site counts across all samples.
    
    Args:
        total_counts: 1D array of total modification counts per sample
        save_path: Path to save the figure
    """
    plot_single_violin(
        data_array=total_counts,
        title='Human Dataset: Total Modification Sites per Sequence',
        xlabel='',
        ylabel='Number of Modification Sites',
        save_path=save_path,
        figsize=(8, 8)
    )


def plot_total_mod_count_bar(total_counts, save_path='output/bar_human_total_mod_sites.png'):
    """
    Create a bar chart showing the sample proportion distribution of total
    modification site counts with a probability density curve overlay.
    
    Args:
        total_counts: 1D array of total modification counts per sample
        save_path: Path to save the figure
    """
    # Calculate discrete distribution
    unique_counts, frequencies = np.unique(total_counts, return_counts=True)
    proportions = frequencies / np.sum(frequencies)

    # Set up figure
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_color = MORANDI_PALETTE[0]
    curve_color = MORANDI_PALETTE[4]

    # Draw sample proportion bars
    ax.bar(
        unique_counts,
        proportions,
        width=0.8,
        color=bar_color,
        edgecolor='#5A6370',
        linewidth=1.0,
        alpha=0.6,
        label='Sample Proportion'
    )

    # Draw KDE on a secondary axis so density stays statistically meaningful
    ax2 = ax.twinx()
    kde = stats.gaussian_kde(total_counts, bw_method='scott')
    x_smooth = np.linspace(min(unique_counts) - 1, max(unique_counts) + 1, 500)
    density = kde(x_smooth)

    ax2.plot(
        x_smooth,
        density,
        color=curve_color,
        linewidth=2.5,
        label='Density Curve',
        zorder=5
    )

    # Set title and labels with 18pt font
    ax.set_title(
        'Human Dataset: Distribution of Total Modification Sites per Sequence',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Number of Modification Sites', fontsize=18, fontweight='medium')
    ax.set_ylabel('Sample Proportion', fontsize=18, fontweight='medium')
    ax2.set_ylabel('Probability Density', fontsize=18, fontweight='medium')

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=18)

    # Set background color
    ax.set_facecolor('#FAF9F6')
    fig.patch.set_facecolor('#FAF9F6')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#8E9AAF')
    ax.spines['bottom'].set_color('#8E9AAF')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_color('#B8A8C4')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#B8A8C4')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#B8A8C4')

    # Set x-axis ticks to show all integer values
    ax.set_xticks(unique_counts)

    ax.set_ylim(0, proportions.max() * 1.15)
    ax2.set_ylim(0, density.max() * 1.15)

    # Combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper right',
        fontsize=18,
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAF9F6')
    
    # Also save as PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='#FAF9F6')
    plt.close()
    print(f"Bar chart with density curve saved to: {save_path}")
    print(f"Bar chart with density curve saved to: {pdf_path}")


def plot_total_mod_count_nested_donut(total_counts, save_path='output/donut_human_total_mod_sites.png'):
    """
    Create a nested donut chart emphasizing the share of sequences with exactly
    one modification site versus more than one, and break down the >1 group.

    Args:
        total_counts: 1D array of total modification counts per sample
        save_path: Path to save the figure
    """
    total_samples = len(total_counts)
    one_count = int(np.sum(total_counts == 1))
    more_than_one_mask = total_counts > 1
    more_than_one_count = int(np.sum(more_than_one_mask))

    outer_sizes = [one_count, more_than_one_count]
    # Refined palette for the nested donut:
    # inner ring = coarse grouping, outer ring = detailed >1-site breakdown.
    inner_ring_palette = ['#8FA7B3', '#C89B8A']
    outer_detail_palette = [
        '#A8BBC4', '#B8A8C4', '#C3B0B7', '#CDB8AB',
        '#D6C1A2', '#B7C8B5', '#9FB5C1', '#D3AE9A'
    ]
    empty_outer_color = '#EEE7DD'

    # Restore full label for outer ring (">1 sites" stays on the ring)
    outer_labels = [
        f'1 site\n{one_count / total_samples * 100:.2f}%',
        f'>1 sites\n{more_than_one_count / total_samples * 100:.2f}%'
    ]
    outer_colors = inner_ring_palette

    gt1_counts = total_counts[more_than_one_mask]
    detailed_labels = []
    detailed_sizes = []

    if len(gt1_counts) > 0:
        unique_gt1, freq_gt1 = np.unique(gt1_counts, return_counts=True)
        overflow_count = 0
        for count_value, freq in zip(unique_gt1, freq_gt1):
            if count_value <= 8:
                detailed_labels.append(str(int(count_value)))
                detailed_sizes.append(int(freq))
            else:
                overflow_count += int(freq)
        if overflow_count > 0:
            detailed_labels.append('>8')
            detailed_sizes.append(overflow_count)

    # Put the coarse grouping on the inner ring, and move the >1-site detailed wedges
    # to the outer ring so the subcategories live on the perimeter.
    inner_sizes = outer_sizes
    inner_labels = outer_labels
    inner_colors = outer_colors

    outer_detailed_sizes = [one_count] + detailed_sizes
    outer_detailed_labels = [''] + [''] * len(detailed_labels)
    outer_detailed_colors = [empty_outer_color] + outer_detail_palette[:len(detailed_sizes)]
    if len(detailed_sizes) > len(outer_detail_palette):
        repeats = len(detailed_sizes) - len(outer_detail_palette)
        outer_detailed_colors.extend((outer_detail_palette * ((repeats // len(outer_detail_palette)) + 1))[:repeats])

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14, 11))  # Larger figure for external annotations
    fig.patch.set_facecolor('#FAF9F6')
    ax.set_facecolor('#FAF9F6')

    # Mirror the chart by changing counterclock to True while keeping startangle=90
    # This reverses the order: "1 site" now starts from top-left and goes clockwise,
    # while ">1 sites" starts from top-right, resulting in left-right mirror
    outer_wedges, _ = ax.pie(
        outer_detailed_sizes,
        radius=1.0,
        labels=outer_detailed_labels,
        colors=outer_detailed_colors,
        startangle=90,
        counterclock=True,  # Changed from False to True for left-right mirror
        wedgeprops=dict(width=0.28, edgecolor='#FAF9F6', linewidth=2),
        textprops=dict(fontsize=18, fontweight='medium', color='#4F5866')
    )

    # Inner ring also uses counterclock=True for consistency
    # Set labeldistance to put labels inside the ring (we'll add external annotations instead)
    inner_wedges, _ = ax.pie(
        inner_sizes,
        radius=0.72,
        labels=inner_labels,
        colors=inner_colors,
        startangle=90,
        counterclock=True,  # Changed from False to True for consistency
        wedgeprops=dict(width=0.28, edgecolor='#FAF9F6', linewidth=2),
        labeldistance=0.60,  # Keep small to avoid collision
        textprops=dict(fontsize=18, fontweight='medium', color='#4F5866')
    )

    # Add external annotations for the outer-ring detailed wedges (2, 3, ..., 8, >8).
    if detailed_labels:
        outer_radius = 1.0
        anchor_radius = outer_radius + 0.02
        elbow_radius = 1.12
        label_x_extent = 1.36
        min_label_gap = 0.11
        connector_color = '#8B908F'
        label_fill = '#FBF7F1'

        annotation_specs = []
        for i, (label, wedge) in enumerate(zip(detailed_labels, outer_wedges[1:])):
            mid_angle_rad = np.radians((wedge.theta1 + wedge.theta2) / 2.0)
            side = 1 if np.cos(mid_angle_rad) >= 0 else -1

            anchor_x = anchor_radius * np.cos(mid_angle_rad)
            anchor_y = anchor_radius * np.sin(mid_angle_rad)
            elbow_x = elbow_radius * np.cos(mid_angle_rad)
            elbow_y = elbow_radius * np.sin(mid_angle_rad)

            annotation_specs.append({
                'label': label,
                'pct': detailed_sizes[i] / total_samples * 100,
                'side': side,
                'anchor_x': anchor_x,
                'anchor_y': anchor_y,
                'elbow_x': elbow_x,
                'elbow_y': elbow_y,
                'text_x': side * label_x_extent,
                'text_y': elbow_y,
            })

        for side in (-1, 1):
            side_specs = [spec for spec in annotation_specs if spec['side'] == side]
            side_specs.sort(key=lambda spec: spec['text_y'])
            for idx in range(1, len(side_specs)):
                side_specs[idx]['text_y'] = max(
                    side_specs[idx]['text_y'],
                    side_specs[idx - 1]['text_y'] + min_label_gap
                )
            for idx in range(len(side_specs) - 2, -1, -1):
                side_specs[idx]['text_y'] = min(
                    side_specs[idx]['text_y'],
                    side_specs[idx + 1]['text_y'] - min_label_gap
                )
            for spec in side_specs:
                spec['text_y'] = np.clip(spec['text_y'], -1.18, 1.18)

        for spec in annotation_specs:
            line_end_x = spec['text_x'] - 0.05 * spec['side']
            line_end_y = spec['text_y']

            ax.plot(
                [spec['anchor_x'], spec['elbow_x']],
                [spec['anchor_y'], spec['elbow_y']],
                color=connector_color,
                linewidth=1.8,
                solid_capstyle='round',
                alpha=0.95,
                zorder=5
            )
            ax.plot(
                [spec['elbow_x'], line_end_x],
                [spec['elbow_y'], line_end_y],
                color=connector_color,
                linewidth=1.8,
                solid_capstyle='round',
                alpha=0.95,
                zorder=5
            )
            ax.scatter(
                [spec['anchor_x']],
                [spec['anchor_y']],
                s=20,
                color=connector_color,
                edgecolors='#FAF9F6',
                linewidths=0.8,
                zorder=6
            )
            ax.text(
                spec['text_x'],
                spec['text_y'],
                f"{spec['label']}\n{spec['pct']:.2f}%",
                fontsize=15,
                fontweight='semibold',
                color='#4F5866',
                ha='left' if spec['side'] > 0 else 'right',
                va='center',
                linespacing=1.1,
                bbox=dict(
                    boxstyle='round,pad=0.24,rounding_size=0.12',
                    facecolor=label_fill,
                    edgecolor='none',
                    alpha=0.96
                ),
                zorder=7
            )

    # Emphasize the center message instead of labeling the blank inner wedge.
    ax.text(
        0, 0.03,
        'Total\nModification\nDistribution',
        ha='center',
        va='center',
        fontsize=18,
        fontweight='bold',
        color='#4F5866'
    )

    # Add an explanatory note for the outer ring.
    if more_than_one_count > 0:
        ax.text(
            0, -1.32,
            'Outer ring: detailed breakdown of sequences with more than 1 site (2, 3, ..., 8, >8)',
            ha='center',
            va='center',
            fontsize=18,
            color='#6B7280'
        )

    ax.set_title(
        'Human Dataset: Nested Donut of Total Modification Sites per Sequence',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax.set_aspect('equal')

    # Use constrained_layout for better handling of external annotations
    plt.tight_layout()
    # Use larger bbox_inches to ensure external annotations are not clipped
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAF9F6')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='#FAF9F6')
    plt.close()
    print(f"Nested donut chart saved to: {save_path}")
    print(f"Nested donut chart saved to: {pdf_path}")


if __name__ == "__main__":
    analyze_human_total_modification_sites()
