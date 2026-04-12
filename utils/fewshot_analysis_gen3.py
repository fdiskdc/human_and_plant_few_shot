"""
Few-shot Analysis Gen3 Module

This module contains functions for 3-generation data analysis.
Includes zero-shot alignment metrics comparing gen3 with human data.
"""

import csv
import os

import numpy as np
from scipy.spatial.distance import cdist

from dataset.gen3_zero import Gen3ZeroDataset, MOD_NAMES
from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES
from utils.fewshot_analysis_features import load_model_from_checkpoint, extract_dataset_features
from utils.fewshot_analysis_zeroshot import reduce_umap, get_class_mask
from utils.fewshot_analysis_utils import save_figure, apply_plot_style, save_json, stable_mean

import matplotlib.pyplot as plt


def run_gen3_analysis(config, checkpoint_path, output_dir, layer_name, logger):
    """Run analysis on 3-generation (gen3) data with human comparison."""
    logger.info("\n" + "=" * 80)
    logger.info(f"3-GENERATION DATA ANALYSIS ({layer_name})")
    logger.info("=" * 80)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, config.device)
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    # Load gen3 dataset
    gen3_dataset = Gen3ZeroDataset(
        mode='test',
        data_dir='npy',
    )
    logger.info(f"Gen3 dataset loaded: {len(gen3_dataset)} samples")

    # Precompute structures if needed
    cache_stats = gen3_dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        logger.info("Batch cache not found, precomputing all secondary structures...")
        gen3_dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)

    # Extract features for gen3 data
    gen3_indices = list(range(len(gen3_dataset)))
    gen3_features, gen3_labels = extract_dataset_features(
        model, gen3_dataset, gen3_indices, config.device, layer_name=layer_name
    )

    # Save gen3 sample counts
    save_gen3_sample_count_summary(gen3_labels, output_dir, layer_name, logger)

    # Build gen3 UMAP
    build_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger)

    # Load human dataset for comparison
    from dataset.human import Mer100Dataset
    human_dataset = Mer100Dataset(
        mode='train',
        data_dir=config.data.human_data_dir,
        cache_dir=config.data.cache_dir,
        use_human3=True,
        use_cache=True,
    )
    logger.info(f"Human dataset loaded: {len(human_dataset)} samples")

    # Extract features for human data
    human_indices = list(range(len(human_dataset)))
    human_features, human_labels = extract_dataset_features(
        model, human_dataset, human_indices, config.device, layer_name=layer_name
    )

    # Calculate gen3 alignment metrics (comparing gen3 with human)
    gen3_alignment_metrics = calculate_gen3_alignment_metrics(
        human_features, human_labels, gen3_features, gen3_labels, logger
    )

    # Save gen3 alignment metrics
    save_gen3_alignment_metrics_csv(gen3_alignment_metrics, output_dir, layer_name)
    save_json(gen3_alignment_metrics, os.path.join(output_dir, f'gen3_alignment_metrics_{layer_name}.json'))

    # Build joint UMAP for gen3 and human
    build_gen3_joint_umap(
        human_features, human_labels, gen3_features, gen3_labels,
        output_dir, layer_name, logger
    )

    return {
        'gen3_features': gen3_features,
        'gen3_labels': gen3_labels,
        'gen3_dataset': gen3_dataset,
        'human_features': human_features,
        'human_labels': human_labels,
        'gen3_alignment_metrics': gen3_alignment_metrics,
    }


def save_gen3_sample_count_summary(gen3_labels, output_dir, layer_name, logger):
    """Save sample count summary for gen3 data."""
    csv_path = os.path.join(output_dir, f'gen3_sample_count_summary_{layer_name}.csv')
    fieldnames = ['class_idx', 'class_name', 'positive_samples', 'negative_samples', 'total_samples']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_idx, class_name in MOD_NAMES.items():
            if class_idx >= gen3_labels.shape[1]:
                continue
            pos_count = int(np.sum(gen3_labels[:, class_idx] == 1))
            neg_count = int(np.sum(gen3_labels[:, class_idx] == 0))
            writer.writerow({
                'class_idx': class_idx,
                'class_name': class_name,
                'positive_samples': pos_count,
                'negative_samples': neg_count,
                'total_samples': pos_count + neg_count,
            })
    logger.info(f"Gen3 sample count summary saved to {csv_path}")
    return csv_path


def calculate_gen3_alignment_metrics(human_features, human_labels, gen3_features, gen3_labels, logger):
    """
    Calculate alignment metrics comparing gen3 data with human data.
    Similar to plant vs human alignment metrics.
    """
    metrics = {}

    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        human_mask = get_class_mask(human_labels, class_idx)
        gen3_mask = get_class_mask(gen3_labels, class_idx)

        human_class_features = human_features[human_mask]
        gen3_class_features = gen3_features[gen3_mask]

        if len(human_class_features) == 0 or len(gen3_class_features) == 0:
            logger.warning(f"  Skipping {class_name}: human={len(human_class_features)}, gen3={len(gen3_class_features)}")
            continue

        human_centroid = human_class_features.mean(axis=0)
        gen3_centroid = gen3_class_features.mean(axis=0)
        centroid_distance = np.linalg.norm(human_centroid - gen3_centroid)
        centroid_cosine = float(
            np.dot(human_centroid, gen3_centroid) /
            (np.linalg.norm(human_centroid) * np.linalg.norm(gen3_centroid) + 1e-8)
        )

        human_intra = np.linalg.norm(human_class_features - human_centroid, axis=1)
        gen3_intra = np.linalg.norm(gen3_class_features - gen3_centroid, axis=1)
        nn_distances = cdist(gen3_class_features, human_class_features, metric='euclidean').min(axis=1)

        metrics[class_name] = {
            'human_samples': int(len(human_class_features)),
            'gen3_samples': int(len(gen3_class_features)),
            'centroid_distance': float(centroid_distance),
            'centroid_cosine': centroid_cosine,
            'human_compactness': float(np.mean(human_intra)),
            'gen3_compactness': float(np.mean(gen3_intra)),
            'gen3_to_human_nn_distance': float(np.mean(nn_distances)),
        }

    metrics['overall'] = {
        'avg_centroid_distance': stable_mean(
            [metrics[name]['centroid_distance'] for name in CLASS_NAMES if name in metrics]
        ),
        'avg_centroid_cosine': stable_mean(
            [metrics[name]['centroid_cosine'] for name in CLASS_NAMES if name in metrics]
        ),
        'avg_gen3_to_human_nn_distance': stable_mean(
            [metrics[name]['gen3_to_human_nn_distance'] for name in CLASS_NAMES if name in metrics]
        ),
    }

    # Log metrics
    logger.info("\n" + "=" * 80)
    logger.info("GEN3 ZERO-SHOT ALIGNMENT METRICS")
    logger.info("=" * 80)
    for class_name in CLASS_NAMES:
        if class_name not in metrics:
            continue
        m = metrics[class_name]
        logger.info(
            f"  {class_name}: human={m['human_samples']}, gen3={m['gen3_samples']}, "
            f"centroid_dist={m['centroid_distance']:.4f}, cosine={m['centroid_cosine']:.4f}, "
            f"human_compact={m['human_compactness']:.4f}, gen3_compact={m['gen3_compactness']:.4f}, "
            f"nn_dist={m['gen3_to_human_nn_distance']:.4f}"
        )
    logger.info(
        f"  Overall: avg_centroid_dist={metrics['overall']['avg_centroid_distance']:.4f}, "
        f"avg_cosine={metrics['overall']['avg_centroid_cosine']:.4f}, "
        f"avg_nn_dist={metrics['overall']['avg_gen3_to_human_nn_distance']:.4f}"
    )

    return metrics


def save_gen3_alignment_metrics_csv(metrics, output_dir, layer_name):
    """Save gen3 alignment metrics as tidy CSV."""
    csv_path = os.path.join(output_dir, f'gen3_alignment_metrics_{layer_name}.csv')
    fieldnames = [
        'class_name', 'human_samples', 'gen3_samples',
        'centroid_distance', 'centroid_cosine',
        'human_compactness', 'gen3_compactness',
        'gen3_to_human_nn_distance'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in CLASS_NAMES:
            if class_name not in metrics:
                continue
            m = metrics[class_name]
            writer.writerow({
                'class_name': class_name,
                'human_samples': m['human_samples'],
                'gen3_samples': m['gen3_samples'],
                'centroid_distance': m['centroid_distance'],
                'centroid_cosine': m['centroid_cosine'],
                'human_compactness': m['human_compactness'],
                'gen3_compactness': m['gen3_compactness'],
                'gen3_to_human_nn_distance': m['gen3_to_human_nn_distance'],
            })
    return csv_path


def build_gen3_joint_umap(human_features, human_labels, gen3_features, gen3_labels, output_dir, layer_name, logger):
    """Build joint UMAP visualization for gen3 and human data."""
    from utils.fewshot_analysis_constants import MORANDI_CLASS_COLORS, MORANDI_SPECIES_COLORS

    combined_features = np.vstack([human_features, gen3_features])
    combined_labels = np.vstack([human_labels, gen3_labels])
    species = np.array(['Human'] * len(human_features) + ['Gen3'] * len(gen3_features))

    # Filter for target classes only
    target_mask = np.any(combined_labels[:, TARGET_CLASSES] == 1, axis=1)
    filtered_features = combined_features[target_mask]
    filtered_labels = combined_labels[target_mask]
    filtered_species = species[target_mask]

    if len(filtered_features) == 0:
        logger.warning("No target class samples found for gen3 joint UMAP.")
        return

    embedding = reduce_umap(filtered_features)

    # Save joint UMAP points CSV
    csv_path = os.path.join(output_dir, f'gen3_joint_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y'])
        for i in range(n_points):
            class_name = 'Unknown'
            class_idx = -1
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if filtered_labels[i, ci] == 1:
                    class_name = cn
                    class_idx = ci
                    break
            writer.writerow([
                i, filtered_species[i], class_name, class_idx,
                float(embedding[i, 0]), float(embedding[i, 1])
            ])

    # Plot by class - rasterize points for efficient PDF editing
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#FBF8F3')

    ax = axes[0]
    # Collect all points for rasterization
    scatter_points = []
    scatter_colors = []
    scatter_sizes = []
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = filtered_labels[:, class_idx] == 1
        if np.any(mask):
            scatter_points.append(embedding[mask])
            scatter_colors.extend([MORANDI_CLASS_COLORS.get(class_name, '#999999')] * np.sum(mask))
            scatter_sizes.extend([24] * np.sum(mask))
    if scatter_points:
        all_points = np.vstack(scatter_points)
        ax.scatter(
            all_points[:, 0], all_points[:, 1],
            s=scatter_sizes, alpha=0.72, c=scatter_colors,
            edgecolors='none', rasterized=True  # Key: rasterize for PDF editing
        )
    # Add legend manually
    legend_elements = []
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = filtered_labels[:, class_idx] == 1
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=MORANDI_CLASS_COLORS.get(class_name, '#999999'),
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Gen3/Human Joint UMAP by Class')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    # Plot by species - rasterize points for efficient PDF editing
    ax = axes[1]
    species_colors = {'Human': '#8E8A84', 'Gen3': '#A69C87'}
    scatter_points = []
    scatter_colors = []
    scatter_sizes = []
    for species_name in ['Human', 'Gen3']:
        mask = filtered_species == species_name
        if np.any(mask):
            scatter_points.append(embedding[mask])
            scatter_colors.extend([species_colors[species_name]] * np.sum(mask))
            scatter_sizes.extend([24] * np.sum(mask))
    if scatter_points:
        all_points = np.vstack(scatter_points)
        ax.scatter(
            all_points[:, 0], all_points[:, 1],
            s=scatter_sizes, alpha=0.72, c=scatter_colors,
            edgecolors='none', rasterized=True  # Key: rasterize for PDF editing
        )
    # Add legend manually
    legend_elements = []
    for species_name in ['Human', 'Gen3']:
        mask = filtered_species == species_name
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=species_colors[species_name],
                                           markersize=8, label=species_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Gen3/Human Joint UMAP by Species')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    fig.suptitle(f'Gen3/Human Joint UMAP ({layer_name})', y=1.02, fontsize=14)
    save_figure(fig, os.path.join(output_dir, f'gen3_joint_umap_{layer_name}'))

    # Also save individual species UMAP
    for species_name, features, labels in [
        ('human', human_features, human_labels),
        ('gen3', gen3_features, gen3_labels),
    ]:
        mask = np.any(labels[:, TARGET_CLASSES] == 1, axis=1)
        if not np.any(mask):
            continue
        embedding = reduce_umap(features[mask])
        filtered_labels_ind = labels[mask]

        fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='#FBF8F3')
        # Rasterize points for efficient PDF editing
        scatter_points = []
        scatter_colors = []
        scatter_sizes = []
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            class_mask = filtered_labels_ind[:, class_idx] == 1
            if np.any(class_mask):
                scatter_points.append(embedding[class_mask])
                scatter_colors.extend([MORANDI_CLASS_COLORS.get(class_name, '#999999')] * np.sum(class_mask))
                scatter_sizes.extend([24] * np.sum(class_mask))
        if scatter_points:
            all_points = np.vstack(scatter_points)
            ax.scatter(
                all_points[:, 0], all_points[:, 1],
                s=scatter_sizes, alpha=0.72, c=scatter_colors,
                edgecolors='none', rasterized=True  # Key: rasterize for PDF editing
            )
        # Add legend manually
        legend_elements = []
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            class_mask = filtered_labels_ind[:, class_idx] == 1
            if np.any(class_mask):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor=MORANDI_CLASS_COLORS.get(class_name, '#999999'),
                                               markersize=8, label=class_name))
        ax.legend(handles=legend_elements, frameon=False)
        ax.set_title(f'{species_name.capitalize()} UMAP by Class')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)
        save_figure(fig, os.path.join(output_dir, f'gen3_{species_name}_umap_{layer_name}'))

    logger.info(f"Gen3 joint UMAP saved to {output_dir}")


def build_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger):
    """Build UMAP visualization for gen3 data."""
    # Only use samples with at least one positive label
    pos_mask = np.any(gen3_labels != 0, axis=1)
    if not np.any(pos_mask):
        logger.warning("No positive samples found in gen3 data, skipping UMAP.")
        return

    filtered_features = gen3_features[pos_mask]
    filtered_labels = gen3_labels[pos_mask]

    # Reduce dimensions with UMAP
    embedding = reduce_umap(filtered_features)

    # Save UMAP points CSV
    csv_path = os.path.join(output_dir, f'gen3_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'class_idx', 'class_name', 'umap_x', 'umap_y'])
        for i in range(n_points):
            # Find the primary class for this sample
            primary_class_idx = int(np.argmax(filtered_labels[i]))
            primary_class_name = MOD_NAMES.get(primary_class_idx, 'Unknown')
            writer.writerow([
                i, primary_class_idx, primary_class_name,
                float(embedding[i, 0]), float(embedding[i, 1])
            ])

    # Plot by class - rasterize points for efficient PDF editing
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='#FBF8F3')

    # Use a color map for 12 classes
    class_colors = plt.cm.tab20(np.linspace(0, 1, 12))

    # Rasterize points for efficient PDF editing
    scatter_points = []
    scatter_colors = []
    scatter_sizes = []
    for class_idx, class_name in MOD_NAMES.items():
        class_mask = filtered_labels[:, class_idx] == 1
        if np.any(class_mask):
            scatter_points.append(embedding[class_mask])
            scatter_colors.extend([class_colors[class_idx]] * np.sum(class_mask))
            scatter_sizes.extend([24] * np.sum(class_mask))
    if scatter_points:
        all_points = np.vstack(scatter_points)
        ax.scatter(
            all_points[:, 0], all_points[:, 1],
            s=scatter_sizes, alpha=0.72, c=scatter_colors,
            edgecolors='none', rasterized=True  # Key: rasterize for PDF editing
        )
    # Add legend manually
    legend_elements = []
    for class_idx, class_name in MOD_NAMES.items():
        class_mask = filtered_labels[:, class_idx] == 1
        if np.any(class_mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=class_colors[class_idx],
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_title(f'Gen3 UMAP by Class ({layer_name})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'gen3_umap_by_class_{layer_name}'))
    logger.info(f"Gen3 UMAP saved to {output_dir}")
