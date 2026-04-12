"""
Few-shot Analysis Zero-shot Module

This module contains functions for zero-shot feature alignment analysis.
"""

import csv
import os

import numpy as np
import umap
from matplotlib.lines import Line2D
from prettytable import PrettyTable
from scipy.spatial.distance import cdist

from dataset.human import Mer100Dataset
from dataset.plant_single import PlantSingleDataset
from utils.fewshot_analysis_constants import (
    TARGET_CLASSES, CLASS_NAMES, MORANDI_CLASS_COLORS, MORANDI_SPECIES_COLORS
)
from utils.fewshot_analysis_features import load_model_from_checkpoint, extract_dataset_features
from utils.fewshot_analysis_utils import save_figure, apply_plot_style, save_json, stable_mean

import matplotlib.pyplot as plt


def prepare_datasets(config):
    """Load human and plant datasets."""
    human_dataset = Mer100Dataset(
        mode='train',
        data_dir=config.data.human_data_dir,
        cache_dir=config.data.cache_dir,
        use_human3=True,
        use_cache=True,
    )
    plant_dataset = PlantSingleDataset(
        plant_dir=config.data.plant_data_dir,
        zero_dir=getattr(config.data, 'zero_data_dir', 'npy/zero'),
        cache_dir=config.data.cache_dir,
        use_cache=True,
        preload_cache=True,
    )
    return human_dataset, plant_dataset


def sample_human_reference_indices(human_dataset, random_seed, max_per_class=500):
    """Sample human reference indices for zero-shot analysis."""
    rng = np.random.RandomState(random_seed)
    indices = []
    y12 = human_dataset.y_12class
    for class_idx in TARGET_CLASSES:
        class_pos = np.where(y12[:, class_idx] == 1)[0]
        if len(class_pos) > max_per_class:
            class_pos = rng.choice(class_pos, size=max_per_class, replace=False)
        indices.extend(class_pos.tolist())
    return sorted(set(indices))


def get_class_mask(labels, class_idx):
    """Get mask for samples belonging to a specific class."""
    return labels[:, class_idx] == 1


def reduce_umap(features, n_neighbors=20, min_dist=0.18, random_state=42):
    """Reduce features using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def calculate_alignment_metrics(human_features, human_labels, plant_features, plant_labels):
    """Calculate alignment metrics between human and plant features."""
    metrics = {}

    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        human_mask = get_class_mask(human_labels, class_idx)
        plant_mask = get_class_mask(plant_labels, class_idx)

        human_class_features = human_features[human_mask]
        plant_class_features = plant_features[plant_mask]

        if len(human_class_features) == 0 or len(plant_class_features) == 0:
            continue

        human_centroid = human_class_features.mean(axis=0)
        plant_centroid = plant_class_features.mean(axis=0)
        centroid_distance = np.linalg.norm(human_centroid - plant_centroid)
        centroid_cosine = float(
            np.dot(human_centroid, plant_centroid) /
            (np.linalg.norm(human_centroid) * np.linalg.norm(plant_centroid) + 1e-8)
        )

        human_intra = np.linalg.norm(human_class_features - human_centroid, axis=1)
        plant_intra = np.linalg.norm(plant_class_features - plant_centroid, axis=1)
        nn_distances = cdist(plant_class_features, human_class_features, metric='euclidean').min(axis=1)

        metrics[class_name] = {
            'human_samples': int(len(human_class_features)),
            'plant_samples': int(len(plant_class_features)),
            'centroid_distance': float(centroid_distance),
            'centroid_cosine': centroid_cosine,
            'human_compactness': float(np.mean(human_intra)),
            'plant_compactness': float(np.mean(plant_intra)),
            'plant_to_human_nn_distance': float(np.mean(nn_distances)),
        }

    metrics['overall'] = {
        'avg_centroid_distance': stable_mean(
            [metrics[name]['centroid_distance'] for name in CLASS_NAMES if name in metrics]
        ),
        'avg_centroid_cosine': stable_mean(
            [metrics[name]['centroid_cosine'] for name in CLASS_NAMES if name in metrics]
        ),
        'avg_plant_to_human_nn_distance': stable_mean(
            [metrics[name]['plant_to_human_nn_distance'] for name in CLASS_NAMES if name in metrics]
        ),
    }
    return metrics


def log_alignment_metrics(logger, metrics, title):
    """Log alignment metrics as a table."""
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)

    table = PrettyTable()
    table.field_names = [
        'Class', 'Human N', 'Plant N', 'Centroid Dist',
        'Cosine', 'Human Compact', 'Plant Compact', 'Plant->Human NN'
    ]
    table.align = 'r'

    for class_name in CLASS_NAMES:
        if class_name not in metrics:
            continue
        m = metrics[class_name]
        table.add_row([
            class_name,
            m['human_samples'],
            m['plant_samples'],
            f"{m['centroid_distance']:.4f}",
            f"{m['centroid_cosine']:.4f}",
            f"{m['human_compactness']:.4f}",
            f"{m['plant_compactness']:.4f}",
            f"{m['plant_to_human_nn_distance']:.4f}",
        ])

    logger.info(f"\n{table}")
    logger.info(
        "Overall: "
        f"avg_centroid_distance={metrics['overall']['avg_centroid_distance']:.4f}, "
        f"avg_centroid_cosine={metrics['overall']['avg_centroid_cosine']:.4f}, "
        f"avg_plant_to_human_nn_distance={metrics['overall']['avg_plant_to_human_nn_distance']:.4f}"
    )


def build_joint_zero_shot_umap(human_features, human_labels, plant_features, plant_labels, output_dir, layer_name):
    """Build joint UMAP visualization for zero-shot analysis."""
    combined_features = np.vstack([human_features, plant_features])
    combined_labels = np.vstack([human_labels, plant_labels])
    species = np.array(['Human'] * len(human_features) + ['Plant'] * len(plant_features))

    target_mask = np.any(combined_labels[:, TARGET_CLASSES] == 1, axis=1)
    filtered_features = combined_features[target_mask]
    filtered_labels = combined_labels[target_mask]
    filtered_species = species[target_mask]
    embedding = reduce_umap(filtered_features)

    # Save joint UMAP points CSV
    save_joint_umap_points_csv(embedding, filtered_labels, filtered_species, output_dir, layer_name)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#FBF8F3')

    ax = axes[0]
    # Create a single rasterized scatter collection for all points
    scatter_points = []
    scatter_colors = []
    scatter_sizes = []
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = filtered_labels[:, class_idx] == 1
        if np.any(mask):
            scatter_points.append(embedding[mask])
            scatter_colors.extend([MORANDI_CLASS_COLORS[class_name]] * np.sum(mask))
            scatter_sizes.extend([24] * np.sum(mask))
    if scatter_points:
        all_points = np.vstack(scatter_points)
        # Rasterize the scatter points for efficient PDF editing
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            s=scatter_sizes,
            alpha=0.72,
            c=scatter_colors,
            edgecolors='none',
            rasterized=True  # Key parameter for rasterization
        )
    # Add legend manually
    legend_elements = []
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = filtered_labels[:, class_idx] == 1
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=MORANDI_CLASS_COLORS[class_name],
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Joint UMAP by Class')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    ax = axes[1]
    # Create a single rasterized scatter collection for all points
    scatter_points = []
    scatter_colors = []
    scatter_sizes = []
    for species_name in ['Human', 'Plant']:
        mask = filtered_species == species_name
        if np.any(mask):
            scatter_points.append(embedding[mask])
            scatter_colors.extend([MORANDI_SPECIES_COLORS[species_name]] * np.sum(mask))
            scatter_sizes.extend([24] * np.sum(mask))
    if scatter_points:
        all_points = np.vstack(scatter_points)
        # Rasterize the scatter points for efficient PDF editing
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            s=scatter_sizes,
            alpha=0.72,
            c=scatter_colors,
            edgecolors='none',
            rasterized=True  # Key parameter for rasterization
        )
    # Add legend manually
    legend_elements = []
    for species_name in ['Human', 'Plant']:
        mask = filtered_species == species_name
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=MORANDI_SPECIES_COLORS[species_name],
                                           markersize=8, label=species_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Joint UMAP by Species')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    fig.suptitle(f'Zero-shot Human/Plant Joint UMAP ({layer_name})', y=1.02, fontsize=14)
    save_figure(fig, os.path.join(output_dir, f'zeroshot_joint_umap_{layer_name}'))

    for species_name, features, labels in [
        ('human', human_features, human_labels),
        ('plant', plant_features, plant_labels),
    ]:
        mask = np.any(labels[:, TARGET_CLASSES] == 1, axis=1)
        embedding = reduce_umap(features[mask])
        filtered_labels = labels[mask]

        fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='#FBF8F3')
        # Create a single rasterized scatter collection for all points
        scatter_points = []
        scatter_colors = []
        scatter_sizes = []
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            class_mask = filtered_labels[:, class_idx] == 1
            if np.any(class_mask):
                scatter_points.append(embedding[class_mask])
                scatter_colors.extend([MORANDI_CLASS_COLORS[class_name]] * np.sum(class_mask))
                scatter_sizes.extend([24] * np.sum(class_mask))
        if scatter_points:
            all_points = np.vstack(scatter_points)
            # Rasterize the scatter points for efficient PDF editing
            ax.scatter(
                all_points[:, 0],
                all_points[:, 1],
                s=scatter_sizes,
                alpha=0.72,
                c=scatter_colors,
                edgecolors='none',
                rasterized=True  # Key parameter for rasterization
            )
        # Add legend manually
        legend_elements = []
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            class_mask = filtered_labels[:, class_idx] == 1
            if np.any(class_mask):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor=MORANDI_CLASS_COLORS[class_name],
                                               markersize=8, label=class_name))
        ax.legend(handles=legend_elements, frameon=False)
        ax.set_title(f'{species_name.capitalize()} UMAP by Class')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)
        save_figure(fig, os.path.join(output_dir, f'zeroshot_{species_name}_umap_{layer_name}'))


def save_alignment_metrics_csv(metrics, output_dir, layer_name):
    """Save zero-shot alignment metrics as tidy CSV."""
    csv_path = os.path.join(output_dir, f'zeroshot_alignment_metrics_{layer_name}.csv')
    fieldnames = [
        'class_name', 'human_samples', 'plant_samples',
        'centroid_distance', 'centroid_cosine',
        'human_compactness', 'plant_compactness',
        'plant_to_human_nn_distance'
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
                'plant_samples': m['plant_samples'],
                'centroid_distance': m['centroid_distance'],
                'centroid_cosine': m['centroid_cosine'],
                'human_compactness': m['human_compactness'],
                'plant_compactness': m['plant_compactness'],
                'plant_to_human_nn_distance': m['plant_to_human_nn_distance'],
            })
    return csv_path


def save_joint_umap_points_csv(embedding, filtered_labels, filtered_species, output_dir, layer_name):
    """Save joint UMAP point coordinates as CSV for R re-plotting."""
    csv_path = os.path.join(output_dir, f'zeroshot_joint_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    class_names = []
    class_indices = []
    for i in range(n_points):
        assigned = False
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            if filtered_labels[i, class_idx] == 1:
                class_names.append(class_name)
                class_indices.append(class_idx)
                assigned = True
                break
        if not assigned:
            class_names.append('Unknown')
            class_indices.append(-1)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y'])
        for i in range(n_points):
            writer.writerow([
                i,
                filtered_species[i],
                class_names[i],
                class_indices[i],
                float(embedding[i, 0]),
                float(embedding[i, 1]),
            ])
    return csv_path


def save_sample_count_summary(human_labels, plant_labels, output_dir, layer_name):
    """Save sample count summary CSV for supplementary bar chart."""
    csv_path = os.path.join(output_dir, f'sample_count_summary_{layer_name}.csv')
    fieldnames = ['analysis_type', 'class_name', 'human_samples', 'plant_samples']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            human_count = int(np.sum(human_labels[:, class_idx] == 1))
            plant_count = int(np.sum(plant_labels[:, class_idx] == 1))
            writer.writerow({
                'analysis_type': 'zero_shot',
                'class_name': class_name,
                'human_samples': human_count,
                'plant_samples': plant_count,
            })
    return csv_path


def run_zero_shot_analysis(config, checkpoint_path, output_dir, layer_name, logger):
    """Run zero-shot feature alignment analysis."""
    logger.info("\n" + "=" * 80)
    logger.info(f"ZERO-SHOT FEATURE ALIGNMENT ({layer_name})")
    logger.info("=" * 80)

    human_dataset, plant_dataset = prepare_datasets(config)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, config.device)

    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    human_indices = sample_human_reference_indices(human_dataset, config.random_seed)
    plant_indices = list(range(plant_dataset.num_plant))

    human_features, human_labels = extract_dataset_features(
        model, human_dataset, human_indices, config.device, layer_name=layer_name
    )
    plant_features, plant_labels = extract_dataset_features(
        model, plant_dataset, plant_indices, config.device, layer_name=layer_name
    )

    build_joint_zero_shot_umap(
        human_features, human_labels, plant_features, plant_labels,
        output_dir=output_dir, layer_name=layer_name
    )

    metrics = calculate_alignment_metrics(human_features, human_labels, plant_features, plant_labels)
    log_alignment_metrics(logger, metrics, f'ZERO-SHOT ALIGNMENT METRICS ({layer_name})')

    save_json(metrics, os.path.join(output_dir, f'zeroshot_alignment_metrics_{layer_name}.json'))
    save_alignment_metrics_csv(metrics, output_dir, layer_name)
    save_sample_count_summary(human_labels, plant_labels, output_dir, layer_name)
    return {
        'metrics': metrics,
        'human_features': human_features,
        'human_labels': human_labels,
        'plant_features': plant_features,
        'plant_labels': plant_labels,
    }
