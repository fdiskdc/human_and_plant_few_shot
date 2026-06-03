"""
utils/fewshot_analysis_zeroshot.py - 零样本对齐分析 / Zero-shot Alignment Analysis

零样本对齐:数据集准备、UMAP 降维、合成点生成、对齐指标、联合 UMAP。
Zero-shot alignment: dataset preparation, UMAP reduction, synthetic point generation, alignment metrics, joint UMAP.

功能模块 / Modules:
- 数据集准备 / Dataset preparation
- UMAP 降维 / UMAP reduction
- 合成点生成 / Synthetic point generation
- 对齐指标 / Alignment metrics
- 联合 UMAP / Joint UMAP
- main 分析函数 / Main analysis functions

输入 / Inputs:
- human3/, plant3/ 数据 / human, plant data
- features: 提取的特征 / Extracted features
- labels: 标签 / Labels

输出 / Outputs:
- 对齐指标 (MMD, Wasserstein) / Alignment metrics
- 联合 UMAP 图 / Joint UMAP plot
- 合成插值点 / Synthetic interpolation points

数据流 / Data Flow:
1. 准备人类+植物数据 / Prepare human+plant data
2. UMAP 降维 / UMAP reduction
3. 合成点生成 / Generate synthetic points
4. 对齐指标 / Alignment metrics
5. 联合 UMAP / Joint UMAP

相关文件 / Related Files:
- 调用 / Calls: utils.fewshot_analysis_constants, sklearn, umap
- 被调用 / Called by: zero_shot_fewshot_analysis.py

使用示例 / Usage Example:
    from utils.fewshot_analysis_zeroshot import run_zeroshot_alignment
    run_zeroshot_alignment(features, labels)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import csv
import os

import numpy as np
import umap
from matplotlib.lines import Line2D
from prettytable import PrettyTable
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from dataset.human import Mer100Dataset
from dataset.plant_single import PlantSingleDataset
from utils.fewshot_analysis_constants import (
    TARGET_CLASSES, CLASS_NAMES, MORANDI_CLASS_COLORS, MORANDI_SPECIES_COLORS,
    HIGH_CONTRAST_MOD_COLORS, HIGH_CONTRAST_SPECIES_COLORS,
    INTERPOLATION_PARAMS, POINT_STYLE_PARAMS
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


def generate_synthetic_points_by_group(points, groups, n_synthetic=5, jitter_strength=0.01):
    """
    Generate synthetic points via interpolation within same group + Gaussian jitter.
    
    Args:
        points: numpy array of shape (n_points, 2) - UMAP 2D coordinates
        groups: numpy array of shape (n_points,) - group labels (e.g., 'Human_m6A', 'Plant_m6A')
        n_synthetic: number of synthetic points to generate per real point
        jitter_strength: fraction of coordinate range to use for jitter (default 1.0%)
    
    Returns:
        synthetic_points: numpy array of synthetic points
        synthetic_groups: numpy array of group labels for synthetic points
        original_indices: indices of original points used to generate each synthetic point
    """
    synthetic_points = []
    synthetic_groups = []
    original_indices = []
    
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(42)
    
    # Compute jitter range based on actual point distribution
    if len(points) > 0:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        jitter_x = x_range * jitter_strength if x_range > 0 else jitter_strength
        jitter_y = y_range * jitter_strength if y_range > 0 else jitter_strength
    else:
        jitter_x = jitter_y = jitter_strength
    
    for group in unique_groups:
        group_mask = groups == group
        group_points = points[group_mask]
        
        if len(group_points) < 2:
            # Not enough points for interpolation, just add jittered copies
            for idx in np.where(group_mask)[0]:
                for _ in range(n_synthetic):
                    synthetic_points.append(group_points[0] + rng.randn(2) * np.array([jitter_x, jitter_y]))
                    synthetic_groups.append(group)
                    original_indices.append(idx)
            continue
        
        # Build KD-tree for finding neighbors within group
        tree = KDTree(group_points)
        
        # Get global indices for this group
        group_global_indices = np.where(group_mask)[0]
        
        for local_idx, point in enumerate(group_points):
            global_idx = group_global_indices[local_idx]
            neighbors_idx = tree.query(point, k=min(INTERPOLATION_PARAMS['n_neighbors_for_interpolation'] + 1, len(group_points)))[1]
            
            for _ in range(n_synthetic):
                # Randomly select 2 different neighbors for interpolation
                # Exclude self (local_idx) from neighbor choices
                valid_neighbors = neighbors_idx[neighbors_idx != local_idx]
                if len(valid_neighbors) == 0:
                    valid_neighbors = neighbors_idx
                neighbor_local_idx = rng.choice(valid_neighbors)
                
                # Linear interpolation between point and neighbor
                alpha = rng.uniform(0.2, 0.8)
                synthetic = point + alpha * (group_points[neighbor_local_idx] - point)
                
                # Add Gaussian jitter
                synthetic += rng.randn(2) * np.array([jitter_x, jitter_y])
                
                synthetic_points.append(synthetic)
                synthetic_groups.append(group)
                original_indices.append(global_idx)
    
    if synthetic_points:
        return np.array(synthetic_points), np.array(synthetic_groups), np.array(original_indices)
    else:
        return np.array([]), np.array([]), np.array([])


def filter_by_modification(features, labels, species, target_modification=None):
    """
    Filter data to only include samples of target modification.
    
    Args:
        features: numpy array of features
        labels: numpy array of labels (one-hot encoded)
        species: numpy array of species labels
        target_modification: string like 'm6A', 'm5C', 'Y', or None for all
    
    Returns:
        filtered_features, filtered_labels, filtered_species, source_groups
    """
    if target_modification is None:
        # Return all target classes
        target_mask = np.any(labels[:, TARGET_CLASSES] == 1, axis=1)
        filtered_features = features[target_mask]
        filtered_labels = labels[target_mask]
        filtered_species = species[target_mask]
        
        # Create source groups
        source_groups = []
        for i in range(len(filtered_species)):
            sp = filtered_species[i]
            for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
                if filtered_labels[i, class_idx] == 1:
                    source_groups.append(f"{sp}_{class_name}")
                    break
        return filtered_features, filtered_labels, filtered_species, np.array(source_groups)
    
    # Find class index for target modification
    if target_modification not in CLASS_NAMES:
        raise ValueError(f"Unknown modification: {target_modification}. Must be one of {CLASS_NAMES}")
    
    target_class_idx = CLASS_NAMES.index(target_modification)
    target_class_target = TARGET_CLASSES[target_class_idx]
    
    # Filter to only this modification
    mod_mask = labels[:, target_class_target] == 1
    filtered_features = features[mod_mask]
    filtered_labels = labels[mod_mask]
    filtered_species = species[mod_mask]
    
    # Create source groups: species + modification
    source_groups = np.array([f"{sp}_{target_modification}" for sp in filtered_species])
    
    return filtered_features, filtered_labels, filtered_species, source_groups


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


def save_joint_umap_points_csv(embedding, filtered_labels, filtered_species, source_groups,
                                 output_dir, layer_name, is_synthetic=None,
                                 target_modification=None):
    """Save joint UMAP point coordinates as CSV for R re-plotting."""
    mod_str = f'_{target_modification}' if target_modification else ''
    csv_path = os.path.join(output_dir, f'zeroshot_joint_umap_points_{layer_name}{mod_str}.csv')
    n_points = embedding.shape[0]
    
    # Determine is_synthetic array
    if is_synthetic is None:
        is_synthetic = np.array([False] * n_points)
    
    # Determine point_role array
    point_role = np.array(['real' if not syn else 'synthetic' for syn in is_synthetic])
    
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
        writer.writerow(['sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y', 
                        'is_synthetic', 'point_role', 'source_group'])
        for i in range(n_points):
            writer.writerow([
                i,
                filtered_species[i],
                class_names[i],
                class_indices[i],
                float(embedding[i, 0]),
                float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                point_role[i],
                source_groups[i] if i < len(source_groups) else 'Unknown'
            ])
    return csv_path


def build_joint_zero_shot_umap(human_features, human_labels, plant_features, plant_labels, 
                                output_dir, layer_name, target_modification=None,
                                use_high_contrast=True, use_enhancement=True):
    """
    Build joint UMAP visualization for zero-shot analysis.
    
    Args:
        human_features: Human feature array
        human_labels: Human labels array
        plant_features: Plant feature array
        plant_labels: Plant labels array
        output_dir: Output directory
        layer_name: Layer name for file naming
        target_modification: If specified, only show this modification (e.g., 'm6A')
        use_high_contrast: Use high contrast colors instead of Morandi
        use_enhancement: Generate synthetic points for visual enhancement
    """
    combined_features = np.vstack([human_features, plant_features])
    combined_labels = np.vstack([human_labels, plant_labels])
    species = np.array(['Human'] * len(human_features) + ['Plant'] * len(plant_features))
    
    # Filter by target modification if specified
    filtered_features, filtered_labels, filtered_species, source_groups = filter_by_modification(
        combined_features, combined_labels, species, target_modification
    )
    
    if len(filtered_features) == 0:
        print(f"Warning: No samples found for target_modification={target_modification}")
        return
    
    # Generate UMAP on filtered data only (no other modifications involved)
    embedding = reduce_umap(filtered_features)
    
    # Track which points are synthetic
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points for visual enhancement
    if use_enhancement and len(filtered_features) > 0:
        synthetic_emb, synthetic_groups, _ = generate_synthetic_points_by_group(
            embedding, source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        
        if len(synthetic_emb) > 0:
            # Combine real and synthetic
            embedding = np.vstack([embedding, synthetic_emb])
            is_synthetic = np.array([False] * n_real + [True] * len(synthetic_emb))
            # Extend source_groups for synthetic points
            source_groups = np.concatenate([source_groups, synthetic_groups])
            # Extend filtered_species (not really needed for synthetic but for consistency)
            filtered_species = np.concatenate([filtered_species, np.array(['Synthetic'] * len(synthetic_emb))])
            # Extend filtered_labels for synthetic points by inferring from synthetic_groups
            synthetic_labels = np.zeros((len(synthetic_emb), filtered_labels.shape[1]))
            for i, sg in enumerate(synthetic_groups):
                # synthetic_groups are like 'Human_m6A' or 'Plant_m5C'
                for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
                    if sg.endswith(f'_{class_name}'):
                        synthetic_labels[i, class_idx] = 1
                        break
            filtered_labels = np.vstack([filtered_labels, synthetic_labels])
    
    # Save joint UMAP points CSV
    save_joint_umap_points_csv(
        embedding, filtered_labels, filtered_species, source_groups,
        output_dir, layer_name, is_synthetic, target_modification=target_modification
    )
    
    # Select color scheme
    if use_high_contrast:
        species_colors = {
            'Human': HIGH_CONTRAST_SPECIES_COLORS['Human']['primary'],
            'Plant': HIGH_CONTRAST_SPECIES_COLORS['Plant']['primary'],
        }
        class_colors = {mod: HIGH_CONTRAST_MOD_COLORS[mod]['primary'] for mod in CLASS_NAMES}
    else:
        species_colors = MORANDI_SPECIES_COLORS.copy()
        class_colors = MORANDI_CLASS_COLORS.copy()

    allowed_class_names = [target_modification] if target_modification else CLASS_NAMES
    
    # Point sizes and alphas
    real_size = int(24 * POINT_STYLE_PARAMS['real_point_size_ratio'])  # ~12
    synthetic_size = int(24 * POINT_STYLE_PARAMS['synthetic_point_size_ratio'])  # ~8
    real_alpha = POINT_STYLE_PARAMS['real_point_alpha']  # 0.55
    synthetic_alpha = POINT_STYLE_PARAMS['synthetic_point_alpha']  # 0.15
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#FBF8F3')
    
    # ===== Left plot: by class =====
    ax = axes[0]
    
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        if class_name not in allowed_class_names:
            continue
        real_mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        
        # Real points for this class
        if np.any(real_mask):
            ax.scatter(
                embedding[:n_real][real_mask, 0],
                embedding[:n_real][real_mask, 1],
                s=real_size,
                alpha=real_alpha,
                c=class_colors.get(class_name, '#999999'),
                edgecolors='none',
                rasterized=True,
                marker='o'
            )
        
        # Synthetic points for this class
        if use_enhancement and len(synthetic_emb) > 0:
            syn_class_mask = np.array([f.endswith(f'_{class_name}') for f in source_groups[n_real:]])
            if np.any(syn_class_mask):
                if use_high_contrast and class_name in HIGH_CONTRAST_MOD_COLORS:
                    syn_color = HIGH_CONTRAST_MOD_COLORS[class_name]['secondary']
                else:
                    syn_color = class_colors.get(class_name, '#999999')
                ax.scatter(
                    embedding[n_real:][syn_class_mask, 0],
                    embedding[n_real:][syn_class_mask, 1],
                    s=synthetic_size,
                    alpha=synthetic_alpha,
                    c=syn_color,
                    edgecolors='none',
                    rasterized=True,
                    marker='o'
                )
    
    # Legend
    legend_elements = []
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        if class_name not in allowed_class_names:
            continue
        mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=class_colors.get(class_name, '#999999'),
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Joint UMAP by Class' + (f' ({target_modification})' if target_modification else ''))
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)
    
    # ===== Right plot: by species =====
    ax = axes[1]
    
    # Plot real points for each species
    for species_name in ['Human', 'Plant']:
        real_mask = filtered_species[:n_real] == species_name if n_real > 0 else np.zeros(len(filtered_species), dtype=bool)
        
        if np.any(real_mask):
            ax.scatter(
                embedding[:n_real][real_mask, 0],
                embedding[:n_real][real_mask, 1],
                s=real_size,
                alpha=real_alpha,
                c=species_colors.get(species_name, '#999999'),
                edgecolors='none',
                rasterized=True,
                marker='o'
            )
        
        # Synthetic points for this species
        if use_enhancement and len(synthetic_emb) > 0:
            syn_species_mask = np.array([f.startswith(f'{species_name}_') for f in source_groups[n_real:]])
            if np.any(syn_species_mask):
                if use_high_contrast and species_name in HIGH_CONTRAST_SPECIES_COLORS:
                    syn_color = HIGH_CONTRAST_SPECIES_COLORS[species_name]['secondary']
                else:
                    syn_color = species_colors.get(species_name, '#999999')
                ax.scatter(
                    embedding[n_real:][syn_species_mask, 0],
                    embedding[n_real:][syn_species_mask, 1],
                    s=synthetic_size,
                    alpha=synthetic_alpha,
                    c=syn_color,
                    edgecolors='none',
                    rasterized=True,
                    marker='o'
                )
    
    # Legend
    legend_elements = []
    for species_name in ['Human', 'Plant']:
        mask = filtered_species[:n_real] == species_name if n_real > 0 else np.zeros(len(filtered_species), dtype=bool)
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=species_colors.get(species_name, '#999999'),
                                           markersize=8, label=species_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Joint UMAP by Species')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    mod_str = f'_{target_modification}' if target_modification else ''
    fig.suptitle(f'Zero-shot Human/Plant Joint UMAP ({layer_name}{mod_str})', y=1.02, fontsize=14)
    save_figure(fig, os.path.join(output_dir, f'zeroshot_joint_umap_{layer_name}{mod_str}'))


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


def run_zero_shot_analysis(config, checkpoint_path, output_dir, layer_name, logger,
                           model=None, checkpoint=None):
    """Run zero-shot feature alignment analysis."""
    logger.info("\n" + "=" * 80)
    logger.info(f"ZERO-SHOT FEATURE ALIGNMENT ({layer_name})")
    logger.info("=" * 80)

    human_dataset, plant_dataset = prepare_datasets(config)
    if model is None or checkpoint is None:
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

    # Build joint UMAP with all classes (for backward compatibility)
    build_joint_zero_shot_umap(
        human_features, human_labels, plant_features, plant_labels,
        output_dir=output_dir, layer_name=layer_name,
        target_modification=None, use_high_contrast=True, use_enhancement=True
    )
    
    # Also build per-modification UMAPs for m6A, m5C, Y
    for mod in CLASS_NAMES:
        logger.info(f"Building UMAP for {mod} only...")
        build_joint_zero_shot_umap(
            human_features, human_labels, plant_features, plant_labels,
            output_dir=output_dir, layer_name=layer_name,
            target_modification=mod, use_high_contrast=True, use_enhancement=True
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
        'plant_dataset': plant_dataset,
    }
