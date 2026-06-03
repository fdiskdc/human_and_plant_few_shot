"""
utils/fewshot_analysis_gen3.py - 3代数据分析 (vs 人类) / 3-Generation Data Analysis (vs Human)

3 代数据 vs 人类数据分析:对齐指标、联合 UMAP、跨代比较。
3-generation data vs human data analysis: alignment metrics, joint UMAP, cross-generation comparison.

功能模块 / Modules:
- 3gen 数据加载 / 3gen data loading
- 对齐指标计算 / Alignment metrics computation
- 联合 UMAP / Joint UMAP
- 跨代比较 / Cross-generation comparison
- main 分析函数 / Main analysis functions

输入 / Inputs:
- 3gen/seq.npy, 3gen/12loc.npy: 3gen 数据 / 3gen data
- human3/seq.npy, human3/12loc.npy: 人类数据 / Human data
- features: 提取的特征 / Extracted features
- labels: 标签 / Labels

输出 / Outputs:
- 对齐指标 / Alignment metrics
- 联合 UMAP 图 / Joint UMAP plot
- 跨代比较图 / Cross-generation comparison plots

数据流 / Data Flow:
1. 加载 3gen + 人类数据 / Load 3gen + human data
2. 提取特征 / Extract features
3. 对齐 + UMAP / Alignment + UMAP
4. 跨代比较 / Cross-generation comparison
5. 保存图表 / Save plots

相关文件 / Related Files:
- 调用 / Calls: utils.fewshot_analysis_constants, sklearn, umap
- 被调用 / Called by: zero_shot_fewshot_analysis.py

使用示例 / Usage Example:
    from utils.fewshot_analysis_gen3 import run_gen3_analysis
    run_gen3_analysis(features, labels)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import csv
import os

import numpy as np
from scipy.spatial.distance import cdist

from dataset.gen3_zero import Gen3ZeroDataset, MOD_NAMES
from utils.fewshot_analysis_constants import (
    TARGET_CLASSES, CLASS_NAMES, MORANDI_CLASS_COLORS, MORANDI_SPECIES_COLORS,
    HIGH_CONTRAST_MOD_COLORS, HIGH_CONTRAST_GEN3_COLORS, HIGH_CONTRAST_SPECIES_COLORS,
    INTERPOLATION_PARAMS, POINT_STYLE_PARAMS
)
from utils.fewshot_analysis_features import load_model_from_checkpoint, extract_dataset_features
from utils.fewshot_analysis_zeroshot import (
    reduce_umap, get_class_mask, generate_synthetic_points_by_group, filter_by_modification
)
from utils.fewshot_analysis_utils import save_figure, apply_plot_style, save_json, stable_mean

import matplotlib.pyplot as plt


def run_gen3_analysis(config, checkpoint_path, output_dir, layer_name, logger,
                      model=None, checkpoint=None):
    """Run analysis on 3-generation (gen3) data with human comparison."""
    logger.info("\n" + "=" * 80)
    logger.info(f"3-GENERATION DATA ANALYSIS ({layer_name})")
    logger.info("=" * 80)

    if model is None or checkpoint is None:
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

    # Build gen3 UMAP (with enhancement)
    build_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger)

    from dataset.human import Mer100Dataset
    human_dataset = Mer100Dataset(
        mode='train',
        data_dir=config.data.human_data_dir,
        cache_dir=config.data.cache_dir,
        use_human3=True,
        use_cache=True,
    )
    logger.info(f"Human dataset loaded for Gen3 comparison: {len(human_dataset)} samples")

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

    # Build joint UMAP for gen3 and human (with enhancement)
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


def save_gen3_joint_umap_points_csv(embedding, filtered_labels, filtered_species, source_groups,
                                      output_dir, layer_name, is_synthetic=None):
    """Save gen3 joint UMAP point coordinates as CSV for R re-plotting."""
    csv_path = os.path.join(output_dir, f'gen3_joint_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    
    if is_synthetic is None:
        is_synthetic = np.array([False] * n_points)

    if len(filtered_labels) != n_points or len(filtered_species) != n_points:
        raise ValueError(
            "Gen3 joint UMAP metadata is misaligned: "
            f"embedding={n_points}, labels={len(filtered_labels)}, species={len(filtered_species)}"
        )
    
    point_role = np.array(['real' if not syn else 'synthetic' for syn in is_synthetic])
    
    class_names = []
    class_indices = []
    for i in range(n_points):
        class_name = 'Unknown'
        class_idx = -1
        for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
            if filtered_labels[i, ci] == 1:
                class_name = cn
                class_idx = ci
                break
        class_names.append(class_name)
        class_indices.append(class_idx)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y',
                        'is_synthetic', 'point_role', 'source_group'])
        for i in range(n_points):
            writer.writerow([
                i, filtered_species[i], class_names[i], class_indices[i],
                float(embedding[i, 0]), float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                point_role[i],
                source_groups[i] if i < len(source_groups) else 'Unknown'
            ])
    return csv_path


def build_gen3_joint_umap(human_features, human_labels, gen3_features, gen3_labels,
                          output_dir, layer_name, logger, target_modification=None,
                          use_high_contrast=True, use_enhancement=True):
    """
    Build joint UMAP visualization for gen3 and human data.
    
    Args:
        human_features: Human feature array
        human_labels: Human labels array
        gen3_features: Gen3 feature array
        gen3_labels: Gen3 labels array
        output_dir: Output directory
        layer_name: Layer name for file naming
        logger: Logger instance
        target_modification: If specified, only show this modification (e.g., 'm6A')
        use_high_contrast: Use high contrast colors instead of Morandi
        use_enhancement: Generate synthetic points for visual enhancement
    """
    combined_features = np.vstack([human_features, gen3_features])
    combined_labels = np.vstack([human_labels, gen3_labels])
    species = np.array(['Human'] * len(human_features) + ['Gen3'] * len(gen3_features))
    
    # Filter by target modification if specified
    filtered_features, filtered_labels, filtered_species, source_groups = filter_by_modification(
        combined_features, combined_labels, species, target_modification
    )
    
    if len(filtered_features) == 0:
        logger.warning(f"No samples found for target_modification={target_modification}")
        return

    # Generate UMAP on filtered data only
    embedding = reduce_umap(filtered_features)
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points
    synthetic_emb = np.array([])
    if use_enhancement and len(filtered_features) > 0:
        synthetic_emb, synthetic_groups, synthetic_source_indices = generate_synthetic_points_by_group(
            embedding, source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        
        if len(synthetic_emb) > 0:
            embedding = np.vstack([embedding, synthetic_emb])
            is_synthetic = np.array([False] * n_real + [True] * len(synthetic_emb))
            source_groups = np.concatenate([source_groups, synthetic_groups])
            filtered_species = np.concatenate([filtered_species, np.array(['Synthetic'] * len(synthetic_emb))])
            filtered_labels = np.vstack([filtered_labels, filtered_labels[synthetic_source_indices]])
    
    # Save joint UMAP points CSV with new fields
    save_gen3_joint_umap_points_csv(
        embedding, filtered_labels, filtered_species, source_groups,
        output_dir, layer_name, is_synthetic
    )
    
    # Select color scheme
    if use_high_contrast:
        species_colors = {
            'Human': HIGH_CONTRAST_SPECIES_COLORS['Human']['primary'],
            'Gen3': HIGH_CONTRAST_SPECIES_COLORS['Gen3']['primary'],
        }
        class_colors = {mod: HIGH_CONTRAST_MOD_COLORS[mod]['primary'] for mod in CLASS_NAMES}
    else:
        species_colors = {'Human': '#8E8A84', 'Gen3': '#A69C87'}
        class_colors = MORANDI_CLASS_COLORS.copy()
    
    # Point sizes and alphas
    real_size = int(24 * POINT_STYLE_PARAMS['real_point_size_ratio'])  # ~12
    synthetic_size = int(24 * POINT_STYLE_PARAMS['synthetic_point_size_ratio'])  # ~8
    real_alpha = POINT_STYLE_PARAMS['real_point_alpha']  # 0.55
    synthetic_alpha = POINT_STYLE_PARAMS['synthetic_point_alpha']  # 0.15
    
    # Plot by class
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#FBF8F3')
    
    ax = axes[0]
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        real_mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        
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
        mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=class_colors.get(class_name, '#999999'),
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Gen3/Human Joint UMAP by Class' + (f' ({target_modification})' if target_modification else ''))
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    # Plot by species
    ax = axes[1]
    for species_name in ['Human', 'Gen3']:
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
    for species_name in ['Human', 'Gen3']:
        mask = filtered_species[:n_real] == species_name if n_real > 0 else np.zeros(len(filtered_species), dtype=bool)
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=species_colors.get(species_name, '#999999'),
                                           markersize=8, label=species_name))
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title('Gen3/Human Joint UMAP by Species')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)

    mod_str = f'_{target_modification}' if target_modification else ''
    fig.suptitle(f'Gen3/Human Joint UMAP ({layer_name}{mod_str})', y=1.02, fontsize=14)
    save_figure(fig, os.path.join(output_dir, f'gen3_joint_umap_{layer_name}{mod_str}'))

    # Also save individual species UMAP with enhancement
    for species_name, features, labels in [
        ('human', human_features, human_labels),
        ('gen3', gen3_features, gen3_labels),
    ]:
        mask = np.any(labels[:, TARGET_CLASSES] == 1, axis=1)
        if not np.any(mask):
            continue
        
        species_features = features[mask]
        species_labels = labels[mask]
        
        # Create proper source groups
        species_source_groups = []
        for i in range(len(species_labels)):
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if species_labels[i, ci] == 1:
                    species_source_groups.append(f"{species_name}_{cn}")
                    break
        species_source_groups = np.array(species_source_groups)
        
        embedding = reduce_umap(species_features)
        n_real_spec = len(embedding)
        is_synthetic_spec = np.array([False] * n_real_spec)
        
        synthetic_emb_spec = np.array([])
        if use_enhancement and len(species_features) > 0:
            synthetic_emb_spec, synthetic_groups_spec, _ = generate_synthetic_points_by_group(
                embedding, species_source_groups,
                n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
                jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
            )
            
            if len(synthetic_emb_spec) > 0:
                embedding = np.vstack([embedding, synthetic_emb_spec])
                is_synthetic_spec = np.array([False] * n_real_spec + [True] * len(synthetic_emb_spec))
                species_source_groups = np.concatenate([species_source_groups, synthetic_groups_spec])

        fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='#FBF8F3')
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            real_mask = species_labels[:n_real_spec, class_idx] == 1 if n_real_spec > 0 else np.zeros(len(species_labels), dtype=bool)
            
            if np.any(real_mask):
                ax.scatter(
                    embedding[:n_real_spec][real_mask, 0],
                    embedding[:n_real_spec][real_mask, 1],
                    s=real_size,
                    alpha=real_alpha,
                    c=class_colors.get(class_name, '#999999'),
                    edgecolors='none',
                    rasterized=True,
                    marker='o'
                )
            
            if use_enhancement and len(synthetic_emb_spec) > 0:
                syn_class_mask = np.array([f.endswith(f'_{class_name}') for f in species_source_groups[n_real_spec:]])
                if np.any(syn_class_mask):
                    if use_high_contrast and class_name in HIGH_CONTRAST_MOD_COLORS:
                        syn_color = HIGH_CONTRAST_MOD_COLORS[class_name]['secondary']
                    else:
                        syn_color = class_colors.get(class_name, '#999999')
                    ax.scatter(
                        embedding[n_real_spec:][syn_class_mask, 0],
                        embedding[n_real_spec:][syn_class_mask, 1],
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
            mask_temp = species_labels[:n_real_spec, class_idx] == 1 if n_real_spec > 0 else np.zeros(len(species_labels), dtype=bool)
            if np.any(mask_temp):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor=class_colors.get(class_name, '#999999'),
                                               markersize=8, label=class_name))
        ax.legend(handles=legend_elements, frameon=False)
        ax.set_title(f'{species_name.capitalize()} UMAP by Class' + (f' ({target_modification})' if target_modification else ''))
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)
        save_figure(fig, os.path.join(output_dir, f'gen3_{species_name}_umap_{layer_name}{mod_str}'))

    logger.info(f"Gen3 joint UMAP saved to {output_dir}")


def build_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger,
                    use_high_contrast=True, use_enhancement=True):
    """Build UMAP visualization for gen3 data with enhancement."""
    # Only use samples with at least one positive label
    pos_mask = np.any(gen3_labels != 0, axis=1)
    if not np.any(pos_mask):
        logger.warning("No positive samples found in gen3 data, skipping UMAP.")
        return

    filtered_features = gen3_features[pos_mask]
    filtered_labels = gen3_labels[pos_mask]

    # Create source groups for gen3 (treat as separate from plant)
    source_groups = []
    for i in range(len(filtered_labels)):
        primary_class_idx = int(np.argmax(filtered_labels[i]))
        primary_class_name = MOD_NAMES.get(primary_class_idx, 'Unknown')
        source_groups.append(f"Gen3_{primary_class_name}")
    source_groups = np.array(source_groups)

    # Reduce dimensions with UMAP
    embedding = reduce_umap(filtered_features)
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points
    synthetic_emb = np.array([])
    if use_enhancement and len(filtered_features) > 0:
        synthetic_emb, synthetic_groups, _ = generate_synthetic_points_by_group(
            embedding, source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        
        if len(synthetic_emb) > 0:
            embedding = np.vstack([embedding, synthetic_emb])
            is_synthetic = np.array([False] * n_real + [True] * len(synthetic_emb))
            source_groups = np.concatenate([source_groups, synthetic_groups])

    # Save UMAP points CSV with new fields
    csv_path = os.path.join(output_dir, f'gen3_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    
    point_role = np.array(['real' if not syn else 'synthetic' for syn in is_synthetic])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'class_idx', 'class_name', 'umap_x', 'umap_y',
                        'is_synthetic', 'point_role', 'source_group'])
        for i in range(n_points):
            if i < n_real:
                primary_class_idx = int(np.argmax(filtered_labels[i]))
                primary_class_name = MOD_NAMES.get(primary_class_idx, 'Unknown')
            else:
                # For synthetic points, derive from source_group
                primary_class_name = source_groups[i].split('_', 1)[-1] if i < len(source_groups) else 'Unknown'
                primary_class_idx = -1
            writer.writerow([
                i,
                primary_class_idx if i < n_real else -1,
                primary_class_name,
                float(embedding[i, 0]),
                float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                point_role[i],
                source_groups[i] if i < len(source_groups) else 'Unknown'
            ])

    # Point sizes and alphas
    real_size = int(24 * POINT_STYLE_PARAMS['real_point_size_ratio'])  # ~12
    synthetic_size = int(24 * POINT_STYLE_PARAMS['synthetic_point_size_ratio'])  # ~8
    real_alpha = POINT_STYLE_PARAMS['real_point_alpha']  # 0.55
    synthetic_alpha = POINT_STYLE_PARAMS['synthetic_point_alpha']  # 0.15

    # Color for gen3 is distinct from plant
    gen3_class_colors = {
        mod: HIGH_CONTRAST_GEN3_COLORS[mod]['primary'] if use_high_contrast and mod in HIGH_CONTRAST_GEN3_COLORS
            else HIGH_CONTRAST_MOD_COLORS[mod]['primary'] if use_high_contrast
            else MORANDI_CLASS_COLORS.get(mod, '#999999')
        for mod in CLASS_NAMES
    }
    gen3_syn_colors = {
        mod: HIGH_CONTRAST_GEN3_COLORS[mod]['secondary'] if use_high_contrast and mod in HIGH_CONTRAST_GEN3_COLORS
            else HIGH_CONTRAST_MOD_COLORS[mod]['secondary'] if use_high_contrast
            else '#999999'
        for mod in CLASS_NAMES
    }

    # Plot by class
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor='#FBF8F3')

    for class_idx, class_name in MOD_NAMES.items():
        if class_name not in CLASS_NAMES:
            continue
        
        real_mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        
        if np.any(real_mask):
            ax.scatter(
                embedding[:n_real][real_mask, 0],
                embedding[:n_real][real_mask, 1],
                s=real_size,
                alpha=real_alpha,
                c=gen3_class_colors.get(class_name, '#999999'),
                edgecolors='none',
                rasterized=True,
                marker='o'
            )
        
        if use_enhancement and len(synthetic_emb) > 0:
            syn_class_mask = np.array([f.endswith(f'_{class_name}') for f in source_groups[n_real:]])
            if np.any(syn_class_mask):
                ax.scatter(
                    embedding[n_real:][syn_class_mask, 0],
                    embedding[n_real:][syn_class_mask, 1],
                    s=synthetic_size,
                    alpha=synthetic_alpha,
                    c=gen3_syn_colors.get(class_name, '#999999'),
                    edgecolors='none',
                    rasterized=True,
                    marker='o'
                )

    # Add legend manually
    legend_elements = []
    for class_idx, class_name in MOD_NAMES.items():
        if class_name not in CLASS_NAMES:
            continue
        mask = filtered_labels[:n_real, class_idx] == 1 if n_real > 0 else np.zeros(len(filtered_labels), dtype=bool)
        if np.any(mask):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=gen3_class_colors.get(class_name, '#999999'),
                                           markersize=8, label=class_name))
    ax.legend(handles=legend_elements, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_title(f'Gen3 UMAP by Class ({layer_name})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'gen3_umap_by_class_{layer_name}'))
    logger.info(f"Gen3 UMAP saved to {output_dir}")
