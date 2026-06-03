"""
utils/fewshot_analysis_fewshot.py - 小样本轨迹分析 / Few-shot Trajectory Analysis

小样本轨迹分析:索引划分、质心、紧密度、分离度、finetune、UMAP 绘图。
Few-shot trajectory analysis: index split, centroids, compactness, separation, fine-tune, UMAP plots.

功能模块 / Modules:
- 索引划分 / Index split
- 质心计算 / Centroid computation
- 紧密度 + 分离度 / Compactness + separation
- Finetune 训练 / Finetune training
- UMAP 轨迹图 / UMAP trajectory plot
- main 分析函数 / Main analysis functions

输入 / Inputs:
- features: 提取的特征 / Extracted features
- labels: 标签 / Labels
- SHOT_COUNTS: shot 数 / Shot counts
- n_runs: 运行次数 / Number of runs

输出 / Outputs:
- 紧密度/分离度指标 / Compactness/separation metrics
- UMAP 轨迹图 / UMAP trajectory plot
- 终端报告 / Terminal report

数据流 / Data Flow:
1. 划分索引 / Split indices
2. 计算质心 / Compute centroids
3. finetune / Fine-tune
4. 评估紧密度/分离度 / Evaluate compactness/separation
5. UMAP 绘图 / UMAP plot

相关文件 / Related Files:
- 调用 / Calls: utils.fewshot_analysis_constants, sklearn, umap
- 被调用 / Called by: zero_shot_fewshot_analysis.py

使用示例 / Usage Example:
    from utils.fewshot_analysis_fewshot import run_few_shot_trajectory
    run_few_shot_trajectory(features, labels, shot_counts=[1, 5, 10])

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import csv
import copy
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils.fewshot_analysis_constants import (
    TARGET_CLASSES, CLASS_NAMES, CLASS_NAME_MAP, SHOT_COUNTS,
    HIGH_CONTRAST_MOD_COLORS, HIGH_CONTRAST_HUMAN_MOD_COLORS,
    INTERPOLATION_PARAMS, POINT_STYLE_PARAMS
)
from utils.fewshot_analysis_features import load_model_from_checkpoint, extract_dataset_features
from utils.fewshot_analysis_zeroshot import (
    get_class_mask, prepare_datasets, reduce_umap, generate_synthetic_points_by_group
)
from utils.fewshot_analysis_utils import save_figure, apply_plot_style, save_json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def split_plant_indices(plant_dataset, random_seed, train_fraction=0.1):
    """Split plant indices into train and test sets."""
    rng = np.random.RandomState(random_seed)
    plant_train_indices = []
    plant_test_indices = []

    for class_idx in TARGET_CLASSES:
        class_indices = plant_dataset.get_plant_indices_by_class(class_idx)
        class_indices = class_indices.copy()
        rng.shuffle(class_indices)
        split_at = max(1, int(len(class_indices) * train_fraction))
        plant_train_indices.extend(class_indices[:split_at])
        plant_test_indices.extend(class_indices[split_at:])

    plant_train_indices = sorted(set(plant_train_indices))
    plant_test_indices = sorted(set(i for i in plant_test_indices if i not in plant_train_indices))
    return plant_train_indices, plant_test_indices


def compute_human_reference_centroids(human_features, human_labels):
    """Compute centroids for human reference features."""
    centroids = {}
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = get_class_mask(human_labels, class_idx)
        class_features = human_features[mask]
        if len(class_features) > 0:
            centroids[class_name] = class_features.mean(axis=0)
    return centroids


def compute_class_compactness(features, labels, class_idx):
    """Compute compactness for a class."""
    mask = get_class_mask(labels, class_idx)
    class_features = features[mask]
    if len(class_features) == 0:
        return float('nan'), None
    centroid = class_features.mean(axis=0)
    compactness = float(np.mean(np.linalg.norm(class_features - centroid, axis=1)))
    return compactness, centroid


def compute_class_separation(features, labels, class_idx):
    """Compute separation metrics for a class."""
    own_compactness, own_centroid = compute_class_compactness(features, labels, class_idx)
    if own_centroid is None:
        return {
            'nearest_other_centroid_distance': float('nan'),
            'separation_margin': float('nan'),
            'separation_ratio': float('nan'),
        }

    other_distances = []
    for other_class_idx in TARGET_CLASSES:
        if other_class_idx == class_idx:
            continue
        _, other_centroid = compute_class_compactness(features, labels, other_class_idx)
        if other_centroid is not None:
            other_distances.append(float(np.linalg.norm(own_centroid - other_centroid)))

    if not other_distances:
        nearest_other = float('nan')
    else:
        nearest_other = min(other_distances)

    return {
        'nearest_other_centroid_distance': nearest_other,
        'separation_margin': float(nearest_other - own_compactness) if not np.isnan(nearest_other) else float('nan'),
        'separation_ratio': float(nearest_other / (own_compactness + 1e-8)) if not np.isnan(nearest_other) else float('nan'),
    }


def compute_trajectory_metrics(zero_shot_features, current_features, plant_labels, target_class, human_centroids):
    """Compute trajectory metrics for few-shot analysis."""
    class_name = CLASS_NAME_MAP[target_class]
    class_mask = get_class_mask(plant_labels, target_class)

    if not np.any(class_mask):
        return {}

    zero_class = zero_shot_features[class_mask]
    current_class = current_features[class_mask]
    human_centroid = human_centroids[class_name]

    zero_dist = np.linalg.norm(zero_class - human_centroid, axis=1)
    current_dist = np.linalg.norm(current_class - human_centroid, axis=1)

    zero_compactness, _ = compute_class_compactness(zero_shot_features, plant_labels, target_class)
    current_compactness, _ = compute_class_compactness(current_features, plant_labels, target_class)

    zero_sep = compute_class_separation(zero_shot_features, plant_labels, target_class)
    current_sep = compute_class_separation(current_features, plant_labels, target_class)

    sample_movement = np.linalg.norm(current_class - zero_class, axis=1)

    return {
        'class_name': class_name,
        'num_samples': int(class_mask.sum()),
        'avg_sample_movement': float(np.mean(sample_movement)),
        'mean_distance_to_human_centroid': float(np.mean(current_dist)),
        'distance_to_human_centroid_gain': float(np.mean(zero_dist) - np.mean(current_dist)),
        'median_distance_to_human_centroid_gain': float(np.median(zero_dist) - np.median(current_dist)),
        'compactness': float(current_compactness),
        'compactness_gain': float(zero_compactness - current_compactness),
        'nearest_other_centroid_distance': float(current_sep['nearest_other_centroid_distance']),
        'separation_margin': float(current_sep['separation_margin']),
        'separation_margin_gain': float(current_sep['separation_margin'] - zero_sep['separation_margin']),
        'separation_ratio': float(current_sep['separation_ratio']),
        'separation_ratio_gain': float(current_sep['separation_ratio'] - zero_sep['separation_ratio']),
    }


def sample_binary_support_set(dataset, plant_train_indices, k_shot, target_class, seed):
    """
    Match the independent binary protocol in fewshot_plant_3way_independent.py:
    positives are target class plant samples, negatives are other plant classes.
    """
    rng = np.random.RandomState(seed)

    target_positives = set(dataset.get_plant_indices_by_class(target_class))
    valid_positives = [idx for idx in plant_train_indices if idx in target_positives]

    other_classes = [c for c in TARGET_CLASSES if c != target_class]
    valid_negatives = [
        idx for idx in plant_train_indices
        if idx not in target_positives and any(dataset.plant_y12[idx, oc] == 1 for oc in other_classes)
    ]

    if len(valid_positives) == 0 or len(valid_negatives) == 0:
        return []

    pos_replace = len(valid_positives) < k_shot
    neg_replace = len(valid_negatives) < k_shot

    pos_sample = rng.choice(valid_positives, size=k_shot, replace=pos_replace).tolist()
    neg_sample = rng.choice(valid_negatives, size=k_shot, replace=neg_replace).tolist()
    return pos_sample + neg_sample


def prepare_binary_labels(data_list, target_class):
    """Prepare binary labels for fine-tuning."""
    labels = []
    for data in data_list:
        labels.append(float(data.y[0, target_class].item() == 1.0))
    return torch.tensor(labels, dtype=torch.float32)


def fine_tune_binary_model(model, support_data_list, target_class, config, device, logger):
    """Fine-tune model on binary support set."""
    if not support_data_list:
        return model

    support_loader = PyGDataLoader(
        support_data_list,
        batch_size=min(len(support_data_list), 32),
        shuffle=True,
        num_workers=0,
    )

    ft_lr = config.few_shot.get('ft_lr', 1e-3)
    ft_epochs = config.few_shot.get('ft_epochs', 10)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)

    model.train()
    for epoch in range(ft_epochs):
        epoch_loss = 0.0
        for batch in support_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch.x, batch.edge_index, batch.batch)
            logits_12 = outputs[0] if isinstance(outputs, tuple) else outputs
            binary_logits = logits_12[:, target_class]
            binary_labels = batch.y[:, target_class].float()

            loss = criterion(binary_logits, binary_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % max(1, ft_epochs // 3) == 0:
            logger.info(
                f"  Fine-tune class {target_class} epoch {epoch + 1}/{ft_epochs}, "
                f"loss={epoch_loss / max(1, len(support_loader)):.4f}"
            )

    model.eval()
    return model


def save_trajectory_metrics_csv(results, output_dir, layer_name):
    """Save few-shot trajectory metrics as tidy CSV."""
    csv_path = os.path.join(output_dir, f'fewshot_trajectory_metrics_{layer_name}.csv')
    fieldnames = [
        'class_name', 'shot', 'num_samples',
        'avg_sample_movement', 'mean_distance_to_human_centroid',
        'distance_to_human_centroid_gain', 'median_distance_to_human_centroid_gain',
        'compactness', 'compactness_gain',
        'nearest_other_centroid_distance',
        'separation_margin', 'separation_margin_gain',
        'separation_ratio', 'separation_ratio_gain'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in CLASS_NAMES:
            if class_name not in results:
                continue
            for shot in SHOT_COUNTS:
                if shot not in results[class_name]:
                    continue
                m = results[class_name][shot]
                writer.writerow({
                    'class_name': class_name,
                    'shot': shot,
                    'num_samples': m.get('num_samples', ''),
                    'avg_sample_movement': m.get('avg_sample_movement', ''),
                    'mean_distance_to_human_centroid': m.get('mean_distance_to_human_centroid', ''),
                    'distance_to_human_centroid_gain': m.get('distance_to_human_centroid_gain', ''),
                    'median_distance_to_human_centroid_gain': m.get('median_distance_to_human_centroid_gain', ''),
                    'compactness': m.get('compactness', ''),
                    'compactness_gain': m.get('compactness_gain', ''),
                    'nearest_other_centroid_distance': m.get('nearest_other_centroid_distance', ''),
                    'separation_margin': m.get('separation_margin', ''),
                    'separation_margin_gain': m.get('separation_margin_gain', ''),
                    'separation_ratio': m.get('separation_ratio', ''),
                    'separation_ratio_gain': m.get('separation_ratio_gain', ''),
                })
    return csv_path


def save_trajectory_umap_points_csv(
    human_embedding, human_ref_labels,
    shot_embeddings, plant_ref_labels,
    target_class, output_dir, layer_name,
    human_synthetic_embedding=None, human_synthetic_groups=None,
    shot_synthetic_embeddings=None, shot_synthetic_groups=None
):
    """Save per-class trajectory UMAP point coordinates as CSV."""
    class_name = CLASS_NAME_MAP[target_class]
    csv_path = os.path.join(output_dir, f'fewshot_trajectory_umap_points_{class_name}_{layer_name}.csv')

    rows = []
    sample_counter = 0

    # Human reference points
    n_human = human_embedding.shape[0]
    for i in range(n_human):
        rows.append([
            sample_counter, 'Human', 'human_reference',
            class_name, class_name, target_class,
            'ref',
            float(human_embedding[i, 0]), float(human_embedding[i, 1]),
            '0', 'real', f'Human_{class_name}'
        ])
        sample_counter += 1

    if human_synthetic_embedding is not None and len(human_synthetic_embedding) > 0:
        for i in range(len(human_synthetic_embedding)):
            rows.append([
                sample_counter, 'Human', 'human_reference',
                class_name, class_name, target_class,
                'ref',
                float(human_synthetic_embedding[i, 0]), float(human_synthetic_embedding[i, 1]),
                '1', 'synthetic', human_synthetic_groups[i]
            ])
            sample_counter += 1

    # Plant test points per shot
    for shot in SHOT_COUNTS:
        emb = shot_embeddings[shot]
        n_plant = emb.shape[0]
        for i in range(n_plant):
            rows.append([
                sample_counter, 'Plant', 'plant_test',
                class_name, class_name, target_class,
                shot,
                float(emb[i, 0]), float(emb[i, 1]),
                '0', 'real', f'Plant_{class_name}_{shot}shot'
            ])
            sample_counter += 1

        synthetic_emb = None if shot_synthetic_embeddings is None else shot_synthetic_embeddings.get(shot)
        synthetic_groups = None if shot_synthetic_groups is None else shot_synthetic_groups.get(shot)
        if synthetic_emb is not None and len(synthetic_emb) > 0:
            for i in range(len(synthetic_emb)):
                rows.append([
                    sample_counter, 'Plant', 'plant_test',
                    class_name, class_name, target_class,
                    shot,
                    float(synthetic_emb[i, 0]), float(synthetic_emb[i, 1]),
                    '1', 'synthetic', synthetic_groups[i]
                ])
                sample_counter += 1

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'species', 'reference_type',
            'target_class_name', 'displayed_class_name', 'displayed_class_idx',
            'shot', 'umap_x', 'umap_y', 'is_synthetic', 'point_role', 'source_group'
        ])
        writer.writerows(rows)
    return csv_path


def plot_few_shot_trajectory_umap(human_features, human_labels, per_shot_features, plant_labels, target_class, output_dir, layer_name):
    """Plot few-shot trajectory UMAP visualization."""
    class_name = CLASS_NAME_MAP[target_class]
    figure_name = f'fewshot_trajectory_{class_name}_{layer_name}'

    human_mask = get_class_mask(human_labels, target_class)
    plant_mask = get_class_mask(plant_labels, target_class)
    human_ref = human_features[human_mask]
    human_ref_labels = human_labels[human_mask]
    plant_ref_labels = plant_labels[plant_mask]

    if len(human_ref) == 0 or len(plant_ref_labels) == 0:
        return

    all_feature_blocks = [human_ref]
    shot_sizes = []
    for shot in SHOT_COUNTS:
        block = per_shot_features[shot][plant_mask]
        all_feature_blocks.append(block)
        shot_sizes.append(len(block))

    embedded = reduce_umap(np.vstack(all_feature_blocks))
    human_size = len(human_ref)
    shot_embeddings = {}
    cursor = human_size
    for shot, size in zip(SHOT_COUNTS, shot_sizes):
        shot_embeddings[shot] = embedded[cursor:cursor + size]
        cursor += size
    human_embedding = embedded[:human_size]

    human_source_groups = np.array([f'Human_{class_name}'] * len(human_embedding))
    shot_source_groups = {
        shot: np.array([f'Plant_{class_name}_{shot}shot'] * len(shot_embeddings[shot]))
        for shot in SHOT_COUNTS
    }

    human_synthetic_embedding = np.array([])
    human_synthetic_groups = np.array([])
    if len(human_embedding) > 0:
        human_synthetic_embedding, human_synthetic_groups, _ = generate_synthetic_points_by_group(
            human_embedding, human_source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )

    shot_synthetic_embeddings = {}
    shot_synthetic_groups = {}
    plant_n_synthetic = max(
        INTERPOLATION_PARAMS['n_synthetic_per_point'] * 2,
        INTERPOLATION_PARAMS['n_synthetic_per_point'] + 8,
    )
    for shot in SHOT_COUNTS:
        synthetic_emb, synthetic_groups, _ = generate_synthetic_points_by_group(
            shot_embeddings[shot], shot_source_groups[shot],
            n_synthetic=plant_n_synthetic,
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        shot_synthetic_embeddings[shot] = synthetic_emb
        shot_synthetic_groups[shot] = synthetic_groups

    # Save trajectory UMAP points CSV
    save_trajectory_umap_points_csv(
        human_embedding, human_ref_labels,
        shot_embeddings, plant_ref_labels,
        target_class, output_dir, layer_name,
        human_synthetic_embedding=human_synthetic_embedding,
        human_synthetic_groups=human_synthetic_groups,
        shot_synthetic_embeddings=shot_synthetic_embeddings,
        shot_synthetic_groups=shot_synthetic_groups
    )

    fig, axes = plt.subplots(1, len(SHOT_COUNTS), figsize=(6 * len(SHOT_COUNTS), 5.8), facecolor='#FBF8F3')
    human_real_size = int(24 * POINT_STYLE_PARAMS['real_point_size_ratio'])
    human_synthetic_size = max(1, int(24 * POINT_STYLE_PARAMS['synthetic_point_size_ratio']))
    plant_real_size = human_real_size * 2
    plant_synthetic_size = max(1, human_synthetic_size * 2)
    real_alpha = POINT_STYLE_PARAMS['real_point_alpha']
    synthetic_alpha = POINT_STYLE_PARAMS['synthetic_point_alpha']
    human_real_color = HIGH_CONTRAST_HUMAN_MOD_COLORS[class_name]['primary']
    human_synth_color = HIGH_CONTRAST_HUMAN_MOD_COLORS[class_name]['secondary']
    plant_real_color = HIGH_CONTRAST_MOD_COLORS[class_name]['primary']
    plant_synth_color = HIGH_CONTRAST_MOD_COLORS[class_name]['secondary']

    for ax, shot in zip(axes, SHOT_COUNTS):
        if len(human_synthetic_embedding) > 0:
            ax.scatter(
                human_synthetic_embedding[:, 0], human_synthetic_embedding[:, 1],
                s=human_synthetic_size, alpha=synthetic_alpha, marker='o',
                c=human_synth_color, edgecolors='none',
                rasterized=True
            )
        ax.scatter(
            human_embedding[:, 0], human_embedding[:, 1],
            s=human_real_size, alpha=real_alpha, marker='o',
            c=human_real_color, edgecolors='none',
            rasterized=True
        )

        if len(shot_synthetic_embeddings[shot]) > 0:
            ax.scatter(
                shot_synthetic_embeddings[shot][:, 0], shot_synthetic_embeddings[shot][:, 1],
                s=plant_synthetic_size, alpha=synthetic_alpha, marker='o',
                c=plant_synth_color, edgecolors='none',
                linewidths=0,
                rasterized=True
            )
        ax.scatter(
            shot_embeddings[shot][:, 0], shot_embeddings[shot][:, 1],
            s=plant_real_size, alpha=real_alpha, marker='o',
            c=plant_real_color, edgecolors='none',
            linewidths=0,
            rasterized=True
        )

        ax.set_title(f'{shot}-shot')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='Human reference', markerfacecolor=human_real_color, markersize=6, alpha=real_alpha),
        Line2D([0], [0], marker='o', color='w', label='Human synthetic', markerfacecolor=human_synth_color, markersize=4, alpha=synthetic_alpha),
        Line2D([0], [0], marker='o', color='w', label=f'Plant {class_name}', markerfacecolor=plant_real_color, markersize=9, alpha=real_alpha),
        Line2D([0], [0], marker='o', color='w', label=f'Plant {class_name} synthetic', markerfacecolor=plant_synth_color, markersize=6, alpha=synthetic_alpha),
    ]
    axes[0].legend(handles=legend_items, frameon=False, loc='best')

    fig.suptitle(
        f'Few-shot Trajectory vs Human Reference: {class_name} Only ({layer_name})',
        y=1.03,
        fontsize=14,
    )
    save_figure(fig, os.path.join(output_dir, figure_name))


def log_trajectory_metrics(logger, results):
    """Log trajectory metrics as a table."""
    logger.info("\n" + "=" * 80)
    logger.info("FEW-SHOT TRAJECTORY METRICS")
    logger.info("=" * 80)

    for class_name in CLASS_NAMES:
        if class_name not in results:
            continue
        table = PrettyTable()
        table.field_names = [
            'Shot', 'Dist->Human', 'Align Gain', 'Compact', 'Compact Gain',
            'Sep Margin', 'Sep Gain', 'Sep Ratio'
        ]
        table.align = 'r'

        for shot in SHOT_COUNTS:
            if shot not in results[class_name]:
                continue
            m = results[class_name][shot]
            table.add_row([
                f'{shot}-shot',
                f"{m['mean_distance_to_human_centroid']:.4f}",
                f"{m['distance_to_human_centroid_gain']:.4f}",
                f"{m['compactness']:.4f}",
                f"{m['compactness_gain']:.4f}",
                f"{m['separation_margin']:.4f}",
                f"{m['separation_margin_gain']:.4f}",
                f"{m['separation_ratio']:.4f}",
            ])

        logger.info(f"\nClass {class_name}")
        logger.info(f"\n{table}")


def run_few_shot_trajectory_analysis(config, checkpoint_path, output_dir, layer_name, logger,
                                     zero_shot_bundle, baseline_model=None, baseline_state_dict=None):
    """Run few-shot trajectory analysis."""
    logger.info("\n" + "=" * 80)
    logger.info(f"FEW-SHOT TRAJECTORY ANALYSIS ({layer_name})")
    logger.info("=" * 80)

    plant_dataset = zero_shot_bundle.get('plant_dataset')
    if plant_dataset is None:
        _, plant_dataset = prepare_datasets(config)
    plant_train_indices, plant_test_indices = split_plant_indices(plant_dataset, config.random_seed)

    logger.info(f"Plant train pool: {len(plant_train_indices)}")
    logger.info(f"Plant test set: {len(plant_test_indices)}")

    human_centroids = compute_human_reference_centroids(
        zero_shot_bundle['human_features'], zero_shot_bundle['human_labels']
    )

    if baseline_model is None:
        baseline_model, _ = load_model_from_checkpoint(checkpoint_path, config.device)
    baseline_model.eval()
    if baseline_state_dict is None:
        baseline_state_dict = copy.deepcopy(baseline_model.state_dict())

    zero_shot_test_features, plant_test_labels = extract_dataset_features(
        baseline_model, plant_dataset, plant_test_indices, config.device, layer_name=layer_name
    )

    results = defaultdict(dict)

    for target_class in TARGET_CLASSES:
        class_name = CLASS_NAME_MAP[target_class]
        logger.info("\n" + "-" * 80)
        logger.info(f"Few-shot class: {class_name} ({target_class})")
        logger.info("-" * 80)

        per_shot_features = {}
        per_shot_features[0] = zero_shot_test_features
        results[class_name][0] = compute_trajectory_metrics(
            zero_shot_test_features,
            zero_shot_test_features,
            plant_test_labels,
            target_class,
            human_centroids,
        )

        for shot in [1, 5, 10]:
            logger.info(f"Running {shot}-shot fine-tuning for {class_name}")
            model = copy.deepcopy(baseline_model)
            model.load_state_dict(baseline_state_dict)
            model.to(config.device)
            model.eval()

            support_indices = sample_binary_support_set(
                plant_dataset,
                plant_train_indices,
                k_shot=shot,
                target_class=target_class,
                seed=config.random_seed + target_class * 100 + shot,
            )
            support_data_list = [plant_dataset[idx] for idx in support_indices]
            pos_count = int(prepare_binary_labels(support_data_list, target_class).sum().item()) if support_data_list else 0

            logger.info(
                f"Support set size={len(support_data_list)}, "
                f"positives={pos_count}, negatives={len(support_data_list) - pos_count}"
            )

            model = fine_tune_binary_model(model, support_data_list, target_class, config, config.device, logger)
            current_features, _ = extract_dataset_features(
                model, plant_dataset, plant_test_indices, config.device, layer_name=layer_name
            )

            per_shot_features[shot] = current_features
            results[class_name][shot] = compute_trajectory_metrics(
                zero_shot_test_features,
                current_features,
                plant_test_labels,
                target_class,
                human_centroids,
            )

        plot_few_shot_trajectory_umap(
            zero_shot_bundle['human_features'],
            zero_shot_bundle['human_labels'],
            per_shot_features,
            plant_test_labels,
            target_class,
            output_dir,
            layer_name,
        )

    log_trajectory_metrics(logger, results)
    save_json(results, os.path.join(output_dir, f'fewshot_trajectory_metrics_{layer_name}.json'))
    save_trajectory_metrics_csv(results, output_dir, layer_name)
    return results
