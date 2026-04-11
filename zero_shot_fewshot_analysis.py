"""
Zero-shot Feature Alignment & Few-shot Trajectory Analysis

This script performs two experiment groups:
1. Zero-shot feature alignment between human and plant samples.
2. Few-shot trajectory analysis using the existing independent binary protocol
   (Y / m5C / m6A, one-vs-rest on plant classes).

Outputs are written under output/zero_fewshot_analysis as both PNG and PDF
whenever figures are generated.
"""

import argparse
import copy
import json
import os
import random
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap
from matplotlib.lines import Line2D
from prettytable import PrettyTable
from scipy.spatial.distance import cdist
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from dataset.human import Mer100Dataset
from dataset.plant_single import PlantSingleDataset
from model.main_model import RNA_ClassQuery_Model
from utils import load_config, setup_logging

warnings.filterwarnings('ignore')

# Keep the runtime GPU behavior consistent with the rest of the repo.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

TARGET_CLASSES = [5, 8, 9]
CLASS_NAMES = ['Y', 'm5C', 'm6A']
CLASS_NAME_MAP = dict(zip(TARGET_CLASSES, CLASS_NAMES))
SHOT_COUNTS = [0, 1, 5, 10]

MORANDI_CLASS_COLORS = {
    'Y': '#B58A83',
    'm5C': '#9AAA91',
    'm6A': '#8EA3B0',
}
MORANDI_SPECIES_COLORS = {
    'Human': '#8E8A84',
    'Plant': '#A69C87',
}
MORANDI_NEUTRAL = '#C7C0B7'
MORANDI_GRID = '#E7E0D8'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, base_path, dpi=300):
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{base_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def apply_plot_style(ax):
    ax.grid(True, color=MORANDI_GRID, linewidth=0.7, alpha=0.8)
    ax.set_facecolor('#FBF8F3')
    for spine in ax.spines.values():
        spine.set_color('#D7CFC4')
    ax.tick_params(colors='#6E675F')


def stable_mean(values):
    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', {})

    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=model_cfg.get('cnn_hidden_dim', 128),
        cnn_kernel_sizes=tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7])),
        cnn_dropout=model_cfg.get('cnn_dropout', 0.1),
        gcn_hidden_dim=model_cfg.get('gcn_hidden_dim', 256),
        gcn_out_channels=model_cfg.get('gcn_out_channels', 256),
        gcn_num_layers=model_cfg.get('gcn_num_layers', 3),
        gcn_dropout=model_cfg.get('gcn_dropout', 0.3),
        num_classes=model_cfg.get('num_classes', 12),
        num_attn_heads=model_cfg.get('num_attn_heads', 8),
        attn_dropout=model_cfg.get('attn_dropout', 0.1),
        use_simple_pooling=model_cfg.get('use_simple_pooling', False),
        use_hierarchical=model_cfg.get('use_hierarchical', True),
        use_layer_norm=model_cfg.get('use_layer_norm', True),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


class FeatureExtractor:
    """
    Extract graph-level or class-aware features from intermediate layers.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.features = {}
        self.hooks = []
        self.attention_features = None

    def register_hooks(self):
        def tensor_hook(name):
            def _hook(_module, _inputs, output):
                self.features[name] = output.detach()
            return _hook

        def attention_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                self.attention_features = output[0].detach()
            else:
                self.attention_features = output.detach()

        self.hooks.append(self.model.cnn_block.register_forward_hook(tensor_hook('cnn_output')))
        self.hooks.append(self.model.gcn_block.register_forward_hook(tensor_hook('gcn_output')))

        if hasattr(self.model.class_query_head, 'mha_12'):
            self.hooks.append(
                self.model.class_query_head.mha_12.register_forward_hook(attention_hook)
            )
        elif hasattr(self.model.class_query_head, 'cross_attention'):
            self.hooks.append(
                self.model.class_query_head.cross_attention.register_forward_hook(attention_hook)
            )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, data_loader, layer_name='gcn_output'):
        self.model.eval()
        self.register_hooks()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {layer_name}", leave=False):
                batch = batch.to(self.device)
                self.features = {}
                self.attention_features = None

                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                if isinstance(outputs, tuple):
                    logits_12 = outputs[0]
                else:
                    logits_12 = outputs

                if layer_name == 'attention_output':
                    if self.attention_features is None:
                        raise RuntimeError("Attention features were not captured by hooks.")
                    graph_features = self.attention_features.mean(dim=1)
                else:
                    if layer_name not in self.features:
                        raise RuntimeError(f"Layer '{layer_name}' was not captured by hooks.")
                    node_features = self.features[layer_name]
                    batch_size = int(batch.batch.max().item()) + 1
                    pooled = []
                    for b in range(batch_size):
                        mask = batch.batch == b
                        pooled.append(node_features[mask].mean(dim=0))
                    graph_features = torch.stack(pooled, dim=0)

                all_features.append(graph_features.cpu().numpy())
                all_labels.append(batch.y.detach().cpu().numpy())

                # Touch logits to keep forward contract explicit and validated.
                _ = logits_12

        self.remove_hooks()
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def extract_dataset_features(model, dataset, indices, device, batch_size=64, layer_name='gcn_output'):
    loader = PyGDataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    extractor = FeatureExtractor(model, device)
    return extractor.extract_features(loader, layer_name=layer_name)


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
    labels = []
    for data in data_list:
        labels.append(float(data.y[0, target_class].item() == 1.0))
    return torch.tensor(labels, dtype=torch.float32)


def fine_tune_binary_model(model, support_data_list, target_class, config, device, logger):
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


def reduce_umap(features, n_neighbors=20, min_dist=0.18, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def get_class_mask(labels, class_idx):
    return labels[:, class_idx] == 1


def build_joint_zero_shot_umap(human_features, human_labels, plant_features, plant_labels, output_dir, layer_name):
    combined_features = np.vstack([human_features, plant_features])
    combined_labels = np.vstack([human_labels, plant_labels])
    species = np.array(['Human'] * len(human_features) + ['Plant'] * len(plant_features))

    target_mask = np.any(combined_labels[:, TARGET_CLASSES] == 1, axis=1)
    filtered_features = combined_features[target_mask]
    filtered_labels = combined_labels[target_mask]
    filtered_species = species[target_mask]
    embedding = reduce_umap(filtered_features)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#FBF8F3')

    ax = axes[0]
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = filtered_labels[:, class_idx] == 1
        if np.any(mask):
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=24,
                alpha=0.72,
                c=MORANDI_CLASS_COLORS[class_name],
                label=class_name,
                edgecolors='none',
            )
    ax.set_title('Joint UMAP by Class')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)
    ax.legend(frameon=False)

    ax = axes[1]
    for species_name in ['Human', 'Plant']:
        mask = filtered_species == species_name
        if np.any(mask):
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=24,
                alpha=0.72,
                c=MORANDI_SPECIES_COLORS[species_name],
                label=species_name,
                edgecolors='none',
            )
    ax.set_title('Joint UMAP by Species')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    apply_plot_style(ax)
    ax.legend(frameon=False)

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
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            class_mask = filtered_labels[:, class_idx] == 1
            if np.any(class_mask):
                ax.scatter(
                    embedding[class_mask, 0],
                    embedding[class_mask, 1],
                    s=24,
                    alpha=0.72,
                    c=MORANDI_CLASS_COLORS[class_name],
                    label=class_name,
                    edgecolors='none',
                )
        ax.set_title(f'{species_name.capitalize()} UMAP by Class')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)
        ax.legend(frameon=False)
        save_figure(fig, os.path.join(output_dir, f'zeroshot_{species_name}_umap_{layer_name}'))


def calculate_alignment_metrics(human_features, human_labels, plant_features, plant_labels):
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


def prepare_datasets(config):
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
    rng = np.random.RandomState(random_seed)
    indices = []
    y12 = human_dataset.y_12class
    for class_idx in TARGET_CLASSES:
        class_pos = np.where(y12[:, class_idx] == 1)[0]
        if len(class_pos) > max_per_class:
            class_pos = rng.choice(class_pos, size=max_per_class, replace=False)
        indices.extend(class_pos.tolist())
    return sorted(set(indices))


def split_plant_indices(plant_dataset, random_seed, train_fraction=0.1):
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
    centroids = {}
    for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
        mask = get_class_mask(human_labels, class_idx)
        class_features = human_features[mask]
        if len(class_features) > 0:
            centroids[class_name] = class_features.mean(axis=0)
    return centroids


def compute_class_compactness(features, labels, class_idx):
    mask = get_class_mask(labels, class_idx)
    class_features = features[mask]
    if len(class_features) == 0:
        return float('nan'), None
    centroid = class_features.mean(axis=0)
    compactness = float(np.mean(np.linalg.norm(class_features - centroid, axis=1)))
    return compactness, centroid


def compute_class_separation(features, labels, class_idx):
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


def convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    return obj


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(convert_for_json(data), f, indent=2)


def plot_few_shot_trajectory_umap(human_features, human_labels, per_shot_features, plant_labels, target_class, output_dir, layer_name):
    class_name = CLASS_NAME_MAP[target_class]
    figure_name = f'fewshot_trajectory_{class_name}_{layer_name}'

    human_mask = np.any(human_labels[:, TARGET_CLASSES] == 1, axis=1)
    plant_mask = np.any(plant_labels[:, TARGET_CLASSES] == 1, axis=1)
    human_ref = human_features[human_mask]
    human_ref_labels = human_labels[human_mask]

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
    plant_ref_labels = plant_labels[plant_mask]

    fig, axes = plt.subplots(1, len(SHOT_COUNTS), figsize=(6 * len(SHOT_COUNTS), 5.8), facecolor='#FBF8F3')

    for ax, shot in zip(axes, SHOT_COUNTS):
        for class_idx, current_class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            human_class_mask = human_ref_labels[:, class_idx] == 1
            if np.any(human_class_mask):
                ax.scatter(
                    human_embedding[human_class_mask, 0],
                    human_embedding[human_class_mask, 1],
                    s=18,
                    alpha=0.26,
                    marker='o',
                    c=MORANDI_CLASS_COLORS[current_class_name],
                    edgecolors='none',
                )

            plant_class_mask = plant_ref_labels[:, class_idx] == 1
            if np.any(plant_class_mask):
                ax.scatter(
                    shot_embeddings[shot][plant_class_mask, 0],
                    shot_embeddings[shot][plant_class_mask, 1],
                    s=28 if class_idx == target_class else 22,
                    alpha=0.80 if class_idx == target_class else 0.42,
                    marker='^',
                    c=MORANDI_CLASS_COLORS[current_class_name],
                    edgecolors='white' if class_idx == target_class else 'none',
                    linewidths=0.4,
                )

        ax.set_title(f'{shot}-shot')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        apply_plot_style(ax)

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='Human reference', markerfacecolor=MORANDI_NEUTRAL, markersize=7, alpha=0.7),
        Line2D([0], [0], marker='^', color='w', label='Plant test', markerfacecolor=MORANDI_NEUTRAL, markersize=8, alpha=0.9),
    ]
    for class_name_iter in CLASS_NAMES:
        legend_items.append(
            Line2D([0], [0], marker='s', color='w', label=class_name_iter,
                   markerfacecolor=MORANDI_CLASS_COLORS[class_name_iter], markersize=8)
        )
    axes[0].legend(handles=legend_items, frameon=False, loc='best')

    fig.suptitle(
        f'Few-shot Trajectory vs Human Reference: {class_name} ({layer_name})',
        y=1.03,
        fontsize=14,
    )
    save_figure(fig, os.path.join(output_dir, figure_name))


def log_trajectory_metrics(logger, results):
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


def run_zero_shot_analysis(config, checkpoint_path, output_dir, layer_name, logger):
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
    return {
        'metrics': metrics,
        'human_features': human_features,
        'human_labels': human_labels,
        'plant_features': plant_features,
        'plant_labels': plant_labels,
    }


def run_few_shot_trajectory_analysis(config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle):
    logger.info("\n" + "=" * 80)
    logger.info(f"FEW-SHOT TRAJECTORY ANALYSIS ({layer_name})")
    logger.info("=" * 80)

    _, plant_dataset = prepare_datasets(config)
    plant_train_indices, plant_test_indices = split_plant_indices(plant_dataset, config.random_seed)

    logger.info(f"Plant train pool: {len(plant_train_indices)}")
    logger.info(f"Plant test set: {len(plant_test_indices)}")

    human_centroids = compute_human_reference_centroids(
        zero_shot_bundle['human_features'], zero_shot_bundle['human_labels']
    )

    baseline_model, _ = load_model_from_checkpoint(checkpoint_path, config.device)
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
            model, _ = load_model_from_checkpoint(checkpoint_path, config.device)

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
    return results


def main(config_path='json/plant_single.json', checkpoint_path=None, layer_name='gcn_output'):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    output_dir = ensure_dir(os.path.join('output', 'zero_fewshot_analysis'))
    Config.output_dir = output_dir

    logger = setup_logging(Config.log_dir, 'zero_fewshot_analysis')
    logger.info("\n" + "=" * 80)
    logger.info("ZERO-SHOT FEATURE ALIGNMENT & FEW-SHOT TRAJECTORY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Feature layer: {layer_name}")

    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)

    if checkpoint_path is None:
        checkpoint_path = 'logs/old/rna_classification_20260129_164810/checkpoints/epoch_030.pt'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    zero_shot_bundle = run_zero_shot_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger
    )
    few_shot_metrics = run_few_shot_trajectory_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle
    )

    logger.info("\nAnalysis complete.")
    logger.info(f"Results saved to {output_dir}")
    return zero_shot_bundle, few_shot_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot feature alignment and few-shot trajectory analysis'
    )
    parser.add_argument('--config', type=str, default='json/plant_single.json')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='logs/old/rna_classification_20260129_164810/checkpoints/epoch_030.pt'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='gcn_output',
        choices=['cnn_output', 'gcn_output', 'attention_output'],
        help='Intermediate feature representation used for analysis.',
    )

    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint, layer_name=args.layer)
