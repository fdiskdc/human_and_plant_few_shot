"""
Few-shot Analysis Export Helpers

This module contains helper functions for exporting analysis data in tidy CSV/JSON format
for consumption by R visualization scripts.

Key design principles:
- All UMAP point CSVs have consistent schema: sample_id, species, class_name, class_idx,
  umap_x, umap_y, is_synthetic, point_role, source_group, shot, reference_type, target_class_name
- Real and synthetic points are distinguished via is_synthetic flag
- Colors are fixed per the global color specification
"""

import csv
import os
from datetime import datetime

import numpy as np

from utils.fewshot_analysis_constants import (
    TARGET_CLASSES, CLASS_NAMES, CLASS_NAME_MAP, SHOT_COUNTS,
    HIGH_CONTRAST_MOD_COLORS, HIGH_CONTRAST_SPECIES_COLORS, HIGH_CONTRAST_GEN3_COLORS,
    INTERPOLATION_PARAMS
)


def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def build_export_output_dir(base='output', prefix='zero_fewshot_export'):
    """Create and return a timestamped export output directory path."""
    ts = get_timestamp()
    out_dir = os.path.join(base, f'{prefix}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def export_zeroshot_alignment_metrics(metrics, output_dir, layer_name):
    """
    Export zero-shot alignment metrics as both CSV and JSON.
    
    Args:
        metrics: Dict of alignment metrics from calculate_alignment_metrics
        output_dir: Output directory path
        layer_name: Layer name for file naming
    
    Returns:
        List of exported file paths
    """
    import json
    
    exported = []
    
    # Export JSON (full metrics with 'overall' key)
    json_path = os.path.join(output_dir, f'zeroshot_alignment_metrics_{layer_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    exported.append(json_path)
    
    # Export CSV (tidy format, per-class only)
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
    exported.append(csv_path)
    
    return exported


def export_fewshot_trajectory_metrics(results, output_dir, layer_name):
    """
    Export few-shot trajectory metrics as both CSV and JSON.
    
    Args:
        results: Dict of trajectory results from run_few_shot_trajectory_analysis
        output_dir: Output directory path
        layer_name: Layer name for file naming
    
    Returns:
        List of exported file paths
    """
    import json
    
    exported = []
    
    # Export JSON (full nested structure)
    json_path = os.path.join(output_dir, f'fewshot_trajectory_metrics_{layer_name}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    exported.append(json_path)
    
    # Export CSV (tidy format)
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
    exported.append(csv_path)
    
    return exported


def export_sample_count_summary(human_labels, plant_labels, output_dir, layer_name):
    """Export sample count summary CSV."""
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


def export_zeroshot_joint_umap_points(embedding, filtered_labels, filtered_species, source_groups,
                                      output_dir, layer_name, is_synthetic=None,
                                      target_modification=None):
    """
    Export zero-shot joint UMAP points as tidy CSV.
    
    Consistent schema:
    - sample_id: Unique identifier
    - species: Human, Plant, Gen3, or Synthetic
    - class_name: Y, m5C, m6A
    - class_idx: Numeric class index
    - umap_x, umap_y: UMAP coordinates
    - is_synthetic: 0 or 1
    - point_role: 'real' or 'synthetic'
    - source_group: e.g., 'Human_m6A', 'Plant_m6A'
    - shot: NA for zero-shot (保留shot维度便于一致性)
    - reference_type: 'ref' for human, 'test' for plant
    - target_class_name: Target modification class
    """
    mod_str = f'_{target_modification}' if target_modification else ''
    csv_path = os.path.join(output_dir, f'zeroshot_joint_umap_points_{layer_name}{mod_str}.csv')
    n_points = embedding.shape[0]
    
    if is_synthetic is None:
        is_synthetic = np.array([False] * n_points)
    
    # Determine reference_type and target_class_name
    reference_types = []
    target_class_names = []
    for sg in source_groups:
        if sg.startswith('Human_'):
            reference_types.append('ref')
            target_class_names.append(sg.split('_', 1)[-1] if '_' in sg else 'Unknown')
        elif sg.startswith('Plant_'):
            reference_types.append('test')
            target_class_names.append(sg.split('_', 1)[-1] if '_' in sg else 'Unknown')
        else:
            reference_types.append('unknown')
            target_class_names.append('Unknown')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y',
            'is_synthetic', 'point_role', 'source_group', 'shot', 'reference_type', 'target_class_name'
        ])
        for i in range(n_points):
            # Derive class info from labels
            class_name = 'Unknown'
            class_idx = -1
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if filtered_labels[i, ci] == 1:
                    class_name = cn
                    class_idx = ci
                    break
            
            writer.writerow([
                i,
                filtered_species[i],
                class_name,
                class_idx,
                float(embedding[i, 0]),
                float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                'synthetic' if is_synthetic[i] else 'real',
                source_groups[i] if i < len(source_groups) else 'Unknown',
                'NA',  # shot not applicable for zero-shot
                reference_types[i] if i < len(reference_types) else 'unknown',
                target_class_names[i] if i < len(target_class_names) else 'Unknown'
            ])
    return csv_path


def export_fewshot_trajectory_umap_points(
    human_embedding, human_ref_labels,
    shot_embeddings, plant_ref_labels,
    target_class, output_dir, layer_name,
    human_synthetic_embedding=None, human_synthetic_groups=None,
    shot_synthetic_embeddings=None, shot_synthetic_groups=None
):
    """
    Export few-shot trajectory UMAP points as tidy CSV.
    
    Consistent schema includes:
    - sample_id, species, class_name, class_idx, umap_x, umap_y
    - is_synthetic, point_role, source_group
    - shot: 0, 1, 5, 10
    - reference_type: 'ref' for human, 'test' for plant
    - target_class_name: Current target class
    """
    class_name = CLASS_NAME_MAP[target_class]
    csv_path = os.path.join(output_dir, f'fewshot_trajectory_umap_points_{class_name}_{layer_name}.csv')
    
    rows = []
    sample_counter = 0
    
    # Human reference points (real)
    n_human = human_embedding.shape[0]
    for i in range(n_human):
        rows.append({
            'sample_id': sample_counter,
            'species': 'Human',
            'class_name': class_name,
            'class_idx': target_class,
            'umap_x': float(human_embedding[i, 0]),
            'umap_y': float(human_embedding[i, 1]),
            'is_synthetic': '0',
            'point_role': 'real',
            'source_group': f'Human_{class_name}',
            'shot': 'NA',
            'reference_type': 'ref',
            'target_class_name': class_name
        })
        sample_counter += 1
    
    # Human synthetic points
    if human_synthetic_embedding is not None and len(human_synthetic_embedding) > 0:
        for i in range(len(human_synthetic_embedding)):
            rows.append({
                'sample_id': sample_counter,
                'species': 'Human',
                'class_name': class_name,
                'class_idx': target_class,
                'umap_x': float(human_synthetic_embedding[i, 0]),
                'umap_y': float(human_synthetic_embedding[i, 1]),
                'is_synthetic': '1',
                'point_role': 'synthetic',
                'source_group': human_synthetic_groups[i] if i < len(human_synthetic_groups) else f'Human_{class_name}',
                'shot': 'NA',
                'reference_type': 'ref',
                'target_class_name': class_name
            })
            sample_counter += 1
    
    # Plant test points per shot
    for shot in SHOT_COUNTS:
        emb = shot_embeddings[shot]
        n_plant = emb.shape[0]
        for i in range(n_plant):
            rows.append({
                'sample_id': sample_counter,
                'species': 'Plant',
                'class_name': class_name,
                'class_idx': target_class,
                'umap_x': float(emb[i, 0]),
                'umap_y': float(emb[i, 1]),
                'is_synthetic': '0',
                'point_role': 'real',
                'source_group': f'Plant_{class_name}_{shot}shot',
                'shot': str(shot),
                'reference_type': 'test',
                'target_class_name': class_name
            })
            sample_counter += 1
        
        # Plant synthetic points for this shot
        synthetic_emb = None if shot_synthetic_embeddings is None else shot_synthetic_embeddings.get(shot)
        synthetic_groups = None if shot_synthetic_groups is None else shot_synthetic_groups.get(shot)
        if synthetic_emb is not None and len(synthetic_emb) > 0:
            for i in range(len(synthetic_emb)):
                rows.append({
                    'sample_id': sample_counter,
                    'species': 'Plant',
                    'class_name': class_name,
                    'class_idx': target_class,
                    'umap_x': float(synthetic_emb[i, 0]),
                    'umap_y': float(synthetic_emb[i, 1]),
                    'is_synthetic': '1',
                    'point_role': 'synthetic',
                    'source_group': synthetic_groups[i] if (synthetic_groups is not None and i < len(synthetic_groups)) else f'Plant_{class_name}_{shot}shot',
                    'shot': str(shot),
                    'reference_type': 'test',
                    'target_class_name': class_name
                })
                sample_counter += 1
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    return csv_path


def export_gen3_alignment_metrics(metrics, output_dir, layer_name):
    """Export gen3 alignment metrics as both CSV and JSON."""
    import json
    
    exported = []
    
    # Export JSON
    json_path = os.path.join(output_dir, f'gen3_alignment_metrics_{layer_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    exported.append(json_path)
    
    # Export CSV
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
    exported.append(csv_path)
    
    return exported


def export_gen3_umap_points(embedding, filtered_labels, source_groups, is_synthetic,
                             output_dir, layer_name):
    """Export gen3-only UMAP points as tidy CSV."""
    csv_path = os.path.join(output_dir, f'gen3_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'class_idx', 'class_name', 'umap_x', 'umap_y',
            'is_synthetic', 'point_role', 'source_group', 'shot', 'reference_type', 'target_class_name'
        ])
        for i in range(n_points):
            # Derive class info from labels
            if i < len(filtered_labels):
                class_idx = int(np.argmax(filtered_labels[i]))
                class_name = CLASS_NAME_MAP.get(class_idx, 'Unknown')
            else:
                class_idx = -1
                class_name = 'Unknown'
            
            writer.writerow([
                i,
                class_idx,
                class_name,
                float(embedding[i, 0]),
                float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                'synthetic' if is_synthetic[i] else 'real',
                source_groups[i] if i < len(source_groups) else 'Unknown',
                'NA',
                'test',  # gen3 is treated as test data
                class_name
            ])
    return csv_path


def export_gen3_joint_umap_points(embedding, filtered_labels, filtered_species, source_groups,
                                  is_synthetic, output_dir, layer_name):
    """
    Export gen3/human joint UMAP points as tidy CSV.
    
    This uses the FULL human dataset (not sampled) for proper comparison.
    """
    csv_path = os.path.join(output_dir, f'gen3_joint_umap_points_{layer_name}.csv')
    n_points = embedding.shape[0]
    
    # Determine reference_type and target_class_name
    reference_types = []
    target_class_names = []
    for sg in source_groups:
        if sg.startswith('Human_'):
            reference_types.append('ref')
            target_class_names.append(sg.split('_', 1)[-1] if '_' in sg else 'Unknown')
        elif sg.startswith('Gen3_'):
            reference_types.append('test')
            target_class_names.append(sg.split('_', 1)[-1] if '_' in sg else 'Unknown')
        else:
            reference_types.append('unknown')
            target_class_names.append('Unknown')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y',
            'is_synthetic', 'point_role', 'source_group', 'shot', 'reference_type', 'target_class_name'
        ])
        for i in range(n_points):
            # Derive class info from labels
            class_name = 'Unknown'
            class_idx = -1
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if filtered_labels[i, ci] == 1:
                    class_name = cn
                    class_idx = ci
                    break
            
            writer.writerow([
                i,
                filtered_species[i],
                class_name,
                class_idx,
                float(embedding[i, 0]),
                float(embedding[i, 1]),
                '1' if is_synthetic[i] else '0',
                'synthetic' if is_synthetic[i] else 'real',
                source_groups[i] if i < len(source_groups) else 'Unknown',
                'NA',
                reference_types[i] if i < len(reference_types) else 'unknown',
                target_class_names[i] if i < len(target_class_names) else 'Unknown'
            ])
    return csv_path


def export_gen3_sample_count_summary(gen3_labels, output_dir, layer_name):
    """Export gen3 sample count summary CSV."""
    from dataset.gen3_zero import MOD_NAMES
    
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
    return csv_path


def export_metadata(output_dir, layer_name, config_info):
    """
    Export metadata JSON with information about the export run.
    
    Args:
        output_dir: Output directory
        layer_name: Layer name used
        config_info: Dict with config information
    """
    import json
    
    metadata = {
        'layer_name': layer_name,
        'timestamp': get_timestamp(),
        'target_classes': CLASS_NAMES,
        'shot_counts': list(SHOT_COUNTS),
        'color_scheme': {
            'mod_colors': {k: v for k, v in HIGH_CONTRAST_MOD_COLORS.items()},
            'species_colors': {k: v for k, v in HIGH_CONTRAST_SPECIES_COLORS.items()},
            'gen3_colors': {k: v for k, v in HIGH_CONTRAST_GEN3_COLORS.items()},
        },
        'config_info': config_info
    }
    
    json_path = os.path.join(output_dir, f'export_metadata_{layer_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return json_path
