"""
Zero-shot & Few-shot Data Export Pipeline

This script is a data-only export pipeline that extracts features, computes metrics,
generates UMAP coordinates, and produces synthetic interpolation points - all 
exported to structured CSV/JSON files for R visualization.

Key differences from zero_shot_fewshot_analysis.py:
- NO matplotlib figure generation
- Exports ALL intermediate data to output directory
- Generates consistent CSV schemas for R consumption
- Uses existing analysis modules but bypasses their plotting functions

Usage:
    python zero_shot_fewshot_extract_only.py --config json/plant_single.json --layer attention_output
    conda run -n learn python zero_shot_fewshot_extract_only.py --config json/plant_single.json
"""

import argparse
import copy
import os
import random
import warnings

import numpy as np
import torch

from utils import load_config, setup_logging
from utils.fewshot_analysis_constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH
from utils.fewshot_analysis_utils import build_output_dir, save_json
from utils.fewshot_analysis_features import load_model_from_checkpoint
from utils.fewshot_analysis_zeroshot import (
    run_zero_shot_analysis, prepare_datasets, sample_human_reference_indices,
    reduce_umap, generate_synthetic_points_by_group, filter_by_modification,
    calculate_alignment_metrics, save_alignment_metrics_csv, save_sample_count_summary
)
from utils.fewshot_analysis_fewshot import (
    split_plant_indices, compute_human_reference_centroids,
    compute_trajectory_metrics, sample_binary_support_set,
    prepare_binary_labels, fine_tune_binary_model,
    extract_dataset_features, save_trajectory_metrics_csv
)
from utils.fewshot_analysis_gen3 import (
    run_gen3_analysis, save_gen3_sample_count_summary, 
    calculate_gen3_alignment_metrics, save_gen3_alignment_metrics_csv
)
from dataset.gen3_zero import MOD_NAMES

warnings.filterwarnings('ignore')

# Keep the runtime GPU behavior consistent with the rest of the repo.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def export_zeroshot_data(config, checkpoint_path, output_dir, layer_name, logger,
                         model=None, checkpoint=None):
    """
    Export zero-shot analysis data (features, metrics, UMAP points).
    
    Reuses logic from utils/fewshot_analysis_zeroshot.py but exports 
    all intermediate data instead of generating plots.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"EXPORTING ZERO-SHOT DATA ({layer_name})")
    logger.info("=" * 80)

    human_dataset, plant_dataset = prepare_datasets(config)
    if model is None or checkpoint is None:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, config.device)

    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    # Sample human reference indices for zero-shot analysis
    human_indices = sample_human_reference_indices(human_dataset, config.random_seed)
    plant_indices = list(range(plant_dataset.num_plant))

    human_features, human_labels = extract_dataset_features(
        model, human_dataset, human_indices, config.device, layer_name=layer_name
    )
    plant_features, plant_labels = extract_dataset_features(
        model, plant_dataset, plant_indices, config.device, layer_name=layer_name
    )

    # Calculate and save alignment metrics
    metrics = calculate_alignment_metrics(human_features, human_labels, plant_features, plant_labels)
    save_json(metrics, os.path.join(output_dir, f'zeroshot_alignment_metrics_{layer_name}.json'))
    save_alignment_metrics_csv(metrics, output_dir, layer_name)
    save_sample_count_summary(human_labels, plant_labels, output_dir, layer_name)
    
    logger.info("Zero-shot alignment metrics exported.")

    # Export joint UMAP with synthetic points
    export_joint_umap(
        human_features, human_labels, plant_features, plant_labels,
        output_dir, layer_name, logger, target_modification=None
    )
    
    # Export per-modification UMAPs
    from utils.fewshot_analysis_constants import CLASS_NAMES
    for mod in CLASS_NAMES:
        logger.info(f"Exporting UMAP for {mod}...")
        export_joint_umap(
            human_features, human_labels, plant_features, plant_labels,
            output_dir, layer_name, logger, target_modification=mod
        )

    return {
        'human_features': human_features,
        'human_labels': human_labels,
        'plant_features': plant_features,
        'plant_labels': plant_labels,
        'plant_dataset': plant_dataset,
        'human_dataset': human_dataset,
        'human_indices': human_indices,
        'plant_indices': plant_indices,
    }


def export_joint_umap(human_features, human_labels, plant_features, plant_labels,
                      output_dir, layer_name, logger, target_modification=None):
    """
    Export joint UMAP coordinates and synthetic points as CSV.
    
    Generates:
    - zeroshot_joint_umap_points_{layer_name}[_mod].csv
    """
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES, INTERPOLATION_PARAMS
    
    combined_features = np.vstack([human_features, plant_features])
    combined_labels = np.vstack([human_labels, plant_labels])
    species = np.array(['Human'] * len(human_features) + ['Plant'] * len(plant_features))
    
    # Filter by target modification if specified
    filtered_features, filtered_labels, filtered_species, source_groups = filter_by_modification(
        combined_features, combined_labels, species, target_modification
    )
    
    if len(filtered_features) == 0:
        logger.warning(f"No samples found for target_modification={target_modification}")
        return
    
    # Generate UMAP
    embedding = reduce_umap(filtered_features)
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points for visual enhancement
    synthetic_emb = np.array([])
    synthetic_groups = np.array([])
    
    if len(filtered_features) > 0:
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
            # Extend filtered_species (Synthetic for synthetic points)
            filtered_species = np.concatenate([filtered_species, np.array(['Synthetic'] * len(synthetic_emb))])
            # Extend filtered_labels for synthetic points
            synthetic_labels = np.zeros((len(synthetic_emb), filtered_labels.shape[1]))
            for i, sg in enumerate(synthetic_groups):
                for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
                    if sg.endswith(f'_{class_name}'):
                        synthetic_labels[i, class_idx] = 1
                        break
            filtered_labels = np.vstack([filtered_labels, synthetic_labels])
    
    # Save joint UMAP points CSV
    mod_str = f'_{target_modification}' if target_modification else ''
    csv_path = os.path.join(output_dir, f'zeroshot_joint_umap_points_{layer_name}{mod_str}.csv')
    save_umap_points_csv(
        embedding, filtered_labels, filtered_species, source_groups,
        is_synthetic, output_dir, layer_name, target_modification=target_modification,
        extra_cols={'analysis_type': 'zeroshot'}
    )
    logger.info(f"Exported: {csv_path}")


def save_umap_points_csv(embedding, labels, species, source_groups, is_synthetic,
output_dir, layer_name, target_modification=None, extra_cols=None):
    """
    Save UMAP point coordinates as tidy CSV with consistent schema.
    
    Schema:
    - sample_id: unique integer ID
    - species: Human, Plant, Gen3, Synthetic
    - class_name: m6A, m5C, Y, Unknown
    - class_idx: integer class index (-1 for synthetic)
    - umap_x, umap_y: UMAP coordinates
    - is_synthetic: 0 or 1
    - point_role: real or synthetic
    - source_group: Human_m6A, Plant_m5C, etc.
    - analysis_type: zeroshot, fewshot, gen3 (optional)
    - shot: 0, 1, 5, 10, or NA (optional)
    - reference_type: ref, plant_test (optional)
    - target_class_name: for fewshot (optional)
    """
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES
    
    mod_str = f'_{target_modification}' if target_modification else ''
    
    # Determine filename based on context
    # NOTE: gen3_umap (standalone) vs gen3_joint_umap are differentiated by caller
    if extra_cols and extra_cols.get('analysis_type') == 'fewshot':
        target_class_name = extra_cols.get('target_class_name', 'unknown')
        filename = f'fewshot_trajectory_umap_points_{target_class_name}_{layer_name}.csv'
    elif extra_cols and extra_cols.get('is_gen3_joint', False):
        # gen3 joint UMAP with human uses is_gen3_joint flag
        filename = f'gen3_joint_umap_points_{layer_name}.csv'
    elif extra_cols and extra_cols.get('analysis_type') == 'gen3' and not extra_cols.get('is_gen3_joint', False):
        # standalone gen3 UMAP (no human)
        filename = f'gen3_umap_points_{layer_name}.csv'
    else:
        filename = f'zeroshot_joint_umap_points_{layer_name}{mod_str}.csv'
    
    csv_path = os.path.join(output_dir, filename)
    n_points = embedding.shape[0]
    
    point_role = np.array(['real' if not syn else 'synthetic' for syn in is_synthetic])
    
    class_names = []
    class_indices = []
    for i in range(n_points):
        assigned = False
        for class_idx, class_name in zip(TARGET_CLASSES, CLASS_NAMES):
            if labels[i, class_idx] == 1:
                class_names.append(class_name)
                class_indices.append(class_idx)
                assigned = True
                break
        if not assigned:
            class_names.append('Unknown')
            class_indices.append(-1)
    
    # Build rows with consistent schema
    rows = []
    for i in range(n_points):
        row = {
            'sample_id': i,
            'species': str(species[i]) if hasattr(species[i], '__str__') else species[i],
            'class_name': class_names[i],
            'class_idx': class_indices[i],
            'umap_x': float(embedding[i, 0]),
            'umap_y': float(embedding[i, 1]),
            'is_synthetic': '1' if is_synthetic[i] else '0',
            'point_role': point_role[i],
            'source_group': str(source_groups[i]) if i < len(source_groups) else 'Unknown',
        }
        # Add optional columns if provided
        if extra_cols:
            for key, val in extra_cols.items():
                # Skip internal flags used for routing
                if key in ('target_class_name', 'is_gen3_joint'):
                    continue
                # Per-point columns should be indexed row-by-row instead of stringifying whole arrays.
                if isinstance(val, np.ndarray):
                    row[key] = str(val[i]) if len(val) == n_points else str(val.tolist())
                elif isinstance(val, (list, tuple)):
                    row[key] = str(val[i]) if len(val) == n_points else ','.join(str(v) for v in val)
                else:
                    row[key] = val
        rows.append(row)
    
    # Write CSV
    import csv as csv_module
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(rows[0].keys()) if rows else [
            'sample_id', 'species', 'class_name', 'class_idx', 'umap_x', 'umap_y',
            'is_synthetic', 'point_role', 'source_group'
        ]
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return csv_path


def export_fewshot_data(config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle,
                       baseline_model=None, baseline_state_dict=None):
    """
    Export few-shot trajectory analysis data.
    
    Reuses logic from utils/fewshot_analysis_fewshot.py but exports
    all intermediate data instead of generating plots.
    """
    from collections import defaultdict
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES, CLASS_NAME_MAP, SHOT_COUNTS, INTERPOLATION_PARAMS
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    logger.info("\n" + "=" * 80)
    logger.info(f"EXPORTING FEW-SHOT DATA ({layer_name})")
    logger.info("=" * 80)

    plant_dataset = zero_shot_bundle.get('plant_dataset')
    if plant_dataset is None:
        _, plant_dataset = prepare_datasets(config)
    plant_train_indices, plant_test_indices = split_plant_indices(plant_dataset, config.random_seed)

    logger.info(f"Plant train pool: {len(plant_train_indices)}")
    logger.info(f"Plant test set: {len(plant_test_indices)}")

    human_features = zero_shot_bundle['human_features']
    human_labels = zero_shot_bundle['human_labels']
    
    human_centroids = compute_human_reference_centroids(human_features, human_labels)

    if baseline_model is None:
        baseline_model, _ = load_model_from_checkpoint(checkpoint_path, config.device)
    baseline_model.eval()
    if baseline_state_dict is None:
        baseline_state_dict = copy.deepcopy(baseline_model.state_dict())

    zero_shot_test_features, plant_test_labels = extract_dataset_features(
        baseline_model, plant_dataset, plant_test_indices, config.device, layer_name=layer_name
    )

    results = defaultdict(dict)
    all_per_shot_features = {}

    for target_class in TARGET_CLASSES:
        class_name = CLASS_NAME_MAP[target_class]
        logger.info(f"\nProcessing few-shot class: {class_name} ({target_class})")

        per_shot_features = {}
        per_shot_features[0] = zero_shot_test_features
        results[class_name][0] = compute_trajectory_metrics(
            zero_shot_test_features,
            zero_shot_test_features,
            plant_test_labels,
            target_class,
            human_centroids,
        )

        for shot in SHOT_COUNTS:
            if shot == 0:
                continue  # Already computed above
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

        # Export trajectory UMAP for this class
        export_fewshot_trajectory_umap(
            human_features, human_labels, per_shot_features, plant_test_labels,
            target_class, output_dir, layer_name, logger
        )
        
        all_per_shot_features[class_name] = per_shot_features

    # Save trajectory metrics
    save_json(results, os.path.join(output_dir, f'fewshot_trajectory_metrics_{layer_name}.json'))
    save_trajectory_metrics_csv(results, output_dir, layer_name)
    logger.info("Few-shot trajectory metrics exported.")
    
    return results


def export_fewshot_trajectory_umap(human_features, human_labels, per_shot_features, 
                                   plant_labels, target_class, output_dir, layer_name, logger):
    """
    Export few-shot trajectory UMAP with synthetic points.
    """
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES, CLASS_NAME_MAP, SHOT_COUNTS, INTERPOLATION_PARAMS
    
    class_name = CLASS_NAME_MAP[target_class]

    human_mask = human_labels[:, target_class] == 1
    plant_mask = plant_labels[:, target_class] == 1
    human_ref = human_features[human_mask]
    human_ref_labels = human_labels[human_mask]
    plant_ref_labels = plant_labels[plant_mask]

    if len(human_ref) == 0 or len(plant_ref_labels) == 0:
        logger.warning(f"No samples for class {class_name}, skipping trajectory UMAP export.")
        return

    # Stack all features for joint UMAP
    all_feature_blocks = [human_ref]
    shot_sizes = []
    for shot in SHOT_COUNTS:
        if shot in per_shot_features:
            block = per_shot_features[shot][plant_mask]
            all_feature_blocks.append(block)
            shot_sizes.append(len(block))

    embedded = reduce_umap(np.vstack(all_feature_blocks))
    human_size = len(human_ref)
    
    shot_embeddings = {}
    cursor = human_size
    for shot, size in zip(SHOT_COUNTS, shot_sizes):
        if shot in per_shot_features:
            shot_embeddings[shot] = embedded[cursor:cursor + size]
            cursor += size
    human_embedding = embedded[:human_size]

    # Create source groups
    human_source_groups = np.array([f'Human_{class_name}'] * len(human_embedding))
    shot_source_groups = {
        shot: np.array([f'Plant_{class_name}_{shot}shot'] * len(shot_embeddings[shot]))
        for shot in SHOT_COUNTS if shot in shot_embeddings
    }

    # Generate synthetic points for human reference
    human_synthetic_embedding = np.array([])
    human_synthetic_groups = np.array([])
    if len(human_embedding) > 0:
        human_synthetic_embedding, human_synthetic_groups, _ = generate_synthetic_points_by_group(
            human_embedding, human_source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )

    # Generate synthetic points for plant (denser than human - "比 human 更密" strategy)
    shot_synthetic_embeddings = {}
    shot_synthetic_groups = {}
    plant_n_synthetic = max(
        INTERPOLATION_PARAMS['n_synthetic_per_point'] * 2,
        INTERPOLATION_PARAMS['n_synthetic_per_point'] + 8,
    )
    for shot in SHOT_COUNTS:
        if shot not in shot_embeddings:
            continue
        synthetic_emb, synthetic_groups, _ = generate_synthetic_points_by_group(
            shot_embeddings[shot], shot_source_groups[shot],
            n_synthetic=plant_n_synthetic,
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        shot_synthetic_embeddings[shot] = synthetic_emb
        shot_synthetic_groups[shot] = synthetic_groups

    # Build combined data for CSV export
    all_embeddings = [human_embedding]
    all_is_synthetic = [np.array([False] * len(human_embedding))]
    all_source_groups = [human_source_groups]
    all_species = [np.array(['Human'] * len(human_embedding))]
    all_shots = [np.array(['NA'] * len(human_embedding))]  # NA for human ref
    all_ref_types = [np.array(['ref'] * len(human_embedding))]
    all_labels = [human_ref_labels]
    
    if len(human_synthetic_embedding) > 0:
        all_embeddings.append(human_synthetic_embedding)
        all_is_synthetic.append(np.array([True] * len(human_synthetic_embedding)))
        all_source_groups.append(human_synthetic_groups)
        all_species.append(np.array(['Human'] * len(human_synthetic_embedding)))
        all_shots.append(np.array(['NA'] * len(human_synthetic_embedding)))
        all_ref_types.append(np.array(['synthetic'] * len(human_synthetic_embedding)))
        # Create labels for synthetic human points
        synth_human_labels = np.zeros((len(human_synthetic_embedding), human_ref_labels.shape[1]))
        for i, sg in enumerate(human_synthetic_groups):
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if sg.endswith(f'_{cn}'):
                    synth_human_labels[i, ci] = 1
                    break
        all_labels.append(synth_human_labels)

    for shot in SHOT_COUNTS:
        if shot not in shot_embeddings:
            continue
        all_embeddings.append(shot_embeddings[shot])
        all_is_synthetic.append(np.array([False] * len(shot_embeddings[shot])))
        all_source_groups.append(shot_source_groups[shot])
        all_species.append(np.array(['Plant'] * len(shot_embeddings[shot])))
        all_shots.append(np.array([str(shot)] * len(shot_embeddings[shot])))
        all_ref_types.append(np.array(['plant_test'] * len(shot_embeddings[shot])))
        # Create labels for plant test points
        plant_shot_labels = plant_ref_labels.copy()  # Use the filtered labels
        all_labels.append(plant_shot_labels)
        
        if len(shot_synthetic_embeddings.get(shot, [])) > 0:
            all_embeddings.append(shot_synthetic_embeddings[shot])
            all_is_synthetic.append(np.array([True] * len(shot_synthetic_embeddings[shot])))
            all_source_groups.append(shot_synthetic_groups[shot])
            all_species.append(np.array(['Plant'] * len(shot_synthetic_embeddings[shot])))
            all_shots.append(np.array([str(shot)] * len(shot_synthetic_embeddings[shot])))
            all_ref_types.append(np.array(['synthetic'] * len(shot_synthetic_embeddings[shot])))
            # Create labels for synthetic plant points
            synth_plant_labels = np.zeros((len(shot_synthetic_embeddings[shot]), plant_ref_labels.shape[1]))
            for i, sg in enumerate(shot_synthetic_groups[shot]):
                for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                    if sg.endswith(f'_{cn}'):
                        synth_plant_labels[i, ci] = 1
                        break
            all_labels.append(synth_plant_labels)

    # Concatenate all
    combined_embedding = np.vstack(all_embeddings)
    combined_is_synthetic = np.concatenate(all_is_synthetic)
    combined_source_groups = np.concatenate(all_source_groups)
    combined_species = np.concatenate(all_species)
    combined_shots = np.concatenate(all_shots)
    combined_ref_types = np.concatenate(all_ref_types)
    combined_labels = np.vstack(all_labels)

    # Save to CSV with extended schema
    csv_path = save_umap_points_csv(
        combined_embedding, combined_labels, combined_species, combined_source_groups,
        combined_is_synthetic, output_dir, layer_name,
        extra_cols={
            'analysis_type': 'fewshot',
            'shot': combined_shots,
            'reference_type': combined_ref_types,
            'target_class_name': class_name,
        }
    )
    logger.info(f"Exported few-shot trajectory UMAP: {csv_path}")


def export_gen3_data(config, checkpoint_path, output_dir, layer_name, logger,
                     model=None, checkpoint=None):
    """
    Export 3-generation (gen3) analysis data.
    
    IMPORTANT: Uses FULL human dataset for joint UMAP, not sampled reference.
    """
    from dataset.gen3_zero import Gen3ZeroDataset
    from dataset.human import Mer100Dataset
    
    logger.info("\n" + "=" * 80)
    logger.info(f"EXPORTING GEN3 DATA ({layer_name})")
    logger.info("=" * 80)

    if model is None or checkpoint is None:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, config.device)
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    # Load gen3 dataset
    gen3_dataset = Gen3ZeroDataset(mode='test', data_dir='npy')
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
    
    # Export gen3 standalone UMAP
    export_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger)

    # Load FULL human dataset (not sampled) for joint UMAP
    human_dataset = Mer100Dataset(
        mode='train',
        data_dir=config.data.human_data_dir,
        cache_dir=config.data.cache_dir,
        use_human3=True,
        use_cache=True,
    )
    logger.info(f"Human dataset loaded for Gen3 comparison (FULL): {len(human_dataset)} samples")

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

    # Export gen3/human joint UMAP with FULL human dataset
    export_gen3_joint_umap(
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


def export_gen3_umap(gen3_features, gen3_labels, output_dir, layer_name, logger):
    """Export standalone gen3 UMAP with synthetic points."""
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES, INTERPOLATION_PARAMS
    
    # Only use samples with at least one positive label
    pos_mask = np.any(gen3_labels != 0, axis=1)
    if not np.any(pos_mask):
        logger.warning("No positive samples found in gen3 data, skipping UMAP export.")
        return

    filtered_features = gen3_features[pos_mask]
    filtered_labels = gen3_labels[pos_mask]

    # Create source groups for gen3
    source_groups = []
    for i in range(len(filtered_labels)):
        primary_class_idx = int(np.argmax(filtered_labels[i]))
        if primary_class_idx < len(MOD_NAMES):
            primary_class_name = MOD_NAMES[primary_class_idx]
        else:
            primary_class_name = 'Unknown'
        source_groups.append(f"Gen3_{primary_class_name}")
    source_groups = np.array(source_groups)
    
    species = np.array(['Gen3'] * len(filtered_features))

    # Reduce dimensions with UMAP
    embedding = reduce_umap(filtered_features)
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points
    synthetic_emb = np.array([])
    synthetic_groups = np.array([])
    if len(filtered_features) > 0:
        synthetic_emb, synthetic_groups, _ = generate_synthetic_points_by_group(
            embedding, source_groups,
            n_synthetic=INTERPOLATION_PARAMS['n_synthetic_per_point'],
            jitter_strength=INTERPOLATION_PARAMS['jitter_strength']
        )
        
        if len(synthetic_emb) > 0:
            embedding = np.vstack([embedding, synthetic_emb])
            is_synthetic = np.array([False] * n_real + [True] * len(synthetic_emb))
            source_groups = np.concatenate([source_groups, synthetic_groups])
            species = np.concatenate([species, np.array(['Synthetic'] * len(synthetic_emb))])

    # Save UMAP points CSV
    csv_path = os.path.join(output_dir, f'gen3_umap_points_{layer_name}.csv')
    
    # Create labels for synthetic points
    if len(synthetic_emb) > 0:
        synth_labels = np.zeros((len(synthetic_emb), filtered_labels.shape[1]))
        for i, sg in enumerate(synthetic_groups):
            for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                if sg.endswith(f'_{cn}'):
                    synth_labels[i, ci] = 1
                    break
        combined_labels = np.vstack([filtered_labels, synth_labels])
    else:
        combined_labels = filtered_labels

    save_umap_points_csv(
        embedding, combined_labels, species, source_groups, is_synthetic,
        output_dir, layer_name, extra_cols={'analysis_type': 'gen3'}
    )
    logger.info(f"Exported gen3 UMAP: {csv_path}")


def export_gen3_joint_umap(human_features, human_labels, gen3_features, gen3_labels,
                           output_dir, layer_name, logger):
    """Export gen3/human joint UMAP with FULL human dataset."""
    from utils.fewshot_analysis_constants import TARGET_CLASSES, CLASS_NAMES, INTERPOLATION_PARAMS
    
    # Combine full human dataset with gen3
    combined_features = np.vstack([human_features, gen3_features])
    combined_labels = np.vstack([human_labels, gen3_labels])
    species = np.array(['Human'] * len(human_features) + ['Gen3'] * len(gen3_features))
    
    # Filter by target classes only
    filtered_features, filtered_labels, filtered_species, source_groups = filter_by_modification(
        combined_features, combined_labels, species, target_modification=None
    )
    
    if len(filtered_features) == 0:
        logger.warning("No samples found for gen3/human joint UMAP.")
        return

    # Generate UMAP
    embedding = reduce_umap(filtered_features)
    n_real = len(embedding)
    is_synthetic = np.array([False] * n_real)
    
    # Generate synthetic points
    synthetic_emb = np.array([])
    synthetic_groups = np.array([])
    if len(filtered_features) > 0:
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
            # Create labels for synthetic points
            synthetic_labels = np.zeros((len(synthetic_emb), filtered_labels.shape[1]))
            for i, sg in enumerate(synthetic_groups):
                for ci, cn in zip(TARGET_CLASSES, CLASS_NAMES):
                    if sg.endswith(f'_{cn}'):
                        synthetic_labels[i, ci] = 1
                        break
            filtered_labels = np.vstack([filtered_labels, synthetic_labels])
    
    # Save joint UMAP points CSV - use is_gen3_joint flag to differentiate from standalone gen3 UMAP
    csv_path = save_umap_points_csv(
        embedding, filtered_labels, filtered_species, source_groups, is_synthetic,
        output_dir, layer_name, extra_cols={'analysis_type': 'gen3', 'is_gen3_joint': True}
    )
    logger.info(f"Exported gen3/human joint UMAP: {csv_path}")


def main(config_path=DEFAULT_CONFIG_PATH, checkpoint_path=None, layer_name='gcn_output'):
    """
    Main export pipeline entry point.
    
    Runs all analysis and exports data to timestamped output directory.
    """
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    # Create output directory with export-specific prefix
    output_dir = build_output_dir(base='output', prefix='zero_fewshot_export')
    Config.output_dir = output_dir

    logger = setup_logging(Config.log_dir, 'zero_fewshot_extract_only')
    logger.info("\n" + "=" * 80)
    logger.info("ZERO-SHOT & FEW-SHOT DATA EXPORT PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Feature layer: {layer_name}")

    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    baseline_model, checkpoint = load_model_from_checkpoint(checkpoint_path, Config.device)
    baseline_model.eval()
    baseline_state_dict = copy.deepcopy(baseline_model.state_dict())
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    # Export zero-shot data
    zero_shot_bundle = export_zeroshot_data(
        Config, checkpoint_path, output_dir, layer_name, logger,
        model=baseline_model, checkpoint=checkpoint
    )

    # Export few-shot trajectory data
    few_shot_metrics = export_fewshot_data(
        Config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle,
        baseline_model=baseline_model, baseline_state_dict=baseline_state_dict
    )

    # Export gen3 data (uses FULL human dataset for joint UMAP)
    logger.info("\n" + "=" * 80)
    logger.info("Exporting 3-generation data...")
    logger.info("=" * 80)
    gen3_bundle = export_gen3_data(
        Config, checkpoint_path, output_dir, layer_name, logger,
        model=baseline_model, checkpoint=checkpoint
    )

    # Write metadata file
    metadata = {
        'layer_name': layer_name,
        'checkpoint_path': checkpoint_path,
        'config_path': config_path,
        'random_seed': Config.random_seed,
        'output_dir': output_dir,
        'export_timestamp': output_dir.split('_')[-1] if '_' in output_dir else 'unknown',
        'num_zeroshot_human_samples': len(zero_shot_bundle['human_indices']),
        'num_zeroshot_plant_samples': len(zero_shot_bundle['plant_indices']),
        'num_gen3_samples': len(gen3_bundle['gen3_features']),
        'num_full_human_samples': len(gen3_bundle['human_features']),
        'shot_counts': [0, 1, 5, 10],
        'target_classes': ['Y', 'm5C', 'm6A'],
    }
    save_json(metadata, os.path.join(output_dir, 'export_metadata.json'))

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All data exported to: {output_dir}")
    logger.info("Run R script to generate figures:")
    logger.info(f"  conda run -n learn-new Rscript plot_zero_fewshot_export.R --input_dir {output_dir} --output_dir {output_dir}")

    # Automatically call R script to generate figures
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING R VISUALIZATION")
    logger.info("=" * 80)
    
    r_script_path = os.path.join(os.path.dirname(__file__), 'plot_zero_fewshot_export.R')
    if not os.path.exists(r_script_path):
        logger.warning(f"R script not found at {r_script_path}, skipping R plots.")
    else:
        try:
            import subprocess
            abs_output_dir = os.path.abspath(output_dir)
            abs_r_script_path = os.path.abspath(r_script_path)
            
            cmd = ['conda', 'run', '-n', 'learn-new', 'Rscript', abs_r_script_path,
                   '--input_dir', abs_output_dir, '--output_dir', abs_output_dir]
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.warning(f"R script failed with return code {result.returncode}")
                if result.stderr:
                    logger.warning(f"R stderr: {result.stderr[:1000]}")
            else:
                logger.info("R visualization completed successfully.")
                if result.stdout:
                    for line in result.stdout.strip().split('\n')[:20]:  # Limit output
                        logger.info(f"  {line}")
        except FileNotFoundError:
            logger.warning("conda or Rscript not found in PATH. Skipping R plots.")
        except subprocess.TimeoutExpired:
            logger.warning("R script timed out after 600 seconds.")
        except Exception as e:
            logger.warning(f"Failed to run R script: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    return zero_shot_bundle, few_shot_metrics, gen3_bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export zero-shot and few-shot analysis data to CSV/JSON'
    )
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        '--layer',
        type=str,
        default='attention_output',
        choices=['cnn_output', 'gcn_output', 'attention_output'],
        help='Intermediate feature representation used for analysis.',
    )

    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        layer_name=args.layer,
    )
