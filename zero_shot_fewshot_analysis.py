"""
Zero-shot Feature Alignment & Few-shot Trajectory Analysis

This script is the main entry point for the few-shot analysis pipeline.
It performs:
1. Zero-shot feature alignment between human and plant samples.
2. Few-shot trajectory analysis using the existing independent binary protocol
   (Y / m5C / m6A, one-vs-rest on plant classes).
3. 3-generation (gen3) data visualization (always run alongside plant analysis).
4. Spatial motif analysis for plant and gen3 data (always run).

All analysis results are stored in the same timestamped output directory.

Outputs are written under output/zero_fewshot_analysis as both PNG and PDF
whenever figures are generated.

Usage:
    python zero_shot_fewshot_analysis.py --config json/plant_single.json --layer attention_output
    python zero_shot_fewshot_analysis.py --conda_env learn-new --spatial_motif_classes m6A m5C Y
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch

from utils import load_config, setup_logging
from utils.fewshot_analysis_constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH
from utils.fewshot_analysis_utils import build_output_dir, call_r_script
from utils.fewshot_analysis_features import load_model_from_checkpoint
from utils.fewshot_analysis_zeroshot import run_zero_shot_analysis, prepare_datasets
from utils.fewshot_analysis_fewshot import run_few_shot_trajectory_analysis
from utils.fewshot_analysis_gen3 import run_gen3_analysis
from dataset.gen3_zero import MOD_NAMES

# Try importing spatial motif module
try:
    from utils.fewshot_analysis_spatial_motif import run_spatial_motif_analysis
    HAS_SPATIAL_MOTIF = True
except ImportError:
    HAS_SPATIAL_MOTIF = False

warnings.filterwarnings('ignore')

# Keep the runtime GPU behavior consistent with the rest of the repo.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def main(config_path=DEFAULT_CONFIG_PATH, checkpoint_path=None, layer_name='gcn_output',
         spatial_motif_classes=None,
         n_clusters=3, pca_components=50, node_num=10, conda_env='learn-new'):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    output_dir = build_output_dir(base='output', prefix='zero_fewshot_analysis')
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
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    # Run standard zero-shot and few-shot analysis (plant data)
    zero_shot_bundle = run_zero_shot_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger
    )
    few_shot_metrics = run_few_shot_trajectory_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle
    )

    # Run 3-generation analysis (always run alongside plant analysis)
    logger.info("\n" + "=" * 80)
    logger.info("Running 3-generation data analysis...")
    logger.info("=" * 80)
    gen3_bundle = run_gen3_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger
    )

    # Run spatial motif analysis for both plant and gen3 data (always run)
    if not HAS_SPATIAL_MOTIF:
        logger.warning("Spatial motif module not available (logomaker or captum not installed).")
    else:
        model, _ = load_model_from_checkpoint(checkpoint_path, Config.device)

        # Determine target classes for spatial motif
        if spatial_motif_classes is None:
            target_classes = list(MOD_NAMES.items())
        else:
            target_classes = [(k, v) for k, v in MOD_NAMES.items() if v in spatial_motif_classes]

        # Run spatial motif on plant data
        logger.info("\n" + "=" * 80)
        logger.info("Running spatial motif analysis on plant data...")
        logger.info("=" * 80)
        _, plant_dataset = prepare_datasets(Config)
        run_spatial_motif_analysis(
            model, plant_dataset, 'plant', target_classes, output_dir, Config.device,
            n_clusters=n_clusters, pca_components=pca_components, node_num=node_num
        )

        # Run spatial motif on gen3 data
        logger.info("\n" + "=" * 80)
        logger.info("Running spatial motif analysis on gen3 data...")
        logger.info("=" * 80)
        run_spatial_motif_analysis(
            model, gen3_bundle['gen3_dataset'], 'gen3', target_classes, output_dir, Config.device,
            n_clusters=n_clusters, pca_components=pca_components, node_num=node_num
        )

    logger.info("\nAnalysis complete.")
    logger.info(f"Results saved to {output_dir}")

    # Try to call R plotting script
    call_r_script(input_dir=output_dir, output_dir=output_dir, logger=logger, conda_env=conda_env)

    return zero_shot_bundle, few_shot_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot feature alignment and few-shot trajectory analysis'
    )
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='attention_output',
        choices=['cnn_output', 'gcn_output', 'attention_output'],
        help='Intermediate feature representation used for analysis.',
    )
    parser.add_argument(
        '--spatial_motif_classes',
        nargs='+',
        default=None,
        help='Classes for spatial motif analysis (e.g., m6A m5C Y). If not specified, all classes are used.',
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=3,
        help='Number of clusters for spatial motif analysis.',
    )
    parser.add_argument(
        '--pca_components',
        type=int,
        default=50,
        help='PCA components for dimensionality reduction in spatial motif analysis.',
    )
    parser.add_argument(
        '--node_num',
        type=int,
        default=10,
        help='Number of context positions for spatial motif.',
    )
    parser.add_argument(
        '--conda_env',
        type=str,
        default='learn-new',
        help='Conda environment name for running R script.',
    )

    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        layer_name=args.layer,
        spatial_motif_classes=args.spatial_motif_classes,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        node_num=args.node_num,
        conda_env=args.conda_env,
    )
