"""
zero_shot_fewshot_analysis.py - 零样本特征对齐与小样本轨迹分析 / Zero-shot Feature Alignment & Few-shot Trajectory Analysis

小样本分析流水线主入口:零样本特征对齐 (人类 vs 植物) + 小样本轨迹分析 (Y/m5C/m6A 一对多) +
3 代数据可视化 + 植物/3代空间 motif 分析。所有结果输出到同一时间戳目录。
Main entry for few-shot analysis pipeline: zero-shot feature alignment (human vs plant) + few-shot trajectory
(one-vs-rest on Y/m5C/m6A) + 3gen visualization + spatial motif analysis. All results in same timestamped output dir.

功能模块 / Modules:
- 零样本特征对齐 / Zero-shot feature alignment
- 小样本轨迹分析 / Few-shot trajectory analysis
- 3 代数据可视化 / 3-generation data visualization
- 空间 motif 分析 / Spatial motif analysis
- R 脚本调用 / R script invocation
- main: 主入口 / Main entry point

输入 / Inputs:
- json/plant_single.json: 配置 / Config
- checkpoints/best_model.pt: 训练好的模型 / Trained model
- human3/, plant3/, 3gen/ 数据 / human, plant, 3gen data
- 命令行参数 / CLI: --config, --layer, --conda_env, --spatial_motif_classes

输出 / Outputs:
- output/zero_fewshot_analysis/figures/*.pdf, *.png: 图表 / Figures
- output/zero_fewshot_analysis/data/*.csv, *.json: 数据 / Data
- 时间戳子目录 / Timestamped subdirectory

数据流 / Data Flow:
1. 加载配置与模型 / Load config and model
2. 零样本特征提取 / Zero-shot feature extraction
3. 小样本轨迹 / Few-shot trajectory
4. UMAP + motif 分析 / UMAP + motif analysis
5. 图表 + 数据保存 / Save figures and data

相关文件 / Related Files:
- 调用 / Calls: utils.fewshot_analysis_*, model.mrmodn, plot_zero_fewshot_analysis.R
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python zero_shot_fewshot_analysis.py --config json/plant_single.json --layer attention_output

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
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

    baseline_model, checkpoint = load_model_from_checkpoint(checkpoint_path, Config.device)
    baseline_model.eval()
    baseline_state_dict = copy.deepcopy(baseline_model.state_dict())
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")

    # Run standard zero-shot and few-shot analysis (plant data)
    zero_shot_bundle = run_zero_shot_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger,
        model=baseline_model, checkpoint=checkpoint
    )
    few_shot_metrics = run_few_shot_trajectory_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger, zero_shot_bundle,
        baseline_model=baseline_model, baseline_state_dict=baseline_state_dict
    )

    # Run 3-generation analysis (always run alongside plant analysis)
    logger.info("\n" + "=" * 80)
    logger.info("Running 3-generation data analysis...")
    logger.info("=" * 80)
    gen3_bundle = run_gen3_analysis(
        Config, checkpoint_path, output_dir, layer_name, logger,
        model=baseline_model, checkpoint=checkpoint
    )

    # Run spatial motif analysis for both plant and gen3 data (always run)
    if not HAS_SPATIAL_MOTIF:
        logger.warning("Spatial motif module not available (logomaker or captum not installed).")
    else:
        # Determine target classes for spatial motif
        if spatial_motif_classes is None:
            target_classes = list(MOD_NAMES.items())
        else:
            target_classes = [(k, v) for k, v in MOD_NAMES.items() if v in spatial_motif_classes]

        # Run spatial motif on plant data
        logger.info("\n" + "=" * 80)
        logger.info("Running spatial motif analysis on plant data...")
        logger.info("=" * 80)
        plant_dataset = zero_shot_bundle.get('plant_dataset')
        if plant_dataset is None:
            _, plant_dataset = prepare_datasets(Config)
        run_spatial_motif_analysis(
            baseline_model, plant_dataset, 'plant', target_classes, output_dir, Config.device,
            n_clusters=n_clusters, pca_components=pca_components, node_num=node_num
        )

        # Run spatial motif on gen3 data
        logger.info("\n" + "=" * 80)
        logger.info("Running spatial motif analysis on gen3 data...")
        logger.info("=" * 80)
        run_spatial_motif_analysis(
            baseline_model, gen3_bundle['gen3_dataset'], 'gen3', target_classes, output_dir, Config.device,
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
