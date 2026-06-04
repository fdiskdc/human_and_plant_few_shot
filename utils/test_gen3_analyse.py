"""
utils/test_gen3_analyse.py - 3gen 推理 + Top-K 定位分布图 / 3gen Inference + Top-K Localization Distribution

3gen 数据集推理 + Top-K 定位分布图 + tolerant-M (tolerateM) 指标变体。
3gen dataset inference + Top-K localization distribution plots + tolerant-M (tolerateM) metric variants.

功能模块 / Modules:
- 3gen 推理 / 3gen inference
- Top-K 定位 / Top-K localization
- Tolerant-M 指标 / Tolerant-M metrics
- 分布图 / Distribution plots
- main: 主入口 / Main entry point

输入 / Inputs:
- json/gen3.json: 配置 / Config
- checkpoints/best_gen3.pt: 预训练模型 / Pretrained model
- 3gen/seq.npy, 3gen/1001loc.npy: 3gen 数据 / 3gen data
- 命令行参数 / CLI: --config, --checkpoint, --output

输出 / Outputs:
- logs/gen3_analyse_*/results.json: 评估结果 / Evaluation results
- fig/gen3_topk_distribution_*.png: Top-K 分布图 / Top-K distribution plots
- tolerant-M 指标 / tolerant-M metrics

数据流 / Data Flow:
1. 加载数据 / Load data
2. 模型推理 / Model inference
3. Top-K 定位 / Top-K localization
4. 计算 tolerant-M / Compute tolerant-M
5. 绘图 + 保存 / Plot + save

相关文件 / Related Files:
- 调用 / Calls: dataset.gen3.Gen3Dataset, model.mrmodn, utils.{common,metrics}
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python -m utils.test_gen3_analyse --config json/gen3.json

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import random
import json
import numpy as np
import torch
import warnings
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch_geometric.loader import DataLoader
warnings.filterwarnings('ignore')

from model.mrmodn import RNA_ClassQuery_Model
from dataset.gen3_zero import Gen3ZeroDataset, LABEL_MAPPING, MOD_NAMES
from utils import (
    setup_logging,
    evaluate_unbalance,
    evaluate_balanceb,
    evaluate_balanceb_th,
    evaluate_group_balanceb,
    evaluate_with_optimal_threshold,
    evaluate_4class_with_optimal_threshold,
    load_config,
    print_evaluation_results,
    get_all_predictions,
    get_all_predictions_and_attention,
    calculate_topk_recall,
    print_topk_table,
    calculate_comprehensive_localization_metrics,
    print_comprehensive_table,
    load_checkpoint
)
from utils.common import print_topk_table_tolerateM,calculate_topk_recall_tolerateM,print_comprehensive_table_tolerateM,calculate_comprehensive_localization_metrics_tolerateM


def plot_localization_distributions(attn_weights, y_site, k_list=[1, 10, 50], save_dir='att_fig', logger=None):
    """
    绘制 Top-K 预测位置的概率密度分布图 (PDF)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if logger: logger.info(f"Created directory: {save_dir}")

    # 转换数据到 CPU numpy
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if isinstance(y_site, torch.Tensor):
        y_site = y_site.detach().cpu().numpy()

    # 这里的 y_site 形状处理
    if y_site.ndim == 1:
        N, _, L = attn_weights.shape
        y_site = y_site.reshape(N, L)

    for class_idx in range(12):
        mod_name = MOD_NAMES.get(class_idx, f"Class_{class_idx}")
        
        # 查找原始 label ID 用于过滤拥有该修饰的样本
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break
        
        if original_label_id is None: continue

        # 过滤出包含该修饰的样本
        has_mod = np.any(y_site == original_label_id, axis=1)
        if np.sum(has_mod) == 0:
            continue

        target_attn = attn_weights[has_mod, class_idx, :] # [M, 1001]
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 绘制不同 K 值下的分布
        colors = sns.color_palette("husl", len(k_list))
        for i, k in enumerate(k_list):
            # 获取每个样本中 attention 最高的 top-k 个位置
            topk_indices = np.argsort(target_attn, axis=1)[:, -k:] 
            all_preds = topk_indices.flatten()
            
            sns.kdeplot(all_preds, label=f'Top-{k}', color=colors[i], fill=True, alpha=0.1)

        # 绘制 Ground Truth 的分布（真实修饰位点通常在中心 500 附近）
        # 这里提取真实标签的位置
        gt_positions = []
        rows, cols = np.where(y_site[has_mod] == original_label_id)
        gt_positions = cols
        if len(gt_positions) > 0:
            sns.kdeplot(gt_positions, label='Ground Truth', color='black', linestyle='--', linewidth=2)

        plt.title(f"Localization Distribution - {mod_name} (Class {class_idx})")
        plt.xlabel("Sequence Position (0-1000)")
        plt.ylabel("Probability Density")
        plt.xlim(0, 1001)
        plt.legend()
        
        save_path = os.path.join(save_dir, f"class_{class_idx}_{mod_name.replace('/', '_')}_dist.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    if logger:
        logger.info(f"Localization density plots saved to {save_dir}")


def main(config_path='json/3gen.json', checkpoint_path=None):
    """Main evaluation function"""
    global Config, config_dict
    Config, config_dict = load_config(config_path)
    Config.device='cuda'

    logger = setup_logging(Config.log_dir, f'{Config.experiment_name}_test')

    logger.info(f"\n{'='*60}")
    logger.info("RNA Multi-label Classification Full-Dataset Evaluation")
    logger.info(f"{'='*60}")

    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    dataset = Gen3ZeroDataset(mode='test', data_dir='npy')
    test_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=Config.num_classes,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=getattr(Config, 'use_hierarchical', False),
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")

    # Inference and Eval... (省略中间 evaluate_unbalance 等逻辑以保持简洁，保留核心 localization 部分)
    y_true, y_prob, y_4class, y_4prob = get_all_predictions(model, test_loader, Config.device, Config.use_hierarchical)
    
    # 核心修改部分：Localization Evaluation
    if Config.use_attention_supervision:
        logger.info(f"\n{'='*60}")
        logger.info("Localization Evaluation & Density Plotting")
        logger.info(f"{'='*60}")
        
        y_true, y_prob, y_4class, y_4prob, attn_weights, y_site = \
            get_all_predictions_and_attention(model, test_loader, Config.device, Config.use_hierarchical)
        
        if attn_weights is not None and y_site is not None:
            N = attn_weights.shape[0]
            L = attn_weights.shape[2]
            
            if y_site.dim() == 1:
                y_site = y_site.view(N, L)

            if torch.is_tensor(y_true):
                pos_mask = torch.any(y_true > 0, dim=1)
            else:
                pos_mask = np.any(y_true > 0, axis=1)
            
            attn_weights_pos = attn_weights[pos_mask]
            y_site_pos = y_site[pos_mask]
            
            # 1. 打印统计表格
            analyze_position_stats(attn_weights_pos, y_site_pos, k_list=[1, 10], logger=logger)

            # 2. 新增：生成概率密度图 (Top 1, 10, 50)
            plot_localization_distributions(
                attn_weights_pos, 
                y_site_pos, 
                k_list=[1, 10, 50], 
                save_dir='att_fig', 
                logger=logger
            )

            # 3. 计算常规指标
            topk_results = calculate_topk_recall(attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 7, 10, 20, 50])
            print_topk_table(topk_results, k_list=[1, 3, 5, 7, 10, 20, 50], logger=logger)
            
            comp_results_M = calculate_comprehensive_localization_metrics_tolerateM(
                attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 10], M=5
            )
            print_comprehensive_table_tolerateM(comp_results_M, k_list=[1, 3, 5, 10], M=5, logger=logger)

    logger.info("\nEvaluation complete!")

# ... (保持原有的 analyze_position_stats 和 print_balanceb_th_table 函数不变)

def analyze_position_stats(attn_weights, y_site, k_list=[1, 10], logger=None):
    """
    Analyze the distribution of predicted positions (Median & Mode) for each class.
    """
    from prettytable import PrettyTable
    
    # Move to CPU/Numpy
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if isinstance(y_site, torch.Tensor):
        y_site = y_site.detach().cpu().numpy()
        
    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        seq_len = attn_weights.shape[2]
        y_site = y_site.reshape(batch_size, seq_len)
        
    table = PrettyTable()
    field_names = ["Class", "Name", "Count"]
    for k in k_list:
        field_names.extend([f"Top-{k} Median", f"Top-{k} Mode"])
    table.field_names = field_names
    
    for class_idx in range(12):
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break
        if original_label_id is None: continue
            
        has_mod = np.any(y_site == original_label_id, axis=1)
        count = np.sum(has_mod)
        if count == 0: continue
            
        target_attn = attn_weights[has_mod, class_idx, :]
        row = [class_idx, MOD_NAMES.get(class_idx, str(class_idx)), count]
        
        for k in k_list:
            topk_indices = np.argsort(target_attn, axis=1)[:, -k:] 
            all_preds = topk_indices.flatten()
            median = np.median(all_preds)
            mode_res = stats.mode(all_preds, keepdims=False)
            try:
                mode_val = mode_res.mode
                if isinstance(mode_val, np.ndarray): mode_val = mode_val[0]
            except:
                mode_val = mode_res[0]
            row.extend([f"{int(median)}", f"{int(mode_val)}"])
        table.add_row(row)
    if logger:
        logger.info("\n" + str(table))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='json/3gen.json')
    parser.add_argument('--checkpoint', type=str, default="logs/rna_classification_20260119_163921/checkpoints/epoch_040.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)