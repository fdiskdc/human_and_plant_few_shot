import os
import random
import json
import numpy as np
import torch
import warnings
from datetime import datetime
import scipy.stats as stats  # Added for mode calculation

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch_geometric.loader import DataLoader
warnings.filterwarnings('ignore')

from model.main_model import RNA_ClassQuery_Model
from dataset.gen3_zero import Gen3ZeroDataset, LABEL_MAPPING, MOD_NAMES  # Added mappings
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


def main(config_path='json/3gen.json', checkpoint_path=None):
    """Main evaluation function"""
    # Load configuration from JSON
    global Config, config_dict
    Config, config_dict = load_config(config_path)
    Config.device='cuda'

    # Setup logging
    logger = setup_logging(Config.log_dir, f'{Config.experiment_name}_test')

    # Log basic info
    logger.info(f"\n{'='*60}")
    logger.info("RNA Multi-label Classification Full-Dataset Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Random seed: {Config.random_seed}")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load full dataset (no train/test split)
    logger.info(f"\nLoading full dataset from {Config.data.human_data_dir}...")
    dataset = Gen3ZeroDataset(
        mode='test',  # Use 'test' mode to load all data
        data_dir='npy',
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Precompute structures if batch cache doesn't exist
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        logger.info("\n" + "="*60)
        logger.info("Batch cache not found, precomputing all secondary structures...")
        logger.info("="*60)
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    else:
        logger.info(f"\nBatch cache exists: {cache_stats['batch_cache']['path']}")
        logger.info(f"  File size: {cache_stats['batch_cache']['size_mb']:.2f} MB")
        if cache_stats['batch_cache'].get('loaded_in_memory', False):
            logger.info(f"  Status: Loaded in memory")
        else:
            logger.info(f"  Status: Not loaded in memory")

    # Create dataloader for full dataset
    test_loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    logger.info(f"DataLoader created:")
    logger.info(f"  Batch size: {Config.batch_size}, Batches: {len(test_loader)}")

   

    # Create model
    logger.info(f"\nCreating model...")
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Load model weights
    if checkpoint_path is None:
        # Default to best_model.pt from checkpoint directory
        checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"\nLoading model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model weights loaded successfully")
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"Checkpoint macro_f1: {checkpoint['metrics'].get('group_macro_f1', 'N/A'):.4f}")
    else:
        logger.warning(f"\nCheckpoint not found at {checkpoint_path}")
        logger.warning("Proceeding with randomly initialized model (for testing only)")
    
    model.eval()

    # ========================================================================
    # Evaluation Phase
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Starting Evaluation...")
    logger.info(f"{'='*60}")

    # 1. Get predictions for all data
    logger.info(f"\nRunning inference on full dataset...")
    y_true, y_prob, y_4class, y_4prob = get_all_predictions(
        model, test_loader, Config.device, Config.use_hierarchical
    )
    # 在调用 evaluate_unbalance 之前
    print(f"m6A (Class 9) positive count: {y_true[:, 9].sum()}")
    print(f"m6A (Class 9) negative count: {(1 - y_true[:, 9]).sum()}")

    # 2. Compute evaluation metrics
    # ========================================================================
    
    # 2.1 Unbalance: Real distribution
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation 1: Unbalance (Real Distribution)")
    logger.info(f"{'='*60}")
    metrics_unbalance = evaluate_unbalance(
        y_true, y_prob, Config.device, y_4class, Config.random_seed, y_4prob
    )
    logger.info(f"Unbalance Macro-F1: {metrics_unbalance['group_macro_f1']:.4f}")

    # 2.2 BalanceB: Optimal threshold (1:1 balance)
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation 2: BalanceB (Optimal Threshold, 1:1 Balance)")
    logger.info(f"{'='*60}")
    metrics_balanceb = evaluate_balanceb(
        y_true, y_prob, y_4class, Config.device, Config.random_seed, y_4prob
    )
    if metrics_balanceb is not None:
        logger.info(f"BalanceB Macro-F1: {metrics_balanceb['group_macro_f1']:.4f}")
    else:
        logger.warning("BalanceB evaluation returned None, skipping Macro-F1 logging")


    # 2.3 BalanceB Fixed Threshold: Fixed threshold (0.5, 1:1 balance)
    logger.info(f"\n{'='*60}")
    th=0.43
    logger.info("Evaluation 3: BalanceB (Fixed Threshold %f, 1:1 Balance)"%(th))
    logger.info(f"{'='*60}")
    metrics_balanceb_th = evaluate_balanceb_th(
        y_true, y_prob, y_4class, Config.device, threshold=0.430, 
        random_seed=Config.random_seed, y_4prob=y_4prob
    )
    logger.info(f"BalanceB (th={th}) Macro-F1: {metrics_balanceb_th['group_th_macro_f1']:.4f}")

    # 2.4 Group Balanced: Positive vs Pure Negative samples
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation 4: Group Balanced (Negative from Pure Negative)")
    logger.info(f"{'='*60}")
    metrics_group_balanceb = evaluate_group_balanceb(
        y_true, y_prob, y_4class, Config.random_seed
    )
    logger.info(f"Group Balanced Macro-F1: {metrics_group_balanceb['group_balanced_macro_f1']:.4f}")

    # 2.5 Additional metrics with optimal threshold
    metrics_opt = evaluate_with_optimal_threshold(
        model, test_loader, Config.device, Config.use_hierarchical
    )
    metrics_4class = evaluate_4class_with_optimal_threshold(
        model, test_loader, Config.device, Config.use_hierarchical
    )

    # ========================================================================
    # Localization Evaluation (if attention supervision enabled)
    # ========================================================================
    topk_results = {}
    comprehensive_results = {}
    
    if Config.use_attention_supervision:
        logger.info(f"\n{'='*60}")
        logger.info("Localization Evaluation (Positive Samples Only)")
        logger.info(f"{'='*60}")
        
        # 1. 获取所有数据（含负样本）的预测
        y_true, y_prob, y_4class, y_4prob, attn_weights, y_site = \
            get_all_predictions_and_attention(
                model, test_loader, Config.device, Config.use_hierarchical
            )
        
        if attn_weights is not None and y_site is not None:
            # 获取基本维度
            # attn_weights 形状通常为 [N, 12, 1001]
            N = attn_weights.shape[0]
            L = attn_weights.shape[2]
            
            # --- 修复核心：先 Reshape y_site ---
            if y_site.dim() == 1:
                y_site = y_site.view(N, L) # 将 57337280 还原为 [57280, 1001]

            # 2. 创建正样本掩码 (y_true 形状为 [N, 12])
            if torch.is_tensor(y_true):
                pos_mask = torch.any(y_true > 0, dim=1)
            else:
                pos_mask = np.any(y_true > 0, axis=1)
            
            # 3. 应用掩码过滤数据
            attn_weights_pos = attn_weights[pos_mask] # 形状变为 [N_pos, 12, 1001]
            y_site_pos = y_site[pos_mask]           # 形状变为 [N_pos, 1001]
            
            num_pos = pos_mask.sum().item() if torch.is_tensor(pos_mask) else pos_mask.sum()
            logger.info(f"Filtered for localization: {num_pos} positive samples remaining.")

            # --- 4. 使用过滤后的正样本进行计算 ---
            # 计算 Top-K site recall
            topk_results = calculate_topk_recall(
                attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 7, 10, 20, 50]
            )
            
            # 计算综合指标
            comprehensive_results = calculate_comprehensive_localization_metrics(
                attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 7, 10]
            )
            
            # 位置统计分析 (Median/Mode)
            analyze_position_stats(attn_weights_pos, y_site_pos, k_list=[1, 10], logger=logger)

            # 宽容度评估 (M=2)
            M_RADIUS = 5
            topk_results_M = calculate_topk_recall_tolerateM(
                attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 10, 20, 50], M=M_RADIUS
            )
            print_topk_table_tolerateM(topk_results_M, k_list=[1, 3, 5, 10, 20, 50], M=M_RADIUS, logger=logger)
            
            comp_results_M = calculate_comprehensive_localization_metrics_tolerateM(
                attn_weights_pos, y_site_pos, k_list=[1, 3, 5, 10], M=M_RADIUS
            )
            print_comprehensive_table_tolerateM(comp_results_M, k_list=[1, 3, 5, 10], M=M_RADIUS, logger=logger)
            
        else:
            logger.info(f"Attention weights or site labels not available, skipping Top-K evaluation")

    # ========================================================================
    # Print Results
    # ========================================================================
    
    logger.info(f"\n{'='*120}")
    logger.info("FINAL EVALUATION RESULTS")
    logger.info(f"{'='*120}")
    
    # Print Unbalance and BalanceB tables
    print_evaluation_results(
        metrics_unbalance=metrics_unbalance,
        metrics_balanceb=metrics_balanceb,
        epoch=0,
        logger=logger,
        metrics_opt=metrics_opt,
        metrics_group_balanceb=metrics_group_balanceb,
        metrics_4class=metrics_4class
    )
    
    # Print BalanceB Fixed Threshold results
    logger.info(f"\n### BalanceB Fixed Threshold (threshold=0.5) Results ###")
    print_balanceb_th_table(metrics_balanceb_th, logger=logger)
    
    # Print Top-K results (if available)
    if topk_results:
        print_topk_table(topk_results, k_list=[1, 3, 5, 7, 10, 20, 50], logger=logger)
    
    # Print Comprehensive Localization results (if available)
    if comprehensive_results:
        print_comprehensive_table(comprehensive_results, k_list=[1, 3, 5, 7, 10], logger=logger)
    
    # Final summary
    logger.info(f"\n{'='*120}")
    logger.info("SUMMARY")
    logger.info(f"{'='*120}")
    logger.info(f"Unbalance Macro-F1:          {metrics_unbalance['group_macro_f1']:.4f}")
    if metrics_balanceb is not None:
        logger.info(f"BalanceB Macro-F1:           {metrics_balanceb['group_macro_f1']:.4f}")
    logger.info(f"BalanceB (th=0.340) Macro-F1:  {metrics_balanceb_th['group_th_macro_f1']:.4f}")
    logger.info(f"Group Balanced Macro-F1:      {metrics_group_balanceb['group_balanced_macro_f1']:.4f}")
    logger.info(f"{'='*120}")
    
    # Close logger
    logger.info("\nEvaluation complete!")


def print_balanceb_th_table(metrics, logger=None):
    """
    Print formatted table for BalanceB Fixed Threshold results.
    """
    from prettytable import PrettyTable
    from utils.common import INDEX_TO_GROUP
    
    # 12-Class Table
    output = f"\n### BalanceB Select Threshold  - 12 Class ###\n"
    table = PrettyTable()
    table.field_names = [
        "Class", "Mod Name", "Acc", "Precision", "Recall", "F1", "MCC", 
        "Sn", "Sp", "TP", "TN", "FP", "FN"
    ]
    table.align = "r"
    table.align["Class"] = "l"
    table.align["Mod Name"] = "l"
    
    for c in range(12):
        mod_name = MOD_NAMES.get(c, f"Class_{c}")
        
        acc = metrics.get(f'group_class_{c}_th_accuracy', 0.0)
        precision = metrics.get(f'group_class_{c}_th_precision', 0.0)
        recall = metrics.get(f'group_class_{c}_th_recall', 0.0)
        f1 = metrics.get(f'group_class_{c}_th_f1', 0.0)
        mcc = metrics.get(f'group_class_{c}_th_mcc', 0.0)
        sensitivity = metrics.get(f'group_class_{c}_th_sensitivity', 0.0)
        specificity = metrics.get(f'group_class_{c}_th_specificity', 0.0)
        tp = int(metrics.get(f'group_class_{c}_th_tp', 0))
        tn = int(metrics.get(f'group_class_{c}_th_tn', 0))
        fp = int(metrics.get(f'group_class_{c}_th_fp', 0))
        fn = int(metrics.get(f'group_class_{c}_th_fn', 0))
        
        table.add_row([
            f"{c} ({mod_name})",
            mod_name,
            f"{acc:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{mcc:.4f}",
            f"{sensitivity:.4f}",
            f"{specificity:.4f}",
            tp, tn, fp, fn
        ])
    
    output += str(table) + "\n"
    output += f"BalanceB (th=0.5) Macro-F1: {metrics.get('group_th_macro_f1', 0.0):.4f}\n"
    
    # 4-Class Table
    output += f"\n### BalanceB Fixed Threshold (th=0.5) - 4 Class ###\n"
    table_4 = PrettyTable()
    table_4.field_names = [
        "Class", "Nucleotide", "Acc", "Precision", "Recall", "F1", "MCC",
        "Sn", "Sp", "TP", "TN", "FP", "FN"
    ]
    table_4.align = "r"
    table_4.align["Class"] = "l"
    table_4.align["Nucleotide"] = "l"
    
    for g in range(4):
        group_name = INDEX_TO_GROUP[g]
        
        acc = metrics.get(f'group_4class_{g}_th_accuracy', 0.0)
        precision = metrics.get(f'group_4class_{g}_th_precision', 0.0)
        recall = metrics.get(f'group_4class_{g}_th_recall', 0.0)
        f1 = metrics.get(f'group_4class_{g}_th_f1', 0.0)
        mcc = metrics.get(f'group_4class_{g}_th_mcc', 0.0)
        sensitivity = metrics.get(f'group_4class_{g}_th_sensitivity', 0.0)
        specificity = metrics.get(f'group_4class_{g}_th_specificity', 0.0)
        tp = int(metrics.get(f'group_4class_{g}_th_tp', 0))
        tn = int(metrics.get(f'group_4class_{g}_th_tn', 0))
        fp = int(metrics.get(f'group_4class_{g}_th_fp', 0))
        fn = int(metrics.get(f'group_4class_{g}_th_fn', 0))
        
        table_4.add_row([
            f"{g}",
            group_name,
            f"{acc:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{mcc:.4f}",
            f"{sensitivity:.4f}",
            f"{specificity:.4f}",
            tp, tn, fp, fn
        ])
    
    output += str(table_4) + "\n"
    output += f"BalanceB (th=0.5) 4-Class Macro-F1: {metrics.get('group_4class_th_macro_f1', 0.0):.4f}\n"
    
    print(output)
    if logger:
        logger.info(output)

def analyze_position_stats(attn_weights, y_site, k_list=[1, 10], logger=None):
    """
    Analyze the distribution of predicted positions (Median & Mode) for each class.
    Considers the Top-K predicted positions across all samples containing the modification.
    """
    from prettytable import PrettyTable
    
    logger.info(f"\n{'='*80}")
    logger.info("Predicted Position Statistics (Median & Mode in Top-K)")
    logger.info(f"{'='*80}")
    
    # Move to CPU/Numpy
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if isinstance(y_site, torch.Tensor):
        y_site = y_site.detach().cpu().numpy()
        
    # Reshape y_site [N*L] -> [N, L]
    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        seq_len = attn_weights.shape[2]
        y_site = y_site.reshape(batch_size, seq_len)
        
    table = PrettyTable()
    # Dynamic columns based on K
    field_names = ["Class", "Name", "Count"]
    for k in k_list:
        field_names.extend([f"Top-{k} Median", f"Top-{k} Mode"])
    table.field_names = field_names
    table.align = "r"
    table.align["Class"] = "l"
    table.align["Name"] = "l"
    
    for class_idx in range(12):
        # Find Label ID
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break
        
        if original_label_id is None: 
            continue
            
        # Filter samples that actually have this modification
        has_mod = np.any(y_site == original_label_id, axis=1)
        count = np.sum(has_mod)
        if count == 0:
            continue
            
        target_attn = attn_weights[has_mod, class_idx, :] # [M, SeqLen]
        
        row = [class_idx, MOD_NAMES.get(class_idx, str(class_idx)), count]
        
        for k in k_list:
            # Get Top-K indices for all samples
            # argsort returns ascending, so take last k
            # flattened to get a distribution of "where the model looks"
            topk_indices = np.argsort(target_attn, axis=1)[:, -k:] 
            
            # Flatten to get distribution of all predicted positions
            all_preds = topk_indices.flatten()
            
            median = np.median(all_preds)
            
            # Calculate mode safely
            mode_res = stats.mode(all_preds, keepdims=False)
            try:
                # Scipy 1.11+ returns a struct with .mode
                mode_val = mode_res.mode
                if isinstance(mode_val, np.ndarray):
                    mode_val = mode_val[0]
            except:
                # Older scipy returns array directly
                mode_val = mode_res[0]
                
            row.extend([f"{int(median)}", f"{int(mode_val)}"])
            
        table.add_row(row)
        
    print(table)
    if logger:
        logger.info("\n" + str(table))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full-dataset evaluation for RNA multi-label classification')
    parser.add_argument('--config', type=str, default='json/3gen.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--checkpoint', type=str, default="logs/rna_classification_20260129_164810/checkpoints/epoch_020.pt",
                        help='Path to model checkpoint (default: checkpoints/best_model.pt)')
    
    args = parser.parse_args()
    
    main(config_path=args.config, checkpoint_path=args.checkpoint)