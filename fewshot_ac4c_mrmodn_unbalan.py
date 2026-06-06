"""
fewshot_ac4c_unbalan.py - ac4C 非平衡数据集训练脚本(剪枝模式) / AC4C Unbalanced Dataset Training Script (Pruned Mode)

ac4C 非平衡数据集训练入口:剪枝 RGCNFormer 主模型,仅计算 ac4C 单类查询 (class index 6)。
处理模型 1 维输出与 12 维标签的不匹配,使用自定义训练/测试循环。
AC4C unbalanced training entry: prunes RGCNFormer to compute only the AC4C class query (class index 6).
Handles 1-dim model output vs 12-dim label mismatch with custom training/test loops.

功能模块 / Modules:
- 模型剪枝 / Model pruning (only ac4C query)
- 自定义训练/测试循环 / Custom training/test loops
- PrettyTable 结果输出 / PrettyTable result output
- main: 主入口 / Main entry point

输入 / Inputs:
- json/ac4c_unbalan.json: 训练配置 / Training config
- ac4c_unbalan/seq.npy, ac4c_unbalan/label.npy: ac4C 非平衡数据 / AC4C unbalanced data
- 命令行参数 / CLI: --config, --gpu, --seed

输出 / Outputs:
- checkpoints/best_ac4c_unbalan.pt: 最佳 ac4C 非平衡模型 / Best ac4C unbalanced model
- logs/ac4c_unbalan_*/results.json: 评估结果 / Evaluation results
- PrettyTable 输出 / PrettyTable output

数据流 / Data Flow:
1. 加载配置 / Load config
2. 加载 ac4C 非平衡数据 / Load ac4C unbalanced data
3. 初始化剪枝模型 / Init pruned model
4. 自定义训练循环 / Custom training loop
5. 评估 + PrettyTable 输出 / Evaluate and PrettyTable output

相关文件 / Related Files:
- 调用 / Calls: model.mrmodn.RNA_ClassQuery_Model, dataset.ac4c.AC4CDataset, prettytable
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python fewshot_ac4c_unbalan.py --config json/ac4c_unbalan.json --gpu 1

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import random
import json
import numpy as np
import torch
import argparse
from datetime import datetime
from prettytable import PrettyTable

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.mrmodn import RNA_ClassQuery_Model
from dataset.ac4c import AC4CDataset
from utils import (
    setup_logging, setup_tensorboard, log_metrics_to_tensorboard,
    get_smoothed_pos_weights,
    save_checkpoint, load_config,
    print_evaluation_results, print_few_shot_results,
    run_few_shot_benchmark_ac4c,
    MOD_NAMES, get_all_predictions
)
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑 BCE 损失 / Label-smoothed BCE loss.

    在二分类场景下将硬标签 {0, 1} 平滑为 (smoothing/2, 1 - smoothing/2)，
    以缓解过拟合并提升模型校准。底层使用 `BCEWithLogitsLoss`。
    Smooths hard labels {0, 1} to (smoothing/2, 1 - smoothing/2) to mitigate
    overfitting and improve calibration. Backed by `BCEWithLogitsLoss`.

    Attributes / 属性:
        smoothing (float): [中文] 平滑系数, 取值 (0, 1) / [English] smoothing factor in (0, 1).
        bce (nn.BCEWithLogitsLoss): [中文] 底层 BCE 损失 / [English] underlying BCE loss.
    """

    def __init__(self, smoothing=0.1):
        """
        初始化标签平滑损失 / Initialize the label-smoothing loss.

        Args / 参数:
            smoothing (float, optional): [中文] 标签平滑系数 / [English] label-smoothing
                factor. Defaults to 0.1.
        """

        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # 将 [0, 1] 标签转换为 [0.05, 0.95] (假设 smoothing=0.1)
        # targets: (Batch,)
        """
        计算平滑后的 BCE 损失 / Compute smoothed BCE loss.

        将 targets 从 {0, 1} 平滑到 (smoothing/2, 1 - smoothing/2) 后与 logits 计算 BCE。
        Smooths targets from {0, 1} to (smoothing/2, 1 - smoothing/2) then applies BCE.

        Args / 参数:
            logits (Tensor): [中文] 模型原始输出, 形状 (Batch,) / [English] raw logits, shape (Batch,).
            targets (Tensor): [中文] 0/1 标签, 形状 (Batch,) / [English] 0/1 labels, shape (Batch,).

        Returns / 返回:
            Tensor: [中文] 标量损失值 / [English] scalar loss value.
        """

        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smooth_targets)
        return loss
# Constants for AC4C
AC4C_CLASS_IDX = 6
AC4C_GROUP_IDX = 1  # Group C

class PrunedModelWrapper(nn.Module):
    """
    剪枝模型包装器: 回填 1 维输出到 12 维 / Wrapper padding pruned 1-d output to 12-d.

    将剪枝后仅含 ac4C 头的 1 维输出回填为 12 维 (非激活位填 -1e9)，
    以便复用现有 12 维评估管线, 包括 few-shot 基准与混淆矩阵统计。
    Pads the 1-d pruned ac4C-only output back to 12-d (inactive slots = -1e9)
    so the existing 12-d evaluation pipeline (few-shot benchmark, confusion
    matrices) can be reused unchanged.

    Attributes / 属性:
        model (nn.Module): [中文] 内部剪枝模型 / [English] inner pruned model.
        class_idx (int): [中文] ac4C 在 12 类中的索引 / [English] ac4C index in 12-class head.
        group_idx (int): [中文] ac4C 在 4 分组中的索引 / [English] ac4C index in 4-group head.
        use_hierarchical (bool): [中文] 是否使用分层头 / [English] whether hierarchical head is on.
    """
    def __init__(self, pruned_model, class_idx=AC4C_CLASS_IDX, group_idx=AC4C_GROUP_IDX):
        """
        初始化包装器 / Initialize the pruned-model wrapper.

        Args / 参数:
            pruned_model (nn.Module): [中文] 剪枝后的单类输出模型 / [English] pruned single-class model.
            class_idx (int, optional): [中文] ac4C 在 12 类中的索引 / [English] ac4C class index
                in the full 12-class head. Defaults to AC4C_CLASS_IDX.
            group_idx (int, optional): [中文] ac4C 在 4 分组中的索引 / [English] ac4C group index
                in the 4-group head. Defaults to AC4C_GROUP_IDX.
        """

        super().__init__()
        self.model = pruned_model
        self.class_idx = class_idx
        self.group_idx = group_idx
        self.use_hierarchical = pruned_model.use_hierarchical

    def forward(self, x, edge_index, batch=None):
        """
        前向传播, 将 1 维输出回填到 12 维 / Forward pass, padding 1-d output back to 12-d.

        非激活类的 logits 用 `-1e9` 填充, 以保证 sigmoid 后趋近 0, 从而与原有 12 维
        评估流程完全兼容。分层模式下同时回填 4 维分组输出。
        Fills inactive class logits with `-1e9` so sigmoid(output) -> 0, keeping
        compatibility with the existing 12-d evaluation pipeline. In hierarchical
        mode the 4-d group output is padded as well.

        Args / 参数:
            x (Tensor): [中文] 节点特征 / [English] node features.
            edge_index (LongTensor): [中文] 边索引 / [English] edge index.
            batch (LongTensor, optional): [中文] 批索引 / [English] batch index.

        Returns / 返回:
            Tuple[Tensor, Tensor] | Tensor: [中文] (12 维 logits [, 4 维 logits]) /
                [English] 12-d logits, optionally followed by 4-d logits.
        """

        device = x.device
        
        # Get pruned output (Batch, 1) or (Batch, 1), (Batch, 1)
        out = self.model(x, edge_index, batch)
        
        if self.use_hierarchical:
            logits_1, logits_4_1 = out
            batch_size = logits_1.size(0)
            
            # Pad 12-class logits
            # Use -1e9 for non-active classes so sigmoid(logits) -> 0
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_idx] = logits_1.view(-1)
            
            # Pad 4-class logits
            padded_logits_4 = torch.full((batch_size, 4), -1e9, device=device)
            padded_logits_4[:, self.group_idx] = logits_4_1.view(-1)
            
            return padded_logits_12, padded_logits_4
        else:
            logits_1 = out
            batch_size = logits_1.size(0)
            
            # Pad 12-class logits
            padded_logits_12 = torch.full((batch_size, 12), -1e9, device=device)
            padded_logits_12[:, self.class_idx] = logits_1.view(-1)
            
            return padded_logits_12

def train_epoch_pruned(model, dataloader, criterion, optimizer, scheduler, device, logger, use_hierarchical, use_amp):
    """
    Optimized training loop using the pruned model directly (avoiding wrapper overhead).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training (Pruned)", leave=True)
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get target for AC4C only
        target_12 = batch.y[:, AC4C_CLASS_IDX].float()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                if use_hierarchical:
                    logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)
                    logits_12 = logits_12.view(-1)
                    logits_4 = logits_4.view(-1)
                    
                    loss_12 = criterion(logits_12, target_12)
                    target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                    loss = loss_12 + loss_4
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    logits = logits.view(-1)
                    loss = criterion(logits, target_12)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_hierarchical:
                logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)
                logits_12 = logits_12.view(-1)
                logits_4 = logits_4.view(-1)
                
                loss_12 = criterion(logits_12, target_12)
                target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                loss = loss_12 + loss_4
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
                logits = logits.view(-1)
                loss = criterion(logits, target_12)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    return total_loss / num_batches

def test_epoch(model, dataloader, criterion, device, phase, logger, use_hierarchical, use_amp):
    """
    Local test_epoch that handles the Wrapped model correctly.
    Calculates loss only on the relevant class to avoid noise from padded values.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            target_12 = batch.y[:, AC4C_CLASS_IDX].float()
            
            # If using wrapper, we can access the internal pruned model for efficiency/correctness in loss
            if isinstance(model, PrunedModelWrapper):
                internal_model = model.model
            else:
                internal_model = model
                
            if use_hierarchical:
                logits_12, logits_4 = internal_model(batch.x, batch.edge_index, batch.batch)
                logits_12 = logits_12.view(-1)
                logits_4 = logits_4.view(-1)
                
                loss_12 = criterion(logits_12, target_12)
                target_4 = batch.y_4class[:, AC4C_GROUP_IDX].float()
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, target_4)
                loss = loss_12 + loss_4
            else:
                logits = internal_model(batch.x, batch.edge_index, batch.batch)
                logits = logits.view(-1)
                loss = criterion(logits, target_12)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches

def main(config_path='json/ac4c_unbalan.json', checkpoint_path=None):
    """
    ac4C 非平衡数据集训练主入口 / ac4C unbalanced-dataset training main entry.

    流程: 加载配置 -> 加载 ac4C 非平衡数据 -> 剪枝模型仅保留 ac4C 头 ->
    自定义训练循环 -> 按 `test_interval` 周期性评估 -> 保存最优 checkpoint。
    Pipeline: load config -> load ac4C unbalanced data -> prune model to
    ac4C-only head -> custom train loop -> periodic eval -> save best checkpoint.

    Args / 参数:
        config_path (str, optional): [中文] 训练配置文件路径 / [English] training config path.
            Defaults to 'json/ac4c_unbalan.json'.
        checkpoint_path (str, optional): [中文] 预训练权重路径, 用于热启动 /
            [English] pretrained checkpoint path for warm start. Defaults to None.

    Called by / 被调用:
        - __main__ 块: [中文] 通过 argparse 解析参数后调用 / [English] called from CLI after argparse.
    """

    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name)
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_ac4c_pruned')

    logger.info(f"\n{'='*60}")
    logger.info("AC4C Unbalanced Dataset Training (PRUNED MODE)")
    logger.info(f"{'='*60}")
    logger.info(f"Target Class: Index {AC4C_CLASS_IDX} (ac4c)")
    
    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Data directory
    ac4c_data_dir = 'npy/ac4c_processed/unbalanced_ac4c'

    # Load datasets
    logger.info(f"Loading AC4C Datasets...")
    train_dataset = AC4CDataset(mode='train', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir, use_cache=True, preload_cache=True)
    test_dataset = AC4CDataset(mode='test', data_dir=ac4c_data_dir, cache_dir=Config.data.cache_dir, use_cache=True, preload_cache=True)
    
    # Calculate weights based on FULL 12 classes first
    train_indices = list(range(len(train_dataset)))
    pos_weight_full = get_smoothed_pos_weights(
        train_dataset, train_indices, num_classes=12, logger=logger
    )
    
    # Extract only the weight for AC4C
    pos_weight_ac4c = pos_weight_full[AC4C_CLASS_IDX].view(1) # Shape (1,)
    logger.info(f"Pruned Pos Weight for AC4C: {pos_weight_ac4c.item():.4f}")

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
        num_classes=Config.num_classes, # Init with 12
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=getattr(Config, 'use_hierarchical', False),
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    # PRUNE THE MODEL HEADS
    logger.info(f"\n{'!'*60}")
    logger.info(f"PRUNING MODEL HEADS TO AC4C ONLY")
    logger.info(f"{'!'*60}")
    model.prune_heads(
        valid_class_indices=[AC4C_CLASS_IDX], 
        valid_group_indices=[AC4C_GROUP_IDX] if getattr(Config, 'use_hierarchical', False) else None
    )

    # Load checkpoint if exists (Handle strict loading manually if needed)
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=Config.device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("Loaded state dict (strict=False)")
        except Exception as e:
            logger.warning(f"Could not load checkpoint directly: {e}")

    # Wrap the pruned model for evaluation compatibility
    wrapped_model = PrunedModelWrapper(model)

    # Loss function for single class
    pos_weight_ac4c = pos_weight_ac4c.to(Config.device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ac4c)
    criterion = LabelSmoothingLoss(smoothing=0.1).to(Config.device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, Config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{Config.num_epochs}")
        
        # Train: Use raw pruned model for efficiency
        train_loss = train_epoch_pruned(
            model, train_loader, criterion, optimizer, scheduler, Config.device, logger,
            use_hierarchical=getattr(Config, 'use_hierarchical', False),
            use_amp=Config.use_amp and torch.cuda.is_available()
        )
        
        tb_writer.add_scalar('train/loss', train_loss, epoch)

        # Verification Logic (Preserved from user request)
        should_eval = (epoch % Config.test_interval == 0)
        if should_eval:
            # Evaluate
            logger.info(f"\nEvaluating...")
            # Note: Passing wrapped_model here to handle interface compatibility
            test_loss = test_epoch(
                wrapped_model, test_loader, criterion, Config.device, "test", logger,
                use_hierarchical=getattr(Config, 'use_hierarchical', False),
                use_amp=Config.use_amp and torch.cuda.is_available()
            )

            # Log test loss to tensorboard
            tb_writer.add_scalar('test/loss', test_loss, epoch)

            # Get predictions
            logger.info(f"\n  Getting predictions for test data...")
            y_true, y_prob, y_4class, y_4prob = get_all_predictions(wrapped_model, test_loader, Config.device, getattr(Config, 'use_hierarchical', False))

            # Import AC4C evaluation function
            from utils.metrics import evaluate_ac4c
            metrics_unbalance = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                             dataset_mode='unbalanced', y_4prob=y_4prob)
            metrics_balanced = evaluate_ac4c(y_true, y_prob, y_4class, Config.device, Config.random_seed,
                                           dataset_mode='balanced', y_4prob=y_4prob)

            # Print results
            def print_ac4c_table(metrics, title, logger):
                """
                打印 ac4C 评估结果表 / Pretty-print the ac4C evaluation table.

                使用 PrettyTable 输出 ac4C 单类的 F1/Prec/Rec/Acc/AUC/AUPRC/Sn/Sp
                以及混淆矩阵, 同时附 macro 平均行。
                Emits F1/Prec/Rec/Acc/AUC/AUPRC/Sn/Sp and confusion matrix for
                the ac4C class via PrettyTable, with a macro-average row.

                Args / 参数:
                    metrics (dict): [中文] `evaluate_ac4c` 返回的指标字典 / [English] metrics dict
                        returned by `evaluate_ac4c`.
                    title (str): [中文] 表标题, 含 "Unbalanced" / "Balanced" 标识 /
                        [English] table title indicating "Unbalanced" or "Balanced".
                    logger (logging.Logger): [中文] 日志记录器 / [English] logger instance.
                """

                table = PrettyTable()
                table.field_names = ["Class", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
                table.align = "r"
                table.align["Class"] = "l"
                
                # Get metrics for AC4C class (index 6)
                c = AC4C_CLASS_IDX
                mod_name = "ac4c"
                
                # Extract metrics
                f1 = metrics.get(f'group_ac4c_class_{c}_opt_f1', 0.0)
                prec = metrics.get(f'group_ac4c_class_{c}_opt_precision', 0.0)
                rec = metrics.get(f'group_ac4c_class_{c}_opt_recall', 0.0)
                acc = metrics.get(f'group_ac4c_class_{c}_opt_accuracy', 0.0)
                auc = metrics.get(f'group_ac4c_class_{c}_auc', 0.0)
                auprc = metrics.get(f'group_ac4c_class_{c}_auprc', 0.0)
                sn = metrics.get(f'group_ac4c_class_{c}_opt_sensitivity', 0.0)
                sp = metrics.get(f'group_ac4c_class_{c}_opt_specificity', 0.0)
                
                tp = int(metrics.get(f'group_ac4c_class_{c}_opt_tp', 0))
                tn = int(metrics.get(f'group_ac4c_class_{c}_opt_tn', 0))
                fp = int(metrics.get(f'group_ac4c_class_{c}_opt_fp', 0))
                fn = int(metrics.get(f'group_ac4c_class_{c}_opt_fn', 0))
                
                # Add row
                table.add_row([
                    f"{c}({mod_name})",
                    f"{f1:.4f}",
                    f"{prec:.4f}",
                    f"{rec:.4f}",
                    f"{acc:.4f}",
                    f"{auc:.4f}",
                    f"{auprc:.4f}",
                    f"{sn:.4f}",
                    f"{sp:.4f}",
                    tp, tn, fp, fn
                ])
                
                # Add macro average row
                macro_f1 = metrics.get(f'group_ac4c_opt_macro_f1', 0.0)
                macro_p = metrics.get(f'group_ac4c_opt_macro_precision', 0.0)
                macro_r = metrics.get(f'group_ac4c_opt_macro_recall', 0.0)
                table.add_row(["Avg", f"{macro_f1:.4f}", f"{macro_p:.4f}", f"{macro_r:.4f}", "-", "-", "-", "-", "-", "-", "-", "-"])
                
                # Print table
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch} - {title}")
                logger.info(f"{'='*80}")
                logger.info(table)
            
            # Print tables
            print_ac4c_table(metrics_unbalance, "AC4C Unbalanced Dataset Test Results", logger)
            print_ac4c_table(metrics_balanced, "AC4C Balanced Dataset Test Results", logger)
            
            # Also print the brief summary as before
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch} - AC4C Test Results Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Unbalance Mode - Macro F1: {metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_unbalance.get('group_ac4c_micro_f1', 0.0):.4f}")
            logger.info(f"Balance Mode - Macro F1: {metrics_balanced.get('group_ac4c_opt_macro_f1', 0.0):.4f}, "
                       f"Micro F1: {metrics_balanced.get('group_ac4c_micro_f1', 0.0):.4f}")

            # Log metrics to tensorboard
            log_metrics_to_tensorboard(tb_writer, metrics_unbalance, 'test_ac4c_unbalanced_unbalance', epoch, key_prefix='group_ac4c_')
            log_metrics_to_tensorboard(tb_writer, metrics_balanced, 'test_ac4c_unbalanced_balance', epoch, key_prefix='group_ac4c_')

            # Run Few-Shot Benchmark
            logger.info(f"\n>>> Running AC4C Unbalanced Few-Shot Benchmark <<<")


            # Check for best model
            current_macro_f1 = metrics_unbalance.get('group_ac4c_opt_macro_f1', 0.0)
            logger.info(f"Current Macro-F1: {current_macro_f1:.4f}")
            logger.info(f"Best Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")
            
            # Check best
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, metrics_unbalance, 
                              os.path.join(Config.checkpoint_dir, 'ac4c_pruned_best.pt'), logger, config_dict)

    logger.info(f"Training Complete. Best F1: {best_macro_f1:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='json/ac4c_unbalan.json')
    parser.add_argument('--checkpoint', type=str, default="logs/rna_classification_20260111_111223/checkpoints/best_model.pt")
    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
