"""
utils/logging.py - 日志配置与格式化输出 / Logging Configuration & Formatted Output

日志设置 (文件 + 控制台)、TensorBoard 设置与记录、PrettyTable 格式化的评估/小样本结果输出。
Logger setup (file + console), TensorBoard setup and logging, PrettyTable formatted evaluation/few-shot result output.

功能模块 / Modules:
- setup_logging: 文件+控制台日志设置 / File+console logger setup
- setup_tensorboard: TensorBoard SummaryWriter / TensorBoard writer
- log_metrics_to_tensorboard: 记录指标到 TB / Log metrics to TB
- print_evaluation_results: 评估结果 PrettyTable / Evaluation PrettyTable
- print_few_shot_results: 小样本结果 PrettyTable / Few-shot PrettyTable
- print_comprehensive_table: 综合表输出 / Comprehensive table output
- print_topk_table: Top-K 表输出 / Top-K table output
- logger 配置类 / Logger config classes

输入 / Inputs:
- 日志目录 / Log directory
- 评估结果字典 / Evaluation result dict
- PrettyTable 样式 / PrettyTable style

输出 / Outputs:
- train_*.log 文件 / train log file
- TensorBoard events 文件 / TensorBoard event files
- 终端 PrettyTable / Terminal PrettyTable

数据流 / Data Flow:
1. 设置 logger / Setup logger
2. 设置 TB writer / Setup TB writer
3. 训练时记录 / Record during training
4. 评估时输出 / Output during evaluation

相关文件 / Related Files:
- 调用 / Calls: logging, tensorboardX, prettytable
- 被调用 / Called by: train_*.py, test_*.py, evaluation scripts

使用示例 / Usage Example:
    from utils import setup_logging, setup_tensorboard
    logger, log_file = setup_logging(log_dir='logs/')
    writer = setup_tensorboard(log_dir='logs/tb/')

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import logging
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from typing import Dict, Optional

# Import constants from common
from utils.common import MOD_NAMES, INDEX_TO_NUCLEOTIDE


def setup_logging(log_dir: str, experiment_name: str = 'rna_classification') -> logging.Logger:
    """
    Setup logging configuration to record all training/testing outputs.

    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment for the log file

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{experiment_name}.log')

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_tensorboard(log_dir: str, experiment_name: str = 'rna_classification') -> SummaryWriter:
    """
    Setup Tensorboard writer for logging metrics.

    Args:
        log_dir: Directory to save tensorboard logs
        experiment_name: Name of the experiment

    Returns:
        SummaryWriter instance
    """
    tb_dir = os.path.join(log_dir, 'tensorboard', experiment_name)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(tb_dir)
    return writer


def log_metrics_to_tensorboard(writer: SummaryWriter, metrics: Dict, phase: str,
                                epoch: int, class_metrics: bool = True, key_prefix: str = ''):
    """
    Log metrics to Tensorboard.

    Args:
        writer: SummaryWriter instance
        metrics: Dictionary of metrics
        phase: 'train' or 'test'
        epoch: Current epoch number
        class_metrics: Whether to log per-class metrics
        key_prefix: Prefix for metric keys (e.g., 'plant_' for plant metrics)
    """
    # Helper function to get metric key with optional prefix
    def get_key(base_key):
        """
        拼接带前缀的指标键 / Compose metric key with optional prefix.

        Args / 参数:
            base_key (str): [中文] 基础键名 / [English] base key name.

        Returns / 返回:
            str: [中文] `key_prefix + base_key` / [English] `key_prefix + base_key`.
        """

        if key_prefix:
            return f'{key_prefix}{base_key}'
        else:
            return f'group_{base_key}'
    
    # Helper function to safely get metric value
    def get_metric(base_key: str, default=0.0):
        """
        从 metrics 字典中安全取值 / Safely fetch a metric from the dict.

        Args / 参数:
            base_key (str): [中文] 基础键名 / [English] base key.
            default (Any, optional): [中文] 缺省值 / [English] default value. Defaults to 0.0.

        Returns / 返回:
            Any: [中文] 命中则返回对应值, 否则 `default` /
                [English] the value if the key exists, else `default`.
        """

        key = get_key(base_key)
        if key in metrics:
            return metrics[key]
        return default

    # Overall metrics
    for avg in ['macro', 'micro', 'weighted']:
        # Try with _opt suffix first, then without
        key_opt = get_key(f'opt_{avg}_f1')
        key_std = get_key(f'{avg}_f1')
        val = metrics.get(key_opt, metrics.get(key_std, 0.0))
        writer.add_scalar(f'{phase}/{avg}_f1', val, epoch)
        
        key_opt = get_key(f'opt_{avg}_precision')
        key_std = get_key(f'{avg}_precision')
        val = metrics.get(key_opt, metrics.get(key_std, 0.0))
        writer.add_scalar(f'{phase}/{avg}_precision', val, epoch)
        
        key_opt = get_key(f'opt_{avg}_recall')
        key_std = get_key(f'{avg}_recall')
        val = metrics.get(key_opt, metrics.get(key_std, 0.0))
        writer.add_scalar(f'{phase}/{avg}_recall', val, epoch)

    # Overall accuracy
    accuracy_key = get_key('opt_accuracy') if key_prefix == '' else get_key('opt_accuracy')
    if accuracy_key not in metrics:
        accuracy_key = get_key('accuracy')
    if accuracy_key in metrics:
        writer.add_scalar(f'{phase}/accuracy', metrics[accuracy_key], epoch)

    # Per-class metrics
    if class_metrics:
        num_classes = 12
        for c in range(num_classes):
            # Try to get keys with and without _opt_ suffix
            # Priority: _opt_ version first, then standard version
            for metric_name in ['f1', 'precision', 'recall', 'auc', 'auprc', 'mcc', 'sensitivity', 'specificity']:
                key_opt = get_key(f'class_{c}_opt_{metric_name}')
                key_std = get_key(f'class_{c}_{metric_name}')
                
                if key_opt in metrics:
                    writer.add_scalar(f'{phase}/class_{c}_{metric_name}', metrics[key_opt], epoch)
                elif key_std in metrics:
                    writer.add_scalar(f'{phase}/class_{c}_{metric_name}', metrics[key_std], epoch)


def create_evaluation_table(metrics: Dict, table_type: str = "12class", 
                       mode: str = "unbalance", dataset: str = "Human",
                       only_classes: Optional[list] = None,
                       use_balanced_prefix: bool = False) -> PrettyTable:
    """
    Create a formatted table for evaluation results.

    Args:
        metrics: Dictionary of metrics
        table_type: '12class' or '4class'
        mode: 'unbalance' or 'balanceb'
        dataset: 'Human' or 'Plant'
        only_classes: Optional list of class indices to include (for Plant filtering)
        use_balanced_prefix: If True, use 'group_balanced_' prefix for Group Balanced metrics

    Returns:
        PrettyTable object
    """
    # Import INDEX_TO_GROUP for 4-class evaluation
    from utils.common import INDEX_TO_GROUP
    
    # Determine prefix
    # For Group Balanced metrics, use 'group_balanced_' prefix
    if use_balanced_prefix and dataset == "Human":
        prefix = "group_balanced_"
    else:
        prefix = "group_plant_" if dataset == "Plant" else "group_"
    
    # Determine if using optimal threshold metrics
    # For unbalance mode, metrics have "_opt" prefix (e.g., "group_class_0_opt_tp")
    # For balanceb mode, metrics have "_opt" prefix (e.g., "group_class_0_opt_tp")
    # So _opt goes before the metric name (tp, tn, fp, fn, accuracy, etc.)
    
    # Helper function to safely get metric value
    def get_metric(base_key: str, default=0.0):
        # The key format in metrics.py is: {prefix}{class/group}_{opt}_{metric}
        # e.g., group_class_0_opt_tp, group_4class_0_opt_tp
        # So we need to insert "_opt" before the metric name

        # For class metrics like "class_0_tp" -> "class_0_opt_tp"
        # For 4class metrics like "4class_0_tp" -> "4class_0_opt_tp"

        # Try with _opt first (format: class_{c}_opt_{metric})
        """
        从 metrics 中按 `{prefix}class_{c}_opt_{metric}` / `{prefix}4class_{c}_opt_{metric}`
        形式取值, 失败则回退到无 `_opt` 形式, 最终回退到 `default` /
        Fetch a metric via `{prefix}class_{c}_opt_{metric}` /
        `{prefix}4class_{c}_opt_{metric}`, falling back to the non-`_opt` form
        and finally to `default`.

        Args / 参数:
            base_key (str): [中文] 基础键名 (如 "class_0_tp" 或 "4class_0_accuracy") /
                [English] base key (e.g. "class_0_tp" or "4class_0_accuracy").
            default (Any, optional): [中文] 缺省值 / [English] default value. Defaults to 0.0.

        Returns / 返回:
            Any: [中文] 命中值或 `default` / [English] matched value or `default`.
        """

        if "class_" in base_key:
            parts = base_key.split("_")
            # parts will be ["class", "0", "tp"] or ["class", "0", "accuracy"]
            if len(parts) >= 3:
                metric_name = parts[-1]  # "tp", "accuracy", etc.
                prefix_part = "_".join(parts[:-1])  # "class_0"
                key_opt = f'{prefix}{prefix_part}_opt_{metric_name}'
                if key_opt in metrics:
                    return metrics[key_opt]
        elif "4class_" in base_key:
            parts = base_key.split("_")
            # parts will be ["4class", "0", "tp"]
            if len(parts) >= 3:
                metric_name = parts[-1]
                prefix_part = "_".join(parts[:-1])  # "4class_0"
                key_opt = f'{prefix}{prefix_part}_opt_{metric_name}'
                if key_opt in metrics:
                    return metrics[key_opt]
        
        # Try without _opt suffix (fallback)
        key_std = f'{prefix}{base_key}'
        if key_std in metrics:
            return metrics[key_std]
        return default
    
    # Create table
    table = PrettyTable()
    
    if table_type == "12class":
        table.field_names = [
            "Class", "Mod Name", "Acc", "AUC", "AUPRC", "Precision", 
            "Recall", "F1", "MCC", "Sn", "Sp", "TP", "TN", "FP", "FN", "Optimal_Threshold"
        ]
        table.align = "r"
        table.align["Class"] = "l"
        table.align["Mod Name"] = "l"
        
        # Determine classes to iterate
        if only_classes is not None:
            class_indices = only_classes
        else:
            class_indices = range(12)
        
        for c in class_indices:
            mod_name = MOD_NAMES.get(c, f"Class_{c}")
            
            # Get metrics
            acc = get_metric(f'class_{c}_accuracy', 0.0)
            auc = get_metric(f'class_{c}_auc', 0.0)
            auprc = get_metric(f'class_{c}_auprc', 0.0)
            precision = get_metric(f'class_{c}_precision', 0.0)
            recall = get_metric(f'class_{c}_recall', 0.0)
            f1 = get_metric(f'class_{c}_f1', 0.0)
            mcc = get_metric(f'class_{c}_mcc', 0.0)
            sensitivity = get_metric(f'class_{c}_sensitivity', 0.0)
            specificity = get_metric(f'class_{c}_specificity', 0.0)
            tp = int(get_metric(f'class_{c}_tp', 0))
            tn = int(get_metric(f'class_{c}_tn', 0))
            fp = int(get_metric(f'class_{c}_fp', 0))
            fn = int(get_metric(f'class_{c}_fn', 0))
            threshold = get_metric(f'class_{c}_opt_threshold', 0.5)
            
            table.add_row([
                f"{c} ({mod_name})",
                mod_name,
                f"{acc:.4f}",
                f"{auc:.4f}",
                f"{auprc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{mcc:.4f}",
                f"{sensitivity:.4f}",
                f"{specificity:.4f}",
                tp, tn, fp, fn,
                f"{threshold:.3f}"
            ])
    
    elif table_type == "4class":
        table.field_names = [
            "Class", "Nucleotide", "Acc", "AUC", "AUPRC", "Precision", 
            "Recall", "F1", "MCC", "Sn", "Sp", "TP", "TN", "FP", "FN", "Optimal_Threshold"
        ]
        table.align = "r"
        table.align["Class"] = "l"
        table.align["Nucleotide"] = "l"
        
        for g in range(4):
            # Use INDEX_TO_GROUP for 4-class indices (0-3) -> nucleotides (A, C, G, U)
            group_name = INDEX_TO_GROUP[g]
            
            # Get metrics
            acc = get_metric(f'4class_{g}_accuracy', 0.0)
            auc = get_metric(f'4class_{g}_auc', 0.0)
            auprc = get_metric(f'4class_{g}_auprc', 0.0)
            precision = get_metric(f'4class_{g}_precision', 0.0)
            recall = get_metric(f'4class_{g}_recall', 0.0)
            f1 = get_metric(f'4class_{g}_f1', 0.0)
            mcc = get_metric(f'4class_{g}_mcc', 0.0)
            sensitivity = get_metric(f'4class_{g}_sensitivity', 0.0)
            specificity = get_metric(f'4class_{g}_specificity', 0.0)
            tp = int(get_metric(f'4class_{g}_tp', 0))
            tn = int(get_metric(f'4class_{g}_tn', 0))
            fp = int(get_metric(f'4class_{g}_fp', 0))
            fn = int(get_metric(f'4class_{g}_fn', 0))
            threshold = get_metric(f'4class_{g}_opt_threshold', 0.5)
            
            table.add_row([
                f"{g}",
                group_name,
                f"{acc:.4f}",
                f"{auc:.4f}",
                f"{auprc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{mcc:.4f}",
                f"{sensitivity:.4f}",
                f"{specificity:.4f}",
                tp, tn, fp, fn,
                f"{threshold:.3f}"
            ])
    
    return table


def print_evaluation_results(
    metrics_unbalance: Dict,
    metrics_balanceb: Dict,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    metrics_opt: Optional[Dict] = None,
    plant_metrics_unbalance: Optional[Dict] = None,
    plant_metrics_balanceb: Optional[Dict] = None,
    metrics_4class: Optional[Dict] = None,
    plant_metrics_4class: Optional[Dict] = None,
    plant_metrics_opt: Optional[Dict] = None,
    metrics_group_balanceb: Optional[Dict] = None
) -> None:
    """
    Print and log evaluation results in formatted tables.
    
    Generates 9 tables:
    1. Human - 12 Class (Unbalanced)
    2. Human - 4 Class (Unbalanced)
    3. Human - 12 Class (Balanced)
    4. Human - 4 Class (Balanced)
    5. Group Human - 12 Class (Balanced) - NEW: negative samples from same group only
    6. Plant - 12 Class (Unbalanced) - filtered to [5,8,9]
    7. Plant - 4 Class (Unbalanced)
    8. Plant - 12 Class (Balanced) - filtered to [5,8,9]
    9. Plant - 4 Class (Balanced)

    Args:
        metrics_unbalance: Metrics from Unbalance mode (human)
        metrics_balanceb: Metrics from BalanceB mode (human)
        epoch: Current epoch number
        logger: Optional logger instance
        metrics_opt: Optional metrics from optimal threshold evaluation (human)
        plant_metrics_unbalance: Optional metrics from Unbalance mode (plant)
        plant_metrics_balanceb: Optional metrics from BalanceB mode (plant)
        metrics_4class: Optional metrics from 4-class evaluation (human)
        plant_metrics_4class: Optional metrics from 4-class evaluation (plant)
        plant_metrics_opt: Optional metrics from optimal threshold evaluation (plant)
        metrics_group_balanceb: Optional metrics from Group Balanced mode (human) - NEW
    """
    
    output = f"\n{'='*120}\n"
    output += f"Evaluation Results - Epoch {epoch}\n"
    output += f"{'='*120}\n"

    # Table 0: Human - 12 Class (Optimal Threshold)
    if metrics_opt:
        output += f"\n### Table 0: Human - 12 Class (Optimal Threshold) ###\n"
        table = create_evaluation_table(metrics_opt, "12class", "unbalance", "Human")
        output += str(table) + "\n"
        output += f"Human - Optimal Threshold Macro F1: {metrics_opt.get('group_opt_macro_f1', 0.0):.4f}\n"

    # Table 1: Human - 12 Class (Unbalanced)
    if metrics_unbalance:
        output += f"\n### Table 1: Human - 12 Class (Unbalanced/Real Distribution) ###\n"
        table = create_evaluation_table(metrics_unbalance, "12class", "unbalance", "Human")
        output += str(table) + "\n"
        output += f"Human - Macro F1: {metrics_unbalance.get('group_opt_macro_f1', 0.0):.4f}\n"

    # Table 2: Human - 4 Class (Unbalanced)
    if metrics_4class:
        output += f"\n### Table 2: Human - 4 Class (Unbalanced/Real Distribution) ###\n"
        table = create_evaluation_table(metrics_4class, "4class", "unbalance", "Human")
        output += str(table) + "\n"
        output += f"Human - 4-Class Macro F1: {metrics_4class.get('group_4class_opt_macro_f1', 0.0):.4f}\n"

    # Table 3: Human - 12 Class (Balanced)
    if metrics_balanceb:
        output += f"\n### Table 3: Human - 12 Class (Balanced) ###\n"
        table = create_evaluation_table(metrics_balanceb, "12class", "balanceb", "Human")
        output += str(table) + "\n"
        output += f"Human - Macro F1: {metrics_balanceb.get('group_macro_f1', 0.0):.4f}\n"

    # Table 4: Human - 4 Class (Balanced)
    if metrics_balanceb:
        output += f"\n### Table 4: Human - 4 Class (Balanced) ###\n"
        table = create_evaluation_table(metrics_balanceb, "4class", "balanceb", "Human")
        output += str(table) + "\n"
        output += f"Human - 4-Class Macro F1: {metrics_balanceb.get('group_4class_macro_f1', 0.0):.4f}\n"

    # Table 5: Group Human - 12 Class (Balanced) - NEW: negative samples from same group only
    if metrics_group_balanceb:
        output += f"\n### Table 5: Group Human - 12 Class (Balanced) - Negative samples from same nucleotide group ###\n"
        table = create_evaluation_table(metrics_group_balanceb, "12class", "balanceb", "Human", use_balanced_prefix=True)
        output += str(table) + "\n"
        output += f"Group Human - Macro F1: {metrics_group_balanceb.get('group_balanced_opt_macro_f1', 0.0):.4f}\n"

    # Table 6: Plant - 12 Class (Unbalanced) - Filtered to [5, 8, 9]
    if plant_metrics_unbalance:
        output += f"\n### Table 5: Plant - 12 Class (Unbalanced) - Only Valid Classes [5, 8, 9] ###\n"
        table = create_evaluation_table(plant_metrics_unbalance, "12class", "unbalance", "Plant", only_classes=[5, 8, 9])
        output += str(table) + "\n"
        output += f"Plant - Macro F1 (filtered): {plant_metrics_unbalance.get('group_plant_opt_macro_f1', 0.0):.4f}\n"

    # Table 6: Plant - 4 Class (Unbalanced)
    if plant_metrics_4class:
        output += f"\n### Table 6: Plant - 4 Class (Unbalanced/Real Distribution) ###\n"
        table = create_evaluation_table(plant_metrics_4class, "4class", "unbalance", "Plant")
        output += str(table) + "\n"
        output += f"Plant - 4-Class Macro F1: {plant_metrics_4class.get('group_plant_4class_opt_macro_f1', 0.0):.4f}\n"

    # Table 7: Plant - 12 Class (Balanced) - Filtered to [5, 8, 9]
    if plant_metrics_balanceb:
        output += f"\n### Table 7: Plant - 12 Class (Balanced) - Only Valid Classes [5, 8, 9] ###\n"
        table = create_evaluation_table(plant_metrics_balanceb, "12class", "balanceb", "Plant", only_classes=[5, 8, 9])
        output += str(table) + "\n"
        output += f"Plant - Macro F1 (filtered): {plant_metrics_balanceb.get('group_plant_opt_macro_f1', 0.0):.4f}\n"

    # Table 8: Plant - 4 Class (Balanced)
    if plant_metrics_balanceb:
        output += f"\n### Table 8: Plant - 4 Class (Balanced) ###\n"
        table = create_evaluation_table(plant_metrics_balanceb, "4class", "balanceb", "Plant")
        output += str(table) + "\n"
        output += f"Plant - 4-Class Macro F1: {plant_metrics_balanceb.get('group_plant_4class_opt_macro_f1', 0.0):.4f}\n"

    output += f"{'='*120}\n"

    # Print to console
    print(output)

    # Log to file if logger is available
    if logger:
        logger.info(output)


def print_few_shot_results(all_shot_metrics, epoch, logger=None):
    """
    Print summary tables for Few-Shot results with extended metrics.
    all_shot_metrics structure: {k_shot: {'unbalance': metrics, 'balanceb': metrics}}

    Extended metrics include: TP, TN, FP, FN, AUPRC, Sn (Sensitivity), Sp (Specificity)

    Args:
        all_shot_metrics (dict): Dictionary of metrics for each shot count
            Format: {k_shot: {'unbalance': metrics_dict, 'balanceb': metrics_dict}}
        epoch (int): Current epoch number
        logger: Optional logger instance for logging to file
    """
    from prettytable import PrettyTable
    from utils.common import MOD_NAMES

    output = f"\n{'='*150}\n"  # Increased width for more columns
    output += f"Plant Few-Shot Adaptation Results - Epoch {epoch}\n"
    output += f"{'='*150}\n"

    # Define table configurations
    table_configs = [
        {
            "title": "Table 5+: Plant - 12 Class (Unbalanced) - Valid Classes [5, 8, 9] per Shot",
            "metric_type": "unbalance",
            "prefix": "group_plant_"
        },
        {
            "title": "Table 7+: Plant - 12 Class (Balanced) - Valid Classes [5, 8, 9] per Shot",
            "metric_type": "balanceb",
            "prefix": "group_plant_"
        }
    ]
    valid_classes = [5, 8, 9]

    for config in table_configs:
        output += f"\n### {config['title']} ###\n"
        table = PrettyTable()
        # Extended table with additional metrics: TP, TN, FP, FN, AUC, AUPRC, Sn, Sp
        table.field_names = ["Shot", "Class", "F1", "Prec", "Rec", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
        table.align = "r"
        table.align["Class"] = "l"

        prefix = config['prefix']

        # Iterate through sorted shots
        for shot in sorted(all_shot_metrics.keys()):
            metrics = all_shot_metrics[shot][config['metric_type']]

            # Separator
            if shot != sorted(all_shot_metrics.keys())[0]:
                table.add_row(["-"*4, "-"*5, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*4, "-"*4, "-"*4, "-"*4])

            # Macro Average Row
            macro_f1 = metrics.get(f'{prefix}opt_macro_f1', 0.0)
            macro_p = metrics.get(f'{prefix}opt_macro_precision', 0.0)
            macro_r = metrics.get(f'{prefix}opt_macro_recall', 0.0)
            # Counts/AUC/AUPRC don't sum meaningfully for Macro Avg in this context, using placeholders
            table.add_row([f"{shot}-s", "Avg", f"{macro_f1:.4f}", f"{macro_p:.4f}", f"{macro_r:.4f}", "-", "-", "-", "-", "-", "-", "-", "-"])

            # Class Rows
            for c in valid_classes:
                mod_name = MOD_NAMES.get(c, f"C{c}")

                # Fetch metrics
                f1 = metrics.get(f'{prefix}class_{c}_opt_f1', 0.0)
                p = metrics.get(f'{prefix}class_{c}_opt_precision', 0.0)
                r = metrics.get(f'{prefix}class_{c}_opt_recall', 0.0)
                auc = metrics.get(f'{prefix}class_{c}_auc', 0.0)
                auprc = metrics.get(f'{prefix}class_{c}_auprc', 0.0)
                sn = metrics.get(f'{prefix}class_{c}_opt_sensitivity', 0.0)
                sp = metrics.get(f'{prefix}class_{c}_opt_specificity', 0.0)

                tp = int(metrics.get(f'{prefix}class_{c}_opt_tp', 0))
                tn = int(metrics.get(f'{prefix}class_{c}_opt_tn', 0))
                fp = int(metrics.get(f'{prefix}class_{c}_opt_fp', 0))
                fn = int(metrics.get(f'{prefix}class_{c}_opt_fn', 0))

                table.add_row([
                    "",
                    f"{c}({mod_name})",
                    f"{f1:.4f}",
                    f"{p:.4f}",
                    f"{r:.4f}",
                    f"{auc:.4f}",
                    f"{auprc:.4f}",
                    f"{sn:.4f}",
                    f"{sp:.4f}",
                    tp, tn, fp, fn
                ])

        output += str(table) + "\n"

    print(output)
    if logger:
        logger.info(output)


def print_few_shot_results_ac4c(all_shot_metrics, epoch, logger=None):
    """
    Print summary tables for Few-Shot results for ac4c dataset with extended metrics.
    all_shot_metrics structure: {k_shot: {'unbalance': metrics, 'balanceb': metrics}}

    Extended metrics include: TP, TN, FP, FN, AUPRC, Sn (Sensitivity), Sp (Specificity)

    Args:
        all_shot_metrics (dict): Dictionary of metrics for each shot count
            Format: {k_shot: {'unbalance': metrics_dict, 'balanceb': metrics_dict}}
        epoch (int): Current epoch number
        logger: Optional logger instance for logging to file
    """
    from prettytable import PrettyTable
    from utils.common import MOD_NAMES

    output = f"\n{'='*150}\n"  # Increased width for more columns
    output += f"ac4c Few-Shot Adaptation Results - Epoch {epoch}\n"
    output += f"{'='*150}\n"

    # Define table configurations
    table_configs = [
        {
            "title": "ac4c - 12 Class (Unbalanced) per Shot",
            "metric_type": "unbalance",
            "prefix": "group_"
        },
        {
            "title": "ac4c - 12 Class (Balanced) per Shot",
            "metric_type": "balanceb",
            "prefix": "group_"
        }
    ]
    valid_classes = [6]  # ac4C is class index 6 in the human dataset

    for config in table_configs:
        output += f"\n### {config['title']} ###\n"
        table = PrettyTable()
        # Extended table with additional metrics: TP, TN, FP, FN, AUC, AUPRC, Sn, Sp
        table.field_names = ["Shot", "Class", "F1", "Prec", "Rec", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
        table.align = "r"
        table.align["Class"] = "l"

        prefix = config['prefix']

        # Iterate through sorted shots
        for shot in sorted(all_shot_metrics.keys()):
            metrics = all_shot_metrics[shot][config['metric_type']]

            # Separator
            if shot != sorted(all_shot_metrics.keys())[0]:
                table.add_row(["-"*4, "-"*5, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*6, "-"*4, "-"*4, "-"*4, "-"*4])

            # Macro Average Row
            macro_f1 = metrics.get(f'{prefix}opt_macro_f1', 0.0)
            macro_p = metrics.get(f'{prefix}opt_macro_precision', 0.0)
            macro_r = metrics.get(f'{prefix}opt_macro_recall', 0.0)
            # Counts/AUC/AUPRC don't sum meaningfully for Macro Avg in this context, using placeholders
            table.add_row([f"{shot}-s", "Avg", f"{macro_f1:.4f}", f"{macro_p:.4f}", f"{macro_r:.4f}", "-", "-", "-", "-", "-", "-", "-", "-"])

            # Class Rows
            for c in valid_classes:
                mod_name = MOD_NAMES.get(c, f"C{c}")

                # Fetch metrics
                f1 = metrics.get(f'{prefix}class_{c}_opt_f1', 0.0)
                p = metrics.get(f'{prefix}class_{c}_opt_precision', 0.0)
                r = metrics.get(f'{prefix}class_{c}_opt_recall', 0.0)
                auc = metrics.get(f'{prefix}class_{c}_auc', 0.0)
                auprc = metrics.get(f'{prefix}class_{c}_auprc', 0.0)
                sn = metrics.get(f'{prefix}class_{c}_opt_sensitivity', 0.0)
                sp = metrics.get(f'{prefix}class_{c}_opt_specificity', 0.0)

                tp = int(metrics.get(f'{prefix}class_{c}_opt_tp', 0))
                tn = int(metrics.get(f'{prefix}class_{c}_opt_tn', 0))
                fp = int(metrics.get(f'{prefix}class_{c}_opt_fp', 0))
                fn = int(metrics.get(f'{prefix}class_{c}_opt_fn', 0))

                table.add_row([
                    "",
                    f"{c}({mod_name})",
                    f"{f1:.4f}",
                    f"{p:.4f}",
                    f"{r:.4f}",
                    f"{auc:.4f}",
                    f"{auprc:.4f}",
                    f"{sn:.4f}",
                    f"{sp:.4f}",
                    tp, tn, fp, fn
                ])

        output += str(table) + "\n"

    print(output)
    if logger:
        logger.info(output)
