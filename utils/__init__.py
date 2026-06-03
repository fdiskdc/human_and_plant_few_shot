"""
utils/__init__.py - Utils 包初始化与符号重导出 / Utils Package Init & Symbol Re-export

RGCNFormer 项目的 utils 包入口:从所有 utils 子模块重导出公共符号供外部直接导入。
Utils package entry: re-exports all public symbols from utils submodules for direct external import.

功能模块 / Modules:
- 符号重导出 / Symbol re-export
- 包级文档 / Package-level documentation

输入 / Inputs:
- 无 (包初始化) / None (package init)

输出 / Outputs:
- 可导入符号: setup_logging, load_config, save_checkpoint, etc.
- 所有 utils 公共 API / All utils public API

数据流 / Data Flow:
1. 导入子模块 / Import submodules
2. 重导出符号 / Re-export symbols

相关文件 / Related Files:
- 调用 / Calls: utils.common, utils.metrics, utils.logging, utils.few_shot 等
- 被调用 / Called by: 几乎所有项目脚本 (from utils import ...)

使用示例 / Usage Example:
    from utils import setup_logging, load_config, save_checkpoint, MOD_NAMES

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

# Import all constants
from .common import (
    MOD_NAMES,
    NUCLEOTIDE_GROUP_NAMES,
    INDEX_TO_NUCLEOTIDE,
    NUCLEOTIDE_GROUPS,
    GROUP_TO_INDEX,
    INDEX_TO_GROUP,
    GROUP_SIZES,
    GROUP_TO_CLASS_INDICES,
    PLANT_VALID_CLASS_INDICES,
    get_center_nucleotide
)

# Import configuration and checkpoint functions
from .common import (
    load_config,
    save_checkpoint,
    load_checkpoint
)

# Import sampler classes and split functions (now in common.py)
from .common import (
    MultilabelBalancedBatchSampler,
    DynamicBalancedBatchSampler,
    get_smoothed_pos_weights,
    multi_label_disjoint_split
)

# Import metrics functions (including group-based metrics)
from .metrics import (
    calculate_metrics,
    find_optimal_threshold,
    evaluate_unbalance,
    evaluate_balanceb,
    evaluate_balanceb_th,
    evaluate_with_optimal_threshold,
    evaluate_4class_with_optimal_threshold,
    evaluate_plant_unbalance,
    evaluate_plant_balanceb,
    evaluate_group_balanceb,
    get_all_predictions,
    get_all_predictions_and_attention
)

# Import logging functions
from .logging import (
    setup_logging,
    setup_tensorboard,
    log_metrics_to_tensorboard,
    print_evaluation_results,
    print_few_shot_results
)

# Import training functions
from .common import (
    train_epoch,
    test_epoch,
    compute_attention_supervision_loss,
    calculate_topk_recall,
    print_topk_table,
    calculate_comprehensive_localization_metrics,
    print_comprehensive_table
)

# Import few-shot learning functions
from .few_shot import (
    run_few_shot_benchmark,
    run_few_shot_benchmark_ac4c,
    apply_advanced_augmentation
)

__all__ = [
    # Constants
    'MOD_NAMES',
    'NUCLEOTIDE_GROUP_NAMES',
    'INDEX_TO_NUCLEOTIDE',
    'NUCLEOTIDE_GROUPS',
    'GROUP_TO_INDEX',
    'INDEX_TO_GROUP',
    'GROUP_SIZES',
    'GROUP_TO_CLASS_INDICES',
    'PLANT_VALID_CLASS_INDICES',
    'get_center_nucleotide',

    # Common utilities
    'load_config',
    'save_checkpoint',
    'load_checkpoint',

    # Samplers
    'MultilabelBalancedBatchSampler',
    'DynamicBalancedBatchSampler',
    'get_smoothed_pos_weights',

    # Split
    'multi_label_disjoint_split',

    # Metrics
    'calculate_metrics',
    'find_optimal_threshold',
    'evaluate_unbalance',
    'evaluate_balanceb',
    'evaluate_balanceb_th',
    'evaluate_with_optimal_threshold',
    'evaluate_4class_with_optimal_threshold',
    'evaluate_plant_unbalance',
    'evaluate_plant_balanceb',
    'evaluate_group_balanceb',
    'get_all_predictions',
    'get_all_predictions_and_attention',

    # Training
    'train_epoch',
    'test_epoch',
    'compute_attention_supervision_loss',
    'calculate_topk_recall',
    'print_topk_table',
    'calculate_comprehensive_localization_metrics',
    'print_comprehensive_table',

    # Few-shot learning
    'run_few_shot_benchmark',
    'run_few_shot_benchmark_ac4c',
    'apply_advanced_augmentation',

    # Logging
    'setup_logging',
    'setup_tensorboard',
    'log_metrics_to_tensorboard',
    'print_evaluation_results',
    'print_few_shot_results',
]
