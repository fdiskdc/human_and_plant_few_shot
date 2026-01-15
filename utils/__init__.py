"""
Utils package for RNA Multi-label Classification Training

This package contains helper functions for:
- Multi-label disjoint data split
- Smoothed class weighting
- Evaluation metrics calculation
- Model checkpointing
- Tensorboard logging
- Batch sampling for imbalanced data
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
