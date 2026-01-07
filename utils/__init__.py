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

# Import sampler classes
from .sampler import (
    MultilabelBalancedBatchSampler,
    DynamicBalancedBatchSampler,
    get_smoothed_pos_weights
)

# Import split functions
from .split import multi_label_disjoint_split

# Import metrics functions
from .metrics import (
    calculate_metrics,
    find_optimal_threshold,
    evaluate_unbalance,
    evaluate_balanceb,
    evaluate_with_optimal_threshold,
    evaluate_4class_with_optimal_threshold,
    evaluate_plant_unbalance,
    evaluate_plant_balanceb,
    get_all_predictions
)

# Import group-based metrics functions
from .group_metrics import (
    evaluate_group_balanceb
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
    test_epoch
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

    # Training
    'train_epoch',
    'test_epoch',

    # Logging
    'setup_logging',
    'setup_tensorboard',
    'log_metrics_to_tensorboard',
    'print_evaluation_results',
    'print_few_shot_results',
]
