"""
Common utilities for RNA Multi-label Classification Training

This module contains:
- Hierarchical classification constants
- Configuration loading
- Model checkpointing
- Batch samplers for multi-label data
- Data split functions
- Training/testing functions
"""

import os
import json
import hashlib
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

# ============================================================================
# Hierarchical Classification Constants
# ============================================================================

# 12类修饰名称映射 (模型索引 -> 修饰名称)
# 与 human_make_npy.py 中的 MOD_TO_INDEX 一致 (mod_index - 1 转换后)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',      # 索引 0-2
    3: 'Gm',     4: 'Tm',     5: 'Y',       # 索引 3-5
    6: 'ac4C',   7: 'm1A',    8: 'm5C',     # 索引 6-8
    9: 'm6A',    10: 'm6Am',  11: 'm7G'     # 索引 9-11
}

# 4-Class Group Names (Nucleotide Groups)
NUCLEOTIDE_GROUP_NAMES = ['A', 'C', 'G', 'U']

# Model Index (0-11) to Nucleotide Group Name
INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',          # Am, Atol
    2: 'C',                  # Cm
    3: 'G',                  # Gm
    4: 'U', 5: 'U',          # Tm, Y
    6: 'C',                  # ac4C
    7: 'A',                  # m1A
    8: 'C',                  # m5C
    9: 'A', 10: 'A',         # m6A, m6Am
    11: 'G'                  # m7G
}

# Group Name to original IDs (for reference)
NUCLEOTIDE_GROUPS = {
    'A': [1, 2, 8, 10, 11],
    'C': [3, 7, 9],
    'G': [4, 12],
    'U': [5, 6]
}

# Group to 4-Class Index Mapping (Order for y_4class)
GROUP_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# Index to Group Mapping (reverse of above)
INDEX_TO_GROUP = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Number of classes per group (for hierarchical derivation)
GROUP_SIZES = {
    'A': 5,  # Am(0), Atol(1), m1A(7), m6A(9), m6Am(10)
    'C': 3,  # Cm(2), ac4C(6), m5C(8)
    'G': 2,  # Gm(3), m7G(11)
    'U': 2   # Tm(4), Y(5)
}

# Mapping from group to the indices (0-11) of classes in that group
GROUP_TO_CLASS_INDICES = {
    'A': [0, 1, 7, 9, 10],    # Am, Atol, m1A, m6A, m6Am
    'C': [2, 6, 8],           # Cm, ac4C, m5C
    'G': [3, 11],             # Gm, m7G
    'U': [4, 5]               # Tm, Y
}

# ============================================================================
# Plant Dataset Constants
# ============================================================================

# Plant dataset only has 3 valid classes (Y, m5C, m6A)
# These are the class indices (0-11) that should be evaluated for Plant data
PLANT_VALID_CLASS_INDICES = [5, 8, 9]  # Y (class 5), m5C (class 8), m6A (class 9)


# ============================================================================
# Helper Functions
# ============================================================================

def get_center_nucleotide(sequence: str) -> str:
    """
    Get the center nucleotide (index 500) from a sequence.

    Args:
        sequence: RNA sequence string (length 1001)

    Returns:
        Nucleotide character ('A', 'C', 'G', or 'U')
    """
    if len(sequence) >= 501:
        return sequence[500].upper()
    return 'A'  # Default fallback


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = 'model.json') -> Tuple:
    """
    Load configuration from JSON file and create timestamped experiment folder.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Config object and config_dict
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create timestamped experiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config_dict.get('experiment_name', 'rna_classification')
    exp_folder = f"{exp_name}_{timestamp}"

    # Update paths with timestamped folder
    base_log_dir = config_dict.get('paths', {}).get('log_dir', './logs')

    log_dir = os.path.join(base_log_dir, exp_folder)
    # Save checkpoints inside the log folder
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration to experiment folder for reproducibility
    config_save_path = os.path.join(log_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Define Config class
    class Config:
        """Global configuration class"""
        pass

    # Set paths - support both old and new config format
    data_cfg = config_dict.get('data', {})

    # For backward compatibility, if 'data' section doesn't exist, use old format
    if data_cfg:
        # New format with separate data paths
        Config.data_dir = data_cfg.get('human_data_dir', './human3')
        Config.human_data_dir = data_cfg.get('human_data_dir', './human3')
        Config.plant_data_dir = data_cfg.get('plant_data_dir', './plant')
        Config.cache_dir = data_cfg.get('cache_dir', './cache')
    else:
        # Old format (backward compatibility)
        Config.data_dir = config_dict.get('data_dir', './human3')
        Config.human_data_dir = Config.data_dir
        Config.plant_data_dir = './plant'
        Config.cache_dir = './cache'

    Config.checkpoint_dir = checkpoint_dir
    Config.log_dir = log_dir
    Config.experiment_name = exp_name
    Config.timestamp = timestamp

    # Create Config.data object for easier access
    class DataConfig:
        pass
    DataConfig.human_data_dir = Config.human_data_dir
    DataConfig.plant_data_dir = Config.plant_data_dir
    DataConfig.cache_dir = Config.cache_dir
    Config.data = DataConfig

    # Set model parameters
    model_cfg = config_dict.get('model', {})
    Config.cnn_hidden_dim = model_cfg.get('cnn_hidden_dim', 64)
    Config.cnn_kernel_sizes = tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7]))
    Config.cnn_dropout = model_cfg.get('cnn_dropout', 0.1)
    Config.gcn_hidden_dim = model_cfg.get('gcn_hidden_dim', 128)
    Config.gcn_out_channels = model_cfg.get('gcn_out_channels', 128)
    Config.gcn_num_layers = model_cfg.get('gcn_num_layers', 3)
    Config.gcn_dropout = model_cfg.get('gcn_dropout', 0.3)
    Config.num_classes = model_cfg.get('num_classes', 12)
    Config.num_attn_heads = model_cfg.get('num_attn_heads', 4)
    Config.attn_dropout = model_cfg.get('attn_dropout', 0.1)
    Config.use_simple_pooling = model_cfg.get('use_simple_pooling', False)
    Config.use_hierarchical = model_cfg.get('use_hierarchical', False)
    Config.use_layer_norm = model_cfg.get('use_layer_norm', True)

    # Set training parameters
    train_cfg = config_dict.get('training', {})
    Config.batch_size = train_cfg.get('batch_size', 32)
    Config.num_epochs = train_cfg.get('num_epochs', 100)
    Config.learning_rate = train_cfg.get('learning_rate', 1e-3)
    Config.weight_decay = train_cfg.get('weight_decay', 1e-4)
    Config.warmup_epochs = train_cfg.get('warmup_epochs', 5)
    Config.train_ratio = train_cfg.get('train_ratio', 0.7)
    Config.random_seed = train_cfg.get('random_seed', 42)
    Config.eval_threshold = train_cfg.get('eval_threshold', 0.5)
    Config.test_interval = train_cfg.get('test_interval', 1)
    Config.save_every_epoch = train_cfg.get('save_every_epoch', True)
    Config.save_best_only = train_cfg.get('save_best_only', False)

    # Dynamic sampler parameters
    Config.use_dynamic_sampler = train_cfg.get('use_dynamic_sampler', True)
    Config.balance_ratio = train_cfg.get('balance_ratio', 0.3)

    # AMP (Automatic Mixed Precision) settings
    Config.use_amp = train_cfg.get('use_amp', True)
    # -----------------------------------------------------------
    # [新增] 读取 Attention Supervision 配置 (这里是你漏掉的部分)
    # -----------------------------------------------------------
    Config.use_attention_supervision = train_cfg.get('use_attention_supervision', False)
    Config.attention_lambda = train_cfg.get('attention_lambda', 1.0)
    # -----------------------------------------------------------

    # Few-shot learning parameters
    few_shot_cfg = config_dict.get('few_shot', {})
    Config.few_shot = few_shot_cfg

    # Device
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return Config, config_dict


# ============================================================================
# Model Checkpointing
# ============================================================================

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict, filepath: str,
                   logger: Optional[logging.Logger] = None,
                   config_dict: Optional[Dict] = None):
    """
    Save model checkpoint with model parameters.

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        logger: Optional logger instance
        config_dict: Optional configuration dictionary to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config_dict
    }
    torch.save(checkpoint, filepath)

    msg = f"Checkpoint saved to {filepath}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   filepath: str, device: torch.device) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        filepath: Path to checkpoint file
        device: Device to load checkpoint onto

    Returns:
        Dictionary containing epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


# ============================================================================
# Data Split Functions
# ============================================================================

def multi_label_disjoint_split(dataset, train_ratio: float = 0.7,
                               random_seed: int = 42, logger: Optional[logging.Logger] = None,
                               use_cache: bool = True, cache_dir: str = './cache') -> Tuple[List[int], List[int]]:
    """
    Split dataset into train and test sets with disjoint samples for multi-label data.

    For each class, samples are split into pre-train and pre-test indices.
    The final train_indices is the union of all pre-train indices.
    The final test_indices is the union of all pre-test indices.
    If a sample appears in both unions, it is forced into train set.

    Args:
        dataset: Mer100Dataset object
        train_ratio: Ratio for training split (default 0.7)
        random_seed: Random seed for reproducibility
        logger: Optional logger instance
        use_cache: Whether to use cached labels (default True)
        cache_dir: Directory to store cache files (default './cache')

    Returns:
        train_indices: List of training sample indices
        test_indices: List of test sample indices
    """
    rng = np.random.default_rng(random_seed)
    num_samples = len(dataset)
    num_classes = 12

    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key based on dataset properties and parameters
    cache_key = hashlib.md5(f"{dataset.data_dir}_{num_samples}_{train_ratio}_{random_seed}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'labels_cache_{cache_key}.pkl')

    # Try to load from cache
    all_labels = None
    if use_cache:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Verify cache is valid
                    if cache_data.get('num_samples') == num_samples:
                        all_labels = cache_data['labels']
                        if logger:
                            logger.info(f"Loaded cached labels from {cache_file}")
                        else:
                            print(f"Loaded cached labels from {cache_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load cache: {e}")
                else:
                    print(f"Failed to load cache: {e}")

    # Get all labels (y_12class) with tqdm progress monitoring
    # OPTIMIZATION: Directly access y_12class array instead of calling __getitem__
    # which avoids expensive LinearFold calls for each sample
    if all_labels is None:
        if logger:
            logger.info(f"Collecting labels from {num_samples} samples...")
        else:
            print(f"Collecting labels from {num_samples} samples...")

        # Directly read from the pre-loaded y_12class array
        # This is much faster than calling dataset[idx] which triggers LinearFold
        all_labels = dataset.y_12class[:num_samples].copy()  # (N, 12)

        if logger:
            logger.info(f"Loaded {len(all_labels)} labels directly from dataset array")
        else:
            print(f"Loaded {len(all_labels)} labels directly from dataset array")

        # Save to cache for future runs
        if use_cache:
            try:
                cache_data = {
                    'labels': all_labels,
                    'num_samples': num_samples,
                    'train_ratio': train_ratio,
                    'random_seed': random_seed
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                if logger:
                    logger.info(f"Saved cached labels to {cache_file}")
                else:
                    print(f"Saved cached labels to {cache_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save cache: {e}")
                else:
                    print(f"Failed to save cache: {e}")

    log_msg = f"\n{'='*60}\nMulti-label Disjoint Split\n{'='*60}\n"
    log_msg += f"Total samples: {num_samples}\n"
    log_msg += f"Number of classes: {num_classes}\n"
    log_msg += f"Train ratio: {train_ratio}\n"

    # For each class, collect samples containing that class
    class_sample_indices = {}
    for class_idx in range(num_classes):
        # Find samples with this class (label == 1)
        indices = np.where(all_labels[:, class_idx] == 1)[0]
        class_sample_indices[class_idx] = indices.tolist()
        mod_name = MOD_NAMES.get(class_idx, f'Class{class_idx}')
        log_msg += f"Class {class_idx} ({mod_name}): {len(indices)} positive samples\n"

    # Split each class independently
    pre_train_indices_per_class = []
    pre_test_indices_per_class = []

    for class_idx, indices in class_sample_indices.items():
        if len(indices) == 0:
            pre_train_indices_per_class.append(set())
            pre_test_indices_per_class.append(set())
            continue

        # Shuffle and split
        shuffled = rng.permutation(indices).tolist()
        split_point = int(len(shuffled) * train_ratio)

        pre_train = set(shuffled[:split_point])
        pre_test = set(shuffled[split_point:])

        pre_train_indices_per_class.append(pre_train)
        pre_test_indices_per_class.append(pre_test)

        mod_name = MOD_NAMES.get(class_idx, f'Class{class_idx}')
        log_msg += f"Class {class_idx} ({mod_name}) split: {len(pre_train)} train, {len(pre_test)} pre-test\n"

    # Take union across all classes
    train_indices_union = set()
    test_indices_union = set()

    for pre_train in pre_train_indices_per_class:
        train_indices_union.update(pre_train)

    for pre_test in pre_test_indices_per_class:
        test_indices_union.update(pre_test)

    log_msg += f"\nUnion sizes:\n"
    log_msg += f"  Train union: {len(train_indices_union)}\n"
    log_msg += f"  Test union: {len(test_indices_union)}\n"

    # Handle conflicts: samples in both unions go to train set
    conflicts = train_indices_union & test_indices_union
    if conflicts:
        log_msg += f"  Conflicts (samples in both unions): {len(conflicts)}\n"
        test_indices_union -= conflicts  # Remove conflicts from test set
        train_indices_union.update(conflicts)  # Ensure they're in train set

    # Verify disjoint
    assert len(train_indices_union & test_indices_union) == 0, "Train and test sets are not disjoint!"

    train_indices = sorted(list(train_indices_union))
    test_indices = sorted(list(test_indices_union))

    log_msg += f"\nFinal split sizes:\n"
    log_msg += f"  Train: {len(train_indices)} samples\n"
    log_msg += f"  Test: {len(test_indices)} samples\n"
    log_msg += f"  Total: {len(train_indices) + len(test_indices)} samples\n"
    log_msg += f"{'='*60}\n"

    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return train_indices, test_indices


# ============================================================================
# Multi-label Balanced Batch Sampler
# ============================================================================

class MultilabelBalancedBatchSampler(Sampler):
    """
    A batch sampler that ensures each batch contains balanced representation
    of all classes in multi-label classification.

    This sampler addresses extreme class imbalance by enforcing that every batch
    contains exactly n_samples_per_class samples from each of the 12 classes.

    Key features:
    - Multi-label aware: A single sample can appear in multiple class buckets
    - Infinite cyclic iterator: When a class bucket runs out, it reshuffles
      and cycles back to the beginning
    - Effective oversampling: Rare classes (e.g., Class 0 with ~1.4k samples)
      are oversampled significantly compared to common classes (e.g., Class 9
      with ~86k samples) within a single epoch

    Example:
        With batch_size=96 and num_classes=12:
        - n_samples_per_class = 96 // 12 = 8
        - Each batch contains 8 samples from each of the 12 classes
        - Total: 12 * 8 = 96 samples per batch

        For Class 0 (1,400 samples):
        - Can form 1,400 / 8 = 175 unique batches
        - After 175 batches, the bucket reshuffles and cycles

        For Class 9 (86,000 samples):
        - Can form 86,000 / 8 = 10,750 unique batches
        - Much more diversity, no cycling needed in typical epoch

    Args:
        dataset: The dataset (typically Mer100Dataset)
        train_indices: List of training sample indices
        batch_size: Total batch size (must be divisible by num_classes)
        num_classes: Number of classes (default 12)
        random_seed: Random seed for reproducibility (default 42)
    """

    def __init__(self, dataset, train_indices: List[int], batch_size: int,
                 num_classes: int = 12, random_seed: int = 42):
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_classes ({num_classes})"
            )

        self.dataset = dataset
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.n_samples_per_class = batch_size // num_classes
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Organize train_indices into class buckets
        # Each bucket contains indices of samples that have that class label
        self.class_buckets = self._build_class_buckets()

        # Calculate the number of batches per epoch
        # This is determined by the rarest class
        min_bucket_size = min(len(bucket) for bucket in self.class_buckets)
        self.num_batches = min_bucket_size // self.n_samples_per_class

    def _build_class_buckets(self) -> List[np.ndarray]:
        """
        Organize train_indices into class-specific buckets.

        For multi-label data, a single sample index can appear in multiple buckets.
        This is intentional and correct for multi-label classification.

        Returns:
            List of num_classes arrays, where each array contains RELATIVE indices
            (positions within train_indices/subset) for compatibility with DataLoader
        """
        # Get labels for all training samples
        # OPTIMIZATION: Directly access y_12class array to avoid expensive __getitem__ calls
        train_labels = self.dataset.y_12class[self.train_indices]  # (N_train, 12)

        # Build buckets: for each class, find indices where label == 1
        class_buckets = []
        for class_idx in range(self.num_classes):
            # Find samples with this class label
            mask = train_labels[:, class_idx] == 1
            # IMPORTANT: Use relative indices (positions in train_indices)
            # NOT the original dataset indices, for Subset compatibility
            class_indices = np.where(mask)[0]  # Returns positions relative to train_indices
            class_buckets.append(class_indices)

        # Shuffle each bucket for randomness
        for bucket in class_buckets:
            np.random.shuffle(bucket)

        return class_buckets

    def _get_samples_from_bucket(self, bucket_idx: int, n: int) -> np.ndarray:
        """
        Get n samples from a class bucket with cycling/oversampling.

        If the bucket runs out of samples, it reshuffles and starts from
        the beginning (infinite cyclic iterator logic).

        Args:
            bucket_idx: Index of the class bucket
            n: Number of samples to fetch

        Returns:
            Array of n sample indices
        """
        bucket = self.class_buckets[bucket_idx]
        bucket_size = len(bucket)

        if bucket_size >= n:
            # Enough samples available, take first n
            samples = bucket[:n]
            # Move used samples to end (for cycling)
            self.class_buckets[bucket_idx] = np.concatenate([bucket[n:], bucket[:n]])
        else:
            # Not enough samples (rare class case), use cycling/oversampling
            # Reshuffle and concatenate to get n samples
            np.random.shuffle(bucket)
            times = n // bucket_size + 1
            samples = np.tile(bucket, times)[:n]

        return samples

    def __iter__(self):
        """
        Generate batches with balanced class representation.

        Each batch contains exactly n_samples_per_class samples from each class.
        """
        for batch_idx in range(self.num_batches):
            batch_indices = []

            # Collect n_samples_per_class from each class bucket
            for class_idx in range(self.num_classes):
                samples = self._get_samples_from_bucket(class_idx, self.n_samples_per_class)
                batch_indices.extend(samples.tolist())

            # Convert to numpy array and shuffle within batch
            # This prevents the model from learning class order patterns
            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)

            yield batch_indices.tolist()

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches


class DynamicBalancedBatchSampler(MultilabelBalancedBatchSampler):
    """
    A dynamic batch sampler that adapts sampling strategy during training.

    Training phases:
    1. Balanced phase (early training): Prioritizes class balance, shorter epochs
       - Rare classes determine epoch length
       - Prevents overfitting on rare classes
       - Model learns balanced representations

    2. Full coverage phase (late training): Prioritizes complete data coverage
       - Common classes determine epoch length
       - All samples get sampled
       - Better convergence on full dataset

    The transition is controlled by balance_ratio (e.g., 0.3 means first 30%
    of epochs use balanced strategy, remaining 70% use full coverage).

    Args:
        dataset: The dataset (typically Mer100Dataset)
        train_indices: List of training sample indices
        batch_size: Total batch size (must be divisible by num_classes)
        num_classes: Number of classes (default 12)
        balance_ratio: Fraction of epochs to use balanced strategy (0.0-1.0)
        total_epochs: Total number of training epochs for phase calculation
        random_seed: Random seed for reproducibility (default 42)

    Example:
        With balance_ratio=0.3, total_epochs=100:
        - Epochs 1-30: Balanced mode (rare class determines batches)
        - Epochs 31-100: Full coverage mode (common class determines batches)
    """

    def __init__(
        self,
        dataset,
        train_indices: List[int],
        batch_size: int,
        num_classes: int = 12,
        balance_ratio: float = 0.3,
        total_epochs: int = 100,
        random_seed: int = 42
    ):
        # Initialize parent class
        super().__init__(dataset, train_indices, batch_size, num_classes, random_seed)

        self.balance_ratio = balance_ratio
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Calculate number of batches for each mode
        self.min_bucket_size = min(len(bucket) for bucket in self.class_buckets)
        self.max_bucket_size = max(len(bucket) for bucket in self.class_buckets)

        # Balanced mode: determined by rarest class
        self.num_batches_balanced = self.min_bucket_size // self.n_samples_per_class

        # Full coverage mode: determined by commonest class
        self.num_batches_full = self.max_bucket_size // self.n_samples_per_class

        # Current mode
        self.use_balanced_mode = True

    def set_epoch(self, epoch: int):
        """
        Set the current epoch to determine sampling strategy.

        Args:
            epoch: Current epoch number (1-indexed)
        """
        self.current_epoch = epoch
        # Use balanced mode for early epochs, full coverage for later epochs
        transition_epoch = int(self.total_epochs * self.balance_ratio)
        self.use_balanced_mode = (epoch <= transition_epoch)

        # Update num_batches based on mode
        if self.use_balanced_mode:
            self.num_batches = self.num_batches_balanced
        else:
            self.num_batches = self.num_batches_full

    def get_mode_info(self) -> str:
        """Get current mode information for logging."""
        mode = "BALANCED" if self.use_balanced_mode else "FULL_COVERAGE"
        if self.use_balanced_mode:
            return f"{mode} (rare class: {self.num_batches} batches)"
        else:
            coverage_pct = (self.num_batches_balanced / self.num_batches_full) * 100
            return f"{mode} (all classes: {self.num_batches} batches, {coverage_pct:.1f}% of balanced mode batches)"

    def __iter__(self):
        """
        Generate batches with balanced class representation.
        The number of batches depends on the current mode (balanced vs full coverage).
        """
        for batch_idx in range(self.num_batches):
            batch_indices = []

            # Collect n_samples_per_class from each class bucket
            for class_idx in range(self.num_classes):
                samples = self._get_samples_from_bucket(class_idx, self.n_samples_per_class)
                batch_indices.extend(samples.tolist())

            # Convert to numpy array and shuffle within batch
            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)

            yield batch_indices.tolist()


# ============================================================================
# Smoothed Class Weighting
# ============================================================================

def get_smoothed_pos_weights(dataset, train_indices: List[int], num_classes: int = 12,
                            epsilon: float = 1e-6, logger: Optional[logging.Logger] = None,
                            use_balanced_weights: bool = False) -> torch.Tensor:
    """
    Calculate smoothed positive weights for BCEWithLogitsLoss.

    If use_balanced_weights=True (for BALANCED sampler mode):
        Returns equal weights of 1.0 for all classes (no class weighting)

    If use_balanced_weights=False (for FULL_COVERAGE/UNBALANCED sampler mode):
        Uses square root reciprocal smoothing to handle extreme class imbalance.
        weight[i] = 1 / sqrt(count[i] + epsilon)
        Then normalized so minimum weight is 1.0.

    Args:
        dataset: Mer100Dataset object
        train_indices: List of training sample indices
        num_classes: Number of classes (12)
        epsilon: Small value to avoid division by zero
        logger: Optional logger instance
        use_balanced_weights: If True, use equal weights (1.0) for all classes

    Returns:
        pos_weight: Tensor of shape (num_classes,) with positive weights
    """
    # If using balanced weights, return equal weights
    if use_balanced_weights:
        pos_weights = np.ones(num_classes, dtype=np.float32)

        log_msg = f"\n{'='*60}\nClass Weights: BALANCED (equal weights for all classes)\n{'='*60}\n"
        log_msg += f"All classes: weight = 1.0\n"
        log_msg += f"{'='*60}\n"

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        return torch.FloatTensor(pos_weights)

    # Count positive samples for each class
    # OPTIMIZATION: Directly access y_12class array instead of calling __getitem__
    # which avoids expensive LinearFold calls for each sample
    if logger:
        logger.info(f"Counting positive samples from {len(train_indices)} training samples...")
    else:
        print(f"Counting positive samples from {len(train_indices)} training samples...")

    # Directly read from the pre-loaded y_12class array and count positives
    # This is much faster than calling dataset[idx] which triggers LinearFold
    train_labels = dataset.y_12class[train_indices]  # (len(train_indices), 12)
    pos_counts = train_labels.sum(axis=0).astype(np.float32)  # Sum along sample dimension

    # Calculate raw weights using sqrt reciprocal
    raw_weights = 1.0 / np.sqrt(pos_counts + epsilon)

    # Normalize so minimum weight is 1.0
    min_weight = np.min(raw_weights)
    pos_weights = raw_weights / min_weight

    log_msg = f"\n{'='*60}\nSmoothed Class Weights (Square Root Reciprocal)\n{'='*60}\n"
    log_msg += f"{'Class':<12} {'Pos Count':<12} {'Weight':<10}\n"
    log_msg += f"{'-'*40}\n"
    for i in range(num_classes):
        mod_name = MOD_NAMES.get(i, f'Class{i}')
        log_msg += f"{i:<3} ({mod_name:<6}) {int(pos_counts[i]):<12} {pos_weights[i]:<10.4f}\n"
    log_msg += f"{'='*60}\n"

    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return torch.FloatTensor(pos_weights)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False,
    use_amp: bool = False,
    use_attention_supervision: bool = False,
    attention_lambda: float = 1.0
) -> float:
    """
    Train for one epoch with TQDM progress monitoring.

    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Optional logger instance
        use_hierarchical: If True, use multi-task learning (4-class + 12-class)
        use_amp: If True, use automatic mixed precision
        use_attention_supervision: If True, use attention supervision loss
        attention_lambda: Weight for attention supervision loss

    Returns:
        Average loss for the epoch
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    total_loss_attn = 0.0
    num_batches = 0

    # Create GradScaler for AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training", leave=True)
    for batch in pbar:
        # Ensure labels are tensor before moving to device
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float32)

        batch = batch.to(device)

        # Ensure labels are on same device
        batch.y = batch.y.to(device)

        # Forward pass
        optimizer.zero_grad()
        
        # Determine if we need attention weights
        # Only request attention if supervision is enabled AND dataset has site labels
        should_return_attention = use_attention_supervision and hasattr(batch, 'y_site')

        if use_amp:
            # Use AMP for forward pass
            with torch.cuda.amp.autocast():
                if use_hierarchical:
                    # === Hierarchical Mode (AMP) ===
                    if should_return_attention:
                        # Multi-task learning: 12-class + 4-class + Attention Weights
                        logits_12, logits_4, attn_weights = model(
                            batch.x, batch.edge_index, batch.batch, return_attention=True
                        )
                    else:
                        # Multi-task learning: 12-class + 4-class
                        logits_12, logits_4 = model(
                            batch.x, batch.edge_index, batch.batch, return_attention=False
                        )

                    # Generate 4-class labels from 12-class labels
                    y_12 = batch.y
                    y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                    for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                        class_indices = GROUP_TO_CLASS_INDICES[group_name]
                        y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                    # Classification Losses
                    loss_12 = criterion(logits_12, y_12)
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                    
                    # Attention Supervision Loss
                    loss_attn = torch.tensor(0.0, device=device)
                    if should_return_attention:
                        loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                        total_loss_attn += loss_attn.item()

                    # Total Loss
                    loss = loss_12 + loss_4 + (attention_lambda * loss_attn)

                    total_loss_12 += loss_12.item()
                    total_loss_4 += loss_4.item()
                    
                else:
                    # === Single Task Mode (AMP) ===
                    if should_return_attention:
                        logits, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                        loss_cls = criterion(logits, batch.y)
                        loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                        loss = loss_cls + attention_lambda * loss_attn
                        total_loss_attn += loss_attn.item()
                    else:
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            # No AMP (Standard FP32)
            if use_hierarchical:
                # === Hierarchical Mode (Standard) ===
                if should_return_attention:
                    logits_12, logits_4, attn_weights = model(
                        batch.x, batch.edge_index, batch.batch, return_attention=True
                    )
                else:
                    logits_12, logits_4 = model(
                        batch.x, batch.edge_index, batch.batch, return_attention=False
                    )

                y_12 = batch.y
                y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                    class_indices = GROUP_TO_CLASS_INDICES[group_name]
                    y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                loss_12 = criterion(logits_12, y_12)
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                
                loss_attn = torch.tensor(0.0, device=device)
                if should_return_attention:
                    loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                    total_loss_attn += loss_attn.item()

                loss = loss_12 + loss_4 + (attention_lambda * loss_attn)

                total_loss_12 += loss_12.item()
                total_loss_4 += loss_4.item()
                
            else:
                # === Single Task Mode (Standard) ===
                if should_return_attention:
                    logits, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                    loss_cls = criterion(logits, batch.y)
                    loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                    loss = loss_cls + attention_lambda * loss_attn
                    total_loss_attn += loss_attn.item()
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(logits, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update Progress Bar
        postfix_dict = {"loss": f"{loss.item():.4f}"}
        
        if use_hierarchical:
            postfix_dict["l12"] = f"{loss_12.item():.4f}"
            postfix_dict["l4"] = f"{loss_4.item():.4f}"
            
        if should_return_attention and total_loss_attn > 0:
            postfix_dict["lattn"] = f"{loss_attn.item():.4f}"
            
        pbar.set_postfix(postfix_dict)

    if scheduler is not None:
        scheduler.step()

    # Final Logging
    avg_loss = total_loss / num_batches
    msg_parts = [f"Train loss: {avg_loss:.4f}"]
    
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg_parts.append(f"12-class: {avg_loss_12:.4f}")
        msg_parts.append(f"4-class: {avg_loss_4:.4f}")
        
    if use_attention_supervision and total_loss_attn > 0:
        avg_loss_attn = total_loss_attn / num_batches
        msg_parts.append(f"attn: {avg_loss_attn:.4f}")
        
    msg = ", ".join(msg_parts)

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss


def test_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    phase: str = "test",
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False,
    use_amp: bool = False
) -> float:
    """
    Test for one epoch with TQDM progress monitoring.

    Args:
        model: The model to evaluate
        dataloader: Test/validation dataloader
        criterion: Loss function
        device: Device to run evaluation on
        phase: Phase name ('test' or 'val')
        logger: Optional logger instance
        use_hierarchical: If True, use multi-task learning (4-class + 12-class)
        use_amp: If True, use automatic mixed precision

    Returns:
        Average loss for the epoch
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    num_batches = 0

    # Determine autocast context for AMP
    autocast = torch.cuda.amp.autocast if use_amp else torch.no_grad

    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    if use_amp:
        with torch.no_grad():
            for batch in pbar:
                # Ensure labels are tensor before moving to device
                if not isinstance(batch.y, torch.Tensor):
                    batch.y = torch.tensor(batch.y, dtype=torch.float32)

                batch = batch.to(device)

                # Ensure labels are on same device
                batch.y = batch.y.to(device)

                # Forward pass with AMP
                with torch.cuda.amp.autocast():
                    if use_hierarchical:
                        # Multi-task learning: get both 12-class and 4-class logits
                        logits_12, logits_4,_ = model(batch.x, batch.edge_index, batch.batch)

                        # Generate 4-class labels from 12-class labels
                        y_12 = batch.y  # (Batch, 12)
                        y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                        # For each group (A=0, C=1, G=2, U=3)
                        for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                            class_indices = GROUP_TO_CLASS_INDICES[group_name]
                            y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                        # Calculate losses for both tasks
                        # Note: criterion has pos_weight for 12 classes, so we can only use it for 12-class loss
                        # For 4-class loss, we use BCEWithLogitsLoss without pos_weight
                        loss_12 = criterion(logits_12, y_12)
                        loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                        loss = loss_12 + loss_4

                        total_loss_12 += loss_12.item()
                        total_loss_4 += loss_4.item()
                    else:
                        # Single-task learning: only 12-class
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y)

                total_loss += loss.item()
                num_batches += 1

                if use_hierarchical:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    else:
        with torch.no_grad():
            for batch in pbar:
                # Ensure labels are tensor before moving to device
                if not isinstance(batch.y, torch.Tensor):
                    batch.y = torch.tensor(batch.y, dtype=torch.float32)

                batch = batch.to(device)

                # Ensure labels are on same device
                batch.y = batch.y.to(device)

                # Forward pass
                if use_hierarchical:
                    # Multi-task learning: get both 12-class and 4-class logits
                    logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)

                    # Generate 4-class labels from 12-class labels
                    y_12 = batch.y  # (Batch, 12)
                    y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                    # For each group (A=0, C=1, G=2, U=3)
                    for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                        class_indices = GROUP_TO_CLASS_INDICES[group_name]
                        y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                    # Calculate losses for both tasks
                    # Note: criterion has pos_weight for 12 classes, so we can only use it for 12-class loss
                    # For 4-class loss, we use BCEWithLogitsLoss without pos_weight
                    loss_12 = criterion(logits_12, y_12)
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                    loss = loss_12 + loss_4

                    total_loss_12 += loss_12.item()
                    total_loss_4 += loss_4.item()
                else:
                    # Single-task learning: only 12-class
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(logits, batch.y)

                total_loss += loss.item()
                num_batches += 1

                if use_hierarchical:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})"
    else:
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f}"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss


# ============================================================================
# Attention Supervision Loss
# ============================================================================

# Import LABEL_MAPPING from human for consistency
# This mapping converts original label IDs (1-12) to model indices (0-11)
from human import LABEL_MAPPING


def compute_attention_supervision_loss(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    num_classes: int = 12,
    seq_len: int = 1001
) -> torch.Tensor:
    """
    Compute attention supervision loss using KL divergence.

    This loss encourages the model to attend to positions where modifications
    actually occur (as indicated by y_site labels).

    Args:
        attn_weights: Attention weights [Batch_Size, Num_Classes, Seq_Len]
        y_site: Site-level labels [Batch_Size * Seq_Len] (from PyG batch)
        num_classes: Number of classes (default 12)
        seq_len: Sequence length (default 1001)

    Returns:
        loss_attn: Attention supervision loss (scalar)
    """
    import torch.nn.functional as F

    batch_size = attn_weights.size(0)
    device = attn_weights.device

    # Reshape y_site from [Batch * Seq_Len] to [Batch, Seq_Len]
    y_site_reshaped = y_site.view(batch_size, seq_len)

    loss_attn = 0.0
    num_valid_classes = 0

    # Iterate over each class
    for class_idx in range(num_classes):
        # Find the original label ID for this class_idx
        # LABEL_MAPPING: {1: 0, 2: 1, ..., 12: 11}
        # We need to reverse this: find k where LABEL_MAPPING[k] == class_idx
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            continue

        # Generate binary mask: 1 if position has this modification, 0 otherwise
        target_mask = (y_site_reshaped == original_label_id).float()

        # Only compute loss for samples that have this modification
        has_mod_mask = target_mask.sum(dim=1) > 0

        if has_mod_mask.sum() > 0:
            num_valid_classes += 1

            # Get predictions and targets for samples with this modification
            pred_attn = attn_weights[has_mod_mask, class_idx, :]  # [M, Seq_Len]
            target_mask_filtered = target_mask[has_mod_mask]  # [M, Seq_Len]

            # Normalize target to probability distribution for KL divergence
            target_dist = target_mask_filtered / (target_mask_filtered.sum(dim=1, keepdim=True) + 1e-10)

            # Use log_softmax for prediction (KL divergence expects log probabilities)
            pred_log_dist = torch.log(pred_attn + 1e-10)

            # Compute KL divergence
            loss_kl = F.kl_div(pred_log_dist, target_dist, reduction='batchmean')

            loss_attn += loss_kl

    # Average over classes that have valid samples
    if num_valid_classes > 0:
        loss_attn = loss_attn / num_valid_classes
    else:
        loss_attn = torch.tensor(0.0, device=device)

    return loss_attn


# ============================================================================
# Top-K Site Recall Evaluation
# ============================================================================

def calculate_topk_recall(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    k_list: list = [1, 5, 10, 20, 50],
    num_classes: int = 12,
    seq_len: int = 1001
) -> dict:
    """
    Calculate Top-K site recall based on attention weights.

    For each class, compute the recall rate: how many of the true modification
    sites appear in the top-K positions ranked by attention weights.

    Args:
        attn_weights: Attention weights [N, Num_Classes, Seq_Len]
        y_site: Site-level labels [N * Seq_Len] or [N, Seq_Len]
        k_list: List of K values for top-K recall
        num_classes: Number of classes (default 12)
        seq_len: Sequence length (default 1001)

    Returns:
        dict: {class_idx: {k: recall_value}}
    """
    import numpy as np

    # Convert to CPU numpy
    attn_weights = attn_weights.detach().cpu().numpy()
    y_site = y_site.detach().cpu().numpy()

    # Reshape y_site if needed
    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        y_site = y_site.reshape(batch_size, seq_len)

    results = {}
    max_k = max(k_list)

    # Iterate over each class
    for class_idx in range(num_classes):
        # Find the original label ID for this class_idx
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            results[class_idx] = {k: 0.0 for k in k_list}
            continue

        # Find samples that have this modification
        has_mod_samples = np.any(y_site == original_label_id, axis=1)

        if np.sum(has_mod_samples) == 0:
            results[class_idx] = {k: 0.0 for k in k_list}
            continue

        # Get attention weights and labels for samples with this modification
        target_attn = attn_weights[has_mod_samples, class_idx, :]  # [M, Seq_Len]
        target_labels = y_site[has_mod_samples]  # [M, Seq_Len]

        class_recalls = {k: [] for k in k_list}

        # For each sample, compute top-K recall
        for i in range(len(target_attn)):
            true_indices = np.where(target_labels[i] == original_label_id)[0]
            num_true = len(true_indices)

            if num_true == 0:
                continue

            # Get top-K predictions (indices with highest attention)
            pred_indices = np.argsort(-target_attn[i])[:max_k]

            # Compute recall for each K
            for k in k_list:
                topk_pred = pred_indices[:k]
                hit_count = len(np.intersect1d(topk_pred, true_indices))
                recall = hit_count / num_true
                class_recalls[k].append(recall)

        # Average recall across samples
        results[class_idx] = {}
        for k in k_list:
            if len(class_recalls[k]) > 0:
                results[class_idx][k] = np.mean(class_recalls[k])
            else:
                results[class_idx][k] = 0.0

    return results


def print_topk_table(
    topk_results: dict,
    k_list: list = [1, 5, 10, 20, 50],
    logger=None
):
    """
    Print Top-K site recall results in a formatted table.

    Args:
        topk_results: Results from calculate_topk_recall
        k_list: List of K values
        logger: Optional logger instance
    """
    from prettytable import PrettyTable

    output = f"\n{'='*80}\n"
    output += f"Top-K Site Localization Recall (Attention Analysis - Macro-Average)\n"
    output += f"{'='*80}\n"

    table = PrettyTable()
    field_names = ["Class", "Name"] + [f"Top-{k}" for k in k_list]
    table.field_names = field_names
    table.align = "r"
    table.align["Class"] = "l"
    table.align["Name"] = "l"

    for c in range(12):
        row = [c, MOD_NAMES.get(c, str(c))]
        metrics = topk_results.get(c, {})
        for k in k_list:
            rec = metrics.get(k, 0.0)
            row.append(f"{rec:.4f}")
        table.add_row(row)

    output += str(table) + "\n"

    print(output)
    if logger:
        logger.info(output)


# ============================================================================
# Comprehensive Localization Metrics Evaluation (Global/Micro-Average)
# ============================================================================

def calculate_comprehensive_localization_metrics(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    k_list: list = [1, 3, 5, 10],
    num_classes: int = 12,
    seq_len: int = 1001
) -> dict:
    """
    Calculate comprehensive localization metrics based on attention weights.
    This function computes multiple evaluation metrics:
    1.  Global Recall@K (Micro-Average): Sum of all hits / Sum of all true sites
    2.  Mean Average Precision (mAP): Average AP across all valid samples
    3.  Mean Reciprocal Rank (MRR): Average of 1/Rank for first correct prediction
    4.  R-Precision: Precision at R, where R is the number of true sites for the sample.
    5.  NDCG@K: Normalized Discounted Cumulative Gain, measuring ranking quality.
    6.  MDE (Mean Distance Error): Average distance of Top-1 false positives to the nearest true site.
    Args:
        attn_weights: Attention weights [N, Num_Classes, Seq_Len]
        y_site: Site-level labels [N * Seq_Len] or [N, Seq_Len]
        k_list: List of K values for top-K recall/precision
        num_classes: Number of classes (default 12)
        seq_len: Sequence length (default 1001)
    Returns:
        dict: {class_idx: {'mAP': float, 'MRR': float, 'R@K': float, 'R-Precision': float, 'NDCG@K': float, 'MDE': float}}
    """
    import numpy as np

    # Convert to CPU numpy
    attn_weights = attn_weights.detach().cpu().numpy()
    y_site = y_site.detach().cpu().numpy()

    # Reshape y_site if needed
    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        y_site = y_site.reshape(batch_size, seq_len)

    results = {}

    # Iterate over each class
    for class_idx in range(num_classes):
        # Find the original label ID for this class_idx
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            # No valid label mapping, return zeros for all metrics
            results[class_idx] = {
                'mAP': 0.0, 'MRR': 0.0, 'R-Precision': 0.0, 'MDE': 0.0
            }
            for k in k_list:
                results[class_idx][f'R@{k}'] = 0.0
                results[class_idx][f'NDCG@{k}'] = 0.0
            continue

        # Find samples that have this modification
        has_mod_samples = np.any(y_site == original_label_id, axis=1)

        if np.sum(has_mod_samples) == 0:
            # No samples with this modification
            results[class_idx] = {
                'mAP': 0.0, 'MRR': 0.0, 'R-Precision': 0.0, 'MDE': 0.0
            }
            for k in k_list:
                results[class_idx][f'R@{k}'] = 0.0
                results[class_idx][f'NDCG@{k}'] = 0.0
            continue

        # Get attention weights and labels for samples with this modification
        target_attn = attn_weights[has_mod_samples, class_idx, :]
        target_labels = y_site[has_mod_samples]

        # Initialize accumulators
        ap_list, mrr_list, r_precision_list, mde_list = [], [], [], []
        ndcg_scores = {k: [] for k in k_list}
        global_hits = {k: 0 for k in k_list}
        global_true_count = 0

        # For each sample, compute metrics
        for i in range(len(target_attn)):
            true_indices = np.where(target_labels[i] == original_label_id)[0]
            num_true = len(true_indices)

            if num_true == 0:
                continue

            global_true_count += num_true
            pred_ranks = np.argsort(-target_attn[i])

            # R-Precision
            r = num_true
            top_r_preds = pred_ranks[:r]
            r_precision_hits = np.sum(np.isin(top_r_preds, true_indices))
            r_precision_list.append(r_precision_hits / r)

            # NDCG@K
            relevance = np.zeros_like(target_attn[i])
            relevance[true_indices] = 1
            
            def dcg_at_k(r, k):
                r = np.asarray(r)[:k]
                if r.size:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                return 0.

            for k in k_list:
                pred_rel_at_k = relevance[pred_ranks]
                dcg_val = dcg_at_k(pred_rel_at_k, k)
                
                ideal_rel_at_k = np.sort(relevance)[::-1]
                idcg_val = dcg_at_k(ideal_rel_at_k, k)

                if idcg_val > 0:
                    ndcg_scores[k].append(dcg_val / idcg_val)

            # MDE (Mean Distance Error) for Top-1 False Positives
            top_1_pred = pred_ranks[0]
            if top_1_pred not in true_indices:
                distances = np.abs(true_indices - top_1_pred)
                mde_list.append(np.min(distances))

            # mAP and MRR
            precisions_at_k = []
            for rank_idx, pred_pos in enumerate(pred_ranks):
                if pred_pos in true_indices:
                    # Found a true positive, calculate precision at this rank
                    hit_count = np.sum(np.isin(pred_ranks[:rank_idx+1], true_indices))
                    precisions_at_k.append(hit_count / (rank_idx + 1))
            
            if precisions_at_k:
                ap_list.append(np.mean(precisions_at_k))

            for rank, pos_idx in enumerate(pred_ranks):
                if pos_idx in true_indices:
                    mrr_list.append(1.0 / (rank + 1))
                    break
            
            # Global Hits for R@K
            for k in k_list:
                hit_count = len(np.intersect1d(pred_ranks[:k], true_indices))
                global_hits[k] += hit_count

        # ===== Compute Final Metrics =====
        class_results = {}
        class_results['mAP'] = np.mean(ap_list) if ap_list else 0.0
        class_results['MRR'] = np.mean(mrr_list) if mrr_list else 0.0
        class_results['R-Precision'] = np.mean(r_precision_list) if r_precision_list else 0.0
        class_results['MDE'] = np.mean(mde_list) if mde_list else 0.0

        for k in k_list:
            class_results[f'R@{k}'] = global_hits[k] / global_true_count if global_true_count > 0 else 0.0
            class_results[f'NDCG@{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores.get(k) else 0.0

        results[class_idx] = class_results

    return results


def print_comprehensive_table(
    comprehensive_results: dict,
    k_list: list = [1, 3, 5, 10],
    logger=None
):
    """
    Print comprehensive localization metrics in multiple formatted tables.
    - Table A: General Accuracy (mAP, MRR, R-Precision)
    - Table B: Recall Analysis (R@K)
    - Table C: Ranking Quality (NDCG@K)
    - Table D: Error Analysis (MDE)
    """
    from prettytable import PrettyTable

    # Helper to print a table
    def _print_table(title, table, note=""):
        output = f"\n{'='*80}\n"
        output += f"=== {title} ===\n"
        output += f"{'='*80}\n"
        output += str(table) + "\n"
        if note:
            output += f"Note: {note}\n"
        print(output)
        if logger:
            logger.info(output)

    # --- Table A: General Accuracy ---
    table_a = PrettyTable()
    table_a.field_names = ["Class", "Name", "mAP", "MRR", "R-Prec"]
    table_a.align = "r"
    table_a.align["Class"] = "l"
    table_a.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        table_a.add_row([
            c, MOD_NAMES.get(c, str(c)),
            f"{metrics.get('mAP', 0.0):.4f}",
            f"{metrics.get('MRR', 0.0):.4f}",
            f"{metrics.get('R-Precision', 0.0):.4f}"
        ])
    _print_table("Table A: Localization Accuracy (mAP, MRR, R-Precision)", table_a, 
                 "mAP: Mean Avg Precision, MRR: Mean Reciprocal Rank, R-Prec: R-Precision")

    # --- Table B: Recall Analysis ---
    table_b = PrettyTable()
    table_b.field_names = ["Class", "Name"] + [f"R@{k}" for k in k_list]
    table_b.align = "r"
    table_b.align["Class"] = "l"
    table_b.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        row = [c, MOD_NAMES.get(c, str(c))]
        row.extend([f"{metrics.get(f'R@{k}', 0.0):.4f}" for k in k_list])
        table_b.add_row(row)
    _print_table("Table B: Recall Analysis (R@K)", table_b, "R@K = Global Recall (Micro-Average)")

    # --- Table C: Ranking Quality ---
    table_c = PrettyTable()
    table_c.field_names = ["Class", "Name"] + [f"NDCG@{k}" for k in k_list]
    table_c.align = "r"
    table_c.align["Class"] = "l"
    table_c.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        row = [c, MOD_NAMES.get(c, str(c))]
        row.extend([f"{metrics.get(f'NDCG@{k}', 0.0):.4f}" for k in k_list])
        table_c.add_row(row)
    _print_table("Table C: Ranking Quality (NDCG@K)", table_c, "NDCG = Normalized Discounted Cumulative Gain")

    # --- Table D: Error Analysis ---
    table_d = PrettyTable()
    table_d.field_names = ["Class", "Name", "MDE (bp)"]
    table_d.align = "r"
    table_d.align["Class"] = "l"
    table_d.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        table_d.add_row([
            c, MOD_NAMES.get(c, str(c)),
            f"{metrics.get('MDE', 0.0):.2f}"
        ])
    _print_table("Table D: Error Analysis", table_d, "MDE = Mean Distance Error for Top-1 False Positives (in base pairs)")
