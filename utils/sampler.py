"""
Batch samplers and class weighting for RNA Multi-label Classification Training

This module contains:
- MultilabelBalancedBatchSampler: Balanced batch sampler for multi-label data
- DynamicBalancedBatchSampler: Dynamic sampler that adapts strategy during training
- get_smoothed_pos_weights: Calculate smoothed positive weights for loss function
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Optional
import logging

# Import constants from common module
from .common import MOD_NAMES


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
