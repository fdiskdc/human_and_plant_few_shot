"""
Data split functions for RNA Multi-label Classification Training

This module contains:
- multi_label_disjoint_split: Split dataset into train/test with multi-label awareness
"""

import os
import hashlib
import pickle
from typing import List, Optional, Tuple
import logging

import numpy as np

# Import constants from common module
from .common import MOD_NAMES


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
