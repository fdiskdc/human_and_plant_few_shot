"""
Plant + Zero Combined Dataset for Single-Class Binary Classification

This module combines the Plant dataset (Positives) and the Zero dataset (Negatives/Background)
into a single interface for "Plant vs. Zero (Background)" binary classification.

Key Features:
1. Virtual Indexing: 0..N-1 are Plant, N..M are Zero
2. Zero Splitting: Deterministically splitting the Zero dataset into Train/Test pools
3. Structure Caching: Separate caching for Zero data to utilize pre-computed Plant structures
4. y_12class Property: Seamless concatenation of Plant and Zero labels
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
import hashlib
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, List

# Re-use constants and functions from human.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from human import (
    ONE_HOT_MAPPING,
    DEFAULT_ONE_HOT,
    TARGET_LENGTH,
    LINEARFOLD_PATH,
    LABEL_MAPPING,
    INDEX_TO_NUCLEOTIDE,
    NUCLEOTIDE_GROUPS,
    MOD_NAMES,
    run_linearfold,
    build_edge_index_from_structure,
    build_sequential_edge_index,
    _BYTE_TO_ONEHOT_MAPPING,
    BATCH_CACHE_FILE
)


class PlantSingleDataset(Dataset):
    """
    Combined Plant + Zero Dataset for Single-Class Binary Classification.

    This dataset provides:
    1. Virtual indexing where:
       - Indices 0 to num_plant-1 return Plant samples (positives)
       - Indices num_plant to total-1 return Zero samples (negatives/background)

    2. Deterministic splitting of Zero dataset to prevent data leakage
    3. Separate caching for Zero structures
    4. y_12class property that seamlessly concatenates Plant and Zero labels
    """

    def __init__(self, plant_dir='plant', zero_dir='npy/zero', cache_dir=None,
                 use_cache=True, preload_cache=True):
        """
        Initialize the Plant+Zero combined dataset.

        Args:
            plant_dir: Directory containing plant data (seq.npy, 12loc.npy, etc.)
            zero_dir: Directory containing zero data (zero_seq.npy, zero_label12.npy, etc.)
            cache_dir: Directory for structure cache files
            use_cache: Whether to enable structure caching
            preload_cache: Whether to preload cache into memory at initialization
        """
        self.plant_dir = plant_dir
        self.zero_dir = zero_dir
        self.use_cache = use_cache
        self._plant_edge_indices = None
        self._zero_edge_indices = None

        # Set up cache directory
        if cache_dir is None:
            self.CACHE_DIR = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'npy/cache'
            )
        else:
            self.CACHE_DIR = cache_dir

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # ---------------- Load Plant Data ----------------
        if not os.path.exists(plant_dir) and os.path.exists(f'../{plant_dir}'):
            self.plant_dir = f'../{plant_dir}'

        print(f"Loading Plant data from: {self.plant_dir}")
        self.plant_seq = np.load(f'{self.plant_dir}/seq.npy', mmap_mode='r')
        self.plant_y12 = np.load(f'{self.plant_dir}/12loc.npy', mmap_mode='r')
        self.plant_y4 = np.load(f'{self.plant_dir}/4loc.npy', mmap_mode='r')

        # Try to load 1001loc (full labels), handle gracefully if not exists
        try:
            self.plant_full_labels = np.load(f'{self.plant_dir}/1001loc.npy', mmap_mode='r')
        except FileNotFoundError:
            self.plant_full_labels = None

        self.num_plant = len(self.plant_seq)

        # ---------------- Load Zero Data ----------------
        if not os.path.exists(zero_dir) and os.path.exists(f'../{zero_dir}'):
            self.zero_dir = f'../{zero_dir}'

        print(f"Loading Zero data from: {self.zero_dir}")
        self.zero_seq = np.load(f'{self.zero_dir}/zero_seq.npy', mmap_mode='r')
        self.zero_y12 = np.load(f'{self.zero_dir}/zero_label12.npy', mmap_mode='r')
        self.zero_y4 = np.load(f'{self.zero_dir}/zero_label4.npy', mmap_mode='r')

        # Try to load zero_label1001, handle gracefully if not exists
        try:
            self.zero_full_labels = np.load(f'{self.zero_dir}/zero_label1001.npy', mmap_mode='r')
        except FileNotFoundError:
            self.zero_full_labels = None

        self.num_zero = len(self.zero_seq)
        self.total_samples = self.num_plant + self.num_zero

        print(f"Dataset Initialized: {self.num_plant} Plant + {self.num_zero} Zero = {self.total_samples} Total")

        # ---------------- Preload Caches ----------------
        if self.use_cache and preload_cache:
            self._load_batch_caches()

    def __len__(self):
        """Return total number of samples (Plant + Zero)."""
        return self.total_samples

    @property
    def y_12class(self):
        """
        Seamless concatenation of Plant and Zero 12-class labels.

        Returns a concatenated numpy array of shape (total_samples, 12).
        This property is needed for the sampler functions.
        """
        # Concatenate Plant and Zero labels along the first axis
        return np.concatenate([self.plant_y12, self.zero_y12], axis=0)

    def _one_hot_encode_optimized(self, sequence_bytes):
        """
        Optimized one-hot encoding that directly processes byte array.

        Args:
            sequence_bytes: |S1 type byte array from mmap_mode='r'

        Returns:
            np.array: One-hot encoded array with shape (1001, 4)
        """
        byte_array = sequence_bytes.view(np.uint8)
        one_hot = _BYTE_TO_ONEHOT_MAPPING[byte_array]
        return one_hot.astype(np.float32)

    def __getitem__(self, idx):
        """
        Get a sample by index with virtual indexing routing.

        Routing Logic:
        - 0 <= idx < num_plant: Return Plant sample
        - num_plant <= idx < total: Return Zero sample (mapped to idx - num_plant)

        Args:
            idx: Global index (0 to total_samples-1)

        Returns:
            Data: PyG Data object with x, edge_index, y, y_4class, is_plant, real_idx
        """
        is_plant = idx < self.num_plant
        full_label = None

        if is_plant:
            # Fetch from Plant
            real_idx = idx
            seq_bytes = self.plant_seq[real_idx].copy()
            y12 = self.plant_y12[real_idx].copy()
            y4 = self.plant_y4[real_idx].copy()
            if self.plant_full_labels is not None:
                full_label = self.plant_full_labels[real_idx].copy()
            # Cache retrieval for Plant
            edge_index = self._get_edge_index(real_idx, is_plant=True, seq_bytes=seq_bytes)
        else:
            # Fetch from Zero
            real_idx = idx - self.num_plant
            seq_bytes = self.zero_seq[real_idx].copy()
            y12 = self.zero_y12[real_idx].copy()
            y4 = self.zero_y4[real_idx].copy()
            if self.zero_full_labels is not None:
                full_label = self.zero_full_labels[real_idx].copy()
            # Cache retrieval for Zero
            edge_index = self._get_edge_index(real_idx, is_plant=False, seq_bytes=seq_bytes)

        # If full_label (site-level) is not available, create a zero array
        if full_label is None:
            full_label = np.zeros(TARGET_LENGTH, dtype=np.int64)

        # Common processing
        one_hot_seq = self._one_hot_encode_optimized(seq_bytes)
        node_features = torch.FloatTensor(one_hot_seq)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y12).unsqueeze(0),
            y_4class=torch.FloatTensor(y4).unsqueeze(0),
            y_site=torch.LongTensor(full_label)  # Add site-level labels
        )

        # Add metadata for tracking
        data.is_plant = is_plant
        data.real_idx = real_idx

        return data

    # ------------------------------------------------------------------------
    # Split Logic
    # ------------------------------------------------------------------------

    def get_zero_split(self, test_ratio=0.2, seed=42):
        """
        Deterministic split of Zero dataset into train/test pools.

        Returns GLOBAL indices (offset by self.num_plant) for the Zero part.
        This ensures no data leakage between training and testing.

        Args:
            test_ratio: Fraction of Zero data to reserve for testing (default 0.2)
            seed: Random seed for reproducibility (default 42)

        Returns:
            tuple: (zero_train_global, zero_test_global)
                - zero_train_global: List of global indices for Zero training samples
                - zero_test_global: List of global indices for Zero test samples
        """
        np.random.seed(seed)
        indices = np.arange(self.num_zero)
        np.random.shuffle(indices)

        split_point = int(self.num_zero * (1 - test_ratio))
        train_local = indices[:split_point]
        test_local = indices[split_point:]

        # Convert local zero indices to global dataset indices
        zero_train_global = train_local + self.num_plant
        zero_test_global = test_local + self.num_plant

        return zero_train_global.tolist(), zero_test_global.tolist()

    def get_plant_indices_by_class(self, target_class):
        """
        Get global indices of Plant samples containing the target class.

        Args:
            target_class: Class index (0-11) to filter by

        Returns:
            list: Global indices of Plant samples with the target class
        """
        # Plant indices are 0...num_plant-1, so local == global
        pos_indices = np.where(self.plant_y12[:, target_class] == 1)[0]
        return pos_indices.tolist()

    # ------------------------------------------------------------------------
    # Caching Logic (Separated for Plant and Zero)
    # ------------------------------------------------------------------------

    def _get_cache_paths(self):
        """Get paths for Plant and Zero batch cache files."""
        # 修改点：根据您的目录结构，Plant 使用 plant_structures_cache.npz
        plant_path = os.path.join(self.CACHE_DIR, "plant_structures_cache.npz")
        
        # Zero 使用 zero_batch_cache.npz
        zero_path = os.path.join(self.CACHE_DIR, "zero_batch_cache.npz")
        
        return plant_path, zero_path

    def _load_batch_caches(self):
        """Load both Plant and Zero batch caches into memory."""
        plant_path, zero_path = self._get_cache_paths()

        # Load Plant Cache
        if os.path.exists(plant_path):
            try:
                data = np.load(plant_path, allow_pickle=True)
                self._plant_edge_indices = data['edge_indices']
                print(f"Loaded Plant structures: {len(self._plant_edge_indices)}")
            except Exception as e:
                print(f"Error loading plant cache: {e}")

        # Load Zero Cache
        if os.path.exists(zero_path):
            try:
                data = np.load(zero_path, allow_pickle=True)
                self._zero_edge_indices = data['edge_indices']
                print(f"Loaded Zero structures: {len(self._zero_edge_indices)}")
            except Exception as e:
                print(f"Error loading zero cache: {e}")

    def _get_edge_index(self, real_idx, is_plant, seq_bytes):
        """
        Get edge index STRICTLY from cache. 
        Raises RuntimeError if cache is missing, out of bounds, or None.
        """
        edge_index = None
        dataset_name = "Plant" if is_plant else "Zero"

        # 1. 尝试从内存缓存读取
        if is_plant:
            # 检查 Plant 缓存是否已加载
            if self._plant_edge_indices is None:
                 raise RuntimeError(f"CRITICAL ERROR: {dataset_name} cache array is None. "
                                    f"Did the cache file load correctly? Check _load_batch_caches.")
            
            # 检查索引越界
            if real_idx >= len(self._plant_edge_indices):
                raise RuntimeError(f"CRITICAL ERROR: {dataset_name} index {real_idx} out of bounds. "
                                   f"Cache size is {len(self._plant_edge_indices)}.")
            
            edge_index = self._plant_edge_indices[real_idx]

        else: # is Zero
            # 检查 Zero 缓存是否已加载
            if self._zero_edge_indices is None:
                 raise RuntimeError(f"CRITICAL ERROR: {dataset_name} cache array is None. "
                                    f"Did the cache file load correctly? Check _load_batch_caches.")

            # 检查索引越界
            if real_idx >= len(self._zero_edge_indices):
                raise RuntimeError(f"CRITICAL ERROR: {dataset_name} index {real_idx} out of bounds. "
                                   f"Cache size is {len(self._zero_edge_indices)}.")
            
            edge_index = self._zero_edge_indices[real_idx]

        # 2. 检查具体值是否有效 (防止缓存中存在空洞/None)
        if edge_index is None:
            raise RuntimeError(f"CRITICAL ERROR: Cache entry is None for {dataset_name} index {real_idx}. "
                               f"This indicates the cache file exists but this specific sample was not computed correctly.")

        return torch.from_numpy(edge_index)

    def precompute_zero_structures(self, batch_size=100, num_workers=None, show_progress=True):
        """
        Precompute structures for the Zero dataset only.

        This saves a separate cache file (zero_batch_cache.npz) so we don't
        re-compute Plant structures.

        Args:
            batch_size: Batch size for LinearFold processing
            num_workers: Number of worker processes (None = CPU count)
            show_progress: Whether to show progress bar
        """
        from multiprocessing import Pool, cpu_count
        from human import _worker_process_batch

        num_samples = self.num_zero
        _, zero_path = self._get_cache_paths()

        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()

        use_multiprocessing = num_workers > 1

        print(f"\n{'='*60}")
        print(f"Precomputing Zero dataset structures...")
        print(f"  Samples: {num_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Workers: {num_workers if use_multiprocessing else 1}")
        print(f"  Cache: {zero_path}")
        print(f"{'='*60}\n")

        # Initialize edge indices array
        edge_indices = np.empty(num_samples, dtype=object)
        stats = {'total': num_samples, 'computed': 0, 'failed': 0}

        # Prepare batch tasks
        batch_tasks = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))
            batch_tasks.append((batch_indices, self.zero_seq, LINEARFOLD_PATH))

        total_batches = len(batch_tasks)

        if use_multiprocessing:
            # Multi-process processing
            print(f"Using {num_workers} processes...\n")

            with Pool(processes=num_workers) as pool:
                results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)

                if show_progress:
                    results_iter = tqdm(results_iter, total=total_batches, desc="Precomputing Zero structures")

                # Collect results
                for batch_results in results_iter:
                    for idx, edge_index_numpy, error in batch_results:
                        if error is None:
                            edge_indices[idx] = edge_index_numpy
                            stats['computed'] += 1
                        else:
                            stats['failed'] += 1
                            if stats['failed'] <= 5:
                                print(f"  Warning: Index {idx} failed: {error}")

        else:
            # Single-process processing
            iterator = range(0, num_samples, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Precomputing Zero structures")

            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))

                # Prepare batch sequence strings
                sequences_str = []
                for idx in batch_indices:
                    sequence_bytes = self.zero_seq[idx].copy()
                    sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                    sequences_str.append(sequence_str)

                # Use LinearFold to compute structures
                try:
                    structures = run_linearfold(sequences_str)

                    # Build edge indices
                    for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
                        edge_index = build_edge_index_from_structure(sequences_str[i], structure)
                        edge_indices[idx] = edge_index.cpu().numpy()
                        stats['computed'] += 1

                except Exception as e:
                    print(f"\nWarning: Batch computation failed (indices {start_idx}-{end_idx}): {e}")
                    for idx in batch_indices:
                        stats['failed'] += 1

        # Save to batch cache file
        print(f"\nSaving Zero batch cache to: {zero_path}")
        try:
            np.savez_compressed(
                zero_path,
                edge_indices=edge_indices,
                mode='zero',
                num_samples=num_samples
            )

            file_size_mb = os.path.getsize(zero_path) / (1024**2)
            print(f"  Cache saved, size: {file_size_mb:.2f} MB")

        except Exception as e:
            print(f"  Error: Failed to save cache: {e}")
            return

        # Load into memory
        self._zero_edge_indices = edge_indices

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Zero precomputation complete!")
        print(f"  Total: {stats['total']}")
        print(f"  Computed: {stats['computed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"{'='*60}\n")

    def get_cache_stats(self):
        """
        Get cache statistics for both Plant and Zero datasets.

        Returns:
            dict: Cache statistics including file sizes and load status
        """
        stats = {
            'cache_dir': self.CACHE_DIR,
            'plant_cache': None,
            'zero_cache': None
        }

        plant_path, zero_path = self._get_cache_paths()

        # Plant cache stats
        if os.path.exists(plant_path):
            try:
                plant_size_mb = os.path.getsize(plant_path) / (1024 * 1024)
                stats['plant_cache'] = {
                    'exists': True,
                    'path': plant_path,
                    'size_mb': plant_size_mb,
                    'loaded_in_memory': self._plant_edge_indices is not None
                }
            except OSError:
                stats['plant_cache'] = {'exists': False}
        else:
            stats['plant_cache'] = {'exists': False}

        # Zero cache stats
        if os.path.exists(zero_path):
            try:
                zero_size_mb = os.path.getsize(zero_path) / (1024 * 1024)
                stats['zero_cache'] = {
                    'exists': True,
                    'path': zero_path,
                    'size_mb': zero_size_mb,
                    'loaded_in_memory': self._zero_edge_indices is not None
                }
            except OSError:
                stats['zero_cache'] = {'exists': False}
        else:
            stats['zero_cache'] = {'exists': False}

        return stats

    def clear_zero_cache(self):
        """
        Clear Zero cache files and in-memory cache.
        Plant cache is preserved.
        """
        _, zero_path = self._get_cache_paths()

        # Clear file cache
        if os.path.exists(zero_path):
            try:
                file_size_mb = os.path.getsize(zero_path) / (1024 * 1024)
                os.remove(zero_path)
                print(f"Deleted Zero batch cache: {zero_path} ({file_size_mb:.2f} MB)")
            except OSError as e:
                print(f"Failed to delete Zero cache: {e}")

        # Clear in-memory cache
        self._zero_edge_indices = None
