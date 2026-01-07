"""
Plant RNA Multi-label Classification Dataset

This module provides a dataset class for loading plant RNA modification data.
The plant data has the same format as human3 data:
- seq.npy: (N, 1001) byte array for RNA sequences
- 12loc.npy: (N, 12) int8 array for 12-class labels
- 1001loc.npy: (N, 1001) int8 array for site-level labels
- 4loc.npy: (N, 4) int8 array for nucleotide group labels
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import os
import hashlib
import pickle
from typing import Optional

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


class PlantDataset(Dataset):
    """
    用于加载Plant RNA序列数据集，执行12类多标签分类预测任务

    与 Mer100Dataset 格式完全一致，使用内存映射加载
    """

    # 缓存目录路径
    CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'npy/cache')

    def __init__(self, plant_dir='plant', use_cache=True, preload_cache=True):
        """
        初始化Plant数据集

        Args:
            plant_dir (str): plant数据目录路径
            use_cache (bool): 是否启用二级结构缓存（默认True）
            preload_cache (bool): 是否在初始化时加载所有边索引到内存（默认True）
        """
        self.plant_dir = plant_dir
        self.use_cache = use_cache
        self._batch_cache = None
        self._edge_indices = None

        # 确保缓存目录存在
        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)

        # 检测plant目录
        if os.path.exists(plant_dir):
            self.data_dir = plant_dir
        elif os.path.exists(f'../{plant_dir}'):
            self.data_dir = f'../{plant_dir}'
        else:
            raise FileNotFoundError(f"找不到plant目录: {plant_dir}")

        print(f"使用内存映射加载plant数据: {self.data_dir}")

        # 使用mmap_mode='r'进行内存映射加载
        self.sequences = np.load(f'{self.data_dir}/seq.npy', mmap_mode='r')
        self.full_labels = np.load(f'{self.data_dir}/1001loc.npy', mmap_mode='r')
        self.y_12class = np.load(f'{self.data_dir}/12loc.npy', mmap_mode='r')
        self.y_4class = np.load(f'{self.data_dir}/4loc.npy', mmap_mode='r')

        print(f"Plant数据集初始化完成:")
        print(f"  总样本数: {len(self.sequences)}")
        print(f"  序列形状: {self.sequences.shape}, dtype: {self.sequences.dtype}")
        print(f"  1001loc形状: {self.full_labels.shape}, dtype: {self.full_labels.dtype}")
        print(f"  12loc形状: {self.y_12class.shape}, dtype: {self.y_12class.dtype}")
        print(f"  4loc形状: {self.y_4class.shape}, dtype: {self.y_4class.dtype}")

        # 如果启用缓存且preload_cache=True，尝试加载批量缓存
        if self.use_cache and preload_cache:
            self._load_batch_cache()

    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        获取单个数据样本，返回包含多层级标签的PyG Data对象

        Args:
            idx (int): 样本索引

        Returns:
            Data: PyG Data对象，包含：
                - x: 节点特征 (1001, 4)
                - edge_index: 边索引 (2, E)
                - y: 12类多标签向量 (1, 12)
        """
        # 获取序列和标签（使用copy()确保多线程安全）
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()
        y_12class = self.y_12class[idx].copy()
        y_4class = self.y_4class[idx].copy()

        # 使用优化的one-hot编码（直接处理字节流）
        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)

        # 将字节序列转换为字符串（用于LinearFold）
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')

        # 获取或计算边索引（优先使用批量缓存）
        edge_index = self._get_or_compute_edge_index(sequence_str, idx)

        # 节点特征
        node_features = torch.FloatTensor(one_hot_seq)

        # 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),  # 形状为 [1, 12]
        )

        return data

    def _get_batch_cache_path(self):
        """
        获取批量缓存文件路径

        Returns:
            str: 批量缓存文件完整路径
        """
        return os.path.join(self.CACHE_DIR, f"plant_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        """
        从批量缓存文件加载所有边索引到内存

        如果缓存文件存在，直接加载；如果不存在，则不进行任何操作
        """
        cache_path = self._get_batch_cache_path()

        if os.path.exists(cache_path):
            print(f"正在从批量缓存加载plant边索引: {cache_path}")
            try:
                self._batch_cache = np.load(cache_path, allow_pickle=True)
                self._edge_indices = self._batch_cache['edge_indices']
                print(f"  已加载 {len(self._edge_indices)} 个边索引")
                print(f"  缓存文件大小: {os.path.getsize(cache_path) / (1024**2):.2f} MB")
            except Exception as e:
                print(f"  警告: 加载批量缓存失败: {e}")
                self._batch_cache = None
                self._edge_indices = None
        else:
            print(f"plant批量缓存文件不存在: {cache_path}")
            print(f"  提示: 请先调用 dataset.precompute_all_structures() 生成缓存")

    def precompute_all_structures(self, batch_size=100, num_workers=None, show_progress=True):
        """
        预计算所有序列的二级结构并保存到批量缓存文件（支持多进程）

        Args:
            batch_size (int): 每次调用LinearFold的序列数量
            num_workers (int): 工作进程数，None表示使用CPU核心数
            show_progress (bool): 是否显示进度条

        Returns:
            dict: 统计信息
        """
        from tqdm import tqdm
        from multiprocessing import Pool, cpu_count

        # Import worker function from human module
        from human import _worker_process_batch

        num_samples = len(self.sequences)
        cache_path = self._get_batch_cache_path()

        # 确定工作进程数
        if num_workers is None:
            num_workers = cpu_count()

        use_multiprocessing = num_workers > 1

        print(f"\n{'='*60}")
        print(f"开始预计算plant所有序列的二级结构...")
        print(f"  样本总数: {num_samples}")
        print(f"  批量大小: {batch_size}")
        print(f"  工作进程数: {num_workers if use_multiprocessing else 1} ({'多进程' if use_multiprocessing else '单进程'})")
        print(f"  缓存路径: {cache_path}")
        print(f"{'='*60}\n")

        # 初始化边索引数组
        edge_indices = np.empty(num_samples, dtype=object)
        stats = {
            'total': num_samples,
            'computed': 0,
            'failed': 0
        }

        # 准备批处理任务
        batch_tasks = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))
            batch_tasks.append((batch_indices, self.sequences, LINEARFOLD_PATH))

        total_batches = len(batch_tasks)

        if use_multiprocessing:
            # 多进程处理
            print(f"使用 {num_workers} 个进程并行处理 {total_batches} 个批次...\n")

            with Pool(processes=num_workers) as pool:
                # 使用imap_unordered获取结果并显示进度
                results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)

                if show_progress:
                    results_iter = tqdm(results_iter, total=total_batches, desc="预计算plant二级结构")

                # 收集结果
                for batch_results in results_iter:
                    for idx, edge_index_numpy, error in batch_results:
                        if error is None:
                            edge_indices[idx] = edge_index_numpy
                            stats['computed'] += 1
                        else:
                            stats['failed'] += 1
                            if stats['failed'] <= 5:
                                print(f"  警告: 索引 {idx} 计算失败: {error}")

        else:
            # 单进程处理
            iterator = range(0, num_samples, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="预计算plant二级结构")

            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))

                # 准备批量序列字符串
                sequences_str = []
                for idx in batch_indices:
                    sequence_bytes = self.sequences[idx].copy()
                    sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                    sequences_str.append(sequence_str)

                # 使用LinearFold批量计算二级结构
                try:
                    structures = run_linearfold(sequences_str)

                    # 构建边索引
                    for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
                        edge_index = build_edge_index_from_structure(sequences_str[i], structure)
                        edge_indices[idx] = edge_index.cpu().numpy()
                        stats['computed'] += 1

                except Exception as e:
                    print(f"\n警告: 批量计算失败 (索引 {start_idx}-{end_idx}): {e}")
                    for idx in batch_indices:
                        stats['failed'] += 1

        # 保存到批量缓存文件
        print(f"\n正在保存plant批量缓存到: {cache_path}")
        try:
            np.savez_compressed(
                cache_path,
                edge_indices=edge_indices,
                mode='plant',
                num_samples=num_samples
            )

            file_size_mb = os.path.getsize(cache_path) / (1024**2)
            print(f"  缓存已保存，文件大小: {file_size_mb:.2f} MB")

        except Exception as e:
            print(f"  错误: 保存批量缓存失败: {e}")
            return

        # 加载到内存
        self._batch_cache = np.load(cache_path, allow_pickle=True)
        self._edge_indices = self._batch_cache['edge_indices']

        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"Plant预计算完成！")
        print(f"  总样本数: {stats['total']}")
        print(f"  成功计算: {stats['computed']}")
        print(f"  失败: {stats['failed']}")
        print(f"{'='*60}\n")

    def _get_cache_key(self, sequence_str, idx):
        """
        生成缓存键值

        Args:
            sequence_str (str): RNA序列字符串
            idx (int): 样本索引

        Returns:
            str: 缓存键值
        """
        sequence_hash = hashlib.md5(sequence_str.encode('utf-8')).hexdigest()
        cache_key = f"plant_{idx}_{sequence_hash}"
        return cache_key

    def _get_cache_path(self, cache_key):
        """
        获取缓存文件路径

        Args:
            cache_key (str): 缓存键值

        Returns:
            str: 缓存文件完整路径
        """
        return os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key):
        """
        从缓存加载二级结构

        Args:
            cache_key (str): 缓存键值

        Returns:
            torch.Tensor or None: 边索引
        """
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['edge_index']
            except (pickle.PickleError, EOFError, KeyError):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return None

        return None

    def _save_to_cache(self, cache_key, edge_index):
        """
        将二级结构保存到缓存

        Args:
            cache_key (str): 缓存键值
            edge_index (torch.Tensor): 边索引张量
        """
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            cached_data = {
                'edge_index': edge_index,
                'mode': 'plant'
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except (pickle.PickleError, OSError):
            pass

    def _get_or_compute_edge_index(self, sequence_str, idx):
        """
        获取或计算边索引（优先使用批量缓存）

        Args:
            sequence_str (str): RNA序列字符串
            idx (int): 样本索引

        Returns:
            torch.Tensor: 边索引张量
        """
        # 1. 优先使用批量缓存
        if self._edge_indices is not None and idx < len(self._edge_indices):
            cached = self._edge_indices[idx]
            if cached is not None:
                return torch.from_numpy(cached)

        # 2. 回退到旧的单文件缓存机制
        cache_key = self._get_cache_key(sequence_str, idx)
        cached_edge_index = self._load_from_cache(cache_key)
        if cached_edge_index is not None:
            return cached_edge_index

        # 3. 缓存未命中，使用LinearFold计算二级结构
        try:
            structures = run_linearfold([sequence_str])
            structure = structures[0]
            edge_index = build_edge_index_from_structure(sequence_str, structure)

            # 保存到旧缓存
            self._save_to_cache(cache_key, edge_index)

            return edge_index
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
            raise

    def _one_hot_encode_optimized(self, sequence_bytes):
        """
        优化的one-hot编码，直接处理|S1字节流

        Args:
            sequence_bytes (np.array): |S1类型的字节数组

        Returns:
            np.array: one-hot编码后的数组，shape为(1001, 4)
        """
        byte_array = sequence_bytes.view(np.uint8)
        one_hot = _BYTE_TO_ONEHOT_MAPPING[byte_array]
        return one_hot.astype(np.float32)

    def clear_cache(self):
        """
        清除当前模式的所有缓存文件
        """
        if not os.path.exists(self.CACHE_DIR):
            return

        removed_count = 0
        prefix = "plant_"

        # 清除单文件缓存
        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    os.remove(cache_path)
                    removed_count += 1
                except OSError:
                    pass

        print(f"已清除 {removed_count} 个plant单文件缓存")

        # 清除批量缓存
        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                os.remove(batch_cache_path)
                file_size_mb = os.path.getsize(batch_cache_path) / (1024**2)
                print(f"已清除plant批量缓存: {batch_cache_path} ({file_size_mb:.2f} MB)")
            except OSError as e:
                print(f"清除批量缓存失败: {e}")

        # 清除内存中的缓存
        self._batch_cache = None
        self._edge_indices = None

    def get_cache_stats(self):
        """
        获取缓存统计信息

        Returns:
            dict: 包含缓存统计信息的字典
        """
        stats = {
            'cache_dir': self.CACHE_DIR,
            'batch_cache': None,
            'single_file_cache': {
                'total_files': 0,
                'total_size_mb': 0.0
            }
        }

        if not os.path.exists(self.CACHE_DIR):
            return stats

        # 统计单文件缓存
        prefix = "plant_"
        total_size = 0
        file_count = 0

        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    file_size = os.path.getsize(cache_path)
                    file_count += 1
                    total_size += file_size
                except OSError:
                    pass

        stats['single_file_cache']['total_files'] = file_count
        stats['single_file_cache']['total_size_mb'] = total_size / (1024 * 1024)

        # 统计批量缓存
        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                batch_size_mb = os.path.getsize(batch_cache_path) / (1024 * 1024)
                stats['batch_cache'] = {
                    'exists': True,
                    'path': batch_cache_path,
                    'size_mb': batch_size_mb,
                    'loaded_in_memory': self._edge_indices is not None
                }
            except OSError:
                stats['batch_cache'] = {'exists': False}
        else:
            stats['batch_cache'] = {'exists': False}

        return stats

    def get_few_shot_split(self, k_shots, valid_classes=[5, 8, 9], test_ratio=0.9, seed=42):
        """
        Generate indices for Few-Shot Support Set and Fixed Query (Test) Set.
        Ratio: 10% Support Pool, 90% Fixed Test Set.

        Args:
            k_shots (int): Number of samples to sample per class (0 for zero-shot)
            valid_classes (list): List of valid class indices to sample from (default: [5, 8, 9])
            test_ratio (float): Ratio of data to reserve as fixed test set (default: 0.9, i.e., 90%)
            seed (int): Random seed for reproducibility

        Returns:
            tuple: (support_indices, test_indices)
                - support_indices (list): Indices of samples selected for fine-tuning (k per valid class)
                - test_indices (list): Indices of samples reserved for fixed testing (90% of data)
        """
        np.random.seed(seed)
        num_samples = len(self.sequences)
        indices = np.arange(num_samples)

        # 1. Fixed Split: Support Pool (10%) vs Fixed Test Set (90%)
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)

        # split_point defines the size of the Support Pool (1 - test_ratio)
        split_point = int(num_samples * (1 - test_ratio))
        pool_indices = shuffled_indices[:split_point]  # 10% for sampling shots
        test_indices = shuffled_indices[split_point:]  # 90% fixed for testing

        if k_shots == 0:
            return [], test_indices.tolist()

        # 2. Sample k_shots from the Support Pool
        support_indices_set = set()
        pool_labels = self.y_12class[pool_indices]

        for cls_idx in valid_classes:
            # Find positive samples for this class within the pool
            cls_pos_relative_indices = np.where(pool_labels[:, cls_idx] == 1)[0]

            if len(cls_pos_relative_indices) < k_shots:
                # If not enough samples, take all available
                selected_relative = cls_pos_relative_indices
            else:
                selected_relative = np.random.choice(cls_pos_relative_indices, k_shots, replace=False)

            # Convert relative pool indices back to global dataset indices
            selected_global = pool_indices[selected_relative]
            support_indices_set.update(selected_global.tolist())

        return list(support_indices_set), test_indices.tolist()
