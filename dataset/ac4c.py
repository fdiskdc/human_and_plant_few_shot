"""
ac4c.py - ac4C 修饰专用的12类多标签分类数据集 / ac4C Modification Specific 12-class Multi-Label Classification Dataset

本模块加载 ac4C (N4-acetylcytidine, 4类位点) 修饰数据并复用12类多标签分类框架。与 human.py
格式类似但不含 1001loc.npy 位点级标签，因此使用全0的虚拟 full_labels 保持接口一致。
数据集分为 balanced 与 unbalanced 两个版本，默认使用 balanced 版本。提供少样本切分工具。
This module loads ac4C (N4-acetylcytidine) modification data and reuses the 12-class multi-label
classification framework. Unlike human.py, it lacks 1001loc.npy (site-level) labels, so a virtual
all-zero full_labels is used to keep the interface consistent. Two versions are available:
balanced (default) and unbalanced, with few-shot splitting support.

功能模块 / Modules:
- AC4CDataset: PyG Dataset类,加载balanced_ac4c或unbalanced_ac4c子目录下train/test的npy数据 / PyG Dataset class loading npy data from balanced_ac4c or unbalanced_ac4c subdirs
- get_few_shot_split: 生成 Few-Shot Support Pool (10%) 与 Fixed Test Set (90%) 索引 / Generates Few-Shot support pool (10%) and fixed test set (90%) indices
- precompute_all_structures: 多进程预计算所有序列二级结构 / Multiprocess precomputation of all sequence secondary structures
- _generate_uniform_attention_masks: AC4C无位点标签,使用均匀分布注意力掩码 / AC4C has no site-level labels, so uniform attention masks are used

输入 / Inputs:
- npy/ac4c_processed/balanced_ac4c/train|test/seq.npy: NumPy字节数组, 形状 (N, 1001) |S1 - ac4C RNA序列 / ac4C RNA sequences
- npy/ac4c_processed/balanced_ac4c/train|test/12loc.npy: NumPy int8数组, 形状 (N, 12) - 12类多标签 / 12-class multi-labels
- npy/ac4c_processed/balanced_ac4c/train|test/4loc.npy: NumPy int8数组, 形状 (N, 4) - 4类核苷酸组标签 / 4-class nucleotide group labels
- (无1001loc.npy - 用全0虚拟标签填充) / (No 1001loc.npy - filled with virtual zero labels)
- 配置文件 / Config: LINEARFOLD_PATH, cache_dir, data_dir - 路径与缓存配置 / Path and cache configuration

输出 / Outputs:
- PyG Data对象 / PyG Data objects: x=(1001,4) one-hot, edge_index=(2,E) 边索引, y=(1,12) 12类多标签, y_4class=(1,4) 4类标签, y_site=(1001,) 全0虚拟标签 / x: one-hot; edge_index: edges; y: 12-class (1,12); y_4class: (1,4); y_site: virtual zeros (1001,)
- 均匀注意力掩码 / Uniform attention masks: attn_mask_A/C/G/U (1001,) - 全1/1001均匀分布 / All uniform 1/1001
- Few-Shot Split: (support_indices, test_indices) - 元组 / Tuple

数据流 / Data Flow:
1. 加载npy / Load npy: 检测balanced/unbalanced目录,加载train或test子目录的seq/12loc/4loc / Detect balanced/unbalanced, load seq/12loc/4loc from train/test subdirs
2. 虚拟full_label生成 / Virtual full_label: 由于没有1001loc,生成全0的 (N, 1001) 数组 / Generate all-zero (N, 1001) since no 1001loc available
3. 字节流one-hot编码 / Byte-to-onehot: 使用_BYTE_TO_ONEHOT_MAPPING查表 / Use lookup table for one-hot encoding
4. 均匀注意力掩码 / Uniform masks: 为4个核苷酸生成均匀分布掩码 / Generate uniform masks for 4 nucleotides
5. 构建PyG Data / Build PyG Data: 整合所有字段返回PyG Data对象 / Integrate all fields into PyG Data

相关文件 / Related Files:
- 调用 / Calls: torch.utils.data.Dataset, torch_geometric.data.Data, subprocess (LinearFold), human.py (复用常量) / Reuses from human.py
- 被调用 / Called by: fewshot_ac4c_balance.py, fewshot_ac4c_unbalan.py, train_human.py, train_human_modx.py, train_human_multirm.py, utils/train_gen3.py

使用示例 / Usage Example:
    from dataset.ac4c import AC4CDataset
    train_set = AC4CDataset(mode='train', data_dir='npy/ac4c_processed/balanced_ac4c')
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    for batch in loader:
        x, edge_index, y = batch.x, batch.edge_index, batch.y

作者 / Author: RGCNFormer Project (C. Deng, DOI:10.3390/app15158626)
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

'''
Author: Chao Deng && chaodeng987@outlook.com
Date: 2026-01-09 16:22:08
LastEditors: Chao Deng && chaodeng987@outlook.com
LastEditTime: 2026-01-09 16:30:28
FilePath: /human_and_plant_few_shot_fastAtten_ac4c/dataset/ac4c.py
Description: 
那只是一场游戏一场梦
 
https://orcid.org/0009-0009-8520-1656
DOI: 10.3390/app15158626
DOI: 10.3390/rs17142354
Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
'''
"""
AC4C RNA Multi-label Classification Dataset

This module provides a dataset class for loading AC4C RNA modification data.
The AC4C data structure:
- seq.npy: (N, 1001) byte array for RNA sequences
- 12loc.npy: (N, 12) int8 array for 12-class labels
- 4loc.npy: (N, 4) int8 array for nucleotide group labels
- Note: No 1001loc.npy file

Data is organized in balanced_ac4c and unbalanced_ac4c directories,
with train/ and test/ subdirectories.
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


class AC4CDataset(Dataset):
    """
    AC4C RNA 修饰数据集 / AC4C RNA modification dataset.

    与 Mer100Dataset 格式类似但不包含 1001loc.npy。分为 balanced / unbalanced 两个版本。
    Same format as Mer100Dataset but without 1001loc.npy. Available in balanced/unbalanced versions.

    Attributes / 属性:
        mode (str): [中文] 'train'/'test' / [English] split mode.
        data_dir (str): [中文] 数据目录 / [English] data directory.
    """

    def __init__(self, mode='train', data_dir='npy/ac4c_processed/balanced_ac4c',
                 cache_dir=None, use_cache=True, preload_cache=True):
        """
        初始化 AC4CDataset / Initialize AC4CDataset.

        Args / 参数:
            mode (str): [中文] 'train'/'test' / [English] split mode. Defaults to 'train'.
            data_dir (str): [中文] 数据目录 / [English] data directory. Defaults to balanced_ac4c.
            cache_dir (Optional[str]): [中文] 缓存目录 / [English] cache directory.
            use_cache (bool): [中文] 启用缓存 / [English] enable cache. Defaults to True.
            preload_cache (bool): [中文] 预加载 / [English] preload. Defaults to True.
        """
        self.mode = mode
        self.data_dir = data_dir
        self.use_cache = use_cache
        self._batch_cache = None
        self._edge_indices = None

        # 设置缓存目录
        if cache_dir is None:
            self.CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'npy/cache')
        else:
            self.CACHE_DIR = cache_dir

        # 确保缓存目录存在
        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)

        # 检测数据目录
        if os.path.exists(data_dir):
            self.base_dir = data_dir
        elif os.path.exists(f'../{data_dir}'):
            self.base_dir = f'../{data_dir}'
        else:
            # 尝试相对于项目根目录的路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt_path = os.path.join(project_root, data_dir)
            if os.path.exists(alt_path):
                self.base_dir = alt_path
            else:
                raise FileNotFoundError(f"找不到AC4C数据目录: {data_dir}")

        # 构建完整的训练/测试集路径
        self.data_path = os.path.join(self.base_dir, mode)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到{mode}数据目录: {self.data_path}")

        print(f"使用内存映射加载AC4C {mode}数据: {self.data_path}")

        # 使用mmap_mode='r'进行内存映射加载
        # 注意：AC4C数据集没有1001loc.npy文件
        self.sequences = np.load(f'{self.data_path}/seq.npy', mmap_mode='r')
        self.y_12class = np.load(f'{self.data_path}/12loc.npy', mmap_mode='r')
        self.y_4class = np.load(f'{self.data_path}/4loc.npy', mmap_mode='r')

        # 生成虚拟的full_label（全0），保持与human.py接口一致
        # AC4C数据集没有位点级别的标签，所以full_label全是0
        self.full_labels = np.zeros((len(self.sequences), TARGET_LENGTH), dtype=np.int8)

        print(f"AC4C数据集初始化完成 (mode={mode}):")
        print(f"  总样本数: {len(self.sequences)}")
        print(f"  序列形状: {self.sequences.shape}, dtype: {self.sequences.dtype}")
        print(f"  12loc形状: {self.y_12class.shape}, dtype: {self.y_12class.dtype}")
        print(f"  4loc形状: {self.y_4class.shape}, dtype: {self.y_4class.dtype}")
        print(f"  注: AC4C数据集没有1001loc.npy文件，使用全0的虚拟标签")

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
                - y_4class: 4类核苷酸组标签 (1, 4)
                - y_site: 1001长度位点标签 (1001,) - AC4C中全为0
        """
        # 获取序列和标签（使用copy()确保多线程安全）
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()  # 虚拟的全0标签
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

        # 生成注意力掩码（用于监督）
        # 由于AC4C没有1001loc，所有注意力掩码都是均匀分布
        attn_masks = self._generate_uniform_attention_masks()

        # 生成N字符掩码（用于标记未知核苷酸位置）
        attn_mask_N = self._extract_attention_masks_N(sequence_bytes)

        # 创建PyG Data对象，整合所有层级标签
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),  # 形状为 [1, 12]
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),  # 形状为 [1, 4]
            y_site=torch.LongTensor(full_label)  # 形状为 [1001]，全为0
        )

        # 将注意力掩码添加为data的属性
        for nuc, mask in attn_masks.items():
            setattr(data, f'attn_mask_{nuc}', mask)

        # 添加N字符掩码
        setattr(data, 'attn_mask_N', attn_mask_N)

        return data

    def _generate_uniform_attention_masks(self):
        """
        生成均匀分布的注意力掩码（AC4C没有位点级别的修饰信息）

        Returns:
            dict: 包含4个核苷酸注意力掩码的字典，都是均匀分布
        """
        attn_masks = {}
        for nuc in ['A', 'C', 'G', 'U']:
            # 创建均匀分布的注意力掩码
            attn_masks[nuc] = torch.ones(1001, dtype=torch.float32) / 1001
        return attn_masks

    def _extract_attention_masks_N(self, sequence_bytes):
        """
        从RNA序列中提取'N'字符的掩码

        Args:
            sequence_bytes (np.array): |S1类型的字节数组，长度为1001

        Returns:
            torch.Tensor: 掩码张量，形状为(1001,)，'N'字符位置为1，其他位置为0
        """
        # 将字节数组转换为整数数组
        byte_array = sequence_bytes.view(np.uint8)

        # 创建掩码：'N'的ASCII码是78和110
        mask = np.zeros(1001, dtype=np.float32)
        mask[(byte_array == 78) | (byte_array == 110)] = 1.0

        return torch.FloatTensor(mask)

    def _get_batch_cache_path(self):
        """
        获取批量缓存文件路径

        Returns:
            str: 批量缓存文件完整路径
        """
        # 根据数据目录名称生成缓存文件名
        # 例如：balanced_ac4c -> balanced, unbalanced_ac4c -> unbalanced
        dir_name = os.path.basename(os.path.normpath(self.base_dir))
        # 去掉_ac4c后缀
        cache_name = dir_name.replace('_ac4c', '')
        # 添加数据集名称前缀
        return os.path.join(self.CACHE_DIR, f"ac4c_{cache_name}_{self.mode}_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        """
        从批量缓存文件加载所有边索引到内存

        如果缓存文件存在，直接加载；如果不存在，则不进行任何操作
        """
        cache_path = self._get_batch_cache_path()

        if os.path.exists(cache_path):
            print(f"正在从批量缓存加载AC4C边索引: {cache_path}")
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
            print(f"AC4C批量缓存文件不存在: {cache_path}")
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
        print(f"开始预计算AC4C {self.mode}所有序列的二级结构...")
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
                    results_iter = tqdm(results_iter, total=total_batches, desc=f"预计算AC4C {self.mode}二级结构")

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
                iterator = tqdm(iterator, desc=f"预计算AC4C {self.mode}二级结构")

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
        print(f"\n正在保存AC4C批量缓存到: {cache_path}")
        try:
            np.savez_compressed(
                cache_path,
                edge_indices=edge_indices,
                mode=self.mode,
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
        print(f"AC4C {self.mode}预计算完成！")
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
        # 使用数据目录名称和模式确保唯一性
        dir_name = os.path.basename(os.path.normpath(self.base_dir))
        cache_key = f"{dir_name}_{self.mode}_{idx}_{sequence_hash}"
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
                'mode': self.mode,
                'dataset': 'ac4c'
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
        # 使用数据目录名称作为前缀
        dir_name = os.path.basename(os.path.normpath(self.base_dir))
        prefix = f"{dir_name}_{self.mode}_"

        # 清除单文件缓存
        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    os.remove(cache_path)
                    removed_count += 1
                except OSError:
                    pass

        print(f"已清除 {removed_count} 个AC4C {self.mode}单文件缓存")

        # 清除批量缓存
        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                os.remove(batch_cache_path)
                file_size_mb = os.path.getsize(batch_cache_path) / (1024**2)
                print(f"已清除AC4C批量缓存: {batch_cache_path} ({file_size_mb:.2f} MB)")
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
        dir_name = os.path.basename(os.path.normpath(self.base_dir))
        prefix = f"{dir_name}_{self.mode}_"
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

    def get_few_shot_split(self, k_shots, valid_classes=None, test_ratio=0.9, seed=42):
        """
        Generate indices for Few-Shot Support Set and Fixed Query (Test) Set.
        Ratio: 10% Support Pool, 90% Fixed Test Set.

        Args:
            k_shots (int): Number of samples to sample per class (0 for zero-shot)
            valid_classes (list): List of valid class indices to sample from.
                                 If None, use all classes with positive samples.
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

        # 2. Determine valid classes if not specified
        if valid_classes is None:
            # Find all classes that have at least one positive sample in the pool
            pool_labels = self.y_12class[pool_indices]
            valid_classes = []
            for cls_idx in range(12):
                if np.any(pool_labels[:, cls_idx] == 1):
                    valid_classes.append(cls_idx)

        # 3. Sample k_shots from the Support Pool
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


# 测试代码
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    print("=" * 60)
    print("=== 测试1: 加载balanced AC4C训练集 ===")
    print("=" * 60)
    
    dataset = AC4CDataset(mode='train', data_dir='npy/ac4c_processed/balanced_ac4c')
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取单个样本
    print("\n测试获取单个样本:")
    sample = dataset[0]
    print(f"  节点特征形状: {sample.x.shape}")
    print(f"  边索引形状: {sample.edge_index.shape}")
    print(f"  y (12类) 形状: {sample.y.shape}, 值: {sample.y}")
    print(f"  y_4class (4类) 形状: {sample.y_4class.shape}, 值: {sample.y_4class}")
    print(f"  y_site (1001位点) 形状: {sample.y_site.shape}, 非零元素: {sample.y_site.sum().item()}")
    print(f"  注意力掩码:")
    for nuc in ['A', 'C', 'G', 'U']:
        mask_attr = f'attn_mask_{nuc}'
        if hasattr(sample, mask_attr):
            mask = getattr(sample, mask_attr)
            if mask is not None and hasattr(mask, 'shape'):
                print(f"    {nuc}: 形状={mask.shape}, 均值={mask.mean().item():.6f}")
            else:
                print(f"    {nuc}: 掩码为None或无效")
        else:
            print(f"    {nuc}: 属性不存在")
    
    print("\n" + "=" * 60)
    print("=== 测试2: 加载unbalanced AC4C测试集 ===")
    print("=" * 60)
    
    dataset_test = AC4CDataset(mode='test', data_dir='npy/ac4c_processed/unbalanced_ac4c')
    print(f"测试集大小: {len(dataset_test)}")
    
    print("\n" + "=" * 60)
    print("=== 测试3: PyG DataLoader测试 ===")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
    print(f"DataLoader配置: batch_size=32, num_workers=0 (单进程)")
    
    print("\n获取第一个批次:")
    batch = next(iter(loader))
    
    print(f"  batch.x 形状: {batch.x.shape}")
    print(f"  batch.edge_index 形状: {batch.edge_index.shape}")
    print(f"  batch.y 形状: {batch.y.shape}")
    print(f"  batch.y_4class 形状: {batch.y_4class.shape}")
    print(f"  batch.y_site 形状: {batch.y_site.shape}")
    
    # 验证维度
    if batch.y is not None and hasattr(batch.y, 'shape'):
        assert batch.y.shape == (32, 12), f"batch.y形状错误: {batch.y.shape}"
        print(f"  ✓ batch.y形状正确: {batch.y.shape}")
    if batch.y_4class is not None and hasattr(batch.y_4class, 'shape'):
        assert batch.y_4class.shape == (32, 4), f"batch.y_4class形状错误: {batch.y_4class.shape}"
        print(f"  ✓ batch.y_4class形状正确: {batch.y_4class.shape}")
    if batch.y_site is not None and hasattr(batch.y_site, 'shape'):
        expected_shape = (32 * 1001,)
        assert batch.y_site.shape == expected_shape, f"batch.y_site形状错误: {batch.y_site.shape}, 期望: {expected_shape}"
        print(f"  ✓ batch.y_site形状正确: {batch.y_site.shape} (32个样本 × 1001个位点)")
    print("\n✓ 所有维度验证通过！")
    
    print("\n" + "=" * 60)
    print("=== 所有测试完成！===")
    print("=" * 60)