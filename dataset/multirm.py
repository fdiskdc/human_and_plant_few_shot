"""
multirm.py - MultIRM 多类别RNA修饰分类数据集 / MultIRM Multi-class RNA Modification Classification Dataset

本模块实现 MultIRM 数据集的高性能加载器，支持 12 种 RNA 修饰的多类分类任务。
采用"预计算 Tensor 驻留"策略：初始化时将所有数据预转为 Tensor 格式，__getitem__ 期间
零计算开销仅做查表。支持训练集 Oversampling with Max-Length Alignment 策略以平衡各类样本数。
This module implements a high-performance loader for the MultIRM dataset, supporting multi-class
classification of 12 RNA modification types. It uses a "precomputed tensor resident" strategy:
all data is pre-converted to tensors at init, so __getitem__ is pure lookup with zero compute.
Training mode also supports Oversampling with Max-Length Alignment to balance class sample counts.

功能模块 / Modules:
- MultirmDataset: PyG Dataset类,加载 npy/multirm/51split/{class}/{pos|neg}/{mode}_in.npy 等数据 / PyG Dataset class loading pos/neg samples per class
- _load_raw_data: 加载所有类 (12种) 的正负样本 {train,test,valid}_in.npy 与 {train,test,valid}_out.npy / Loads pos/neg samples for all 12 classes
- _load_structure_cache: 一次性从 H5 缓存加载所有序列的二级结构 / One-shot load of all sequence structures from H5 cache
- _build_indices_map: 训练模式生成 Oversampling with Max-Length Alignment 虚拟索引 / Generates oversampling virtual indices for training
- precompute_all_structures: 预计算二级结构并保存为 H5 缓存 (区别于 human.py 的 npz) / Precomputes structures and saves as H5 cache

输入 / Inputs:
- npy/multirm/51split/{class_name}/{pos|neg}/{mode}_in.npy: NumPy字节数组, 1001nt RNA序列 / 1001nt RNA sequences per class
- npy/multirm/51split/{class_name}/{pos|neg}/{mode}_out.npy: NumPy数组, 12类多标签 / 12-class multi-labels
- npy/multirm/51split/{class_name}/{pos|neg}/{mode}_out_4class.npy: NumPy数组, 4类核苷酸组标签 (可选) / Optional 4-class nucleotide group labels
- 配置文件 / Config: LINEARFOLD_PATH, cache_dir, numsample, use_4class - 缓存目录与采样参数 / Cache dir and sampling params
- 12类修饰 / 12 classes: Am, Cm, Gm, Um, m1A, m5C, m5U, m6A, m6Am, m7G, Psi, AtoI

输出 / Outputs:
- PyG Data对象 / PyG Data objects: x=(1001,4) one-hot (Tensor), edge_index=(2,E) 边索引, y=(1,12) 12类多标签, class_idx=(1,) 类别索引, y_4=(1,4) 4类组标签 (可选) / Pre-built tensors for zero-compute lookup
- H5缓存 / H5 cache: multirm_{mode}_structures_cache.h5 - 二级结构边索引HDF5缓存 / HDF5 cache of secondary structure edge indices
- 虚拟索引 / Virtual indices: training模式下使用 oversampling 索引, virtual_length = max_class_len * num_buckets

数据流 / Data Flow:
1. 加载原始npy / Load raw npy: 遍历12类×pos/neg加载所有 sequence_bytes / label / label_4class / Iterate 12 classes x pos/neg, load all samples
2. 预处理为Tensor / Preprocess to Tensor: 用 BYTE_TO_INDEX 查表 + ONE_HOT_EMB 嵌入矩阵直接生成 (1001,4) Tensor / Use BYTE_TO_INDEX + ONE_HOT_EMB for direct tensor generation
3. 加载H5边索引缓存 / Load H5 edge cache: 一次性读入所有边索引到内存, 边索引查表零计算 / One-shot load of all edges into memory for zero-compute lookup
4. 训练模式Oversampling / Training oversampling: 按 (class_name, is_pos) 分桶, 各桶扩展到 max_len 形成虚拟索引 / Group by (class, pos/neg), expand each bucket to max_len
5. __getitem__ 查表 / __getitem__ lookup: 根据虚拟索引直接返回预构建 Tensor Data对象 / Return pre-built Tensor Data by virtual index

相关文件 / Related Files:
- 调用 / Calls: torch.utils.data.Dataset, torch_geometric.data.Data, subprocess (LinearFold), h5py (HDF5缓存) / Uses h5py for H5 cache
- 被调用 / Called by: train_multirm_dataset.py, test_multirm_4class.py, test_multirm_oversampling.py

使用示例 / Usage Example:
    from dataset.multirm import MultirmDataset
    train_set = MultirmDataset(numsample=50, mode='train', use_4class=True)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    for batch in loader:
        x, edge_index, y = batch.x, batch.edge_index, batch.y

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import os
import hashlib
import pickle
import h5py
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# MultIRM modification classes (12 classes) - new order
MULTIRM_CLASSES = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Psi', 'AtoI']

# 4-class grouping for hierarchical classification
# Group 0 (A): Am, m1A, m6A, m6Am, AtoI
# Group 1 (C): Cm, m5C
# Group 2 (G): Gm, m7G
# Group 3 (U): Um, m5U, Psi
MULTIRM_4CLASS_NAMES = ['A', 'C', 'G', 'U']

# Mapping from 12-class index to 4-class group index
MULTIRM_12TO4_MAPPING = {
    0: 0,   # Am -> A
    1: 1,   # Cm -> C
    2: 2,   # Gm -> G
    3: 3,   # Um -> U
    4: 0,   # m1A -> A
    5: 1,   # m5C -> C
    6: 3,   # m5U -> U
    7: 0,   # m6A -> A
    8: 0,   # m6Am -> A
    9: 2,   # m7G -> G
    10: 3,  # Psi -> U
    11: 0   # AtoI -> A
}

# Class to index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(MULTIRM_CLASSES)}

# Index to class mapping
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# One-hot编码映射
ONE_HOT_MAPPING = {
    'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.],
    'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
    'U': [0., 0., 0., 1.], 'N': [0., 0., 0., 0.]
}

# 目标序列长度
TARGET_LENGTH = 1001

# LinearFold路径
LINEARFOLD_PATH = '/home/dc/vscode/LinearFold/linearfold'

# 批量缓存文件名
BATCH_CACHE_FILE = 'structures_cache.npz'

# 预计算的 Byte 到 One-Hot 索引映射 (0-4: A,C,G,T,U, N=4)
BYTE_TO_INDEX = np.zeros(256, dtype=np.int64)
BYTE_TO_INDEX[:] = 4  # Default to N
for b, i in zip([65, 97, 67, 99, 71, 103, 84, 116, 85, 117], [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]):
    BYTE_TO_INDEX[b] = i

# 预计算的 One-Hot 嵌入矩阵 (5, 4) - 最后一行全是0对应 'N'
eye = torch.eye(4, dtype=torch.float32)
zero_row = torch.zeros(1, 4, dtype=torch.float32)
ONE_HOT_EMB = torch.cat([eye, zero_row], dim=0)


def run_linearfold(sequences, timeout_seconds=1800):
    """
    使用 LinearFold 预测 RNA 二级结构 / Predict RNA secondary structures with LinearFold.

    Args / 参数:
        sequences (list): [中文] RNA 序列列表 / [English] list of RNA sequences.
        timeout_seconds (int): [中文] 超时 / [English] timeout. Defaults to 1800.

    Returns / 返回:
        list: [中文] 二级结构列表 / [English] list of structures.

    Raises / 异常:
        RuntimeError: [中文] LinearFold 失败 / [English] LinearFold failed.
        FileNotFoundError: [中文] 可执行文件不存在 / [English] missing executable.
    """
    if not sequences:
        return []

    fasta_input = '\n'.join([f'>seq_{i}\n{seq}' for i, seq in enumerate(sequences)])

    try:
        if not os.path.exists(LINEARFOLD_PATH):
            raise FileNotFoundError(f"LinearFold可执行文件不存在: {LINEARFOLD_PATH}")

        process = subprocess.Popen(
            [LINEARFOLD_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        stdout_data, stderr_data = process.communicate(input=fasta_input, timeout=timeout_seconds)

        if process.returncode != 0:
            raise RuntimeError(
                f"LinearFold执行失败，返回码 {process.returncode}。"
                f"错误信息: {stderr_data[:500]}"
            )

        if len(stdout_data.strip()) == 0:
            raise RuntimeError("LinearFold没有返回任何输出")

        lines = stdout_data.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        structures = []
        for i in range(2, len(lines), 3):
            line = lines[i]
            if not line:
                continue
            structure = line.split()[0]
            structures.append(structure)

        if len(structures) != len(sequences):
            raise RuntimeError(
                f"LinearFold返回的结构数量({len(structures)})与输入序列数量({len(sequences)})不匹配"
            )

    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired:
        raise
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold执行失败: {e}")
    except Exception as e:
        raise RuntimeError(f"LinearFold执行过程中发生未知错误: {e}")

    return structures


def build_edge_index_from_structure(sequence, structure):
    """
    根据 RNA 二级结构构建边索引 / Build edge index from RNA secondary structure.

    Args / 参数:
        sequence (str): [中文] RNA 序列 / [English] RNA sequence.
        structure (str): [中文] 二级结构 (点括号) / [English] dot-bracket structure.

    Returns / 返回:
        torch.Tensor: [中文] 边索引 `(2, E)` / [English] edge index tensor.
    """
    if not structure:
        return build_sequential_edge_index(sequence)

    stack = []
    pairs = {}

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j

    edge_list = []

    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])

    for i, j in pairs.items():
        if i < j:
            edge_list.extend([(i, j), (j, i)])

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


def build_sequential_edge_index(sequence):
    """
    仅构建顺序边 `(i, i+1)` / Build only sequential edges.

    Args / 参数:
        sequence (str): [中文] RNA 序列 / [English] RNA sequence.

    Returns / 返回:
        torch.Tensor: [中文] 边索引 `(2, E)` / [English] edge index tensor.
    """
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


class MultirmDataset(Dataset):
    """
    优化版 MultIRM 数据集加载器 / Optimized MultIRM dataset loader.

    核心优化 / Core optimizations:
    1. 内存驻留 Tensor (init 阶段预计算)
    2. 预计算索引映射 (Oversampling 策略)
    3. __getitem__ 纯查表 (零计算开销)

    Key optimizations:
    1. In-memory tensors (pre-computed at __init__)
    2. Pre-computed index maps (for oversampling)
    3. Pure table lookup in __getitem__

    Attributes / 属性:
        mode (str): [中文] 'train'/'test'/'valid' / [English] split mode.
        use_4class (bool): [中文] 是否 4 类模式 / [English] use 4-class mode.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        numsample: int = 50,
        mode: str = 'train',
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        preload_cache: bool = True,
        seed: int = 42,
        use_4class: bool = True
    ):
        """
        初始化 MultirmDataset / Initialize MultirmDataset.

        Args / 参数:
            data_dir (Optional[str]): [中文] 数据目录 / [English] data directory.
            numsample (int): [中文] 每类正/负样本数 / [English] pos/neg samples per class. Defaults to 50.
            mode (str): [中文] split 模式 / [English] split mode. Defaults to 'train'.
            cache_dir (Optional[str]): [中文] 缓存目录 / [English] cache directory.
            use_cache (bool): [中文] 启用缓存 / [English] enable cache. Defaults to True.
            preload_cache (bool): [中文] 预加载 / [English] preload. Defaults to True.
            seed (int): [中文] 随机种子 / [English] random seed. Defaults to 42.
            use_4class (bool): [中文] 4 类模式 / [English] 4-class mode. Defaults to True.
        """
        self.mode = mode
        self.use_4class = use_4class
        self.use_cache = use_cache
        self.seed = seed
        
        # 自动定位数据目录
        if data_dir is None:
            possible_paths = [
                'npy/multirm/51split',
                '../npy/multirm/51split',
                '/home/dc/vscode/vscode20260124/rgcnformer_sum/npy/multirm/51split'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_dir = path
                    break
            
            if data_dir is None:
                raise RuntimeError(
                    f"无法找到multirm数据目录。请手动指定data_dir参数。\n"
                    f"尝试过的路径: {possible_paths}"
                )
        
        self.data_dir = data_dir
        
        # 设置缓存目录
        if cache_dir is None:
            self.cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'npy', 'cache', 'multirm'
            )
        else:
            self.cache_dir = cache_dir
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # ========== 第1步：加载原始数据到内存 ==========
        print(f"\n{'='*60}")
        print(f"加载MultIRM数据集 (mode={self.mode}, numsample={numsample})")
        print(f"{'='*60}")
        raw_samples = self._load_raw_data()
        print(f"加载了 {len(raw_samples)} 个原始样本")
        
        # ========== 第2步：加载结构缓存 ==========
        print(f"正在加载结构缓存...")
        edge_cache = self._load_structure_cache()
        print(f"加载了 {len(edge_cache)} 个缓存条目")
        
        # ========== 第3步：预处理数据为 Tensor 格式（关键加速步骤）==========
        print(f"正在将 {len(raw_samples)} 个样本转换为 Tensor...")
        self.data_x = []
        self.data_edge_index = []
        self.data_y = []
        self.data_y_4 = []
        self.data_class_idx = []
        
        for sample in tqdm(raw_samples, desc="预处理数据"):
            # A. 处理序列特征
            seq_bytes = sample['sequence_bytes'].view(np.uint8)
            seq_indices = BYTE_TO_INDEX[seq_bytes]  # Map bytes to 0-4
            x = ONE_HOT_EMB[torch.from_numpy(seq_indices)]  # Vectorized lookup
            self.data_x.append(x)
            
            # B. 处理边索引 (Look up from loaded cache)
            cache_key = sample['cache_key']
            if cache_key in edge_cache:
                self.data_edge_index.append(edge_cache[cache_key])
            else:
                # Fallback: 仅生成线性边
                seq_len = len(seq_bytes)
                row = torch.arange(seq_len - 1, dtype=torch.long)
                col = torch.arange(1, seq_len, dtype=torch.long)
                edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
                self.data_edge_index.append(edge_index)
            
            # C. 处理标签
            self.data_y.append(torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(0))
            
            if sample['label_4class'] is not None:
                self.data_y_4.append(torch.tensor(sample['label_4class'], dtype=torch.float32).unsqueeze(0))
            else:
                self.data_y_4.append(None)
            
            self.data_class_idx.append(torch.tensor([CLASS_TO_IDX[sample['class_name']]], dtype=torch.long))
        
        # ========== 第4步：构建索引映射 (Oversampling Logic) ==========
        self.indices_map = self._build_indices_map(raw_samples)
        print(f"数据集初始化完成. 虚拟长度: {len(self.indices_map)}, 真实样本数: {len(raw_samples)}")
        
        # 存储原始样本信息用于后续操作
        self.raw_samples = raw_samples
        print(f"{'='*60}\n")
    
    def _load_raw_data(self):
        """
        加载原始 .npy 文件 / Load raw .npy files.

        Returns / 返回:
            list: [中文] 原始样本列表 / [English] list of raw samples.
        """
        samples = []
        file_prefix = self.mode
        
        for class_name in MULTIRM_CLASSES:
            for is_pos in [True, False]:
                sub_dir = os.path.join(self.data_dir, class_name, 'pos' if is_pos else 'neg')
                in_path = os.path.join(sub_dir, f'{file_prefix}_in.npy')
                out_path = os.path.join(sub_dir, f'{file_prefix}_out.npy')
                out_4_path = os.path.join(sub_dir, f'{file_prefix}_out_4class.npy')
                
                if os.path.exists(in_path):
                    try:
                        seqs = np.load(in_path, allow_pickle=True)
                        labels = np.load(out_path, allow_pickle=True)
                        labels_4 = np.load(out_4_path, allow_pickle=True) if self.use_4class and os.path.exists(out_4_path) else None
                        
                        for i in range(len(seqs)):
                            samples.append({
                                'sequence_bytes': seqs[i],
                                'label': labels[i],
                                'label_4class': labels_4[i] if labels_4 is not None and i < len(labels_4) else None,
                                'class_name': class_name,
                                'is_pos': is_pos,
                                'cache_key': f"{class_name}_{self.mode}_{'pos' if is_pos else 'neg'}_{i}"
                            })
                    except Exception as e:
                        print(f"  Warning: 加载 {class_name} {'pos' if is_pos else 'neg'} 失败: {e}")
        
        return samples
    
    def _load_structure_cache(self):
        """一次性加载 H5 缓存"""
        cache_path = os.path.join(self.cache_dir, f"multirm_{self.mode}_structures_cache.h5")
        cache = {}
        
        if os.path.exists(cache_path):
            try:
                with h5py.File(cache_path, 'r') as f:
                    for k in tqdm(f.keys(), desc="加载结构缓存"):
                        cache[k] = torch.from_numpy(f[k][:])
                print(f"成功从 {cache_path} 加载缓存")
            except Exception as e:
                print(f"警告: 加载缓存失败: {e}")
        
        return cache
    
    def _build_indices_map(self, raw_samples):
        """
        构建虚拟索引到真实物理索引的映射数组
        实现 Oversampling with Max-Length Alignment 策略
        """
        if self.mode != 'train':
            # 测试/验证模式：直接使用原始索引
            return np.arange(len(raw_samples))
        
        # 训练模式：按类分组
        class_groups = {}
        for idx, sample in enumerate(raw_samples):
            key = (sample['class_name'], sample['is_pos'])
            if key not in class_groups:
                class_groups[key] = []
            class_groups[key].append(idx)
        
        # 打印统计信息
        print(f"\n各桶数据统计:")
        for (class_name, is_pos), indices in sorted(class_groups.items()):
            pos_neg = "Pos" if is_pos else "Neg"
            print(f"  {class_name:6s} {pos_neg:3s}: {len(indices):5d}")
        
        # 找到最大桶
        max_len = max(len(indices) for indices in class_groups.values()) if class_groups else 0
        print(f"\n训练模式 - Epoch对齐基准 (Max Class Len): {max_len}")
        print(f"总虚拟样本数: {max_len * len(class_groups)} ({len(class_groups)}个桶 × max_len)")
        
        final_indices = []
        # 对每个桶进行循环填充
        for key in class_groups:
            indices = class_groups[key]
            if len(indices) == 0:
                continue
            
            # 扩展到 max_len
            extended = []
            while len(extended) < max_len:
                extended.extend(indices)
            final_indices.extend(extended[:max_len])
        
        np.random.seed(self.seed)
        np.random.shuffle(final_indices)  # Shuffle 一次
        return np.array(final_indices)
    
    def __len__(self):
        return len(self.indices_map)
    
    def __getitem__(self, idx):
        # 极速版: 只有查表，没有计算
        real_idx = self.indices_map[idx]
        
        data = Data(
            x=self.data_x[real_idx],
            edge_index=self.data_edge_index[real_idx],
            y=self.data_y[real_idx],
            class_idx=self.data_class_idx[real_idx]
        )
        
        if self.data_y_4[real_idx] is not None:
            data.y_4 = self.data_y_4[real_idx]
        
        return data
    
    def _get_batch_cache_path(self) -> str:
        """获取批量缓存文件路径"""
        cache_filename = f"multirm_{self.mode}_structures_cache.h5"
        return os.path.join(self.cache_dir, cache_filename)
    
    def precompute_all_structures(
        self,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        预计算所有可能序列的二级结构并缓存
        
        Args:
            batch_size (int): 每次调用LinearFold的序列数量
            num_workers (int): 工作进程数，None表示使用CPU核心数
            show_progress (bool): 是否显示进度条
        
        Returns:
            dict: 统计信息
        """
        print(f"\n{'='*60}")
        print(f"开始预计算所有序列的二级结构...")
        print(f"  模式: {self.mode}")
        print(f"  批量大小: {batch_size}")
        print(f"{'='*60}\n")
        
        stats = {
            'total': 0,
            'computed': 0,
            'failed': 0
        }
        
        all_sequences = []
        all_keys = []
        
        # 收集所有序列
        for sample in self.raw_samples:
            sequence_bytes = sample['sequence_bytes'].copy()
            sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
            all_sequences.append(sequence_str)
            all_keys.append(sample['cache_key'])
        
        stats['total'] = len(all_sequences)
        print(f"  需要计算的序列总数: {stats['total']}")
        
        # 批量计算
        iterator = range(0, len(all_sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="预计算二级结构")
        
        edge_cache = {}
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(all_sequences))
            batch_sequences = all_sequences[start_idx:end_idx]
            batch_keys = all_keys[start_idx:end_idx]
            
            try:
                structures = run_linearfold(batch_sequences)
                
                for seq_str, structure, key in zip(batch_sequences, structures, batch_keys):
                    edge_index = build_edge_index_from_structure(seq_str, structure)
                    edge_cache[key] = edge_index
                    stats['computed'] += 1
            
            except Exception as e:
                print(f"\n警告: 批量计算失败 (索引 {start_idx}-{end_idx}): {e}")
                for key in batch_keys:
                    stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print(f"预计算完成！")
        print(f"  总样本数: {stats['total']}")
        print(f"  成功计算: {stats['computed']}")
        print(f"  失败: {stats['failed']}")
        print(f"  缓存大小: {len(edge_cache)} 条目")
        
        # 保存缓存到磁盘
        if len(edge_cache) > 0:
            self._save_cache_to_disk(edge_cache)
        
        print(f"{'='*60}\n")
        
        return stats
    
    def _save_cache_to_disk(self, edge_cache):
        """将边索引缓存保存到磁盘（HDF5格式）"""
        cache_path = self._get_batch_cache_path()
        
        print(f"\n正在保存缓存到磁盘: {cache_path}")
        
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            with h5py.File(cache_path, 'w') as f:
                for key, edge_index in edge_cache.items():
                    edge_index_numpy = edge_index.cpu().numpy()
                    f.create_dataset(key, data=edge_index_numpy, compression='gzip', compression_opts=4)
            
            print(f"缓存已保存: {len(edge_cache)} 个条目")
        
        except Exception as e:
            print(f"警告: 保存缓存失败: {e}")
    
    def clear_cache(self):
        """清除内存中的边索引缓存"""
        self.data_x.clear()
        self.data_edge_index.clear()
        self.data_y.clear()
        self.data_y_4.clear()
        self.data_class_idx.clear()
        print("已清除内存缓存")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cache_dir': self.cache_dir,
            'mode': self.mode,
            'virtual_length': len(self.indices_map),
            'real_length': len(self.raw_samples) if hasattr(self, 'raw_samples') else 0
        }


# 测试代码
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    print("=" * 60)
    print("=== 测试1: 单线程加载数据集 ===")
    print("=" * 60)

    dataset = MultirmDataset(numsample=50, mode='train')
    print(f"数据集大小: {len(dataset)}")

    print("\n测试获取单个样本:")
    sample = dataset[0]
    print(f"  节点特征形状: {sample.x.shape}")
    print(f"  边索引形状: {sample.edge_index.shape}")
    print(f"  y (12类) 形状: {sample.y.shape}")
    print(f"  类别索引: {sample.class_idx.item()}")
    print(f"  类别名称: {MULTIRM_CLASSES[sample.class_idx.item()]}")

    print("\n" + "=" * 60)
    print("=== 测试2: PyG DataLoader测试 ===")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
    print(f"DataLoader配置: batch_size=32, num_workers=0")

    print("\n获取第一个批次:")
    batch = next(iter(loader))

    print(f"  batch.x 形状: {batch.x.shape}")
    print(f"  batch.edge_index 形状: {batch.edge_index.shape}")
    print(f"  batch.y 形状: {batch.y.shape}")

    print("\n" + "=" * 60)
    print("=== 测试3: 缓存统计 ===")
    print("=" * 60)

    cache_stats = dataset.get_cache_stats()
    print(f"缓存目录: {cache_stats['cache_dir']}")
    print(f"虚拟长度: {cache_stats['virtual_length']}")
    print(f"真实长度: {cache_stats['real_length']}")

    print("\n" + "=" * 60)
    print("=== 所有测试完成！===")
    print("=" * 60)