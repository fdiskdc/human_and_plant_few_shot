"""
human_with_seq.py - 带序列字符串的人类RNA数据集 (推理加速版) / Human RNA Dataset with Sequence String (Inference-Accelerated)

本模块继承 Mer100Dataset，在每个 PyG Data 对象中额外附加 seq_str 字段 (序列字符串)，
避免推理时从 one-hot 反向转换为字符序列的昂贵计算。其他行为与 human.py 完全一致。
This module inherits Mer100Dataset and adds a seq_str field (sequence string) to each PyG Data
object, avoiding the expensive one-hot to character conversion during inference. All other
behavior is identical to human.py.

功能模块 / Modules:
- Mer100DatasetWithSeq: 继承 Mer100Dataset,重写 __getitem__ 添加 seq_str 字段 / Inherits Mer100Dataset and overrides __getitem__ to add seq_str
- human as human_module: 复用 human.py 中的全部常量与函数 / Reuses all constants and functions from human.py

输入 / Inputs:
- human3/seq.npy: NumPy字节数组, 形状 (N, 1001) |S1 - RNA序列 (从 human.py 复用) / RNA sequences (reused from human.py)
- human3/1001loc.npy: NumPy int8数组, 形状 (N, 1001) - 位点级标签 / Site-level labels
- human3/12loc.npy: NumPy int8数组, 形状 (N, 12) - 12类多标签 / 12-class multi-labels
- human3/4loc.npy: NumPy int8数组, 形状 (N, 4) - 4类组标签 / 4-class group labels
- 配置文件 / Config: mode, data_dir, cache_dir, use_human3, use_cache, preload_cache - 与 Mer100Dataset 相同 / Same as Mer100Dataset

输出 / Outputs:
- PyG Data对象 / PyG Data objects: 包含 human.py 所有字段 + 额外 seq_str (str, 长度1001) / All human.py fields + extra seq_str (str, len 1001)
- 字段列表 / Fields: x, edge_index, y, y_4class, y_site, attn_mask_A/C/G/U, attn_mask_N, seq_str / All human.py fields plus seq_str

数据流 / Data Flow:
1. 复用 human.py 加载流程 / Reuse human.py loading: 调用 super().__init__() 完成 npy 加载与缓存初始化 / Call super().__init__() to load npy and init cache
2. 字节转字符串 (一次性) / Byte-to-string (one-time): 在 __getitem__ 中将 sequence_bytes 转换为字符串并保存到 data.seq_str / Convert sequence_bytes to string and save as data.seq_str
3. 字节流one-hot编码 / Byte-to-onehot: 使用 _BYTE_TO_ONEHOT_MAPPING 查表 / Use lookup table
4. LinearFold二级结构 / Secondary structure: 复用 human.py 缓存与计算 / Reuse human.py cache and computation
5. 构建PyG Data + seq_str / Build PyG Data + seq_str: 组装所有字段,附加 seq_str 用于推理 / Assemble all fields and attach seq_str

相关文件 / Related Files:
- 调用 / Calls: Mer100Dataset (继承), human.py 全部依赖 / Inherits from Mer100Dataset, reuses human.py
- 被调用 / Called by: collect_human.py, collect_human_atten.py, collect_modx_atten.py, collect_multirm_atten.py, prepare_umap_data.py - 注意力收集与UMAP数据准备脚本 / Attention collection and UMAP data preparation scripts

使用示例 / Usage Example:
    from dataset.human_with_seq import Mer100DatasetWithSeq
    train_set = Mer100DatasetWithSeq(mode='train', use_human3=True)
    sample = train_set[0]
    print(sample.seq_str[:50])  # 直接访问序列字符串,避免one-hot转换 / Direct access avoids one-hot conversion
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_set, batch_size=32, shuffle=False)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

"""
Enhanced Mer100Dataset with sequence string support for efficient data collection

This module extends the original human.py to include sequence string in Data objects,
avoiding the need for one-hot to sequence conversion during inference.

Based on: dataset/human.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import subprocess
import tempfile
import os
import uuid
import hashlib
import pickle

# Import the original dataset module
try:
    from . import human as human_module
except ImportError:
    import sys
    import os
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import dataset.human as human_module

# Import constants and functions from original module
LABEL_MAPPING = human_module.LABEL_MAPPING
INDEX_TO_NUCLEOTIDE = human_module.INDEX_TO_NUCLEOTIDE
NUCLEOTIDE_GROUPS = human_module.NUCLEOTIDE_GROUPS
MOD_NAMES = human_module.MOD_NAMES
ONE_HOT_MAPPING = human_module.ONE_HOT_MAPPING
DEFAULT_ONE_HOT = human_module.DEFAULT_ONE_HOT
TARGET_LENGTH = human_module.TARGET_LENGTH
LINEARFOLD_PATH = human_module.LINEARFOLD_PATH
BATCH_CACHE_FILE = human_module.BATCH_CACHE_FILE

_worker_process_batch = human_module._worker_process_batch
_BYTE_TO_ONEHOT_MAPPING = human_module._BYTE_TO_ONEHOT_MAPPING
run_linearfold = human_module.run_linearfold
build_edge_index_from_structure = human_module.build_edge_index_from_structure
build_sequential_edge_index = human_module.build_sequential_edge_index

# Import the base class
Mer100Dataset = human_module.Mer100Dataset


class Mer100DatasetWithSeq(Mer100Dataset):
    """
    Mer100Dataset 扩展：Data 对象中包含序列字符串 / Extended Mer100Dataset including sequence string.

    避免推理时昂贵的 one-hot → string 转换。
    Avoids expensive one-hot to string conversion during inference.

    Attributes / 属性:
        继承自 Mer100Dataset / Inherits from Mer100Dataset.
    """

    def __init__(self, mode='train', data_dir='../npy', cache_dir=None,
                 use_human3=True, use_cache=True, preload_cache=True):
        """
        初始化 Mer100DatasetWithSeq / Initialize the enhanced dataset.

        Args / 参数:
            mode (str): [中文] 'train'/'test' / [English] 'train' or 'test'. Defaults to 'train'.
            data_dir (str): [中文] 数据目录 / [English] data directory. Defaults to '../npy'.
            cache_dir (Optional[str]): [中文] 缓存目录 / [English] cache directory.
            use_human3 (bool): [中文] 使用 human3 / [English] use human3. Defaults to True.
            use_cache (bool): [中文] 启用缓存 / [English] enable cache. Defaults to True.
            preload_cache (bool): [中文] 预加载 / [English] preload. Defaults to True.
        """
        # Initialize parent class to load all data and structures
        super().__init__(
            mode=mode,
            data_dir=data_dir,
            cache_dir=cache_dir,
            use_human3=use_human3,
            use_cache=use_cache,
            preload_cache=preload_cache
        )
    
    def __getitem__(self, idx):
        """
        获取数据样本（含序列字符串） / Get single data sample with sequence string included.

        Args / 参数:
            idx (int): [中文] 样本索引 / [English] sample index.

        Returns / 返回:
            Data: [中文] PyG Data 对象，含 `seq_str` / [English] PyG Data with `seq_str` attribute.
        """
        # Get sequence bytes and labels (using copy() for thread safety)
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()
        y_12class = self.y_12class[idx].copy()
        y_4class = self.y_4class[idx].copy()
        
        # Convert bytes to sequence string ONCE during data loading
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        
        # Use optimized one-hot encoding
        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)
        
        # Get or compute edge index
        edge_index = self._get_or_compute_edge_index(sequence_str, idx)
        
        # Node features
        node_features = torch.FloatTensor(one_hot_seq)
        
        # Generate attention masks
        attn_masks = self._extract_attention_masks(full_label)
        attn_mask_N = self._extract_attention_masks_N(sequence_bytes)
        
        # Create PyG Data object with sequence string
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),
            y_site=torch.LongTensor(full_label)
        )
        
        # Add sequence string as attribute (avoiding one-hot conversion later)
        data.seq_str = sequence_str
        
        # Add attention masks
        for nuc, mask in attn_masks.items():
            setattr(data, f'attn_mask_{nuc}', mask)
        setattr(data, 'attn_mask_N', attn_mask_N)
        
        return data


# Test code
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    print("=" * 60)
    print("Testing Enhanced Dataset with Sequence String")
    print("=" * 60)
    
    dataset = Mer100DatasetWithSeq(mode='train', use_human3=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting single sample
    print("\nTesting single sample:")
    sample = dataset[0]
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge index shape: {sample.edge_index.shape}")
    print(f"  Sequence string length: {len(sample.seq_str)}")
    print(f"  Sequence string (first 50 chars): {sample.seq_str[:50]}...")
    print(f"  y (12-class) shape: {sample.y.shape}")
    print(f"  y_4class (4-class) shape: {sample.y_4class.shape}")
    
    # Test with DataLoader
    print("\n" + "=" * 60)
    print("Testing with DataLoader")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    
    print(f"  Batch size: {batch.batch.max().item() + 1}")
    print(f"  Batch.x shape: {batch.x.shape}")
    print(f"  Has seq_str: {hasattr(batch, 'seq_str')}")
    if hasattr(batch, 'seq_str'):
        print(f"  seq_str type: {type(batch.seq_str)}")
        print(f"  seq_str length: {len(batch.seq_str)}")
        print(f"  First sequence (50 chars): {batch.seq_str[0][:50]}...")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)