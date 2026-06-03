"""
human_motif.py - 人类RNA全数据集Motif可视化加载器 / Human RNA Full-Dataset Motif Visualization Loader

本模块是 human.py 的轻量化变体，专用于 Motif (RNA修饰上下文基序) 可视化任务。
关键差异：加载所有可用数据 (合并 train+test 或直接读取 human3 完整文件)，不做 train/test 切分。
用于分析模型在不同位点上下文上的注意力模式与基序偏好。复用 human.py 的标签体系与辅助函数。
This module is a lightweight variant of human.py specifically for Motif (RNA modification context
motif) visualization. Key difference: it loads ALL available data (merging train+test or reading
the full human3 directory) without train/test split. Used to analyze model attention patterns
and motif preferences at modification sites. Reuses label system and helper functions from human.py.

功能模块 / Modules:
- Mer100DatasetMotif: PyG Dataset类,加载 human3 全部数据或合并 train+test,专用于 Motif 可视化 / PyG Dataset class loading all data for motif visualization
- precompute_all_structures: 预计算全部数据的二级结构 (强制 mode='all' 缓存文件名) / Precomputes all structures with mode='all' cache filename
- _load_legacy_full_data: 向后兼容模式,合并 train+test 旧 npz 数据 / Legacy mode merging train+test npz data
- run_linearfold / build_edge_index_from_structure: 复用 human.py 简化版 / Reused from human.py

输入 / Inputs:
- human3/seq.npy: NumPy字节数组, 形状 (N, 1001) |S1 - 全部 human3 RNA序列 / Full human3 RNA sequences
- human3/1001loc.npy: NumPy int8数组, 形状 (N, 1001) - 全部位点级标签 / All site-level labels
- human3/12loc.npy: NumPy int8数组, 形状 (N, 12) - 全部12类多标签 / All 12-class multi-labels
- human3/4loc.npy: NumPy int8数组, 形状 (N, 4) - 全部4类组标签 / All 4-class group labels
- (可选) Legacy npz: {nuc}_expert_train.npy / {nuc}_expert_test.npy, 4 nucleotides x 2 modes / Optional legacy npz
- 配置文件 / Config: LINEARFOLD_PATH, cache_dir, use_human3 - 路径与缓存 / Path and cache

输出 / Outputs:
- PyG Data对象 / PyG Data objects: x=(1001,4) one-hot, edge_index=(2,E) 边索引, y=(1,12) 12类, y_4class=(1,4) 4类, y_site=(1001,) 位点级 / x: one-hot; edge_index: edges; y: 12-class; y_4class: 4-class; y_site: site-level
- 缓存文件名 / Cache filename: human_all_structures_cache.npz - 区别于 human.py 的 human_{train|test}_structures_cache.npz / Differs from human.py's per-mode cache

数据流 / Data Flow:
1. 加载全量数据 / Load full data: 默认从 human3 加载全部 N 个样本, 或从 legacy npz 合并 train+test / Load all N from human3, or merge train+test from legacy npz
2. 字节流one-hot编码 / Byte-to-onehot: 使用 _BYTE_TO_ONEHOT_MAPPING 查表 / Use lookup table
3. LinearFold二级结构 / Secondary structure: 调用 LinearFold 或读取 human_all_*.npz 缓存 / Use LinearFold or read human_all_*.npz cache
4. 构建PyG Data / Build PyG Data: 组装节点特征、边索引、标签为 PyG Data 对象 / Assemble features, edges, labels

相关文件 / Related Files:
- 调用 / Calls: torch.utils.data.Dataset, torch_geometric.data.Data, subprocess (LinearFold), numpy/pickle / Standard utilities
- 被调用 / Called by: SpatialMotif.py, SpatialMotif_nobackground.py - 空间Motif可视化脚本 / Spatial motif visualization scripts

使用示例 / Usage Example:
    from dataset.human_motif import Mer100DatasetMotif
    full_set = Mer100DatasetMotif(use_human3=True, preload_cache=True)
    print(f"Total samples for motif analysis: {len(full_set)}")
    from torch_geometric.loader import DataLoader
    loader = DataLoader(full_set, batch_size=32, shuffle=False)
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

# ============================================================================
# Constants & Mappings (Copied from human.py to ensure standalone capability)
# ============================================================================

LABEL_MAPPING = {
    1: 0,   # Am
    2: 1,   # Atol
    3: 2,   # Cm
    4: 3,   # Gm
    5: 4,   # Tm
    6: 5,   # Y
    7: 6,   # ac4C
    8: 7,   # m1A
    9: 8,   # m5C
    10: 9,  # m6A
    11: 10, # m6Am
    12: 11  # m7G
}

INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A', 2: 'C', 3: 'G', 4: 'U', 5: 'U',
    6: 'C', 7: 'A', 8: 'C', 9: 'A', 10: 'A', 11: 'G'
}

ONE_HOT_MAPPING = {
    'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.],
    'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
    'U': [0., 0., 0., 1.], 'N': [0., 0., 0., 0.]
}

LINEARFOLD_PATH = '/home/dc/vscode/LinearFold/linearfold'
BATCH_CACHE_FILE = 'structures_cache.npz'

# ============================================================================
# Helper Functions
# ============================================================================

def _create_byte_to_onehot_mapping():
    mapping = np.zeros((256, 4), dtype=np.float32)
    mapping[65] = [1., 0., 0., 0.]  # A
    mapping[97] = [1., 0., 0., 0.]  # a
    mapping[67] = [0., 1., 0., 0.]  # C
    mapping[99] = [0., 1., 0., 0.]  # c
    mapping[71] = [0., 0., 1., 0.]  # G
    mapping[103] = [0., 0., 1., 0.] # g
    mapping[84] = [0., 0., 0., 1.]  # T
    mapping[116] = [0., 0., 0., 1.] # t
    mapping[85] = [0., 0., 0., 1.]  # U
    mapping[117] = [0., 0., 0., 1.] # u
    return mapping

_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()

def run_linearfold(sequences, timeout_seconds=1800):
    if not sequences: return []
    fasta_input = '\n'.join([f'>seq_{i}\n{seq}' for i, seq in enumerate(sequences)])
    
    try:
        if not os.path.exists(LINEARFOLD_PATH):
            raise FileNotFoundError(f"LinearFold not found: {LINEARFOLD_PATH}")
        
        process = subprocess.Popen(
            [LINEARFOLD_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, encoding='utf-8'
        )
        stdout_data, stderr_data = process.communicate(input=fasta_input, timeout=timeout_seconds)
        
        if process.returncode != 0:
            raise RuntimeError(f"LinearFold failed: {stderr_data[:500]}")
            
        lines = [line.strip() for line in stdout_data.strip().split('\n') if line.strip()]
        structures = []
        for i in range(2, len(lines), 3):
            structures.append(lines[i].split()[0])
            
        return structures
    except Exception as e:
        raise RuntimeError(f"LinearFold execution error: {e}")

def build_edge_index_from_structure(sequence, structure):
    if not structure: return build_sequential_edge_index(sequence)
    stack = []
    pairs = {}
    for i, char in enumerate(structure):
        if char == '(': stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs[j] = i; pairs[i] = j
            
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    for i, j in pairs.items():
        if i < j: edge_list.extend([(i, j), (j, i)])
        
    if edge_list: return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else: return torch.empty((2, 0), dtype=torch.long)

def build_sequential_edge_index(sequence):
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

def _worker_process_batch(args):
    batch_indices, sequences_bytes_array, linearfold_path = args
    sequences_str = []
    for idx in batch_indices:
        sequences_str.append(sequences_bytes_array[idx].tobytes().decode('ascii', errors='ignore'))
    
    results = []
    try:
        structures = run_linearfold(sequences_str)
        for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            results.append((idx, edge_index.cpu().numpy(), None))
    except Exception as e:
        for idx in batch_indices:
            results.append((idx, None, str(e)))
    return results

# ============================================================================
# Mer100DatasetMotif Class (Full Dataset Version)
# ============================================================================

class Mer100DatasetMotif(Dataset):
    """
    Dataset class specifically for Motif Visualization.
    Crucially: It loads ALL available data (combining train+test or using full arrays),
    ignoring any split logic.
    """

    def __init__(self, data_dir='../npy', cache_dir=None, use_human3=True, use_cache=True, preload_cache=True):
        # Forced mode to 'all' to indicate full dataset usage in cache filenames
        self.mode = 'all' 
        self.data_dir = data_dir
        self.use_cache = use_cache
        self._batch_cache = None
        self._edge_indices = None

        if cache_dir is None:
            self.CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        else:
            self.CACHE_DIR = cache_dir

        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        if use_human3:
            # Human3 usually contains the full dataset in single .npy files
            if os.path.exists('human3'): human3_dir = 'human3'
            elif os.path.exists('../human3'): human3_dir = '../human3'
            else: human3_dir = '/home/dc/vscode/vscode20251230/human_and_plant/human3'
            
            print(f"[MotifDataset] Loading FULL dataset from human3: {human3_dir}")
            
            self.sequences = np.load(f'{human3_dir}/seq.npy', mmap_mode='r')
            self.full_labels = np.load(f'{human3_dir}/1001loc.npy', mmap_mode='r')
            self.y_12class = np.load(f'{human3_dir}/12loc.npy', mmap_mode='r')
            self.y_4class = np.load(f'{human3_dir}/4loc.npy', mmap_mode='r')
            
            if self.use_cache and preload_cache:
                self._load_batch_cache()
        else:
            # Legacy mode: Must load BOTH train and test and concatenate
            self._load_legacy_full_data(data_dir)

    def _load_legacy_full_data(self, data_dir):
        print(f"[MotifDataset] Loading and merging legacy TRAIN + TEST data from: {data_dir}")
        nucleotides = ['A', 'C', 'G', 'U']
        modes = ['train', 'test']
        
        all_seq, all_full, all_y12, all_y4 = [], [], [], []
        
        for mode in modes:
            suffix = '_train.npy' if mode == 'train' else '_test.npy'
            for nuc in nucleotides:
                file_path = f'{data_dir}/{nuc}_expert{suffix}'
                try:
                    data = np.load(file_path, allow_pickle=True)
                    sequences = data['seq']
                    full_labels = data['full_label']
                    
                    y_12class = np.zeros((len(sequences), 12), dtype=np.int8)
                    y_4class = np.zeros((len(sequences), 4), dtype=np.int8)
                    
                    for i, label in enumerate(full_labels):
                        for label_id in np.unique(label):
                            if label_id == 0: continue
                            if label_id in LABEL_MAPPING:
                                idx = LABEL_MAPPING[label_id]
                                y_12class[i, idx] = 1
                                nuc_group = INDEX_TO_NUCLEOTIDE[idx]
                                group_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}[nuc_group]
                                y_4class[i, group_idx] = 1
                    
                    all_seq.append(sequences)
                    all_full.append(full_labels)
                    all_y12.append(y_12class)
                    all_y4.append(y_4class)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

        self.sequences = np.concatenate(all_seq, axis=0)
        self.full_labels = np.concatenate(all_full, axis=0)
        self.y_12class = np.concatenate(all_y12, axis=0)
        self.y_4class = np.concatenate(all_y4, axis=0)
        
        print(f"[MotifDataset] Merged Total Samples: {len(self.sequences)}")
        if self.use_cache: self._load_batch_cache()

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()
        y_12class = self.y_12class[idx].copy()
        y_4class = self.y_4class[idx].copy()

        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        
        # Use existing cache or compute
        edge_index = self._get_or_compute_edge_index(sequence_str, idx)

        node_features = torch.FloatTensor(one_hot_seq)
        
        # Simple Data object (Motif viz doesn't strictly need masks, but keeping for compatibility)
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),
            y_site=torch.LongTensor(full_label)
        )
        return data

    def _one_hot_encode_optimized(self, sequence_bytes):
        byte_array = sequence_bytes.view(np.uint8)
        return _BYTE_TO_ONEHOT_MAPPING[byte_array].astype(np.float32)

    def _get_batch_cache_path(self):
        # Unique cache file for the FULL dataset
        return os.path.join(self.CACHE_DIR, f"human_{self.mode}_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        cache_path = self._get_batch_cache_path()
        if os.path.exists(cache_path):
            try:
                self._batch_cache = np.load(cache_path, allow_pickle=True)
                self._edge_indices = self._batch_cache['edge_indices']
                print(f"[Cache] Loaded {len(self._edge_indices)} structures from {cache_path}")
            except:
                self._batch_cache = None
        else:
            print(f"[Cache] No batch cache found at {cache_path}. Precomputation recommended.")

    def precompute_all_structures(self, batch_size=100, num_workers=None, show_progress=True):
        from tqdm import tqdm
        from multiprocessing import Pool, cpu_count
        
        cache_path = self._get_batch_cache_path()
        if os.path.exists(cache_path):
            print(f"[Precompute] Cache already exists at {cache_path}. Skipping.")
            self._load_batch_cache()
            return

        num_samples = len(self.sequences)
        if num_workers is None: num_workers = cpu_count()
        
        print(f"[Precompute] Starting for {num_samples} samples using {num_workers} workers...")
        
        edge_indices = np.empty(num_samples, dtype=object)
        batch_tasks = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_tasks.append((list(range(start_idx, end_idx)), self.sequences, LINEARFOLD_PATH))
            
        with Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)
            if show_progress: results_iter = tqdm(results_iter, total=len(batch_tasks))
            
            for batch_results in results_iter:
                for idx, edge_idx_np, err in batch_results:
                    if err is None: edge_indices[idx] = edge_idx_np
                    
        np.savez_compressed(cache_path, edge_indices=edge_indices, mode=self.mode, num_samples=num_samples)
        print(f"[Precompute] Saved cache to {cache_path}")
        self._load_batch_cache()

    def _get_or_compute_edge_index(self, sequence_str, idx):
        if self._edge_indices is not None and idx < len(self._edge_indices):
            cached = self._edge_indices[idx]
            if cached is not None: return torch.from_numpy(cached)
            
        # Fallback
        structures = run_linearfold([sequence_str])
        return build_edge_index_from_structure(sequence_str, structures[0])