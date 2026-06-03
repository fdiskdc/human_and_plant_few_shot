"""
gen3_zero.py - 3gen正样本 + Zero负样本混合数据集 / 3gen Positives + Zero Negatives Hybrid Dataset

本模块将 3gen 数据 (12loc 不全为零的样本) 作为正样本,与 zero 数据 (背景/无修饰样本)
作为负样本,按 1:1 数量合并用于零样本/小样本学习任务。处理正负样本维度不一致的边界情况
(负样本可能为 1D 字节字符串,正样本为 2D S1 数组)。结构与 human.py 高度相似。
This module merges 3gen data (samples with non-zero 12loc) as positives and zero data (background
samples) as negatives at 1:1 ratio for zero-shot/few-shot learning tasks. Handles dimension
mismatches between pos/neg samples. Structurally similar to human.py.

功能模块 / Modules:
- Gen3ZeroDataset: PyG Dataset类,从 data_dir/3gen_alignment 与 data_dir/zero 加载并合并正负样本 / PyG Dataset class loading and merging pos/neg from 3gen_alignment and zero subdirs
- 正样本筛选 / Positive filtering: 从 3gen 提取 12loc 任一位点非零的样本 (np.any(g3_12_loaded != 0, axis=1)) / Extract samples with any non-zero 12loc entry
- 负样本配对 / Negative pairing: 提取等量 (num_pos) 的 zero 样本作为负样本 / Extract equal number of zero samples
- 维度修复 / Dimension fix: 检测并修复正负样本维度不一致 (1D vs 2D) / Detect and fix dimension mismatch
- precompute_all_structures: 多进程预计算二级结构 / Multiprocess precomputation

输入 / Inputs:
- data_dir/3gen_alignment/seq.npy: NumPy字节数组, 形状 (N_pos, 1001) - 3gen正样本 / 3gen positive samples
- data_dir/3gen_alignment/12loc.npy: NumPy int8数组, 形状 (N_pos, 12) - 12类多标签 / 12-class multi-labels
- data_dir/3gen_alignment/4loc.npy: NumPy int8数组, 形状 (N_pos, 4) - 4类组标签 / 4-class group labels
- data_dir/3gen_alignment/1001loc.npy: NumPy int8数组, 形状 (N_pos, 1001) - 位点级标签 / Site-level labels
- data_dir/zero/zero_seq.npy: NumPy字节数组 - 零样本/背景序列 / Zero/background sequences
- data_dir/zero/zero_label12.npy: NumPy int8数组 - 12类多标签 (通常为全0) / 12-class multi-labels (usually all-zero)
- data_dir/zero/zero_label4.npy: NumPy int8数组 - 4类组标签 / 4-class group labels
- data_dir/zero/zero_label1001.npy: NumPy int8数组 - 位点级标签 / Site-level labels
- 配置文件 / Config: LINEARFOLD_PATH, cache_dir='cache/gen3_zero' - 路径与缓存 / Path and cache

输出 / Outputs:
- PyG Data对象 / PyG Data objects: x=(1001,4) one-hot, edge_index=(2,E) 边索引, y=(1,12) 12类, y_4class=(1,4) 4类, y_site=(1001,) 位点级 / x: one-hot; edge_index: edges; y: 12-class; y_4class: 4-class; y_site: site-level
- 注意力掩码 / Attention masks: attn_mask_A/C/G/U (1001,) - 4个核苷酸组归一化目标 / Per-nucleotide normalized targets
- N字符掩码 / N-mask: attn_mask_N (1001,) - 'N'位置 / 'N' character positions
- 缓存前缀 / Cache prefix: gen3_zero_{mode}_structures_cache.npz / gen3_zero mode cache

数据流 / Data Flow:
1. 加载3gen数据 / Load 3gen: 读取3gen_alignment子目录的seq/12loc/4loc/1001loc / Read 3gen_alignment subdir data
2. 筛选正样本 / Filter positives: 使用 np.any 找出 12loc 不全为0的样本索引 / Use np.any to find positive sample indices
3. 加载Zero数据 / Load Zero: 读取zero子目录的zero_seq/zero_label12/zero_label4/zero_label1001 / Read zero subdir data
4. 配对负样本 / Pair negatives: 取前 num_pos 个 zero 样本作为负样本 / Take first num_pos zero samples as negatives
5. 维度修复 / Dimension fix: 若 neg_seq 是 2D 而 pos_seq 是 1D,使用 astype('S1').view('S{len}').ravel() 转换 / Fix dimension mismatch via astype/view/ravel
6. 合并数据 / Merge: np.concatenate 拼接正负样本得到 self.sequences 等 / Concatenate pos/neg samples
7. 字节流one-hot + LinearFold + 构建PyG Data / Byte-to-onehot + LinearFold + Build PyG Data

相关文件 / Related Files:
- 调用 / Calls: torch.utils.data.Dataset, torch_geometric.data.Data, subprocess (LinearFold), numpy/pickle / Standard utilities
- 被调用 / Called by: test_gen3.py, utils/fewshot_analysis_gen3.py, utils/fewshot_analysis_spatial_motif.py, utils/fewshot_export_helpers.py, utils/test_gen3_analyse.py, zero_shot_fewshot_analysis.py, zero_shot_fewshot_extract_only.py

使用示例 / Usage Example:
    from dataset.gen3_zero import Gen3ZeroDataset
    hybrid_set = Gen3ZeroDataset(data_dir='../npy', mode='train')
    print(f"Hybrid samples: {len(hybrid_set)} (pos + neg)")
    from torch_geometric.loader import DataLoader
    loader = DataLoader(hybrid_set, batch_size=4, shuffle=True)
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

# 12类标签映射表 (与 human_make_npy.py 中的 MOD_TO_INDEX 一致)
LABEL_MAPPING = {
    # mod_index (1-12) -> 模型索引 (0-11)
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

# 反向映射：模型索引 -> 核苷酸组
INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',                          # Am, Atol
    2: 'C',                                  # Cm
    3: 'G',                                  # Gm
    4: 'U', 5: 'U',                          # Tm, Y
    6: 'C',                                  # ac4C
    7: 'A',                                  # m1A
    8: 'C',                                  # m5C
    9: 'A', 10: 'A',                         # m6A, m6Am
    11: 'G'                                  # m7G
}

# 每个核苷酸组包含的ID (1-12)
NUCLEOTIDE_GROUPS = {
    'A': [1, 2, 8, 10, 11],  # Am, Atol, m1A, m6A, m6Am
    'C': [3, 7, 9],          # Cm, ac4C, m5C
    'G': [4, 12],            # Gm, m7G
    'U': [5, 6]              # Tm, Y
}

# 12类修饰名称映射
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',      # 索引 0-2
    3: 'Gm',     4: 'Tm',     5: 'Y',       # 索引 3-5
    6: 'ac4C',   7: 'm1A',    8: 'm5C',     # 索引 6-8
    9: 'm6A',    10: 'm6Am',  11: 'm7G'     # 索引 9-11
}

# One-hot编码映射
ONE_HOT_MAPPING = {
    'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.],
    'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
    'U': [0., 0., 0., 1.], 'N': [0., 0., 0., 0.]
}

# 默认One-Hot编码（用于未知核苷酸）
DEFAULT_ONE_HOT = [0., 0., 0., 0.]

# 目标序列长度
TARGET_LENGTH = 1001

# LinearFold路径（可能需要根据实际情况调整）
LINEARFOLD_PATH = '/home/dc/vscode/LinearFold/linearfold'

# 批量缓存文件名
BATCH_CACHE_FILE = 'structures_cache.npz'

# 多进程预计算的工作进程函数
def _worker_process_batch(args):
    """
    工作进程函数：处理一批序列的二级结构计算

    Args:
        args: tuple (batch_indices, sequences_bytes_array, linearfold_path)

    Returns:
        list: [(idx, edge_index_numpy), ...] 或 [(idx, None, error_msg), ...]
    """
    batch_indices, sequences_bytes_array, linearfold_path = args

    # 准备批量序列字符串
    sequences_str = []
    for idx in batch_indices:
        sequence_bytes = sequences_bytes_array[idx].copy()
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        sequences_str.append(sequence_str)

    results = []
    try:
        # 使用LinearFold批量计算二级结构
        structures = run_linearfold(sequences_str)

        # 构建边索引
        for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            # 将torch.Tensor转换为numpy数组存储
            edge_index_numpy = edge_index.cpu().numpy()
            results.append((idx, edge_index_numpy, None))

    except Exception as e:
        # 失败时返回错误信息
        for idx in batch_indices:
            results.append((idx, None, str(e)))

    return results

# 创建字节到one-hot的映射表（用于快速转换）
def _create_byte_to_onehot_mapping():
    """
    创建字节值到one-hot编码的映射表，避免字符串处理开销
    
    Returns:
        np.array: 形状为(256, 4)的映射表
    """
    mapping = np.zeros((256, 4), dtype=np.float32)
    
    # A (ASCII 65, 97)
    mapping[65] = [1., 0., 0., 0.]  # 'A'
    mapping[97] = [1., 0., 0., 0.]  # 'a'
    
    # C (ASCII 67, 99)
    mapping[67] = [0., 1., 0., 0.]  # 'C'
    mapping[99] = [0., 1., 0., 0.]  # 'c'
    
    # G (ASCII 71, 103)
    mapping[71] = [0., 0., 1., 0.]  # 'G'
    mapping[103] = [0., 0., 1., 0.]  # 'g'
    
    # T (ASCII 84, 116) - RNA中优先使用U
    mapping[84] = [0., 0., 0., 1.]  # 'T'
    mapping[116] = [0., 0., 0., 1.]  # 't'
    
    # U (ASCII 85, 117) - RNA中使用U
    mapping[85] = [0., 0., 0., 1.]  # 'U'
    mapping[117] = [0., 0., 0., 1.]  # 'u'
    
    # N (ASCII 78, 110) - 未知核苷酸
    mapping[78] = [0., 0., 0., 0.]  # 'N'
    mapping[110] = [0., 0., 0., 0.]  # 'n'
    
    return mapping

# 全局映射表
_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()

def one_hot_to_sequence(one_hot_array):
    """
    将One-Hot编码的RNA序列转换回字符序列
    """
    reverse_mapping = {
        tuple([1., 0., 0., 0.]): 'A',
        tuple([0., 1., 0., 0.]): 'C',
        tuple([0., 0., 1., 0.]): 'G',
        tuple([0., 0., 0., 1.]): 'U',
        tuple([0., 0., 0., 0.]): 'N'
    }
    
    sequence = []
    for one_hot in one_hot_array:
        key = tuple(one_hot)
        nucleotide = reverse_mapping.get(key, 'N')
        sequence.append(nucleotide)
    
    return ''.join(sequence)

def run_linearfold(sequences, timeout_seconds=1800):
    """
    使用LinearFold预测RNA序列的二级结构（线程安全版本）
    """
    if not sequences:
        return []
    
    # 构建FASTA格式的输入字符串
    fasta_input = '\n'.join([f'>seq_{i}\n{seq}' for i, seq in enumerate(sequences)])
    
    structures = []
    
    try:
        # 检查LinearFold可执行文件是否存在
        if not os.path.exists(LINEARFOLD_PATH):
            raise FileNotFoundError(f"LinearFold可执行文件不存在: {LINEARFOLD_PATH}")
        
        # 调用LinearFold，通过stdin输入
        process = subprocess.Popen(
            [LINEARFOLD_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # 获取输出
        stdout_data, stderr_data = process.communicate(input=fasta_input, timeout=timeout_seconds)
        
        # 检查返回码
        if process.returncode != 0:
            raise RuntimeError(
                f"LinearFold执行失败，返回码 {process.returncode}。"
                f"错误信息: {stderr_data[:500]}"
            )
        
        # 解析输出
        lines = stdout_data.strip().split('\n')
        structures = []
        
        # 过滤掉空行
        lines = [line.strip() for line in lines if line.strip()]
        
        # 每个序列对应3行：header, sequence, structure+energy
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
        
    except FileNotFoundError as e:
        raise
    except subprocess.TimeoutExpired as e:
        raise subprocess.TimeoutExpired(e.cmd, e.timeout, output=e.output, stderr=e.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold执行失败: {e}")
    except Exception as e:
        raise RuntimeError(f"LinearFold执行过程中发生未知错误: {e}")
    
    return structures

def build_edge_index_from_structure(sequence, structure):
    """
    根据RNA二级结构构建边索引
    """
    if not structure:
        return build_sequential_edge_index(sequence)
    
    stack = []
    pairs = {}
    
    # 解析括号结构，找到配对
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    
    edge_list = []
    
    # 添加顺序边 (i, i+1)
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    
    # 添加配对边
    for i, j in pairs.items():
        if i < j:  # 避免重复添加
            edge_list.extend([(i, j), (j, i)])
    
    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)

def build_sequential_edge_index(sequence):
    """
    仅构建顺序边 (i, i+1)
    """
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    
    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)

class Gen3ZeroDataset(Dataset):
    """
    混合 3gen (正样本) 和 zero (负样本) 的数据集。
    
    逻辑：
    1. 从 3gen 文件夹提取 12loc.npy 不全为零的样本作为正样本。
    2. 从 zero 文件夹提取与正样本等量的样本作为负样本。
    3. zero文件夹映射关系：
       - zero_label12 -> 12loc
       - zero_label4 -> 4loc
       - zero_label1001 -> 1001loc
       - zero_seq -> seq
    """

    def __init__(self, mode='train', data_dir='../npy', cache_dir=None, use_cache=True, preload_cache=True):
        """
        初始化混合数据集

        Args:
            mode (str): 'train' 或 'test'
            data_dir (str): 数据根目录 (包含 3gen 和 zero 子文件夹)
            cache_dir (str): 缓存目录路径
            use_cache (bool): 是否启用二级结构缓存
            preload_cache (bool): 是否预加载缓存
        """
        self.mode = mode
        self.data_dir = data_dir
        self.use_cache = use_cache
        self._batch_cache = None
        self._edge_indices = None

        # 设置缓存目录
        if cache_dir is None:
            self.CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'gen3_zero')
        else:
            self.CACHE_DIR = cache_dir

        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # 路径配置
        gen3_dir = os.path.join(data_dir, '3gen_alignment')
        zero_dir = os.path.join(data_dir, 'zero')
        
        print(f"初始化 Gen3Zero 数据集 (mode={mode})...")
        print(f"  3gen 目录: {gen3_dir}")
        print(f"  zero 目录: {zero_dir}")

        # --- 1. 加载 3gen 数据 ---
        print("  正在加载 3gen 数据...")
        g3_seq = np.load(os.path.join(gen3_dir, 'seq.npy'), mmap_mode='r')
        g3_12 = np.load(os.path.join(gen3_dir, '12loc.npy'), mmap_mode='r')
        g3_4 = np.load(os.path.join(gen3_dir, '4loc.npy'), mmap_mode='r')
        g3_1001 = np.load(os.path.join(gen3_dir, '1001loc.npy'), mmap_mode='r')

        # --- 2. 筛选正样本 (12loc 不全为 0) ---
        print("  正在筛选正样本...")
        # 注意：这里需要将数据读入内存进行比较，如果内存不足可能需要分块处理
        # 假设数据量可以放入内存
        g3_12_loaded = np.array(g3_12) 
        # 只要行中有一个非零值，即视为正样本
        pos_mask = np.any(g3_12_loaded != 0, axis=1)
        
        pos_seq = np.array(g3_seq[pos_mask])
        pos_12 = g3_12_loaded[pos_mask]
        pos_4 = np.array(g3_4[pos_mask])
        pos_1001 = np.array(g3_1001[pos_mask])

        num_pos = len(pos_seq)
        print(f"  -> 找到 {num_pos} 个正样本")
        print(f"  -> 正样本序列维度: {pos_seq.ndim}D, 形状: {pos_seq.shape}")

        # --- 3. 加载 Zero 数据 ---
        print("  正在加载 Zero 数据...")
        z_seq = np.load(os.path.join(zero_dir, 'zero_seq.npy'), mmap_mode='r')
        z_12 = np.load(os.path.join(zero_dir, 'zero_label12.npy'), mmap_mode='r')
        z_4 = np.load(os.path.join(zero_dir, 'zero_label4.npy'), mmap_mode='r')
        z_1001 = np.load(os.path.join(zero_dir, 'zero_label1001.npy'), mmap_mode='r')

        # --- 4. 提取等量负样本 ---
        num_avail_neg = len(z_seq)
        if num_avail_neg < num_pos:
            print(f"  警告: 负样本数量 ({num_avail_neg}) 少于正样本 ({num_pos})，使用全部负样本。")
            num_neg = num_avail_neg
        else:
            num_neg = num_pos
            
        print(f"  -> 提取 {num_neg} 个负样本")
        
        # 简单取前 N 个样本
        neg_seq = np.array(z_seq[:num_neg])
        neg_12 = np.array(z_12[:num_neg])
        neg_4 = np.array(z_4[:num_neg])
        neg_1001 = np.array(z_1001[:num_neg])

        print(f"  -> 负样本序列原始维度: {neg_seq.ndim}D, 形状: {neg_seq.shape}")

        # === 修复: 统一序列维度 ===
        # 检查正负样本维度是否一致，如果不一致则尝试转换负样本
        if pos_seq.ndim == 1 and neg_seq.ndim == 2:
            print(f"  [修复] 检测到维度不匹配: 正样本 1D, 负样本 2D。正在将负样本转换为 1D 字节字符串...")
            
            # 确保内存连续
            neg_seq = np.ascontiguousarray(neg_seq)
            
            # 获取序列长度 (dim 1)
            seq_len = neg_seq.shape[1]
            
            # 尝试转换为 dtype='|S{seq_len}' (假设长度为seq_len)
            try:
                # 假设是 'S1' 或 'U1'
                target_dtype = f'S{seq_len}'
                
                # 如果原类型不是字节串(S)，先转为 S1
                if neg_seq.dtype.kind != 'S':
                    neg_seq = neg_seq.astype('S1')
                
                # View as 1D array of strings
                neg_seq = neg_seq.view(target_dtype).ravel()
                print(f"  [修复成功] 转换后负样本形状: {neg_seq.shape}")
                
            except Exception as e:
                print(f"  [修复失败] 无法转换负样本维度: {e}")
                print("  尝试备用方案: 逐行 join (速度较慢)...")
                # 备用方案: 慢速但稳健
                # neg_seq = np.array([b''.join(row) for row in neg_seq])

        # --- 5. 合并数据集 ---
        print("  正在合并数据集...")
        try:
            self.sequences = np.concatenate([pos_seq, neg_seq], axis=0)
            self.y_12class = np.concatenate([pos_12, neg_12], axis=0)
            self.y_4class = np.concatenate([pos_4, neg_4], axis=0)
            self.full_labels = np.concatenate([pos_1001, neg_1001], axis=0)
        except ValueError as e:
            print(f"  [严重错误] 合并数据集失败: {e}")
            print(f"  Debug: pos_seq shape={pos_seq.shape}, neg_seq shape={neg_seq.shape}")
            raise e
        
        print(f"数据集构建完成:")
        print(f"  总样本数: {len(self.sequences)} (正: {num_pos}, 负: {num_neg})")
        print(f"  序列形状: {self.sequences.shape}")

        # 如果启用缓存且preload_cache=True，尝试加载批量缓存
        if self.use_cache and preload_cache:
            self._load_batch_cache()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
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

        # 生成注意力掩码（用于监督）
        attn_masks = self._extract_attention_masks(full_label)

        # 生成N字符掩码（用于标记未知核苷酸位置）
        attn_mask_N = self._extract_attention_masks_N(sequence_bytes)

        # 创建PyG Data对象，整合所有层级标签
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),  # 形状为 [1, 12]
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),  # 形状为 [1, 4]
            y_site=torch.LongTensor(full_label)  # 形状为 [1001]
        )

        # 将注意力掩码添加为data的属性
        for nuc, mask in attn_masks.items():
            setattr(data, f'attn_mask_{nuc}', mask)

        # 添加N字符掩码
        setattr(data, 'attn_mask_N', attn_mask_N)

        return data
    
    def _extract_attention_masks(self, full_label):
        attn_masks = {}
        for nuc in ['A', 'C', 'G', 'U']:
            attn_masks[nuc] = np.zeros(1001, dtype=np.float32)
        
        for i, label_id in enumerate(full_label):
            if label_id == 0: continue
            if label_id in LABEL_MAPPING:
                model_idx = LABEL_MAPPING[label_id]
                nucleotide = INDEX_TO_NUCLEOTIDE[model_idx]
                attn_masks[nucleotide][i] = 1.0
        
        for nuc in ['A', 'C', 'G', 'U']:
            mask = attn_masks[nuc]
            if np.sum(mask) > 0:
                attn_masks[nuc] = mask / np.sum(mask)
            else:
                attn_masks[nuc] = np.ones(1001, dtype=np.float32) / 1001
            attn_masks[nuc] = torch.FloatTensor(attn_masks[nuc])
        
        return attn_masks
    
    def _extract_attention_masks_N(self, sequence_bytes):
        if sequence_bytes.ndim == 0:
            byte_array = np.frombuffer(sequence_bytes, dtype=np.uint8)
        else:
            byte_array = sequence_bytes.view(np.uint8)

        if len(byte_array) > 1001:
            byte_array = byte_array[:1001]
        elif len(byte_array) < 1001:
            byte_array = np.pad(byte_array, (0, 1001 - len(byte_array)), mode='constant')

        mask = np.zeros(1001, dtype=np.float32)
        mask[(byte_array == 78) | (byte_array == 110)] = 1.0
        return torch.FloatTensor(mask)

    def _get_batch_cache_path(self):
        # 修改缓存文件名前缀为 gen3_zero_
        return os.path.join(self.CACHE_DIR, f"gen3_zero_{self.mode}_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        cache_path = self._get_batch_cache_path()
        if os.path.exists(cache_path):
            print(f"正在从批量缓存加载边索引: {cache_path}")
            try:
                self._batch_cache = np.load(cache_path, allow_pickle=True)
                self._edge_indices = self._batch_cache['edge_indices']
                print(f"  已加载 {len(self._edge_indices)} 个边索引")
            except Exception as e:
                print(f"  警告: 加载批量缓存失败: {e}")
                self._batch_cache = None
                self._edge_indices = None
        else:
            print(f"批量缓存文件不存在: {cache_path}")

    def precompute_all_structures(self, batch_size=100, num_workers=None, show_progress=True):
        from tqdm import tqdm
        from multiprocessing import Pool, cpu_count

        num_samples = len(self.sequences)
        cache_path = self._get_batch_cache_path()

        if num_workers is None:
            num_workers = cpu_count()

        print(f"开始预计算所有序列的二级结构 (gen3_zero)...")
        print(f"  样本总数: {num_samples}")
        
        edge_indices = np.empty(num_samples, dtype=object)
        stats = {'total': num_samples, 'computed': 0, 'failed': 0}

        batch_tasks = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))
            batch_tasks.append((batch_indices, self.sequences, LINEARFOLD_PATH))

        total_batches = len(batch_tasks)
        use_multiprocessing = num_workers > 1

        if use_multiprocessing:
            with Pool(processes=num_workers) as pool:
                results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)
                if show_progress:
                    results_iter = tqdm(results_iter, total=total_batches, desc="预计算二级结构")
                
                for batch_results in results_iter:
                    for idx, edge_index_numpy, error in batch_results:
                        if error is None:
                            edge_indices[idx] = edge_index_numpy
                            stats['computed'] += 1
                        else:
                            stats['failed'] += 1
        else:
            # 单进程回退
            iterator = range(0, num_samples, batch_size)
            if show_progress: iterator = tqdm(iterator, desc="预计算")
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))
                # 构造序列列表
                seqs = []
                for idx in batch_indices:
                    seqs.append(self.sequences[idx].tobytes().decode('ascii', errors='ignore'))
                try:
                    structures = run_linearfold(seqs)
                    for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
                        edge_index = build_edge_index_from_structure(seqs[i], structure)
                        edge_indices[idx] = edge_index.cpu().numpy()
                        stats['computed'] += 1
                except Exception:
                    stats['failed'] += len(batch_indices)

        print(f"保存批量缓存到: {cache_path}")
        np.savez_compressed(cache_path, edge_indices=edge_indices, mode=self.mode, num_samples=num_samples)
        
        self._batch_cache = np.load(cache_path, allow_pickle=True)
        self._edge_indices = self._batch_cache['edge_indices']

    def _get_cache_key(self, sequence_str, idx):
        # 缓存键增加 gen3_zero 前缀
        sequence_hash = hashlib.md5(sequence_str.encode('utf-8')).hexdigest()
        cache_key = f"gen3_zero_{self.mode}_{idx}_{sequence_hash}"
        return cache_key

    def _get_cache_path(self, cache_key):
        return os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key):
        if not self.use_cache: return None
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)['edge_index']
            except:
                return None
        return None

    def _save_to_cache(self, cache_key, edge_index):
        if not self.use_cache: return
        try:
            with open(self._get_cache_path(cache_key), 'wb') as f:
                pickle.dump({'edge_index': edge_index, 'mode': self.mode}, f)
        except: pass

    def _get_or_compute_edge_index(self, sequence_str, idx):
        if self._edge_indices is not None and idx < len(self._edge_indices):
            cached = self._edge_indices[idx]
            if cached is not None:
                return torch.from_numpy(cached)
        
        cache_key = self._get_cache_key(sequence_str, idx)
        cached = self._load_from_cache(cache_key)
        if cached is not None: return cached

        try:
            structures = run_linearfold([sequence_str])
            edge_index = build_edge_index_from_structure(sequence_str, structures[0])
            self._save_to_cache(cache_key, edge_index)
            return edge_index
        except: raise

    def get_cache_stats(self):
        """
        获取缓存统计信息（包括批量缓存和单文件缓存）
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

        # 统计单文件缓存 (Prefix adjusted for this class)
        prefix = f"gen3_zero_{self.mode}_"
        mode_files = []
        total_size = 0

        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    file_size = os.path.getsize(cache_path)
                    mode_files.append(filename)
                    total_size += file_size
                except OSError:
                    pass

        stats['single_file_cache']['total_files'] = len(mode_files)
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

    def clear_cache(self):
        if not os.path.exists(self.CACHE_DIR): return
        prefix = f"gen3_zero_{self.mode}_"
        for f in os.listdir(self.CACHE_DIR):
            if f.startswith(prefix) and f.endswith('.pkl'):
                try: os.remove(os.path.join(self.CACHE_DIR, f))
                except: pass
        
        batch_cache = self._get_batch_cache_path()
        if os.path.exists(batch_cache):
            try: os.remove(batch_cache)
            except: pass
        self._batch_cache = None
        self._edge_indices = None

    def _one_hot_encode_optimized(self, sequence_bytes):
        if sequence_bytes.ndim == 0:
            bytes_data = np.frombuffer(sequence_bytes, dtype=np.uint8)
        else:
            bytes_data = sequence_bytes.view(np.uint8)
        
        if len(bytes_data) != TARGET_LENGTH:
            if len(bytes_data) < TARGET_LENGTH:
                padding = np.full(TARGET_LENGTH - len(bytes_data), 78, dtype=np.uint8)
                bytes_data = np.concatenate([bytes_data, padding])
            else:
                bytes_data = bytes_data[:TARGET_LENGTH]
        
        one_hot = _BYTE_TO_ONEHOT_MAPPING[bytes_data]
        return one_hot.astype(np.float32)

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    print("=" * 60)
    print("=== 测试 Gen3Zero 数据集 ===")
    print("=" * 60)
    
    # 假设数据根目录在 ../npy
    dataset = Gen3ZeroDataset(data_dir='../npy')
    
    print("\n获取第一个批次测试:")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"  Batch y (12类) 形状: {batch.y.shape}")
    print(f"  Batch x 形状: {batch.x.shape}")
    
    # Test get_cache_stats
    print("\n测试 get_cache_stats:")
    stats = dataset.get_cache_stats()
    print(stats)