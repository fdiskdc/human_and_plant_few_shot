import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import tempfile
import os
import uuid
import hashlib
import pickle

# 12类标签映射表 (与 human_make_npy.py 中的 MOD_TO_INDEX 一致)
# human_make_npy.py 中的 MOD_TO_INDEX:
#   'Am': 1, 'Atol': 2, 'Cm': 3, 'Gm': 4, 'Tm': 5, 'Y': 6,
#   'ac4C': 7, 'm1A': 8, 'm5C': 9, 'm6A': 10, 'm6Am': 11, 'm7G': 12
#
# 1001loc.npy 中存储的值是 1-12 (mod_index)
# LABEL_MAPPING 将这些 ID 映射到 0-11 的模型索引 (mod_index - 1)

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
# 根据 human_make_npy.py 中的核苷酸分组转换
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

# 每个核苷酸组包含的ID (1-12, 对应 human_make_npy.py)
NUCLEOTIDE_GROUPS = {
    'A': [1, 2, 8, 10, 11],  # Am, Atol, m1A, m6A, m6Am
    'C': [3, 7, 9],          # Cm, ac4C, m5C
    'G': [4, 12],            # Gm, m7G
    'U': [5, 6]              # Tm, Y
}

# 12类修饰名称映射 (模型索引 -> 修饰名称)
# 与 human_make_npy.py 中的 MOD_TO_INDEX 一致 (mod_index - 1 转换后)
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
    
    Args:
        one_hot_array (np.array): One-Hot编码的数组，形状为(N, 4)
        
    Returns:
        str: RNA序列字符串
    """
    # 创建反向映射
    reverse_mapping = {
        tuple([1., 0., 0., 0.]): 'A',
        tuple([0., 1., 0., 0.]): 'C',
        tuple([0., 0., 1., 0.]): 'G',
        tuple([0., 0., 0., 1.]): 'U',  # RNA中优先使用U而不是T
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
    
    Args:
        sequences (list): RNA序列字符串列表
        timeout_seconds (int): 超时时间（秒）
        
    Returns:
        list: 二级结构字符串列表
        
    Raises:
        RuntimeError: 如果LinearFold执行失败
        FileNotFoundError: 如果LinearFold可执行文件不存在
        subprocess.TimeoutExpired: 如果执行超时
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
        
        # 调试：打印原始输出
        if len(stdout_data.strip()) == 0:
            print(f"LinearFold调试信息:")
            print(f"  返回码: {process.returncode}")
            print(f"  stdout长度: {len(stdout_data)}")
            print(f"  stderr长度: {len(stderr_data)}")
            if stderr_data:
                print(f"  stderr内容: {stderr_data[:500]}")
            print(f"  输入序列数: {len(sequences)}")
            print(f"  输入前200字符: {fasta_input[:200]}")
            
            raise RuntimeError("LinearFold没有返回任何输出")
        
        # 解析输出
        # LinearFold输出格式：
        # >seq_0
        # ACGUACGU
        # ........ (-0.08)
        
        lines = stdout_data.strip().split('\n')
        structures = []
        
        # 过滤掉空行
        lines = [line.strip() for line in lines if line.strip()]
        
        # 每个序列对应3行：header, sequence, structure+energy
        # 结构在第3行（索引2），然后每隔3行
        for i in range(2, len(lines), 3):
            line = lines[i]
            if not line:
                continue
            # 结构在空格前，例如：........ (-0.08) -> ........
            structure = line.split()[0]
            structures.append(structure)
        
        # 验证结构数量
        if len(structures) != len(sequences):
            # 打印调试信息
            print(f"LinearFold输出调试信息:")
            print(f"  输入序列数: {len(sequences)}")
            print(f"  输出行数: {len(lines)}")
            print(f"  解析到的结构数: {len(structures)}")
            print(f"  输出内容前10行:")
            for i, line in enumerate(lines[:10]):
                print(f"    [{i}]: {line}")
            
            raise RuntimeError(
                f"LinearFold返回的结构数量({len(structures)})与输入序列数量({len(sequences)})不匹配"
            )
        
    except FileNotFoundError as e:
        # 重新抛出FileNotFoundError
        raise
    except subprocess.TimeoutExpired as e:
        # 重新抛出TimeoutExpired
        raise subprocess.TimeoutExpired(e.cmd, e.timeout, output=e.output, stderr=e.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold执行失败: {e}")
    except Exception as e:
        raise RuntimeError(f"LinearFold执行过程中发生未知错误: {e}")
    
    return structures

def build_edge_index_from_structure(sequence, structure):
    """
    根据RNA二级结构构建边索引
    
    Args:
        sequence (str): RNA序列
        structure (str): 二级结构（点括号表示法）
        
    Returns:
        torch.Tensor: 边索引，形状为[2, E]
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
    
    Args:
        sequence (str): RNA序列
        
    Returns:
        torch.Tensor: 边索引，形状为[2, E]
    """
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    
    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)

class Mer100Dataset(Dataset):
    """
    用于加载4个核苷酸（A, C, G, U）的RNA序列数据集，执行12类细粒度分类预测任务
    支持注意力监督和排序正则化

    使用内存映射加载，支持多线程DataLoader
    """

    def __init__(self, mode='train', data_dir='../npy', cache_dir=None, use_human3=True, use_cache=True, preload_cache=True, lazy_cache=False):
        """
        初始化数据集（支持内存映射和多线程）

        Args:
            mode (str): 'train' 或 'test'，指定加载训练集还是测试集
            data_dir (str): 数据文件目录路径
            cache_dir (str): 缓存目录路径（默认None，使用默认路径）
            use_human3 (bool): 是否使用human3目录数据（默认True）
            use_cache (bool): 是否启用二级结构缓存（默认True）
            preload_cache (bool): 是否在初始化时加载所有边索引到内存（默认True）
            lazy_cache (bool): 是否按需从磁盘读取边索引（默认False，内存优化）
        """
        self.mode = mode
        self.data_dir = data_dir
        self.use_cache = use_cache
        self._batch_cache = None  # 批量缓存数据
        self._edge_indices = None  # 内存中的边索引列表
        self._lazy_cache = lazy_cache  # 按需加载模式
        self._cache_path = None  # 缓存文件路径（lazy模式用）
        self._lru_cache = {}  # LRU缓存（lazy模式用）
        self._lru_max_size = 256  # LRU缓存最大条目数

        # 设置缓存目录
        if cache_dir is None:
            self.CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        else:
            self.CACHE_DIR = cache_dir

        # 确保缓存目录存在
        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # 使用human3目录的数据
        if use_human3:
            # 检测是从项目根目录还是dataset子目录运行
            if os.path.exists('human3'):
                human3_dir = 'human3'
            elif os.path.exists('../human3'):
                human3_dir = '../human3'
            else:
                # 使用绝对路径作为最后的回退
                human3_dir = 'npy/human3'
            
            self._human3_dir = human3_dir
            print(f"使用内存映射加载human3数据: {human3_dir}")
            
            # 使用mmap_mode='r'进行内存映射加载，支持多进程共享内存
            self.sequences = np.load(f'{human3_dir}/seq.npy', mmap_mode='r')
            self.full_labels = np.load(f'{human3_dir}/1001loc.npy', mmap_mode='r')
            self.y_12class = np.load(f'{human3_dir}/12loc.npy', mmap_mode='r')
            self.y_4class = np.load(f'{human3_dir}/4loc.npy', mmap_mode='r')
            
            print(f"数据集初始化完成 (mode={mode}):")
            print(f"  总样本数: {len(self.sequences)}")
            print(f"  序列形状: {self.sequences.shape}, dtype: {self.sequences.dtype}")
            print(f"  1001loc形状: {self.full_labels.shape}, dtype: {self.full_labels.dtype}")
            print(f"  12loc形状: {self.y_12class.shape}, dtype: {self.y_12class.dtype}")
            print(f"  4loc形状: {self.y_4class.shape}, dtype: {self.y_4class.dtype}")

            # 如果启用缓存且preload_cache=True，尝试加载批量缓存
            if self.use_cache and preload_cache and not lazy_cache:
                self._load_batch_cache()
            elif self.use_cache and lazy_cache:
                # Lazy模式：仅记录缓存路径，不加载到内存
                self._cache_path = self._get_batch_cache_path()
                if os.path.exists(self._cache_path):
                    print(f"Lazy缓存模式: 边索引将按需从磁盘读取: {self._cache_path}")
                else:
                    print(f"Lazy缓存文件不存在: {self._cache_path}")
                    print(f"  提示: 请先调用 dataset.precompute_all_structures() 生成缓存")
        else:
            # 使用原来的np数据（保持向后兼容）
            self._load_legacy_data(mode, data_dir)
    
    def _load_legacy_data(self, mode, data_dir):
        """
        加载旧格式的npy数据（向后兼容）
        
        Args:
            mode (str): 'train' 或 'test'
            data_dir (str): 数据文件目录路径
        """
        print(f"使用旧格式加载数据: {data_dir}")
        
        # 定义核苷酸文件名
        nucleotides = ['A', 'C', 'G', 'U']
        suffix = '_train.npy' if mode == 'train' else '_test.npy'
        
        # 加载并合并4个核苷酸的数据
        all_sequences = []
        all_full_labels = []
        all_y_12class = []
        all_y_4class = []
        
        for nuc in nucleotides:
            file_path = f'{data_dir}/{nuc}_expert{suffix}'
            try:
                data = np.load(file_path, allow_pickle=True)
                
                # 提取序列和标签
                sequences = data['seq']
                full_labels = data['full_label']
                
                # 从full_label生成12类和4类标签
                y_12class = np.zeros((len(sequences), 12), dtype=np.int8)
                y_4class = np.zeros((len(sequences), 4), dtype=np.int8)
                
                for i, label in enumerate(full_labels):
                    for label_id in np.unique(label):
                        if label_id == 0:
                            continue
                        if label_id in LABEL_MAPPING:
                            idx = LABEL_MAPPING[label_id]
                            y_12class[i, idx] = 1
                            nuc_group = INDEX_TO_NUCLEOTIDE[idx]
                            group_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}[nuc_group]
                            y_4class[i, group_idx] = 1
                
                all_sequences.append(sequences)
                all_full_labels.append(full_labels)
                all_y_12class.append(y_12class)
                all_y_4class.append(y_4class)
                
                print(f"已加载 {file_path}: {len(sequences)} 个样本")
                
            except Exception as e:
                raise RuntimeError(f"加载文件 {file_path} 时出错: {e}")
        
        # 垂直堆叠所有数据
        self.sequences = np.concatenate(all_sequences, axis=0)
        self.full_labels = np.concatenate(all_full_labels, axis=0)
        self.y_12class = np.concatenate(all_y_12class, axis=0)
        self.y_4class = np.concatenate(all_y_4class, axis=0)
        
        print(f"数据集初始化完成 (mode={mode}):")
        print(f"  总样本数: {len(self.sequences)}")
        print(f"  full_label形状: {self.full_labels.shape}")
        print(f"  12loc形状: {self.y_12class.shape}")
        print(f"  4loc形状: {self.y_4class.shape}")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in ('sequences', 'full_labels', 'y_12class', 'y_4class', '_batch_cache', '_lru_cache'):
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lru_cache = {}
        human3_dir = self._human3_dir
        self.sequences = np.load(f'{human3_dir}/seq.npy', mmap_mode='r')
        self.full_labels = np.load(f'{human3_dir}/1001loc.npy', mmap_mode='r')
        self.y_12class = np.load(f'{human3_dir}/12loc.npy', mmap_mode='r')
        self.y_4class = np.load(f'{human3_dir}/4loc.npy', mmap_mode='r')
        cache_path = self._get_batch_cache_path()
        if getattr(self, '_lazy_cache', False) and os.path.exists(cache_path):
            # Lazy模式：不加载到内存，仅记录路径
            self._cache_path = cache_path
            self._edge_indices = None
        elif os.path.exists(cache_path):
            cache = np.load(cache_path, allow_pickle=True)
            self._edge_indices = cache['edge_indices']
        else:
            self._edge_indices = None

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
                - y_site: 1001长度位点标签 (1001,)
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
        """
        从full_label提取注意力掩码
        
        Args:
            full_label (np.array): 长度为1001的完整标签
            
        Returns:
            dict: 包含4个核苷酸注意力掩码的字典
        """
        # 初始化4个核苷酸的注意力掩码
        attn_masks = {}
        for nuc in ['A', 'C', 'G', 'U']:
            attn_masks[nuc] = np.zeros(1001, dtype=np.float32)
        
        # 遍历full_label，提取注意力掩码
        for i, label_id in enumerate(full_label):
            if label_id == 0:  # 跳过无修饰位置
                continue
                
            if label_id in LABEL_MAPPING:
                # 获取对应的核苷酸组
                model_idx = LABEL_MAPPING[label_id]
                nucleotide = INDEX_TO_NUCLEOTIDE[model_idx]
                
                # 更新对应核苷酸的注意力掩码
                attn_masks[nucleotide][i] = 1.0
        
        # 归一化注意力掩码（如果非零）
        for nuc in ['A', 'C', 'G', 'U']:
            mask = attn_masks[nuc]
            if np.sum(mask) > 0:
                attn_masks[nuc] = mask / np.sum(mask)  # 归一化为概率分布
            else:
                # 如果没有修饰位点，使用均匀分布
                attn_masks[nuc] = np.ones(1001, dtype=np.float32) / 1001
        
        # 将注意力掩码转换为张量
        for nuc in ['A', 'C', 'G', 'U']:
            attn_masks[nuc] = torch.FloatTensor(attn_masks[nuc])
        
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
        # 添加数据集名称前缀
        return os.path.join(self.CACHE_DIR, f"human_{self.mode}_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        """
        从批量缓存文件加载所有边索引到内存

        如果缓存文件存在，直接加载；如果不存在，则不进行任何操作
        """
        cache_path = self._get_batch_cache_path()

        if os.path.exists(cache_path):
            print(f"正在从批量缓存加载边索引: {cache_path}")
            try:
                self._batch_cache = np.load(cache_path, allow_pickle=True)
                self._edge_indices = self._batch_cache['edge_indices']
                self._batch_cache = None
                print(f"  已加载 {len(self._edge_indices)} 个边索引")
                print(f"  缓存文件大小: {os.path.getsize(cache_path) / (1024**2):.2f} MB")
            except Exception as e:
                print(f"  警告: 加载批量缓存失败: {e}")
                self._batch_cache = None
                self._edge_indices = None
        else:
            print(f"批量缓存文件不存在: {cache_path}")
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
        import itertools

        num_samples = len(self.sequences)
        cache_path = self._get_batch_cache_path()

        # 确定工作进程数
        if num_workers is None:
            num_workers = cpu_count()

        use_multiprocessing = num_workers > 1

        print(f"\n{'='*60}")
        print(f"开始预计算所有序列的二级结构...")
        print(f"  样本总数: {num_samples}")
        print(f"  批量大小: {batch_size}")
        print(f"  工作进程数: {num_workers if use_multiprocessing else 1} ({'多进程' if use_multiprocessing else '单进程'})")
        print(f"  缓存路径: {cache_path}")
        print(f"{'='*60}\n")

        # 初始化边索引数组（使用object类型存储不同形状的数组）
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
                    results_iter = tqdm(results_iter, total=total_batches, desc="预计算二级结构")

                # 收集结果
                for batch_results in results_iter:
                    for idx, edge_index_numpy, error in batch_results:
                        if error is None:
                            edge_indices[idx] = edge_index_numpy
                            stats['computed'] += 1
                        else:
                            stats['failed'] += 1
                            if stats['failed'] <= 5:  # 只打印前5个错误
                                print(f"  警告: 索引 {idx} 计算失败: {error}")

        else:
            # 单进程处理（原逻辑）
            iterator = range(0, num_samples, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="预计算二级结构")

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
                        # 将torch.Tensor转换为numpy数组存储
                        edge_indices[idx] = edge_index.cpu().numpy()
                        stats['computed'] += 1

                except Exception as e:
                    print(f"\n警告: 批量计算失败 (索引 {start_idx}-{end_idx}): {e}")
                    for idx in batch_indices:
                        stats['failed'] += 1

        # 保存到批量缓存文件
        print(f"\n正在保存批量缓存到: {cache_path}")
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
        self._batch_cache = None

        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"预计算完成！")
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
            str: 缓存键值（基于序列内容哈希）
        """
        # 使用序列内容的MD5哈希作为缓存键，确保相同序列使用相同缓存
        sequence_hash = hashlib.md5(sequence_str.encode('utf-8')).hexdigest()
        # 结合模式(train/test)和索引确保唯一性
        cache_key = f"{self.mode}_{idx}_{sequence_hash}"
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
            torch.Tensor or None: 边索引，如果缓存不存在则返回None
        """
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['edge_index']
            except (pickle.PickleError, EOFError, KeyError) as e:
                # 缓存文件损坏，删除并返回None
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
                'mode': self.mode
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except (pickle.PickleError, OSError) as e:
            # 缓存保存失败，静默处理
            pass

    def _get_or_compute_edge_index(self, sequence_str, idx):
        """
        获取或计算边索引（优先使用批量缓存）

        优先级：
        1. 批量缓存（内存中 或 lazy磁盘读取）
        2. 单文件缓存（旧机制）
        3. LinearFold实时计算

        Args:
            sequence_str (str): RNA序列字符串
            idx (int): 样本索引

        Returns:
            torch.Tensor: 边索引张量

        Raises:
            FileNotFoundError: 如果LinearFold可执行文件不存在
            subprocess.TimeoutExpired: 如果LinearFold执行超时
            RuntimeError: 如果LinearFold执行失败
        """
        # 1. 优先使用批量缓存（内存模式）
        if self._edge_indices is not None and idx < len(self._edge_indices):
            cached = self._edge_indices[idx]
            if cached is not None:
                return torch.from_numpy(cached)

        # 1b. Lazy缓存模式：从磁盘按需读取
        if self._lazy_cache and self._cache_path and os.path.exists(self._cache_path):
            # Check LRU cache first
            if idx in self._lru_cache:
                return self._lru_cache[idx]
            # Read from disk
            try:
                cache_data = np.load(self._cache_path, allow_pickle=True)
                edge_indices = cache_data['edge_indices']
                if idx < len(edge_indices):
                    cached = edge_indices[idx]
                    if cached is not None:
                        edge_tensor = torch.from_numpy(cached)
                        # Add to LRU cache (with size limit)
                        if len(self._lru_cache) < self._lru_max_size:
                            self._lru_cache[idx] = edge_tensor
                        return edge_tensor
            except Exception:
                pass  # Fall through to other methods

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

            # 保存到旧缓存（向后兼容）
            self._save_to_cache(cache_key, edge_index)

            return edge_index
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
            # 重新抛出异常，不允许自动回退到顺序边
            raise

    def clear_cache(self):
        """
        清除当前模式的所有缓存文件（包括批量缓存和单文件缓存）
        """
        if not os.path.exists(self.CACHE_DIR):
            return

        removed_count = 0
        prefix = f"{self.mode}_"

        # 清除单文件缓存
        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    os.remove(cache_path)
                    removed_count += 1
                except OSError:
                    pass

        print(f"已清除 {removed_count} 个单文件缓存 (mode={self.mode})")

        # 清除批量缓存
        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                os.remove(batch_cache_path)
                file_size_mb = os.path.getsize(batch_cache_path) / (1024**2)
                print(f"已清除批量缓存: {batch_cache_path} ({file_size_mb:.2f} MB)")
            except OSError as e:
                print(f"清除批量缓存失败: {e}")

        # 清除内存中的缓存
        self._batch_cache = None
        self._edge_indices = None

    def get_cache_stats(self):
        """
        获取缓存统计信息（包括批量缓存和单文件缓存）

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
        prefix = f"{self.mode}_"
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
    
    def _one_hot_encode_optimized(self, sequence_bytes):
        """
        优化的one-hot编码，直接处理|S1字节流（高性能版本）
        
        Args:
            sequence_bytes (np.array): |S1类型的字节数组
            
        Returns:
            np.array: one-hot编码后的数组，shape为(len(seq), 4)
        """
        # 将字节数组转换为整数数组
        byte_array = sequence_bytes.view(np.uint8)
        
        # 使用预建的映射表进行快速查找
        # byte_array是(1001,)数组，映射表是(256,4)
        # 输出形状为(1001, 4)
        one_hot = _BYTE_TO_ONEHOT_MAPPING[byte_array]
        
        return one_hot.astype(np.float32)
    
    def _one_hot_encode(self, seq):
        """
        对RNA序列进行one-hot编码（保留用于兼容性）
        
        Args:
            seq (str or np.array): RNA序列字符串或数组
            
        Returns:
            np.array: one-hot编码后的数组，shape为(len(seq), 4)
        """
        # 如果已经是one-hot编码的数组，直接返回
        if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] == 4:
            return seq
            
        # 如果是|S1字节数组，使用优化版本
        if isinstance(seq, np.ndarray) and seq.dtype == np.dtype('|S1'):
            return self._one_hot_encode_optimized(seq)
        
        # 如果是字符串，进行one-hot编码
        if isinstance(seq, str):
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            
            for i, nucleotide in enumerate(seq):
                nuc_str = nucleotide.upper()
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        # 如果是其他类型的数组，假设是字符数组
        if isinstance(seq, np.ndarray):
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            
            for i, nucleotide in enumerate(seq):
                nuc_str = str(nucleotide).upper()
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        raise ValueError(f"不支持的序列类型: {type(seq)}")


# 测试代码
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    print("=" * 60)
    print("=== 测试1: 单线程加载数据集 ===")
    print("=" * 60)
    
    dataset = Mer100Dataset(mode='train', use_human3=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取单个样本
    print("\n测试获取单个样本:")
    sample = dataset[0]
    print(f"  节点特征形状: {sample.x.shape}")
    print(f"  边索引形状: {sample.edge_index.shape}")
    print(f"  y (12类) 形状: {sample.y.shape}, 值: {sample.y}")
    print(f"  y_4class (4类) 形状: {sample.y_4class.shape}, 值: {sample.y_4class}")
    print(f"  y_site (1001位点) 形状: {sample.y_site.shape}")
    print(f"  注意力掩码:")
    for nuc in ['A', 'C', 'G', 'U']:
        mask_attr = f'attn_mask_{nuc}'
        if hasattr(sample, mask_attr):
            mask = getattr(sample, mask_attr)
            if mask is not None and hasattr(mask, 'shape'):
                print(f"    {nuc}: 形状={mask.shape}, 非零元素={mask.sum().item()}")
            else:
                print(f"    {nuc}: 掩码为None或无效")
        else:
            print(f"    {nuc}: 属性不存在")
    # 测试 N 掩码
    if hasattr(sample, 'attn_mask_N'):
        mask_n = sample.attn_mask_N
        if mask_n is not None and hasattr(mask_n, 'shape'):
            print(f"    N: 形状={mask_n.shape}, 'N'字符数量={mask_n.sum().item()}")
        else:
            print(f"    N: 掩码为None或无效")
    else:
        print(f"    N: 属性不存在")
    
    print("\n" + "=" * 60)
    print("=== 测试2: PyG DataLoader测试 ===")
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
    # y_site是节点级别的，会被PyG展平为 (batch_size * 1001,)
    if batch.y_site is not None and hasattr(batch.y_site, 'shape'):
        expected_shape = (32 * 1001,)
        assert batch.y_site.shape == expected_shape, f"batch.y_site形状错误: {batch.y_site.shape}, 期望: {expected_shape}"
        print(f"  ✓ batch.y_site形状正确: {batch.y_site.shape} (32个样本 × 1001个位点)")
    print("\n✓ 所有维度验证通过！")
    
    print("\n" + "=" * 60)
    print("=== 测试3: 错误处理测试 ===")
    print("=" * 60)
    
    print("\n测试: LinearFold路径错误时的异常抛出")
    print("（注意：此测试需要实际修改代码中的LINEARFOLD_PATH路径）")
    print("当前LinearFold路径:", LINEARFOLD_PATH)
    print("如需测试错误处理，请手动修改LINEARFOLD_PATH为不存在的路径后运行")
    
    # 简化测试：只测试run_linearfold函数的异常处理
    print("\n测试run_linearfold函数的输入验证:")
    try:
        result = run_linearfold([])
        print(f"✓ 空列表输入返回空结果: {result}")
    except Exception as e:
        print(f"✗ 异常: {e}")
    
    print("\n" + "=" * 60)
    print("=== 所有测试完成！===")
    print("=" * 60)