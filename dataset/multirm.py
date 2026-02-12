import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import os
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional

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


def _create_byte_to_onehot_mapping():
    """
    创建字节值到one-hot编码的映射表，避免字符串处理开销

    Returns:
        np.array: 形状为(256, 4)的映射表
    """
    mapping = np.zeros((256, 4), dtype=np.float32)

    # A (ASCII 65, 97)
    mapping[65] = [1., 0., 0., 0.]
    mapping[97] = [1., 0., 0., 0.]

    # C (ASCII 67, 99)
    mapping[67] = [0., 1., 0., 0.]
    mapping[99] = [0., 1., 0., 0.]

    # G (ASCII 71, 103)
    mapping[71] = [0., 0., 1., 0.]
    mapping[103] = [0., 0., 1., 0.]

    # T (ASCII 84, 116)
    mapping[84] = [0., 0., 0., 1.]
    mapping[116] = [0., 0., 0., 1.]

    # U (ASCII 85, 117)
    mapping[85] = [0., 0., 0., 1.]
    mapping[117] = [0., 0., 0., 1.]

    # N (ASCII 78, 110)
    mapping[78] = [0., 0., 0., 0.]
    mapping[110] = [0., 0., 0., 0.]

    return mapping


_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()


def run_linearfold(sequences, timeout_seconds=1800):
    """
    使用LinearFold预测RNA序列的二级结构

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


def _worker_process_batch(args):
    """
    工作进程函数：处理一批序列的二级结构计算

    Args:
        args: tuple (batch_indices, sequences_bytes_array, linearfold_path)

    Returns:
        list: [(idx, edge_index_numpy), ...] 或 [(idx, None, error_msg), ...]
    """
    batch_indices, sequences_bytes_array, linearfold_path = args

    sequences_str = []
    for idx in batch_indices:
        sequence_bytes = sequences_bytes_array[idx].copy()
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        sequences_str.append(sequence_str)

    results = []
    try:
        structures = run_linearfold(sequences_str)

        for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            edge_index_numpy = edge_index.cpu().numpy()
            results.append((idx, edge_index_numpy, None))

    except Exception as e:
        for idx in batch_indices:
            results.append((idx, None, str(e)))

    return results


class MultirmDataset(Dataset):
    """
    用于加载MultIRM RNA修饰数据集
    支持对每个类进行正负样本采样

    Args:
        data_dir (str): 数据目录路径，默认为 '../npy/multirm/split'
        numsample (int): 每个类抽取的正样本和负样本数量，默认50
        mode (str): 'train', 'test', 或 'valid'，默认为 'train'
        cache_dir (str): 缓存目录路径，默认为 '../npy/cache/multirm'
        use_cache (bool): 是否启用二级结构缓存，默认为True
        preload_cache (bool): 是否在初始化时加载所有边索引到内存，默认为True
        seed (int): 随机种子，用于可重复采样，默认为42
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
        # 自动检测数据目录路径
        if data_dir is None:
            # 检测是从项目根目录还是dataset子目录运行
            # 现在需要指向 51split 目录（包含按类分组的子目录）
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
        self.numsample = numsample
        self.mode = mode
        self.use_cache = use_cache
        self.seed = seed
        self.use_4class = use_4class
        self._batch_cache = None
        self._edge_indices = None
        self._rng = np.random.default_rng(self.seed)

        # 设置缓存目录
        if cache_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.CACHE_DIR = os.path.join(project_root, 'npy', 'cache', 'multirm')
        else:
            self.CACHE_DIR = cache_dir

        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)

        # 加载全部数据集
        self._load_data_meta()

        # 加载缓存
        if self.use_cache and preload_cache:
            self._load_batch_cache()

    def _load_data_meta(self):
        """加载全部数据集到内存"""
        print(f"\n{'='*60}")
        print(f"加载MultIRM数据集 (mode={self.mode}, numsample={self.numsample})")
        print(f"{'='*60}")

        # 读取按类别分组的文件
        self.all_samples = []
        self.class_data = {}
        total_samples = 0

        for class_name in MULTIRM_CLASSES:
            pos_dir = os.path.join(self.data_dir, class_name, 'pos')
            neg_dir = os.path.join(self.data_dir, class_name, 'neg')

            # 检查目录是否存在
            if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
                print(f"  {class_name}: 目录不存在，跳过")
                continue

            pos_in_path = os.path.join(pos_dir, f'{self.mode}_in.npy')
            pos_out_path = os.path.join(pos_dir, f'{self.mode}_out.npy')
            pos_out_4class_path = os.path.join(pos_dir, f'{self.mode}_out_4class.npy')
            
            neg_in_path = os.path.join(neg_dir, f'{self.mode}_in.npy')
            neg_out_path = os.path.join(neg_dir, f'{self.mode}_out.npy')
            neg_out_4class_path = os.path.join(neg_dir, f'{self.mode}_out_4class.npy')

            try:
                # 读取正样本
                pos_seqs = np.load(pos_in_path, allow_pickle=True)
                pos_labels = np.load(pos_out_path, allow_pickle=True)
                num_pos = len(pos_seqs)

                # 读取负样本
                neg_seqs = np.load(neg_in_path, allow_pickle=True)
                neg_labels = np.load(neg_out_path, allow_pickle=True)
                num_neg = len(neg_seqs)

                self.class_data[class_name] = {
                    'num_pos': num_pos,
                    'num_neg': num_neg
                }

                # 加载4类标签
                if self.use_4class:
                    pos_labels_4class = np.load(pos_out_4class_path, allow_pickle=True)
                    neg_labels_4class = np.load(neg_out_4class_path, allow_pickle=True)
                else:
                    pos_labels_4class = None
                    neg_labels_4class = None

                # 将正样本添加到全部样本列表
                for i in range(num_pos):
                    self.all_samples.append({
                        'class_name': class_name,
                        'is_pos': True,
                        'index': i,
                        'sequence_bytes': pos_seqs[i],
                        'label': pos_labels[i],
                        'label_4class': pos_labels_4class[i] if pos_labels_4class is not None else None
                    })
                    total_samples += 1

                # 将负样本添加到全部样本列表
                for i in range(num_neg):
                    self.all_samples.append({
                        'class_name': class_name,
                        'is_pos': False,
                        'index': i,
                        'sequence_bytes': neg_seqs[i],
                        'label': neg_labels[i],
                        'label_4class': neg_labels_4class[i] if neg_labels_4class is not None else None
                    })
                    total_samples += 1

                print(f"  {class_name}: 正样本{num_pos}, 负样本{num_neg}")

            except Exception as e:
                print(f"  {class_name}: 加载数据失败: {e}，跳过")
                self.class_data[class_name] = {
                    'num_pos': 0,
                    'num_neg': 0
                }
                continue
        
        self.using_array_mode = False

        print(f"\n数据集加载完成:")
        print(f"  总样本数: {total_samples}")
        print(f"  使用4类模式: {self.use_4class}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        """返回数据集大小（全部样本数）"""
        if self.using_array_mode:
            return len(self.all_sequences)
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Data:
        """
        获取单个数据样本

        Args:
            idx (int): 样本索引

        Returns:
            Data: PyG Data对象，包含：
                - x: 节点特征 (1001, 4)
                - edge_index: 边索引 (2, E)
                - y: 12类标签向量 (1, 12)
                - y_4: 4类标签向量 (1, 4) - 当use_4class=True时
                - class_idx: 类别索引（用于批量加载）
        """
        if not self.using_array_mode:
            # 直接从内存中获取样本
            sample = self.all_samples[idx]
            
            class_name = sample['class_name']
            class_idx = CLASS_TO_IDX[class_name]
            is_pos = sample['is_pos']
            actual_idx = sample['index']
            
            sequence_bytes = sample['sequence_bytes'].copy()
            label_12 = sample['label'].copy()
            
            # 使用预生成的4类标签
            label_4 = sample['label_4class'].copy() if sample['label_4class'] is not None else None
            
            cache_key = f"{class_name}_{'pos' if is_pos else 'neg'}_{actual_idx}"

        # One-hot编码
        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)

        # 将字节序列转换为字符串（用于LinearFold）
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')

        # 获取或计算边索引
        edge_index = self._get_or_compute_edge_index_cached(sequence_str, cache_key)

        # 节点特征
        node_features = torch.FloatTensor(one_hot_seq)

        # 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(label_12).unsqueeze(0),
            class_idx=torch.tensor([idx], dtype=torch.long)  # 使用全局索引
        )
        
        # 如果使用4类模式，添加4类标签
        if self.use_4class and label_4 is not None:
            data.y_4 = torch.FloatTensor(label_4).unsqueeze(0)

        return data

    def _one_hot_encode_optimized(self, sequence_bytes: np.ndarray) -> np.ndarray:
        """
        优化的one-hot编码，直接处理|S1字节流

        Args:
            sequence_bytes (np.array): |S1类型的字节数组

        Returns:
            np.array: one-hot编码后的数组，shape为(len(seq), 4)
        """
        byte_array = sequence_bytes.view(np.uint8)
        one_hot = _BYTE_TO_ONEHOT_MAPPING[byte_array]
        return one_hot.astype(np.float32)

    def _get_batch_cache_path(self) -> str:
        """
        获取批量缓存文件路径

        Returns:
            str: 批量缓存文件完整路径
        """
        cache_filename = f"multirm_{self.mode}_{BATCH_CACHE_FILE}"
        return os.path.join(self.CACHE_DIR, cache_filename)

    def _load_batch_cache(self):
        """
        从批量缓存文件加载所有边索引到内存

        如果磁盘上存在缓存文件，则加载到内存中；否则跳过
        """
        cache_path = self._get_batch_cache_path()

        if os.path.exists(cache_path):
            print(f"发现磁盘缓存文件: {cache_path}")
            print(f"正在加载到内存...")

            try:
                # 加载磁盘缓存
                cache_data = np.load(cache_path, allow_pickle=True)

                # 初始化内存缓存
                if not hasattr(self, '_edge_index_cache'):
                    self._edge_index_cache = {}

                # 将磁盘缓存加载到内存
                # 假设缓存文件中存储的是 {key: edge_index}
                if hasattr(cache_data, 'items'):
                    for key in cache_data.files:
                        edge_index = cache_data[key]
                        self._edge_index_cache[key] = edge_index

                print(f"已加载 {len(self._edge_index_cache)} 个缓存条目到内存")

                self._batch_cache = cache_data
                self._edge_indices = self._edge_index_cache

            except Exception as e:
                print(f"警告: 加载磁盘缓存失败: {e}")
                print(f"将在首次访问时重新计算二级结构")
                self._batch_cache = None
                self._edge_indices = None
        else:
            print(f"磁盘缓存文件不存在: {cache_path}")
            print(f"首次运行时将创建缓存文件")
            self._batch_cache = None
            self._edge_indices = None

    def _save_cache_to_disk(self):
        """
        将内存中的边索引缓存保存到磁盘

        使用npz格式保存所有边索引
        """
        cache_path = self._get_batch_cache_path()

        if not hasattr(self, '_edge_index_cache') or not self._edge_index_cache:
            print("内存缓存为空，无需保存")
            return

        print(f"\n正在保存缓存到磁盘: {cache_path}")

        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 将边索引转换为numpy数组并保存
            save_dict = {}
            for key, edge_index in self._edge_index_cache.items():
                save_dict[key] = edge_index.cpu().numpy()

            np.savez(cache_path, **save_dict)
            print(f"缓存已保存: {len(save_dict)} 个条目")

        except Exception as e:
            print(f"警告: 保存缓存失败: {e}")

    def precompute_all_structures(
        self,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        预计算所有可能序列的二级结构并缓存

        由于使用动态采样，此方法会遍历所有类别的所有样本进行预计算

        Args:
            batch_size (int): 每次调用LinearFold的序列数量
            num_workers (int): 工作进程数，None表示使用CPU核心数
            show_progress (bool): 是否显示进度条

        Returns:
            dict: 统计信息
        """
        from tqdm import tqdm

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
        for sample in self.all_samples:
            class_name = sample['class_name']
            is_pos = sample['is_pos']
            actual_idx = sample['index']
            
            sequence_bytes = sample['sequence_bytes'].copy()
            sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
            all_sequences.append(sequence_str)
            all_keys.append(f"{class_name}_{'pos' if is_pos else 'neg'}_{actual_idx}")

        stats['total'] = len(all_sequences)
        print(f"  需要计算的序列总数: {stats['total']}")

        # 初始化缓存
        if not hasattr(self, '_edge_index_cache'):
            self._edge_index_cache = {}

        # 批量计算
        iterator = range(0, len(all_sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="预计算二级结构")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(all_sequences))
            batch_sequences = all_sequences[start_idx:end_idx]
            batch_keys = all_keys[start_idx:end_idx]

            try:
                structures = run_linearfold(batch_sequences)

                for seq_str, structure, key in zip(batch_sequences, structures, batch_keys):
                    edge_index = build_edge_index_from_structure(seq_str, structure)
                    self._edge_index_cache[key] = edge_index
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
        print(f"  缓存大小: {len(self._edge_index_cache)} 条目")

        # 保存缓存到磁盘
        if len(self._edge_index_cache) > 0:
            self._save_cache_to_disk()

        print(f"{'='*60}\n")

        return stats

    def _get_or_compute_edge_index_cached(self, sequence_str: str, cache_key: str) -> torch.Tensor:
        """
        获取或计算边索引（使用字符串缓存键）

        优先级：
        1. LRU缓存（内存中）
        2. 磁盘缓存（懒加载）
        3. LinearFold实时计算

        Args:
            sequence_str (str): RNA序列字符串
            cache_key (str): 缓存键

        Returns:
            torch.Tensor: 边索引张量
        """
        # 1. 检查内存缓存
        if not hasattr(self, '_edge_index_cache'):
            self._edge_index_cache = {}

        if cache_key in self._edge_index_cache:
            return self._edge_index_cache[cache_key]

        # 2. 尝试从磁盘缓存加载（懒加载）
        if self.use_cache:
            cache_path = self._get_batch_cache_path()
            if os.path.exists(cache_path):
                try:
                    # 只加载需要的单个条目，而不是整个文件
                    cache_data = np.load(cache_path, allow_pickle=True)
                    if cache_key in cache_data.files:
                        edge_index_numpy = cache_data[cache_key]
                        edge_index = torch.from_numpy(edge_index_numpy)
                        self._edge_index_cache[cache_key] = edge_index
                        return edge_index
                except Exception as e:
                    # 磁盘加载失败，继续计算
                    pass

        # 3. 使用LinearFold计算二级结构
        try:
            structures = run_linearfold([sequence_str])
            structure = structures[0]
            edge_index = build_edge_index_from_structure(sequence_str, structure)

            # 缓存结果
            self._edge_index_cache[cache_key] = edge_index
            return edge_index
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
            raise

    def clear_cache(self):
        """清除内存中的边索引缓存"""
        if hasattr(self, '_edge_index_cache'):
            cache_size = len(self._edge_index_cache)
            self._edge_index_cache.clear()
            print(f"已清除内存缓存: {cache_size} 条目")
        else:
            print("内存缓存为空")

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            dict: 包含缓存统计信息的字典
        """
        stats = {
            'cache_dir': self.CACHE_DIR,
            'memory_cache': {
                'entries': 0,
                'keys': []
            }
        }

        if hasattr(self, '_edge_index_cache') and self._edge_index_cache:
            stats['memory_cache']['entries'] = len(self._edge_index_cache)
            stats['memory_cache']['keys'] = list(self._edge_index_cache.keys())

        return stats


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
    print(f"内存缓存条目数: {cache_stats['memory_cache']['entries']}")
    if cache_stats['memory_cache']['entries'] > 0:
        print(f"  前5个缓存键: {cache_stats['memory_cache']['keys'][:5]}")

    print("\n" + "=" * 60)
    print("=== 所有测试完成！===")
    print("=" * 60)
