import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import tempfile
import os
import hashlib
import pickle

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
LINEARFOLD_PATH = '/home/dc/vscode/vscode20251112/human2/LinearFold/build/linearfold'

# Plant3Class 标签映射表
# 原始标签 -> 模型类别索引
# 1 (Pseudouridine) -> 0
# 2 (m6A) -> 1
# 3 (m5C) -> 2
PLANT_LABEL_MAPPING = {
    1: 0,  # Pseudouridine
    2: 1,  # m6A
    3: 2   # m5C
}

# 类别名称映射
CLASS_NAMES = {
    0: 'Pseudouridine',
    1: 'm6A',
    2: 'm5C'
}


def run_linearfold(sequences, timeout_seconds=1800):
    """
    使用LinearFold预测RNA序列的二级结构
    
    Args:
        sequences (list): RNA序列字符串列表
        timeout_seconds (int): 超时时间（秒）
        
    Returns:
        list: 二级结构字符串列表，失败时对应位置为None
    """
    if not sequences:
        return []
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as tmp_file:
        tmp_filename = tmp_file.name
        # 写入序列到临时文件
        for i, seq in enumerate(sequences):
            tmp_file.write(f'>seq_{i}\n{seq}\n')
    
    structures = []
    try:
        # 调用LinearFold
        process = subprocess.Popen(
            [LINEARFOLD_PATH, tmp_filename],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # 获取输出
        stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
        
        # 检查返回码
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=stdout_data,
                stderr=stderr_data
            )
        
        # 解析输出
        lines = stdout_data.strip().split('\n')
        structures = [lines[i].split(' ')[0] for i in range(2, len(lines), 3)]
        
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        error_message = f"LinearFold error: "
        if isinstance(e, subprocess.TimeoutExpired):
            error_message += f"Timeout ({timeout_seconds}s) expired."
        elif isinstance(e, FileNotFoundError):
            error_message += f"Command not found at {LINEARFOLD_PATH}."
        elif isinstance(e, subprocess.CalledProcessError):
            error_message += f"Return code {e.returncode}. Stderr: {e.stderr[:200]}..."
        else:
            error_message += f"An unexpected error occurred: {e}"
        
        print(error_message + " Defaulting to sequential edges.")
        structures = [None] * len(sequences)
    finally:
        # 删除临时文件
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    
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


class MerPlantDataset(Dataset):
    """
    用于加载Plant RNA序列数据集，执行3类细粒度分类预测任务
    类别: Pseudouridine(0), m6A(1), m5C(2)
    """
    
    def __init__(self, mode='train', data_dir='../npy'):
        """
        初始化数据集
        
        Args:
            mode (str): 'train' 或 'test'，指定加载训练集还是测试集
            data_dir (str): 数据文件目录路径
        """
        self.mode = mode
        self.data_dir = data_dir
        
        # 创建缓存目录
        cache_dir = os.path.join(tempfile.gettempdir(), 'merPlant_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名（基于数据目录和模式）
        data_dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
        cache_file = os.path.join(cache_dir, f'merPlant_{mode}_{data_dir_hash}.pkl')
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                print(f"尝试从缓存加载数据: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.sequences = cached_data['sequences']
                    self.full_labels = cached_data['full_labels']
                    
                # 验证缓存数据
                assert len(self.sequences) == len(self.full_labels), "缓存数据序列和标签数量不匹配"
                assert self.full_labels.shape[1] == 1001, f"缓存full_label维度应为1001，实际为{self.full_labels.shape[1]}"
                
                print(f"成功从缓存加载数据 (mode={mode}):")
                print(f"  总样本数: {len(self.sequences)}")
                print(f"  full_label形状: {self.full_labels.shape}")
                return
            except Exception as e:
                print(f"缓存加载失败，将重新加载数据: {e}")
        
        # 缓存不存在或加载失败，重新加载数据
        print("缓存未命中，开始重新加载数据...")
        
        # 根据模式确定数据路径
        if mode == 'train':
            seq_file = os.path.join(data_dir, 'train', '1001seq.npy')
            label_file = os.path.join(data_dir, 'train', '1001label.npy')
        elif mode == 'test':
            seq_file = os.path.join(data_dir, 'test', '1001seq.npy')
            label_file = os.path.join(data_dir, 'test', '1001label.npy')
        else:
            raise ValueError(f"不支持的mode: {mode}，必须是 'train' 或 'test'")
        
        # 加载数据
        try:
            print(f"加载序列文件: {seq_file}")
            self.sequences = np.load(seq_file, allow_pickle=True)
            
            print(f"加载标签文件: {label_file}")
            self.full_labels = np.load(label_file, allow_pickle=True)
            
            # 验证数据
            assert len(self.sequences) == len(self.full_labels), "序列和标签数量不匹配"
            assert self.full_labels.shape[1] == 1001, f"full_label维度应为1001，实际为{self.full_labels.shape[1]}"
            
            print(f"数据集初始化完成 (mode={mode}):")
            print(f"  总样本数: {len(self.sequences)}")
            print(f"  full_label形状: {self.full_labels.shape}")
            
        except Exception as e:
            raise RuntimeError(f"加载数据文件时出错: {e}")
        
        # 保存到缓存
        try:
            print(f"保存数据到缓存: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'sequences': self.sequences,
                    'full_labels': self.full_labels
                }, f)
            print("缓存保存成功")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    @staticmethod
    def clear_cache(data_dir='../npy'):
        """
        清除指定数据目录的缓存
        
        Args:
            data_dir (str): 数据文件目录路径
        """
        cache_dir = os.path.join(tempfile.gettempdir(), 'merPlant_cache')
        if not os.path.exists(cache_dir):
            print("缓存目录不存在")
            return
            
        # 生成数据目录的哈希值
        data_dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
        
        # 查找并删除匹配的缓存文件
        deleted_count = 0
        for filename in os.listdir(cache_dir):
            if data_dir_hash in filename:
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"已删除缓存文件: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除缓存文件失败 {file_path}: {e}")
        
        if deleted_count == 0:
            print(f"未找到与数据目录 {data_dir} 相关的缓存文件")
        else:
            print(f"共删除 {deleted_count} 个缓存文件")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本，返回包含3类标签的图数据
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Data: PyG Data对象，包含节点特征、边索引和类别标签
        """
        # 获取序列和full_label
        sequence = self.sequences[idx]
        full_label = self.full_labels[idx]  # 长度为1001的完整标签
        
        # 确保序列是字符串格式
        if not isinstance(sequence, str):
            sequence = str(sequence)
        
        # 使用LinearFold预测二级结构
        structures = run_linearfold([sequence])
        structure = structures[0] if structures else None
        
        # 构建边索引
        edge_index = build_edge_index_from_structure(sequence, structure) if structure else build_sequential_edge_index(sequence)
        
        # 对序列进行One-Hot编码
        one_hot_seq = self._one_hot_encode(sequence)
        
        # 节点特征
        node_features = torch.FloatTensor(one_hot_seq)
        
        # 从full_label提取3类分类标签
        class_label = self._extract_class_label(full_label)
        
        # 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.LongTensor([class_label])  # 形状为 [1]，包含类别索引 (0, 1, 或 2)
        )
        
        return data
    
    def _extract_class_label(self, full_label):
        """
        从full_label提取3类分类标签
        
        Args:
            full_label (np.array): 长度为1001的完整标签
            
        Returns:
            int: 类别索引 (0, 1, 或 2)
        """
        # 遍历full_label，找到唯一的非零值
        for i, label_id in enumerate(full_label):
            # 确保 label_id 是标量值
            if isinstance(label_id, np.ndarray):
                if label_id.size == 1:
                    label_id = label_id.item()
                else:
                    continue
            
            if label_id != 0:  # 找到非零值（修饰位点）
                if label_id in PLANT_LABEL_MAPPING:
                    return PLANT_LABEL_MAPPING[label_id]
                else:
                    raise ValueError(f"未知的标签值: {label_id}，期望值为 1, 2, 或 3")
        
        # 如果没有找到非零值，抛出异常
        raise ValueError(f"full_label中没有找到非零值（修饰位点），索引: {idx}")
    
    def _one_hot_encode(self, seq):
        """
        对RNA序列进行one-hot编码
        
        Args:
            seq (str or np.array): RNA序列字符串或数组
            
        Returns:
            np.array: one-hot编码后的数组，shape为(len(seq), 4)
        """
        # 如果已经是one-hot编码的数组，直接返回
        if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] == 4:
            return seq
            
        # 如果是字符串，进行one-hot编码
        if isinstance(seq, str):
            # 初始化one-hot编码数组
            one_hot = np.zeros((len(seq), 4))
            
            for i, nucleotide in enumerate(seq):
                # 将核苷酸转换为大写字符串
                nuc_str = str(nucleotide).upper()
                
                # 获取对应的one-hot编码
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    # 未知字符使用全零编码
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        # 如果是其他类型的数组，假设是字符数组
        if isinstance(seq, np.ndarray):
            # 初始化one-hot编码数组
            one_hot = np.zeros((len(seq), 4))
            
            for i, nucleotide in enumerate(seq):
                # 将核苷酸转换为大写字符串
                nuc_str = str(nucleotide).upper()
                
                # 获取对应的one-hot编码
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    # 未知字符使用全零编码
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        # 不支持的类型
        raise ValueError(f"不支持的序列类型: {type(seq)}")


# 测试代码
if __name__ == "__main__":
    # 清除缓存（可选）
    print("=== 清除缓存 ===")
    MerPlantDataset.clear_cache()
    
    # 测试训练集（首次加载，会创建缓存）
    print("\n=== 测试训练集（首次加载）===")
    import time
    start_time = time.time()
    train_dataset = MerPlantDataset(mode='train', data_dir='npy')
    first_load_time = time.time() - start_time
    print(f"首次加载耗时: {first_load_time:.2f} 秒")
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 测试获取单个样本
    sample_data = train_dataset[0]
    print(f"样本数据类型: {type(sample_data)}")
    print(f"节点特征形状: {sample_data.x.shape}")
    print(f"边索引形状: {sample_data.edge_index.shape}")
    print(f"类别标签形状: {sample_data.y.shape}")
    print(f"类别标签值: {sample_data.y.item()}")
    print(f"类别名称: {CLASS_NAMES[sample_data.y.item()]}")
    
    # 测试缓存效果（第二次加载，应该更快）
    print("\n=== 测试训练集（缓存加载）===")
    start_time = time.time()
    train_dataset_cached = MerPlantDataset(mode='train', data_dir='npy')
    cached_load_time = time.time() - start_time
    print(f"缓存加载耗时: {cached_load_time:.2f} 秒")
    print(f"速度提升: {first_load_time/cached_load_time:.2f}x")
    
    print("\n=== 测试测试集 ===")
    test_dataset = MerPlantDataset(mode='test', data_dir='npy')
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 测试获取单个样本
    test_data = test_dataset[0]
    print(f"测试样本节点特征形状: {test_data.x.shape}")
    print(f"测试样本边索引形状: {test_data.edge_index.shape}")
    print(f"测试样本类别标签形状: {test_data.y.shape}")
    print(f"测试样本类别标签值: {test_data.y.item()}")
    print(f"测试样本类别名称: {CLASS_NAMES[test_data.y.item()]}")
    
    # 测试多个样本，检查类别分布
    print("\n=== 类别分布统计 ===")
    from collections import Counter
    train_labels = [train_dataset[i].y.item() for i in range(min(100, len(train_dataset)))]
    label_counter = Counter(train_labels)
    print(f"训练集前100个样本的类别分布:")
    for class_idx in range(3):
        count = label_counter.get(class_idx, 0)
        print(f"  {CLASS_NAMES[class_idx]} (索引 {class_idx}): {count} 个样本")
    
    print("\n数据集类测试成功！")
