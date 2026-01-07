import numpy as np
import os
import subprocess
import sys

# --- 0. 确保依赖库已安装 ---
def install_package(package):
    """一个辅助函数，用于安装缺失的包。"""
    print(f"库 '{package}' 未找到。正在尝试自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"'{package}' 安装成功。")
    except Exception as e:
        print(f"自动安装 '{package}' 失败: {e}")
        print(f"请手动安装: pip install {package}")
        sys.exit(1)

try:
    from Bio import SeqIO
    print("Biopython 已安装。")
except ImportError:
    install_package("biopython")
    from Bio import SeqIO

try:
    from tqdm import tqdm
    print("tqdm 已安装。")
except ImportError:
    install_package("tqdm")
    from tqdm import tqdm

# --- 1. 硬编码映射定义 (不使用动态扫描) ---
print("\n--- 使用硬编码映射定义 ---")

# 修饰类型到索引 (1-12) 的映射
MOD_TO_INDEX = {
    'Am': 1, 'Atol': 2, 'Cm': 3, 'Gm': 4, 'Tm': 5, 'Y': 6,
    'ac4C': 7, 'm1A': 8, 'm5C': 9, 'm6A': 10, 'm6Am': 11, 'm7G': 12
}

# 核苷酸分组 (4个类别: A, C, G, U)
GROUP_A = [1, 2, 8, 10, 11]  # Am, Atol, m1A, m6A, m6Am
GROUP_C = [3, 7, 9]          # Cm, ac4C, m5C
GROUP_G = [4, 12]            # Gm, m7G
GROUP_U = [5, 6]             # Tm, Y

# 组索引映射: 组名 -> 索引 (0-3)
GROUP_TO_INDEX = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3
}

# 索引到组映射: 修饰索引 -> 组索引
MOD_INDEX_TO_GROUP_INDEX = {}
for idx in GROUP_A:
    MOD_INDEX_TO_GROUP_INDEX[idx] = 0
for idx in GROUP_C:
    MOD_INDEX_TO_GROUP_INDEX[idx] = 1
for idx in GROUP_G:
    MOD_INDEX_TO_GROUP_INDEX[idx] = 2
for idx in GROUP_U:
    MOD_INDEX_TO_GROUP_INDEX[idx] = 3

print(f"修饰类型映射 (12种): {MOD_TO_INDEX}")
print(f"分组映射: A={GROUP_A}, C={GROUP_C}, G={GROUP_G}, U={GROUP_U}")

# --- 2. 辅助函数 (用于解析 Header) ---
def get_mods_from_header(header, file_type):
    """
    根据文件名类型从header中解析修饰信息。
    Returns: list of (location, mod_type) tuples
    """
    mods = []
    parts = header.split(';')
    
    if file_type == 'modomics':
        # 格式: ...;target:loc-type;other:loc-type;...
        for part in parts:
            if part.startswith('target:') or part.startswith('other:'):
                try:
                    info = part.split(':', 1)[1]
                    loc_str, mod_type = info.split('-', 1)
                    mods.append((int(loc_str), mod_type))
                except (ValueError, IndexError):
                    pass # 解析失败
                    
    elif file_type in ['m7Ghubv2', 'multiRM_v1','ac4C']:
        # 格式: ...;loc;type
        try:
            loc = int(parts[-2])
            mod_type = parts[-1]
            if not mod_type.isnumeric() and mod_type:
                 mods.append((loc, mod_type))
        except (ValueError, IndexError):
            pass # 没有修饰或格式不符
            
    return mods

# --- 3. 核心处理函数: 中心对齐与填充 ---
def process_sequence(sequence, mods):
    """
    将序列和修饰信息处理为固定长度1001。
    
    参数:
        sequence: 原始序列字符串
        mods: [(loc, mod_type), ...] 修饰位置和类型列表
    
    返回:
        padded_seq: 长度为1001的序列字符串
        site_labels: 长度为1001的int8数组，site-level labels (0-12)
        seq_labels_12: 长度为12的int8数组，sequence-level multi-labels (0/1)
        seq_labels_4: 长度为4的int8数组，sequence-level category labels (0/1)
    """
    orig_len = len(sequence)
    target_len = 1001
    
    # 初始化输出
    padded_seq = ''
    site_labels = np.zeros(target_len, dtype=np.int8)
    seq_labels_12 = np.zeros(12, dtype=np.int8)
    seq_labels_4 = np.zeros(4, dtype=np.int8)
    
    if orig_len == target_len:
        # 长度正好，无需处理
        padded_seq = sequence
        start_offset = 0
    elif orig_len > target_len:
        # 长度超过1001，执行中心裁剪
        # start = (original_len - 1001) // 2
        # 如果原始长度 - 1001 是奇数，右边会多丢弃1个字符
        start = (orig_len - target_len) // 2
        end = start + target_len
        padded_seq = sequence[start:end]
        start_offset = start
    else:
        # 长度不足1001，执行中心填充
        # left_pad = (1001 - original_len) // 2
        # 如果 1001 - original_len 是奇数，右边会多填充1个字符
        diff = target_len - orig_len
        left_pad = diff // 2
        right_pad = diff - left_pad
        padded_seq = 'N' * left_pad + sequence + 'N' * right_pad
        start_offset = -left_pad  # 负值表示左移
    
    # 处理修饰标签
    for loc, mod_type in mods:
        # 跳过不在映射中的修饰类型
        if mod_type not in MOD_TO_INDEX:
            continue
        
        mod_index = MOD_TO_INDEX[mod_type]
        
        # 计算在新序列中的位置
        if orig_len > target_len:
            # 裁剪情况: 新位置 = 原位置 - start_offset
            new_loc = loc - start_offset
        else:
            # 填充情况: 新位置 = 原位置 + left_pad (注意start_offset是负的)
            new_loc = loc - start_offset  # 因为start_offset = -left_pad
        
        # 检查新位置是否在有效范围内 [0, 1000]
        if 0 <= new_loc < target_len:
            # 设置site-level label
            site_labels[new_loc] = mod_index
            
            # 设置sequence-level 12-label (索引从1开始，所以-1)
            seq_labels_12[mod_index - 1] = 1
            
            # 设置sequence-level 4-label
            group_idx = MOD_INDEX_TO_GROUP_INDEX[mod_index]
            seq_labels_4[group_idx] = 1
    
    return padded_seq, site_labels, seq_labels_12, seq_labels_4

# --- 4. 主处理逻辑 ---

# 文件路径配置
files_to_process = [
    ('fasta/human3_fasta_class12/remove_redundant_8v3_m7Ghubv2.fasta', 'm7Ghubv2'),
    ('fasta/human3_fasta_class12/remove_redundant_8v3_modomics.fasta', 'modomics'),
    ('fasta/human3_fasta_class12/remove_redundant_8v3_multiRM_v1.fasta', 'multiRM_v1'),
    ('fasta/human3_fasta_class12/ac4C_sequence_v1.fasta', 'ac4C')
]

# 检查输出目录
output_dir = 'human3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

print("\n--- 开始处理FASTA文件 ---")

# 收集所有数据
all_sequences = []      # 保存原始序列和修饰信息
all_raw_data = []       # [(sequence, mods), ...]

for filepath, file_type in files_to_process:
    if not os.path.exists(filepath):
        print(f"!! 警告: 文件未找到 {filepath}。将跳过此文件。")
        continue
        
    print(f"\n正在处理 {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            record_iterator = SeqIO.parse(handle, "fasta")
            
            for record in tqdm(record_iterator, desc=f"Reading {os.path.basename(filepath)}", unit=" records"):
                sequence_as_string = str(record.seq)
                mods = get_mods_from_header(record.description, file_type)
                all_raw_data.append((sequence_as_string, mods))
                    
    except Exception as e:
        print(f"处理 {filepath} 时发生错误: {e}")

num_sequences = len(all_raw_data)
print(f"\n共读取到 {num_sequences} 条序列")

# --- 5. 构建并保存 .npy 文件 ---
print("\n--- 开始构建固定长度 (1001bp) 的 NumPy 矩阵 ---")

# 预分配内存以提高性能
# seq.npy: (N, 1001), 每个元素是单个字符
# 1001loc.npy: (N, 1001), int8
# 12loc.npy: (N, 12), int8
# 4loc.npy: (N, 4), int8

seq_matrix = np.zeros((num_sequences, 1001), dtype='|S1')
loc_1001_matrix = np.zeros((num_sequences, 1001), dtype=np.int8)
loc_12_matrix = np.zeros((num_sequences, 12), dtype=np.int8)
loc_4_matrix = np.zeros((num_sequences, 4), dtype=np.int8)

print(f"正在处理 {num_sequences} 条序列，标准化为1001bp...")

for i in tqdm(range(num_sequences), desc="Processing sequences", unit=" seqs"):
    sequence, mods = all_raw_data[i]
    
    # 处理序列和修饰标签
    padded_seq, site_labels, seq_labels_12, seq_labels_4 = process_sequence(sequence, mods)
    
    # 保存到矩阵
    seq_matrix[i, :] = np.array(list(padded_seq), dtype='|S1')
    loc_1001_matrix[i, :] = site_labels
    loc_12_matrix[i, :] = seq_labels_12
    loc_4_matrix[i, :] = seq_labels_4

# --- 6. 保存文件 ---
print("\n--- 保存 .npy 文件 ---")

np.save(f'{output_dir}/seq.npy', seq_matrix)
print(f"已保存 {output_dir}/seq.npy，形状: {seq_matrix.shape}，dtype: {seq_matrix.dtype}")

np.save(f'{output_dir}/1001loc.npy', loc_1001_matrix)
print(f"已保存 {output_dir}/1001loc.npy，形状: {loc_1001_matrix.shape}，dtype: {loc_1001_matrix.dtype}")

np.save(f'{output_dir}/12loc.npy', loc_12_matrix)
print(f"已保存 {output_dir}/12loc.npy，形状: {loc_12_matrix.shape}，dtype: {loc_12_matrix.dtype}")

np.save(f'{output_dir}/4loc.npy', loc_4_matrix)
print(f"已保存 {output_dir}/4loc.npy，形状: {loc_4_matrix.shape}，dtype: {loc_4_matrix.dtype}")

# --- 7. 输出映射关系和说明 ---
print("\n" + "="*60)
print("编码与修饰类型的对应关系 (硬编码)")
print("="*60)

print("\n修饰类型 -> 索引 (1-12):")
for mod_name, idx in sorted(MOD_TO_INDEX.items(), key=lambda x: x[1]):
    group_name = list(GROUP_TO_INDEX.keys())[MOD_INDEX_TO_GROUP_INDEX[idx]]
    print(f"  {idx:2d}: {mod_name:6s} -> 组: {group_name}")

print("\n核苷酸分组 (4个类别):")
print(f"  组 A (索引 0): 修饰索引 {GROUP_A}")
print(f"  组 C (索引 1): 修饰索引 {GROUP_C}")
print(f"  组 G (索引 2): 修饰索引 {GROUP_G}")
print(f"  组 U (索引 3): 修饰索引 {GROUP_U}")

print("\n" + "="*60)
print(".npy 文件格式说明")
print("="*60)

print(f"\n1. seq.npy: {seq_matrix.shape}, dtype={seq_matrix.dtype}")
print("   - 每一行是一个长度为1001的序列")
print("   - 每个元素是一个单字符核苷酸 (A, C, G, T/U, N)")

print(f"\n2. 1001loc.npy: {loc_1001_matrix.shape}, dtype={loc_1001_matrix.dtype}")
print("   - Site-level labels: 每个位置是否有修饰")
print("   - 0 = 无修饰, 1-12 = 修饰类型索引")

print(f"\n3. 12loc.npy: {loc_12_matrix.shape}, dtype={loc_12_matrix.dtype}")
print("   - Sequence-level multi-labels: 每条序列包含哪些修饰类型")
print("   - 1 = 存在该修饰, 0 = 不存在")
print("   - 12列分别对应: Am, Atol, Cm, Gm, Tm, Y, ac4C, m1A, m5C, m6A, m6Am, m7G")

print(f"\n4. 4loc.npy: {loc_4_matrix.shape}, dtype={loc_4_matrix.dtype}")
print("   - Sequence-level category labels: 每条序列包含哪些核苷酸修饰组")
print("   - 1 = 存在该组修饰, 0 = 不存在")
print("   - 4列分别对应: A, C, G, U 组")

print("\n" + "="*60)
print("中心对齐奇偶性处理说明")
print("="*60)
print("\n对于长度 > 1001 的序列:")
print("  start = (len - 1001) // 2")
print("  如果 (len - 1001) 是奇数，右边会多丢弃1个字符")
print("  例如: len=1003, start=1, 截取索引[1:1002] (丢弃左边1个，右边1个)")
print("       len=1004, start=1, 截取索引[1:1002] (丢弃左边1个，右边2个)")

print("\n对于长度 < 1001 的序列:")
print("  left_pad = (1001 - len) // 2")
print("  right_pad = 1001 - len - left_pad")
print("  如果 (1001 - len) 是奇数，右边会多填充1个'N'")
print("  例如: len=999, left_pad=1, right_pad=1 (各填充1个)")
print("       len=998, left_pad=1, right_pad=2 (左填充1个，右填充2个)")

print("\n" + "="*60)
print("脚本执行完毕！")
print("="*60)
