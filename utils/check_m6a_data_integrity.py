"""
utils/check_m6a_data_integrity.py - m6A 标签+序列完整性检查 / m6A Label+Sequence Integrity Check

m6A 修饰标签 + 序列完整性检查:验证 m6A 修饰位点都是 A 核苷酸。
m6A modification label + sequence integrity check: verify m6A modification sites are all A nucleotides.

功能模块 / Modules:
- m6A 标签加载 / m6A label loading
- 序列加载 / Sequence loading
- A 核苷酸验证 / A nucleotide verification
- main: 主入口 / Main entry point

输入 / Inputs:
- human3/seq.npy: RNA 序列 / RNA sequences
- human3/1001loc.npy: 1001 位点级标签 / 1001 site-level labels

输出 / Outputs:
- 终端检查结果 / Terminal check results
- 不匹配位点报告 / Mismatch site report

数据流 / Data Flow:
1. 加载序列与标签 / Load sequences and labels
2. 提取 m6A 位点 / Extract m6A sites
3. 验证 A 核苷酸 / Verify A nucleotides
4. 报告不匹配 / Report mismatches

相关文件 / Related Files:
- 调用 / Calls: numpy
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python -m utils.check_m6a_data_integrity

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
from collections import defaultdict

# 标签映射表（与 dataset/gen3.py 一致）
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

# 模型索引到核苷酸的映射
INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',
    2: 'C',
    3: 'G',
    4: 'U', 5: 'U',
    6: 'C',
    7: 'A',
    8: 'C',
    9: 'A', 10: 'A',
    11: 'G'
}

# 修饰名称
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}

# One-hot编码映射
ONE_HOT_MAPPING = {
    'A': [1., 0., 0., 0.],
    'C': [0., 1., 0., 0.],
    'G': [0., 0., 1., 0.],
    'T': [0., 0., 0., 1.],
    'U': [0., 0., 0., 1.],
    'N': [0., 0., 0., 0.]
}

def check_m6a_integrity(data_dir='npy/3gen'):
    """
    检查 m6A 数据的完整性
    
    Args:
        data_dir: 数据目录路径
    """
    print("=" * 80)
    print("m6A 数据完整性检查")
    print("=" * 80)
    print(f"\n数据路径: {data_dir}")
    
    # 目标参数
    TARGET_MOD_INDEX = 10  # 原始标签
    TARGET_MODEL_INDEX = 9  # 模型索引
    TARGET_NUCLEOTIDE = 'A'
    TARGET_ONE_HOT = [1., 0., 0., 0.]
    
    # 第一步：加载数据
    print("\n[第一步] 加载数据...")
    try:
        sequences = np.load(f'{data_dir}/seq.npy', mmap_mode='r')
        full_labels = np.load(f'{data_dir}/1001loc.npy', mmap_mode='r')
        y_12class = np.load(f'{data_dir}/12loc.npy', mmap_mode='r')
        
        print(f"  ✓ 成功加载 seq.npy: {sequences.shape}")
        print(f"  ✓ 成功加载 1001loc.npy: {full_labels.shape}")
        print(f"  ✓ 成功加载 12loc.npy: {y_12class.shape}")
    except FileNotFoundError as e:
        print(f"  ✗ 文件未找到: {e}")
        return
    
    # 第二步：标签过滤 - 找出所有 m6A 样本
    print("\n[第二步] 筛选 m6A 样本...")
    
    m6a_sample_indices = []
    m6a_positions = []  # 存储每个样本中 m6A 位点的位置
    
    for idx in range(len(sequences)):
        # 检查 12loc 中的标签
        y12 = y_12class[idx]
        if y12[TARGET_MODEL_INDEX] == 1:  # 模型索引 9 对应 m6A
            m6a_sample_indices.append(idx)
            
            # 查找该样本中所有 m6A 修饰的位置
            full_label = full_labels[idx]
            m6a_pos_in_sample = []
            for pos, label_id in enumerate(full_label):
                if label_id == TARGET_MOD_INDEX:
                    m6a_pos_in_sample.append(pos)
            m6a_positions.append(m6a_pos_in_sample)
    
    m6a_count = len(m6a_sample_indices)
    print(f"  ✓ 找到 {m6a_count} 个 m6A 样本")
    
    if m6a_count == 0:
        print("\n⚠ 警告: 未找到任何 m6A 样本")
        print("  可能原因:")
        print("    - 数据集路径错误")
        print("    - 标签映射不匹配")
        print("    - 数据集确实不包含 m6A 样本")
        return
    
    # 第三步：空间与序列一致性检查
    print("\n[第三步] 检查 m6A 样本的序列一致性...")
    print(f"  目标中心碱基: '{TARGET_NUCLEOTIDE}'")
    print(f"  目标 One-hot 编码: {TARGET_ONE_HOT}")
    
    # 统计信息
    total_m6a_sites = 0
    invalid_sites = []  # 存储无效位点信息 (sample_idx, position, actual_nuc)
    nucleotide_distribution = defaultdict(int)
    
    # 检查每个 m6A 样本
    for i, sample_idx in enumerate(m6a_sample_indices):
        sequence_bytes = sequences[sample_idx].copy()
        
        # 转换为字符串
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        
        # 获取该样本的 m6A 位点
        positions = m6a_positions[i]
        total_m6a_sites += len(positions)
        
        # 检查每个 m6A 位点
        for pos in positions:
            if pos >= len(sequence_str):
                invalid_sites.append({
                    'sample_idx': sample_idx,
                    'position': pos,
                    'error': 'position out of bounds'
                })
                continue
            
            actual_nuc = sequence_str[pos]
            nucleotide_distribution[actual_nuc] += 1
            
            # 检查是否为目标碱基
            if actual_nuc != TARGET_NUCLEOTIDE:
                invalid_sites.append({
                    'sample_idx': sample_idx,
                    'position': pos,
                    'expected': TARGET_NUCLEOTIDE,
                    'actual': actual_nuc,
                    'one_hot': ONE_HOT_MAPPING.get(actual_nuc, [0., 0., 0., 0.])
                })
    
    # 第四步：生成报告
    print("\n" + "=" * 80)
    print("检查报告")
    print("=" * 80)
    
    print(f"\n[1] 样本统计")
    print(f"  m6A 样本总数: {m6a_count}")
    print(f"  m6A 修饰位点总数: {total_m6a_sites}")
    print(f"  平均每个样本的 m6A 位点数: {total_m6a_sites / m6a_count:.2f}")
    
    print(f"\n[2] 碱基分布")
    print(f"  {'碱基':<10} {'数量':<10} {'占比':<10}")
    print(f"  {'-'*30}")
    for nuc in sorted(nucleotide_distribution.keys()):
        count = nucleotide_distribution[nuc]
        percentage = (count / total_m6a_sites) * 100 if total_m6a_sites > 0 else 0
        expected_mark = " ✓" if nuc == TARGET_NUCLEOTIDE else " ✗"
        print(f"  {nuc:<10} {count:<10} {percentage:>6.2f}%{expected_mark}")
    
    print(f"\n[3] 纯净度校验")
    valid_count = nucleotide_distribution.get(TARGET_NUCLEOTIDE, 0)
    invalid_count = total_m6a_sites - valid_count
    purity = (valid_count / total_m6a_sites) * 100 if total_m6a_sites > 0 else 0
    
    print(f"  有效 m6A 位点 (中心碱基 = 'A'): {valid_count}")
    print(f"  无效 m6A 位点 (中心碱基 ≠ 'A'): {invalid_count}")
    print(f"  数据纯净度: {purity:.2f}%")
    
    if invalid_count > 0:
        print(f"\n[4] 异常定位")
        print(f"  发现 {len(invalid_sites)} 个无效位点:")
        print(f"\n  前 20 个异常位点详情:")
        print(f"  {'样本索引':<15} {'位点位置':<15} {'预期':<10} {'实际':<10} {'One-hot'}")
        print(f"  {'-'*80}")
        
        for i, site in enumerate(invalid_sites[:20]):
            if 'error' in site:
                print(f"  {site['sample_idx']:<15} {site['position']:<15} {'-':<10} {'-':<10} {site['error']}")
            else:
                one_hot_str = str(site['one_hot'])
                print(f"  {site['sample_idx']:<15} {site['position']:<15} {site['expected']:<10} {site['actual']:<10} {one_hot_str}")
        
        if len(invalid_sites) > 20:
            print(f"  ... 还有 {len(invalid_sites) - 20} 个异常位点未显示")
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    if invalid_count == 0:
        print("  [PASS] 所有 m6A 位点的中心碱基均为 'A'，数据完整无误")
    elif invalid_count <= 10:
        print(f"  [WARN] 发现少量 ({invalid_count}) 异常位点，建议检查")
    else:
        print(f"  [FAIL] 发现大量 ({invalid_count}) 异常位点，需要清理数据")
    
    print("=" * 80)
    
    # 保存详细报告到文件
    output_file = 'm6a_integrity_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("m6A 数据完整性检查报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n数据路径: {data_dir}\n")
        f.write(f"\n目标参数:\n")
        f.write(f"  原始标签索引: {TARGET_MOD_INDEX}\n")
        f.write(f"  模型索引: {TARGET_MODEL_INDEX}\n")
        f.write(f"  目标中心碱基: '{TARGET_NUCLEOTIDE}'\n")
        f.write(f"  目标 One-hot 编码: {TARGET_ONE_HOT}\n")
        
        f.write(f"\n[1] 样本统计\n")
        f.write(f"  m6A 样本总数: {m6a_count}\n")
        f.write(f"  m6A 修饰位点总数: {total_m6a_sites}\n")
        f.write(f"  平均每个样本的 m6A 位点数: {total_m6a_sites / m6a_count:.2f}\n")
        
        f.write(f"\n[2] 碱基分布\n")
        f.write(f"  {'碱基':<10} {'数量':<10} {'占比':<10}\n")
        f.write(f"  {'-'*30}\n")
        for nuc in sorted(nucleotide_distribution.keys()):
            count = nucleotide_distribution[nuc]
            percentage = (count / total_m6a_sites) * 100 if total_m6a_sites > 0 else 0
            f.write(f"  {nuc:<10} {count:<10} {percentage:>6.2f}%\n")
        
        f.write(f"\n[3] 纯净度校验\n")
        f.write(f"  有效 m6A 位点 (中心碱基 = 'A'): {valid_count}\n")
        f.write(f"  无效 m6A 位点 (中心碱基 ≠ 'A'): {invalid_count}\n")
        f.write(f"  数据纯净度: {purity:.2f}%\n")
        
        if invalid_count > 0:
            f.write(f"\n[4] 所有异常位点详情\n")
            f.write(f"  {'样本索引':<15} {'位点位置':<15} {'预期':<10} {'实际':<10} {'One-hot'}\n")
            f.write(f"  {'-'*80}\n")
            
            for site in invalid_sites:
                if 'error' in site:
                    f.write(f"  {site['sample_idx']:<15} {site['position']:<15} {'-':<10} {'-':<10} {site['error']}\n")
                else:
                    one_hot_str = str(site['one_hot'])
                    f.write(f"  {site['sample_idx']:<15} {site['position']:<15} {site['expected']:<10} {site['actual']:<10} {one_hot_str}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("结论\n")
        f.write("=" * 80 + "\n")
        
        if invalid_count == 0:
            f.write("  [PASS] 所有 m6A 位点的中心碱基均为 'A'，数据完整无误\n")
        elif invalid_count <= 10:
            f.write(f"  [WARN] 发现少量 ({invalid_count}) 异常位点，建议检查\n")
        else:
            f.write(f"  [FAIL] 发现大量 ({invalid_count}) 异常位点，需要清理数据\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存到: {output_file}")
    
    return {
        'total_samples': m6a_count,
        'total_sites': total_m6a_sites,
        'valid_sites': valid_count,
        'invalid_sites': invalid_count,
        'purity': purity,
        'invalid_site_details': invalid_sites
    }


if __name__ == "__main__":
    # 执行检查
    result = check_m6a_integrity('npy/3gen')