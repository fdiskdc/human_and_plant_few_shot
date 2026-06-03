"""
utils/audit_12loc_structure.py - 12类标签矩阵深度审计 / 12-class Label Matrix Deep Audit

对 npy/3gen/12loc.npy 进行 12 类标签矩阵的深度审计:共现矩阵、列激活统计。
Deep audit of 12-class label matrix from npy/3gen/12loc.npy: co-occurrence matrix, column activation statistics.

功能模块 / Modules:
- 共现矩阵 / Co-occurrence matrix
- 列激活统计 / Column activation stats
- 类别相关性 / Class correlation
- main: 主入口 / Main entry point

输入 / Inputs:
- npy/3gen/12loc.npy: 12 类多热标签 / 12-class multi-hot labels

输出 / Outputs:
- 终端详细报告 / Detailed terminal report
- 可选图表 (PNG/PDF) / Optional plots

数据流 / Data Flow:
1. 加载 12loc 矩阵 / Load 12loc matrix
2. 计算共现 / Compute co-occurrence
3. 列激活统计 / Column activation stats
4. 输出报告 / Output report

相关文件 / Related Files:
- 调用 / Calls: numpy
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python -m utils.audit_12loc_structure

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
from collections import defaultdict, Counter
import random

# 12类标签映射表（与 dataset/gen3.py 一致）
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

# 修饰名称
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}

# 反向映射：修饰名称 -> 模型索引
MOD_TO_INDEX = {v: k for k, v in MOD_NAMES.items()}


def audit_12loc_structure(data_dir='npy/3gen'):
    """
    深度审计 12loc.npy 的内部结构
    
    Args:
        data_dir: 数据目录路径
    """
    print("=" * 80)
    print("12loc 标签矩阵深度审计")
    print("=" * 80)
    print(f"\n数据路径: {data_dir}")
    
    # 加载数据
    print("\n[加载] 正在加载数据文件...")
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
    
    total_samples = y_12class.shape[0]
    print(f"\n  总样本数: {total_samples}")
    print(f"  标签维度: {y_12class.shape[1]}")
    
    # ========================================================================
    # A. 修饰共存性检查 (Co-occurrence Analysis)
    # ========================================================================
    print("\n" + "=" * 80)
    print("A. 修饰共存性检查 (Co-occurrence Analysis)")
    print("=" * 80)
    
    # 统计每行的非零元素个数（即每个样本有多少种修饰）
    row_sums = np.sum(y_12class > 0, axis=1)
    
    # 统计修饰数量分布
    modification_count_distribution = Counter(row_sums)
    
    print("\n样本修饰数量分布:")
    print(f"  {'修饰数量':<12} {'样本数':<15} {'占比':<15}")
    print(f"  {'-'*45}")
    
    for count in sorted(modification_count_distribution.keys()):
        num_samples = modification_count_distribution[count]
        percentage = (num_samples / total_samples) * 100
        label = "多修饰" if count > 1 else ("无修饰" if count == 0 else f"{count}种修饰")
        print(f"  {label:<12} {num_samples:<15} {percentage:>6.2f}%")
    
    # 统计单修饰和多修饰样本
    single_mod_samples = np.sum(row_sums == 1)
    multi_mod_samples = np.sum(row_sums > 1)
    no_mod_samples = np.sum(row_sums == 0)
    
    print(f"\n共存性统计:")
    print(f"  无修饰样本 (sum=0): {no_mod_samples} ({no_mod_samples/total_samples*100:.2f}%)")
    print(f"  单修饰样本 (sum=1): {single_mod_samples} ({single_mod_samples/total_samples*100:.2f}%)")
    print(f"  多修饰样本 (sum>1): {multi_mod_samples} ({multi_mod_samples/total_samples*100:.2f}%)")
    
    # 分析多修饰样本的修饰组合
    if multi_mod_samples > 0:
        print(f"\n多修饰组合分析 (Top 20):")
        print(f"  {'修饰组合':<50} {'样本数':<10}")
        print(f"  {'-'*65}")
        
        mod_combinations = defaultdict(int)
        multi_mod_indices = np.where(row_sums > 1)[0]
        
        for idx in multi_mod_indices:
            mods = []
            for col in range(12):
                if y_12class[idx, col] > 0:
                    mods.append(MOD_NAMES[col])
            mod_combo = ' + '.join(sorted(mods))
            mod_combinations[mod_combo] += 1
        
        # 按出现次数排序，显示前20
        sorted_combos = sorted(mod_combinations.items(), key=lambda x: x[1], reverse=True)
        for combo, count in sorted_combos[:20]:
            print(f"  {combo:<50} {count:<10}")
        
        if len(sorted_combos) > 20:
            print(f"  ... 还有 {len(sorted_combos) - 20} 种组合未显示")
    
    # ========================================================================
    # B. 标签列激活统计 (Column-wise Activation)
    # ========================================================================
    print("\n" + "=" * 80)
    print("B. 标签列激活统计 (Column-wise Activation)")
    print("=" * 80)
    
    print("\n各列（修饰类型）激活统计:")
    print(f"  {'索引':<6} {'修饰名称':<12} {'激活样本数':<15} {'占比':<12} {'稀疏度':<12}")
    print(f"  {'-'*70}")
    
    column_stats = {}
    for col in range(12):
        activated = np.sum(y_12class[:, col] > 0)
        percentage = (activated / total_samples) * 100
        sparsity = 100 - percentage
        column_stats[col] = {
            'name': MOD_NAMES[col],
            'activated': int(activated),
            'percentage': percentage,
            'sparsity': sparsity
        }
        print(f"  {col:<6} {MOD_NAMES[col]:<12} {activated:<15} {percentage:>6.2f}%    {sparsity:>6.2f}%")
    
    # 检查是否有全零列
    zero_columns = [col for col in range(12) if column_stats[col]['activated'] == 0]
    if zero_columns:
        print(f"\n⚠ 警告: 以下列（修饰类型）在数据中完全缺失: {zero_columns}")
    else:
        print(f"\n✓ 所有 12 类修饰都有数据")
    
    # 特别关注 m6A (列 9)
    print(f"\n重点检查: m6A 修饰 (列 9)")
    print(f"  激活样本数: {column_stats[9]['activated']}")
    print(f"  占比: {column_stats[9]['percentage']:.2f}%")
    
    # ========================================================================
    # C. 值的有效性校验 (Value Range)
    # ========================================================================
    print("\n" + "=" * 80)
    print("C. 值的有效性校验 (Value Range)")
    print("=" * 80)
    
    # 检查矩阵中的数值范围
    min_val = np.min(y_12class)
    max_val = np.max(y_12class)
    unique_vals = np.unique(y_12class)
    
    print(f"\n数值统计:")
    print(f"  最小值: {min_val}")
    print(f"  最大值: {max_val}")
    print(f"  唯一值: {unique_vals}")
    
    # 分析数值含义
    print(f"\n数值含义分析:")
    if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
        print(f"  ✓ 这是一个 Binary (0/1) 矩阵")
        print(f"    - 0: 该样本不存在该修饰类型")
        print(f"    - 1: 该样本存在该修饰类型")
    elif 0 in unique_vals and 1 in unique_vals and len(unique_vals) > 2:
        print(f"  ⚠ 这可能是一个 Count 矩阵（0 和 >1 的值）")
        print(f"    - 0: 该样本不存在该修饰类型")
        print(f"    - >1: 该样本存在多个该修饰类型的位点")
        print(f"  唯一值分布: {dict(Counter(y_12class.flatten()))}")
    else:
        print(f"  ⚠ 未知矩阵类型")
        print(f"  唯一值分布: {dict(Counter(y_12class.flatten()))}")
    
    # 检查是否存在 >1 的值
    values_gt_1 = y_12class[y_12class > 1]
    if len(values_gt_1) > 0:
        print(f"\n⚠ 发现 {len(values_gt_1)} 个值 > 1 的元素")
        print(f"  这些值可能代表该区域内有多个同类修饰位点")
        print(f"  值分布: {dict(Counter(values_gt_1.flatten()))}")
    
    # ========================================================================
    # D. 与 1001loc 的联动校验 (Cross-reference)
    # ========================================================================
    print("\n" + "=" * 80)
    print("D. 与 1001loc 的联动校验 (Cross-reference)")
    print("=" * 80)
    
    # 随机抽取样本进行检查
    sample_size = min(1000, total_samples)
    random_indices = random.sample(range(total_samples), sample_size)
    
    print(f"\n随机抽取 {sample_size} 个样本进行一致性检查...")
    
    # 检查 1001loc 中的 m6A (label 10) 与 12loc 中的 m6A (column 9) 的一致性
    m6a_label_10_count = 0
    m6a_col_9_count = 0
    consistent_count = 0
    inconsistent_12loc_0 = 0  # 1001loc有m6A但12loc第9列为0
    inconsistent_12loc_1 = 0  # 1001loc无m6A但12loc第9列为1
    
    m6a_mod10_samples = []  # 存储1001loc中有m6A的样本索引
    m6a_col9_samples = []   # 存储12loc第9列为1的样本索引
    
    for idx in random_indices:
        full_label = full_labels[idx]
        y12 = y_12class[idx]
        
        # 检查 1001loc 中是否有 m6A (label 10)
        has_m6a_in_1001loc = np.any(full_label == 10)
        if has_m6a_in_1001loc:
            m6a_label_10_count += 1
            m6a_mod10_samples.append(idx)
        
        # 检查 12loc 第9列是否有 m6A
        has_m6a_in_12loc = y12[9] > 0
        if has_m6a_in_12loc:
            m6a_col_9_count += 1
            m6a_col9_samples.append(idx)
        
        # 检查一致性
        if has_m6a_in_1001loc and has_m6a_in_12loc:
            consistent_count += 1
        elif has_m6a_in_1001loc and not has_m6a_in_12loc:
            inconsistent_12loc_0 += 1
        elif not has_m6a_in_1001loc and has_m6a_in_12loc:
            inconsistent_12loc_1 += 1
    
    print(f"\n在随机抽取的 {sample_size} 个样本中:")
    print(f"  1001loc 中有 m6A (label=10) 的样本: {m6a_label_10_count} ({m6a_label_10_count/sample_size*100:.2f}%)")
    print(f"  12loc 第9列有 m6A 的样本: {m6a_col_9_count} ({m6a_col_9_count/sample_size*100:.2f}%)")
    print(f"\n一致性分析:")
    print(f"  ✓ 一致的样本: {consistent_count} ({consistent_count/sample_size*100:.2f}%)")
    print(f"  ✗ 不一致 - 1001loc有m6A但12loc第9列为0: {inconsistent_12loc_0} ({inconsistent_12loc_0/sample_size*100:.2f}%)")
    print(f"  ✗ 不一致 - 1001loc无m6A但12loc第9列为1: {inconsistent_12loc_1} ({inconsistent_12loc_1/sample_size*100:.2f}%)")
    
    # 计算重合率
    if m6a_label_10_count > 0 or m6a_col_9_count > 0:
        # Jaccard相似度
        intersection = len(set(m6a_mod10_samples) & set(m6a_col9_samples))
        union = len(set(m6a_mod10_samples) | set(m6a_col9_samples))
        jaccard = intersection / union if union > 0 else 0
        
        print(f"\n重合率指标:")
        print(f"  交集大小: {intersection}")
        print(f"  并集大小: {union}")
        print(f"  Jaccard 相似度: {jaccard:.4f}")
        
        if m6a_label_10_count > 0:
            recall = intersection / m6a_label_10_count
            print(f"  召回率 (1001loc -> 12loc): {recall:.4f}")
        
        if m6a_col_9_count > 0:
            precision = intersection / m6a_col_9_count
            print(f"  精确率 (12loc -> 1001loc): {precision:.4f}")
    
    # ========================================================================
    # 生成数据透视表
    # ========================================================================
    print("\n" + "=" * 80)
    print("数据透视表汇总")
    print("=" * 80)
    
    print(f"\n[1] 各列激活数表:")
    print(f"  {'索引':<6} {'修饰名称':<12} {'激活样本数':<15} {'占比':<12}")
    print(f"  {'-'*50}")
    for col in range(12):
        print(f"  {col:<6} {MOD_NAMES[col]:<12} {column_stats[col]['activated']:<15} {column_stats[col]['percentage']:>6.2f}%")
    
    print(f"\n[2] 多标签冲突表:")
    print(f"  单修饰样本: {single_mod_samples} ({single_mod_samples/total_samples*100:.2f}%)")
    print(f"  多修饰样本: {multi_mod_samples} ({multi_mod_samples/total_samples*100:.2f}%)")
    if multi_mod_samples > 0:
        print(f"\n  Top 10 多修饰组合:")
        for combo, count in sorted_combos[:10]:
            print(f"    {combo:<50} {count}")
    
    print(f"\n[3] 1001loc 与 12loc 对比一致性 (基于 {sample_size} 个随机样本):")
    print(f"  1001loc m6A (label=10) 样本数: {m6a_label_10_count}")
    print(f"  12loc m6A (column=9) 样本数: {m6a_col_9_count}")
    print(f"  一致样本数: {consistent_count} ({consistent_count/sample_size*100:.2f}%)")
    if m6a_label_10_count > 0 or m6a_col_9_count > 0:
        intersection = len(set(m6a_mod10_samples) & set(m6a_col9_samples))
        union = len(set(m6a_mod10_samples) | set(m6a_col9_samples))
        jaccard = intersection / union if union > 0 else 0
        print(f"  Jaccard 相似度: {jaccard:.4f}")
    
    # ========================================================================
    # 保存详细报告
    # ========================================================================
    output_file = '12loc_audit_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("12loc 标签矩阵深度审计报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n数据路径: {data_dir}\n")
        f.write(f"总样本数: {total_samples}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("A. 修饰共存性检查 (Co-occurrence Analysis)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n样本修饰数量分布:\n")
        for count in sorted(modification_count_distribution.keys()):
            num_samples = modification_count_distribution[count]
            percentage = (num_samples / total_samples) * 100
            label = "多修饰" if count > 1 else ("无修饰" if count == 0 else f"{count}种修饰")
            f.write(f"  {label}: {num_samples} ({percentage:.2f}%)\n")
        
        f.write(f"\n共存性统计:\n")
        f.write(f"  无修饰样本: {no_mod_samples} ({no_mod_samples/total_samples*100:.2f}%)\n")
        f.write(f"  单修饰样本: {single_mod_samples} ({single_mod_samples/total_samples*100:.2f}%)\n")
        f.write(f"  多修饰样本: {multi_mod_samples} ({multi_mod_samples/total_samples*100:.2f}%)\n")
        
        if multi_mod_samples > 0:
            f.write(f"\n多修饰组合 (Top 20):\n")
            for combo, count in sorted_combos[:20]:
                f.write(f"  {combo}: {count}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("B. 标签列激活统计 (Column-wise Activation)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n各列激活统计:\n")
        for col in range(12):
            f.write(f"  列 {col} ({MOD_NAMES[col]}): {column_stats[col]['activated']} 样本 ({column_stats[col]['percentage']:.2f}%)\n")
        
        if zero_columns:
            f.write(f"\n⚠ 以下列完全缺失: {zero_columns}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("C. 值的有效性校验 (Value Range)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n数值统计:\n")
        f.write(f"  最小值: {min_val}\n")
        f.write(f"  最大值: {max_val}\n")
        f.write(f"  唯一值: {unique_vals}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("D. 与 1001loc 的联动校验 (Cross-reference)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n基于 {sample_size} 个随机样本:\n")
        f.write(f"  1001loc m6A (label=10): {m6a_label_10_count} 样本\n")
        f.write(f"  12loc m6A (column=9): {m6a_col_9_count} 样本\n")
        f.write(f"  一致样本: {consistent_count} ({consistent_count/sample_size*100:.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("数据透视表汇总\n")
        f.write("=" * 80 + "\n")
        
        f.write("\n[1] 各列激活数表:\n")
        for col in range(12):
            f.write(f"  列 {col} ({MOD_NAMES[col]}): {column_stats[col]['activated']} ({column_stats[col]['percentage']:.2f}%)\n")
        
        f.write("\n[2] 多标签冲突表:\n")
        f.write(f"  单修饰样本: {single_mod_samples} ({single_mod_samples/total_samples*100:.2f}%)\n")
        f.write(f"  多修饰样本: {multi_mod_samples} ({multi_mod_samples/total_samples*100:.2f}%)\n")
        
        f.write("\n[3] 1001loc 与 12loc 对比一致性:\n")
        f.write(f"  1001loc m6A: {m6a_label_10_count}\n")
        f.write(f"  12loc m6A: {m6a_col_9_count}\n")
        f.write(f"  一致率: {consistent_count/sample_size*100:.2f}%\n")
        if m6a_label_10_count > 0 or m6a_col_9_count > 0:
            f.write(f"  Jaccard 相似度: {jaccard:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n详细审计报告已保存到: {output_file}")
    
    return {
        'total_samples': total_samples,
        'column_stats': column_stats,
        'modification_count_distribution': modification_count_distribution,
        'multi_mod_samples': multi_mod_samples,
        'cross_reference': {
            'sample_size': sample_size,
            'm6a_label_10_count': m6a_label_10_count,
            'm6a_col_9_count': m6a_col_9_count,
            'consistent_count': consistent_count,
            'jaccard': jaccard if m6a_label_10_count > 0 or m6a_col_9_count > 0 else 0
        }
    }


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 执行审计
    result = audit_12loc_structure('npy/3gen')