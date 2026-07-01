#!/usr/bin/env python3
"""
cal_mean_median_mode.py - 12类修饰位点计数统计 (mean/median/mode) / Modification Site Count Statistics

统计每类 RNA 修饰在每个序列中的位点数量中位数、平均数和众数,使用 1001loc 数据。
对于 m6A: 统计所有有 m6A 修饰的序列,每个序列有多少个 m6A 位点,然后计算这些数量的平均值、中位数和众数。
Statistics of mean, median, mode of modification site counts per class per sequence using 1001loc data.

功能模块 / Modules:
- 12 类位点数统计 / 12-class site count statistics
- mean/median/mode 计算 / mean/median/mode computation
- main: 主入口 / Main entry point

输入 / Inputs:
- human3/1001loc.npy: 1001 位点级标签 / 1001 site-level labels
- LABEL_MAPPING from dataset.human

输出 / Outputs:
- 终端统计输出 / Terminal statistics output
- att_fig/mean_median_mode_*.png: 可选图表 / Optional figures

数据流 / Data Flow:
1. 加载位点标签 / Load site labels
2. 统计每序列每类位点数 / Count sites per class per sequence
3. 计算 mean/median/mode / Compute mean/median/mode
4. 输出结果 / Output results

相关文件 / Related Files:
- 调用 / Calls: numpy, pandas, collections.Counter
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python cal_mean_median_mode.py

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import pandas as pd
import os
from collections import Counter


# 12类标签映射表 (与 human.py 中的 LABEL_MAPPING 一致)
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

# 12类修饰名称
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    3: 'Gm',     4: 'Tm',     5: 'Y',
    6: 'ac4C',   7: 'm1A',    8: 'm5C',
    9: 'm6A',    10: 'm6Am',  11: 'm7G'
}


def get_human3_dir():
    """获取human3数据目录路径"""
    if os.path.exists('human3'):
        return 'human3'
    elif os.path.exists('../human3'):
        return '../human3'
    else:
        return '/home/dc/vscode/vscode20251230/human_and_plant/human3'


def load_data(human3_dir=None):
    """
    加载数据

    Args:
        human3_dir: human3目录路径，None则自动检测

    Returns:
        full_labels: 1001loc数据，形状为(N, 1001)
    """
    if human3_dir is None:
        human3_dir = get_human3_dir()

    print(f"加载数据: {human3_dir}")

    full_labels = np.load(f'{human3_dir}/1001loc.npy', mmap_mode='r')

    print(f"  总样本数: {len(full_labels)}")
    print(f"  1001loc形状: {full_labels.shape}")

    return full_labels


def calculate_statistics_per_class(full_labels, max_samples=None):
    """
    计算每个class的修饰位点数量统计（平均数、中位数、众数）

    Args:
        full_labels: 1001loc数据，形状为(N, 1001)
        max_samples: 最大处理样本数，None表示全部处理

    Returns:
        dict: 每个class的统计信息
    """
    num_samples = len(full_labels)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    print(f"\n开始统计 {num_samples} 个序列...")

    # 存储每个class的所有序列的修饰位点数量
    class_counts = {class_name: [] for class_name in MOD_NAMES.values()}

    for seq_idx in range(num_samples):
        if (seq_idx + 1) % 10000 == 0 or seq_idx == 0:
            print(f"  处理进度: {seq_idx + 1}/{num_samples}")

        full_label = full_labels[seq_idx]

        for class_idx, class_name in MOD_NAMES.items():
            mod_index = class_idx + 1
            count = np.sum(full_label == mod_index)
            if count > 0:
                class_counts[class_name].append(count)

    # 计算每个class的统计信息
    results = {}

    for class_name, counts in class_counts.items():
        if len(counts) == 0:
            # 没有该修饰的序列
            results[class_name] = {
                'class': class_name,
                'seq_count': 0,
                'total_sites': 0,
                'mean': np.nan,
                'median': np.nan,
                'mode': np.nan,
                'mode_count': 0,
                'min': np.nan,
                'max': np.nan,
                'std': np.nan,
                'distribution': {}
            }
        else:
            counts_array = np.array(counts)

            # 计算众数
            counter = Counter(counts)
            mode_value, mode_freq = counter.most_common(1)[0]

            # 计算分布
            total_seqs = len(counts)
            distribution = {
                k: round(v / total_seqs * 100, 2) for k, v in sorted(counter.items())
            }

            results[class_name] = {
                'class': class_name,
                'seq_count': len(counts),
                'total_sites': int(np.sum(counts_array)),
                'mean': round(np.mean(counts_array), 2),
                'median': round(float(np.median(counts_array)), 2),
                'mode': mode_value,
                'mode_percent': round(mode_freq / total_seqs * 100, 2),
                'min': int(np.min(counts_array)),
                'max': int(np.max(counts_array)),
                'std': round(float(np.std(counts_array)), 2),
                'distribution': distribution
            }

    print(f"  统计完成！")
    return results


def print_summary_table(results):
    """
    打印汇总统计表格

    Args:
        results: 每个class的统计信息字典
    """
    print("\n" + "=" * 120)
    print("每个Class的修饰位点数量统计（每个序列中该class修饰位点的数量）")
    print("=" * 120)

    # 构建表格数据
    table_data = []
    for class_name in MOD_NAMES.values():
        r = results[class_name]
        table_data.append({
            'Class': class_name,
            '序列数': r['seq_count'],
            '修饰位点数': r['total_sites'],
            '平均数': r['mean'],
            '中位数': r['median'],
            '众数': r['mode'],
            '众数占比(%)': r['mode_percent'],
            '最小值': r['min'],
            '最大值': r['max'],
            '标准差': r['std']
        })

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print("=" * 120)


def print_class_distribution(results, class_name):
    """
    打印某个class的修饰位点数量分布

    Args:
        results: 每个class的统计信息字典
        class_name: 类名
    """
    print(f"\n{'=' * 80}")
    print(f"{class_name} 类修饰位点数量分布")
    print("=" * 80)

    r = results[class_name]
    if r['seq_count'] == 0:
        print(f"  没有找到 {class_name} 类的修饰数据")
        return

    print(f"统计信息:")
    print(f"  含该修饰的序列数: {r['seq_count']}")
    print(f"  平均每个序列修饰位点数: {r['mean']}")
    print(f"  中位数: {r['median']}")
    print(f"  众数: {r['mode']} (占比 {r['mode_percent']}%)")
    print(f"  范围: {r['min']} ~ {r['max']}")
    print(f"  标准差: {r['std']}")

    print(f"\n位点数量分布:")
    print(f"{'位点数':<10} {'序列数':<15} {'占比(%)':<15} {'累计占比(%)':<15}")
    print("-" * 80)

    cumsum = 0
    for count, percent in sorted(r['distribution'].items()):
        cumsum += percent
        count_num = int(r['seq_count'] * percent / 100)
        print(f"{count:<10} {count_num:<15} {percent:<15.2f} {cumsum:<15.2f}")

    print("=" * 80)


def save_to_csv(results, output_path='statistics_summary.csv'):
    """
    保存统计结果到CSV文件

    Args:
        results: 每个class的统计信息字典
        output_path: 输出文件路径
    """
    # 保存汇总表
    table_data = []
    for class_name in MOD_NAMES.values():
        r = results[class_name]
        table_data.append({
            'Class': class_name,
            '序列数': r['seq_count'],
            '修饰位点数': r['total_sites'],
            '平均数': r['mean'],
            '中位数': r['median'],
            '众数': r['mode'],
            '众数占比(%)': r['mode_percent'],
            '最小值': r['min'],
            '最大值': r['max'],
            '标准差': r['std']
        })

    df = pd.DataFrame(table_data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n统计结果已保存到: {output_path}")

    # 保存详细分布数据
    dist_output_path = output_path.replace('.csv', '_distribution.csv')
    dist_data = []

    for class_name in MOD_NAMES.values():
        r = results[class_name]
        for count, percent in r['distribution'].items():
            count_num = int(r['seq_count'] * percent / 100)
            dist_data.append({
                'Class': class_name,
                '位点数': count,
                '序列数': count_num,
                '占比(%)': percent
            })

    dist_df = pd.DataFrame(dist_data)
    dist_df.to_csv(dist_output_path, index=False, encoding='utf-8-sig')
    print(f"分布数据已保存到: {dist_output_path}")


def main():
    """主函数"""
    print("=" * 120)
    print("RNA修饰位点数量统计工具 - 统计每个序列中每个class修饰位点的数量")
    print("=" * 120)

    # 加载数据
    human3_dir = get_human3_dir()
    full_labels = load_data(human3_dir)

    # 计算统计信息
    results = calculate_statistics_per_class(full_labels)

    # 打印汇总表格
    print_summary_table(results)

    # 打印每个class的详细分布
    for class_name in MOD_NAMES.values():
        print_class_distribution(results, class_name)

    # 保存到CSV
    output_path = os.path.join(os.path.dirname(__file__), 'statistics_summary.csv')
    save_to_csv(results, output_path)

    print("\n" + "=" * 120)
    print("统计完成！")
    print("=" * 120)


if __name__ == "__main__":
    main()
