"""
count_modification_sites.py - 统计Human3数据集各类RNA修饰位点数量 / Count RNA Modification Sites in Human3 Dataset

统计npy/human3数据集中12类RNA修饰的位点分布情况，支持位点级（1001loc.npy）和样本级（12loc.npy）
两种统计维度。输出每种修饰的出现次数、占比，以及整体数据集的统计摘要。
Counts the distribution of 12 RNA modification types in the npy/human3 dataset, supporting both
site-level (1001loc.npy) and sample-level (12loc.npy) statistics. Outputs occurrence counts,
percentages, and overall dataset summary.

功能模块 / Modules:
- load_human3_data: 加载human3目录下的npy数据文件 / Load npy files from human3 directory
- count_site_level_modifications: 统计1001个位置中每种修饰的出现次数 / Count modification occurrences across 1001 positions
- count_sample_level_modifications: 统计每种修饰的阳性样本数 / Count positive samples per modification type
- print_statistics: 格式化输出统计结果表格 / Formatted output of statistics table
- main: 主函数，执行完整统计流程 / Main function executing full statistics pipeline

输入 / Inputs:
- npy/human3/1001loc.npy: NumPy int8数组 (N, 1001) - 位点级修饰标签 (0=无修饰, 1-12=修饰类型) / Site-level modification labels (0=none, 1-12=modification type)
- npy/human3/12loc.npy: NumPy int8数组 (N, 12) - 12类多标签二值向量 (0/1) / 12-class multi-label binary vectors

输出 / Outputs:
- 终端输出 / Console output: 每种修饰的位点数、样本数、占比表格 / Per-modification site count, sample count, percentage table
- 可选CSV / Optional CSV: statistics_results.csv - 结构化统计结果 / Structured statistics results

数据流 / Data Flow:
1. 加载npy数据 / Load npy data: 使用mmap_mode内存映射加载大型数组 / Memory-map large arrays with mmap_mode
2. 位点级统计 / Site-level counting: 遍历1001loc.npy统计每种修饰(1-12)出现次数 / Iterate 1001loc.npy counting each modification (1-12) occurrences
3. 样本级统计 / Sample-level counting: 统计12loc.npy中每列的阳性样本数 / Count positive samples per column in 12loc.npy
4. 结果汇总 / Result summary: 计算占比、总位点数、平均每样本位点数 / Calculate percentages, total sites, average sites per sample
5. 格式化输出 / Formatted output: 打印表格并可选保存CSV / Print table and optionally save CSV

相关文件 / Related Files:
- 调用 / Calls: numpy, pandas (可选), os, argparse
- 被调用 / Called by: 手动执行或集成到分析流程 / Manual execution or integrated into analysis pipeline
- 数据来源 / Data source: dataset/human.py (Mer100Dataset类使用相同数据) / Same data used by Mer100Dataset class

使用示例 / Usage Example:
    # 基本统计 / Basic statistics
    python analysis/count_modification_sites.py

    # 指定数据目录 / Specify data directory
    python analysis/count_modification_sites.py --data-dir npy/human3

    # 保存CSV结果 / Save CSV results
    python analysis/count_modification_sites.py --output-csv analysis/modification_stats.csv

    # 仅显示位点级统计 / Show site-level statistics only
    python analysis/count_modification_sites.py --site-only

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-06
版本 / Version: 1.0
"""

import os
import argparse
import numpy as np
from typing import Dict, Tuple, Optional

# ============================================================================
# 常量定义 / Constants Definition
# ============================================================================

# 12类修饰名称映射 (标签值 1-12 -> 修饰名称)
# 与 dataset/human.py 中的 LABEL_MAPPING 一致
# 12-class modification name mapping (label value 1-12 -> modification name)
# Consistent with LABEL_MAPPING in dataset/human.py
MOD_NAMES = {
    1: 'Am',      2: 'Atol',    3: 'Cm',
    4: 'Gm',      5: 'Tm',      6: 'Y',
    7: 'ac4C',    8: 'm1A',     9: 'm5C',
    10: 'm6A',    11: 'm6Am',   12: 'm7G'
}

# 核苷酸组映射 (修饰类型 -> 核苷酸组)
# Nucleotide group mapping (modification type -> nucleotide group)
MOD_TO_NUCLEOTIDE = {
    1: 'A', 2: 'A',                    # Am, Atol -> A
    3: 'C',                            # Cm -> C
    4: 'G',                            # Gm -> G
    5: 'U', 6: 'U',                    # Tm, Y -> U
    7: 'C',                            # ac4C -> C
    8: 'A',                            # m1A -> A
    9: 'C',                            # m5C -> C
    10: 'A', 11: 'A',                  # m6A, m6Am -> A
    12: 'G'                            # m7G -> G
}


# ============================================================================
# 数据加载函数 / Data Loading Functions
# ============================================================================

def load_human3_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载human3目录下的npy数据文件 / Load npy files from human3 directory.

    使用内存映射模式加载大型数组，减少内存占用。
    Uses memory-mapped mode to load large arrays, reducing memory footprint.

    Args / 参数:
        data_dir (str): [中文] human3数据目录路径 / [English] path to human3 data directory.

    Returns / 返回值:
        Tuple[np.ndarray, np.ndarray]: [中文] (loc_1001, loc_12) 两个numpy数组 / [English] (loc_1001, loc_12) two numpy arrays.

    Raises / 异常:
        FileNotFoundError: [中文] 数据文件不存在 / [English] data file not found.
    """
    loc_1001_path = os.path.join(data_dir, '1001loc.npy')
    loc_12_path = os.path.join(data_dir, '12loc.npy')

    # 检查文件是否存在 / Check if files exist
    if not os.path.exists(loc_1001_path):
        raise FileNotFoundError(f"找不到文件 / File not found: {loc_1001_path}")
    if not os.path.exists(loc_12_path):
        raise FileNotFoundError(f"找不到文件 / File not found: {loc_12_path}")

    # 使用内存映射加载 / Load with memory mapping
    print(f"加载数据 / Loading data from: {data_dir}")
    loc_1001 = np.load(loc_1001_path, mmap_mode='r')
    loc_12 = np.load(loc_12_path, mmap_mode='r')

    print(f"  1001loc.npy 形状 / shape: {loc_1001.shape}")
    print(f"  12loc.npy 形状 / shape: {loc_12.shape}")

    return loc_1001, loc_12


# ============================================================================
# 统计函数 / Statistics Functions
# ============================================================================

def count_site_level_modifications(loc_1001: np.ndarray) -> Dict[int, int]:
    """
    统计1001个位置中每种修饰的出现次数 / Count modification occurrences across 1001 positions.

    遍历所有样本的所有位置，统计每种修饰类型(1-12)出现的总次数。
    Iterates through all positions of all samples, counting total occurrences of each modification type (1-12).

    Args / 参数:
        loc_1001 (np.ndarray): [中文] (N, 1001) 位点级标签数组，值为0-12 / [English] (N, 1001) site-level label array with values 0-12.

    Returns / 返回值:
        Dict[int, int]: [中文] {修饰标签: 出现次数} 字典 / [English] {modification_label: occurrence_count} dictionary.
    """
    # 初始化计数字典 (标签1-12) / Initialize count dictionary (labels 1-12)
    site_counts = {i: 0 for i in range(1, 13)}

    # 统计每种修饰的出现次数 / Count occurrences of each modification
    for mod_label in range(1, 13):
        site_counts[mod_label] = int(np.sum(loc_1001 == mod_label))

    return site_counts


def count_sample_level_modifications(loc_12: np.ndarray) -> Dict[int, int]:
    """
    统计每种修饰的阳性样本数 / Count positive samples per modification type.

    统计12loc.npy中每列的阳性样本数（值为1的样本数）。
    Counts positive samples (samples with value 1) per column in 12loc.npy.

    Args / 参数:
        loc_12 (np.ndarray): [中文] (N, 12) 12类多标签二值向量 / [English] (N, 12) 12-class multi-label binary vectors.

    Returns / 返回值:
        Dict[int, int]: [中文] {修饰索引(0-11): 阳性样本数} 字典 / [English] {modification_index(0-11): positive_sample_count} dictionary.
    """
    # 初始化计数字典 (索引0-11) / Initialize count dictionary (indices 0-11)
    sample_counts = {}

    # 统计每列的阳性样本数 / Count positive samples per column
    for col_idx in range(12):
        sample_counts[col_idx] = int(np.sum(loc_12[:, col_idx] == 1))

    return sample_counts


# ============================================================================
# 输出函数 / Output Functions
# ============================================================================

def print_statistics(
    site_counts: Dict[int, int],
    sample_counts: Dict[int, int],
    total_samples: int,
    total_positions: int
) -> None:
    """
    格式化输出统计结果表格 / Formatted output of statistics table.

    打印包含修饰名称、位点数、样本数、占比的完整统计表格。
    Prints complete statistics table with modification names, site counts, sample counts, and percentages.

    Args / 参数:
        site_counts (Dict[int, int]): [中文] 位点级统计结果 / [English] site-level statistics.
        sample_counts (Dict[int, int]): [中文] 样本级统计结果 / [English] sample-level statistics.
        total_samples (int): [中文] 总样本数 / [English] total number of samples.
        total_positions (int): [中文] 总位置数 (N * 1001) / [English] total positions (N * 1001).
    """
    # 表头 / Header
    print("\n" + "=" * 80)
    print("Human3 数据集 RNA 修饰位点统计 / Human3 Dataset RNA Modification Site Statistics")
    print("=" * 80)

    # 列标题 / Column headers
    header = f"{'标签':<6} {'修饰名称':<10} {'核苷酸':<8} {'位点数':<12} {'样本数':<12} {'位点占比':<10} {'样本占比':<10}"
    print(header)
    print("-" * 80)

    # 统计数据行 / Statistics data rows
    total_sites = sum(site_counts.values())
    total_positive_samples = sum(sample_counts.values())

    for mod_label in range(1, 13):
        mod_name = MOD_NAMES[mod_label]
        nucleotide = MOD_TO_NUCLEOTIDE[mod_label]
        site_count = site_counts[mod_label]
        sample_count = sample_counts[mod_label - 1]  # 样本计数使用0-11索引 / Sample count uses 0-11 index

        # 计算占比 / Calculate percentages
        site_pct = (site_count / total_positions * 100) if total_positions > 0 else 0
        sample_pct = (sample_count / total_samples * 100) if total_samples > 0 else 0

        print(f"{mod_label:<6} {mod_name:<10} {nucleotide:<8} {site_count:<12,} {sample_count:<12,} {site_pct:<10.4f} {sample_pct:<10.4f}")

    # 汇总行 / Summary row
    print("-" * 80)
    print(f"{'总计':<6} {'':<10} {'':<8} {total_sites:<12,} {total_positive_samples:<12,} {'100.0000':<10} {'100.0000':<10}")

    # 额外统计信息 / Additional statistics
    print("\n" + "=" * 80)
    print("数据集摘要 / Dataset Summary")
    print("=" * 80)
    print(f"总样本数 / Total samples: {total_samples:,}")
    print(f"总位置数 / Total positions: {total_positions:,} ({total_samples:,} × 1001)")
    print(f"修饰位点总数 / Total modification sites: {total_sites:,}")
    print(f"平均每样本修饰位点 / Avg sites per sample: {total_sites / total_samples:.2f}")
    print(f"平均每样本阳性修饰类型 / Avg positive types per sample: {total_positive_samples / total_samples:.2f}")
    print(f"无修饰位点数 / Unmodified positions: {total_positions - total_sites:,}")
    print(f"无修饰占比 / Unmodified percentage: {(total_positions - total_sites) / total_positions * 100:.4f}%")
    print("=" * 80)


def save_csv_results(
    site_counts: Dict[int, int],
    sample_counts: Dict[int, int],
    total_samples: int,
    total_positions: int,
    output_path: str
) -> None:
    """
    保存统计结果到CSV文件 / Save statistics to CSV file.

    将统计结果保存为结构化的CSV文件，便于后续分析。
    Saves statistics as structured CSV file for further analysis.

    Args / 参数:
        site_counts (Dict[int, int]): [中文] 位点级统计结果 / [English] site-level statistics.
        sample_counts (Dict[int, int]): [中文] 样本级统计结果 / [English] sample-level statistics.
        total_samples (int): [中文] 总样本数 / [English] total number of samples.
        total_positions (int): [中文] 总位置数 / [English] total positions.
        output_path (str): [中文] CSV输出路径 / [English] CSV output path.
    """
    import csv

    total_sites = sum(site_counts.values())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头 / Write header
        writer.writerow([
            'Label', 'Modification', 'Nucleotide',
            'Site_Count', 'Sample_Count',
            'Site_Percentage', 'Sample_Percentage'
        ])

        # 写入每种修饰的数据 / Write data for each modification
        for mod_label in range(1, 13):
            mod_name = MOD_NAMES[mod_label]
            nucleotide = MOD_TO_NUCLEOTIDE[mod_label]
            site_count = site_counts[mod_label]
            sample_count = sample_counts[mod_label - 1]

            site_pct = (site_count / total_positions * 100) if total_positions > 0 else 0
            sample_pct = (sample_count / total_samples * 100) if total_samples > 0 else 0

            writer.writerow([
                mod_label, mod_name, nucleotide,
                site_count, sample_count,
                f"{site_pct:.4f}", f"{sample_pct:.4f}"
            ])

        # 写入汇总行 / Write summary row
        writer.writerow([
            'Total', '', '',
            total_sites, sum(sample_counts.values()),
            '100.0000', '100.0000'
        ])

    print(f"\n结果已保存至 / Results saved to: {output_path}")


# ============================================================================
# 主函数 / Main Function
# ============================================================================

def main(
    data_dir: str = 'npy/human3',
    output_csv: Optional[str] = None,
    site_only: bool = False
) -> None:
    """
    主函数，执行完整统计流程 / Main function executing full statistics pipeline.

    协调数据加载、统计计算和结果输出的完整流程。
    Coordinates the complete pipeline of data loading, statistics calculation, and result output.

    Args / 参数:
        data_dir (str): [中文] 数据目录路径 / [English] data directory path. Defaults to 'npy/human3'.
        output_csv (Optional[str]): [中文] CSV输出路径（None则不保存） / [English] CSV output path (None to skip). Defaults to None.
        site_only (bool): [中文] 是否仅显示位点级统计 / [English] show site-level statistics only. Defaults to False.
    """
    # 加载数据 / Load data
    loc_1001, loc_12 = load_human3_data(data_dir)

    # 获取数据集维度 / Get dataset dimensions
    total_samples = loc_1001.shape[0]
    total_positions = total_samples * loc_1001.shape[1]

    # 位点级统计 / Site-level statistics
    print("\n正在统计位点级修饰分布... / Counting site-level modification distribution...")
    site_counts = count_site_level_modifications(loc_1001)

    # 样本级统计 / Sample-level statistics (unless --site-only)
    if site_only:
        # 仅位点级统计时，样本计数设为0 / Set sample counts to 0 for site-only mode
        sample_counts = {i: 0 for i in range(12)}
    else:
        print("正在统计样本级修饰分布... / Counting sample-level modification distribution...")
        sample_counts = count_sample_level_modifications(loc_12)

    # 输出统计结果 / Output statistics
    print_statistics(site_counts, sample_counts, total_samples, total_positions)

    # 可选保存CSV / Optionally save CSV
    if output_csv:
        save_csv_results(site_counts, sample_counts, total_samples, total_positions, output_csv)


# ============================================================================
# 命令行入口 / Command Line Entry
# ============================================================================

if __name__ == '__main__':
    # 命令行参数解析 / Command line argument parsing
    parser = argparse.ArgumentParser(
        description='统计Human3数据集RNA修饰位点分布 / Count RNA modification site distribution in Human3 dataset'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='npy/human3',
        help='数据目录路径 / Data directory path (default: npy/human3)'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default='analysis/modification_sites.csv',
        help='CSV输出路径 / CSV output path (optional)'
    )

    parser.add_argument(
        '--site-only',
        action='store_true',
        help='仅显示位点级统计 / Show site-level statistics only'
    )

    args = parser.parse_args()

    # 执行主函数 / Execute main function
    main(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        site_only=args.site_only
    )
