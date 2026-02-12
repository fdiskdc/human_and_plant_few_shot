#!/usr/bin/env python3
"""
预计算 MultIRM 数据集的二级结构
使用多进程并行计算，充分利用所有 CPU 核心
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.multirm import (
    MULTIRM_CLASSES,
    CLASS_TO_IDX,
    run_linearfold,
    build_edge_index_from_structure,
    LINEARFOLD_PATH,
    TARGET_LENGTH
)

# 批量缓存文件名
BATCH_CACHE_FILE = 'structures_cache.npz'


def worker_process_batch(args):
    """
    工作进程函数：处理一批序列的二级结构计算

    Args:
        args: tuple (batch_items, linearfold_path)

    Returns:
        list: [(cache_key, edge_index_numpy, error), ...]
    """
    batch_items, linearfold_path = args

    results = []
    sequences_str = []
    cache_keys = []

    for cache_key, sequence_str in batch_items:
        sequences_str.append(sequence_str)
        cache_keys.append(cache_key)

    try:
        structures = run_linearfold(sequences_str)

        for i, (cache_key, structure) in enumerate(zip(cache_keys, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            edge_index_numpy = edge_index.cpu().numpy()
            results.append((cache_key, edge_index_numpy, None))

    except Exception as e:
        for cache_key in cache_keys:
            results.append((cache_key, None, str(e)))

    return results


def collect_all_sequences(data_dir, mode='train'):
    """
    收集所有需要计算的序列

    Args:
        data_dir: 数据目录
        mode: 'train', 'test', 或 'valid'

    Returns:
        list: [(cache_key, sequence_str), ...]
    """
    all_items = []

    print(f"\n收集数据集序列...")
    print(f"  数据目录: {data_dir}")
    print(f"  模式: {mode}")
    print(f"  类别数: {len(MULTIRM_CLASSES)}")

    for class_name in MULTIRM_CLASSES:
        pos_dir = os.path.join(data_dir, class_name, 'pos')
        neg_dir = os.path.join(data_dir, class_name, 'neg')

        # 检查目录是否存在
        if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
            print(f"  跳过 {class_name}: 目录不存在")
            continue

        # 正样本
        pos_in_path = os.path.join(pos_dir, f'{mode}_in.npy')
        if os.path.exists(pos_in_path):
            try:
                pos_seqs = np.load(pos_in_path, allow_pickle=True)
                for i, seq in enumerate(pos_seqs):
                    sequence_bytes = seq.copy()
                    sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                    cache_key = f"{class_name}_pos_{i}"
                    all_items.append((cache_key, sequence_str))
                print(f"  {class_name} 正样本: {len(pos_seqs)}")
            except Exception as e:
                print(f"  警告: 加载 {class_name} 正样本失败: {e}")
        else:
            print(f"  {class_name} 正样本文件不存在: {pos_in_path}")

        # 负样本
        neg_in_path = os.path.join(neg_dir, f'{mode}_in.npy')
        if os.path.exists(neg_in_path):
            try:
                neg_seqs = np.load(neg_in_path, allow_pickle=True)
                for i, seq in enumerate(neg_seqs):
                    sequence_bytes = seq.copy()
                    sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                    cache_key = f"{class_name}_neg_{i}"
                    all_items.append((cache_key, sequence_str))
                print(f"  {class_name} 负样本: {len(neg_seqs)}")
            except Exception as e:
                print(f"  警告: 加载 {class_name} 负样本失败: {e}")
        else:
            print(f"  {class_name} 负样本文件不存在: {neg_in_path}")

    return all_items


def precompute_structures(
    data_dir,
    mode='train',
    cache_dir=None,
    batch_size=100,
    num_workers=None,
    show_progress=True
):
    """
    预计算所有序列的二级结构并保存到缓存文件

    Args:
        data_dir: 数据目录
        mode: 'train', 'test', 或 'valid'
        cache_dir: 缓存目录，默认为数据目录下的cache文件夹
        batch_size: 每批处理的序列数量
        num_workers: 工作进程数，None表示使用所有CPU核心
        show_progress: 是否显示进度条

    Returns:
        dict: 统计信息
    """
    # 收集所有序列
    all_items = collect_all_sequences(data_dir, mode)

    if not all_items:
        print("错误: 没有找到任何序列")
        return {'total': 0, 'computed': 0, 'failed': 0}

    # 设置缓存目录
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, '..', 'cache', 'multirm')
    os.makedirs(cache_dir, exist_ok=True)

    cache_filename = f"multirm_{mode}_{BATCH_CACHE_FILE}"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 设置工作进程数
    if num_workers is None:
        num_workers = cpu_count()

    print(f"\n{'='*70}")
    print(f"开始预计算二级结构")
    print(f"{'='*70}")
    print(f"  序列总数: {len(all_items)}")
    print(f"  批量大小: {batch_size}")
    print(f"  工作进程数: {num_workers}")
    print(f"  使用CPU核心: {num_workers} / {cpu_count()}")
    print(f"  LinearFold路径: {LINEARFOLD_PATH}")
    print(f"  缓存路径: {cache_path}")
    print(f"{'='*70}\n")

    # 准备批处理任务
    batch_tasks = []
    for start_idx in range(0, len(all_items), batch_size):
        end_idx = min(start_idx + batch_size, len(all_items))
        batch_items = all_items[start_idx:end_idx]
        batch_tasks.append((batch_items, LINEARFOLD_PATH))

    total_batches = len(batch_tasks)
    print(f"总共 {total_batches} 个批次\n")

    # 初始化结果存储
    edge_indices_dict = {}
    stats = {
        'total': len(all_items),
        'computed': 0,
        'failed': 0,
        'errors': []
    }

    # 使用多进程池
    use_multiprocessing = num_workers > 1

    if use_multiprocessing:
        print(f"使用 {num_workers} 个进程并行处理...\n")

        with Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 获取更实时的进度更新
            results_iter = pool.imap_unordered(worker_process_batch, batch_tasks)

            if show_progress:
                results_iter = tqdm(results_iter, total=total_batches, desc="计算二级结构")

            for batch_results in results_iter:
                for cache_key, edge_index_numpy, error in batch_results:
                    if error is None:
                        edge_indices_dict[cache_key] = edge_index_numpy
                        stats['computed'] += 1
                    else:
                        stats['failed'] += 1
                        if len(stats['errors']) < 10:  # 只记录前10个错误
                            stats['errors'].append(f"{cache_key}: {error}")
    else:
        # 单进程模式
        iterator = range(0, len(all_items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="计算二级结构")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(all_items))
            batch_items = all_items[start_idx:end_idx]

            batch_results = worker_process_batch((batch_items, LINEARFOLD_PATH))

            for cache_key, edge_index_numpy, error in batch_results:
                if error is None:
                    edge_indices_dict[cache_key] = edge_index_numpy
                    stats['computed'] += 1
                else:
                    stats['failed'] += 1
                    if len(stats['errors']) < 10:
                        stats['errors'].append(f"{cache_key}: {error}")

    # 保存缓存
    print(f"\n{'='*70}")
    print(f"保存缓存到: {cache_path}")

    try:
        # 方案：将每个 edge_index 作为独立的数组保存到同一个 npz 文件中
        # 这样可以避免形状不一致的问题，同时所有数据都在一个文件里
        
        # 获取所有的键
        keys = list(edge_indices_dict.keys())
        
        # 创建保存字典
        save_dict = {
            'keys': keys,
            'mode': mode,
            'num_samples': len(all_items)
        }
        
        # 将每个 edge_index 添加到保存字典中
        # 使用前缀 'edge_' 避免与其他键冲突
        for i, key in enumerate(keys):
            safe_key = f'edge_{i}'  # 使用索引作为键，避免特殊字符问题
            save_dict[safe_key] = edge_indices_dict[key]

        # 保存到 npz 文件（单个文件）
        np.savez_compressed(cache_path, **save_dict)

        file_size_mb = os.path.getsize(cache_path) / (1024**2)
        print(f"  缓存已保存到单个文件: {cache_path}")
        print(f"  文件大小: {file_size_mb:.2f} MB")
        print(f"  已缓存边索引数量: {len(edge_indices_dict)}")

    except Exception as e:
        print(f"  错误: 保存缓存失败: {e}")
        import traceback
        traceback.print_exc()
        return stats

    # 打印统计信息
    print(f"\n{'='*70}")
    print(f"预计算完成！")
    print(f"  总样本数: {stats['total']}")
    print(f"  成功计算: {stats['computed']}")
    print(f"  失败: {stats['failed']}")

    if stats['errors']:
        print(f"\n前 {len(stats['errors'])} 个错误:")
        for error in stats['errors']:
            print(f"  - {error}")

    print(f"{'='*70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='预计算 MultIRM 数据集的二级结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用所有CPU核心预计算训练集
  python calc_multirm.py

  # 指定数据目录和模式
  python calc_multirm.py --data_dir /path/to/data --mode test

  # 使用4个工作进程，批量大小50
  python calc_multirm.py --num_workers 4 --batch_size 50

  # 指定缓存目录
  python calc_multirm.py --cache_dir /path/to/cache
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='npy/multirm/51split',
        help='数据目录路径 (默认: npy/multirm/split)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'valid', 'all'],
        default='all',
        help='数据集模式 (默认: all, 计算所有模式)'
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default='npy/cache/multirm/51',
        help='缓存目录路径 (默认: 数据目录/../cache/multirm)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='每批处理的序列数量 (默认: 100)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=32,
        help='工作进程数，默认使用所有CPU核心'
    )

    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='不显示进度条'
    )

    args = parser.parse_args()

    # 检查 LinearFold 是否存在
    if not os.path.exists(LINEARFOLD_PATH):
        print(f"错误: LinearFold 不存在: {LINEARFOLD_PATH}")
        print("请先安装 LinearFold 或修改脚本中的路径")
        sys.exit(1)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # 确定要处理的模式
    if args.mode == 'all':
        modes = ['train', 'test', 'valid']
        print(f"\n计算所有模式的数据集: {', '.join(modes)}\n")
    else:
        modes = [args.mode]

    # 运行预计算
    all_stats = {}
    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"# 处理模式: {mode}")
        print(f"{'#'*70}\n")
        stats = precompute_structures(
            data_dir=args.data_dir,
            mode=mode,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            show_progress=not args.no_progress
        )
        all_stats[mode] = stats

    # 打印总体统计
    print(f"\n{'='*70}")
    print(f"所有模式计算完成！")
    print(f"{'='*70}")
    total_computed = sum(s['computed'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())
    total_samples = sum(s['total'] for s in all_stats.values())

    print(f"  总样本数: {total_samples}")
    print(f"  成功计算: {total_computed}")
    print(f"  失败: {total_failed}")

    for mode, stats in all_stats.items():
        print(f"\n  {mode}: {stats['computed']}/{stats['total']} 成功, {stats['failed']} 失败")

    print(f"{'='*70}\n")

    # 根据结果返回退出码
    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
