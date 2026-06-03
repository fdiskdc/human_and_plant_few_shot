"""
select_representative_sequences.py - 按m6A修饰密度选代表性序列 / Select Representative Sequences by m6A Density

按 m6A 修饰位点数对序列排序,选择高密度与低密度两组,用于注意力对比可视化。
Sorts sequences by number of m6A sites, selects both high-density and low-density groups for attention comparison.

功能模块 / Modules:
- m6A 位点数统计 / m6A site count statistics
- 高/低密度分组 / High/low density grouping
- 序列与标签选择 / Sequence and label selection
- main: 主入口 / Main entry point

输入 / Inputs:
- human3/seq.npy: 1001nt RNA 序列 / 1001nt RNA sequences
- human3/1001loc.npy: 1001 位点级标签 / 1001 site-level labels
- human3/12loc.npy: 12 类多热标签 / 12-class multi-hot labels
- M6A_LABEL = 10 (m6A 索引) / m6A index

输出 / Outputs:
- npy/selected_indices.npy: 选中序列的索引 / Selected indices
- npy/selected_sequences.npy: one-hot 序列 (deprecated) / One-hot sequences
- npy/selected_labels.npy: 12 类多热标签 / 12-class multi-hot labels
- npy/selected_sites.npy: 1001 位点标签 / 1001 site-level labels
- npy/selected_seqs_str.npy: 序列字符串 / Sequence strings
- npy/selected_group.npy: 组标签 (0=高密度, 1=低密度) / Group label

数据流 / Data Flow:
1. 加载序列与位点标签 / Load sequences and site labels
2. 统计每序列 m6A 位点数 / Count m6A sites per sequence
3. 排序 + 选 top-N / Sort + select top-N
4. 保存选中索引与数据 / Save selected indices and data

相关文件 / Related Files:
- 调用 / Calls: numpy
- 被调用 / Called by: run_attention_comparison.py, visualize_attention_comparison.py

使用示例 / Usage Example:
    python select_representative_sequences.py --top_n 50

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import sys
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

M6A_LABEL = 10

MOD_NAMES = [
    'Am', 'Atol', 'Cm', 'Gm', 'Tm', 'Y',
    'ac4C', 'm1A', 'm5C', 'm6A', 'm6Am', 'm7G'
]


def main(top_n=100, low_n=100, output_dir='npy'):
    title = 'Selecting Representative Sequences by m6A Density'
    print(f"{'='*60}")
    print(title)
    print(f"{'='*60}")

    loc_1001 = np.load('npy/human3/1001loc.npy', mmap_mode='r')
    loc_12 = np.load('npy/human3/12loc.npy', mmap_mode='r')
    seq_bytes = np.load('npy/human3/seq.npy', mmap_mode='r')

    n_samples = loc_1001.shape[0]
    print(f"Total samples: {n_samples}")
    print(f"Sequence length: {loc_1001.shape[1]}")

    m6a_per_seq = np.sum(loc_1001 == M6A_LABEL, axis=1)
    has_m6a = m6a_per_seq > 0

    print(f"\nm6A density stats (all {n_samples} sequences):")
    print(f"  sequences with m6A: {has_m6a.sum()} ({has_m6a.sum()/n_samples*100:.2f}%)")
    print(f"  m6A sites per seq — min: {m6a_per_seq.min()}, max: {m6a_per_seq.max()}")
    print(f"  m6A sites per seq — mean: {m6a_per_seq.mean():.2f}, median: {np.median(m6a_per_seq):.0f}")

    sorted_high = np.argsort(m6a_per_seq)[::-1]
    selected_high = sorted_high[:top_n]

    print(f"\n=== High-density group (top-{top_n} by m6A sites) ===")
    for rank, idx in enumerate(selected_high):
        count = m6a_per_seq[idx]
        classes = np.where(loc_12[idx] > 0)[0]
        class_names = [MOD_NAMES[c] for c in classes]
        print(f"  #{rank+1}: index={idx}, m6A={count}, classes={class_names}")

    selected_low = np.array([], dtype=selected_high.dtype)
    if low_n > 0:
        m6a_positive = np.where(has_m6a)[0]
        m6a_counts_positive = m6a_per_seq[m6a_positive]
        sorted_low_local = m6a_positive[np.argsort(m6a_counts_positive)]
        selected_low = sorted_low_local[:low_n]

        print(f"\n=== Low-density group (bottom-{low_n} by m6A sites, >=1 site) ===")
        for rank, idx in enumerate(selected_low):
            count = m6a_per_seq[idx]
            classes = np.where(loc_12[idx] > 0)[0]
            class_names = [MOD_NAMES[c] for c in classes]
            print(f"  #{rank+1}: index={idx}, m6A={count}, classes={class_names}")

    all_indices = np.concatenate([selected_high, selected_low])
    groups = np.concatenate([
        np.zeros(len(selected_high), dtype=np.int8),
        np.ones(len(selected_low), dtype=np.int8)
    ])
    n_total = len(all_indices)
    print(f"\nTotal selected: {n_total} (high={len(selected_high)}, low={len(selected_low)})")

    print(f"\nLoading selected data...")
    selected_sites = loc_1001[all_indices].astype(np.int8)
    selected_labels = loc_12[all_indices].astype(np.float32)

    selected_seqs = []
    for idx in all_indices:
        raw = seq_bytes[idx]
        seq_str = raw.tobytes().decode('ascii', errors='ignore').upper()
        selected_seqs.append(seq_str)

    selected_seqs_str = np.array(selected_seqs, dtype=object)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'selected_indices.npy'), all_indices)
    np.save(os.path.join(output_dir, 'selected_sites.npy'), selected_sites)
    np.save(os.path.join(output_dir, 'selected_labels.npy'), selected_labels)
    np.save(os.path.join(output_dir, 'selected_seqs_str.npy'), selected_seqs_str)
    np.save(os.path.join(output_dir, 'selected_group.npy'), groups)

    print(f"\nSaved to {output_dir}/:")
    print(f"  selected_indices.npy:  {all_indices.shape}")
    print(f"  selected_sites.npy:    {selected_sites.shape}")
    print(f"  selected_labels.npy:   {selected_labels.shape}")
    print(f"  selected_seqs_str.npy: {selected_seqs_str.shape} (object array)")
    print(f"  selected_group.npy:    {groups.shape} (0=high, 1=low)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of high-m6A-density sequences')
    parser.add_argument('--low_n', type=int, default=100,
                        help='Number of low-m6A-density sequences (>=1 m6A site)')
    parser.add_argument('--output_dir', type=str, default='npy')
    args = parser.parse_args()
    main(top_n=args.top_n, low_n=args.low_n, output_dir=args.output_dir)
