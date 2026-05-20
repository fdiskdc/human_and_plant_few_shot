"""
Select Representative Sequences by m6A Modification Density

Sorts sequences by number of m6A sites, selects both high-density and low-density
groups for attention comparison visualization.

Output:
    - npy/selected_indices.npy  — indices of selected sequences
    - npy/selected_sequences.npy — one-hot sequences (deprecated)
    - npy/selected_labels.npy   — 12-class multi-hot labels
    - npy/selected_sites.npy    — per-position site labels (1001)
    - npy/selected_seqs_str.npy — sequence strings
    - npy/selected_group.npy    — group label (0=high, 1=low) per sequence
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
