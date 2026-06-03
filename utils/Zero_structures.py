#!/usr/bin/env python3
"""
utils/Zero_structures.py - Zero 数据集二级结构预计算 / Zero Dataset Structure Precomputation

使用 LinearFold 预计算 Zero 数据集的 RNA 二级结构,缓存到磁盘以加速训练。
Uses LinearFold to precompute RNA secondary structures for the Zero dataset, cached to disk for faster training.

功能模块 / Modules:
- LinearFold 调用 / LinearFold invocation
- 多进程预计算 / Multiprocessing precomputation
- 缓存保存到 NPZ / Cache saved to NPZ
- main: 主入口 / Main entry point

输入 / Inputs:
- zero/zero_seq.npy: Zero RNA 序列 / Zero RNA sequences
- 命令行参数 / CLI: --batch_size, --num_workers

输出 / Outputs:
- npy/cache/zero_batch_cache.npz: 二级结构缓存 / Structure cache
  * 包含 / Contains: structures (list), edge_indices (object array)

数据流 / Data Flow:
1. 加载 Zero 序列 / Load Zero sequences
2. 多进程调用 LinearFold / Multiprocess LinearFold
3. 构建 edge_indices / Build edge_indices
4. 缓存到 NPZ / Cache to NPZ

相关文件 / Related Files:
- 调用 / Calls: numpy, LinearFold executable, multiprocessing
- 被调用 / Called by: train_*.py with Zero data, dataset.plant_single

使用示例 / Usage Example:
    python -m utils.Zero_structures --batch_size 100 --num_workers 4

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import torch
import argparse
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import subprocess


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ZERO_DIR = 'npy/zero'
DEFAULT_CACHE_DIR = 'npy/cache'
# Zero dataset: 154,607 samples, sequence length 1001
# For 192-core ARM processor: use all cores with moderate batch size
DEFAULT_BATCH_SIZE = 500
DEFAULT_NUM_WORKERS = 128
DEFAULT_LINEARFOLD_PATH = '/home/dc/vscode/LinearFold/linearfold'

# Cache file name
ZERO_CACHE_FILE = 'zero_batch_cache.npz'


# =============================================================================
# LinearFold Functions
# =============================================================================

def run_linearfold(sequences, linearfold_path, timeout_seconds=1800):
    """
    Predict RNA secondary structures using LinearFold.

    Args:
        sequences (list): List of RNA sequence strings
        linearfold_path (str): Path to LinearFold executable
        timeout_seconds (int): Timeout in seconds

    Returns:
        list: List of secondary structure strings

    Raises:
        RuntimeError: If LinearFold execution fails
        FileNotFoundError: If LinearFold executable not found
        subprocess.TimeoutExpired: If execution times out
    """
    if not sequences:
        return []

    # Build FASTA format input
    fasta_input = '\n'.join([f'>seq_{i}\n{seq}' for i, seq in enumerate(sequences)])

    try:
        # Check if LinearFold exists
        if not os.path.exists(linearfold_path):
            raise FileNotFoundError(f"LinearFold executable not found: {linearfold_path}")

        # Call LinearFold with stdin input
        process = subprocess.Popen(
            [linearfold_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        # Get output
        stdout_data, stderr_data = process.communicate(input=fasta_input, timeout=timeout_seconds)

        # Check return code
        if process.returncode != 0:
            raise RuntimeError(
                f"LinearFold execution failed with return code {process.returncode}. "
                f"Error: {stderr_data[:500]}"
            )

        if len(stdout_data.strip()) == 0:
            raise RuntimeError("LinearFold returned no output")

        # Parse output
        # LinearFold output format:
        # >seq_0
        # ACGUACGU
        # ........ (-0.08)

        lines = stdout_data.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        structures = []
        for i in range(2, len(lines), 3):
            line = lines[i]
            if not line:
                continue
            # Structure is before the space, e.g.: ........ (-0.08) -> ........
            structure = line.split()[0]
            structures.append(structure)

        # Validate structure count
        if len(structures) != len(sequences):
            raise RuntimeError(
                f"LinearFold returned {len(structures)} structures, "
                f"expected {len(sequences)}"
            )

    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired as e:
        raise subprocess.TimeoutExpired(e.cmd, e.timeout, output=e.output, stderr=e.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold execution failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unknown error during LinearFold execution: {e}")

    return structures


def build_edge_index_from_structure(sequence, structure):
    """
    Build edge index from RNA secondary structure.

    Args:
        sequence (str): RNA sequence
        structure (str): Secondary structure in dot-bracket notation

    Returns:
        torch.Tensor: Edge index with shape [2, E]
    """
    if not structure:
        return build_sequential_edge_index(sequence)

    stack = []
    pairs = {}

    # Parse bracket structure to find pairs
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j

    edge_list = []

    # Add sequential edges (i, i+1)
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])

    # Add pairing edges
    for i, j in pairs.items():
        if i < j:  # Avoid duplicates
            edge_list.extend([(i, j), (j, i)])

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


def build_sequential_edge_index(sequence):
    """
    Build only sequential edges (i, i+1).

    Args:
        sequence (str): RNA sequence

    Returns:
        torch.Tensor: Edge index with shape [2, E]
    """
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


# =============================================================================
# Worker Function for Multiprocessing
# =============================================================================

def _worker_process_batch(args):
    """
    Worker process function for processing a batch of sequences.

    Args:
        args: tuple (batch_indices, sequences_bytes_array, linearfold_path)

    Returns:
        list: [(idx, edge_index_numpy, error), ...]
    """
    batch_indices, sequences_bytes_array, linearfold_path = args

    # Prepare batch sequence strings
    sequences_str = []
    for idx in batch_indices:
        sequence_bytes = sequences_bytes_array[idx].copy()
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        sequences_str.append(sequence_str)

    results = []
    try:
        # Use LinearFold to compute structures
        structures = run_linearfold(sequences_str, linearfold_path)

        # Build edge indices
        for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            edge_index_numpy = edge_index.cpu().numpy()
            results.append((idx, edge_index_numpy, None))

    except Exception as e:
        # Return error info on failure
        for idx in batch_indices:
            results.append((idx, None, str(e)))

    return results


# =============================================================================
# Main Precomputation Function
# =============================================================================

def precompute_zero_structures(zero_dir, cache_dir, batch_size=100, num_workers=None,
                               linearfold_path=None, show_progress=True):
    """
    Precompute structures for the Zero dataset.

    Args:
        zero_dir: Directory containing zero data (zero_seq.npy, etc.)
        cache_dir: Directory for cache files
        batch_size: Batch size for LinearFold processing
        num_workers: Number of worker processes (None = CPU count)
        linearfold_path: Path to LinearFold executable
        show_progress: Whether to show progress bar
    """
    if linearfold_path is None:
        linearfold_path = DEFAULT_LINEARFOLD_PATH

    # Resolve paths
    if not os.path.exists(zero_dir) and os.path.exists(f'../{zero_dir}'):
        zero_dir = f'../{zero_dir}'

    zero_seq_path = os.path.join(zero_dir, 'zero_seq.npy')

    # Load Zero sequences
    print(f"\n{'='*60}")
    print(f"Loading Zero sequences from: {zero_seq_path}")
    zero_seq = np.load(zero_seq_path, mmap_mode='r')
    num_samples = len(zero_seq)
    print(f"Loaded {num_samples} sequences")

    # Cache file path
    cache_path = os.path.join(cache_dir, ZERO_CACHE_FILE)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache will be saved to: {cache_path}")

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    use_multiprocessing = num_workers > 1

    print(f"\n{'='*60}")
    print(f"Precomputing Zero dataset structures...")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers if use_multiprocessing else 1}")
    print(f"  LinearFold: {linearfold_path}")
    print(f"{'='*60}\n")

    # Initialize edge indices array
    edge_indices = np.empty(num_samples, dtype=object)
    stats = {'total': num_samples, 'computed': 0, 'failed': 0}

    # Prepare batch tasks
    batch_tasks = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = list(range(start_idx, end_idx))
        batch_tasks.append((batch_indices, zero_seq, linearfold_path))

    total_batches = len(batch_tasks)

    if use_multiprocessing:
        # Multi-process processing
        print(f"Using {num_workers} processes...\n")

        with Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)

            if show_progress:
                results_iter = tqdm(results_iter, total=total_batches,
                                   desc="Precomputing Zero structures")

            # Collect results
            for batch_results in results_iter:
                for idx, edge_index_numpy, error in batch_results:
                    if error is None:
                        edge_indices[idx] = edge_index_numpy
                        stats['computed'] += 1
                    else:
                        stats['failed'] += 1
                        if stats['failed'] <= 5:
                            print(f"  Warning: Index {idx} failed: {error}")

    else:
        # Single-process processing
        iterator = range(0, num_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Precomputing Zero structures")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            # Prepare batch sequence strings
            sequences_str = []
            for idx in batch_indices:
                sequence_bytes = zero_seq[idx].copy()
                sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                sequences_str.append(sequence_str)

            # Use LinearFold to compute structures
            try:
                structures = run_linearfold(sequences_str, linearfold_path)

                # Build edge indices
                for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
                    edge_index = build_edge_index_from_structure(sequences_str[i], structure)
                    edge_indices[idx] = edge_index.cpu().numpy()
                    stats['computed'] += 1

            except Exception as e:
                print(f"\nWarning: Batch computation failed (indices {start_idx}-{end_idx}): {e}")
                for idx in batch_indices:
                    stats['failed'] += 1

    # Save to batch cache file
    print(f"\nSaving Zero batch cache to: {cache_path}")
    try:
        np.savez_compressed(
            cache_path,
            edge_indices=edge_indices,
            mode='zero',
            num_samples=num_samples
        )

        file_size_mb = os.path.getsize(cache_path) / (1024**2)
        print(f"  Cache saved, size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"  Error: Failed to save cache: {e}")
        return

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Zero precomputation complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Computed: {stats['computed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Precompute Zero dataset RNA structures using LinearFold'
    )
    parser.add_argument(
        '--zero_dir',
        type=str,
        default=DEFAULT_ZERO_DIR,
        help='Directory containing zero data (default: npy/zero)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=DEFAULT_CACHE_DIR,
        help='Directory for cache files (default: npy/cache)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for LinearFold processing (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f'Number of worker processes (default: {DEFAULT_NUM_WORKERS})'
    )
    parser.add_argument(
        '--linearfold_path',
        type=str,
        default=DEFAULT_LINEARFOLD_PATH,
        help='Path to LinearFold executable'
    )

    args = parser.parse_args()

    precompute_zero_structures(
        zero_dir=args.zero_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        linearfold_path=args.linearfold_path,
        show_progress=True
    )


if __name__ == '__main__':
    main()
