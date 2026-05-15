"""
Common sliding-window stitching utilities for attention comparison experiments.
"""

import numpy as np


def gaussian_weighted_stitch(window_attn, start, seq_len, window_size, num_classes, sigma=10.0):
    """
    Stitch a single window's attention into the full-length array using
    Gaussian weighting (center = weight 1.0, edges decay by exp).

    This ensures all models have the same effective window width (±3σ),
    regardless of their actual sliding window size.

    Args:
        window_attn: [num_classes, window_size] attention from one window
        start: start position of this window in the full sequence
        seq_len: full sequence length (1001)
        window_size: e.g. 51, 101, 41
        num_classes: 12
        sigma: Gaussian standard deviation in nt (default 10)

    Returns:
        weighted_attn: [num_classes, actual_len] Gaussian-weighted attention
        ws: window start in full sequence
        we: window end in full sequence
    """
    half_w = window_size // 2
    center = start + half_w
    end = min(start + window_size, seq_len)
    actual_len = end - start

    dists = np.abs(np.arange(start, end) - center).astype(np.float32)
    weights = np.exp(-0.5 * (dists / sigma) ** 2)

    weighted = window_attn[:, :actual_len] * weights[np.newaxis, :]
    return weighted, start, end


def make_gaussian_weight(start, seq_len, window_size, sigma=10.0):
    """
    Create the Gaussian weight array for a given window (for accumulation).
    """
    half_w = window_size // 2
    center = start + half_w
    end = min(start + window_size, seq_len)
    dists = np.abs(np.arange(start, end) - center).astype(np.float32)
    return np.exp(-0.5 * (dists / sigma) ** 2)
