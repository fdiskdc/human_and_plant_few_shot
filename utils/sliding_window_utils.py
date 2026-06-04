"""
sliding_window_utils.py - 滑动窗口注意力拼接通用工具 / Common Sliding-Window Stitching Utilities

为注意力对比实验提供高斯加权滑动窗口拼接:确保所有模型具有相同的有效窗口宽度 (±3σ),无论实际窗口大小。
高斯权重:中心 = 1.0,边缘按 exp 衰减。
Common Gaussian-weighted sliding window stitching for attention comparison experiments.
Ensures all models have the same effective window width (±3σ), regardless of their actual sliding window size.

功能模块 / Modules:
- gaussian_weighted_stitch: 单窗口高斯加权拼接 / Gaussian-weighted single-window stitch
- make_gaussian_weight: 生成高斯权重 / Make Gaussian weight
- main 工具 / Utility functions

输入 / Inputs:
- window_attn: [num_classes, window_size] 单窗口注意力 / Single window attention
- start: 窗口起始位置 / Window start position
- seq_len: 全长 (1001) / Full sequence length
- window_size: 51, 101, 41 等 / 51, 101, 41 etc.
- sigma: 高斯标准差 (默认 10) / Gaussian std

输出 / Outputs:
- weighted_attn: [num_classes, actual_len] 高斯加权注意力 / Gaussian-weighted attention
- ws, we: 窗口起止 / Window start/end

数据流 / Data Flow:
1. 接收单窗口注意力 / Receive single window attention
2. 计算高斯权重 / Compute Gaussian weight
3. 加权叠加到全长数组 / Weighted add to full-length array
4. 返回加权结果 / Return weighted result

相关文件 / Related Files:
- 调用 / Calls: numpy
- 被调用 / Called by: inference_modx_segmented, inference_multirm_segmented, inference_evormd_segmented

使用示例 / Usage Example:
    from sliding_window_utils import gaussian_weighted_stitch

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
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
