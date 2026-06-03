"""
inference_evormd_segmented.py - EvoRMD 41nt滑动窗口分段推理 / EvoRMD 41nt Sliding-Window Segmented Inference

使用适配 EvoRMD 检查点 (Conv1dEmbedder + TrainableAttention + MLP) 在 1001nt 序列上以 41nt 窗口 (stride=20) 滑动推理。
收集每窗口注意力并通过高斯加权 (gaussian_weighted_stitch) 拼回全长 [N, 12, 1001]。
Uses the adapted EvoRMD checkpoint (Conv1dEmbedder + TrainableAttention + MLP) on 1001nt sequences with 41nt sliding windows (stride=20).
Collects per-window attention and stitches back to full-length [N, 12, 1001] via Gaussian-weighted stitching.

功能模块 / Modules:
- sliding_window_inference: 滑动窗口推理 / Sliding window inference
- 高斯加权拼接 / Gaussian-weighted stitching
- main: 主入口 / Main entry point

输入 / Inputs:
- checkpoints/best_evormd.pt: PyTorch state_dict / Model weights
- json/evormd_inference.json: 推理配置 (window_size=41, stride=20) / Inference config
- npy/selected_*.npy: 预选序列 / Pre-selected sequences
- 命令行参数 / CLI: --checkpoint, --config, --output_dir

输出 / Outputs:
- npy/evormd_segmented_atten.npz: NumPy 压缩注意力 / NumPy compressed attention
  * 包含 / Contains: attn_weights [N, 12, 1001], labels [N, 12], seqs [N]

数据流 / Data Flow:
1. 加载模型 / Load model
2. 加载配置与序列 / Load config and sequences
3. 41nt 滑动窗口推理 / 41nt sliding window inference
4. 高斯加权拼接 / Gaussian-weighted stitching
5. 保存到 npz / Save to npz

相关文件 / Related Files:
- 调用 / Calls: model.evormd_human, sliding_window_utils, utils.common
- 被调用 / Called by: run_attention_comparison.py, manual CLI

使用示例 / Usage Example:
    python inference_evormd_segmented.py --checkpoint checkpoints/best_evormd.pt --output npy/evormd_atten.npz

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.evormd_human import EvoRMDForHuman
from sliding_window_utils import gaussian_weighted_stitch, make_gaussian_weight
from utils import load_config


def sliding_window_inference(model, sequence_onehot, window_size=41, stride=20, device='cuda'):
    """
    Run EvoRMD model on sliding windows using distance-weighted stitching.

    Args:
        model: EvoRMDForHuman
        sequence_onehot: [1001, 4] numpy array
        window_size: 41
        stride: 20
        device: torch device

    Returns:
        attn_full: [12, 1001] attention weights stitched from all windows
    """
    seq_len = sequence_onehot.shape[0]
    num_classes = 12

    attn_accum = np.zeros((num_classes, seq_len), dtype=np.float32)
    weight_accum = np.zeros((num_classes, seq_len), dtype=np.float32)

    model.eval()

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        window = sequence_onehot[start:end, :]

        x = torch.FloatTensor(window).unsqueeze(0).to(device)

        with torch.no_grad():
            token_embeddings = model.embedder(x)
            attn_weights = model.attention(token_embeddings)
            attn_w_np = attn_weights.squeeze(0).cpu().numpy()

        attn_broadcast = np.broadcast_to(attn_w_np, (num_classes, attn_w_np.shape[0]))
        weighted, ws, we = gaussian_weighted_stitch(
            attn_broadcast, start, seq_len, window_size, num_classes
        )
        attn_accum[:, ws:we] += weighted
        weight_accum[:, ws:we] += make_gaussian_weight(start, seq_len, window_size)[np.newaxis, :]

    weight_accum[weight_accum == 0] = 1.0
    attn_full = attn_accum / weight_accum

    return attn_full


def main(config_path='json/human_evormd.json',
         checkpoint_path='logs_evormd/evormd_human_20260508_104349/checkpoints/best_model.pt',
         output_path='npy/evormd_segmented_atten.npz',
         window_size=41, stride=20, top_n=20):

    Config, config_dict = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("EvoRMD Segmented Inference (41nt sliding window)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Window: {window_size}, Stride: {stride}")

    selected_indices = np.load('npy/selected_indices.npy')
    selected_sites = np.load('npy/selected_sites.npy')
    selected_labels = np.load('npy/selected_labels.npy')
    selected_seqs_str = np.load('npy/selected_seqs_str.npy', allow_pickle=True)
    print(f"Selected sequences: {len(selected_indices)}")

    seq_bytes = np.load('npy/human3/seq.npy', mmap_mode='r')

    model_cfg = config_dict.get('model', {})
    model = EvoRMDForHuman(
        num_task=12,
        d_fm=model_cfg.get('d_fm', 640),
        mlp_depth=model_cfg.get('mlp_depth', 2),
        conv_kernel_size=model_cfg.get('conv_kernel_size', 7),
        conv_dropout=model_cfg.get('conv_dropout', 0.1),
        use_hierarchical=model_cfg.get('use_hierarchical', True)
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return

    model.eval()

    ONEHOT = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'U': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
    }

    all_attn = []
    print(f"\nRunning sliding window inference on {len(selected_indices)} sequences...")
    for rank, idx in enumerate(tqdm(selected_indices, desc="EvoRMD sliding window")):
        raw = seq_bytes[idx]
        seq_str = raw.tobytes().decode('ascii', errors='ignore').upper()

        onehot = np.zeros((1001, 4), dtype=np.float32)
        for i, ch in enumerate(seq_str[:1001]):
            if ch in ONEHOT:
                onehot[i] = ONEHOT[ch]

        attn = sliding_window_inference(model, onehot, window_size, stride, device)
        all_attn.append(attn)

    all_attn_np = np.stack(all_attn, axis=0)
    print(f"\nAttention shape: {all_attn_np.shape}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez(
        output_path,
        attn_weights=all_attn_np,
        indices=selected_indices,
        sites=selected_sites,
        labels=selected_labels,
        seqs=selected_seqs_str
    )
    print(f"Saved to {output_path}")
    print(f"  attn_weights: {all_attn_np.shape} [N, 12, 1001]")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='json/human_evormd.json')
    parser.add_argument('--checkpoint', type=str,
                        default='logs_evormd/evormd_human_20260508_104349/checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='npy/evormd_segmented_atten.npz')
    parser.add_argument('--window_size', type=int, default=41)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.window_size, args.stride, args.top_n)
