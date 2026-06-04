"""
inference_modx_segmented.py - modX 101nt滑动窗口分段推理 / modX 101nt Sliding-Window Segmented Inference

使用适配 modX 检查点 (BiLSTM + BahdanauAttention) 在 1001nt 序列上以 101nt 窗口 (stride=50) 滑动推理。
收集每窗口注意力并通过高斯加权 (gaussian_weighted_stitch) 拼回全长 [N, 12, 1001]。
Uses the adapted modX checkpoint (BiLSTM + BahdanauAttention) on 1001nt sequences with 101nt sliding windows (stride=50).
Collects per-window attention and stitches back to full-length [N, 12, 1001] via Gaussian-weighted stitching.

功能模块 / Modules:
- build_model_from_checkpoint: 从 checkpoint 重建模型 / Rebuild model from checkpoint
- sliding_window_inference: 滑动窗口推理 / Sliding window inference
- 高斯加权拼接 / Gaussian-weighted stitching
- main: 主入口 / Main entry point

输入 / Inputs:
- checkpoints/best_modx.pt: PyTorch state_dict / Model weights
- json/modx_inference.json: 推理配置 (window_size=101, stride=50) / Inference config
- npy/selected_*.npy: 预选序列 / Pre-selected sequences
- 命令行参数 / CLI: --checkpoint, --config, --output_dir

输出 / Outputs:
- npy/modx_segmented_atten.npz: NumPy 压缩注意力结果 / NumPy compressed attention
  * 包含 / Contains: attn_weights [N, 12, 1001], labels [N, 12], seqs [N]

数据流 / Data Flow:
1. 加载模型 / Load model
2. 加载配置与序列 / Load config and sequences
3. 101nt 滑动窗口推理 / 101nt sliding window inference
4. 高斯加权拼接 / Gaussian-weighted stitching
5. 保存到 npz / Save to npz

相关文件 / Related Files:
- 调用 / Calls: model.modx_collect_atten, sliding_window_utils, utils.common
- 被调用 / Called by: run_attention_comparison.py, manual CLI

使用示例 / Usage Example:
    python inference_modx_segmented.py --checkpoint checkpoints/best_modx.pt --output npy/modx_atten.npz

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

from model.modx_collect_atten import RNAClassifierWithWord2Vec_Collect_Atten
from sliding_window_utils import gaussian_weighted_stitch, make_gaussian_weight
from utils import load_config


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint.get('config') or {}
    model_cfg = config.get('model', {})

    model = RNAClassifierWithWord2Vec_Collect_Atten(
        input_dim=4,
        embedding_dim=model_cfg.get('cnn_hidden_dim', 64),
        hidden_dim=model_cfg.get('gcn_hidden_dim', 128),
        num_layers=model_cfg.get('gcn_num_layers', 3),
        output_dim=model_cfg.get('num_classes', 12),
        dropout=model_cfg.get('cnn_dropout', 0.1),
    ).to(device)

    print("Model config:")
    print(f"  embedding_dim: {model.input_proj.out_features}")
    print(f"  hidden_dim:    {model.hidden_dim}")
    print(f"  num_layers:    {model.num_layers}")
    print(f"  output_dim:    {model.output_dim}")

    return model


def sliding_window_inference(model, sequence_onehot, window_size=101, stride=50, device='cuda'):
    """
    Run modx model on sliding windows using distance-weighted stitching.

    Args:
        model: RNAClassifierWithWord2Vec_Collect_Atten
        sequence_onehot: [1001, 4] numpy array
        window_size: 101
        stride: 50
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
            x_proj = model.input_proj(x)
            lstm_out, (h, c) = model.lstm(x_proj)
            last_h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
            context_vector, attention_weights = model.attention(last_h, lstm_out)

            attn_w = attention_weights.squeeze(-1).squeeze(0).cpu().numpy()
            if attn_w.ndim > 1:
                attn_w = attn_w[:, 0]

        attn_broadcast = np.broadcast_to(attn_w, (num_classes, attn_w.shape[0]))
        weighted, ws, we = gaussian_weighted_stitch(
            attn_broadcast, start, seq_len, window_size, num_classes
        )
        attn_accum[:, ws:we] += weighted
        weight_accum[:, ws:we] += make_gaussian_weight(start, seq_len, window_size)[np.newaxis, :]

    weight_accum[weight_accum == 0] = 1.0
    attn_full = attn_accum / weight_accum

    return attn_full


def main(config_path='json/human.json',
         checkpoint_path='logs/old/modx_rna_classification_20260203_135128/checkpoints/best_model.pt',
         output_path='npy/modx_segmented_atten.npz',
         window_size=101, stride=50, top_n=20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("ModX Segmented Inference (101nt sliding window)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Window: {window_size}, Stride: {stride}")

    selected_indices = np.load('npy/selected_indices.npy')
    selected_sites = np.load('npy/selected_sites.npy')
    selected_labels = np.load('npy/selected_labels.npy')
    selected_seqs_str = np.load('npy/selected_seqs_str.npy', allow_pickle=True)
    print(f"Selected sequences: {len(selected_indices)}")

    seq_bytes = np.load('npy/human3/seq.npy', mmap_mode='r')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model_from_checkpoint(checkpoint, device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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
    for rank, idx in enumerate(tqdm(selected_indices, desc="ModX sliding window")):
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
    parser.add_argument('--config', type=str, default='json/human.json')
    parser.add_argument('--checkpoint', type=str,
                        default='logs/old/modx_rna_classification_20260203_135128/checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='npy/modx_segmented_atten.npz')
    parser.add_argument('--window_size', type=int, default=101)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.window_size, args.stride, args.top_n)
