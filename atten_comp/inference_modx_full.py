"""
atten_comp/inference_modx_full.py - modX 1001nt全长单次推理 / modX 1001nt Full-Length Single-Pass Inference

对 1001nt 全长序列运行 modX (BiLSTM + BahdanauAttention) 单次前向 (无滑动窗口),收集 1001 位置注意力。
矢量化 one-hot、直接 Data 构造 (无 per-sequence 循环),输出 modx_full_atten.npz。
Runs modX (BiLSTM + BahdanauAttention) on full 1001nt sequences with single forward (no sliding window).
Vectorized one-hot, direct Data construction, outputs modx_full_atten.npz.

功能模块 / Modules:
- 矢量化 one-hot 编码 / Vectorized one-hot encoding
- 直接 Data 构造 / Direct Data construction
- 1001nt 单次前向 / Single forward pass
- main: 主入口 / Main entry point

输入 / Inputs:
- checkpoints/best_modx.pt: PyTorch state_dict / ModX model weights
- json/modx_inference.json: 推理配置 / Inference config
- npy/selected_*.npy: 预选序列 / Pre-selected sequences
- 命令行参数 / CLI: --checkpoint, --config, --output_dir

输出 / Outputs:
- npy/modx_full_atten.npz: NumPy 压缩格式 / NumPy compressed format
  * 包含 / Contains: attn_weights [N, 12, 1001], labels [N, 12], seqs [N]

数据流 / Data Flow:
1. 加载 modX 模型 / Load modX model
2. 矢量化 one-hot 编码 / Vectorized one-hot encoding
3. 1001nt 单次前向 / Single forward pass on 1001nt
4. 收集注意力 / Collect attention
5. 保存到 npz / Save to npz

相关文件 / Related Files:
- 调用 / Calls: model.modx_collect_atten, torch_geometric.data.Data
- 被调用 / Called by: atten_comp/run_attention_comparison_v2.py

使用示例 / Usage Example:
    python atten_comp/inference_modx_full.py --checkpoint checkpoints/best_modx.pt

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
from tqdm import tqdm
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modx_collect_atten import RNAClassifierWithWord2Vec_Collect_Atten

ONEHOT_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'U': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],
}


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


def main(checkpoint_path='logs/old/modx_rna_classification_20260203_135128/checkpoints/best_model.pt',
         output_path='npy/modx_full_atten.npz'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("ModX Full-Length Inference (1001nt, no sliding window)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    selected_indices = np.load('npy/selected_indices.npy')
    selected_sites = np.load('npy/selected_sites.npy')
    selected_labels = np.load('npy/selected_labels.npy')
    selected_seqs_str = np.load('npy/selected_seqs_str.npy', allow_pickle=True)
    n_samples = len(selected_indices)
    print(f"Selected sequences: {n_samples}")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model_from_checkpoint(checkpoint, device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return

    model.eval()

    print(f"\nPreparing {n_samples} samples (vectorized)...")

    onehots = np.zeros((n_samples, 1001, 4), dtype=np.float32)
    for i, seq_str in enumerate(tqdm(selected_seqs_str, desc="  One-hot encoding")):
        for pos, ch in enumerate(seq_str[:1001]):
            if ch in ONEHOT_MAP:
                onehots[i, pos] = ONEHOT_MAP[ch]

    x = torch.FloatTensor(onehots).reshape(-1, 4)
    batch = torch.arange(n_samples).repeat_interleave(1001)

    data = Data(x=x, batch=batch).to(device)

    print(f"  x: {x.shape}, batch: {batch.shape}")

    print(f"\nRunning modx full-length inference on {n_samples} sequences...")
    with torch.no_grad():
        logits, context_vector, attn_weights = model(data, return_attention=True)
        probs = torch.sigmoid(logits)

    attn_np = attn_weights.cpu().numpy()
    probs_np = probs.cpu().numpy()

    print(f"\nAttention shape: {attn_np.shape}")
    print(f"Probs shape: {probs_np.shape}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez(
        output_path,
        attn_weights=attn_np,
        probs=probs_np,
        indices=selected_indices,
        sites=selected_sites,
        labels=selected_labels,
        seqs=selected_seqs_str
    )
    print(f"Saved to {output_path}")
    print(f"  attn_weights: {attn_np.shape} [N, 12, 1001]")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='logs/old/modx_rna_classification_20260203_135128/checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='npy/modx_full_atten.npz')
    args = parser.parse_args()
    main(args.checkpoint, args.output)
