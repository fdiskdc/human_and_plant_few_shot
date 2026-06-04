"""
inference_mrmodn_full.py - mRModN 1001nt全长推理与注意力收集 / mRModN Full-Length 1001nt Inference with Attention

在 1001nt 全长序列上运行 mRModN (GCN + per-class MHA),收集 attn_weights_12: [N, 12, 1001] — 每个类别对全长序列的注意力。
优化:矢量化 one-hot,带 per-graph 偏移的批处理 edge_index,直接 tensor 输入 (无 Data/Batch 开销)。
Runs mRModN (GCN + per-class MHA) on full 1001nt sequences, collecting attn_weights_12: [N, 12, 1001].
Optimizations: vectorized one-hot, batched edge_index with per-graph offsets, direct tensor feed.

功能模块 / Modules:
- model: 模型前向 + 注意力收集 / Model forward + attention collection
- one-hot 矢量化编码 / Vectorized one-hot encoding
- main: 主入口 / Main entry point

输入 / Inputs:
- checkpoints/best_mrmodn.pt: PyTorch state_dict / Model weights
- json/mrmodn_inference.json: 推理配置 / Inference config
- npy/selected_*.npy: 预选序列 / Pre-selected sequences
- 命令行参数 / CLI: --checkpoint, --config, --output_dir

输出 / Outputs:
- npy/mrmodn_full_atten.npz: NumPy 压缩注意力 / NumPy compressed attention
  * 包含 / Contains: attn_weights_12 [N, 12, 1001], labels [N, 12], seqs [N]

数据流 / Data Flow:
1. 加载模型 / Load model
2. 矢量化 one-hot 编码 / Vectorized one-hot encoding
3. 单次前向 1001nt / Single forward pass on 1001nt
4. 收集 per-class 注意力 / Collect per-class attention
5. 保存到 npz / Save to npz

相关文件 / Related Files:
- 调用 / Calls: model.mrmodn_collect_atten, utils.common
- 被调用 / Called by: run_attention_comparison.py, manual CLI

使用示例 / Usage Example:
    python inference_mrmodn_full.py --checkpoint checkpoints/best_mrmodn.pt --output npy/mrmodn_atten.npz

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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mrmodn_collect_atten import RNA_ClassQuery_Model_Collect_Atten
from utils import load_config

ONEHOT_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'U': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],
}


def build_sequential_edge_index(seq_len):
    src = list(range(seq_len - 1)) + list(range(1, seq_len))
    dst = list(range(1, seq_len)) + list(range(seq_len - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def load_edge_index_from_cache(cache_path, idx):
    cache = np.load(cache_path, allow_pickle=True)
    if 'edge_indices' in cache:
        ei = cache['edge_indices'][idx]
        if isinstance(ei, np.ndarray):
            return torch.from_numpy(ei).long()
    return None


def main(config_path='json/human.json',
         checkpoint_path='logs/old/rna_classification_20260129_195404/checkpoints/best_model.pt',
         output_path='npy/mrmodn_full_atten.npz',
         top_n=20):

    Config, _ = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("mRModN Full-Length Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")

    selected_indices = np.load('npy/selected_indices.npy')
    selected_sites = np.load('npy/selected_sites.npy')
    selected_labels = np.load('npy/selected_labels.npy')
    selected_seqs_str = np.load('npy/selected_seqs_str.npy', allow_pickle=True)
    n_samples = len(selected_indices)
    print(f"Selected sequences: {n_samples}")

    cache_path = os.path.join(Config.data.cache_dir, 'human_train_structures_cache.npz')
    use_cache = os.path.exists(cache_path)
    if use_cache:
        print(f"Using structure cache: {cache_path}")
    else:
        print("No cache found, using sequential edges")

    model = RNA_ClassQuery_Model_Collect_Atten(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=Config.num_classes,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=Config.use_hierarchical,
        use_layer_norm=Config.use_layer_norm
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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

    x = torch.FloatTensor(onehots).reshape(-1, 4).to(device)
    batch = torch.arange(n_samples, device=device).repeat_interleave(1001)

    base_ei = build_sequential_edge_index(1001)
    if use_cache:
        cache = np.load(cache_path, allow_pickle=True)
        cached_ei = cache.get('edge_indices', None)
        ei_parts = []
        for i, idx in enumerate(selected_indices):
            ei = None
            if cached_ei is not None and idx < len(cached_ei):
                ei_candidate = cached_ei[idx]
                if isinstance(ei_candidate, np.ndarray):
                    ei = torch.from_numpy(ei_candidate).long()
            if ei is None:
                ei = base_ei
            ei_parts.append(ei + i * 1001)
        edge_index = torch.cat(ei_parts, dim=1).to(device)
    else:
        n_edges = base_ei.size(1)
        offsets = torch.arange(n_samples).repeat_interleave(n_edges) * 1001
        edge_index = (base_ei.repeat(1, n_samples) + offsets).to(device)

    print(f"  x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

    print(f"\nRunning mRModN inference...")
    with torch.no_grad():
        logits_12, logits_4, attn_out_12, attn_out_4, attn_weights_12 = model(
            x, edge_index, batch, return_attention=True
        )
        probs_12 = torch.sigmoid(logits_12)

    attn_np = attn_weights_12.cpu().numpy()
    probs_np = probs_12.cpu().numpy()

    print(f"\nAttention weights shape: {attn_np.shape}")
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
    parser.add_argument('--config', type=str, default='json/human.json')
    parser.add_argument('--checkpoint', type=str,
                        default='logs/old/rna_classification_20260129_195404/checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='npy/mrmodn_full_atten.npz')
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.top_n)
