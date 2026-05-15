"""
MultiRM Segmented Inference — 51nt Sliding Window

Uses the adapted MultiRM checkpoint (BiLSTM + BahdanauAttention) trained on human data.
Runs 51nt sliding windows (stride=1) over 1001nt sequences.
Collects attention per window and stitches back to full-length [N, 12, 1001].

Output: npy/multirm_segmented_atten.npz
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

from model.multirm_collect_atten import model_v3_Collect_Atten
from sliding_window_utils import gaussian_weighted_stitch, make_gaussian_weight
from utils import load_config


def sliding_window_inference(model, sequence_onehot, window_size=51, stride=1, device='cuda'):
    """
    Run model on sliding windows of a single sequence using distance-weighted stitching.

    Args:
        model: model_v3_Collect_Atten (adapted, trained on human data)
        sequence_onehot: [1001, 4] numpy array
        window_size: 51
        stride: 1
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
            logits, logits_4, ctx, attn_weights = model(
                x, edge_index=None, batch=None, return_attention=True
            )
            attn_w_np = attn_weights.squeeze(0).cpu().numpy()

        weighted, ws, we = gaussian_weighted_stitch(
            attn_w_np, start, seq_len, window_size, num_classes
        )
        attn_accum[:, ws:we] += weighted
        weight_accum[:, ws:we] += make_gaussian_weight(start, seq_len, window_size)[np.newaxis, :]

    weight_accum[weight_accum == 0] = 1.0
    attn_full = attn_accum / weight_accum

    return attn_full


def main(config_path='json/human.json',
         checkpoint_path='logs/old/multirm_rna_classification_20260203_230348/checkpoints/best_model.pt',
         output_path='npy/multirm_segmented_atten.npz',
         window_size=51, stride=1, top_n=20):

    Config, _ = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("MultiRM Segmented Inference (51nt sliding window)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Window: {window_size}, Stride: {stride}")

    selected_indices = np.load('npy/selected_indices.npy')
    selected_sites = np.load('npy/selected_sites.npy')
    selected_labels = np.load('npy/selected_labels.npy')
    selected_seqs_str = np.load('npy/selected_seqs_str.npy', allow_pickle=True)
    print(f"Selected sequences: {len(selected_indices)}")

    seq_bytes = np.load('npy/human3/seq.npy', mmap_mode='r')

    model = model_v3_Collect_Atten(
        num_task=12,
        use_embedding=False,
        use_hierarchical=Config.use_hierarchical
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
    for rank, idx in enumerate(tqdm(selected_indices, desc="MultiRM sliding window")):
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
                        default='logs/old/multirm_rna_classification_20260203_230348/checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='npy/multirm_segmented_atten.npz')
    parser.add_argument('--window_size', type=int, default=51)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.window_size, args.stride, args.top_n)
