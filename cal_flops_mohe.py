"""
cal_flops_mohe.py - 16消融配置 FLOPs 计算 (无外部库) / FLOPs Counter for 16 Ablation Configs (No External Lib)

手工计算 16 个消融配置 (4 query types x 4 dims) 的 FLOPs,无需外部 FLOPs 库。
统计 Linear、Conv1d、GCNConv、MultiheadAttention、TransformerEncoder 层的乘加运算。
Manual FLOPs counter for 16 ablation configs (4 query types x 4 dims), no external FLOPs library.
Counts multiply-accumulate ops for Linear, Conv1d, GCNConv, MHA, TransformerEncoder.

功能模块 / Modules:
- 手工 FLOPs 计数器 / Manual FLOPs counter
- 16 配置循环 / 16-config loop
- CSV 输出 / CSV output
- main: 主入口 / Main entry point

输入 / Inputs:
- json/abla_human.json: 消融配置 / Ablation config
- 命令行参数 / CLI: --config

输出 / Outputs:
- logs_abla/flops_results.csv: FLOPs 结果 CSV / FLOPs results CSV
- 终端详细输出 / Detailed terminal output

数据流 / Data Flow:
1. 加载配置 / Load config
2. 16 配置循环 / 16-config loop
3. 构造模型并计算 FLOPs / Build model and compute FLOPs
4. 累加每层 FLOPs / Sum per-layer FLOPs
5. 输出 CSV / Output CSV

相关文件 / Related Files:
- 调用 / Calls: model.abla_model.AblationModel
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python cal_flops_mohe.py
    python cal_flops_mohe.py --config json/abla_human.json

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import sys
import csv
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from model.abla_model import AblationModel

# ============================================================================
# Manual FLOPs counter
# ============================================================================

def _count_linear(module, input_shape, output_shape):
    """FLOPs for nn.Linear: 2 * in_features * out_features (+ bias)"""
    in_f = module.in_features
    out_f = module.out_features
    macs = in_f * out_f
    if module.bias is not None:
        macs += out_f  # bias addition
    return 2 * macs


def _count_conv1d(module, input_shape, output_shape):
    """FLOPs for nn.Conv1d: 2 * C_in * C_out * K * L_out"""
    c_in = module.in_channels
    c_out = module.out_channels
    k = module.kernel_size[0]
    # output spatial length from output_shape
    l_out = output_shape[-1]
    macs = c_in * c_out * k * l_out
    if module.bias is not None:
        macs += c_out * l_out
    return 2 * macs


def _count_layernorm(module, input_shape, output_shape):
    """LayerNorm: ~5 * N (mean, var, normalize, scale, shift)"""
    n = 1
    for s in input_shape:
        n *= s
    return 5 * n


def _count_batchnorm(module, input_shape, output_shape):
    """BatchNorm: ~5 * N"""
    n = 1
    for s in input_shape:
        n *= s
    return 5 * n


def _count_gcnconv(module, num_nodes, num_edges):
    """
    GCNConv internally uses a linear transform (module.lin) for x @ W,
    plus sparse aggregation (negligible FLOPs).
    FLOPs ≈ 2 * in_channels * out_channels * num_nodes
    """
    if hasattr(module, 'lin') and module.lin is not None:
        in_f = module.in_channels
        out_f = module.out_channels
        return 2 * in_f * out_f * num_nodes
    return 0


def _count_mha(seq_len, embed_dim, num_heads):
    """
    MultiheadAttention FLOPs:
    - Q, K, V projections: 3 * (2 * embed_dim^2 * seq_len)
    - Attention scores: 2 * seq_len^2 * embed_dim
    - Attention * V: 2 * seq_len^2 * embed_dim
    - Output projection: 2 * embed_dim^2 * seq_len
    Total = 8 * embed_dim^2 * seq_len + 4 * seq_len^2 * embed_dim
    """
    proj = 4 * (2 * embed_dim * embed_dim * seq_len)  # Q, K, V, out
    attn = 2 * (2 * seq_len * seq_len * embed_dim)    # score + weighted sum
    return proj + attn


def _count_transformer_encoder(num_layers, seq_len, d_model, dim_ff):
    """
    TransformerEncoderLayer FLOPs per layer:
    - Self-attention: 8 * d_model^2 * seq_len + 4 * seq_len^2 * d_model
    - FFN: 2 * (d_model * dim_ff + dim_ff * d_model) * seq_len = 4 * d_model * dim_ff * seq_len
    - LayerNorm (x2): ~10 * d_model * seq_len (negligible)
    """
    attn = 8 * d_model * d_model * seq_len + 4 * seq_len * seq_len * d_model
    ffn = 4 * d_model * dim_ff * seq_len
    return num_layers * (attn + ffn)


def count_model_flops(model, seq_len=1001, batch_size=1, override_head_dim=None):
    """
    Count FLOPs for one forward pass of AblationModel.

    Args:
        model: AblationModel instance
        seq_len: sequence length (nodes per sample)
        batch_size: batch size (FLOPs scale linearly)
        override_head_dim: if set, use this dim for head computation instead of
                           model's hidden_dim. Used for fullattn to make
                           group_query_dim effective in FLOPs calculation.
    """
    total_flops = 0
    num_nodes = seq_len

    query_type = model.query_type

    # --- 1. CNN Block ---
    cnn = model.cnn_block
    for conv in cnn.conv_branches:
        cin = conv.in_channels
        cout = conv.out_channels
        k = conv.kernel_size[0]
        macs = cin * cout * k * seq_len
        if conv.bias is not None:
            macs += cout * seq_len
        total_flops += 2 * macs

    # CNN norm
    if isinstance(cnn.norm, (nn.LayerNorm, nn.BatchNorm1d)):
        total_flops += 5 * cnn.out_channels * seq_len

    # --- 2. GCN Block (if exists) ---
    if model.gcn_block is not None:
        gcn = model.gcn_block
        if gcn.input_proj is not None:
            total_flops += _count_linear(gcn.input_proj, None, None)

        for i, gcn_layer in enumerate(gcn.gcn_layers):
            in_ch = gcn_layer.in_channels
            out_ch = gcn_layer.out_channels
            total_flops += 2 * in_ch * out_ch * num_nodes
            norm = gcn.norms[i]
            if isinstance(norm, nn.LayerNorm):
                total_flops += 5 * out_ch * num_nodes

    # --- 3. Backbone projection (1query only) ---
    if model.backbone_proj is not None:
        total_flops += _count_linear(model.backbone_proj, None, None)

    # --- 4. Classification Head ---
    head = model.class_query_head

    if query_type == "1query":
        total_flops += _count_linear(head.classifier_12, None, None) * num_nodes
        total_flops += _count_linear(head.classifier_4, None, None)

    elif query_type == "4query":
        hidden = head.hidden_dim
        if not isinstance(head.query_proj, nn.Identity):
            total_flops += _count_linear(head.query_proj, None, None)
        for proj in head.group_projectors:
            for m in proj:
                if isinstance(m, nn.Linear):
                    total_flops += _count_linear(m, None, None)
        total_flops += _count_mha(seq_len, hidden, head.num_heads)
        total_flops += _count_mha(seq_len, hidden, head.num_heads)
        for m in head.output_proj_12:
            if isinstance(m, nn.Linear):
                total_flops += _count_linear(m, None, None)
        for m in head.output_proj_4:
            if isinstance(m, nn.Linear):
                total_flops += _count_linear(m, None, None)

    elif query_type == "12query":
        hidden = head.hidden_dim
        num_heads = head.per_class_mha[0].num_heads
        if not isinstance(head.query_proj, nn.Identity):
            total_flops += _count_linear(head.query_proj, None, None)
        for i in range(head.num_classes):
            total_flops += _count_mha(seq_len, hidden, num_heads)
            for m in head.per_class_proj[i]:
                if isinstance(m, nn.Linear):
                    total_flops += _count_linear(m, None, None)
        for m in head.output_proj_4:
            if isinstance(m, nn.Linear):
                total_flops += _count_linear(m, None, None)

    elif query_type == "fullattn":
        # Use override_head_dim (group_query_dim) as d_model for TransformerEncoder
        d_model = override_head_dim if override_head_dim else head.hidden_dim
        dim_ff = d_model * 4
        total_flops += _count_transformer_encoder(
            num_layers=8, seq_len=seq_len, d_model=d_model, dim_ff=dim_ff
        )
        # classifiers: LayerNorm + Linear(d_model -> d_model//2) + ReLU + Linear(d_model//2 -> 12/4)
        for out_dim in [12, 4]:
            total_flops += 5 * d_model * seq_len  # LayerNorm (over seq)
            total_flops += 2 * d_model * (d_model // 2) * seq_len  # Linear
            total_flops += 2 * (d_model // 2) * out_dim * seq_len   # Linear

    # Scale by batch size
    total_flops *= batch_size

    return total_flops


def count_fullattn_params(group_query_dim, model_cfg):
    """Compute parameter count for fullattn head using group_query_dim as d_model."""
    d = group_query_dim
    d_ff = d * 4
    num_classes = model_cfg["num_classes"]
    num_heads = model_cfg["num_attn_heads"]

    # --- Backbone (same for all fullattn configs) ---
    cnn_hidden = model_cfg["cnn_hidden_dim"]
    kernel_sizes = model_cfg["cnn_kernel_sizes"]
    branch_out = cnn_hidden // len(kernel_sizes)
    backbone = 0
    # Conv1d branches
    for k in kernel_sizes:
        backbone += 4 * branch_out * k + branch_out  # weight + bias
    # LayerNorm
    backbone += 2 * cnn_hidden  # weight + bias
    # GCN: input_proj
    backbone += cnn_hidden * model_cfg["gcn_hidden_dim"] + model_cfg["gcn_hidden_dim"]
    # GCN layers
    gcn_hidden = model_cfg["gcn_hidden_dim"]
    gcn_out = model_cfg["gcn_out_channels"]
    for i in range(model_cfg["gcn_num_layers"]):
        in_ch = gcn_hidden if i > 0 else gcn_hidden
        out_ch = gcn_out if i == model_cfg["gcn_num_layers"] - 1 else gcn_hidden
        backbone += in_ch * out_ch + out_ch  # GCNConv weight + bias (via lin)
        backbone += 2 * out_ch  # LayerNorm

    # --- Head (d_model = group_query_dim) ---
    head = 0
    # TransformerEncoder: 8 layers
    for _ in range(8):
        # Self-attention: Q, K, V, out projections
        head += 4 * (d * d + d)  # weight + bias
        # FFN: Linear(d, d_ff) + Linear(d_ff, d)
        head += d * d_ff + d_ff + d_ff * d + d
        # 2 LayerNorm per layer
        head += 4 * d  # weight + bias * 2 norms

    # classifier_12: LayerNorm + Linear(d, d//2) + Linear(d//2, 12)
    head += 2 * d + d * (d // 2) + (d // 2) + (d // 2) * 12 + 12
    # classifier_4: LayerNorm + Linear(d, d//2) + Linear(d//2, 4)
    head += 2 * d + d * (d // 2) + (d // 2) + (d // 2) * 4 + 4

    return backbone + head


# ============================================================================
# Analytical helpers for fullattn (group_query_dim as d_model)
# ============================================================================

def _count_backbone_flops(model_cfg, seq_len=1001):
    """CNN + GCN backbone FLOPs (shared by all query_types with full backbone)."""
    flops = 0
    cnn_hidden = model_cfg["cnn_hidden_dim"]
    kernel_sizes = model_cfg["cnn_kernel_sizes"]
    branch_out = cnn_hidden // len(kernel_sizes)

    # Conv1d branches
    for k in kernel_sizes:
        macs = 4 * branch_out * k * seq_len + branch_out * seq_len
        flops += 2 * macs
    # LayerNorm
    flops += 5 * cnn_hidden * seq_len

    # GCN input_proj (64 -> 128)
    gcn_hidden = model_cfg["gcn_hidden_dim"]
    gcn_out = model_cfg["gcn_out_channels"]
    flops += 2 * cnn_hidden * gcn_hidden  # Linear

    # GCN layers
    for i in range(model_cfg["gcn_num_layers"]):
        in_ch = gcn_hidden
        out_ch = gcn_out if i == model_cfg["gcn_num_layers"] - 1 else gcn_hidden
        flops += 2 * in_ch * out_ch * seq_len  # GCNConv
        flops += 5 * out_ch * seq_len           # LayerNorm

    return flops


def count_fullattn_head_flops(d_model, seq_len=1001, num_classes=12):
    """TransformerEncoder head FLOPs with arbitrary d_model."""
    dim_ff = d_model * 4
    flops = _count_transformer_encoder(num_layers=8, seq_len=seq_len,
                                        d_model=d_model, dim_ff=dim_ff)
    # classifier_12: LN + Linear(d, d//2) + Linear(d//2, 12)
    flops += 5 * d_model * seq_len
    flops += 2 * d_model * (d_model // 2) * seq_len
    flops += 2 * (d_model // 2) * num_classes * seq_len
    # classifier_4: LN + Linear(d, d//2) + Linear(d//2, 4)
    flops += 5 * d_model * seq_len
    flops += 2 * d_model * (d_model // 2) * seq_len
    flops += 2 * (d_model // 2) * 4 * seq_len
    return flops


# ============================================================================
# Ablation configurations
# ============================================================================

QUERY_TYPES = ["1query", "4query", "12query", "fullattn"]
QUERY_DIMS  = [128, 256, 512, 1001]

QUERY_TYPE_DESC = {
    "1query":   "1 global query + 12-way MLP",
    "4query":   "4-group hierarchical (baseline)",
    "12query":  "12 independent queries",
    "fullattn": "12 queries + self-attention",
}


def main(config_path='json/abla_human.json'):
    print("=" * 70)
    print("  FLOPs Calculation for 16 Ablation Configurations")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    device = torch.device('cpu')  # FLOPs counting on CPU
    model_cfg = config_dict["model"]

    results = []

    for qt in QUERY_TYPES:
        for qd in QUERY_DIMS:
            cfg_name = f"{qt}_d{qd}"
            print(f"\n  [{cfg_name}] {QUERY_TYPE_DESC[qt]}, dim={qd}")

            try:
                if qt == "fullattn":
                    # fullattn: group_query_dim acts as TransformerEncoder d_model
                    # Compute params & FLOPs analytically (model ignores group_query_dim)
                    total_params = count_fullattn_params(qd, model_cfg)
                    trainable_params = total_params
                    flops = _count_backbone_flops(model_cfg, seq_len=1001)
                    flops += count_fullattn_head_flops(qd, seq_len=1001, num_classes=model_cfg["num_classes"])
                else:
                    model = AblationModel(
                        query_type=qt,
                        group_query_dim=qd,
                        cnn_hidden_dim=model_cfg["cnn_hidden_dim"],
                        cnn_kernel_sizes=tuple(model_cfg["cnn_kernel_sizes"]),
                        cnn_dropout=model_cfg["cnn_dropout"],
                        gcn_hidden_dim=model_cfg["gcn_hidden_dim"],
                        gcn_out_channels=model_cfg["gcn_out_channels"],
                        gcn_num_layers=model_cfg["gcn_num_layers"],
                        gcn_dropout=model_cfg["gcn_dropout"],
                        num_classes=model_cfg["num_classes"],
                        num_attn_heads=model_cfg["num_attn_heads"],
                        attn_dropout=model_cfg["attn_dropout"],
                        use_layer_norm=model_cfg["use_layer_norm"],
                        seq_len=1001
                    ).to(device)

                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    flops = count_model_flops(model, seq_len=1001, batch_size=1)

                flops_m = flops / 1e6
                flops_g = flops / 1e9

                print(f"    Params: {total_params:,} | FLOPs: {flops_m:,.1f}M ({flops_g:.3f}G)")

                results.append({
                    "name": cfg_name,
                    "query_type": qt,
                    "group_query_dim": qd,
                    "desc": f"{QUERY_TYPE_DESC[qt]}, dim={qd}",
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "flops": flops,
                    "flops_m": round(flops_m, 2),
                    "flops_g": round(flops_g, 4),
                })

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"    [OOM] Skipping {cfg_name}")
                    torch.cuda.empty_cache()
                else:
                    raise

    # Write CSV
    output_dir = "logs_abla"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "flops_results.csv")

    fieldnames = ["name", "query_type", "group_query_dim", "desc",
                  "total_params", "trainable_params", "flops", "flops_m", "flops_g"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Config':<18} {'Query':<10} {'Dim':>5} {'Params':>12} {'FLOPs(M)':>12} {'FLOPs(G)':>10}")
    print(f"{'-' * 90}")
    for r in results:
        print(f"{r['name']:<18} {r['query_type']:<10} {r['group_query_dim']:>5d} "
              f"{r['total_params']:>12,} {r['flops_m']:>12.1f} {r['flops_g']:>10.4f}")
    print(f"{'=' * 90}")
    print(f"\nResults saved to: {csv_path}")

    # Print 4x4 heatmap-style summary
    print(f"\n{'=' * 55}")
    print("  FLOPs (M) — 4x4 Summary")
    print(f"{'=' * 55}")
    header = f"{'':>10}" + "".join(f"{d:>12}" for d in QUERY_DIMS)
    print(header)
    print(f"{'-' * 55}")
    for qt in QUERY_TYPES:
        row = f"{qt:>10}"
        for qd in QUERY_DIMS:
            r = next((x for x in results if x['query_type'] == qt and x['group_query_dim'] == qd), None)
            if r:
                row += f"{r['flops_m']:>12.1f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='json/abla_human.json')
    args = parser.parse_args()
    main(args.config)
