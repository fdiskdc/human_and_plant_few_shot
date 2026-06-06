# Algorithm Architecture

## 1. Overall Data Flow

```
RNA sequence (B, 1001, 4) one-hot
        │
        ▼
┌──────────────────────┐
│   Embedding Layer    │  one-hot → dense vector
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   M2D Multi-scale CNN │  4 parallel 1D convs (k=1,3,5,7)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   GCN Graph Prop.     │  multi-layer GCNConv + LayerNorm + residual
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   MoHE Hierarchical  │  4 nucleotide groups → 12 class queries
│   Expert Mixture     │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   A2P Multi-anchor   │  class query ↔ sequence node attention
│   Pooling            │
└──────────────────────┘
        │
        ▼
   12-class logits (B, 12)
```

## 2. Embedding Module

The input is a fixed-length 1001nt RNA one-hot `(B, 1001, 4)`, with 4 channels for `A/C/G/U` (`N` is zero). Preprocessing scripts in `npy/` convert raw FASTA files to `np.float32` arrays.

## 3. M2D - Multi-view Motif Discovery

**M2D** is the local feature extraction module. Its core is **ParallelCNNBlock**, which contains 4 parallel 1D conv branches:

| Branch | Kernel | Pattern |
|---|---|---|
| 1 | k=1 | Single-nucleotide identity |
| 2 | k=3 | Triplet motif (e.g., m6A's RRACH) |
| 3 | k=5 | Quintuplet context |
| 4 | k=7 | Long-range local dependency |

Each branch outputs `hidden_dim/4` channels, concatenated and passed through `LayerNorm → ReLU → Dropout`. **Structure embedding** (RNAfold base-pairing probability) is fed as an extra channel.

## 4. MoHE - Mixture of Hierarchical Experts

MoHE is the **core innovation** of mRModN, comprising two sub-modules:

### 4.1 Hierarchical Query Routing (HQR)

The 12 RNA modifications are grouped by base type into **4 groups**:

| Group | Modifications |
|---|---|
| A (idx 0) | m6A, m1A, I |
| C (idx 1) | m5C, ac4C, m3C |
| G (idx 2) | m2G, m7G, Nm |
| U (idx 3) | Ψ, s2U, D |

Each group shares a **group-level learnable query**, and the 12 **class-level queries** are linearly derived from their group query (`class_query = Linear(group_query)`), forming a hierarchical semantic structure.

### 4.2 Hierarchical Cross-Attention (HCA)

HCA uses the hierarchical queries to perform cross-attention with GCN node features:

```
attn = softmax(Q · K^T / sqrt(d))
output = attn · V
```

`Q` comes from hierarchical queries; `K`/`V` come from GCN node features.

## 5. A2P - Anchor-to-Positive Pooling

**A2P** is the pooling strategy at the classification head output. Traditional pooling (mean/max) is replaced by **learnable anchors** that attend to sequence nodes:

- Each class maintains an anchor vector (class query)
- Anchors compute scaled dot-product attention with node features
- Attention weights are further supervised by **KL divergence** against biological priors (e.g., motif positions)

## 6. ABS - Adaptive Balanced Sampler

The 12 RNA modification classes are **heavily imbalanced** in real datasets (some are 1/100 of others). mRModN uses an **ABS Adaptive Balanced Sampler**:

- Each epoch evaluates per-class F1
- Sampling probabilities are inversely proportional to F1
- Long-tail classes are oversampled; short-head classes are undersampled

## 7. Loss Function

Total loss has three components:

```
L_total = L_cls + λ_4 · L_group + λ_kl · L_kl
```

- `L_cls` — 12-class cross-entropy
- `L_group` — 4-group cross-entropy (auxiliary)
- `L_kl` — KL divergence for A2P anchor attention

## 8. Model Variants

| Variant | File | Specialty |
|---|---|---|
| **mRModN** | `model/mrmodn.py` | Full version (M2D + MoHE + A2P + ABS) |
| **mRModN_collect_atten** | `model/mrmodn_collect_atten.py` | Collects attention weights for visualization |
| **mRModN_multirm** | `model/mrmodn_multirm.py` | MultiRM 51nt variant |
| **MultiRM** | `model/multirm.py` | Multi-task multi-modification variant |
| **ModX** | `model/modx.py` | Modification-type ablation variant |
| **EvoRMD** | `model/evormd_human.py` | Integrates evolutionary features (Evo2) |
| **abla_model** | `model/abla_model.py` | Ablation experiment model baseline |

## 9. Key Hyperparameters

| Param | Default | Description |
|---|---|---|
| `cnn_hidden_dim` | 64 | Multi-scale CNN hidden dim |
| `gcn_hidden_dim` | 128 | GCN hidden dim |
| `num_classes` | 12 | Number of classes |
| `num_gcn_layers` | 3 | Number of GCN layers |
| `num_heads` | 4 | Number of MHA heads |
| `dropout` | 0.1-0.3 | Dropout rate |
| `seq_len` | 1001 | Sequence length |
| `use_hierarchical` | True | Whether to use hierarchical head |
| `kernel_sizes` | (1,3,5,7) | Multi-scale kernels |

## 10. Next Steps

Read [Model Details](/en/guide/model) for per-class implementation; [Training Pipeline](/en/guide/training) for the training loop.
