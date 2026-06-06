# Model Details

> This chapter walks through the core components of the mRModN end-to-end model in `model/mrmodn.py`.

## 1. ParallelCNNBlock - Multi-scale CNN Block

**Location**: `model/mrmodn.py` class `ParallelCNNBlock` (line 64)

```python
class ParallelCNNBlock(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, kernel_sizes=(1,3,5,7), ...):
        # 4 parallel nn.Conv1d branches
        # Concat then: LayerNorm → ReLU → Dropout
```

**Input**: `(B, 1001, 4)` one-hot or `(Total_Nodes, 4)` flat tensor
**Output**: `(Total_Nodes, hidden_dim)` node-level features

### Key Points

- Each branch uses `padding='same'` to preserve sequence length
- 4 branches output `hidden_dim/4` channels each, concatenated to `hidden_dim`
- `LayerNorm` normalizes over `(C, L)` dimensions
- `Dropout(0.1)` improves generalization

## 2. GCNBlock - Graph Convolutional Block

**Location**: `model/mrmodn.py` class `GCNBlock` (line 139)

```python
class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_layers=3, use_residual=True):
        # Multi-layer GCNConv + LayerNorm + residual
```

**Input**: node features `(Total_Nodes, in_channels)` + edge indices `(2, E)`
**Output**: `(Total_Nodes, out_channels)`

### Key Points

- 3 GCN layers, each followed by `LayerNorm` + `ReLU` + `Dropout(0.3)`
- Residual connections (`use_residual=True`) to mitigate over-smoothing
- The last layer only normalizes (no activation)

## 3. ClassQueryHead - Class-Query Attention Head

**Location**: `model/mrmodn.py` class `ClassQueryHead` (line 204)

```python
class ClassQueryHead(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=12, num_heads=4, use_decoder=True):
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))
        if use_decoder:
            self.cross_attention = nn.TransformerDecoder(...)
        else:
            self.cross_attention = nn.MultiheadAttention(...)
```

**Input**: GCN node features `(Total_Nodes, hidden_dim)` + `batch` index
**Output**: 12-class logits `(B, 12)`

### Key Points

- `class_queries`: 12 learnable `(hidden_dim,)` parameters
- `use_decoder=True` uses `TransformerDecoder` (with FFN); otherwise pure `MultiheadAttention`
- `prune_heads()` can be used for few-shot transfer

## 4. ClassQueryHeadPooling - Simplified Pooling Head

**Location**: `model/mrmodn.py` class `ClassQueryHeadPooling`

```python
class ClassQueryHeadPooling(nn.Module):
    # Scaled dot-product attention + 12-class output
    # Returns (logits, attn_weights)
```

**Input**: GCN node features + `batch`
**Output**: `(logits (B,12), attn_weights (B,12,1001))`

### Key Points

- Returns **attention weights** for visualization and explanation
- Suitable for `collect_atten_*.py` feature/attention collection scripts

## 5. HierarchicalClassQueryHeadPooling - Hierarchical Query Head

**Location**: `model/mrmodn.py` class `HierarchicalClassQueryHeadPooling`

**Input**: GCN node features + `batch`
**Output**: `(logits_12 (B,12), logits_4 (B,4), attn_weights_12 (B,12,1001))`

### Key Points

- 4 learnable group queries, each corresponding to 3 classes
- 12 class queries are linearly derived from group queries
- Outputs 4-group logits as auxiliary supervision signal

## 6. RNA_ClassQuery_Model - End-to-End Integration

**Location**: `model/mrmodn.py` class `RNA_ClassQuery_Model`

```python
class RNA_ClassQuery_Model(nn.Module):
    def __init__(self, num_classes=12, use_hierarchical=True):
        self.cnn = ParallelCNNBlock(...)
        self.gcn = GCNBlock(...)
        self.head = (HierarchicalClassQueryHeadPooling(...)
                     if use_hierarchical else ClassQueryHead(...))
```

**Input**: `(B, 1001, 4)` one-hot + `(2, E)` edge indices + `(Total_Nodes,)` batch
**Output**: depends on head mode

## 7. Variant Comparison

| Variant | Main Change | File |
|---|---|---|
| mRModN (base) | Full M2D + GCN + hierarchical head | `model/mrmodn.py` |
| mRModN_collect_atten | Explicitly returns attention weights | `model/mrmodn_collect_atten.py` |
| mRModN_multirm | 51nt window + multi-task head | `model/mrmodn_multirm.py` |
| MultiRM | Multi-task multi-modification | `model/multirm.py` |
| ModX | Single-modification ablation | `model/modx.py` |
| EvoRMD | Integrates Evo2 evolutionary features | `model/evormd_human.py` |
| abla_model | Ablation baseline (no M2D/MoHE) | `model/abla_model.py` |

## 8. Key Parameter Reference

| Param | Suggested | Description |
|---|---|---|
| `cnn_hidden_dim` | 64 | Too small → underfit; too large → overfit |
| `gcn_hidden_dim` | 128 | GCN hidden dim |
| `num_gcn_layers` | 3 | Too many → over-smoothing |
| `num_heads` | 4 | Must be a factor of `hidden_dim` |
| `dropout` | 0.1-0.3 | 0.1 at CNN, 0.3 at GCN |
| `use_layer_norm` | True | Recommended |
| `use_residual` | True | Required for GCN |
| `use_decoder` | True | TransformerDecoder > pure MHA |
| `use_hierarchical` | True | Default for main model |

## 9. Debugging Tips

- **NaN Loss**: Lower learning rate to 5e-4; check input for invalid characters
- **No convergence**: Check data augmentation strength; try disabling ABS
- **GPU OOM**: Reduce `gcn_hidden_dim` to 64; turn off `use_decoder`
- **All-zero attention**: Check `K`/`V` dim match; verify `batch` index alignment

## 10. Next Steps

- Training loop: [Training Pipeline](/en/guide/training)
- Inference scripts: [Inference Pipeline](/en/guide/inference)
