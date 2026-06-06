# 模型详解 / Model Details

> 本章以 `model/mrmodn.py` 为主线，详解 mRModN 端到端模型的每个核心组件。
> This chapter walks through the core components of the mRModN end-to-end model in `model/mrmodn.py`.

## 1. ParallelCNNBlock - 多尺度卷积块 / Multi-scale CNN Block

**位置 / Location**: `model/mrmodn.py` 类 `ParallelCNNBlock` (line 64)

```python
class ParallelCNNBlock(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, kernel_sizes=(1,3,5,7), ...):
        # 4 个并行的 nn.Conv1d 分支
        # 拼接后: LayerNorm → ReLU → Dropout
```

**输入 / Input**: `(B, 1001, 4)` one-hot 或 `(Total_Nodes, 4)` 平铺张量
**输出 / Output**: `(Total_Nodes, hidden_dim)` 节点级特征

### 关键点 / Key Points

- 每个分支 `padding='same'`，保持序列长度不变
- 4 个分支输出 `hidden_dim/4` 通道，拼接得到 `hidden_dim`
- `LayerNorm` 在 `(C, L)` 维度上做归一化
- 训练时使用 `Dropout(0.1)` 提升泛化

## 2. GCNBlock - 图卷积块 / Graph Convolutional Block

**位置 / Location**: `model/mrmodn.py` 类 `GCNBlock` (line 139)

```python
class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_layers=3, use_residual=True):
        # 多层 GCNConv + LayerNorm + 残差
```

**输入 / Input**: 节点特征 `(Total_Nodes, in_channels)` + 边索引 `(2, E)`
**输出 / Output**: `(Total_Nodes, out_channels)`

### 关键点 / Key Points

- 3 层 GCN，每层后接 `LayerNorm` + `ReLU` + `Dropout(0.3)`
- 残差连接（`use_residual=True`）缓解过平滑
- 最后一层不做激活，仅做归一化

## 3. ClassQueryHead - 类查询注意力头 / Class-Query Attention Head

**位置 / Location**: `model/mrmodn.py` 类 `ClassQueryHead` (line 204)

```python
class ClassQueryHead(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=12, num_heads=4, use_decoder=True):
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))
        if use_decoder:
            self.cross_attention = nn.TransformerDecoder(...)
        else:
            self.cross_attention = nn.MultiheadAttention(...)
```

**输入 / Input**: GCN 节点特征 `(Total_Nodes, hidden_dim)` + `batch` 索引
**输出 / Output**: 12 类 logits `(B, 12)`

### 关键点 / Key Points

- `class_queries`：12 个可学习的 `(hidden_dim,)` 参数
- `use_decoder=True` 时使用 `TransformerDecoder`（带 FFN），否则使用纯 `MultiheadAttention`
- `prune_heads()` 可在迁移到少样本场景时剪枝

## 4. ClassQueryHeadPooling - 简化池化头 / Simplified Pooling Head

**位置 / Location**: `model/mrmodn.py` 类 `ClassQueryHeadPooling`

```python
class ClassQueryHeadPooling(nn.Module):
    # 缩放点积注意力 + 12 类输出
    # 返回 (logits, attn_weights)
```

**输入 / Input**: GCN 节点特征 + `batch`
**输出 / Output**: `(logits (B,12), attn_weights (B,12,1001))`

### 关键点 / Key Points

- 返回**注意力权重**，便于可视化和解释
- 适合 `collect_atten_*.py` 特征/注意力收集脚本

## 5. HierarchicalClassQueryHeadPooling - 层级查询头 / Hierarchical Query Head

**位置 / Location**: `model/mrmodn.py` 类 `HierarchicalClassQueryHeadPooling`

**输入 / Input**: GCN 节点特征 + `batch`
**输出 / Output**: `(logits_12 (B,12), logits_4 (B,4), attn_weights_12 (B,12,1001))`

### 关键点 / Key Points

- 4 组可学习组查询，每组对应 3 个类
- 12 个类查询由组查询线性派生
- 同时输出 4 组 logits 作为辅助监督信号

## 6. RNA_ClassQuery_Model - 端到端整合 / End-to-End Integration

**位置 / Location**: `model/mrmodn.py` 类 `RNA_ClassQuery_Model`

```python
class RNA_ClassQuery_Model(nn.Module):
    def __init__(self, num_classes=12, use_hierarchical=True):
        self.cnn = ParallelCNNBlock(...)
        self.gcn = GCNBlock(...)
        self.head = (HierarchicalClassQueryHeadPooling(...)
                     if use_hierarchical else ClassQueryHead(...))
```

**输入 / Input**: `(B, 1001, 4)` one-hot + `(2, E)` 边索引 + `(Total_Nodes,)` batch
**输出 / Output**: 取决于头部模式

## 7. 变体对比 / Variant Comparison

| 变体 / Variant | 主要改动 / Main Change | 文件 / File |
|---|---|---|
| mRModN (base) | 完整 M2D + GCN + 层级头 | `model/mrmodn.py` |
| mRModN_collect_atten | 显式返回注意力权重 | `model/mrmodn_collect_atten.py` |
| mRModN_multirm | 51nt 窗口 + 多任务头 | `model/mrmodn_multirm.py` |
| MultiRM | 多任务多修饰并行 | `model/multirm.py` |
| ModX | 单修饰类型消融 | `model/modx.py` |
| EvoRMD | 集成 Evo2 进化特征 | `model/evormd_human.py` |
| abla_model | 消融基线 (无 M2D/MoHE) | `model/abla_model.py` |

## 8. 关键参数详解 / Key Parameter Reference

| 参数 / Param | 推荐值 / Suggested | 说明 / Description |
|---|---|---|
| `cnn_hidden_dim` | 64 | 太小欠拟合，太大过拟合 |
| `gcn_hidden_dim` | 128 | GCN 隐藏层维度 |
| `num_gcn_layers` | 3 | 太多导致过平滑 |
| `num_heads` | 4 | 必须是 `hidden_dim` 的因数 |
| `dropout` | 0.1-0.3 | CNN 端 0.1，GCN 端 0.3 |
| `use_layer_norm` | True | 推荐使用 LayerNorm |
| `use_residual` | True | GCN 必须开 |
| `use_decoder` | True | TransformerDecoder 优于纯 MHA |
| `use_hierarchical` | True | 主模型默认开启 |

## 9. 调试技巧 / Debugging Tips

- **NaN Loss**: 降低学习率到 5e-4，检查输入是否有 `N` 以外的非法字符
- **不收敛**: 检查数据增强是否过强，禁用 ABS 采样器试试
- **显存爆**: 减小 `gcn_hidden_dim` 到 64，关闭 `use_decoder`
- **注意力全 0**: 检查 `K`/`V` 维度匹配，确认 `batch` 索引对齐

## 10. 下一步 / Next Steps

- 了解训练循环：[训练流程](/guide/training)
- 了解推理脚本：[推理流程](/guide/inference)
