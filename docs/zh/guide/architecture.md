# 算法架构 / Algorithm Architecture

## 1. 整体数据流 / Overall Data Flow

```
RNA 序列 (B, 1001, 4) one-hot
        │
        ▼
┌──────────────────────┐
│   Embedding 层       │  one-hot → dense vector
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   M2D 多尺度 CNN     │  4 个并行的 1D 卷积 (k=1,3,5,7)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   GCN 图传播         │  多层 GCNConv + LayerNorm + 残差
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   MoHE 层级专家混合  │  4 个核苷酸组 → 12 个类查询
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   A2P 多锚点池化     │  类查询 ↔ 序列节点注意力
└──────────────────────┘
        │
        ▼
   12 类 logits (B, 12)
```

## 2. Embedding 模块 / Embedding Module

输入是固定长度 1001nt 的 RNA 序列 one-hot 编码 `(B, 1001, 4)`，其中 4 个通道分别对应 `A/C/G/U`（`N` 用零向量表示）。`npy/` 目录下的预处理脚本将原始 FASTA 文件转换为 `np.float32` 数组。

The input is a fixed-length 1001nt RNA one-hot `(B, 1001, 4)`, with 4 channels for `A/C/G/U` (`N` is zero). Preprocessing scripts in `npy/` convert raw FASTA files to `np.float32` arrays.

## 3. M2D - Multi-view Motif Discovery

**M2D** 是 mRModN 的局部特征提取模块，核心是 **ParallelCNNBlock**（见 [model/mrmodn.py](https://github.com/))，包含 4 个并行的 1D 卷积支路：

**M2D** is the local feature extraction module. Its core is **ParallelCNNBlock**, which contains 4 parallel 1D conv branches:

| 分支 / Branch | 核大小 / Kernel | 捕获模式 / Pattern |
|---|---|---|
| 1 | k=1 | 单核苷酸身份 / Single-nucleotide identity |
| 2 | k=3 | 三联体 motif (如 m6A 的 RRACH) / Triplet motif |
| 3 | k=5 | 五联体上下文 / Quintuplet context |
| 4 | k=7 | 长程局部依赖 / Long-range local dependency |

每个分支输出 `hidden_dim/4` 通道，拼接后经过 `LayerNorm → ReLU → Dropout` 产生最终特征。**结构嵌入**（RNAfold 预测的配对概率）作为额外通道一并输入。

Each branch outputs `hidden_dim/4` channels, concatenated and passed through `LayerNorm → ReLU → Dropout`. **Structure embedding** (RNAfold base-pairing probability) is fed as an extra channel.

## 4. MoHE - Mixture of Hierarchical Experts

MoHE 是 mRModN 的**核心创新**，由两个子模块组成：

MoHE is the **core innovation** of mRModN, comprising two sub-modules:

### 4.1 层级查询路由 (HQR, Hierarchical Query Routing)

将 12 类 RNA 修饰按碱基类型聚为 **4 组**：

12 RNA modifications are grouped by base type into **4 groups**:

| 组 / Group | 修饰 / Modifications |
|---|---|
| A (idx 0) | m6A, m1A, I |
| C (idx 1) | m5C, ac4C, m3C |
| G (idx 2) | m2G, m7G, Nm |
| U (idx 3) | Ψ, s2U, D |

每组共享一个**组级可学习查询 (group query)**，12 个**类级查询 (class queries)** 通过组级查询线性派生 (`class_query = Linear(group_query)`)，形成层级语义结构。

Each group shares a **group-level learnable query**, and the 12 **class-level queries** are linearly derived from their group query (`class_query = Linear(group_query)`), forming a hierarchical semantic structure.

### 4.2 层级交叉注意力 (HCA, Hierarchical Cross-Attention)

HCA 使用上述层级查询与 GCN 输出的节点特征做交叉注意力：

HCA uses the hierarchical queries to perform cross-attention with GCN node features:

```
attn = softmax(Q · K^T / sqrt(d))
output = attn · V
```

其中 `Q` 来自层级查询，`K`/`V` 来自 GCN 节点特征。

`Q` comes from hierarchical queries; `K`/`V` come from GCN node features.

## 5. A2P - Anchor-to-Positive Pooling

**A2P** 是分类头输出阶段的池化策略。传统做法是对序列做 `mean pooling` 或 `max pooling`，A2P 改用**可学习锚点 (anchors)** 与序列节点做注意力：

**A2P** is the pooling strategy at the classification head output. Traditional pooling (mean/max) is replaced by **learnable anchors** that attend to sequence nodes:

- 每个类维护一个锚点向量 (class query)
- 锚点与节点特征计算缩放点积注意力
- 注意力权重再以 **KL 散度** 与生物学先验（如 motif 位置）监督

Each class maintains an anchor vector. Anchor-to-node attention is supervised by a **KL divergence** loss against biological priors (e.g., motif positions).

## 6. ABS - Adaptive Balanced Sampler

12 类 RNA 修饰在数据集中天然存在**严重不平衡**（某些修饰样本数是其他修饰的 1/100）。mRModN 采用 **ABS 自适应平衡采样器**：

The 12 RNA modification classes are **heavily imbalanced** in real datasets (some are 1/100 of others). mRModN uses an **ABS Adaptive Balanced Sampler**:

- 每个 epoch 评估各类别的 F1
- 根据 F1 倒数为各类分配采样概率
- 长尾类别采样概率被放大，短头类别被压缩

Each epoch evaluates per-class F1. Sampling probabilities are inversely proportional to F1, amplifying long-tail and compressing short-head classes.

## 7. 损失函数 / Loss Function

总损失由三部分组成：

Total loss has three components:

```
L_total = L_cls + λ_4 · L_group + λ_kl · L_kl
```

- `L_cls` — 12 类交叉熵 / 12-class cross-entropy
- `L_group` — 4 组交叉熵（辅助损失）/ 4-group cross-entropy (auxiliary)
- `L_kl` — A2P 锚点注意力的 KL 散度 / KL divergence for A2P anchor attention

## 8. 模型变体 / Model Variants

| 变体 / Variant | 文件 / File | 特点 / Specialty |
|---|---|---|
| **mRModN** | `model/mrmodn.py` | 完整版 (M2D + MoHE + A2P + ABS) |
| **mRModN_collect_atten** | `model/mrmodn_collect_atten.py` | 收集注意力权重，可视化用 |
| **mRModN_multirm** | `model/mrmodn_multirm.py` | MultiRM 51nt 变体 |
| **MultiRM** | `model/multirm.py` | 多任务多修饰变体 |
| **ModX** | `model/modx.py` | 修饰类型消融变体 |
| **EvoRMD** | `model/evormd_human.py` | 集成进化特征 (Evo2) |
| **abla_model** | `model/abla_model.py` | 消融实验模型基线 |

## 9. 关键超参数 / Key Hyperparameters

| 参数 / Param | 默认值 / Default | 说明 / Description |
|---|---|---|
| `cnn_hidden_dim` | 64 | 多尺度 CNN 隐藏维度 |
| `gcn_hidden_dim` | 128 | GCN 隐藏维度 |
| `num_classes` | 12 | 分类数 |
| `num_gcn_layers` | 3 | GCN 层数 |
| `num_heads` | 4 | 多头注意力头数 |
| `dropout` | 0.1-0.3 | dropout 比率 |
| `seq_len` | 1001 | 序列长度 |
| `use_hierarchical` | True | 是否使用层级头 |
| `kernel_sizes` | (1,3,5,7) | 多尺度卷积核 |

## 10. 下一步 / Next Steps

- 阅读 [模型详解](/guide/model) 了解每个类与方法的实现细节
- 阅读 [训练流程](/guide/training) 了解训练循环

Read [Model Details](/guide/model) for per-class implementation; [Training Pipeline](/guide/training) for the training loop.
