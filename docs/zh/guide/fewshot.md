# 少/零样本分析 / Few/Zero-shot Analysis

## 1. 背景 / Background

RNA 修饰的实验鉴定成本高昂，导致**许多修饰类型的标注样本极少**（如某些稀有修饰可能仅有 10-100 个阳性样本）。mRModN 通过**少样本学习**（Few-shot Learning）和**零样本迁移**（Zero-shot Transfer）来应对这一挑战。

Experimental identification of RNA modifications is expensive, so **many modification types have very few labeled samples** (e.g., rare modifications may have only 10-100 positive samples). mRModN tackles this via **Few-shot Learning** and **Zero-shot Transfer**.

## 2. 核心算法 / Core Algorithm: `utils/few_shot.py`

**文件 / File**: `utils/few_shot.py`

核心组件 / Core components:

| 组件 / Component | 类 / Class | 作用 / Purpose |
|---|---|---|
| Prototypical Networks | `PrototypicalNetwork` | 原型网络少样本学习 |
| Adaptive Balanced Sampler | `AdaptiveBalancedSampler` | 自适应平衡采样 |
| N-way K-shot Sampler | `NWayKShotSampler` | N 类 K 样本采样器 |
| Episode Sampler | `EpisodeSampler` | Episode 采样 |

### 2.1 原型网络 / Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        self.encoder = encoder  # mRModN 编码器（去掉分类头）
        self.hidden_dim = hidden_dim

    def forward(self, support_x, support_y, query_x, n_way, k_shot):
        # 1. 编码支持集和查询集
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)

        # 2. 计算类原型（支持集各类均值）
        prototypes = []
        for c in range(n_way):
            prototypes.append(support_emb[support_y == c].mean(dim=0))
        prototypes = torch.stack(prototypes)

        # 3. 查询样本到原型的距离
        dists = -torch.cdist(query_emb, prototypes)  # (Q, n_way)
        return F.log_softmax(dists, dim=-1)
```

### 2.2 自适应平衡采样器 / Adaptive Balanced Sampler

```python
class AdaptiveBalancedSampler:
    """
    根据各类 F1 倒数动态调整采样概率
    Dynamically adjusts sampling probabilities based on inverse per-class F1
    """
    def __init__(self, dataset, num_classes, beta=0.999):
        self.dataset = dataset
        self.num_classes = num_classes
        self.beta = beta
        self.f1_history = torch.ones(num_classes)  # 初始 F1=1

    def update_f1(self, f1_per_class):
        # EMA 平滑
        self.f1_history = (
            self.beta * self.f1_history + (1 - self.beta) * f1_per_class
        )

    def get_weights(self):
        # 权重 = 1 / F1
        weights = 1.0 / (self.f1_history + 1e-6)
        # 归一化
        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        weights = self.get_weights()
        # 按样本所属类别分配权重
        sample_weights = torch.tensor([
            weights[self.dataset.label_of(i)]
            for i in range(len(self.dataset))
        ])
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(self.dataset), replacement=True
        )
        yield from iter(sampler)
```

## 3. ac4C 平衡少样本 / ac4C Balanced Few-shot

**文件 / File**: `fewshot_ac4c_mrmodn_balance.py`

**任务**: 在 5-shot / 10-shot / 20-shot 三个规模下评估 ac4C 修饰预测

**Task**: Evaluate ac4C modification prediction at 5-shot / 10-shot / 20-shot scales.

**特点 / Specialty**: **平衡采样** — 每类采样数量相等

**Balanced sampling** — equal number of samples per class.

```bash
python fewshot_ac4c_mrmodn_balance.py --n-shot 5 --n-way 2
python fewshot_ac4c_mrmodn_balance.py --n-shot 10 --n-way 2
python fewshot_ac4c_mrmodn_balance.py --n-shot 20 --n-way 2
```

**输出 / Output**: `output/fewshot/ac4c_balance_{n_shot}shot.csv`

| 规模 / Shots | ACC | F1 | AUC |
|---|---|---|---|
| 5-shot | 0.71 | 0.68 | 0.79 |
| 10-shot | 0.78 | 0.76 | 0.84 |
| 20-shot | 0.84 | 0.82 | 0.89 |

## 4. ac4C 非平衡少样本 / ac4C Unbalanced Few-shot

**文件 / File**: `fewshot_ac4c_mrmodn_unbalan.py`

**任务**: 模拟真实场景，正负样本比例 1:10

**Task**: Simulates real-world scenario with 1:10 positive-negative ratio.

```bash
python fewshot_ac4c_mrmodn_unbalan.py --n-shot 5 --neg-ratio 10
```

**特点**: 启用 ABS 采样器处理不平衡

**Specialty**: ABS sampler handles imbalance.

## 5. Plant 三向独立少样本 / Plant 3-way Independent Few-shot

**文件 / File**: `fewshot_plant_mrmodn_3way.py`

**任务**: 在 3 个独立物种（拟南芥、水稻、玉米）上做 5-way 5-shot 学习

**Task**: 5-way 5-shot learning on 3 independent species (Arabidopsis, rice, maize).

```bash
python fewshot_plant_mrmodn_3way.py --species arabidopsis rice maize
```

**关键点 / Key points**:
- 3 个物种的训练彼此独立
- 支持跨物种迁移（用 Arabidopsis 训练，评估 Rice）

## 6. 零样本综合分析 / Zero-shot Comprehensive Analysis

**文件 / File**: `zeroshot_human_mrmodn_analysis.py`

**任务**: 评估在 Human 上训练的模型在未见过物种/位点上的泛化能力

**Task**: Evaluate generalization to unseen species/sites.

```bash
python zeroshot_human_mrmodn_analysis.py \
    --pretrained output/human_mrmodn/best.pt \
    --target gen3 plant
```

**输出 / Output**:
- `output/zeroshot/gen3_metrics.csv` — Gen3 各类 F1
- `output/zeroshot/plant_metrics.csv` — Plant 各类 F1
- `output/zeroshot/comparison.csv` — Human 训练 → Gen3/Plant 迁移对比

## 7. 零样本特征抽取 / Zero-shot Feature Extraction

**文件 / File**: `zeroshot_human_mrmodn_extract.py`

抽取预训练模型的中间特征用于可视化或下游任务。

Extract intermediate features from a pretrained model for visualization or downstream tasks.

```bash
python zeroshot_human_mrmodn_extract.py \
    --pretrained output/human_mrmodn/best.pt \
    --output output/zeroshot/features.npz
```

**输出 features.npz**:
- `cnn_features` (N, 64) — CNN 阶段特征
- `gcn_features` (N, 128) — GCN 阶段特征
- `attn_features` (N, 12, 1001) — 注意力权重

## 8. 性能对比 / Performance Comparison

| 方法 / Method | 5-shot ACC | 20-shot ACC | 0-shot (Gen3) F1 |
|---|---|---|---|
| Baseline (no FS) | 0.62 | 0.78 | 0.31 |
| Fine-tune | 0.68 | 0.81 | - |
| Prototypical Net | 0.71 | 0.84 | - |
| **mRModN + ABS** | **0.75** | **0.86** | **0.42** |

## 9. 下一步 / Next Steps

- 附录：[附录](/guide/appendix)
- GitHub Issues 提交问题
