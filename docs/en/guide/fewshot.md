# Few/Zero-shot Analysis

## 1. Background

Experimental identification of RNA modifications is expensive, so **many modification types have very few labeled samples** (e.g., rare modifications may have only 10-100 positive samples). mRModN tackles this via **Few-shot Learning** and **Zero-shot Transfer**.

## 2. Core Algorithm: `utils/few_shot.py`

**File**: `utils/few_shot.py`

Core components:

| Component | Class | Purpose |
|---|---|---|
| Prototypical Networks | `PrototypicalNetwork` | Few-shot via prototype network |
| Adaptive Balanced Sampler | `AdaptiveBalancedSampler` | Adaptive class-balanced sampling |
| N-way K-shot Sampler | `NWayKShotSampler` | N-way K-shot sampler |
| Episode Sampler | `EpisodeSampler` | Episode sampler |

### 2.1 Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        self.encoder = encoder  # mRModN encoder (head removed)
        self.hidden_dim = hidden_dim

    def forward(self, support_x, support_y, query_x, n_way, k_shot):
        # 1. Encode support and query
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)

        # 2. Compute class prototypes (mean of support per class)
        prototypes = []
        for c in range(n_way):
            prototypes.append(support_emb[support_y == c].mean(dim=0))
        prototypes = torch.stack(prototypes)

        # 3. Query-to-prototype distance
        dists = -torch.cdist(query_emb, prototypes)  # (Q, n_way)
        return F.log_softmax(dists, dim=-1)
```

### 2.2 Adaptive Balanced Sampler

```python
class AdaptiveBalancedSampler:
    """
    Dynamically adjusts sampling probabilities based on inverse per-class F1
    """
    def __init__(self, dataset, num_classes, beta=0.999):
        self.dataset = dataset
        self.num_classes = num_classes
        self.beta = beta
        self.f1_history = torch.ones(num_classes)  # initial F1=1

    def update_f1(self, f1_per_class):
        # EMA smoothing
        self.f1_history = (
            self.beta * self.f1_history + (1 - self.beta) * f1_per_class
        )

    def get_weights(self):
        # weights = 1 / F1
        weights = 1.0 / (self.f1_history + 1e-6)
        # Normalize
        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        weights = self.get_weights()
        sample_weights = torch.tensor([
            weights[self.dataset.label_of(i)]
            for i in range(len(self.dataset))
        ])
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(self.dataset), replacement=True
        )
        yield from iter(sampler)
```

## 3. ac4C Balanced Few-shot

**File**: `fewshot_ac4c_mrmodn_balance.py`

**Task**: Evaluate ac4C modification prediction at 5-shot / 10-shot / 20-shot scales.

**Specialty**: **Balanced sampling** — equal number of samples per class.

```bash
python fewshot_ac4c_mrmodn_balance.py --n-shot 5 --n-way 2
python fewshot_ac4c_mrmodn_balance.py --n-shot 10 --n-way 2
python fewshot_ac4c_mrmodn_balance.py --n-shot 20 --n-way 2
```

**Output**: `output/fewshot/ac4c_balance_{n_shot}shot.csv`

| Shots | ACC | F1 | AUC |
|---|---|---|---|
| 5-shot | 0.71 | 0.68 | 0.79 |
| 10-shot | 0.78 | 0.76 | 0.84 |
| 20-shot | 0.84 | 0.82 | 0.89 |

## 4. ac4C Unbalanced Few-shot

**File**: `fewshot_ac4c_mrmodn_unbalan.py`

**Task**: Simulates real-world scenario with 1:10 positive-negative ratio.

```bash
python fewshot_ac4c_mrmodn_unbalan.py --n-shot 5 --neg-ratio 10
```

**Specialty**: ABS sampler handles imbalance.

## 5. Plant 3-way Independent Few-shot

**File**: `fewshot_plant_mrmodn_3way.py`

**Task**: 5-way 5-shot learning on 3 independent species (Arabidopsis, rice, maize).

```bash
python fewshot_plant_mrmodn_3way.py --species arabidopsis rice maize
```

**Key points**:
- 3 species trained independently
- Supports cross-species transfer (train on Arabidopsis, evaluate on Rice)

## 6. Zero-shot Comprehensive Analysis

**File**: `zeroshot_human_mrmodn_analysis.py`

**Task**: Evaluate generalization to unseen species/sites.

```bash
python zeroshot_human_mrmodn_analysis.py \
    --pretrained output/human_mrmodn/best.pt \
    --target gen3 plant
```

**Output**:
- `output/zeroshot/gen3_metrics.csv` — Gen3 per-class F1
- `output/zeroshot/plant_metrics.csv` — Plant per-class F1
- `output/zeroshot/comparison.csv` — Human → Gen3/Plant transfer comparison

## 7. Zero-shot Feature Extraction

**File**: `zeroshot_human_mrmodn_extract.py`

Extract intermediate features from a pretrained model for visualization or downstream tasks.

```bash
python zeroshot_human_mrmodn_extract.py \
    --pretrained output/human_mrmodn/best.pt \
    --output output/zeroshot/features.npz
```

**Output `features.npz`**:
- `cnn_features` (N, 64) — CNN-stage features
- `gcn_features` (N, 128) — GCN-stage features
- `attn_features` (N, 12, 1001) — attention weights

## 8. Performance Comparison

| Method | 5-shot ACC | 20-shot ACC | 0-shot (Gen3) F1 |
|---|---|---|---|
| Baseline (no FS) | 0.62 | 0.78 | 0.31 |
| Fine-tune | 0.68 | 0.81 | - |
| Prototypical Net | 0.71 | 0.84 | - |
| **mRModN + ABS** | **0.75** | **0.86** | **0.42** |

## 9. Next Steps

- Appendix: [Appendix](/en/guide/appendix)
- Submit an issue on GitHub
