# Datasets

## 1. Overview

| Dataset | # Mods | Source | Loader | Length |
|---|---|---|---|---|
| HRMD-m | 12 | Human (HEK293, HeLa, etc.) | `dataset/human.py` | 1001 |
| PRMD-m | 12 | Plant (Arabidopsis, Rice) | `dataset/plant.py` | 1001 |
| ac4C | 1 | Human / Yeast ac4C | `dataset/ac4c.py` | 415 |
| MultiRM | 12 | Human multi-task | `dataset/multirm.py` | 51-1001 |
| Gen3 | 12 | Third-gen sequencing (ONT/PacBio) | `dataset/gen3.py` | variable |
| Gen3-zero | 12 | Zero-shot evaluation | `dataset/gen3_zero.py` | variable |
| Human-motif | 12 | With motif annotation | `dataset/human_motif.py` | 1001 |
| Human-seq | 12 | With raw sequence | `dataset/human_with_seq.py` | 1001 |
| Plant-single | 1 | Single plant species | `dataset/plant_single.py` | 1001 |

## 2. HRMD-m (Human 12-Modification)

**Source**: Human mRNA modification data, aggregated from RMBase, MODOMICS, etc.
**Samples**: ~250,000 (positive) + ~2,000,000 (negative)
**Preprocessing**:

```python
# dataset/human.py:Mer100Dataset
seq_id, pos, neg = preprocess(...)
# CD-HIT de-redundancy (threshold 0.8)
# Truncate/pad to 1001nt
# Convert to (N, 1001, 4) one-hot
```

**Label mapping** (12-class multi-label):

```python
MOD_LABELS = {
    'm6A': 0, 'm5C': 1, 'm1A': 2, 'm2G': 3,
    'Psi': 4, 'ac4C': 5, 'm7G': 6, 'm3C': 7,
    'I': 8, 's2U': 9, 'D': 10, 'Nm': 11
}
```

**Data format**:

```
npy/
├── train_pos.npy    # (N_pos, 1001, 4) float32
├── train_neg.npy    # (N_neg, 1001, 4) float32
├── test_pos.npy
├── test_neg.npy
└── train_labels.npy # (N_pos, 12) multi-label
```

## 3. PRMD-m (Plant 12-Modification)

**Source**: Plant mRNA modifications aggregated from RMBase Plant subset
**Specialty**: High species diversity (Arabidopsis, rice, maize, tomato, etc.)
**Loader**: `dataset/plant.py`

## 4. ac4C Dataset

**Source**: ac4C (N4-acetylcytosine) public dataset
**Samples**: ~14,000 (positive) + ~140,000 (negative)
**Length**: 415nt (fixed)
**Use case**: Few-shot learning benchmark

**Loader**: `dataset/ac4c.py:AC4CDataset`

```python
class AC4CDataset(Dataset):
    def __init__(self, npy_dir, split='train'):
        # Load 415nt sequences + 1 label (ac4C pos/neg)
        pass

    def __getitem__(self, idx):
        # Return Data(x=[415,4], edge_index, y=int)
        pass
```

## 5. MultiRM Dataset

**Source**: Multi-task RNA modification data
**Specialty**: Each sample may belong to multiple modification classes (multi-label)
**Loader**: `dataset/multirm.py`
**Key param**: `seq_lens=[51, 100, 1001]` supports multi-scale

## 6. Gen3 Dataset

**Source**: Third-generation sequencing (Nanopore / PacBio) modification data
**Specialty**:
- Variable sequence length (200-3000nt)
- Higher error rate than NGS (~10-15%)
- Includes base-calling quality scores

**Loader**: `dataset/gen3.py:Gen3Dataset`

## 7. Gen3-zero Dataset

**Purpose**: Zero-shot evaluation — model has not seen this species/sites during training
**Loader**: `dataset/gen3_zero.py`

## 8. Data Format Details

### 8.1 npy Files

All datasets are ultimately converted to `np.float32` arrays in `npy/`:

- **Sequence arrays**: `(N, L, 4)` one-hot, `L` = sequence length
- **Label arrays**:
  - Single-label: `(N,)` int
  - Multi-label: `(N, 12)` float (0/1 binary) or `(N, 12)` logits
  - Multi-class: `(N, 12)` one-hot

### 8.2 PyG Data Object

Each `__getitem__` returns a PyG `Data` object:

```python
Data(
    x=torch.Tensor([L, 4]),           # one-hot
    edge_index=torch.LongTensor([2, E]),  # graph edges
    y=torch.Tensor([12])             # 12-class label
)
```

`edge_index` is typically built from sequence adjacency (i ↔ i+1) with optional long-range edges (k-NN).

### 8.3 Data Augmentation

- **Random reverse complement** (50% probability)
- **Random shift** (±10nt)
- **Random masking** (5% nucleotides replaced with N)

## 9. Loader Comparison

| Loader | Key Class | Length | Task |
|---|---|---|---|
| `dataset/human.py` | `Mer100Dataset` | 1001 | 12-class |
| `dataset/plant.py` | `PlantDataset` | 1001 | 12-class |
| `dataset/ac4c.py` | `AC4CDataset` | 415 | 2-class (single mod) |
| `dataset/multirm.py` | `MultiRMDataset` | 51-1001 | 12 multi-task |
| `dataset/gen3.py` | `Gen3Dataset` | var | 12-class |
| `dataset/gen3_zero.py` | `Gen3ZeroDataset` | var | 12 zero-shot |
| `dataset/human_motif.py` | `HumanMotifDataset` | 1001 | 12 + motif |
| `dataset/human_with_seq.py` | `HumanWithSeqDataset` | 1001 | 12 + seq |
| `dataset/plant_single.py` | `PlantSingleDataset` | 1001 | single species 12-class |

## 10. Custom Dataset

To add a new dataset, use this template:

```python
from torch_geometric.data import Data, Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, npy_dir, split='train'):
        super().__init__()
        self.x = np.load(f"{npy_dir}/{split}_x.npy")  # (N, L, 4)
        self.y = np.load(f"{npy_dir}/{split}_y.npy")  # (N, num_classes)
        self.edge_index = self._build_edges(L)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Data(
            x=torch.from_numpy(self.x[idx]).float(),
            edge_index=self.edge_index,
            y=torch.from_numpy(self.y[idx]).float()
        )

    def _build_edges(self, L):
        edges = [[i, i+1] for i in range(L-1)]
        return torch.LongTensor(edges).t().contiguous()
```

## 11. Next Steps

- Training loop: [Training Pipeline](/en/guide/training)
- Few-shot analysis: [Few/Zero-shot Analysis](/en/guide/fewshot)
