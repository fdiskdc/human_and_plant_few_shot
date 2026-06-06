# 数据集 / Datasets

## 1. 总览 / Overview

| 数据集 / Dataset | 修饰数 / Mods | 来源 / Source | 加载器 / Loader | 序列长度 / Length |
|---|---|---|---|---|
| HRMD-m | 12 | Human (HEK293, HeLa 等) | `dataset/human.py` | 1001 |
| PRMD-m | 12 | Plant (Arabidopsis, Rice) | `dataset/plant.py` | 1001 |
| ac4C | 1 | Human / Yeast ac4C | `dataset/ac4c.py` | 415 |
| MultiRM | 12 | Human 多任务 | `dataset/multirm.py` | 51-1001 |
| Gen3 | 12 | 第三代测序 (ONT/PacBio) | `dataset/gen3.py` | variable |
| Gen3-zero | 12 | 零样本评估 | `dataset/gen3_zero.py` | variable |
| Human-motif | 12 | 含 motif 标注 | `dataset/human_motif.py` | 1001 |
| Human-seq | 12 | 含原始序列 | `dataset/human_with_seq.py` | 1001 |
| Plant-single | 1 | 单植物样本 | `dataset/plant_single.py` | 1001 |

## 2. HRMD-m (Human 12-Modification)

**来源 / Source**: Human mRNA 修饰数据集，整合自 RMBase、MODOMICS 等公共数据库
**样本数 / Samples**: ~250,000 (positive) + ~2,000,000 (negative)
**预处理 / Preprocessing**:

```python
# dataset/human.py:Mer100Dataset
seq_id, pos, neg = preprocess(...)
# CD-HIT 去冗余 (阈值 0.8)
# 截取/填充至 1001nt
# 转换为 (N, 1001, 4) one-hot 数组
```

**标签映射 / Label Mapping** (12-class multi-label):

```python
MOD_LABELS = {
    'm6A': 0, 'm5C': 1, 'm1A': 2, 'm2G': 3,
    'Psi': 4, 'ac4C': 5, 'm7G': 6, 'm3C': 7,
    'I': 8, 's2U': 9, 'D': 10, 'Nm': 11
}
```

**数据格式 / Data Format**:

```
npy/
├── train_pos.npy    # (N_pos, 1001, 4) float32
├── train_neg.npy    # (N_neg, 1001, 4) float32
├── test_pos.npy
├── test_neg.npy
└── train_labels.npy # (N_pos, 12) multi-label
```

## 3. PRMD-m (Plant 12-Modification)

**来源 / Source**: Plant mRNA 修饰，整合自 RMBase Plant 子库
**特点 / Specialty**: 物种多样性高（拟南芥、水稻、玉米、番茄等）
**加载器 / Loader**: `dataset/plant.py`

## 4. ac4C 数据集 / ac4C Dataset

**来源 / Source**: ac4C (N4-acetylcytosine) 公开数据集
**样本数 / Samples**: ~14,000 (positive) + ~140,000 (negative)
**序列长度 / Length**: 415nt (固定 / fixed)
**用途 / Use Case**: 少样本学习基准 / Few-shot learning benchmark

**加载器 / Loader**: `dataset/ac4c.py:AC4CDataset`

```python
class AC4CDataset(Dataset):
    def __init__(self, npy_dir, split='train'):
        # 加载 415nt 序列 + 1 标签 (ac4C 阳性/阴性)
        pass

    def __getitem__(self, idx):
        # 返回 Data(x=[415,4], edge_index, y=int)
        pass
```

## 5. MultiRM 数据集 / MultiRM Dataset

**来源 / Source**: 多任务 RNA 修饰数据
**特点 / Specialty**: 每个样本可能属于多个修饰类别（多标签）
**加载器 / Loader**: `dataset/multirm.py`
**关键参数 / Key Param**: `seq_lens=[51, 100, 1001]` 支持多尺度

## 6. Gen3 数据集 / Gen3 Dataset

**来源 / Source**: 第三代测序 (Nanopore / PacBio) 修饰数据
**特点 / Specialty**:
- 序列长度可变（200-3000nt）
- 错误率较二代高 (~10-15%)
- 包含 base-calling 质量分数

**加载器 / Loader**: `dataset/gen3.py:Gen3Dataset`

## 7. Gen3-zero 数据集 / Gen3-zero Dataset

**目的 / Purpose**: 零样本评估 — 模型在训练时未见过此数据集的物种/位点
**加载器 / Loader**: `dataset/gen3_zero.py`

## 8. 数据格式详解 / Data Format Details

### 8.1 npy 文件 / npy Files

所有数据集最终都转换为 `np.float32` 数组保存到 `npy/`：

All datasets are ultimately converted to `np.float32` arrays in `npy/`:

- **序列数组 / Sequence arrays**: `(N, L, 4)` one-hot，`L` 为序列长度
- **标签数组 / Label arrays**:
  - 单标签: `(N,)` int
  - 多标签: `(N, 12)` float (0/1 二值) 或 `(N, 12)` logits
  - 多分类: `(N, 12)` one-hot

### 8.2 PyG Data 对象 / PyG Data Object

每个 `__getitem__` 返回 PyG `Data` 对象：

```python
Data(
    x=torch.Tensor([L, 4]),           # one-hot
    edge_index=torch.LongTensor([2, E]),  # graph edges
    y=torch.Tensor([12])             # 12-class label
)
```

`edge_index` 通常基于序列邻接构建（节点 i 与 i+1 相连）+ 可选长程边（k-NN）。

`edge_index` is typically built from sequence adjacency (i ↔ i+1) with optional long-range edges (k-NN).

### 8.3 数据增强 / Data Augmentation

- **随机反向互补 / Random reverse complement** (50% 概率)
- **随机平移 / Random shift** (±10nt)
- **随机 mask / Random masking** (5% 核苷酸替换为 N)

## 9. 各数据集加载器对比 / Loader Comparison

| 加载器 / Loader | 关键类 / Class | 序列长度 / Length | 任务 / Task |
|---|---|---|---|
| `dataset/human.py` | `Mer100Dataset` | 1001 | 12 分类 |
| `dataset/plant.py` | `PlantDataset` | 1001 | 12 分类 |
| `dataset/ac4c.py` | `AC4CDataset` | 415 | 2 分类 (单修饰) |
| `dataset/multirm.py` | `MultiRMDataset` | 51-1001 | 12 多任务 |
| `dataset/gen3.py` | `Gen3Dataset` | var | 12 分类 |
| `dataset/gen3_zero.py` | `Gen3ZeroDataset` | var | 12 零样本 |
| `dataset/human_motif.py` | `HumanMotifDataset` | 1001 | 12 分类 + motif |
| `dataset/human_with_seq.py` | `HumanWithSeqDataset` | 1001 | 12 分类 + seq |
| `dataset/plant_single.py` | `PlantSingleDataset` | 1001 | 单物种 12 分类 |

## 10. 自定义数据集 / Custom Dataset

如需添加新数据集，参考以下模板：

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
        # 序列邻接边 + 长程 k-NN
        edges = [[i, i+1] for i in range(L-1)]
        return torch.LongTensor(edges).t().contiguous()
```

## 11. 下一步 / Next Steps

- 训练流程：[训练流程](/guide/training)
- 少样本分析：[少/零样本分析](/guide/fewshot)
