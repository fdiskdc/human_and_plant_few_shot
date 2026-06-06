# 训练流程 / Training Pipeline

## 1. 训练脚本总览 / Training Script Overview

| 脚本 / Script | 数据集 / Dataset | 模型 / Model | 用途 / Purpose |
|---|---|---|---|
| `train_human_mrmodn.py` | Human | mRModN | 主训练入口 / Main training |
| `train_plant_mrmodn.py` | Plant | mRModN | 植物数据训练 |
| `train_multirm_mrmodn.py` | MultiRM | mRModN | 多任务训练 |
| `train_human_multirm.py` | Human | MultiRM | 多任务头 |
| `train_human_modx.py` | Human | ModX | ModX 消融训练 |
| `train_human_evormd.py` | Human | EvoRMD | 进化特征训练 |

## 2. `train_human_mrmodn.py` 详解 / Walkthrough

`train_human_mrmodn.py` 是项目的主训练脚本，结构如下：

`train_human_mrmodn.py` is the main training script with the following structure:

### 2.1 导入与配置 / Imports & Config

```python
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from model.mrmodn import RNA_ClassQuery_Model
from dataset.human import Mer100Dataset
from utils.metrics import compute_metrics
from utils.common import set_seed
```

### 2.2 命令行参数 / Command-line Arguments

| 参数 / Arg | 默认值 / Default | 说明 / Description |
|---|---|---|
| `--epochs` | 40 | 训练轮数 / Number of epochs |
| `--batch-size` | 32 | 批大小 / Batch size |
| `--lr` | 1e-3 | 初始学习率 / Initial learning rate |
| `--weight-decay` | 1e-4 | 权重衰减 / Weight decay |
| `--cnn-hidden-dim` | 64 | CNN 隐藏维度 |
| `--gcn-hidden-dim` | 128 | GCN 隐藏维度 |
| `--num-gcn-layers` | 3 | GCN 层数 |
| `--use-hierarchical` | True | 是否使用层级头 |
| `--use-abs` | True | 是否使用 ABS 采样器 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | `output/human_mrmodn` | 输出目录 |
| `--log-dir` | `logs/human_mrmodn` | 日志目录 |

### 2.3 数据加载 / Data Loading

```python
train_dataset = Mer100Dataset(npy_dir="npy/human", split="train")
val_dataset = Mer100Dataset(npy_dir="npy/human", split="val")
test_dataset = Mer100Dataset(npy_dir="npy/human", split="test")

train_loader = PyGDataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=4
)
```

### 2.4 模型初始化 / Model Initialization

```python
model = RNA_ClassQuery_Model(
    num_classes=12,
    cnn_hidden_dim=args.cnn_hidden_dim,
    gcn_hidden_dim=args.gcn_hidden_dim,
    num_gcn_layers=args.num_gcn_layers,
    use_hierarchical=args.use_hierarchical
).to(device)
```

### 2.5 优化器与调度器 / Optimizer & Scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs
)
```

### 2.6 训练循环 / Training Loop

```python
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        logits_12, logits_4, attn = model(batch.x, batch.edge_index, batch.batch)
        loss = compute_loss(logits_12, logits_4, batch.y, attn)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs

    scheduler.step()

    # 验证
    val_metrics = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}: train_loss={train_loss/len(train_dataset):.4f} "
          f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")

    # 保存 checkpoint
    if val_metrics['f1'] > best_f1:
        torch.save(model.state_dict(),
                   f"{args.output_dir}/epoch_{epoch:03d}.pt")
        best_f1 = val_metrics['f1']
```

### 2.7 评估函数 / Evaluation

```python
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits_12, _, _ = model(batch.x, batch.edge_index, batch.batch)
            preds = (logits_12.sigmoid() > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch.y.cpu().numpy())
    return compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
```

## 3. 训练产物 / Training Outputs

训练过程中会生成以下文件 / The training process produces:

```
output/human_mrmodn/
├── epoch_001.pt
├── epoch_002.pt
├── ...
├── epoch_040.pt           # 最终权重
├── best.pt                # 验证 F1 最高权重
├── config.json            # 训练配置
└── training.log           # 训练日志

logs/human_mrmodn/
├── events.out.tfevents.*  # TensorBoard 日志
└── metrics.csv            # 逐 epoch 指标
```

## 4. TensorBoard 可视化 / TensorBoard

```bash
tensorboard --logdir logs/human_mrmodn
# 浏览器打开 http://localhost:6006
```

可观察曲线 / Observed curves:
- `train/loss` — 训练损失 / Training loss
- `val/acc` — 验证准确率 / Validation accuracy
- `val/f1_macro` — 验证宏 F1 / Validation macro F1
- `val/f1_per_class` — 各类 F1 / Per-class F1

## 5. ABS 自适应平衡采样器 / ABS Sampler

启用 ABS 时使用 `torch.utils.data.WeightedRandomSampler`：

When ABS is enabled, use `WeightedRandomSampler`:

```python
from utils.few_shot import AdaptiveBalancedSampler

sampler = AdaptiveBalancedSampler(
    dataset=train_dataset,
    num_classes=12,
    beta=0.999  # 平滑系数
)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

**采样概率公式**:
```
p_c = (1 / f1_c) ^ beta / sum_c (1 / f1_c) ^ beta
```

`f1_c` 是上一 epoch 类 `c` 的 F1 分数。

## 6. 自定义训练 / Custom Training

如需训练新数据集，参考以下步骤：

1. 创建 `dataset/my_data.py`，继承 `torch_geometric.data.Dataset`
2. 在 `train_my_data.py` 中仿照 `train_human_mrmodn.py`
3. 注册到命名规范：`<task>_<dataset>_<model>[_<suffix>].py`

To train on a new dataset:

1. Create `dataset/my_data.py` extending `torch_geometric.data.Dataset`
2. Model `train_my_data.py` after `train_human_mrmodn.py`
3. Follow the naming convention

## 7. 训练技巧 / Training Tips

- **学习率 / Learning rate**: mRModN 推荐 `1e-3`，ModX 推荐 `5e-4`
- **Warmup**: 前 5 个 epoch 用线性 warmup，从 0 升到 `lr`
- **梯度裁剪 / Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)`
- **混合精度 / Mixed precision**: 用 `torch.cuda.amp` 加速 ~30%
- **多 GPU / Multi-GPU**: 用 `nn.DataParallel` 或 `DistributedDataParallel`

## 8. 下一步 / Next Steps

- 推理脚本：[推理流程](/guide/inference)
- 少样本分析：[少/零样本分析](/guide/fewshot)
