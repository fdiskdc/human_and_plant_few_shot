# Training Pipeline

## 1. Training Script Overview

| Script | Dataset | Model | Purpose |
|---|---|---|---|
| `train_human_mrmodn.py` | Human | mRModN | Main training entry |
| `train_plant_mrmodn.py` | Plant | mRModN | Plant data training |
| `train_multirm_mrmodn.py` | MultiRM | mRModN | Multi-task training |
| `train_human_multirm.py` | Human | MultiRM | Multi-task head |
| `train_human_modx.py` | Human | ModX | ModX ablation training |
| `train_human_evormd.py` | Human | EvoRMD | Evolutionary feature training |

## 2. `train_human_mrmodn.py` Walkthrough

`train_human_mrmodn.py` is the main training script with the following structure:

### 2.1 Imports & Config

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

### 2.2 Command-line Arguments

| Arg | Default | Description |
|---|---|---|
| `--epochs` | 40 | Number of epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--weight-decay` | 1e-4 | Weight decay |
| `--cnn-hidden-dim` | 64 | CNN hidden dim |
| `--gcn-hidden-dim` | 128 | GCN hidden dim |
| `--num-gcn-layers` | 3 | Number of GCN layers |
| `--use-hierarchical` | True | Use hierarchical head |
| `--use-abs` | True | Use ABS sampler |
| `--seed` | 42 | Random seed |
| `--output-dir` | `output/human_mrmodn` | Output directory |
| `--log-dir` | `logs/human_mrmodn` | Log directory |

### 2.3 Data Loading

```python
train_dataset = Mer100Dataset(npy_dir="npy/human", split="train")
val_dataset = Mer100Dataset(npy_dir="npy/human", split="val")
test_dataset = Mer100Dataset(npy_dir="npy/human", split="test")

train_loader = PyGDataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=4
)
```

### 2.4 Model Initialization

```python
model = RNA_ClassQuery_Model(
    num_classes=12,
    cnn_hidden_dim=args.cnn_hidden_dim,
    gcn_hidden_dim=args.gcn_hidden_dim,
    num_gcn_layers=args.num_gcn_layers,
    use_hierarchical=args.use_hierarchical
).to(device)
```

### 2.5 Optimizer & Scheduler

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

### 2.6 Training Loop

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

    # Validation
    val_metrics = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}: train_loss={train_loss/len(train_dataset):.4f} "
          f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")

    # Save checkpoint
    if val_metrics['f1'] > best_f1:
        torch.save(model.state_dict(),
                   f"{args.output_dir}/epoch_{epoch:03d}.pt")
        best_f1 = val_metrics['f1']
```

### 2.7 Evaluation

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

## 3. Training Outputs

```
output/human_mrmodn/
├── epoch_001.pt
├── epoch_002.pt
├── ...
├── epoch_040.pt           # final weights
├── best.pt                # best val F1 weights
├── config.json            # training config
└── training.log           # training log

logs/human_mrmodn/
├── events.out.tfevents.*  # TensorBoard logs
└── metrics.csv            # per-epoch metrics
```

## 4. TensorBoard

```bash
tensorboard --logdir logs/human_mrmodn
# Open http://localhost:6006 in your browser
```

Curves to observe:
- `train/loss` — training loss
- `val/acc` — validation accuracy
- `val/f1_macro` — validation macro F1
- `val/f1_per_class` — per-class F1

## 5. ABS Adaptive Balanced Sampler

When ABS is enabled, use `WeightedRandomSampler`:

```python
from utils.few_shot import AdaptiveBalancedSampler

sampler = AdaptiveBalancedSampler(
    dataset=train_dataset,
    num_classes=12,
    beta=0.999  # smoothing factor
)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

**Sampling probability formula**:
```
p_c = (1 / f1_c) ^ beta / sum_c (1 / f1_c) ^ beta
```

`f1_c` is the F1 score of class `c` from the previous epoch.

## 6. Custom Training

To train on a new dataset:

1. Create `dataset/my_data.py` extending `torch_geometric.data.Dataset`
2. Model `train_my_data.py` after `train_human_mrmodn.py`
3. Follow the naming convention: `<task>_<dataset>_<model>[_<suffix>].py`

## 7. Training Tips

- **Learning rate**: mRModN recommended `1e-3`; ModX recommended `5e-4`
- **Warmup**: Linear warmup over the first 5 epochs from 0 to `lr`
- **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)`
- **Mixed precision**: Use `torch.cuda.amp` for ~30% speedup
- **Multi-GPU**: `nn.DataParallel` or `DistributedDataParallel`

## 8. Next Steps

- Inference: [Inference Pipeline](/en/guide/inference)
- Few-shot: [Few/Zero-shot Analysis](/en/guide/fewshot)
