# Inference Pipeline

## 1. Inference Script Overview

| Script | Data | Model | Length | Type |
|---|---|---|---|---|
| `inference_human_mrmodn_full.py` | Human | mRModN | 1001 | Full |
| `inference_human_modx_segmented.py` | Human | ModX | 51 window | Sliding |
| `inference_human_evormd_segmented.py` | Human | EvoRMD | 51 window | Sliding |
| `inference_multirm_multirm_segmented.py` | MultiRM | MultiRM | 51 window | Sliding |

## 2. Full-length vs Sliding Window

### 2.1 Full-length Inference

For fixed-length (1001nt) inputs. The model processes the entire sequence at once.

**Pros**:
- Simple and direct
- Fully exploits GCN's global receptive field

**Cons**:
- Large GPU memory
- Does not support variable-length sequences

**Use case**: Human mRModN full-length prediction

### 2.2 Sliding Window Inference

Split long sequences into 51nt windows (stride 25nt), infer per window, then stitch.

**Pros**:
- Supports arbitrary length
- Low GPU memory

**Cons**:
- Information loss at window boundaries
- Requires post-processing (Gaussian-weighted stitching)

**Use case**: ModX, EvoRMD, MultiRM segmented tasks

## 3. `inference_human_mrmodn_full.py` Walkthrough

```python
import torch
from model.mrmodn import RNA_ClassQuery_Model
from dataset.human import Mer100Dataset
from torch_geometric.loader import DataLoader

# 1. Load model
model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
checkpoint = torch.load("output/human_mrmodn/best.pt")
model.load_state_dict(checkpoint)
model.eval().cuda()

# 2. Load test data
test_dataset = Mer100Dataset(npy_dir="npy/human", split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Inference
all_logits_12, all_logits_4, all_attn = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.cuda()
        logits_12, logits_4, attn = model(batch.x, batch.edge_index, batch.batch)
        all_logits_12.append(logits_12.cpu())
        all_logits_4.append(logits_4.cpu())
        all_attn.append(attn.cpu())

# 4. Save results
torch.save({
    "logits_12": torch.cat(all_logits_12),
    "logits_4": torch.cat(all_logits_4),
    "attn": torch.cat(all_attn),
    "labels": test_dataset.y
}, "output/inference/human_mrmodn_full.pt")
```

## 4. Sliding Window Walkthrough

Use utilities from `utils/sliding_window_utils.py`:

```python
from utils.sliding_window_utils import (
    split_into_windows,        # split into windows
    gaussian_weighted_stitch,  # Gaussian-weighted stitching
    batch_predict
)

# 1. Split into windows
windows = split_into_windows(sequence, window_size=51, stride=25)
# windows: List[(start, end, sub_seq)]

# 2. Batch prediction
preds = batch_predict(model, windows, batch_size=64, device="cuda")
# preds: (num_windows, num_classes)

# 3. Gaussian-weighted stitching
final_logits = gaussian_weighted_stitch(
    preds, sequence_length=len(sequence), window_size=51, stride=25
)
# final_logits: (num_classes, sequence_length)
```

**Gaussian weighting**:
```
w_i = exp(-(i - center)^2 / (2 * sigma^2))
logits[pos] = sum_i (w_i * preds[i]) / sum_i w_i
```

## 5. Zero-shot Inference

```python
# Load pretrained model (Human)
model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
model.load_state_dict(torch.load("output/human_mrmodn/best.pt"))

# Zero-shot inference on Gen3
from dataset.gen3 import Gen3Dataset
gen3_dataset = Gen3Dataset(npy_dir="npy/gen3", split="test")
# Note: Gen3 species/sites are unseen during training
gen3_loader = DataLoader(gen3_dataset, batch_size=32)

# Inference (no fine-tuning)
zeroshot_results = evaluate(model, gen3_loader, device="cuda")
print(zeroshot_results)
```

## 6. Inference Output Format

| Key | Shape | Meaning |
|---|---|---|
| `logits_12` | (N, 12) | 12-class logits |
| `logits_4` | (N, 4) | 4-group logits (hierarchical head) |
| `attn` | (N, 12, L) | Class-node attention weights |
| `labels` | (N, 12) | True labels |

Convert to probabilities:

```python
probs_12 = torch.sigmoid(logits_12)  # (N, 12)
preds_12 = (probs_12 > 0.5).int()     # binarize
```

## 7. Performance Benchmark

| Model | Dataset | Time/epoch | GPU |
|---|---|---|---|
| mRModN | Human (250k) | ~12 min | V100 |
| mRModN | Plant (200k) | ~10 min | V100 |
| ModX | Human | ~8 min | V100 |
| EvoRMD | Human | ~25 min | A100 |

Inference speed: ~1000 samples/s on V100

## 8. Next Steps

- Visualization: [Visualization](/en/guide/visualization)
- Ablation: [Ablation Studies](/en/guide/ablation)
