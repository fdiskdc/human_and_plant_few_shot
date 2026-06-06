# 推理流程 / Inference Pipeline

## 1. 推理脚本总览 / Inference Script Overview

| 脚本 / Script | 数据 / Data | 模型 / Model | 长度 / Length | 类型 / Type |
|---|---|---|---|---|
| `inference_human_mrmodn_full.py` | Human | mRModN | 1001 | 全长 / Full |
| `inference_human_modx_segmented.py` | Human | ModX | 51 窗口 | 滑窗 / Sliding |
| `inference_human_evormd_segmented.py` | Human | EvoRMD | 51 窗口 | 滑窗 |
| `inference_multirm_multirm_segmented.py` | MultiRM | MultiRM | 51 窗口 | 滑窗 |

## 2. 全长推理 vs 滑窗推理 / Full vs Sliding Window

### 2.1 全长推理 / Full-length Inference

适用于固定长度（1001nt）的输入。模型一次性处理整条序列。

For fixed-length (1001nt) inputs. The model processes the entire sequence at once.

**优点 / Pros**:
- 简单直接
- 充分利用 GCN 的全局感受野

**缺点 / Cons**:
- 显存占用大
- 不支持变长序列

**使用场景 / Use case**: Human mRModN 全长预测

### 2.2 滑窗推理 / Sliding Window Inference

将长序列切分为 51nt 窗口（步长 25nt），逐窗口推理后拼接。

Split long sequences into 51nt windows (stride 25nt), infer per window, then stitch.

**优点 / Pros**:
- 支持任意长度
- 显存占用小

**缺点 / Cons**:
- 边界处可能信息丢失
- 需要后处理（高斯加权拼接）

**使用场景 / Use case**: ModX、EvoRMD、MultiRM 分段任务

## 3. `inference_human_mrmodn_full.py` 详解 / Walkthrough

```python
import torch
from model.mrmodn import RNA_ClassQuery_Model
from dataset.human import Mer100Dataset
from torch_geometric.loader import DataLoader

# 1. 加载模型
model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
checkpoint = torch.load("output/human_mrmodn/best.pt")
model.load_state_dict(checkpoint)
model.eval().cuda()

# 2. 加载测试数据
test_dataset = Mer100Dataset(npy_dir="npy/human", split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 推理
all_logits_12, all_logits_4, all_attn = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.cuda()
        logits_12, logits_4, attn = model(batch.x, batch.edge_index, batch.batch)
        all_logits_12.append(logits_12.cpu())
        all_logits_4.append(logits_4.cpu())
        all_attn.append(attn.cpu())

# 4. 保存结果
torch.save({
    "logits_12": torch.cat(all_logits_12),
    "logits_4": torch.cat(all_logits_4),
    "attn": torch.cat(all_attn),
    "labels": test_dataset.y
}, "output/inference/human_mrmodn_full.pt")
```

## 4. 滑窗推理详解 / Sliding Window Walkthrough

使用 `utils/sliding_window_utils.py` 中的工具：

Use utilities from `utils/sliding_window_utils.py`:

```python
from utils.sliding_window_utils import (
    split_into_windows,        # 切窗
    gaussian_weighted_stitch,  # 高斯加权拼接
    batch_predict
)

# 1. 切窗
windows = split_into_windows(sequence, window_size=51, stride=25)
# windows: List[(start, end, sub_seq)]

# 2. 批量推理
preds = batch_predict(model, windows, batch_size=64, device="cuda")
# preds: (num_windows, num_classes)

# 3. 高斯加权拼接
final_logits = gaussian_weighted_stitch(
    preds, sequence_length=len(sequence), window_size=51, stride=25
)
# final_logits: (num_classes, sequence_length)
```

**高斯加权公式**:
```
w_i = exp(-(i - center)^2 / (2 * sigma^2))
logits[pos] = sum_i (w_i * preds[i]) / sum_i w_i
```

**Gaussian weighting**:
```
w_i = exp(-(i - center)^2 / (2 * sigma^2))
logits[pos] = sum_i (w_i * preds[i]) / sum_i w_i
```

## 5. 零样本推理 / Zero-shot Inference

```python
# 加载预训练模型 (Human)
model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
model.load_state_dict(torch.load("output/human_mrmodn/best.pt"))

# 在 Gen3 数据集上做零样本推理
from dataset.gen3 import Gen3Dataset
gen3_dataset = Gen3Dataset(npy_dir="npy/gen3", split="test")
# 注意：训练时未见过 Gen3 物种/位点
gen3_loader = DataLoader(gen3_dataset, batch_size=32)

# 推理（不微调）
zeroshot_results = evaluate(model, gen3_loader, device="cuda")
print(zeroshot_results)
```

## 6. 推理输出格式 / Inference Output Format

推理结果保存为 `torch.save` 的字典：

| Key | 形状 / Shape | 含义 / Meaning |
|---|---|---|
| `logits_12` | (N, 12) | 12 类 logits |
| `logits_4` | (N, 4) | 4 组 logits（层级头） |
| `attn` | (N, 12, L) | 类-节点注意力权重 |
| `labels` | (N, 12) | 真实标签 |

转换为概率：

```python
probs_12 = torch.sigmoid(logits_12)  # (N, 12)
preds_12 = (probs_12 > 0.5).int()     # 二值化
```

## 7. 性能基准 / Performance Benchmark

| 模型 / Model | 数据集 / Dataset | 时间/epoch | GPU |
|---|---|---|---|
| mRModN | Human (250k) | ~12 min | V100 |
| mRModN | Plant (200k) | ~10 min | V100 |
| ModX | Human | ~8 min | V100 |
| EvoRMD | Human | ~25 min | A100 |

推理速度 / Inference speed: ~1000 samples/s on V100

## 8. 下一步 / Next Steps

- 可视化工具：[可视化](/guide/visualization)
- 消融实验：[消融实验](/guide/ablation)
