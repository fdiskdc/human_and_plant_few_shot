# Quickstart

## 1. Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.8+ (3.10 recommended) | Runtime |
| PyTorch | 1.10+ | Deep learning framework |
| PyTorch Geometric | 2.0+ | Graph convolution |
| numpy | 1.20+ | Numerical computing |
| pandas | 1.3+ | Data analysis |
| scikit-learn | 1.0+ | Evaluation metrics |
| matplotlib | 3.4+ | Visualization |
| R (optional) | 4.0+ | R visualization scripts |
| Node.js (optional) | 18+ | Build VitePress docs locally |

## 2. Installation

```bash
# Clone the repo
git clone https://github.com/<your-org>/human_and_plant_few_shot.git
cd human_and_plant_few_shot

# Python deps
pip install torch torch-geometric numpy pandas scikit-learn matplotlib

# (Optional) Docs deps
cd docs && npm install
```

## 3. Data Preparation

Data is provided via the `npy/` symlink:

```bash
ls -la npy
# npy -> /home/dc/vscode/npyForTrain
```

If the symlink is missing, run preprocessing scripts in `dataset/`:

```bash
python dataset/preprocess_human.py
# Produces train_pos.npy, train_neg.npy, test_pos.npy, test_neg.npy
```

## 4. Minimal Example

```python
import subprocess
subprocess.run([
    "python", "train_human_mrmodn.py",
    "--epochs", "30",
    "--batch-size", "32",
    "--lr", "1e-3"
])
```

Or directly:

```bash
python train_human_mrmodn.py --epochs 30 --batch-size 32
```

After training, weights are saved to `output/human_mrmodn/epoch_030.pt`, logs to `logs/`.

## 5. Quick Inference

```bash
# Human + mRModN full-length 1001nt inference
python inference_human_mrmodn_full.py

# Sliding window (segmented)
python inference_human_modx_segmented.py --input sequences.fasta
```

## 6. FAQ

**Q: GPU OOM?**
A: Reduce `--batch-size` (try 16 or 8) and enable gradient accumulation.

**Q: Data path error?**
A: Verify the `npy/` symlink points to a valid directory or re-run `dataset/preprocess_*.py`.

**Q: How to train on a subset of classes?**
A: Edit the label-filter logic in `dataset/human.py` or `num_classes` in `model/mrmodn.py`.

**Q: Loss not decreasing?**
A: Check learning rate (try 1e-3 ~ 5e-4), warmup steps, and data augmentation.

## 7. Directory Tour

| Dir | Purpose |
|---|---|
| `model/` | Model definitions |
| `dataset/` | Dataset loaders |
| `utils/` | Utilities |
| `train_*.py` | Training entry points |
| `inference_*.py` | Inference entry points |
| `fewshot_*.py` / `zeroshot_*.py` | Few/zero-shot analysis |
| `collect_*.py` | Feature/attention collection |
| `ablation/` | Ablation studies |
| `visualization/` | Visualization scripts |
| `tests/` | pytest tests |
| `docs/` | This tutorial (VitePress) |
