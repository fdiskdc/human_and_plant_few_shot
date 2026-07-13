# 快速开始 / Quickstart

## 1. 环境要求 / Requirements

| 依赖 / Dependency | 版本 / Version | 用途 / Purpose |
|---|---|---|
| Python | 3.11 | 运行时 / Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Python 包管理器 / Package manager |
| PyTorch | 2.0+ | 深度学习框架 / Deep learning framework |
| PyTorch Geometric | 2.4+ | 图卷积 / Graph convolution |
| numpy | 1.20+ | 数值计算 / Numerical computing |
| pandas | 1.3+ | 数据分析 / Data analysis |
| scikit-learn | 1.0+ | 评估指标 / Evaluation metrics |
| matplotlib | 3.4+ | 可视化 / Visualization |
| R (可选) | 4.0+ | R 脚本可视化 / R visualization |
| Node.js (可选) | 18+ | 本地构建 VitePress 文档 / Build VitePress docs |

## 2. 安装 / Installation

```bash
# 克隆仓库 / Clone the repo
git clone https://github.com/<your-org>/human_and_plant_few_shot.git
cd human_and_plant_few_shot

# Python 依赖 / Python deps
uv sync --locked

# (可选) 分析工具 / (Optional) Analysis tools
uv sync --locked --group analysis

# (可选) 文档依赖 / (Optional) Docs deps
cd docs && npm ci
```

## 3. 数据准备 / Data Preparation

数据通过软链接 `npy/` 提供：

Data is provided via the `npy/` symlink:

```bash
ls -la npy
# npy -> /home/dc/vscode/npyForTrain
```

如果软链接不存在，需要从 `dataset/` 运行预处理脚本生成 `*.npy`：

If the symlink is missing, run preprocessing scripts in `dataset/`:

```bash
uv run python dataset/preprocess_human.py
# 生成 train_pos.npy, train_neg.npy, test_pos.npy, test_neg.npy
```

## 4. 最小可运行示例 / Minimal Example

```python
# 训练 Human mRModN 30 个 epoch
import subprocess
subprocess.run([
    "uv", "run", "python", "train_human_mrmodn.py",
    "--epochs", "30",
    "--batch-size", "32",
    "--lr", "1e-3"
])
```

或直接命令行：

```bash
uv run python train_human_mrmodn.py --epochs 30 --batch-size 32
```

训练完成后，模型权重保存到 `output/human_mrmodn/epoch_030.pt`，日志写入 `logs/`。

After training, weights are saved to `output/human_mrmodn/epoch_030.pt`, logs to `logs/`.

## 5. 快速推理 / Quick Inference

```bash
# Human + mRModN 全长 1001nt 推理
uv run python inference_human_mrmodn_full.py

# 滑窗推理 (segmented)
uv run python inference_human_modx_segmented.py --input sequences.fasta
```

## 6. 常见问题 / FAQ

**Q: GPU 显存不足怎么办？**
A: 减小 `--batch-size`（推荐 16 或 8），并启用梯度累积。

**Q: 数据路径报错？**
A: 确认 `npy/` 软链接指向正确目录，或重新运行 `dataset/preprocess_*.py`。

**Q: 如何只跑部分类？**
A: 修改 `dataset/human.py` 的标签筛选逻辑或 `model/mrmodn.py` 的 `num_classes`。

**Q: 训练时 Loss 不下降？**
A: 检查学习率（建议 1e-3 ~ 5e-4），warmup 步数和数据增强。

## 7. 目录速览 / Directory Tour

| 目录 / Dir | 用途 / Purpose |
|---|---|
| `model/` | 模型定义 |
| `dataset/` | 数据集加载器 |
| `utils/` | 工具函数 |
| `train_*.py` | 训练入口 |
| `inference_*.py` | 推理入口 |
| `fewshot_*.py` / `zeroshot_*.py` | 少/零样本分析 |
| `collect_*.py` | 特征/注意力收集 |
| `ablation/` | 消融实验 |
| `visualization/` | 可视化脚本 |
| `tests/` | pytest 测试 |
| `docs/` | 本教程（VitePress） |
