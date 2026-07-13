# RGCNFormer 汇总项目 / RGCNFormer Summary Project

## 项目背景 / Project Background

本项目是 **RGCNFormer** 模型的**主项目**,一个面向 RNA 修饰位点分类的深度学习系统。RGCNFormer 结合 **RGCN(Relational Graph Convolutional Network)** 与 **Transformer(多头自注意力)** 架构,用于在 RNA 序列上精准预测 12 类常见 RNA 修饰(如 m6A、m5C、Ψ、ac4C 等)。

This is the **main project** for the **RGCNFormer** model — a deep learning system for RNA modification site classification. RGCNFormer combines **RGCN (Relational Graph Convolutional Network)** with **Transformer (multi-head self-attention)** to predict 12 common RNA modifications (m6A, m5C, Ψ, ac4C, etc.) on RNA sequences.

**研究背景** / **Research context**:
- RNA 修饰在转录后调控中起关键作用,与疾病、发育、表观遗传密切相关
- RNA modifications play critical roles in post-transcriptional regulation, linked to disease, development, and epigenetics
- 传统实验(iCLIP、miCLIP、SCARLET)成本高、通量低;深度学习可高通量、低成本预测
- Traditional experiments (iCLIP, miCLIP, SCARLET) are costly and low-throughput; deep learning enables high-throughput, low-cost prediction
- RGCNFormer 通过图结构(碱基配对、二级结构)与序列上下文的联合建模,显著提升少样本与零样本场景下的性能
- RGCNFormer jointly models graph structure (base pairing, secondary structure) and sequence context, significantly improving few-shot and zero-shot performance

## 项目作用 / Purpose

本项目是**整个研究系统的核心**,包含:

This project is the **core of the entire research system**, containing:

| 模块 / Module | 作用 / Purpose |
|---|---|
| **模型定义** / Model definition | mRModN (RGCNFormer) 主网络 + 多任务变体 (ModX、MultiRM、EvoRMD) / mRModN backbone + variants |
| **数据集加载** / Dataset loading | Human / Plant / ac4C / MultiRM / gen3 数据集 / Datasets |
| **训练流程** / Training pipeline | 6 个训练脚本 (单修饰、多修饰、零样本等) / 6 training scripts |
| **推理流程** / Inference pipeline | 4 个推理脚本 (分段、全长) / 4 inference scripts (segmented, full) |
| **数据收集** / Data collection | 4 个 attention/特征收集脚本 / 4 attention collection scripts |
| **少样本/零样本分析** / Few-shot/Zero-shot | 平衡/非平衡 ac4C、Plant 3-way、Human 零样本 |
| **可视化** / Visualization | `visualization/` — 按数据集/模型二级组织 / organized by dataset/model |
| **消融实验** / Ablation studies | `ablation/` — 3×3 矩阵、MoHE、FLOPs 计算 |
| **R 可视化** / R visualization | `analysis/` — 零样本/少样本分析图表 / Zero-shot/few-shot plots |
| **测试** / Tests | `tests/` — pytest 框架, model/dataset 导入回归 + 命名规范审计 |

## 目录结构 / Directory Layout

```
rgcnformer_sum/
├── model/                          # 模型定义 (10 个 .py) / Model definitions
│   ├── mrmodn.py                   # mRModN (RGCNFormer) 主网络 — 重命名自 main_model.py
│   ├── mrmodn_collect_atten.py     # 注意力收集变体 — 重命名自 main_model_collect_atten.py
│   ├── mrmodn_multirm.py           # MultIRM 51nt 变体 — 重命名自 main_model_multirm.py
│   ├── multirm.py                  # 多任务多修饰变体
│   ├── modx.py                     # 修饰类型消融变体
│   ├── evormd_human.py             # EvoRMD 集成
│   ├── abla_model.py               # 消融实验模型
│   ├── modx_collect_atten.py       # ModX 注意力收集
│   └── multirm_collect_atten.py    # MultiRM 注意力收集
│
├── dataset/                        # 数据集加载 (9 个 .py) / Dataset loaders
│   ├── human.py                    # Human (标准 12 修饰)
│   ├── plant.py                    # Plant
│   ├── ac4c.py                     # ac4C 专用
│   ├── multirm.py                  # 多 RM
│   ├── gen3.py / gen3_zero.py      # 第 3 代数据集
│   ├── human_motif.py / human_with_seq.py
│   └── plant_single.py
│
├── utils/                          # 工具与少样本分析 / Utilities
│   ├── common.py / metrics.py / logging.py
│   ├── sliding_window_utils.py     # 滑窗工具 (从根目录迁入)
│   ├── few_shot.py / fewshot_analysis_*.py
│   ├── train_gen3.py / test_gen3_analyse.py
│   └── rna_visualization.py
│
├── 根目录脚本 (19 个) / Root scripts (19) — 命名: <task>_<dataset>_<model>[_<suffix>].py
│   ├── train_human_mrmodn.py       # ← train_human.py
│   ├── train_plant_mrmodn.py       # ← train_plant.py
│   ├── train_multirm_mrmodn.py     # ← train_multirm_dataset.py
│   ├── train_human_evormd.py / train_human_modx.py / train_human_multirm.py
│   ├── inference_human_mrmodn_full.py        # ← inference_mrmodn_full.py
│   ├── inference_human_evormd_segmented.py   # ← inference_evormd_segmented.py
│   ├── inference_human_modx_segmented.py     # ← inference_modx_segmented.py
│   ├── inference_multirm_multirm_segmented.py # ← inference_multirm_segmented.py
│   ├── collect_human_mrmodn.py / collect_atten_human_mrmodn.py
│   ├── collect_atten_human_modx.py / collect_atten_multirm_multirm.py
│   ├── fewshot_ac4c_mrmodn_balance.py / fewshot_ac4c_mrmodn_unbalan.py
│   ├── fewshot_plant_mrmodn_3way.py
│   └── zeroshot_human_mrmodn_analysis.py / zeroshot_human_mrmodn_extract.py
│
├── ablation/                       # 消融实验 (5 个) / Ablation experiments
│   ├── ablation_3x3_human_mrmodn.py / ablation_3x3_v2_human_mrmodn.py
│   ├── ablation_mohe_human_mrmodn.py
│   ├── cal_flops_human_mrmodn.py / cal_stats_human_mrmodn.py
│   └── README.md
│
├── visualization/                  # 可视化 — 按数据集/模型二级组织
│   ├── human/                      # mrmodn/, modx/, evormd/, multirm/
│   ├── plant/  multirm/  ac4c/  gen3/
│   ├── tools/                      # view_npz.py, prepare_umap_*.py
│   ├── notebooks/                  # Jupyter notebooks
│   ├── fig/ / att_fig/ / motif_logo/  # 静态图片 (gitignored)
│   └── README.md
│
├── analysis/                       # R 脚本和分析报告 / R scripts and reports
│   ├── plot_zero_fewshot_analysis.R
│   ├── plot_zero_fewshot_export.R
│   └── Rplots.pdf
│
├── tests/                          # pytest 测试套件 / pytest test suite
│   ├── conftest.py / __init__.py
│   ├── test_infrastructure.py      # 框架自检
│   ├── test_model_import.py        # model.* 导入回归
│   ├── test_dataset_import.py      # dataset.* 导入回归
│   ├── test_config.py              # json/ 配置验证
│   ├── test_naming_convention.py   # 命名规范审计
│   ├── test_gen3.py / test_multirm_4class.py / test_multirm_oversampling.py
│
├── pytest.ini                      # pytest 配置 (Wave 5 新增)
│
├── output/                         # 训练输出
├── logs/                           # 训练日志
├── npy/                            # 数据软链接 (/home/dc/vscode/npyForTrain) — 不可移动 / DO NOT MOVE
├── json/                           # 中间 JSON
├── logs_abla/                      # 消融日志
└── .omo/                           # 项目计划与证据 / project plans and evidence
```

## 启动方式 / Getting Started

### 环境要求 / Requirements

- **Python 3.11** / Python 3.11
- **[uv](https://docs.astral.sh/uv/)**（Python 包管理器）/ uv (Python package manager)
- **PyTorch 2.0+** + **PyTorch Geometric**（用于图卷积）/ PyTorch + PyG
- **numpy / pandas / scikit-learn / matplotlib**
- **R 4.0+**（用于 R 可视化脚本）/ R 4.0+ (for R viz scripts)
- **JupyterLab / Notebook**（用于 `visualization/notebooks/`）/ Jupyter for notebooks
- 训练用 GPU（推荐 NVIDIA V100/A100，显存 ≥ 16GB）/ Training GPU recommended

### 安装 / Installation

```bash
cd human_and_plant_few_shot
uv sync --locked

# (可选) 分析工具 / (Optional) Analysis tools
uv sync --locked --group analysis

# (可选) 文档依赖 / (Optional) Docs deps
cd docs && npm ci
```

### 数据准备 / Data Preparation

数据通过软链接 `npy/` 提供:
```bash
ls -la npy
# npy -> /home/dc/vscode/npyForTrain
```

如链接不存在,需从 `dataset/` 目录运行预处理脚本生成 `*.npy`:/ If link missing, run preprocessing scripts in `dataset/` to generate `*.npy`.

### 训练示例 / Training Example

```bash
# 训练 Human 数据集上的 mRModN (RGCNFormer)
uv run python train_human_mrmodn.py --epochs 40 --batch-size 32

# 训练 Plant 数据集
uv run python train_plant_mrmodn.py

# 训练多 RM 数据集
uv run python train_human_multirm.py
uv run python train_multirm_mrmodn.py

# 训练 ModX / EvoRMD 变体
uv run python train_human_modx.py
uv run python train_human_evormd.py
```

训练产物保存到 `output/`,日志保存到 `logs/`。/ Outputs to `output/`, logs to `logs/`.

### 推理示例 / Inference Example

```bash
# Human + mRModN 全长 1001nt 推理
uv run python inference_human_mrmodn_full.py

# 滑窗推理 (segmented)
uv run python inference_human_modx_segmented.py --input sequences.fasta
uv run python inference_human_evormd_segmented.py
uv run python inference_multirm_multirm_segmented.py
```

### 可视化 / Visualization

```bash
# 注意力对比 (从 atten_comp/ 迁移到 visualization/)
uv run python visualization/human/mrmodn/run_attention_comparison_v2.py
uv run python visualization/human/mrmodn/attention_comparison.py

# 空间 motif
uv run python visualization/human/mrmodn/spatial_motif.py

# UMAP 工具
uv run python visualization/tools/prepare_umap_data.py

# R 脚本: 零样本/少样本图表 (从根目录迁移到 analysis/)
Rscript analysis/plot_zero_fewshot_analysis.R
```

### 测试 / Testing

```bash
# 运行完整测试套件 (pytest)
uv run pytest tests/ -v

# 仅运行命名规范审计
uv run pytest tests/test_naming_convention.py -v

# 仅运行模型导入回归
uv run pytest tests/test_model_import.py -v
```

### Jupyter Notebooks

```bash
uv run jupyter lab visualization/notebooks/
# 打开 view_gen3.ipynb 等进行交互式分析
```

## 关键文件说明 / Key Files

### 模型 / Models
- `model/mrmodn.py` - mRModN (RGCNFormer) 主网络: ParallelCNNBlock (多尺度卷积) + GCNBlock (图卷积) + ClassQueryHead (类查询注意力) — 重命名自 `main_model.py`
- `model/mrmodn_collect_atten.py` - 收集注意力变体 — 重命名自 `main_model_collect_atten.py`
- `model/mrmodn_multirm.py` - MultIRM 51nt 变体 — 重命名自 `main_model_multirm.py`
- `model/multirm.py` - 多任务多修饰变体
- `model/modx.py` - 修饰类型消融变体
- `model/evormd_human.py` - EvoRMD 集成(进化特征)/ Evolutionary feature integration
- `model/abla_model.py` - 消融实验模型

### 数据集 / Datasets
- `dataset/human.py` - Human 标准 12 修饰(主用)/ Human standard 12-mod (primary)
- `dataset/plant.py` - Plant 数据集
- `dataset/multirm.py` - 多 RM 数据集
- `dataset/gen3.py` - 第 3 代数据集

### 训练 / Training
- `train_human_mrmodn.py` - 主训练脚本(Human)/ Main training (Human)
- `train_plant_mrmodn.py` - Plant 训练
- `train_multirm_mrmodn.py` - 多 RM 训练
- `train_human_multirm.py` / `train_human_modx.py` / `train_human_evormd.py` - 变体模型训练

### 推理 / Inference
- `inference_human_mrmodn_full.py` - Human + mRModN 全长 1001nt 推理
- `inference_human_modx_segmented.py` - ModX 分段推理
- `inference_human_evormd_segmented.py` - EvoRMD 分段推理
- `inference_multirm_multirm_segmented.py` - 多 RM 分段

### 少样本分析 / Few-shot Analysis
- `fewshot_ac4c_mrmodn_balance.py` / `fewshot_ac4c_mrmodn_unbalan.py` - ac4C 平衡/非平衡
- `fewshot_plant_mrmodn_3way.py` - Plant 三向独立
- `zeroshot_human_mrmodn_analysis.py` - 零样本+少样本综合
- `zeroshot_human_mrmodn_extract.py` - 零样本特征抽取

### 工具 / Utils
- `utils/common.py` - 公共常量(NUCLEOTIDE_MAP、k-mer)、评估函数
- `utils/metrics.py` - ACC / F1 / MCC 等指标
- `utils/few_shot.py` - 少样本核心算法
- `utils/sliding_window_utils.py` - 滑窗工具 (从根目录迁入)

## 与其他项目的关系 / Relation to Other Projects

```
rgcnformer_sum (本项目)
    │  训练并导出 ONNX
    ▼
Cluster_WebAndWx_backend  ←── HTTP API 入口
    │
    ├──→ Cluster_WebAndWx_WxFrontend (微信小程序)
    └──→ RGCNFormer_WebAndWx_WebFrontend (网页前端)
```

- **Cluster_WebAndWx_backend** - 加载本项目导出的 `epoch_040.pt` 与 ONNX 模型,提供 HTTP API
- **Cluster_WebAndWx_WxFrontend** - 微信小程序,经后端间接使用本项目模型
- **RGCNFormer_WebAndWx_WebFrontend** - 网页前端,经后端间接使用本项目模型

## 详细文档 / Detailed Documentation

所有 93+ 个代码文件均已添加**中英双语**顶部注释,涵盖用途、I/O、数据流、相关文件、使用示例。/ All 93+ code files have **bilingual** header comments covering purpose, I/O, data flow, related files, and usage examples.

详细计划见 `.omo/plans/codebase-documentation.md`。/ Detailed plan: `.omo/plans/codebase-documentation.md`.

## 注意事项 / Notes

- 数据通过软链接 `npy/` 共享, 删除软链接会导致训练失败 / Data via `npy/` symlink; breaking it will fail training
- 训练前确保 `dataset/*.py` 与 `npy/` 中的数据对应 / Ensure `dataset/*.py` matches data in `npy/`
- GPU 显存不足时可减小 `batch_size` / Reduce `batch_size` if GPU OOM
- 重构后所有 `model.main_model*` 导入已统一替换为 `model.mrmodn*`; 旧路径不再可用 /
  After refactor, all `model.main_model*` imports are now `model.mrmodn*`; old paths no longer work
- 命名规范见 `tests/test_naming_convention.py`: `<task>_<dataset>_<model>[_<suffix>].py` /
  Naming convention enforced in `tests/test_naming_convention.py`
