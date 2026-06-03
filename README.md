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
| **模型定义** / Model definition | RGCNFormer 主网络(RGCN + Transformer)、多任务变体(ModX、MultiRM、EvoRMD)/ RGCNFormer backbone + variants |
| **数据集加载** / Dataset loading | Human / Plant / ac4C / MultiRM / gen3 数据集 / Datasets |
| **训练流程** / Training pipeline | 5 个训练脚本(单修饰、多修饰、零样本等)/ 5 training scripts |
| **推理流程** / Inference pipeline | 4 个推理脚本(分段、批量)/ 4 inference scripts |
| **数据收集** / Data collection | 4 个 attention 收集脚本 / 4 attention collection scripts |
| **少样本分析** / Few-shot analysis | 零样本/少样本场景分析 / Zero-shot / few-shot analysis |
| **可视化** / Visualization | 注意力对比、序列选择、空间 motif / Attention comparison, sequence selection, spatial motifs |
| **消融实验** / Ablation studies | 3×3 矩阵、FLOPs 计算、mohe 消融 / 3×3 matrix, FLOPs, mohe ablation |
| **R 可视化** / R visualization | 零样本/少样本分析图表 / Zero-shot / few-shot plots |
| **Jupyter Notebooks** | 34 个分析 notebook / 34 analysis notebooks |

## 目录结构 / Directory Layout

```
rgcnformer_sum/
├── model/                          # 模型定义(10 个 .py)
│   ├── main_model.py               # RGCNFormer 主网络(ParallelCNNBlock + GCNBlock + ClassQueryHead)
│   ├── multirm.py                  # 多任务多修饰变体
│   ├── modx.py                     # 修饰类型消融变体
│   ├── evormd_human.py             # EvoRMD 集成
│   ├── abla_model.py               # 消融实验模型
│   ├── main_model_multirm.py       # 多 RM 主模型
│   ├── main_model_collect_atten.py # 主模型(收集注意力)
│   ├── modx_collect_atten.py       # ModX 注意力收集
│   └── multirm_collect_atten.py    # MultiRM 注意力收集
│
├── dataset/                        # 数据集加载(9 个 .py)
│   ├── human.py                    # Human(标准 12 修饰)
│   ├── plant.py                    # Plant(植物)
│   ├── ac4c.py                     # ac4C 专用
│   ├── multirm.py                  # 多 RM
│   ├── gen3.py                     # 第 3 代数据集
│   ├── human_motif.py              # 带 motif 的 Human
│   ├── human_with_seq.py           # 带序列特征的 Human
│   ├── gen3_zero.py                # 零样本 gen3
│   └── plant_single.py             # 单修饰 Plant
│
├── utils/                          # 工具与少样本分析(18 个 .py)
│   ├── common.py                   # 公共常量、评估函数
│   ├── metrics.py                  # 指标计算
│   ├── logging.py                  # 日志
│   ├── train_gen3.py               # gen3 训练
│   ├── test_gen3_analyse.py        # gen3 测试分析
│   ├── few_shot.py                 # 少样本核心
│   ├── fewshot_analysis_*.py       # 6 个少样本分析脚本
│   ├── rna_visualization.py        # RNA 可视化
│   ├── audit_12loc_structure.py    # 12 定位结构审计
│   ├── check_m6a_data_integrity.py # m6A 数据完整性检查
│   └── Zero_structures.py          # 零样本结构
│
├── 顶层脚本 / Top-level scripts:
│   ├── train_*.py                  # 6 个训练脚本(human / plant / multirm / evormd / modx / multirm_dataset)
│   ├── inference_*.py              # 4 个推理脚本(分段 + 全量)
│   ├── collect_*.py                # 4 个数据/注意力收集脚本
│   ├── fewshot_*.py                # 3 个少样本分析脚本
│   ├── 3x3.py / 3x3_2.py           # 3×3 矩阵消融
│   ├── abla_mohe.py                # MoHE 消融
│   ├── cal_flops_mohe.py           # FLOPs 计算
│   ├── cal_mean_median_mode.py     # 统计计算
│   ├── view_*.py                   # 4 个可视化脚本
│   ├── visualize_*.py / run_*.py / select_*.py  # 注意力对比
│   ├── test_*.py                   # 3 个测试脚本
│   ├── SpatialMotif*.py            # 空间 motif 提取
│   ├── prepare_umap_*.py           # UMAP 数据准备
│   ├── sliding_window_utils.py     # 滑窗工具
│   ├── plot_zero_fewshot_*.R       # 2 个 R 可视化脚本
│   └── zero_shot_*.py              # 零样本分析
│
├── atten_comp/                     # 注意力对比子模块(3 个 .py)
├── gen3process/                    # gen3 数据预处理(9 个 .py)
├── multirmprocess/                 # 多 RM 数据预处理(10 个 .py)
│
├── ipynb/                          # 34 个分析 notebook
│   ├── check_data.ipynb
│   ├── cls_compare*.ipynb          # 分类对比
│   ├── loc_compare*.ipynb          # 定位对比
│   ├── rgcnformer_*.ipynb          # 主模型分析
│   ├── umap*.ipynb                 # UMAP 嵌入
│   ├── violin.ipynb                # 小提琴图
│   └── ... (34 total)
│
├── model/EvoRMD/                   # EvoRMD 子模块(7 个 .py)
│   └── Script/
│       ├── dataset.py
│       ├── utils.py
│       ├── main.py
│       ├── preprocess_data.py
│       ├── embedding.py
│       ├── downsampling.py
│       ├── train_val_test.py
│       └── model.py
│
├── output/                         # 训练输出
├── logs/                           # 训练日志
├── npy/                            # 数据软链接(/home/dc/vscode/npyForTrain)
├── docs/ / fig/ / figs_atten/ / motif_logo*/  # 文档与图表
├── dataset/                        # 数据集子目录
├── cache/                          # 缓存
├── json/                           # 中间 JSON
└── .omo/plans/codebase-documentation.md  # 本项目的代码级文档计划
```

## 启动方式 / Getting Started

### 环境要求 / Requirements

- **Python 3.8+**(推荐 3.10)/ Python 3.8+ (3.10 recommended)
- **PyTorch 1.10+** + **PyTorch Geometric**(用于图卷积)/ PyTorch + PyG
- **numpy / pandas / scikit-learn / matplotlib**
- **R 4.0+**(用于 R 可视化脚本)/ R 4.0+ (for R viz scripts)
- **JupyterLab / Notebook**(用于 ipynb/)/ Jupyter for notebooks
- 训练用 GPU(推荐 NVIDIA V100/A100,显存 ≥ 16GB)/ Training GPU recommended

### 安装 / Installation

```bash
cd rgcnformer_sum
pip install torch torch-geometric numpy pandas scikit-learn matplotlib
# 或使用项目根目录的 requirements(如有)/ or use project requirements if present
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
# 训练 Human 数据集上的 RGCNFormer
python train_human.py --epochs 40 --batch-size 32

# 训练多 RM 数据集
python train_human_multirm.py

# 训练 ModX(消融)
python train_human_modx.py
```

训练产物保存到 `output/`,日志保存到 `logs/`。/ Outputs to `output/`, logs to `logs/`.

### 推理示例 / Inference Example

```bash
# 对新序列推理(分段)
python inference_modx_segmented.py --input sequences.fasta

# 多 RM 推理
python inference_mrmodn_full.py
```

### 可视化 / Visualization

```bash
# 注意力对比
python atten_comp/run_attention_comparison_v2.py

# R 脚本:零样本/少样本图表
Rscript plot_zero_fewshot_analysis.R
```

### Jupyter Notebooks

```bash
jupyter lab ipynb/
# 打开 rgcnformer_cls_res.ipynb 等进行交互式分析
```

## 关键文件说明 / Key Files

### 模型 / Models
- `model/main_model.py` - RGCNFormer 主网络:ParallelCNNBlock(多尺度卷积)+ GCNBlock(图卷积)+ ClassQueryHead(类查询注意力)
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
- `train_human.py` - 主训练脚本(Human)/ Main training (Human)
- `train_plant.py` - Plant 训练
- `train_human_multirm.py` - 多 RM 训练

### 推理 / Inference
- `inference_modx_segmented.py` - ModX 分段推理
- `inference_mrmodn_full.py` - 多 RM 全量推理
- `inference_multirm_segmented.py` - 多 RM 分段

### 少样本分析 / Few-shot Analysis
- `fewshot_ac4c_balance.py` / `fewshot_ac4c_unbalan.py` - ac4C 平衡/非平衡
- `fewshot_plant_3way_independent.py` - Plant 三向独立
- `zero_shot_fewshot_analysis.py` - 零样本+少样本综合

### 工具 / Utils
- `utils/common.py` - 公共常量(NUCLEOTIDE_MAP、k-mer)、评估函数
- `utils/metrics.py` - ACC / F1 / MCC 等指标
- `utils/few_shot.py` - 少样本核心算法

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

- 数据通过软链接 `npy/` 共享,删除软链接会导致训练失败 / Data via `npy/` symlink; breaking it will fail training
- 训练前确保 `dataset/*.py` 与 `npy/` 中的数据对应 / Ensure `dataset/*.py` matches data in `npy/`
- GPU 显存不足时可减小 `batch_size` / Reduce `batch_size` if GPU OOM
- 34 个 ipynb 文件中部分可能引用旧的路径/模型,运行前请检查 / Some ipynb may reference old paths/models; check before running
