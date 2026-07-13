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

## 注释标准 / Documentation Standards

本项目采用**中英双语**注释标准,确保代码可读性和国际化。所有新增和修改的代码必须遵循以下规范。/ This project uses **bilingual (Chinese-English)** documentation standards. All new and modified code must follow these conventions.

### 1. 文件头注释 / File Header Comments

每个 `.py` 文件必须以模块级 docstring 开头,包含以下 7 个部分:

```python
"""
<filename>.py - <中文简述> / <English Summary>

<中文详细描述: 本模块的功能、在项目中的角色、核心算法或技术要点>
<English detailed description: module purpose, role in project, key algorithms or technical points>

功能模块 / Modules:
- <ClassName>: <中文描述> / <English description>
- <function_name>: <中文描述> / <English description>

输入 / Inputs:
- <数据源>: <格式说明> - <中文描述> / <English description>
- 命令行参数 / CLI: <参数列表>

输出 / Outputs:
- <输出内容>: <格式说明> - <中文描述> / <English description>

数据流 / Data Flow:
1. <步骤1中文> / <Step 1 English>
2. <步骤2中文> / <Step 2 English>

相关文件 / Related Files:
- 调用 / Calls: <依赖的模块或库>
- 被调用 / Called by: <调用本模块的文件>

使用示例 / Usage Example:
    from <module> import <class_or_function>
    <示例代码>

作者 / Author: RGCNFormer Project
日期 / Date: YYYY-MM-DD
版本 / Version: X.Y
"""
```

**示例** (参考 `model/mrmodn.py`):
```python
"""
mrmodn.py - RGCNFormer 多尺度类查询分类模型 / RGCNFormer Multi-scale Class-Query Model

实现RNA 12类多标签修饰分类主模型，结合多尺度CNN局部特征提取、GCN图结构特征传播,
以及基于可学习类查询的注意力分类头。
Implements the primary RNA 12-class multi-label modification classification model,
combining multi-scale CNN local feature extraction, GCN graph propagation, and
learnable class-query attention classification head.

功能模块 / Modules:
- ParallelCNNBlock: 多尺度并行一维卷积块 / Multi-scale parallel 1D CNN block
- GCNBlock: 残差图卷积块 / Residual GCN block
- ClassQueryHead: 类查询注意力头 / Class-query attention head
...
"""
```

### 2. 类注释 / Class Comments

每个类必须有 docstring,包含功能描述和属性说明:

```python
class MyClass(nn.Module):
    """
    <中文类描述: 功能、用途、技术要点>
    <English class description: purpose, usage, technical details>

    Attributes / 属性:
        attr1 (type): [中文] <描述> / [English] <description>.
        attr2 (type): [中文] <描述> / [English] <description>.
    """
```

**示例** (参考 `model/mrmodn.py`):
```python
class ParallelCNNBlock(nn.Module):
    """
    多尺度并行一维卷积块 (M2D 模块核心) / Multi-scale parallel 1D CNN block (M2D module core).

    使用 4 个不同核大小的并行 1D 卷积捕获 RNA 序列的 k-mer 局部模式 (k=1,3,5,7)。
    Uses 4 parallel 1D convolutions with different kernel sizes (k=1,3,5,7) to capture
    k-mer local patterns.

    Attributes / 属性:
        in_channels (int): [中文] 输入通道数 (固定 4) / [English] input channels (fixed at 4).
        hidden_dim (int): [中文] 隐藏维度 / [English] hidden dimension.
    """
```

### 3. 函数/方法注释 / Function/Method Comments

每个函数必须有 docstring,包含功能、参数、返回值和异常:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    <中文功能描述>
    <English function description>

    Args / 参数:
        param1 (int): [中文] <描述> / [English] <description>.
        param2 (str): [中文] <描述> / [English] <description>. Defaults to "default".

    Returns / 返回值:
        bool: [中文] <描述> / [English] <description>.

    Raises / 异常:
        ValueError: [中文] <触发条件> / [English] <trigger condition>.
    """
```

**简化版** (用于简短函数):
```python
def helper(x: int) -> int:
    """计算平方 / Calculate square."""  # 单行中英双语
    return x * x
```

### 4. 行内注释 / Inline Comments

- **必须使用中英双语** (除非代码逻辑显而易见)
- 注释以 `#` 开头,中英文用 `/` 分隔
- 复杂逻辑、算法步骤、数值含义必须注释

```python
# 常量定义 (与 human_make_npy.py 中的 MOD_TO_INDEX 一致)
# Constants (consistent with MOD_TO_INDEX in human_make_npy.py)
MOD_NAMES = {
    0: 'Am', 1: 'Atol', 2: 'Cm',  # 索引 0-2 / Indices 0-2
    ...
}

# 计算类别权重 (平滑处理以避免极端值)
# Calculate class weights (smoothed to avoid extreme values)
weights = 1.0 / (class_counts + 1e-6)
```

### 5. 常量/全局变量注释 / Constants/Global Variables

```python
# ============================================================================
# Section Name (中英双语)
# ============================================================================

# 12类修饰名称映射 (模型索引 -> 修饰名称)
# 与 human_make_npy.py 中的 MOD_TO_INDEX 一致 (mod_index - 1 转换后)
# 12-class modification name mapping (model index -> modification name)
# Consistent with MOD_TO_INDEX in human_make_npy.py (after mod_index - 1 conversion)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',
    ...
}
```

### 6. 数据结构注释 / Data Structure Comments

```python
# PyG Data 对象字段说明 / PyG Data object field descriptions
# - x: (1001, 4) one-hot 编码的 RNA 序列 / one-hot encoded RNA sequence
# - edge_index: (2, E) 图边索引 (PyG 格式) / graph edge indices (PyG format)
# - y: (1, 12) 12 类多标签二值向量 / 12-class multi-label binary vector
# - y_site: (1001,) 位点级标签 / site-level labels
```

### 7. 特殊标记 / Special Markers

使用以下标记突出重要信息:

```python
# TODO: <待完成任务描述> / <task description>
# FIXME: <已知问题描述> / <known issue description>
# HACK: <临时解决方案> / <workaround description>
# NOTE: <重要说明> / <important note>
# WARNING: <警告信息> / <warning message>
```

### 8. 注释质量检查清单 / Documentation Quality Checklist

新增或修改代码时,确保:

- [ ] 文件头 docstring 包含全部 7 个部分
- [ ] 所有类有完整的 Attributes 文档
- [ ] 所有公共函数有 Args/Returns 文档
- [ ] 复杂逻辑有行内注释
- [ ] 常量有来源说明和用途解释
- [ ] 中英文描述准确对应,无机械翻译痕迹
- [ ] 使用示例可直接运行

### 9. 快速参考 / Quick Reference

| 场景 / Scenario | 格式 / Format |
|---|---|
| 文件头 / File header | `"""filename.py - 中文 / English\n...\n"""` |
| 类 / Class | `"""中文描述\nEnglish description\n\nAttributes / 属性:\n..."""` |
| 函数 / Function | `"""中文功能\nEnglish purpose\n\nArgs / 参数:\n..."""` |
| 单行注释 / Single line | `# 中文说明 / English explanation` |
| 常量 / Constants | `# 中文来源说明 / English source note` |

## 注意事项 / Notes

- 数据通过软链接 `npy/` 共享, 删除软链接会导致训练失败 / Data via `npy/` symlink; breaking it will fail training
- 训练前确保 `dataset/*.py` 与 `npy/` 中的数据对应 / Ensure `dataset/*.py` matches data in `npy/`
- GPU 显存不足时可减小 `batch_size` / Reduce `batch_size` if GPU OOM
- 重构后所有 `model.main_model*` 导入已统一替换为 `model.mrmodn*`; 旧路径不再可用 /
  After refactor, all `model.main_model*` imports are now `model.mrmodn*`; old paths no longer work
- 命名规范见 `tests/test_naming_convention.py`: `<task>_<dataset>_<model>[_<suffix>].py` /
  Naming convention enforced in `tests/test_naming_convention.py`
