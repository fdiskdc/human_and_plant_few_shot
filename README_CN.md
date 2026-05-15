# RGCNFormer

基于 CNN + GCN + Class-Query Attention 的 RNA 修饰多标签分类框架

## 项目概述

RGCNFormer 是一个深度学习框架，用于从 1001 核苷酸序列中预测 12 种 RNA 转录后修饰类型。核心模型 (`RNA_ClassQuery_Model`) 结合了：

- **并行 CNN**：多尺度卷积核 (1, 3, 5, 7) 提取局部序列模式
- **GCN**：图卷积网络沿 RNA 二级结构边传播信息
- **Class-Query Attention**：每类独立的交叉注意力头，层级式 4 组 (A/C/G/U) -> 12 类推导

### 支持的修饰类型

| 索引 | 类型  | 核苷酸 | 索引 | 类型  | 核苷酸 |
|------|-------|--------|------|-------|--------|
| 0    | Am    | A      | 6    | ac4C  | C      |
| 1    | Atol  | A      | 7    | m1A   | A      |
| 2    | Cm    | C      | 8    | m5C   | C      |
| 3    | Gm    | G      | 9    | m6A   | A      |
| 4    | Tm    | U      | 10   | m6Am  | A      |
| 5    | Y     | U      | 11   | m7G   | G      |

### 基线模型

- **MultiRM** (`model/multirm.py`)：BiLSTM + Bahdanau 注意力
- **ModX** (`model/modx.py`)：BiLSTM + Bahdanau 注意力 + Word2Vec 风格输入
- **EvoRMD** (`model/evormd_human.py`)：Conv1d 嵌入器 + 可训练注意力 + MLP
- **mRModN**：GCN + 每类 MHA（全长推理）

---

## 项目结构

```
rgcnformer_sum/
├── model/                          # 模型定义
│   ├── main_model.py               # 核心 RGCNFormer 模型 (CNN + GCN + Class-Query)
│   ├── main_model_multirm.py       # MultIRM 数据集变体
│   ├── main_model_collect_atten.py # 返回中间注意力的变体
│   ├── multirm.py                  # MultiRM 基线 (BiLSTM + Bahdanau)
│   ├── multirm_collect_atten.py    # MultiRM 注意力收集版
│   ├── modx.py                     # ModX 基线 (BiLSTM + Word2Vec)
│   ├── modx_collect_atten.py       # ModX 注意力收集版
│   ├── evormd_human.py             # EvoRMD 基线 (Conv1d + Attention)
│   ├── abla_model.py               # 消融实验变体
│   └── EvoRMD/Script/              # 原始 EvoRMD 参考实现
│
├── dataset/                        # 数据集类
│   ├── human.py                    # 人类 RNA 修饰数据集（主数据集）
│   ├── human_with_seq.py           # 增强版（含序列字符串）
│   ├── human_motif.py              # 空间模体分析用数据集
│   ├── plant.py                    # 植物 RNA 修饰数据集
│   ├── plant_single.py             # 植物 + 零样本组合数据集
│   ├── multirm.py                  # MultIRM 12 类数据集 (HDF5)
│   ├── ac4c.py                     # AC4C 修饰数据集
│   ├── gen3.py                     # 三代数据集
│   └── gen3_zero.py                # 三代 + 背景样本数据集
│
├── utils/                          # 工具函数
│   ├── common.py                   # 核心：常量、配置、采样器、数据分割、训练
│   ├── metrics.py                  # 评估指标与阈值优化
│   ├── logging.py                  # 日志与格式化表格输出
│   ├── few_shot.py                 # 少样本学习基准
│   ├── fewshot_analysis_*.py       # 少样本分析管线模块
│   ├── fewshot_export_helpers.py   # CSV/JSON 导出（供 R 可视化）
│   ├── rna_visualization.py        # RNA 结构与注意力可视化
│   ├── train_gen3.py               # 三代数据集训练脚本
│   └── Zero_structures.py          # 零样本数据集二级结构预计算
│
├── atten_comp/                     # 注意力比较脚本 (v2)
│   ├── run_attention_comparison_v2.py
│   ├── visualize_attention_comparison_v2.py
│   └── inference_modx_full.py
│
├── json/                           # 配置文件
├── npy/                            # 数据集文件 (.npy)
├── cache/                          # 结构缓存文件
├── logs/                           # 训练日志
│
├── train_human.py                  # 主训练脚本（人类数据）
├── train_plant.py                  # 植物数据训练脚本
├── train_multirm_dataset.py        # MultIRM 数据集训练脚本
├── train_human_multirm.py          # MultiRM 模型训练脚本
├── train_human_modx.py             # ModX 模型训练脚本
├── train_human_evormd.py           # EvoRMD 模型训练脚本
│
├── inference_modx_segmented.py     # ModX 滑动窗口推理
├── inference_mrmodn_full.py        # mRModN 全长推理
├── inference_multirm_segmented.py  # MultiRM 滑动窗口推理
├── inference_evormd_segmented.py   # EvoRMD 滑动窗口推理
│
├── fewshot_plant_3way_independent.py   # 植物三路少样本
├── fewshot_ac4c_balance.py             # AC4C 平衡少样本
├── fewshot_ac4c_unbalan.py             # AC4C 非平衡少样本
├── zero_shot_fewshot_analysis.py       # 零/少样本分析管线
├── zero_shot_fewshot_extract_only.py   # 数据导出（供 R 可视化）
│
├── run_attention_comparison.py     # 注意力比较编排器
├── visualize_attention_comparison.py
├── select_representative_sequences.py
├── sliding_window_utils.py         # 高斯加权拼接工具
├── SpatialMotif.py                 # 空间模体可视化
├── SpatialMotif_nobackground.py    # 空间模体（干净版）
├── prepare_umap_data.py            # UMAP 管线
├── prepare_umap_from_npz.py        # 从预收集数据生成 UMAP
├── abla_mohe.py                    # 消融实验运行器
├── cal_flops_mohe.py               # FLOPs 计算器
├── collect_human_atten.py          # 收集人类注意力输出
├── collect_human.py                # 收集人类输出到 Excel
├── collect_modx_atten.py           # 收集 ModX 注意力
├── collect_multirm_atten.py        # 收集 MultiRM 注意力
├── view_human.py                   # 小提琴图（每类）
├── view_human_total.py             # 小提琴图（汇总）
├── view_npz.py                     # NPZ 文件检查器
├── view3.py                        # 多数据集小提琴图
├── 3x3.py                          # 人类到植物零样本迁移
├── 3x3_2.py                        # 迁移分析 v2
├── test_gen3.py                    # 三代数据评估脚本
├── test_multirm_4class.py          # MultIRM 4 类测试
├── test_multirm_oversampling.py    # MultIRM 过采样测试
├── cal_mean_median_mode.py         # 每类统计量计算
│
├── plot_zero_fewshot_analysis.R    # R：从 CSV 生成出版级图表
└── plot_zero_fewshot_export.R      # R：从 CSV/JSON 导出并生成图表
```

---

## Python 脚本说明

### 训练脚本

| 脚本 | 说明 |
|------|------|
| `train_human.py` | 人类 12 类 RNA 修饰分类主训练脚本。多标签不相交分割 (7:3)、平滑类别权重、动态平衡批次采样、双模式评估 (Unbalance & BalanceB)、TensorBoard 日志。 |
| `train_plant.py` | 植物数据集训练，支持少样本基准。使用 `RNA_ClassQuery_Model` 与支持集采样。 |
| `train_multirm_dataset.py` | MultIRM 12 类数据集训练 (51nt 序列)。正/负样本配对、每类独立损失、HDF5 数据。 |
| `train_human_multirm.py` | 使用 MultiRM 基线模型 (`model_v3`: BiLSTM + Bahdanau) 的人类数据训练。 |
| `train_human_modx.py` | 使用 ModX 基线模型 (`RNAClassifierWithWord2Vec`: BiLSTM + Bahdanau) 的人类数据训练。 |
| `train_human_evormd.py` | 使用 EvoRMD 基线模型 (`EvoRMDForHuman`: Conv1d + 可训练注意力) 的人类数据训练。 |

### 推理脚本

| 脚本 | 说明 |
|------|------|
| `inference_modx_segmented.py` | ModX 101nt 滑动窗口推理。收集每类注意力 `[N, 12, 1001]`，高斯加权拼接。 |
| `inference_mrmodn_full.py` | mRModN 全长 1001nt 推理。GCN + 每类 MHA，向量化 one-hot，批量 edge_index。 |
| `inference_multirm_segmented.py` | MultiRM 51nt 滑动窗口推理 (stride=1)。BiLSTM + Bahdanau，高斯加权拼接。 |
| `inference_evormd_segmented.py` | EvoRMD 41nt 滑动窗口推理 (stride=20)。Conv1d 嵌入器，距离加权拼接。 |

### 少样本脚本

| 脚本 | 说明 |
|------|------|
| `fewshot_plant_3way_independent.py` | 植物 3 路分类，使用独立二元一对多模型 (Y/m5C/m6A)。训练 3 个独立模型，二元焦点损失。 |
| `fewshot_ac4c_balance.py` | AC4C 平衡数据集训练。裁剪模型仅计算 AC4C 查询 (类索引 6)，标签平滑损失。 |
| `fewshot_ac4c_unbalan.py` | AC4C 非平衡数据集训练。与平衡版相同架构。 |
| `zero_shot_fewshot_analysis.py` | 零样本特征对齐与少样本轨迹分析主入口。输出 PNG+PDF 图表。 |
| `zero_shot_fewshot_extract_only.py` | 仅数据导出管线。提取特征、计算指标、生成 UMAP 坐标，导出 CSV/JSON 供 R 可视化。 |

### 分析与可视化

| 脚本 | 说明 |
|------|------|
| `run_attention_comparison.py` | 编排器：运行完整注意力比较管线（选序列 -> 4 模型推理 -> 可视化）。 |
| `visualize_attention_comparison.py` | 生成每序列 m6A 注意力比较图（4 模型）。期刊级 PDF+PNG 输出。 |
| `sliding_window_utils.py` | 滑动窗口推理的高斯加权拼接工具。 |
| `select_representative_sequences.py` | 按 m6A 修饰密度选择代表性序列（高/低密度组）。 |
| `SpatialMotif.py` | 使用 Top-K 序列标志可视化"空间模体"，潜空间聚类 (PCA + K-Means)。 |
| `SpatialMotif_nobackground.py` | 空间模体可视化（硬置零版，无背景噪声）。 |
| `abla_mohe.py` | `HierarchicalClassQueryHeadPooling` 消融实验。运行 8 种配置，输出 CSV + 3 张图。 |
| `cal_flops_mohe.py` | 16 种消融配置的手动 FLOPs 计算器。 |
| `prepare_umap_data.py` | 完整 UMAP 管线：模型推理 -> 特征收集 -> UMAP -> JSON 导出。 |
| `prepare_umap_from_npz.py` | 从预收集的 `human_atten.npz` 准备 UMAP 数据。 |
| `collect_human_atten.py` | 收集人类数据集的注意力输出。保存到 `human_atten.npz`。 |
| `collect_human.py` | 收集模型输出（logits + 注意力）到 Excel。 |
| `collect_modx_atten.py` | 收集 ModX 模型注意力输出。保存到 `modx_atten.npy`。 |
| `collect_multirm_atten.py` | 收集 MultiRM 模型注意力输出。保存到 `multirm_atten.npy`。 |
| `view_human.py` | 人类数据每类修饰位点分布小提琴图。 |
| `view_human_total.py` | 总修饰位点（12 类求和）小提琴图。 |
| `view_npz.py` | NPZ 文件检查器 / 完整性检查工具。 |
| `view3.py` | 多数据集小提琴图可视化，莫兰迪配色。 |
| `3x3.py` | 3x3 人类到植物零样本迁移分析（一对多）。 |
| `3x3_2.py` | 迁移分析 v2（植物正样本 + 植物其他修饰负样本）。 |
| `test_gen3.py` | 三代 (gen3) 数据集评估脚本，含 top-k 召回率。 |
| `test_multirm_4class.py` | MultIRM 4 类修饰映射测试脚本。 |
| `test_multirm_oversampling.py` | MultIRM 过采样与最大长度对齐策略测试。 |
| `cal_mean_median_mode.py` | 计算每类修饰位点数的均值/中位数/众数。 |

### 模型定义 (`model/`)

| 脚本 | 说明 |
|------|------|
| `main_model.py` | **核心 RGCNFormer 模型**：`ParallelCNNBlock`（多尺度卷积核）、`GCNBlock`（残差 GCNConv）、`HierarchicalClassQueryHeadPooling`（4 组层级 MHA）、`RNA_ClassQuery_Model`。 |
| `main_model_multirm.py` | 适配 MultIRM 数据集的变体 (51nt 序列，4 组 A/C/G/U)。 |
| `main_model_collect_atten.py` | 修改版 main_model，返回中间注意力输出用于分析。 |
| `multirm.py` | MultiRM 基线：`NaiveNet`（纯 CNN）、`model_v3`（BiLSTM + Bahdanau 共享注意力）。 |
| `multirm_collect_atten.py` | MultiRM 变体，返回注意力权重和上下文向量。 |
| `modx.py` | ModX 基线：`BahdanauAttention`、`RNAClassifierWithWord2Vec`（输入投影 + BiLSTM）。 |
| `modx_collect_atten.py` | ModX 变体，返回注意力权重和上下文向量。 |
| `evormd_human.py` | EvoRMD 基线：`Conv1dEmbedder`、`TrainableAttention`（MIL 池化）、`MulticlassClassifier`。 |
| `abla_model.py` | 消融变体：`1Query`、`4Query`、`12Query`、`FullAttn`。 |

### 数据集类 (`dataset/`)

| 脚本 | 说明 |
|------|------|
| `human.py` | 主要人类 RNA 修饰数据集 (`Mer100Dataset`)。加载 `seq.npy`/`12loc.npy`/`1001loc.npy`/`4loc.npy`，LinearFold 二级结构计算与缓存，PyG Data 对象。 |
| `human_with_seq.py` | 增强版数据集，Data 对象中包含原始序列字符串。 |
| `human_motif.py` | 空间模体分析的独立数据集。 |
| `plant.py` | 植物 RNA 修饰数据集 (`PlantDataset`)。与人类数据集同格式。 |
| `plant_single.py` | 植物 + 零样本组合数据集，用于二分类。 |
| `multirm.py` | MultIRM 12 类数据集。基于 HDF5，正/负样本配对，过采样。 |
| `ac4c.py` | AC4C 修饰数据集（平衡和非平衡变体）。 |
| `gen3.py` | 三代 RNA 修饰数据集。 |
| `gen3_zero.py` | 三代数据集 + 背景/负样本。 |

### 工具函数 (`utils/`)

| 脚本 | 说明 |
|------|------|
| `common.py` | 核心工具：常量 (`MOD_NAMES`, `GROUP_TO_CLASS_INDICES`)、配置加载、检查点保存、批次采样器 (`MultilabelBalancedBatchSampler`, `DynamicBalancedBatchSampler`)、数据分割、训练/测试循环。 |
| `metrics.py` | 评估指标：`evaluate_unbalance()`、`evaluate_balanceb()`、`evaluate_with_optimal_threshold()`、`evaluate_4class_with_optimal_threshold()`、top-k 召回率、综合定位指标。 |
| `logging.py` | 日志设置（文件 + 控制台）、TensorBoard 集成、格式化表格输出。 |
| `few_shot.py` | 少样本学习基准：`run_few_shot_benchmark()`、`run_few_shot_benchmark_ac4c()`、支持集构建。 |
| `fewshot_analysis_constants.py` | 少样本分析管线常量（目标类、shot 数、颜色）。 |
| `fewshot_analysis_utils.py` | 共享工具：输出目录管理、图表保存、绘图样式。 |
| `fewshot_analysis_features.py` | 从模型中间层提取特征（基于 hook）。 |
| `fewshot_analysis_zeroshot.py` | 零样本特征对齐分析与 UMAP。 |
| `fewshot_analysis_fewshot.py` | 少样本轨迹分析（微调、距离指标）。 |
| `fewshot_analysis_gen3.py` | 三代数据分析与人类比较。 |
| `fewshot_analysis_spatial_motif.py` | 使用积分梯度 (captum) 的空间模体分析。 |
| `fewshot_export_helpers.py` | CSV/JSON 导出工具（供 R 可视化）。 |
| `rna_visualization.py` | RNA 结构与注意力可视化 (forgi/matplotlib)。 |
| `train_gen3.py` | 三代数据集训练脚本（与 train_human.py 同流程）。 |
| `test_gen3_analyse.py` | 三代测试与概率密度分布图。 |
| `check_m6a_data_integrity.py` | m6A 数据完整性检查器。 |
| `audit_12loc_structure.py` | 12loc.npy 标签矩阵深度审计。 |
| `Zero_structures.py` | 零样本数据集 RNA 二级结构预计算。 |

### 注意力比较 (`atten_comp/`)

| 脚本 | 说明 |
|------|------|
| `run_attention_comparison_v2.py` | 编排器 v2：4 模型完整注意力比较管线。 |
| `visualize_attention_comparison_v2.py` | 注意力比较 v2，使用背景色块替代虚线。 |
| `inference_modx_full.py` | ModX 全长 1001nt 推理（单次前向，无滑动窗口）。 |

---

## R 脚本说明

| 脚本 | 说明 |
|------|------|
| `plot_zero_fewshot_analysis.R` | 从 Python 导出的 CSV 数据生成出版级图表。处理植物和三代数据可视化。使用 ggplot2、dplyr、tidyr、viridis。 |
| `plot_zero_fewshot_export.R` | 从 Python 导出的 CSV/JSON 数据生成出版级图表。导出 + 可视化管线。 |

---

## 快速开始

### 训练

```bash
# 在人类数据集上训练 RGCNFormer
python train_human.py --config json/human.json

# 使用 MultiRM 基线训练
python train_human_multirm.py --config json/human.json

# 使用 ModX 基线训练
python train_human_modx.py --config json/human.json

# 使用 EvoRMD 基线训练
python train_human_evormd.py --config json/human.json
```

### 推理

```bash
# 收集注意力输出
python collect_human_atten.py

# 运行 4 模型注意力比较
python run_attention_comparison.py
```

### 少样本分析

```bash
# 运行零样本和少样本分析
python zero_shot_fewshot_analysis.py

# 导出数据供 R 可视化
python zero_shot_fewshot_extract_only.py

# 生成出版级图表 (R)
Rscript plot_zero_fewshot_analysis.R --input_dir output --output_dir fig
```

---

## 依赖

### Python

```
torch >= 1.12
torch-geometric >= 2.1
numpy
scikit-learn
pandas
tqdm
prettytable
tensorboard
captum (用于 SpatialMotif)
logomaker (用于 SpatialMotif)
matplotlib
```

### R

```
ggplot2
dplyr
tidyr
readr
scales
forcats
viridis
ggrastr (可选，用于 PDF 中的栅格化散点)
```

---

## 许可

本项目仅供研究使用。
