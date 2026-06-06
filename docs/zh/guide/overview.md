# 项目概述 / Project Overview

## 1. 研究背景 / Research Background

RNA 修饰是近年来表观转录组学（epitranscriptomics）研究的核心议题之一。迄今为止，研究人员已经在各类 RNA 分子上鉴定出 **170 余种**化学修饰类型，其中以 `m6A`、`m5C`、`Ψ`、`ac4C`、`m1A`、`m2G` 等为代表的 12 类修饰在 mRNA 上分布最广、功能研究最深入。这些修饰广泛参与 mRNA 的剪接、出核、翻译、降解等关键过程，与癌症、神经发育疾病、代谢紊乱等密切相关。

RNA modifications are a central topic in epitranscriptomics research. To date, more than **170 distinct chemical modifications** have been identified on various RNA molecules. Among them, 12 well-characterized types — `m6A`, `m5C`, `Ψ`, `ac4C`, `m1A`, `m2G`, etc. — are the most abundant on mRNA and the most deeply studied. These modifications are involved in splicing, nuclear export, translation and degradation, and are closely linked to cancer, neurodevelopmental disorders and metabolic diseases.

## 2. 研究动机 / Motivation

传统 RNA 修饰检测方法（如 `iCLIP`、`miCLIP`、`SCARLET`、`MAZTER-seq`）虽然准确度高，但存在以下不足：

Traditional RNA modification detection methods (`iCLIP`, `miCLIP`, `SCARLET`, `MAZTER-seq`) are accurate but have notable limitations:

- **成本高 / Costly**: 单次实验需要数千元到数万元的试剂与测序费用
- **通量低 / Low-throughput**: 一次仅能检测一种修饰或一个位点
- **抗体依赖 / Antibody-dependent**: 抗体特异性差异导致假阳性/假阴性
- **位点分辨率有限 / Limited single-site resolution**

深度学习可基于 RNA 序列本身的碱基组合和上下文特征进行高通量、低成本预测，是 RNA 修饰组学的重要补充手段。

Deep learning enables high-throughput, low-cost prediction from RNA sequence alone, providing a powerful complement to wet-lab assays.

## 3. mRModN 的定位 / Position of mRModN

**mRModN (Mixture of Hierarchical Experts with self-adaptive balanced sampling for RNA modification prediction across species)** 是本项目背后的核心方法，其关键创新点包括：

**mRModN (Mixture of Hierarchical Experts with self-adaptive balanced sampling for RNA modification prediction across species)** is the core method of this project. Its key innovations are:

| 创新模块 / Module | 全称 / Full Name | 主要作用 / Purpose |
|---|---|---|
| **M2D** | Multi-view Motif Discovery | 多尺度卷积捕获 k-mer 局部模式 / Multi-scale CNN for local k-mer patterns |
| **MoHE** | Mixture of Hierarchical Experts | 层级专家混合 + 路由 / Hierarchical experts with query routing |
| **ABS** | Adaptive Balanced Sampler | 自适应类别平衡采样 / Adaptive class-balanced sampling |
| **A2P** | Anchor-to-Positive Pooling | 多锚点注意力池化 / Multi-anchor attention pooling |

## 4. 支持的 12 类修饰 / Supported 12 Modifications

| 索引 / Index | 修饰 / Modification | 含义 / Meaning |
|---|---|---|
| 0 | m6A | N6-methyladenosine |
| 1 | m5C | 5-methylcytosine |
| 2 | m1A | N1-methyladenosine |
| 3 | m2G | N2-methylguanosine |
| 4 | Ψ | Pseudouridine |
| 5 | ac4C | N4-acetylcytosine |
| 6 | m7G | N7-methylguanosine |
| 7 | m3C | 3-methylcytosine |
| 8 | I | Inosine (A-to-I editing) |
| 9 | s2U | 2-thiouridine |
| 10 | D | Dihydrouridine |
| 11 | Nm | 2'-O-methylation |

## 5. 仓库组织 / Repository Organization

本仓库 `human_and_plant_few_shot` 是整个 RGCNFormer 研究系统的**核心训练与推理代码库**，包含：

This repository `human_and_plant_few_shot` is the **core training & inference codebase** of the entire RGCNFormer system:

- `model/` — 10 个模型定义文件（主模型 + 4 个变体 + 消融）/ 10 model files
- `dataset/` — 9 个数据集加载器 / 9 dataset loaders
- `utils/` — 20 个工具脚本（含少样本分析）/ 20 utility scripts
- 根目录 `train_*.py` / `inference_*.py` / `fewshot_*.py` / `zeroshot_*.py` / `collect_*.py` 共 20 个训练/推理/分析脚本
- `ablation/` — 5 个消融实验脚本 / 5 ablation scripts
- `visualization/` — 14 个可视化脚本 / 14 visualization scripts
- `tests/` — 10 个 pytest 测试用例 / 10 pytest test cases

## 6. 与其他项目的关系 / Relation to Other Projects

```
rgcnformer_sum (本项目 / this repo)
    │  训练并导出 ONNX / train and export ONNX
    ▼
Cluster_WebAndWx_backend  ←── HTTP API 入口 / HTTP API entry
    │
    ├──→ Cluster_WebAndWx_WxFrontend (微信小程序 / WeChat MiniApp)
    └──→ RGCNFormer_WebAndWx_WebFrontend (网页前端 / Web Frontend)
```

- **Cluster_WebAndWx_backend**：加载本项目导出的 `epoch_040.pt` 与 ONNX 模型，提供 HTTP API
- **Cluster_WebAndWx_WxFrontend**：微信小程序，通过后端间接使用本项目模型
- **RGCNFormer_WebAndWx_WebFrontend**：网页前端，通过后端间接使用本项目模型

Loads checkpoints exported from this project and provides HTTP APIs. / WeChat MiniApp via backend. / Web frontend via backend.

## 7. 下一步 / Next Steps

- 继续阅读 [算法架构](/guide/architecture) 了解 mRModN 的四大创新模块
- 跳转 [快速开始](/guide/quickstart) 跑通最小训练流程

Continue to [Algorithm Architecture](/guide/architecture) to understand the four innovation modules, or jump to [Quickstart](/guide/quickstart) to run a minimal training pipeline.
