---
layout: home
title: RGCNFormer 教程
hero:
  name: RGCNFormer
  text: mRModN RNA 修饰位点分类教程
  tagline: Mixture of Hierarchical Experts · 自适应平衡采样 · 多锚点注意力池化
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/quickstart
    - theme: alt
      text: 算法架构
      link: /guide/architecture
features:
  - title: 12 类 RNA 修饰预测
    details: 支持 m6A、m5C、Ψ、ac4C 等 12 种常见 RNA 修饰类型的高精度分类。
  - title: 多尺度类查询架构
    details: 融合多尺度 CNN、GCN 图传播和可学习类查询注意力头的端到端模型。
  - title: 少样本与零样本分析
    details: 内置 ac4C 平衡/非平衡、Plant 三向独立、Human 零样本等少零样本分析流程。
  - title: 完整可视化体系
    details: 提供注意力对比、空间 motif、UMAP 降维等可视化工具，便于模型解释。
  - title: 配套 Web 服务
    details: 训练好的 ONNX 模型可由 Cluster_WebAndWx 后端加载并对外提供 HTTP API。
  - title: 中英双语
    details: 教程页面提供中英双语切换，方便不同背景的读者阅读。
---

## 教程目录 / Tutorial Contents

1. [项目概述](/guide/overview) — 了解 RGCNFormer 的研究背景、核心创新与代码组织
2. [算法架构](/guide/architecture) — 深入理解 M2D、MoHE、ABS、A2P 四大创新模块
3. [快速开始](/guide/quickstart) — 环境配置、数据准备与最小可运行示例
4. [模型详解](/guide/model) — ParallelCNNBlock、GCNBlock、ClassQueryHead 等核心模块
5. [数据集](/guide/dataset) — HRMD-m、PRMD-m、ac4C 等数据集与加载器
6. [训练流程](/guide/training) — 6 个训练脚本的用法与超参数
7. [推理流程](/guide/inference) — 全长推理、滑窗推理与零样本推理
8. [可视化](/guide/visualization) — 注意力、motif、UMAP 等可视化工具
9. [消融实验](/guide/ablation) — 3×3 矩阵消融、MoHE 消融、FLOPs 计算
10. [少/零样本分析](/guide/fewshot) — ac4C、Plant 3-way、Human 零样本分析
11. [附录](/guide/appendix) — 目录结构、命名规范、常见问题与引用

::: tip 切换语言 / Switch Language
点击右上角语言切换按钮可在「简体中文」与「English」之间切换。
Click the language switcher in the top-right corner to toggle between 简体中文 and English.
:::
