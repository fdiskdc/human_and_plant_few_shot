---
layout: home
title: RGCNFormer Tutorial
hero:
  name: RGCNFormer
  text: mRModN RNA Modification Site Classification Tutorial
  tagline: Mixture of Hierarchical Experts · Adaptive Balanced Sampling · Anchor-to-Positive Attention Pooling
  actions:
    - theme: brand
      text: Quickstart
      link: /en/guide/quickstart
    - theme: alt
      text: Architecture
      link: /en/guide/architecture
features:
  - title: 12-Class RNA Modification Prediction
    details: High-accuracy classification for 12 common RNA modification types (m6A, m5C, Ψ, ac4C, etc.).
  - title: Multi-scale Class-Query Architecture
    details: End-to-end model combining multi-scale CNN, GCN graph propagation, and learnable class-query attention heads.
  - title: Few-shot & Zero-shot Analysis
    details: Built-in ac4C balanced/unbalanced, Plant 3-way, and Human zero-shot analysis pipelines.
  - title: Complete Visualization Suite
    details: Attention comparison, spatial motif, UMAP dimensionality reduction and other explainability tools.
  - title: Companion Web Service
    details: Trained ONNX models can be loaded by the Cluster_WebAndWx backend to provide HTTP APIs.
  - title: Bilingual Documentation
    details: Pages are available in both Chinese and English, switchable from the top-right corner.
---

## Tutorial Contents

1. [Project Overview](/en/guide/overview) — Research background, core innovations, and code organization
2. [Algorithm Architecture](/en/guide/architecture) — Deep dive into M2D, MoHE, ABS, and A2P modules
3. [Quickstart](/en/guide/quickstart) — Environment setup, data preparation, and minimal runnable example
4. [Model Details](/en/guide/model) — ParallelCNNBlock, GCNBlock, ClassQueryHead and other core modules
5. [Datasets](/en/guide/dataset) — HRMD-m, PRMD-m, ac4C and dataset loaders
6. [Training Pipeline](/en/guide/training) — Six training scripts and their hyperparameters
7. [Inference Pipeline](/en/guide/inference) — Full-length inference, sliding window, and zero-shot inference
8. [Visualization](/en/guide/visualization) — Attention, motif, UMAP, and other visualization tools
9. [Ablation Studies](/en/guide/ablation) — 3×3 matrix ablation, MoHE ablation, FLOPs calculation
10. [Few/Zero-shot Analysis](/en/guide/fewshot) — ac4C, Plant 3-way, Human zero-shot experiments
11. [Appendix](/en/guide/appendix) — Directory structure, naming conventions, FAQ, and citation

::: tip Switch Language / 切换语言
Click the language switcher in the top-right corner to toggle between 简体中文 and English.
点击右上角语言切换按钮可在「简体中文」与「English」之间切换。
:::
