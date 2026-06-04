"""
model/__init__.py - model 包初始化 / model 包初始化 (model package init)

model 包初始化:model package init。
model package init.

功能模块 / Modules:
- 无(包标记文件)/ None, package marker
- (详见源代码 / see source code)

输入 / Inputs:
- 命令行参数(超参、数据路径等)/ CLI args (hyperparams, data paths, etc.)
- 配置文件(json/yaml) / config files (json/yaml)
- 上一阶段产物(.npy/.pt/.json)/ prior-stage outputs

输出 / Outputs:
- checkpoint(.pt) / 指标(.json) / 图(.png/.pdf) / 注意力文件
- checkpoints (.pt) / metrics (.json) / figures (.png/.pdf) / attention files

数据流 / Data Flow:
1. 读数据 → 构建 DataLoader / Load data → build DataLoader
2. 实例化模型 / Instantiate model
3. 训练/推理/分析主循环 / train/inference/analysis main loop
4. 保存产物 / save outputs

相关文件 / Related Files:
- 调用 / Calls: model/*.py
- 被调用 / Called by: model/*.py 相关的训练/推理/分析脚本 / related training/inference/analysis scripts

使用示例 / Usage Example:
    python model/__init__.py --epochs 40 --batch-size 32

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

'''
Author: Chao Deng && chaodeng987@outlook.com
Date: 2026-01-11 10:37:11
LastEditors: Chao Deng && chaodeng987@outlook.com
LastEditTime: 2026-01-11 15:48:33
FilePath: /human_and_plant_few_shot_fastAtten_ac4c/model/__init__.py
Description: 
那只是一场游戏一场梦
 
https://orcid.org/0009-0009-8520-1656
DOI: 10.3390/app15158626
DOI: 10.3390/rs17142354
Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
'''
"""
RNA_ClassQuery_Model Package

This package implements a multi-scale Class-Query classification model for RNA
secondary structure prediction and 12-class multi-label classification.

Main Model:
    RNA_ClassQuery_Model: Base model with CNN + GCN + Class-Query attention
    RNA_ClassQuery_Model_Large: Larger version with more capacity

Sub-modules (from mrmodn.py):
    ParallelCNNBlock: Multi-scale parallel CNN feature extraction
    GCNBlock: Graph Convolutional Network block with residual connections
    ClassQueryHead: Cross-attention based classification head
    ClassQueryHeadPooling: Simplified attention pooling version
    HierarchicalClassQueryHeadPooling: Hierarchical head with Group-to-Class derivation

Example Usage:
    >>> from model import RNA_ClassQuery_Model
    >>> model = RNA_ClassQuery_Model()
    >>> logits = model(x, edge_index, batch)
    >>> predictions, probs = model.predict(x, edge_index, batch)
"""

from .mrmodn import (
    RNA_ClassQuery_Model,
    # RNA_ClassQuery_Model_Large,
    ParallelCNNBlock,
    GCNBlock,
    ClassQueryHead,
    ClassQueryHeadPooling,
    HierarchicalClassQueryHeadPooling
)

__all__ = [
    # Main models
    'RNA_ClassQuery_Model',
    'RNA_ClassQuery_Model_Large',
    # Sub-modules
    'ParallelCNNBlock',
    'GCNBlock',
    'ClassQueryHead',
    'ClassQueryHeadPooling',
    'HierarchicalClassQueryHeadPooling',
]

__version__ = '1.0.0'
