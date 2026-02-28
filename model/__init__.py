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

Sub-modules (from main_model.py):
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

from .main_model import (
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
