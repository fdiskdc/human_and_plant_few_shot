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
    RNA_ClassQuery_Model_Large,
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
