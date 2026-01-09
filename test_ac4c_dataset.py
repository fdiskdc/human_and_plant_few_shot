"""
简单的AC4C数据集测试脚本
"""

import sys
import os

# 添加dataset目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    
    # 导入AC4C数据集
    from dataset.ac4c import AC4CDataset
    
