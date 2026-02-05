"""
Enhanced Mer100Dataset with sequence string support for efficient data collection

This module extends the original human.py to include sequence string in Data objects,
avoiding the need for one-hot to sequence conversion during inference.

Based on: dataset/human.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import subprocess
import tempfile
import os
import uuid
import hashlib
import pickle

# Import the original dataset module
try:
    from . import human as human_module
except ImportError:
    import sys
    import os
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import dataset.human as human_module

# Import constants and functions from original module
LABEL_MAPPING = human_module.LABEL_MAPPING
INDEX_TO_NUCLEOTIDE = human_module.INDEX_TO_NUCLEOTIDE
NUCLEOTIDE_GROUPS = human_module.NUCLEOTIDE_GROUPS
MOD_NAMES = human_module.MOD_NAMES
ONE_HOT_MAPPING = human_module.ONE_HOT_MAPPING
DEFAULT_ONE_HOT = human_module.DEFAULT_ONE_HOT
TARGET_LENGTH = human_module.TARGET_LENGTH
LINEARFOLD_PATH = human_module.LINEARFOLD_PATH
BATCH_CACHE_FILE = human_module.BATCH_CACHE_FILE

_worker_process_batch = human_module._worker_process_batch
_BYTE_TO_ONEHOT_MAPPING = human_module._BYTE_TO_ONEHOT_MAPPING
run_linearfold = human_module.run_linearfold
build_edge_index_from_structure = human_module.build_edge_index_from_structure
build_sequential_edge_index = human_module.build_sequential_edge_index

# Import the base class
Mer100Dataset = human_module.Mer100Dataset


class Mer100DatasetWithSeq(Mer100Dataset):
    """
    Extended Mer100Dataset that includes sequence string in Data objects.
    
    This avoids expensive one-hot to string conversion during inference.
    """
    
    def __init__(self, mode='train', data_dir='../npy', cache_dir=None, 
                 use_human3=True, use_cache=True, preload_cache=True):
        """
        Initialize the enhanced dataset.
        
        Args:
            mode (str): 'train' or 'test'
            data_dir (str): Data file directory path
            cache_dir (str): Cache directory path
            use_human3 (bool): Whether to use human3 directory data
            use_cache (bool): Whether to enable secondary structure cache
            preload_cache (bool): Whether to load all edge indices to memory
        """
        # Initialize parent class to load all data and structures
        super().__init__(
            mode=mode,
            data_dir=data_dir,
            cache_dir=cache_dir,
            use_human3=use_human3,
            use_cache=use_cache,
            preload_cache=preload_cache
        )
    
    def __getitem__(self, idx):
        """
        Get single data sample with sequence string included.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Data: PyG Data object with additional 'seq_str' attribute
        """
        # Get sequence bytes and labels (using copy() for thread safety)
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()
        y_12class = self.y_12class[idx].copy()
        y_4class = self.y_4class[idx].copy()
        
        # Convert bytes to sequence string ONCE during data loading
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        
        # Use optimized one-hot encoding
        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)
        
        # Get or compute edge index
        edge_index = self._get_or_compute_edge_index(sequence_str, idx)
        
        # Node features
        node_features = torch.FloatTensor(one_hot_seq)
        
        # Generate attention masks
        attn_masks = self._extract_attention_masks(full_label)
        attn_mask_N = self._extract_attention_masks_N(sequence_bytes)
        
        # Create PyG Data object with sequence string
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),
            y_site=torch.LongTensor(full_label)
        )
        
        # Add sequence string as attribute (avoiding one-hot conversion later)
        data.seq_str = sequence_str
        
        # Add attention masks
        for nuc, mask in attn_masks.items():
            setattr(data, f'attn_mask_{nuc}', mask)
        setattr(data, 'attn_mask_N', attn_mask_N)
        
        return data


# Test code
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    print("=" * 60)
    print("Testing Enhanced Dataset with Sequence String")
    print("=" * 60)
    
    dataset = Mer100DatasetWithSeq(mode='train', use_human3=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting single sample
    print("\nTesting single sample:")
    sample = dataset[0]
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge index shape: {sample.edge_index.shape}")
    print(f"  Sequence string length: {len(sample.seq_str)}")
    print(f"  Sequence string (first 50 chars): {sample.seq_str[:50]}...")
    print(f"  y (12-class) shape: {sample.y.shape}")
    print(f"  y_4class (4-class) shape: {sample.y_4class.shape}")
    
    # Test with DataLoader
    print("\n" + "=" * 60)
    print("Testing with DataLoader")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    
    print(f"  Batch size: {batch.batch.max().item() + 1}")
    print(f"  Batch.x shape: {batch.x.shape}")
    print(f"  Has seq_str: {hasattr(batch, 'seq_str')}")
    if hasattr(batch, 'seq_str'):
        print(f"  seq_str type: {type(batch.seq_str)}")
        print(f"  seq_str length: {len(batch.seq_str)}")
        print(f"  First sequence (50 chars): {batch.seq_str[0][:50]}...")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)