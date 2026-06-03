"""
collect_human.py - 收集人类数据集的模型输出 / Collect Model Outputs for Human Dataset

对人类数据集中所有样本运行模型推理,收集 12 类 logits、4 类 logits 和注意力权重,保存为 Excel 文件。
优化:批累积减少 GPU-CPU 传输 90%、序列字符串预存避免 one-hot→string 转换、AMP 混合精度加速。
Runs model inference on all human dataset samples, collecting 12-class logits, 4-class logits, and attention weights, saved as Excel.
Optimizations: batch accumulation reduces GPU-CPU transfer by 90%, pre-stored sequence string, AMP mixed precision.

功能模块 / Modules:
- load_model_from_checkpoint: 从 checkpoint 加载模型 / Load model from checkpoint
- AMP 混合精度推理 / AMP mixed precision inference
- main: 主入口 / Main entry point

输入 / Inputs:
- checkpoints/best_model.pt: PyTorch state_dict / Model weights
- json/human.json: 推理配置 / Inference config
- human3/seq.npy, human3/1001loc.npy, human3/12loc.npy: 人类数据 / Human data
- 命令行参数 / CLI: --checkpoint, --config, --output

输出 / Outputs:
- npy/human.xlsx: pandas DataFrame 包含 / pandas DataFrame containing:
  * seq: RNA 序列字符串 (1001nt) / RNA sequence string
  * l12: 12 类 logits / 12-class logits
  * l4: 4 类 logits / 4-class logits
  * atten: 注意力权重 / Attention weights

数据流 / Data Flow:
1. 加载模型与数据 / Load model and data
2. 批累积推理 / Batched inference with accumulation
3. AMP 加速 / AMP speedup
4. 收集 logits + 注意力 / Collect logits and attention
5. 保存为 xlsx / Save as xlsx

相关文件 / Related Files:
- 调用 / Calls: dataset.human_with_seq.Mer100DatasetWithSeq, model.main_model, torch.cuda.amp
- 被调用 / Called by: analysis pipelines, manual CLI

使用示例 / Usage Example:
    python collect_human.py --checkpoint checkpoints/best_model.pt --config json/human.json --output npy/human.xlsx

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.human_with_seq import Mer100DatasetWithSeq
from model.main_model import RNA_ClassQuery_Model
from utils import load_config


def load_model_from_checkpoint(checkpoint_path, config, device):
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        config: Configuration object
        device: torch device
        
    Returns:
        model: Loaded model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=config.cnn_hidden_dim,
        cnn_kernel_sizes=config.cnn_kernel_sizes,
        cnn_dropout=config.cnn_dropout,
        gcn_hidden_dim=config.gcn_hidden_dim,
        gcn_out_channels=config.gcn_out_channels,
        gcn_num_layers=config.gcn_num_layers,
        gcn_dropout=config.gcn_dropout,
        num_classes=config.num_classes,
        num_attn_heads=config.num_attn_heads,
        attn_dropout=config.attn_dropout,
        use_simple_pooling=config.use_simple_pooling,
        use_hierarchical=config.use_hierarchical,
        use_layer_norm=config.use_layer_norm
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"  Warning: Checkpoint not found, using randomly initialized model")
    
    model.eval()
    return model


def collect_predictions(model, dataloader, device, use_hierarchical=False, accumulate_batches=10):
    """
    Collect predictions for all samples in the dataset with batch accumulation.
    
    OPTIMIZATION: Accumulate multiple batches before GPU-CPU transfer.
    This reduces GPU-to-CPU transfer overhead by ~90%.
    
    Args:
        model: The model to use for predictions
        dataloader: DataLoader for the dataset
        device: torch device
        use_hierarchical: Whether model uses hierarchical head
        accumulate_batches: Number of batches to accumulate before CPU transfer (default: 10)
                          Higher values = fewer transfers but more memory usage
        
    Returns:
        list: List of tuples (seq, l12, l4, attention) for each sample
    """
    model.eval()
    
    results = []
    
    # Determine if AMP is available (CUDA only)
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    autocast = torch.cuda.amp.autocast if use_amp else torch.no_grad
    
    # Accumulators for batched processing
    accumulated_logits_12 = []
    accumulated_logits_4 = []
    accumulated_attention = []
    accumulated_y_12 = []
    accumulated_y_4 = []
    accumulated_seq_str = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing samples")):
            batch = batch.to(device)
            
            # Use AMP for faster inference if CUDA is available
            with autocast():
                # Get model predictions
                if use_hierarchical:
                    # Hierarchical mode: returns (logits_12, logits_4, attention)
                    logits_12, logits_4, attention = model(
                        batch.x, batch.edge_index, batch.batch, return_attention=True
                    )
                elif hasattr(model, 'class_query_head') and hasattr(model.class_query_head, 'use_simple_pooling'):
                    if model.class_query_head.use_simple_pooling:
                        # Simple pooling mode: returns (logits, attention)
                        logits, attention = model(
                            batch.x, batch.edge_index, batch.batch, return_attention=True
                        )
                        # 12-class logits only, no 4-class
                        logits_12 = logits
                        logits_4 = None
                    else:
                        # Standard mode: returns only logits
                        logits_12 = model(batch.x, batch.edge_index, batch.batch)
                        logits_4 = None
                        attention = None
                else:
                    # Default: standard mode
                    logits_12 = model(batch.x, batch.edge_index, batch.batch)
                    logits_4 = None
                    attention = None
            
            # Detach tensors and accumulate (keep on GPU for now)
            accumulated_logits_12.append(logits_12.detach())
            accumulated_logits_4.append(logits_4.detach() if logits_4 is not None else None)
            accumulated_attention.append(attention.detach() if attention is not None else None)
            
            # Accumulate ground truth labels from dataset
            accumulated_y_12.append(batch.y.detach() if hasattr(batch, 'y') else None)
            accumulated_y_4.append(batch.y_4class.detach() if hasattr(batch, 'y_4class') else None)
            
            # Get sequence strings directly from batch (no conversion needed!)
            # OPTIMIZATION: Sequence string is already in batch, no one-hot conversion
            if hasattr(batch, 'seq_str'):
                accumulated_seq_str.extend(list(batch.seq_str))
            else:
                # Fallback: generate placeholder strings
                batch_size = logits_12.size(0)
                accumulated_seq_str.extend(['N' * 1001] * batch_size)
            
            # Process accumulated batches when reaching threshold
            # OPTIMIZATION: Batch CPU transfer reduces transfers from N to N/accumulate_batches
            if len(accumulated_logits_12) >= accumulate_batches or batch_idx == len(dataloader) - 1:
                # Batch CPU transfer (optimized: single transfer per N batches)
                all_logits_12 = torch.cat(accumulated_logits_12, dim=0)
                logits_12_np = all_logits_12.cpu().numpy().astype(np.float32)
                
                logits_4_np = None
                if accumulated_logits_4[0] is not None:
                    all_logits_4 = torch.cat(accumulated_logits_4, dim=0)
                    logits_4_np = all_logits_4.cpu().numpy().astype(np.float32)
                
                # Batch CPU transfer for ground truth labels
                all_y_12 = torch.cat(accumulated_y_12, dim=0)
                accumulated_y_12_np = all_y_12.cpu().numpy().astype(np.float32)
                
                all_y_4 = torch.cat(accumulated_y_4, dim=0)
                accumulated_y_4_np = all_y_4.cpu().numpy().astype(np.float32)
                
                # Compute labels using torch.argmax before numpy conversion (keep on GPU)
                label12_indices = torch.argmax(all_y_12, dim=1).cpu().numpy()
                label4_indices = torch.argmax(all_y_4, dim=1).cpu().numpy()
                
                attention_np = None
                if accumulated_attention[0] is not None:
                    all_attention = torch.cat(accumulated_attention, dim=0)
                    attention_np = all_attention.cpu().numpy().astype(np.float32)
                
                # Process all samples in accumulated batches
                batch_size = len(logits_12_np)
                for i in range(batch_size):
                    # OPTIMIZATION: Direct string access, no conversion needed
                    seq = accumulated_seq_str[i]
                    
                    # Format logits as space-separated strings
                    l12_str = ' '.join([f'{v:.6f}' for v in logits_12_np[i]])
                    l4_str = ' '.join([f'{v:.6f}' for v in logits_4_np[i]]) if logits_4_np is not None else ''
                    
                    # Get ground truth labels from dataset
                    # label12: 12-class ground truth label (0-11)
                    label12 = int(label12_indices[i])
                    # label4: 4-class ground truth label (0-3)
                    label4 = int(label4_indices[i])
                    
                    # Format attention weights as space-separated strings
                    if attention_np is not None:
                        # attention shape: [total_samples, num_classes, seq_len] or [total_samples, seq_len]
                        if attention_np.ndim == 3:
                            atten_str = ' '.join([f'{v:.6f}' for v in attention_np[i].flatten()])
                        else:
                            atten_str = ' '.join([f'{v:.6f}' for v in attention_np[i]])
                    else:
                        atten_str = ''
                    
                    results.append((seq, l12_str, l4_str, atten_str, label12, label4))
                
                # Clear accumulators
                accumulated_logits_12.clear()
                accumulated_logits_4.clear()
                accumulated_attention.clear()
                accumulated_y_12.clear()
                accumulated_y_4.clear()
                accumulated_seq_str.clear()
    
    return results


def main(config_path='json/human.json', checkpoint_path=None, output_path='human.xlsx', batch_size=32, accumulate_batches=10):
    """
    Main function to collect model outputs.
    
    Args:
        config_path (str): Path to configuration file
        checkpoint_path (str): Path to model checkpoint (optional)
        output_path (str): Path to output Excel file
        batch_size (int): Batch size for inference
        accumulate_batches (int): Number of batches to accumulate before CPU transfer (default: 10)
    """
    # Load configuration
    Config, config_dict = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("Human Dataset Model Output Collector (Optimized)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Batch accumulation: {accumulate_batches} batches")
    print(f"Optimizations enabled:")
    print(f"  - Batch accumulation (reduces GPU-CPU transfers by ~{100 * (1 - 1/accumulate_batches):.0f}%)")
    print(f"  - Sequence string in dataset (avoids one-hot conversion)")
    print(f"  - AMP support (mixed precision inference)")
    
    # Load dataset (with sequence string support)
    print(f"\nLoading dataset (with sequence string support)...")
    dataset = Mer100DatasetWithSeq(
        mode='train',
        data_dir=Config.data.human_data_dir,
        cache_dir=Config.data.cache_dir,
        use_human3=True,
        use_cache=True
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Check cache status
    cache_stats = dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
        print("\nWarning: Batch cache does not exist.")
        print("Precomputing structures (this may take a while)...")
        dataset.precompute_all_structures(batch_size=100, num_workers=None, show_progress=True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    if checkpoint_path is None:
        # Try to find best_model.pt in checkpoint directory
        checkpoint_dir = os.path.join(Config.log_dir, 'checkpoints')
        potential_paths = [
            os.path.join(checkpoint_dir, 'best_model.pt'),
            os.path.join(checkpoint_dir, 'checkpoint.pt'),
            'best_model.pt'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"\nWarning: No checkpoint found, using randomly initialized model")
        checkpoint_path = None
    
    model = load_model_from_checkpoint(checkpoint_path, Config, device)
    
    # Collect predictions
    print(f"\nCollecting predictions...")
    results = collect_predictions(
        model, dataloader, device, use_hierarchical=Config.use_hierarchical, 
        accumulate_batches=accumulate_batches
    )
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    df = pd.DataFrame(results, columns=['seq', 'l12', 'l4', 'atten', 'label12', 'label4'])
    
    # Save to Excel
    print(f"Saving to {output_path}...")
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"\n{'='*60}")
    print("Collection Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Output file: {output_path}")
    print(f"\nColumns:")
    print(f"  - seq: RNA sequence (length 1001)")
    print(f"  - l12: 12-class logits (space-separated)")
    print(f"  - l4: 4-class logits (space-separated, may be empty)")
    print(f"  - atten: attention weights (space-separated, may be empty)")
    print(f"  - label12: 12-class ground truth label (0-11)")
    print(f"  - label4: 4-class ground truth label (0-3)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect model outputs for human dataset (optimized version)')
    parser.add_argument('--config', type=str, default='json/human.json',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='logs/rna_classification_20260129_195404/checkpoints/epoch_080.pt',
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='human.xlsx',
                        help='Path to output Excel file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--accumulate_batches', type=int, default=30,
                        help='Number of batches to accumulate before CPU transfer (default: 10)')
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        accumulate_batches=args.accumulate_batches
    )