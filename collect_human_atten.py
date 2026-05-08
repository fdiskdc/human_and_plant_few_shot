"""
Collect Attention Outputs for Human Dataset

This script collects attention outputs (attn_out_12, attn_out_4) for all samples
in human dataset and saves them to an NPZ file.

Output format (human_atten.npz):
    - attn_out_12: Attention outputs for 12-class task [num_samples, 12, hidden_dim]
    - attn_out_4: Attention outputs for 4-class task [num_samples, 4, hidden_dim]
    - label12: 12-class ground truth labels (multilabel) [num_samples, 12]
    - label4: 4-class ground truth labels (multilabel) [num_samples, 4]
    - seqs: RNA sequences as strings [num_samples,]
    - probs12: 12-class softmax probabilities [num_samples, 12]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.human_with_seq import Mer100DatasetWithSeq
from model.main_model_collect_atten import RNA_ClassQuery_Model_Collect_Atten
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
    print(f"Loading model from checkpoint: {checkpoint_path or '(none, using random init)'}")
    
    # Initialize model
    model = RNA_ClassQuery_Model_Collect_Atten(
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
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"  Warning: Checkpoint not found, using randomly initialized model")
    
    model.eval()
    return model


def collect_attention_outputs(model, dataloader, device, accumulate_batches=10):
    """
    Collect attention outputs for all samples in the dataset with batch accumulation.

    Args:
        model: The model to use for predictions
        dataloader: DataLoader for the dataset
        device: torch device
        accumulate_batches: Number of batches to accumulate before CPU transfer (default: 10)

    Returns:
        dict: Dictionary with keys:
            - attn_out_12: [num_samples, 12, hidden_dim]
            - attn_out_4: [num_samples, 4, hidden_dim]
            - label12: [num_samples, 12] (multilabel)
            - label4: [num_samples, 4] (multilabel)
            - seqs: [num_samples] strings
            - probs12: [num_samples, 12] softmax probabilities
    """
    model.eval()

    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    autocast = torch.cuda.amp.autocast if use_amp else torch.no_grad

    accumulated_attn_out_12 = []
    accumulated_attn_out_4 = []
    accumulated_y_12 = []
    accumulated_y_4 = []
    accumulated_seqs = []
    accumulated_probs12 = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing samples")):
            batch = batch.to(device)

            with autocast():
                logits_12, logits_4, attn_out_12, attn_out_4, _ = model(
                    batch.x, batch.edge_index, batch.batch, return_attention=True
                )
                probs12 = F.softmax(logits_12, dim=-1)

            accumulated_attn_out_12.append(attn_out_12.detach())
            accumulated_attn_out_4.append(attn_out_4.detach())
            accumulated_y_12.append(batch.y.detach() if hasattr(batch, 'y') else None)
            accumulated_y_4.append(batch.y_4class.detach() if hasattr(batch, 'y_4class') else None)
            accumulated_probs12.append(probs12.detach())

            if hasattr(batch, 'seq_str'):
                if isinstance(batch.seq_str, list):
                    accumulated_seqs.extend(batch.seq_str)
                else:
                    accumulated_seqs.extend(batch.seq_str.tolist())
            else:
                accumulated_seqs.extend([''] * batch.num_graphs)

            if len(accumulated_attn_out_12) >= accumulate_batches or batch_idx == len(dataloader) - 1:
                all_attn_out_12 = torch.cat(accumulated_attn_out_12, dim=0)
                all_attn_out_4 = torch.cat(accumulated_attn_out_4, dim=0)
                all_probs12 = torch.cat(accumulated_probs12, dim=0)
                all_y_12 = torch.cat(accumulated_y_12, dim=0)
                all_y_4 = torch.cat(accumulated_y_4, dim=0)

                yield {
                    'attn_out_12': all_attn_out_12.cpu().numpy().astype(np.float32),
                    'attn_out_4': all_attn_out_4.cpu().numpy().astype(np.float32),
                    'label12': all_y_12.cpu().numpy().astype(np.float32),
                    'label4': all_y_4.cpu().numpy().astype(np.float32),
                    'seqs': list(accumulated_seqs),
                    'probs12': all_probs12.cpu().numpy().astype(np.float32),
                }

                accumulated_attn_out_12.clear()
                accumulated_attn_out_4.clear()
                accumulated_y_12.clear()
                accumulated_y_4.clear()
                accumulated_probs12.clear()
                accumulated_seqs.clear()


def main(config_path='json/human.json', checkpoint_path=None, output_path='npy/human_atten.npy', 
         batch_size=32, accumulate_batches=10):
    """
    Main function to collect attention outputs.
    
    Args:
        config_path (str): Path to configuration file
        checkpoint_path (str): Path to model checkpoint (optional)
        output_path (str): Path to output NPY file
        batch_size (int): Batch size for inference
        accumulate_batches (int): Number of batches to accumulate before CPU transfer (default: 10)
    """
    # Load configuration
    Config, config_dict = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("Human Dataset Attention Output Collector")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Batch accumulation: {accumulate_batches} batches")
    print(f"Optimizations enabled:")
    print(f"  - Batch accumulation (reduces GPU-CPU transfers by ~{100 * (1 - 1/accumulate_batches):.0f}%)")
    print(f"  - AMP support (mixed precision inference)")
    
    # Load dataset
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
    
    # Collect attention outputs in batches
    print(f"\nCollecting attention outputs...")

    all_attn_out_12 = []
    all_attn_out_4 = []
    all_label12 = []
    all_label4 = []
    all_seqs = []
    all_probs12 = []

    for batch_output in collect_attention_outputs(
        model, dataloader, device, accumulate_batches=accumulate_batches
    ):
        all_attn_out_12.append(batch_output['attn_out_12'])
        all_attn_out_4.append(batch_output['attn_out_4'])
        all_label12.append(batch_output['label12'])
        all_label4.append(batch_output['label4'])
        all_probs12.append(batch_output['probs12'])
        all_seqs.extend(batch_output['seqs'])

    print(f"\nConcatenating batches...")
    final_attn_out_12 = np.concatenate(all_attn_out_12, axis=0)
    final_attn_out_4 = np.concatenate(all_attn_out_4, axis=0)
    final_label12 = np.concatenate(all_label12, axis=0)
    final_label4 = np.concatenate(all_label4, axis=0)
    final_probs12 = np.concatenate(all_probs12, axis=0)
    final_seqs = np.array(all_seqs, dtype=object)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    print(f"Saving to {output_path}...")
    np.savez(
        output_path,
        attn_out_12=final_attn_out_12,
        attn_out_4=final_attn_out_4,
        label12=final_label12,
        label4=final_label4,
        seqs=final_seqs,
        probs12=final_probs12
    )

    print(f"\n{'='*60}")
    print("Collection Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_label12)}")
    print(f"Output file: {output_path}")
    print(f"\nData shapes:")
    print(f"  - attn_out_12: {final_attn_out_12.shape}")
    print(f"  - attn_out_4: {final_attn_out_4.shape}")
    print(f"  - label12: {final_label12.shape}")
    print(f"  - label4: {final_label4.shape}")
    print(f"  - seqs: {final_seqs.shape} (object array)")
    print(f"  - probs12: {final_probs12.shape}")
    print(f"\nTo load the data:")
    print(f"  data = np.load('{output_path}', allow_pickle=True)")
    print(f"  attn_out_12 = data['attn_out_12']")
    print(f"  attn_out_4 = data['attn_out_4']")
    print(f"  label12 = data['label12']")
    print(f"  label4 = data['label4']")
    print(f"  seqs = data['seqs']")
    print(f"  probs12 = data['probs12']")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect attention outputs for human dataset')
    parser.add_argument('--config', type=str, default='json/human.json',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='/home/dc/vscode/vscode20260424/rgcnformer_sum/logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt',
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='npy/human_atten.npz',
                        help='Path to output NPY file')
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