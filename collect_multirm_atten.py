"""
Collect Attention Outputs for Human Dataset using MultiRM Model

This script collects attention outputs (context_vector, attention_weights) for all samples
in human dataset using the multirm model and saves them to an NPY file.

Output format (multirm_atten.npy):
    - context_vector: Context vectors from attention mechanism [num_samples, num_task, hidden_dim]
    - attention_weights: Attention weights for all classes [num_samples, num_task, seq_len]
    - label12: 12-class ground truth labels (multilabel) [num_samples, 12]
    - label4: 4-class ground truth labels (multilabel) [num_samples, 4]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.human_with_seq import Mer100DatasetWithSeq
from model.multirm_collect_atten import model_v3_Collect_Atten
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
    model = model_v3_Collect_Atten(
        num_task=12,
        use_embedding=False,
        cnn_hidden_dim=config.cnn_hidden_dim,
        cnn_kernel_sizes=config.cnn_kernel_sizes,
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
            - context_vector: [num_samples, num_task, hidden_dim]
            - attention_weights: [num_samples, num_task, seq_len]
            - label12: [num_samples, 12] (multilabel)
            - label4: [num_samples, 4] (multilabel)
    """
    model.eval()

    # Determine if AMP is available (CUDA only)
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    autocast = torch.cuda.amp.autocast if use_amp else torch.no_grad

    # Accumulators for batched processing
    accumulated_context_vector = []
    accumulated_attention_weights = []
    accumulated_y_12 = []
    accumulated_y_4 = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing samples")):
            batch = batch.to(device)

            # Use AMP for faster inference if CUDA is available
            with autocast():
                # Get model outputs: returns (logits, context_vector, attention_weights)
                # or (logits, logits_4class, context_vector, attention_weights) if hierarchical
                outputs = model(
                    batch.x, batch.edge_index, batch.batch, return_attention=True
                )

                # Handle both hierarchical (4 outputs) and non-hierarchical (3 outputs) models
                if len(outputs) == 4:
                    # Hierarchical model: (logits, logits_4class, context_vector, attention_weights)
                    _, _, context_vector, attention_weights = outputs
                else:
                    # Non-hierarchical model: (logits, context_vector, attention_weights)
                    _, context_vector, attention_weights = outputs

            # Detach tensors and accumulate
            accumulated_context_vector.append(context_vector.detach())
            accumulated_attention_weights.append(attention_weights.detach())

            # Accumulate ground truth labels from dataset
            accumulated_y_12.append(batch.y.detach() if hasattr(batch, 'y') else None)
            accumulated_y_4.append(batch.y_4class.detach() if hasattr(batch, 'y_4class') else None)

            # Process accumulated batches when reaching threshold
            if len(accumulated_context_vector) >= accumulate_batches or batch_idx == len(dataloader) - 1:
                # Batch CPU transfer
                all_context_vector = torch.cat(accumulated_context_vector, dim=0)
                all_context_vector_np = all_context_vector.cpu().numpy().astype(np.float32)

                all_attention_weights = torch.cat(accumulated_attention_weights, dim=0)
                all_attention_weights_np = all_attention_weights.cpu().numpy().astype(np.float32)

                # Batch CPU transfer for ground truth labels (multilabel format)
                all_y_12 = torch.cat(accumulated_y_12, dim=0)
                label12_indices = all_y_12.cpu().numpy().astype(np.float32)

                all_y_4 = torch.cat(accumulated_y_4, dim=0)
                label4_indices = all_y_4.cpu().numpy().astype(np.float32)

                yield {
                    'context_vector': all_context_vector_np,
                    'attention_weights': all_attention_weights_np,
                    'label12': label12_indices,
                    'label4': label4_indices
                }

                # Clear accumulators
                accumulated_context_vector.clear()
                accumulated_attention_weights.clear()
                accumulated_y_12.clear()
                accumulated_y_4.clear()


def main(config_path='json/human.json', checkpoint_path=None, output_path='npy/multirm_atten.npz',
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
    print("MultiRM Model Attention Output Collector")
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

    all_context_vector = []
    all_attention_weights = []
    all_label12 = []
    all_label4 = []

    for batch_output in collect_attention_outputs(
        model, dataloader, device, accumulate_batches=accumulate_batches
    ):
        all_context_vector.append(batch_output['context_vector'])
        all_attention_weights.append(batch_output['attention_weights'])
        all_label12.append(batch_output['label12'])
        all_label4.append(batch_output['label4'])

    # Concatenate all batches
    print(f"\nConcatenating batches...")
    final_context_vector = np.concatenate(all_context_vector, axis=0)
    final_attention_weights = np.concatenate(all_attention_weights, axis=0)
    final_label12 = np.concatenate(all_label12, axis=0)
    final_label4 = np.concatenate(all_label4, axis=0)

    # Save to NPY
    print(f"Saving to {output_path}...")
    np.savez(
        output_path,
        context_vector=final_context_vector,
        attention_weights=final_attention_weights,
        label12=final_label12,
        label4=final_label4
    )

    print(f"\n{'='*60}")
    print("Collection Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_label12)}")
    print(f"Output file: {output_path}")
    print(f"\nData shapes:")
    print(f"  - context_vector: {final_context_vector.shape}")
    print(f"  - attention_weights: {final_attention_weights.shape}")
    print(f"  - label12: {final_label12.shape}")
    print(f"  - label4: {final_label4.shape}")
    print(f"\nTo load the data:")
    print(f"  data = np.load('{output_path}')")
    print(f"  context_vector = data['context_vector']")
    print(f"  attention_weights = data['attention_weights']")
    print(f"  label12 = data['label12']")
    print(f"  label4 = data['label4']")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect attention outputs for multirm model')
    parser.add_argument('--config', type=str, default='json/human.json',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='logs/multirm_rna_classification_20260203_230348/checkpoints/best_model.pt',
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--output', type=str, default='npy/multirm_atten.npz',
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
