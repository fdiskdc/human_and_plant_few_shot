"""
Test script for MultIRM 4-class modifications
"""

import torch
from torch_geometric.loader import DataLoader
from dataset.multirm import MultirmDataset, MULTIRM_CLASSES, MULTIRM_4CLASS_NAMES, MULTIRM_12TO4_MAPPING
from model.main_model_multirm import RNA_ClassQuery_Model

def main():
    print("=" * 60)
    print("Test 1: Check class mappings")
    print("=" * 60)
    
    print(f"\n12 classes (new order):")
    for i, cls in enumerate(MULTIRM_CLASSES):
        group_idx = MULTIRM_12TO4_MAPPING[i]
        print(f"  {i}: {cls:8s} -> Group {group_idx} ({MULTIRM_4CLASS_NAMES[group_idx]})")
    
    print("=" * 60)
    print("Test 2: Load dataset with 4-class mode")
    print("=" * 60)
    
    dataset = MultirmDataset(mode='train', use_4class=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  x shape: {sample.x.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  y (12-class) shape: {sample.y.shape}")
    if hasattr(sample, 'y_4'):
        print(f"  y_4 (4-class) shape: {sample.y_4.shape}")
        print(f"  12-class label: {sample.y[0].numpy()}")
        print(f"  4-class label: {sample.y_4[0].numpy()}")
    else:
        print("  Warning: y_4 not found")
    
    print("=" * 60)
    print("Test 3: Test DataLoader")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"\nBatch structure:")
    print(f"  batch.x shape: {batch.x.shape}")
    print(f"  batch.edge_index shape: {batch.edge_index.shape}")
    print(f"  batch.y shape: {batch.y.shape}")
    if hasattr(batch, 'y_4'):
        print(f"  batch.y_4 shape: {batch.y_4.shape}")
    
    print("=" * 60)
    print("Test 4: Test model with 4-class grouping")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=64,
        gcn_hidden_dim=128,
        gcn_out_channels=128,
        num_classes=12,
        use_hierarchical=True
    )
    
    model = model.to(device)
    batch = batch.to(device)
    
    print("\nModel structure:")
    print(model)
    
    print("\nForward pass (without attention):")
    with torch.no_grad():
        logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch, return_attention=False)
    
    print(f"  logits_12 shape: {logits_12.shape}")
    print(f"  logits_4 shape: {logits_4.shape}")
    
    print("\nForward pass (with attention):")
    with torch.no_grad():
        logits_12, logits_4, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
    
    print(f"  logits_12 shape: {logits_12.shape}")
    print(f"  logits_4 shape: {logits_4.shape}")
    print(f"  attn_weights shape: {attn_weights.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()