"""
Generate 4-class labels for MultIRM dataset
According to the rule:
- A (Adenine): m6A, m1A, m6Am, Am, AtoI
- C (Cytosine): m5C, Cm
- G (Guanine): m7G, Gm
- U (Uracil): m5U, Psi (Ψ), Um
"""

import numpy as np
import os

# Data directory
DATA_DIR = 'npy/multirm/51split'

# 12 classes in the new order (class 0 to class 11)
MULTIRM_CLASSES = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Psi', 'AtoI']

# Create class to index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(MULTIRM_CLASSES)}

# 4-class grouping rule
GROUP_MAPPING = {
    0: 0,  # Am -> A (0)
    1: 1,  # Cm -> C (1)
    2: 2,  # Gm -> G (2)
    3: 3,  # Um -> U (3)
    4: 0,  # m1A -> A (0)
    5: 1,  # m5C -> C (1)
    6: 3,  # m5U -> U (3)
    7: 0,  # m6A -> A (0)
    8: 0,  # m6Am -> A (0)
    9: 2,  # m7G -> G (2)
    10: 3, # Psi -> U (3)
    11: 0  # AtoI -> A (0)
}

GROUP_NAMES = ['A', 'C', 'G', 'U']


def process_mode(mode):
    """
    Generate 4-class labels for a specific mode (train/test/valid)
    
    Args:
        mode: 'train', 'test', or 'valid'
    """
    print(f"\nProcessing {mode} mode...")
    
    # Initialize arrays to store all labels
    # We need to process all 12 classes and collect their labels
    all_sequences = []
    all_labels_12 = []
    all_labels_4 = []
    
    for class_name in MULTIRM_CLASSES:
        class_idx = CLASS_TO_IDX[class_name]
        
        # Paths for positive samples
        pos_dir = os.path.join(DATA_DIR, class_name, 'pos')
        pos_in_path = os.path.join(pos_dir, f'{mode}_in.npy')
        pos_out_path = os.path.join(pos_dir, f'{mode}_out.npy')
        
        # Paths for negative samples
        neg_dir = os.path.join(DATA_DIR, class_name, 'neg')
        neg_in_path = os.path.join(neg_dir, f'{mode}_in.npy')
        neg_out_path = os.path.join(neg_dir, f'{mode}_out.npy')
        
        # Check if files exist
        if not os.path.exists(pos_in_path) or not os.path.exists(pos_out_path):
            print(f"  Warning: {class_name}/pos files not found, skipping")
            continue
        if not os.path.exists(neg_in_path) or not os.path.exists(neg_out_path):
            print(f"  Warning: {class_name}/neg files not found, skipping")
            continue
        
        # Load positive samples
        pos_seqs = np.load(pos_in_path, allow_pickle=True)
        pos_labels_12 = np.load(pos_out_path, allow_pickle=True)
        
        # Create 4-class labels for positive samples
        pos_labels_4 = np.zeros((len(pos_seqs), 4), dtype=np.float32)
        for i, label_12 in enumerate(pos_labels_12):
            # For positive samples of this class, label_12[class_idx] = 1
            # So we set the corresponding 4-class label to 1
            group_idx = GROUP_MAPPING[class_idx]
            pos_labels_4[i, group_idx] = 1.0
        
        # Load negative samples
        neg_seqs = np.load(neg_in_path, allow_pickle=True)
        neg_labels_12 = np.load(neg_out_path, allow_pickle=True)
        
        # Create 4-class labels for negative samples
        # For negative samples, all labels are 0 (no modification)
        neg_labels_4 = np.zeros((len(neg_seqs), 4), dtype=np.float32)
        
        # Store all data
        all_sequences.extend(pos_seqs)
        all_labels_12.extend(pos_labels_12)
        all_labels_4.extend(pos_labels_4)
        
        all_sequences.extend(neg_seqs)
        all_labels_12.extend(neg_labels_12)
        all_labels_4.extend(neg_labels_4)
        
        print(f"  {class_name}: pos={len(pos_seqs)}, neg={len(neg_seqs)}")
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels_12 = np.array(all_labels_12)
    all_labels_4 = np.array(all_labels_4)
    
    # Save to split directory
    output_dir = os.path.join(DATA_DIR, 'split')
    os.makedirs(output_dir, exist_ok=True)
    
    in_path = os.path.join(output_dir, f'{mode}_in.npy')
    out_path_12 = os.path.join(output_dir, f'{mode}_out_12class.npy')
    out_path_4 = os.path.join(output_dir, f'{mode}_out_4class.npy')
    
    np.save(in_path, all_sequences)
    np.save(out_path_12, all_labels_12)
    np.save(out_path_4, all_labels_4)
    
    print(f"  Saved {len(all_sequences)} samples to:")
    print(f"    {in_path}")
    print(f"    {out_path_12}")
    print(f"    {out_path_4}")


def main():
    print("=" * 60)
    print("Generating 4-class labels for MultIRM dataset")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"\n12 classes (new order):")
    for i, cls in enumerate(MULTIRM_CLASSES):
        group_idx = GROUP_MAPPING[i]
        print(f"  {i}: {cls:8s} -> Group {group_idx} ({GROUP_NAMES[group_idx]})")
    
    # Process all modes
    for mode in ['train', 'test', 'valid']:
        process_mode(mode)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()