#!/usr/bin/env python3
"""
generate_cache.py - Generate the structures cache for the dataset
"""

from dataset.human import Mer100Dataset

def main():
    print("=" * 60)
    print("Generating structures cache for test dataset...")
    print("=" * 60)

    # Initialize dataset with mode='test' and preload_cache=False
    # preload_cache=False is important so we don't try to load the cache that doesn't exist yet
    dataset = Mer100Dataset(mode='test', use_human3=True, use_cache=True, preload_cache=False)

    print(f"\nDataset initialized: {len(dataset)} samples")
    print(f"Cache directory: {dataset.CACHE_DIR}")
    print(f"Expected cache file: {dataset._get_batch_cache_path()}")

    # Precompute all structures
    print("\nStarting precomputation...")
    dataset.precompute_all_structures(
        batch_size=100,
        num_workers=None,  # Use all CPU cores
        show_progress=True
    )

    print("\n" + "=" * 60)
    print("Cache generation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
