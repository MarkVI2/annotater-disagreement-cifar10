"""
Main orchestrator script for data pipeline.
Verifies all stages: loading, sanity checks, statistics, splits, and dataloaders.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.config import (
    FIXED_SEED, DATA_DIR, CIFAR10H_PROBS_FILE,
    BATCH_SIZE, NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from data import (
    set_seed,
    get_transforms,
    load_full_dataset,
    create_splits,
    create_dataloaders,
    run_sanity_checks,
    compute_all_statistics,
    generate_all_data_visualizations,
)


def main():
    print("\n" + "="*80)
    print("PHASE 1: DATA PIPELINE")
    print("="*80)
    
    # Step 1: Set seed
    print("\n[1/6] Setting random seed...")
    set_seed(FIXED_SEED)
    print(f"✓ Seed set to {FIXED_SEED}")
    
    # Step 2: Load dataset
    print("\n[2/6] Loading CIFAR-10H dataset...")
    dataset = load_full_dataset(DATA_DIR, CIFAR10H_PROBS_FILE)
    
    # Step 3: Sanity checks
    print("\n[3/6] Running sanity checks...")
    run_sanity_checks(dataset)
    
    # Step 4: Compute statistics
    print("\n[4/6] Computing dataset statistics...")
    stats = compute_all_statistics(dataset)
    
    # Step 4b: Generate visualizations
    print("\n[4b/6] Generating data visualizations...")
    from training.config import PLOTS_DIR
    generate_all_data_visualizations(dataset, stats, PLOTS_DIR)
    
    # Step 5: Create splits
    print("\n[5/7] Creating train/val/test splits...")
    splits = create_splits(dataset, TRAIN_RATIO, VAL_RATIO, FIXED_SEED)
    
    # Step 6: Create dataloaders
    print("\n[6/7] Creating dataloaders...")
    train_tf = get_transforms(is_train=True)
    test_tf = get_transforms(is_train=False)
    transforms_dict = {
        "train": train_tf,
        "val": test_tf,
        "test": test_tf,
    }
    # Note: Using num_workers=0 for Windows compatibility with multiprocessing
    loaders = create_dataloaders(splits, BATCH_SIZE, 0, transforms_dict)
    
    # Verify loaders
    print("\n" + "-"*80)
    print("VERIFICATION: Sampling one batch from each loader")
    print("-"*80)
    
    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        images, soft_labels, hard_labels = batch
        
        print(f"\n{split_name.upper()} LOADER:")
        print(f"  Images shape:    {images.shape}")
        print(f"  Soft labels shape: {soft_labels.shape}")
        print(f"  Hard labels shape: {hard_labels.shape}")
        print(f"  Soft labels sum:   {soft_labels.sum(dim=1).mean():.6f} ± {soft_labels.sum(dim=1).std():.6f}")
        print(f"  Image range:       [{images.min():.3f}, {images.max():.3f}]")
        
        # Verify soft labels sum to 1
        sums_to_1 = (soft_labels.sum(dim=1) - 1.0).abs().max().item()
        if sums_to_1 < 1e-5:
            print(f"  ✓ All soft labels sum to 1.0")
        else:
            print(f"  ⚠ Soft labels don't perfectly sum to 1.0 (max error: {sums_to_1:.6f})")
    
    print("\n" + "="*80)
    print("✓ DATA PIPELINE COMPLETE - Ready for model training")
    print("="*80 + "\n")
    
    return dataset, splits, loaders, stats


if __name__ == "__main__":
    dataset, splits, loaders, stats = main()
