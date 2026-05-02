"""
Minimal verification that data pipeline works.
Runs in < 30 seconds.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Data Pipeline Verification\n" + "="*60)

try:
    # 1. Test imports
    print("1. Testing imports...", end=" ")
    from data import (set_seed, load_full_dataset, create_splits,
                     run_sanity_checks, compute_all_statistics)
    from training.config import FIXED_SEED
    print("✓")
    
    # 2. Set seed
    print("2. Setting seed...", end=" ")
    set_seed(FIXED_SEED)
    print("✓")
    
    # 3. Load dataset
    print("3. Loading CIFAR-10H...", end=" ")
    dataset = load_full_dataset()
    assert len(dataset) == 10000
    print(f"✓ ({len(dataset)} images)")
    
    # 4. Sanity checks
    print("4. Running sanity checks...", end=" ")
    run_sanity_checks(dataset)
    print("✓")
    
    # 5. Statistics
    print("5. Computing statistics...", end=" ")
    stats = compute_all_statistics(dataset)
    assert "entropies" in stats
    assert "class_entropies" in stats
    assert "confusion_matrix" in stats
    print("✓")
    
    # 6. Splits
    print("6. Creating splits...", end=" ")
    from training.config import TRAIN_RATIO, VAL_RATIO
    splits = create_splits(dataset)
    assert len(splits['train']) == 6000
    assert len(splits['val']) == 2000
    assert len(splits['test']) == 2000
    print(f"✓ (6k/2k/2k)")
    
    print("\n" + "="*60)
    print("✓ DATA PIPELINE VERIFIED - All components functional")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
