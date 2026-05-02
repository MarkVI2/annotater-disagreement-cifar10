"""
Centralized configuration for the entire project.
All hyperparameters and paths defined here to avoid hardcoding.
"""

import os
from pathlib import Path

# ============================================================================
# RANDOM SEED & REPRODUCIBILITY
# ============================================================================
FIXED_SEED: int = 42
"""Fixed seed for all RNGs. Used in report."""

# ============================================================================
# DATA PATHS & CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR: str = str(PROJECT_ROOT / "data")
"""Root directory for all datasets."""

CIFAR10H_PROBS_FILE: str = "cifar10h-probs.npy"
"""File containing normalized soft labels (10000, 10)."""

# Split ratios for CIFAR-10H
TRAIN_RATIO: float = 0.6  # 6000 images
VAL_RATIO: float = 0.2    # 2000 images
TEST_RATIO: float = 0.2   # 2000 images

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9, \
    f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}"

# ============================================================================
# MODEL & TRAINING
# ============================================================================
BATCH_SIZE: int = 128
NUM_WORKERS: int = 8
"""Number of workers for DataLoader."""

# ============================================================================
# NORMALIZATION (CIFAR-10 standard)
# ============================================================================
CIFAR10_MEAN: tuple = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple = (0.2470, 0.2435, 0.2616)

# ============================================================================
# CLASS NAMES
# ============================================================================
CLASS_NAMES: list = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

assert len(CLASS_NAMES) == 10, "Must have exactly 10 class names"

# ============================================================================
# PATHS FOR OUTPUTS
# ============================================================================
CHECKPOINTS_DIR: str = str(PROJECT_ROOT / "outputs" / "checkpoints")
PLOTS_DIR: str = str(PROJECT_ROOT / "outputs" / "plots")
LOGS_DIR: str = str(PROJECT_ROOT / "outputs" / "logs")

# Create output directories if they don't exist
for dir_path in [CHECKPOINTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# VALIDATION (run at import time)
# ============================================================================
def validate_config():
    """Validate critical configuration."""
    assert os.path.exists(DATA_DIR), f"DATA_DIR does not exist: {DATA_DIR}"
    assert NUM_WORKERS >= 0, f"NUM_WORKERS must be >= 0, got {NUM_WORKERS}"
    assert BATCH_SIZE > 0, f"BATCH_SIZE must be > 0, got {BATCH_SIZE}"
    

validate_config()


if __name__ == "__main__":
    print("Configuration validation passed ✓")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Sum of ratios: {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")
