# Predicting Human Annotator Disagreement on CIFAR-10

## Overview

This project predicts the full distribution of human annotator labels for CIFAR-10 images. Rather than predicting a single hard class, the model outputs a 10-dimensional probability distribution representing disagreement among ~50 human annotators per image.

**Task**: Given an image, predict the soft label distribution (how annotators voted across 10 classes).

---

## Data Pipeline

The data pipelining phase provides a complete, production-ready data loading and processing infrastructure.

### Core Components

#### 1. **Configuration** (`training/config.py`)
Centralized configuration with runtime validation:
- Random seed: `FIXED_SEED = 42` (PyTorch, NumPy, CUDA deterministic)
- Train/Val/Test split: 60% / 20% / 20% (6000 / 2000 / 2000 images)
- Data augmentation: RandomHorizontalFlip + RandomCrop (train only)
- Normalization: CIFAR-10 mean/std constants
- Output directories: Auto-created checkpoints, plots, logs

#### 2. **Dataset Loading** (`data/dataset.py`)
`CIFAR10HWrapper` class handles:
- Loading CIFAR-10 images from torchvision
- Aligning with CIFAR-10H soft labels (10-dimensional probability vectors)
- Soft label validation (sum-to-1 check, normalization)
- Entropy computation per image (Shannon entropy in bits)

#### 3. **Data Pipeline** (`data/pipeline.py`)
Core functions:
- `set_seed()` - Reproducibility across all libraries
- `get_transforms()` - Per-split augmentation (train vs. val/test)
- `load_full_dataset()` - Load CIFAR-10H wrapper
- `create_splits()` - Stratified train/val/test splitting (preserves class distribution)
- `create_dataloaders()` - Create PyTorch DataLoaders with optimal settings
- `run_sanity_checks()` - 7-point validation (soft label sums, NaN, entropy bounds, etc.)

#### 4. **Statistics & Analysis** (`data/statistics.py`)
Compute dataset statistics:
- Per-image entropy (Shannon: H = -Σ p·log₂(p))
- Per-class average entropy
- Annotator confusion matrix (true class × annotator response)
- Majority agreement fraction
- High/low entropy examples identification

#### 5. **Visualizations** (`data/visualization.py`)
Generate 4 required plots (saved to `outputs/plots/`):
1. **Entropy histogram** - Distribution of disagreement across images
2. **Per-class entropy bar chart** - Which classes have highest disagreement
3. **Annotator confusion matrix** - Systematic annotation patterns
4. **Extreme examples grid** - Visual inspection of high/low entropy images

### Usage

**One-command data pipeline orchestration:**
```bash
python main_data.py
```

This runs all 7 steps:
1. Set reproducible seed
2. Load CIFAR-10H (10,000 images)
3. Run sanity checks
4. Compute statistics
5. Generate visualizations
6. Create stratified splits (6000/2000/2000)
7. Create DataLoaders

**Output:**
- Console: Dataset statistics and verification results
- Files: 4 PNG plots in `outputs/plots/`
- Python: DataLoaders ready for training

### Quick Verification

```bash
python verify_pipeline.py
```

Runs 30-second sanity checks without generating visualizations (useful for development).

### Data Module Interface

```python
from data import (
    CIFAR10HWrapper,           # Dataset class
    set_seed,                  # Reproducibility
    get_transforms,            # Augmentation
    load_full_dataset,         # Load CIFAR-10H
    create_splits,             # Stratified splits
    create_dataloaders,        # PyTorch DataLoaders
    run_sanity_checks,         # Validation
    compute_all_statistics,    # Statistics
    generate_all_data_visualizations  # Plots
)
```

---

## Project Structure

```
annotater-disagreement-cifar10-main/
├── data/                          # Data pipeline (COMPLETE)
│   ├── __init__.py
│   ├── dataset.py                 # CIFAR10HWrapper class
│   ├── pipeline.py                # Loading, splitting, dataloaders
│   ├── statistics.py              # Entropy, confusion matrix
│   ├── utils.py
│   └── visualization.py           # 4 required plots
├── training/                      # Training configuration
│   ├── config.py                  # Centralized settings
│   ├── train.py                   # Training loop
│   ├── metrics_logger.py
│   └── __init__.py
├── models/                        # Pre-built model
│   ├── cifar_resnet.py
│   ├── heads.py
│   ├── weights/
│   └── __init__.py
├── losses/                        # Loss functions (to implement)
│   ├── __init__.py
│   └── utils.py
├── evaluation/                    # Evaluation metrics (to implement)
│   ├── __init__.py
│   ├── metrics.py
│   └── robustness.py
├── explainability/                # Grad-CAM (to implement)
│   ├── grad_cam.py
│   └── __init__.py
├── utils/                         # Utilities
│   └── device.py
├── outputs/                       # Generated outputs
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── plots/                     # Visualization outputs
├── experiments/                   # Config-driven experiments
│   ├── run.py
│   └── configs/
├── notebooks/                     # Analysis notebooks
│   └── 00_data_exploration.ipynb
├── tests/                         # Unit tests
│   ├── test_data.py
│   ├── test_losses.py
│   └── test_models.py
├── requirements.txt
├── main_data.py                   # Data pipeline orchestrator
├── verify_pipeline.py             # Quick verification
└── README.md
```

---

## Requirements

```
torch>=2.0
torchvision
numpy
scikit-learn
matplotlib
```

Install:
```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

### 1. Soft Label Validation
- All soft labels normalized to sum to 1.0 (atol=1e-5)
- NaN/inf detection prevents silent failures

### 2. Deterministic Reproducibility
- Fixed seed (42) set at module import time
- CUDA determinism enabled
- Stratified splitting preserves class balance

### 3. Data Augmentation Strategy
- **Train**: RandomHorizontalFlip (50%) + RandomCrop(32, pad=4)
- **Val/Test**: No augmentation (identity transforms only)
- Prevents data leakage during evaluation

### 4. Per-Split Transforms
- Different transforms applied per split using `TransformWrapper`
- Ensures test set never sees augmented versions

### 5. Windows Compatibility
- `num_workers=0` in DataLoaders (Windows multiprocessing limitation)
- All path handling uses `pathlib` for cross-platform compatibility

---

## Dataset Statistics (CIFAR-10H)

From `main_data.py` output:
- Total images: 10,000
- Training: 6,000 (60%)
- Validation: 2,000 (20%)
- Test: 2,000 (20%)
- Soft labels: 10-dimensional probability vectors (~50 annotators per image)
- Majority agreement: 99.2%
- Mean entropy: 0.2228 bits
- Max entropy: 2.8602 bits (theoretical max: 3.3219 bits)

---

## Workflow for Next Phase

To integrate this pipeline with training:

```python
from training.config import FIXED_SEED, BATCH_SIZE, NUM_WORKERS
from data import (
    set_seed,
    load_full_dataset,
    create_splits,
    create_dataloaders,
    run_sanity_checks,
)

# Initialize
set_seed(FIXED_SEED)

# Load and validate
dataset = load_full_dataset()
run_sanity_checks(dataset)

# Create splits and loaders
splits = create_splits(dataset, train_ratio=0.6, val_ratio=0.2)
dataloaders = create_dataloaders(splits, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Now use dataloaders in training loop
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

---

## Sources & Citations

- **CIFAR-10H Dataset**: https://github.com/jcpeterson/cifar-10h
- **CIFAR-10 (Original)**: https://www.cs.toronto.edu/~kriz/cifar.html
- **ResNet-18 Pretrained Weights**:
  - https://github.com/huyvnphan/PyTorch_CIFAR10#how-to-use-pretrained-models
  - https://www.kaggle.com/datasets/pytorch/resnet18
