# Predicting Human Annotator Disagreement on CIFAR-10

**3rd Year Deep Neural Networks Course Project** Atharv A. Garg, Vedh Chengappa,
Chirag Goyal, Mayak M. Thakre, Dhruva Mekala SE23UCSE{223, 184, 050, 218, 219}

Complete code and results:
[GitHub](https://github.com/MarkVI2/annotater-disagreement-cifar10)

Project report can be found at
[docs/G_223_184_050_218_219.pdf](docs/G_223_184_050_218_219.pdf)

Overview of the codebase (file based documentation) can be found at
[docs/overview.md](docs/overview.md)

## Overview

Standard deep learning treats annotator disagreement as noise to be discarded
via majority voting. This project inverts that assumption: **disagreement is the
signal.**

Given a 32×32 CIFAR-10 image, the model outputs a 10-dimensional probability
distribution representing how a population of approximately 50 human annotators
would distribute their votes across the 10 CIFAR-10 classes — rather than
predicting a single hard label.

**Example:** where a standard classifier outputs `cat with 95% confidence`, our
model outputs `[cat: 0.60, dog: 0.25, deer: 0.10, horse: 0.05, ...]`, capturing
the full structure of human disagreement.

**Best result:** KL divergence of **0.4962**, cosine similarity of **0.8553**
(KL loss, linear head, ImageNet pretraining).

---

## Problem Formulation

| Symbol          | Meaning                                            |
| --------------- | -------------------------------------------------- |
| x ∈ ℝ^(3×32×32) | A CIFAR-10 image                                   |
| p(y\|x) ∈ Δ⁹    | Ground-truth annotator distribution from CIFAR-10H |
| qθ(y\|x) ∈ Δ⁹   | Model's predicted distribution                     |
| Nₐ ≈ 50         | Number of annotators per image                     |

**Objective:** minimise Eₓ~D[L(p(y|x), qθ(y|x))] where L is a
distribution-matching loss.

**Key metrics:** KL divergence, Jensen-Shannon divergence, cosine similarity,
Pearson/Spearman correlation on predicted entropy, and Precision@K (K = 100,
200, 500).

---

## Dataset: CIFAR-10H

CIFAR-10H contains 511,400 crowdsourced annotations over the 10,000-image
CIFAR-10 test set, collected via Amazon Mechanical Turk. Each image received
approximately 51 independent labels (range 47–63). Annotators scoring below 75%
on attention-check trials were excluded (14 of 2,571 total).

### Data Splits

All splits use stratified sampling (preserving per-class ratio) with fixed
seed 42.

| Split      | Images | Percentage | Purpose                                  |
| ---------- | ------ | ---------- | ---------------------------------------- |
| Training   | 6,000  | 60%        | Soft-label fine-tuning                   |
| Validation | 2,000  | 20%        | Early stopping, hyperparameter selection |
| Test       | 2,000  | 20%        | Final held-out evaluation only           |

### Dataset Statistics

| Statistic                               | Value                  |
| --------------------------------------- | ---------------------- |
| Total images                            | 10,000                 |
| Soft label dimensionality               | 10                     |
| Majority vote agreement with hard label | 99.2%                  |
| Mean Shannon entropy                    | 0.2228 bits            |
| Observed maximum entropy                | 2.8602 bits            |
| Theoretical maximum entropy             | log₂(10) ≈ 3.3219 bits |

The entropy distribution is heavily right-skewed — approximately 70% of images
have entropy below 0.1 bits (near-unanimous annotator agreement), with
disagreement concentrated in a long tail of genuinely ambiguous images.

### Sanity Checks

Seven checks were applied and passed:

1. All soft-label vectors sum to 1.0 (tolerance 10⁻⁵)
2. No NaN or Inf values in soft-label tensors
3. Entropy values in [0, log₂(10) + 10⁻⁵]
4. Hard labels in {0, ..., 9}
5. Image pixel values in a reasonable normalised range
6. No index overlap between splits
7. Stratification — per-class counts approximately equal across splits

---

## Data Pipeline

### Core Components

**Configuration** (`training/config.py`) — Centralised settings with runtime
validation: fixed seed (42), split ratios, augmentation policy, normalisation
constants, and auto-created output directories.

**Dataset Loading** (`data/dataset.py`) — `CIFAR10HWrapper` loads CIFAR-10
images via torchvision (`train=False`), aligns with CIFAR-10H soft labels,
validates label sums, and computes per-image Shannon entropy.

**Pipeline** (`data/pipeline.py`) — Reproducibility seeding, per-split
transforms, stratified splitting, DataLoader creation, and 7-point sanity
checks.

**Statistics** (`data/statistics.py`) — Per-image and per-class entropy,
annotator confusion matrix, majority agreement fraction, and high/low entropy
example identification.

**Visualisations** (`data/visualization.py`) — Entropy histogram, per-class
entropy bar chart, annotator confusion matrix, and extreme-example image grid
(saved to `outputs/plots/`).

### Data Augmentation Policy

Augmentation applied to training split only. Augmentations that alter class
semantics (e.g. 90° rotation, severe colour jitter) were explicitly excluded.

| Transform                      | Train | Val/Test |
| ------------------------------ | ----- | -------- |
| ToTensor (PIL → float tensor)  | ✓     | ✓        |
| Normalize (μ, σ)               | ✓     | ✓        |
| RandomHorizontalFlip (p=0.5)   | ✓     | —        |
| RandomCrop(32, pad=4, reflect) | ✓     | —        |

Normalisation constants (CIFAR-10 channel statistics):

- μ = (0.4914, 0.4822, 0.4465)
- σ = (0.2023, 0.1994, 0.2010)

### Usage

```bash
# Full data pipeline (7 steps: seed → load → sanity checks → stats → plots → splits → dataloaders)
python main_data.py

# Quick sanity check only (no plots, ~30 seconds)
python tests/verify_pipeline.py

# Run training (configure loss/head/init in experiments/configs/)
python main.py

# Run a specific experiment configuration
python experiments/run.py
```

---

## Model Architecture

The model has two components: a **backbone** for feature extraction and a
**prediction head** mapping features to a 10-dimensional probability
distribution via softmax.

### Backbone: CIFAR-adapted ResNet-18

Standard ImageNet ResNet-18 uses a 7×7 convolution with stride 2 and a 3×3
max-pool, reducing a 32×32 input to 4×4 after the stem alone. We replaced the
stem with a 3×3 convolution (stride 1) and removed the max-pool layer.

| Layer           | Output Size | Notes                                      |
| --------------- | ----------- | ------------------------------------------ |
| Input           | 3×32×32     | RGB image                                  |
| Stem (adapted)  | 64×32×32    | Conv 3×3, stride 1, BN, ReLU — no max-pool |
| Layer 1         | 64×32×32    | 2× BasicBlock                              |
| Layer 2         | 128×16×16   | 2× BasicBlock, stride 2                    |
| Layer 3         | 256×8×8     | 2× BasicBlock, stride 2                    |
| Layer 4         | 512×4×4     | 2× BasicBlock, stride 2                    |
| Global Avg Pool | 512         | Spatial → vector                           |
| Prediction Head | 10          | Linear/MLP + Softmax                       |

### Prediction Heads

Two head architectures were compared:

**Linear Head** — `qθ(y|x) = softmax(Wz + b)`, W ∈ ℝ^(10×512). Parameters:
5,130. Low capacity, low overfitting risk on 6,000 soft-label examples.

**MLP Head** — `h₁ = ReLU(W₁z + b₁)`, `qθ(y|x) = softmax(W₂h₁ + b₂)`, W₁ ∈
ℝ^(256×512). Parameters: 133,902. More expressive but higher overfitting risk
with limited data.

| Component                          | Parameters     |
| ---------------------------------- | -------------- |
| ResNet-18 backbone (CIFAR adapted) | 11,173,952     |
| Linear head (512→10)               | 5,130          |
| MLP head (512→256→10)              | 133,902        |
| **Total (linear head)**            | **11,179,082** |
| **Total (MLP head)**               | **11,307,854** |

---

## Loss Functions

Four loss functions were implemented and compared.

| Loss           | Symmetric | Bounded | Entropy-aware | Semantic |
| -------------- | --------- | ------- | ------------- | -------- |
| KL Divergence  | No        | No      | No            | No       |
| Jensen-Shannon | Yes       | Yes     | No            | No       |
| Custom (ours)  | No        | No      | Yes           | No       |
| EMD (bonus)    | Yes       | Yes     | No            | Yes      |

**KL Divergence (mandatory baseline):** Measures extra bits per symbol when
using the model distribution q to encode data drawn from p. Predictions clamped
to 10⁻⁹ to prevent log(0).

**Jensen-Shannon Divergence:** Symmetric, bounded in [0,1], never infinite even
under support mismatch. Provides a smoother gradient landscape near p = q.

**Custom Composite Loss (ours):**

`L_custom = (1/N) Σ wᵢ · KL(pᵢ ‖ qθ,ᵢ) + λ · (1/N) Σ (H(qθ,ᵢ) − H(pᵢ))²`

where `wᵢ = 1 + α · H(pᵢ)`, α = 2.0, λ = 0.5.

The focal weight up-weights high-disagreement images (which dominate less than
30% of the dataset but are most informative). The entropy error penalty
explicitly penalises cases where the model's predicted uncertainty level differs
from the true level — something KL and JSD do not capture.

**Earth Mover's Distance (bonus):** Accounts for semantic class structure via a
class-distance matrix, but optimises a fundamentally different objective from
pointwise probability matching.

---

## Training Protocol

### Two-Stage Pipeline

**Stage 1 — Pre-training:** ResNet-18 trained on all 50,000 CIFAR-10 images with
hard-label cross-entropy (Adam, lr=0.1, weight decay=5×10⁻⁴, 100 epochs, batch
size 128). This gives the backbone rich class-discriminative features that 6,000
soft-label images alone cannot provide.

**Stage 2 — Fine-tuning:** Backbone and head fine-tuned on CIFAR-10H (Adam,
lr=10⁻⁴, early stopping with patience=15 on validation KL divergence). Best
checkpoint saved at minimum validation KL.

### Fixed Hyperparameters (All Runs)

| Hyperparameter          | Value                    |
| ----------------------- | ------------------------ |
| Optimiser               | Adam                     |
| Weight decay            | 5×10⁻⁴                   |
| LR schedule             | Cosine annealing to 10⁻⁶ |
| Batch size              | 128                      |
| Random seed             | 42                       |
| Early stopping patience | 15 epochs                |
| Early stopping metric   | Validation KL divergence |

### Backbone Initialisation Strategies

Three strategies were investigated (see Ablation A): random (Xavier-uniform),
CIFAR-10 hard-label pre-training, and ImageNet pre-training.

---

## Results

### Core Performance (Best Model)

Best model: KL loss, linear head, ImageNet pre-training.

| Metric                         | Value  |
| ------------------------------ | ------ |
| KL Divergence                  | 0.4962 |
| Jensen-Shannon Divergence      | 0.1099 |
| Cosine Similarity              | 0.8553 |
| Pearson Correlation (entropy)  | 0.2996 |
| Spearman Correlation (entropy) | 0.2856 |
| Precision@100                  | 0.170  |
| Precision@200                  | 0.255  |
| Precision@500                  | 0.382  |

### Ablation Studies

**Backbone Initialisation (Ablation A)** — KL loss, linear head:

| Initialisation       | KL Divergence |
| -------------------- | ------------- |
| ImageNet pre-trained | 0.4962        |
| CIFAR-10 pre-trained | 0.6129        |
| Random               | 0.7158        |

ImageNet pre-training reduces KL divergence by ~31% compared to random
initialisation.

**Training Data Strategy (Ablation C)** — KL loss, linear head:

| Strategy                        | KL Divergence |
| ------------------------------- | ------------- |
| Soft-label only (random init)   | 0.7158        |
| Hard pre-train + soft fine-tune | 0.4962        |

**Prediction Head Architecture (Ablation D)** — KL loss, ImageNet pre-training:

| Head Type     | KL Divergence |
| ------------- | ------------- |
| Linear        | 0.4962        |
| MLP (2-layer) | 0.5241        |

The linear head outperforms the MLP — with only 6,000 soft-label examples, the
extra MLP capacity leads to overfitting despite early stopping.

### Key Findings

- KL divergence serves as a strong baseline for pointwise distribution matching.
- Linear heads generalise better than MLP heads in this limited-data regime.
- ImageNet pre-training reduces KL by 31% vs. random initialisation.
- Hard-label pre-training substantially outperforms soft-label-only training.
- EMD optimises semantic structural distance rather than pointwise probability
  matching — its high KL (5.29) is therefore informative rather than simply a
  failure.
- Custom composite loss improves Pearson correlation (0.3238 vs. 0.2996) at the
  cost of raw KL.
- Temperature scaling (T=2.0) worsens performance, indicating the model's logits
  are already well-calibrated.

### Robustness

- **Annotator subsampling:** Performance improves rapidly from 6 to ~10
  annotators, then plateaus, suggesting ~10 annotators provide sufficient signal
  for reliable soft-label estimation.
- **OOD corruptions (Gaussian noise):** Mean predicted entropy increases with
  noise severity, saturating at ~1.25 bits — expected behaviour once images
  become fully corrupted.

### Explainability (Grad-CAM)

- **Low-entropy images** (H(p) < 0.1 bits): heatmaps are tightly focused on the
  dominant object.
- **High-entropy images** (H(p) > 1.5 bits): entropy-weighted heatmaps are
  diffuse across the entire image, as no single region drives the prediction.

Failure cases fall into three types: (1) collapsed prediction on an ambiguous
image — the most common; (2) spread prediction on a clear image; (3) wrong modal
class.

---

## Project Structure

```
annotater-disagreement-cifar10/
├── analysis/
│   ├── cifar10h_analysis.py    # In-depth dataset analysis scripts
│   └── manual_disagreement.py  # Manual high-entropy image inspection
├── data/
│   ├── __init__.py
│   ├── dataset.py              # CIFAR10HWrapper class
│   ├── pipeline.py             # Loading, splitting, DataLoaders, sanity checks
│   ├── statistics.py           # Entropy computations, confusion matrix
│   ├── visualization.py        # Plot generation
│   ├── cifar10h-probs.npy      # Pre-computed soft-label probability vectors
│   ├── cifar10h-counts.npy     # Raw annotator count vectors
│   └── cifar10h-raw.csv        # Full raw annotation data
├── docs/
│   ├── overview.md             # Detailed file-by-file documentation
│   └── cifar_resnet18_architecture_white.png
├── evaluation/
│   ├── eval.py                 # Full test-set evaluation loop
│   ├── metrics.py              # All test metrics (KL, JSD, cosine, Pearson, P@K)
│   ├── robustness.py           # Corruption and subsampling experiments
│   └── visualize.py            # Scatter plots, metric comparisons, qualitative grids
├── experiments/
│   ├── configs/                # Per-experiment YAML/JSON configuration files
│   └── run.py                  # Experiment runner
├── explainability/
│   └── grad_cam.py             # Grad-CAM implementation and plotting
├── losses/
│   ├── __init__.py             # Loss factory function
│   ├── kl_divergence.py
│   ├── js_divergence.py
│   ├── emd_loss.py
│   └── custom_losses.py        # Focal-weighted KL + entropy penalty
├── models/
│   ├── cifar_resnet.py         # CIFAR-adapted ResNet-18 + prediction heads
│   └── weights/                # Stored pre-trained weight files
├── notebooks/
│   └── 00_data_exploration.ipynb
├── outputs/
│   ├── ablation_results.csv    # Aggregated ablation metrics
│   ├── eval/                   # Per-model evaluation outputs
│   ├── explainability/         # Grad-CAM outputs
│   ├── logs/                   # Per-epoch training logs
│   ├── manual_disagreement/    # Manual inspection outputs
│   └── plots/                  # Generated visualisation plots
├── tests/
│   ├── verify_pipeline.py      # Quick pipeline sanity check (~30 seconds)
│   └── run_ce_check.py         # Cross-entropy sanity check
├── training/
│   ├── config.py               # Centralised settings (seed, splits, paths)
│   ├── train.py                # Main training loop with early stopping
│   └── metrics_logger.py       # CSV/NPY per-epoch logging
├── utils/
│   └── device.py               # Device detection and configuration
├── visualisations/
│   ├── data_plots.py           # Dataset visualisation plots
│   ├── eval_plots.py           # Evaluation and metric plots
│   ├── grad_cam_plots.py       # Grad-CAM grids
│   ├── model_diagram.py        # Architecture diagram generation
│   └── training_curves.py      # Train/val loss curve plotting
├── docs/
│   └── G_223_184_050_218_219.pdf   # Project report
├── abalation.py                # CSV aggregation across ablation runs
├── main_data.py                # Data pipeline orchestrator (7-step)
├── main.py                     # Main training entry point
├── README.md
└── requirements.txt
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

```bash
pip install -r requirements.txt
```

**Note:** On Windows, DataLoaders use `num_workers=0` due to multiprocessing
constraints. On Linux/macOS, `num_workers=4` is used for faster training. All
paths use `pathlib` for cross-platform compatibility.

---

## Sources & Citations

- **CIFAR-10H Dataset**: Peterson, J. C., Battleday, R. M., Griffiths, T. L., &
  Russakovsky, O. (2019). Human uncertainty makes classification more robust.
  _Proc. IEEE/CVF ICCV_. https://github.com/jcpeterson/cifar-10h
- **CIFAR-10 (Original)**: Krizhevsky, A., & Hinton, G. (2009). Learning
  multiple layers of features from tiny images. _Technical Report, University of
  Toronto_. https://www.cs.toronto.edu/~kriz/cifar.html
- **ResNet-18**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual
  learning for image recognition. _Proc. IEEE CVPR_.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations
  from deep networks via gradient-based localization. _Proc. IEEE ICCV_.
- **ResNet-18 Pretrained Weights**:
  - https://github.com/huyvnphan/PyTorch_CIFAR10
  - https://www.kaggle.com/datasets/pytorch/resnet18
