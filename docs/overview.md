# Predicting Human Annotator Disagreement on CIFAR-10: Codebase Overview

This document provides a complete file-by-file reference for the project
codebase. The model takes CIFAR-10H soft labels — probability distributions over
10 classes from approximately 50 human annotators per image — and trains a
CIFAR-adapted ResNet-18 to predict that distribution rather than a single hard
class.

---

## Directory Map

```
annotater-disagreement-cifar10/
├── analysis/           Dataset analysis scripts (entropy, manual inspection)
├── data/               Data loading, splitting, statistics, visualisation
├── docs/               This file, architecture diagram, and project report
├── evaluation/         Inference, metrics, robustness checks, eval plots
├── experiments/        Experiment runner and per-run YAML configs
├── explainability/     Grad-CAM implementation
├── losses/             KL, JSD, custom composite, and EMD loss functions
├── models/             CIFAR-adapted ResNet-18 with linear/MLP heads
├── notebooks/          Data exploration notebook
├── outputs/            All generated artefacts (logs, plots, CSVs, evals)
├── tests/              Pipeline verification scripts
├── training/           Config, training loop, metrics logger
├── utils/              Device selection and mixed-precision helpers
├── visualisations/     Standalone plotting scripts (data, training, eval, Grad-CAM)
├── abalation.py        Aggregates ablation CSVs into a single summary
├── main_data.py        Data pipeline orchestrator (7-step)
├── main.py             Training entry point
└── requirements.txt    Python dependencies
```

---

## Root Files

### `main.py`

Entry point for training. Parses CLI arguments (loss function, head type,
backbone initialisation strategy) and delegates to `training/train.py`.

### `main_data.py`

Orchestrator for the complete data pipeline. Runs six sequential stages: set
seed → load dataset → run sanity checks → compute statistics → create stratified
splits → create DataLoaders.

- `main()` → `tuple[CIFAR10HWrapper, dict, dict, dict]` Returns the dataset
  wrapper, splits dict, loaders dict, and statistics dict.

### `abalation.py`

Reads per-experiment CSV logs from `outputs/logs/{exp}_metrics.csv` for every
combination of loss function and head type, then writes a consolidated summary
to `outputs/ablation_results.csv`.

### `requirements.txt`

Pip dependencies: `torch>=2.0`, `torchvision`, `numpy`, `scikit-learn`,
`matplotlib`.

---

## `data/`

The most complete subsystem. Handles all data loading, alignment, splitting,
statistics, and required visualisations.

### `data/dataset.py` — `CIFAR10HWrapper`

Pairs torchvision CIFAR-10 images (`train=False`) with CIFAR-10H soft-label
probability arrays. Enforces that every soft-label row sums to 1.0 (tolerance
10⁻⁵). Hard-label alignment is verified by checking that the majority vote of
the soft distribution matches the CIFAR-10 hard label for ≥99% of images.

| Method               | Signature                                    | Returns                             |
| -------------------- | -------------------------------------------- | ----------------------------------- |
| `__init__`           | `(root, soft_labels_path, train, transform)` | —                                   |
| `__getitem__`        | `(idx: int)`                                 | `tuple[PIL.Image, Tensor[10], int]` |
| `get_entropy`        | `(idx: int)`                                 | `float` in [0, 3.32]                |
| `get_majority_label` | `(idx: int)`                                 | `int`                               |
| `get_all_entropies`  | `()`                                         | `np.ndarray[N]`                     |

### `data/pipeline.py` — Core transform and batching factory

Handles reproducibility seeding, per-split transform creation, stratified
splitting, and DataLoader construction. Hardcodes CIFAR-10 normalisation
constants (μ = (0.4914, 0.4822, 0.4465), σ = (0.2023, 0.1994, 0.2010)).

| Function             | Signature                                                 | Returns              |
| -------------------- | --------------------------------------------------------- | -------------------- |
| `set_seed`           | `(seed: int)`                                             | `None`               |
| `get_transforms`     | `(is_train: bool)`                                        | `transforms.Compose` |
| `load_full_dataset`  | `(data_dir, soft_labels_path)`                            | `CIFAR10HWrapper`    |
| `create_splits`      | `(dataset, train_ratio, val_ratio, seed)`                 | `dict`               |
| `create_dataloaders` | `(splits_dict, batch_size, num_workers, transforms_dict)` | `dict`               |
| `run_sanity_checks`  | `(dataset: CIFAR10HWrapper)`                              | `bool`               |

Augmentation policy: training split receives `RandomHorizontalFlip(p=0.5)` and
`RandomCrop(32, padding=4, padding_mode='reflect')`; validation and test receive
normalisation only.

### `data/statistics.py` — Entropy and agreement computations

| Function                             | Signature                                 | Returns              |
| ------------------------------------ | ----------------------------------------- | -------------------- |
| `compute_entropies`                  | `(soft_labels: np.ndarray[N, 10])`        | `np.ndarray[N]`      |
| `compute_class_average_entropy`      | `(entropies, hard_labels, num_classes)`   | `dict`               |
| `compute_annotator_confusion_matrix` | `(soft_labels, hard_labels, num_classes)` | `np.ndarray[10, 10]` |
| `compute_majority_agreement`         | `(soft_labels, hard_labels)`              | `float`              |
| `identify_extreme_examples`          | `(entropies, n)`                          | `dict`               |
| `compute_all_statistics`             | `(dataset: CIFAR10HWrapper)`              | `dict`               |

### `data/visualization.py` — Required data analysis plots

Generates the four plots required by the assignment brief, saved to
`outputs/plots/`.

| Function                          | Output                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| `plot_entropy_histogram`          | Distribution of Shannon entropy across all 10,000 images                             |
| `plot_class_entropy_bar`          | Per-class average annotator disagreement                                             |
| `plot_annotator_confusion_matrix` | 10×10 matrix of annotator response patterns                                          |
| `plot_extreme_examples`           | Image grid of lowest and highest entropy examples with their annotator distributions |

### `data/__init__.py`

Exposes pipeline and statistics functions as the public `data` package API.

---

## `models/`

### `models/cifar_resnet.py` — CIFAR-adapted ResNet-18

Replaces the standard ResNet-18 stem (7×7 conv, stride 2, max-pool) with a 3×3
conv at stride 1 and no pooling, preserving spatial resolution through the early
layers for 32×32 inputs. The backbone produces a `[B, 512]` feature vector via
global average pooling, which is then passed to the prediction head.

**`CIFAR10ResNet18Backbone`**

- `forward(x: Tensor[B, 3, 32, 32])` → `Tensor[B, 512]`

**`SoftLabelPredictor`**

- `__init__(backbone, head_type: str)` — `head_type` is `'linear'` or `'mlp'`
- `forward(x: Tensor[B, 3, 32, 32])` → `Tensor[B, 10]` (softmax probabilities)

**Linear head** (`head_type='linear'`): single `Linear(512, 10)` + softmax.
5,130 parameters. Lower overfitting risk on 6,000 soft-label examples.

**MLP head** (`head_type='mlp'`): `Linear(512, 256)` → ReLU →
`Linear(256, 10)` + softmax. 133,902 parameters. More expressive but overfits
more readily.

| Component          | Parameters |
| ------------------ | ---------- |
| ResNet-18 backbone | 11,173,952 |
| Linear head        | 5,130      |
| MLP head           | 133,902    |

---

## `losses/`

### `losses/__init__.py` — Loss factory

`get_loss_function(loss_name: str, beta: float, epsilon: float)` → `Callable`
Binds hyperparameters and returns the requested loss function by name.

### `losses/kl_divergence.py`

`kl_divergence(pred_probs: Tensor[B, 10], target_probs: Tensor[B, 10])` →
`Tensor[]` Wrapper around `F.kl_div` in log-space mode. Predictions are clamped
to 10⁻⁹ before taking logarithms to prevent undefined values where `q(y) = 0`.

### `losses/js_divergence.py`

`js_divergence(pred_probs: Tensor[B, 10], target_probs: Tensor[B, 10])` →
`Tensor[]` Symmetric version of KL: computes the mixture `M = (p + q) / 2`, then
averages `KL(p‖M)` and `KL(q‖M)`. Bounded in [0, 1], never infinite.

### `losses/custom_losses.py` — Focal-weighted KL + entropy error penalty

`custom_composite_loss(pred_probs, target_probs, beta)` → `Tensor[]`

The loss has two terms:

1. **Focal-weighted KL:** each image's KL contribution is scaled by
   `wᵢ = 1 + α · H(pᵢ)` (α = 2.0), up-weighting the high-disagreement images
   that standard KL deprioritises.
2. **Entropy error penalty:** `λ · (H(q) − H(p))²` (λ = 0.5), directly
   penalising miscalibrated uncertainty regardless of distributional distance.

### `losses/emd_loss.py` — Differentiable Earth Mover's Distance

`sinkhorn_loss(pred_probs, target_probs, cost_matrix: Tensor[10, 10], epsilon, max_iter)`
→ `Tensor[]` Solves optimal transport via Sinkhorn iterations using a
pre-defined semantic cost matrix (cat–dog distance < cat–truck distance). Runs
slower than KL/JSD but captures class-level semantic structure.

---

## `training/`

### `training/config.py`

Central configuration: fixed seed (42), train/val/test split ratios (60/20/20),
batch size (128), optimiser (Adam), learning rate (10⁻⁴ fine-tuning), weight
decay (5×10⁻⁴), cosine annealing schedule, early stopping patience (15 epochs),
and output directory paths. Raises assertions on import if values are invalid.

### `training/train.py`

Main training script. Handles CLI argument parsing (`argparse`) for loss
function, head type, and backbone initialisation strategy. Wraps the model in
`torch.compile()` and runs the train/val loop with mixed-precision `autocast`.

| Function                                                                           | Returns                                             |
| ---------------------------------------------------------------------------------- | --------------------------------------------------- |
| `train_one_epoch(model, loader, optimizer, device, scaler, autocast_ctx, loss_fn)` | `float` (mean train loss)                           |
| `validate(model, loader, device, loss_fn)`                                         | `float` (mean val loss)                             |
| `main()`                                                                           | Outer epoch loop, early stopping, checkpoint saving |

Checkpoints are saved to `outputs/` whenever validation KL divergence improves.
Early stopping triggers after 15 epochs without improvement.

### `training/metrics_logger.py` — `MetricsLogger`

Appends per-epoch `train_loss` and `val_loss` rows to a CSV file at
`outputs/logs/{exp}_metrics.csv`.

- `log(epoch: int, metrics_dict: dict)` → `None`
- `close()` → `None`

---

## `evaluation/`

### `evaluation/metrics.py`

All test-set evaluation mathematics in one place.

| Function                       | Signature                         | Returns                                       |
| ------------------------------ | --------------------------------- | --------------------------------------------- |
| `compute_entropy`              | `(probs: Tensor[B, 10])`          | `Tensor[B]`                                   |
| `compute_kl`                   | `(p, q: Tensor[B, 10])`           | `float`                                       |
| `compute_entropy_correlations` | `(true_p, pred_q: Tensor[N, 10])` | `tuple[float, float]` (Pearson r, Spearman ρ) |
| `compute_precision_at_k`       | `(true_p, pred_q, k: int)`        | `float`                                       |
| `run_all_metrics`              | `(true_p, pred_q: Tensor[N, 10])` | `dict`                                        |

### `evaluation/eval.py`

Loads all trained model checkpoints, runs inference over the 2,000-image test
set, computes the full metric suite, and triggers evaluation plots.

| Function                                         | Returns                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `load_model(checkpoint_path, head_type, device)` | `SoftLabelPredictor`                                                                                      |
| `run_inference(model, test_loader, device)`      | `tuple[Tensor[N,10], Tensor[N,10], Tensor[N], Tensor[N,3,32,32]]` — (pred_q, true_p, hard_labels, images) |

### `evaluation/robustness.py`

Checks model stability under two conditions.

| Function                                                           | Returns                                            |
| ------------------------------------------------------------------ | -------------------------------------------------- |
| `annotator_subsampling_check(raw_counts, pred_q, subsample_sizes)` | `dict` — Pearson correlation per annotator count   |
| `ood_corruption_check(model, test_images, severities)`             | `dict` — mean predicted entropy per severity level |
| `per_class_kl(true_p, pred_q, hard_labels)`                        | `dict` — per-class mean KL divergence              |

### `evaluation/visualize.py`

Generates all evaluation plots from raw tensors and saves them as PNGs.

| Function                     | Output                                           |
| ---------------------------- | ------------------------------------------------ |
| `plot_entropy_scatter`       | Predicted vs. true Shannon entropy scatter plot  |
| `plot_metrics_comparison`    | Grouped bar chart across loss functions          |
| `plot_qualitative_examples`  | Image grid across low-to-high disagreement range |
| `plot_per_class_predictions` | Per-class KL divergence bar chart                |

---

## `explainability/`

### `explainability/grad_cam.py`

Implements Grad-CAM for soft-label models. Three target strategies are
supported: `top_pred` (gradient w.r.t. argmax of q), `top_true` (gradient w.r.t.
dominant class in p), and `entropy_weighted` (weighted sum of per-class CAMs
using `wᵧ = −p(y) log p(y)`). The entropy-weighted strategy is most informative
for high-entropy images.

Target layer: the final residual block (`layer4[-1]`).

---

## `analysis/`

### `analysis/cifar10h_analysis.py`

Standalone entropy analysis without a neural network. Computes classical Shannon
entropy over the raw CIFAR-10H probability arrays and generates
true-vs-predicted histograms for reference.

- `compute_shannon_entropy(probs: np.ndarray[N, 10], eps: float)` →
  `np.ndarray[N]`
- `plot_entropy_analysis(true_probs, pred_probs, save_path)` → `None`

### `analysis/manual_disagreement.py`

Loads high-entropy images for manual inspection and categorisation. Supports the
compulsory manual disagreement source analysis (ambiguous object identity, poor
image quality, multi-object content, boundary case, unusual viewpoint). Results
saved to `outputs/manual_disagreement/`.

---

## `visualisations/`

Standalone plotting scripts decoupled from the evaluation and training loops.

| File                 | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `data_plots.py`      | Dataset-level plots (entropy histogram, class entropy bar chart) |
| `eval_plots.py`      | Metric comparison charts and scatter plots                       |
| `grad_cam_plots.py`  | Grad-CAM heatmap grids for low- and high-entropy images          |
| `model_diagram.py`   | Architecture diagram generation                                  |
| `training_curves.py` | Train/val loss curves per experiment run                         |

---

## `utils/`

### `utils/device.py`

Device detection and mixed-precision context helpers.

| Function                                                         | Returns                                            |
| ---------------------------------------------------------------- | -------------------------------------------------- |
| `get_device()`                                                   | `torch.device` — CUDA, MPS (Apple Silicon), or CPU |
| `get_autocast_context(device)`                                   | `torch.amp.autocast` context manager               |
| `move_batch_to_device(images, soft_labels, hard_labels, device)` | `tuple[Tensor, Tensor, Tensor]`                    |

---

## `experiments/`

### `experiments/run.py`

Reads a YAML config from `experiments/configs/` and launches a training run with
the specified loss function, head type, and initialisation strategy. Used to
systematically execute all ablation conditions.

### `experiments/configs/`

One YAML file per experimental condition. Each file specifies `loss`,
`head_type`, `backbone_init`, and any loss-specific hyperparameters (e.g.
`beta`, `epsilon`).

---

## `outputs/`

All generated artefacts. Nothing in this directory is checked into version
control.

| Subdirectory                   | Contents                                             |
| ------------------------------ | ---------------------------------------------------- |
| `outputs/plots/`               | Data pipeline visualisation plots (4 required plots) |
| `outputs/logs/`                | Per-epoch CSV logs from each training run            |
| `outputs/eval/`                | Evaluation metric outputs and plots per model        |
| `outputs/explainability/`      | Grad-CAM heatmap images                              |
| `outputs/manual_disagreement/` | Manual inspection outputs                            |
| `outputs/ablation_results.csv` | Consolidated ablation summary table                  |

---

## `tests/`

### `tests/verify_pipeline.py`

Runs the data pipeline end-to-end and checks all seven sanity conditions without
generating plots. Completes in approximately 30 seconds. Run this after any
change to `data/` to confirm nothing is broken.

### `tests/run_ce_check.py`

Verifies that the cross-entropy baseline (hard-label-trained model on soft
targets) produces sensible values before beginning soft-label fine-tuning.

---

## `notebooks/`

### `notebooks/00_data_exploration.ipynb`

Interactive data exploration: entropy distributions, class-level disagreement
patterns, example images at various entropy levels, and annotator confusion
matrix. The plots here informed the design of the custom focal-weighted loss.
