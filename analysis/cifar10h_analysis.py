"""
============================================================================
CIFAR-10H Disagreement Prediction — Analysis & Visualization Suite
=============================================================================
Repo structure assumed (from your README):
  data/dataset.py          → CIFAR10HWrapper  (images + soft labels)
  data/pipeline.py         → create_dataloaders, load_full_dataset, etc.
  models/cifar_resnet.py   → your ResNet/EfficientNet backbone
  evaluation/metrics.py    → KL, JSD, cosine (you fill these in)
  explainability/grad_cam.py → this file houses GradCAM
  visualisations/          → drop output PNGs here
 
Each section below maps to one of your requested tasks.
Call each function AFTER your normal evaluation loop.
=============================================================================
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — safe in training environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy as scipy_entropy
 
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
 
 
# =============================================================================
# SECTION 1 — Shannon Entropy Analysis
# WHERE TO CALL: right after your test-loop collects all_true_probs /
#                all_pred_probs (numpy arrays, shape [N, 10])
# =============================================================================
 
def compute_shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy (bits) for a batch of distributions.
    probs : (N, 10) — each row sums to 1
    Returns: (N,) entropy values
    """
    clipped = np.clip(probs, eps, 1.0)
    return -np.sum(clipped * np.log2(clipped), axis=1)
 
 
def plot_entropy_analysis(
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    save_path: str = "visualisations/entropy_analysis.png"
):
    """
    Plots:
      (a) Histogram of true entropy
      (b) Histogram of predicted entropy
      (c) Scatter: true entropy vs predicted entropy
 
    WHERE TO CALL in your pipeline:
      After evaluate() in training/train.py or in evaluation/metrics.py.
      Example:
        true_probs, pred_probs = collect_test_predictions(model, test_loader)
        plot_entropy_analysis(true_probs, pred_probs)
 
    Args:
        true_probs : np.ndarray (N, 10) — CIFAR-10H annotator distributions
        pred_probs : np.ndarray (N, 10) — model softmax outputs
        save_path  : where to write the PNG
    """
    H_true = compute_shannon_entropy(true_probs)
    H_pred = compute_shannon_entropy(pred_probs)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Shannon Entropy: True vs Predicted Distributions", fontsize=13)
 
    # --- (a) True entropy histogram ---
    ax = axes[0]
    ax.hist(H_true, bins=50, color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.axvline(H_true.mean(), color="navy", linestyle="--",
               label=f"mean = {H_true.mean():.3f}")
    ax.set_xlabel("Shannon Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("True Entropy H(p)")
    ax.legend(fontsize=8)
 
    # --- (b) Predicted entropy histogram ---
    ax = axes[1]
    ax.hist(H_pred, bins=50, color="#FF5722", edgecolor="white", linewidth=0.4)
    ax.axvline(H_pred.mean(), color="darkred", linestyle="--",
               label=f"mean = {H_pred.mean():.3f}")
    ax.set_xlabel("Shannon Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Entropy H(q)")
    ax.legend(fontsize=8)
 
    # --- (c) Scatter ---
    ax = axes[2]
    ax.scatter(H_true, H_pred, alpha=0.25, s=6, color="#607D8B")
    # Perfect prediction line
    lim = max(H_true.max(), H_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="perfect prediction")
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(H_true, H_pred)
    r_s, _ = spearmanr(H_true, H_pred)
    ax.set_xlabel("True Entropy (bits)")
    ax.set_ylabel("Predicted Entropy (bits)")
    ax.set_title(
        f"True vs Predicted Entropy\n"
        f"Pearson r={r_p:.3f}  Spearman ρ={r_s:.3f}"
    )
    ax.legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[entropy_analysis] Saved → {save_path}")
    return H_true, H_pred
 
 
# =============================================================================
# SECTION 2 — Distribution Bar Plots (low-entropy vs high-entropy examples)
# WHERE TO CALL: same place as entropy analysis, or in a notebook
# =============================================================================
 
def plot_distribution_comparison(
    images,                       # (N, 3, 32, 32) torch.Tensor or numpy — raw test images (un-normalized)
    true_probs: np.ndarray,       # (N, 10)
    pred_probs: np.ndarray,       # (N, 10)
    n_examples: int = 4,          # 4 is good for a report (2 low + 2 high)
    save_path: str = "visualisations/distribution_comparison.png"
):
    """
    Selects n_examples//2 low-entropy and n_examples//2 high-entropy images.
    For each shows the image + side-by-side bar charts p(y|x) vs q(y|x).
    """
    H_true = compute_shannon_entropy(true_probs)
    half = n_examples // 2

    # Pick examples
    sorted_idx = np.argsort(H_true)
    low_idx  = sorted_idx[:half]
    high_idx = sorted_idx[-half:]
    selected = list(low_idx) + list(high_idx)
    labels   = (["Low Entropy"] * half) + (["High Entropy"] * half)

    fig, axes = plt.subplots(n_examples, 3, figsize=(10, 2.8 * n_examples))
    fig.suptitle("True vs Predicted Distributions (p vs q)", fontsize=13, y=1.01)

    x_pos = np.arange(10)
    bar_width = 0.38

    for row, (idx, label) in enumerate(zip(selected, labels)):
        img_ax  = axes[row, 0]
        bar_ax  = axes[row, 1]
        diff_ax = axes[row, 2]

        # --- Image ---
        img = images[idx]
        # Convert to (H, W, 3) numpy array for imshow
        if hasattr(img, 'permute'):                 # torch Tensor
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            if img.shape[0] == 3:                   # (3, H, W) -> (H, W, 3)
                img_np = img.transpose(1, 2, 0)
            else:
                img_np = img
        else:
            img_np = np.asarray(img)
        img_np = np.clip(img_np, 0, 1)

        img_ax.imshow(img_np, interpolation="nearest")
        img_ax.set_title(
            f"{label}\nH_true={H_true[idx]:.2f} bits",
            fontsize=8
        )
        img_ax.axis("off")

        # --- Bar chart ---
        p = true_probs[idx]
        q = pred_probs[idx]
        bar_ax.bar(x_pos - bar_width/2, p, bar_width,
                   label="p(y|x) true", color="#2196F3", alpha=0.85)
        bar_ax.bar(x_pos + bar_width/2, q, bar_width,
                   label="q(y|x) pred", color="#FF5722", alpha=0.85)
        bar_ax.set_xticks(x_pos)
        bar_ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=7)
        bar_ax.set_ylabel("Probability")
        bar_ax.set_title("Distribution p vs q", fontsize=8)
        bar_ax.legend(fontsize=7)
        bar_ax.set_ylim(0, 1.05)

        # --- Difference plot ---
        diff = q - p
        colors = ["#4CAF50" if d >= 0 else "#F44336" for d in diff]
        diff_ax.bar(x_pos, diff, color=colors, alpha=0.85)
        diff_ax.axhline(0, color="black", linewidth=0.8)
        diff_ax.set_xticks(x_pos)
        diff_ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=7)
        diff_ax.set_ylabel("q − p")
        diff_ax.set_title("Prediction Error per Class", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[distribution_comparison] Saved → {save_path}")
 
 
# =============================================================================
# SECTION 3 — Grad-CAM
# FILE:  explainability/grad_cam.py  (paste class there)
# WHERE TO CALL: after training, pass your eval-mode model
# =============================================================================
 
class GradCAM:
    """
    Grad-CAM for a ResNet/EfficientNet that outputs a 10-d soft distribution.
 
    WHICH CLASS TO USE FOR GRAD-CAM IN A DISTRIBUTION MODEL?
    ─────────────────────────────────────────────────────────
    Three valid strategies:
      1. Top predicted class  → arg max q(y|x)
         Best for: "what did the model focus on most?"
      2. True top class       → arg max p(y|x)
         Best for: "did the model look at the right thing?"
      3. Entropy-weighted sum → Σ_y H(q) * CAM_y
         Best for: "what regions drove the model's uncertainty?"
         (useful for high-entropy images — use strategy 3 there)
 
    We implement strategies 1 & 3 and let you choose per image.
 
    USAGE:
        grad_cam = GradCAM(model, target_layer=model.layer4[-1])  # ResNet
        # or for EfficientNet:
        # grad_cam = GradCAM(model, target_layer=model.features[-1])
 
        cam = grad_cam.generate(image_tensor, target_class=None)  # strategy 1
        overlay = grad_cam.overlay(raw_image_np, cam)
    """
 
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None
        self._register_hooks()
 
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()
 
        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()
 
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
 
    def generate(
        self,
        image: torch.Tensor,          # (1, 3, 32, 32) — normalized
        target_class: int | None = None,
        strategy: str = "top_pred"    # "top_pred" | "entropy_weighted"
    ) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1].
 
        strategy="top_pred"        → gradient w.r.t. argmax class
        strategy="entropy_weighted" → sum of gradients weighted by –p*log(p)
        """
        self.model.eval()
        image = image.requires_grad_(True)
 
        logits = self.model(image)          # (1, 10) pre-softmax or post-softmax
        probs  = F.softmax(logits, dim=1)   # ensure we have probs
 
        if strategy == "top_pred" or target_class is not None:
            cls = target_class if target_class is not None else probs.argmax(dim=1).item()
            score = probs[0, cls]
            self.model.zero_grad()
            score.backward()
            cam = self._compute_cam()
 
        elif strategy == "entropy_weighted":
            # Σ_c H_c * output_c   where H_c = -p_c * log(p_c)
            eps = 1e-12
            weights = -(probs * torch.log(probs + eps))  # (1, 10)
            score = (weights * probs).sum()
            self.model.zero_grad()
            score.backward()
            cam = self._compute_cam()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
 
        return cam
 
    def _compute_cam(self) -> np.ndarray:
        grads  = self._gradients[0]      # (C, H, W)
        acts   = self._activations[0]    # (C, H, W)
        weights = grads.mean(dim=(1, 2)) # global average pooling of gradients
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
 
    @staticmethod
    def overlay(
        raw_image: np.ndarray,      # (H, W, 3) uint8 or float in [0,1]
        cam: np.ndarray,            # (h, w) in [0,1]
        alpha: float = 0.45
    ) -> np.ndarray:
        """Overlay heatmap on CIFAR-10 image (bilinearly upsample cam to image size)."""
        import cv2
        h, w = raw_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        img_float = raw_image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0
        overlay = (1 - alpha) * img_float + alpha * heatmap
        return np.clip(overlay, 0, 1)
 
 
def plot_gradcam_grid(
    model: nn.Module,
    target_layer: nn.Module,
    images_norm: torch.Tensor,      # (N, 3, 32, 32) normalized
    images_raw: np.ndarray,         # (N, 32, 32, 3) un-normalized [0,1]
    true_probs: np.ndarray,         # (N, 10)
    pred_probs: np.ndarray,         # (N, 10)
    n_each: int = 3,
    save_path: str = "visualisations/gradcam_grid.png"
):
    """
    Shows Grad-CAM for:
      - n_each lowest-disagreement images  (low H_true)
      - n_each highest-disagreement images (high H_true)
 
    Each row: [original image] [top-class CAM] [entropy-weighted CAM]
 
    WHERE TO CALL:
      After loading your best checkpoint, call this once on the test set.
      Pass the same test_loader images you used for evaluation.
    """
    H_true  = compute_shannon_entropy(true_probs)
    sorted_idx = np.argsort(H_true)
    low_idx    = sorted_idx[:n_each]
    high_idx   = sorted_idx[-n_each:]
    selected   = list(low_idx) + list(high_idx)
    labels     = (["Low Disagree"] * n_each) + (["High Disagree"] * n_each)
 
    grad_cam = GradCAM(model, target_layer)
    n_total  = len(selected)
    fig, axes = plt.subplots(n_total, 3, figsize=(9, 2.5 * n_total))
    fig.suptitle("Grad-CAM: Low vs High Disagreement Images", fontsize=12)
 
    for row, (idx, label) in enumerate(zip(selected, labels)):
        img_tensor = images_norm[idx].unsqueeze(0)   # (1,3,32,32)
        raw_img    = images_raw[idx]                  # (32,32,3)
 
        cam_top     = grad_cam.generate(img_tensor, strategy="top_pred")
        cam_entropy = grad_cam.generate(img_tensor, strategy="entropy_weighted")
 
        overlay_top     = GradCAM.overlay(raw_img, cam_top)
        overlay_entropy = GradCAM.overlay(raw_img, cam_entropy)
 
        top_cls  = pred_probs[idx].argmax()
        true_cls = true_probs[idx].argmax()
 
        axes[row, 0].imshow(np.clip(raw_img, 0, 1))
        axes[row, 0].set_title(
            f"{label} | H={H_true[idx]:.2f}\n"
            f"true={CIFAR10_CLASSES[true_cls]}  pred={CIFAR10_CLASSES[top_cls]}",
            fontsize=7
        )
        axes[row, 0].axis("off")
 
        axes[row, 1].imshow(overlay_top)
        axes[row, 1].set_title(f"CAM: top class\n({CIFAR10_CLASSES[top_cls]})", fontsize=7)
        axes[row, 1].axis("off")
 
        axes[row, 2].imshow(overlay_entropy)
        axes[row, 2].set_title("CAM: entropy-weighted", fontsize=7)
        axes[row, 2].axis("off")
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[gradcam_grid] Saved → {save_path}")
 
 
# =============================================================================
# SECTION 4 — Failure Case Analysis (worst KL divergence)
# WHERE TO CALL: inside evaluation/metrics.py or after test loop
# =============================================================================
 
def plot_failure_cases(
    images_raw: np.ndarray,        # (N, 32, 32, 3) in [0,1]
    true_probs: np.ndarray,        # (N, 10)
    pred_probs: np.ndarray,        # (N, 10)
    top_k: int = 10,
    save_path: str = "visualisations/failure_cases.png"
):
    """
    Finds top_k worst predictions by KL(p || q), prints stats, saves figure.
 
    REPORT SUGGESTION:
      Show 10 cases. Use a 2-row × 5-col grid: each cell = image + colour-coded
      bar comparison. This fits one page and is easy to discuss in your report.
 
    WHERE TO CALL:
      After collecting all test outputs:
        plot_failure_cases(raw_imgs, true_probs, pred_probs)
    """
    eps = 1e-12
    kl = np.sum(
        true_probs * np.log((true_probs + eps) / (pred_probs + eps)),
        axis=1
    )
    H_true = compute_shannon_entropy(true_probs)
    H_pred = compute_shannon_entropy(pred_probs)
 
    worst_idx = np.argsort(kl)[-top_k:][::-1]
 
    print("\n=== Top Failure Cases (worst KL divergence) ===")
    for rank, idx in enumerate(worst_idx):
        true_top = CIFAR10_CLASSES[true_probs[idx].argmax()]
        pred_top = CIFAR10_CLASSES[pred_probs[idx].argmax()]
        print(
            f"  Rank {rank+1:2d} | idx={idx:4d} | KL={kl[idx]:.4f} | "
            f"H_true={H_true[idx]:.3f} | H_pred={H_pred[idx]:.3f} | "
            f"true_top={true_top:<12s} | pred_top={pred_top}"
        )
 
    # Visualize: 2 rows × 5 cols
    cols = 5
    rows = (top_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.5))
    axes = axes.flatten()
 
    x_pos = np.arange(10)
    bar_w = 0.4
 
    for rank, idx in enumerate(worst_idx):
        ax = axes[rank]
        p = true_probs[idx]
        q = pred_probs[idx]
 
        # Mini image as inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="35%", height="35%", loc="upper right")
        axins.imshow(np.clip(images_raw[idx], 0, 1))
        axins.axis("off")
 
        ax.bar(x_pos - bar_w/2, p, bar_w, color="#2196F3", alpha=0.8, label="true")
        ax.bar(x_pos + bar_w/2, q, bar_w, color="#FF5722", alpha=0.8, label="pred")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [c[:3] for c in CIFAR10_CLASSES], rotation=45, ha="right", fontsize=6
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"KL={kl[idx]:.3f}\n"
            f"Ht={H_true[idx]:.2f}  Hp={H_pred[idx]:.2f}",
            fontsize=7
        )
        if rank == 0:
            ax.legend(fontsize=6, loc="upper left")
 
    # Hide unused cells
    for ax in axes[top_k:]:
        ax.axis("off")
 
    fig.suptitle("Top-10 Failure Cases by KL Divergence", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[failure_cases] Saved → {save_path}")
 
 
# =============================================================================
# SECTION 5 — Loss Comparison Bar Chart + Training Curves
# WHERE TO CALL: in experiments/run.py or a notebook after all runs finish
# =============================================================================
 
def plot_loss_comparison(
    results: dict,
    save_path: str = "visualisations/loss_comparison.png"
):
    """
    Grouped bar chart comparing KL, JSD, cosine similarity across loss functions.
 
    Args:
        results : dict with structure:
          {
            "KL Loss":  {"kl": 0.12, "jsd": 0.08, "cosine": 0.91},
            "JSD Loss": {"kl": 0.14, "jsd": 0.07, "cosine": 0.90},
            "Custom":   {"kl": 0.11, "jsd": 0.07, "cosine": 0.92},
          }
        save_path: output PNG path
 
    WHERE TO CALL:
      After all experiment runs, aggregate your metrics dict and call:
        plot_loss_comparison(results_dict)
    """
    loss_names = list(results.keys())
    metrics    = ["kl", "jsd", "cosine"]
    metric_labels = {"kl": "KL Divergence ↓", "jsd": "JSD ↓", "cosine": "Cosine Sim ↑"}
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
 
    n_groups = len(loss_names)
    n_metrics = len(metrics)
    bar_w = 0.22
    x = np.arange(n_groups)
 
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle("Loss Function Comparison on Test Set", fontsize=13)
 
    for mi, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[mi]
        vals = [results[ln][metric] for ln in loss_names]
        bars = ax.bar(x, vals, width=0.55, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
 
        # Annotate values on top of bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8
            )
 
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names, rotation=15, ha="right", fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=10)
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(vals) * 1.15)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[loss_comparison] Saved → {save_path}")
 
 
def plot_training_curves(
    history: dict,
    save_path: str = "visualisations/training_curves.png"
):
    """
    Plots train/val loss curves and a validation metric curve.
 
    Args:
        history: dict with keys — "train_loss", "val_loss", "val_kl"
                 each is a list of per-epoch values.
 
    WHERE TO CALL:
      In training/train.py after training loop ends:
        plot_training_curves(trainer.history)
    """
    epochs = range(1, len(history["train_loss"]) + 1)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training & Validation Curves", fontsize=12)
 
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color="#2196F3", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",
            color="#FF5722", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
 
    # Validation KL curve
    ax = axes[1]
    ax.plot(epochs, history["val_kl"], color="#4CAF50", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Validation KL Divergence")
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[training_curves] Saved → {save_path}")
 
 
# =============================================================================
# SECTION 6 — Robustness: Corruption + Entropy vs Severity
# WHERE TO CALL: in evaluation/robustness.py (your existing stub)
# =============================================================================
 
import torchvision.transforms.functional as TF
 
def apply_corruption(
    image: torch.Tensor,      # (3, 32, 32) normalized
    corruption: str,          # "gaussian_noise" | "gaussian_blur" | "contrast"
    severity: int,            # 1–5
    mean: list = [0.4914, 0.4822, 0.4465],
    std:  list = [0.2023, 0.1994, 0.2010]
) -> torch.Tensor:
    """
    Applies one of three corruptions at a given severity level (1=mild, 5=severe).
    Returns a normalized tensor of the same shape.
 
    Severity scales:
      gaussian_noise : sigma = severity * 0.05
      gaussian_blur  : kernel_size = 2*severity+1, sigma = severity * 0.5
      contrast       : factor = 1 - severity * 0.15   (reduces contrast)
    """
    # Denormalize
    t_mean = torch.tensor(mean).view(3, 1, 1)
    t_std  = torch.tensor(std).view(3, 1, 1)
    img    = image * t_std + t_mean                      # [0, ~1]
    img    = img.clamp(0, 1)
 
    if corruption == "gaussian_noise":
        sigma = severity * 0.05
        noise = torch.randn_like(img) * sigma
        img   = (img + noise).clamp(0, 1)
 
    elif corruption == "gaussian_blur":
        k = 2 * severity + 1
        sigma = severity * 0.5
        img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
 
    elif corruption == "contrast":
        factor = max(0.1, 1.0 - severity * 0.15)
        img = TF.adjust_contrast(img, factor)
 
    else:
        raise ValueError(f"Unknown corruption: {corruption}")
 
    # Re-normalize
    return (img - t_mean) / t_std
 
 
def plot_robustness_curves(
    model: nn.Module,
    images_norm: torch.Tensor,      # (N, 3, 32, 32) normalized
    device: torch.device,
    corruptions: list = ["gaussian_noise", "gaussian_blur", "contrast"],
    severities:  list = [1, 2, 3, 4, 5],
    save_path: str = "visualisations/robustness_entropy.png"
):
    """
    For each corruption × severity, computes the mean predicted entropy
    and plots entropy vs severity.
 
    WHERE TO CALL:
      In evaluation/robustness.py, after loading your best checkpoint:
        plot_robustness_curves(model, test_images_norm, device)
    """
    model.eval()
    results = {c: [] for c in corruptions}
 
    with torch.no_grad():
        for corruption in corruptions:
            for sev in severities:
                corrupted = torch.stack([
                    apply_corruption(img, corruption, sev)
                    for img in images_norm
                ]).to(device)
 
                logits = model(corrupted)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                H_pred = compute_shannon_entropy(probs)
                results[corruption].append(H_pred.mean())
 
    # Also compute clean entropy as severity-0 baseline
    with torch.no_grad():
        logits_clean = model(images_norm.to(device))
        probs_clean  = F.softmax(logits_clean, dim=1).cpu().numpy()
        H_clean      = compute_shannon_entropy(probs_clean).mean()
 
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_map = {
        "gaussian_noise": "#2196F3",
        "gaussian_blur":  "#FF5722",
        "contrast":       "#4CAF50"
    }
    ax.axhline(H_clean, color="gray", linestyle="--",
               linewidth=1.2, label=f"Clean baseline ({H_clean:.3f} bits)")
 
    for corruption, vals in results.items():
        ax.plot(severities, vals, marker="o", linewidth=2,
                color=colors_map[corruption], label=corruption.replace("_", " ").title())
 
    ax.set_xlabel("Corruption Severity", fontsize=11)
    ax.set_ylabel("Mean Predicted Entropy (bits)", fontsize=11)
    ax.set_title("Robustness: Predicted Entropy vs Corruption Severity", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(severities)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[robustness_curves] Saved → {save_path}")
 
 
# =============================================================================
# SECTION 7 — Three Advanced / Insightful Visualizations
# =============================================================================
 
# --- 7a. Per-Class Entropy Calibration Plot ---
def plot_per_class_entropy_calibration(
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    save_path: str = "visualisations/per_class_entropy_calibration.png"
):
    """
    INSIGHT: Shows how well the model calibrates entropy per true majority class.
    For each class, plots a violin/box of (H_pred - H_true) distribution.
 
    WHY INSIGHTFUL:
      Reveals which CLASSES the model systematically over/under-estimates
      uncertainty for. E.g., 'cat' images might have systematically under-
      predicted entropy, showing the model is over-confident on hard classes.
    """
    H_true = compute_shannon_entropy(true_probs)
    H_pred = compute_shannon_entropy(pred_probs)
    entropy_error = H_pred - H_true
    true_class = true_probs.argmax(axis=1)
 
    fig, ax = plt.subplots(figsize=(12, 5))
    data_by_class = [entropy_error[true_class == c] for c in range(10)]
 
    parts = ax.violinplot(data_by_class, positions=range(10),
                          showmedians=True, showmeans=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.6)
 
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2,
               label="perfect calibration")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("H_pred − H_true (bits)")
    ax.set_title(
        "Entropy Calibration Error per True Class\n"
        "Positive = model over-estimates uncertainty; Negative = under-estimates",
        fontsize=11
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[per_class_entropy_calibration] Saved → {save_path}")
 
 
# --- 7b. JSD Confusion Heatmap ---
def plot_jsd_confusion_heatmap(
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    n_bins: int = 10,
    save_path: str = "visualisations/jsd_confusion_heatmap.png"
):
    """
    INSIGHT: A 10×10 heatmap where cell (i, j) = mean JSD between images
    where true top-class=i and predicted top-class=j.
 
    WHY INSIGHTFUL:
      Goes beyond raw accuracy. Shows WHICH class confusions are also
      distribution confusions. You can identify: "When the model mistakes
      cats for dogs, how badly does its entire distribution diverge?"
      This reveals structured failure modes not visible in a hard-label
      confusion matrix.
    """
    true_class = true_probs.argmax(axis=1)
    pred_class = pred_probs.argmax(axis=1)
 
    eps = 1e-12
 
    def jsd(p, q):
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + \
               0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
 
    heatmap = np.zeros((10, 10))
    counts  = np.zeros((10, 10))
 
    for idx in range(len(true_probs)):
        i = true_class[idx]
        j = pred_class[idx]
        heatmap[i, j] += jsd(true_probs[idx], pred_probs[idx])
        counts[i, j]  += 1
 
    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap),
                        where=counts > 0)
 
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean JSD")
 
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted Top Class")
    ax.set_ylabel("True Top Class")
    ax.set_title(
        "Mean JSD per (True, Predicted) Class Pair\n"
        "High off-diagonal = distribution + label confusion",
        fontsize=11
    )
 
    # Annotate cells
    for i in range(10):
        for j in range(10):
            if counts[i, j] > 0:
                ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if heatmap[i, j] < 0.3 else "white")
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[jsd_confusion_heatmap] Saved → {save_path}")
 
 
# --- 7c. Entropy Reliability Diagram (Calibration) ---
def plot_entropy_reliability_diagram(
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    n_bins: int = 15,
    save_path: str = "visualisations/entropy_reliability_diagram.png"
):
    """
    INSIGHT: Bins images by predicted entropy, then plots mean true entropy
    vs mean predicted entropy per bin — a reliability diagram adapted
    to the distribution-prediction setting.
 
    WHY INSIGHTFUL:
      Standard calibration plots (confidence vs accuracy) don't apply here.
      This adaptation shows whether the model is: (a) well-calibrated in
      uncertainty, (b) systematically overconfident at low entropy,
      (c) underconfident at high entropy. This directly addresses one of the
      core evaluation criteria in your project spec.
    """
    H_true = compute_shannon_entropy(true_probs)
    H_pred = compute_shannon_entropy(pred_probs)
 
    bin_edges = np.linspace(H_pred.min(), H_pred.max(), n_bins + 1)
    bin_centers  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_true_mean = []
    bin_pred_mean = []
    bin_counts    = []
 
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (H_pred >= lo) & (H_pred < hi)
        if mask.sum() > 0:
            bin_true_mean.append(H_true[mask].mean())
            bin_pred_mean.append(H_pred[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_true_mean.append(np.nan)
            bin_pred_mean.append(np.nan)
            bin_counts.append(0)
 
    bin_true_mean = np.array(bin_true_mean)
    bin_pred_mean = np.array(bin_pred_mean)
    bin_counts    = np.array(bin_counts)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
 
    lim = max(H_true.max(), H_pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="perfect calibration")
    valid = ~np.isnan(bin_true_mean)
    ax1.plot(bin_pred_mean[valid], bin_true_mean[valid],
             "o-", color="#E91E63", linewidth=2, markersize=6, label="model")
 
    # Shade miscalibration region
    ax1.fill_between(
        bin_pred_mean[valid], bin_pred_mean[valid], bin_true_mean[valid],
        alpha=0.15, color="#E91E63", label="miscalibration gap"
    )
    ax1.set_xlabel("Mean Predicted Entropy per Bin (bits)")
    ax1.set_ylabel("Mean True Entropy per Bin (bits)")
    ax1.set_title("Entropy Reliability Diagram\n"
                  "Points above line = model under-estimates uncertainty", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
 
    # Histogram of sample counts per bin
    ax2.bar(bin_pred_mean[valid], bin_counts[valid],
            width=(bin_edges[1] - bin_edges[0]) * 0.8,
            color="#9E9E9E", alpha=0.7)
    ax2.set_xlabel("Predicted Entropy Bin (bits)")
    ax2.set_ylabel("Sample Count")
    ax2.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[entropy_reliability_diagram] Saved → {save_path}")
 
 
# =============================================================================
# HELPER: Collect test predictions (plug this into your eval loop)
# WHERE TO ADD: evaluation/metrics.py or inline in training/train.py
# =============================================================================
 
def collect_test_predictions(
    model: nn.Module,
    test_loader,
    device: torch.device
):
    """
    Runs the model over test_loader and returns:
      true_probs  : (N, 10) numpy  — CIFAR-10H soft labels
      pred_probs  : (N, 10) numpy  — model softmax outputs
      images_raw  : (N, 32, 32, 3) numpy [0,1] — denormalized images
 
    WHERE TO CALL in training/train.py:
      After your training loop / before calling any plot function:
 
        true_probs, pred_probs, images_raw = collect_test_predictions(
            model, dataloaders["test"], device
        )
 
    Your existing DataLoader returns: (images_norm, soft_labels)
    where soft_labels is the CIFAR-10H distribution.
    """
    model.eval()
    all_true, all_pred, all_imgs = [], [], []
 
    # CIFAR-10 normalization constants — adjust if you use different ones
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
 
    with torch.no_grad():
        for images_norm, soft_labels in test_loader:
            images_norm = images_norm.to(device)
            logits = model(images_norm)
            probs  = F.softmax(logits, dim=1).cpu()
 
            # Denormalize for visualization
            images_denorm = (images_norm.cpu() * std + mean).clamp(0, 1)
            images_denorm = images_denorm.permute(0, 2, 3, 1).numpy()
 
            all_true.append(soft_labels.numpy())
            all_pred.append(probs.numpy())
            all_imgs.append(images_denorm)
 
    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_imgs, axis=0)
    )
 
 
# =============================================================================
# QUICK-START: Call everything in sequence
# =============================================================================
 
def run_full_analysis(model, test_loader, device, target_layer=None):
    """
    One-shot entry point that runs all 7 sections.
 
    Example usage (add to the bottom of main.py or experiments/run.py):
 
        from cifar10h_analysis import run_full_analysis
        run_full_analysis(
            model        = model,
            test_loader  = dataloaders["test"],
            device       = device,
            target_layer = model.layer4[-1]    # or model.features[-1] for EfficientNet
        )
    """
    print("\n=== Running Full CIFAR-10H Analysis ===")
 
    true_probs, pred_probs, images_raw = collect_test_predictions(
        model, test_loader, device
    )
    images_norm = torch.stack([
        torch.tensor(img).permute(2, 0, 1) for img in images_raw
    ])  # raw (denorm); re-norm if needed for GradCAM
 
    # Section 1 — Entropy analysis
    H_true, H_pred = plot_entropy_analysis(true_probs, pred_probs)
 
    # Section 2 — Distribution bar plots
    plot_distribution_comparison(
        torch.tensor(images_raw).permute(0, 3, 1, 2),
        true_probs, pred_probs
    )
 
    # Section 3 — Grad-CAM (requires target_layer)
    if target_layer is not None:
        plot_gradcam_grid(
            model, target_layer, images_norm, images_raw,
            true_probs, pred_probs
        )
 
    # Section 4 — Failure cases
    plot_failure_cases(images_raw, true_probs, pred_probs)
 
    # Section 6 — Robustness
    plot_robustness_curves(model, images_norm.to(device), device)
 
    # Section 7 — Advanced visualizations
    plot_per_class_entropy_calibration(true_probs, pred_probs)
    plot_jsd_confusion_heatmap(true_probs, pred_probs)
    plot_entropy_reliability_diagram(true_probs, pred_probs)
 
    print("\n=== Analysis Complete. Check visualisations/ ===")
