"""
visualisations/grad_cam_plots.py
==================================
Grad-CAM visualization helpers.

The GradCAM class lives in  explainability/grad_cam.py  (see below).
This file contains the *plotting* functions that sit in the visualisations tab:
    - overlay_grad_cam          : produce a single heatmap overlay image
    - plot_grad_cam_grid        : grid of Grad-CAM panels for multiple images
    - plot_entropy_vs_gradcam   : compare low vs high entropy CAM patterns

Additionally, this file contains the complete GradCAM class so that your
explainability/grad_cam.py  can simply do:
    from visualisations.grad_cam_plots import GradCAM
or you can copy the class there directly.

WHERE TO CALL:
    from visualisations.grad_cam_plots import GradCAM, plot_grad_cam_grid
    grad_cam = GradCAM(model, target_layer=model.layer4[-1])
    plot_grad_cam_grid(grad_cam, test_images, true_probs, pred_probs,
                       n_low=3, n_high=3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2               # pip install opencv-python-headless

os.makedirs("outputs/explainability", exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse","ship","truck",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  GradCAM class  (also lives here so visualisations/ is self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    """
    Grad-CAM for soft-label distribution models (ResNet / EfficientNet).

    Strategies for choosing the target class in a distribution model:
        "top_pred"        → gradient w.r.t. argmax q(y|x)   — most confident class
        "top_true"        → gradient w.r.t. argmax p(y|x)   — ground-truth dominant
        "entropy_weighted"→ weighted sum of class CAMs by –p·log(p)

    USAGE:
        grad_cam = GradCAM(model, target_layer=model.layer4[-1])   # ResNet-18
        # EfficientNet: model.features[-1]
        cam = grad_cam.generate(image_tensor)       # (H, W) in [0, 1]
        overlay = grad_cam.overlay(raw_image_np, cam)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(fwd_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(bwd_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image: torch.Tensor,             # (1, 3, H, W) — normalized
        target_class: int | None = None,
        strategy: str = "top_pred",      # "top_pred" | "top_true" | "entropy_weighted"
        true_probs: torch.Tensor | None = None,   # needed for "top_true" / "entropy_weighted"
    ) -> np.ndarray:
        """
        Returns a (H, W) heatmap array, values in [0, 1].
        """
        self.model.eval()
        image = image.clone().requires_grad_(True)

        # Forward
        logits = self.model(image)          # (1, 10) — already softmax'd
        probs  = logits                     # model outputs probabilities

        # Pick target class
        if strategy == "top_pred" or target_class is not None:
            cls = target_class if target_class is not None else int(probs.argmax(dim=1))
            score = probs[0, cls]
            self.model.zero_grad()
            score.backward(retain_graph=True)
            cam = self._cam_from_grads()

        elif strategy == "top_true":
            assert true_probs is not None, "Pass true_probs for strategy='top_true'"
            cls   = int(true_probs.argmax())
            score = probs[0, cls]
            self.model.zero_grad()
            score.backward(retain_graph=True)
            cam = self._cam_from_grads()

        elif strategy == "entropy_weighted":
            # Σ_y  –p(y)·log(p(y))  *  ∂logit_y/∂A
            p   = probs[0].detach()
            eps = 1e-12
            weights = -(p * (p + eps).log())          # (10,)
            cam = np.zeros(
                (self._activations.shape[-2], self._activations.shape[-1]),
                dtype=np.float32,
            )
            for cls_i in range(10):
                self.model.zero_grad()
                probs[0, cls_i].backward(retain_graph=True)
                cam += weights[cls_i].item() * self._cam_from_grads()
            cam = np.clip(cam, 0, None)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        return cam

    def _cam_from_grads(self) -> np.ndarray:
        """Global-average-pool gradients → weighted activation sum → ReLU → normalize."""
        grads = self._gradients[0]       # (C, H, W)
        acts  = self._activations[0]     # (C, H, W)
        alpha = grads.mean(dim=(1, 2))   # (C,)
        cam   = (alpha[:, None, None] * acts).sum(dim=0)
        cam   = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        raw_image: np.ndarray,      # (H, W, 3)  uint8 or float [0,1]
        cam: np.ndarray,            # (h, w) [0,1]
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Returns (H, W, 3) uint8 with the heatmap overlaid on the raw image.
        """
        if raw_image.dtype != np.uint8:
            raw_image = (np.clip(raw_image, 0, 1) * 255).astype(np.uint8)

        h, w = raw_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap_u8  = (cam_resized * 255).astype(np.uint8)
        heatmap_rgb = cv2.applyColorMap(heatmap_u8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        blended = (alpha * heatmap_rgb + (1 - alpha) * raw_image).clip(0, 255).astype(np.uint8)
        return blended


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def overlay_grad_cam(
    raw_image: np.ndarray,
    cam: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> np.ndarray:
    """
    Show (and optionally save) a single Grad-CAM overlay.
    Returns the overlay array.
    """
    overlay = GradCAM.overlay(raw_image, cam)
    if save_path or title:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(np.clip(raw_image / 255.0
                               if raw_image.max() > 1 else raw_image, 0, 1))
        axes[0].set_title("Original", fontsize=9);  axes[0].axis("off")
        axes[1].imshow(cam, cmap="jet");  axes[1].set_title("CAM heatmap", fontsize=9); axes[1].axis("off")
        axes[2].imshow(overlay);          axes[2].set_title("Overlay", fontsize=9);      axes[2].axis("off")
        if title:
            fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[grad_cam_plots] Saved → {save_path}")
        plt.close()
    return overlay


def plot_grad_cam_grid(
    grad_cam: GradCAM,
    images: torch.Tensor,       # (N, 3, 32, 32)  normalized
    true_probs: torch.Tensor,   # (N, 10)
    pred_probs: torch.Tensor,   # (N, 10)
    n_low: int = 3,
    n_high: int = 3,
    strategy: str = "top_pred",
    save_path: str = "outputs/explainability/grad_cam_grid.png",
) -> None:
    """
    Grid layout:
        rows = selected images (n_low low-entropy + n_high high-entropy)
        cols = [original | Grad-CAM | overlay | bar chart p vs q]

    WHERE TO CALL:
        After load_model() + run_inference():
            from visualisations.grad_cam_plots import GradCAM, plot_grad_cam_grid
            gc = GradCAM(model, model.layer4[-1])
            plot_grad_cam_grid(gc, test_images, true_p, pred_q)
    """
    from evaluation.metrics import compute_entropy as _cE
    import torch as _torch

    # Pick images
    H      = _cE(true_probs).numpy()
    sort_i = np.argsort(H)
    lo_idx = list(sort_i[:n_low])
    hi_idx = list(sort_i[-n_high:])
    sel    = lo_idx + hi_idx
    tags   = [f"Low  H={H[i]:.2f}" for i in lo_idx] + \
             [f"High H={H[i]:.2f}" for i in hi_idx]

    n_rows = len(sel)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.2 * n_rows))
    fig.suptitle(f"Grad-CAM Analysis  (strategy={strategy})", fontsize=12)

    x_pos  = np.arange(10)
    bwidth = 0.38

    for row, (idx, tag) in enumerate(zip(sel, tags)):
        img_t  = images[idx:idx+1]        # (1, 3, 32, 32)
        tp     = true_probs[idx]
        qp     = pred_probs[idx]

        # Grad-CAM
        cam = grad_cam.generate(
            img_t, strategy=strategy,
            true_probs=tp if "true" in strategy else None
        )

        # Un-normalise for display  (undo CIFAR-10 mean/std)
        MEAN = _torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        STD  = _torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        raw  = (img_t[0].cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        raw_u8 = (raw * 255).astype(np.uint8)
        overlay = GradCAM.overlay(raw_u8, cam)

        row_axes = axes[row] if n_rows > 1 else axes

        # Col 0: original
        row_axes[0].imshow(raw)
        row_axes[0].set_title(tag, fontsize=8)
        row_axes[0].axis("off")

        # Col 1: heatmap
        row_axes[1].imshow(cam, cmap="jet")
        row_axes[1].set_title("CAM heatmap", fontsize=8)
        row_axes[1].axis("off")

        # Col 2: overlay
        row_axes[2].imshow(overlay)
        row_axes[2].set_title("Overlay", fontsize=8)
        row_axes[2].axis("off")

        # Col 3: p vs q bar chart
        p_np = tp.numpy() if isinstance(tp, _torch.Tensor) else tp
        q_np = qp.numpy() if isinstance(qp, _torch.Tensor) else qp
        row_axes[3].bar(x_pos - bwidth/2, p_np, bwidth, color="#2196F3", alpha=0.85, label="p true")
        row_axes[3].bar(x_pos + bwidth/2, q_np, bwidth, color="#FF5722", alpha=0.85, label="q pred")
        row_axes[3].set_xticks(x_pos)
        row_axes[3].set_xticklabels([c[:3] for c in CIFAR10_CLASSES],
                                     rotation=45, ha="right", fontsize=6)
        row_axes[3].set_ylim(0, 1.05)
        row_axes[3].legend(fontsize=6)
        row_axes[3].set_title("p vs q", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grad_cam_plots] Saved → {save_path}")


def plot_entropy_vs_gradcam(
    grad_cam: GradCAM,
    images: torch.Tensor,
    true_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    strategy: str = "entropy_weighted",
    n_per_group: int = 4,
    save_path: str = "outputs/explainability/entropy_vs_gradcam.png",
) -> None:
    """
    Two-row comparison: top row = low-entropy images with their CAMs,
    bottom row = high-entropy images. Demonstrates that diffuse CAMs
    correlate with high annotator disagreement.

    WHERE TO CALL: same as plot_grad_cam_grid.
    """
    from evaluation.metrics import compute_entropy as _cE
    import torch as _torch

    H      = _cE(true_probs).numpy()
    sort_i = np.argsort(H)
    lo_idx = list(sort_i[:n_per_group])
    hi_idx = list(sort_i[-n_per_group:])

    MEAN = _torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    STD  = _torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    fig, axes = plt.subplots(4, n_per_group, figsize=(3 * n_per_group, 11))
    fig.suptitle(
        f"Grad-CAM vs Entropy  (strategy={strategy})\n"
        "Top rows: low disagreement → focused attention | Bottom rows: high disagreement → diffuse",
        fontsize=10,
    )
    row_labels = ["Low Entropy\nImage", "Low Entropy\nCAM Overlay",
                  "High Entropy\nImage", "High Entropy\nCAM Overlay"]

    for col, (lo, hi) in enumerate(zip(lo_idx, hi_idx)):
        for grp_i, idx in enumerate([lo, hi]):
            img_t = images[idx:idx+1]
            cam   = grad_cam.generate(
                img_t, strategy=strategy,
                true_probs=true_probs[idx] if "true" in strategy or "entropy" in strategy else None,
            )
            raw   = (img_t[0].cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
            raw_u8 = (raw * 255).astype(np.uint8)
            ov    = GradCAM.overlay(raw_u8, cam)

            axes[grp_i * 2,     col].imshow(raw)
            axes[grp_i * 2,     col].set_title(f"H={H[idx]:.2f}", fontsize=8)
            axes[grp_i * 2,     col].axis("off")
            axes[grp_i * 2 + 1, col].imshow(ov)
            axes[grp_i * 2 + 1, col].axis("off")

    for row_i, lbl in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(lbl, fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grad_cam_plots] Saved → {save_path}")
