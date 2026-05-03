"""
explainability/grad_cam.py
============================
Full Grad-CAM implementation for the soft-label disagreement model.

This file IS your explainability tab deliverable.
It imports GradCAM from visualisations/grad_cam_plots.py (DRY principle)
and adds the standalone run_grad_cam_analysis() entry-point that produces
all Grad-CAM output figures in one call.

USAGE (standalone):
    python explainability/grad_cam.py  --checkpoint outputs/checkpoints/kl_linear/best.pth
                                       --head linear
                                       --n_low 4 --n_high 4

USAGE (imported):
    from explainability.grad_cam import GradCAM, run_grad_cam_analysis
    gc = GradCAM(model, model.layer4[-1])
    run_grad_cam_analysis(gc, model, test_loader)
"""

# ── Re-export GradCAM so the rest of the codebase can do
#    `from explainability.grad_cam import GradCAM`
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualisations.grad_cam_plots import (
    GradCAM,
    plot_grad_cam_grid,
    plot_entropy_vs_gradcam,
    overlay_grad_cam,
)

__all__ = [
    "GradCAM",
    "plot_grad_cam_grid",
    "plot_entropy_vs_gradcam",
    "overlay_grad_cam",
    "run_grad_cam_analysis",
]

import os
import torch
import argparse


def run_grad_cam_analysis(
    grad_cam: GradCAM,
    model: torch.nn.Module,
    test_loader,
    n_low: int = 4,
    n_high: int = 4,
    strategy: str = "top_pred",
    output_dir: str = "outputs/explainability",
    max_batches: int = 4,           # limit how many batches to load (saves time)
) -> None:
    """
    Full Grad-CAM analysis:
      1. Collects test images + predictions (up to max_batches × batch_size)
      2. Generates  outputs/explainability/grad_cam_grid.png
      3. Generates  outputs/explainability/entropy_vs_gradcam.png

    Args:
        grad_cam    : GradCAM instance already hooked to a model layer
        model       : the eval-mode model
        test_loader : DataLoader (images, soft_labels, hard_labels)
        n_low       : number of low-entropy examples
        n_high      : number of high-entropy examples
        strategy    : "top_pred" | "top_true" | "entropy_weighted"
        output_dir  : directory for output PNGs
        max_batches : cap on batches to process (keep GPU memory safe)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    all_imgs, all_true, all_pred = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_i, (imgs, soft_labels, _) in enumerate(test_loader):
            if batch_i >= max_batches:
                break
            imgs        = imgs.to(device)
            soft_labels = soft_labels.to(device)
            preds       = model(imgs)
            all_imgs.append(imgs.cpu())
            all_true.append(soft_labels.cpu())
            all_pred.append(preds.cpu())

    images    = torch.cat(all_imgs,  dim=0)
    true_probs = torch.cat(all_true, dim=0)
    pred_probs = torch.cat(all_pred, dim=0)

    print(f"[grad_cam] Collected {len(images)} images for analysis.")

    plot_grad_cam_grid(
        grad_cam, images, true_probs, pred_probs,
        n_low=n_low, n_high=n_high, strategy=strategy,
        save_path=os.path.join(output_dir, "grad_cam_grid.png"),
    )

    plot_entropy_vs_gradcam(
        grad_cam, images, true_probs, pred_probs,
        strategy="entropy_weighted",
        n_per_group=min(4, n_low, n_high),
        save_path=os.path.join(output_dir, "entropy_vs_gradcam.png"),
    )

    print(f"[grad_cam] All outputs saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--head",       default="linear", choices=["linear", "mlp"])
    parser.add_argument("--n_low",      type=int, default=4)
    parser.add_argument("--n_high",     type=int, default=4)
    parser.add_argument("--strategy",   default="top_pred",
                        choices=["top_pred", "top_true", "entropy_weighted"])
    parser.add_argument("--output_dir", default="outputs/explainability")
    parser.add_argument("--max_batches",type=int, default=4)
    args = parser.parse_args()

    # ── Lazy imports (avoid loading torch until needed)
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from models.cifar_resnet import build_soft_label_model
    from data.pipeline import load_full_dataset, create_splits, create_dataloaders
    from utils.device import get_device

    device = get_device()

    # Load model
    model = build_soft_label_model(head_type=args.head, pretrained_type='random')
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # Target layer: last residual block of ResNet-18
    target_layer = model.layer4[-1]
    gc = GradCAM(model, target_layer)

    # Data
    dataset     = load_full_dataset()
    splits      = create_splits(dataset)
    loaders     = create_dataloaders(splits)
    test_loader = loaders["test"]

    run_grad_cam_analysis(
        gc, model, test_loader,
        n_low=args.n_low, n_high=args.n_high,
        strategy=args.strategy,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
    )
