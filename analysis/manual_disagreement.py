#!/usr/bin/env python3
"""
Generates composite images for high‑entropy samples:
  Left side  : image + hard label
  Right side : image overlaid with entropy‑weighted Grad‑CAM
  Below      : human soft‑label bar chart

Outputs saved to outputs/manual_disagreement/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cifar_resnet import build_soft_label_model
from data.pipeline import load_full_dataset, create_splits, create_dataloaders
from utils.device import get_device
from visualisations.grad_cam_plots import GradCAM
from analysis.cifar10h_analysis import compute_shannon_entropy

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def find_best_checkpoint():
    """Re‑use logic from main.py to find the best model."""
    import pandas as pd
    best_loss = float("inf")
    best_path = None
    logs_dir = Path("outputs/logs")
    checkpoints_dir = Path("outputs/checkpoints")
    for csv_path in logs_dir.rglob("*_metrics.csv"):
        try:
            df = pd.read_csv(csv_path)
            min_val = df["val_loss"].min()
            exp_name = csv_path.stem.replace("_metrics", "")
            ckpt_path = checkpoints_dir / exp_name / (exp_name + "_best.pth")
            if min_val < best_loss and ckpt_path.exists():
                best_loss = min_val
                best_path = ckpt_path
        except:
            continue
    if best_path is None:
        for p in checkpoints_dir.rglob("*_best.pth"):
            best_path = p
            break
    return str(best_path) if best_path else None

def main():
    device = get_device()
    ckpt_path = find_best_checkpoint()
    if not ckpt_path:
        print("No checkpoint found. Run training first.")
        return

    # Determine architecture from filename
    exp_name = Path(ckpt_path).stem.replace("_best", "")
    head = "mlp" if "mlp" in exp_name else "linear"
    use_temp = "_temp" in exp_name

    model = build_soft_label_model(
        head_type=head,
        pretrained_type="random",
        use_temperature=use_temp,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    # Data
    dataset = load_full_dataset()
    splits = create_splits(dataset)
    loaders = create_dataloaders(splits)
    test_loader = loaders["test"]

    # Gather all test predictions
    all_images = []
    all_true = []
    with torch.no_grad():
        for images, soft_labels, _ in test_loader:
            images = images.to(device)
            preds = model(images)
            all_images.append(images.cpu())
            all_true.append(soft_labels.cpu())

    images = torch.cat(all_images, dim=0)      # (N, 3, 32, 32) normalized
    true_probs = torch.cat(all_true, dim=0)    # (N, 10)

    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    images_raw = (images * std + mean).clamp(0, 1)

    # Select 30 highest entropy images
    H_true = compute_shannon_entropy(true_probs.numpy())
    high_idx = np.argsort(H_true)[-30:]  # 30 most ambiguous

    # Grad‑CAM setup
    target_layer = model.backbone.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    os.makedirs("outputs/manual_disagreement", exist_ok=True)

    for i, idx in enumerate(high_idx):
        img_tensor = images[idx].unsqueeze(0).to(device)   # normalized
        raw = images_raw[idx].permute(1, 2, 0).numpy()
        raw_u8 = (raw * 255).astype(np.uint8)
        true_dist = true_probs[idx].numpy()
        hard_label = int(np.argmax(true_dist))

        # Grad‑CAM (entropy‑weighted)
        cam = grad_cam.generate(img_tensor, strategy="entropy_weighted")
        overlay = GradCAM.overlay(raw_u8, cam)

        # Build composite figure
        fig = plt.figure(figsize=(10, 4))
        # Left: original image + hard label
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(raw)
        ax1.set_title(f"Image (True class: {CIFAR10_CLASSES[hard_label]})", fontsize=9)
        ax1.axis("off")

        # Middle: Grad‑CAM overlay
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(overlay)
        ax2.set_title("Grad‑CAM (entropy‑weighted)", fontsize=9)
        ax2.axis("off")

        # Right: soft‑label bar chart
        ax3 = fig.add_subplot(1, 3, 3)
        x = np.arange(10)
        ax3.bar(x, true_dist, color="coral", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([c[:3] for c in CIFAR10_CLASSES], rotation=45, ha="right", fontsize=7)
        ax3.set_ylim(0, 1.05)
        ax3.set_title(f"H = {H_true[idx]:.2f}", fontsize=9)

        plt.suptitle(f"High‑entropy example #{i+1} (index {idx})", fontsize=11)
        plt.tight_layout()
        save_path = f"outputs/manual_disagreement/high_entropy_{i+1:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

    print(f"Done. {len(high_idx)} composite images saved to outputs/manual_disagreement/")

if __name__ == "__main__":
    main()