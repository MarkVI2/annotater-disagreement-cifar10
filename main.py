#!/usr/bin/env python3
"""
One-command pipeline: data → training → evaluation → visualisations → explainability.
Usage:   python main.py [--skip-data] [--skip-train] [--skip-eval] [--skip-post]
"""

import argparse
import subprocess
import sys
import os
import json
import glob
from pathlib import Path


def run_data_pipeline():
    print("\n" + "="*60)
    print("STAGE 1: DATA PIPELINE")
    print("="*60)
    result = subprocess.run([sys.executable, "main_data.py"], capture_output=False)
    if result.returncode != 0:
        print("Data pipeline failed. Exiting.")
        sys.exit(1)


def run_training():
    print("\n" + "="*60)
    print("STAGE 2: TRAINING ALL EXPERIMENTS")
    print("="*60)
    result = subprocess.run([sys.executable, "experiments/run.py"], capture_output=False)
    if result.returncode != 0:
        print("Training failed. Check experiments/run.py for errors.")
        sys.exit(1)


def run_evaluation():
    print("\n" + "="*60)
    print("STAGE 3: EVALUATION")
    print("="*60)
    # The new evaluation/eval.py produces per‑model plots AND aggregated plots
    result = subprocess.run([sys.executable, "evaluation/eval.py"], capture_output=False)
    if result.returncode != 0:
        print("Evaluation failed.")
        sys.exit(1)


def run_post_evaluation():
    """
    STAGE 4: Post‑evaluation visualisations and explainability.
    Produces:
      - Architecture diagram + parameter table (visualisations/model_diagram.py)
      - Overlay of all training curves (visualisations/training_curves.py)
      - Grad‑CAM analysis (explainability/grad_cam.py CLI)
      - Composite high‑entropy disagreement images (manual_disagreement.py)
    """
    print("\n" + "="*60)
    print("STAGE 4: POST‑EVALUATION VISUALISATIONS")
    print("="*60)

    # 4.1 Architecture diagram & parameter table
    print("\n--- Architecture diagram ---")
    subprocess.run([sys.executable, "visualisations/model_diagram.py"], check=False)

    # 4.2 Training curve overlay (all experiments)
    print("\n--- Training curve overlay ---")
    from visualisations.training_curves import plot_all_runs
    plot_all_runs("outputs/logs", save_path="outputs/plots/all_val_curves.png")
    # Also generate individual curves for a couple of key experiments if needed
    # (the training script already does this per experiment)

    # 4.3 Grad‑CAM analysis (using the best model)
    print("\n--- Grad‑CAM analysis ---")
    best_ckpt = find_best_checkpoint()  # helper to pick the checkpoint with lowest validation loss
    if best_ckpt:
        # Determine head type and temp from the checkpoint path
        name = Path(best_ckpt).stem.replace("_best", "")
        head = "mlp" if "mlp" in name else "linear"
        subprocess.run(
            [sys.executable, "explainability/grad_cam.py",
             "--checkpoint", best_ckpt,
             "--head", head,
             "--n_low", "4", "--n_high", "4",
             "--output_dir", "outputs/explainability"],
            check=False,
        )

    # 4.4 Manual disagreement composite images (30 highest entropy images)
    print("\n--- Composite high‑entropy images ---")
    subprocess.run([sys.executable, "analysis/manual_disagreement.py"], check=False)

    print("\n===== ALL STAGES COMPLETE =====")


def find_best_checkpoint():
    """Scan logs for the experiment with the lowest validation loss and return the best.pth path."""
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
        except Exception:
            continue
    if best_path is None:
        # Fallback: use one checkpoint that exists
        for p in checkpoints_dir.rglob("*_best.pth"):
            best_path = p
            break
    return str(best_path) if best_path else None


def main():
    parser = argparse.ArgumentParser(description="Full pipeline for CIFAR‑10H disagreement project")
    parser.add_argument("--skip-data", action="store_true", help="Skip data pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-post", action="store_true", help="Skip post‑evaluation visualisations")
    args = parser.parse_args()

    if not args.skip_data:
        run_data_pipeline()
    if not args.skip_train:
        run_training()
    if not args.skip_eval:
        run_evaluation()
    if not args.skip_post:
        run_post_evaluation()

    print("\nDONE. All outputs are in outputs/.")


if __name__ == "__main__":
    main()