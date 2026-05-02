"""
visualisations/training_curves.py
===================================
Plots training and validation loss curves from the MetricsLogger CSV files.

WHERE TO CALL:
    After training finishes (end of training/train.py main(), or in a notebook).
    Example:
        from visualisations.training_curves import plot_training_curves, plot_all_runs
        plot_training_curves("outputs/logs/kl_linear_metrics.csv", label="KL Linear")
        plot_all_runs("outputs/logs/")   # overlays all runs on one figure
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────
# 1. Single-run plot  (train loss + val loss)
# ─────────────────────────────────────────────
def plot_training_curves(
    csv_path: str,
    label: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot train_loss and val_loss from a MetricsLogger CSV file.

    Args:
        csv_path  : path to the CSV produced by MetricsLogger
        label     : experiment name shown in the title
        save_path : where to save the PNG; if None, derives from csv_path
    """
    import csv

    epochs, train_losses, val_losses = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    epochs      = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses   = np.array(val_losses)

    best_epoch = epochs[np.argmin(val_losses)]
    best_val   = val_losses.min()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, color="#2196F3", linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_losses,   color="#FF5722", linewidth=1.8, label="Val loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
               label=f"Best val @ epoch {best_epoch}  ({best_val:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss{f'  ({label})' if label else ''}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        base = os.path.splitext(csv_path)[0]
        save_path = base + "_curves.png"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[training_curves] Saved → {save_path}")


# ─────────────────────────────────────────────
# 2. Multi-run overlay (all experiments together)
# ─────────────────────────────────────────────
def plot_all_runs(
    logs_dir: str = "outputs/logs",
    save_path: str = "outputs/plots/all_val_curves.png",
) -> None:
    """
    Overlay validation loss curves for every CSV in logs_dir on a single figure.
    Useful for comparing KL / JS / Custom / EMD experiments at a glance.

    Args:
        logs_dir  : directory containing MetricsLogger CSVs
        save_path : output PNG path
    """
    import csv

    csv_files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))
    if not csv_files:
        print(f"[training_curves] No CSV files found in {logs_dir}")
        return

    colors = cm.tab10(np.linspace(0, 1, len(csv_files)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for csv_path, color in zip(csv_files, colors):
        label = os.path.splitext(os.path.basename(csv_path))[0]
        label = label.replace("_metrics", "")      # strip suffix added by MetricsLogger

        epochs, val_losses = [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_loss"]))

        ax.plot(epochs, val_losses, color=color, linewidth=1.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss — All Experiments")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[training_curves] Saved → {save_path}")


# ─────────────────────────────────────────────
# 3. Metric curve over epochs (e.g. Pearson r)
# ─────────────────────────────────────────────
def plot_metric_curve(
    epochs: list[int],
    values: list[float],
    metric_name: str = "Pearson r",
    label: str = "",
    save_path: str = "outputs/plots/metric_curve.png",
) -> None:
    """
    Plot any scalar metric tracked per-epoch during validation.

    WHERE TO CALL:
        If you compute e.g. Pearson r on val set each epoch in train.py,
        collect it in a list and call this at the end of training.

        from visualisations.training_curves import plot_metric_curve
        plot_metric_curve(epoch_list, pearson_list, metric_name="Pearson r (val)",
                          label="KL Linear", save_path="outputs/plots/pearson_kl_linear.png")
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, values, color="#4CAF50", linewidth=1.8, label=metric_name)
    ax.axhline(max(values), color="gray", linestyle="--", linewidth=0.8,
               label=f"Best = {max(values):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} over Training{f'  ({label})' if label else ''}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[training_curves] Saved → {save_path}")


# ─────────────────────────────────────────────
# Quick self-test with dummy data
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, csv, os

    # Create a dummy CSV to test single-run plot
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for ep in range(1, 31):
            writer.writerow({
                "epoch":      ep,
                "train_loss": 0.9 * np.exp(-ep / 15) + np.random.uniform(0, 0.02),
                "val_loss":   0.95 * np.exp(-ep / 15) + np.random.uniform(0, 0.03),
            })
        tmp = f.name

    os.makedirs("outputs/plots", exist_ok=True)
    plot_training_curves(tmp, label="Dummy KL Linear",
                         save_path="outputs/plots/test_training_curves.png")
    plot_metric_curve(
        list(range(1, 31)),
        [0.3 + 0.5 * (1 - np.exp(-i / 10)) + np.random.uniform(0, 0.02) for i in range(30)],
        metric_name="Pearson r (val)",
        label="Dummy",
        save_path="outputs/plots/test_metric_curve.png",
    )
    print("Self-test complete — check outputs/plots/")
    os.unlink(tmp)

