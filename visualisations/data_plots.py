"""
visualisations/data_plots.py
==============================
Thin wrappers that re-export the four required data-stage plots so your
team can call them from this tab without touching data/visualization.py.

Also adds one extra: side-by-side true vs predicted entropy histogram,
which belongs to the visualization tab (not just the data tab).

WHERE TO CALL:
    from visualisations.data_plots import (
        plot_entropy_histogram,
        plot_class_entropy_bar,
        plot_annotator_confusion_matrix,
        plot_extreme_examples,
        plot_entropy_comparison,
    )

    # After main_data.py has run:
    generate_all_data_visualizations(dataset, statistics, save_dir="outputs/plots")
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── re-export the four canonical data plots ─────────────────────────────────
from data.visualization import (
    plot_entropy_histogram,
    plot_class_entropy_bar,
    plot_annotator_confusion_matrix,
    plot_extreme_examples,
    generate_all_data_visualizations,
)

__all__ = [
    "plot_entropy_histogram",
    "plot_class_entropy_bar",
    "plot_annotator_confusion_matrix",
    "plot_extreme_examples",
    "generate_all_data_visualizations",
    "plot_entropy_comparison",
    "plot_soft_label_overview",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────────────────────
# Extra: True vs Predicted entropy histogram side-by-side
# ─────────────────────────────────────────────────────────────────────────────
def plot_entropy_comparison(
    true_entropies: np.ndarray,
    pred_entropies: np.ndarray,
    save_path: str = "outputs/plots/entropy_comparison.png",
) -> None:
    """
    Side-by-side histograms of true and predicted entropy.

    WHERE TO CALL:
        After running evaluation on the test set.
        true_entropies / pred_entropies are (N,) numpy arrays.

        from visualisations.data_plots import plot_entropy_comparison
        plot_entropy_comparison(H_true, H_pred)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Shannon Entropy Distribution: True vs Predicted", fontsize=13)

    for ax, vals, color, title in zip(
        axes,
        [true_entropies, pred_entropies],
        ["#2196F3", "#FF5722"],
        ["True Entropy  H(p)", "Predicted Entropy  H(q)"],
    ):
        ax.hist(vals, bins=50, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean = {vals.mean():.3f}")
        ax.axvline(np.log2(10), color="gray", linestyle=":", linewidth=1,
                   label=f"max = {np.log2(10):.3f}")
        ax.set_xlabel("Shannon Entropy (bits)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[data_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra: Soft-label distribution overview for a sample of images
# ─────────────────────────────────────────────────────────────────────────────
def plot_soft_label_overview(
    soft_labels: np.ndarray,
    n_samples: int = 12,
    save_path: str = "outputs/plots/soft_label_overview.png",
    seed: int = 42,
) -> None:
    """
    Show bar-chart soft labels for n_samples randomly chosen images.
    Useful as a sanity-check panel in the report.

    WHERE TO CALL:
        dataset = load_full_dataset()
        soft = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
        plot_soft_label_overview(soft)
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(soft_labels), size=n_samples, replace=False)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = axes.flatten()

    for ax_i, idx in enumerate(indices):
        sl = soft_labels[idx]
        dominant = int(np.argmax(sl))
        ax = axes[ax_i]
        colors = ["#FF5722" if j == dominant else "#90CAF9" for j in range(10)]
        ax.bar(range(10), sl, color=colors, edgecolor="white")
        ax.set_xticks(range(10))
        ax.set_xticklabels(
            [c[:3] for c in CIFAR10_CLASSES], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"#{idx} — {CIFAR10_CLASSES[dominant]}", fontsize=8)
        ax.set_ylabel("p", fontsize=8)

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle("Soft Label Overview (random sample)", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[data_plots] Saved → {save_path}")
