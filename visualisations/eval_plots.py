"""
visualisations/eval_plots.py
==============================
All post-training evaluation visualizations.

Functions:
    plot_entropy_scatter          – true vs predicted entropy scatter
    plot_metrics_comparison_bar   – grouped bar chart across loss functions
    plot_metrics_table            – print-ready table + PNG heatmap
    plot_distribution_examples    – per-image p vs q bar charts
    plot_precision_at_k           – Precision@K for K=100,200,500
    plot_per_class_kl             – per-class KL breakdown
    plot_robustness_curves        – OOD noise / annotator subsampling

WHERE TO CALL:
    All functions accept numpy arrays or torch tensors (N, 10).
    Typical call site: evaluation/eval.py — after run_inference() returns.

    from visualisations.eval_plots import (
        plot_entropy_scatter, plot_metrics_comparison_bar,
        plot_distribution_examples, plot_precision_at_k,
        plot_per_class_kl, plot_robustness_curves,
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from scipy.stats import pearsonr, spearmanr

os.makedirs("outputs/eval", exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",      "frog",       "horse","ship","truck",
]


def _to_numpy(x):
    """Accept torch.Tensor or np.ndarray, return np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Entropy scatter  (required: Section 6.1 Metric 2)
# ─────────────────────────────────────────────────────────────────────────────
def plot_entropy_scatter(
    true_p,
    pred_q,
    label: str = "",
    save_path: str = "outputs/eval/entropy_scatter.png",
) -> None:
    """
    Scatter plot of true vs predicted entropy with Pearson/Spearman annotations.

    WHERE TO CALL:
        plot_entropy_scatter(true_p, pred_q, label="KL Linear")
    """
    p, q   = _to_numpy(true_p), _to_numpy(pred_q)
    H_true = _entropy(p)
    H_pred = _entropy(q)
    r_p, _ = pearsonr(H_true, H_pred)
    r_s, _ = spearmanr(H_true, H_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(H_true, H_pred, alpha=0.25, s=7, color="steelblue", label="Test images")
    lim = max(H_true.max(), H_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("True Entropy H(p)  [bits]")
    ax.set_ylabel("Predicted Entropy H(q)  [bits]")
    ax.set_title(
        f"True vs Predicted Entropy{f'  ({label})' if label else ''}\n"
        f"Pearson r = {r_p:.3f}   Spearman ρ = {r_s:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grouped bar chart across loss functions  (required: Section 6.1 Metric 4)
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison_bar(
    results_dict: dict,
    metrics_to_show: list[str] | None = None,
    save_path: str = "outputs/eval/metrics_comparison.png",
) -> None:
    """
    Grouped bar chart comparing metrics across multiple models / loss functions.

    Args:
        results_dict  : { 'KL Linear': {'kl_divergence': ..., 'jsd': ..., ...}, ... }
        metrics_to_show : subset of metric keys to display; defaults to main 5
        save_path     : output PNG

    WHERE TO CALL (from eval.py):
        plot_metrics_comparison_bar(
            {name: res["metrics"] for name, res in all_results.items()}
        )
    """
    if metrics_to_show is None:
        metrics_to_show = [
            "kl_divergence", "jsd", "cosine_sim", "pearson_r", "spearman_r"
        ]

    models = list(results_dict.keys())
    x      = np.arange(len(metrics_to_show))
    width  = 0.75 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(models):
        vals   = [results_dict[model_name].get(m, 0) for m in metrics_to_show]
        offset = (i - len(models) / 2) * width + width / 2
        bars   = ax.bar(x + offset, vals, width, label=model_name, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics_to_show], fontsize=9
    )
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Loss Functions")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Metrics heatmap table  (nice for the report)
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_table(
    results_dict: dict,
    save_path: str = "outputs/eval/metrics_table.png",
) -> None:
    """
    Render a colored heatmap table of all models × all metrics.

    WHERE TO CALL:
        plot_metrics_table({name: res["metrics"] for name, res in all_results.items()})
    """
    import matplotlib.colors as mcolors

    metric_order = [
        "kl_divergence", "jsd", "cosine_sim",
        "pearson_r", "spearman_r",
        "precision@100", "precision@200", "precision@500",
    ]
    lower_better = {"kl_divergence", "jsd"}

    models  = list(results_dict.keys())
    headers = metric_order

    data = np.array([
        [results_dict[m].get(h, np.nan) for h in headers]
        for m in models
    ])

    # Normalize each column for color (0=worst, 1=best)
    norm_data = np.zeros_like(data)
    for col_i, h in enumerate(headers):
        col = data[:, col_i]
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi == lo:
            norm_data[:, col_i] = 0.5
        else:
            n = (col - lo) / (hi - lo)
            norm_data[:, col_i] = (1 - n) if h in lower_better else n

    fig, ax = plt.subplots(figsize=(len(headers) * 1.5, len(models) * 0.7 + 1.2))
    ax.axis("off")
    cmap = plt.get_cmap("RdYlGn")

    cell_colors = [[cmap(norm_data[r, c]) for c in range(len(headers))]
                   for r in range(len(models))]
    cell_text   = [[f"{data[r, c]:.4f}" for c in range(len(headers))]
                   for r in range(len(models))]

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=models,
        colLabels=[h.replace("_", "\n") for h in headers],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    ax.set_title("Full Metrics Table (green = better)", fontsize=11, pad=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-image distribution comparison  (required: Section 5.2 & 6 qualitative)
# ─────────────────────────────────────────────────────────────────────────────
def plot_distribution_examples(
    images,
    true_probs,
    pred_probs,
    n_examples: int = 6,
    save_path: str = "outputs/eval/distribution_examples.png",
) -> None:
    """
    For n_examples//2 low-entropy and n_examples//2 high-entropy images, show:
      col 0: image
      col 1: side-by-side bar chart p(y|x) vs q(y|x)
      col 2: error  q(y|x) − p(y|x)

    Args:
        images     : torch.Tensor (N, 3, 32, 32) or np.ndarray, un-normalized
        true_probs : (N, 10)
        pred_probs : (N, 10)
        n_examples : total examples (half low, half high)
        save_path  : output PNG

    WHERE TO CALL:
        plot_distribution_examples(all_images, true_p, pred_q, n_examples=6)
    """
    imgs  = _to_numpy(images)     # (N, 3, 32, 32)
    p     = _to_numpy(true_probs)
    q     = _to_numpy(pred_probs)
    H     = _entropy(p)

    half      = n_examples // 2
    sorted_ix = np.argsort(H)
    low_ix    = sorted_ix[:half]
    high_ix   = sorted_ix[-half:]
    selected  = list(low_ix) + list(high_ix)
    labels    = ["Low Entropy"] * half + ["High Entropy"] * half

    fig, axes = plt.subplots(n_examples, 3, figsize=(11, 2.9 * n_examples))
    fig.suptitle("True (p) vs Predicted (q) Distributions", fontsize=13)
    x_pos  = np.arange(10)
    bwidth = 0.36

    for row, (idx, lbl) in enumerate(zip(selected, labels)):
        img_ax, bar_ax, diff_ax = axes[row]

        # Image
        img = imgs[idx].transpose(1, 2, 0)           # (H, W, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        img_ax.set_title(f"{lbl}\nH_true = {H[idx]:.2f} bits", fontsize=8)
        img_ax.axis("off")

        # Bar chart
        bar_ax.bar(x_pos - bwidth / 2, p[idx], bwidth,
                   color="#2196F3", alpha=0.85, label="p(y|x) true")
        bar_ax.bar(x_pos + bwidth / 2, q[idx], bwidth,
                   color="#FF5722", alpha=0.85, label="q(y|x) pred")
        bar_ax.set_xticks(x_pos)
        bar_ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=7)
        bar_ax.set_ylim(0, 1.05)
        bar_ax.set_ylabel("Probability")
        bar_ax.legend(fontsize=7)
        bar_ax.set_title("p vs q", fontsize=8)

        # Difference
        diff = q[idx] - p[idx]
        cols = ["#4CAF50" if d >= 0 else "#F44336" for d in diff]
        diff_ax.bar(x_pos, diff, color=cols, alpha=0.85)
        diff_ax.axhline(0, color="black", linewidth=0.8)
        diff_ax.set_xticks(x_pos)
        diff_ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=7)
        diff_ax.set_ylabel("q − p")
        diff_ax.set_title("Prediction Error per Class", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Precision@K bar chart  (required: Section 6.1 Metric 3)
# ─────────────────────────────────────────────────────────────────────────────
def plot_precision_at_k(
    results_dict: dict,
    ks: list[int] | None = None,
    save_path: str = "outputs/eval/precision_at_k.png",
) -> None:
    """
    Grouped bar chart of Precision@K for K = 100, 200, 500 across models.

    Args:
        results_dict : { model_name: {'precision@100': ..., ...}, ... }
        ks           : list of K values to plot

    WHERE TO CALL:
        plot_precision_at_k({name: res["metrics"] for name, res in all_results.items()})
    """
    if ks is None:
        ks = [100, 200, 500]

    models = list(results_dict.keys())
    x      = np.arange(len(ks))
    width  = 0.7 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_name in enumerate(models):
        vals   = [results_dict[model_name].get(f"precision@{k}", 0) for k in ks]
        offset = (i - len(models) / 2) * width + width / 2
        bars   = ax.bar(x + offset, vals, width, label=model_name, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in ks], fontsize=10)
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision@K — Identifying High-Disagreement Images")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Per-class KL breakdown  (bonus / robustness)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_kl(
    per_class_kl_dict: dict,
    label: str = "",
    save_path: str = "outputs/eval/per_class_kl.png",
) -> None:
    """
    Horizontal bar chart of per-class KL divergence.

    Args:
        per_class_kl_dict : { 'cat': 0.32, 'dog': 0.28, ... }
                             as returned by evaluation.robustness.per_class_kl()

    WHERE TO CALL:
        from evaluation.robustness import per_class_kl
        pkl = per_class_kl(true_p, pred_q, hard_labels)
        plot_per_class_kl(pkl, label="KL Linear")
    """
    classes = list(per_class_kl_dict.keys())
    values  = list(per_class_kl_dict.values())
    sorted_pairs = sorted(zip(values, classes), reverse=True)
    values, classes = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(classes)))
    bars   = ax.barh(classes, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean KL Divergence ↓")
    ax.set_title(f"Per-Class KL Divergence{f'  ({label})' if label else ''}")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Robustness curves  (OOD noise / annotator subsampling)
# ─────────────────────────────────────────────────────────────────────────────
def plot_robustness_curves(
    ood_results: dict | None = None,
    subsampling_results: dict | None = None,
    save_path: str = "outputs/eval/robustness.png",
) -> None:
    """
    Plot OOD noise robustness and/or annotator subsampling robustness.

    Args:
        ood_results         : { 'sigma=0.05': mean_entropy, ... }
                              from evaluation.robustness.ood_corruption_check()
        subsampling_results : { 'k=5': pearson_r, ... }
                              from evaluation.robustness.annotator_subsampling_check()

    WHERE TO CALL:
        ood = ood_corruption_check(model, test_images)
        sub = annotator_subsampling_check(raw_counts, pred_q)
        plot_robustness_curves(ood_results=ood, subsampling_results=sub)
    """
    n_plots = sum([ood_results is not None, subsampling_results is not None])
    if n_plots == 0:
        print("[eval_plots] plot_robustness_curves: nothing to plot.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    if ood_results is not None:
        labels = list(ood_results.keys())
        values = list(ood_results.values())
        sigmas = [float(k.split("=")[1]) for k in labels]
        axes[ax_idx].plot(sigmas, values, marker="o", color="#E91E63", linewidth=2)
        axes[ax_idx].set_xlabel("Gaussian noise σ")
        axes[ax_idx].set_ylabel("Mean Predicted Entropy (bits)")
        axes[ax_idx].set_title("OOD Robustness: Predicted Entropy under Noise")
        axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    if subsampling_results is not None:
        ks     = [int(k.split("=")[1]) for k in subsampling_results]
        rs     = list(subsampling_results.values())
        axes[ax_idx].plot(ks, rs, marker="s", color="#9C27B0", linewidth=2)
        axes[ax_idx].set_xlabel("Number of Annotators (k)")
        axes[ax_idx].set_ylabel("Pearson r  (entropy)")
        axes[ax_idx].set_title("Robustness to Annotator Subsampling")
        axes[ax_idx].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test with dummy data
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    N  = 200
    p  = torch.softmax(torch.randn(N, 10), dim=1)
    q  = torch.softmax(torch.randn(N, 10), dim=1)

    # Fake images: (N, 3, 32, 32) uniform noise
    imgs = torch.rand(N, 3, 32, 32)

    dummy_metrics = {
        "kl_divergence":  0.42, "jsd": 0.19, "cosine_sim": 0.71,
        "pearson_r": 0.61, "spearman_r": 0.59,
        "precision@100": 0.55, "precision@200": 0.52, "precision@500": 0.48,
    }
    results = {"KL Linear": dummy_metrics,
               "JSD MLP": {**dummy_metrics, "kl_divergence": 0.38, "jsd": 0.16,
                            "cosine_sim": 0.75, "pearson_r": 0.65}}

    os.makedirs("outputs/eval", exist_ok=True)
    plot_entropy_scatter(p, q, label="Dummy", save_path="outputs/eval/test_scatter.png")
    plot_metrics_comparison_bar(results,         save_path="outputs/eval/test_bar.png")
    plot_metrics_table(results,                  save_path="outputs/eval/test_table.png")
    plot_distribution_examples(imgs, p, q,
                                n_examples=4,   save_path="outputs/eval/test_dist.png")
    plot_precision_at_k(results,                save_path="outputs/eval/test_pak.png")
    plot_per_class_kl({"cat": 0.32, "dog": 0.28, "bird": 0.45,
                        "airplane": 0.12, "ship": 0.15, "automobile": 0.18,
                        "deer": 0.22, "frog": 0.30, "horse": 0.25, "truck": 0.20},
                      label="Dummy",            save_path="outputs/eval/test_pkl.png")
    plot_robustness_curves(
        ood_results={"sigma=0.05": 0.31, "sigma=0.1": 0.45,
                     "sigma=0.2": 0.71, "sigma=0.3": 0.95, "sigma=0.5": 1.40},
        subsampling_results={"k=5": 0.48, "k=10": 0.61, "k=20": 0.73},
        save_path="outputs/eval/test_robust.png",
    )
    print("Self-test complete — check outputs/eval/")
