import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from scipy.stats import pearsonr, spearmanr

os.makedirs("outputs/eval", exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def plot_entropy_scatter(true_p, pred_q, label: str = "", save_path: str = "outputs/eval/entropy_scatter.png"):
    p, q = _to_numpy(true_p), _to_numpy(pred_q)
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
    ax.set_title(f"True vs Predicted Entropy{f'  ({label})' if label else ''}\n"
                 f"Pearson r = {r_p:.3f}   Spearman ρ = {r_s:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ----  Clutter free comparison bar chart ----
def plot_metrics_comparison_bar(results_dict: dict,
                                metrics_to_show: list[str] | None = None,
                                save_path: str = "outputs/eval/metrics_comparison.png"):
    if metrics_to_show is None:
        metrics_to_show = ["kl_divergence", "jsd", "cosine_sim", "pearson_r", "spearman_r"]
    models = list(results_dict.keys())
    x = np.arange(len(metrics_to_show))
    width = 0.7 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        vals = [results_dict[model_name].get(m, 0) for m in metrics_to_show]
        offset = (i - len(models) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=model_name, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_show], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Loss Functions")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ---- Clutter free precision@k ----
def plot_precision_at_k(results_dict: dict,
                        ks: list[int] | None = None,
                        save_path: str = "outputs/eval/precision_at_k.png"):
    if ks is None:
        ks = [100, 200, 500]
    models = list(results_dict.keys())
    x = np.arange(len(ks))
    width = 0.6 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_name in enumerate(models):
        vals = [results_dict[model_name].get(f"precision@{k}", 0) for k in ks]
        offset = (i - len(models) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=model_name, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in ks], fontsize=10)
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision@K — Identifying High-Disagreement Images")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ---- Metrics table with scaling ----
def plot_metrics_table(results_dict: dict,
                       save_path: str = "outputs/eval/metrics_table.png",
                       max_rows: int = 25):
    import matplotlib.colors as mcolors

    metric_order = [
        "kl_divergence", "jsd", "cosine_sim",
        "pearson_r", "spearman_r",
        "precision@100", "precision@200", "precision@500",
    ]
    lower_better = {"kl_divergence", "jsd"}

    models = sorted(results_dict.keys())
    if len(models) > max_rows:
        models = sorted(models, key=lambda n: results_dict[n].get('kl_divergence', 999))[:max_rows]
        print(f"[metrics_table] Showing top {max_rows} models by KL.")

    headers = metric_order
    data = np.array([[results_dict[m].get(h, np.nan) for h in headers] for m in models])

    norm_data = np.zeros_like(data)
    for col_i, h in enumerate(headers):
        col = data[:, col_i]
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi == lo:
            norm_data[:, col_i] = 0.5
        else:
            n = (col - lo) / (hi - lo)
            norm_data[:, col_i] = (1 - n) if h in lower_better else n

    fig, ax = plt.subplots(figsize=(len(headers) * 1.5, len(models) * 0.5 + 1.5))
    ax.axis("off")
    cmap = plt.get_cmap("RdYlGn")
    cell_colours = [[cmap(norm_data[r, c]) for c in range(len(headers))] for r in range(len(models))]
    cell_text = [[f"{data[r, c]:.4f}" for c in range(len(headers))] for r in range(len(models))]

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=models,
        colLabels=[h.replace("_", "\n") for h in headers],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title("Full Metrics Table (green = better, top by KL)", fontsize=11, pad=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval_plots] Saved → {save_path}")


# ---- Per-class KL (unchanged, already clean) ----
def plot_per_class_kl(per_class_kl_dict: dict, label: str = "", save_path: str = "outputs/eval/per_class_kl.png"):
    classes = list(per_class_kl_dict.keys())
    values = list(per_class_kl_dict.values())
    sorted_pairs = sorted(zip(values, classes), reverse=True)
    values, classes = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(classes)))
    bars = ax.barh(classes, values, color=colors, edgecolor="white")
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


# ---- Robustness curves (unchanged) ----
def plot_robustness_curves(ood_results=None, subsampling_results=None, save_path="outputs/eval/robustness.png"):
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
        ks = [int(k.split("=")[1]) for k in subsampling_results]
        rs = list(subsampling_results.values())
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