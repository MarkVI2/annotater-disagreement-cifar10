import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from evaluation.metrics import compute_entropy

os.makedirs("outputs", exist_ok=True)
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']


def plot_entropy_scatter(true_p: torch.Tensor, pred_q: torch.Tensor, save_path="outputs/entropy_scatter.png"):
    H_true = compute_entropy(true_p).numpy()
    H_pred = compute_entropy(pred_q).numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(H_true, H_pred, alpha=0.3, s=8, color='steelblue', label='Test images')
    lims = [0, np.log2(10) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect prediction')
    ax.set_xlabel("True Entropy H(p)  [bits]")
    ax.set_ylabel("Predicted Entropy H(q)  [bits]")
    ax.set_title("True vs Predicted Annotator Entropy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_comparison(results_dict: dict, save_path="outputs/metrics_comparison.png"):
    """
    results_dict = {
        'KL Loss':     {'kl_divergence': ..., 'jsd': ..., 'cosine_sim': ..., 'pearson_r': ...},
        'JSD Loss':    {...},
        'Custom Loss': {...},
    }
    """
    metrics_to_show = ['kl_divergence', 'jsd', 'cosine_sim', 'pearson_r', 'spearman_r']
    models = list(results_dict.keys())
    x = np.arange(len(metrics_to_show))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, model_name in enumerate(models):
        vals = [results_dict[model_name][m] for m in metrics_to_show]
        ax.bar(x + i * width, vals, width, label=model_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_show, rotation=15, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Loss Functions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_qualitative_examples(images, true_dists, pred_dists, entropies, save_path="outputs/qualitative_grid.png"):
    """3x3 grid: cols = low/medium/high entropy, rows = image + true dist + pred dist."""
    import torch as _torch
    sorted_idx = _torch.argsort(entropies)
    n = len(sorted_idx)
    low_idx    = sorted_idx[:3]
    medium_idx = sorted_idx[n//2 - 1 : n//2 + 2]
    high_idx   = sorted_idx[-3:]
    groups     = [low_idx, medium_idx, high_idx]
    col_titles = ['Low Entropy', 'Medium Entropy', 'High Entropy']

    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    for col, (idx_group, col_title) in enumerate(zip(groups, col_titles)):
        for row, idx in enumerate(idx_group):
            ax = fig.add_subplot(outer[row, col])
            if images is not None:
                img = images[idx]
                # Handle tensor vs numpy, and ensure (H, W, C) for imshow
                if hasattr(img, 'permute'):
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                elif isinstance(img, np.ndarray):
                    if img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                        img_np = img.transpose(1, 2, 0)
                    else:
                        img_np = img
                else:
                    img_np = img
                # Normalize if needed
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-9)
                ax.imshow(img_np.clip(0,1))
            else:
                ax.text(0.5, 0.5, f'img {idx.item()}', ha='center', va='center')
            H_val = entropies[idx].item()
            ax.set_title(f"{col_title}\nH={H_val:.2f}", fontsize=8)
            ax.axis('off')

    plt.suptitle("Qualitative Examples by Entropy Level", fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_summary(ablation_results: dict, save_path="outputs/ablation_summary.png"):
    """
    ablation_results = {'Variant Name': kl_score, ...}
    """
    names = list(ablation_results.keys())
    scores = list(ablation_results.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylabel("Mean KL Divergence ↓")
    ax.set_title("Ablation Study — Primary Metric (KL)")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    N = 2000
    p = torch.softmax(torch.randn(N, 10), dim=1)
    q = torch.softmax(torch.randn(N, 10), dim=1)
    H = compute_entropy(p)

    print("Generating plots with dummy data...")
    plot_entropy_scatter(p, q)
    plot_metrics_comparison({
        'KL Loss':     {'kl_divergence':0.41,'jsd':0.19,'cosine_sim':0.72,'pearson_r':0.61,'spearman_r':0.59},
        'JSD Loss':    {'kl_divergence':0.38,'jsd':0.17,'cosine_sim':0.75,'pearson_r':0.64,'spearman_r':0.62},
        'Custom Loss': {'kl_divergence':0.35,'jsd':0.15,'cosine_sim':0.78,'pearson_r':0.68,'spearman_r':0.66},
    })
    plot_qualitative_examples(None, p, q, H)
    plot_ablation_summary({
        'No pretrain': 0.55, 'Linear head': 0.41,
        'MLP head': 0.38,    'Frozen backbone': 0.47,
    })
    print("Done. Check outputs/ folder.")

def plot_per_class_predictions(images: torch.Tensor, true_dists: torch.Tensor,
                               pred_dists: torch.Tensor, hard_labels: torch.Tensor,
                               model_names: list, all_pred_dists: dict,
                               save_path="outputs/eval/per_class_predictions.png"):
    """
    For each of the 10 CIFAR classes, pick 1 random sample.
    Show: image | true dist bar | top-3 model pred dist bars.
    """
    CLASSES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    
    n_models = len(model_names)
    fig, axes = plt.subplots(10, 2 + n_models, figsize=(5 * (2 + n_models), 30))
    colors = ['steelblue', 'darkorange', 'seagreen']

    for cls_idx in range(10):
        # Pick a random sample from this class
        cls_mask = (hard_labels == cls_idx).nonzero(as_tuple=True)[0]
        rand_pick = cls_mask[torch.randint(len(cls_mask), (1,))].item()

        # Column 0: Image
        ax_img = axes[cls_idx, 0]
        if images is not None:
            img = images[rand_pick].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            ax_img.imshow(img)
        ax_img.set_title(f"Class: {CLASSES[cls_idx]}", fontsize=9, fontweight='bold')
        ax_img.axis('off')

        # Column 1: True distribution
        ax_true = axes[cls_idx, 1]
        true_vals = true_dists[rand_pick].numpy()
        bars = ax_true.bar(range(10), true_vals, color='gray', alpha=0.7)
        ax_true.set_xticks(range(10))
        ax_true.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=6)
        ax_true.set_ylim(0, 1)
        ax_true.set_title("True Dist", fontsize=8)
        ax_true.set_ylabel("Prob", fontsize=7)

        # Columns 2+: Model predictions
        for m_idx, model_name in enumerate(model_names):
            ax_pred = axes[cls_idx, 2 + m_idx]
            pred_vals = all_pred_dists[model_name][rand_pick].numpy()
            ax_pred.bar(range(10), pred_vals, color=colors[m_idx], alpha=0.8)
            ax_pred.set_xticks(range(10))
            ax_pred.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=6)
            ax_pred.set_ylim(0, 1)
            ax_pred.set_title(f"{model_name}", fontsize=8)

    plt.suptitle("Per-Class Predictions: Image + True vs Predicted Distributions (Top 3 Models)",
                 fontsize=13, y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")