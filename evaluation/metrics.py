import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in bits. probs: (N, 10) → returns (N,)"""
    p = probs.clamp(min=1e-9)
    return -(p * p.log2()).sum(dim=1)


def compute_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean KL(p||q) over batch. p=true, q=predicted."""
    log_q = (q + 1e-9).log()
    return F.kl_div(log_q, p, reduction='batchmean').item()


def compute_jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """Symmetric Jensen-Shannon divergence, bounded [0, 1]."""
    M = (p + q) / 2.0
    kl_pm = (p * ((p + 1e-9).log() - (M + 1e-9).log())).sum(dim=1).mean()
    kl_qm = (q * ((q + 1e-9).log() - (M + 1e-9).log())).sum(dim=1).mean()
    return ((kl_pm + kl_qm) / 2).item()


def compute_cosine(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean cosine similarity between distribution vectors."""
    return F.cosine_similarity(p, q, dim=1).mean().item()


def compute_entropy_correlations(true_p: torch.Tensor, pred_q: torch.Tensor):
    """Pearson and Spearman r between true and predicted entropy."""
    H_true = compute_entropy(true_p).numpy()
    H_pred = compute_entropy(pred_q).numpy()
    pearson_r,  _ = pearsonr(H_true, H_pred)
    spearman_r, _ = spearmanr(H_true, H_pred)
    return float(pearson_r), float(spearman_r)


def compute_precision_at_k(true_p: torch.Tensor, pred_q: torch.Tensor, k: int) -> float:
    """Overlap of top-k most ambiguous images by entropy."""
    H_true = compute_entropy(true_p).numpy()
    H_pred = compute_entropy(pred_q).numpy()
    top_true = set(np.argsort(H_true)[-k:])
    top_pred = set(np.argsort(H_pred)[-k:])
    return len(top_true & top_pred) / k


def run_all_metrics(true_p: torch.Tensor, pred_q: torch.Tensor) -> dict:
    """Run all metrics and return a summary dict."""
    pearson_r, spearman_r = compute_entropy_correlations(true_p, pred_q)
    return {
        "kl_divergence":  compute_kl(true_p, pred_q),
        "jsd":            compute_jsd(true_p, pred_q),
        "cosine_sim":     compute_cosine(true_p, pred_q),
        "pearson_r":      pearson_r,
        "spearman_r":     spearman_r,
        "precision@100":  compute_precision_at_k(true_p, pred_q, 100),
        "precision@200":  compute_precision_at_k(true_p, pred_q, 200),
        "precision@500":  compute_precision_at_k(true_p, pred_q, 500),
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    p = torch.softmax(torch.randn(2000, 10), dim=1)
    q = torch.softmax(torch.randn(2000, 10), dim=1)
    results = run_all_metrics(p, q)
    print("\n=== Metrics (dummy data) ===")
    for k, v in results.items():
        print(f"  {k:<20s}: {v:.4f}")