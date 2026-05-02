import torch
import numpy as np
from evaluation.metrics import compute_entropy, compute_kl
from scipy.stats import pearsonr


def annotator_subsampling_check(raw_counts: np.ndarray, pred_q: torch.Tensor,
                                 subsample_sizes=[5, 10, 20]) -> dict:
    """
    raw_counts: (N, 10) integer array — how many annotators picked each class
    pred_q:     (N, 10) model predictions
    Returns pearson_r for each subsample size.
    """
    results = {}
    rng = np.random.default_rng(42)
    total_annotators = raw_counts.sum(axis=1, keepdims=True)

    for k in subsample_sizes:
        # Simulate k annotators by resampling from counts
        resampled = np.array([
            rng.multinomial(k, raw_counts[i] / total_annotators[i])
            for i in range(len(raw_counts))
        ]).astype(np.float32)
        resampled_probs = resampled / resampled.sum(axis=1, keepdims=True)
        resampled_t = torch.from_numpy(resampled_probs)

        H_true = compute_entropy(resampled_t).numpy()
        H_pred = compute_entropy(pred_q).numpy()
        r, _ = pearsonr(H_true, H_pred)
        results[f"k={k}"] = float(r)
        print(f"  Annotators={k:>2d} → Pearson r = {r:.4f}")

    return results

def ood_corruption_check(model, test_images: torch.Tensor,
                         severities=[0.05, 0.1, 0.2, 0.3, 0.5]) -> dict:
    model.eval()
    device = next(model.parameters()).device
    results = {}
    batch_size = 128

    with torch.no_grad():
        for sigma in severities:
            all_H = []
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i+batch_size].to(device)
                noisy = batch + torch.randn_like(batch) * sigma
                noisy = noisy.clamp(-3, 3)
                q_noisy = model(noisy).cpu()
                all_H.append(compute_entropy(q_noisy))
            mean_H = torch.cat(all_H).mean().item()
            results[f"sigma={sigma}"] = mean_H
            print(f"  sigma={sigma:.2f} → mean predicted entropy = {mean_H:.4f}")
    return results

def per_class_kl(true_p: torch.Tensor, pred_q: torch.Tensor,
                  hard_labels: torch.Tensor) -> dict:
    """KL divergence broken down per true class."""
    import torch.nn.functional as F
    CLASSES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    results = {}
    for c in range(10):
        mask = (hard_labels == c)
        if mask.sum() == 0:
            continue
        kl = F.kl_div((pred_q[mask]+1e-9).log(), true_p[mask],
                       reduction='batchmean').item()
        results[CLASSES[c]] = kl
        print(f"  {CLASSES[c]:<12s}: KL = {kl:.4f}")
    return results