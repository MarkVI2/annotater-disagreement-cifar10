import torch
import torch.nn.functional as F

def custom_composite_loss(pred_probs, target_probs, beta=0.5):
    """
    Composite loss: KL divergence + beta * |H_pred - H_true|.
    H = Shannon entropy (in bits, base 2).
    beta: weight of the entropy error term.
    """
    # KL part
    log_pred = (pred_probs + 1e-9).log()
    kl = F.kl_div(log_pred, target_probs, reduction='batchmean')

    # Entropy error part
    def entropy(probs):
        # probs: (batch, 10)
        plog = probs * (probs + 1e-9).log2()
        return -plog.sum(dim=1)  # (batch,)
    H_pred = entropy(pred_probs)
    H_true = entropy(target_probs)
    entropy_error = (H_pred - H_true).abs().mean()

    return kl + beta * entropy_error