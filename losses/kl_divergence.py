import torch.nn.functional as F

def kl_divergence(pred_probs, target_probs):
    """KL divergence between target and predicted distributions.
    pred_probs, target_probs: (batch, 10) softmax probabilities.
    Returns scalar loss.
    """
    log_preds = (pred_probs + 1e-9).log()
    return F.kl_div(log_preds, target_probs, reduction='batchmean')