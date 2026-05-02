import torch

def js_divergence(pred_probs, target_probs):
    """Jensen-Shannon divergence (symmetric, bounded [0, log 2]).
    JS = (KL(P||M) + KL(Q||M)) / 2, where M = (P+Q)/2.
    """
    M = (pred_probs + target_probs) / 2.0
    log_pred = (pred_probs + 1e-9).log()
    log_target = (target_probs + 1e-9).log()
    log_M = (M + 1e-9).log()
    kl_pm = (target_probs * (log_target - log_M)).sum(dim=1).mean()
    kl_qm = (pred_probs * (log_pred - log_M)).sum(dim=1).mean()
    return (kl_pm + kl_qm) / 2