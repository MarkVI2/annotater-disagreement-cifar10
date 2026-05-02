import torch

def sinkhorn_loss(pred_probs, target_probs, cost_matrix, epsilon=0.1, max_iter=50):
    """
    Sinkhorn-Knopp regularized OT distance (differentiable approximation of EMD).
    pred_probs, target_probs: (batch, 10) softmax probabilities.
    cost_matrix: (10, 10) tensor of pairwise distances.
    epsilon: entropic regularization strength (smaller -> closer to exact EMD).
    Returns scalar loss.
    """
    batch_size = pred_probs.shape[0]
    device = pred_probs.device

    # Move cost matrix to correct device
    cost_matrix = cost_matrix.to(device)

    # Ensure positive masses
    a = pred_probs.clamp(min=1e-9)
    b = target_probs.clamp(min=1e-9)

    # Gibbs kernel K = exp(-C / epsilon), shape (1, 10, 10)
    K = torch.exp(-cost_matrix / epsilon).unsqueeze(0)
    # Expand to match batch size for bmm
    K = K.expand(batch_size, -1, -1)

    # Sinkhorn iterations
    v = torch.ones(batch_size, 10, device=device)
    for _ in range(max_iter):
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)
        v = b / (torch.bmm(K.transpose(1,2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)

    # Transport plan T = diag(u) * K * diag(v)
    T = u.unsqueeze(2) * K * v.unsqueeze(1)  # (batch, 10, 10)

    # Sinkhorn distance = sum(T * C)
    cost = T * cost_matrix.unsqueeze(0)
    return cost.sum(dim=(1,2)).mean()

# Keep the same cost matrix you already defined
EMD_COST_MATRIX = torch.tensor([
    [0.0, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.7],
    [0.8, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.2],
    [0.9, 0.9, 0.0, 0.4, 0.6, 0.5, 0.3, 0.5, 0.9, 0.9],
    [0.9, 0.9, 0.4, 0.0, 0.4, 0.2, 0.5, 0.3, 0.9, 0.9],
    [0.9, 0.9, 0.6, 0.4, 0.0, 0.4, 0.5, 0.3, 0.9, 0.9],
    [0.9, 0.9, 0.5, 0.2, 0.4, 0.0, 0.5, 0.2, 0.9, 0.9],
    [0.9, 0.9, 0.3, 0.5, 0.5, 0.5, 0.0, 0.5, 0.9, 0.9],
    [0.9, 0.9, 0.5, 0.3, 0.3, 0.2, 0.5, 0.0, 0.9, 0.9],
    [0.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.7],
    [0.7, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.0],
], dtype=torch.float32)