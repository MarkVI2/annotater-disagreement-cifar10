import torch

# Example semantic distance matrix (10x10) – you can improve with WordNet or manual reasoning.
# Values in [0,1]; diagonal = 0.
EMD_COST_MATRIX = torch.tensor([
    # airplan autom  bird   cat   deer   dog   frog  horse  ship  truck
    [0.0, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.7],  # airplane
    [0.8, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.2],  # automobile
    [0.9, 0.9, 0.0, 0.4, 0.6, 0.5, 0.3, 0.5, 0.9, 0.9],  # bird
    [0.9, 0.9, 0.4, 0.0, 0.4, 0.2, 0.5, 0.3, 0.9, 0.9],  # cat
    [0.9, 0.9, 0.6, 0.4, 0.0, 0.4, 0.5, 0.3, 0.9, 0.9],  # deer
    [0.9, 0.9, 0.5, 0.2, 0.4, 0.0, 0.5, 0.2, 0.9, 0.9],  # dog
    [0.9, 0.9, 0.3, 0.5, 0.5, 0.5, 0.0, 0.5, 0.9, 0.9],  # frog
    [0.9, 0.9, 0.5, 0.3, 0.3, 0.2, 0.5, 0.0, 0.9, 0.9],  # horse
    [0.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.7],  # ship
    [0.7, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.0],  # truck
], dtype=torch.float32)


def emd_loss(pred_probs, target_probs, cost_matrix=None):
    """
    Earth Mover's Distance (1-Wasserstein) for discrete distributions.
    pred_probs, target_probs: (batch, 10) float tensors.
    cost_matrix: (10,10) tensor of distances between classes.
    Returns scalar loss.
    """
    if cost_matrix is None:
        cost_matrix = EMD_COST_MATRIX.to(pred_probs.device)
    
    # Define an ordering that groups similar classes together (manual)
    order = [0, 8, 9, 1, 2, 6, 3, 5, 7, 4]  # airplane, ship, truck, automobile, bird, frog, cat, dog, horse, deer
    
    # Reorder both distributions
    p = pred_probs[:, order]
    t = target_probs[:, order]
    
    # Cumulative distributions
    cdf_p = torch.cumsum(p, dim=1)
    cdf_t = torch.cumsum(t, dim=1)
    
    # EMD approximation via 1D embedding
    loss = torch.abs(cdf_p - cdf_t).mean(dim=1).mean()
    return loss
