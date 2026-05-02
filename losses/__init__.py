from losses.kl_divergence import kl_divergence
from losses.js_divergence import js_divergence
from losses.custom_losses import custom_composite_loss
from losses.emd_loss import sinkhorn_loss, EMD_COST_MATRIX

def get_loss_function(loss_name, beta=0.5, epsilon=0.1):
    if loss_name == 'kl':
        return kl_divergence
    elif loss_name == 'js':
        return js_divergence
    elif loss_name == 'emd':
        return lambda pred, target: sinkhorn_loss(pred, target, EMD_COST_MATRIX, epsilon=epsilon)
    elif loss_name == 'custom_composite':
        return lambda pred, target: custom_composite_loss(pred, target, beta=beta)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")