from losses.kl_divergence import kl_divergence
from losses.js_divergence import js_divergence
from losses.custom_losses import custom_composite_loss
from losses.emd_loss import emd_loss

def get_loss_function(loss_name, **kwargs):
    if loss_name == 'kl':
        return kl_divergence
    elif loss_name == 'js':
        return js_divergence
    elif loss_name == 'emd':
        return emd_loss
    elif loss_name == 'custom_composite':
        beta = kwargs.get('beta', 0.5)
        return lambda pred, target: custom_composite_loss(pred, target, beta=beta)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")