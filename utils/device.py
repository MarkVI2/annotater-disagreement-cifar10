import torch
import torch.amp as ac

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def get_autocast_context(device):
    device_type = device.type
    if device_type in ('cuda', 'mps'):
        return ac.autocast(device_type=device_type)
    else:
        from contextlib import nullcontext
        return nullcontext()