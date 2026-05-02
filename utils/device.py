import torch

def get_device():
    """
    Returns the best available device:
    - CUDA  (NVIDIA GPU) 
    - MPS   (Apple Silicon GPU)
    - CPU   (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS — Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU — no GPU found")
    return device


def get_autocast_context(device: torch.device):
    """
    Returns the appropriate autocast context for mixed precision.
    Used during training only — not needed for eval.
    """
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=torch.float16)
    else:
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)


def move_batch_to_device(images, soft_labels, hard_labels, device: torch.device):
    """
    Moves a full batch tuple to the target device cleanly.
    Use this in your eval loop instead of calling .to(device) everywhere.
    """
    return (
        images.to(device, non_blocking=True),
        soft_labels.to(device, non_blocking=True),
        hard_labels.to(device, non_blocking=True),
    )