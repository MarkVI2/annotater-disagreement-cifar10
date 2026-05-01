"""
CIFAR-10H Dataset wrapper that pairs CIFAR-10 images with soft label distributions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from pathlib import Path as PathlibPath


class CIFAR10HWrapper(Dataset):
    """
    Wraps CIFAR-10 dataset and pairs each image with its soft label distribution
    from CIFAR-10H (human annotator disagreement dataset).
    
    Returns: (image, soft_label, hard_label) tuples
    """

    def __init__(self, root: str, soft_labels_path: str, train: bool = True, transform=None):
        """
        Args:
            root: Path to torchvision dataset storage
            soft_labels_path: Path to the .npy file containing soft labels
            train: If True, use CIFAR-10 training set; else test set
            transform: Optional torchvision transform to apply to images
        """
        # Load CIFAR-10 (try different download paths)
        # First try without downloading in case data already exists
        try:
            self.cifar10 = CIFAR10(root=root, train=train, download=False, transform=None)
        except (FileNotFoundError, RuntimeError) as e:
            # Try looking in a subfolder (cifar-10-python)
            cifar10_python_path = PathlibPath(root) / "cifar-10-python"
            if cifar10_python_path.exists():
                try:
                    self.cifar10 = CIFAR10(root=str(cifar10_python_path), train=train, download=False, transform=None)
                except (FileNotFoundError, RuntimeError):
                    print(f"CIFAR-10 data not found. Attempting to download to {root}...")
                    self.cifar10 = CIFAR10(root=root, train=train, download=True, transform=None)
            else:
                print(f"CIFAR-10 data not found at {root}. Attempting to download...")
                self.cifar10 = CIFAR10(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.train = train
        
        # Load soft labels
        soft_labels_full_path = PathlibPath(root) / soft_labels_path
        if not soft_labels_full_path.exists():
            raise FileNotFoundError(f"Soft labels file not found: {soft_labels_full_path}")
        
        soft_labels = np.load(soft_labels_full_path)
        
        # Handle shape mismatch
        if soft_labels.shape == (10, len(self.cifar10)):
            print(f"⚠ Soft labels were transposed from (10, N) to (N, 10)")
            soft_labels = soft_labels.T
        
        # Validate alignment
        if soft_labels.shape[0] != len(self.cifar10):
            raise ValueError(
                f"Soft labels length ({soft_labels.shape[0]}) does not match "
                f"CIFAR-10 dataset length ({len(self.cifar10)})"
            )
        
        if soft_labels.shape[1] != 10:
            raise ValueError(
                f"Soft labels must have 10 classes, got {soft_labels.shape[1]}"
            )
        
        # Convert to float32
        self.soft_labels = torch.from_numpy(soft_labels.astype(np.float32))
        
        # Validate that all rows sum to 1.0
        row_sums = self.soft_labels.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones(len(self)), atol=1e-5):
            failing_indices = (~torch.allclose(
                row_sums.unsqueeze(1), 
                torch.ones_like(row_sums.unsqueeze(1)), 
                atol=1e-5
            )).nonzero(as_tuple=True)[0]
            print(f"⚠ WARNING: {len(failing_indices)} rows don't sum to 1.0")
            print(f"   Failing indices (first 10): {failing_indices[:10].tolist()}")
            print(f"   Their sums: {row_sums[failing_indices[:10]].tolist()}")
            print(f"   Normalizing all soft labels...")
            self.soft_labels = self.soft_labels / self.soft_labels.sum(dim=1, keepdim=True)

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns: (image, soft_label, hard_label)
            - image: PIL Image or Tensor depending on transform
            - soft_label: Tensor of shape (10,)
            - hard_label: int in range [0, 9]
        """
        image, hard_label = self.cifar10[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        soft_label = self.soft_labels[idx]
        
        return image, soft_label, hard_label

    def get_entropy(self, idx: int) -> float:
        """
        Compute Shannon entropy of the soft label distribution.
        
        Formula: H = -sum(p * log2(p)) for p > 0
        
        Returns: Scalar float (entropy in bits)
        """
        probs = self.soft_labels[idx].numpy()
        # Ignore zero probabilities
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def get_majority_label(self, idx: int) -> int:
        """Returns the class with highest annotator agreement (argmax)."""
        return int(torch.argmax(self.soft_labels[idx]).item())

    def get_majority_confidence(self, idx: int) -> float:
        """Returns the fraction of annotators on the majority class."""
        return float(torch.max(self.soft_labels[idx]).item())

    def get_all_entropies(self) -> np.ndarray:
        """
        Compute entropies for all samples at once (more efficient).
        
        Returns: np.ndarray of shape (N,)
        """
        probs = self.soft_labels.numpy()
        # Compute entropy: H = -sum(p * log2(p)) for p > 0
        entropies = np.zeros(len(self))
        for i, p in enumerate(probs):
            p_nonzero = p[p > 0]
            entropies[i] = -np.sum(p_nonzero * np.log2(p_nonzero))
        return entropies
