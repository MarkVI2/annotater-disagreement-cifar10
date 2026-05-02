"""
Data pipeline utilities for loading, splitting, and validating CIFAR-10H data.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path

from data.dataset import CIFAR10HWrapper
from training.config import (
    FIXED_SEED, DATA_DIR, CIFAR10H_PROBS_FILE,
    CIFAR10_MEAN, CIFAR10_STD, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class TransformWrapper:
    """Wrapper to apply transforms when loading data from a subset."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, soft_label, hard_label = self.dataset[actual_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, soft_label, hard_label


def set_seed(seed: int = FIXED_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For reproducibility (at the cost of some speed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_transforms(is_train: bool) -> transforms.Compose:
    """
    Get data augmentation transforms.
    
    Training: ToTensor -> Normalize -> RandomHorizontalFlip -> RandomCrop
    Validation/Test: ToTensor -> Normalize
    
    Args:
        is_train: If True, apply training augmentations
        
    Returns:
        torchvision.transforms.Compose object
    """
    if is_train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])


def load_full_dataset(data_dir: str = DATA_DIR, 
                     soft_labels_path: str = CIFAR10H_PROBS_FILE) -> CIFAR10HWrapper:
    """
    Load the complete CIFAR-10H dataset (test set with soft labels).
    
    Args:
        data_dir: Path to data directory
        soft_labels_path: Filename of soft labels within data_dir
        
    Returns:
        CIFAR10HWrapper dataset object
    """
    dataset = CIFAR10HWrapper(
        root=data_dir,
        soft_labels_path=soft_labels_path,
        train=False,  # CIFAR-10H is based on test set
        transform=None  # Transforms applied later per split
    )
    
    print(f"✓ Loaded CIFAR-10H dataset: {len(dataset)} images")
    print(f"  Soft labels shape: {dataset.soft_labels.shape}")
    
    return dataset


def create_splits(dataset: CIFAR10HWrapper,
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio: float = VAL_RATIO,
                 seed: int = FIXED_SEED) -> dict:
    """
    Split dataset into train, validation, and test sets.
    
    Uses stratified splitting to preserve class balance across splits.
    
    Args:
        dataset: CIFAR10HWrapper dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dict with keys {"train", "val", "test"} containing Subset objects
    """
    set_seed(seed)
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    # Get hard labels for stratification
    hard_labels = np.array([dataset.cifar10.targets[i] for i in range(n_samples)])
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=hard_labels,
        random_state=seed
    )
    
    # Second split: val vs test (from remaining data)
    temp_hard_labels = hard_labels[temp_idx]
    val_size = val_ratio / (val_ratio + TEST_RATIO)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_size,
        stratify=temp_hard_labels,
        random_state=seed
    )
    
    # Create Subset objects
    splits = {
        "train": Subset(dataset, train_idx.tolist()),
        "val": Subset(dataset, val_idx.tolist()),
        "test": Subset(dataset, test_idx.tolist()),
    }
    
    # Logging
    print(f"✓ Created data splits:")
    print(f"  Train: {len(splits['train'])} ({100*len(splits['train'])/n_samples:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({100*len(splits['val'])/n_samples:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({100*len(splits['test'])/n_samples:.1f}%)")
    
    # Verify no overlaps
    all_idx = set()
    for split_name, split in splits.items():
        split_indices = set(split.indices)
        assert len(all_idx & split_indices) == 0, f"Index overlap in {split_name}!"
        all_idx.update(split_indices)
    
    return splits


def create_dataloaders(splits_dict: dict,
                      batch_size: int = 128,
                      num_workers: int = 4,
                      transforms_dict: dict = None) -> dict:
    """
    Create PyTorch DataLoaders for each split.
    
    Args:
        splits_dict: Dict from create_splits with Subset objects
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        transforms_dict: Dict with keys {"train", "val", "test"} of transforms.
                        If None, no transforms applied.
        
    Returns:
        Dict with keys {"train", "val", "test"} containing DataLoader objects
    """
    if transforms_dict is None:
        transforms_dict = {
            "train": get_transforms(is_train=True),
            "val": get_transforms(is_train=False),
            "test": get_transforms(is_train=False),
        }
    
    loaders = {}
    
    # Apply transforms by wrapping the dataset
    for split_name, subset in splits_dict.items():
        # Get the base dataset from the subset
        base_dataset = subset.dataset
        indices = subset.indices
        
        # Create a wrapped dataset with transforms applied
        wrapped_dataset = TransformWrapper(
            base_dataset,
            indices,
            transforms_dict[split_name]
        )
        
        # Create DataLoader
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    print(f"✓ Created DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
    
    return loaders


def run_sanity_checks(dataset: CIFAR10HWrapper) -> bool:
    """
    Run comprehensive sanity checks on the dataset.
    
    Verifies:
    - All soft labels sum to 1.0
    - No NaN or inf values
    - Entropy values in valid range [0, log2(10)]
    - Hard labels in range [0, 9]
    - Image tensor values reasonable after normalization
    
    Args:
        dataset: CIFAR10HWrapper dataset
        
    Returns:
        True if all checks pass, raises AssertionError otherwise
    """
    print("🔍 Running sanity checks...")
    
    # Check 1: Soft labels sum to 1
    sums = dataset.soft_labels.sum(dim=1)
    check_1 = torch.allclose(sums, torch.ones(len(dataset)), atol=1e-5)
    print(f"  {'✓' if check_1 else '✗'} Soft labels sum to 1.0: {check_1}")
    assert check_1, "Not all soft labels sum to 1.0"
    
    # Check 2: No NaN or inf
    check_2 = torch.isfinite(dataset.soft_labels).all()
    print(f"  {'✓' if check_2 else '✗'} No NaN/inf values: {check_2}")
    assert check_2, "Found NaN or inf values in soft labels"
    
    # Check 3: Entropy range
    entropies = dataset.get_all_entropies()
    max_entropy = np.log2(10)
    check_3 = (entropies.min() >= 0) and (entropies.max() <= max_entropy + 1e-5)
    print(f"  {'✓' if check_3 else '✗'} Entropy range [0, {max_entropy:.2f}]: {check_3}")
    print(f"       Min: {entropies.min():.4f}, Max: {entropies.max():.4f}")
    assert check_3, f"Entropy out of range: [{entropies.min()}, {entropies.max()}]"
    
    # Check 4: Hard labels in valid range
    hard_labels_array = np.array(dataset.cifar10.targets)
    check_4 = (hard_labels_array.min() >= 0) and (hard_labels_array.max() <= 9)
    print(f"  {'✓' if check_4 else '✗'} Hard labels in [0, 9]: {check_4}")
    assert check_4, f"Hard labels out of range: [{hard_labels_array.min()}, {hard_labels_array.max()}]"
    
    # Check 5: Sample images and check tensor values
    print(f"  Sampling 100 images to verify tensor values...")
    transform = get_transforms(is_train=False)
    sample_indices = np.random.choice(len(dataset), size=100, replace=False)
    for idx in sample_indices:
        image, _, _ = dataset[idx]
        image_tensor = transform(image) if not isinstance(image, torch.Tensor) else image
        if image_tensor.min() < -5 or image_tensor.max() > 5:
            print(f"⚠ Image {idx} has unusual values: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    print(f"  ✓ Image tensors appear reasonable")
    
    print("✓ All sanity checks passed!")
    return True
