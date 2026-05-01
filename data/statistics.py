"""
Compute statistics and analysis on CIFAR-10H dataset.
"""

import numpy as np
import torch
from data.dataset import CIFAR10HWrapper


def compute_entropies(soft_labels_numpy: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy for each sample.
    
    Formula: H(p) = -sum(p * log2(p)) for p > 0
    
    Args:
        soft_labels_numpy: np.ndarray of shape (N, 10)
        
    Returns:
        np.ndarray of shape (N,) with entropy values
    """
    entropies = np.zeros(len(soft_labels_numpy))
    for i, probs in enumerate(soft_labels_numpy):
        probs_nonzero = probs[probs > 0]
        entropies[i] = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    return entropies


def compute_class_average_entropy(entropies: np.ndarray,
                                  hard_labels: np.ndarray,
                                  num_classes: int = 10) -> dict:
    """
    Compute average entropy per class.
    
    Args:
        entropies: np.ndarray of shape (N,) with entropy values
        hard_labels: np.ndarray of shape (N,) with class labels
        num_classes: Number of classes
        
    Returns:
        Dict mapping class_idx -> average_entropy
    """
    class_entropies = {}
    for class_idx in range(num_classes):
        mask = hard_labels == class_idx
        if mask.sum() > 0:
            class_entropies[class_idx] = entropies[mask].mean()
        else:
            class_entropies[class_idx] = 0.0
            print(f"⚠ Class {class_idx} has no samples")
    
    return class_entropies


def compute_annotator_confusion_matrix(soft_labels: np.ndarray,
                                       hard_labels: np.ndarray,
                                       num_classes: int = 10) -> np.ndarray:
    """
    Compute confusion matrix based on annotator distributions.
    
    For each hard class, compute the average soft label distribution.
    Result: confusion_matrix[true_class, predicted_class]
    
    Args:
        soft_labels: np.ndarray of shape (N, 10)
        hard_labels: np.ndarray of shape (N,)
        num_classes: Number of classes
        
    Returns:
        np.ndarray of shape (10, 10) where each row sums to ~1.0
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for class_idx in range(num_classes):
        mask = hard_labels == class_idx
        if mask.sum() > 0:
            # Average soft labels for this class
            confusion_matrix[class_idx, :] = soft_labels[mask].mean(axis=0)
    
    return confusion_matrix


def compute_majority_agreement(soft_labels: np.ndarray,
                               hard_labels: np.ndarray) -> float:
    """
    Compute fraction of images where majority vote matches hard label.
    
    Args:
        soft_labels: np.ndarray of shape (N, 10)
        hard_labels: np.ndarray of shape (N,)
        
    Returns:
        Float in [0, 1]
    """
    majority_votes = np.argmax(soft_labels, axis=1)
    agreement = (majority_votes == hard_labels).mean()
    return float(agreement)


def identify_extreme_examples(entropies: np.ndarray,
                             n: int = 10) -> dict:
    """
    Identify n lowest and n highest entropy images.
    
    Args:
        entropies: np.ndarray of shape (N,)
        n: Number of examples to return for each extreme
        
    Returns:
        Dict with keys "lowest_indices" and "highest_indices"
    """
    lowest_indices = np.argsort(entropies)[:n]
    highest_indices = np.argsort(entropies)[-n:][::-1]  # Reverse to get highest first
    
    return {
        "lowest_indices": lowest_indices.tolist(),
        "highest_indices": highest_indices.tolist(),
    }


def compute_all_statistics(dataset: CIFAR10HWrapper) -> dict:
    """
    Compute all statistics for the dataset.
    
    Returns a comprehensive dictionary with all computed statistics.
    
    Args:
        dataset: CIFAR10HWrapper dataset
        
    Returns:
        Dict containing all statistics
    """
    print("\n📊 Computing dataset statistics...")
    
    # Get data arrays
    soft_labels = dataset.soft_labels.numpy()
    hard_labels = np.array(dataset.cifar10.targets)
    
    # Compute entropies
    entropies = compute_entropies(soft_labels)
    
    # Compute per-class statistics
    class_entropies = compute_class_average_entropy(entropies, hard_labels)
    confusion_matrix = compute_annotator_confusion_matrix(soft_labels, hard_labels)
    majority_agreement = compute_majority_agreement(soft_labels, hard_labels)
    
    # Identify extreme examples
    extreme_examples = identify_extreme_examples(entropies, n=10)
    
    # Compile statistics dictionary
    stats = {
        "entropies": entropies,
        "class_entropies": class_entropies,
        "confusion_matrix": confusion_matrix,
        "majority_agreement": majority_agreement,
        "extreme_examples": extreme_examples,
        "soft_labels": soft_labels,
        "hard_labels": hard_labels,
    }
    
    # Print summary
    print(f"  Total samples: {len(dataset)}")
    print(f"  Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"  Mean entropy: {entropies.mean():.4f} bits")
    print(f"  Median entropy: {np.median(entropies):.4f} bits")
    print(f"  Majority agreement: {100*majority_agreement:.1f}%")
    print(f"\n  Per-class average entropy:")
    for class_idx, avg_entropy in class_entropies.items():
        print(f"    Class {class_idx:2d}: {avg_entropy:.4f}")
    
    return stats
