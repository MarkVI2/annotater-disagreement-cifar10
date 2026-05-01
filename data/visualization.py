"""
Visualization module for CIFAR-10H data analysis.
Creates required plots for the data pipeline report.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_entropy_histogram(entropies: np.ndarray, save_path: str) -> None:
    """
    Plot histogram of entropy distribution across dataset.
    
    Args:
        entropies: np.ndarray of entropy values
        save_path: Path to save the figure
    """
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(entropies, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    mean_entropy = entropies.mean()
    max_entropy = np.log2(10)
    
    ax.axvline(mean_entropy, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_entropy:.3f}')
    ax.axvline(max_entropy, color='gray', linestyle=':', linewidth=2, label=f'Max possible: {max_entropy:.3f}')
    
    ax.set_xlabel('Shannon Entropy (bits)', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Distribution of Human Annotator Disagreement across CIFAR-10H', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved entropy histogram to {save_path}")


def plot_class_entropy_bar(class_entropy_dict: dict, class_names: list, save_path: str) -> None:
    """
    Plot average entropy per class.
    
    Args:
        class_entropy_dict: Dict mapping class_idx -> average_entropy
        class_names: List of class names
        save_path: Path to save the figure
    """
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = sorted(class_entropy_dict.keys())
    entropies = [class_entropy_dict[c] for c in classes]
    import seaborn as sns
    colors = sns.color_palette("husl", len(classes))
    
    bars = ax.bar([class_names[c] for c in classes], entropies, color=colors, edgecolor='black', alpha=0.8)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Average Entropy (bits)', fontsize=12)
    ax.set_title('Per-Class Average Annotator Disagreement', fontsize=13, fontweight='bold')
    ax.set_xticklabels([class_names[c] for c in classes], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved class entropy bar plot to {save_path}")


def plot_annotator_confusion_matrix(confusion_matrix: np.ndarray, class_names: list, save_path: str) -> None:
    """
    Plot confusion matrix based on annotator distributions.
    
    Args:
        confusion_matrix: np.ndarray of shape (10, 10)
        class_names: List of class names
        save_path: Path to save the figure
    """
    import seaborn as sns
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_ylabel('True (CIFAR-10) Class', fontsize=12)
    ax.set_xlabel('Annotator Response Class', fontsize=12)
    ax.set_title('Annotator Confusion Matrix', fontsize=13, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(class_names))-.5, minor=True)
    ax.set_yticks(np.arange(len(class_names))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # Annotate cells with values > 0.05
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if confusion_matrix[i, j] > 0.05:
                text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Probability', fontsize=11)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrix to {save_path}")


def plot_extreme_examples(dataset, extreme_indices_dict: dict, class_names: list, save_path: str) -> None:
    """
    Plot grid of low and high entropy examples with their distributions.
    
    Args:
        dataset: CIFAR10HWrapper dataset
        extreme_indices_dict: Dict with "lowest_indices" and "highest_indices"
        class_names: List of class names
        save_path: Path to save the figure
    """
    lowest_idx = extreme_indices_dict["lowest_indices"]
    highest_idx = extreme_indices_dict["highest_indices"]
    n = len(lowest_idx)
    
    fig, axes = plt.subplots(4, n, figsize=(3*n, 12))
    
    # Normalize for imshow
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    for i, idx in enumerate(lowest_idx):
        image, soft_label, hard_label = dataset[idx]
        entropy = dataset.get_entropy(idx)
        
        # Row 0: Low entropy images
        if isinstance(image, np.ndarray):
            axes[0, i].imshow(normalize(image.transpose(1, 2, 0)))
        else:
            axes[0, i].imshow(image)
        axes[0, i].set_title(f'H={entropy:.2f}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 1: Low entropy distributions
        axes[1, i].bar(range(10), soft_label.numpy(), color='steelblue', alpha=0.7)
        axes[1, i].set_ylim([0, 1])
        axes[1, i].set_xticks([])
        if i == 0:
            axes[1, i].set_ylabel('Probability', fontsize=10)
    
    for i, idx in enumerate(highest_idx):
        image, soft_label, hard_label = dataset[idx]
        entropy = dataset.get_entropy(idx)
        
        # Row 2: High entropy images
        if isinstance(image, np.ndarray):
            axes[2, i].imshow(normalize(image.transpose(1, 2, 0)))
        else:
            axes[2, i].imshow(image)
        axes[2, i].set_title(f'H={entropy:.2f}', fontsize=10)
        axes[2, i].axis('off')
        
        # Row 3: High entropy distributions
        axes[3, i].bar(range(10), soft_label.numpy(), color='coral', alpha=0.7)
        axes[3, i].set_ylim([0, 1])
        axes[3, i].set_xticks([])
        if i == 0:
            axes[3, i].set_ylabel('Probability', fontsize=10)
    
    # Set row labels
    axes[0, 0].set_ylabel('Low Entropy\nImages', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Distributions', fontsize=11)
    axes[2, 0].set_ylabel('High Entropy\nImages', fontsize=11, fontweight='bold')
    axes[3, 0].set_ylabel('Distributions', fontsize=11)
    
    fig.suptitle('Representative Low and High Disagreement Examples', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved extreme examples grid to {save_path}")


def generate_all_data_visualizations(dataset, statistics: dict, save_dir: str = "outputs/plots") -> None:
    """
    Generate all required data visualizations.
    
    Args:
        dataset: CIFAR10HWrapper dataset
        statistics: Dict from compute_all_statistics()
        save_dir: Directory to save plots
    """
    from training.config import CLASS_NAMES
    
    print(f"\n📊 Generating data visualizations...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Entropy histogram
    plot_entropy_histogram(
        statistics["entropies"],
        f"{save_dir}/entropy_histogram.png"
    )
    
    # 2. Per-class entropy bar
    plot_class_entropy_bar(
        statistics["class_entropies"],
        CLASS_NAMES,
        f"{save_dir}/class_entropy_bar.png"
    )
    
    # 3. Annotator confusion matrix
    plot_annotator_confusion_matrix(
        statistics["confusion_matrix"],
        CLASS_NAMES,
        f"{save_dir}/annotator_confusion_matrix.png"
    )
    
    # 4. Extreme examples grid
    plot_extreme_examples(
        dataset,
        statistics["extreme_examples"],
        CLASS_NAMES,
        f"{save_dir}/extreme_examples_grid.png"
    )
    
    print(f"✓ All visualizations saved to {save_dir}/")
