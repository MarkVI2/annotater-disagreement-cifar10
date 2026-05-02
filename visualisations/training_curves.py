import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(log_dir, exp_name):
    """
    Reads the metrics CSV from log_dir/{exp_name}_metrics.csv and saves
    training/validation loss curves + validation KL (or primary loss) curve.
    """
    csv_path = os.path.join(log_dir, f'{exp_name}_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"No log file found at {csv_path}, skipping training curves.")
        return

    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training Curves: {exp_name}')

    # Loss curves
    ax1.plot(epochs, df['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='#FF5722', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Validation loss (which for KL is KL divergence)
    ax2.plot(epochs, df['val_loss'], color='#4CAF50', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss over Epochs')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    save_path = os.path.join('outputs/plots', f'{exp_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")