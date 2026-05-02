"""
Experiments runner: loops over loss functions, heads, pretraining options,
calls training for each configuration, and records final metrics.
"""
import subprocess
import csv
import os
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can use project_root directly; no need to import
PROJECT_ROOT = project_root

# Define experiment grid
losses = ['kl', 'js', 'custom_composite', 'emd']
heads = ['linear', 'mlp']
pretrain_options = [True]
epochs = 60
batch_size = 128
lr = 1e-3
weight_decay = 1e-4
loss_beta = 0.5        # for custom_composite
loss_epsilon = 0.1     # for EMD

results_file = PROJECT_ROOT / 'outputs' / 'ablation_results.csv'
os.makedirs(results_file.parent, exist_ok=True)

with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['loss', 'head', 'pretrained', 'best_val_loss', 'save_dir'])

    for loss in losses:
        for head in heads:
            for pretrained in pretrain_options:
                exp_name = f'{loss}_{head}'
                save_dir = PROJECT_ROOT / 'outputs' / 'checkpoints' / exp_name
                pretrain_flag = '--pretrained' if pretrained else ''
                cmd = (
                    f"python -m training.train "
                    f"--loss {loss} "
                    f"--head {head} "
                    f"--epochs {epochs} "
                    f"--batch_size {batch_size} "
                    f"--lr {lr} "
                    f"--weight_decay {weight_decay} "
                    f"--save_dir {save_dir} "
                    f"--loss_beta {loss_beta} "
                    f"--loss_epsilon {loss_epsilon} "
                    f"{pretrain_flag}"
                )
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error: {result.stderr}")
                    continue
                # Read best val loss from the CSV log
                log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
                try:
                    with open(log_csv, 'r') as mf:
                        reader = csv.DictReader(mf)
                        val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                        best_val = min(val_losses) if val_losses else 999.0
                except Exception as e:
                    print(f"Could not read log: {e}")
                    best_val = 999.0
                writer.writerow([loss, head, pretrained, best_val, save_dir])