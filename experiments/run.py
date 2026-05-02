"""
Parallel experiments runner: distributes independent configs across available GPUs.
"""
import subprocess
import csv
import os
import sys
import itertools
from pathlib import Path
from multiprocessing import Pool

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

PROJECT_ROOT = project_root

# ---- Experiment grid ----
losses = ['kl', 'js', 'custom_composite', 'emd']
heads = ['linear', 'mlp']
pretrain_options = [True]
epochs = 60
batch_size = 128
lr = 1e-3
weight_decay = 1e-4
loss_beta = 0.5
loss_epsilon = 0.1

# ---- Prepare a list of configs ----
all_configs = []
for loss, head, pretrained in itertools.product(losses, heads, pretrain_options):
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
    all_configs.append((exp_name, cmd, save_dir))


def run_one(gpu_id, configs_for_gpu):
    """
    Set CUDA_VISIBLE_DEVICES = gpu_id, then sequentially run all assigned configs.
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    for exp_name, cmd, save_dir in configs_for_gpu:
        print(f"[GPU {gpu_id}] Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"Error on GPU {gpu_id}: {result.stderr}")
            # Still record a failure value
        # Read best val loss from log
        log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
        try:
            with open(log_csv, 'r') as mf:
                reader = csv.DictReader(mf)
                val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                best_val = min(val_losses) if val_losses else 999.0
        except Exception as e:
            print(f"Could not read log for {exp_name}: {e}")
            best_val = 999.0
        print(f"[GPU {gpu_id}] {exp_name} best val loss = {best_val:.4f}")
        # We'll collect results later globally


def main():
    # Split configs between the two GPUs
    gpu_count = 2
    config_chunks = [[] for _ in range(gpu_count)]
    for i, cfg in enumerate(all_configs):
        config_chunks[i % gpu_count].append(cfg)

    # Run in parallel
    with Pool(processes=gpu_count) as pool:
        pool.starmap(run_one, [(gpu_id, config_chunks[gpu_id]) for gpu_id in range(gpu_count)])

    # After all finished, write the summary CSV
    results_file = PROJECT_ROOT / 'outputs' / 'ablation_results.csv'
    os.makedirs(results_file.parent, exist_ok=True)
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'head', 'pretrained', 'best_val_loss', 'save_dir'])
        for exp_name, cmd, save_dir in all_configs:
            loss, head = exp_name.split('_', 1)
            log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
            try:
                with open(log_csv, 'r') as mf:
                    reader = csv.DictReader(mf)
                    val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                    best_val = min(val_losses) if val_losses else 999.0
            except Exception:
                best_val = 999.0
            writer.writerow([loss, head, True, best_val, save_dir])
    print(f"All experiments finished. Summary written to {results_file}")


if __name__ == '__main__':
    main()