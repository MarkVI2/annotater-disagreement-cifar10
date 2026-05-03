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

losses = ['kl', 'js', 'custom_composite', 'emd']
heads = ['linear', 'mlp']
pretrained_types = ['random', 'cifar10', 'imagenet']
use_temperature = [False, True]
INIT_TEMP = 2.0
epochs = 60
batch_size = 128
lr = 1e-3
weight_decay = 1e-4
loss_beta = 0.5
loss_epsilon = 0.1

all_configs = []
for loss, head, pt, temp in itertools.product(losses, heads, pretrained_types, use_temperature):
    exp_name = f'{loss}_{head}_pt_{pt}'
    if temp:
        exp_name += f'_temp{INIT_TEMP}'
    save_dir = PROJECT_ROOT / 'outputs' / 'checkpoints' / exp_name
    cmd = (
        f"python -m training.train "
        f"--loss {loss} "
        f"--head {head} "
        f"--pretrained_type {pt} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--lr {lr} "
        f"--weight_decay {weight_decay} "
        f"--save_dir {save_dir} "
        f"--loss_beta {loss_beta} "
        f"--loss_epsilon {loss_epsilon} "
    )
    if temp:
        cmd += f" --use_temperature --init_temp {INIT_TEMP}"
    all_configs.append((exp_name, cmd, save_dir))


def run_one(gpu_id, configs_for_gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    for exp_name, cmd, save_dir in configs_for_gpu:
        best_path = save_dir / (exp_name + '_best.pth')
        latest_path = save_dir / (exp_name + '_latest.pth')

        if best_path.exists():
            print(f"[GPU {gpu_id}] {exp_name} already completed (best.pth found). Skipping.")
            # Read best validation loss from existing log
            log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
            try:
                with open(log_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                    best_val = min(val_losses) if val_losses else 999.0
            except:
                best_val = 999.0
            print(f"[GPU {gpu_id}] {exp_name} best val loss = {best_val:.4f}")
            continue

        if latest_path.exists():
            print(f"[GPU {gpu_id}] {exp_name} has incomplete run; resuming from latest checkpoint.")
            cmd += " --resume"
        else:
            print(f"[GPU {gpu_id}] Starting fresh training for {exp_name}.")

        full_cmd = cmd  # already includes --resume if needed
        print(f"[GPU {gpu_id}] Running: {full_cmd}")
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"[GPU {gpu_id}] ERROR: Training failed for {exp_name}")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

        # Read best val loss from log
        log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
        try:
            with open(log_csv, 'r') as f:
                reader = csv.DictReader(f)
                val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                best_val = min(val_losses) if val_losses else 999.0
        except:
            best_val = 999.0
        print(f"[GPU {gpu_id}] {exp_name} best val loss = {best_val:.4f}")


def main():
    gpu_count = 2
    config_chunks = [[] for _ in range(gpu_count)]
    for i, cfg in enumerate(all_configs):
        config_chunks[i % gpu_count].append(cfg)

    with Pool(processes=gpu_count) as pool:
        pool.starmap(run_one, [(gpu_id, config_chunks[gpu_id]) for gpu_id in range(gpu_count)])

    # Write summary CSV
    results_file = PROJECT_ROOT / 'outputs' / 'ablation_results.csv'
    os.makedirs(results_file.parent, exist_ok=True)
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_name', 'loss', 'head', 'pretrained_type', 'temperature', 'best_val_loss', 'save_dir'])
        for exp_name, cmd, save_dir in all_configs:
            log_csv = PROJECT_ROOT / 'outputs' / 'logs' / f'{exp_name}_metrics.csv'
            best_val = 999.0
            try:
                with open(log_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    val_losses = [float(row['val_loss']) for row in reader if 'val_loss' in row]
                    best_val = min(val_losses) if val_losses else 999.0
            except:
                pass
            loss, head = exp_name.split('_')[0], exp_name.split('_')[1]
            pt = exp_name.split('_')[3] if len(exp_name.split('_')) > 3 else 'unknown'
            temp = '_temp' in exp_name
            writer.writerow([exp_name, loss, head, pt, temp, best_val, str(save_dir)])
    print(f"All experiments finished. Summary written to {results_file}")


if __name__ == '__main__':
    main()