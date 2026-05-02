import csv
from pathlib import Path

results = []
for exp in ['kl_linear', 'kl_mlp', 'js_linear', 'js_mlp',
            'custom_composite_linear', 'custom_composite_mlp',
            'emd_linear', 'emd_mlp']:
    log_csv = Path(f'outputs/logs/{exp}_metrics.csv')
    if log_csv.exists():
        with open(log_csv) as f:
            reader = csv.DictReader(f)
            vals = [float(row['val_loss']) for row in reader if 'val_loss' in row]
            best_val = min(vals) if vals else 999.0
        loss, head = exp.split('_', 1)
        results.append([loss, head, True, best_val, f'outputs/checkpoints/{exp}'])

with open('outputs/ablation_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['loss', 'head', 'pretrained', 'best_val_loss', 'save_dir'])
    writer.writerows(results)
