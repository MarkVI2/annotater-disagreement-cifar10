import os
import numpy as np
import csv

class MetricsLogger:
    def __init__(self, log_dir, filename_prefix):
        os.makedirs(log_dir, exist_ok=True)
        self.prefix = filename_prefix
        self.csv_path = os.path.join(log_dir, f'{filename_prefix}_metrics.csv')
        self.npy_path = os.path.join(log_dir, f'{filename_prefix}_metrics.npy')
        self.metrics = {'epoch': []}
        # Write CSV header
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.header_written = False

    def log(self, epoch, metrics_dict):
        self.metrics['epoch'].append(epoch)
        row = [epoch]
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
            row.append(v)
        # Write CSV
        if not self.header_written:
            header = ['epoch'] + list(metrics_dict.keys())
            self.csv_writer.writerow(header)
            self.header_written = True
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        # Also save as npy every epoch (overwrites, fine)
        np.save(self.npy_path, self.metrics)

    def close(self):
        self.csv_file.close()
        np.save(self.npy_path, self.metrics)