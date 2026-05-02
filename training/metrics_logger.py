import csv
import os

class MetricsLogger:
    def __init__(self, log_dir, filename='metrics.csv'):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.file = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.header_written = False

    def log(self, epoch, metrics_dict):
        if not self.header_written:
            header = ['epoch'] + list(metrics_dict.keys())
            self.writer.writerow(header)
            self.header_written = True
        row = [epoch] + [metrics_dict[k] for k in metrics_dict]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()