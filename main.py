#!/usr/bin/env python3
"""
One-shot pipeline: data -> training -> evaluation.
Usage:   python main.py [--skip-data] [--skip-train] [--skip-eval]
"""

import argparse
import subprocess
import sys
import os
import json
import glob
from pathlib import Path

def run_data_pipeline():
    print("\n" + "="*60)
    print("STAGE 1: DATA PIPELINE")
    print("="*60)
    # The main_data.py script loads, checks, splits, and creates dataloaders
    result = subprocess.run([sys.executable, "main_data.py"], capture_output=False)
    if result.returncode != 0:
        print("Data pipeline failed. Exiting.")
        sys.exit(1)

def run_training():
    print("\n" + "="*60)
    print("STAGE 2: TRAINING ALL EXPERIMENTS")
    print("="*60)
    # The experiments/run.py script uses multiprocessing for GPU assignment
    result = subprocess.run([sys.executable, "experiments/run.py"], capture_output=False)
    if result.returncode != 0:
        print("Training failed. Check experiments/run.py for errors.")
        sys.exit(1)

def find_best_checkpoints(checkpoint_dir="outputs/checkpoints"):
    """Find all *_best.pth files and return a dict of exp_name -> path."""
    best = {}
    for path in Path(checkpoint_dir).rglob("*_best.pth"):
        # exp_name is the file name without _best.pth
        name = path.stem.replace("_best", "")
        best[name] = str(path)
    return best

def run_evaluation():
    print("\n" + "="*60)
    print("STAGE 3: EVALUATION")
    print("="*60)
    result = subprocess.run([sys.executable, "evaluation/eval.py"], capture_output=False)
    if result.returncode != 0:
        print("Evaluation failed.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Full pipeline for CIFAR-10H disagreement project")
    parser.add_argument("--skip-data", action="store_true", help="Skip data pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    if not args.skip_data:
        run_data_pipeline()
    if not args.skip_train:
        run_training()
    if not args.skip_eval:
        run_evaluation()

    print("\n===== END OF PIPELINE =====")

if __name__ == "__main__":
    main()