import torch
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cifar_resnet import build_soft_label_model
from data.pipeline import load_full_dataset, create_splits, create_dataloaders
from evaluation.metrics import run_all_metrics, compute_entropy
from utils.device import get_device
from evaluation.robustness import (
    annotator_subsampling_check,
    ood_corruption_check,
    per_class_kl,
)
from evaluation.visualize import (
    plot_entropy_scatter,
    plot_metrics_comparison,
    plot_qualitative_examples,
    plot_ablation_summary,
)
from visualisations.eval_plots import (
    plot_metrics_comparison_bar,
    plot_metrics_table,
    plot_precision_at_k,
    plot_per_class_kl as plot_per_class_kl_bar,
    plot_robustness_curves as plot_robustness_curves_eval,
)
from analysis.cifar10h_analysis import (
    plot_entropy_analysis,
    plot_distribution_comparison,
    plot_gradcam_grid,
    plot_failure_cases,
    plot_robustness_curves,
    plot_per_class_entropy_calibration,
    plot_entropy_reliability_diagram,
    GradCAM,
)

def find_best_checkpoints(checkpoint_dir="outputs/checkpoints"):
    """Return dict {exp_name: ckpt_path} for all *_best.pth files."""
    best = {}
    for path in Path(checkpoint_dir).rglob("*_best.pth"):
        name = path.stem.replace("_best", "")
        best[name] = str(path)
    return best

def load_model_from_ckpt(ckpt_path, device, head_type='linear', use_temperature=False):
    model = build_soft_label_model(
        head_type=head_type,
        pretrained_type='random',
        use_temperature=use_temperature,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt['model_state_dict']
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model

def run_analysis_for_model(model, test_loader, device, output_dir='outputs/eval'):
    """Runs all per-model plots and returns metrics dict."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds, all_true, all_hard, all_images = [], [], [], []
    with torch.no_grad():
        for images, soft_labels, hard_labels in test_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            pred_q = model(images)
            all_preds.append(pred_q.cpu())
            all_true.append(soft_labels.cpu())
            all_hard.append(hard_labels)
            all_images.append(images.cpu())

    pred_q = torch.cat(all_preds, dim=0)
    true_p = torch.cat(all_true, dim=0)
    hard_labels = torch.cat(all_hard, dim=0)
    raw_images = torch.cat(all_images, dim=0)  # normalized

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
    images_raw = (raw_images * std + mean).clamp(0,1).permute(0,2,3,1).numpy()

    metrics = run_all_metrics(true_p, pred_q)
    print(f"\n=== Metrics for {output_dir} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Per-model plots
    plot_entropy_scatter(true_p, pred_q, save_path=os.path.join(output_dir, "entropy_scatter.png"))
    plot_qualitative_examples(images_raw[:1000], true_p[:1000], pred_q[:1000],
                          compute_entropy(true_p[:1000]), save_path=os.path.join(output_dir, "qualitative_grid.png"))
    plot_failure_cases(images_raw, true_p.numpy(), pred_q.numpy(), save_path=os.path.join(output_dir, "failure_cases.png"))
    plot_entropy_analysis(true_p.numpy(), pred_q.numpy(), save_path=os.path.join(output_dir, "entropy_analysis.png"))
    plot_distribution_comparison(images_raw[:1000], true_p.numpy()[:1000], pred_q.numpy()[:1000],
                                 save_path=os.path.join(output_dir, "distribution_comparison.png"))

    # Grad‑CAM
    backbone = model.backbone
    target_layer = backbone.layer4[-1]
    plot_gradcam_grid(model, target_layer, raw_images[:1000], images_raw[:1000],
                      true_p.numpy()[:1000], pred_q.numpy()[:1000],
                      save_path=os.path.join(output_dir, "gradcam_grid.png"))

    return metrics, true_p, pred_q, hard_labels, raw_images

def main():
    from pathlib import Path
    device = get_device()
    dataset = load_full_dataset()
    splits = create_splits(dataset)
    loaders = create_dataloaders(splits)
    test_loader = loaders["test"]

    best_ckpts = find_best_checkpoints()
    if not best_ckpts:
        print("No best checkpoints found. Aborting.")
        return

    all_metrics = {}
    per_model_data = {}  # keep true/pred/etc. for best model later

    # Evaluate all models
    for exp_name, ckpt_path in best_ckpts.items():
        print(f"\n===== Evaluating {exp_name} =====")
        head = "mlp" if "mlp" in exp_name else "linear"
        use_temp = "_temp" in exp_name
        model = load_model_from_ckpt(ckpt_path, device, head_type=head, use_temperature=use_temp)
        out_dir = f"outputs/eval/{exp_name}"
        metrics, true_p, pred_q, hard_labels, raw_images = run_analysis_for_model(model, test_loader, device, out_dir)
        all_metrics[exp_name] = metrics
        per_model_data[exp_name] = (true_p, pred_q, hard_labels, raw_images)

    # Aggregate plots
    os.makedirs("outputs/eval/aggregated", exist_ok=True)

    # Grouped bar chart
    if all_metrics:
        plot_metrics_comparison_bar(all_metrics, save_path="outputs/eval/aggregated/metrics_comparison.png")
        plot_metrics_table(all_metrics, save_path="outputs/eval/aggregated/metrics_table.png")
        plot_precision_at_k(all_metrics, save_path="outputs/eval/aggregated/precision_at_k.png")

        # Ablation summary (using KL as primary metric)
        ablation_kl = {name: m["kl_divergence"] for name, m in all_metrics.items()}
        plot_ablation_summary(ablation_kl, save_path="outputs/eval/aggregated/ablation_summary.png")

    # Per‑class KL for the best model (lowest KL)
    best_model_name = min(all_metrics, key=lambda n: all_metrics[n]["kl_divergence"])
    print(f"\nBest model by KL: {best_model_name}")
    true_p_best, pred_q_best, hard_labels_best, images_best = per_model_data[best_model_name]

    # Per‑class KL bar
    from evaluation.robustness import per_class_kl
    pkl_dict = per_class_kl(true_p_best, pred_q_best, hard_labels_best)
    plot_per_class_kl_bar(pkl_dict, label=best_model_name, save_path="outputs/eval/aggregated/per_class_kl.png")

    # Robustness: annotator subsampling (requires raw counts, which we don't have in the test set;
    # if you have cifar10h-counts.npy, you can load it. We'll skip or mock.)
    # For completeness, load counts file if available:
    counts_path = "data/cifar10h-counts.npy"
    if os.path.exists(counts_path):
        raw_counts_full = np.load(counts_path)
        # Select only the test split indices (we have test_indices from splits)
        test_indices = splits['test'].indices   # list of indices in the full dataset
        raw_counts = raw_counts_full[test_indices]   # shape (2000, 10)
        print("Annotator subsampling check running...")
        sub_results = annotator_subsampling_check(raw_counts, pred_q_best)
        plot_robustness_curves_eval(subsampling_results=sub_results,
                                    save_path="outputs/eval/aggregated/robustness_subsampling.png")
    else:
        print("cifar10h-counts.npy not found. Skipping annotator subsampling robustness check.")

    # OOD corruption check and plot
    print("Running OOD corruption check on best model...")
    best_model = load_model_from_ckpt(best_ckpts[best_model_name], device,
                                  head_type="mlp" if "mlp" in best_model_name else "linear",
                                  use_temperature="_temp" in best_model_name)
    ood_results = ood_corruption_check(best_model, images_best, severities=[0.05, 0.1, 0.2, 0.3, 0.5])
    plot_robustness_curves_eval(ood_results=ood_results,
                                save_path="outputs/eval/aggregated/robustness_ood.png")

    print("\nAll evaluation plots are in outputs/eval/aggregated/ and per-model folders.")

if __name__ == "__main__":
    main()