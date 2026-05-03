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


def compute_metrics_only(model, test_loader, device):
    """Run inference and return metrics dict without any plotting."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, soft_labels, _ in test_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            pred_q = model(images)
            all_preds.append(pred_q.cpu())
            all_true.append(soft_labels.cpu())
    pred_q = torch.cat(all_preds, dim=0)
    true_p = torch.cat(all_true, dim=0)
    return run_all_metrics(true_p, pred_q)


def run_analysis_for_model(model, test_loader, device, output_dir='outputs/eval'):
    """Runs all per-model plots and returns metrics, tensors for robustness."""
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

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    images_raw = (raw_images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    metrics = run_all_metrics(true_p, pred_q)
    print(f"\n=== Metrics for {output_dir} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Per-model plots
    plot_entropy_scatter(true_p, pred_q,
                         save_path=os.path.join(output_dir, "entropy_scatter.png"))
    plot_qualitative_examples(images_raw[:1000], true_p[:1000], pred_q[:1000],
                              compute_entropy(true_p[:1000]),
                              save_path=os.path.join(output_dir, "qualitative_grid.png"))
    plot_failure_cases(images_raw, true_p.numpy(), pred_q.numpy(),
                       save_path=os.path.join(output_dir, "failure_cases.png"))
    plot_entropy_analysis(true_p.numpy(), pred_q.numpy(),
                          save_path=os.path.join(output_dir, "entropy_analysis.png"))
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
    device = get_device()
    dataset = load_full_dataset()
    splits = create_splits(dataset)
    loaders = create_dataloaders(splits)
    test_loader = loaders["test"]

    best_ckpts = find_best_checkpoints()
    if not best_ckpts:
        print("No best checkpoints found. Aborting.")
        return

    # ------------------------------------------------------------
    # 1. Compute metrics for ALL checkpoints (no per-model plots)
    # ------------------------------------------------------------
    all_metrics = {}
    print("\n[1/3] Computing metrics for all checkpoints...")
    for exp_name, ckpt_path in best_ckpts.items():
        head = "mlp" if "mlp" in exp_name else "linear"
        use_temp = "_temp" in exp_name
        model = load_model_from_ckpt(ckpt_path, device, head_type=head, use_temperature=use_temp)
        metrics = compute_metrics_only(model, test_loader, device)
        all_metrics[exp_name] = metrics
        print(f"  {exp_name:50s}  KL={metrics['kl_divergence']:.4f}")

    # ------------------------------------------------------------
    # 2. Select representative models for per-model figures
    #    (best linear/cifar10 for each loss function)
    # ------------------------------------------------------------
    representative_models = {}
    loss_functions = ['kl', 'js', 'custom_composite', 'emd']
    for loss in loss_functions:
        # Prefer e.g. "kl_linear_pt_cifar10", fallback to any linear/cifar10 variant
        pattern = f'{loss}_linear_pt_cifar10'
        candidates = [n for n in best_ckpts if pattern in n]
        if not candidates:
            candidates = [n for n in best_ckpts if loss in n]  # any variant of that loss
        if candidates:
            # Pick the one with lowest KL from all_metrics
            best_candidate = min(candidates, key=lambda n: all_metrics[n]['kl_divergence'])
            representative_models[loss] = best_candidate
            print(f"  Representative for {loss}: {best_candidate}")
        else:
            print(f"  No model found for loss {loss}")

    # Generate per-model plots for each representative
    per_model_data = {}
    for loss, exp_name in representative_models.items():
        head = "mlp" if "mlp" in exp_name else "linear"
        use_temp = "_temp" in exp_name
        model = load_model_from_ckpt(best_ckpts[exp_name], device, head_type=head, use_temperature=use_temp)
        out_dir = f"outputs/eval/representative/{exp_name}"
        metrics, true_p, pred_q, hard_labels, raw_images = run_analysis_for_model(
            model, test_loader, device, out_dir
        )
        per_model_data[exp_name] = (true_p, pred_q, hard_labels, raw_images)

    # ------------------------------------------------------------
    # 3. Aggregate plots (using metrics from all or representative)
    # ------------------------------------------------------------
    os.makedirs("outputs/eval/aggregated", exist_ok=True)

    # Use only representative models for comparison bar chart (cleaner)
    rep_metrics = {name: all_metrics[name] for name in representative_models.values()}
    plot_metrics_comparison_bar(rep_metrics,
                                save_path="outputs/eval/aggregated/metrics_comparison.png")
    # Full metrics heatmap table can use all models
    plot_metrics_table(all_metrics,
                       save_path="outputs/eval/aggregated/metrics_table.png")
    plot_precision_at_k(rep_metrics,
                        save_path="outputs/eval/aggregated/precision_at_k.png")

    # Ablation summary (using KL from all models)
    ablation_kl = {name: m["kl_divergence"] for name, m in all_metrics.items()}
    plot_ablation_summary(ablation_kl,
                          save_path="outputs/eval/aggregated/ablation_summary.png")

    # ------------------------------------------------------------
    # 4. Robustness & per-class analysis on the best overall model
    # ------------------------------------------------------------
    best_model_name = min(all_metrics, key=lambda n: all_metrics[n]["kl_divergence"])
    print(f"\nBest model by KL: {best_model_name}")

    # Load best model and run full analysis (or reuse from per_model_data if it's among representatives)
    head = "mlp" if "mlp" in best_model_name else "linear"
    use_temp = "_temp" in best_model_name
    best_model = load_model_from_ckpt(best_ckpts[best_model_name], device,
                                      head_type=head, use_temperature=use_temp)

    # Get data for best model
    if best_model_name in per_model_data:
        true_p_best, pred_q_best, hard_labels_best, images_best = per_model_data[best_model_name]
    else:
        # Need to collect data
        best_model.eval()
        all_preds, all_true, all_hard, all_images = [], [], [], []
        with torch.no_grad():
            for images, soft_labels, hard_labels in test_loader:
                images = images.to(device)
                soft_labels = soft_labels.to(device)
                pred_q = best_model(images)
                all_preds.append(pred_q.cpu())
                all_true.append(soft_labels.cpu())
                all_hard.append(hard_labels)
                all_images.append(images.cpu())
        pred_q_best = torch.cat(all_preds, dim=0)
        true_p_best = torch.cat(all_true, dim=0)
        hard_labels_best = torch.cat(all_hard, dim=0)
        images_best = torch.cat(all_images, dim=0)

    # Per-class KL
    pkl_dict = per_class_kl(true_p_best, pred_q_best, hard_labels_best)
    plot_per_class_kl_bar(pkl_dict, label=best_model_name,
                          save_path="outputs/eval/aggregated/per_class_kl.png")

    # Annotator subsampling robustness
    counts_path = "data/cifar10h-counts.npy"
    if os.path.exists(counts_path):
        raw_counts_full = np.load(counts_path)
        test_indices = splits['test'].indices
        raw_counts = raw_counts_full[test_indices]
        print("Annotator subsampling check running...")
        sub_results = annotator_subsampling_check(raw_counts, pred_q_best)
        plot_robustness_curves_eval(subsampling_results=sub_results,
                                    save_path="outputs/eval/aggregated/robustness_subsampling.png")
    else:
        print("cifar10h-counts.npy not found. Skipping annotator subsampling.")

    # OOD corruption
    print("Running OOD corruption check on best model...")
    ood_results = ood_corruption_check(best_model, images_best,
                                       severities=[0.05, 0.1, 0.2, 0.3, 0.5])
    plot_robustness_curves_eval(ood_results=ood_results,
                                save_path="outputs/eval/aggregated/robustness_ood.png")

    print("\nAll evaluation plots are in outputs/eval/aggregated/ and outputs/eval/representative/.")


if __name__ == "__main__":
    main()