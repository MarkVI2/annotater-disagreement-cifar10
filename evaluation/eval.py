import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cifar_resnet import build_soft_label_model
from data.pipeline import load_full_dataset, create_splits, create_dataloaders
from evaluation.metrics import run_all_metrics, compute_entropy
from utils.device import get_device
from evaluation.robustness import (
    annotator_subsampling_check,
    ood_corruption_check,
    per_class_kl
)
from evaluation.visualize import (
    plot_entropy_scatter,
    plot_metrics_comparison,
    plot_qualitative_examples,
    plot_ablation_summary,
)
from analysis.cifar10h_analysis import (
    plot_entropy_analysis,
    plot_distribution_comparison,
    plot_gradcam_grid,
    plot_failure_cases,
    plot_robustness_curves,
    plot_per_class_entropy_calibration,
    plot_entropy_reliability_diagram,
    plot_training_curves,
    GradCAM,
)

# Map your checkpoints to their file paths. Adjust as needed.
CHECKPOINTS = {
    "kl_linear_pt_cifar10":   "outputs/checkpoints/kl_linear_pt_cifar10_best.pth",
    "kl_mlp_pt_cifar10":      "outputs/checkpoints/kl_mlp_pt_cifar10_best.pth",
    "js_linear_pt_cifar10":   "outputs/checkpoints/js_linear_pt_cifar10_best.pth",
    "js_mlp_pt_cifar10":      "outputs/checkpoints/js_mlp_pt_cifar10_best.pth",
    "custom_linear_pt_cifar10": "outputs/checkpoints/custom_composite_linear_pt_cifar10_best.pth",
    "custom_mlp_pt_cifar10":    "outputs/checkpoints/custom_composite_mlp_pt_cifar10_best.pth",
    # Add temperature variants if trained: e.g., "custom_mlp_pt_cifar10_temp": ...
}

def load_model_from_ckpt(ckpt_path, device, head_type='linear', use_temperature=False):
    model = build_soft_label_model(head_type=head_type, pretrained_type='random',
                                   use_temperature=use_temperature)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state_dict']
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model

def run_analysis(model, test_loader, device, output_dir='outputs/eval'):
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

    # Denormalize images for Grad-CAM and display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
    images_raw = (raw_images * std + mean).clamp(0,1).permute(0,2,3,1).numpy()

    # Core metrics
    metrics = run_all_metrics(true_p, pred_q)
    print("\n=== Test Metrics ===")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Required plots
    plot_entropy_scatter(true_p, pred_q, save_path=os.path.join(output_dir, "entropy_scatter.png"))
    plot_qualitative_examples(raw_images.permute(0,3,1,2)[:1000], true_p[:1000], pred_q[:1000],
                              compute_entropy(true_p[:1000]), save_path=os.path.join(output_dir, "qualitative_grid.png"))
    plot_failure_cases(images_raw, true_p.numpy(), pred_q.numpy(), save_path=os.path.join(output_dir, "failure_cases.png"))
    plot_entropy_analysis(true_p.numpy(), pred_q.numpy(), save_path=os.path.join(output_dir, "entropy_analysis.png"))
    plot_distribution_comparison(images_raw[:1000], true_p.numpy()[:1000], pred_q.numpy()[:1000],
                                 save_path=os.path.join(output_dir, "distribution_comparison.png"))

    # Grad-CAM (requires target layer)
    backbone = model.backbone
    target_layer = backbone.layer4[-1]  # last basic block
    plot_gradcam_grid(model, target_layer, raw_images[:1000], images_raw[:1000], true_p.numpy()[:1000],
                      pred_q.numpy()[:1000], save_path=os.path.join(output_dir, "gradcam_grid.png"))

    # Robustness: annotation subsampling
    # Need raw counts – we don't have them in this script. We'll skip if not available.
    # But we can still call ood corruption and per-class KL.
    print("\n=== OOD Corruption Check ===")
    ood_results = ood_corruption_check(model, raw_images, severities=[0.05,0.1,0.2,0.3,0.5])
    plot_robustness_curves(model, raw_images, device, save_path=os.path.join(output_dir, "robustness_entropy.png"))

    print("\n=== Per-Class KL ===")
    per_class_kl(true_p, pred_q, hard_labels)

    # Advanced plots
    plot_per_class_entropy_calibration(true_p.numpy(), pred_q.numpy(),
                                       save_path=os.path.join(output_dir, "per_class_entropy_cal.png"))
    plot_entropy_reliability_diagram(true_p.numpy(), pred_q.numpy(),
                                    save_path=os.path.join(output_dir, "entropy_reliability.png"))

def main():
    device = get_device()
    # Load test data
    dataset = load_full_dataset()
    splits = create_splits(dataset)
    loaders = create_dataloaders(splits)
    test_loader = loaders['test']

    # Evaluate each model
    results_summary = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping {name}")
            continue
        print(f"\n====== Evaluating {name} ======")
        head = 'mlp' if 'mlp' in name else 'linear'
        use_temp = '_temp' in name
        model = load_model_from_ckpt(ckpt_path, device, head, use_temperature=use_temp)
        run_analysis(model, test_loader, device, output_dir=f'outputs/eval/{name}')

    # Global comparison plot (if multiple models evaluated)
    # You can collect metrics from each run and call plot_metrics_comparison() here.

if __name__ == '__main__':
    main()