import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cifar_resnet import build_soft_label_model
from data.pipeline import load_full_dataset, create_splits, create_dataloaders
from evaluation.metrics import run_all_metrics, compute_entropy
from utils.device import get_device
from evaluation.robustness import ood_corruption_check, per_class_kl
from evaluation.visualize import (
    plot_entropy_scatter,
    plot_metrics_comparison,
    plot_qualitative_examples,
    plot_ablation_summary,
)

CHECKPOINTS = {
    "kl_linear":               "outputs/checkpoints/kl_linear/best.pth",
    "kl_mlp":                  "outputs/checkpoints/kl_mlp/best.pth",
    "js_linear":               "outputs/checkpoints/js_linear/best.pth",
    "js_mlp":                  "outputs/checkpoints/js_mlp/best.pth",
    "custom_composite_linear": "outputs/checkpoints/custom_composite_linear/best.pth",
    "custom_composite_mlp":    "outputs/checkpoints/custom_composite_mlp/best.pth",
    "emd_linear":              "outputs/checkpoints/emd_linear/best.pth",
    "emd_mlp":                 "outputs/checkpoints/emd_mlp/best.pth",
}


def load_model(checkpoint_path, head_type, device):
    model = build_soft_label_model(head_type=head_type, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = ckpt['model_state_dict']
    
    # Strip _orig_mod. prefix added by torch.compile()
    cleaned = {
        k.replace("_orig_mod.", ""): v 
        for k, v in state_dict.items()
    }
    
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    print(f"  Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f})")
    return model


def run_inference(model, test_loader, device):
    all_preds, all_true, all_hard, all_images = [], [], [], []
    with torch.no_grad():
        for images, soft_labels, hard_labels in test_loader:
            images      = images.to(device, non_blocking=True)
            soft_labels = soft_labels.to(device, non_blocking=True)
            pred_q      = model(images)
            all_preds.append(pred_q.cpu())
            all_true.append(soft_labels.cpu())
            all_hard.append(hard_labels)
            all_images.append(images.cpu())        
    return (
        torch.cat(all_preds,  dim=0),   # (N, 10)
        torch.cat(all_true,   dim=0),   # (N, 10)
        torch.cat(all_hard,   dim=0),   # (N,)
        torch.cat(all_images, dim=0),   # (N, 3, 32, 32)  
    )


def main():
    device = get_device()
    os.makedirs("outputs/eval", exist_ok=True)

    # Load test data 
    print("\n[1/4] Loading test data...")
    dataset  = load_full_dataset()
    splits   = create_splits(dataset)
    loaders  = create_dataloaders(splits)
    test_loader = loaders["test"]
    print(f"  Test set size: {len(splits['test'])} images")

    # Run eval for every checkpoint 
    print("\n[2/4] Running inference on all models...")
    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n  → {name}")
        head_type = "mlp" if "mlp" in name else "linear"
        model  = load_model(ckpt_path, head_type, device)
        pred_q, true_p, hard_labels, images = run_inference(model, test_loader, device)
        metrics = run_all_metrics(true_p, pred_q)
        all_results[name] = {
            "metrics":     metrics,
            "pred_q":      pred_q,
            "true_p":      true_p,
            "hard_labels": hard_labels,
            "images":      images,      
        }
        print(f"     KL={metrics['kl_divergence']:.4f}  "
              f"JSD={metrics['jsd']:.4f}  "
              f"Pearson={metrics['pearson_r']:.4f}  "
              f"P@100={metrics['precision@100']:.4f}")

    # Print full metrics table 
    print("\n[3/4] Full Metrics Table")
    print(f"{'Model':<30} {'KL':>7} {'JSD':>7} {'Cos':>7} {'Pear':>7} {'Spear':>7} {'P@100':>7} {'P@500':>7}")
    print("-" * 90)
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<30} {m['kl_divergence']:>7.4f} {m['jsd']:>7.4f} "
              f"{m['cosine_sim']:>7.4f} {m['pearson_r']:>7.4f} "
              f"{m['spearman_r']:>7.4f} {m['precision@100']:>7.4f} "
              f"{m['precision@500']:>7.4f}")

    # Generate plots
    print("\n[4/4] Generating plots...")

    # Pick best model by KL for scatter + qualitative plots
    best_name = min(all_results, key=lambda n: all_results[n]["metrics"]["kl_divergence"])
    print(f"  Best model by KL: {best_name}")
    best = all_results[best_name]

    plot_entropy_scatter(
        best["true_p"], best["pred_q"],
        save_path="outputs/eval/entropy_scatter.png"
    )
    plot_metrics_comparison(
        {name: res["metrics"] for name, res in all_results.items()},
        save_path="outputs/eval/metrics_comparison.png"
    )
    best_entropy = compute_entropy(best["true_p"])
    plot_qualitative_examples(
    best["images"], best["true_p"], best["pred_q"],   
    best_entropy,
    save_path="outputs/eval/qualitative_grid.png"
    )
    plot_ablation_summary(
        {name: res["metrics"]["kl_divergence"] for name, res in all_results.items()},
        save_path="outputs/eval/ablation_summary.png"
    )

    print("\n[Bonus 1] OOD Corruption Check (best model)...")
    ood_results = ood_corruption_check(
        model=load_model(CHECKPOINTS["kl_linear"], "linear", device),
        test_images=best["images"],
        severities=[0.05, 0.1, 0.2, 0.3, 0.5]
    )

    print("\n[Bonus 2] Per-class KL (best model)...")
    per_class_kl(best["true_p"], best["pred_q"], best["hard_labels"])

    print("\n Evaluation complete. Results saved to outputs/eval/") 

if __name__ == "__main__":
    main()