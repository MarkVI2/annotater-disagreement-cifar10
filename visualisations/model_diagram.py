"""
visualisations/model_diagram.py
=================================
Generates an architecture diagram (matplotlib, no external tools needed)
and a parameter-count summary table.

Functions:
    draw_architecture_diagram   – block-diagram PNG of backbone → head → output
    print_model_summary         – pretty-print + PNG table of param counts

WHERE TO CALL:
    from visualisations.model_diagram import (
        draw_architecture_diagram, print_model_summary
    )
    draw_architecture_diagram(save_path="outputs/plots/architecture.png")
    print_model_summary(model, save_path="outputs/plots/param_table.png")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


os.makedirs("outputs/plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Architecture block diagram
# ─────────────────────────────────────────────────────────────────────────────
def draw_architecture_diagram(
    head_type: str = "mlp",
    save_path: str = "outputs/plots/architecture.png",
) -> None:
    """
    Draw a clear block diagram:
        [Input 32×32 RGB] → [ResNet-18 Backbone] → [Global Avg Pool]
        → [Linear / MLP head] → [Softmax] → [10-d distribution q(y|x)]

    Args:
        head_type : "linear" or "mlp"
        save_path : output PNG
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    # ── Blocks: (x_center, y_center, width, height, label, sub, color)
    blocks = [
        (1.1,  2.0, 1.6, 1.2, "Input",          "32×32 RGB",          "#90CAF9"),
        (3.2,  2.0, 2.0, 1.2, "ResNet-18",       "Backbone\n(pretrained)",   "#64B5F6"),
        (5.7,  2.0, 1.8, 1.2, "Global\nAvg Pool","512-d feature",      "#42A5F5"),
        (8.1,  2.0, 2.0, 1.2,
         "MLP Head" if head_type == "mlp" else "Linear Head",
         "512→256→10\n(ReLU + Dropout)" if head_type == "mlp" else "512→10",
         "#1E88E5"),
        (10.5, 2.0, 1.6, 1.2, "Softmax",         "Temperature τ",      "#1565C0"),
        (12.7, 2.0, 1.8, 1.2, "Output",          "q(y|x) ∈ Δ¹⁰",      "#FF8F00"),
    ]

    for (cx, cy, w, h, title, sub, color) in blocks:
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor="#333", facecolor=color, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.12, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=7, color="white", alpha=0.9)

    # ── Arrows between blocks
    arrow_style = dict(
        arrowstyle="->", color="#555", lw=1.6,
        connectionstyle="arc3,rad=0.0",
    )
    xs = [b[0] for b in blocks]
    for i in range(len(xs) - 1):
        x0 = xs[i]  + blocks[i][2]  / 2
        x1 = xs[i+1] - blocks[i+1][2] / 2
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=arrow_style)

    # ── Loss label below
    ax.text(7.0, 0.5,
            "Training target: KL / JSD / Custom (KL + entropy error) / EMD  →  minimize divergence from  p(y|x)",
            ha="center", va="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FBC02D"))

    ax.set_title(
        f"Model Architecture — ResNet-18 Backbone + {'MLP' if head_type == 'mlp' else 'Linear'} Head",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model_diagram] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parameter-count summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_model_summary(
    model=None,
    save_path: str = "outputs/plots/param_table.png",
    extra_rows: list[dict] | None = None,
) -> None:
    """
    Print a param-count table to console and save as a PNG.

    Args:
        model      : a PyTorch nn.Module; if None, uses hard-coded typical values
        save_path  : output PNG
        extra_rows : list of dicts { "Model": ..., "Backbone": ...,
                                     "Head": ..., "Total Params": ..., "Trainable": ... }
                     to append (for comparing variants without loading each model).

    WHERE TO CALL:
        from models.cifar_resnet import build_soft_label_model
        m = build_soft_label_model(head_type="mlp", pretrained=True)
        print_model_summary(m, extra_rows=[
            {"Model": "ResNet-18 Linear", "Backbone": "11.17M",
             "Head": "5.12K", "Total Params": "11.18M", "Trainable": "11.18M"},
        ])
    """
    rows = []

    if model is not None:
        import torch
        total   = sum(p.numel() for p in model.parameters())
        trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append({
            "Model":        "Current model",
            "Total Params": f"{total / 1e6:.3f}M",
            "Trainable":    f"{trained / 1e6:.3f}M",
        })

    # Typical ResNet-18 values for the report
    if not rows and extra_rows is None:
        rows = [
            {"Model": "ResNet-18 + Linear head",  "Backbone": "11.17 M", "Head": "5.1 K",  "Total": "11.18 M", "Trainable": "11.18 M"},
            {"Model": "ResNet-18 + MLP head",      "Backbone": "11.17 M", "Head": "132.9 K","Total": "11.30 M", "Trainable": "11.30 M"},
            {"Model": "ResNet-18 (frozen bb) + MLP","Backbone": "11.17 M", "Head": "132.9 K","Total": "11.30 M", "Trainable": "132.9 K"},
        ]

    if extra_rows:
        rows.extend(extra_rows)

    # Console print
    if rows:
        keys = list(rows[0].keys())
        col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
        header = "  ".join(k.ljust(col_widths[k]) for k in keys)
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for r in rows:
            print("  ".join(str(r.get(k, "")).ljust(col_widths[k]) for k in keys))
        print("=" * len(header) + "\n")

    # PNG table
    col_labels = list(rows[0].keys()) if rows else []
    cell_text  = [[str(r.get(k, "")) for k in col_labels] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 2), len(rows) * 0.6 + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=["#1E88E5"] * len(col_labels),
    )
    for (row_i, col_i), cell in table.get_celld().items():
        if row_i == 0:
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_fontsize(9)
    table.auto_set_font_size(False)
    table.scale(1, 1.6)
    ax.set_title("Model Parameter Summary", fontsize=11, pad=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model_diagram] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    draw_architecture_diagram(head_type="mlp",
                               save_path="outputs/plots/arch_mlp.png")
    draw_architecture_diagram(head_type="linear",
                               save_path="outputs/plots/arch_linear.png")
    print_model_summary(save_path="outputs/plots/param_table.png")
    print("Self-test complete — check outputs/plots/")
