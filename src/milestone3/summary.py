"""Layer-by-layer model summaries and an architecture diagram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from torch import nn


def _parameters(module: nn.Module, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def model_summary(
    model: nn.Module,
    input_shape: tuple[int, ...],
    modules: dict[str, nn.Module] | None = None,
) -> pd.DataFrame:
    """Run one dummy forward pass and record each module's output shape and parameters.

    ``modules`` selects which modules to report (default: all leaf modules).
    """
    if modules is None:
        modules = {
            name: module
            for name, module in model.named_modules()
            if name and not list(module.children())
        }
    rows: list[dict[str, object]] = []
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            shape = tuple(output.shape[1:]) if isinstance(output, torch.Tensor) else None
            rows.append(
                {
                    "layer": name,
                    "type": module.__class__.__name__,
                    "output_shape": " x ".join(str(size) for size in shape) if shape else "",
                    "trainable_parameters": _parameters(module, trainable_only=True),
                    "total_parameters": _parameters(module, trainable_only=False),
                }
            )

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name, module)))
    was_training = model.training
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, *input_shape), device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return pd.DataFrame(rows)


def cnn_block_summary(model: nn.Module, input_shape: tuple[int, int, int]) -> pd.DataFrame:
    """Block-level summary for ``SkinLesionCNN`` (one row per stage)."""
    selected: dict[str, nn.Module] = {}
    for index, block in enumerate(model.features):
        selected[f"block{index + 1}"] = block
    selected["global_avg_pool"] = model.pool
    for name, module in model.classifier.named_children():
        selected[f"classifier.{name}"] = module
    summary = model_summary(model, input_shape, selected)
    descriptions = []
    for row in summary.itertuples():
        module = selected[row.layer]
        if row.layer.startswith("block"):
            convs = sum(isinstance(child, nn.Conv2d) for child in module)
            descriptions.append(f"{convs} x [Conv3x3 + BN + ReLU] + MaxPool2x2")
        elif isinstance(module, nn.Linear):
            descriptions.append(f"Linear {module.in_features} -> {module.out_features}")
        elif isinstance(module, nn.Dropout):
            descriptions.append(f"Dropout p={module.p}")
        else:
            descriptions.append(row.type)
    summary.insert(2, "description", descriptions)
    input_row = pd.DataFrame(
        [{"layer": "input", "type": "Input", "description": "RGB image",
          "output_shape": " x ".join(map(str, input_shape)),
          "trainable_parameters": 0, "total_parameters": 0}]
    )
    return pd.concat([input_row, summary], ignore_index=True)


def draw_architecture(summary: pd.DataFrame, title: str, path=None):
    """Draw a vertical flow diagram from ``cnn_block_summary`` output."""
    rows = summary.reset_index(drop=True)
    height = 0.62 * len(rows) + 1.2
    fig, axis = plt.subplots(figsize=(9, height), dpi=130)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, len(rows) + 0.6)
    axis.axis("off")

    colours = {
        "Input": "#E8E8E8",
        "ConvBlock": "#BBD5EE",
        "AdaptiveAvgPool2d": "#D7EBC9",
        "Flatten": "#F2F2F2",
        "Dropout": "#F6DDC7",
        "ReLU": "#F2F2F2",
        "Linear": "#F4C7C3",
    }
    for position, row in enumerate(rows.itertuples()):
        y = len(rows) - position - 0.2
        box = FancyBboxPatch(
            (0.4, y - 0.42), 9.2, 0.62,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=colours.get(row.type, "#FFFFFF"), edgecolor="#4A4A4A", linewidth=0.8,
        )
        axis.add_patch(box)
        axis.text(0.6, y - 0.11, row.layer, fontsize=9, weight="bold", va="center")
        axis.text(2.7, y - 0.11, row.description, fontsize=8.5, va="center")
        axis.text(7.0, y - 0.11, row.output_shape, fontsize=8.5, va="center", family="monospace")
        params = f"{row.trainable_parameters:,}" if row.trainable_parameters else "0"
        axis.text(9.45, y - 0.11, params, fontsize=8.5, va="center", ha="right", family="monospace")
        if position < len(rows) - 1:
            axis.annotate("", xy=(5.0, y - 0.56), xytext=(5.0, y - 0.42),
                          arrowprops={"arrowstyle": "->", "color": "#4A4A4A", "lw": 0.8})

    header_y = len(rows) + 0.35
    for x, label, align in ((0.6, "Stage", "left"), (2.7, "Operation", "left"),
                            (7.0, "Output shape", "left"), (9.45, "Params", "right")):
        axis.text(x, header_y, label, fontsize=9, weight="bold", ha=align, color="#333333")
    total = int(rows["trainable_parameters"].sum())
    axis.set_title(f"{title}\nTotal trainable parameters: {total:,}", fontsize=11, weight="bold")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    return fig
