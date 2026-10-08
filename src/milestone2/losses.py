"""Classification losses used in the Milestone 2 comparison."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss operating on raw logits."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        losses = -(1.0 - target_probabilities).pow(self.gamma) * target_log_probabilities
        if self.class_weights is not None:
            losses = losses * self.class_weights[targets]
        return losses.mean()


def build_loss(
    name: str,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.10,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Create one of the three required classification losses."""
    key = name.lower()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if key == "label_smoothing":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if key == "focal":
        return FocalLoss(class_weights=class_weights, gamma=focal_gamma)
    raise ValueError(
        f"Unsupported loss '{name}'. Choices: cross_entropy, label_smoothing, focal"
    )
