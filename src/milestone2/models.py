"""Flexible MLP architectures for image classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(activations)}")
    return activations[key]()


class MLPClassifier(nn.Module):
    """Configurable fully connected classifier with raw-logit output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.30,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim must be positive and num_classes must exceed one.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = [nn.Flatten()]
        previous_width = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous_width, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_width = width
        layers.append(nn.Linear(previous_width, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
