"""Reusable orchestration for one controlled MLP experiment."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from .data import make_dataloaders
from .losses import build_loss
from .models import MLPClassifier, count_trainable_parameters
from .training import FitResult, build_optimizer, fit_model, set_global_seed


@dataclass
class ExperimentArtifacts:
    model: MLPClassifier
    fit: FitResult
    parameter_count: int


def train_configuration(
    *,
    split_manifest: pd.DataFrame,
    image_cache: torch.Tensor,
    label_to_index: dict[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    class_weights: torch.Tensor,
    hidden_dims: list[int],
    input_dim: int,
    loss_name: str,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    activation: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    label_smoothing: float = 0.10,
    focal_gamma: float = 2.0,
    verbose: bool = True,
) -> ExperimentArtifacts:
    """Train one configuration without touching the test partition."""
    set_global_seed(seed)
    loaders = make_dataloaders(
        split_manifest=split_manifest,
        image_cache=image_cache,
        label_to_index=label_to_index,
        mean=mean,
        std=std,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
        include_test=False,
    )
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=len(label_to_index),
        activation=activation,
        dropout=dropout,
        batch_norm=True,
    )
    criterion = build_loss(
        name=loss_name,
        class_weights=class_weights.to(device),
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )
    optimizer = build_optimizer(
        name=optimizer_name,
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    fit = fit_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
    )
    return ExperimentArtifacts(
        model=model,
        fit=fit,
        parameter_count=count_trainable_parameters(model),
    )
