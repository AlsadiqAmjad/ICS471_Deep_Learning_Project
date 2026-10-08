"""Reproducible training, evaluation, optimization, and checkpoint helpers."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


@dataclass
class FitResult:
    history: pd.DataFrame
    best_epoch: int
    best_validation_macro_f1: float
    best_validation_loss: float
    best_state_dict: dict[str, torch.Tensor]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(
    name: str,
    parameters: object,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if key == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            alpha=0.99,
            weight_decay=weight_decay,
        )
    if key == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{name}'. Choices: sgd, rmsprop, adam")


def _metrics(targets: list[int], predictions: list[int]) -> tuple[float, float]:
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return float(accuracy), float(macro_f1)


def _run_epoch(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Encountered non-finite loss: {float(loss.detach())}. "
                    "Stop the experiment and inspect the learning rate and inputs."
                )
            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if total_examples == 0:
        raise ValueError("The DataLoader yielded no examples.")
    accuracy, macro_f1 = _metrics(all_targets, all_predictions)
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "targets": np.asarray(all_targets),
        "predictions": np.asarray(all_predictions),
    }


def evaluate_model(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return _run_epoch(model, loader, criterion, device, optimizer=None)


def fit_model(
    model: nn.Module,
    train_loader: object,
    validation_loader: object,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    minimum_improvement: float = 1e-4,
    verbose: bool = True,
) -> FitResult:
    """Train with early stopping selected by validation macro-F1."""
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive.")
    model.to(device)
    criterion.to(device)

    rows: list[dict[str, float | int]] = []
    best_epoch = 0
    best_f1 = -np.inf
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation = evaluate_model(model, validation_loader, criterion, device)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "train_macro_f1": train["macro_f1"],
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )

        improved_f1 = validation["macro_f1"] > best_f1 + minimum_improvement
        tied_f1 = abs(validation["macro_f1"] - best_f1) <= minimum_improvement
        improved_loss = validation["loss"] < best_loss
        if improved_f1 or (tied_f1 and improved_loss):
            best_epoch = epoch
            best_f1 = validation["macro_f1"]
            best_loss = validation["loss"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:02d}/{max_epochs} | "
                f"train loss {train['loss']:.4f}, F1 {train['macro_f1']:.4f} | "
                f"val loss {validation['loss']:.4f}, F1 {validation['macro_f1']:.4f}"
            )
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after epoch {epoch}; best epoch: {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")
    model.load_state_dict(best_state)
    return FitResult(
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_f1),
        best_validation_loss=float(best_loss),
        best_state_dict=best_state,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    fit_result: FitResult,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": copy.deepcopy(fit_result.best_state_dict),
            "history": fit_result.history.to_dict(orient="list"),
            "best_epoch": fit_result.best_epoch,
            "best_validation_macro_f1": fit_result.best_validation_macro_f1,
            "best_validation_loss": fit_result.best_validation_loss,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
