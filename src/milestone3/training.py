"""Training with a configurable early-stopping monitor, mixed precision, and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


MONITOR_MODES = {
    "train_loss": "min",
    "train_accuracy": "max",
    "train_macro_f1": "max",
    "validation_loss": "min",
    "validation_accuracy": "max",
    "validation_macro_f1": "max",
}


@dataclass
class EarlyStopping:
    """Stop when ``monitor`` has not improved by ``min_delta`` for ``patience`` epochs.

    The weights from the best monitored epoch are restored at the end of training.
    """

    monitor: str = "validation_macro_f1"
    patience: int = 8
    min_delta: float = 1e-4
    mode: str = field(init=False)

    def __post_init__(self) -> None:
        if self.monitor not in MONITOR_MODES:
            raise ValueError(f"Unsupported monitor '{self.monitor}'. Choices: {sorted(MONITOR_MODES)}")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        self.mode = MONITOR_MODES[self.monitor]

    def initial(self) -> float:
        return -np.inf if self.mode == "max" else np.inf

    def is_improvement(self, value: float, best: float) -> bool:
        if self.mode == "max":
            return value > best + self.min_delta
        return value < best - self.min_delta


@dataclass
class TrainingResult:
    history: pd.DataFrame
    monitor: str
    best_epoch: int
    best_value: float
    stopped_epoch: int
    elapsed_seconds: float
    best_state_dict: dict[str, torch.Tensor]

    def best_row(self) -> pd.Series:
        return self.history.loc[self.history["epoch"] == self.best_epoch].iloc[0]


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def run_epoch(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = True,
    return_probabilities: bool = False,
) -> dict[str, Any]:
    """One pass over ``loader``; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_examples = 0.0, 0
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(inputs)
            logits = logits.float()
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {float(loss.detach())}; check the learning rate.")
            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            targets_all.append(targets.detach().cpu().numpy())
            predictions_all.append(logits.argmax(dim=1).detach().cpu().numpy())
            if return_probabilities:
                probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    if total_examples == 0:
        raise ValueError("The DataLoader yielded no examples.")
    targets_np = np.concatenate(targets_all)
    predictions_np = np.concatenate(predictions_all)
    result = {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(targets_np, predictions_np)),
        "macro_f1": float(f1_score(targets_np, predictions_np, average="macro", zero_division=0)),
        "targets": targets_np,
        "predictions": predictions_np,
    }
    if return_probabilities:
        result["probabilities"] = np.concatenate(probabilities_all)
    return result


def evaluate(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, Any]:
    model.to(device)
    return run_epoch(model, loader, criterion, device, None, use_amp, return_probabilities=True)


def fit(
    model: nn.Module,
    train_loader: object,
    validation_loader: object,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    early_stopping: EarlyStopping,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    log_path: Path | None = None,
    use_amp: bool = True,
    verbose: bool = True,
    scheduler_metric: str = "validation_loss",
) -> TrainingResult:
    """Train, log every epoch, and restore the weights of the best monitored epoch.

    A ``ReduceLROnPlateau`` scheduler is stepped with ``scheduler_metric``; any other
    scheduler is stepped once per epoch.
    """
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if scheduler_metric not in MONITOR_MODES:
        raise ValueError(f"Unsupported scheduler metric '{scheduler_metric}'.")
    model.to(device)
    criterion.to(device)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    best_value = early_stopping.initial()
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, criterion, device, optimizer, use_amp)
        validation = run_epoch(model, validation_loader, criterion, device, None, use_amp)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "train_macro_f1": train["macro_f1"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        rows.append(row)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(row[scheduler_metric])
        elif scheduler is not None:
            scheduler.step()
        if log_path is not None:
            pd.DataFrame(rows).to_csv(log_path, index=False)

        value = row[early_stopping.monitor]
        marker = ""
        if early_stopping.is_improvement(value, best_value):
            best_value, best_epoch, stale_epochs = value, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            marker = " *"
        else:
            stale_epochs += 1

        if verbose:
            print(
                f"Epoch {epoch:02d}/{max_epochs} | lr {learning_rate:.2e} | "
                f"train loss {train['loss']:.4f} acc {train['accuracy']:.4f} F1 {train['macro_f1']:.4f} | "
                f"val loss {validation['loss']:.4f} acc {validation['accuracy']:.4f} "
                f"F1 {validation['macro_f1']:.4f} | {row['epoch_seconds']:.1f}s{marker}"
            )
        if stale_epochs >= early_stopping.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}: {early_stopping.monitor} "
                      f"did not improve for {early_stopping.patience} epochs (best epoch {best_epoch}).")
            break

    if best_state is None:
        raise RuntimeError("Training finished without a checkpoint.")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=pd.DataFrame(rows),
        monitor=early_stopping.monitor,
        best_epoch=best_epoch,
        best_value=float(best_value),
        stopped_epoch=int(rows[-1]["epoch"]),
        elapsed_seconds=time.perf_counter() - started,
        best_state_dict=best_state,
    )


def trainable_state_keys(model: nn.Module) -> set[str]:
    """State-dict keys of every module that owns at least one trainable parameter."""
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False))
        if any(parameter.requires_grad for _, parameter in own):
            prefix = f"{module_name}." if module_name else ""
            keys.update(prefix + name for name, _ in own)
            keys.update(prefix + name for name, _ in module.named_buffers(recurse=False))
    return keys


def save_checkpoint(
    path: Path,
    model: nn.Module,
    result: TrainingResult,
    metadata: dict[str, Any],
    trainable_only: bool = False,
) -> None:
    """Save the restored best weights (optionally only trainable modules) and the log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = result.best_state_dict
    if trainable_only:
        keep = trainable_state_keys(model)
        state = {key: value for key, value in state.items() if key in keep}
    torch.save(
        {
            "model_state_dict": state,
            "trainable_only": trainable_only,
            "history": result.history.to_dict(orient="list"),
            "monitor": result.monitor,
            "best_epoch": result.best_epoch,
            "best_value": result.best_value,
            "stopped_epoch": result.stopped_epoch,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("trainable_only"):
        missing_from_model = set(checkpoint["model_state_dict"]) - set(model.state_dict())
        if missing_from_model:
            raise KeyError(f"Checkpoint keys not found in model: {sorted(missing_from_model)[:5]}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
