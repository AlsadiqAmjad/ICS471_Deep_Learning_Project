"""Resumable experiment bookkeeping: one log, checkpoint, and summary per run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from torch import nn

from .training import TrainingResult, save_checkpoint


class ExperimentStore:
    """Files for run ``name``: ``logs/<name>.csv``, ``checkpoints/<name>.pt``, ``summaries/<name>.json``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.log_dir = root / "logs"
        self.checkpoint_dir = root / "checkpoints"
        self.summary_dir = root / "summaries"
        for directory in (self.log_dir, self.checkpoint_dir, self.summary_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def log_path(self, name: str) -> Path:
        return self.log_dir / f"{name}.csv"

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.pt"

    def summary_path(self, name: str) -> Path:
        return self.summary_dir / f"{name}.json"

    def is_complete(self, name: str, require_checkpoint: bool = True) -> bool:
        if not self.summary_path(name).exists():
            return False
        return self.checkpoint_path(name).exists() or not require_checkpoint

    def record(
        self,
        name: str,
        model: nn.Module,
        result: TrainingResult,
        config: dict[str, Any],
        trainable_only: bool = False,
        save_weights: bool = True,
    ) -> dict[str, Any]:
        """Persist the log, the best checkpoint, and a JSON summary of the run."""
        result.history.to_csv(self.log_path(name), index=False)
        if save_weights:
            save_checkpoint(self.checkpoint_path(name), model, result, config, trainable_only)
        best = result.best_row()
        summary = {
            "name": name,
            **config,
            "monitor": result.monitor,
            "best_epoch": int(result.best_epoch),
            "stopped_epoch": int(result.stopped_epoch),
            "best_monitor_value": float(result.best_value),
            "train_loss": float(best["train_loss"]),
            "train_accuracy": float(best["train_accuracy"]),
            "validation_loss": float(best["validation_loss"]),
            "validation_accuracy": float(best["validation_accuracy"]),
            "validation_macro_f1": float(best["validation_macro_f1"]),
            "elapsed_seconds": float(result.elapsed_seconds),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
        self.summary_path(name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def summary(self, name: str) -> dict[str, Any]:
        return json.loads(self.summary_path(name).read_text(encoding="utf-8"))

    def history(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.log_path(name))

    def table(self, names: list[str]) -> pd.DataFrame:
        return pd.DataFrame([self.summary(name) for name in names])
