"""Reusable components for ICS 471 Milestone 3 (CNNs and transfer learning)."""

from .data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_augmentation,
    build_normalization,
    build_transform_pair,
    load_or_build_cache,
    make_loaders,
)
from .experiments import ExperimentStore
from .models import ConvBlock, SkinLesionCNN
from .summary import cnn_block_summary, draw_architecture, model_summary
from .training import (
    MONITOR_MODES,
    EarlyStopping,
    TrainingResult,
    evaluate,
    fit,
    load_checkpoint,
    run_epoch,
    save_checkpoint,
    trainable_state_keys,
)
from .transfer import (
    BACKBONES,
    STRATEGIES,
    TransferClassifier,
    build_extended_head,
    build_transfer_model,
    extract_features,
    parameter_report,
    trainable_layer_table,
    weight_layers,
)

__all__ = [
    "BACKBONES",
    "ConvBlock",
    "EarlyStopping",
    "ExperimentStore",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MONITOR_MODES",
    "STRATEGIES",
    "SkinLesionCNN",
    "TrainingResult",
    "TransferClassifier",
    "build_augmentation",
    "build_extended_head",
    "build_normalization",
    "build_transfer_model",
    "build_transform_pair",
    "cnn_block_summary",
    "draw_architecture",
    "evaluate",
    "extract_features",
    "fit",
    "load_checkpoint",
    "load_or_build_cache",
    "make_loaders",
    "model_summary",
    "parameter_report",
    "run_epoch",
    "save_checkpoint",
    "trainable_layer_table",
    "trainable_state_keys",
    "weight_layers",
]
