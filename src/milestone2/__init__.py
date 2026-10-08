"""Reusable components for ICS 471 Milestone 2."""

from .data import (
    CLASS_NAMES,
    CLASS_ORDER,
    CachedHAM10000Dataset,
    build_image_cache,
    build_transforms,
    compute_channel_stats,
    compute_class_weights,
    fit_label_mapping,
    index_images,
    make_dataloaders,
    validate_image_coverage,
)
from .experiments import ExperimentArtifacts, train_configuration
from .losses import FocalLoss, build_loss
from .models import MLPClassifier, count_trainable_parameters
from .splitting import (
    create_milestone2_splits,
    save_split_manifest,
    split_class_table,
    split_summary,
    validate_metadata,
    validate_split_manifest,
)
from .training import (
    FitResult,
    build_optimizer,
    evaluate_model,
    fit_model,
    load_checkpoint,
    save_checkpoint,
    set_global_seed,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_ORDER",
    "CachedHAM10000Dataset",
    "ExperimentArtifacts",
    "FitResult",
    "FocalLoss",
    "MLPClassifier",
    "build_image_cache",
    "build_loss",
    "build_optimizer",
    "build_transforms",
    "compute_channel_stats",
    "compute_class_weights",
    "count_trainable_parameters",
    "create_milestone2_splits",
    "evaluate_model",
    "fit_label_mapping",
    "fit_model",
    "index_images",
    "load_checkpoint",
    "make_dataloaders",
    "save_checkpoint",
    "save_split_manifest",
    "set_global_seed",
    "split_class_table",
    "split_summary",
    "train_configuration",
    "validate_image_coverage",
    "validate_metadata",
    "validate_split_manifest",
]
