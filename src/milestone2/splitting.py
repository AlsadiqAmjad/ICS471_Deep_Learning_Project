"""Leakage-aware, stratified train/validation/test splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}
SPLIT_ORDER = ("train", "validation", "test")


def validate_metadata(frame: pd.DataFrame) -> None:
    """Validate the metadata fields required by this project."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required metadata columns contain missing values.")
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"Duplicate image IDs found in metadata: {duplicates}")

    labels_per_lesion = frame.groupby("lesion_id", observed=True)["dx"].nunique()
    if (labels_per_lesion > 1).any():
        bad = labels_per_lesion[labels_per_lesion > 1].index[:5].tolist()
        raise ValueError(f"Lesion IDs with conflicting diagnoses: {bad}")


def create_milestone2_splits(
    metadata: pd.DataFrame,
    outer_seed: int = 471,
    inner_seed: int = 472,
) -> pd.DataFrame:
    """Create an approximately 64/16/20 group-stratified split.

    The outer five-fold split holds out one fold (about 20%) for testing.
    The inner five-fold split holds out one fold (20% of the remaining 80%)
    for validation. All images sharing a lesion_id remain together.
    """
    validate_metadata(metadata)
    result = metadata.reset_index(drop=True).copy()
    result["cache_index"] = np.arange(len(result), dtype=np.int64)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    development_positions, test_positions = next(
        outer.split(result, y=result["dx"], groups=result["lesion_id"])
    )

    development = result.iloc[development_positions]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)
    train_relative, validation_relative = next(
        inner.split(
            development,
            y=development["dx"],
            groups=development["lesion_id"],
        )
    )
    train_positions = development_positions[train_relative]
    validation_positions = development_positions[validation_relative]

    result["split"] = ""
    split_column = result.columns.get_loc("split")
    result.iloc[train_positions, split_column] = "train"
    result.iloc[validation_positions, split_column] = "validation"
    result.iloc[test_positions, split_column] = "test"

    validate_split_manifest(result)
    return result


def validate_split_manifest(frame: pd.DataFrame) -> None:
    """Fail if a split is incomplete, misses a class, or leaks a lesion."""
    validate_metadata(frame)
    if "split" not in frame.columns:
        raise ValueError("Split manifest has no 'split' column.")
    unknown = set(frame["split"].unique()) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"Unknown or empty split labels: {sorted(unknown)}")

    all_classes = set(frame["dx"].unique())
    lesion_sets: dict[str, set[str]] = {}
    for split_name in SPLIT_ORDER:
        subset = frame[frame["split"] == split_name]
        if subset.empty:
            raise ValueError(f"Split '{split_name}' is empty.")
        missing_classes = all_classes - set(subset["dx"].unique())
        if missing_classes:
            raise ValueError(
                f"Split '{split_name}' is missing classes: {sorted(missing_classes)}"
            )
        lesion_sets[split_name] = set(subset["lesion_id"])

    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1 :]:
            overlap = lesion_sets[left] & lesion_sets[right]
            if overlap:
                raise ValueError(
                    f"Lesion leakage between {left} and {right}: {len(overlap)} groups"
                )


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return image and lesion counts plus overall percentages."""
    rows = []
    for name in SPLIT_ORDER:
        subset = frame[frame["split"] == name]
        rows.append(
            {
                "split": name,
                "images": len(subset),
                "image_percentage": 100.0 * len(subset) / len(frame),
                "lesions": subset["lesion_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def split_class_table(frame: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """Return class counts or row-wise percentages for each split."""
    table = pd.crosstab(frame["split"], frame["dx"]).reindex(SPLIT_ORDER)
    if normalize:
        table = table.div(table.sum(axis=1), axis=0).mul(100.0)
    return table


def save_split_manifest(frame: pd.DataFrame, output_directory: Path) -> None:
    """Save the full manifest and one CSV file per partition."""
    validate_split_manifest(frame)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_directory / "all_splits.csv", index=False)
    for name in SPLIT_ORDER:
        frame[frame["split"] == name].to_csv(
            output_directory / f"{name}.csv", index=False
        )
