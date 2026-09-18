from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SEED = 471
N_FOLDS = 20
TRAIN_FOLDS = set(range(0, 14))
VALIDATION_FOLDS = set(range(14, 17))
TEST_FOLDS = set(range(17, 20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-aware HAM10000 splits.")
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/HAM10000_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    return parser.parse_args()


def validate_metadata(df: pd.DataFrame) -> None:
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    diagnoses_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    if (diagnoses_per_lesion > 1).any():
        bad = diagnoses_per_lesion[diagnoses_per_lesion > 1].index.tolist()[:10]
        raise ValueError(f"Some lesion IDs have more than one diagnosis: {bad}")


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold, (_, held_out_indices) in enumerate(
        splitter.split(df.index.to_numpy(), y=df["dx"], groups=df["lesion_id"])
    ):
        fold_ids[held_out_indices] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("At least one row was not assigned to a fold.")

    result = df.copy()
    result["fold"] = fold_ids
    result["split"] = result["fold"].map(
        lambda fold: "train"
        if fold in TRAIN_FOLDS
        else "validation"
        if fold in VALIDATION_FOLDS
        else "test"
    )
    return result


def validate_splits(df: pd.DataFrame) -> None:
    names = ("train", "validation", "test")
    lesion_sets = {name: set(df.loc[df["split"] == name, "lesion_id"]) for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = lesion_sets[left] & lesion_sets[right]
            if overlap:
                raise RuntimeError(f"Lesion leakage between {left} and {right}: {len(overlap)} groups")

    all_classes = set(df["dx"].unique())
    for name in names:
        present = set(df.loc[df["split"] == name, "dx"].unique())
        missing = all_classes - present
        if missing:
            raise RuntimeError(f"Split {name} is missing classes: {sorted(missing)}")


def print_summary(df: pd.DataFrame) -> None:
    totals = df["split"].value_counts().reindex(["train", "validation", "test"])
    print("Image counts")
    for name, count in totals.items():
        print(f"  {name:10s} {count:5d} ({100 * count / len(df):5.2f}%)")
    print("\nClass counts")
    print(pd.crosstab(df["split"], df["dx"]).reindex(["train", "validation", "test"]))


def main() -> None:
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    validate_metadata(metadata)
    result = assign_folds(metadata)
    validate_splits(result)
    print_summary(result)

    result.to_csv(args.output / "all_splits.csv", index=False)
    for name in ("train", "validation", "test"):
        result[result["split"] == name].to_csv(args.output / f"{name}.csv", index=False)
    print(f"\nSaved split manifests to {args.output.resolve()}")


if __name__ == "__main__":
    main()
