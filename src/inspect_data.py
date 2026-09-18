from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
SEED = 471


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HAM10000 and create milestone figures.")
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/HAM10000_metadata.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--samples-per-class", type=int, default=2)
    return parser.parse_args()


def index_images(root: Path) -> dict[str, Path]:
    image_paths = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths[path.stem] = path
    return image_paths


def validate_metadata(df: pd.DataFrame) -> None:
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    unknown = set(df["dx"].dropna().unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown class codes: {sorted(unknown)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Required metadata columns contain missing values.")


def create_distribution(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    counts = df["dx"].value_counts().reindex(ORDER)
    summary = pd.DataFrame(
        {
            "code": counts.index,
            "category": [LABELS[code] for code in counts.index],
            "count": counts.values,
            "percentage": (100 * counts.values / len(df)).round(2),
        }
    )
    labels = [f"{row.code}  {row.category}" for row in summary.itertuples()][::-1]
    values = summary["count"].tolist()[::-1]
    percentages = summary["percentage"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=180)
    bars = ax.barh(labels, values, color="#2F5D8A", height=0.66)
    ax.set_xlabel("Number of images")
    ax.set_title("HAM10000 class distribution", loc="left", weight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.set_xlim(0, max(values) * 1.15)
    for bar, count, percentage in zip(bars, values, percentages):
        ax.text(count + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{count:,} ({percentage:.2f}%)", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output / "class_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary.to_csv(output / "dataset_summary.csv", index=False)
    return summary


def create_sample_grid(df: pd.DataFrame, images: dict[str, Path], output: Path, samples_per_class: int) -> None:
    available = df[df["image_id"].isin(images)].copy()
    missing_count = len(df) - len(available)
    if missing_count:
        print(f"Warning: {missing_count} metadata rows have no matching image file.")

    selected = []
    for code in ORDER:
        group = available[available["dx"] == code]
        if len(group) < samples_per_class:
            raise ValueError(f"Class {code} has only {len(group)} available images.")
        selected.append(group.sample(samples_per_class, random_state=SEED))
    samples = pd.concat(selected, ignore_index=True)

    fig, axes = plt.subplots(samples_per_class, len(ORDER), figsize=(15, 2.8 * samples_per_class), dpi=180)
    if samples_per_class == 1:
        axes = [axes]
    for row_index in range(samples_per_class):
        for col_index, code in enumerate(ORDER):
            record = samples[samples["dx"] == code].iloc[row_index]
            ax = axes[row_index][col_index]
            with Image.open(images[record.image_id]) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")
            if row_index == 0:
                ax.set_title(f"{code}\n{LABELS[code]}", fontsize=9, weight="bold")
    fig.suptitle("Representative HAM10000 samples", fontsize=14, weight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(output / "representative_samples.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    df = pd.read_csv(args.metadata)
    validate_metadata(df)
    image_index = index_images(args.images)
    if not image_index:
        raise FileNotFoundError(f"No image files found under: {args.images}")
    summary = create_distribution(df, args.output)
    create_sample_grid(df, image_index, args.output, args.samples_per_class)
    print(summary.to_string(index=False))
    print(f"Found {len(image_index):,} image files.")
    print(f"Saved figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
