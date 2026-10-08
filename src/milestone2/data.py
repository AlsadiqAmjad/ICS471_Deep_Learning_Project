"""HAM10000 image indexing, preprocessing, caching, and DataLoaders."""

from __future__ import annotations

import hashlib
import random
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
CLASS_ORDER = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def index_images(root: Path) -> dict[str, Path]:
    """Index images by stem and reject conflicting duplicate image IDs."""
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")

    candidates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.setdefault(path.stem, []).append(path)
    if not candidates:
        raise FileNotFoundError(f"No supported image files found under: {root}")

    image_index: dict[str, Path] = {}
    identical_duplicate_count = 0
    for image_id, paths in candidates.items():
        selected = paths[0]
        if len(paths) > 1:
            selected_size = selected.stat().st_size
            selected_hash = _sha256(selected)
            for duplicate in paths[1:]:
                same = (
                    duplicate.stat().st_size == selected_size
                    and _sha256(duplicate) == selected_hash
                )
                if not same:
                    raise ValueError(
                        f"Conflicting files share image ID '{image_id}': {paths}"
                    )
                identical_duplicate_count += 1
        image_index[image_id] = selected

    if identical_duplicate_count:
        warnings.warn(
            f"Ignored {identical_duplicate_count} byte-identical duplicate image files.",
            stacklevel=2,
        )
    return image_index


def validate_image_coverage(metadata: pd.DataFrame, image_index: Mapping[str, Path]) -> None:
    """Ensure that every metadata image has exactly one resolved file."""
    metadata_ids = set(metadata["image_id"].astype(str))
    missing = sorted(metadata_ids - set(image_index))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} metadata images are missing. First IDs: {missing[:5]}"
        )


def build_image_cache(
    metadata: pd.DataFrame,
    image_index: Mapping[str, Path],
    image_size: int,
    show_progress: bool = True,
) -> torch.Tensor:
    """Load and resize each image once into a compact uint8 tensor cache."""
    validate_image_coverage(metadata, image_index)
    cache = torch.empty(
        (len(metadata), 3, image_size, image_size),
        dtype=torch.uint8,
    )
    iterator = metadata.reset_index(drop=True).itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(metadata), desc="Caching resized images")

    resampling = Image.Resampling.LANCZOS
    for cache_index, record in enumerate(iterator):
        path = image_index[str(record.image_id)]
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((image_size, image_size), resampling)
            array = np.asarray(resized, dtype=np.uint8).copy()
        cache[cache_index].copy_(torch.from_numpy(array).permute(2, 0, 1))
    return cache


def compute_channel_stats(
    image_cache: torch.Tensor,
    train_cache_indices: np.ndarray | list[int],
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit RGB mean and standard deviation using training images only."""
    indices = torch.tensor(train_cache_indices, dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("Training cache indices are empty.")

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for start in range(0, len(indices), chunk_size):
        batch = image_cache[indices[start : start + chunk_size]].to(torch.float64) / 255.0
        channel_sum += batch.sum(dim=(0, 2, 3))
        channel_squared_sum += batch.square().sum(dim=(0, 2, 3))
        pixel_count += batch.shape[0] * batch.shape[2] * batch.shape[3]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.square()
    std = variance.clamp_min(0.0).sqrt()
    if (std <= 0).any():
        raise ValueError(f"Invalid channel standard deviation: {std.tolist()}")
    return mean.to(torch.float32), std.to(torch.float32)


def fit_label_mapping(train_labels: pd.Series) -> dict[str, int]:
    """Fit a deterministic label encoder from training labels only."""
    classes = sorted(train_labels.dropna().astype(str).unique().tolist())
    if not classes:
        raise ValueError("No training labels were provided.")
    return {label: index for index, label in enumerate(classes)}


def compute_class_weights(encoded_train_labels: list[int] | np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute balanced class weights from encoded training labels only."""
    labels = torch.tensor(encoded_train_labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training data is missing encoded classes: {missing}")
    return len(labels) / (num_classes * counts)


def build_transforms(
    mean: torch.Tensor | list[float],
    std: torch.Tensor | list[float],
    training: bool,
) -> transforms.Compose:
    """Build tensor transforms; augmentation is applied only during training."""
    operations: list[object] = [transforms.ConvertImageDtype(torch.float32)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=15,
                    interpolation=InterpolationMode.BILINEAR,
                ),
            ]
        )
    operations.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(operations)


class CachedHAM10000Dataset(Dataset):
    """Dataset backed by a shared uint8 cache of resized images."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_cache: torch.Tensor,
        label_to_index: Mapping[str, int],
        transform: object,
    ) -> None:
        required = {"cache_index", "dx"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.image_cache = image_cache
        self.label_to_index = dict(label_to_index)
        self.transform = transform

        unknown = set(self.frame["dx"].astype(str).unique()) - set(self.label_to_index)
        if unknown:
            raise ValueError(f"Unknown labels in dataset frame: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.frame.iloc[index]
        image = self.image_cache[int(record["cache_index"])]
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.label_to_index[str(record["dx"])], dtype=torch.long)
        return image, label


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloaders(
    split_manifest: pd.DataFrame,
    image_cache: torch.Tensor,
    label_to_index: Mapping[str, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """Create deterministic training/validation loaders and optionally test."""
    train_transform = build_transforms(mean, std, training=True)
    evaluation_transform = build_transforms(mean, std, training=False)
    names = ["train", "validation"] + (["test"] if include_test else [])
    loaders: dict[str, DataLoader] = {}

    for name in names:
        frame = split_manifest[split_manifest["split"] == name]
        dataset = CachedHAM10000Dataset(
            frame=frame,
            image_cache=image_cache,
            label_to_index=label_to_index,
            transform=train_transform if name == "train" else evaluation_transform,
        )
        generator = torch.Generator().manual_seed(seed)
        drop_last = name == "train" and len(dataset) % batch_size == 1
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
    return loaders
