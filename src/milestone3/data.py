"""Milestone 3 data utilities: persistent image caches, augmentation, and loaders.

The fixed Milestone 2 split manifest and its training-only preprocessing rules
are reused unchanged. Augmentation is applied only to the training partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

from milestone2.data import CachedHAM10000Dataset, _seed_worker, build_image_cache


# Statistics of the ImageNet training set; required by torchvision pretrained weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_or_build_cache(
    metadata: pd.DataFrame,
    image_index: Mapping[str, Path],
    image_size: int,
    cache_dir: Path,
    show_progress: bool = True,
) -> torch.Tensor:
    """Return a uint8 image cache, reusing a saved copy when it matches the metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"ham10000_{image_size}px_uint8.pt"
    image_ids = metadata.reset_index(drop=True)["image_id"].astype(str).tolist()

    if path.exists():
        payload = torch.load(path, weights_only=True)
        images = payload["images"]
        if payload["image_ids"] == image_ids and tuple(images.shape[1:]) == (3, image_size, image_size):
            return images

    images = build_image_cache(metadata, image_index, image_size, show_progress=show_progress)
    torch.save({"image_ids": image_ids, "images": images}, path)
    return images


def build_augmentation(
    image_size: int,
    crop_scale: tuple[float, float] = (0.75, 1.0),
    rotation_degrees: float = 20.0,
    color_jitter: bool = True,
) -> v2.Compose:
    """Label-preserving geometric and mild photometric augmentation on uint8 tensors.

    Dermoscopic lesions have no canonical orientation, so flips and small rotations
    preserve the diagnosis. Random resized crops simulate changes in zoom and
    framing. Colour jitter is deliberately mild because colour is diagnostic.
    """
    operations: list[object] = [
        v2.RandomResizedCrop(
            image_size,
            scale=crop_scale,
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(rotation_degrees, interpolation=InterpolationMode.BILINEAR),
    ]
    if color_jitter:
        operations.append(v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02))
    return v2.Compose(operations)


def build_normalization(mean: Sequence[float], std: Sequence[float]) -> v2.Compose:
    """Convert uint8 images to [0, 1] floats, subtract the channel mean, divide by std."""
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[float(x) for x in mean], std=[float(x) for x in std]),
        ]
    )


def build_transform_pair(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    augment: bool = True,
) -> tuple[v2.Compose, v2.Compose]:
    """Return (training transform, evaluation transform)."""
    normalization = build_normalization(mean, std)
    if augment:
        train_transform = v2.Compose([build_augmentation(image_size), normalization])
    else:
        train_transform = normalization
    return train_transform, normalization


def make_loaders(
    split_manifest: pd.DataFrame,
    image_cache: torch.Tensor,
    label_to_index: Mapping[str, int],
    train_transform: object,
    eval_transform: object,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    splits: Sequence[str] = ("train", "validation"),
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Create deterministic loaders; only the training split receives ``train_transform``."""
    loaders: dict[str, DataLoader] = {}
    for name in splits:
        frame = split_manifest[split_manifest["split"] == name]
        if frame.empty:
            raise ValueError(f"Split '{name}' is empty.")
        is_train = name == "train"
        dataset = CachedHAM10000Dataset(
            frame=frame,
            image_cache=image_cache,
            label_to_index=label_to_index,
            transform=train_transform if is_train else eval_transform,
        )
        shuffle = is_train and shuffle_train
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle and len(dataset) % batch_size == 1,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders
