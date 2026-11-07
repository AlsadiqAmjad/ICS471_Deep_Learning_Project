"""Pretrained ImageNet backbones adapted to HAM10000 with three freezing strategies.

Strategies
----------
``head``      Freeze every pretrained parameter; train only a new linear classifier.
``last3``     Freeze everything except the last three weight layers (Conv2d/Linear),
              counting the new classifier as one of them.
``extended``  Freeze every pretrained parameter; replace the classifier with a new
              deeper head (Linear-BN-ReLU-Dropout blocks) and train only that head.

Frozen BatchNorm layers are kept in evaluation mode during training so their
ImageNet running statistics are not overwritten by HAM10000 batches.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import models


BACKBONES = {
    "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}
STRATEGIES = ("head", "last3", "extended")


def _head_slot(network: nn.Module, backbone: str) -> tuple[nn.Module, str]:
    """Return (parent module, child name) of the ImageNet classification layer."""
    if backbone == "vgg19":
        return network.classifier, "6"
    if backbone == "resnet50":
        return network, "fc"
    if backbone == "mobilenet_v3_large":
        return network.classifier, "3"
    raise ValueError(f"Unsupported backbone '{backbone}'. Choices: {sorted(BACKBONES)}")


def build_extended_head(
    in_features: int,
    num_classes: int,
    hidden: Sequence[int] = (512, 256),
    dropout: float = 0.5,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        width = size
    layers.append(nn.Linear(width, num_classes))
    return nn.Sequential(*layers)


def weight_layers(module: nn.Module) -> list[tuple[str, nn.Module]]:
    """Conv2d and Linear layers in registration (forward) order."""
    return [(name, child) for name, child in module.named_modules() if isinstance(child, (nn.Conv2d, nn.Linear))]


class TransferClassifier(nn.Module):
    """Wrapper that records the strategy and keeps frozen BatchNorm in eval mode."""

    def __init__(self, network: nn.Module, final_linear: nn.Linear, backbone: str, strategy: str,
                 trainable_layers: list[str]) -> None:
        super().__init__()
        self.network = network
        self.backbone = backbone
        self.strategy = strategy
        self.trainable_layers = trainable_layers
        self._final_linear_name = next(name for name, module in network.named_modules() if module is final_linear)

    @property
    def final_linear(self) -> nn.Linear:
        return self.network.get_submodule(self._final_linear_name)

    def train(self, mode: bool = True) -> "TransferClassifier":
        super().train(mode)
        if mode:
            for module in self.network.modules():
                if isinstance(module, _BatchNorm) and not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_transfer_model(
    backbone: str,
    strategy: str,
    num_classes: int = 7,
    pretrained: bool = True,
    extended_hidden: Sequence[int] = (512, 256),
    extended_dropout: float = 0.5,
    unfreeze_last: int = 3,
) -> TransferClassifier:
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choices: {sorted(BACKBONES)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strategy}'. Choices: {STRATEGIES}")

    constructor, weights = BACKBONES[backbone]
    network = constructor(weights=weights if pretrained else None)
    for parameter in network.parameters():
        parameter.requires_grad = False

    parent, slot = _head_slot(network, backbone)
    old_head = parent.get_submodule(slot)
    in_features = old_head.in_features
    if strategy == "extended":
        new_head: nn.Module = build_extended_head(in_features, num_classes, extended_hidden, extended_dropout)
        final_linear = new_head[-1]
    else:
        new_head = nn.Linear(in_features, num_classes)
        final_linear = new_head
    if isinstance(parent, nn.Sequential):
        parent[int(slot)] = new_head
    else:
        setattr(parent, slot, new_head)

    # New layers are trainable; in ``last3`` the two preceding weight layers are too.
    for parameter in new_head.parameters():
        parameter.requires_grad = True
    if strategy == "last3":
        for _, layer in weight_layers(network)[-unfreeze_last:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    trainable = [name for name, module in network.named_modules()
                 if any(p.requires_grad for p in module.parameters(recurse=False))]
    return TransferClassifier(network, final_linear, backbone, strategy, trainable)


def parameter_report(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable}


def trainable_layer_table(model: TransferClassifier) -> pd.DataFrame:
    rows = []
    for name in model.trainable_layers:
        module = model.network.get_submodule(name)
        rows.append({
            "layer": name,
            "type": module.__class__.__name__,
            "shape": str(tuple(module.weight.shape)) if hasattr(module, "weight") else "",
            "trainable_parameters": sum(p.numel() for p in module.parameters(recurse=False)),
        })
    return pd.DataFrame(rows)


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    loader: object,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the input of ``model.final_linear`` (penultimate features) and labels."""
    captured: list[torch.Tensor] = []
    handle = model.final_linear.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach().float().cpu())
    )
    features, labels = [], []
    model.to(device)
    model.eval()
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp and device.type == "cuda"):
                model(images)
            features.append(captured.pop())
            labels.append(targets.numpy())
    finally:
        handle.remove()
    return torch.cat(features).numpy(), np.concatenate(labels)
