"""Convolutional network trained from scratch for HAM10000."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """``convs`` x (3x3 Conv -> BatchNorm -> ReLU) followed by 2x2 max pooling."""

    def __init__(self, in_channels: int, out_channels: int, convs: int = 2) -> None:
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(convs):
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)


class SkinLesionCNN(nn.Module):
    """VGG-style CNN: widening conv blocks, global average pooling, dropout MLP head.

    With the defaults (4 blocks x 2 convs) the network has eight convolutional
    layers. Each block halves the spatial resolution and doubles the number of
    filters, so a 128x128 input becomes an 8x8 map before global pooling.
    """

    def __init__(
        self,
        num_classes: int = 7,
        base_filters: int = 32,
        num_blocks: int = 4,
        convs_per_block: int = 2,
        dropout: float = 0.3,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        if num_blocks * convs_per_block < 4:
            raise ValueError("The milestone requires at least four convolutional layers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        widths = [base_filters * 2**index for index in range(num_blocks)]
        blocks: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            blocks.append(ConvBlock(in_channels, width, convs_per_block))
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self._initialize()

    @property
    def final_linear(self) -> nn.Linear:
        return self.classifier[-1]

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))
