"""A small but modern CNN for image classification (CIFAR-10 sized inputs).

Conv-BN-ReLU blocks with max-pooling, global average pooling, and a linear
head. Batch-norm and GAP (instead of large fully-connected layers) keep the
parameter count low and training stable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) x2, then optional 2x2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.block(x))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64),    # 32 -> 16
            ConvBlock(64, 128),            # 16 -> 8
            ConvBlock(128, 256),           # 8  -> 4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(y.shape))   # (2, 10)
    print("parameters  :", f"{n_params:,}")
