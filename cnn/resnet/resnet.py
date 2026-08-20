"""ResNet from scratch (He et al., 2015).

Implements both the two-layer BasicBlock (ResNet-18/34) and the three-layer
Bottleneck (ResNet-50/101/152), with a CIFAR-friendly small-image stem toggle.

Key idea: each block learns a *residual* F(x) and outputs F(x) + x. The identity
shortcut lets gradients flow directly, enabling very deep networks to train.
"""
from __future__ import annotations

from typing import List, Type, Union

import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # matches shortcut dims when they change

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = conv1x1(out_ch, out_ch * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        small_input: bool = False,
    ):
        super().__init__()
        self.in_ch = 64

        if small_input:
            # CIFAR stem: 3x3 conv, no aggressive downsampling.
            self.stem = nn.Sequential(
                conv3x3(3, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            # ImageNet stem: 7x7 stride-2 conv + max-pool.
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_ch, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_ch, out_ch * block.expansion, stride),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
        layers = [block(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes=1000, small_input=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, small_input)


def resnet34(num_classes=1000, small_input=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, small_input)


def resnet50(num_classes=1000, small_input=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, small_input)


if __name__ == "__main__":
    for name, ctor in [("resnet18", resnet18), ("resnet34", resnet34), ("resnet50", resnet50)]:
        model = ctor(num_classes=10, small_input=True)
        y = model(torch.randn(2, 3, 32, 32))
        n = sum(p.numel() for p in model.parameters())
        print(f"{name}: output {tuple(y.shape)}, params {n / 1e6:.1f}M")
