# resnet

**ResNet** (He et al., 2015) implemented from scratch, with both block types and
constructors for ResNet-18/34/50.

## Files
- `resnet.py` — `BasicBlock`, `Bottleneck`, the `ResNet` backbone, and
  `resnet18/34/50` factory functions.

## Run
```bash
python resnet.py
```
```
resnet18: output (2, 10), params 11.2M
resnet34: output (2, 10), params 21.3M
resnet50: output (2, 10), params 23.5M
```

## The residual idea
Each block computes `F(x) + x`. The identity shortcut gives gradients a direct
path back through the network, which is what makes training very deep nets
feasible without vanishing gradients.

- **BasicBlock** (2×3×3 convs) — used in ResNet-18/34.
- **Bottleneck** (1×1 → 3×3 → 1×1, ×4 expansion) — used in ResNet-50+.
- `small_input=True` swaps the 7×7 stride-2 ImageNet stem for a 3×3 stem, which
  works much better on 32×32 CIFAR images.

Drop `resnet18(num_classes=10, small_input=True)` into the training loop in
`../image_classifier/train.py` to compare against the plain CNN.
