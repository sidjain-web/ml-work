# augmentation

Framework-agnostic **image augmentations** in pure NumPy. Operate on `HxWxC`
arrays in `[0, 1]` — no torchvision or PIL required — so you can plug them into
any pipeline.

## Files
- `transforms.py` — `Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Cutout`,
  `Normalize`, and a `mixup` helper.

## Run
```bash
python transforms.py
```

## Example
```python
from transforms import Compose, RandomCrop, RandomHorizontalFlip, Cutout, Normalize

aug = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(0.5),
    Cutout(size=8),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
augmented = aug(image)   # image: (32, 32, 3) float array in [0, 1]
```

These are the standard CIFAR augmentations; combine `Cutout`/`mixup` with the
models in `../resnet` and `../vision_transformer` to reduce overfitting.
