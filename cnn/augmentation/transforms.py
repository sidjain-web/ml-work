"""Image augmentation transforms in pure NumPy.

Framework-agnostic augmentations that operate on HxWxC arrays in [0, 1]. Compose
them into a pipeline. These mirror the standard CIFAR recipe (pad+crop, flip,
cutout, normalize) without depending on torchvision or PIL.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None):
        self.p = p
        self.rng = rng or np.random.default_rng()

    def __call__(self, img):
        if self.rng.random() < self.p:
            return img[:, ::-1, :].copy()
        return img


class RandomCrop:
    """Zero-pad by `padding` on each side, then crop a random HxW window."""

    def __init__(self, size: int, padding: int = 4, rng: np.random.Generator | None = None):
        self.size = size
        self.padding = padding
        self.rng = rng or np.random.default_rng()

    def __call__(self, img):
        p = self.padding
        padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
        H, W = padded.shape[:2]
        top = int(self.rng.integers(0, H - self.size + 1))
        left = int(self.rng.integers(0, W - self.size + 1))
        return padded[top:top + self.size, left:left + self.size, :]


class Cutout:
    """Zero out a random square patch (a.k.a. random erasing)."""

    def __init__(self, size: int = 8, rng: np.random.Generator | None = None):
        self.size = size
        self.rng = rng or np.random.default_rng()

    def __call__(self, img):
        img = img.copy()
        H, W = img.shape[:2]
        cy = int(self.rng.integers(0, H))
        cx = int(self.rng.integers(0, W))
        h = self.size // 2
        y0, y1 = max(0, cy - h), min(H, cy + h)
        x0, x1 = max(0, cx - h), min(W, cx + h)
        img[y0:y1, x0:x1, :] = 0.0
        return img


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        return (img - self.mean) / self.std


def mixup(x1, y1, x2, y2, alpha=0.2, rng=None):
    """Convex combination of two (image, one-hot label) pairs."""
    rng = rng or np.random.default_rng()
    lam = rng.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = (rng.random((32, 32, 3)) * 0.5 + 0.25).astype(np.float32)  # all pixels > 0

    # Check cutout alone actually blanks a region.
    cut = Cutout(size=8, rng=np.random.default_rng(0))(img)
    blanked = int((cut == 0).all(axis=-1).sum())

    pipeline = Compose([
        RandomCrop(32, padding=4, rng=rng),
        RandomHorizontalFlip(p=1.0, rng=rng),
        Cutout(size=8, rng=rng),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    out = pipeline(img)
    print("input shape       :", img.shape)
    print("output shape      :", out.shape)
    print("cutout blanks      :", blanked, "pixels set to 0")
    print("shapes preserved   :", out.shape == img.shape)
