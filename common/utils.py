"""Small helpers shared across projects.

Kept dependency-light: only `random`/`os` are always imported. `numpy` and
`torch` are imported lazily so that pure-NumPy projects don't require torch.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device ('cuda' > 'mps' > 'cpu')."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module, trainable_only: bool = True) -> int:
    """Count parameters of a torch module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def human_format(n: float) -> str:
    """Format a number like 12_300_000 -> '12.3M'."""
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:3.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}P"
