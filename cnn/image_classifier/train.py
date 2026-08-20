"""Train SimpleCNN on CIFAR-10 (downloaded via torchvision).

    python train.py --epochs 15 --data ./data

If torchvision or the dataset is unavailable, the script falls back to a small
random-tensor dataset so the training loop still runs end-to-end for smoke tests.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device, set_seed, human_format  # noqa: E402
from model import SimpleCNN  # noqa: E402


def get_dataloaders(data_dir: str, batch_size: int):
    try:
        import torchvision
        import torchvision.transforms as T

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, num_workers=2),
            10,
        )
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[train] torchvision/CIFAR unavailable ({e}); using random fallback data.")
        xt = torch.randn(512, 3, 32, 32)
        yt = torch.randint(0, 10, (512,))
        xv = torch.randn(128, 3, 32, 32)
        yv = torch.randint(0, 10, (128,))
        return (
            DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(xv, yv), batch_size=batch_size),
            10,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", type=str, default="./data")
    args = p.parse_args()

    set_seed(0)
    device = get_device()
    print(f"[train] device = {device}")

    train_dl, test_dl, num_classes = get_dataloaders(args.data, args.batch_size)
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"[train] parameters: {human_format(sum(p.numel() for p in model.parameters()))}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = total = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
        model.train()
        return correct / total

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        acc = evaluate()
        print(f"epoch {epoch:2d} | loss {running / len(train_dl):.4f} | test acc {acc:.3f}")


if __name__ == "__main__":
    main()
