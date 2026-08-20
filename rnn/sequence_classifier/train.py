"""Train the BiLSTM classifier on the same synthetic task as
`llm/text_classifier`, so the recurrent and transformer models are directly
comparable.

Task: label = 1 if the sequence has more "positive" marker tokens than
"negative" ones.

    python train.py --epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device, set_seed  # noqa: E402
from model import BiLSTMClassifier  # noqa: E402

PAD, POS_TOKEN, NEG_TOKEN, VOCAB = 0, 1, 2, 50


class SyntheticDataset(Dataset):
    def __init__(self, n: int, max_len: int = 24, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(n):
            length = int(torch.randint(6, max_len, (1,), generator=g))
            seq = torch.randint(3, VOCAB, (length,), generator=g)
            n_pos = int(torch.randint(0, 5, (1,), generator=g))
            n_neg = int(torch.randint(0, 5, (1,), generator=g))
            for i in range(min(n_pos, length)):
                seq[i] = POS_TOKEN
            for i in range(min(n_neg, length - n_pos)):
                seq[n_pos + i] = NEG_TOKEN
            seq = seq[torch.randperm(length, generator=g)]
            label = 1 if n_pos > n_neg else 0
            padded = torch.full((max_len,), PAD, dtype=torch.long)
            padded[:length] = seq
            self.samples.append((padded, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return x, torch.tensor(y, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()

    set_seed(0)
    device = get_device()
    print(f"[train] device = {device}")

    train_dl = DataLoader(SyntheticDataset(4000, seed=1), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(SyntheticDataset(1000, seed=2), batch_size=args.batch_size)

    model = BiLSTMClassifier(vocab_size=VOCAB, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        print(f"epoch {epoch} | loss {running / len(train_dl):.4f} | val acc {evaluate():.3f}")


if __name__ == "__main__":
    main()
