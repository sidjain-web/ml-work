"""Train the GPT on a character-level corpus.

Point `--data` at any UTF-8 text file (e.g. the tiny-shakespeare dataset).
If the file is missing, a small built-in sample is used so the script runs
out of the box.

    python train.py --data input.txt --iters 2000
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device, set_seed, human_format  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

_SAMPLE = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles.\n"
) * 200


def load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[train] '{path}' not found — using built-in sample text.")
    return _SAMPLE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="input.txt")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--out", type=str, default="ckpt.pt")
    args = p.parse_args()

    set_seed(1337)
    device = get_device()
    print(f"[train] device = {device}")

    text = load_text(args.data)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split: str):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    cfg = GPTConfig(
        vocab_size=len(chars), block_size=args.block_size,
        n_layer=4, n_head=4, n_embd=128, dropout=0.1,
    )
    model = GPT(cfg).to(device)
    print(f"[train] parameters: {human_format(model.num_params())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def estimate_loss(iters: int = 50):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(iters)
            for k in range(iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    for it in range(args.iters + 1):
        if it % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__, "stoi": stoi, "itos": itos},
        args.out,
    )
    print(f"[train] saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
