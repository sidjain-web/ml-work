"""Train the character-level RNN/GRU/LSTM language model.

Reuses the same corpus as nano_gpt (datasets/tiny_corpus.txt by default), so you
can train both and compare recurrent vs transformer next-character modeling.

    python train.py --rnn_type lstm --iters 3000
    python train.py --rnn_type gru  --data ../../datasets/input.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device, set_seed, human_format  # noqa: E402
from model import CharRNN  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUNDLED_CORPUS = os.path.join(_REPO_ROOT, "datasets", "tiny_corpus.txt")
_SAMPLE = ("the quick brown fox jumps over the lazy dog.\n" * 400)


def load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.exists(_BUNDLED_CORPUS):
        print(f"[train] '{path}' not found — using bundled corpus {_BUNDLED_CORPUS}.")
        with open(_BUNDLED_CORPUS, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[train] '{path}' not found and no bundled corpus — using built-in sample.")
    return _SAMPLE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="../../datasets/tiny_corpus.txt")
    p.add_argument("--rnn_type", type=str, default="lstm", choices=["rnn", "gru", "lstm"])
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--out", type=str, default="ckpt.pt")
    args = p.parse_args()

    set_seed(1337)
    device = get_device()
    print(f"[train] device = {device} | rnn_type = {args.rnn_type}")

    text = load_text(args.data)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.seq_len - 1, (args.batch_size,))
        x = torch.stack([d[i:i + args.seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.seq_len] for i in ix])
        return x.to(device), y.to(device)

    model = CharRNN(
        vocab_size=len(chars), embed_dim=128, hidden_size=256,
        num_layers=2, rnn_type=args.rnn_type,
    ).to(device)
    print(f"[train] parameters: {human_format(sum(p.numel() for p in model.parameters()))}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def estimate_loss(iters=50):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(iters)
            for k in range(iters):
                x, y = get_batch(split)
                logits, _ = model(x)
                losses[k] = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            out[split] = losses.mean().item()
        model.train()
        return out

    for it in range(args.iters + 1):
        if it % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        x, y = get_batch("train")
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # RNNs need clipping
        opt.step()

    torch.save(
        {"model": model.state_dict(), "stoi": stoi, "itos": itos,
         "config": {"vocab_size": len(chars), "rnn_type": args.rnn_type,
                    "embed_dim": 128, "hidden_size": 256, "num_layers": 2}},
        args.out,
    )
    print(f"[train] saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
