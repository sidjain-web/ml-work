"""Generate text from a trained CharRNN checkpoint.

    python sample.py --ckpt ckpt.pt --prompt "The fox" --tokens 300
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device  # noqa: E402
from model import CharRNN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--prompt", type=str, default="The ")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = CharRNN(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        rnn_type=cfg["rnn_type"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    seed = args.prompt if args.prompt else "\n"
    idx = torch.tensor([[stoi.get(c, 0) for c in seed]], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[int(i)] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
