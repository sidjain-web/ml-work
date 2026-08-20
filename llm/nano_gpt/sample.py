"""Generate text from a trained nano_gpt checkpoint.

    python sample.py --ckpt ckpt.pt --prompt "To be" --tokens 300
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import get_device  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    start = args.prompt if args.prompt else "\n"
    idx = torch.tensor([[stoi.get(c, 0) for c in start]], dtype=torch.long, device=device)

    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[int(i)] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
