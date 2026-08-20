"""Train a BPE tokenizer on a text file and save the merges.

    python train_tokenizer.py --data corpus.txt --vocab_size 1024 --out merges.txt
"""
from __future__ import annotations

import argparse

from tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="UTF-8 text file")
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--out", type=str, default="merges.txt")
    args = p.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)
    tok.save(args.out)

    # Quick sanity check on the training text itself.
    sample = text[:200]
    ok = tok.decode(tok.encode(sample)) == sample
    print(f"[train] vocab size = {tok.vocab_size}, roundtrip ok = {ok}")
    print(f"[train] saved merges -> {args.out}")


if __name__ == "__main__":
    main()
