"""A minimal byte-level Byte-Pair Encoding (BPE) tokenizer.

Pure standard-library implementation (no external deps), in the spirit of
Karpathy's minbpe. Trains merges on raw UTF-8 bytes so it can encode *any*
string without an <unk> token.

    tok = BPETokenizer()
    tok.train(text, vocab_size=512)
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def _get_stats(ids: List[int]) -> Counter:
    """Count occurrences of adjacent id pairs."""
    counts: Counter = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        # (int, int) -> int   the learned merges, in application order
        self.merges: Dict[Tuple[int, int], int] = {}
        # int -> bytes        maps token ids back to byte strings
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte alphabet)"
        n_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(n_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # most frequent adjacent pair
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{n_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}) had {stats[pair]} occurrences")

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        # Greedily apply merges in the order they were learned.
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # Choose the pair whose merge was learned earliest (lowest new_id).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no more applicable merges
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # --- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for (a, b), idx in self.merges.items():
                f.write(f"{a} {b} {idx}\n")

    def load(self, path: str) -> None:
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                a, b, idx = map(int, line.split())
                self.merges[(a, b)] = idx
                self.vocab[idx] = self.vocab[a] + self.vocab[b]


if __name__ == "__main__":
    sample = ("the quick brown fox jumps over the lazy dog. " * 50) + \
             "unicode: café, naïve, 日本語, 😀"
    tok = BPETokenizer()
    tok.train(sample, vocab_size=350, verbose=False)
    text = "the lazy dog 😀"
    ids = tok.encode(text)
    print("vocab size :", tok.vocab_size)
    print("encoded    :", ids)
    print("roundtrip  :", tok.decode(ids) == text)
