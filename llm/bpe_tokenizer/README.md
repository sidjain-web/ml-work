# bpe_tokenizer

A minimal **byte-level Byte-Pair Encoding** tokenizer in pure Python (no
dependencies). Because it operates on raw UTF-8 bytes, it can encode any string
— including emoji and non-Latin scripts — with no `<unk>` token.

## Files
- `tokenizer.py` — `BPETokenizer` with `train`, `encode`, `decode`, `save`, `load`.
- `train_tokenizer.py` — CLI to train on a file and persist merges.

## Quick start
```bash
python tokenizer.py                      # runs a built-in self-test
python train_tokenizer.py --data datasets/tiny_corpus.txt --vocab_size 1024 --out datasets/merges.txt
```

```python
from tokenizer import BPETokenizer
tok = BPETokenizer()
tok.train(open("corpus.txt").read(), vocab_size=1024)
ids = tok.encode("hello world")
assert tok.decode(ids) == "hello world"
```

## How it works
1. Start from the 256 single-byte tokens.
2. Repeatedly find the most frequent adjacent pair and merge it into a new token.
3. `encode` greedily applies learned merges in the order they were discovered.

Drop the resulting vocab into `nano_gpt` to train on sub-word units instead of
characters.
