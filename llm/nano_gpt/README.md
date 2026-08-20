# nano_gpt

A compact, from-scratch **decoder-only transformer** (GPT) for character-level
language modeling. Pre-norm blocks, causal multi-head self-attention, weight-tied
embeddings, and PyTorch's fused attention when available.

## Files
- `model.py` — `GPTConfig`, `GPT`, attention/MLP/block modules, `generate()`.
- `train.py` — character-level training loop with train/val loss estimation.
- `sample.py` — autoregressive sampling from a checkpoint.

## Quick start
```bash
# optional: grab a corpus
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python train.py --data input.txt --iters 2000
python sample.py --ckpt ckpt.pt --prompt "To be" --tokens 300
```
If `input.txt` is absent the script falls back to a small built-in sample so it
still runs end-to-end.

## Notes
- `block_size` is the context window; positions beyond it are cropped at
  generation time.
- Swap the char-level vocab for the `bpe_tokenizer` in this repo to train on
  sub-word units.
