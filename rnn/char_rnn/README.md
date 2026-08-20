# char_rnn

A character-level **recurrent** language model — the RNN counterpart to this
repo's `nano_gpt`. Same task (predict the next character), swappable backbone:
vanilla RNN, GRU, or LSTM.

## Files
- `model.py` — `CharRNN` wrapping `nn.RNN` / `nn.GRU` / `nn.LSTM`, with a
  `generate()` method that streams the hidden state one token at a time.
- `train.py` — training loop (reuses `datasets/tiny_corpus.txt`) with gradient
  clipping, which recurrent nets need to avoid exploding gradients.
- `sample.py` — text generation from a checkpoint.

## Run
```bash
python model.py                              # shape check for all three backbones
python train.py --rnn_type lstm --iters 3000
python sample.py --ckpt ckpt.pt --prompt "The fox" --tokens 300
```

## Compare backbones
Train each on the same corpus and watch the validation loss:
```bash
python train.py --rnn_type rnn  --iters 3000
python train.py --rnn_type gru  --iters 3000
python train.py --rnn_type lstm --iters 3000
```
Typically GRU/LSTM beat the vanilla RNN because their gating carries information
over longer spans. For a deeper contrast, train `../../llm/nano_gpt` on the same
file and compare the transformer against these recurrent models.

## Notes
- Generation feeds only the newest token each step and keeps the hidden state,
  so sampling is O(1) work per character rather than reprocessing the whole
  context (unlike the transformer, which re-attends over the window).
