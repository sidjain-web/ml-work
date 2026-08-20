# rnn — recurrent sequence models

RNNs and LSTMs, from the raw gate equations up to full language models and
classifiers. These complement the transformer projects in `../llm`: same tasks,
recurrent backbones, so you can compare the two families directly.

## Projects

| Project | What it is | Needs torch |
| --- | --- | --- |
| `cells_from_scratch/` | Vanilla RNN & LSTM cells implemented by hand (NumPy + PyTorch), with tests | NumPy part: no |
| `char_rnn/` | Character-level language model — RNN / GRU / LSTM (counterpart to `nano_gpt`) | Yes |
| `sequence_classifier/` | BiLSTM classifier (counterpart to `llm/text_classifier`) | Yes |

## Suggested tour
1. **Understand the cell** — run `cells_from_scratch/numpy_cells.py` (no
   dependencies) to see exactly what an LSTM computes, including a demo of how
   its gating preserves memory across timesteps.
2. **Build a language model** — train `char_rnn` on the bundled corpus and
   sample from it; swap `--rnn_type` between `rnn`, `gru`, `lstm`.
3. **Classify sequences** — train `sequence_classifier` and compare it against
   the transformer in `../llm/text_classifier` on the same synthetic task.

## RNN vs Transformer, in one line
Recurrent models process a sequence step by step and carry a hidden state, giving
O(1) memory per step but a sequential (hard-to-parallelize) computation.
Transformers attend over the whole context at once — more parallel, but with
cost that grows with the context length. This repo lets you train both on
identical data and see the trade-offs yourself.
