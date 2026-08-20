# sequence_classifier

A **bidirectional-LSTM** sequence classifier — the recurrent counterpart to
`llm/text_classifier` (a transformer encoder). Same interface and same synthetic
training task, so you can compare the two model families head-to-head.

## Files
- `model.py` — `BiLSTMClassifier`: embedding → BiLSTM → masked mean pooling →
  linear head.
- `train.py` — trains on the same synthetic marker-counting task used by the
  transformer classifier.

## Run
```bash
python model.py           # forward-pass shape check
python train.py --epochs 5
```

## Compare with the transformer
```bash
python train.py                              # this BiLSTM
python ../../llm/text_classifier/train.py    # transformer encoder
```
Both should approach high accuracy on this easy task; the interesting part is
watching how quickly each gets there and how they scale as you make the task
harder (longer sequences, more distractor tokens).

## Notes
- **Bidirectional**: reads the sequence both forward and backward, so each
  position's representation sees the whole context — the output dim is
  `2 × hidden_size`.
- **Masked mean pooling** ignores padding, matching the transformer classifier's
  pooling so the comparison is fair.
- Gradient clipping is applied during training, as is standard for recurrent nets.
