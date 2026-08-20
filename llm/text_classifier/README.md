# text_classifier

A **transformer-encoder** sequence classifier: token + positional embeddings →
pre-norm encoder stack → masked mean pooling → linear head. Correctly ignores
padding via a `src_key_padding_mask`.

## Files
- `model.py` — `TransformerClassifier` (built on `nn.TransformerEncoder`).
- `train.py` — training loop on a self-contained synthetic task, so it runs
  and demonstrably learns without any dataset download.

## Run
```bash
python model.py       # forward-pass shape check
python train.py --epochs 5
```

## Using your own data
Replace `SyntheticDataset` with a dataset that yields `(input_ids, label)` where
`input_ids` are integer token ids padded to a fixed length with `PAD = 0`. Pair
it with the `bpe_tokenizer` in this repo to tokenize raw text.
