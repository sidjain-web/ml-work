# attention

The attention mechanism, built from scratch and explained in two forms.

## Files
- `attention.py` — PyTorch `scaled_dot_product_attention` and a
  `MultiHeadAttention` module supporting self-attention, cross-attention, and
  causal masking.
- `reference_numpy.py` — a dependency-free NumPy version of the same math,
  handy for study and as a correctness reference.

## Run
```bash
python attention.py          # PyTorch demo (needs torch)
python reference_numpy.py    # NumPy demo (needs only numpy)
```

## The core idea
```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```
- **Scaling** by `√d_k` keeps the softmax gradients well-behaved as dimension grows.
- **Multiple heads** let the model attend to different subspaces in parallel;
  outputs are concatenated and projected back to `d_model`.
- **Causal masking** sets future scores to `-∞` before the softmax so position
  *t* can only see positions `≤ t` — this is what makes a GPT autoregressive.
