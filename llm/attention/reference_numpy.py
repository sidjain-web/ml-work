"""Dependency-free NumPy reference for scaled dot-product & multi-head attention.

Useful for understanding the math without a deep-learning framework, and as a
ground-truth to check the PyTorch version against.
"""
from __future__ import annotations

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., seq, d_k). mask: (..., seq_q, seq_k) with 0 = masked."""
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn = softmax(scores, axis=-1)
    return np.matmul(attn, v), attn


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_head, causal=False):
    """Single-tensor self-attention. x: (B, T, d_model); W*: (d_model, d_model)."""
    B, T, d_model = x.shape
    d_head = d_model // n_head

    def split(mat):  # (B, T, d_model) -> (B, H, T, d_head)
        return (x @ mat).reshape(B, T, n_head, d_head).transpose(0, 2, 1, 3)

    q, k, v = split(Wq), split(Wk), split(Wv)

    mask = None
    if causal:
        mask = np.tril(np.ones((T, T)))

    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)  # merge heads
    return out @ Wo, attn


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, d_model, n_head = 2, 5, 32, 4
    x = rng.standard_normal((B, T, d_model))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d_model, d_model)) * 0.1 for _ in range(4))

    out, attn = multi_head_attention(x, Wq, Wk, Wv, Wo, n_head, causal=True)
    print("output shape     :", out.shape)                       # (2, 5, 32)
    print("attn shape       :", attn.shape)                      # (2, 4, 5, 5)
    print("rows sum to 1    :", np.allclose(attn.sum(-1), 1.0))
    upper = np.triu(attn[0, 0], k=1)
    print("causal respected :", np.allclose(upper, 0.0))
