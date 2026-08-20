"""Attention, from scratch, in PyTorch.

Educational implementations of the building blocks used throughout modern
transformers:
  - scaled dot-product attention (with optional mask)
  - multi-head attention (self- or cross-attention)

See `reference_numpy.py` for a dependency-free NumPy version of the core op.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Shapes: q,k,v are (..., seq, d_k). `mask` broadcasts to the score matrix
    (..., seq_q, seq_k); positions that are 0/False are masked out.
    """
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting self- and cross-attention.

    If `causal=True`, a lower-triangular mask is applied so each position may
    only attend to earlier positions (used in decoders / GPT).
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.causal = causal
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

    def forward(self, query, key=None, value=None, mask=None):
        # Self-attention when key/value are omitted.
        key = query if key is None else key
        value = query if value is None else value

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if self.causal:
            Tq, Tk = q.size(2), k.size(2)
            causal = torch.tril(torch.ones(Tq, Tk, device=q.device)).bool()
            mask = causal if mask is None else (mask.bool() & causal)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
        )
        B, H, T, d = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * d)  # merge heads
        return self.w_o(out), attn


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, n_head = 2, 5, 32, 4
    x = torch.randn(B, T, d_model)

    mha = MultiHeadAttention(d_model, n_head, causal=True)
    out, attn = mha(x)
    print("output shape    :", tuple(out.shape))     # (2, 5, 32)
    print("attn shape      :", tuple(attn.shape))     # (2, 4, 5, 5)
    # In a causal map, row t must place zero weight on future positions.
    upper = attn[0, 0].triu(diagonal=1)
    print("causal respected:", bool(torch.all(upper == 0)))
