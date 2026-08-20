"""Vanilla RNN and LSTM cells from scratch in PyTorch (trainable via autograd).

These implement the gate equations explicitly rather than calling `nn.LSTMCell`,
so the internals are visible. The `__main__` block verifies the scratch LSTM
matches PyTorch's built-in `nn.LSTMCell` exactly when given the same weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class VanillaRNNCellScratch(nn.Module):
    """h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, h):
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


class LSTMCellScratch(nn.Module):
    """A single LSTM step. Gate order matches PyTorch: [i, f, g, o]."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # One projection for all four gates from the input and from the hidden state.
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTM(nn.Module):
    """A minimal single-layer LSTM built on LSTMCellScratch (batch_first)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellScratch(input_size, hidden_size)

    def forward(self, x, state=None):
        # x: (B, T, input_size)
        B, T, _ = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = state
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)  # (B, T, H), (h_T, c_T)


def _verify_against_pytorch():
    """Copy weights from nn.LSTMCell into the scratch cell; outputs must match."""
    torch.manual_seed(0)
    D, H, B = 6, 8, 4
    ref = nn.LSTMCell(D, H)
    mine = LSTMCellScratch(D, H)

    with torch.no_grad():
        mine.weight_ih.weight.copy_(ref.weight_ih)
        mine.weight_ih.bias.copy_(ref.bias_ih)
        mine.weight_hh.weight.copy_(ref.weight_hh)
        mine.weight_hh.bias.copy_(ref.bias_hh)

    x = torch.randn(B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)
    h_ref, c_ref = ref(x, (h0, c0))
    h_mine, c_mine = mine(x, (h0, c0))

    ok = torch.allclose(h_ref, h_mine, atol=1e-6) and torch.allclose(c_ref, c_mine, atol=1e-6)
    print(f"scratch LSTMCell matches nn.LSTMCell: {ok}")
    print(f"max |Δh| = {(h_ref - h_mine).abs().max():.2e}, "
          f"max |Δc| = {(c_ref - c_mine).abs().max():.2e}")


if __name__ == "__main__":
    _verify_against_pytorch()
    x = torch.randn(2, 5, 6)
    out, (hT, cT) = LSTM(6, 8)(x)
    print("LSTM sequence output:", tuple(out.shape), "| final h:", tuple(hT.shape))
