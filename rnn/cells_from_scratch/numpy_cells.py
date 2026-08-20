"""Vanilla RNN and LSTM cells from scratch, in pure NumPy.

Dependency-free forward passes that make the recurrence explicit. Useful for
understanding exactly what a recurrent cell computes at each timestep, and as a
ground-truth reference for the PyTorch versions in `torch_cells.py`.

Vanilla RNN:
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)

LSTM (gates in order i, f, g, o):
    i_t = sigmoid(W_i · [x_t, h_{t-1}] + b_i)   input gate
    f_t = sigmoid(W_f · [x_t, h_{t-1}] + b_f)   forget gate
    g_t = tanh   (W_g · [x_t, h_{t-1}] + b_g)   candidate cell input
    o_t = sigmoid(W_o · [x_t, h_{t-1}] + b_o)   output gate
    c_t = f_t * c_{t-1} + i_t * g_t             new cell state
    h_t = o_t * tanh(c_t)                        new hidden state
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VanillaRNNCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.W_xh = rng.uniform(-scale, scale, (hidden_size, input_size))
        self.W_hh = rng.uniform(-scale, scale, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def step(self, x, h):
        return np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

    def forward(self, x_seq, h0=None):
        """x_seq: (T, input_size). Returns hs: (T, hidden_size)."""
        h = np.zeros(self.hidden_size) if h0 is None else h0
        hs = []
        for x in x_seq:
            h = self.step(x, h)
            hs.append(h)
        return np.stack(hs)


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(hidden_size)
        self.hidden_size = hidden_size
        # Stack the four gates' weights into one matrix of shape (4H, input+H).
        self.W = rng.uniform(-scale, scale, (4 * hidden_size, input_size + hidden_size))
        self.b = np.zeros(4 * hidden_size)

    def step(self, x, h, c):
        H = self.hidden_size
        z = self.W @ np.concatenate([x, h]) + self.b
        i = sigmoid(z[0:H])          # input gate
        f = sigmoid(z[H:2 * H])      # forget gate
        g = np.tanh(z[2 * H:3 * H])  # candidate
        o = sigmoid(z[3 * H:4 * H])  # output gate
        c = f * c + i * g
        h = o * np.tanh(c)
        return h, c, {"i": i, "f": f, "g": g, "o": o}

    def forward(self, x_seq, h0=None, c0=None):
        """x_seq: (T, input_size). Returns (hs, cs) each (T, hidden_size)."""
        h = np.zeros(self.hidden_size) if h0 is None else h0
        c = np.zeros(self.hidden_size) if c0 is None else c0
        hs, cs = [], []
        for x in x_seq:
            h, c, _ = self.step(x, h, c)
            hs.append(h)
            cs.append(c)
        return np.stack(hs), np.stack(cs)


def _test():
    rng = np.random.default_rng(1)
    T, D, H = 7, 4, 5
    x_seq = rng.standard_normal((T, D))

    rnn = VanillaRNNCell(D, H, seed=0)
    hs = rnn.forward(x_seq)
    assert hs.shape == (T, H)
    assert np.all(np.abs(hs) <= 1.0 + 1e-9), "tanh outputs must lie in [-1, 1]"
    print(f"VanillaRNN: hs {hs.shape}, values in [-1,1] ok")

    lstm = LSTMCell(D, H, seed=0)
    hs, cs = lstm.forward(x_seq)
    assert hs.shape == (T, H) and cs.shape == (T, H)
    # Every gate value must be a valid probability / activation.
    _, _, gates = lstm.step(x_seq[0], np.zeros(H), np.zeros(H))
    for name in ("i", "f", "o"):
        assert np.all((gates[name] >= 0) & (gates[name] <= 1)), f"{name} gate out of [0,1]"
    assert np.all(np.abs(gates["g"]) <= 1.0 + 1e-9), "candidate g must be in [-1,1]"
    print(f"LSTM: hs {hs.shape}, cs {cs.shape}, gates in valid ranges ok")

    # Memory property: force forget-gate ~1 and input-gate ~0 -> cell state is held.
    mem = LSTMCell(D, H, seed=0)
    mem.b[H:2 * H] = 50.0     # forget gate -> sigmoid(+50) ~ 1
    mem.b[0:H] = -50.0        # input  gate -> sigmoid(-50) ~ 0
    c_start = rng.standard_normal(H)
    h = np.zeros(H)
    c = c_start.copy()
    for x in x_seq:
        h, c, _ = mem.step(x, h, c)
    assert np.allclose(c, c_start, atol=1e-6), "cell state should be preserved"
    print("LSTM memory gate behavior ok (cell state preserved across 7 steps)")


if __name__ == "__main__":
    _test()
    print("\nAll NumPy cell tests passed.")
