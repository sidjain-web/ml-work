# cells_from_scratch

The internals of recurrent cells, written out explicitly — no `nn.RNN` /
`nn.LSTM` black box.

## Files
- `numpy_cells.py` — pure-NumPy forward passes for a vanilla RNN cell and an
  LSTM cell, with self-tests (including a gate-behavior test).
- `torch_cells.py` — trainable PyTorch versions, plus a check that the scratch
  LSTM matches `nn.LSTMCell` exactly given identical weights.

## Run
```bash
python numpy_cells.py    # needs only numpy
python torch_cells.py    # needs torch; verifies equivalence with nn.LSTMCell
```

## The equations

**Vanilla RNN**
```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
```
Simple, but struggles to carry information over long sequences — repeated
multiplication by `W_hh` makes gradients vanish or explode.

**LSTM** — adds a cell state `c_t` and three gates that control it:
```
i_t = σ(...)              # how much new candidate info to write
f_t = σ(...)              # how much old cell state to keep
g_t = tanh(...)           # the candidate values
o_t = σ(...)              # how much of the cell to expose as output
c_t = f_t·c_{t-1} + i_t·g_t
h_t = o_t·tanh(c_t)
```
The additive cell-state update (`f_t·c_{t-1} + ...`) is the key: it gives
gradients a near-linear path through time, which is why LSTMs handle long-range
dependencies far better than vanilla RNNs. The NumPy self-test demonstrates this
by forcing `f≈1, i≈0` and showing the cell state is preserved unchanged across
timesteps.

Gate order here is `[i, f, g, o]`, matching PyTorch, so weights are directly
transferable between the scratch and built-in cells.
