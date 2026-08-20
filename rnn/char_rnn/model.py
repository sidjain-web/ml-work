"""A character-level recurrent language model (RNN / GRU / LSTM).

The recurrent counterpart to this repo's `nano_gpt`: same task (predict the next
character), different backbone. Switch `rnn_type` to compare a vanilla RNN, GRU,
and LSTM on identical data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in ("rnn", "gru", "lstm"), "rnn_type must be rnn|gru|lstm"
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(
            embed_dim, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T) integer tokens
        emb = self.dropout(self.embed(x))
        out, hidden = self.rnn(emb, hidden)          # out: (B, T, hidden)
        logits = self.head(self.dropout(out))         # (B, T, vocab)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample characters. `idx`: (1, T) seed tokens."""
        self.eval()
        hidden = None
        # Warm up the hidden state on the full prompt.
        logits, hidden = self(idx, hidden)
        logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # (1, 1)
            idx = torch.cat([idx, nxt], dim=1)
            logits, hidden = self(nxt, hidden)              # feed only the new token
            logits = logits[:, -1, :]
        return idx


if __name__ == "__main__":
    for rnn_type in ("rnn", "gru", "lstm"):
        model = CharRNN(vocab_size=65, rnn_type=rnn_type)
        x = torch.randint(0, 65, (4, 16))
        logits, _ = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"{rnn_type.upper():4s}: logits {tuple(logits.shape)}, params {n/1e6:.2f}M")
