"""A bidirectional-LSTM sequence classifier.

The recurrent counterpart to `llm/text_classifier` (a transformer encoder). Same
interface — integer token ids in, class logits out — so you can train both on the
same data and compare. Handles padding via masked mean pooling over the LSTM
outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, num_classes)  # 2x for both directions

    def forward(self, input_ids):
        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()  # (B, T, 1)
        emb = self.embed(input_ids)
        out, _ = self.lstm(emb)                                    # (B, T, 2*hidden)

        # Masked mean pooling over real (non-pad) timesteps.
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(summed / counts)
        return self.head(pooled)


if __name__ == "__main__":
    model = BiLSTMClassifier(vocab_size=1000, num_classes=2)
    ids = torch.randint(1, 1000, (4, 16))
    ids[:, 10:] = 0  # padding
    logits = model(ids)
    print("logits shape:", tuple(logits.shape))  # (4, 2)
