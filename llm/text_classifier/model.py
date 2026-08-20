"""A transformer-encoder text classifier.

Embedding + learned positional encoding + N encoder blocks + masked mean
pooling + linear head. Handles variable-length batches via a padding mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_head: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(pos))

        pad_mask = input_ids == self.pad_idx           # True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Masked mean pooling over real (non-pad) tokens.
        keep = (~pad_mask).unsqueeze(-1).float()
        summed = (x * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.head(pooled)


if __name__ == "__main__":
    model = TransformerClassifier(vocab_size=1000, num_classes=2)
    ids = torch.randint(1, 1000, (4, 16))
    ids[:, 10:] = 0  # simulate padding
    logits = model(ids)
    print("logits shape:", tuple(logits.shape))  # (4, 2)
