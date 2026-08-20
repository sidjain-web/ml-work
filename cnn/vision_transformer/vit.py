"""Vision Transformer (ViT) from scratch — Dosovitskiy et al., 2020.

Bridges the CNN and LLM halves of this repo: an image is split into patches,
each patch is linearly embedded like a "token", a learnable [CLS] token and
positional embeddings are added, and a standard transformer encoder does the
rest. Classification reads off the [CLS] token.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly embed them.

    Implemented as a strided conv: kernel = stride = patch_size turns each
    patch into one embedding vector in a single op.
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with multi-head self-attention."""

    def __init__(self, dim, n_head, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        n_head=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, n_head, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                             # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        x = torch.cat([cls, x], dim=1) + self.pos_embed     # prepend CLS + add pos
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])                            # classify from CLS


if __name__ == "__main__":
    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10)
    y = model(torch.randn(2, 3, 32, 32))
    n = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(y.shape))   # (2, 10)
    print("parameters  :", f"{n / 1e6:.1f}M")
