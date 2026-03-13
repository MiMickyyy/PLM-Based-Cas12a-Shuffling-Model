from __future__ import annotations

import torch
from torch import nn


class TransformerAutoregressiveLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_positions: int = 4096,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if max_positions < 2:
            raise ValueError("max_positions must be >= 2")

        self.pad_idx = int(pad_idx)
        self.max_positions = int(max_positions)
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (int(seq_len), device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            self._causal_mask_cache[key] = mask
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_positions={self.max_positions}"
            )

        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)

        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        causal_mask = self._causal_mask(seq_len, input_ids.device)
        key_padding_mask = input_ids.eq(self.pad_idx)
        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.final_norm(h)
        return self.lm_head(h)
