"""Hashed n-gram embeddings injected at a single transformer layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import NgramConfig
from .norm import RMSNorm


class NgramEmbedding(nn.Module):
    """A compact, collision-tolerant n-gram lookup.

    For every order n in ``orders`` a rolling hash is computed from the last n
    token ids.  The corresponding rows of a per-order embedding table are
    summed, projected to hidden size, and added to the residual stream at
    ``inject_layer`` (the screenshot's "Layer 2 only").
    """

    def __init__(self, config: NgramConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.orders = tuple(config.orders)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(config.buckets_per_order, config.dim)
                for _ in self.orders
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        self.norm = RMSNorm(config.dim)
        self.proj = nn.Linear(config.dim, hidden_size, bias=False)
        nn.init.normal_(self.proj.weight, std=config.scale / (hidden_size**0.5))
        self.dropout = nn.Dropout(config.dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def _hashes(self, input_ids: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len = input_ids.shape
        ids = input_ids.to(torch.int64)
        result = []
        for order_idx, order in enumerate(self.orders):
            padded = F.pad(ids, (order - 1, 0))
            hashed = torch.zeros_like(ids)
            for offset in range(order):
                token = padded[:, offset : offset + seq_len]
                hashed = (
                    hashed * 1_000_003
                    + token
                    + self.config.hash_seed
                    + order_idx * 97_409
                    + offset * 31_337
                ) & 0x7FFF_FFFF
            valid = torch.arange(seq_len, device=input_ids.device)[None, :] >= (
                order - 1
            )
            buckets = hashed % self.config.buckets_per_order
            result.append((buckets, valid))
        return result

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if not self.orders:
            return input_ids.new_zeros(batch, seq_len, self.proj.out_features)
        hashes = self._hashes(input_ids)
        summary = None
        for embedding, (buckets, valid) in zip(self.embeddings, hashes):
            current = embedding(buckets) * valid.unsqueeze(-1)
            summary = current if summary is None else summary + current
        assert summary is not None
        summary = self.norm(summary)
        return self.dropout(self.proj(summary)) * self.scale
