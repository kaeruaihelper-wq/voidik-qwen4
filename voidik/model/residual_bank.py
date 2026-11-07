"""Expanded residual stream with GR-style read/write gating.

The residual bank gives every token several low-dimensional latent slots in
addition to the main hidden state.  Each hybrid layer reads a gated summary of
the bank, runs its mixer/MoE, then writes a gated update back.  This is the
"Expanded Residual" path from the diagram; unlike a recurrent memory it keeps
no cross-token state, so it is fully compatible with teacher forcing and
incremental decoding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ResidualBankConfig
from .norm import RMSNorm


class ResidualBank(nn.Module):
    def __init__(self, config: ResidualBankConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.slots = config.slots
        self.slot_dim = config.slot_dim
        self.bank_dim = self.slots * self.slot_dim

        self.write_proj = nn.Linear(hidden_size, self.bank_dim, bias=False)
        self.read_proj = nn.Linear(self.bank_dim, hidden_size, bias=False)
        self.write_gate = nn.Linear(hidden_size, self.slots, bias=True)
        self.read_gate = nn.Linear(hidden_size, 1, bias=True)
        self.bank_norm = RMSNorm(self.bank_dim)

        # Start with a small read/write path.  The bank itself is expanded from
        # the token embedding, so all projections receive gradients from step 0.
        nn.init.normal_(self.write_proj.weight, std=0.02)
        nn.init.normal_(self.read_proj.weight, std=0.02)
        nn.init.constant_(self.read_gate.bias, config.read_bias_init)
        nn.init.constant_(self.write_gate.bias, config.write_bias_init)

    def initial_bank(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.write_proj(hidden_states).view(
            *hidden_states.shape[:-1], self.slots, self.slot_dim
        )

    def read(self, hidden_states: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        flat_bank = self.bank_norm(bank.reshape(*bank.shape[:-2], self.bank_dim))
        summary = self.read_proj(F.silu(flat_bank))
        gate = torch.sigmoid(self.read_gate(hidden_states))
        return hidden_states + gate * summary

    def write(self, hidden_states: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        update = self.write_proj(hidden_states).view(
            *hidden_states.shape[:-1], self.slots, self.slot_dim
        )
        gate = torch.sigmoid(self.write_gate(hidden_states)).unsqueeze(-1)
        return (1.0 - gate) * bank + gate * update

    def merge(self, bank: torch.Tensor) -> torch.Tensor:
        return self.read_proj(F.silu(self.bank_norm(bank.reshape(*bank.shape[:-2], self.bank_dim))))
