"""Multi-token-prediction (MTP) modules."""

from __future__ import annotations

import torch
from torch import nn

from ..config import MTPConfig
from .gdn import GatedDeltaNet
from .norm import RMSNorm, SwiGLU


class MTPModule(nn.Module):
    """Predict token t+d+1 from the main hidden state at t and token t+1.

    The module is deliberately small (one GDN mixer + dense SwiGLU), because
    MTP is an auxiliary training objective, not the main generator.
    """

    def __init__(self, config: MTPConfig, model_config) -> None:
        super().__init__()
        hidden_size = model_config.hidden_size
        self.hidden_norm = RMSNorm(hidden_size, model_config.rms_norm_eps)
        self.next_norm = RMSNorm(hidden_size, model_config.rms_norm_eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.block_norm = RMSNorm(hidden_size, model_config.rms_norm_eps)
        self.mixer = GatedDeltaNet(model_config.gdn, hidden_size)
        self.ffn_norm = RMSNorm(hidden_size, model_config.rms_norm_eps)
        self.ffn = SwiGLU(hidden_size, config.ffn_ff)
        self.out_norm = RMSNorm(hidden_size, model_config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, next_embeddings: torch.Tensor):
        combined = torch.cat(
            (self.hidden_norm(hidden_states), self.next_norm(next_embeddings)), dim=-1
        )
        hidden_states = self.proj(combined)
        hidden_states = hidden_states + self.mixer(self.block_norm(hidden_states))
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return self.out_norm(hidden_states)
