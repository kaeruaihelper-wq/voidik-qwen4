"""The repeating hybrid block: [Expanded Residual / GR] + mixer + MoE."""

from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .attention import QwenSparseAttention
from .gdn import GatedDeltaNet
from .moe import MoE
from .norm import RMSNorm, SwiGLU
from .residual_bank import ResidualBank


class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_type == "gdn":
            self.mixer = GatedDeltaNet(config.gdn, config.hidden_size)
        elif layer_type == "qsa":
            self.mixer = QwenSparseAttention(config.attention, config.hidden_size)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.post_moe_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.moe.enabled:
            self.moe: nn.Module = MoE(config.moe, config.hidden_size)
        else:
            self.moe = SwiGLU(config.hidden_size, 4 * config.hidden_size)

        self.residual_bank = (
            ResidualBank(config.residual_bank, config.hidden_size)
            if config.residual_bank.enabled
            else None
        )
        self.last_aux_loss = None
        self.last_z_loss = None

    def forward(self, hidden_states, bank=None):
        if self.residual_bank is not None:
            hidden_states = self.residual_bank.read(hidden_states, bank)

        hidden_states = hidden_states + self.mixer(self.input_norm(hidden_states))

        if self.residual_bank is not None:
            bank = self.residual_bank.write(hidden_states, bank)

        hidden_states = hidden_states + self.moe(self.post_moe_norm(hidden_states))

        if isinstance(self.moe, MoE):
            self.last_aux_loss = self.moe.last_aux_loss
            self.last_z_loss = self.moe.last_z_loss
        else:
            self.last_aux_loss = None
            self.last_z_loss = None
        # Auxiliary losses are returned instead of read from module attributes:
        # activation checkpointing reruns the forward pass, and outputs keep the
        # gradient path intact.
        return hidden_states, bank, self.last_aux_loss, self.last_z_loss
