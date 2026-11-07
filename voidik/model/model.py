"""Voidik hybrid decoder-only language model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .block import HybridBlock
from .mtp import MTPModule
from .ngram import NgramEmbedding
from .norm import RMSNorm


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    mtp_logits: list[torch.Tensor] = field(default_factory=list)
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


class VoidikForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                HybridBlock(config, layer_type)
                for layer_type in config.layer_types()
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.ngram = (
            NgramEmbedding(config.ngram, config.hidden_size)
            if config.ngram.enabled
            else None
        )

        if config.residual_bank.enabled and config.residual_bank.merge_at_end:
            self.bank_final_norm = RMSNorm(
                config.residual_bank.slots * config.residual_bank.slot_dim
            )
            self.bank_merge = nn.Linear(
                config.residual_bank.slots * config.residual_bank.slot_dim,
                config.hidden_size,
                bias=False,
            )
        else:
            self.bank_final_norm = None
            self.bank_merge = None

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp_modules = nn.ModuleList(
            [
                MTPModule(config.mtp, config)
                for _ in range(config.mtp.depth if config.mtp.enabled else 0)
            ]
        )
        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        self._post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _post_init(self) -> None:
        # The generic initializer resets these deliberately biased gates.
        for layer in self.layers:
            if layer.residual_bank is not None:
                nn.init.constant_(
                    layer.residual_bank.read_gate.bias,
                    self.config.residual_bank.read_bias_init,
                )
                nn.init.constant_(
                    layer.residual_bank.write_gate.bias,
                    self.config.residual_bank.write_bias_init,
                )
            if hasattr(layer.moe, "router"):
                nn.init.normal_(layer.moe.router.weight, std=0.02)
            # Residual output projections are scaled at initialization, as in
            # GPT-NeoX / Qwen models, to keep activation variance stable.
            scale = 1.0 / math.sqrt(2.0 * max(self.config.num_layers, 1))
            layer.mixer.o_proj.weight.data.mul_(scale)
            if hasattr(layer.moe, "experts"):
                for expert in layer.moe.experts:
                    expert.down_proj.weight.data.mul_(scale)
                layer.moe.shared_expert.down_proj.weight.data.mul_(scale)
            elif hasattr(layer.moe, "down_proj"):
                layer.moe.down_proj.weight.data.mul_(scale)

        for module in self.mtp_modules:
            module.mixer.o_proj.weight.data.mul_(
                1.0 / math.sqrt(2.0 * max(self.config.num_layers, 1))
            )
            module.ffn.down_proj.weight.data.mul_(
                1.0 / math.sqrt(2.0 * max(self.config.num_layers, 1))
            )

        if self.ngram is not None:
            nn.init.normal_(
                self.ngram.proj.weight,
                std=self.config.ngram.scale / math.sqrt(self.config.hidden_size),
            )
        if self.bank_merge is not None:
            nn.init.normal_(self.bank_merge.weight, std=0.02)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def num_parameters(self, only_trainable: bool = True) -> int:
        parameters = self.parameters()
        if only_trainable:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def active_parameters(self) -> int:
        """Total minus the unselected expert parameters in every MoE layer."""
        inactive = 0
        for layer in self.layers:
            moe = getattr(layer, "moe", None)
            if isinstance(moe, nn.Module) and hasattr(moe, "experts"):
                for expert_idx, expert in enumerate(moe.experts):
                    if expert_idx >= moe.top_k:
                        inactive += sum(p.numel() for p in expert.parameters())
        return self.num_parameters() - inactive

    def _apply_ngram(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        if self.ngram is None:
            return hidden_states
        return hidden_states + self.ngram(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        del labels  # loss is computed by the training loop, where MTP alignment lives.
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        ngram_inject_layer = (
            self.config.ngram.inject_layer if self.ngram is not None else -1
        )
        if ngram_inject_layer == 0:
            hidden_states = self._apply_ngram(hidden_states, input_ids)

        bank = None
        if self.layers and self.layers[0].residual_bank is not None:
            bank = self.layers[0].residual_bank.initial_bank(hidden_states)

        aux_losses: list[torch.Tensor] = []
        z_losses: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == ngram_inject_layer and layer_idx != 0:
                hidden_states = self._apply_ngram(hidden_states, input_ids)
            if self.gradient_checkpointing and self.training:
                hidden_states, bank, aux, z_loss = checkpoint(
                    layer,
                    hidden_states,
                    bank,
                    use_reentrant=False,
                )
            else:
                hidden_states, bank, aux, z_loss = layer(hidden_states, bank)
            if aux is not None:
                aux_losses.append(aux)
            if z_loss is not None:
                z_losses.append(z_loss)

        if bank is not None and self.bank_merge is not None:
            bank_flat = bank.reshape(*bank.shape[:-2], -1)
            hidden_states = hidden_states + self.bank_merge(
                F.silu(self.bank_final_norm(bank_flat))
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        mtp_logits: list[torch.Tensor] = []
        if self.training and self.mtp_modules:
            for depth_idx, module in enumerate(self.mtp_modules, start=1):
                keep = seq_len - depth_idx - 1
                if keep <= 0:
                    continue
                h = hidden_states[:, :keep]
                # Teacher forcing: token t+1 is known during training.
                next_emb = self.embed_tokens(input_ids[:, 1 : 1 + keep])
                mtp_hidden = module(h, next_emb)
                mtp_logits.append(self.lm_head(mtp_hidden))

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        z_loss = torch.stack(z_losses).mean() if z_losses else None
        return CausalLMOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            aux_loss=aux_loss,
            z_loss=z_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Reference decoder without KV cache.

        This is intentionally simple and is useful for smoke tests only; the
        training path does not depend on it.  A cached decoder can be added
        once the architecture is frozen.
        """
        was_training = self.training
        self.eval()
        finished = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=input_ids.device
        )
        for _ in range(max_new_tokens):
            outputs = self(input_ids)
            logits = outputs.logits[:, -1, :].float()
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k, dim=-1)
                    probs = F.softmax(values, dim=-1)
                    sampled = torch.multinomial(probs, 1)
                    next_token = indices.gather(-1, sampled)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
            if eos_token_id is not None and finished.any():
                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1), eos_token_id
                )
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1) == eos_token_id
                if finished.all():
                    break
        self.train(was_training)
        return input_ids
