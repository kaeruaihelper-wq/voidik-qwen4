"""Dropless top-k mixture of experts with a shared expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MoEConfig
from .norm import SwiGLU


class MoE(nn.Module):
    def __init__(self, config: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_size, config.expert_ff) for _ in range(self.num_experts)]
        )
        self.shared_expert = SwiGLU(hidden_size, config.shared_ff)
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_router_probs: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)

        router_logits = self.router(flat)
        if self.training and self.config.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * (
                self.config.router_jitter
            )
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(
            router_probs, self.top_k, dim=-1, sorted=False
        )
        if self.config.normalize_topk:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(
                1e-9
            )

        output = self.shared_expert(flat)
        for expert_idx in range(self.num_experts):
            token_mask = top_indices == expert_idx
            if not token_mask.any():
                continue
            token_idx, slot_idx = token_mask.nonzero(as_tuple=True)
            expert_output = self.experts[expert_idx](flat.index_select(0, token_idx))
            weight = top_weights[token_idx, slot_idx].unsqueeze(-1).to(
                expert_output.dtype
            )
            output.index_add_(0, token_idx, expert_output * weight)

        # Switch-style load-balancing loss and ST-MoE router z-loss.  Both are
        # accumulated by the model and added to the language-modelling loss.
        tokens_per_expert = torch.bincount(
            top_indices.reshape(-1), minlength=self.num_experts
        ).float()
        fraction = tokens_per_expert / max(flat.shape[0] * self.top_k, 1)
        mean_prob = router_probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * (fraction * mean_prob).sum()
        log_z = torch.logsumexp(router_logits, dim=-1)
        self.last_z_loss = log_z.pow(2).mean()
        self.last_router_probs = mean_prob.detach()

        return output.view(batch_size, seq_len, hidden_size)
