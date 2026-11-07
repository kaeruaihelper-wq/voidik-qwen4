"""Qwen-style sparse attention (QSA).

QSA is implemented as a fixed block-sparse causal attention pattern:
every token sees a local window plus a small number of attention-sink tokens
at the beginning of the sequence.  This keeps the asymptotic cost linear in
sequence length while retaining the "global" information path provided by the
sinks.  The implementation uses PyTorch SDPA and does not require FlashAttention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AttentionConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb


def build_sparse_mask(
    q_positions: torch.Tensor,
    key_positions: torch.Tensor,
    window_size: int,
    sink_tokens: int,
) -> torch.Tensor:
    """Boolean mask [q, k]; True means the key is visible."""
    causal = key_positions[None, :] <= q_positions[:, None]
    local = (q_positions[:, None] - key_positions[None, :]) < window_size
    mask = causal & local
    if sink_tokens > 0:
        mask = mask | (key_positions[None, :] < sink_tokens)
    return mask


def _causal_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_tokens: int,
    dropout_p: float,
    scale: float,
) -> torch.Tensor:
    seq_len = q.shape[-2]
    positions = torch.arange(seq_len, device=q.device)
    mask = build_sparse_mask(positions, positions, window_size, sink_tokens)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask[None, None],
        dropout_p=dropout_p if q.requires_grad else 0.0,
        scale=scale,
    )


def _block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_tokens: int,
    block_size: int,
    dropout_p: float,
    scale: float,
) -> torch.Tensor:
    seq_len = q.shape[-2]
    outputs: list[torch.Tensor] = []
    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        key_start = max(0, q_start - window_size + 1)
        local_keys = torch.arange(key_start, q_end, device=q.device)
        if sink_tokens > 0:
            sink_keys = torch.arange(0, min(sink_tokens, seq_len), device=q.device)
            key_indices = torch.cat((sink_keys, local_keys)).unique(sorted=True)
        else:
            key_indices = local_keys

        q_block = q[:, :, q_start:q_end]
        k_block = k.index_select(-2, key_indices)
        v_block = v.index_select(-2, key_indices)
        q_pos = torch.arange(q_start, q_end, device=q.device)
        mask = build_sparse_mask(q_pos, key_indices, window_size, sink_tokens)
        block_out = F.scaled_dot_product_attention(
            q_block,
            k_block,
            v_block,
            attn_mask=mask[None, None],
            dropout_p=dropout_p if q.requires_grad else 0.0,
            scale=scale,
        )
        outputs.append(block_out)
    return torch.cat(outputs, dim=-2)


class QwenSparseAttention(nn.Module):
    def __init__(self, config: AttentionConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.output_gate = (
            nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
            if config.output_gate
            else None
        )
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)

        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary(position_ids, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Dense SDPA is faster for short sequences; the block loop matters for
        # long-context training and avoids materialising a T x T mask.
        if seq_len <= self.config.window_size + self.config.sink_tokens:
            attn_output = _causal_dense_attention(
                q,
                k,
                v,
                self.config.window_size,
                self.config.sink_tokens,
                self.config.attention_dropout,
                self.scaling,
            )
        else:
            attn_output = _block_sparse_attention(
                q,
                k,
                v,
                self.config.window_size,
                self.config.sink_tokens,
                self.config.block_size,
                self.config.attention_dropout,
                self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        if self.output_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.output_gate(hidden_states))
        return self.o_proj(attn_output)
