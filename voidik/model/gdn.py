"""Gated DeltaNet (GDN) mixer.

The training path uses a chunk-parallel form of the delta rule.  The
recurrence for one head is

    S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

where S is a d_k x d_v state.  Within a chunk the lower-triangular system is
solved directly; chunk states are then propagated sequentially.  This is the
same algebra used by Qwen3-Next's Gated DeltaNet, written in a compact and
dependency-free way.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import GDNConfig
from .norm import RMSNormGated


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps))


def gated_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Reference O(T) implementation.

    Shapes: q/k [B,H,T,Dk], v [B,H,T,Dv], beta/g [B,H,T].
    """
    batch, heads, seq_len, _ = q.shape
    state = q.new_zeros(batch, heads, q.shape[-1], v.shape[-1])
    outputs: list[torch.Tensor] = []
    for t in range(seq_len):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        beta_t = beta[:, :, t].unsqueeze(-1)
        decay = g[:, :, t].exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
        state = decay.unsqueeze(-1) * state + beta_t.unsqueeze(-1) * k_t.unsqueeze(
            -1
        ) * (v_t - decay * prediction).unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
    return torch.stack(outputs, dim=2)


def _chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Chunk-parallel implementation; returns [B,H,T,Dv]."""
    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    original_dtype = q.dtype

    # The triangular solve is sensitive to precision, so the state algebra is
    # always performed in fp32 even under bf16 autocast.
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    g = g.float()

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    total_len = q.shape[2]
    state = q.new_zeros(batch, heads, key_dim, value_dim)
    output = q.new_zeros(batch, heads, total_len, value_dim)

    lower_incl = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    )
    lower_strict = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=-1,
    )
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)

    for start in range(0, total_len, chunk_size):
        end = start + chunk_size
        q_c = q[:, :, start:end]
        k_c = k[:, :, start:end]
        v_c = v[:, :, start:end]
        beta_c = beta[:, :, start:end]
        g_c = g[:, :, start:end]

        log_decay = torch.cumsum(g_c, dim=-1)
        decay_t = log_decay.exp()
        # exp(log_decay_t - log_decay_j) for j <= t; clamp before exp so the
        # masked upper triangle cannot overflow.
        delta = (log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)).clamp(max=0.0)
        pairwise_decay = delta.exp()

        kk = torch.matmul(k_c, k_c.transpose(-1, -2))
        a_matrix = (
            kk
            * beta_c.unsqueeze(-2)
            * pairwise_decay
            * lower_strict.unsqueeze(0).unsqueeze(0)
        )
        rhs = v_c - decay_t.unsqueeze(-1) * torch.matmul(k_c, state)
        u = torch.linalg.solve_triangular(
            eye + a_matrix, rhs, upper=False, unitriangular=True
        )

        qk = torch.matmul(q_c, k_c.transpose(-1, -2))
        output_coef = (
            qk
            * beta_c.unsqueeze(-2)
            * pairwise_decay
            * lower_incl.unsqueeze(0).unsqueeze(0)
        )
        output[:, :, start:end] = (
            decay_t.unsqueeze(-1) * torch.matmul(q_c, state)
            + torch.matmul(output_coef, u)
        )

        last_decay = log_decay[:, :, -1].exp()
        state = last_decay.unsqueeze(-1).unsqueeze(-1) * state + torch.matmul(
            (
                k_c
                * (beta_c * (log_decay[:, :, -1:] - log_decay).exp()).unsqueeze(-1)
            ).transpose(-1, -2),
            u,
        )

    return output[:, :, :seq_len].to(original_dtype)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    return _chunk_gated_delta_rule(q, k, v, beta, g, chunk_size)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: GDNConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel = config.conv_kernel

        self.qk_proj = nn.Linear(hidden_size, 2 * self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.z_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if self.conv_kernel > 1:
            conv_dim = 2 * self.key_dim + self.value_dim
            self.conv1d = nn.Conv1d(
                conv_dim,
                conv_dim,
                kernel_size=self.conv_kernel,
                groups=conv_dim,
                padding=self.conv_kernel - 1,
                bias=True,
            )
        else:
            self.conv1d = None

        self.a_log = nn.Parameter(
            torch.empty(self.num_v_heads).uniform_(
                math.log(config.a_log_init_min), math.log(config.a_log_init_max)
            )
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads).uniform_(
                config.dt_bias_init_min, config.dt_bias_init_max
            )
        )
        self.o_norm = RMSNormGated(self.head_v_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids  # GDN is position-agnostic; RoPE is used by QSA only.
        batch_size, seq_len, _ = hidden_states.shape

        qk = self.qk_proj(hidden_states)
        q, k = qk.chunk(2, dim=-1)
        v = self.v_proj(hidden_states)
        z = self.z_proj(hidden_states)

        if self.conv1d is not None:
            qkv = torch.cat((q, k, v), dim=-1)
            mixed = self.conv1d(qkv.transpose(1, 2))[:, :, :seq_len]
            q, k, v = mixed.transpose(1, 2).split(
                (self.key_dim, self.key_dim, self.value_dim), dim=-1
            )
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)

        q = q.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        q = l2norm(q)
        k = l2norm(k)
        q = q * (1.0 / math.sqrt(self.head_k_dim))

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        beta = torch.sigmoid(self.b_proj(hidden_states))
        gate = -self.a_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias.float()
        )

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()
        gate = gate.transpose(1, 2).contiguous()

        core = chunk_gated_delta_rule(
            q,
            k,
            v,
            beta,
            gate,
            chunk_size=self.config.chunk_size,
        )
        core = core.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        z = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        core = self.o_norm(core, z).reshape(batch_size, seq_len, self.value_dim)
        return self.o_proj(core)
