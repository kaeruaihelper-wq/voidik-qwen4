"""Configuration objects for Voidik-0.8B.

The model is intentionally described by plain dataclasses so that a YAML file
can be loaded without pulling in a second configuration framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


LayerType = Literal["gdn", "qsa"]


@dataclass
class AttentionConfig:
    num_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int = 128
    # Fixed block-sparse pattern used by QSA. A token can attend to:
    #   * the first `sink_tokens` tokens of the sequence;
    #   * the previous `window_size` tokens (including itself).
    window_size: int = 512
    sink_tokens: int = 4
    block_size: int = 64
    qk_norm: bool = True
    rope_theta: float = 10_000_000.0
    attention_dropout: float = 0.0
    # Qwen-style gated attention output.
    output_gate: bool = True


@dataclass
class GDNConfig:
    num_k_heads: int = 4
    num_v_heads: int = 12
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    chunk_size: int = 64
    # The decay is g = -exp(A_log) * softplus(a + dt_bias).
    a_log_init_min: float = 1.0
    a_log_init_max: float = 16.0
    dt_bias_init_min: float = 0.001
    dt_bias_init_max: float = 0.1


@dataclass
class MoEConfig:
    enabled: bool = True
    num_experts: int = 8
    top_k: int = 2
    expert_ff: int = 384
    shared_ff: int = 384
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter: float = 0.0
    normalize_topk: bool = True
    # If > 0, each expert receives at most ceil(capacity_factor * tokens / experts)
    # tokens. 0 means dropless routing (recommended for language modelling).
    capacity_factor: float = 0.0


@dataclass
class ResidualBankConfig:
    """Expanded residual stream / GR read-write memory.

    The screenshot calls this "Expanded Residual" and controls access with
    GR Read / GR Write.  Every token owns a small bank of latent slots.  The
    bank is expanded from the token embedding once, read before each mixer,
    and updated after it.  No cross-token state is kept, which keeps training
    and incremental decoding simple.
    """

    enabled: bool = True
    slots: int = 4
    slot_dim: int = 128
    read_bias_init: float = -2.0
    write_bias_init: float = -2.0
    # Add the final bank contents back to the hidden stream before lm_head.
    merge_at_end: bool = True


@dataclass
class NgramConfig:
    enabled: bool = True
    orders: tuple[int, ...] = (2, 3)
    buckets_per_order: int = 131_072
    dim: int = 32
    # Zero-based transformer layer at which the n-gram vector is injected.
    # The paper/screenshot says "Layer 2 only", hence 1.
    inject_layer: int = 1
    scale: float = 1.0
    dropout: float = 0.0
    hash_seed: int = 0


@dataclass
class MTPConfig:
    enabled: bool = True
    depth: int = 1
    loss_weight: float = 0.1
    # MTP uses a lightweight GDN block and a dense SwiGLU FFN.
    ffn_ff: int = 4096
    aux_loss_coef: float = 0.0


@dataclass
class ModelConfig:
    vocab_size: int = 151_936
    hidden_size: int = 1536
    num_layers: int = 20
    # A pattern of 3 GDN layers followed by 1 QSA layer, repeated 5 times.
    layer_pattern: tuple[LayerType, ...] = ("gdn", "gdn", "gdn", "qsa")
    tie_embeddings: bool = True
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = False

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    residual_bank: ResidualBankConfig = field(default_factory=ResidualBankConfig)
    ngram: NgramConfig = field(default_factory=NgramConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.layer_pattern:
            raise ValueError("layer_pattern cannot be empty")
        if self.hidden_size != self.attention.num_heads * self.attention.head_dim:
            raise ValueError(
                "hidden_size must equal attention.num_heads * attention.head_dim"
            )
        if self.attention.num_heads % self.attention.num_kv_heads:
            raise ValueError("attention.num_heads must be divisible by num_kv_heads")
        if self.gdn.num_v_heads % self.gdn.num_k_heads:
            raise ValueError("gdn.num_v_heads must be divisible by gdn.num_k_heads")
        if self.gdn.num_v_heads * self.gdn.head_v_dim != self.hidden_size:
            raise ValueError(
                "gdn.num_v_heads * gdn.head_v_dim must equal hidden_size"
            )
        if self.ngram.inject_layer < 0:
            raise ValueError("ngram.inject_layer must be >= 0")
        if self.ngram.inject_layer >= self.num_layers:
            raise ValueError("ngram.inject_layer must be < num_layers")
        if self.mtp.depth < 0:
            raise ValueError("mtp.depth must be >= 0")

    def layer_types(self) -> list[LayerType]:
        return [
            self.layer_pattern[i % len(self.layer_pattern)]
            for i in range(self.num_layers)
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        values = dict(values)
        attention = AttentionConfig(**values.pop("attention", {}))
        gdn = GDNConfig(**values.pop("gdn", {}))
        moe = MoEConfig(**values.pop("moe", {}))
        residual_bank = ResidualBankConfig(**values.pop("residual_bank", {}))
        ngram_values = dict(values.pop("ngram", {}))
        if "orders" in ngram_values:
            ngram_values["orders"] = tuple(ngram_values["orders"])
        ngram = NgramConfig(**ngram_values)
        mtp = MTPConfig(**values.pop("mtp", {}))
        if "layer_pattern" in values:
            values["layer_pattern"] = tuple(values["layer_pattern"])
        unknown = set(values) - {field_.name for field_ in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown ModelConfig fields: {sorted(unknown)}")
        return cls(
            **values,
            attention=attention,
            gdn=gdn,
            moe=moe,
            residual_bank=residual_bank,
            ngram=ngram,
            mtp=mtp,
        )


@dataclass
class DataConfig:
    train_glob: str = "data/fineweb/train_*.bin"
    val_glob: str = "data/fineweb/val_*.bin"
    block_size: int = 1024
    token_dtype: str = "uint32"
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    # Random crops per nominal epoch. Only used by the sampler's length.
    tokens_per_epoch: int = 1_000_000_000


@dataclass
class TrainConfig:
    output_dir: str = "runs/voidik-0.8b"
    seed: int = 42
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    optimizer: Literal["adamw", "adafactor"] = "adamw"
    fused_adamw: bool = True
    batch_size: int = 1
    grad_accum: int = 32
    max_steps: int = 100_000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 1000
    eval_batches: int = 20
    save_every: int = 1000
    keep_last_checkpoints: int = 3
    gradient_checkpointing: bool = True
    compile: bool = False
    tensorboard: bool = False
    log_csv: bool = True
    resume: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = dict(raw)
        return cls(
            model=ModelConfig.from_dict(raw.pop("model", {})),
            data=DataConfig(**raw.pop("data", {})),
            train=TrainConfig(**raw.pop("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
