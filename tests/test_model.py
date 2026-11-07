import torch
import torch.nn.functional as F

from voidik.config import (
    AttentionConfig,
    GDNConfig,
    ModelConfig,
    MoEConfig,
    MTPConfig,
    NgramConfig,
    ResidualBankConfig,
)
from voidik.model import VoidikForCausalLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        hidden_size=128,
        num_layers=4,
        layer_pattern=("gdn", "gdn", "gdn", "qsa"),
        attention=AttentionConfig(
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            window_size=32,
            sink_tokens=2,
            block_size=16,
        ),
        gdn=GDNConfig(
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=64,
            head_v_dim=64,
            chunk_size=16,
        ),
        moe=MoEConfig(
            num_experts=4,
            top_k=2,
            expert_ff=64,
            shared_ff=64,
        ),
        residual_bank=ResidualBankConfig(slots=2, slot_dim=32),
        ngram=NgramConfig(buckets_per_order=1024, dim=16),
        mtp=MTPConfig(depth=1, ffn_ff=128),
    )


def test_forward_backward_and_mtp():
    torch.manual_seed(0)
    model = VoidikForCausalLM(tiny_config())
    input_ids = torch.randint(0, 512, (2, 33))
    outputs = model(input_ids)
    assert outputs.logits.shape == (2, 33, 512)
    assert len(outputs.mtp_logits) == 1
    assert outputs.mtp_logits[0].shape == (2, 31, 512)

    main = F.cross_entropy(
        outputs.logits[:, :32].reshape(-1, 512), input_ids[:, 1:33].reshape(-1)
    )
    mtp = F.cross_entropy(
        outputs.mtp_logits[0].reshape(-1, 512),
        input_ids[:, 2:33].reshape(-1),
    )
    total = main + 0.1 * mtp
    if outputs.aux_loss is not None:
        total = total + outputs.aux_loss * 0.01
    total.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "mtp_modules" in name:
            assert parameter.grad is not None, name
    assert model.active_parameters() < model.num_parameters()


def test_ngram_only_affects_expected_layer():
    torch.manual_seed(1)
    config = tiny_config()
    model = VoidikForCausalLM(config)
    ids_a = torch.randint(1, 512, (1, 16))
    ids_b = ids_a.clone()
    ids_b[0, -1] = 0
    # The final token cannot influence an earlier position.
    out_a = model(ids_a).logits
    out_b = model(ids_b).logits
    torch.testing.assert_close(out_a[:, :7], out_b[:, :7])


def test_causality():
    torch.manual_seed(2)
    model = VoidikForCausalLM(tiny_config())
    model.eval()
    ids_a = torch.randint(0, 512, (1, 12))
    ids_b = ids_a.clone()
    ids_b[0, 8:] = torch.randint(0, 512, (4,))
    with torch.no_grad():
        out_a = model(ids_a).logits
        out_b = model(ids_b).logits
    torch.testing.assert_close(out_a[:, :8], out_b[:, :8], rtol=1e-4, atol=1e-4)
