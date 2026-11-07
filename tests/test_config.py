from pathlib import Path

from voidik.config import Config
from voidik.model import VoidikForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def test_reference_config_loads_and_validates():
    config = Config.load(ROOT / "configs" / "voidik_0.8b.yaml")
    assert config.model.num_layers == 20
    assert config.model.layer_types().count("gdn") == 15
    assert config.model.layer_types().count("qsa") == 5
    assert config.model.ngram.inject_layer == 1


def test_smoke_config_loads():
    config = Config.load(ROOT / "configs" / "smoke.yaml")
    assert config.data.block_size == 64
    assert config.model.mtp.enabled


def test_58m_config_preserves_hybrid_architecture_and_parameter_budget():
    config = Config.load(ROOT / "configs" / "voidik_58m.yaml")
    model = VoidikForCausalLM(config.model)

    assert config.model.num_layers == 8
    assert config.model.layer_types().count("gdn") == 6
    assert config.model.layer_types().count("qsa") == 2
    assert config.model.moe.enabled
    assert config.model.ngram.enabled
    assert config.model.residual_bank.enabled
    assert config.model.mtp.enabled
    assert 58_000_000 <= model.num_parameters() <= 60_000_000
