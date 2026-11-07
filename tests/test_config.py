from pathlib import Path

from voidik.config import Config


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
