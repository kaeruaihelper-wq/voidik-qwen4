import torch

from voidik.config import NgramConfig
from voidik.model.ngram import NgramEmbedding


def test_ngram_is_deterministic_and_causal():
    torch.manual_seed(0)
    layer = NgramEmbedding(NgramConfig(orders=(2, 3), buckets_per_order=64, dim=8), 16)
    ids = torch.randint(1, 100, (2, 20))
    out_a = layer(ids)
    out_b = layer(ids)
    torch.testing.assert_close(out_a, out_b)

    changed_future = ids.clone()
    changed_future[:, 10:] = torch.randint(1, 100, changed_future[:, 10:].shape)
    out_c = layer(changed_future)
    torch.testing.assert_close(out_a[:, :10], out_c[:, :10])


def test_ngram_first_positions_are_zero():
    layer = NgramEmbedding(NgramConfig(orders=(3,), buckets_per_order=64, dim=8), 16)
    ids = torch.randint(1, 100, (1, 10))
    out = layer(ids)
    # A trigram needs two preceding tokens.
    torch.testing.assert_close(out[:, 0], torch.zeros_like(out[:, 0]))
    torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))
    assert out[:, 2].abs().sum() > 0
