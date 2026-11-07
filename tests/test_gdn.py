import torch

from voidik.model.gdn import chunk_gated_delta_rule, gated_delta_rule_recurrent


def test_chunk_matches_recurrent():
    torch.manual_seed(0)
    batch, heads, seq_len, dim = 1, 2, 37, 8
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)
    beta = torch.sigmoid(torch.randn(batch, heads, seq_len))
    g = -torch.rand(batch, heads, seq_len) * 0.2

    reference = gated_delta_rule_recurrent(q, k, v, beta, g)
    actual = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=16)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-3)


def test_chunk_gradients_are_finite():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 12, 4, requires_grad=True)
    k = torch.randn(1, 1, 12, 4, requires_grad=True)
    v = torch.randn(1, 1, 12, 4, requires_grad=True)
    beta = torch.rand(1, 1, 12, requires_grad=True)
    g = torch.empty(1, 1, 12).uniform_(-0.1, 0.0).requires_grad_(True)
    out = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=4)
    out.square().mean().backward()
    for tensor in (q, k, v, beta, g):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
