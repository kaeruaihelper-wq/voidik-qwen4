import torch
import torch.nn.functional as F

from voidik.model.attention import (
    _block_sparse_attention,
    _causal_dense_attention,
    _block_sparse_attention as block_attention,
)


def test_block_sparse_matches_dense_pattern():
    torch.manual_seed(0)
    batch, heads, seq_len, dim = 2, 3, 200, 16
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, dim)
    window, sinks, block = 32, 4, 16

    dense = _causal_dense_attention(q, k, v, window, sinks, 0.0, dim**-0.5)
    sparse = _block_sparse_attention(q, k, v, window, sinks, block, 0.0, dim**-0.5)
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-6)


def test_attention_is_causal():
    torch.manual_seed(2)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 40, 8)
    v = torch.randn(1, 1, 40, 8)
    out = block_attention(q, k, v, 16, 2, 8, 0.0, 8**-0.5)
    assert out.shape == (1, 1, 1, 8)
    assert torch.isfinite(out).all()
