import pytest
import torch
import torch.nn.functional as F

from qwen27b.model import gated_delta_chunked, gated_delta_recurrent, l2norm


def inputs(T, H=6, dk=16, dv=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(T, H, dk, generator=g, dtype=torch.float64)) * dk ** -0.5
    k = l2norm(torch.randn(T, H, dk, generator=g, dtype=torch.float64))
    v = torch.randn(T, H, dv, generator=g, dtype=torch.float64)
    gate = -F.softplus(torch.randn(T, H, generator=g, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(T, H, generator=g, dtype=torch.float64))
    s0 = torch.randn(H, dk, dv, generator=g, dtype=torch.float64) * 0.1
    return q, k, v, gate, beta, s0


@pytest.mark.parametrize("T", [1, 7, 64, 100, 200])
def test_chunked_matches_recurrent(T):
    q, k, v, g, b, s0 = inputs(T)
    o1, s1 = gated_delta_recurrent(q, k, v, g, b, s0)
    o2, s2 = gated_delta_chunked(q, k, v, g, b, s0, chunk=64)
    assert torch.allclose(o1, o2, atol=1e-9), (o1 - o2).abs().max()
    assert torch.allclose(s1, s2, atol=1e-9), (s1 - s2).abs().max()


def test_split_calls_equal_one_call():
    q, k, v, g, b, s0 = inputs(150, seed=1)
    o, s = gated_delta_chunked(q, k, v, g, b, s0)
    oa, sa = gated_delta_chunked(q[:70], k[:70], v[:70], g[:70], b[:70], s0)
    ob, sb = gated_delta_recurrent(q[70:], k[70:], v[70:], g[70:], b[70:], sa)
    assert torch.allclose(torch.cat((oa, ob)), o, atol=1e-9)
    assert torch.allclose(sb, s, atol=1e-9)
