import torch

from qwen27b.quant import (E2M1_VALUES, QLinear, quant_act_nvfp4, quant_act_nvfp4_terms, round_e2m1, to_e4m3,
                           unpack_e2m1)


def test_unpack_low_nibble_first():
    packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)  # codes 1,2 | 8 (-0), 15 (-6)
    assert unpack_e2m1(packed).tolist() == [[0.5, 1.0, -0.0, -6.0]]


def test_round_e2m1_grid_and_ties():
    grid = torch.tensor(E2M1_VALUES)
    assert torch.equal(round_e2m1(grid), grid)
    assert torch.equal(round_e2m1(-grid), -grid)
    # ties go to the even code: 0, 1, 2, 4 are even magnitudes
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert round_e2m1(ties).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert round_e2m1(torch.tensor([0.26, 2.51, 7.0, -100.0])).tolist() == [0.5, 3.0, 6.0, -6.0]


def test_quant_act_nvfp4_properties():
    torch.manual_seed(0)
    x = torch.randn(4, 256) * 3
    scale = torch.tensor(0.01)
    q = quant_act_nvfp4(x, scale)
    # idempotent: a quantized tensor quantizes to itself
    assert torch.allclose(quant_act_nvfp4(q, scale), q)
    # each 16-element block has at most 8 distinct magnitudes, and the block max is preserved within e4m3 rounding
    blocks = q.reshape(-1, 16)
    assert max(len(b.abs().unique()) for b in blocks) <= 8
    rel = (x - q).norm() / x.norm()
    assert rel < 0.15, rel  # e2m1 has a 1-bit mantissa; ~10% RMS error is expected


def test_e4m3_saturates():
    assert to_e4m3(torch.tensor([1000.0, -1000.0])).tolist() == [448.0, -448.0]


def test_two_term_activations():
    torch.manual_seed(0)
    x, scale = torch.randn(8, 512), torch.tensor(0.021)
    one, two = quant_act_nvfp4_terms(x, scale, 1), quant_act_nvfp4_terms(x, scale, 2)
    assert torch.equal(one, quant_act_nvfp4(x, scale))
    assert torch.equal(two, one + quant_act_nvfp4(x - one, scale))
    err = lambda v: float((v - x).norm() / x.norm())
    assert err(two) < 0.02 < 0.05 < err(one)
    w = torch.randn(64, 512)
    assert torch.equal(QLinear("nvfp4", w, scale)(x), one @ w.T)
    assert torch.equal(QLinear("nvfp4", w, scale, act_terms=2)(x), two @ w.T)
