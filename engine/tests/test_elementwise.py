"""Elementwise kernels against the reference numerics (reference/qwen27b)."""

import pytest
import torch
import torch.nn.functional as F

from q27lib import call, lib
from qwen27b.model import rms_norm
from qwen27b.quant import quant_act_fp8, quant_act_nvfp4, unpack_e2m1

dev = "cuda"


def sf_index(rows, kblocks):
    """Swizzled offset of every (row, kblock), as a [rows, kblocks] long tensor (mirror of sf_offset)."""
    r = torch.arange(rows)[:, None]
    kb = torch.arange(kblocks)[None, :]
    katoms = (kblocks + 3) // 4
    return ((r // 128) * katoms + kb // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (kb % 4)


def unswizzle(sf, rows, kblocks):
    return sf.cpu()[sf_index(rows, kblocks).flatten()].view(rows, kblocks)


def test_rmsnorm():
    x = torch.randn(7, 5120, device=dev) * 3
    w = (torch.randn(5120, device=dev) * 0.1).bfloat16()
    out = torch.empty_like(x)
    call("q27t_rmsnorm", x, w, 7, 5120, 1e-6, 1, out)
    torch.testing.assert_close(out, rms_norm(x, w, 1e-6), rtol=1e-5, atol=1e-5)
    call("q27t_rmsnorm", x, w, 7, 5120, 1e-6, 0, out)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_embed_gather_device_and_pinned():
    table = torch.randn(1000, 5120).bfloat16()
    ids = torch.tensor([3, 999, 0, 3], dtype=torch.int32, device=dev)
    out = torch.empty(4, 5120, device=dev)
    for t in (table.to(dev), table.pin_memory()):
        call("q27t_embed_gather", t, ids, 4, 5120, out)
        torch.testing.assert_close(out.cpu(), table[ids.long().cpu()].float(), rtol=0, atol=0)


def test_quant_fp4_matches_reference():
    M, K = 37, 5120
    x = torch.randn(M, K, device=dev) * torch.logspace(-2, 1, M, device=dev)[:, None]
    x[3, :16] = 0  # an all-zero block
    s = torch.tensor(0.0123)
    q = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    sf = torch.zeros(lib().q27t_sf_bytes(M, K // 16), dtype=torch.uint8, device=dev)
    call("q27t_quant_fp4", x, M, K, float(s), q, sf)
    vals = unpack_e2m1(q.cpu())
    scales = unswizzle(sf, M, K // 16).view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    deq = vals * scales / (1.0 / s)  # q * sf / gs, in the reference's order of operations
    ref = quant_act_nvfp4(x.cpu(), s)
    torch.testing.assert_close(deq, ref, rtol=0, atol=0)


def test_quant_fp8_matches_reference():
    x = torch.randn(3, 6144, device=dev) * 50
    s = torch.tensor(0.07)
    q = torch.empty(3, 6144, dtype=torch.uint8, device=dev)
    call("q27t_quant_fp8", x, x.numel(), float(s), q)
    deq = q.cpu().view(torch.float8_e4m3fn).float() * s
    torch.testing.assert_close(deq, quant_act_fp8(x.cpu(), s), rtol=0, atol=0)


def test_silu_mul():
    g, u = torch.randn(2, 17408, device=dev) * 4, torch.randn(2, 17408, device=dev)
    out = torch.empty_like(g)
    call("q27t_silu_mul", g, u, g.numel(), out)
    torch.testing.assert_close(out, F.silu(g) * u, rtol=1e-5, atol=1e-6)


def test_linear_bf16():
    x = torch.randn(5, 5120, device=dev)
    w = torch.randn(48, 5120, device=dev).bfloat16()
    out = torch.empty(5, 48, device=dev)
    call("q27t_linear_bf16", x, w, 5, 48, 5120, out)
    torch.testing.assert_close(out, x @ w.float().T, rtol=1e-4, atol=1e-3)


def test_swizzle_sf_roundtrip():
    rows, kb = 300, 1088
    src = torch.randint(0, 255, (rows, kb), dtype=torch.uint8, device=dev)
    dst = torch.zeros(lib().q27t_sf_bytes(rows, kb), dtype=torch.uint8, device=dev)
    call("q27t_swizzle_sf", src, rows, kb, dst)
    assert torch.equal(unswizzle(dst, rows, kb), src.cpu())


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_rmsnorm_quant(kind):
    T, D = 5, 5120
    x = torch.randn(T, D, device=dev) * 3
    w = (torch.randn(D, device=dev) * 0.1).bfloat16()
    s = 0.0123
    out = torch.empty(T, D, device=dev)
    ref = rms_norm(x, w, 1e-6)
    if kind == "fp8":
        q = torch.empty(T, D, dtype=torch.uint8, device=dev)
        call("q27t_rmsnorm_quant", x, w, T, 1e-6, out, 1, s, q, None)
        deq = q.cpu().view(torch.float8_e4m3fn).float() * s
        qref = quant_act_fp8(out.cpu(), torch.tensor(s))
    else:
        q = torch.empty(T, D // 2, dtype=torch.uint8, device=dev)
        sf = torch.zeros(lib().q27t_sf_bytes(T, D // 16), dtype=torch.uint8, device=dev)
        call("q27t_rmsnorm_quant", x, w, T, 1e-6, out, 2, s, q, sf)
        scales = unswizzle(sf, T, D // 16).view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        deq = unpack_e2m1(q.cpu()) * scales / (1.0 / torch.tensor(s))
        qref = quant_act_nvfp4(out.cpu(), torch.tensor(s))
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    # The quantized copy is exactly the quantization of the fp32 output.
    torch.testing.assert_close(deq, qref, rtol=0, atol=0)


def test_silu_mul_quant_fp4_matches_separate_kernels():
    T, I = 3, 17408
    gu = torch.randn(T, 2 * I, device=dev) * 4  # gate | up per row (the fused GEMM's output)
    s = 0.02
    q = torch.empty(T, I // 2, dtype=torch.uint8, device=dev)
    sf = torch.zeros(lib().q27t_sf_bytes(T, I // 16), dtype=torch.uint8, device=dev)
    call("q27t_silu_mul_quant_fp4", gu, gu[:, I:], 2 * I, T, I, s, q, sf)
    h = torch.empty(T, I, device=dev)
    g, u = gu[:, :I].contiguous(), gu[:, I:].contiguous()
    call("q27t_silu_mul", g, u, g.numel(), h)
    q2, sf2 = torch.empty_like(q), torch.zeros_like(sf)
    call("q27t_quant_fp4", h, T, I, s, q2, sf2)
    assert torch.equal(q, q2) and torch.equal(sf, sf2)


def test_linear_bf16_pair():
    T, N, K = 3, 48, 5120
    x = torch.randn(T, K, device=dev)
    wa, wb = torch.randn(N, K, device=dev).bfloat16(), torch.randn(N, K, device=dev).bfloat16()
    oa, ob = torch.empty(T, N, device=dev), torch.empty(T, N, device=dev)
    call("q27t_linear_bf16_pair", x, wa, wb, T, N, K, oa, ob)
    torch.testing.assert_close(oa, x @ wa.float().T, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(ob, x @ wb.float().T, rtol=1e-4, atol=1e-3)


def test_linear_bf16_pair_prefill():
    """T > 8 (tensor cores, x split into bf16 hi + lo): near fp32, rows independent of T and placement."""
    N, K = 48, 5120
    x = torch.randn(300, K, device=dev)
    wa, wb = (torch.randn(N, K, device=dev) * 0.02).bfloat16(), (torch.randn(N, K, device=dev) * 0.02).bfloat16()
    outs = []
    for lo, hi in ((0, 300), (0, 9), (7, 300), (100, 233)):
        oa, ob = torch.empty(hi - lo, N, device=dev), torch.empty(hi - lo, N, device=dev)
        call("q27t_linear_bf16_pair", x[lo:hi], wa, wb, hi - lo, N, K, oa, ob)
        outs.append((lo, hi, oa, ob))
    _, _, oa, ob = outs[0]
    ref_a, ref_b = (x.double() @ wa.double().T), (x.double() @ wb.double().T)
    assert ((oa.double() - ref_a).norm() / ref_a.norm()).item() < 2e-5
    assert ((ob.double() - ref_b).norm() / ref_b.norm()).item() < 2e-5
    for lo, hi, a, b in outs[1:]:
        assert torch.equal(a, oa[lo:hi]) and torch.equal(b, ob[lo:hi]), (lo, hi)


def test_silu_mul_quant_fp4_bf16_matches_fp32_on_bf16_values():
    """The BF16-input version (prefill MLP) is the fp32 version applied to the same (BF16) values."""
    T, I = 5, 17408
    g, u = (torch.randn(T, I, device=dev) * 4).bfloat16(), (torch.randn(T, I, device=dev) * 4).bfloat16()
    s = 0.02
    q, q2 = (torch.empty(T, I // 2, dtype=torch.uint8, device=dev) for _ in range(2))
    sf, sf2 = (torch.zeros(lib().q27t_sf_bytes(T, I // 16), dtype=torch.uint8, device=dev) for _ in range(2))
    call("q27t_silu_mul_quant_fp4_bf16", g, u, I, T, I, s, q, sf)
    call("q27t_silu_mul_quant_fp4", g.float(), u.float(), I, T, I, s, q2, sf2)
    assert torch.equal(q, q2) and torch.equal(sf, sf2)
