"""Skinny decode GEMMs (mma.sync) against fp64 matmuls of the dequantized operands, and their speed."""

import ctypes
import os

import pytest
import torch

from q27lib import call, lib
from test_gemm import fp4_operand, rel_err

from qwen27b.quant import quant_act_nvfp4, unpack_e2m1

dev = "cuda"
WS = 32 << 20


def swizzled(sf, rows, kblocks):
    buf = torch.zeros(lib().q27t_sf_bytes(rows, kblocks), dtype=torch.uint8, device=dev)
    call("q27t_swizzle_sf", sf.view(torch.uint8).to(dev), rows, kblocks, buf)
    return buf


SHAPES_FP4 = [(5120, 17408), (17408, 5120), (34816, 5120), (128, 5120), (248320, 5120)]
SHAPES_FP8 = [(10240, 5120), (5120, 6144), (1024, 5120), (16384, 5120), (12288, 5120)]


@pytest.mark.parametrize("M", [1, 2, 3, 4, 8, 13, 16, 29, 32, 64])
@pytest.mark.parametrize("N,K", SHAPES_FP4)
def test_skinny_fp4(M, N, K):
    a, asf, adq = fp4_operand(M, K, 1.0)
    b, bsf, bdq = fp4_operand(N, K, 1.0)
    asw, bsw = swizzled(asf, M, K // 16), swizzled(bsf, N, K // 16)
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    c = torch.randn(M, N, device=dev)
    d = c.clone()
    alpha = 0.37
    ref = alpha * (adq.double() @ bdq.double().T)
    for _ in range(2):  # the second call checks that the split-K counters were reset
        d.copy_(c)
        call("q27t_skinny_fp4", M, N, K, a.to(dev), asw, b.to(dev), bsw, alpha, d, 1.0, d, ws, WS)
        assert rel_err(d.cpu().double(), ref + c.cpu().double()) < 1e-5
    call("q27t_skinny_fp4", M, N, K, a.to(dev), asw, b.to(dev), bsw, alpha, None, 0.0, d, ws, WS)
    assert rel_err(d.cpu().double(), ref) < 1e-5


@pytest.mark.parametrize("M", [1, 3, 8, 16, 31, 64])
@pytest.mark.parametrize("N,K", SHAPES_FP8)
def test_skinny_fp8(M, N, K):
    a = (torch.randn(M, K) * 30).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn)
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    c = torch.randn(M, N, device=dev) * 1000
    d = c.clone()
    alpha = 0.011
    call("q27t_skinny_fp8", M, N, K, a.view(torch.uint8).to(dev), b.view(torch.uint8).to(dev), alpha, d, 1.0, d,
         ws, WS)
    assert rel_err(d.cpu().double(), alpha * (a.double() @ b.double().T) + c.cpu().double()) < 1e-5


@pytest.mark.parametrize("fmt", ["fp4", "fp8", "bf16"])
def test_rows_independent_of_m(fmt):
    """A token's row is bit-identical whatever the number of tokens (speculative verify vs decode), with the
    M-independent split policy (Q27_SKINNY_EXACT=1, read at every launch)."""
    os.environ["Q27_SKINNY_EXACT"] = "1"
    N, K = 5120, 17408
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    outs = {}
    if fmt == "fp4":
        a, asf, _ = fp4_operand(64, K, 1.0)
        b, bsf, _ = fp4_operand(N, K, 1.0)
        asw, bsw = swizzled(asf, 64, K // 16), swizzled(bsf, N, K // 16)
        a, b = a.to(dev), b.to(dev)
    elif fmt == "fp8":
        a = (torch.randn(64, K) * 30).to(torch.float8_e4m3fn).view(torch.uint8).to(dev)
        b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn).view(torch.uint8).to(dev)
    else:
        a = torch.randn(64, K, device=dev).to(torch.bfloat16)
        b = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    for M in (1, 5, 8, 9, 16, 32, 64):
        d = torch.empty(M, N, device=dev)
        if fmt == "fp4":
            call("q27t_skinny_fp4", M, N, K, a, asw, b, bsw, 1.0, None, 0.0, d, ws, WS)
        elif fmt == "fp8":
            call("q27t_skinny_fp8", M, N, K, a, b, 1.0, None, 0.0, d, ws, WS)
        else:
            call("q27t_skinny_bf16", M, N, K, a, b, d, ws, WS)
        outs[M] = d.cpu()
    os.environ.pop("Q27_SKINNY_EXACT")
    for M, d in outs.items():
        assert torch.equal(d, outs[64][:M]), f"M={M}"


@pytest.mark.parametrize("fmt", ["fp4", "fp8"])
def test_rows_independent_of_m_residual(fmt):
    """The same with the residual added in place (d == c, as the model's out projections): the epilogue's
    rounding must not depend on the code path of the M class (a batched step's rows equal a single
    sequence's, P6-10). The default split policy (M > 1 splits alike)."""
    N, K = 5120, 17408 if fmt == "fp4" else 6144
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    if fmt == "fp4":
        a, asf, _ = fp4_operand(64, K, 1.0)
        b, bsf, _ = fp4_operand(N, K, 1.0)
        asw, bsw = swizzled(asf, 64, K // 16), swizzled(bsf, N, K // 16)
        a, b = a.to(dev), b.to(dev)
    else:
        a = (torch.randn(64, K) * 30).to(torch.float8_e4m3fn).view(torch.uint8).to(dev)
        b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn).view(torch.uint8).to(dev)
    c = torch.randn(64, N, device=dev) * 100
    outs = {}
    for M in (2, 8, 9, 16, 29, 32, 41, 64):
        d = c[:M].clone()
        if fmt == "fp4":
            call("q27t_skinny_fp4", M, N, K, a, asw, b, bsw, 0.37, d, 1.0, d, ws, WS)
        else:
            call("q27t_skinny_fp8", M, N, K, a, b, 0.01, d, 1.0, d, ws, WS)
        outs[M] = d.cpu()
    for M, d in outs.items():
        assert torch.equal(d, outs[64][:M]), f"M={M}"


def test_shared_workspace():
    """Shapes with different tile counts and splits, interleaved on one workspace (as in the model)."""
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    cases = []
    for N, K in [(17408, 5120), (5120, 17408), (5120, 5120), (248320, 5120)]:
        a, asf, adq = fp4_operand(3, K, 1.0)
        b, bsf, bdq = fp4_operand(N, K, 1.0)
        cases.append((N, K, a.to(dev), swizzled(asf, 3, K // 16), b.to(dev), swizzled(bsf, N, K // 16),
                      adq.double() @ bdq.double().T))
    for _ in range(3):
        for N, K, a, asw, b, bsw, ref in cases:
            d = torch.empty(3, N, device=dev)
            call("q27t_skinny_fp4", 3, N, K, a, asw, b, bsw, 1.0, None, 0.0, d, ws, WS)
            assert rel_err(d.cpu().double(), ref) < 1e-5


def test_deterministic():
    M, N, K = 1, 5120, 17408
    a, asf, _ = fp4_operand(M, K, 1.0)
    b, bsf, _ = fp4_operand(N, K, 1.0)
    asw, bsw = swizzled(asf, M, K // 16), swizzled(bsf, N, K // 16)
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    outs = []
    for _ in range(3):
        d = torch.empty(M, N, device=dev)
        call("q27t_skinny_fp4", M, N, K, a.to(dev), asw, b.to(dev), bsw, 1.0, None, 0.0, d, ws, WS)
        outs.append(d)
    assert all(torch.equal(outs[0], o) for o in outs[1:])


BENCH = [(1, 5120, 17408), (1, 17408, 5120), (1, 34816, 5120), (1, 248320, 5120), (4, 34816, 5120), (8, 5120, 17408)]
BENCH8 = [(1, 10240, 5120), (1, 6144, 5120), (1, 5120, 6144), (1, 1024, 5120), (1, 16384, 5120), (4, 16384, 5120)]


@pytest.mark.skipif(not os.environ.get("Q27_BENCH"), reason="set Q27_BENCH=1 for the bandwidth report")
def test_bench(capsys):
    lines = []
    for fp4, shapes in ((1, BENCH), (0, BENCH8)):
        for M, N, K in shapes:
            kbytes = K // 2 if fp4 else K
            wbytes = N * kbytes
            sfb = lib().q27t_sf_bytes(N, K // 16) if fp4 else 0
            nbuf = max(2, int(400e6 // (wbytes + sfb)) + 1)  # > L2 (96 MB) per round
            w = torch.randint(0, 256, (nbuf, wbytes), dtype=torch.uint8, device=dev)
            wsf = torch.full((nbuf, max(sfb, 1)), 0x38, dtype=torch.uint8, device=dev)
            a = torch.randint(0, 256, (M, kbytes), dtype=torch.uint8, device=dev)
            asf = torch.full((lib().q27t_sf_bytes(M, K // 16),), 0x38, dtype=torch.uint8, device=dev)
            d = torch.empty(M, N, device=dev)
            ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
            res = []
            for many in (0, 1):
                us = ctypes.c_float()
                call("q27t_bench_gemm", fp4, many, M, N, K, a, asf, w, wsf, wbytes, max(sfb, 1), nbuf, 50, d, ws,
                     WS, ctypes.addressof(us))
                res.append(us.value)
            gbs = (wbytes + sfb) / (res[0] * 1e-6) / 1e9
            lines.append(f"{'fp4' if fp4 else 'fp8'} M={M} N={N} K={K}: skinny {res[0]:.1f} us ({gbs:.0f} GB/s), "
                         f"many-row {res[1]:.1f} us")
    with capsys.disabled():
        print("\n" + "\n".join(lines))


@pytest.mark.parametrize("M", [1, 3, 8, 16, 31, 64])
@pytest.mark.parametrize("N,K", SHAPES_FP8 + [(34816, 5120), (5120, 17408), (248320, 5120)])
def test_skinny_fp8_scaled(M, N, K):
    """Per-row activation and per-channel weight scales with a residual (the target's per-channel FP8 layers), and
    the many-row GEMM's values up to the order of the fp32 sums."""
    a = (torch.randn(M, K) * 30).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn)
    asc, wsc = (torch.rand(M) * 0.02 + 1e-3).to(dev), (torch.rand(N) * 1e-3 + 1e-5).to(dev)
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    prod = (a.double() @ b.double().T) * wsc.cpu().double()[None, :] * asc.cpu().double()[:, None]
    c = torch.randn(M, N, device=dev) * prod.abs().mean().item()
    a8, b8 = a.view(torch.uint8).to(dev), b.view(torch.uint8).to(dev)
    for _ in range(2):  # (the split-K counters are reset)
        d = c.clone()
        call("q27t_skinny_fp8_scaled", M, N, K, a8, asc, b8, wsc, d, 1.0, d, ws, WS)
        assert rel_err(d.cpu().double(), prod + c.cpu().double()) < 1e-5
    call("q27t_skinny_fp8_scaled", M, N, K, a8, asc, b8, wsc, None, 0.0, d, ws, WS)
    assert rel_err(d.cpu().double(), prod) < 1e-5
    many = torch.empty_like(d)
    call("q27t_gemm_fp8_scaled", M, N, K, a8, asc, b8, wsc, None, 0.0, many)
    assert rel_err(d.cpu().double(), many.cpu().double()) < 5e-6  # (fp32 sums in another order)


def x2_operand(M, K):
    """Two random NVFP4 terms per token (row 2 tok + term; scales row-major) and the value they add up to."""
    q, sf, deq = fp4_operand(2 * M, K, 1.0)
    return q.to(dev), sf.view(torch.uint8).to(dev).contiguous(), (deq[0::2].double() + deq[1::2].double()).to(dev)


@pytest.mark.parametrize("M", [1, 3, 8, 13, 16, 29, 32, 33, 57, 64])
@pytest.mark.parametrize("N,K", [(128, 5120), (5120, 5120), (32768, 5120), (1024, 2560)])
def test_skinny_fp4_a4x2(M, N, K):
    """W4A4x2: against the fp64 product of the two terms' sum and the dequantized weights."""
    q, qsf, x = x2_operand(M, K)
    w, wsf, wdq = fp4_operand(N, K, 1.0)
    d = torch.empty(M, N, device=dev)
    call("q27t_skinny_fp4_a4x2", M, N, K, q, qsf, w.to(dev), swizzled(wsf, N, K // 16), 0.37, d)
    assert rel_err(d.double(), 0.37 * (x @ wdq.to(dev).double().T)) < 1e-5


@pytest.mark.parametrize("M", [1, 8, 16, 24, 32, 40, 48, 56, 64])
def test_skinny_fp4_a4x2_first_term_is_w4a4(M):
    """With a zero second term: bit for bit skinny_fp4 on the first (the lm_head's shape: no split-K there)."""
    N, K = 248320, 5120
    q1, sf1, _ = fp4_operand(M, K, 1.0)
    w, wsf, _ = fp4_operand(N, K, 1.0)
    w, wsw = w.to(dev), swizzled(wsf, N, K // 16)
    q = torch.zeros(2 * M, K // 2, dtype=torch.uint8, device=dev)
    qsf = torch.full((2 * M, K // 16), 0x38, dtype=torch.uint8, device=dev)
    q[0::2], qsf[0::2] = q1.to(dev), sf1.view(torch.uint8).to(dev)
    ws = torch.zeros(WS, dtype=torch.uint8, device=dev)
    a, b = torch.empty(M, N, device=dev), torch.empty(M, N, device=dev)
    call("q27t_skinny_fp4_a4x2", M, N, K, q, qsf, w, wsw, 0.37, a)
    call("q27t_skinny_fp4", M, N, K, q1.to(dev), swizzled(sf1, M, K // 16), w, wsw, 0.37, None, 0.0, b, ws, WS)
    assert torch.equal(a, b)


def test_skinny_fp4_a4x2_rows_independent_of_m():
    N, K = 248320, 5120
    q, qsf, _ = x2_operand(64, K)
    w, wsf, _ = fp4_operand(N, K, 1.0)
    w, wsw = w.to(dev), swizzled(wsf, N, K // 16)
    full = torch.empty(64, N, device=dev)
    call("q27t_skinny_fp4_a4x2", 64, N, K, q, qsf, w, wsw, 0.013, full)
    for M in (1, 2, 5, 8, 9, 16, 17, 31, 32, 33, 41, 63):
        d = torch.full((M + 1, N), 7.0, device=dev)
        call("q27t_skinny_fp4_a4x2", M, N, K, q, qsf, w, wsw, 0.013, d)
        assert torch.equal(d[:M], full[:M]), f"M={M}"
        assert bool((d[M] == 7.0).all()), f"M={M}: wrote past its rows"


@pytest.mark.parametrize("normed", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.05, 1.0, 4.0])
def test_quant_fp4x2(normed, scale):
    """The first term is rmsnorm_quant's / quant_fp4's bit for bit, the second the quantization of what it left."""
    T, K, ins = 64, 5120, 0.021
    kb = K // 16
    x = torch.randn(T, K, device=dev) * scale
    w = (torch.randn(K, device=dev) * 0.1).to(torch.bfloat16)
    q = torch.empty(2 * T, K // 2, dtype=torch.uint8, device=dev)
    sf = torch.empty(2 * T, kb, dtype=torch.uint8, device=dev)
    call("q27t_quant_fp4x2", x, w if normed else None, T, K, 1e-6, ins, q, sf)
    q1 = torch.empty(T, K // 2, dtype=torch.uint8, device=dev)
    sf1 = torch.zeros(lib().q27t_sf_bytes(T, kb), dtype=torch.uint8, device=dev)
    if normed:
        v = torch.empty(T, K, device=dev)
        call("q27t_rmsnorm_quant", x, w, T, 1e-6, v, 2, ins, q1, sf1)
    else:
        v = x
        call("q27t_quant_fp4", x, T, K, ins, q1, sf1)
    assert torch.equal(q[0::2], q1)
    assert torch.equal(swizzled(sf[0::2].contiguous().view(torch.float8_e4m3fn), T, kb)[: sf1.numel()], sf1)
    s = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    terms = unpack_e2m1(q) * s * ins
    left = v - terms[0::2]
    want = quant_act_nvfp4(left, torch.tensor(ins, device=dev))
    # (fp32 rounding can move a tie of the second term: a few elements in 100,000, by one step)
    assert float((terms[1::2] - want).abs().gt(1e-5 + 1e-5 * want.abs()).float().mean()) < 1e-3
    if scale:
        assert rel_err(terms[0::2] + terms[1::2], v) < 0.02
