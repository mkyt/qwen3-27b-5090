"""Skinny decode GEMMs (mma.sync) against fp64 matmuls of the dequantized operands, and their speed."""

import ctypes
import os

import pytest
import torch

from q27lib import call, lib
from test_gemm import fp4_operand, rel_err

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
