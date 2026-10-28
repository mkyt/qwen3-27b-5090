"""The prefill GEMMs (gemm_mma.cu) against fp32 matmuls of the dequantized operands."""

import pytest
import torch

from q27lib import call, lib
from qwen27b.quant import unpack_e2m1
from test_elementwise import unswizzle

dev = "cuda"


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def fp4_operand(rows, K, scale):
    """Random NVFP4 operand (codes, row-major e4m3 scales) and its dequantized fp32 value."""
    codes = torch.randint(0, 256, (rows, K // 2), dtype=torch.uint8)
    sf = (torch.rand(rows, K // 16) * 2 + 0.25).to(torch.float8_e4m3fn)
    deq = unpack_e2m1(codes) * sf.float().repeat_interleave(16, -1) * scale
    return codes, sf, deq


@pytest.mark.parametrize("M", [1, 5, 128, 300, 2048])
@pytest.mark.parametrize("N,K", [(5120, 17408), (17408, 5120), (384, 5120)])
def test_gemm_fp4(M, N, K):
    """NVFP4 GEMM: residual in place, plain, C apart from D, and the BF16 output (the fp32 result rounded)."""
    a, asf, adq = fp4_operand(M, K, 1.0)
    b, bsf, bdq = fp4_operand(N, K, 1.0)
    swz = []
    for rows, s in ((M, asf), (N, bsf)):
        buf = torch.zeros(lib().q27t_sf_bytes(rows, K // 16), dtype=torch.uint8, device=dev)
        call("q27t_swizzle_sf", s.view(torch.uint8).to(dev), rows, K // 16, buf)
        swz.append(buf)
    a, b = a.to(dev), b.to(dev)
    alpha = 0.37
    prod = alpha * (adq.double() @ bdq.double().T)
    c = torch.randn(M, N, device=dev)
    d = c.clone()
    call("q27t_gemm_fp4", M, N, K, a, swz[0], b, swz[1], alpha, d, 1.0, d)  # residual in place
    assert rel_err(d.cpu().double(), prod + c.cpu().double()) < 1e-5
    call("q27t_gemm_fp4", M, N, K, a, swz[0], b, swz[1], alpha, None, 0.0, d)
    assert rel_err(d.cpu().double(), prod) < 1e-5
    d2 = torch.empty_like(d)
    call("q27t_gemm_fp4", M, N, K, a, swz[0], b, swz[1], alpha, c, 0.5, d2)  # C apart from D
    assert rel_err(d2.cpu().double(), prod + 0.5 * c.cpu().double()) < 1e-5
    db = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    call("q27t_gemm_fp4_bf16", M, N, K, a, swz[0], b, swz[1], alpha, db)
    assert torch.equal(db, d.bfloat16())


def glu_rows(I):
    """Rows of the interleaved gate | up (kernels.h glu_row) that hold gate rows 0..I-1, then up rows 0..I-1."""
    r = torch.arange(I)
    return torch.cat([(r // 64) * 128 + r % 64, (r // 64) * 128 + 64 + r % 64])


@pytest.mark.parametrize("M", [1, 77, 300, 2048])
@pytest.mark.parametrize("I,K", [(384, 5120), (17408, 5120)])
def test_gemm_fp4_glu(M, I, K):
    """gate | up GEMM (interleaved weights) with silu(gate) * up NVFP4-quantized in the epilogue, against the
    separate fp32 GEMMs and silu_mul_quant_fp4 (gate / up in fp32 there too)."""
    a, asf, _ = fp4_operand(M, K, 1.0)
    w, wsf, _ = fp4_operand(2 * I, K, 1.0)  # gate rows, then up rows
    rows = glu_rows(I)
    wi, wsfi = torch.empty_like(w), torch.empty_like(wsf)
    wi[rows], wsfi[rows] = w, wsf
    swz = []
    for n, sf in ((M, asf), (2 * I, wsf), (2 * I, wsfi)):
        buf = torch.zeros(lib().q27t_sf_bytes(n, K // 16), dtype=torch.uint8, device=dev)
        call("q27t_swizzle_sf", sf.view(torch.uint8).to(dev), n, K // 16, buf)
        swz.append(buf)
    # The loader's path: gate and up scales swizzled into their interleaved places.
    sfl = torch.zeros_like(swz[2])
    for h in range(2):
        call("q27t_swizzle_sf_glu", wsf[h * I:(h + 1) * I].contiguous().view(torch.uint8).to(dev), I, K // 16, sfl, h)
    assert torch.equal(sfl, swz[2])
    a, w, wi = a.to(dev), w.to(dev), wi.to(dev)
    ag, au, out_scale = 0.013, 0.021, 0.05
    gu = torch.empty(M, 2 * I, device=dev)
    call("q27t_gemm_fp4", M, 2 * I, K, a, swz[0], w, swz[1], 1.0, None, 0.0, gu)
    gate, up = (gu[:, :I] * ag).contiguous(), (gu[:, I:] * au).contiguous()
    sfb = lib().q27t_sf_bytes(M, I // 16)
    q_ref = torch.zeros(M, I // 2, dtype=torch.uint8, device=dev)
    sf_ref = torch.zeros(sfb, dtype=torch.uint8, device=dev)
    call("q27t_silu_mul_quant_fp4", gate, up, I, M, I, out_scale, q_ref, sf_ref)
    # silu_mul_quant_fp4_glu on the interleaved GEMM's output (alpha 1): the same values, the same codes.
    gui = torch.empty(M, 2 * I, device=dev)
    call("q27t_gemm_fp4", M, 2 * I, K, a, swz[0], wi, swz[2], 1.0, None, 0.0, gui)
    assert torch.equal(gui[:, rows.to(dev)], gu)
    q = torch.zeros_like(q_ref)
    sf = torch.zeros_like(sf_ref)
    call("q27t_silu_mul_quant_fp4_glu", gui, M, I, ag, au, out_scale, q, sf)
    assert torch.equal(q, q_ref) and torch.equal(sf, sf_ref)
    q.zero_(), sf.zero_()
    call("q27t_gemm_fp4_glu", M, I, K, a, swz[0], wi, swz[2], ag, au, out_scale, q, sf)
    # Accumulation order and __expf: a few values on a rounding boundary may land in the next code.
    assert (sf != sf_ref).float().mean().item() < 1e-3
    lo, hi = q & 15, q >> 4
    lo_r, hi_r = q_ref & 15, q_ref >> 4
    assert ((lo != lo_r).float().mean().item() + (hi != hi_r).float().mean().item()) / 2 < 2e-3
    # The BF16 path (Q27_MLP_FUSED=0): a BF16 gate | up, then the silu step with the alphas.
    gub = gui.to(torch.bfloat16)
    q.zero_(), sf.zero_()
    call("q27t_silu_mul_quant_fp4_glu_bf16", gub, M, I, ag, au, out_scale, q, sf)
    gb, ub = (gub.float()[:, rows[:I].to(dev)] * ag), (gub.float()[:, rows[I:].to(dev)] * au)
    q_ref.zero_(), sf_ref.zero_()
    call("q27t_silu_mul_quant_fp4", gb.contiguous(), ub.contiguous(), I, M, I, out_scale, q_ref, sf_ref)
    assert torch.equal(q, q_ref) and torch.equal(sf, sf_ref)


@pytest.mark.parametrize("M", [1, 7, 256, 333, 2048])
@pytest.mark.parametrize("N,K", [(10240, 5120), (5120, 6144), (1024, 5120)])
def test_gemm_fp8(M, N, K):
    """E4M3 GEMM: residual in place, plain, C apart from D."""
    a = (torch.randn(M, K) * 30).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn)
    a8, b8 = a.view(torch.uint8).to(dev), b.view(torch.uint8).to(dev)
    alpha = 0.011
    prod = alpha * (a.double() @ b.double().T)
    c = torch.randn(M, N, device=dev) * 1000
    d = c.clone()
    call("q27t_gemm_fp8", M, N, K, a8, b8, alpha, d, 1.0, d)  # residual in place
    assert rel_err(d.cpu().double(), prod + c.cpu().double()) < 1e-5
    call("q27t_gemm_fp8", M, N, K, a8, b8, alpha, None, 0.0, d)
    assert rel_err(d.cpu().double(), prod) < 1e-5
    d2 = torch.empty_like(d)
    call("q27t_gemm_fp8", M, N, K, a8, b8, alpha, c, 0.5, d2)  # C apart from D
    assert rel_err(d2.cpu().double(), prod + 0.5 * c.cpu().double()) < 1e-5


@pytest.mark.parametrize("M", [1, 70, 1000, 2048])
@pytest.mark.parametrize("N,K", [(4304, 1152), (1152, 4304), (3456, 1152), (1152, 1536), (1024, 5120)])
def test_gemm_bf16(M, N, K):
    """The BF16 GEMM (draft context, vision): any N, K multiple of 8 (TMA edges), the residual in place."""
    a = torch.randn(M, K, device=dev).bfloat16()
    w = torch.randn(N, K, device=dev).bfloat16()
    prod = 0.5 * (a.double() @ w.double().T)
    c = torch.randn(M, N, device=dev)
    d = c.clone()
    call("q27t_gemm_bf16_ex", M, N, K, a, w, 0.5, d, 1.0, d)
    assert rel_err(d.double(), prod + c.double()) < 1e-5
    d0 = torch.full((M, N), float("nan"), device=dev)
    call("q27t_gemm_bf16_ex", M, N, K, a, w, 0.5, None, 0.0, d0)
    assert rel_err(d0.double(), prod) < 1e-5
