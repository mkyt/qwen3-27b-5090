"""CUTLASS GEMMs against fp32 matmuls of the dequantized operands."""

import pytest
import torch

from q27lib import call, lib
from qwen27b.quant import unpack_e2m1
from test_elementwise import unswizzle

dev = "cuda"
WS = 4 << 20


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def fp4_operand(rows, K, scale):
    """Random NVFP4 operand (codes, row-major e4m3 scales) and its dequantized fp32 value."""
    codes = torch.randint(0, 256, (rows, K // 2), dtype=torch.uint8)
    sf = (torch.rand(rows, K // 16) * 2 + 0.25).to(torch.float8_e4m3fn)
    deq = unpack_e2m1(codes) * sf.float().repeat_interleave(16, -1) * scale
    return codes, sf, deq


@pytest.mark.parametrize("M", [1, 5, 128, 300])
@pytest.mark.parametrize("N,K", [(5120, 17408), (17408, 5120), (384, 5120)])
def test_gemm_fp4(M, N, K):
    a, asf, adq = fp4_operand(M, K, 1.0)
    b, bsf, bdq = fp4_operand(N, K, 1.0)
    swz = []
    for rows, s in ((M, asf), (N, bsf)):
        buf = torch.zeros(lib().q27t_sf_bytes(rows, K // 16), dtype=torch.uint8, device=dev)
        call("q27t_swizzle_sf", s.view(torch.uint8).to(dev), rows, K // 16, buf)
        swz.append(buf)
    c = torch.randn(M, N, device=dev)
    d = c.clone()
    ws = torch.empty(WS, dtype=torch.uint8, device=dev)
    alpha = 0.37
    call("q27t_gemm_fp4", M, N, K, a.to(dev), swz[0], b.to(dev), swz[1], alpha, d, 1.0, d, ws, WS)
    ref = alpha * (adq.double() @ bdq.double().T) + c.cpu().double()
    assert rel_err(d.cpu().double(), ref) < 1e-5
    call("q27t_gemm_fp4", M, N, K, a.to(dev), swz[0], b.to(dev), swz[1], alpha, None, 0.0, d, ws, WS)
    assert rel_err(d.cpu().double(), alpha * (adq.double() @ bdq.double().T)) < 1e-5
    # BF16 output: the fp32 result rounded to BF16.
    db = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    call("q27t_gemm_fp4_bf16", M, N, K, a.to(dev), swz[0], b.to(dev), swz[1], alpha, db, ws, WS)
    assert (db.float() - d.bfloat16().float()).abs().max().item() <= (d.abs().max().item() * 2 ** -8)
    assert rel_err(db.cpu().double(), alpha * (adq.double() @ bdq.double().T)) < 4e-3


@pytest.mark.parametrize("M", [1, 7, 256, 333])
@pytest.mark.parametrize("N,K", [(10240, 5120), (5120, 6144), (1024, 5120)])
def test_gemm_fp8(M, N, K):
    a = (torch.randn(M, K) * 30).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 30).to(torch.float8_e4m3fn)
    c = torch.randn(M, N, device=dev) * 1000
    d = c.clone()
    ws = torch.empty(WS, dtype=torch.uint8, device=dev)
    alpha = 0.011
    call("q27t_gemm_fp8", M, N, K, a.view(torch.uint8).to(dev), b.view(torch.uint8).to(dev), alpha, d, 1.0, d, ws, WS)
    ref = alpha * (a.double() @ b.double().T) + c.cpu().double()
    assert rel_err(d.cpu().double(), ref) < 1e-5
    # A workspace with the unit scales runs the block-scaled MXFP8 kernel: bit-identical to the plain one (P5-11).
    wsb = lib().q27t_gemm_workspace_bytes()
    ws2 = torch.empty(wsb, dtype=torch.uint8, device=dev)
    call("q27t_gemm_workspace_init", ws2, wsb)
    d2 = c.clone()
    call("q27t_gemm_fp8", M, N, K, a.view(torch.uint8).to(dev), b.view(torch.uint8).to(dev), alpha, d2, 1.0, d2, ws2, wsb)
    assert torch.equal(d2, d)
