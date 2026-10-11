"""Attention kernels (q/k norm + RoPE + paged KV write, causal attention with the output gate) against
the reference model's attention, over several calls that cross page boundaries."""

import os

import pytest
import torch

from q27lib import call
from qwen27b.model import TextConfig, apply_rope, rms_norm, rope_cos_sin
from qwen27b.quant import quant_act_nvfp4, to_e4m3, unpack_e2m1

dev = "cuda"
H, HKV, D, PAGE = 24, 4, 256, 64
CFG = TextConfig(hidden=5120, vocab=248320, layer_types=[], n_heads=H, n_kv_heads=HKV, head_dim=D, rotary_dim=64,
                 rope_theta=1e7, eps=1e-6, k_heads=16, v_heads=48, k_dim=128, v_dim=128, conv_k=4)


# E4M3 P (and Q) on these random inputs: about 2% (3.5% with Q), as a torch emulation of the same
# quantization.
F8_TOL = {1: 0.03, 2: 0.05}
F8_TOL_FP4 = {2: 0.07, 3: 0.13}  # NVFP4 KV rounded to E4M3 for mode 2: about 4.1% (emulation 4.2%); mode 3 (Q in
# NVFP4 for the FP4 Q K^T, P7-8): about 8.8% (emulation 8.9%)
# Decode / verify with FP8 MMAs (Q27_ATTN_TC_F8, P7-7): the prefill's mode-2 numerics.
TC8_TOL = {"fp8": 0.05, "nvfp4": 0.07}


KV_CODE = {"bf16": 0, "fp8": 1, "nvfp4": 2}
FP4_ROW = D // 2 + D // 16  # NVFP4 cache row per (token, KV head): 128 bytes of e2m1 codes + 16 e4m3 scales


def new_cache(kv, pages):
    if kv == "nvfp4":
        return torch.zeros(pages, PAGE, HKV, FP4_ROW, dtype=torch.uint8, device=dev)
    return torch.zeros(pages, PAGE, HKV, D, dtype=torch.bfloat16 if kv == "bf16" else torch.uint8, device=dev)


def cache_values(c, kv):
    """Stored values [..., D] of a cache tensor (or rows of it)."""
    if kv == "bf16":
        return c.float()
    if kv == "fp8":
        return c.view(torch.float8_e4m3fn).float()
    sf = c[..., D // 2:].contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return unpack_e2m1(c[..., :D // 2].contiguous()) * sf


def store_ref(kv):
    if kv == "bf16":
        return lambda t: t.bfloat16().float()
    if kv == "fp8":
        return to_e4m3
    return lambda t: quant_act_nvfp4(t, torch.ones((), device=t.device))


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("decode,tc8", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("kv,f8", [("bf16", 0), ("fp8", 0), ("fp8", 1), ("fp8", 2), ("nvfp4", 0), ("nvfp4", 2),
                                  ("nvfp4", 3)])
def test_attention_calls(kv, f8, decode, tc8, monkeypatch):
    if f8 and decode:
        pytest.skip("Q27_ATTN_FP8 is the prefill kernel only")
    if tc8 and kv == "bf16":
        pytest.skip("Q27_ATTN_TC_F8 needs FP8 or NVFP4 KV")
    monkeypatch.setenv("Q27_ATTN_FP8", str(f8))
    monkeypatch.setenv("Q27_ATTN_TC_F8", str(tc8))
    pages = 48
    page_table = torch.randperm(pages, dtype=torch.int32).to(dev)  # scrambled physical pages
    kc = new_cache(kv, pages)
    vc = torch.zeros_like(kc)
    q_norm, k_norm = (torch.randn(D, device=dev) * 0.1).bfloat16(), (torch.randn(D, device=dev) * 0.1).bfloat16()
    store = store_ref(kv)
    K_ref = torch.empty(0, HKV, D, device=dev)
    V_ref = torch.empty(0, HKV, D, device=dev)
    pos = 0
    ws = torch.zeros(100 << 20, dtype=torch.uint8, device=dev)
    # (1100 .. 1288 crosses the key split of the prefill kernel: blocks ending past 1024.)
    calls = (100, 1, 37, 1, 150, 1, 1100) if not decode else (2000, 1, 37, 1, 8, 3, 150, 1, 2)
    for T in calls:
        qg = torch.randn(T, H, 2 * D, device=dev) * 2
        k, v = torch.randn(T, HKV, D, device=dev) * 2, torch.randn(T, HKV, D, device=dev)
        q_out, out = torch.empty(T, H, D, device=dev), torch.empty(T, H * D, device=dev)
        call("q27t_attn_prep", qg, k, v, q_norm, k_norm, T, pos, kc, vc, KV_CODE[kv], page_table, q_out)
        if decode and T <= 8:
            call("q27t_attention_decode", q_out, qg, T, pos, kc, vc, KV_CODE[kv], page_table, out, ws, 26 << 20)
        else:
            call("q27t_attention", q_out, qg, T, pos, kc, vc, KV_CODE[kv], page_table, out, ws, ws.numel())

        cos, sin = rope_cos_sin(torch.arange(pos, pos + T), CFG, dev)
        q = apply_rope(rms_norm(qg[..., :D], q_norm, 1e-6), cos, sin)
        kk = apply_rope(rms_norm(k, k_norm, 1e-6), cos, sin)
        torch.testing.assert_close(q_out, q, rtol=1e-4, atol=1e-4)
        K_ref, V_ref = torch.cat((K_ref, store(kk))), torch.cat((V_ref, store(v)))
        L = K_ref.shape[0]
        Kf, Vf = K_ref.repeat_interleave(H // HKV, 1), V_ref.repeat_interleave(H // HKV, 1)
        scores = torch.einsum("thd,lhd->htl", q, Kf) * D ** -0.5
        mask = torch.arange(L, device=dev)[None, :] > torch.arange(pos, pos + T, device=dev)[:, None]
        o = torch.einsum("htl,lhd->thd", scores.masked_fill(mask, float("-inf")).softmax(-1), Vf)
        ref = (o.reshape(T, H * D) * torch.sigmoid(qg[..., D:].reshape(T, H * D)))
        # Both paths run on tensor cores with Q and P rounded to BF16 (Q27_ATTN_PREFILL_TC=0: the
        # CUDA-core prefill kernel in fp32).
        # FP8 MMAs (Q27_ATTN_FP8): P (and Q) in E4M3.
        tol = 1e-4 if os.environ.get("Q27_ATTN_PREFILL_TC") == "0" and not (decode and T <= 8) else 1e-2
        if f8 and not (decode and T <= 8):
            tol = F8_TOL_FP4[f8] if kv == "nvfp4" else F8_TOL[f8]
        if tc8 and decode and T <= 8:
            tol = TC8_TOL[kv]
        assert rel_err(out, ref) < tol, (kv, pos, T, rel_err(out, ref))

        # The cache holds exactly the stored K/V at their logical positions.
        flat = lambda c: c[page_table.long()].reshape(-1, HKV, c.shape[-1])[:L]
        kq, vq = cache_values(flat(kc), kv), cache_values(flat(vc), kv)
        if kv == "nvfp4":  # the reference numerics: rounding flips only where q/k norm + RoPE differ by an ulp
            for got, want in ((kq, K_ref), (vq, V_ref)):
                assert (got != want).float().mean().item() < 2e-3, (got != want).float().mean().item()
        assert rel_err(kq, K_ref) < 1e-3
        pos += T


@pytest.mark.parametrize("kv,tc8", [("bf16", ""), ("fp8", ""), ("fp8", "1"), ("nvfp4", "0"), ("nvfp4", "")])
@pytest.mark.parametrize("pos", [5, 63, 1000, 4097, 30000])
def test_decode_rows_independent_of_t(kv, tc8, pos, monkeypatch):
    """A chain of T new tokens (the speculative verify) gives each token exactly the decode result
    (the decode runs a token on 4 warps, the verify a token per warp; 30000: many chunks per split).
    tc8: Q27_ATTN_TC_F8 ("": the default, FP8 MMAs with NVFP4 KV only)."""
    monkeypatch.setenv("Q27_ATTN_TC_F8", tc8)
    pages = max(72, (pos + 8) // PAGE + 8)
    page_table = torch.randperm(pages, dtype=torch.int32).to(dev)
    kc = (torch.randn(pages, PAGE, HKV, D, device=dev)).bfloat16() if kv == "bf16" else \
        torch.randint(0, 120, (pages, PAGE, HKV, FP4_ROW if kv == "nvfp4" else D), dtype=torch.uint8, device=dev)
    vc = kc.clone()
    q_norm, k_norm = (torch.randn(D, device=dev) * 0.1).bfloat16(), (torch.randn(D, device=dev) * 0.1).bfloat16()
    ws = torch.zeros(26 << 20, dtype=torch.uint8, device=dev)
    T = 8
    qg = torch.randn(T, H, 2 * D, device=dev) * 2
    k, v = torch.randn(T, HKV, D, device=dev) * 2, torch.randn(T, HKV, D, device=dev)
    q_out, out = torch.empty(T, H, D, device=dev), torch.empty(T, H * D, device=dev)
    call("q27t_attn_prep", qg, k, v, q_norm, k_norm, T, pos, kc, vc, KV_CODE[kv], page_table, q_out)
    call("q27t_attention_decode", q_out, qg, T, pos, kc, vc, KV_CODE[kv], page_table, out, ws, 26 << 20)
    for t in range(T):
        one = torch.empty(1, H * D, device=dev)
        call("q27t_attention_decode", q_out[t:t + 1].contiguous(), qg[t:t + 1].contiguous(), 1, pos + t, kc, vc,
             KV_CODE[kv], page_table, one, ws, 26 << 20)
        assert torch.equal(one[0], out[t]), (t, (one[0] - out[t]).abs().max().item())


@pytest.mark.parametrize("kv,f8", [("bf16", 0), ("fp8", 0), ("fp8", 1), ("fp8", 2), ("nvfp4", 0), ("nvfp4", 2),
                                  ("nvfp4", 3)])
@pytest.mark.parametrize("T,cut", [(300, 1), (300, 70), (300, 128), (300, 200), (2200, 1000), (2200, 1029),
                                   (2200, 1985)])
def test_prefill_rows_independent_of_tiling(kv, f8, T, cut, monkeypatch):
    """A prefill cut anywhere (the prefix cache resumes there) gives the same rows as an uncut one, also
    across the key split (blocks ending past 1024 keys)."""
    monkeypatch.setenv("Q27_ATTN_FP8", str(f8))
    pages = (T + PAGE - 1) // PAGE
    page_table = torch.randperm(pages, dtype=torch.int32).to(dev)
    kc = new_cache(kv, pages)
    vc = torch.zeros_like(kc)
    q_norm, k_norm = (torch.randn(D, device=dev) * 0.1).bfloat16(), (torch.randn(D, device=dev) * 0.1).bfloat16()
    qg = torch.randn(T, H, 2 * D, device=dev) * 2
    k, v = torch.randn(T, HKV, D, device=dev) * 2, torch.randn(T, HKV, D, device=dev)
    q_out, full = torch.empty(T, H, D, device=dev), torch.empty(T, H * D, device=dev)
    call("q27t_attn_prep", qg, k, v, q_norm, k_norm, T, 0, kc, vc, KV_CODE[kv], page_table, q_out)
    ws = torch.empty(T * H * 2 * (D + 2) * 4, dtype=torch.uint8, device=dev)
    call("q27t_attention", q_out, qg, T, 0, kc, vc, KV_CODE[kv], page_table, full, ws, ws.numel())
    part = torch.empty(T - cut, H * D, device=dev)
    call("q27t_attention", q_out[cut:].contiguous(), qg[cut:].contiguous(), T - cut, cut, kc, vc, KV_CODE[kv],
         page_table, part, ws, ws.numel())
    assert torch.equal(part, full[cut:]), (part - full[cut:]).abs().max().item()


@pytest.mark.parametrize("T,pos", [(300, 0), (2048, 0), (2048, 2048), (700, 3333), (3, 37)])
def test_prefill_nvfp4_stage_same_rows(T, pos, monkeypatch):
    """P7-8: NVFP4 KV, mode 2: the keys rounded to E4M3 once into the stage give the rows of the per-CTA
    conversion bit for bit."""
    monkeypatch.setenv("Q27_ATTN_FP8", "2")
    n = pos + T
    pages = (n + PAGE - 1) // PAGE
    page_table = torch.randperm(pages + 3, dtype=torch.int32)[:pages].to(dev)
    kc = new_cache("nvfp4", pages + 3)
    vc = torch.zeros_like(kc)
    q_norm, k_norm = (torch.randn(D, device=dev) * 0.1).bfloat16(), (torch.randn(D, device=dev) * 0.1).bfloat16()
    k, v = torch.randn(n, HKV, D, device=dev) * 2, torch.randn(n, HKV, D, device=dev)
    qg = torch.randn(n, H, 2 * D, device=dev) * 2
    q_out = torch.empty(n, H, D, device=dev)
    call("q27t_attn_prep", qg, k, v, q_norm, k_norm, n, 0, kc, vc, KV_CODE["nvfp4"], page_table, q_out)
    ws = torch.empty(T * H * 2 * (D + 2) * 4, dtype=torch.uint8, device=dev)
    outs = []
    for stage in ("0", "1"):
        monkeypatch.setenv("Q27_ATTN_FP4_STAGE", stage)
        out = torch.full((T, H * D), float("nan"), device=dev)
        call("q27t_attention", q_out[pos:].contiguous(), qg[pos:].contiguous(), T, pos, kc, vc, KV_CODE["nvfp4"],
             page_table, out, ws, ws.numel())
        outs.append(out)
    assert torch.equal(outs[0], outs[1]), (outs[0] - outs[1]).abs().max().item()
