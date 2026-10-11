"""GDN kernels against the reference model's definitions (conv, l2norm, gates, delta rule, gated norm)."""

import torch
import torch.nn.functional as F

from q27lib import call, lib
from qwen27b.model import gated_delta_recurrent, gated_rms_norm, l2norm

dev = "cuda"
CONV, KH, VH, DK, DV = 10240, 16, 48, 128, 128


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def reference_prep(qkv, state, conv_w, a, b, A_log, dt_bias):
    T = qkv.shape[0]
    seq = torch.cat((state, qkv.T), dim=-1)
    conv = F.silu(F.conv1d(seq[None], conv_w.float().view(CONV, 1, 4), groups=CONV)[0]).T
    new_state = seq[:, -3:].clone()
    q, k, v = conv.split([KH * DK, KH * DK, VH * DV], dim=-1)
    q = l2norm(q.reshape(T, KH, DK)) * DK ** -0.5
    k = l2norm(k.reshape(T, KH, DK))
    g = -A_log.float().exp() * F.softplus(a + dt_bias.float())
    return q, k, v.reshape(T, VH, DV), g, torch.sigmoid(b), new_state


def run_prep(T, state):
    qkv = torch.randn(T, CONV, device=dev) * 2
    conv_w = (torch.randn(CONV, 4, device=dev) * 0.5).bfloat16()
    a, b = torch.randn(T, VH, device=dev) * 3, torch.randn(T, VH, device=dev) * 3
    A_log, dt_bias = torch.randn(VH, device=dev).bfloat16(), torch.randn(VH, device=dev).bfloat16()
    ref = reference_prep(qkv, state.clone(), conv_w, a, b, A_log, dt_bias)
    q, k = torch.empty(T, KH, DK, device=dev), torch.empty(T, KH, DK, device=dev)
    v, g, beta = torch.empty(T, VH, DV, device=dev), torch.empty(T, VH, device=dev), torch.empty(T, VH, device=dev)
    call("q27t_gdn_prep", qkv, state, conv_w, a, b, A_log, dt_bias, T, q, k, v, g, beta)
    return (q, k, v, g, beta, state), ref


def test_gdn_prep_and_conv_state():
    state = torch.randn(CONV, 3, device=dev)
    for T in (1, 2, 5, 300):  # T < 3 shifts part of the old state
        ours, ref = run_prep(T, state)
        for name, x, y in zip(("q", "k", "v", "g", "beta", "conv_state"), ours, ref):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5, msg=name)


def test_gdn_recurrent_chained_calls():
    S = torch.randn(VH, DK, DV, device=dev) * 0.1
    S_ref = S.clone()
    for T in (5, 1, 70):
        q = l2norm(torch.randn(T, KH, DK, device=dev)) * DK ** -0.5
        k = l2norm(torch.randn(T, KH, DK, device=dev))
        v = torch.randn(T, VH, DV, device=dev)
        g = -torch.rand(T, VH, device=dev) * 2
        beta = torch.rand(T, VH, device=dev)
        o = torch.empty(T, VH, DV, device=dev)
        call("q27t_gdn_recurrent", q, k, v, g, beta, T, S, o)
        o_ref, S_ref = gated_delta_recurrent(q.repeat_interleave(3, 1), k.repeat_interleave(3, 1), v, g, beta, S_ref)
        assert rel_err(o, o_ref) < 1e-5, T
        assert rel_err(S, S_ref) < 1e-5, T


def chunk_inputs(T):
    q = l2norm(torch.randn(T, KH, DK, device=dev)) * DK ** -0.5
    k = l2norm(torch.randn(T, KH, DK, device=dev))
    v = torch.randn(T, VH, DV, device=dev)
    g = -torch.rand(T, VH, device=dev) * torch.rand(VH, device=dev) * 0.5  # per head, some decay fast
    beta = torch.rand(T, VH, device=dev)
    return q, k, v, g, beta


def run_chunked(q, k, v, g, beta, pos0, S):
    T = q.shape[0]
    ws = torch.empty(lib().q27t_gdn_chunk_ws_bytes(T), dtype=torch.uint8, device=dev)
    o = torch.empty(T, VH, DV, device=dev)
    call("q27t_gdn_chunked", q, k, v.clone(), g, beta, T, pos0, S, o, ws)
    return o


def test_gdn_chunked_matches_recurrent():
    """bf16 tensor-core chunks against the fp64 definition, chained calls starting inside chunks."""
    S = torch.randn(VH, DK, DV, device=dev) * 0.1
    S_ref = S.double()
    pos = 37
    for T in (5, 1, 70, 300, 64, 27):
        q, k, v, g, beta = chunk_inputs(T)
        o = run_chunked(q, k, v, g, beta, pos, S)
        o_ref, S_ref = gated_delta_recurrent(*(x.double() for x in (q.repeat_interleave(3, 1), k.repeat_interleave(3, 1),
                                                                    v, g, beta)), S_ref)
        assert rel_err(o.double(), o_ref) < 5e-3, (T, rel_err(o.double(), o_ref))
        assert rel_err(S.double(), S_ref) < 5e-3, (T, rel_err(S.double(), S_ref))
        S_ref = S.double()  # compare each call from the same start
        pos += T


def test_gdn_chunked_cut_at_multiples_of_64_is_bit_identical():
    q, k, v, g, beta = chunk_inputs(300)
    S0 = torch.randn(VH, DK, DV, device=dev) * 0.1
    S = S0.clone()
    o = run_chunked(q, k, v, g, beta, 10, S)
    S_cut, parts = S0.clone(), []
    for a, b in ((0, 54), (54, 182), (182, 300)):  # positions 10..64, 64..192, 192..310
        parts.append(run_chunked(q[a:b], k[a:b], v[a:b], g[a:b], beta[a:b], 10 + a, S_cut))
    assert torch.equal(torch.cat(parts), o)
    assert torch.equal(S_cut, S)


def test_gdn_gated_norm():
    T = 9
    o, z = torch.randn(T, VH, DV, device=dev), torch.randn(T, VH, DV, device=dev)
    w = torch.randn(DV, device=dev).bfloat16()
    out = torch.empty(T, VH * DV, device=dev)
    q8 = torch.empty(T, VH * DV, dtype=torch.uint8, device=dev)
    call("q27t_gdn_gated_norm", o, z, w, T, 1e-6, out, q8, 0.05)
    torch.testing.assert_close(out, gated_rms_norm(o, z, w, 1e-6).reshape(T, -1), rtol=1e-5, atol=1e-5)
    q8_ref = torch.empty_like(q8)
    call("q27t_quant_fp8", out, out.numel(), 0.05, q8_ref)
    assert torch.equal(q8, q8_ref)


def test_gdn_decode_matches_the_general_kernels():
    """The fused one-token kernels against linear_bf16_pair + gdn_prep + gdn_recurrent + gated norm, 3 steps."""
    conv_w = (torch.randn(CONV, 4, device=dev) * 0.5).bfloat16()
    wa, wb = (torch.randn(VH, 5120, device=dev) * 0.02).bfloat16(), (torch.randn(VH, 5120, device=dev) * 0.02).bfloat16()
    A_log, dt_bias = (torch.randn(VH, device=dev) * 0.5).bfloat16(), (torch.randn(VH, device=dev) * 0.5).bfloat16()
    norm_w = torch.randn(DV, device=dev).bfloat16()
    cs1 = torch.randn(CONV, 3, device=dev)
    cs2 = cs1.clone()
    S1 = torch.randn(VH, DK, DV, device=dev) * 0.1
    S2 = S1.clone()
    f = lambda *shape: torch.empty(*shape, device=dev)
    for _ in range(3):
        qkv, h, z = torch.randn(1, CONV, device=dev) * 2, torch.randn(1, 5120, device=dev), torch.randn(1, VH * DV, device=dev)
        a1, b1, q1, k1, v1, g1, be1, o1, out1 = f(1, VH), f(1, VH), f(1, KH * DK), f(1, KH * DK), f(1, VH * DV), \
            f(1, VH), f(1, VH), f(1, VH * DV), f(1, VH * DV)
        q81 = torch.empty(1, VH * DV, dtype=torch.uint8, device=dev)
        call("q27t_linear_bf16_pair", h, wa, wb, 1, VH, 5120, a1, b1)
        call("q27t_gdn_prep", qkv, cs1, conv_w, a1, b1, A_log, dt_bias, 1, q1, k1, v1, g1, be1)
        call("q27t_gdn_recurrent", q1, k1, v1, g1, be1, 1, S1, o1)
        call("q27t_gdn_gated_norm", o1, z, norm_w, 1, 1e-6, out1, q81, 0.05)

        a2, b2, q2, k2, v2, out2 = f(VH), f(VH), f(KH * DK), f(KH * DK), f(VH * DV), f(VH * DV)
        q82 = torch.empty(VH * DV, dtype=torch.uint8, device=dev)
        call("q27t_gdn_decode_prep", qkv, cs2, conv_w, h, wa, wb, q2, k2, v2, a2, b2)
        call("q27t_gdn_decode", q2, k2, v2, a2, b2, A_log, dt_bias, S2, z, norm_w, 1e-6, out2, q82, 0.05)

        assert torch.equal(cs1, cs2)
        assert rel_err(q2, q1[0]) < 1e-6 and rel_err(k2, k1[0]) < 1e-6 and torch.equal(v2, v1[0])
        assert rel_err(S2, S1) < 1e-5
        assert rel_err(out2, out1[0]) < 1e-5
        assert (q82 != q81[0]).float().mean() < 1e-3  # FP8 copies agree up to rare rounding-boundary flips
