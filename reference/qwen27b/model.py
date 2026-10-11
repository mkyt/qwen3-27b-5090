"""Reference forward pass of Qwen3.8-27B (text) for correctness tests and golden data.

Plain PyTorch, fp32 by default, written for clarity rather than speed. Layers are streamed from the
checkpoint on each forward (the dequantized model does not fit in RAM or VRAM), so it runs on the CPU
for short prompts and on the GPU for long ones.

Numerics follow the checkpoint: NVFP4 W4A4 (MLP, lm_head) and FP8 W8A8 (GDN / attention projections)
with activation quantization emulated (`act_quant=True`), BF16 for the rest. Turning `act_quant` off
gives the weight-only-dequantized model, which is what transformers computes.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .quant import quant_act_nvfp4, to_e4m3
from .weights import Checkpoint

LM = "model.language_model"


@dataclass
class TextConfig:
    hidden: int
    vocab: int
    layer_types: list
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_k: int

    @classmethod
    def from_hf(cls, cfg: dict) -> "TextConfig":
        t = cfg.get("text_config", cfg)
        rope = t["rope_parameters"]
        return cls(
            hidden=t["hidden_size"], vocab=t["vocab_size"], layer_types=t["layer_types"],
            n_heads=t["num_attention_heads"], n_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
            rotary_dim=int(t["head_dim"] * rope.get("partial_rotary_factor", 1.0)), rope_theta=rope["rope_theta"],
            eps=t["rms_norm_eps"], k_heads=t["linear_num_key_heads"], v_heads=t["linear_num_value_heads"],
            k_dim=t["linear_key_head_dim"], v_dim=t["linear_value_head_dim"], conv_k=t["linear_conv_kernel_dim"],
        )

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim


# ---------------------------------------------------------------- building blocks


def rms_norm(x, weight, eps):
    """Zero-centred RMSNorm: x / rms(x) * (1 + w), computed in fp32."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())


def gated_rms_norm(x, z, weight, eps):
    """GDN output norm: x / rms(x) * w * silu(z) (the weight is not zero-centred here)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float() * F.silu(z.float())


def rope_cos_sin(positions, cfg: TextConfig, device):
    """cos/sin [T, rotary_dim] for text positions (M-RoPE reduces to plain RoPE when t = h = w)."""
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2, dtype=torch.float32, device=device) / cfg.rotary_dim))
    freqs = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x [T, H, D]: rotate the first rotary_dim dims (half-split layout), pass the rest through."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    out = xr * cos[:, None, :] + rotated * sin[:, None, :]
    return torch.cat((out, xp), dim=-1)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_recurrent(q, k, v, g, beta, state):
    """Token-by-token gated delta rule (the definition).

    q, k: [T, H, dk] (already l2-normalized, q scaled); v: [T, H, dv]; g, beta: [T, H]; state: [H, dk, dv] fp32.
    Returns (o [T, H, dv], new state).
    """
    S = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv_mem = torch.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + torch.einsum("hk,hv->hkv", k[t], delta)
        out[t] = torch.einsum("hkv,hk->hv", S, q[t])
    return out, S


def gated_delta_chunked(q, k, v, g, beta, state, chunk=64):
    """Chunk-parallel gated delta rule (WY / UT form), identical in exact arithmetic to the recurrent one.

    Within a chunk of length C, with cumulative log-decay c_i and D_ij = exp(c_i - c_j) for j <= i:
      A = I + tril(diag(beta) K K^T * D, -1)          (unit lower triangular)
      U = A^-1 diag(beta) V,   W = A^-1 diag(beta) (K * exp(c))
      V' = U - W S0
      O  = (Q * exp(c)) S0 + tril(Q K^T * D) V'
      S1 = exp(c_C) S0 + (K * exp(c_C - c))^T V'
    """
    T, H, dk = k.shape
    out = torch.empty_like(v)
    S = state.clone()
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qc, kc, vc = q[s:e].transpose(0, 1), k[s:e].transpose(0, 1), v[s:e].transpose(0, 1)  # [H, C, d]
        gc, bc = g[s:e].transpose(0, 1), beta[s:e].transpose(0, 1)  # [H, C]
        C = e - s
        c = gc.cumsum(-1)
        D = (c[:, :, None] - c[:, None, :])
        causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
        D = torch.where(causal, D, torch.full_like(D, float("-inf"))).exp()  # [H, C, C]
        kb = kc * bc[..., None]
        A = (kb @ kc.transpose(-1, -2)) * D
        A = torch.eye(C, device=q.device) + A.tril(-1)
        U = torch.linalg.solve_triangular(A, vc * bc[..., None], upper=False, unitriangular=True)
        W = torch.linalg.solve_triangular(A, kb * c.exp()[..., None], upper=False, unitriangular=True)
        Vn = U - W @ S
        o = (qc * c.exp()[..., None]) @ S + ((qc @ kc.transpose(-1, -2)) * D) @ Vn
        out[s:e] = o.transpose(0, 1)
        last = c[:, -1:]
        S = S * last.exp()[..., None] + (kc * (last - c).exp()[..., None]).transpose(-1, -2) @ Vn
    return out, S


# ---------------------------------------------------------------- state


@dataclass
class SeqState:
    """Everything a sequence carries between forward calls."""

    pos: int = 0
    kv: dict = field(default_factory=dict)  # attn layer -> (K [L, Hkv, D], V [L, Hkv, D])
    conv: dict = field(default_factory=dict)  # gdn layer -> [conv_dim, conv_k - 1] (last raw inputs)
    ssm: dict = field(default_factory=dict)  # gdn layer -> [Hv, dk, dv] fp32


# ---------------------------------------------------------------- model


class Qwen38Reference:
    def __init__(self, ckpt: Checkpoint, device="cpu", dtype=torch.float32, act_quant=True,
                 kv_dtype="bf16", state_dtype=torch.float32, gdn="chunked"):
        self.ckpt = ckpt
        self.cfg = TextConfig.from_hf(ckpt.config)
        self.device, self.dtype = torch.device(device), dtype
        self.act_quant = act_quant
        self.kv_dtype = kv_dtype  # "bf16" | "fp8" (e4m3, scale 1.0, as stored by the engines) | "fp32"
        self.state_dtype = state_dtype  # storage precision of the GDN state between calls
        self.gdn = gdn  # "chunked" | "recurrent"
        self.attn_block = 512  # queries per attention block (bounds the scores buffer)

    # --- weights -------------------------------------------------------
    def _get(self, name):
        return self.ckpt.get(name, self.device)

    def _linear(self, prefix):
        return self.ckpt.linear(prefix, self.device, self.dtype, self.act_quant)

    # --- layers ---------------------------------------------------------
    def mlp(self, i, x):
        p = f"{LM}.layers.{i}.mlp"
        gate, up, down = self._linear(f"{p}.gate_proj"), self._linear(f"{p}.up_proj"), self._linear(f"{p}.down_proj")
        return down(F.silu(gate(x)) * up(x))

    def gdn_layer(self, i, x, st: SeqState):
        cfg, p = self.cfg, f"{LM}.layers.{i}.linear_attn"
        T = x.shape[0]
        qkv = self._linear(f"{p}.in_proj_qkv")(x)  # [T, conv_dim]
        z = self._linear(f"{p}.in_proj_z")(x).view(T, cfg.v_heads, cfg.v_dim)
        b = x.float() @ self.ckpt.bf16_weight(f"{p}.in_proj_b", self.device).float().T  # BF16, no activation quant
        a = x.float() @ self.ckpt.bf16_weight(f"{p}.in_proj_a", self.device).float().T

        # Causal depthwise conv over [previous conv_k-1 inputs, this call's inputs], then SiLU.
        prev = st.conv.get(i, torch.zeros(cfg.conv_dim, cfg.conv_k - 1, device=self.device))
        seq = torch.cat((prev, qkv.float().T), dim=-1)  # [conv_dim, conv_k-1+T]
        w = self._get(f"{p}.conv1d.weight").float()  # [conv_dim, 1, conv_k]
        conv = F.silu(F.conv1d(seq[None], w, groups=cfg.conv_dim)[0]).T  # [T, conv_dim]
        st.conv[i] = seq[:, -(cfg.conv_k - 1):].clone()

        kd, vd = cfg.k_heads * cfg.k_dim, cfg.v_heads * cfg.v_dim
        q, k, v = conv.split([kd, kd, vd], dim=-1)
        rep = cfg.v_heads // cfg.k_heads
        q = l2norm(q.reshape(T, cfg.k_heads, cfg.k_dim)).repeat_interleave(rep, dim=1) * cfg.k_dim ** -0.5
        k = l2norm(k.reshape(T, cfg.k_heads, cfg.k_dim)).repeat_interleave(rep, dim=1)
        v = v.reshape(T, cfg.v_heads, cfg.v_dim)
        beta = torch.sigmoid(b)
        g = -self._get(f"{p}.A_log").float().exp() * F.softplus(a + self._get(f"{p}.dt_bias").float())

        S0 = st.ssm.get(i, torch.zeros(cfg.v_heads, cfg.k_dim, cfg.v_dim, device=self.device)).float()
        rule = gated_delta_chunked if self.gdn == "chunked" else gated_delta_recurrent
        o, S1 = rule(q, k, v, g, beta, S0)
        st.ssm[i] = S1.to(self.state_dtype)

        o = gated_rms_norm(o, z, self._get(f"{p}.norm.weight"), cfg.eps).reshape(T, -1)
        return self._linear(f"{p}.out_proj")(o)

    def _store_kv(self, t):
        if self.kv_dtype == "fp8":
            return to_e4m3(t)
        if self.kv_dtype == "nvfp4":  # 16-dim blocks of the head dim, e4m3 scales, global scale 1 (as the engine)
            return quant_act_nvfp4(t, torch.ones((), device=t.device))
        if self.kv_dtype == "bf16":
            return t.to(torch.bfloat16).float()
        return t

    def attn_layer(self, i, x, st: SeqState, cos, sin):
        cfg, p = self.cfg, f"{LM}.layers.{i}.self_attn"
        T, H, Hkv, D = x.shape[0], cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        qg = self._linear(f"{p}.q_proj")(x).view(T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(T, H * D)
        k = self._linear(f"{p}.k_proj")(x).view(T, Hkv, D)
        v = self._linear(f"{p}.v_proj")(x).view(T, Hkv, D)
        q = apply_rope(rms_norm(q, self._get(f"{p}.q_norm.weight"), cfg.eps), cos, sin)
        k = apply_rope(rms_norm(k, self._get(f"{p}.k_norm.weight"), cfg.eps), cos, sin)

        k, v = self._store_kv(k), self._store_kv(v.float())
        if i in st.kv:
            k = torch.cat((st.kv[i][0], k)); v = torch.cat((st.kv[i][1], v))
        st.kv[i] = (k, v)

        L = k.shape[0]
        rep = H // Hkv
        kf = k.repeat_interleave(rep, dim=1).float()  # [L, H, D]
        vf = v.repeat_interleave(rep, dim=1).float()
        # Queries in blocks, so the [H, T, L] scores of a 16K prompt (26 GB in fp32) are never materialized.
        o = torch.empty(T, H, D, device=self.device)
        for s in range(0, T, self.attn_block):
            e = min(s + self.attn_block, T)
            scores = torch.einsum("thd,lhd->htl", q[s:e].float(), kf) * D ** -0.5
            qpos = torch.arange(L - T + s, L - T + e, device=self.device)[:, None]
            mask = torch.arange(L, device=self.device)[None, :] > qpos  # [e-s, L]
            scores = scores.masked_fill(mask, float("-inf"))
            o[s:e] = torch.einsum("htl,lhd->thd", scores.softmax(-1), vf)
        o = o.reshape(T, H * D)
        o = o * torch.sigmoid(gate.float())
        return self._linear(f"{p}.o_proj")(o)

    # --- whole model ------------------------------------------------------
    def embed(self, ids):
        if not hasattr(self, "_embed"):
            self._embed = self._get(f"{LM}.embed_tokens.weight")  # BF16, 2.4 GiB; kept after first use
        return self._embed[ids.to(self.device)].float()

    def lm_head(self):
        return self._linear("lm_head")

    def forward(self, ids: torch.Tensor, st: SeqState, all_logits=False, hidden_layers=(), return_hidden=False,
                mm_rows=None, mm_embeds=None, pos3=None):
        """Run `ids` [T] after the tokens already in `st`; returns logits [T or 1, vocab] (fp32).

        hidden_layers: layer indices whose outputs to return as well (DFlash2 reads 5, 19, 33, 47, 61).
        return_hidden: return the final-normed hidden states [T or 1, hidden] instead of logits, for callers
        that apply `lm_head()` in slices (all-position logits of a 16K prompt are 16 GB in fp32).
        Images (vision.py): rows mm_rows of this call take the embeddings mm_embeds (rounded to BF16, as the
        embedding table), and pos3 [3, T] gives the M-RoPE positions of the call's rows (default: st.pos + t).
        """
        cfg = self.cfg
        T = ids.shape[0]
        if pos3 is None:
            positions = torch.arange(st.pos, st.pos + T)
            cos, sin = rope_cos_sin(positions, cfg, self.device)
        else:
            from .vision import mrope_cos_sin
            cos, sin = mrope_cos_sin(pos3, cfg.rotary_dim, cfg.rope_theta, self.device)
        x = self.embed(ids.clamp(max=cfg.vocab - 1))
        if mm_rows is not None and len(mm_rows):
            x[mm_rows.to(self.device)] = mm_embeds.to(self.device).bfloat16().float()
        taps = {}
        for i, kind in enumerate(cfg.layer_types):
            h = rms_norm(x, self._get(f"{LM}.layers.{i}.input_layernorm.weight"), cfg.eps)
            if kind == "linear_attention":
                x = x + self.gdn_layer(i, h, st)
            else:
                x = x + self.attn_layer(i, h, st, cos, sin)
            h = rms_norm(x, self._get(f"{LM}.layers.{i}.post_attention_layernorm.weight"), cfg.eps)
            x = x + self.mlp(i, h)
            if i in hidden_layers:
                taps[i] = x.clone()
        st.pos += T
        x = rms_norm(x if all_logits else x[-1:], self._get(f"{LM}.norm.weight"), cfg.eps)
        out = x if return_hidden else self.lm_head()(x).float()
        return (out, taps) if hidden_layers else out
