"""Reference vision tower of Qwen3.8-27B and the multimodal positions (M-RoPE), for correctness tests.

Plain PyTorch in fp32 on the BF16 checkpoint weights (`model.visual.*`), following transformers 5.18
`modeling_qwen3_5.py` / `vision_utils.py`:

- input: `pixel_values` [patches, 3 * 2 * 16 * 16] and `grid_thw` from the image processor
  (Qwen2VLImageProcessorFast; patches come in 2x2 merge-block order), one image = (1, gh, gw) patches;
- patch embed (a Conv3d with stride = kernel, i.e. a linear layer on the flattened patch) + the learned
  48 x 48 position table resampled bilinearly (align_corners) to the image's patch grid;
- 27 pre-LN blocks: LayerNorm -> qkv -> 2D RoPE (rotate_half over all 72 dims; the first 18 frequencies
  take the patch row, the next 18 the column, repeated) -> non-causal attention within the image -> proj;
  LayerNorm -> fc1 -> GELU (tanh) -> fc2;
- merger: LayerNorm per patch, 4 consecutive patches (one 2x2 block) concatenated -> fc1 -> GELU (erf) -> fc2
  -> one 5120-dim embedding per merged token.

In the language model each image is a run of `<|image_pad|>` tokens (one per merged token, row-major over
the (gh/2, gw/2) grid) whose embeddings are replaced by these. Their M-RoPE positions are (t, h, w) =
(s, s + row, s + col) with s the running position, which then advances by max(gh/2, gw/2) instead of the
token count, so every later token sits `delta` positions before its index.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

VIS = "model.visual"


@dataclass
class VisionConfig:
    hidden: int
    heads: int
    depth: int
    inter: int
    out_hidden: int
    patch: int
    temporal_patch: int
    merge: int
    num_pos: int
    in_channels: int

    @classmethod
    def from_hf(cls, cfg: dict) -> "VisionConfig":
        v = cfg["vision_config"]
        return cls(hidden=v["hidden_size"], heads=v["num_heads"], depth=v["depth"], inter=v["intermediate_size"],
                   out_hidden=v["out_hidden_size"], patch=v["patch_size"], temporal_patch=v["temporal_patch_size"],
                   merge=v["spatial_merge_size"], num_pos=v["num_position_embeddings"], in_channels=v["in_channels"])

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch * self.patch * self.patch


def patch_coords(gh: int, gw: int, merge: int = 2):
    """(row, col) of each patch in the processor's order: 2x2 merge blocks row-major, row-major inside."""
    idx = torch.arange(gh * gw)
    in_col = idx % merge
    in_row = (idx // merge) % merge
    blocks_w = gw // merge
    block_col = (idx // (merge * merge)) % blocks_w
    block_row = idx // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col


def pos_interp(gh: int, gw: int, side: int, merge: int = 2):
    """Bilinear (align_corners) taps into the side x side table: indices, weights [gh*gw, 4]."""
    row, col = patch_coords(gh, gw, merge)

    def axis(i, n):
        src = i.float() * (side - 1) / max(n - 1, 1)
        f = src.floor()
        taps = (f.long()[:, None] + torch.arange(2)).clamp(0, side - 1)
        wts = (1 - (src[:, None] - f[:, None] - torch.arange(2)).abs()).clamp(min=0)
        return taps, wts

    ht, hw = axis(row, gh)
    wt, ww = axis(col, gw)
    idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
    wts = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
    return idx, wts


def vision_rope(gh: int, gw: int, head_dim: int, theta: float = 10000.0, merge: int = 2):
    """cos / sin [gh*gw, head_dim] of the 2D rotary embedding (rotate_half layout)."""
    row, col = patch_coords(gh, gw, merge)
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))  # [head_dim / 4]
    fr = row.float()[:, None] * inv_freq
    fc = col.float()[:, None] * inv_freq
    emb = torch.cat((fr, fc), -1)
    emb = torch.cat((emb, emb), -1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class VisionReference:
    def __init__(self, ckpt, device="cpu", round_bf16=True):
        self.ckpt = ckpt
        # Round the pixels and the position embeddings to BF16 as a BF16 deployment does (transformers in
        # fp32 does not, which is what the comparison with it turns off).
        self.round_bf16 = round_bf16
        self.cfg = VisionConfig.from_hf(ckpt.config)
        self.device = torch.device(device)
        self._w = {}

    def _get(self, name):
        if name not in self._w:
            self._w[name] = self.ckpt.get(f"{VIS}.{name}", self.device).float()
        return self._w[name]

    def _linear(self, x, name):
        return F.linear(x, self._get(f"{name}.weight"), self._get(f"{name}.bias"))

    def _ln(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self._get(f"{name}.weight"), self._get(f"{name}.bias"), 1e-6)

    def patch_embed(self, pixel_values):
        w = self._get("patch_embed.proj.weight").reshape(self.cfg.hidden, -1)
        # (transformers casts the pixels to the weight dtype, BF16, before the conv)
        px = pixel_values.to(self.device).float()
        if self.round_bf16:
            px = px.bfloat16().float()
        return px @ w.T + self._get("patch_embed.proj.bias")

    def encode(self, pixel_values: torch.Tensor, grid_thw, return_blocks=False):
        """pixel_values [P, 1536], grid_thw [(1, gh, gw), ...] -> [P / 4, out_hidden] fp32 (images concatenated).
        return_blocks: also the hidden states after the patch embedding and after each block."""
        cfg = self.cfg
        side = int(cfg.num_pos ** 0.5)
        x = self.patch_embed(pixel_values)
        table = self._get("pos_embed.weight")
        outs, blocks, start = [], [], 0
        cos_l, sin_l, seg = [], [], []
        for t, gh, gw in grid_thw:
            assert t == 1, "images only"
            n = gh * gw
            idx, wts = pos_interp(gh, gw, side, cfg.merge)
            # (transformers sums the interpolated table in fp32 and casts the sum to the hidden dtype)
            pe = (table[idx.to(self.device)] * wts.to(self.device)[:, :, None]).sum(1)
            x[start:start + n] += pe.bfloat16().float() if self.round_bf16 else pe
            c, s = vision_rope(gh, gw, cfg.head_dim, merge=cfg.merge)
            cos_l.append(c), sin_l.append(s), seg.append((start, start + n))
            start += n
        cos, sin = torch.cat(cos_l).to(self.device), torch.cat(sin_l).to(self.device)
        if return_blocks:
            blocks.append(x.clone())
        H, D = cfg.heads, cfg.head_dim
        for i in range(cfg.depth):
            p = f"blocks.{i}"
            h = self._ln(x, f"{p}.norm1")
            qkv = self._linear(h, f"{p}.attn.qkv").view(-1, 3, H, D)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            o = torch.empty_like(q)
            for a, b in seg:
                o[a:b] = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1),
                                                        v[a:b].transpose(0, 1)).transpose(0, 1)
            x = x + self._linear(o.reshape(-1, H * D), f"{p}.attn.proj")
            h = self._ln(x, f"{p}.norm2")
            x = x + self._linear(F.gelu(self._linear(h, f"{p}.mlp.linear_fc1"), approximate="tanh"), f"{p}.mlp.linear_fc2")
            if return_blocks:
                blocks.append(x.clone())
        h = self._ln(x, "merger.norm").reshape(-1, cfg.hidden * cfg.merge ** 2)
        out = self._linear(F.gelu(self._linear(h, "merger.linear_fc1")), "merger.linear_fc2")
        return (out, blocks) if return_blocks else out


# ---------------------------------------------------------------- multimodal positions (M-RoPE)


@dataclass
class ImageSpan:
    """One image in a token sequence: `<|image_pad|>` x (gh * gw) at positions start .. start + gh*gw - 1,
    with the merged grid gh x gw (patch grid / 2)."""

    start: int
    gh: int
    gw: int

    @property
    def length(self) -> int:
        return self.gh * self.gw


def mrope_positions(n: int, spans: list) -> tuple[torch.Tensor, int]:
    """Positions [3, n] (t, h, w) of a sequence of n tokens with images at `spans`, and the final delta
    (position of the next token = n + delta), as transformers' get_rope_index."""
    pos = torch.empty(3, n, dtype=torch.long)
    cur, i = 0, 0
    for sp in sorted(spans, key=lambda s: s.start):
        k = sp.start - i
        pos[:, i:sp.start] = torch.arange(cur, cur + k)
        cur += k
        r = torch.arange(sp.length)
        pos[0, sp.start:sp.start + sp.length] = cur
        pos[1, sp.start:sp.start + sp.length] = cur + r // sp.gw
        pos[2, sp.start:sp.start + sp.length] = cur + r % sp.gw
        cur += max(sp.gh, sp.gw)
        i = sp.start + sp.length
    pos[:, i:] = torch.arange(cur, cur + n - i)
    return pos, int(cur + n - i - n)


def mrope_cos_sin(pos3: torch.Tensor, rotary_dim: int, theta: float, device, sections=(11, 11, 10)):
    """cos / sin [T, rotary_dim] for 3-axis positions [3, T]: interleaved M-RoPE, frequency i takes axis
    1 (h) when i % 3 == 1 and i < 3 * sections[1], axis 2 (w) when i % 3 == 2 and i < 3 * sections[2], else t."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    fi = torch.arange(half, device=device)
    axis = torch.zeros(half, dtype=torch.long, device=device)
    axis[(fi % 3 == 1) & (fi < 3 * sections[1])] = 1
    axis[(fi % 3 == 2) & (fi < 3 * sections[2])] = 2
    p = pos3.to(device=device, dtype=torch.float32)  # [3, T]
    freqs = p[axis].T * inv_freq[None, :]  # [T, half]
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos(), emb.sin()


def expand_image_pads(ids, grids, pad_id: int, merge: int = 2, fill=None):
    """Expand each single `<|image_pad|>` of a templated prompt (one per image, in order) to the image's
    merged token count. grids: [(1, gh, gw)] patch grids from the processor. fill(i, n) gives the n ids of
    image i (default: the pad id n times). Returns (ids, [ImageSpan])."""
    out, spans, k = [], [], 0
    for t in ids:
        if t != pad_id:
            out.append(t)
            continue
        _, gh, gw = (int(v) for v in grids[k])
        sp = ImageSpan(len(out), gh // merge, gw // merge)
        out.extend(fill(k, sp.length) if fill else [pad_id] * sp.length)
        spans.append(sp)
        k += 1
    assert k == len(grids), f"{k} image pads for {len(grids)} images"
    return out, spans


VOCAB = 248320


def image_token_ids(key: bytes, n: int) -> list:
    """The n prompt ids of an image (P4-12): out-of-vocabulary ids derived from the image's content, so that
    the engine's prefix cache matches a prompt only where the same images sit at the same positions."""
    import hashlib

    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    span = 2**31 - 1 - VOCAB
    return [VOCAB + (h + j) % span for j in range(n)]
