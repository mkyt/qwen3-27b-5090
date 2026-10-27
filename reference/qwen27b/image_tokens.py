"""Image runs in a token sequence: expanding each `<|image_pad|>` of a templated prompt to the image's tokens,
and the out-of-vocabulary ids those tokens get (no PyTorch, so the API server can use it)."""

import hashlib
from dataclasses import dataclass


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
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    span = 2**31 - 1 - VOCAB
    return [VOCAB + (h + j) % span for j in range(n)]
