"""The checkpoint's image processor without transformers / PyTorch, as transformers 5.18 `Qwen2VLImageProcessorPil`
does it: smart_resize to multiples of patch * merge and the PIL bicubic resize here; the rest (rescale + normalize,
patches in 2x2 merge-block order, each repeated over the temporal patch) on the device from the resized image's
bytes (q27_mm_encode_rgb; reference/qwen27b/vision.py `patchify` is the same on the CPU). The grids are those of the
default torchvision backend too; its resize rounds differently by at most 1 in 255 on a few pixels.
frontend/tests/test_tokenizer.py compares them where transformers is installed.

The grid needs only the image's size, which PIL reads from the header: the server tokenizes with it, and the
engine process decodes and resizes (rgb) only the images it has to encode (not those in a cached prefix).
"""

import array
import io
import json
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

RESIZE_STRIPS = 8  # threads of a resize (PIL releases the GIL); the result is that of one resize call
_pool = None


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """(height, width) divisible by factor, with min_pixels..max_pixels pixels and about the same aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class ImageProcessor:
    """preprocessor_config.json of a model directory."""

    def __init__(self, path):
        c = json.loads((pathlib.Path(path) / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.patch, self.merge, self.temporal = c["patch_size"], c["merge_size"], c["temporal_patch_size"]
        self.min_pixels = c["size"]["shortest_edge"]
        # (transformers fuses rescale into normalize: (x - mean / rescale) / (std / rescale), in float32)
        rescale = c.get("rescale_factor", 1 / 255)
        mean = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c["image_mean"]]
        std = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c["image_std"]]
        self.norm = tuple(mean + std)

    def size(self, size, max_pixels):
        """The resized (width, height) of an image of `size` (width, height)."""
        w, h = size
        rh, rw = smart_resize(h, w, self.patch * self.merge, self.min_pixels, max_pixels)
        return rw, rh

    def grid(self, size):
        """The (1, gh, gw) patch grid of a resized (width, height)."""
        return 1, size[1] // self.patch, size[0] // self.patch


def rgb(raw, size):
    """The resized image's [height][width][3] uint8 bytes, from the encoded image `raw` and the resized size."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return (resize(img, size) if img.size != tuple(size) else img).tobytes()


def resize(img, size):
    """img.resize(size, BICUBIC), the same pixels, on RESIZE_STRIPS threads (PIL releases the GIL). PIL resizes
    in two passes, horizontal then vertical, with a uint8 image in between: the horizontal pass runs here on bands
    of rows and the vertical one on bands of columns, each a resize along one axis whose length is the whole
    image's, so its filter coefficients are those of the single call. (Bands of output rows through resize's box
    would not be: the box shifts the filter centers by a rounding error.)"""
    global _pool
    w, h = img.size
    rw, rh = size
    if _pool is None:
        _pool = ThreadPoolExecutor(RESIZE_STRIPS, thread_name_prefix="resize")
    if rw != w:
        img = _bands(img, h, lambda b, n: b.resize((rw, n), Image.BICUBIC), rows=True)
    if rh != h:
        img = _bands(img, rw, lambda b, n: b.resize((n, rh), Image.BICUBIC), rows=False)
    return img


def _bands(img, length, f, rows):
    """f(band, its length) over bands of rows (rows) or columns of img, pasted together."""
    n = min(RESIZE_STRIPS, length)
    cuts = [length * i // n for i in range(n + 1)]
    W, H = img.size
    boxes = [(0, a, W, b) if rows else (a, 0, b, H) for a, b in zip(cuts, cuts[1:])]
    parts = list(_pool.map(lambda k: f(img.crop(boxes[k]), cuts[k + 1] - cuts[k]), range(n)))
    out = Image.new("RGB", (parts[0].width, sum(p.height for p in parts)) if rows else (sum(p.width for p in parts),
                                                                                       parts[0].height))
    for (x, y, _, _), part in zip(boxes, parts):
        out.paste(part, (0, y) if rows else (x, 0))
    return out


def _f32(v):
    """v rounded to float32 (the precision of transformers' and the vision tower's arithmetic)."""
    return array.array("f", [v])[0]
