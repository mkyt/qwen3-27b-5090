"""The checkpoint's image processor without transformers / PyTorch: pixel_values and grids as transformers 5.18
`Qwen2VLImageProcessorPil` makes them (smart_resize to multiples of patch * merge, PIL bicubic resize, rescale +
normalize, patches in 2x2 merge-block order, each repeated over the temporal patch), with PIL and numpy. The grids
are those of the default torchvision backend too; its resize rounds differently by at most 1 in 255 on a few pixels.
frontend/tests/test_tokenizer.py compares them where transformers is installed.
"""

import json
import math
import pathlib

import numpy as np
from PIL import Image


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
    """preprocessor_config.json of a model directory; __call__ takes RGB PIL images."""

    def __init__(self, path):
        c = json.loads((pathlib.Path(path) / "preprocessor_config.json").read_text())
        self.patch, self.merge, self.temporal = c["patch_size"], c["merge_size"], c["temporal_patch_size"]
        self.min_pixels = c["size"]["shortest_edge"]
        # (transformers fuses rescale into normalize: (x - mean / rescale) / (std / rescale), in float32)
        rescale = c.get("rescale_factor", 1 / 255)
        self.mean = (np.array(c["image_mean"], np.float32) * (1.0 / rescale)).astype(np.float32)[:, None, None]
        self.std = (np.array(c["image_std"], np.float32) * (1.0 / rescale)).astype(np.float32)[:, None, None]

    def __call__(self, images, max_pixels):
        """(pixel_values float32 [sum of gh * gw, 3 * temporal * patch * patch], [(1, gh, gw)] patch grids)."""
        p, m, t = self.patch, self.merge, self.temporal
        out, grids = [], []
        for img in images:
            w, h = img.size
            rh, rw = smart_resize(h, w, p * m, self.min_pixels, max_pixels)
            if (rh, rw) != (h, w):
                img = img.resize((rw, rh), Image.BICUBIC)
            x = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)
            x = (x - self.mean) / self.std
            c, gh, gw = x.shape[0], rh // p, rw // p
            x = x.reshape(c, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
            x = np.broadcast_to(x[:, :, :, :, :, None], x.shape[:5] + (t, p, p))
            out.append(x.reshape(gh * gw, c * t * p * p))
            grids.append((1, gh, gw))
        return np.concatenate(out), grids
