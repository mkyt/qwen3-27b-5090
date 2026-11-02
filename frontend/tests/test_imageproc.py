"""imageproc.py without transformers: the banded resize is PIL's single resize call, pixel for pixel."""

import io
import pathlib
import random
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import imageproc  # noqa: E402

rng = random.Random(0)
SIZES = [((w, h), (rw, rh)) for w, h, rw, rh in
         [(4032, 3024, 2336, 1760), (3840, 2160, 2720, 1536), (100, 150, 224, 320), (640, 480, 640, 224),
          (37, 900, 32, 896), (5, 3, 64, 32)]]
SIZES += [((rng.randint(8, 900), rng.randint(8, 900)), (rng.randint(8, 700), rng.randint(8, 700))) for _ in range(40)]


@pytest.mark.parametrize("size, out", SIZES)
def test_banded_resize_is_one_resize(size, out):
    img = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
    assert imageproc.resize(img, out).tobytes() == img.resize(out, Image.BICUBIC).tobytes()


def test_rgb_decodes_and_resizes():
    img = Image.frombytes("RGBA", (300, 200), rng.randbytes(300 * 200 * 4))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    assert imageproc.rgb(buf.getvalue(), (300, 200)) == img.convert("RGB").tobytes()
    assert imageproc.rgb(buf.getvalue(), (96, 64)) == img.convert("RGB").resize((96, 64), Image.BICUBIC).tobytes()
