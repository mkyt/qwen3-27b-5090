"""The API server's tokenizer, chat template and image processor (tokenizer.py, imageproc.py) against transformers'
(skipped where transformers or the checkpoint is missing): the same prompt text, ids, decoded text and image grids."""

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))

from chat import normalize_messages, template_kwargs  # noqa: E402
from imageproc import ImageProcessor  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def model():
    from qwen27b.paths import engine_target

    try:
        path = engine_target("gdnfp4")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return path


@pytest.fixture(scope="module")
def toks(model):
    return Tokenizer(model), transformers.AutoTokenizer.from_pretrained(model)


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string", "description": "\u30d1\u30b9"}, "limit": {"type": "integer"}},
    "required": ["path"]}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})}}
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
CHATS = [
    ([{"role": "user", "content": "Hello"}], None, {}),
    ([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "\u65e5\u672c\u8a9e\u3067\u7b54\u3048\u3066"}], None, {}),
    ([{"role": "user", "content": "hi"}], None, {"enable_thinking": False}),
    ([{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"}),
    ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
      {"role": "user", "content": "again <|im_start|> & <b>"}], None, {}),
    ([{"role": "developer", "content": "Use tools."}, {"role": "user", "content": "read a.py"},
      {"role": "assistant", "content": "", "tool_calls": [CALL]},
      {"role": "tool", "tool_call_id": "call_1", "content": "print('x')\n"}], TOOLS, {}),
    ([{"role": "user", "content": [{"type": "text", "text": "What is this?"}, IMAGE, IMAGE]}], None, {}),
    ([{"role": "user", "content": [IMAGE, {"type": "text", "text": "and"}]},
      {"role": "assistant", "content": "A cat."}, {"role": "user", "content": [IMAGE]}], TOOLS, {"enable_thinking": False}),
]


def texts():
    out = [p.read_text() for p in sorted((ROOT / "bench/golden/sources").glob("*.txt"))]
    return out + ["", " ", "a\r\nb\n\n\n  c\t", "<|im_start|>user\nx<|im_end|>", "🎉👍🏽 é \uff71 \U00020bb7", "<think>\n</think>"]


@pytest.mark.parametrize("i", range(len(CHATS)))
def test_render_matches_transformers(toks, i):
    mine, hf = toks
    messages, tools, kw = CHATS[i]
    messages = normalize_messages(messages)
    kw = template_kwargs({"chat_template_kwargs": kw})
    a = mine.render(messages, tools=tools, **kw)
    b = hf.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw)
    assert a == b
    assert mine.encode(a) == hf(a, add_special_tokens=False).input_ids


def test_render_errors_like_transformers(toks):
    mine, hf = toks
    bad = [{"role": "system", "content": [IMAGE]}, {"role": "user", "content": "x"}]
    with pytest.raises(Exception) as a:
        mine.render(bad)
    with pytest.raises(Exception) as b:
        hf.apply_chat_template(bad, add_generation_prompt=True, tokenize=False)
    assert str(a.value) == str(b.value)


def test_encode_decode_match_transformers(toks):
    mine, hf = toks
    for text in texts():
        ids = mine.encode(text)
        assert ids == hf(text, add_special_tokens=False).input_ids
        assert mine.decode(ids) == hf.decode(ids, skip_special_tokens=False)
        for cut in range(0, len(ids), max(1, len(ids) // 7)):  # (incomplete UTF-8 at the ends of slices)
            assert mine.decode(ids[cut:cut + 3]) == hf.decode(ids[cut:cut + 3], skip_special_tokens=False)
    for t in ("<|im_start|>", "<|im_end|>", "<|image_pad|>", "<think>"):
        assert mine.token_id(t) == hf.convert_tokens_to_ids(t)


def test_images_match_transformers(model):
    """Grids are the same; pixels are those of transformers' PIL backend (its default torchvision backend rounds
    the bicubic resize differently: at most 1 in 255 on about 0.5% of the values of noise, none on the photo)."""
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    mine, pil = ImageProcessor(model), Qwen2VLImageProcessorPil.from_pretrained(model)
    tv = transformers.AutoImageProcessor.from_pretrained(model)
    rng = np.random.default_rng(0)
    images = [Image.open(ROOT / "bench/vision/astronaut.jpg").convert("RGB")]
    images += [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((100, 150), (2048, 3000))]
    for img in images:
        for max_pixels in (1 << 20, 4194304):
            size = {"shortest_edge": 65536, "longest_edge": max_pixels}
            pv, grids = mine([img], max_pixels)
            a = pil(images=[img], size=size, return_tensors="np")
            b = tv(images=[img], size=size, return_tensors="np")
            assert a["image_grid_thw"].tolist() == b["image_grid_thw"].tolist() == [list(g) for g in grids]
            np.testing.assert_allclose(pv, a["pixel_values"], rtol=0, atol=1e-6)
            assert np.abs(pv - b["pixel_values"]).max() <= 2 / 127.5 + 1e-6
            assert (pv != b["pixel_values"]).mean() < 0.01
