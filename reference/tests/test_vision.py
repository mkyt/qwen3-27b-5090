"""Our reference vision tower and M-RoPE positions against transformers (CPU, fp32, real checkpoint weights)."""

import pytest
import torch
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding, Qwen3_5VisionModel

from qwen27b.vision import VIS, ImageSpan, VisionReference, mrope_cos_sin, mrope_positions
from qwen27b.weights import TARGET, Checkpoint, snapshot


def make_image(h, w, seed=0):
    """A deterministic RGB test picture (gradients, blocks and noise) as a uint8 [h, w, 3] tensor."""
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, h)[:, None]
    x = torch.linspace(0, 1, w)[None, :]
    img = torch.stack((y.expand(h, w), x.expand(h, w), ((x * 7).floor() + (y * 5).floor()) % 2), -1)
    img = img * 0.8 + 0.2 * torch.rand(h, w, 3, generator=g)
    return (img * 255).round().clamp(0, 255).to(torch.uint8)


@pytest.fixture(scope="module")
def ckpt():
    return Checkpoint(snapshot(*TARGET))


@pytest.fixture(scope="module")
def processor():
    return AutoImageProcessor.from_pretrained(snapshot(*TARGET))


def hf_vision(ckpt):
    cfg = Qwen3_5Config(**{k: v for k, v in ckpt.config.items() if k != "quantization_config"}).vision_config
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(cfg).float().eval()
    sd = {n[len(VIS) + 1:]: ckpt.get(n).float() for n in ckpt.weight_map if n.startswith(VIS + ".")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    return model


def test_vision_matches_transformers(ckpt, processor):
    # Two images in one call (packed, attention separate per image), one not square.
    imgs = [make_image(256, 320, 0), make_image(200, 130, 1)]
    batch = processor(images=[i.permute(2, 0, 1) for i in imgs], return_tensors="pt")
    pv, grid = batch["pixel_values"], batch["image_grid_thw"]
    ours = VisionReference(ckpt, round_bf16=False).encode(pv, grid.tolist())
    with torch.no_grad():
        theirs = hf_vision(ckpt)(pv.float(), grid_thw=grid).pooler_output
    assert ours.shape == theirs.shape == (int(grid.prod(-1).sum()) // 4, 5120)
    err = (ours - theirs).abs().max() / theirs.abs().max()
    assert err < 1e-4, err


def test_mrope_positions_match_transformers(ckpt):
    cfg = Qwen3_5Config(**{k: v for k, v in ckpt.config.items() if k != "quantization_config"})
    model = Qwen3_5Model.__new__(Qwen3_5Model)  # get_rope_index needs only the config
    torch.nn.Module.__init__(model)
    model.config = cfg
    pad = cfg.image_token_id
    # text, image 3x5 (merged grid; patches 6x10), text, image 4x2, text
    spans = [ImageSpan(4, 3, 5), ImageSpan(4 + 15 + 3, 4, 2)]
    n = spans[1].start + spans[1].length + 6
    ids = torch.full((n,), 100)
    types = torch.zeros(n, dtype=torch.int)
    for sp in spans:
        ids[sp.start:sp.start + sp.length] = pad
        types[sp.start:sp.start + sp.length] = 1
    grid = torch.tensor([[1, sp.gh * 2, sp.gw * 2] for sp in spans])
    theirs, delta = model.get_rope_index(ids[None], types[None], image_grid_thw=grid)
    ours, our_delta = mrope_positions(n, spans)
    assert torch.equal(ours, theirs[:, 0])
    assert our_delta == int(delta)

    tcfg = Qwen3_5TextConfig(**ckpt.config["text_config"])
    rot = Qwen3_5TextRotaryEmbedding(tcfg)
    cos, sin = rot(torch.zeros(1, dtype=torch.float32), ours[:, None, :])
    ocos, osin = mrope_cos_sin(ours, 64, tcfg.rope_parameters["rope_theta"], "cpu")
    assert torch.allclose(ocos, cos[0].float(), atol=1e-6) and torch.allclose(osin, sin[0].float(), atol=1e-6)
