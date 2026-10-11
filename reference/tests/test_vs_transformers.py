"""Our reference layers against transformers' Qwen3_5DecoderLayer, on the real checkpoint weights (CPU, fp32).

Weights are dequantized and activation quantization is off on our side, because transformers runs the
dequantized weights as plain linear layers. This pins down layout and formula details (zero-centred norms,
the q_proj gate split, partial RoPE, conv / l2norm / decay of the GDN) independently of quantization.
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from qwen27b.model import LM, Qwen38Reference, SeqState, rms_norm, rope_cos_sin
from qwen27b.weights import TARGET, Checkpoint, snapshot


@pytest.fixture(scope="module")
def ref():
    return Qwen38Reference(Checkpoint(snapshot(*TARGET)), act_quant=False, kv_dtype="fp32")


def hf_layer(ref, i):
    cfg = Qwen3_5TextConfig(**ref.ckpt.config["text_config"])
    cfg._attn_implementation = "sdpa"
    layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
    sd = {}
    prefix = f"{LM}.layers.{i}."
    for name in ref.ckpt.weight_map:
        if not name.startswith(prefix) or name.endswith(("input_scale", "weight_scale", "weight_scale_2")):
            continue
        local = name[len(prefix):]
        if local.endswith(".weight") and local.rsplit(".", 1)[0].endswith(("proj", "proj_qkv", "proj_z")):
            sd[local] = ref._linear(name[: -len(".weight")]).weight.float()
        else:
            sd[local] = ref.ckpt.get(name).float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("rotary" in m for m in missing), missing
    return cfg, layer


def our_layer(ref, i, x, st, cos, sin):
    h = rms_norm(x, ref._get(f"{LM}.layers.{i}.input_layernorm.weight"), ref.cfg.eps)
    if ref.cfg.layer_types[i] == "linear_attention":
        x = x + ref.gdn_layer(i, h, st)
    else:
        x = x + ref.attn_layer(i, h, st, cos, sin)
    h = rms_norm(x, ref._get(f"{LM}.layers.{i}.post_attention_layernorm.weight"), ref.cfg.eps)
    return x + ref.mlp(i, h)


@pytest.mark.parametrize("i", [0, 3])  # first GDN layer, first attention layer
def test_layer_matches_transformers(ref, i):
    torch.manual_seed(i)
    T = 48
    x = torch.randn(T, ref.cfg.hidden) * 0.5
    cfg, layer = hf_layer(ref, i)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(T).view(1, 1, T).expand(3, 1, T)
    with torch.no_grad():
        expected = layer(x[None], position_embeddings=rotary(x[None], pos))[0]

    # ours, in two calls (prefix then the rest) to also exercise the carried state / KV cache
    cos, sin = rope_cos_sin(torch.arange(T), ref.cfg, "cpu")
    st = SeqState()
    a = our_layer(ref, i, x[:29], st, cos[:29], sin[:29])
    b = our_layer(ref, i, x[29:], st, cos[29:], sin[29:])
    got = torch.cat((a, b))
    err = (got - expected).abs().max() / expected.abs().max()
    assert err < 1e-4, err
