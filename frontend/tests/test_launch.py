"""Start settings of the engine (launch.py) and the device memory their plans assume (stats.Hub.plan_device). The
plans themselves need libq27 and the weights' headers (no GPU); without them those tests are skipped."""

import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))

import launch  # noqa: E402
from test_stats import hub  # noqa: E402

GiB = 1 << 30
DEFAULTS = {"weights": "gdnfp4", "kv_dtype": "nvfp4", "spec": "dflash2", "vision": "host", "max_seqs": 8,
            "max_tokens": 262144, "vram_budget_gib": None}
DEVICE = (int(30.20 * GiB), int(31.82 * GiB))  # (free at the start, total) of an engine on the RTX 5090 under WSL2


def settings():
    return launch.Launch({"prefill_chunk": 2048, "engine_python": str(ROOT / "engine/python")}, dict(DEFAULTS))


def test_auto_budget_is_the_engines():
    # memory.cc budget_for: min(free - 0.6, 30.3 - used); the engine logged 28.69 GiB for this device.
    assert abs(launch.auto_budget(*DEVICE) / GiB - 28.68) < 0.01


def test_validate():
    lc = settings()
    assert lc.validate({"kv_dtype": "fp8", "vram_budget_gib": 20}) == dict(DEFAULTS, kv_dtype="fp8", vram_budget_gib=20.0)
    assert lc.settings == DEFAULTS  # (validate does not change them)
    for bad in ({"kv_dtype": "int4"}, {"max_seqs": 9}, {"max_seqs": 2.0}, {"max_tokens": 1 << 20}, {"vision": True},
                {"vram_budget_gib": 2}, {"vram_budget_gib": "auto"}, {"what": 1}, {"weights": "/elsewhere"}):
        with pytest.raises(ValueError):
            lc.validate(bad)
    with pytest.raises(ValueError, match="at most the automatic budget"):
        lc.validate({"vram_budget_gib": 30}, device=DEVICE)
    assert lc.validate({"vram_budget_gib": 28.5}, device=DEVICE)["vram_budget_gib"] == 28.5


def test_plan_device():
    h = hub()
    assert h.plan_device() == (None, None)
    h.samples.append({"t": 0, "used": 1 * GiB, "total": 32 * GiB})
    h.engine_state = "stopped"
    assert h.plan_device() == ((31 * GiB - h.context_bytes, 32 * GiB), "nvml")
    # A started engine: its free memory at the start, and the CUDA context measured against NVML before it.
    h.baseline_used, h.nvml_total = 1 * GiB, 32 * GiB
    h.engine_state = "ready"
    h.set_snapshot({"t": 1.0, "seqs": [], "device": {"free_at_start": 30 * GiB, "total": 31.9 * GiB}})
    assert h.plan_device() == ((30 * GiB, 31.9 * GiB), "engine")
    assert h.context_bytes == 1 * GiB
    h.engine_state = "stopped"  # (another process may take memory meanwhile: NVML now)
    h.samples.append({"t": 2, "used": 3 * GiB, "total": 32 * GiB})
    assert h.plan_device() == ((28 * GiB, 31.9 * GiB), "nvml")


def planner_or_skip():
    lc = settings()
    if not pathlib.Path(os.environ.get("Q27_LIB", "/build/engine/libq27.so")).exists():
        pytest.skip("no libq27.so")
    if lc.weights_choices()["gdnfp4"] is not None:
        pytest.skip("the gdnfp4 weights are not downloaded")
    return lc


def test_plans_match_the_engine():
    lc = planner_or_skip()
    os.environ.setdefault("MAX_TOKENS_FIT", "1")
    os.environ.setdefault("STATE_HOST_CACHE", "4")
    out = lc.plans(lc.settings, DEVICE)
    p = out["plan"]
    # The serve default's startup log on this device: kv pool 9461 pages (605504 tokens), device 28.370 GiB
    # (28.378 until the own GEMMs replaced CUTLASS and its 7.8 MiB workspace).
    assert (p["max_tokens"], p["kv_tokens"], p["fits"]) == (262144, 605504, True)
    assert abs(p["device"] / GiB - 28.370) < 0.001 and abs(p["budget"] / GiB - 28.68) < 0.01
    kv = {v["value"]: v["plan"] for v in out["variants"]["kv_dtype"]}
    assert kv["nvfp4"] == p and kv["fp8"]["kv_tokens"] < kv["nvfp4"]["kv_tokens"]
    small = {v["value"]: v["plan"] for v in out["variants"]["vram_budget_gib"]}[20.0]
    assert small["max_tokens"] < 262144 and small["max_tokens"] % 1024 == 0 and small["fits"]
    seqs = [v["plan"]["kv_tokens"] for v in out["variants"]["max_seqs"]]
    assert seqs == sorted(seqs, reverse=True)  # (each sequence's buffers take some of the pool)


def test_plan_errors():
    lc = planner_or_skip()
    p = lc.plan_many([dict(DEFAULTS, vram_budget_gib=4.0)], DEVICE)[0]
    assert "error" in p and "VRAM budget" in p["error"]


def test_tokenizer_fingerprint(tmp_path):
    # The ids decide: a tokenizer.json rewritten by another transformers (merges as strings, another pre-tokenizer
    # pattern, decoder flags; Huihui's abliterated checkpoint) is the same tokenizer, and so is one with another chat
    # template (unsloth's: prompts are then rendered with it); another vocabulary is not.
    from tokenizer import fingerprint

    def write(name, merges, pattern, vocab=None, template="{{ eos_token }}"):
        d = tmp_path / name
        d.mkdir()
        tok = {"model": {"type": "BPE", "vocab": vocab or {"a": 0, "b": 1, "ab": 2}, "merges": merges},
               "added_tokens": [{"id": 3, "content": "<|im_start|>", "special": True}],
               "pre_tokenizer": {"type": "Split", "pattern": {"Regex": pattern}}, "decoder": {"type": "ByteLevel"}}
        (d / "tokenizer.json").write_text(json.dumps(tok))
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "eos_token": "<|im_end|>"}))
        return d

    base = fingerprint(write("base", [["a", "b"]], r"\p{L}\p{M}+"))
    assert fingerprint(write("rewritten", ["a b"], r"\p{L}+")) == base
    assert fingerprint(write("template", [["a", "b"]], r"\p{L}\p{M}+", template="{{ bos_token }}")) == base
    assert fingerprint(write("other", [["a", "b"]], r"\p{L}\p{M}+", {"a": 0, "b": 1, "ab": 5})) != base


def test_tokenizer_template_of_other_weights():
    # The server's tokenizer with the chat template of the weights the engine runs (Server.use_template_of).
    from tokenizer import Tokenizer

    try:
        own, other = launch.model_dir("gdnfp4"), launch.model_dir("unsloth/Qwen3.8-27B-NVFP4")
    except (FileNotFoundError, KeyError, ValueError) as e:
        pytest.skip(str(e))
    msgs = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "c"}]
    mixed, theirs = Tokenizer(own, template=other), Tokenizer(other)
    assert mixed.path == own and mixed.template_path == other
    assert mixed.render(msgs) == theirs.render(msgs)  # (unsloth's template merges the system messages)
    with pytest.raises(Exception, match="System message must be at the beginning"):
        Tokenizer(own).render(msgs)
    assert mixed.encode("日本語 text") == Tokenizer(own).encode("日本語 text")
