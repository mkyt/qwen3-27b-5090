"""Start settings of the engine (launch.py) and the device memory their plans assume (stats.Hub.plan_device). The
plans themselves need libq27 and the weights' headers (no GPU); without them those tests are skipped."""

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
    # The serve default's startup log on this device: kv pool 9461 pages (605504 tokens), device 28.378 GiB.
    assert (p["max_tokens"], p["kv_tokens"], p["fits"]) == (262144, 605504, True)
    assert abs(p["device"] / GiB - 28.378) < 0.001 and abs(p["budget"] / GiB - 28.68) < 0.01
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
