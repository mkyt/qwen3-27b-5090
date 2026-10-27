"""The stats page's data (stats.Hub, worker.Scheduler.stats) against fake engine messages: no GPU."""

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stats  # noqa: E402
import worker  # noqa: E402
from test_admission import FakeEngine, scheduler, seq  # noqa: E402

GiB = 1 << 30


def hub():
    h = object.__new__(stats.Hub)  # (the constructor would start the NVML sampler)
    h.lock = stats.threading.Condition()
    h.local = stats.threading.local()
    h.info, h.memory, h.baseline_used, h.snap, h.snap_prev = {}, None, None, None, None
    h.rates, h.meta, h.samples, h.log, h.log_n = {}, {}, collections.deque(maxlen=10), collections.deque(maxlen=50), 0
    h.engine_state, h._counters, h.nvml, h.started = "ready", None, None, 0.0
    h.device, h.context_bytes, h.nvml_total = None, stats.CONTEXT_BYTES, None
    return h


REPORT = {"pools": [
    {"name": "weights", "host": False, "bytes": 18 * GiB, "items": [
        ("model.layers.0.linear_attn.in_proj_qkvz.weight", 8 * GiB), ("model.layers.3.self_attn.q_proj.weight", 2 * GiB),
        ("model.layers.0.mlp.down_proj.weight", 4 * GiB), ("lm_head.weight", 1 * GiB), ("draft fc", 3 * GiB)]},
    {"name": "embedding", "host": True, "bytes": 2 * GiB, "items": [("embed_tokens", 2 * GiB)]},
    {"name": "kv", "host": False, "bytes": 10 * GiB, "items": [("K | V pages (and parking areas)", 10 * GiB)]},
    {"name": "state", "host": False, "bytes": 1 * GiB, "items": [("ssm", GiB)]},
    {"name": "activations", "host": False, "bytes": GiB // 2, "items": [("x", GiB // 2)]},
    {"name": "state cache", "host": True, "bytes": 21 * (189 << 20), "items": [("state checkpoints", 21 * (189 << 20))]},
], "slots": 21, "checkpoint_bytes": 189 << 20}


def snapshot(t, seqs, free_pages=500, ckpt=200, device_free=1 * GiB):
    return {"t": t, "seqs": seqs, "running": len(seqs), "waiting": 0, "blocked": None, "target_seqs": 8,
            "cache": {"pool_pages": 1023, "free_pages": free_pages, "checkpoint_only_pages": ckpt, "park_pages": 100,
                      "checkpoints": 3, "prompt_tokens": 1000, "cached_tokens": 900, "slots_allocated": 5},
            "device": {"free": device_free, "total": 32 * GiB, "free_at_start": 31 * GiB, "budget": 30 * GiB},
            "generated": 0, "prefilled": 0}


def test_vram_parts_add_up_to_the_device():
    h = hub()
    h.set_memory(REPORT)
    h.baseline_used = GiB // 4
    h.set_snapshot(snapshot(1.0, [{"rid": "a", "state": "parked", "out": 0, "computed": 0}]))
    v = h.vram()
    parts = {p["name"]: p for p in v["parts"]}
    assert sum(p["bytes"] for p in v["parts"]) == 32 * GiB
    assert parts["weights"]["bytes"] == 18 * GiB and parts["KV cache"]["bytes"] == 10 * GiB
    assert {p["name"] for p in parts["weights"]["parts"]} >= {"GDN layers", "draft model (DFlash2)", "MLP"}
    kv = {p["name"]: p["bytes"] for p in parts["KV cache"]["parts"]}
    page = 10 * GiB / 1024
    assert kv["parked state"] == 100 * page and kv["prefix cache only"] == 200 * page
    assert kv["active sequences"] == (1023 - 500 - 200 - 100) * page and abs(sum(kv.values()) - 10 * GiB) < 1
    # Pools 29.5 GiB of the 31 free at the start, 1 GiB free now: 0.5 GiB of graphs and libraries.
    assert parts["CUDA context & runtime"]["parts"][1]["bytes"] == GiB // 2
    assert parts["other processes"]["bytes"] == GiB // 4
    host = {p["name"]: p for p in v["host"]}
    assert set(host) == {"embedding", "state cache"}
    assert host["state cache"]["bytes"] == 5 * (189 << 20) and host["state cache"]["planned"] == 21 * (189 << 20)


def test_rates_and_log_records():
    h = hub()
    h.local.meta = stats.request_meta("/v1/messages", {"User-Agent": "claude-cli"},
                                      {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi  there"}]}]})
    h.submitted("req-1")
    h.set_snapshot(snapshot(1.0, [{"rid": "req-1", "state": "running", "out": 10, "computed": 100}]))
    h.set_snapshot(snapshot(2.0, [{"rid": "req-1", "state": "running", "out": 110, "computed": 100}]))
    seqs = h.tick()["seqs"]
    assert seqs[0]["decode_now"] == 100 and seqs[0]["endpoint"] == "/v1/messages" and seqs[0]["preview"] == "hi there"
    h.finished("req-1", "stop", {"prompt_tokens": 1000, "completion_tokens": 201, "cached_tokens": 900, "ttft": 0.5,
                                 "total": 2.5, "reusable_tokens": 950, "computed_tokens": 100, "prefill_s": 0.1,
                                 "decode_s": 2.0, "steps": 50, "queued": 0.0, "preempted": 0})
    rec = h.log[-1]["rec"]
    assert rec["hit"] == 0.9 and rec["decode_tps"] == 100 and rec["accept"] == 201 / 50 and rec["prefill_tps"] == 1000
    assert rec["meta"]["client"] == "claude-cli" and "req-1" not in h.meta


def test_scheduler_stats_snapshot():
    eng = FakeEngine(free=1000)
    eng.st.update(pool_pages=2000)
    eng.device_memory = lambda: {"free": 0, "total": 0, "free_at_start": 0, "budget": 0}
    sch = scheduler(eng)
    sch.cur, sch.last_stats = None, 0.0
    sch.m = collections.Counter({"q27:generation_tokens_total": 5})
    a = seq(1, 300, started=True, fed=300, out=5)
    a.phase, a.cached, a.marks = worker.DECODE, 128, [10, 200, 290]
    b = seq(2, 50)
    sch.alive, sch.waiting = [a], collections.deque([b])
    snap = sch.stats()
    assert [s["state"] for s in snap["seqs"]] == ["admitted", "waiting"]
    assert snap["seqs"][0]["cached"] == 128 and snap["seqs"][0]["out"] == 5 and snap["seqs"][0]["marks"] == [10, 200, 290]
    assert snap["seqs"][1]["fed"] == 0 and snap["generated"] == 5
