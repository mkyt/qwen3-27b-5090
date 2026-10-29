"""Live stats of the server for the web UI (GET /ui): GPU samples (NVML), the engine's memory pools and
snapshots (worker.py "memory" / "stats"), and a log of the requests and engine events.

The hub keeps the last STATS_HISTORY_S seconds of samples and the last LOG_SIZE log entries in memory; GET
/ui/stream sends them as server-sent events (an "init" event, then a "tick" every second and the new log entries).
NVML is read through ctypes (libnvidia-ml.so.1, mounted by the NVIDIA container runtime; nvml.dll, installed with the
driver on Windows); without it the GPU fields are null.
"""

import collections
import ctypes
import os
import threading
import time

SAMPLE_S = float(os.environ.get("STATS_SAMPLE_S", 1.0))  # s between GPU samples (0: no sampling)
HISTORY_S = 600   # s of samples kept
LOG_SIZE = 500    # log entries kept
PREVIEW = 160     # characters of the last user message in a log entry
MiB = 1 << 20
CONTEXT_BYTES = int(1.21 * (1 << 30))  # an engine's CUDA context and runtime before its pools (RTX 5090, WSL2)


class Nvml:
    """The few NVML queries the UI shows (device 0)."""

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self.h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) != 0:
            raise OSError("no NVML device 0")
        name = ctypes.create_string_buffer(96)
        self.name = name.value.decode() if self.lib.nvmlDeviceGetName(self.h, name, 96) == 0 else "GPU"

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        return v.value if getattr(self.lib, fn)(self.h, *args, ctypes.byref(v)) == 0 else None

    def sample(self):
        u, m = self._Util(), self._Mem()
        util = self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0
        mem = self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0
        power, limit = self._uint("nvmlDeviceGetPowerUsage"), self._uint("nvmlDeviceGetEnforcedPowerLimit")
        return {"util": u.gpu if util else None, "mem_util": u.memory if util else None,
                "power": power / 1000 if power is not None else None, "power_limit": limit / 1000 if limit else None,
                "sm_clock": self._uint("nvmlDeviceGetClockInfo", 1), "mem_clock": self._uint("nvmlDeviceGetClockInfo", 2),
                "temp": self._uint("nvmlDeviceGetTemperature", 0),
                "used": m.used if mem else None, "total": m.total if mem else None}


def weight_group(what):
    """The part of the model a weight tensor (a reservation of the weights pool) belongs to."""
    if what.startswith("draft"):
        return "draft model (DFlash2)"
    if "visual" in what:
        return "vision tower"
    if "lm_head" in what:
        return "lm_head"
    if "embed" in what:
        return "embedding"
    if "linear_attn" in what:
        return "GDN layers"
    if "self_attn" in what:
        return "attention layers"
    if "mlp" in what:
        return "MLP"
    return "other"


def group_items(items, key, top=6):
    """Sums of (what, bytes) by key(what), largest first; the rest beyond `top` as one "other"."""
    sums = collections.Counter()
    for what, n in items:
        sums[key(what)] += n
    out = sums.most_common()
    if len(out) > top:
        out = out[:top - 1] + [("other", sum(n for _, n in out[top - 1:]))]
    return [{"name": k, "bytes": n} for k, n in out]


def request_meta(path, headers, body):
    """What the log shows of a request: endpoint, client, options and the start of the last user message."""
    msgs = body.get("messages") or body.get("input") or []
    last = ""
    if isinstance(msgs, str):
        last = msgs
    elif isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role", "user") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
                last = c if isinstance(c, str) else ""
                if last:
                    break
    elif isinstance(body.get("prompt"), str):
        last = body["prompt"]
    last = " ".join(last.split())
    return {"endpoint": path, "client": (headers.get("User-Agent") or "")[:80], "model": body.get("model"),
            "stream": bool(body.get("stream")), "messages": len(msgs) if isinstance(msgs, list) else 1,
            "tools": len(body.get("tools") or []),
            "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or body.get("max_output_tokens"),
            "preview": last[:PREVIEW] + ("…" if len(last) > PREVIEW else "")}


class Hub:
    def __init__(self):
        self.lock = threading.Condition()
        self.local = threading.local()      # the HTTP thread's request (request_meta), read by EngineClient.submit
        self.started = time.time()
        self.info = {}                      # the engine's "ready" info and the server configuration
        self.memory = None                  # the engine's pools (Engine.memory_report), grouped for the UI
        self.baseline_used = None           # NVML used bytes before the engine process started
        self.snap = None                    # the last engine snapshot (worker.Scheduler.stats)
        self.snap_prev = None
        self.rates = {}                     # rid -> recent decode / prefill tok/s (from successive snapshots)
        self.meta = {}                      # rid -> request_meta
        self.samples = collections.deque(maxlen=int(HISTORY_S / max(SAMPLE_S, 0.1)))
        self.log = collections.deque(maxlen=LOG_SIZE)
        self.log_n = 0
        self.engine_state = "loading"
        self.device = None                  # (free at the start, total) of the last engine (cudaMemGetInfo)
        self.context_bytes = CONTEXT_BYTES  # what an engine's CUDA context takes (NVML free before it - free at start)
        self._counters = None               # (t, generated, prefilled) at the last sample
        self.nvml = None
        self.nvml_total = None

    def start(self):
        """Starts the GPU sampler (server.main; not at import: the engine process imports server.py again)."""
        try:
            self.nvml = Nvml()
        except OSError:
            return
        if SAMPLE_S > 0:
            threading.Thread(target=self._sampler, daemon=True, name="stats-sampler").start()

    # ------------------------------------------------------------ inputs

    def engine_starting(self):
        with self.lock:
            self.engine_state = "loading"
            self.snap = self.snap_prev = None
            self.device = None
            if self.nvml:
                g = self.nvml.sample()
                self.baseline_used, self.nvml_total = g["used"], g["total"]
        self.add_log("engine", None, "engine process starting")

    def engine_ready(self, info, cfg):
        with self.lock:
            self.engine_state = "ready"
            self.info = dict(info, kv_dtype=cfg.get("kv_dtype"), model=cfg.get("weights") or os.path.basename(str(cfg.get("model"))),
                             draft=bool(cfg.get("draft")), prefill_chunk=cfg.get("prefill_chunk"),
                             gpu=self.nvml.name if self.nvml else None)
        self.add_log("engine", None, f"engine ready: {info}")

    def engine_stopped(self, text):
        with self.lock:
            self.engine_state = "restarting"
        self.add_log("engine", None, text, level="error")

    def engine_state_set(self, state, text):
        """stopping / stopped from the admin API (the engine's snapshot goes: its VRAM is free)."""
        with self.lock:
            self.engine_state = state
            if state == "stopped":
                self.snap = self.snap_prev = None
                self.rates.clear()
            self.lock.notify_all()
        self.add_log("engine", None, text)

    def set_memory(self, report):
        pools = []
        for p in report["pools"]:
            key = weight_group if p["name"] == "weights" else (lambda w: w)
            pools.append({"name": p["name"], "host": p["host"], "bytes": p["bytes"],
                          "parts": group_items(p["items"], key, top=8 if p["name"] == "weights" else 6)})
        with self.lock:
            self.memory = {"pools": pools, "slots": report["slots"], "checkpoint_bytes": report["checkpoint_bytes"]}

    def set_snapshot(self, snap):
        with self.lock:
            d = snap["device"]
            if self.device is None or self.device[0] != d["free_at_start"]:
                self.device = (d["free_at_start"], d["total"])
                if self.baseline_used is not None and self.nvml_total:
                    self.context_bytes = max(0, self.nvml_total - self.baseline_used - d["free_at_start"])
            prev = self.snap
            if prev is not None:
                dt = snap["t"] - prev["t"]
                old = {s["rid"]: s for s in prev["seqs"]}
                for s in snap["seqs"]:
                    o = old.get(s["rid"])
                    r = self.rates.setdefault(s["rid"], {"decode": 0.0, "prefill": 0.0})
                    if o is not None and dt > 0:
                        for k, field in (("decode", "out"), ("prefill", "computed")):
                            v = (s[field] - o[field]) / dt
                            r[k] = v if r[k] == 0 else 0.5 * r[k] + 0.5 * v
            live = {s["rid"] for s in snap["seqs"]}
            for rid in list(self.rates):
                if rid not in live:
                    del self.rates[rid]
            self.snap_prev, self.snap = prev, snap
            self.lock.notify_all()

    def submitted(self, rid):
        meta = getattr(self.local, "meta", None)
        if meta is not None:
            with self.lock:
                self.meta[rid] = meta
            self.add_log("submit", rid, meta.get("preview") or "", meta=meta)

    def finished(self, rid, reason, info):
        with self.lock:
            meta = self.meta.pop(rid, {})
        prompt, cached = info.get("prompt_tokens", 0), info.get("cached_tokens", 0)
        out, steps = info.get("completion_tokens", 0), info.get("steps", 0)
        ttft = info.get("ttft")
        decode_wall = info["total"] - ttft if ttft is not None else 0
        rec = dict(info, reason=reason, meta=meta,
                   hit=cached / prompt if prompt else None,
                   hit_of_reusable=cached / info["reusable_tokens"] if info.get("reusable_tokens") else None,
                   prefill_tps=info["computed_tokens"] / info["prefill_s"] if info.get("prefill_s") else None,
                   decode_tps=(out - 1) / decode_wall if out > 1 and decode_wall > 0 else None,
                   accept=out / steps if steps else None)
        level = "error" if reason == "error" else "warn" if reason == "cancelled" else "info"
        self.add_log("done", rid, reason, level=level, rec=rec)

    def failed(self, rid, msg):
        with self.lock:
            meta = self.meta.pop(rid, {})
        self.add_log("error", rid, msg, level="error", meta=meta)

    def event(self, rid, kind, text):
        self.add_log(kind, rid, text, level="error" if kind == "error" else "warn")

    def http_error(self, path, code, msg):
        self.add_log("http", None, f"{path}: {code} {msg}", level="warn" if code < 500 else "error")

    def add_log(self, kind, rid, text, level="info", **extra):
        with self.lock:
            self.log_n += 1
            self.log.append(dict(extra, n=self.log_n, t=time.time(), kind=kind, rid=rid, text=text, level=level))
            self.lock.notify_all()

    # ------------------------------------------------------------ GPU samples

    def _sampler(self):
        while True:
            t0 = time.time()
            try:
                g = self.nvml.sample()
            except OSError:
                g = {}
            with self.lock:
                snap = self.snap
                gen, pre = (snap["generated"], snap["prefilled"]) if snap else (0, 0)
                decode = prefill = 0.0
                if self._counters is not None and snap is not None:
                    t1, g1, p1 = self._counters
                    if t0 > t1 and gen >= g1 and pre >= p1:
                        decode, prefill = (gen - g1) / (t0 - t1), (pre - p1) / (t0 - t1)
                self._counters = (t0, gen, pre)
                self.samples.append(dict(g, t=t0, decode_tps=decode, prefill_tps=prefill,
                                         running=snap["running"] if snap else 0,
                                         waiting=snap["waiting"] if snap else 0))
                self.lock.notify_all()
            time.sleep(max(0.05, SAMPLE_S - (time.time() - t0)))

    # ------------------------------------------------------------ outputs

    def plan_device(self):
        """(free after the CUDA context, total) bytes an engine started now would see, and how it is known: "engine"
        (the running or last engine's cudaMemGetInfo, the same after a restart), "nvml" (free memory now minus an
        engine's CUDA context), or None without NVML."""
        with self.lock:
            if self.device is not None and self.engine_state != "stopped":
                return self.device, "engine"
            g = self.samples[-1] if self.samples else (self.nvml.sample() if self.nvml else None)
        if not g or g.get("used") is None:
            return (self.device, "engine") if self.device else (None, None)
        total = self.device[1] if self.device else g["total"]
        return (max(0, g["total"] - g["used"] - self.context_bytes), total), "nvml"

    def vram(self):
        """The device memory by owner (bytes; the parts add up to the device total)."""
        snap, mem = self.snap, self.memory
        if not snap or not mem:
            return None
        d, c = snap["device"], snap["cache"]
        pools = {p["name"]: p for p in mem["pools"] if not p["host"]}
        kv = pools.get("kv", {"bytes": 0})
        page = kv["bytes"] / (c["pool_pages"] + 1) if c["pool_pages"] else 0
        parked = sum(1 for s in snap["seqs"] if s["state"] == "parked")
        park = min(parked * c["park_pages"], c["pool_pages"])
        ckpt = c["checkpoint_only_pages"]
        free = c["free_pages"]
        active = max(0, c["pool_pages"] - free - ckpt - park)
        kv_parts = [{"name": "active sequences", "bytes": active * page}, {"name": "prefix cache only", "bytes": ckpt * page},
                    {"name": "parked state", "bytes": park * page}, {"name": "free", "bytes": (free + 1) * page}]
        dev_pools = sum(p["bytes"] for p in pools.values())
        runtime = max(0, d["free_at_start"] - d["free"] - dev_pools)  # CUDA graphs, libraries (since the start)
        before = d["total"] - d["free_at_start"]                       # CUDA context and other processes
        other = min(before, self.baseline_used) if self.baseline_used is not None else 0
        parts = [{"name": "weights", "bytes": pools.get("weights", {}).get("bytes", 0),
                  "parts": pools.get("weights", {}).get("parts", [])},
                 {"name": "KV cache", "bytes": kv["bytes"], "parts": kv_parts},
                 {"name": "sequence state", "bytes": pools.get("state", {}).get("bytes", 0),
                  "parts": pools.get("state", {}).get("parts", [])},
                 {"name": "activations", "bytes": pools.get("activations", {}).get("bytes", 0),
                  "parts": pools.get("activations", {}).get("parts", [])},
                 {"name": "CUDA context & runtime", "bytes": before - other + runtime,
                  "parts": [{"name": "context (at start)", "bytes": before - other},
                            {"name": "graphs & libraries", "bytes": runtime}]},
                 {"name": "other processes", "bytes": other},
                 {"name": "free", "bytes": d["free"]}]
        # (the state cache's slots are allocated as needed: show what is allocated)
        host = [dict(p, bytes=c.get("slots_allocated", 0) * mem["checkpoint_bytes"], planned=p["bytes"])
                if p["name"] == "state cache" else p for p in mem["pools"] if p["host"]]
        return {"total": d["total"], "budget": d["budget"], "free_at_start": d["free_at_start"], "parts": parts,
                "host": host, "kv_page_tokens": 64, "kv_page_bytes": page, "slots": mem["slots"],
                "checkpoints": c["checkpoints"], "checkpoint_bytes": mem["checkpoint_bytes"]}

    def tick(self):
        with self.lock:
            snap = self.snap
            seqs = []
            for s in snap["seqs"] if snap else []:
                r = self.rates.get(s["rid"], {})
                meta = self.meta.get(s["rid"], {})
                seqs.append(dict(s, decode_now=r.get("decode", 0.0), prefill_now=r.get("prefill", 0.0),
                                 endpoint=meta.get("endpoint"), client=meta.get("client"), preview=meta.get("preview")))
            cache = snap["cache"] if snap else None
            return {"t": time.time(), "engine": self.engine_state, "seqs": seqs, "vram": self.vram(), "cache": cache,
                    "sample": self.samples[-1] if self.samples else None,
                    "sched": {k: snap[k] for k in ("running", "waiting", "blocked", "target_seqs")} if snap else None}

    def init(self):
        with self.lock:
            return {"info": self.info, "started": self.started, "sample_s": SAMPLE_S, "history": list(self.samples),
                    "log": list(self.log), "tick": None}

    def log_since(self, n):
        with self.lock:
            return [e for e in self.log if e["n"] > n]

    def wait(self, timeout):
        with self.lock:
            self.lock.wait(timeout)


HUB = Hub()
