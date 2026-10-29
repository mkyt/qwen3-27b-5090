"""Start settings of the engine (the stats page, POST /admin/engine): the weights, KV cache type, speculative
decoding, vision tower, number of sequences, context limit and VRAM budget an engine process starts with, and what
each setting leaves for the context before the engine starts.

The plans come from the engine's own memory planner (q27lib.Engine.plan: the pools of the engine's constructor for a
device with the given free memory, without CUDA; a plan takes 10 to 80 ms), in a helper process: it sets the
environment the engine reads (VISION, MAX_SEQS, VRAM_BUDGET_GIB, ...) for each plan, which the server's own process
must not change while it starts engines (a spawned engine copies it).
"""

import json
import multiprocessing as mp
import os
import threading

from qwen27b.paths import DFLASH2, PROFILES, TARGET, downloaded_repos, engine_target, snapshot
from tokenizer import fingerprint

CHOICES = {"kv_dtype": ("nvfp4", "fp8", "bf16"), "spec": ("dflash2", "none"), "vision": ("host", "fp8", "bf16", "off")}
MAX_SEQS = 8                # (the engine's lanes of a batched step)
MAX_CONTEXT = 262144        # max_position_embeddings of the checkpoint
MIN_BUDGET_GIB = 4.0
# The values the page shows a plan for, besides the current one.
VARIANTS = {"max_seqs": tuple(range(1, MAX_SEQS + 1)), "max_tokens": (32768, 65536, 131072, 196608, 262144),
            "vram_budget_gib": (None, 20.0, 22.0, 24.0, 26.0, 28.0)}  # (gdnfp4 needs about 19 GiB without a KV pool)
LM_HEAD_ENV = os.environ.get("Q27_LM_HEAD_CKPT", "")  # (engine_target("quasar") sets it in this process)
GiB = 1 << 30


def margin_gib():
    return float(os.environ.get("WDDM_MARGIN_GIB", 0.6))


def auto_budget(free, total):
    """The engine's budget without VRAM_BUDGET_GIB (memory.cc budget_for), in bytes."""
    used = (total - free) / GiB
    return int(max(0.0, min(free / GiB - margin_gib(), 30.3 - used)) * GiB)


def model_dir(weights):
    """The model directory of a weight profile (or a directory); FileNotFoundError when it is not downloaded."""
    path = engine_target(weights)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{weights}: no config.json in {path}")
    return path


class Launch:
    """The settings the next engine start uses (changed with a start or restart from the admin API; kept until the
    service restarts) and their plans."""

    KEYS = ("weights", "kv_dtype", "spec", "vision", "max_seqs", "max_tokens", "vram_budget_gib")

    def __init__(self, base, settings, tokenizer=None):
        self.base = base            # the rest of the engine's configuration (prefill chunk, scheduling, ...)
        self.tokenizer = tokenizer  # fingerprint() of the server's tokenizer: weights with another one are refused
        self.lock = threading.Lock()
        self.plans_cache = {}
        self.planner = None
        self.extra_weights = () if settings["weights"] in PROFILES else (settings["weights"],)
        self.settings = self.validate(settings, check_budget=False)

    # ------------------------------------------------------------ settings

    def weights_list(self):
        """The profiles, the server's own weights, and the checkpoints of the model downloaded from other repositories
        (paths.downloaded_repos: "org/name" or "org/name@commit")."""
        listed = PROFILES + self.extra_weights
        return listed + tuple(w for w in downloaded_repos() if w not in listed)

    def weights_choices(self):
        """{weights: None if they can be loaded, else why not}."""
        out = {}
        for w in self.weights_list():
            try:
                path = model_dir(w)
                out[w] = None
                if self.tokenizer and fingerprint(path) != self.tokenizer:
                    out[w] = f"{w}: another tokenizer (vocabulary) than the server's (start the server with WEIGHTS={w})"
            except (FileNotFoundError, KeyError, ValueError) as e:
                out[w] = str(e)
        return out

    def validate(self, changes, base=None, device=None, check_budget=True):
        """The settings with `changes` applied; ValueError names a bad one."""
        s = dict(base or getattr(self, "settings", {}), **changes)
        unknown = set(changes) - set(self.KEYS)
        if unknown:
            raise ValueError(f"unknown setting {sorted(unknown)[0]!r} (settings: {', '.join(self.KEYS)})")
        if s["weights"] not in self.weights_list():
            raise ValueError(f"weights: one of {', '.join(self.weights_list())}")
        if self.tokenizer and s["weights"] != getattr(self, "settings", s)["weights"]:
            why = self.weights_choices()[s["weights"]]
            if why and "tokenizer" in why:
                raise ValueError(f"weights: {why}")
        for k, values in CHOICES.items():
            if s[k] not in values:
                raise ValueError(f"{k}: one of {', '.join(values)}")
        for k, lo, hi in (("max_seqs", 1, MAX_SEQS), ("max_tokens", 2048, MAX_CONTEXT)):
            v = s[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"{k}: an integer from {lo} to {hi}")
        b = s["vram_budget_gib"]
        if b is not None:
            if isinstance(b, bool) or not isinstance(b, (int, float)) or b < MIN_BUDGET_GIB:
                raise ValueError(f"vram_budget_gib: null (automatic) or at least {MIN_BUDGET_GIB:g} GiB")
            s["vram_budget_gib"] = round(float(b), 2)
            if check_budget and device is not None and s["vram_budget_gib"] * GiB > auto_budget(*device) + GiB // 100:
                raise ValueError(f"vram_budget_gib: at most the automatic budget, {auto_budget(*device) / GiB:.2f} GiB "
                                 f"(free memory minus {margin_gib():g} GiB, at most 30.3 GiB for the process)")
        return s

    def config(self, s):
        """The engine's configuration (worker.run) for settings s; FileNotFoundError when weights are missing."""
        path = model_dir(s["weights"])
        draft = snapshot(*DFLASH2) if s["spec"] == "dflash2" else None
        eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
        env = {"VISION": s["vision"], "MAX_SEQS": str(s["max_seqs"]),
               "VRAM_BUDGET_GIB": None if s["vram_budget_gib"] is None else f"{s['vram_budget_gib']:g}",
               "Q27_LM_HEAD_CKPT": str(snapshot(*TARGET)) if s["weights"] == "quasar" else LM_HEAD_ENV}
        return dict(self.base, model=str(path), weights=s["weights"], draft=str(draft) if draft else None,
                    max_tokens=s["max_tokens"], kv_dtype=s["kv_dtype"], max_seqs=s["max_seqs"],
                    eos=eos if isinstance(eos, list) else [eos], env=env, settings=dict(s))

    # ------------------------------------------------------------ plans

    def plan_many(self, settings_list, device):
        """The plans of several settings on a device with (free after the CUDA context, total) bytes: dicts of
        max_tokens (context per request), kv_tokens (the KV pool, shared by the requests), budget, device (bytes of
        the device pools), fits and pools, or {"error"}."""
        free, total = device
        out, jobs = [None] * len(settings_list), {}
        with self.lock:
            for i, s in enumerate(settings_list):
                try:
                    cfg = self.config(s)
                except FileNotFoundError as e:
                    out[i] = {"error": str(e)}
                    continue
                job = {"model": cfg["model"], "draft": cfg["draft"], "max_tokens": s["max_tokens"],
                       "prefill_chunk": cfg["prefill_chunk"], "kv_dtype": s["kv_dtype"], "env": cfg["env"],
                       "free": free, "total": total}
                key = json.dumps(job, sort_keys=True)
                if key in self.plans_cache:
                    out[i] = self.plans_cache[key]
                else:
                    jobs.setdefault(key, (job, []))[1].append(i)
            if jobs:
                results = self._run([job for job, _ in jobs.values()])
                if len(self.plans_cache) > 4096:
                    self.plans_cache.clear()
                for (key, (_, idx)), r in zip(jobs.items(), results):
                    self.plans_cache[key] = r
                    for i in idx:
                        out[i] = r
        return out

    def plans(self, s, device):
        """The plan of settings s and, for each setting, the plans with one value changed (the page's choices)."""
        variants = {"weights": self.weights_list(), **CHOICES, **VARIANTS}
        listed = [s]
        for k, values in variants.items():
            listed += [dict(s, **{k: v}) for v in values]
        res = self.plan_many(listed, device)
        out, i = {"settings": s, "plan": res[0], "variants": {}}, 1
        for k, values in variants.items():
            out["variants"][k] = [{"value": v, "plan": r} for v, r in zip(values, res[i:i + len(values)])]
            i += len(values)
        return out

    def _run(self, jobs):
        if self.planner is None or not self.planner[0].is_alive():
            ctx = mp.get_context("spawn")
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=planner, args=(child, self.base["engine_python"]), daemon=True, name="q27-planner")
            proc.start()
            child.close()
            self.planner = (proc, parent)
        try:
            self.planner[1].send(jobs)
            return self.planner[1].recv()
        except (EOFError, OSError) as e:
            self.planner = None
            return [{"error": f"the planner process failed: {e}"}] * len(jobs)


def planner(conn, engine_python):
    """Process entry point (multiprocessing, spawn): answers lists of plan jobs with lists of plans."""
    import sys

    sys.path.insert(0, engine_python)
    from q27lib import Engine

    devnull = os.open(os.devnull, os.O_WRONLY)  # (the engine logs every plan; errors come back in the answers)
    os.dup2(devnull, 2)

    base = dict(os.environ)
    while True:
        try:
            jobs = conn.recv()
        except EOFError:
            return
        out = []
        for job in jobs:
            os.environ.clear()
            os.environ.update(base)
            for k, v in job["env"].items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            try:
                e = Engine.plan(job["model"], job["free"], job["total"], max_tokens=job["max_tokens"],
                                prefill_chunk=job["prefill_chunk"], kv_dtype=job["kv_dtype"], draft=job["draft"])
            except RuntimeError as err:
                out.append({"error": str(err).removeprefix("q27_plan: ")})
                continue
            try:
                pools = e.memory_report()["pools"]
                c, d = e.cache_stats(), e.device_memory()
                dev = {p["name"]: p["bytes"] for p in pools if not p["host"]}
                out.append({"max_tokens": e.max_tokens, "kv_tokens": c["pool_pages"] * 64, "budget": d["budget"],
                            "device": sum(dev.values()), "fits": sum(dev.values()) <= d["budget"], "pools": dev,
                            "host": sum(p["bytes"] for p in pools if p["host"])})
            finally:
                e.close()
        conn.send(out)
