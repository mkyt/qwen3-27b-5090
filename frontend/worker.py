"""Engine process (Phase 6): owns libq27 and schedules the requests (docs/architecture.md §10).

The API server (server.py) starts this in its own process and talks to it over a multiprocessing pipe,
so a crash of the engine (a CUDA error) leaves the server up: it answers 503 and starts a new engine.

Scheduling: up to MAX_SEQS requests run at once (the rest wait in arrival order). They take turns on the
GPU: the engine parks the running sequence's state in a device area and resumes another one (~0.5 ms,
Model::seq_swap), so a turn is a prefill chunk or a few decode steps. Prefills go first (shortest rest
first: a short warm turn of an agent is not stuck behind a long cold prompt), but each second of prefill
gives the decoding requests DECODE_SHARE seconds, so their output does not stop during a long prompt.
Decoding requests take turns of SCHED_QUANTUM_MS, or with BATCH_DECODE (P6-10, speculative decoding) step
together: one batched step drafts and verifies all of them (Model::spec_step_lanes; each gets the tokens
it would get alone), so the GEMMs read the weights once for up to MAX_SEQS requests. When the KV pool runs
out, the latest request (the one that ran out, if it is the latest) is suspended to the prefix cache (its
state becomes a checkpoint, its pages go back) and goes back to the head of the queue.

Admission (P7-14): the engine has room for MAX_SEQS requests (parking areas, lanes); how many run is decided
while it runs. A waiting request starts only while fewer than the target (max_seqs, MAX_SEQS at the start,
set by /admin/concurrency) run, the KV pool has room for it (ADMIT_KV: free pages and pages only checkpoints
hold cover the rest of every running prompt, KV_RESERVE_TOKENS of output for each request, and one more
parking area), and one more request would not slow each one below MIN_SEQ_TPS (measured tok/s per request
for each number of requests decoding together). With nothing running, the first waiting request always
starts. A request waiting for room rather than one suspended later keeps its turn and its KV.

Messages (pickled tuples). In: ("submit", rid, request), ("cancel", rid), ("config", changes, n), ("stop",).
Out: ("ready", info), ("start", rid, info), ("tok", rid, [ids]), ("done", rid, finish, info),
("error", rid, message), ("metrics", dict), ("config", settings, n: the change answered, or None),
("memory", the pools: once after "ready"), ("stats", snapshot: the requests, KV pool and device memory, every STATS_INTERVAL s while busy),
("event", rid, kind, text: an admission wait, a preemption, an engine error).
"""

import array
import collections
import os
import random
import sys
import time
import traceback

PREFILL, DECODE = "prefill", "decode"
PAGE = 64  # tokens of a KV page (kernels.h kPageTokens)
ADMIT_RECHECK = 0.05  # s between checks of a request kept waiting by the admission
RATE_WINDOW = 10.0    # s of decoding the measured tok/s per request averages over
STATS_INTERVAL = 0.5  # s between stats snapshots (the stats UI) while requests run
STATS_IDLE = 5.0      # s between them while idle


def step_ms(n):
    """Time of a batched step for n requests (short contexts, gdnfp4 + KV nvfp4, P7-11: 12.40 / 13.76 / 16.39 /
    21.99 ms for 1 / 2 / 4 / 8): scales a measured rate per request to a number of requests not measured yet."""
    return 11.1 + 1.36 * n


def log(msg):
    print(f"[sched] {msg}", file=sys.stderr, flush=True)


class Seq:
    """A request: its prompt, sampling settings, progress and place in the engine."""

    def __init__(self, rid, req, now):
        self.rid = rid
        self.prompt = list(req["ids"])
        self.max_new = req["max_new"]
        self.temperature, self.top_k, self.top_p = req["temperature"], req["top_k"], req["top_p"]
        self.rng = random.Random(req.get("seed"))
        self.images = req.get("images")  # (spans [(start, gh, gw)], pixel arrays) or None
        self.arrival = now
        self.out = []            # generated tokens (EOS excluded)
        self.eng_ids = self.prompt  # what the engine runs (prompt + earlier output after a preemption)
        self.started = False     # prefix_begin done (the engine holds its state: running or parked)
        self.area = None         # parking area while parked
        self.phase = PREFILL
        self.fed = 0             # tokens of eng_ids the engine has run
        self.est_cached = 0      # expected cache hit (to order prefills before they start)
        self.at = []             # checkpoint positions of the prompt
        self.cached = None       # tokens restored by the first prefix_begin
        self.reusable = 0
        self.last_run = 0.0
        self.first_token = None
        self.preempted = 0
        self.dead = False
        self.share_with, self.share_len = None, 0  # an earlier request still computing a prefix this one shares
        self.held = False        # kept waiting by the admission (counted once)
        self._bytes = None
        self.begun = None        # time of the first prefix_begin
        self.computed = 0        # prompt tokens prefilled (not restored from the cache)
        self.prefill_s = self.decode_s = 0.0  # time in its prefill turns / decode turns (a batched turn counts for each)
        self.steps = 0           # decode steps (speculative: verify steps)
        self.marks = [0, 0, 0]   # end of the first message, starts of the last message and the generation prompt

    def rest(self):
        return len(self.eng_ids) - (self.fed if self.started else self.est_cached)

    def ids_bytes(self):
        if self._bytes is None or len(self._bytes) != 4 * len(self.eng_ids):
            self._bytes = array.array("i", self.eng_ids).tobytes()
        return self._bytes


class Scheduler:
    def __init__(self, conn, cfg):
        from q27lib import Engine, KvFull

        self.conn, self.cfg, self.KvFull = conn, cfg, KvFull
        os.environ.update(cfg.get("env", {}))
        self.engine = Engine(cfg["model"], max_tokens=cfg["max_tokens"], prefill_chunk=cfg["prefill_chunk"],
                             kv_dtype=cfg["kv_dtype"], draft=cfg.get("draft"))
        self.spec = cfg.get("draft") is not None and self.engine.has_draft
        self.engine_seqs = max(1, min(cfg["max_seqs"], self.engine.park_slots + 1))
        self.max_seqs = self.engine_seqs  # the target number of running requests (<= engine_seqs)
        self.admit_kv, self.kv_reserve, self.min_tps = True, 4096, 0.0
        self.configure({k: cfg[k] for k in ("target_seqs", "admit_kv", "kv_reserve", "min_seq_tps") if k in cfg})
        self.rates = {}          # n -> [tokens per request, seconds] with n requests decoding together (decayed sums)
        self.blocked = None      # why the first waiting request waits ("kv", "speed") or None
        self.held_until = (None, 0.0)  # (request, time): skip checking it again until then
        self.batch = self.spec and cfg.get("batch_decode", True) and self.engine.max_lanes > 1
        self.chunk = cfg["prefill_chunk"]
        self.share = cfg["decode_share"]
        self.quantum = cfg["quantum_ms"] / 1000.0
        self.eos = set(cfg["eos"])
        self.im_start = cfg["im_start"]
        self.waiting = collections.deque()
        self.alive = []          # admitted, unfinished
        self.by_id = {}
        self.cur = None          # the sequence in the engine's working state (may be finished)
        self.credit = 0.0        # decode time owed to decoding requests by the prefills (DECODE_SHARE)
        self.history = collections.deque(maxlen=64)  # recent sequences (prompt + output) as int32 bytes
        self.m = collections.Counter()
        self.last_metrics = self.last_stats = 0.0
        self.cache_full = False  # every state slot of the prefix cache allocated (poll)

    # ------------------------------------------------------------ messages

    def send(self, *msg):
        self.conn.send(msg)

    def poll(self, block):
        # Idle: allocate the prefix cache's state slots one by one (not at startup, so the engine is ready sooner),
        # then wake up now and then for a stats snapshot.
        if block and not self.cache_full:
            self.cache_full = self.engine.cache_grow(1) == 0
            block = self.cache_full
        if block:
            self.conn.poll(STATS_IDLE)
        while self.conn.poll(0):
            msg = self.conn.recv()
            if msg[0] == "submit":
                self.submit(msg[1], msg[2])
            elif msg[0] == "cancel":
                self.cancel(msg[1])
            elif msg[0] == "config":
                self.configure(msg[1])
                self.send("config", self.settings(), msg[2])
            elif msg[0] == "stop":
                raise SystemExit(0)

    def submit(self, rid, req):
        s = Seq(rid, req, time.time())
        s.reusable = self.reusable(s.prompt)
        # Requests that arrive together (an agent and its sub-agents) often share a long prefix (the system
        # prompt and tools) that no checkpoint holds yet: the later one waits until the earlier one has run
        # past it (with a checkpoint planned there) instead of computing it again.
        p = array.array("i", s.prompt).tobytes()
        for o in list(self.alive) + list(self.waiting):
            n = lcp_bytes(p, array.array("i", o.eng_ids).tobytes())
            n = min(n, len(s.prompt) - 1) // 64 * 64
            if n > max(s.reusable, s.share_len) + 256 and not (o.started and o.fed >= n):
                s.share_with, s.share_len = o, n
        if s.share_with is not None:
            o = s.share_with
            if not o.started:
                o.at = sorted(set(o.at) | {s.share_len})
            s.reusable = s.share_len
        s.at = self.checkpoints(s.prompt, s.reusable)
        # (stats: where the first message ends, the last message and the generation prompt start)
        starts = [i for i, t in enumerate(s.prompt) if t == self.im_start]
        s.marks = (starts[1:2] or [0]) + (starts[-2:-1] or [0]) + (starts[-1:] or [0])
        s.est_cached = s.reusable
        self.by_id[rid] = s
        self.waiting.append(s)
        self.m["q27:requests_total"] += 1

    def cancel(self, rid):
        s = self.by_id.get(rid)
        if s is None:
            return
        if s in self.waiting:
            self.waiting.remove(s)
        self.finish(s, "cancelled", save=False)

    def configure(self, c):
        """Admission settings (cfg at the start, then /admin/concurrency); values are clamped to valid ones."""
        if c.get("target_seqs") is not None:
            self.max_seqs = max(1, min(int(c["target_seqs"]), self.engine_seqs))
        if c.get("admit_kv") is not None:
            self.admit_kv = bool(c["admit_kv"])
        if c.get("kv_reserve") is not None:
            self.kv_reserve = max(0, int(c["kv_reserve"]))
        if c.get("min_seq_tps") is not None:
            self.min_tps = max(0.0, float(c["min_seq_tps"]))

    def settings(self):
        return {"target_seqs": self.max_seqs, "engine_seqs": self.engine_seqs, "admit_kv": self.admit_kv,
                "kv_reserve": self.kv_reserve, "min_seq_tps": self.min_tps,
                "rates": {n: round(self.rate(n), 1) for n in sorted(self.rates)},
                "running": len(self.alive), "waiting": len(self.waiting), "blocked": self.blocked}

    # ------------------------------------------------------------ admission (P7-14)

    def admit(self, s):
        """Whether the first waiting request may start now (below the target number of requests)."""
        held, until = self.held_until
        if held is s and time.time() < until:
            return False
        reason = None
        if self.alive:
            # (a suspended request waits for room even without ADMIT_KV: let in at once, it would run out again)
            if (self.admit_kv or s.preempted) and not self.kv_room(s):
                reason = "kv"
            elif self.min_tps > 0 and self.predicted_rate(len(self.alive) + 1) < self.min_tps:
                reason = "speed"
        if reason and not s.held:
            s.held = True
            self.m[f'q27:admission_waits_total{{reason="{reason}"}}'] += 1
            log(f"{s.rid}: waits for room ({reason}), {len(self.alive)} running")
            self.send("event", s.rid, "wait", f"waits for room ({reason}), {len(self.alive)} running")
        self.blocked = reason
        self.held_until = (s, time.time() + ADMIT_RECHECK) if reason else (None, 0.0)
        return reason is None

    def live_prefix(self, s):
        """Tokens of s's prompt whose KV pages a running request holds already (a shared prefix)."""
        p = s.ids_bytes()
        n = max((lcp_bytes(p, o.ids_bytes()[:4 * o.fed]) for o in self.alive if o.started and o is not s), default=0)
        return n // PAGE * PAGE

    def kv_room(self, s):
        """Whether the KV pool holds s next to the running requests: the free pages and the pages only
        checkpoints hold (evicted when needed) cover the rest of each prompt, kv_reserve tokens of output of
        each request, and the parking areas of one more request."""
        st = self.engine.cache_stats()
        room = st["free_pages"] + st["checkpoint_only_pages"]
        parked = sum(1 for o in self.alive if o.area is not None)
        room -= max(0, len(self.alive) - parked) * st["park_pages"]  # (n + 1 requests park n at most)
        need = 0
        for o in self.alive + [s]:
            prompt = len(o.eng_ids) - (o.fed if o.started else self.live_prefix(o))
            out = min(max(0, o.max_new - len(o.out)), self.kv_reserve)
            need += -(-max(0, prompt) // PAGE) + -(-out // PAGE)
        return need <= room

    def predicted_rate(self, n):
        """tok/s per request with n requests decoding together: measured, or scaled from the largest measured
        number below n by the step time; no limit before anything was measured."""
        if n in self.rates:
            return self.rate(n)
        below = [m for m in self.rates if m < n]
        if not below:
            return float("inf")
        m = max(below)
        return self.rate(m) * step_ms(m) / step_ms(n)

    def rate(self, n):
        tokens, seconds = self.rates[n]
        return tokens / seconds

    def measure(self, n, tokens, dt):
        """A decode turn of n requests: tokens in all, dt seconds. The sums decay with a time constant of about
        RATE_WINDOW seconds of decoding with n requests, so a slow turn (a graph capture) weighs by its time."""
        if dt <= 0:
            return
        k = max(0.0, 1.0 - dt / RATE_WINDOW)
        t, sec = self.rates.get(n, (0.0, 0.0))
        self.rates[n] = [t * k + tokens / n, sec * k + dt]

    # ------------------------------------------------------------ prefix cache helpers

    def reusable(self, prompt):
        """Tokens of the prompt an ideal prefix cache could reuse: the longest common prefix with a recent
        prompt + output (the rest is new to the server). cached / reusable measures the cache itself."""
        p = array.array("i", prompt).tobytes()
        best = 0
        for h in self.history:
            lo, hi = best, min(len(p), len(h)) // 4  # binary search on memcmp'd slices
            if p[:4 * lo] != h[:4 * lo]:
                continue
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if p[4 * lo:4 * mid] == h[4 * lo:4 * mid] else (lo, mid - 1)
            best = lo
        return max(0, min(best, len(prompt) - 1))  # (the engine always runs the last prompt token)

    def checkpoints(self, prompt, reusable):
        """Checkpoints of a prompt (docs/architecture.md §9): the end of the first message (the shared system prompt
        and tools), the starts of the last message and of the generation prompt (a retried request, or a
        client that rewrites the last turn), and where the prompt leaves the longest earlier sequence (a
        prefix shared inside a message). The end of the generation is saved after it."""
        starts = [i for i, t in enumerate(prompt) if t == self.im_start]
        at = set(starts[1:2] + starts[-2:])
        if reusable > 0:
            at.add(reusable)
        return sorted(at)

    # ------------------------------------------------------------ scheduling

    def run(self):
        self.send("ready", {"max_tokens": self.engine.max_tokens, "max_seqs": self.engine_seqs, "spec": self.spec,
                            "vision": self.engine.has_vision})
        self.send("memory", self.engine.memory_report())
        self.send("config", self.settings(), None)
        log(f"{self.engine_seqs} sequences at once (target {self.max_seqs}, KV admission {'on' if self.admit_kv else 'off'}, "
            f"min {self.min_tps:g} tok/s per request), decode share {self.share}, quantum {self.quantum * 1000:.0f} ms, "
            f"{'DFlash2' if self.spec else 'no speculative decoding'}, "
            f"{'batched decode steps' if self.batch else 'decode in turns'}")
        while True:
            self.poll(block=not self.alive and not self.waiting)
            if not self.waiting or self.held_until[0] is not self.waiting[0]:
                self.blocked = None
            while self.waiting and len(self.alive) < self.max_seqs and self.admit(self.waiting[0]):
                self.alive.append(self.waiting.popleft())
            job = self.pick()
            if job:
                self.guarded(*job)
            # (stats also once more when the work is done, and after each idle wait)
            self.report(force=not (self.alive or self.waiting))

    def report(self, force=False):
        """Metrics and admission settings every 2 s, stats snapshots every STATS_INTERVAL (also from inside a
        long decode turn: one request alone decodes until it ends or a message arrives)."""
        now = time.time()
        if now - self.last_metrics > 2.0:
            self.send("metrics", self.metrics())
            self.send("config", self.settings(), None)
        if force or now - self.last_stats > STATS_INTERVAL:
            self.send("stats", self.stats())

    def can_prefill(self, s):
        o = s.share_with
        if not s.started and o is not None and not o.dead and o.phase == PREFILL and not (o.started and o.fed >= s.share_len):
            return False
        # Image embeddings of a request live in one shared pinned table until its prompt has run, so only one
        # request with images may be part-way through its prompt.
        if s.images and not s.started:
            return not any(o.images and o.started and o.phase == PREFILL for o in self.alive)
        return True

    def pick(self):
        prefills = [s for s in self.alive if s.phase == PREFILL and self.can_prefill(s)]
        decodes = [s for s in self.alive if s.phase == DECODE]
        if not prefills:
            self.credit = 0.0
        if prefills and (not decodes or self.credit <= 0):
            return ("prefill", min(prefills, key=lambda s: (s.rest(), s.arrival)))
        if self.batch and len(decodes) >= 2:
            return ("batch", decodes)
        if decodes:
            return ("decode", min(decodes, key=lambda s: s.last_run))
        return None

    def guarded(self, kind, s):
        """Runs a turn; KV pool full: preempts another request and retries; engine errors end the request."""
        t0 = time.time()
        try:
            return self._guarded(kind, s)
        finally:
            self.m["q27:engine_busy_seconds_total"] += time.time() - t0

    def _guarded(self, kind, s):
        while True:
            if kind == "batch":  # (s: the decoding requests; a retry after a preemption keeps the rest)
                s = [o for o in s if o.phase == DECODE and not o.dead]
                if len(s) < 2:
                    if not s:
                        return None
                    kind, s = "decode", s[0]
            try:
                if kind == "batch":
                    return self.batch_turn(s)
                return self.prefill_turn(s) if kind == "prefill" else self.decode_turn(s)
            except self.KvFull:
                if kind != "batch" and s.started and s is self.cur:
                    s.fed = self.engine.pos  # (a prompt may have run part of the way)
                # The latest request gives way, also when it is the one that ran out: suspending an earlier
                # one instead lets the two take turns suspending each other (P7-14).
                started = [o for o in self.alive if o.started and not o.dead]
                victim = max(started, key=lambda o: o.arrival, default=None)
                if victim is None or (kind != "batch" and started == [s]):
                    log(f"{s.rid}: the KV pool cannot hold it ({self.engine.pos} tokens)")
                    self.finish(s, "length")
                    return None
                self.suspend(victim)
                if victim is s:
                    return None
            except RuntimeError as e:
                msg = str(e)
                if "check failed" not in msg:  # a CUDA error: the context is unusable, restart the process
                    raise
                for o in s if kind == "batch" else [s]:
                    log(f"{o.rid}: {msg}")
                    self.send("event", o.rid, "error", msg)
                    self.send("error", o.rid, msg)
                    self.finish(o, "error", save=False, notify=False)
                self.engine.reset()
                self.cur = None
                return None

    def load(self, s):
        """Makes s the engine's running sequence (parks the running one, if it is still alive)."""
        if self.cur is s:
            return
        cur = self.cur
        t0 = time.time()
        if cur is not None and not cur.dead and cur.started:
            if s.area is not None:
                a = s.area
                self.engine.seq_swap(a)
                cur.area, s.area = a, None
            else:
                cur.area = self.engine.seq_park()
            self.m["q27:sequence_switches_total"] += 1
            self.m["q27:switch_seconds_total"] += time.time() - t0
        elif s.area is not None:
            self.engine.seq_unpark(s.area)
            s.area = None
        self.cur = s
        if not s.started:
            self.begin(s)

    def begin(self, s):
        """prefix_begin: restores the deepest cached prefix (the engine's state is reset first)."""
        self.cur = None  # (prefix_begin drops whatever the engine held)
        cached = self.engine.prefix_begin(s.eng_ids, at=s.at, spec=self.spec)
        self.cur = s
        s.started, s.fed = True, cached
        if s.cached is None:
            s.cached, s.begun = cached, time.time()
            self.m["q27:prompt_tokens_total"] += len(s.prompt)
            self.m["q27:cached_tokens_total"] += cached
            self.m["q27:reusable_tokens_total"] += s.reusable
            log(f"{s.rid}: prompt {len(s.prompt)} tokens, cached {cached} of {s.reusable} reusable, "
                f"waited {time.time() - s.arrival:.2f} s, checkpoints at {s.at}")
            self.send("start", s.rid, {"cached": cached})
        if s.images:
            spans, pixels = s.images
            self.engine.mm_set(spans)
            t0, encoded = time.time(), 0
            for i, (start, gh, gw) in enumerate(spans):
                if start + gh * gw > cached:  # (images inside the restored prefix need no embeddings)
                    self.engine.mm_encode(i, pixels[i])
                    encoded += 1
            log(f"{s.rid}: images {len(spans)} ({sum(gh * gw for _, gh, gw in spans)} tokens), encoded {encoded} "
                f"in {time.time() - t0:.3f} s")

    def prefill_turn(self, s):
        self.load(s)
        t0, fed0 = time.time(), s.fed
        rest = s.eng_ids[s.fed:]
        if len(rest) > self.chunk:
            s.fed += self.engine.prefill(rest, self.chunk, spec=self.spec)
        else:
            left = s.max_new - len(s.out)
            seed = s.rng.getrandbits(63)
            if self.spec:
                tok = self.engine.spec_begin(rest, s.temperature, s.top_k, s.top_p, seed=seed,
                                             uniform=s.rng.random(), max_new=left)
            else:
                tok = self.engine.forward_sample(rest, s.temperature, s.top_k, s.top_p, s.rng.random())
            s.fed = len(s.eng_ids)
            s.phase = DECODE
            if s.first_token is None:
                s.first_token = time.time()
            self.emit(s, [tok])
        dt = time.time() - t0
        s.last_run = time.time()
        s.prefill_s += dt
        s.computed += s.fed - fed0
        self.m["q27:prefill_tokens_total"] += s.fed - fed0
        if any(o.phase == DECODE for o in self.alive if o is not s):
            self.credit += self.share * dt

    def decode_turn(self, s):
        self.load(s)
        t0 = time.time()
        others = sum(1 for o in self.alive if o is not s) > 0
        has_prefill = any(o.phase == PREFILL for o in self.alive)
        turn = min(self.quantum, max(self.credit, 0.0)) if has_prefill else self.quantum
        step, n0 = 0.0, len(s.out)
        while not s.dead:
            now = time.time()
            last = (others and now + step >= t0 + turn) or self.conn.poll(0)
            ts = time.time()
            if self.spec:
                toks = self.engine.spec_step(ahead=not last)
                self.m["q27:spec_verify_calls_total"] += 1
            else:
                toks = [self.engine.forward_sample([s.out[-1]], s.temperature, s.top_k, s.top_p, s.rng.random())]
            step = max(step * 0.8, time.time() - ts)
            s.steps += 1
            self.emit(s, toks)
            if last:
                break
            self.report()
        dt = time.time() - t0
        s.last_run = time.time()
        s.decode_s += dt
        if not others and self.spec:
            self.measure(1, len(s.out) - n0, dt)
        if has_prefill:
            self.credit -= dt

    def batch_turn(self, seqs):
        """Batched speculative steps of the decoding requests (each one parked or the running one) until the
        turn ends, a request finishes, or a message arrives."""
        t0 = time.time()
        has_prefill = any(o.phase == PREFILL for o in self.alive)
        turn = min(self.quantum, max(self.credit, 0.0)) if has_prefill else self.quantum
        lanes = [None if s is self.cur else s.area for s in seqs]
        n0 = sum(len(s.out) for s in seqs)
        while True:
            new = self.engine.spec_step_lanes(lanes)
            self.m["q27:spec_verify_calls_total"] += len(seqs)
            self.m["q27:batched_steps_total"] += 1
            for s, toks in zip(seqs, new):
                s.steps += 1
                self.emit(s, toks)
            if any(s.dead for s in seqs) or time.time() >= t0 + turn or self.conn.poll(0):
                break
        dt = time.time() - t0
        self.measure(len(seqs), sum(len(s.out) for s in seqs) - n0, dt)
        for s in seqs:
            s.last_run = time.time()
            s.decode_s += dt
        if has_prefill:
            self.credit -= dt

    def emit(self, s, toks):
        """Tokens of a step: stops at EOS and at max_new; the rest goes to the server."""
        out, finish = [], None
        for t in toks:
            if t in self.eos:
                finish = "stop"
                break
            out.append(t)
            if len(s.out) + len(out) >= s.max_new:
                finish = "length"
                break
        if out:
            s.out += out
            self.m["q27:generation_tokens_total"] += len(out)
            self.send("tok", s.rid, out)
        if finish:
            self.finish(s, finish)

    def suspend(self, v):
        """Preemption (KV pool full): v's state becomes a prefix cache checkpoint and its pages go back; it
        waits at the head of the queue (the admission lets it in again) as a prompt of its prompt + output."""
        saved = len(v.eng_ids) if v.phase == DECODE else v.fed
        log(f"{v.rid}: preempted at {saved} tokens (KV pool full)")
        self.send("event", v.rid, "preempt", f"preempted at {saved} tokens (KV pool full)")
        if v is self.cur:
            self.engine.checkpoint()
            self.engine.reset()
            self.cur = None
        elif v.area is not None:
            self.engine.seq_suspend(v.area)
            v.area = None
        v.eng_ids = v.prompt + v.out  # (ids_bytes follows the new length)
        v.started, v.fed, v.phase = False, 0, PREFILL
        v.est_cached = min(saved, len(v.eng_ids) - 1)  # (the checkpoint it resumes from)
        v.preempted += 1
        self.m["q27:preemptions_total"] += 1
        self.alive.remove(v)
        self.waiting.appendleft(v)
        v.held = False

    def finish(self, s, reason, save=True, notify=True):
        if s.dead:
            return
        s.dead = True
        if s in self.alive:
            self.alive.remove(s)
        self.by_id.pop(s.rid, None)
        # The end of the generation: the next turn resumes here when it re-renders this output as is (a parked
        # request, e.g. one of a batched step, is saved from its parking area).
        end = save and reason in ("stop", "length")
        if s.area is not None:
            (self.engine.seq_suspend if end else self.engine.seq_drop)(s.area)
            s.area = None
        elif s is self.cur and end:
            self.engine.checkpoint()
        if s.started or s.out:
            self.history.append(array.array("i", s.prompt + s.out).tobytes())
        now = time.time()
        info = {"prompt_tokens": len(s.prompt), "completion_tokens": len(s.out), "cached_tokens": s.cached or 0,
                "ttft": (s.first_token - s.arrival) if s.first_token else None, "total": now - s.arrival,
                "preempted": s.preempted, "reusable_tokens": s.reusable, "computed_tokens": s.computed,
                "queued": (s.begun - s.arrival) if s.begun else None, "prefill_s": s.prefill_s,
                "decode_s": s.decode_s, "steps": s.steps}
        self.m[f'q27:requests_finished_total{{reason="{reason}"}}'] += 1
        if notify:
            self.send("done", s.rid, reason, dict(info, metrics=self.metrics()))
        if s.first_token and len(s.out) > 1:
            rate = (len(s.out) - 1) / max(now - s.first_token, 1e-6)
            log(f"{s.rid}: {reason}, {len(s.out)} tokens, ttft {info['ttft']:.2f} s, {rate:.1f} tok/s"
                + (f", preempted {s.preempted}x" if s.preempted else ""))

    def metrics(self):
        self.last_metrics = time.time()
        st = self.engine.cache_stats()
        out = dict(self.m)
        out.update({"q27:num_requests_running": len(self.alive), "q27:num_requests_waiting": len(self.waiting),
                    "q27:target_seqs": self.max_seqs, "q27:engine_seqs": self.engine_seqs,
                    'q27:admission_blocked{reason="kv"}': int(self.blocked == "kv"),
                    'q27:admission_blocked{reason="speed"}': int(self.blocked == "speed"),
                    "q27:kv_free_pages": st["free_pages"], "q27:kv_checkpoint_only_pages": st["checkpoint_only_pages"],
                    "q27:prefix_cache_checkpoints": st["checkpoints"], "q27:prefix_cache_evictions_total": st["evicted"]})
        return out


    def stats(self):
        """Snapshot for the stats UI: each request (where its context stands), the KV pool, device memory."""
        self.last_stats = now = time.time()
        st = self.engine.cache_stats()
        seqs = []
        for s in self.alive + list(self.waiting):
            if s.dead:
                continue
            state = "waiting" if s not in self.alive else ("running" if s is self.cur else
                                                          "parked" if s.area is not None else "admitted")
            seqs.append({"rid": s.rid, "state": state, "phase": s.phase, "prompt": len(s.prompt),
                         "cached": s.cached if s.cached is not None else s.est_cached,
                         "fed": min(s.fed, len(s.prompt)) if s.started else 0, "out": len(s.out),
                         "max_new": s.max_new, "computed": s.computed, "reusable": s.reusable,
                         "age": now - s.arrival, "queued": (s.begun or now) - s.arrival,
                         "ttft": (s.first_token - s.arrival) if s.first_token else None,
                         "decode_wall": (now - s.first_token) if s.first_token else 0.0,
                         "prefill_s": s.prefill_s, "decode_s": s.decode_s, "steps": s.steps,
                         "preempted": s.preempted, "held": s.held and s in self.waiting,
                         "images": sum(gh * gw for _, gh, gw in s.images[0]) if s.images else 0,
                         "marks": s.marks})
        return {"t": now, "seqs": seqs, "cache": st, "device": self.engine.device_memory(),
                "running": len(self.alive), "waiting": len(self.waiting), "blocked": self.blocked,
                "target_seqs": self.max_seqs, "busy_s": self.m["q27:engine_busy_seconds_total"],
                "generated": self.m["q27:generation_tokens_total"], "prefilled": self.m["q27:prefill_tokens_total"]}


def lcp_bytes(a, b):
    """Common prefix of two int32 byte strings, in tokens."""
    lo, hi = 0, min(len(a), len(b)) // 4
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if a[4 * lo:4 * mid] == b[4 * lo:4 * mid] else (lo, mid - 1)
    return lo


def run(conn, cfg):
    """Process entry point (multiprocessing, spawn)."""
    sys.path.insert(0, cfg["engine_python"])
    try:
        Scheduler(conn, cfg).run()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - report and exit; the server starts a new engine
        traceback.print_exc()
        try:
            conn.send(("fatal", traceback.format_exc(limit=3)))
        except OSError:
            pass
        os._exit(1)
