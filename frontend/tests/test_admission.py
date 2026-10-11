"""Admission of waiting requests (P7-14, worker.Scheduler.admit) against a fake engine: no GPU."""

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402


class FakeEngine:
    def __init__(self, free, ckpt_only=0, park=171):
        self.st = {"free_pages": free, "checkpoint_only_pages": ckpt_only, "park_pages": park, "checkpoints": 0,
                   "evicted": 0}

    def cache_stats(self):
        return dict(self.st)


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def scheduler(engine, seqs=8, **cfg):
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.engine_seqs, s.max_seqs = engine, Conn(), seqs, seqs
    s.admit_kv, s.kv_reserve, s.min_tps = True, 4096, 0.0
    s.rates, s.blocked, s.held_until = {}, None, (None, 0.0)
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.configure(cfg)
    return s


def seq(rid, n, max_new=10_000, ids=None, started=False, fed=0, area=None, out=0):
    s = worker.Seq(rid, {"ids": ids if ids is not None else [rid * 100_000 + i for i in range(n)], "max_new": max_new,
                         "temperature": 1.0, "top_k": 20, "top_p": 0.95}, 0.0)
    s.started, s.fed, s.area, s.out = started, fed, area, [0] * out
    return s


def test_first_request_always_starts():
    sch = scheduler(FakeEngine(free=0))
    assert sch.admit(seq(1, 300_000))


def test_kv_room_counts_prompts_reserve_and_parking_area():
    run = seq(1, 6400, started=True, fed=6400)  # holds its pages; needs only the output reserve (64 pages)
    new = seq(2, 6400)                          # 100 pages + 64 reserve
    need = 64 + 100 + 64
    park = 171                                  # one more request: the running one may park
    sch = scheduler(FakeEngine(free=need + park))
    sch.alive = [run]
    assert sch.kv_room(new)
    sch.engine.st["free_pages"] -= 1
    assert not sch.kv_room(new)
    sch.engine.st["checkpoint_only_pages"] = 1  # pages only checkpoints hold count as room
    assert sch.kv_room(new)


def test_parked_request_already_holds_its_area():
    run = seq(1, 64, started=True, fed=64, area=0)
    cur = seq(2, 64, started=True, fed=64)
    new = seq(3, 64)
    need = 3 * 64 + 1                           # reserve of 3 requests + the new prompt
    sch = scheduler(FakeEngine(free=need + 171))  # 3 requests park 2 at most, 1 area is taken already
    sch.alive = [run, cur]
    assert sch.kv_room(new)
    sch.engine.st["free_pages"] -= 1
    assert not sch.kv_room(new)


def test_reserve_is_capped_by_max_new_and_prefix_shared_with_a_running_request():
    shared = list(range(12_800))
    run = seq(1, 0, ids=shared + [7] * 64, started=True, fed=12_864, max_new=64)
    new = seq(2, 0, ids=shared + [9] * 640, max_new=64)  # only the 640 new tokens need pages
    sch = scheduler(FakeEngine(free=1 + 10 + 1 + 171))
    sch.alive = [run]
    assert sch.live_prefix(new) == 12_800
    assert sch.kv_room(new)
    sch.engine.st["free_pages"] -= 1
    assert not sch.kv_room(new)


def test_admit_waits_on_kv_and_counts_once():
    sch = scheduler(FakeEngine(free=10))
    sch.alive = [seq(1, 64, started=True, fed=64)]
    new = seq(2, 64_000)
    assert not sch.admit(new)
    assert sch.blocked == "kv"
    sch.held_until = (None, 0.0)
    assert not sch.admit(new)
    assert sch.m['q27:admission_waits_total{reason="kv"}'] == 1
    sch.admit_kv = False
    sch.held_until = (None, 0.0)
    assert sch.admit(new) and sch.blocked is None


def test_admit_skips_rechecks_for_a_while():
    sch = scheduler(FakeEngine(free=10))
    sch.alive = [seq(1, 64, started=True, fed=64)]
    new = seq(2, 64_000)
    assert not sch.admit(new)
    sch.engine.st["free_pages"] = 10**6
    assert not sch.admit(new)  # (within ADMIT_RECHECK)
    sch.held_until = (None, 0.0)
    assert sch.admit(new)


def test_speed_floor():
    sch = scheduler(FakeEngine(free=10**6), min_seq_tps=100)
    sch.alive = [seq(i, 64, started=True, fed=64) for i in range(3)]
    new = seq(9, 64)
    assert sch.admit(new)              # nothing measured yet
    sch.rates = {3: [120.0, 1.0]}
    pred = 120.0 * worker.step_ms(3) / worker.step_ms(4)
    assert abs(sch.predicted_rate(4) - pred) < 1e-9 and pred > 100
    assert sch.admit(new)
    sch.rates[4] = [95.0, 1.0]         # measured beats the scaled estimate
    assert not sch.admit(new) and sch.blocked == "speed"


def test_measure_weighs_turns_by_time():
    sch = scheduler(FakeEngine(free=0))
    sch.measure(2, 200, 1.0)
    assert sch.rate(2) == 100.0
    sch.measure(2, 2, 0.1)             # a slow turn (a graph capture) counts for its 0.1 s only
    k = 1 - 0.1 / worker.RATE_WINDOW
    assert abs(sch.rate(2) - (100 * k + 1) / (k + 0.1)) < 1e-9 and sch.rate(2) > 90
    for _ in range(100):               # the old turns fade
        sch.measure(2, 100, 1.0)
    assert abs(sch.rate(2) - 50.0) < 1.0


def test_configure_clamps():
    sch = scheduler(FakeEngine(free=0), seqs=8, target_seqs=20, kv_reserve=-5, min_seq_tps=-1)
    assert (sch.max_seqs, sch.kv_reserve, sch.min_tps) == (8, 0, 0.0)
    sch.configure({"target_seqs": 0, "admit_kv": False})
    assert sch.max_seqs == 1 and sch.admit_kv is False
    st = sch.settings()
    assert st["target_seqs"] == 1 and st["engine_seqs"] == 8


class KvFull(RuntimeError):
    pass


def kvfull_scheduler(n):
    """n started requests in prefill (arrival order); the turn of any request runs out of KV pages."""
    sch = scheduler(FakeEngine(free=0))
    sch.KvFull, sch.cur, sch.engine.pos = KvFull, None, 0
    sch.engine.seq_suspend = lambda area: None
    sch.alive = [seq(i, 4096, started=True, fed=2048, area=i) for i in range(n)]
    for i, o in enumerate(sch.alive):
        o.arrival = float(i)

    def full(s):
        raise KvFull()
    sch.prefill_turn = full
    sch.finish = lambda s, reason: setattr(s, "dead", reason)
    return sch


def test_kvfull_latest_request_suspends_itself_and_waits_at_the_head():
    sch = kvfull_scheduler(3)
    first, _, last = sch.alive
    sch.waiting.append(seq(9, 64))
    sch._guarded("prefill", last)
    assert last not in sch.alive and sch.waiting[0] is last and not last.started
    assert last.preempted == 1 and first.preempted == 0


def test_kvfull_earlier_request_suspends_the_latest_and_retries():
    sch = kvfull_scheduler(2)
    first, last = sch.alive
    tries = []

    def turn(s):
        tries.append(s)
        if len(tries) == 1:
            raise KvFull()
    sch.prefill_turn = turn
    sch._guarded("prefill", first)
    assert tries == [first, first] and sch.alive == [first] and list(sch.waiting) == [last]


def test_kvfull_alone_ends_with_length():
    sch = kvfull_scheduler(1)
    only = sch.alive[0]
    sch._guarded("prefill", only)
    assert only.dead == "length"


def test_suspended_request_waits_for_room_without_admit_kv():
    sch = scheduler(FakeEngine(free=10), admit_kv=False)
    sch.alive = [seq(1, 64, started=True, fed=64)]
    new = seq(2, 64_000)
    assert sch.admit(new)
    new.preempted = 1
    assert not sch.admit(new) and sch.blocked == "kv"
