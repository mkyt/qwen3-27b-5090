"""A failed video decode in the worker, against a fake engine (no GPU): the client's bad video ends that request
(400); a lost CUDA context (the driver reset the GPU's engine) is not the client's fault and ends the worker, so
the server answers 503 and starts a new engine."""

import collections
import pathlib
import sys
from concurrent.futures import Future

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import videoproc  # noqa: E402
import worker  # noqa: E402


class FakeEngine:
    """The calls a turn makes up to the decode, and after its failure. `lost`: the sticky CUDA error, which every
    call that reaches CUDA then returns."""

    def __init__(self):
        self.lost, self.calls, self.pos = False, [], 0

    def _cuda(self, what):
        self.calls.append(what)
        if self.lost:
            raise RuntimeError(f"{what} failed: an illegal memory access was encountered")

    def prefix_begin(self, ids, at=None, spec=False):
        return 0

    def mm_set(self, spans):
        pass

    def kv_unlend(self, loan):
        self._cuda("kv_unlend")

    def reset(self):
        self._cuda("reset")


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def run_turn(error, lost, prefetched, loan):
    e = FakeEngine()
    e.lost = lost
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.spec, s.cur, s.KvFull = e, Conn(), False, None, type("KvFull", (Exception,), {})
    s.returns, s.alive, s.waiting = collections.deque(), [], collections.deque()
    s.by_id, s.m = {}, collections.Counter()
    r = worker.Seq("r", {"ids": list(range(8)), "max_new": 8, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "images": ([(0, 2, 2)], [("video", 0, 0)], [(b"", None)])}, 0.0)
    s.alive.append(r)

    def decode(*_):
        raise error

    s.decode, s.lend, s.finish = decode, lambda *_: (1, loan), lambda *_, **kw: None
    if prefetched:
        f = Future()
        f.set_exception(error)
        f.first_pair = 0
        r.decoding[("video", 0)] = f
        if loan >= 0:
            s.returns.append(loan)
    raised = None
    try:
        s._guarded("prefill", r)
    except RuntimeError as x:
        raised = x
    return s, e, raised


@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("loan", [-1, 0])
def test_bad_video_is_the_clients_error(prefetched, loan):
    s, e, raised = run_turn(ValueError("the video's bitstream cannot be parsed"), False, prefetched, loan)
    assert raised is None
    assert [m[3] for m in s.conn.sent if m[0] == "error"] == [400]
    assert e.calls == (["kv_unlend"] if loan >= 0 else []) + ["reset"]


@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("loan", [-1, 0])
def test_lost_context_ends_the_worker(prefetched, loan):
    err = videoproc.DeviceLost("q27_video_frames: an illegal memory access was encountered")
    s, e, raised = run_turn(err, True, prefetched, loan)
    assert raised is err  # (out of the scheduler: worker.run reports it and exits)
    assert not [m for m in s.conn.sent if m[0] == "error"]  # (no 400: the server fails the request with 503)
    assert not e.calls  # (nothing more on the dead context)


def test_engine_code_for_a_lost_context(monkeypatch):
    import types

    fake = types.SimpleNamespace(q27_last_error=lambda: b"NVDEC: an illegal memory access was encountered")
    monkeypatch.setitem(sys.modules, "q27lib", types.SimpleNamespace(lib=lambda: fake))
    with pytest.raises(videoproc.DeviceLost):
        videoproc._check(-4, "q27_video_frames")
    with pytest.raises(ValueError):
        videoproc._check(-3, "q27_video_frames")
    with pytest.raises(RuntimeError) as x:
        videoproc._check(-1, "q27_video_frames")
    assert not isinstance(x.value, videoproc.DeviceLost)
