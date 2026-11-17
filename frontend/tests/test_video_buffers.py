"""Frame ownership and peak Python allocation, with a fake native decoder (no model or GPU)."""

import ctypes
import pathlib
import sys
import tracemalloc
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "frontend"), str(ROOT / "engine/python")]
import q27lib
import videoproc


def test_frames_own_their_bytes_and_repeat_missing_tail(monkeypatch):
    def decode(raw, length, want, count, w, h, sessions, buf, stats):
        stats[0] = 2
        for k in range(count):
            ctypes.memset(ctypes.addressof(buf) + k * w * h * 3, k + 1, w * h * 3)
        return 0

    monkeypatch.setattr(q27lib, "lib", lambda: SimpleNamespace(q27_video_frames=decode))
    plan = SimpleNamespace(indices=(0, 1, 1, 9), size=(2, 2))
    frames = videoproc.frames(b"", plan)
    assert frames == [bytes([1]) * 12, bytes([2]) * 12, bytes([2]) * 12, bytes([2]) * 12]
    assert all(isinstance(f, bytes) for f in frames)


def test_frames_do_not_copy_the_whole_buffer_before_slicing(monkeypatch):
    def decode(*args):
        args[-1][0] = 16
        return 0

    monkeypatch.setattr(q27lib, "lib", lambda: SimpleNamespace(q27_video_frames=decode))
    plan = SimpleNamespace(indices=tuple(range(16)), size=(256, 256))
    tracemalloc.start()
    try:
        frames = videoproc.frames(b"", plan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    size = sum(map(len, frames))
    assert size == 16 * 256 * 256 * 3
    assert peak < size * 2.2  # native output + owned frame bytes, not a third whole-video copy
