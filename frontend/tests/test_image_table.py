"""The image table of a request with more image tokens than IMAGE_TOKENS (worker.py Scheduler.mm_ready), against a
fake engine (no GPU): the prompt runs a chunk at a time, and every image position of a chunk must read its own
image's embedding, encoded into rows that no later image has overwritten yet."""

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402


class FakeEngine:
    """Rows of the table hold (span, offset) once encoded; place / encode as the engine checks them."""

    def __init__(self, spans, rows):
        self.spans, self.rows = spans, rows
        self.row = {}                    # span -> its row (mm_place)
        self.table = [None] * rows
        self.encoded = []                # spans in the order they were encoded

    def mm_place(self, i, n, row):
        for k in range(i, i + n):
            _, gh, gw = self.spans[k]
            assert 0 <= row and row + gh * gw <= self.rows
            self.row[k], row = row, row + gh * gw

    def _encode(self, i, n):
        for k in range(i, i + n):
            if k > i:  # (one block of rows)
                assert self.row[k] == self.row[k - 1] + self.spans[k - 1][1] * self.spans[k - 1][2]
            _, gh, gw = self.spans[k]
            for r in range(gh * gw):
                self.table[self.row[k] + r] = (k, r)
            self.encoded.append(k)

    def mm_encode_rgb(self, i, rgb, norm):
        assert rgb == ("pixels", i)
        self._encode(i, 1)

    def mm_encode_frames(self, i, frames, norm):
        k, g = self.srcs[i][1], self.srcs[i][2]
        assert frames == [("frame", k, f) for f in range(2 * g, 2 * g + len(frames))]
        self._encode(i, len(frames) // 2)

    def run(self, lo, hi):
        """The chunk's embedding reads: every image position must find its own image's row."""
        for k, (start, gh, gw) in enumerate(self.spans):
            for p in range(max(lo, start), min(hi, start + gh * gw)):
                assert self.table[self.row[k] + p - start] == (k, p - start), (k, p)


def prompt(rng, n_images, n_videos=0):
    """Spans (start, gh, gw) and their sources: images of random grids and videos of pairs, text between them."""
    spans, srcs, pos = [], [], rng.randrange(0, 40)
    items = ["image"] * n_images + ["video"] * n_videos
    rng.shuffle(items)
    video = 0
    for kind in items:
        if kind == "image":
            gh, gw = rng.randrange(1, 9), rng.randrange(1, 9)
            spans.append((pos, gh, gw))
            srcs.append(("image", b"", None, None))
            pos += gh * gw + rng.randrange(0, 30)
        else:
            gh, gw = rng.randrange(1, 4), rng.randrange(1, 4)
            for g in range(rng.randrange(1, 12)):
                pos += 3  # (the pair's time stamp and vision start)
                spans.append((pos, gh, gw))
                srcs.append(("video", video, g, None))
                pos += gh * gw + 1
            video += 1
            pos += rng.randrange(0, 30)
    return spans, srcs, pos + rng.randrange(1, 40)


def scheduler(engine, rows, chunk):
    s = object.__new__(worker.Scheduler)
    s.engine, s.image_tokens, s.chunk = engine, rows, chunk
    return s


def run_prompt(spans, srcs, length, rows, chunk, cached=0, seed=0):
    """A prefill from `cached`: chunks of random lengths (the engine's cuts) within what mm_ready was told."""
    rng = random.Random(seed)
    e = FakeEngine(spans, rows)
    e.srcs = srcs
    sch = scheduler(e, rows, chunk)
    s = worker.Seq("r", {"ids": [0] * length, "max_new": 1, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
                         "images": (spans, srcs, [])}, 0.0)

    def pixels(_s, key, g=0):
        if isinstance(key, tuple):
            n = sum(1 for src in srcs if src[0] == "video" and src[1] == key[1])
            return [("frame", key[1], f) for f in range(2 * g, 2 * n)]
        return ("pixels", key)

    sch.pixels = pixels
    fed = cached
    while fed < length:
        rest = length - fed
        hi = fed + (rest if rest <= chunk else min(rest, 2 * chunk))
        sch.mm_ready(s, fed, hi)
        end = rng.randrange(fed + 1, hi + 1)
        e.run(fed, end)
        fed = end
    return e


@pytest.mark.parametrize("seed", range(40))
def test_many_images_in_a_small_table(seed):
    rng = random.Random(seed)
    spans, srcs, length = prompt(rng, rng.randrange(1, 40), rng.randrange(0, 3))
    chunk = rng.choice((16, 32, 64))
    biggest = max(gh * gw for _, gh, gw in spans)
    rows = 2 * chunk + 2 * biggest  # (enough for any window of two chunks)
    e = run_prompt(spans, srcs, length, rows, chunk, seed=seed)
    assert set(e.encoded) == set(range(len(spans)))


def test_each_image_is_encoded_once_when_the_table_holds_them_all():
    rng = random.Random(1)
    spans, srcs, length = prompt(rng, 30, 2)
    e = run_prompt(spans, srcs, length, sum(gh * gw for _, gh, gw in spans), 32)
    assert sorted(e.encoded) == list(range(len(spans)))


def test_images_inside_the_restored_prefix_are_not_encoded():
    rng = random.Random(2)
    spans, srcs, length = prompt(rng, 20)
    cached = spans[10][0]  # (images 0..9 lie before it)
    e = run_prompt(spans, srcs, length, 64 + 2 * 64, 32, cached=cached)
    assert sorted(e.encoded) == list(range(10, len(spans)))


def test_too_many_image_tokens_near_one_chunk():
    spans = [(0, 8, 8), (64, 8, 8)]
    srcs = [("image", b"", None, None)] * 2
    with pytest.raises(worker.ImageError, match="more than IMAGE_TOKENS"):
        run_prompt(spans, srcs, 200, 100, 64)
