"""Logprobs of a request (no engine needed): the parameters, which tokens go out with which text, the formats."""

import pathlib
import queue
import sys
import threading

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import ApiError, Generation, chat_logprobs, completion_logprobs, logprobs_of  # noqa: E402
from tokenizer import BYTE_DECODER  # noqa: E402


def test_parameters():
    assert logprobs_of({}, chat=True) == -1
    assert logprobs_of({"logprobs": False}, chat=True) == -1
    assert logprobs_of({"logprobs": True}, chat=True) == 0
    assert logprobs_of({"logprobs": True, "top_logprobs": 5}, chat=True) == 5
    assert logprobs_of({"logprobs": True, "top_logprobs": 0}, chat=True) == 0
    assert logprobs_of({"logprobs": False, "top_logprobs": 0}, chat=True) == -1
    assert logprobs_of({}, chat=False) == -1
    assert logprobs_of({"logprobs": 0}, chat=False) == 0
    assert logprobs_of({"logprobs": 20}, chat=False) == 20


@pytest.mark.parametrize("body, chat", [({"logprobs": True, "top_logprobs": 21}, True),
                                        ({"top_logprobs": 3}, True), ({"logprobs": True, "top_logprobs": -1}, True),
                                        ({"logprobs": 1}, True), ({"logprobs": 21}, False), ({"logprobs": -1}, False),
                                        ({"logprobs": True}, False), ({"logprobs": "5"}, False)])
def test_bad_parameters(body, chat):
    with pytest.raises(ApiError) as e:
        logprobs_of(body, chat=chat)
    assert e.value.code == 400


class FakeTok:
    """Token i is the text PIECES[i]; 9 and 10 are the two halves of "é" (one character of two bytes)."""
    PIECES = {1: "Hello", 2: ",", 3: " wor", 4: "ld", 5: "END", 6: "!", 9: b"\xc3", 10: b"\xa9"}

    def decode(self, ids):
        b = b"".join(p if isinstance(p, bytes) else p.encode() for p in map(self.PIECES.get, ids))
        return b.decode("utf-8", errors="replace")

    def piece(self, i):
        p = self.PIECES[i]
        b = p if isinstance(p, bytes) else p.encode()
        return b.decode("utf-8", errors="replace"), b


class FakeEngine:
    def __init__(self):
        self.q = queue.Queue()
        self.cancelled = False

    def submit(self, req):
        return "req-0000", self.q

    def cancel(self, rid):
        self.cancelled = True


class FakeServer:
    def __init__(self):
        self.engine, self.tok, self.tok_lock = FakeEngine(), FakeTok(), threading.Lock()


def run(steps, stop=(), finish="length"):
    """Feeds the steps (lists of token ids) as "tok" messages with made-up logprobs; returns the pieces with the
    tokens that came out after each, and those taken at the end."""
    srv = FakeServer()
    gen = Generation(srv, [0], 100, {"logprobs": 1}, stop)
    for i, ids in enumerate(steps):
        srv.engine.q.put(("tok", gen.rid, ids, [(-0.5 * t, [(t, -0.5 * t)]) for t in ids]))
    srv.engine.q.put(("done", gen.rid, finish, {"cached_tokens": 0}))
    out = [(piece, [t for _, t, _, _ in gen.take_logprobs()]) for piece in gen]
    return gen, out, [t for _, t, _, _ in gen.take_logprobs()]


def test_tokens_follow_their_text():
    gen, out, rest = run([[1, 2], [3, 4]])
    assert out == [("Hello,", [1, 2]), (" world", [3, 4])] and rest == []
    assert [o for o, *_ in gen.lps] == [0, 5, 6, 10]


def test_split_character():
    # 9 alone is half a character: no text, its token goes out with the piece its character starts
    gen, out, rest = run([[1, 9], [10, 6]])
    assert out == [("Hello", [1]), ("é!", [9, 10, 6])] and rest == []


def test_stop_string_cuts_tokens():
    # "END" stops the text: its token is not returned, the ones before are
    gen, out, rest = run([[1, 2], [5, 6]], stop=["END"])
    text = "".join(p for p, _ in out)
    assert text == "Hello," and gen.stop_sequence == "END" and gen.server.engine.cancelled
    assert [t for _, ts in out for t in ts] + rest == [1, 2]


def test_held_text_released_at_the_end():
    # with a stop string the tail that may start one is held; at the end (no stop) every token is out
    gen, out, rest = run([[1, 3]], stop=["worldly"])
    assert "".join(p for p, _ in out) == "Hello wor"
    assert [t for _, ts in out for t in ts] + rest == [1, 3]


def test_formats():
    tok = FakeTok()
    entries = [(0, 1, -0.25, [(1, -0.25), (2, -2.0)]), (5, 9, float("-inf"), [])]
    chat = chat_logprobs(tok, entries)
    assert chat[0] == {"token": "Hello", "logprob": -0.25, "bytes": list(b"Hello"),
                       "top_logprobs": [{"token": "Hello", "logprob": -0.25, "bytes": list(b"Hello")},
                                        {"token": ",", "logprob": -2.0, "bytes": [44]}]}
    assert chat[1]["token"] == "�" and chat[1]["bytes"] == [0xc3] and chat[1]["logprob"] == -9999.0
    comp = completion_logprobs(tok, entries, 3)
    assert comp == {"tokens": ["Hello", "�"], "token_logprobs": [-0.25, -9999.0],
                    "top_logprobs": [{"Hello": -0.25, ",": -2.0}, {}], "text_offset": [3, 8]}


def test_byte_decoder():
    assert sorted(BYTE_DECODER.values()) == list(range(256))
    assert BYTE_DECODER["Ġ"] == 32 and BYTE_DECODER["Ċ"] == 10
