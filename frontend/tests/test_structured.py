"""Structured output: the request forms (structured.*_format), the grammars (structured.grammar, checked by llguidance),
Server.constrain, and the worker's matching (Scheduler.emit, write_masks) against a fake engine and matcher: no GPU."""

import collections
import ctypes
import pathlib
import sys

import llguidance
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import structured  # noqa: E402
import worker  # noqa: E402
from server import ApiError, Server  # noqa: E402

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


@pytest.mark.parametrize("body, want", [
    ({}, None),
    ({"response_format": {"type": "text"}}, None),
    ({"response_format": {"type": "json_object"}}, ("json", None)),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA, "strict": True}}},
     ("json", SCHEMA)),
    ({"structured_outputs": {"json": SCHEMA}}, ("json", SCHEMA)),
    ({"structured_outputs": {"json": '{"type": "object"}'}}, ("json", {"type": "object"})),
    ({"structured_outputs": {"regex": "[0-9]+"}}, ("regex", "[0-9]+")),
    ({"structured_outputs": {"choice": ["yes", "no"]}}, ("choice", ["yes", "no"])),
    ({"structured_outputs": {"grammar": 'start: "a"'}}, ("grammar", 'start: "a"')),
    ({"structured_outputs": {"json_object": True}}, ("json", None)),
    ({"structured_outputs": {}}, None),
])
def test_chat_forms(body, want):
    assert structured.chat_format(body) == want


@pytest.mark.parametrize("body", [
    {"response_format": {"type": "xml"}}, {"response_format": "json"}, {"response_format": {"type": "json_schema"}},
    {"response_format": {"type": "json_schema", "json_schema": {"schema": "{not json"}}},
    {"response_format": {"type": "json_object"}, "structured_outputs": {"regex": "a"}},
    {"structured_outputs": {"regex": "a", "choice": ["b"]}}, {"structured_outputs": {"choice": []}},
    {"structured_outputs": {"structural_tag": {}}}, {"structured_outputs": {"grammar": ""}},
])
def test_bad_chat_forms(body):
    with pytest.raises(ValueError):
        structured.chat_format(body)


def test_responses_and_anthropic_forms():
    assert structured.responses_format({}) is None
    assert structured.responses_format({"text": {"format": {"type": "text"}}}) is None
    assert structured.responses_format({"text": {"format": {"type": "json_schema", "name": "x", "schema": SCHEMA}}}) \
        == ("json", SCHEMA)
    assert structured.responses_format({"text": {"format": {"type": "json_object"}}}) == ("json", None)
    assert structured.anthropic_format({}) is None
    assert structured.anthropic_format({"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}) \
        == ("json", SCHEMA)
    assert structured.anthropic_format({"output_format": {"type": "json_schema", "schema": SCHEMA}}) == ("json", SCHEMA)
    with pytest.raises(ValueError):
        structured.anthropic_format({"output_config": {"format": {"type": "json_object"}}})


@pytest.mark.parametrize("spec", [("json", None), ("json", SCHEMA), ("regex", r"[0-9]{3}/[a-z]"),
                                  ("choice", ['say "yes"', "いいえ"]), ("grammar", 'start: "a" b\nb: "c"+'),
                                  ("grammar", 'root ::= "x" [0-9]+')])
@pytest.mark.parametrize("lead_ws, calls", [(False, None), (True, None), (True, (500, 501))])
def test_grammars(spec, lead_ws, calls):
    g = structured.grammar(spec, lead_ws=lead_ws, calls=calls)
    assert not llguidance.LLMatcher.validate_grammar_with_warnings(g)[0]


@pytest.mark.parametrize("spec", [("json", {"type": "nonsense"}), ("regex", "(["), ("grammar", 'x: "a"'),
                                  ("grammar", "start: (")])
def test_bad_grammars(spec):
    with pytest.raises(ValueError):
        structured.grammar(spec)


class FakeTok:
    def token_id(self, t):
        return {"</think>": 99, "<tool_call>": 500, "</tool_call>": 501}[t]


def test_server_constrain():
    srv = object.__new__(Server)
    srv.tok = FakeTok()
    fmt = lambda: ("json", None)  # noqa: E731
    assert srv.constrain(lambda: None, ("auto", None), None, True) is None
    assert srv.constrain(fmt, ("required", None), [{}], True) is None  # (a forced call has no format)
    assert srv.constrain(fmt, ("named", "f"), [{}], False) is None
    g = srv.constrain(fmt, ("auto", None), None, True)
    assert g["grammar_after"] == 99 and "<[500]>" not in g["grammar"]
    g = srv.constrain(fmt, ("auto", None), [{}], False)
    assert g["grammar_after"] is None and "<[500]>" in g["grammar"]
    assert "<[500]>" not in srv.constrain(fmt, ("none", None), [{}], False)["grammar"]
    with pytest.raises(ApiError):
        srv.constrain(lambda: structured.chat_format({"response_format": {"type": "xml"}}), ("auto", None), None, True)


# ---------------------------------------------------------------- the worker

THINK, EOS = 99, 7


class Matcher:
    """llguidance's matcher on a toy grammar: tokens below 50, then EOS (accepting after one token)."""

    def __init__(self):
        self.toks, self.log = [], []

    def validate_tokens(self, ts):
        n = 0
        for t in ts:
            if t >= 50 and t != EOS:
                break
            n += 1
        return n

    def consume_token(self, t):
        assert t < 50
        self.toks.append(t)
        self.log.append(("consume", t))
        return True

    def rollback(self, n):
        del self.toks[len(self.toks) - n:]
        self.log.append(("rollback", n))

    def is_accepting(self):
        return bool(self.toks)

    def unsafe_compute_mask_ptr(self, addr, size):
        self.log.append(("mask", len(self.toks)))
        ctypes.memset(addr, 0x11, size)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.rows = (ctypes.c_uint32 * (16 * 4))()

    def checkpoint(self):
        self.calls.append("checkpoint")

    def reset(self):
        self.calls.append("reset")

    def spec_set_masked(self, area=None):
        self.calls.append(("spec_set_masked", area))

    def mask_row(self, i):
        return ctypes.addressof(self.rows) + 16 * i, 16


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def scheduler():
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.eos, s.m, s.spec = FakeEngine(), Conn(), {EOS}, collections.Counter(), True
    s.alive, s.waiting, s.by_id, s.cur = [], collections.deque(), {}, None
    s.finish = lambda q, reason: setattr(q, "dead", reason)
    return s


def seq(after):
    q = worker.Seq("r", {"ids": [1, 2, 3], "max_new": 100, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "grammar": "g", "grammar_after": after}, 0.0)
    q.matcher = Matcher()
    q.gm = q.matcher if after is None else None
    q.started, q.fed, q.phase = True, 3, worker.DECODE
    return q


def toks(sch):
    return [m[2] for m in sch.conn.sent if m[0] == "tok"]


def test_grammar_from_the_start():
    sch = scheduler()
    q = seq(None)
    sch.emit(q, [10, 11])
    sch.emit(q, [12, EOS])
    assert q.matcher.toks == [10, 11, 12] and toks(sch) == [[10, 11], [12]] and q.dead == "stop"


def test_grammar_after_the_reasoning():
    sch = scheduler()
    q = seq(THINK)
    sch.alive, sch.cur = [q], q
    sch.emit(q, [60, 61])  # (the reasoning is not matched)
    assert q.gm is None and q.matcher.toks == []
    sch.emit(q, [62, THINK, 10, 11])  # the rest of the step is in the grammar: kept, and the engine masks from here
    assert q.gm is q.matcher and q.matcher.toks == [10, 11] and toks(sch)[-1] == [62, THINK, 10, 11]
    assert sch.engine.calls == [("spec_set_masked", None)]


def test_grammar_after_the_reasoning_parked():
    sch = scheduler()
    q = seq(THINK)
    q.area = 3
    sch.emit(q, [THINK])
    assert sch.engine.calls == [("spec_set_masked", 3)]


def test_step_past_the_reasoning_outside_the_grammar_restarts():
    sch = scheduler()
    q = seq(THINK)
    sch.alive, sch.cur = [q], q
    sch.emit(q, [62, THINK, 10, 70, 11])  # (70 is outside: the engine holds 70, 11 and runs again without them)
    assert toks(sch)[-1] == [62, THINK, 10] and q.out == [62, THINK, 10] and q.eng_ids == [1, 2, 3, 62, THINK, 10]
    assert q.phase == worker.PREFILL and not q.started and sch.engine.calls == ["checkpoint", "reset"]
    assert sch.m["q27:grammar_restarts_total"] == 1


def test_eos_before_the_grammar_accepts_restarts():
    sch = scheduler()
    q = seq(THINK)
    sch.alive, sch.cur = [q], q
    sch.emit(q, [THINK, EOS])  # (nothing in the grammar yet: EOS is not allowed)
    assert q.out == [THINK] and q.phase == worker.PREFILL and getattr(q, "dead", False) is False


def test_write_masks():
    eng = FakeEngine()
    m = Matcher()
    m.toks = [5]
    worker.write_masks(eng, m, [5, 10, 11, 70, 12, 13, 14, 15], 8, {EOS})
    # rows 8..10: after chain[1..0], [1..1], [1..2]; rows 11..15 (past 70) open; the matcher is back where it was
    assert m.log == [("mask", 1), ("consume", 10), ("mask", 2), ("consume", 11), ("mask", 3), ("rollback", 2)]
    assert m.toks == [5]
    words = list(eng.rows)
    assert words[:32] == [0] * 32 and words[32:44] == [0x11111111] * 12 and words[44:64] == [0xFFFFFFFF] * 20
    m.log = []
    worker.write_masks(eng, m, [5, 10, EOS, 11, 12, 13, 14, 15], 0, {EOS})  # (an EOS is not consumed)
    assert m.log == [("mask", 1), ("consume", 10), ("mask", 2), ("rollback", 1)]
