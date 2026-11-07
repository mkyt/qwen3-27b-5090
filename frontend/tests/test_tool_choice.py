"""tool_choice (forced tool calls): the request forms (chat.tool_choice_of), the forced text (Server.force) and the
worker's forced output (Scheduler.emit / force_output) against a fake engine: no GPU."""

import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402
from chat import forced_call, parse_output, tool_choice_of  # noqa: E402
from server import Server  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}}}}}, {"type": "function", "function": {"name": "ls", "parameters": {}}}]


@pytest.mark.parametrize("choice, want", [
    (None, ("auto", None)), ("auto", ("auto", None)), ("none", ("none", None)), ("required", ("required", None)),
    ({"type": "function", "function": {"name": "ls"}}, ("named", "ls")),       # OpenAI chat
    ({"type": "function", "name": "ls"}, ("named", "ls")),                     # Responses
    ({"type": "custom", "name": "ls"}, ("named", "ls")),
    ({"type": "allowed_tools", "mode": "required", "tools": []}, ("required", None)),
    ({"type": "allowed_tools", "mode": "auto", "tools": []}, ("auto", None)),
    ({"type": "auto"}, ("auto", None)), ({"type": "any"}, ("required", None)),  # Anthropic
    ({"type": "tool", "name": "read_file"}, ("named", "read_file")), ({"type": "none"}, ("none", None)),
])
def test_forms(choice, want):
    assert tool_choice_of(choice, TOOLS) == want


@pytest.mark.parametrize("choice, tools", [
    ({"type": "tool", "name": "write_file"}, TOOLS), ("required", None), ({"type": "tool", "name": "ls"}, []),
    ("sometimes", TOOLS), ({"type": "function"}, TOOLS), ({"type": "tool", "name": ""}, TOOLS),
])
def test_bad_forms(choice, tools):
    with pytest.raises(ValueError):
        tool_choice_of(choice, tools)


def test_forced_text_parses_as_the_call():
    assert forced_call("auto", None) is None and forced_call("none", None) is None
    named = forced_call("named", "read_file")
    assert named == "<tool_call>\n<function=read_file>\n"
    rest = "<parameter=path>\nsrc/main.py\n</parameter>\n</function>\n</tool_call>"
    reasoning, content, calls = parse_output("Let me look.\n</think>\n\n" + named + rest, True, TOOLS)
    assert reasoning == "Let me look." and content == "" and calls == [{"name": "read_file", "arguments": {"path": "src/main.py"}}]
    # Any tool: the model writes "=name>" itself.
    assert forced_call("required", None) == "<tool_call>\n<function"
    _, content, calls = parse_output(forced_call("required", None) + "=ls>\n</function>\n</tool_call>", False, TOOLS)
    assert content == "" and calls == [{"name": "ls", "arguments": {}}]


class FakeTok:
    def token_id(self, t):
        return {"</think>": 99}[t]


def test_server_force():
    srv = object.__new__(Server)
    srv.tok, srv.encode = FakeTok(), lambda text: [ord(c) for c in text]
    assert srv.force(("auto", None), True) is None
    trigger, ids = srv.force(("named", "ls"), True)
    assert trigger == 99 and "".join(map(chr, ids)) == "\n\n<tool_call>\n<function=ls>\n"
    assert srv.force(("required", None), False) == (None, [ord(c) for c in "<tool_call>\n<function"])


# ---------------------------------------------------------------- the worker

THINK, EOS = 99, 7


class FakeEngine:
    def __init__(self):
        self.calls = []

    def checkpoint(self):
        self.calls.append("checkpoint")

    def reset(self):
        self.calls.append("reset")

    def seq_suspend(self, area):
        self.calls.append(("seq_suspend", area))


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def scheduler():
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.eos, s.m = FakeEngine(), Conn(), {EOS}, collections.Counter()
    s.alive, s.waiting, s.by_id, s.cur = [], collections.deque(), {}, None
    s.finish = lambda q, reason: setattr(q, "dead", reason)
    return s


def seq(force, max_new=100, logprobs=-1):
    q = worker.Seq("r", {"ids": [1, 2, 3], "max_new": max_new, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "force": force, "logprobs": logprobs}, 0.0)
    q.started, q.fed, q.phase = True, 3, worker.DECODE
    return q


def toks(sch):
    return [m[2] for m in sch.conn.sent if m[0] == "tok"]


def test_force_after_the_reasoning_restarts_from_a_checkpoint():
    sch = scheduler()
    q = seq((THINK, [50, 51]))
    sch.alive, sch.cur = [q], q
    sch.emit(q, [10, 11])
    assert q.phase == worker.DECODE and toks(sch) == [[10, 11]]
    sch.emit(q, [12, THINK, 13, 14])  # (tokens past the reasoning's end are dropped)
    assert toks(sch)[-1] == [12, THINK]
    assert q.out == [10, 11, 12, THINK, 50, 51] and q.eng_ids == [1, 2, 3] + q.out
    assert q.phase == worker.PREFILL and not q.started and q.fed == 0 and q.force is None and q in sch.alive
    assert sch.engine.calls == ["checkpoint", "reset"] and sch.cur is None
    assert q.est_cached == 3 + 4  # (prompt + output up to the reasoning's end)
    # The forced tokens go out with the first token after the prefill; a second THINK does not force again.
    q.phase = worker.DECODE
    sch.emit(q, [20, THINK])
    assert toks(sch)[-1] == [50, 51, 20, THINK] and q.out[-2:] == [20, THINK]


def test_force_of_a_parked_request_suspends_its_area():
    sch = scheduler()
    q = seq((THINK, [50]))
    q.area = 2
    sch.emit(q, [THINK])
    assert sch.engine.calls == [("seq_suspend", 2)] and q.area is None and q.phase == worker.PREFILL


def test_force_at_the_start():
    sch = scheduler()
    q = worker.Seq("r", {"ids": [1, 2, 3], "max_new": 100, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "force": (None, [50, 51]), "logprobs": 2}, 0.0)
    assert q.out == [50, 51] and q.eng_ids == [1, 2, 3, 50, 51] and q.force is None
    q.phase = worker.DECODE
    sch.emit(q, [10], [(-0.5, [(10, -0.5)])])
    assert sch.conn.sent[-1] == ("tok", "r", [50, 51, 10], [(0.0, []), (0.0, []), (-0.5, [(10, -0.5)])])
    assert q.out == [50, 51, 10]


def test_forced_tokens_go_out_before_an_eos():
    sch = scheduler()
    q = worker.Seq("r", {"ids": [1], "max_new": 100, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "force": (None, [50])}, 0.0)
    sch.emit(q, [EOS])
    assert toks(sch) == [[50]] and q.dead == "stop"


def test_force_up_to_max_new():
    sch = scheduler()
    q = seq((THINK, [50, 51, 52]), max_new=4)
    sch.emit(q, [10, THINK])
    assert q.out == [10, THINK, 50, 51] and toks(sch) == [[10, THINK], [50, 51]] and q.dead == "length"
    assert sch.engine.calls == []  # (no restart)


def test_max_new_at_the_reasoning_end_does_not_force():
    sch = scheduler()
    q = seq((THINK, [50]), max_new=2)
    sch.emit(q, [10, THINK])
    assert q.out == [10, THINK] and q.dead == "length" and sch.engine.calls == []
