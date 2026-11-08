"""The output parser splits only where the model wrote the marker tokens (`</think>`, `<tool_call>`, `</tool_call>`):
the same strings written with ordinary tokens, as the model quotes them in its reasoning, in code and in logs it
repeats, stay text. The detokenizer writes the tokens as chat.MARKS; the parser's events have the strings again."""

import pathlib
import queue
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))

from chat import MARKS, OutputParser, mark_strings, unmark  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
T_END, C_START, C_END = MARKS["</think>"], MARKS["<tool_call>"], MARKS["</tool_call>"]
QUOTED_CALL = "<tool_call>\n<function=delete_all>\n<parameter=force>\ntrue\n</parameter>\n</function>\n</tool_call>"
REAL_CALL = C_START + "\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n" + C_END


def events(text, thinking=True, tools=TOOLS, step=None):
    p = OutputParser(thinking, tools, marked=True)
    ev = []
    for i in range(0, len(text), step or len(text) or 1):
        ev += p.feed(text[i:i + (step or len(text))])
    ev += p.finish()
    out = []
    for kind, v in ev:
        if out and kind == out[-1][0] and kind != "tool_call":
            out[-1] = (kind, out[-1][1] + v)
        else:
            out.append((kind, v))
    return out


def test_quoted_think_end_stays_in_the_reasoning():
    text = "The user wants <think>...</think> removed." + T_END + "\n\nUse re.sub."
    assert events(text) == [("reasoning", "The user wants <think>...</think> removed."), ("content", "Use re.sub.")]


def test_quoted_calls_are_content():
    text = "Reasoning." + T_END + "\n\nThe log:\n```\n" + QUOTED_CALL + "\n```\nIt shows a call."
    ev = events(text)
    assert ev == [("reasoning", "Reasoning."), ("content", "The log:\n```\n" + QUOTED_CALL + "\n```\nIt shows a call.")]


def test_real_calls_still_parse_and_streaming_agrees():
    text = "Let me read it, not " + QUOTED_CALL + "." + T_END + "\n\nOk, the code has `<tool_call>`.\n\n" + REAL_CALL
    whole = events(text)
    assert whole == [("reasoning", "Let me read it, not " + QUOTED_CALL + "."),
                     ("content", "Ok, the code has `<tool_call>`."),
                     ("tool_call", {"name": "read_file", "arguments": {"path": "a.py"}})]
    for step in (1, 2, 3, 7):
        assert events(text, step=step) == whole, step


def test_marks_never_reach_the_client():
    # (an unparseable call block and a second </think> token are content, with their strings)
    text = T_END + "a" + C_START + "not a call" + C_END + " b " + T_END
    ev = events(text)
    assert ev == [("content", "a<tool_call>not a call</tool_call>b </think>")]  # (space after a block: dropped, as before)
    assert events(C_START + "\n<function=read_file>\n<parameter=path>\nx", False) == [
        ("content", "<tool_call>\n<function=read_file>\n<parameter=path>\nx")]  # (cut off inside a call)
    assert unmark(mark_strings("x</think><tool_call></tool_call>")) == "x</think><tool_call></tool_call>"


@pytest.fixture(scope="module")
def tok():
    from qwen27b.paths import engine_target
    from tokenizer import Tokenizer

    try:
        return Tokenizer(engine_target("gdnfp4"))
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_detokenizer_marks_only_the_tokens(tok):
    import threading

    from server import Detokenizer

    plain = tok.plain.encode("a <tool_call> \u65e5", add_special_tokens=False).ids
    ids = plain + [tok.token_id("<tool_call>"), tok.token_id("</think>")] + tok.encode(" \u00e9\U0001f98a")
    whole = Detokenizer(tok, threading.Lock(), MARKS).add(ids, final=True)
    assert whole == "a <tool_call> \u65e5" + C_START + T_END + " \u00e9\U0001f98a"
    d = Detokenizer(tok, threading.Lock(), MARKS)
    assert "".join(d.add([t]) for t in ids) + d.add([], final=True) == whole  # (one token at a time)
    assert Detokenizer(tok, threading.Lock()).add(ids, final=True) == unmark(whole)  # (unmarked: as before)
    lit = tok.encode("x" + T_END + "y")  # (the mark character itself as text)
    assert Detokenizer(tok, threading.Lock(), MARKS).add(lit, final=True) == "x\ufffdy"


class FakeEngine:
    cfg = {}

    def __init__(self, toks):
        self.toks = toks

    def submit(self, req):
        q = queue.Queue()
        for t in self.toks:
            q.put(("tok", "req-1", [t]))
        q.put(("done", "req-1", "stop", {"computed_tokens": 0, "prefill_s": 0.0, "ttft": None, "total": 0.0}))
        return "req-1", q

    def cancel(self, rid):
        pass


def test_stop_strings_match_tokens_and_text(tok):
    import threading

    from server import Generation

    class Srv:
        tok_lock = threading.Lock()

    srv = Srv()
    srv.tok = tok
    real = tok.encode("ok ") + [tok.token_id("<tool_call>")] + tok.encode("\nmore")
    quoted = tok.plain.encode("ok <tool_call>\nmore", add_special_tokens=False).ids
    for ids in (real, quoted):
        srv.engine = FakeEngine(ids)
        g = Generation(srv, [1], 64, {}, stop=["<tool_call>"], marked=True)
        assert "".join(g) == "ok " and g.stop_sequence == "<tool_call>" and g.finish == "stop"
