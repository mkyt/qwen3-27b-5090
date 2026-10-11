"""Output parsing of the API server (no engine needed)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import _partial_suffix, normalize_messages, parse_tool_calls, split_reasoning  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}]


def test_split_reasoning():
    assert split_reasoning("think a bit\n</think>\n\nThe answer.", True) == ("think a bit", "The answer.")
    assert split_reasoning("still thinking", True) == ("still thinking", "")
    assert split_reasoning("plain", False) == (None, "plain")


def test_parse_tool_calls():
    text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.py\n</parameter>\n"
            "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>")
    head, calls = parse_tool_calls(text, TOOLS)
    assert head == "Let me look."
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "src/main.py", "limit": 20}


def test_string_parameter_stays_string():
    text = "<tool_call>\n<function=read_file>\n<parameter=path>\n123\n</parameter>\n</function>\n</tool_call>"
    _, calls = parse_tool_calls(text, TOOLS)
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "123"}


def test_partial_suffix():
    assert _partial_suffix("abc <tool", "<tool_call>") == 5
    assert _partial_suffix("abc", "<tool_call>") == 0


def test_normalize_messages_parses_arguments():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"path\": \"a.py\"}"}}]}]
    out = normalize_messages(msgs)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == "{\"path\": \"a.py\"}"  # input untouched


# ---------------------------------------------------------------- Phase 6: streaming parser, API mapping

from chat import (OutputParser, _convert, anthropic_messages, anthropic_template_kwargs,  # noqa: E402
                  parse_output, template_kwargs)

OUT = ("Let me check the file first.\nThen answer.\n</think>\n\nI'll read it.\n\n<tool_call>\n<function=read_file>\n"
       "<parameter=path>\nsrc/main.py\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>\n"
       "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>")


def _events(text, step, thinking=True, tools=TOOLS):
    p = OutputParser(thinking, tools)
    ev = []
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish()
    # Merge adjacent text events (the split points differ with the step).
    out = []
    for kind, v in ev:
        if out and kind == out[-1][0] and kind != "tool_call":
            out[-1] = (kind, out[-1][1] + v)
        else:
            out.append((kind, v))
    return out


def test_stream_parser_matches_whole_text():
    whole = _events(OUT, len(OUT))
    assert whole == [("reasoning", "Let me check the file first.\nThen answer."), ("content", "I'll read it."),
                     ("tool_call", {"name": "read_file", "arguments": {"path": "src/main.py", "limit": 20}}),
                     ("tool_call", {"name": "read_file", "arguments": {"path": "README.md"}})]
    for step in (1, 2, 3, 5, 7, 13):
        assert _events(OUT, step) == whole, step


def test_parser_without_tools_keeps_xml_and_thinking_off():
    r, c, calls = parse_output("<tool_call>x</tool_call> after", False, None)
    assert (r, c, calls) == (None, "<tool_call>x</tool_call> after", [])


def test_parser_truncated_reasoning_and_call():
    assert parse_output("still thinking\n", True, TOOLS) == ("still thinking", "", [])
    r, c, calls = parse_output("x</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na", True, TOOLS)
    assert calls == [] and "<function=read_file>" in c  # (cut off inside a call: kept as text)


def test_convert_by_schema():
    assert _convert("20", {"type": "integer"}) == 20
    assert _convert("True", {"type": "boolean"}) is True
    assert _convert('{"a": 1}', {"type": "object"}) == {"a": 1}
    assert _convert("{'a': 1}", {"type": "object"}) == {"a": 1}
    assert _convert("123", {"type": ["string", "null"]}) == "123"
    assert _convert("null", {"type": ["string", "null"]}) is None
    assert _convert("hello world", {}) == "hello world"


def test_normalize_roles_and_reasoning():
    out = normalize_messages([{"role": "developer", "content": "a"}, {"role": "system", "content": [{"type": "text", "text": "b"}]},
                              {"role": "user", "content": "q"},
                              {"role": "assistant", "content": None, "reasoning": "r"}, {"role": "system", "content": "late"}])
    assert out[0] == {"role": "system", "content": "a\n\nb"}
    assert out[2]["reasoning_content"] == "r" and out[2]["content"] == ""
    assert out[3]["role"] == "user"


def test_template_kwargs():
    assert template_kwargs({"reasoning_effort": "high"}) == {"reasoning_effort": "xhigh", "enable_thinking": True}
    assert template_kwargs({"reasoning_effort": "none"}) == {"enable_thinking": False}
    assert template_kwargs({"chat_template_kwargs": {"enable_thinking": False}}) == {"enable_thinking": False}
    assert anthropic_template_kwargs({"thinking": {"type": "disabled"}}) == {"enable_thinking": False}
    assert anthropic_template_kwargs({"thinking": {"type": "enabled", "budget_tokens": 1024}}) == {"enable_thinking": True}
    assert anthropic_template_kwargs({}) == {"enable_thinking": True}


def test_anthropic_messages():
    body = {"system": [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object", "properties": {}}},
                      {"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                                                  {"type": "text", "text": "Reading."},
                                                  {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                                   "input": {"path": "a.py"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                              "content": [{"type": "text", "text": "print(1)"}]},
                                             {"type": "text", "text": "<system-reminder>x</system-reminder>"},
                                             {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                          "data": "AAAA"}}]}]}
    msgs, tools = anthropic_messages(body)
    assert [t["function"]["name"] for t in tools] == ["read_file"]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[2] == {"role": "assistant", "content": "Reading.", "reasoning_content": "hmm",
                       "tool_calls": [{"id": "toolu_1", "type": "function",
                                       "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}
    assert msgs[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"}
    assert msgs[4]["role"] == "user" and msgs[4]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_anthropic_system_messages_and_omitted_thinking():
    from chat import thinking_signature
    body = {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1;"}, {"type": "text", "text": "S"}],
            "messages": [{"role": "user", "content": "q"}, {"role": "system", "content": [{"type": "text", "text": "ENV"}]},
                         {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": thinking_signature("why")},
                                                           {"type": "text", "text": "a"}]},
                         {"role": "system", "content": "LATE"}, {"role": "user", "content": "q2"}]}
    msgs, _ = anthropic_messages(body)
    assert msgs[0] == {"role": "system", "content": "S\n\nENV"}
    assert msgs[2] == {"role": "assistant", "content": "a", "reasoning_content": "why"}
    assert msgs[3] == {"role": "user", "content": "LATE"}


def test_responses_input():
    from chat import responses_messages, thinking_signature
    body = {"instructions": "SYS", "input": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "DEV"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": thinking_signature("think")},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Running."}]},
        {"type": "function_call", "call_id": "c1", "name": "spawn_agent", "namespace": "multi_agent_v1", "arguments": "{\"a\": 1}"},
        {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "*** Begin Patch"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": "done"}],
        "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object", "properties": {}}},
                  {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "spawn_agent", "parameters": {}}]},
                  {"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
                  {"type": "web_search"}]}
    msgs, tools, ns, custom = responses_messages(body)
    out = normalize_messages(msgs)
    assert out[0] == {"role": "system", "content": "SYS\n\nDEV"}
    a = out[2]
    assert a["content"] == "Running." and a["reasoning_content"] == "think"
    assert [c["function"]["arguments"] for c in a["tool_calls"]] == [{"a": 1}, {"input": "*** Begin Patch"}]
    assert [m["role"] for m in out[3:]] == ["tool", "tool"]
    assert [t["function"]["name"] for t in tools] == ["exec_command", "spawn_agent", "apply_patch"]
    assert ns == {"spawn_agent": "multi_agent_v1"} and custom == {"apply_patch"}
