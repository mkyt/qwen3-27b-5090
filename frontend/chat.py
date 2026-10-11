"""Request / response translation of the API server (no engine, no tokenizer): output parsing (reasoning in
`<think>`, qwen3_coder XML tool calls), OpenAI message normalization and the Anthropic Messages API mapping.

The model's output is parsed by one state machine (OutputParser) for streaming and non-streaming responses
of both APIs, so they always agree. The behaviour follows vLLM's qwen3 reasoning parser and qwen3_coder tool
parser (the existing environments): reasoning up to `</think>`, content up to the first `<tool_call>`,
each `<tool_call>` block becomes a call with its parameters converted by the tool's JSON schema.
"""

import ast
import base64
import json
import re
import uuid

THINK_END = "</think>"
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"


def new_id(prefix, n=24):
    return f"{prefix}{uuid.uuid4().hex[:n]}"


# ---------------------------------------------------------------- tool calls (qwen3_coder XML)


def _types(schema):
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    out = set()
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or []:
            out |= _types(sub)
    return out


def _convert(value, schema):
    """A parameter's text by its JSON schema: strings stay as they are, other types are parsed (JSON, then
    Python literals such as True / None, as vLLM's qwen3_coder parser does); unparsable text stays a string."""
    types = _types(schema)
    if types == {"string"}:
        return value
    v = value.strip()
    if "boolean" in types and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none") and ("null" in types or not types):
        return None
    try:
        out = json.loads(v)
    except (json.JSONDecodeError, ValueError):
        try:
            out = ast.literal_eval(v)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return value
    if "string" in types and not isinstance(out, (dict, list, bool, int, float)) and out is not None:
        return value
    if types and "string" in types and isinstance(out, (int, float)) and not ({"integer", "number"} & types):
        return value  # e.g. a path "123" of a string parameter
    return out


def tool_schemas(tools):
    """name -> {parameter: schema} of OpenAI-style tool definitions."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            out[fn["name"]] = ((fn.get("parameters") or {}).get("properties") or {})
    return out


def parse_call_block(block, schemas):
    """The calls of one `<tool_call>...</tool_call>` block: [{"name", "arguments" (dict)}]."""
    calls = []
    for m in re.finditer(r"<function=([^>\n]+)>(.*?)</function>", block, re.S):
        name, body = m.group(1).strip(), m.group(2)
        props = schemas.get(name, {})
        args = {}
        for p in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S):
            pname = p.group(1).strip()
            schema = props.get(pname)
            args[pname] = _convert(p.group(2), schema) if schema is not None else _convert(p.group(2), {})
        calls.append({"name": name, "arguments": args})
    return calls


def openai_call(call, index=None):
    out = {"id": new_id("call_"), "type": "function",
           "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
    if index is not None:
        out["index"] = index
    return out


def _partial_suffix(text, marker):
    """Length of the longest suffix of text that is a proper prefix of marker."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


class OutputParser:
    """The generated text, fed piece by piece, as events: ("reasoning", text), ("content", text) and
    ("tool_call", {"name", "arguments"}). Text that may still turn out to be part of a marker, and trailing
    whitespace (dropped before `</think>` and `<tool_call>`), is held back until it is known.

    thinking: the prompt ended in `<think>` (the output starts with reasoning). tools: tool definitions
    (None: tool calls are not parsed and stay in the content)."""

    def __init__(self, thinking, tools=None):
        self.mode = "reasoning" if thinking else "content"
        self.tools = tools is not None
        self.schemas = tool_schemas(tools)
        self.buf = ""
        self.lead = True  # at the start of a reasoning / content part (leading whitespace is dropped)
        self.ncalls = 0

    def _text(self, kind, upto, events, hold_ws=True):
        """Emits buf[:upto] as `kind` (keeping trailing whitespace back when hold_ws)."""
        seg = self.buf[:upto]
        if self.lead:
            seg = seg.lstrip()
            if seg:
                self.lead = False
        keep = seg[len(seg.rstrip()):] if hold_ws else ""
        emit = seg[:len(seg) - len(keep)]
        if emit:
            events.append((kind, emit))
        self.buf = keep + self.buf[upto:]

    def feed(self, text):
        self.buf += text
        events = []
        while True:
            if self.mode == "reasoning":
                i = self.buf.find(THINK_END)
                if i < 0:
                    self._text("reasoning", len(self.buf) - _partial_suffix(self.buf, THINK_END), events)
                    return events
                self._text("reasoning", i, events)
                self.buf = self.buf.lstrip()[len(THINK_END):] if self.buf.lstrip().startswith(THINK_END) else ""
                self.mode, self.lead = "content", True
            elif self.mode == "content":
                i = self.buf.find(TOOL_START) if self.tools else -1
                if i < 0:
                    hold = _partial_suffix(self.buf, TOOL_START) if self.tools else 0
                    self._text("content", len(self.buf) - hold, events)
                    return events
                self._text("content", i, events)
                self.buf = self.buf[self.buf.find(TOOL_START):]  # (held whitespace before the call is dropped)
                self.mode = "tool"
            elif self.mode == "tool":
                i = self.buf.find(TOOL_END)
                if i < 0:
                    return events
                block, self.buf = self.buf[:i + len(TOOL_END)], self.buf[i + len(TOOL_END):]
                calls = parse_call_block(block, self.schemas)
                if not calls:  # (not a call after all: keep the text)
                    events.append(("content", block))
                for c in calls:
                    self.ncalls += 1
                    events.append(("tool_call", c))
                self.mode = "after"
            else:  # "after": between / after calls only another call is expected
                rest = self.buf.lstrip()
                if not rest:
                    return events
                if rest.startswith(TOOL_START):
                    self.buf, self.mode = rest, "tool"
                elif TOOL_START.startswith(rest):
                    return events
                else:  # text after the calls: content
                    self.buf, self.mode, self.lead = rest, "content", False

    def finish(self):
        """Events for what is left at the end of the generation."""
        events = []
        if self.mode == "reasoning":
            self._text("reasoning", len(self.buf), events)
        elif self.mode == "content":
            self._text("content", len(self.buf), events, hold_ws=False)
        elif self.mode == "tool":
            # Cut off inside a call (max_tokens): a complete function still counts, otherwise it is text.
            calls = parse_call_block(self.buf, self.schemas) if "</function>" in self.buf else []
            for c in calls:
                self.ncalls += 1
                events.append(("tool_call", c))
            if not calls and self.buf.strip():
                events.append(("content", self.buf))
        self.buf = ""
        return events


def parse_output(text, thinking, tools=None):
    """(reasoning or None, content, [calls]) of a whole output."""
    p = OutputParser(thinking, tools)
    reasoning, content, calls = [], [], []
    for kind, v in p.feed(text) + p.finish():
        if kind == "reasoning":
            reasoning.append(v)
        elif kind == "content":
            content.append(v)
        else:
            calls.append(v)
    return ("".join(reasoning) if thinking else None), "".join(content), calls


def split_reasoning(text, thinking):
    """(reasoning, content) of an output (tool calls stay in the content)."""
    r, c, _ = parse_output(text, thinking, None)
    return r, c


def parse_tool_calls(content, tools):
    """(content before the calls, [OpenAI tool calls]) of the content part of an output."""
    _, c, calls = parse_output(content, False, tools or [])
    return c, [openai_call(x) for x in calls]


# ---------------------------------------------------------------- OpenAI requests


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None))
    return "" if content is None else str(content)


def normalize_messages(messages):
    """OpenAI chat messages as the chat template takes them: tool-call arguments as mappings (OpenAI sends
    JSON strings), the reasoning of earlier turns as `reasoning_content` (also accepted: `reasoning`),
    `developer` as `system`, leading system messages merged, later ones as user messages (the template
    only takes a system message first), `function` as `tool`."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError("each message needs a role")
        m = dict(m)
        role = {"developer": "system", "function": "tool"}.get(m["role"], m["role"])
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown message role {m['role']!r}")
        m["role"] = role
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if isinstance(p, str):
                    parts.append({"type": "text", "text": p})
                elif isinstance(p, dict) and p.get("type") in ("input_text", "output_text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") == "input_image":  # (Responses API style)
                    parts.append({"type": "image_url", "image_url": {"url": p.get("image_url")}})
                else:
                    parts.append(p)
            m["content"] = parts
        if role == "system":
            m["content"] = _text_of(m.get("content"))
            if out and out[-1]["role"] == "system" and len(out) == 1:
                out[-1] = dict(out[-1], content=out[-1]["content"] + "\n\n" + m["content"])
                continue
            if out:
                m["role"] = "user"
        if role == "assistant":
            r = m.get("reasoning_content")
            if not isinstance(r, str):
                r = m.get("reasoning")
            if isinstance(r, str):
                m["reasoning_content"] = r
            if m.get("content") is None:
                m["content"] = ""
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    fn = dict(c.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except json.JSONDecodeError as e:
                            raise ValueError(f"tool call arguments are not JSON: {e}") from e
                    calls.append(dict(c, function=fn))
                m["tool_calls"] = calls
        if role == "tool" and m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


REASONING_EFFORT = {"low": "low", "minimal": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh",
                    "max": "xhigh"}


def template_kwargs(body, thinking_default=True):
    """chat_template_kwargs of a request (OpenAI: `chat_template_kwargs`, `reasoning_effort`)."""
    kw = dict(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and "reasoning_effort" not in kw and "enable_thinking" not in kw:
        if effort == "none":
            kw["enable_thinking"] = False
        elif effort in REASONING_EFFORT:
            kw["reasoning_effort"] = REASONING_EFFORT[effort]
    if "reasoning_effort" in kw:
        kw["reasoning_effort"] = REASONING_EFFORT.get(kw["reasoning_effort"], kw["reasoning_effort"])
    kw.setdefault("enable_thinking", thinking_default)
    return kw


# ---------------------------------------------------------------- Anthropic Messages API


class AnthropicError(ValueError):
    pass


def _image_url(source):
    if not isinstance(source, dict):
        raise AnthropicError("image source must be an object")
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if source.get("type") == "url":
        return source.get("url")
    raise AnthropicError(f"unsupported image source type {source.get('type')!r}")


def _user_parts(blocks, where):
    """Text / image blocks as OpenAI content parts."""
    parts = []
    for b in blocks:
        if isinstance(b, str):
            parts.append({"type": "text", "text": b})
            continue
        t = b.get("type")
        if t == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        elif t == "image":
            parts.append({"type": "image_url", "image_url": {"url": _image_url(b.get("source"))}})
        elif t == "document":
            src = b.get("source") or {}
            if src.get("type") == "text":
                parts.append({"type": "text", "text": src.get("data", "")})
            elif src.get("type") == "content":
                parts += _user_parts(src.get("content") or [], where)
            else:
                raise AnthropicError(f"{where}: only text documents are supported")
        elif t in ("thinking", "redacted_thinking"):
            continue
        else:
            raise AnthropicError(f"{where}: unsupported content block type {t!r}")
    return parts


def _flat(parts):
    """A list of only text parts as one string (keeps the rendering of plain text as in other clients)."""
    if all(p.get("type") == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def anthropic_tools(tools):
    """Custom tools as OpenAI function tools (server tools such as web search have no input schema and are
    left out: the model cannot run them)."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or not t.get("name") or ("input_schema" not in t and t.get("type") not in (None, "custom")):
            continue
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                     "parameters": t.get("input_schema") or {"type": "object"}}})
    return out


# Thinking with `display: "omitted"` (Claude Code): like Anthropic, the text is left out and an opaque
# signature carries it; sent back, the signature restores the reasoning, so the next turn renders the same
# tokens as generated (prefix cache).
SIGNATURE_PREFIX = "q27r1:"


def thinking_signature(reasoning):
    return SIGNATURE_PREFIX + base64.b64encode(reasoning.encode()).decode()


def _thinking_of(block):
    text, sig = block.get("thinking") or "", block.get("signature") or ""
    if not text and isinstance(sig, str) and sig.startswith(SIGNATURE_PREFIX):
        try:
            return base64.b64decode(sig[len(SIGNATURE_PREFIX):]).decode()
        except (ValueError, UnicodeDecodeError):
            return ""
    return text


def _system_text(system):
    if isinstance(system, list):
        # (Claude Code's first block is client metadata, "x-anthropic-billing-header: cc_version=...;")
        return "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type", "text") == "text"
                           and not b.get("text", "").startswith("x-anthropic-billing-header:"))
    return system or ""


def anthropic_messages(body):
    """(OpenAI-style messages, tools) of an Anthropic Messages request. System messages inside `messages`
    (Claude Code sends its environment that way) join the system prompt while no assistant turn came before
    them, and are user turns after that (the template takes one system message, first; moving later ones up
    would change the start of the prompt and miss the prefix cache)."""
    msgs = []
    system = _system_text(body.get("system"))
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise AnthropicError("messages: at least one message is required")
    seen_assistant = False
    for i, m in enumerate(body["messages"]):
        role, content = m.get("role"), m.get("content")
        if role == "system":
            text = _system_text(content) if isinstance(content, list) else (content or "")
            if not seen_assistant:
                system = f"{system}\n\n{text}" if system else text
            else:
                msgs.append({"role": "user", "content": text})
            continue
        if role not in ("user", "assistant"):
            raise AnthropicError(f"messages.{i}.role must be user, assistant or system")
        seen_assistant |= role == "assistant"
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise AnthropicError(f"messages.{i}.content must be a string or a list")
        if role == "user":
            pending = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    c = _flat(_user_parts(c, f"messages.{i}.tool_result")) if isinstance(c, list) else (c or "")
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": c})
                else:
                    pending.append(b)
            parts = _user_parts(pending, f"messages.{i}")
            if parts:
                msgs.append({"role": "user", "content": _flat(parts)})
        else:
            reasoning, text, calls = [], [], []
            for b in content:
                t = b.get("type") if isinstance(b, dict) else None
                if t == "thinking":
                    reasoning.append(_thinking_of(b))
                elif t == "text":
                    text.append(b.get("text", ""))
                elif t == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
                elif t in ("redacted_thinking", "server_tool_use", "web_search_tool_result"):
                    continue
                else:
                    raise AnthropicError(f"messages.{i}: unsupported assistant content block type {t!r}")
            a = {"role": "assistant", "content": "".join(text)}
            if reasoning:
                a["reasoning_content"] = "\n\n".join(reasoning)
            if calls:
                a["tool_calls"] = calls
            msgs.append(a)
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs, anthropic_tools(body.get("tools"))


def anthropic_template_kwargs(body, thinking_default=True):
    """`thinking` (enabled / adaptive / disabled) as enable_thinking; `chat_template_kwargs` wins."""
    kw = {}
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") in ("enabled", "adaptive", "disabled"):
        kw["enable_thinking"] = th["type"] != "disabled"
    effort = (body.get("output_config") or {}).get("effort") if isinstance(body.get("output_config"), dict) else None
    if isinstance(effort, str) and effort in REASONING_EFFORT:
        kw["reasoning_effort"] = REASONING_EFFORT[effort]
    kw.update(body.get("chat_template_kwargs") or {})
    kw.setdefault("enable_thinking", thinking_default)
    return kw


STOP_REASON = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "stop_sequence": "stop_sequence"}


# ---------------------------------------------------------------- OpenAI Responses API (Codex)


def _response_parts(content):
    """Responses content items as chat content parts (a plain string when it is only text)."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in ("input_text", "output_text", "text", "summary_text", "reasoning_text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t == "input_image":
            url = p.get("image_url")
            if not isinstance(url, str):
                raise ValueError("input_image: only image_url (a data URL or an http(s) URL) is supported")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif t == "input_file":
            raise ValueError("input_file is not supported")
        elif t == "refusal":
            parts.append({"type": "text", "text": p.get("refusal", "")})
    return _flat(parts) if parts else ""


def _reasoning_of_item(item):
    content = item.get("content") or []
    text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    if text:
        return text
    enc = item.get("encrypted_content")
    if isinstance(enc, str) and enc.startswith(SIGNATURE_PREFIX):
        return _thinking_of({"signature": enc})
    return "\n\n".join(s.get("text", "") for s in item.get("summary") or [] if isinstance(s, dict))


def responses_tools(tools):
    """(chat tools, {name: namespace}, {custom tool names}) of Responses tools: function tools, the functions of
    `namespace` groups (named as they are; calls go back with their namespace), custom (freeform) tools as
    functions of one string `input` (their grammar goes into the description). Hosted tools are left out."""
    out, namespaces, custom = [], {}, set()

    def function(t, ns=None):
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                     "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
        if ns:
            namespaces[t["name"]] = ns

    for t in tools or []:
        if not isinstance(t, dict):
            continue
        kind = t.get("type")
        if kind == "function" and t.get("name"):
            function(t)
        elif kind == "namespace":
            for sub in t.get("tools") or []:
                if isinstance(sub, dict) and sub.get("type", "function") == "function" and sub.get("name"):
                    function(sub, t.get("name"))
        elif kind == "custom" and t.get("name"):
            fmt = t.get("format") or {}
            desc = t.get("description", "")
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                desc += f"\n\nThe input is free text in this {fmt.get('syntax', '')} grammar:\n{fmt['definition']}"
            out.append({"type": "function", "function": {"name": t["name"], "description": desc, "parameters": {
                "type": "object", "properties": {"input": {"type": "string", "description": "The raw input text."}},
                "required": ["input"]}}})
            custom.add(t["name"])
    return out, namespaces, custom


def responses_messages(body):
    """(chat messages, tools, namespaces, custom tool names) of a Responses API request (`instructions` and
    `input`). Assistant-side items in a row (reasoning, an assistant message, calls) become one assistant turn."""
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ValueError("input: a string or a non-empty list of items is required")
    turn = None  # the assistant turn being collected

    def assistant():
        nonlocal turn
        if turn is None:
            turn = {"role": "assistant", "content": ""}
            msgs.append(turn)
        return turn

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"input.{i}: an item must be an object")
        kind = it.get("type", "message")
        if kind == "message":
            role = it.get("role")
            if role == "assistant":
                a = assistant()
                text = _response_parts(it.get("content"))
                a["content"] += text if isinstance(text, str) else "".join(p.get("text", "") for p in text)
                continue
            turn = None
            if role not in ("user", "system", "developer"):
                raise ValueError(f"input.{i}.role must be user, assistant, system or developer")
            msgs.append({"role": role, "content": _response_parts(it.get("content"))})
        elif kind == "reasoning":
            a = assistant()
            r = _reasoning_of_item(it)
            a["reasoning_content"] = (a.get("reasoning_content", "") + "\n\n" + r) if a.get("reasoning_content") else r
        elif kind in ("function_call", "custom_tool_call"):
            a = assistant()
            if kind == "function_call":
                args = it.get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError as e:
                    raise ValueError(f"input.{i}.arguments is not JSON: {e}") from e
            else:
                args = {"input": it.get("input", "")}
            a.setdefault("tool_calls", []).append({"id": it.get("call_id"), "type": "function",
                                                   "function": {"name": it.get("name"), "arguments": args}})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            turn = None
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": _response_parts(out) if isinstance(out, list) else (out or "")})
        elif kind in ("item_reference",):
            raise ValueError("item_reference is not supported (the server stores no responses)")
        else:  # hosted tool calls (web search, ...) have no counterpart here
            continue
    tools, namespaces, custom = responses_tools(body.get("tools"))
    return msgs, tools, namespaces, custom


def responses_template_kwargs(body, thinking_default=True):
    kw = dict(body.get("chat_template_kwargs") or {})
    effort = (body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else None
    if isinstance(effort, str) and "enable_thinking" not in kw and "reasoning_effort" not in kw:
        if effort == "none":
            kw["enable_thinking"] = False
        elif effort in REASONING_EFFORT:
            kw["reasoning_effort"] = REASONING_EFFORT[effort]
    kw.setdefault("enable_thinking", thinking_default)
    return kw
