"""Structured output (docs/architecture.md §12): a request's output format as an llguidance grammar.

The API server reads the format (OpenAI's response_format, the Responses API's text.format, vLLM's structured_outputs,
Anthropic's output_format / output_config.format) into a spec, makes the grammar and checks it (an error is the
client's: 400). The engine process (worker.py) matches the output against it and masks the logits of each verify row
with what the grammar allows there.

Specs: ("json", schema or None for any JSON object), ("regex", pattern), ("choice", [strings]), ("grammar", Lark or
GBNF text).
"""

import json
import re

import llguidance
from llguidance.gbnf_to_lark import gbnf_to_lark

# JSON: any whitespace between the tokens (the model's own layout, as vLLM's guidance backend), at most this much in a
# row (a model that only writes newlines would fill max_tokens with them).
JSON_OPTIONS = {"whitespace_flexible": True, "whitespace_pattern": r"[ \t\n\r]{0,32}"}


def _json_spec(schema, what):
    if schema is None:
        return ("json", None)
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            raise ValueError(f"{what}: the schema is not valid JSON") from None
    if not isinstance(schema, dict):
        raise ValueError(f"{what}: the schema must be a JSON object")
    return ("json", schema)


def _openai_format(fmt, what, nested):
    """response_format (chat: {"type", "json_schema": {"name", "schema", "strict"}}) or text.format (Responses:
    {"type", "name", "schema", "strict"}; nested=False)."""
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ValueError(f"{what}: an object is required")
    kind = fmt.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return ("json", None)
    if kind == "json_schema":
        js = fmt.get("json_schema") if nested else fmt
        if not isinstance(js, dict):
            raise ValueError(f"{what}: json_schema is required")
        return _json_spec(js.get("schema"), what)
    raise ValueError(f"{what}: unknown type {kind!r}")


def _vllm_format(so):
    """vLLM's structured_outputs: one of json, regex, choice, grammar, json_object."""
    if so is None:
        return None
    if not isinstance(so, dict):
        raise ValueError("structured_outputs: an object is required")
    given = [k for k in ("json", "regex", "choice", "grammar", "json_object") if so.get(k) not in (None, False)]
    unknown = [k for k in ("structural_tag",) if so.get(k) is not None]
    if unknown:
        raise ValueError(f"structured_outputs: {unknown[0]} is not supported")
    if not given:
        return None
    if len(given) > 1:
        raise ValueError(f"structured_outputs: only one of {', '.join(given)} may be given")
    k = given[0]
    v = so[k]
    if k == "json":
        return _json_spec(v, "structured_outputs.json")
    if k == "json_object":
        return ("json", None)
    if k == "choice":
        if not isinstance(v, list) or not v or not all(isinstance(c, str) and c for c in v):
            raise ValueError("structured_outputs.choice: a list of non-empty strings is required")
        return ("choice", v)
    if not isinstance(v, str) or not v:
        raise ValueError(f"structured_outputs.{k}: a string is required")
    return (k, v)


def chat_format(body):
    """The output format of a chat or completions request (OpenAI response_format or vLLM structured_outputs)."""
    a = _openai_format(body.get("response_format"), "response_format", nested=True)
    b = _vllm_format(body.get("structured_outputs"))
    if a and b:
        raise ValueError("give either response_format or structured_outputs, not both")
    return a or b


def responses_format(body):
    text = body.get("text")
    if text is None:
        return None
    if not isinstance(text, dict):
        raise ValueError("text: an object is required")
    return _openai_format(text.get("format"), "text.format", nested=False)


def anthropic_format(body):
    """Anthropic's structured outputs: output_config.format, or output_format (the beta's name)
    ({"type": "json_schema", "schema": {...}})."""
    cfg = body.get("output_config")
    fmt = cfg.get("format") if isinstance(cfg, dict) else None
    if fmt is None:
        fmt = body.get("output_format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise ValueError("output_config.format: {\"type\": \"json_schema\", \"schema\": {...}} is required")
    return ("json", fmt["schema"])


def grammar(spec, lead_ws=False, calls=None):
    """The llguidance grammar of a spec. lead_ws: whitespace may come first (after the reasoning, the template's blank
    line). calls: (id of <tool_call>, id of </tool_call>) when the model may call tools instead (tool_choice auto):
    the output is then the format or tool calls. Raises ValueError for a format llguidance cannot take."""
    kind, v = spec
    rules = ""
    if kind == "json":
        schema = {"type": "object"} if v is None else dict(v)
        schema["x-guidance"] = dict(JSON_OPTIONS, **(schema.get("x-guidance") or {}))
        out = "%json " + json.dumps(schema, ensure_ascii=False)
    elif kind == "regex":
        out = "/" + v.replace("/", "\\/") + "/"
    elif kind == "choice":
        out = " | ".join(json.dumps(c, ensure_ascii=False) for c in v)
    else:  # Lark as llguidance reads it, or GBNF (xgrammar's grammars): its start rule becomes user_start
        text = gbnf_to_lark(v) if "::=" in v else v
        text = re.sub(r"(?m)^%llguidance \{\}$", "", text)  # (gbnf_to_lark's header: no options)
        rules, n = re.subn(r"(?m)^(\s*)start(\s*):", r"\1user_start\2:", text)
        if n != 1:
            raise ValueError("structured_outputs.grammar: one start rule is required")
        out = "user_start"
    lark = ("start: " + ("/[ \\t\\r\\n]*/ " if lead_ws else "") + "body\n"
            + "body: out" + (" | calls" if calls else "") + "\n"
            + "out: " + out + "\n")
    if calls:
        a, b = calls
        lark += f"calls: <[{a}]> (<[{a}]> | <[{b}]> | /(.|\\n)+/)*\n"
    lark += rules
    g = llguidance.LLMatcher.grammar_from_lark(lark)
    failed, msgs = llguidance.LLMatcher.validate_grammar_with_warnings(g)
    if failed:
        raise ValueError(f"the output format cannot be used: {msgs[0]}")
    return g
