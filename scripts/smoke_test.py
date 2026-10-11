#!/usr/bin/env python3
"""Functional smoke test for the agentic serving path (stdlib only).

Checks: server health, chat (thinking off), reasoning parser (thinking on),
tool-call parser, and a second turn that feeds a tool result back.

    python3 scripts/smoke_test.py [--base-url http://127.0.0.1:8000]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    }
]


def request(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def wait_ready(base, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/health", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(5)
    sys.exit(f"server not ready after {timeout_s}s")


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=int, default=1800, help="seconds to wait for /health")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    wait_ready(base, args.wait)
    model = request(base, "/v1/models")["data"][0]["id"]
    print(f"model: {model}")
    results = []

    t0 = time.time()
    r = request(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    msg = r["choices"][0]["message"]
    results.append(check("chat (thinking off)", "391" in (msg.get("content") or ""),
                         f"{msg.get('content')!r} in {time.time() - t0:.1f}s"))

    r = request(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Is 221 prime? Reply yes or no."}],
        "max_tokens": 4096,
    })
    msg = r["choices"][0]["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    results.append(check("reasoning parser (thinking on)",
                         bool(reasoning) and "<think>" not in (msg.get("content") or ""),
                         f"reasoning={len(reasoning)} chars, content={msg.get('content')!r}"))

    messages = [
        {"role": "system", "content": "You are a coding agent. Use tools to inspect files before answering."},
        {"role": "user", "content": "What does the file src/main.py print?"},
    ]
    r = request(base, "/v1/chat/completions", {
        "model": model, "messages": messages, "tools": TOOLS, "max_tokens": 2048,
    })
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "read_file"
    if ok:
        try:
            ok = "path" in json.loads(calls[0]["function"]["arguments"])
        except json.JSONDecodeError:
            ok = False
    results.append(check("tool call parser", ok, json.dumps(calls)[:200]))

    if ok:
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "reasoning_content": msg.get("reasoning") or msg.get("reasoning_content"),
            "tool_calls": calls,
        })
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": 'print("hello from qwen")\n'})
        r = request(base, "/v1/chat/completions", {
            "model": model, "messages": messages, "tools": TOOLS, "max_tokens": 2048,
        })
        content = r["choices"][0]["message"].get("content") or ""
        results.append(check("tool result turn", "hello from qwen" in content, repr(content[:120])))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
