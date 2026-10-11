#!/usr/bin/env python3
"""API compatibility and concurrency test of the server (Phase 6, stdlib only).

OpenAI: streamed tool calls, include_usage, stop strings, completions. Anthropic: messages (non-streaming and
streaming events), thinking on / off, tool_use -> tool_result turn, stop_sequences, count_tokens, error format.
Server: requests run at once give the same greedy text as one at a time (switching sequences is exact), a
closed connection cancels its request, /v1/models and /metrics.

    python3 scripts/api_test.py [--base-url http://127.0.0.1:8000] [--only concurrency]
"""

import argparse
import concurrent.futures as cf
import http.client
import json
import sys
import time
import urllib.parse
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a UTF-8 text file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path"},
                                                    "limit": {"type": "integer"}}, "required": ["path"]}}}]
A_TOOLS = [{"name": "read_file", "description": "Read a UTF-8 text file from the workspace.",
            "input_schema": TOOLS[0]["function"]["parameters"]}]
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


class Client:
    def __init__(self, base):
        u = urllib.parse.urlparse(base)
        self.host, self.port = u.hostname, u.port or 80

    def conn(self, timeout=600):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def post(self, path, body, headers=None):
        c = self.conn()
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data) if data else None

    def get(self, path):
        c = self.conn()
        c.request("GET", path)
        r = c.getresponse()
        data = r.read().decode()
        c.close()
        return r.status, data

    def stream(self, path, body, on_event=None):
        """SSE events as (event name or None, data dict); on_event can return False to stop reading."""
        c = self.conn()
        c.request("POST", path, json.dumps(dict(body, stream=True)), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"{r.status}: {r.read()[:300]}")
        events, name = [], None
        while True:
            line = r.fp.readline()
            if not line:
                break
            line = line.decode().rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                if line[6:] == "[DONE]":
                    break
                ev = (name, json.loads(line[6:]))
                events.append(ev)
                name = None
                if on_event and on_event(ev) is False:
                    c.close()
                    return events
        c.close()
        return events


def openai_tests(cl):
    msgs = [{"role": "user", "content": "Read src/main.py and tell me what it prints. Use the tool."}]
    ev = cl.stream("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": msgs, "tools": TOOLS,
                                            "max_tokens": 1024, "temperature": 0,
                                            "stream_options": {"include_usage": True}})
    calls = [tc for _, e in ev for ch in e.get("choices", []) for tc in ch["delta"].get("tool_calls", [])]
    finish = [ch["finish_reason"] for _, e in ev for ch in e.get("choices", []) if ch.get("finish_reason")]
    usage = [e["usage"] for _, e in ev if e.get("usage")]
    reasoning = "".join(ch["delta"].get("reasoning_content") or "" for _, e in ev for ch in e.get("choices", []))
    ok = bool(calls) and calls[0]["function"]["name"] == "read_file" and \
        json.loads(calls[0]["function"]["arguments"]).get("path") == "src/main.py"
    check("openai stream: tool call", ok, json.dumps(calls)[:200])
    check("openai stream: reasoning, finish_reason tool_calls, usage chunk",
          bool(reasoning) and finish == ["tool_calls"] and len(usage) == 1 and usage[0]["completion_tokens"] > 0,
          f"reasoning {len(reasoning)} chars, finish {finish}, usage {usage}")

    st, r = cl.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Count from 1 to 20, comma separated."}],
                                             "max_tokens": 200, "temperature": 0, "stop": ["7"],
                                             "chat_template_kwargs": {"enable_thinking": False}})
    content = r["choices"][0]["message"]["content"] if st == 200 else r
    check("openai: stop string ends the text before it", st == 200 and "7" not in content and "6" in content
          and r["choices"][0]["finish_reason"] == "stop", repr(content))

    st, r = cl.post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    check("openai: completions", st == 200 and "Paris" in r["choices"][0]["text"], repr(r["choices"][0]["text"]) if st == 200 else r)

    st, r = cl.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "n": 2})
    check("openai: unsupported n gives 400 in the OpenAI error format", st == 400 and "error" in r, str(r))


def anthropic_tests(cl):
    base = {"model": "claude-sonnet-4-5", "max_tokens": 2048, "temperature": 0}
    st, r = cl.post("/v1/messages", dict(base, messages=[{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                                         thinking={"type": "disabled"}), {"anthropic-version": "2023-06-01"})
    text = "".join(b.get("text", "") for b in r.get("content", [])) if st == 200 else ""
    check("anthropic: message (thinking disabled)", st == 200 and "391" in text and r["stop_reason"] == "end_turn"
          and r["type"] == "message" and not any(b["type"] == "thinking" for b in r["content"])
          and r["usage"]["output_tokens"] > 0, json.dumps(r)[:300])

    st, n = cl.post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                                                  "thinking": {"type": "disabled"}})
    u = r.get("usage", {})
    check("anthropic: count_tokens = input + cache read of the message", st == 200 and
          n["input_tokens"] == u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0), f"{n} vs {u}")

    # A tool turn, streamed, with thinking.
    msgs = [{"role": "user", "content": "Read src/main.py and tell me what it prints. Use the tool."}]
    ev = cl.stream("/v1/messages", dict(base, messages=msgs, tools=A_TOOLS, thinking={"type": "enabled", "budget_tokens": 1024}))
    names = [n for n, _ in ev]
    order_ok = names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    blocks, cur = [], None
    for n_, e in ev:
        if n_ == "content_block_start":
            cur = dict(e["content_block"])
            cur["_json"] = ""
            blocks.append(cur)
        elif n_ == "content_block_delta":
            d = e["delta"]
            if d["type"] == "thinking_delta":
                cur["thinking"] += d["thinking"]
            elif d["type"] == "text_delta":
                cur["text"] += d["text"]
            elif d["type"] == "input_json_delta":
                cur["_json"] += d["partial_json"]
    delta = [e for n_, e in ev if n_ == "message_delta"][0]
    tool = [b for b in blocks if b["type"] == "tool_use"]
    check("anthropic stream: event order, thinking block, tool_use with input_json_delta",
          order_ok and blocks and blocks[0]["type"] == "thinking" and blocks[0]["thinking"] and tool
          and json.loads(tool[0]["_json"]).get("path") == "src/main.py" and delta["delta"]["stop_reason"] == "tool_use",
          f"{names[:3]}..{names[-3:]} blocks {[b['type'] for b in blocks]} stop {delta['delta']}")

    # The tool result turn (the assistant turn sent back as Anthropic content blocks).
    assistant = []
    for b in blocks:
        b = {k: v for k, v in b.items() if k != "_json"}
        if b["type"] == "tool_use":
            b["input"] = json.loads(tool[0]["_json"])
        assistant.append(b)
    msgs2 = msgs + [{"role": "assistant", "content": assistant},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool[0]["id"],
                                                  "content": [{"type": "text", "text": 'print("hello from qwen")\n'}]}]}]
    st, r = cl.post("/v1/messages", dict(base, messages=msgs2, tools=A_TOOLS))
    text = "".join(b.get("text", "") for b in r.get("content", [])) if st == 200 else ""
    cached = r.get("usage", {}).get("cache_read_input_tokens", 0) if st == 200 else 0
    check("anthropic: tool_result turn answers and resumes from the cached prefix",
          st == 200 and "hello from qwen" in text and cached > 0, f"{text[:120]!r} cache_read {cached}")

    st, r = cl.post("/v1/messages", dict(base, messages=[{"role": "user", "content": "Count from 1 to 20, comma separated."}],
                                         stop_sequences=["7,"], thinking={"type": "disabled"}))
    text = "".join(b.get("text", "") for b in r.get("content", [])) if st == 200 else ""
    check("anthropic: stop_sequences", st == 200 and r["stop_reason"] == "stop_sequence" and r["stop_sequence"] == "7,"
          and "7," not in text, f"{text!r} {r.get('stop_reason')}")

    st, r = cl.post("/v1/messages", {"model": "x", "messages": [{"role": "user", "content": "x"}]})
    check("anthropic: missing max_tokens gives 400 in the Anthropic error format",
          st == 400 and r.get("type") == "error" and r["error"]["type"] == "invalid_request_error", str(r))


PROMPTS = ["Write a Python function that checks whether a number is prime, with a docstring.",
           "Explain in three sentences why the sky is blue.",
           "List five prime numbers greater than 100 as a JSON array, nothing else.",
           "日本の首都と、その人口の概算を一文で答えてください。"]


def one(cl, p, max_tokens=256):
    t0 = time.time()
    st, r = cl.post("/v1/chat/completions", {"messages": [{"role": "system", "content": "Be concise."},
                                                          {"role": "user", "content": p}],
                                             "max_tokens": max_tokens, "temperature": 0,
                                             "chat_template_kwargs": {"enable_thinking": False}})
    assert st == 200, r
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t0


def concurrency_tests(cl):
    for p in PROMPTS:  # (warms the prefix cache, so both runs below resume from the same checkpoints)
        one(cl, p)
    seq = [one(cl, p) for p in PROMPTS]
    t0 = time.time()
    with cf.ThreadPoolExecutor(len(PROMPTS)) as ex:
        par = list(ex.map(lambda p: one(cl, p), PROMPTS))
    wall = time.time() - t0
    same = [a[0] == b[0] for a, b in zip(seq, par)]
    tokens = sum(x[1] for x in par)
    check("4 requests at once give the same greedy text as one at a time", all(same),
          f"{same}; {tokens} tokens in {wall:.2f} s at once ({tokens / wall:.0f} tok/s) vs "
          f"{sum(x[2] for x in seq):.2f} s one at a time")
    # Eight at once: four run, four wait.
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda p: one(cl, p, 64), PROMPTS * 2))
    check("8 requests at once all complete", len(res) == 8 and all(r[1] > 0 for r in res))


def cancel_tests(cl):
    def metric(name):
        _, m = cl.get("/metrics")
        return sum(float(line.split()[-1]) for line in m.splitlines() if line.startswith(name))

    before = metric('q27:requests_finished_total{reason="cancelled"}')
    got = {"n": 0}

    def on_event(_):
        got["n"] += 1
        return got["n"] < 20  # close the connection after 20 chunks

    cl.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a very long story about a robot."}],
                                       "max_tokens": 4000, "chat_template_kwargs": {"enable_thinking": False}}, on_event)
    for _ in range(40):
        time.sleep(0.25)
        if metric('q27:requests_finished_total{reason="cancelled"}') > before and metric("q27:num_requests_running") == 0:
            break
    check("a closed stream cancels its request", metric('q27:requests_finished_total{reason="cancelled"}') > before
          and metric("q27:num_requests_running") == 0)

    # Non-streaming: the client gives up (timeout) while the request runs.
    c = cl.conn(timeout=2)
    before = metric('q27:requests_finished_total{reason="cancelled"}')
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Write a very long story."}],
                                                          "max_tokens": 4000,
                                                          "chat_template_kwargs": {"enable_thinking": False}}),
              {"Content-Type": "application/json"})
    try:
        c.getresponse()
    except (TimeoutError, OSError):
        pass
    c.close()
    for _ in range(40):
        time.sleep(0.25)
        if metric('q27:requests_finished_total{reason="cancelled"}') > before:
            break
    check("a closed non-streaming request is cancelled", metric('q27:requests_finished_total{reason="cancelled"}') > before)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default="openai,anthropic,concurrency,cancel")
    args = ap.parse_args()
    cl = Client(args.base_url)
    st, models = cl.get("/v1/models")
    m = json.loads(models)["data"][0]
    check("/v1/models", st == 200 and m["id"] == "qwen3.8-27b" and m["max_model_len"] > 0, f"{m['id']} {m['max_model_len']}")
    for part in args.only.split(","):
        {"openai": openai_tests, "anthropic": anthropic_tests, "concurrency": concurrency_tests,
         "cancel": cancel_tests}[part](cl)
    print("PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
