#!/usr/bin/env python3
"""API compatibility and concurrency test of the server (Phase 6, stdlib only).

OpenAI: streamed tool calls, include_usage, stop strings, completions. Anthropic: messages (non-streaming and
streaming events), thinking on / off, tool_use -> tool_result turn, stop_sequences, count_tokens, error format.
tool_choice: a call forced (a named tool, any tool) where the model would answer in text, in the OpenAI chat,
Anthropic and Responses forms, with thinking on and off.
structured: response_format (json_schema with thinking on and off, json_object streamed), vLLM structured_outputs
(choice, regex, completions), Responses text.format, Anthropic output_config.format, JSON or a tool call with tools,
a bad schema (400), and three masked requests next to a free one giving the text each gives alone.
controls: logit_bias (a banned answer, 400 on a bad one), vLLM min_tokens and ignore_eos, Responses logprobs (include
message.output_text.logprobs, plain and streamed), and biased requests next to a free one giving the text each gives
alone.
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


def timings_ok(t, usage):
    """llama.cpp's timings: the counts agree with usage, and it took time."""
    return bool(t) and t["prompt_n"] + t["cache_n"] == usage["prompt_tokens"] and t["predicted_n"] == usage["completion_tokens"] \
        and t["prompt_ms"] > 0 and t["predicted_ms"] > 0 and t["prompt_per_second"] > 0 and t["predicted_per_second"] > 0 \
        and ("draft_n" not in t or (t["draft_n"] % 7 == 0 and 0 <= t["draft_n_accepted"] <= t["draft_n"]))


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
    check("openai stream: timings on the last chunk only", timings_ok(ev[-1][1].get("timings"), usage[0] if usage else {})
          and sum("timings" in e for _, e in ev) == 1, json.dumps(ev[-1][1])[:300])
    ev = cl.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32,
                                            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    check("openai stream: timings without include_usage, on the finish_reason chunk",
          "timings" in ev[-1][1] and ev[-1][1]["choices"][0]["finish_reason"] is not None
          and sum("timings" in e for _, e in ev) == 1, json.dumps(ev[-1][1])[:300])

    st, r = cl.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Count from 1 to 20, comma separated."}],
                                             "max_tokens": 200, "temperature": 0, "stop": ["7"],
                                             "chat_template_kwargs": {"enable_thinking": False}})
    content = r["choices"][0]["message"]["content"] if st == 200 else r
    check("openai: stop string ends the text before it", st == 200 and "7" not in content and "6" in content
          and r["choices"][0]["finish_reason"] == "stop", repr(content))
    check("openai: timings with a stop string", st == 200 and timings_ok(r.get("timings"), r["usage"]), str(r.get("timings")))

    st, r = cl.post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    check("openai: completions", st == 200 and "Paris" in r["choices"][0]["text"], repr(r["choices"][0]["text"]) if st == 200 else r)
    check("openai: completions timings", st == 200 and timings_ok(r.get("timings"), r["usage"]), str(r.get("timings")))

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


def tool_choice_tests(cl):
    """A question the model answers without tools: the tool_choice forces a call anyway."""
    tools = TOOLS + [{"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
                                                       "parameters": {"type": "object", "properties": {
                                                           "path": {"type": "string"}}, "required": ["path"]}}}]
    q = [{"role": "user", "content": "What is 17 + 25? Answer with the number only."}]
    named = {"type": "function", "function": {"name": "list_dir"}}

    def metric(name):
        _, text = cl.get("/metrics")
        return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name))

    forced0 = metric("q27:forced_outputs_total")
    st, r = cl.post("/v1/chat/completions", {"messages": q, "tools": tools, "tool_choice": named, "max_tokens": 2048,
                                             "temperature": 0, "logprobs": True, "top_logprobs": 2})
    ch = r["choices"][0] if st == 200 else {}
    msg, calls = ch.get("message", {}), ch.get("message", {}).get("tool_calls") or []
    check("tool_choice: openai named, thinking on: one call of that tool after the reasoning",
          st == 200 and len(calls) == 1 and calls[0]["function"]["name"] == "list_dir" and bool(msg.get("reasoning_content"))
          and not msg.get("content") and ch["finish_reason"] == "tool_calls", json.dumps(r)[:400])
    check("tool_choice: logprobs cover the forced tokens too",
          st == 200 and len(ch["logprobs"]["content"]) == r["usage"]["completion_tokens"]
          and any(e["logprob"] == 0.0 and e["token"] == "<tool_call>" for e in ch["logprobs"]["content"]),
          f"{len((ch.get('logprobs') or {}).get('content') or [])} entries, usage {r.get('usage')}")
    check("tool_choice: the forced output counted (q27:forced_outputs_total)", metric("q27:forced_outputs_total") == forced0 + 1)

    ev = cl.stream("/v1/chat/completions", {"messages": q, "tools": tools, "tool_choice": "required", "max_tokens": 512,
                                            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    calls = [tc for _, e in ev for c in e.get("choices", []) for tc in c["delta"].get("tool_calls", [])]
    finish = [c["finish_reason"] for _, e in ev for c in e.get("choices", []) if c.get("finish_reason")]
    check("tool_choice: openai required, stream, thinking off: a call of one of the tools",
          bool(calls) and calls[0]["function"]["name"] in ("read_file", "list_dir") and finish == ["tool_calls"],
          json.dumps(calls)[:300])

    st, r = cl.post("/v1/chat/completions", {"messages": q, "tools": tools, "tool_choice": "none", "max_tokens": 512,
                                             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    check("tool_choice: none answers in text", st == 200 and "42" in (r["choices"][0]["message"]["content"] or "")
          and not r["choices"][0]["message"].get("tool_calls"), json.dumps(r)[:300])

    st, r = cl.post("/v1/chat/completions", {"messages": q, "tools": tools, "max_tokens": 8,
                                             "tool_choice": {"type": "function", "function": {"name": "write_file"}}})
    check("tool_choice: an unknown tool gives 400", st == 400 and "write_file" in r["error"]["message"], str(r))

    a_tools = A_TOOLS + [{"name": "list_dir", "description": "List a directory.",
                          "input_schema": tools[1]["function"]["parameters"]}]
    base = {"model": "x", "max_tokens": 2048, "temperature": 0, "messages": q, "tools": a_tools}
    st, r = cl.post("/v1/messages", dict(base, tool_choice={"type": "any"}, thinking={"type": "enabled", "budget_tokens": 1024}))
    blocks = r.get("content", []) if st == 200 else []
    check("tool_choice: anthropic any, thinking on: thinking then tool_use",
          st == 200 and [b["type"] for b in blocks][:1] == ["thinking"] and any(b["type"] == "tool_use" for b in blocks)
          and r["stop_reason"] == "tool_use", json.dumps(r)[:400])
    ev = cl.stream("/v1/messages", dict(base, tool_choice={"type": "tool", "name": "read_file"}, thinking={"type": "disabled"}))
    starts = [e["content_block"] for _, e in ev if e.get("type") == "content_block_start"]
    stop = [e["delta"]["stop_reason"] for _, e in ev if e.get("type") == "message_delta"]
    check("tool_choice: anthropic tool, stream, thinking off: one tool_use of that tool",
          [b["type"] for b in starts] == ["tool_use"] and starts[0]["name"] == "read_file" and stop == ["tool_use"],
          json.dumps(starts)[:300])

    r_tools = [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
                "parameters": t["function"]["parameters"]} for t in tools]
    st, r = cl.post("/v1/responses", {"input": q[0]["content"], "tools": r_tools, "max_output_tokens": 2048, "temperature": 0,
                                      "tool_choice": {"type": "function", "name": "list_dir"}})
    items = r.get("output", []) if st == 200 else []
    calls = [it for it in items if it["type"] == "function_call"]
    check("tool_choice: responses named function: one function_call of that tool",
          st == 200 and len(calls) == 1 and calls[0]["name"] == "list_dir" and r["status"] == "completed", json.dumps(r)[:400])


def structured_tests(cl):
    """Prompts that would not give the format on their own: the masks make the output follow it."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
                                               "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
              "required": ["city", "population", "landmarks"], "additionalProperties": False}
    q = [{"role": "user", "content": "Tell me about Kyoto in two sentences."}]
    rf = {"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}}

    def fits(text):
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            return False
        return set(obj) == {"city", "population", "landmarks"} and isinstance(obj["population"], int)

    for thinking in (False, True):
        st, r = cl.post("/v1/chat/completions", {"messages": q, "response_format": rf, "max_tokens": 4096, "temperature": 0,
                                                 "chat_template_kwargs": {"enable_thinking": thinking}})
        msg = r["choices"][0]["message"] if st == 200 else {}
        check(f"structured: openai json_schema, thinking {'on' if thinking else 'off'}",
              st == 200 and fits(msg.get("content")) and r["choices"][0]["finish_reason"] == "stop"
              and (bool(msg.get("reasoning_content")) == thinking), json.dumps(r)[:400])
    ev = cl.stream("/v1/chat/completions", {"messages": q, "response_format": {"type": "json_object"}, "max_tokens": 1024,
                                            "temperature": 0.7, "seed": 3, "chat_template_kwargs": {"enable_thinking": False}})
    text = "".join(c["delta"].get("content") or "" for _, e in ev for c in e.get("choices", []))
    try:
        good = isinstance(json.loads(text), dict)
    except ValueError:
        good = False
    check("structured: openai json_object, stream, sampled: a JSON object", good, text[:300])
    q2 = [{"role": "user", "content": "Is Kyoto the capital of Japan? Explain."}]
    st, r = cl.post("/v1/chat/completions", {"messages": q2, "structured_outputs": {"choice": ["yes", "no"]}, "max_tokens": 2048,
                                             "temperature": 0})
    check("structured: vllm choice", st == 200 and r["choices"][0]["message"]["content"] in ("yes", "no"), json.dumps(r)[:300])
    st, r = cl.post("/v1/completions", {"prompt": "The date of the Meiji Restoration was", "max_tokens": 32, "temperature": 0,
                                        "structured_outputs": {"regex": " [0-9]{4}-[0-9]{2}-[0-9]{2}"}})
    text = r["choices"][0]["text"] if st == 200 else ""
    check("structured: vllm regex, completions", st == 200 and len(text) == 11 and text[5] == "-" and text[8] == "-", text)
    st, r = cl.post("/v1/responses", {"input": q[0]["content"], "max_output_tokens": 4096, "temperature": 0,
                                      "text": {"format": {"type": "json_schema", "name": "city", "schema": schema}}})
    out = [c["text"] for it in r.get("output", []) if it["type"] == "message" for c in it["content"]] if st == 200 else []
    check("structured: responses text.format", st == 200 and len(out) == 1 and fits(out[0]), json.dumps(r)[:400])
    st, r = cl.post("/v1/messages", {"model": "x", "max_tokens": 4096, "temperature": 0, "messages": q,
                                     "output_config": {"format": {"type": "json_schema", "schema": schema}}})
    texts = [b["text"] for b in r.get("content", []) if b["type"] == "text"] if st == 200 else []
    check("structured: anthropic output_config.format", st == 200 and len(texts) == 1 and fits(texts[0]), json.dumps(r)[:400])
    q3 = [{"role": "user", "content": "Read README.md and tell me what it says."}]
    st, r = cl.post("/v1/chat/completions", {"messages": q3, "tools": TOOLS, "response_format": rf, "max_tokens": 4096,
                                             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    msg = r["choices"][0]["message"] if st == 200 else {}
    check("structured: with tools, a tool call or the format",
          st == 200 and (bool(msg.get("tool_calls")) or fits(msg.get("content"))), json.dumps(r)[:400])
    st, r = cl.post("/v1/chat/completions", {"messages": q, "max_tokens": 8,
                                             "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "nonsense"}}}})
    check("structured: a bad schema gives 400", st == 400 and "nonsense" in r["error"]["message"], str(r))
    # Masked and free requests at once (batched steps) give the greedy text each gives alone.
    bodies = [{"messages": q, "response_format": rf, "max_tokens": 256, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}},
              {"messages": [{"role": "user", "content": "Write a haiku about Kyoto."}], "max_tokens": 128, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}},
              {"messages": q2, "structured_outputs": {"choice": ["yes", "no"]}, "max_tokens": 64, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}},
              {"messages": [{"role": "user", "content": "List five fruits."}], "response_format": {"type": "json_object"},
               "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}]

    def text_of(b):
        st, r = cl.post("/v1/chat/completions", b)
        return r["choices"][0]["message"]["content"] if st == 200 else f"error {st}"
    alone = [text_of(b) for b in bodies]
    with cf.ThreadPoolExecutor(len(bodies)) as ex:
        together = list(ex.map(text_of, bodies))
    check("structured: 3 masked + 1 free at once give the text of each alone", alone == together,
          json.dumps([a[:80] for a in alone]) + " vs " + json.dumps([t[:80] for t in together]))


def controls_tests(cl):
    off = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
    q = [{"role": "user", "content": "What is the capital of France? Answer with the city's name only."}]

    def chat(**kw):
        st, r = cl.post("/v1/chat/completions", dict(off, messages=q, max_tokens=32, **kw))
        return st, r, (r["choices"][0]["message"]["content"] if st == 200 else "")
    _, _, base = chat()
    ban = {}
    for word in ("Paris", " Paris", "PAR", "Par"):
        _, t = cl.post("/tokenize", {"prompt": word})
        ban[str(t["tokens"][0])] = -100
    st, r, text = chat(logit_bias=ban)
    check("controls: logit_bias bans the answer's first token", st == 200 and "Paris" in base and not text.startswith("Paris"),
          f"{base!r} -> {text!r}")
    st, r, _ = chat(logit_bias={"999999999": 1})
    check("controls: logit_bias out of the vocabulary gives 400", st == 400, str(r)[:200])
    q2 = [{"role": "user", "content": "Say hi."}]
    st, r = cl.post("/v1/chat/completions", dict(off, messages=q2, max_tokens=64))
    n0 = r["usage"]["completion_tokens"] if st == 200 else 0
    st, r = cl.post("/v1/chat/completions", dict(off, messages=q2, max_tokens=64, min_tokens=24))
    n1 = r["usage"]["completion_tokens"] if st == 200 else 0
    check("controls: min_tokens", st == 200 and n0 < 24 <= n1, f"{n0} -> {n1} tokens")
    st, r = cl.post("/v1/chat/completions", dict(off, messages=q2, max_tokens=40, ignore_eos=True))
    msg = r["choices"][0] if st == 200 else {}
    check("controls: ignore_eos runs to max_tokens without the end's text",
          st == 200 and r["usage"]["completion_tokens"] == 40 and msg["finish_reason"] == "length"
          and "<|im_end|>" not in (msg["message"]["content"] or ""), json.dumps(r)[:300])
    st, r = cl.post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 24, "temperature": 0,
                                        "ignore_eos": True, "min_tokens": 24})
    check("controls: completions min_tokens / ignore_eos", st == 200 and r["usage"]["completion_tokens"] == 24, json.dumps(r)[:200])
    # Responses logprobs
    body = {"input": "Name three colors.", "max_output_tokens": 64, "temperature": 0, "reasoning": {"effort": "none"},
            "include": ["message.output_text.logprobs"], "top_logprobs": 2}
    st, r = cl.post("/v1/responses", body)
    parts = [c for it in r.get("output", []) if it["type"] == "message" for c in it["content"]] if st == 200 else []
    lps = parts[0]["logprobs"] if parts else []
    check("controls: responses logprobs", st == 200 and lps and "".join(e["token"] for e in lps) == parts[0]["text"]
          and all(len(e["top_logprobs"]) == 2 and e["logprob"] <= 0 for e in lps), json.dumps(r)[:300])
    ev = cl.stream("/v1/responses", body)
    deltas = [e for _, e in ev if e.get("type") == "response.output_text.delta"]
    text = "".join(d["delta"] for d in deltas)
    check("controls: responses logprobs, streamed", deltas and "".join(x["token"] for d in deltas for x in d["logprobs"]) == text
          and parts and text == parts[0]["text"], text[:200])
    st, r = cl.post("/v1/responses", dict(body, include=[]))
    parts = [c for it in r.get("output", []) if it["type"] == "message" for c in it["content"]] if st == 200 else []
    check("controls: responses without the include: no logprobs", st == 200 and parts and parts[0]["logprobs"] == [])
    # Biased and free requests at once (batched steps) give the text each gives alone.
    bodies = [dict(off, messages=q, max_tokens=32, logit_bias=ban),
              dict(off, messages=[{"role": "user", "content": "Write a haiku about Kyoto."}], max_tokens=96),
              dict(off, messages=q2, max_tokens=48, min_tokens=30),
              dict(off, messages=[{"role": "user", "content": "List five fruits."}], max_tokens=96,
                   logit_bias={k: -5 for k in ban})]

    def text_of(b):
        st, r = cl.post("/v1/chat/completions", b)
        return r["choices"][0]["message"]["content"] if st == 200 else f"error {st}"
    alone = [text_of(b) for b in bodies]
    with cf.ThreadPoolExecutor(len(bodies)) as ex:
        together = list(ex.map(text_of, bodies))
    check("controls: 3 biased + 1 free at once give the text of each alone", alone == together,
          json.dumps([a[:80] for a in alone]) + " vs " + json.dumps([t[:80] for t in together]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default="openai,anthropic,tool_choice,structured,controls,concurrency,cancel")
    args = ap.parse_args()
    cl = Client(args.base_url)
    st, models = cl.get("/v1/models")
    m = json.loads(models)["data"][0]
    check("/v1/models", st == 200 and m["id"] == "qwen3.8-27b" and m["max_model_len"] > 0, f"{m['id']} {m['max_model_len']}")
    for part in args.only.split(","):
        {"openai": openai_tests, "anthropic": anthropic_tests, "tool_choice": tool_choice_tests,
         "structured": structured_tests, "controls": controls_tests, "concurrency": concurrency_tests, "cancel": cancel_tests}[part](cl)
    print("PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
