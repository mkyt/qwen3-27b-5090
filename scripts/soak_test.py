#!/usr/bin/env python3
"""Soak test of the server (P6-7, stdlib only; run from the host against `docker compose up -d serve`).

soak       W workers for M minutes, each picking a random scenario: agent sessions (chat completions with
           tools, a shared ~8K system prompt and a growing history), Anthropic messages and Responses turns,
           cancellations (closing the stream during prefill or decode), long unique prompts (24K-120K tokens,
           several at once overflow the KV pool and get suspended) and images. A sampler records /metrics,
           VRAM and the container's host memory. Before and after: the same greedy request must give the
           same text, and the server must end idle with the KV pool it started with.
longbatch  batched decode at long context: one request alone, then 4 at once, each with a unique prompt of
           --long-tokens tokens (prefill first, then decode together), decode tok/s per request and in total.

    python3 scripts/soak_test.py soak --minutes 30 --workers 6
    python3 scripts/soak_test.py longbatch --long-tokens 32768
"""

import argparse
import base64
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_test import Client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "qwen3-27b-5090"
WORDS = ("self value index result config request buffer token cache layer state return import class async await "
         "yield tensor batch schedule prefix block kernel worker engine server client stream parse render error "
         "retry path file module handler session context manager queue event loop thread offset length window").split()
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a UTF-8 text file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the workspace and return its output.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
METRIC_RE = re.compile(r"^(q27:[a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
GREEDY_CHECK = {"messages": [{"role": "user", "content": "Write a short Python function that merges two sorted lists, "
                                                          "then explain its complexity in two sentences."}],
                "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def filler(rng, approx_tokens):
    lines, n = [], 0
    while n < approx_tokens:
        w = [rng.choice(WORDS) for _ in range(rng.randint(5, 9))]
        lines.append(f"{'    ' * rng.randint(0, 2)}{w[0]}_{w[1]} = {w[2]}.{w[3]}({', '.join(w[4:])})")
        n += 1.83 * len(w)  # measured with the Qwen tokenizer
    return "\n".join(lines)


def data_url(name):
    raw = (ROOT / "bench" / "vision" / name).read_bytes()
    return f"data:{'image/png' if name.endswith('.png') else 'image/jpeg'};base64,{base64.b64encode(raw).decode()}"


def metrics(cl):
    st, text = cl.get("/metrics")
    out = {}
    for name, labels, v in METRIC_RE.findall(text):
        out[name + (labels or "")] = float(v)
    return out


def gpu_mem_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return None


def container_mem():
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
                             capture_output=True, text=True, timeout=30).stdout.split("/")[0].strip()
        num, unit = re.match(r"([0-9.]+)\s*([A-Za-z]+)", out).groups()
        return round(float(num) * {"B": 2**-30, "KiB": 2**-20, "MiB": 2**-10, "GiB": 1}[unit], 3)
    except Exception:
        return None


def stream_chat(cl, body, stop_after=None):
    """Streamed chat completion -> (text, reasoning, tool_calls, finish_reason, usage, ttft, seconds).
    stop_after: close the connection after that many seconds (cancellation)."""
    t0 = time.time()
    st = {"ttft": None, "text": "", "reasoning": "", "calls": {}, "finish": None, "usage": None}

    def on_event(ev):
        _, e = ev
        if e.get("usage"):
            st["usage"] = e["usage"]
        for ch in e.get("choices", []):
            d = ch.get("delta", {})
            if st["ttft"] is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                st["ttft"] = time.time() - t0
            st["text"] += d.get("content") or ""
            st["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = st["calls"].setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                c["name"] += (tc.get("function") or {}).get("name") or ""
                c["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            if ch.get("finish_reason"):
                st["finish"] = ch["finish_reason"]
        return not (stop_after and time.time() - t0 > stop_after)

    if stop_after:
        # The stream may not produce an event before the deadline (long prefill): close from a timer.
        holder = {}

        def run():
            c = cl.conn()
            holder["c"] = c
            c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            try:
                while True:
                    line = r.fp.readline()
                    if not line or line.startswith(b"data: [DONE]"):
                        break
                    if line.startswith(b"data: ") and on_event((None, json.loads(line[6:]))) is False:
                        break
            except Exception:
                pass
        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(stop_after)
        if "c" in holder:
            try:
                holder["c"].sock.shutdown(2)
            except Exception:
                pass
            holder["c"].close()
        th.join(30)
    else:
        cl.stream("/v1/chat/completions", dict(body, stream_options={"include_usage": True}), on_event)
    return st, time.time() - t0


class Soak:
    def __init__(self, args):
        self.args, self.cl = args, Client(args.base_url)
        self.lock = threading.Lock()
        self.records, self.errors, self.samples = [], [], []
        self.stop_at = time.time() + args.minutes * 60
        rng = random.Random(1)
        self.system = ("You are a coding agent working in a Python repository. Use the tools to inspect files and "
                       "run commands, then answer briefly.\n\nRepository overview:\n" + filler(rng, 8000))
        self.images = [data_url("chart.png"), data_url("astronaut.jpg")]

    def record(self, kind, ok, **kw):
        with self.lock:
            self.records.append(dict(kind=kind, ok=ok, t=round(time.time() - self.t0, 1), **kw))
            if not ok:
                self.errors.append(self.records[-1])
                print(f"[soak] FAIL {kind}: {kw}", flush=True)

    # --- scenarios -------------------------------------------------------------------------------------
    def agent(self, rng):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": f"Task {rng.randrange(10**6)}: find where `{rng.choice(WORDS)}_"
                                            f"{rng.choice(WORDS)}` is defined and summarize what it does."}]
        for turn in range(rng.randint(2, 5)):
            st, sec = stream_chat(self.cl, {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": 1024,
                                            "temperature": 1.0, "top_p": 0.95, "top_k": 20})
            u = st["usage"] or {}
            ok = st["finish"] in ("stop", "tool_calls", "length") and u.get("completion_tokens", 0) > 0
            self.record("agent", ok, turn=turn, finish=st["finish"], ttft=st["ttft"], sec=round(sec, 2),
                        prompt=u.get("prompt_tokens"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        out=u.get("completion_tokens"))
            if not ok or st["finish"] != "tool_calls":
                return
            calls = [st["calls"][i] for i in sorted(st["calls"])]
            msgs.append({"role": "assistant", "content": st["text"] or None, "reasoning_content": st["reasoning"],
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for c in calls:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": filler(rng, rng.randint(200, 2500))})

    def anthropic(self, rng):
        body = {"model": "claude-x", "max_tokens": 800, "system": self.system[:20000],
                "messages": [{"role": "user", "content": f"In about {rng.randint(3, 8)} sentences, explain how a "
                                                         f"{rng.choice(WORDS)} {rng.choice(WORDS)} could be cached."}],
                "thinking": {"type": "enabled", "budget_tokens": 512} if rng.random() < 0.5 else {"type": "disabled"}}
        t0 = time.time()
        ev = self.cl.stream("/v1/messages", body)
        names = [n for n, _ in ev]
        out = next((e["usage"]["output_tokens"] for n, e in ev if n == "message_delta"), 0)
        self.record("anthropic", names[-1:] == ["message_stop"] and out > 0, sec=round(time.time() - t0, 2), out=out)

    def responses(self, rng):
        body = {"model": "x", "instructions": "Answer concisely.", "max_output_tokens": 800, "stream": True,
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
                          f"List {rng.randint(3, 7)} pitfalls of {rng.choice(WORDS)} {rng.choice(WORDS)} handling."}]}],
                "reasoning": {"effort": rng.choice(["low", "medium"])}}
        t0 = time.time()
        ev = self.cl.stream("/v1/responses", body)
        last = ev[-1][1] if ev else {}
        self.record("responses", last.get("type") in ("response.completed", "response.incomplete"),
                    sec=round(time.time() - t0, 2), last=last.get("type"),
                    out=((last.get("response") or {}).get("usage") or {}).get("output_tokens"))

    def cancel(self, rng):
        long_prompt = rng.random() < 0.5
        content = (filler(rng, rng.randint(16000, 40000)) + "\n\nSummarize the above.") if long_prompt else \
            f"Write a long story about a {rng.choice(WORDS)} named {rng.choice(WORDS)}."
        after = rng.uniform(0.3, 4.0)
        st, sec = stream_chat(self.cl, {"model": "x", "messages": [{"role": "user", "content": content}],
                                        "max_tokens": 4000}, stop_after=after)
        self.record("cancel", True, long=long_prompt, after=round(after, 2), got_text=bool(st["text"] or st["reasoning"]))

    def long(self, rng):
        n = rng.choice([24000, 48000, 80000, 120000, 160000])
        doc = filler(rng, n)
        key = f"{rng.randrange(10**6):06d}"
        pos = rng.randint(1, max(1, doc.count("\n") - 1))
        lines = doc.split("\n")
        lines.insert(pos, f"secret_code = {key}")
        content = "\n".join(lines) + "\n\nWhat is the value of secret_code? Answer with the number only."
        st, sec = stream_chat(self.cl, {"model": "x", "messages": [{"role": "user", "content": content}],
                                        "max_tokens": 64, "temperature": 0,
                                        "chat_template_kwargs": {"enable_thinking": False}})
        u = st["usage"] or {}
        found = key in st["text"]  # (a miss keeps the answer: the model sometimes drops a key's leading zeros)
        self.record("long", st["finish"] in ("stop", "length") and u.get("prompt_tokens", 0) > 0, target=n,
                    prompt=u.get("prompt_tokens"), sec=round(sec, 2), ttft=st["ttft"], found=found,
                    **({} if found else {"key": key, "answer": st["text"][:80]}))

    def image(self, rng):
        k = rng.randrange(2)
        q = ["Which month has the highest value in this chart? Answer briefly.", "What is in this picture? One sentence."][k]
        st, sec = stream_chat(self.cl, {"model": "x", "max_tokens": 200, "temperature": 0,
                                        "chat_template_kwargs": {"enable_thinking": False},
                                        "messages": [{"role": "user", "content": [
                                            {"type": "image_url", "image_url": {"url": self.images[k]}},
                                            {"type": "text", "text": q}]}]})
        good = ("mar" in st["text"].lower()) if k == 0 else ("astronaut" in st["text"].lower())
        self.record("image", st["finish"] == "stop", sec=round(sec, 2), image=k, answer_ok=good, answer=st["text"][:200])

    # --- driver ----------------------------------------------------------------------------------------
    def worker(self, i):
        rng = random.Random(1000 + i)
        kinds = ["agent", "agent", "agent", "anthropic", "responses", "cancel", "long", "image"]
        while time.time() < self.stop_at:
            kind = rng.choice(kinds)
            try:
                getattr(self, kind)(rng)
            except Exception as e:
                self.record(kind, False, error=repr(e)[:300])
                time.sleep(2)

    def sampler(self):
        while not self.done.is_set():
            try:
                m = metrics(self.cl)
            except Exception as e:
                m = {"error": repr(e)}
            s = {"t": round(time.time() - self.t0, 1), "gpu_mib": gpu_mem_mib(), "host_gib": container_mem(), **m}
            self.samples.append(s)
            print(f"[soak] t={s['t']:.0f}s gpu={s['gpu_mib']} MiB host={s['host_gib']} GiB "
                  f"running={m.get('q27:num_requests_running')} waiting={m.get('q27:num_requests_waiting')} "
                  f"kv_free={m.get('q27:kv_free_pages')} done={len(self.records)} fails={len(self.errors)}", flush=True)
            self.done.wait(self.args.sample_every)

    def greedy(self):
        st, r = self.cl.post("/v1/chat/completions", dict(GREEDY_CHECK, model="x"))
        return r["choices"][0]["message"]["content"] if st == 200 else f"HTTP {st}"

    def run(self):
        self.t0 = time.time()
        self.since = datetime.now().astimezone().isoformat(timespec="seconds")
        before_text, m0 = self.greedy(), metrics(self.cl)
        self.done = threading.Event()
        smp = threading.Thread(target=self.sampler, daemon=True)
        smp.start()
        ws = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(self.args.workers)]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        # Idle: wait for cancelled requests to drain, then compare.
        for _ in range(60):
            m1 = metrics(self.cl)
            if m1.get("q27:num_requests_running") == 0 and m1.get("q27:num_requests_waiting") == 0:
                break
            time.sleep(1)
        self.done.set()
        smp.join()
        after_text = self.greedy()
        # No leaked KV pages: a prompt that needs nearly the whole pool still runs (checkpoints give theirs back).
        full = {}
        full_tokens = self.args.full_tokens
        if full_tokens < 0:  # (95% of the context: 250K at MAX_TOKENS 262144, less when the server lowered it)
            st, models = self.cl.get("/v1/models")
            full_tokens = int(0.95 * json.loads(models)["data"][0]["max_model_len"]) if st == 200 else 0
        if full_tokens:
            st, sec = stream_chat(self.cl, {"model": "x", "max_tokens": 16, "temperature": 0,
                                            "chat_template_kwargs": {"enable_thinking": False},
                                            "messages": [{"role": "user", "content": filler(random.Random(3), full_tokens)
                                                          + "\n\nSay OK."}]})
            full = {"finish": st["finish"], "prompt": (st["usage"] or {}).get("prompt_tokens"), "sec": round(sec, 1)}
            print(f"[soak] full pool request: {full}", flush=True)
        m1 = metrics(self.cl)
        try:
            log = subprocess.run(["docker", "logs", "--since", self.since, CONTAINER], capture_output=True,
                                 text=True, timeout=60)
            log = log.stdout + log.stderr
        except Exception:
            log = ""
        server_log = {"preempted": log.count("(KV pool full)"), "cannot_hold": log.count("cannot hold"),
                      "tracebacks": log.count("Traceback"), "engine_restarts": log.count("engine ready")}
        by_kind = {}
        for r in self.records:
            k = by_kind.setdefault(r["kind"], {"n": 0, "fail": 0, "sec": []})
            k["n"] += 1
            k["fail"] += not r["ok"]
            if r.get("sec") is not None:
                k["sec"].append(r["sec"])
        summary = {k: {"n": v["n"], "fail": v["fail"],
                       "sec_p50": round(statistics.median(v["sec"]), 2) if v["sec"] else None,
                       "sec_max": max(v["sec"]) if v["sec"] else None} for k, v in by_kind.items()}
        warm = [r["ttft"] for r in self.records if r["kind"] == "agent" and r.get("turn", 0) > 0 and r.get("ttft")]
        reuse = [r for r in self.records if r["kind"] == "agent" and r.get("turn", 0) > 0 and r.get("prompt")]
        gpu = [s["gpu_mib"] for s in self.samples if s.get("gpu_mib")]
        host = [s["host_gib"] for s in self.samples if s.get("host_gib")]
        res = {
            "minutes": self.args.minutes, "workers": self.args.workers, "summary": summary,
            "failures": self.errors[:50],
            "greedy_same_before_after": before_text == after_text, "full_pool_request": full, "server_log": server_log,
            "idle_at_end": m1.get("q27:num_requests_running") == 0 and m1.get("q27:num_requests_waiting") == 0,
            "kv_free_pages_start_end": [m0.get("q27:kv_free_pages"), m1.get("q27:kv_free_pages")],
            "gpu_mib_min_max": [min(gpu), max(gpu)] if gpu else None,
            "host_gib_first_last_max": [host[0], host[-1], max(host)] if host else None,
            "agent_warm_ttft_p50_p90": [round(statistics.median(warm), 3),
                                        round(statistics.quantiles(warm, n=10)[-1], 3)] if len(warm) > 10 else None,
            "agent_warm_cached_frac": round(sum(r["cached"] or 0 for r in reuse) / max(1, sum(r["prompt"] for r in reuse)), 4),
            "long_found": [sum(r.get("found", False) for r in self.records if r["kind"] == "long"),
                           sum(r["kind"] == "long" for r in self.records)],
            "image_answer_ok": [sum(r.get("answer_ok", False) for r in self.records if r["kind"] == "image"),
                                sum(r["kind"] == "image" for r in self.records)],
            "metrics_delta": {k: m1[k] - m0.get(k, 0) for k in m1 if k.endswith("_total") or "_total{" in k},
            "samples": self.samples, "records": self.records,
        }
        print(json.dumps({k: v for k, v in res.items() if k not in ("samples", "records", "failures")}, indent=1))
        return res


def longbatch(args):
    cl = Client(args.base_url)
    rng = random.Random(7)

    def body(i):
        doc = filler(random.Random(100 + i), args.long_tokens)
        return {"model": "x", "max_tokens": args.gen, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": doc + "\n\nRewrite the code above as clean, documented "
                                                               "Python, step by step, at length."}]}

    def run(i, out, gate=None):
        t = {"first": None, "n": 0}
        t0 = time.time()

        def on_event(ev):
            e = ev[1]
            for ch in e.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    t["first"] = t["first"] or time.time()
            if e.get("usage"):
                t["usage"] = e["usage"]
        cl.stream("/v1/chat/completions", dict(body(i), stream_options={"include_usage": True}), on_event)
        end = time.time()
        u = t.get("usage", {})
        out[i] = {"prompt": u.get("prompt_tokens"), "out": u.get("completion_tokens"), "ttft": round(t["first"] - t0, 2),
                  "first": t["first"], "end": end,
                  "decode_tok_s": round((u.get("completion_tokens", 1) - 1) / (end - t["first"]), 1)}

    alone = {}
    run(0, alone)
    print(f"[longbatch] alone: {alone[0]}", flush=True)
    together = {}
    ths = [threading.Thread(target=run, args=(i, together)) for i in range(1, 5)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    # The window where all 4 decode together: after the last first token, before the first end.
    lo, hi = max(r["first"] for r in together.values()), min(r["end"] for r in together.values())
    for i, r in sorted(together.items()):
        print(f"[longbatch] together {i}: {r}", flush=True)
    res = {"long_tokens": args.long_tokens, "gen": args.gen, "alone": alone[0],
           "together": [together[i] for i in sorted(together)],
           "together_sum_decode_tok_s": round(sum(r["decode_tok_s"] for r in together.values()), 1),
           "all_decoding_window_s": round(hi - lo, 2)}
    for r in [res["alone"], *res["together"]]:
        r.pop("first"), r.pop("end")
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["soak", "longbatch"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sample-every", type=float, default=15)
    ap.add_argument("--full-tokens", type=int, default=-1,
                    help="soak: prompt of the final full-pool request (-1: 95%% of the server's max_model_len; 0: skip)")
    ap.add_argument("--long-tokens", type=int, default=32768)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = Soak(args).run() if args.mode == "soak" else longbatch(args)
    res["timestamp"] = datetime.now().isoformat(timespec="seconds")
    out = args.out or ROOT / "bench" / "results" / \
        f"{datetime.now():%Y%m%d-%H%M%S}-custom-p6-7-{args.mode}{'-' + args.label if args.label else ''}.json"
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    if args.mode == "soak":
        bad = res["summary"] and any(v["fail"] for v in res["summary"].values())
        full_bad = args.full_tokens != 0 and res["full_pool_request"].get("finish") not in ("stop", "length")
        sys.exit(1 if bad or full_bad or not res["greedy_same_before_after"] or not res["idle_at_end"] else 0)


if __name__ == "__main__":
    main()
