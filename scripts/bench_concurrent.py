#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1"]
# ///
"""Concurrent-decode benchmark for any OpenAI-compatible server (this engine, NInfer, SGLang, vLLM).

For each concurrency C, C streams with distinct seeds start together on the same long-generation prompt.
Reported per C:
  sum_stream_tps  sum over streams of (completion_tokens - 1) / (last_chunk - first_chunk)
  window_s        time all C streams were decoding at once ([max(first_chunk), min(last_chunk)])
  wall_tps        total completion tokens / (last response end - release)

    uv run scripts/bench_concurrent.py --label custom --concurrency 1,2,4 --max-tokens 4096
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PROMPTS = {
    # long reasoning, like NInfer's decode-saturation fixture (AIME-style, thinking on)
    "reason": ("Find the number of ordered pairs of positive integers (a, b) with a, b <= 1000 such that "
               "a^2 + b^2 is divisible by 13 and a + b is a perfect square. Reason step by step.", True),
    "code": ("Write a complete, well-documented Python module implementing a thread-safe LRU cache "
             "with TTL expiry, plus pytest unit tests. Output only code.", False),
}


async def stream(client, model, prompt, thinking, max_tokens, seed, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    t0 = time.perf_counter()
    first = last = None
    n = prompt_tokens = 0
    finish = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:500]!r}")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                n = chunk["usage"]["completion_tokens"]
                prompt_tokens = chunk["usage"]["prompt_tokens"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    last = time.perf_counter()
                    first = first or last
                finish = ch.get("finish_reason") or finish
    end = time.perf_counter()
    return {"t0": t0, "first": first, "last": last, "end": end, "tokens": n, "prompt_tokens": prompt_tokens,
            "finish": finish}


async def run_point(client, model, prompt, thinking, c, max_tokens, extra):
    rel = time.perf_counter()
    rows = await asyncio.gather(*(stream(client, model, prompt, thinking, max_tokens, 1000 + i, extra)
                                  for i in range(c)))
    rates = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in rows]
    w0, w1 = max(r["first"] for r in rows), min(r["last"] for r in rows)
    window = w1 - w0
    out = {
        "concurrency": c,
        "tokens": [r["tokens"] for r in rows],
        "finish": [r["finish"] for r in rows],
        "ttft": [round(r["first"] - r["t0"], 3) for r in rows],
        "stream_tps": [round(x, 2) for x in rates],
        "sum_stream_tps": round(sum(rates), 2),
        "window_s": round(window, 2),
        "wall_s": round(max(r["end"] for r in rows) - rel, 2),
        "wall_tps": round(sum(r["tokens"] for r in rows) / (max(r["end"] for r in rows) - rel), 2),
    }
    print(f"  C={c}: sum {out['sum_stream_tps']} tok/s, per stream p50 {statistics.median(rates):.1f}, "
          f"wall {out['wall_tps']} tok/s, tokens {out['tokens']}, window {out['window_s']} s")
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="run")
    ap.add_argument("--prompts", default="reason,code")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into every request (pin sampling across servers)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "bench" / "results"))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    extra = json.loads(args.extra_body)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=None) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        await stream(client, model, "hi", False, 8, 0, extra)  # warmup
        report = {"label": args.label, "model": model, "started": datetime.now().isoformat(),
                  "args": {k: v for k, v in vars(args).items() if k != "out_dir"}, "results": {}}
        for name in args.prompts.split(","):
            prompt, thinking = PROMPTS[name]
            report["results"][name] = []
            for c in (int(x) for x in args.concurrency.split(",")):
                for rep in range(args.reps):
                    print(f"{name} C={c} rep {rep}")
                    report["results"][name].append(
                        await run_point(client, model, prompt, thinking, c, args.max_tokens, extra))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
