#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1"]
# ///
"""Reasoning-loop probe (PQ-14): how often do responses run away or end in a repetition loop?

Thinking on, low reasoning effort, temperature 0.6 (the setting where NVFP4 checkpoints were reported to loop,
HF Qwen/Qwen3.8-27B discussion 192), GSM8K questions (bench/quality-data, as quality_gsm8k.py) with a few
samples each. A response is a runaway when it hits --max-tokens, and a loop when its last 4,000 characters are
mostly one repeated chunk. Compare two weight profiles on the same questions and seeds.

    uv run scripts/loop_test.py --label custom-nvidia --n 100 --samples 2
"""

import argparse
import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench/quality-data/gsm8k-test.jsonl"
PROMPT = "{q}\n\nSolve step by step, then give the final answer on the last line as `#### <number>`."


def looping(text, tail=4000):
    """True when the tail is mostly copies of one chunk (period 8..1000 characters)."""
    t = text[-tail:]
    if len(t) < 1000:
        return False
    for period in range(8, 1001):
        chunk = t[-period:]
        reps = 0
        while t.endswith(chunk * (reps + 1)):
            reps += 1
        if reps * period >= 0.6 * len(t) and reps >= 3:
            return True
    return False


async def one(client, args, row, sample, sem):
    async with sem:
        body = {"model": args.model, "messages": [{"role": "user", "content": PROMPT.format(q=row["question"])}],
                "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": 0.95, "top_k": 20,
                "seed": sample, "reasoning_effort": args.effort}
        r = (await client.post("/v1/chat/completions", json=body)).raise_for_status().json()
        ch = r["choices"][0]
        msg = ch["message"]
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        return {"q": row["question"][:80], "sample": sample, "finish": ch["finish_reason"],
                "tokens": r["usage"]["completion_tokens"], "loop": looping(text),
                "tail": text[-300:] if ch["finish_reason"] == "length" or looping(text) else ""}


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100, help="questions")
    ap.add_argument("--samples", type=int, default=2, help="samples (seeds) per question")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="low", help="reasoning_effort")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"{DATA} missing: run scripts/quality_gsm8k.py once")
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    random.Random(1).shuffle(rows)
    rows = rows[:args.n]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=3600) as client:
        res = await asyncio.gather(*(one(client, args, r, s, sem) for r in rows for s in range(args.samples)))
    n = len(res)
    runaway = sum(r["finish"] == "length" for r in res)
    loops = sum(r["loop"] for r in res)
    toks = sorted(r["tokens"] for r in res)
    print(f"{args.label}: {n} responses, runaway {runaway} ({runaway / n:.1%}), loop {loops} ({loops / n:.1%}), "
          f"tokens median {toks[n // 2]} p95 {toks[int(n * 0.95)]} in {time.time() - t0:.0f} s")
    out = ROOT / f"bench/results/{datetime.now():%Y%m%d-%H%M%S}-loop-{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "args": vars(args), "runaway": runaway, "loop": loops, "n": n,
                               "results": res}, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
