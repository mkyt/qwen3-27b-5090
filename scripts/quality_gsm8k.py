#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1"]
# ///
"""GSM8K exact-match against an OpenAI-compatible server (quality gate for lossy options, docs/decisions.md "Quality gates").

Thinking off, greedy, the same fixed subset every run, so two configurations are compared on the same
questions. The test set (MIT, openai/grade-school-math) is downloaded once into bench/quality-data/.

    uv run scripts/quality_gsm8k.py --label sglang-mtp4 --n 300
"""

import argparse
import asyncio
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DATA = ROOT / "bench/quality-data/gsm8k-test.jsonl"
PROMPT = "{q}\n\nSolve step by step, then give the final answer on the last line as `#### <number>`."


def load(n, seed):
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(httpx.get(DATA_URL, follow_redirects=True, timeout=60).raise_for_status().content)
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def number(text):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)  # 1024 cut 8 of 300 (long self-checks, not loops)
    args = ap.parse_args()

    rows = load(args.n, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(rows)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:

        async def one(i, row):
            async with sem:
                r = await client.post("/v1/chat/completions", json={
                    "model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": PROMPT.format(q=row["question"])}],
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                choice = r.json()["choices"][0]
                text = choice["message"]["content"] or ""
                gold = row["answer"].split("####")[-1].strip().replace(",", "")
                results[i] = {"gold": gold, "pred": number(text), "ok": same(number(text), gold),
                              "finish": choice.get("finish_reason"), "text": text}

        t0 = time.time()
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    acc = sum(r["ok"] for r in results) / len(results)
    cut = sum(r["finish"] == "length" for r in results)
    print(f"{args.label}: GSM8K {acc * 100:.1f}% ({sum(r['ok'] for r in results)}/{len(results)}) in {time.time() - t0:.0f} s, "
          f"{cut} hit --max-tokens")
    out = ROOT / "bench/results" / f"{datetime.now():%Y%m%d-%H%M%S}-gsm8k-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "n": len(results), "seed": args.seed, "accuracy": acc,
                               "results": results}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
