#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1"]
# ///
"""GPU energy per token for any OpenAI-compatible server, from NVML's total-energy counter.

Runs on the host (it loads libnvidia-ml.so.1 next to the server, not inside a container). Each request, or each
group of concurrent requests, is bracketed by two reads of the board's cumulative energy, so idle power during the
request is included. Reported per row: tokens, seconds, average W and tokens per joule.
  decode      three short prompts, 1,024 tokens, one request at a time (the server's default sampling)
  concurrent  C streams on the reasoning prompt of bench_concurrent.py, tok/s and tok/J until the last one ends
  prefill     a long unique document with max_tokens 1 (prompt tokens per joule)

    uv run scripts/bench_energy.py --label energy
"""

import argparse
import asyncio
import ctypes
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_agentic import filler  # noqa: E402
from bench_concurrent import PROMPTS, stream  # noqa: E402

DECODE = {
    "code": "Write a complete, well-documented Python module implementing a thread-safe LRU cache "
            "with TTL expiry, plus pytest unit tests. Output only code.",
    "json": "Generate a JSON array of 25 fictional employee records with fields id, name, email, "
            "department, salary, start_date, skills (array). Output only JSON.",
    "prose": "Explain in detail how a CPU executes a program, from fetching instructions to "
             "branch prediction and caches.",
}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0}


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.h = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) == 0

    def energy_j(self):
        e = ctypes.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)) == 0
        return e.value / 1000

    def limit_w(self):
        p = ctypes.c_uint()
        assert self.lib.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(p)) == 0
        return p.value / 1000


async def measure(nvml, coro):
    e0, t0 = nvml.energy_j(), time.perf_counter()
    res = await coro
    e1, t1 = nvml.energy_j(), time.perf_counter()
    return res, e1 - e0, t1 - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--decode-reps", type=int, default=3)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--conc-max-tokens", type=int, default=2048)
    ap.add_argument("--prefill", default="19500,73000")
    args = ap.parse_args()
    nvml = Nvml()
    out = {"label": args.label, "started": datetime.now().isoformat(), "power_limit_w": nvml.limit_w(),
           "args": vars(args)}
    async with httpx.AsyncClient(base_url=args.base_url, timeout=None) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        e0 = nvml.energy_j()
        await asyncio.sleep(10)
        out["idle_w"] = round((nvml.energy_j() - e0) / 10, 1)
        print(f"idle {out['idle_w']} W, limit {out['power_limit_w']} W")

        rows = []
        for name, text in DECODE.items():
            for rep in range(args.decode_reps):
                r, j, s = await measure(nvml, stream(client, model, text, False, 1024, rep, {}))
                tps = (r["tokens"] - 1) / (r["last"] - r["first"])
                rows.append({"prompt": name, "rep": rep, "tokens": r["tokens"], "decode_tps": round(tps, 1),
                             "joules": round(j, 1), "seconds": round(s, 2), "avg_w": round(j / s, 1),
                             "tok_per_j": round(r["tokens"] / j, 3)})
                print(f"  decode/{name}#{rep}: {tps:.1f} tok/s, {j / s:.0f} W, {r['tokens'] / j:.3f} tok/J")
        out["decode"] = rows

        rows = []
        prompt, thinking = PROMPTS["reason"]
        for c in map(int, args.concurrency.split(",")):
            res, j, s = await measure(nvml, asyncio.gather(*(
                stream(client, model, prompt, thinking, args.conc_max_tokens, 1000 + i, SAMPLING)
                for i in range(c))))
            tok = sum(r["tokens"] for r in res)
            rows.append({"concurrency": c, "tokens": tok, "wall_tps": round(tok / s, 1), "joules": round(j, 1),
                         "seconds": round(s, 2), "avg_w": round(j / s, 1), "tok_per_j": round(tok / j, 3)})
            print(f"  reason C={c}: {tok / s:.1f} tok/s, {j / s:.0f} W, {tok / j:.3f} tok/J")
        out["concurrent"] = rows

        rows = []
        for target in map(int, args.prefill.split(",")):
            rng = random.Random(f"energy-{target}")
            doc = f"# document {uuid.uuid4().hex}\n{filler(rng, target)}\n\nSummarize this code in one sentence."
            r, j, s = await measure(nvml, stream(client, model, doc, False, 1, 0, {}))
            pt = r["prompt_tokens"]
            rows.append({"target": target, "prompt_tokens": pt, "seconds": round(s, 2),
                         "prefill_tps": round(pt / s, 1), "joules": round(j, 1), "avg_w": round(j / s, 1),
                         "tok_per_j": round(pt / j, 2)})
            print(f"  prefill {pt}: {pt / s:.0f} tok/s, {j / s:.0f} W, {pt / j:.2f} tok/J")
        out["prefill"] = rows

    path = ROOT / "bench/results" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}-energy.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
