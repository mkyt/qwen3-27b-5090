"""Decode speed of the engine (1 sequence, no speculation) at several context lengths, run in-process.

The context is a prefix of the longest golden prompt (ja-prose-16384), repeated when longer.
Greedy decoding; reports prefill tok/s (after a short warm-up prefill; the best of --prefill-reps runs) and decode
ms/token (median of the steps). With --profile the
engine prints the per-op GPU time of the decode steps (Q27_PROFILE=1; the event records add a little).

    docker compose --profile dev run --rm dev python scripts/bench_decode.py --ctx 128,4096,16384 --tokens 128
"""

import argparse
import datetime
import json
import os
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", default="128,4096,16384", help="context lengths (comma separated)")
    ap.add_argument("--tokens", type=int, default=128, help="decode steps per context")
    ap.add_argument("--prefill-reps", type=int, default=1, help="prefill runs per context (the fastest is reported)")
    ap.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="nvidia", help="weight profile (Phase Q): nvidia | quasar | a model directory")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--profile", action="store_true", help="per-op GPU time of the decode steps")
    ap.add_argument("--out", default="", help="result JSON (default bench/results/<UTC time>Z-decode-engine.json)")
    args = ap.parse_args()
    if args.profile:
        os.environ["Q27_PROFILE"] = "1"

    from q27lib import Engine
    from qwen27b.weights import engine_target

    ctxs = [int(x) for x in args.ctx.split(",")]
    prompts = [json.loads(line) for line in open(ROOT / "bench/golden/prompts.jsonl")]
    base = max(prompts, key=lambda p: len(p["ids"]))["ids"]
    eng = Engine(engine_target(args.weights), max_tokens=max(ctxs) + args.tokens + 1, kv_dtype=args.kv_dtype)
    eng.forward_sample(base[:256])  # warm-up: the first call pays one-time setup
    rows = []
    for ctx in ctxs:
        ids = (base * (ctx // len(base) + 1))[:ctx]
        times = []
        for _ in range(args.prefill_reps):
            eng.reset()
            t0 = time.perf_counter()
            nxt = eng.forward_sample(ids)
            times.append(time.perf_counter() - t0)
        prefill_s = min(times)
        if args.profile:
            eng.profile_dump()  # drop the prefill
        steps = []
        for _ in range(args.tokens):
            t0 = time.perf_counter()
            nxt = eng.forward_sample([nxt], args.temperature, args.top_k, args.top_p, random.random())
            steps.append(time.perf_counter() - t0)
        if args.profile:
            print(f"--- profile, context {ctx}", file=sys.stderr, flush=True)
            eng.profile_dump()
        ms = statistics.median(steps) * 1000
        row = {"ctx": ctx, "prefill_tok_s": round(ctx / prefill_s, 1), "prefill_s": [round(t, 3) for t in times], "decode_ms": round(ms, 3),
               "decode_tok_s": round(1000 / ms, 2), "decode_ms_p90": round(sorted(steps)[int(0.9 * len(steps))] * 1000, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    eng.close()
    # The container clock is UTC: the name says so (Z), unlike the host-side scripts' local times.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out = args.out or ROOT / f"bench/results/{stamp}-decode-engine.json"
    if not args.profile:
        pathlib.Path(out).write_text(json.dumps({"weights": args.weights, "kv_dtype": args.kv_dtype, "tokens": args.tokens,
                                                 "sampling": [args.temperature, args.top_k, args.top_p], "rows": rows},
                                                indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
