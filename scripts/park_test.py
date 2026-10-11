"""Parking areas lent to the KV pool (P7-13): seq_park moves the KV pages out of the area it takes, and raises
KvFull, changing nothing, when the pool cannot take them. A small pool without the prefix cache (MAX_SEQS=3,
max_tokens 8,192: KV pages for one sequence plus two areas), plain greedy decoding:

  1. p1 (7,000 tokens) runs and is parked (area 0, at the top of the pool: no pages to move).
  2. p2 (7,000 tokens) runs: its pages reach into area 1. seq_park must raise KvFull (no free page outside it).
  3. p2 continues: its tokens equal those it produces alone.
  4. After p1 is dropped, p2 parks (pages moved if the area it takes holds some), resumes and still matches.

    docker compose --profile dev run --rm dev python scripts/park_test.py --kv-dtype nvfp4
"""

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kv-dtype", default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="gdnfp4", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ["MAX_SEQS"] = "3"
    os.environ["STATE_HOST_CACHE"] = "0"
    os.environ.pop("KV_POOL_TOKENS", None)
    os.environ.pop("Q27_PAGES_HIGH", None)

    from q27lib import Engine, KvFull
    from qwen27b.weights import engine_target

    eng = Engine(engine_target(args.weights), max_tokens=8192, kv_dtype=args.kv_dtype)
    prompts = sorted((json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl")), key=len)
    p1 = (prompts[-1] * 4)[:7000]
    p2 = (prompts[-2] * 4)[:7000]
    fails = 0

    def check(name, ok):
        nonlocal fails
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)

    def run(ids, n):
        out = [eng.forward_sample(ids, 0.0, 1, 1.0)]
        while len(out) < n:
            out.append(eng.forward_sample([out[-1]], 0.0, 1, 1.0))
        return out

    eng.reset()
    ref1 = run(p1, 48)
    eng.reset()
    ref2 = run(p2, 48)
    print(f"free pages, nothing running: {eng.cache_stats()['free_pages']}", flush=True)

    eng.reset()
    out1 = run(p1, 16)
    a1 = eng.seq_park()
    out2 = run(p2, 16)
    try:
        eng.seq_park()
        check("seq_park raises KvFull when the area's pages cannot move", False)
    except KvFull:
        check("seq_park raises KvFull when the area's pages cannot move", True)
    while len(out2) < 32:
        out2.append(eng.forward_sample([out2[-1]], 0.0, 1, 1.0))
    check("the running sequence continues as alone after the failed park", out2 == ref2[:32])

    eng.seq_unpark(a1)  # (drops p2: p1 runs again)
    while len(out1) < 48:
        out1.append(eng.forward_sample([out1[-1]], 0.0, 1, 1.0))
    check("the parked sequence resumes as alone", out1 == ref1)

    eng.reset()
    out2 = run(p2, 16)
    a2 = eng.seq_park()
    eng.reset()
    eng.seq_unpark(a2)
    while len(out2) < 48:
        out2.append(eng.forward_sample([out2[-1]], 0.0, 1, 1.0))
    check("a sequence parked after the failure resumes as alone", out2 == ref2)

    print("PASS" if fails == 0 else f"{fails} FAILED", flush=True)
    eng.close()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
