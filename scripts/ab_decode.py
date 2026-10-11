"""Quick A/B of runtime settings of the decode step in one process (the weights are loaded once).

Each config is a set of Q27_* environment switches ("-" = the defaults); the configs are measured in
turn for several rounds (interleaved, so that clock / thermal drift hits all of them alike), each
round a fresh sequence (prefill of --ctx tokens, then --tokens greedy decode steps). Prints the
median ms/step per config and the difference to the first one. Compile-time variants need another
build: point Q27_LIB at it and run the script once per build.

    docker compose --profile dev run --rm dev python scripts/ab_decode.py - Q27_NO_FUSED_ACT=1
    docker compose --profile dev run --rm dev python scripts/ab_decode.py - Q27_SKIP=norm Q27_SKIP=gdn,attn
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))


def parse(cfg):
    env = {}
    if cfg != "-":
        for kv in cfg.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
            elif env:  # Q27_SKIP=norm,silu: a comma inside the value
                last = list(env)[-1]
                env[last] += "," + kv
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help='"-" or KEY=VALUE[,KEY=VALUE...]')
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from q27lib import Engine
    from qwen27b.weights import TARGET, snapshot

    prompts = [json.loads(line) for line in open(ROOT / "bench/golden/prompts.jsonl")]
    base = max(prompts, key=lambda p: len(p["ids"]))["ids"]
    ids = (base * (args.ctx // len(base) + 1))[: args.ctx]
    t0 = time.perf_counter()
    eng = Engine(snapshot(*TARGET), max_tokens=args.ctx + args.tokens + 1)
    print(f"load {time.perf_counter() - t0:.1f} s", flush=True)

    envs = [parse(c) for c in args.configs]
    keys = sorted({k for e in envs for k in e})
    res = {c: [] for c in args.configs}
    for r in range(args.rounds):
        for cfg, env in zip(args.configs, envs):
            eng.configure(**{k: env.get(k) for k in keys})
            eng.reset()
            nxt = eng.forward_sample(ids)
            steps = []
            for _ in range(args.tokens):
                t = time.perf_counter()
                nxt = eng.forward_sample([nxt])
                steps.append(time.perf_counter() - t)
            ms = statistics.median(steps[2:]) * 1000  # (step 1 eager, step 2 captures the graph)
            res[cfg].append(ms)
            print(f"round {r} {cfg:40s} {ms:.3f} ms", flush=True)
    eng.close()
    ref = statistics.median(res[args.configs[0]])
    print(f"--- ctx {args.ctx}, {args.tokens} steps x {args.rounds} rounds (median of the round medians)")
    for cfg in args.configs:
        m = statistics.median(res[cfg])
        print(f"{cfg:40s} {m:7.3f} ms  {1000 / m:6.2f} tok/s  {m - ref:+.3f}  "
              f"[{' '.join(f'{x:.3f}' for x in res[cfg])}]")


if __name__ == "__main__":
    main()
