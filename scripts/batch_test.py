"""Batched speculative steps (P6-10): correctness against each sequence alone, and throughput.

check: --seqs sequences (4, up to 8; greedy and sampled, different settings per sequence) decoded by batched steps
(Engine.spec_step_lanes) must produce exactly the tokens each produces alone with spec_step. The run mixes
batched steps over random subsets of 2..--seqs sequences in random order with single steps of one sequence (the
batch ends and starts again: replay of the pending commits, scalars back to the sequences, swaps).

kvfull: a small KV pool (KV_POOL_TOKENS, --max-tokens 2048): --seqs sequences in batched steps until the engine
raises KvFull; the latest one is suspended to the prefix cache (seq_suspend) and the batch goes on. The others
must still match their solo tokens; the suspended one resumes later from its checkpoint and finishes.

bench: aggregate decode speed of n = 1..4 sequences (--lanes, up to 8) (n = 1: plain spec_step; n > 1: batched steps), the
bench_spec.py prompts (code, json, prose, ja), the checkpoint's default sampling unless --temperature 0.
Reports ms per step, tokens per step per sequence and the total tok/s. --ctx N puts N tokens of a long
document (the longest golden prompt, repeated and rotated per sequence so no two share a prefix) before each prompt.

    docker compose --profile dev run --rm dev python scripts/batch_test.py check
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --seqs 8
    docker compose --profile dev run --rm dev python scripts/batch_test.py bench --steps 200
    docker compose --profile dev run --rm dev python scripts/batch_test.py bench --ctx 32768 --max-tokens 262144
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

from bench_spec import PROMPTS, chat_ids  # noqa: E402

NAMES = ["code", "json", "prose", "ja"]


def start(eng, ids, smp, seed):
    """prefix_begin + spec_begin of a new running sequence; returns its first token."""
    eng.prefix_begin(ids, spec=True)
    t, k, p = smp
    return eng.spec_begin(ids, t, k, p, seed=seed, uniform=random.Random(seed).random())


def solo(eng, ids, smp, seed, n):
    out = [start(eng, ids, smp, seed)]
    while len(out) < n:
        out += eng.spec_step(ahead=False)
    eng.reset()
    return out[:n]


SAMPLINGS = [(0.0, 1, 1.0), (1.0, 20, 0.95), (0.0, 20, 1.0), (0.7, 40, 0.9)]


def make_cases(tok, n, seed0):
    """n sequences: the prompts in turn, and from the fifth on another sampling per prompt (Phase 7)."""
    return [(chat_ids(tok, PROMPTS[NAMES[i % 4]]), SAMPLINGS[(i + i // 4) % 4], seed0 + i) for i in range(n)]


def case_name(i):
    return NAMES[i % 4] if i < 4 else f"{NAMES[i % 4]}{i // 4 + 1}"


def check(eng, tok, args):
    cases = make_cases(tok, args.seqs, 100)
    t0 = time.time()
    refs = [solo(eng, ids, smp, seed, args.gen) for ids, smp, seed in cases]
    print(f"solo runs {time.time() - t0:.1f} s", flush=True)
    rng = random.Random(args.seed)
    outs = [[] for _ in cases]
    where = {}  # sequence -> None (running) or its parking area
    for i, (ids, smp, seed) in enumerate(cases):
        if i > 0:
            run = next(j for j, a in where.items() if a is None)
            where[run] = eng.seq_park()
        outs[i].append(start(eng, ids, smp, seed))
        where[i] = None
    batched = singles = 0
    while any(len(o) < args.gen for o in outs):
        live = [i for i in range(len(cases)) if len(outs[i]) < args.gen]
        if len(live) >= 2 and rng.random() < 0.85:
            lanes = rng.sample(live, rng.randint(2, len(live)))
            new = eng.spec_step_lanes([where[i] for i in lanes])
            for i, t in zip(lanes, new):
                outs[i] += t
            batched += 1
        else:
            i = rng.choice(live)
            if where[i] is not None:  # make it the running sequence
                run = next(j for j, a in where.items() if a is None)
                a = where[i]
                eng.seq_swap(a)
                where[run], where[i] = a, None
            outs[i] += eng.spec_step(ahead=rng.random() < 0.5)
            singles += 1
    ok = True
    for i, (ref, out) in enumerate(zip(refs, outs)):
        out = out[:args.gen]
        diff = next((k for k, (a, b) in enumerate(zip(ref, out)) if a != b), None)
        ok &= diff is None
        print(f"{'ok  ' if diff is None else 'FAIL'} {case_name(i)} {cases[i][1]}: {args.gen} tokens"
              + ("" if diff is None else f", differs at {diff}: {ref[diff:diff + 8]} vs {out[diff:diff + 8]}"), flush=True)
    print(f"{batched} batched steps, {singles} single steps", flush=True)
    for i, a in where.items():
        if a is not None:
            eng.seq_drop(a)
    eng.reset()
    return ok


def kvfull(eng, tok, args, KvFull):
    cases = make_cases(tok, args.seqs, 200)
    refs = [solo(eng, ids, smp, seed, args.gen) for ids, smp, seed in cases]
    outs = [[] for _ in cases]
    where = {}
    for i, (ids, smp, seed) in enumerate(cases):
        if i > 0:
            run = next(j for j, a in where.items() if a is None)
            where[run] = eng.seq_park()
        outs[i].append(start(eng, ids, smp, seed))
        where[i] = None
    live, suspended, fulls = list(range(len(cases))), [], 0
    while live:
        lanes = [i for i in live if len(outs[i]) < args.gen]
        if not lanes:
            break
        try:
            if len(lanes) >= 2:
                new = eng.spec_step_lanes([where[i] for i in lanes])
            else:
                i = lanes[0]
                if where[i] is not None:
                    run = next((j for j, a in where.items() if a is None), None)
                    if run is None:
                        eng.seq_unpark(where[i])
                    else:
                        a = where[i]
                        eng.seq_swap(a)
                        where[run] = a
                    where[i] = None
                new = [eng.spec_step(ahead=False)]
            for i, t in zip(lanes, new):
                outs[i] += t
        except KvFull:
            fulls += 1
            v = max(live)  # the latest sequence
            if where[v] is None:
                eng.checkpoint()
                eng.reset()
            else:
                eng.seq_suspend(where[v])
            del where[v]
            live.remove(v)
            suspended.append(v)
            print(f"KvFull: suspended {case_name(v)} at {len(outs[v])} tokens", flush=True)
    ok = fulls > 0
    for i in range(len(cases)):
        if i in suspended:
            continue
        good = outs[i][:args.gen] == refs[i]
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {case_name(i)}: {args.gen} tokens", flush=True)
    for a in where.values():
        if a is not None:
            eng.seq_drop(a)
    eng.reset()
    for v in suspended:  # resume from the prefix cache (the continuation may differ in the last bits of the GEMMs)
        ids, smp, seed = cases[v]
        out = outs[v]
        cached = eng.prefix_begin(ids + out[:-1], spec=True)
        rest = (ids + out[:-1])[cached:]
        t, k, p = smp
        out.append(eng.spec_begin(rest, t, k, p, seed=seed + 1))
        while len(out) < args.gen:
            out += eng.spec_step(ahead=False)
        print(f"ok   {case_name(v)}: resumed from {cached} cached tokens, finished at {len(out)}", flush=True)
        eng.reset()
    print(f"{fulls} KvFull", flush=True)
    return ok


def long_doc(n, shift):
    base = max((json.loads(line) for line in open(ROOT / "bench/golden/prompts.jsonl")), key=lambda p: len(p["ids"]))["ids"]
    base = base[shift % len(base):] + base[:shift % len(base)]
    return (base * (n // len(base) + 1))[:n]


def bench(eng, tok, args):
    smp = (args.temperature, args.top_k, args.top_p)
    rows = []
    for n in [int(x) for x in args.lanes.split(",")]:
        for rep in range(args.reps):
            names = [NAMES[(i + rep) % len(NAMES)] for i in range(n)]
            areas = []
            outs = [[] for _ in range(n)]
            for i, name in enumerate(names):
                if i > 0:
                    areas.append(eng.seq_park())
                ids = chat_ids(tok, PROMPTS[name])
                if args.ctx:
                    ids = long_doc(args.ctx, 997 * (i + 1) + 31 * rep) + ids
                outs[i].append(start(eng, ids, smp, 1000 * rep + i))
            lanes = areas + [None]  # (the last one is running)
            order = list(range(n))  # lane z -> sequence (the parked ones in order, then the running one)
            eng.spec_stats()
            if args.profile:
                eng.profile_dump()  # (drops the prompts)
            ms = []
            for k in range(args.steps):
                t0 = time.perf_counter()
                if n == 1:
                    new = [eng.spec_step()]
                else:
                    new = eng.spec_step_lanes(lanes)
                ms.append(1000 * (time.perf_counter() - t0))
                for z, t in enumerate(new):
                    outs[order[z] if n > 1 else 0] += t
            warm = ms[5:]  # (the first steps capture the graph)
            toks = [len(o) - 1 for o in outs]
            tok_s = sum(toks) / (sum(ms) / 1000)
            row = dict(lanes=n, rep=rep, ctx=args.ctx, prompts=names, steps=args.steps, ms_per_step=round(statistics.median(warm), 3),
                       first_ms=[round(x, 1) for x in ms[:3]],
                       tokens_per_step=[round(t / args.steps, 3) for t in toks], total_tok_s=round(tok_s, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.profile:
                print(f"--- profile, {n} lanes", file=sys.stderr, flush=True)
                eng.profile_dump()
            for a in areas:
                eng.seq_drop(a)
            eng.reset()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["check", "kvfull", "bench"])
    ap.add_argument("--gen", type=int, default=400, help="check: tokens per sequence")
    ap.add_argument("--seqs", type=int, default=4, help="check / kvfull: sequences (2..8)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200, help="bench: steps per run")
    ap.add_argument("--lanes", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--kv-dtype", default="fp8", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=0, help="bench: tokens of a long document before each prompt")
    ap.add_argument("--profile", action="store_true", help="bench: per-op GPU time of the steps (Q27_PROFILE=1)")
    ap.add_argument("--out", default="", help="bench: result JSON (default bench/results/<UTC time>-custom-p6-10-batch.json)")
    args = ap.parse_args()
    lanes_max = max(int(x) for x in args.lanes.split(",")) if args.part == "bench" else args.seqs
    os.environ.setdefault("MAX_SEQS", str(max(4, lanes_max)))
    if args.part == "kvfull":  # (a pool of 2,112 tokens: four sequences of --gen 600 do not fit; 512 per sequence)
        os.environ.setdefault("STATE_HOST_CACHE", "2")
        os.environ.setdefault("KV_POOL_TOKENS", str(512 * max(4, args.seqs)))
        args.max_tokens = 2048
    if args.profile:
        os.environ["Q27_PROFILE"] = "1"

    from transformers import AutoTokenizer

    from q27lib import Engine, KvFull
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    t0 = time.time()
    eng = Engine(engine_target(args.weights), max_tokens=args.max_tokens, kv_dtype=args.kv_dtype, draft=snapshot(*DFLASH2))
    print(f"load {time.time() - t0:.1f} s, max lanes {eng.max_lanes}", flush=True)
    if args.part in ("check", "kvfull"):
        ok = check(eng, tok, args) if args.part == "check" else kvfull(eng, tok, args, KvFull)
        eng.close()
        sys.exit(0 if ok else 1)
    rows = bench(eng, tok, args)
    eng.close()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or ROOT / f"bench/results/{stamp}-custom-p6-10-batch.json")
    out.write_text(json.dumps(dict(args=vars(args), rows=rows), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
