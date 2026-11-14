"""Batched speculative steps (P6-10): correctness against each sequence alone, and throughput.

check: --seqs sequences (4, up to 8; greedy and sampled, different settings per sequence) decoded by batched steps
(Engine.spec_step_lanes) must produce exactly the tokens each produces alone with spec_step. The run mixes
batched steps over random subsets of 2..--seqs sequences in random order with single steps of one sequence (the
batch ends and starts again: replay of the pending commits, scalars back to the sequences, swaps).
--widths (DSpine): sequence i verifies blocks of the i-th width (cyclic; Engine.spec_set_block) in its solo run,
its single steps and as a batched lane, so batched steps mix lanes of different rows. The widths of any lanes must
fit the step's row budget (MAX_SEQS 8: 64 rows), or a lane's width is cut and its tokens differ.
--rule (DSpine): the engine picks each lane's rows per step (Model::batch_rows); batched steps only, then each
sequence's recorded rows are replayed alone, pinned per step, and must give the same tokens.

kvfull: a small KV pool (KV_POOL_TOKENS, --max-tokens 2048): --seqs sequences in batched steps until the engine
raises KvFull; the latest one is suspended to the prefix cache (seq_suspend) and the batch goes on. The others
must still match their solo tokens; the suspended one resumes later from its checkpoint and finishes.

bench: aggregate decode speed of n = 1..4 sequences (--lanes, up to 8) (n = 1: plain spec_step; n > 1: batched steps), the
bench_spec.py prompts (code, json, prose, ja), the checkpoint's default sampling unless --temperature 0.
Reports ms per step, tokens per step per sequence and the total tok/s. --ctx N puts N tokens of a long
document (the longest golden prompt, repeated and rotated per sequence so no two share a prefix) before each prompt.

    docker compose --profile dev run --rm dev python scripts/batch_test.py check
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --seqs 8
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --penalties   # min_p, presence / frequency
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --logprobs    # and logprobs (some lanes)
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --draft dspine --widths 4,12,16,8
    docker compose --profile dev run --rm dev python scripts/batch_test.py check --draft dspine --rule --seqs 8
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


def begin(eng, rest, smp, seed, uniform=0.5, prior=()):
    """spec_begin with smp = (temperature, top_k, top_p[, min_p, presence, frequency[, logprobs]]); prior: the output
    so far (the penalty counts)."""
    t, k, p, *more = smp
    extra = dict(zip(("min_p", "presence", "frequency", "logprobs"), more))
    if extra.get("presence") or extra.get("frequency"):
        eng.penalty_init(list(prior))
    return eng.spec_begin(rest, t, k, p, seed=seed, uniform=uniform, **extra)


def start(eng, ids, smp, seed):
    """prefix_begin + spec_begin of a new running sequence; returns its first token."""
    eng.prefix_begin(ids, spec=True)
    return begin(eng, ids, smp, seed, random.Random(seed).random())


def solo(eng, ids, smp, seed, n, lps=None, width=0):
    out = [start(eng, ids, smp, seed)]
    if width:
        eng.spec_set_block(None, width)
    if lps is not None:
        lps += eng.logprobs()
    while len(out) < n:
        out += eng.spec_step(ahead=False)
        if lps is not None:
            lps += eng.logprobs()
    eng.reset()
    return out[:n]


def lp_diff(ref, out, n):
    """The largest difference of the tokens' logprobs and their top entries' (by id; an entry missing from the other
    list counts when it is not a near tie at its end) over the first n tokens; None when one has fewer."""
    if len(ref) < n or len(out) < n:
        return None
    worst = 0.0
    for (a, ta), (b, tb) in zip(ref[:n], out[:n]):
        worst = max(worst, abs(a - b))
        da, db = dict(ta), dict(tb)
        if len(da) != len(db):
            return float("inf")
        floor = min([*da.values(), *db.values()], default=0.0)
        for x in da.keys() | db.keys():
            if x in da and x in db:
                worst = max(worst, abs(da[x] - db[x]))
            elif max(da.get(x, floor), db.get(x, floor)) - floor > 1e-3:
                worst = float("inf")
    return worst


SAMPLINGS = [(0.0, 1, 1.0), (1.0, 20, 0.95), (0.0, 20, 1.0), (0.7, 40, 0.9)]
# --penalties: min_p and presence / frequency penalties on three of the four (and greedy with penalties).
PENALTIES = [(0.0, 1, 1.0, 0.0, 1.5, 0.5), (1.0, 20, 0.95, 0.05, 1.5, 0.0), (0.7, 20, 0.8, 0.0, 0.0, 0.0),
             (0.6, 40, 0.9, 0.1, 0.5, 1.0)]
# --logprobs: the penalties' samplings with logprobs (5, none, 20, 0 top entries).
LOGPROBS = [smp + (n,) for smp, n in zip(PENALTIES, (5, -1, 20, 0))]


def make_cases(tok, n, seed0, samplings=SAMPLINGS):
    """n sequences: the prompts in turn, and from the fifth on another sampling per prompt (Phase 7)."""
    return [(chat_ids(tok, PROMPTS[NAMES[i % 4]]), samplings[(i + i // 4) % 4], seed0 + i) for i in range(n)]


def case_name(i):
    return NAMES[i % 4] if i < 4 else f"{NAMES[i % 4]}{i // 4 + 1}"


def check_rule(eng, tok, args):
    """--rule (DSpine): the engine picks the lanes' rows (batch_rows). Batched steps of random subsets until fewer than
    two sequences are left; each sequence's rows per step are recorded, then replayed alone (spec_set_block per step)."""
    cases = make_cases(tok, args.seqs, 100, PENALTIES if args.penalties else SAMPLINGS)
    rng = random.Random(args.seed)
    outs = [[] for _ in cases]
    rows = [[] for _ in cases]
    where = {}
    for i, (ids, smp, seed) in enumerate(cases):
        if i > 0:
            run = next(j for j, a in where.items() if a is None)
            where[run] = eng.seq_park()
        outs[i].append(start(eng, ids, smp, seed))
        where[i] = None
    steps = 0
    while True:
        live = [i for i in range(len(cases)) if len(outs[i]) < args.gen]
        if len(live) < 2:
            break
        lanes = rng.sample(live, rng.randint(max(2, len(live) - 2), len(live)))
        new = eng.spec_step_lanes([where[i] for i in lanes])
        for z, (i, t) in enumerate(zip(lanes, new)):
            outs[i] += t
            rows[i].append(eng.last_lane_block(z))
        steps += 1
    for i, a in where.items():
        if a is not None:
            eng.seq_drop(a)
    eng.reset()
    ok = True
    for i, (ids, smp, seed) in enumerate(cases):
        ref = [start(eng, ids, smp, seed)]
        for w in rows[i]:
            eng.spec_set_block(None, w)
            ref += eng.spec_step(ahead=False)
        eng.reset()
        diff = next((k for k, (a, b) in enumerate(zip(ref, outs[i])) if a != b), None)
        if diff is None and len(ref) != len(outs[i]):
            diff = min(len(ref), len(outs[i]))
        ok &= diff is None
        hist = {w: rows[i].count(w) for w in sorted(set(rows[i]))}
        changes = sum(a != b for a, b in zip(rows[i], rows[i][1:]))
        print(f"{'ok  ' if diff is None else 'FAIL'} {case_name(i)} {smp}: {len(outs[i])} tokens, rows {hist}, {changes} changes"
              + ("" if diff is None else f", differs at {diff}: {ref[diff:diff + 8]} vs {outs[i][diff:diff + 8]}"), flush=True)
    print(f"{steps} batched steps", flush=True)
    return ok


def check(eng, tok, args):
    if args.rule:
        return check_rule(eng, tok, args)
    cases = make_cases(tok, args.seqs, 100, LOGPROBS if args.logprobs else PENALTIES if args.penalties else SAMPLINGS)
    t0 = time.time()
    ref_lps = [[] for _ in cases]
    widths = [int(x) for x in args.widths.split(",")] if args.widths else []
    width = [widths[i % len(widths)] if widths else 0 for i in range(len(cases))]
    refs = [solo(eng, ids, smp, seed, args.gen, ref_lps[i], width[i]) for i, (ids, smp, seed) in enumerate(cases)]
    print(f"solo runs {time.time() - t0:.1f} s", flush=True)
    rng = random.Random(args.seed)
    outs = [[] for _ in cases]
    lps = [[] for _ in cases]
    where = {}  # sequence -> None (running) or its parking area
    for i, (ids, smp, seed) in enumerate(cases):
        if i > 0:
            run = next(j for j, a in where.items() if a is None)
            where[run] = eng.seq_park()
        outs[i].append(start(eng, ids, smp, seed))
        if width[i]:
            eng.spec_set_block(None, width[i])
        lps[i] += eng.logprobs()
        where[i] = None
    batched = singles = 0
    while any(len(o) < args.gen for o in outs):
        live = [i for i in range(len(cases)) if len(outs[i]) < args.gen]
        if len(live) >= 2 and rng.random() < 0.85:
            lanes = rng.sample(live, rng.randint(2, len(live)))
            new = eng.spec_step_lanes([where[i] for i in lanes])
            for z, (i, t) in enumerate(zip(lanes, new)):
                outs[i] += t
                lps[i] += eng.logprobs(z)
            batched += 1
        else:
            i = rng.choice(live)
            if where[i] is not None:  # make it the running sequence
                run = next(j for j, a in where.items() if a is None)
                a = where[i]
                eng.seq_swap(a)
                where[run], where[i] = a, None
            outs[i] += eng.spec_step(ahead=rng.random() < 0.5)
            lps[i] += eng.logprobs()
            singles += 1
    ok = True
    for i, (ref, out) in enumerate(zip(refs, outs)):
        out = out[:args.gen]
        diff = next((k for k, (a, b) in enumerate(zip(ref, out)) if a != b), None)
        ok &= diff is None
        lp_note = ""
        if args.logprobs:
            want = cases[i][1][6] >= 0
            d = lp_diff(ref_lps[i], lps[i], args.gen) if want else (0.0 if not ref_lps[i] and not lps[i] else None)
            good = d is not None and d <= args.lp_tol
            ok &= good
            lp_note = (f", logprobs {'off' if not want else f'max diff {d:.2e}' if d is not None else 'missing'}"
                       + ("" if good else " FAIL"))
        print(f"{'ok  ' if diff is None else 'FAIL'} {case_name(i)} {cases[i][1]}: {args.gen} tokens"
              + ("" if diff is None else f", differs at {diff}: {ref[diff:diff + 8]} vs {out[diff:diff + 8]}") + lp_note,
              flush=True)
    print(f"{batched} batched steps, {singles} single steps", flush=True)
    for i, a in where.items():
        if a is not None:
            eng.seq_drop(a)
    eng.reset()
    return ok


def kvfull(eng, tok, args, KvFull):
    cases = make_cases(tok, args.seqs, 200, PENALTIES if args.penalties else SAMPLINGS)
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
        out.append(begin(eng, rest, smp, seed + 1, prior=out[:-1]))
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
            names = [args.prompt or NAMES[(i + rep) % len(NAMES)] for i in range(n)]
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
            hist = {}
            prev, changes = None, 0
            for k in range(args.steps):
                t0 = time.perf_counter()
                if n == 1:
                    new = [eng.spec_step()]
                else:
                    new = eng.spec_step_lanes(lanes)
                ms.append(1000 * (time.perf_counter() - t0))
                if n > 1:
                    split = [eng.last_lane_block(z) for z in range(n)]
                    for w in split:
                        hist[w] = hist.get(w, 0) + 1
                    changes += prev is not None and split != prev
                    prev = split
                for z, t in enumerate(new):
                    outs[order[z] if n > 1 else 0] += t
            warm = ms[5:]  # (the first steps capture the graph)
            toks = [len(o) - 1 for o in outs]
            tok_s = sum(toks) / (sum(ms) / 1000)
            row = dict(lanes=n, rep=rep, ctx=args.ctx, prompts=names, steps=args.steps, ms_per_step=round(statistics.median(warm), 3),
                       first_ms=[round(x, 1) for x in ms[:3]],
                       tokens_per_step=[round(t / args.steps, 3) for t in toks], total_tok_s=round(tok_s, 1),
                       lane_rows={str(w): hist[w] for w in sorted(hist)}, row_changes=changes)
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
    ap.add_argument("--penalties", action="store_true", help="check / kvfull: samplings with min_p and penalties")
    ap.add_argument("--logprobs", action="store_true", help="check: the penalties' samplings with logprobs (some lanes)")
    ap.add_argument("--lp-tol", type=float, default=2e-3, help="check --logprobs: largest logprob difference")
    ap.add_argument("--steps", type=int, default=200, help="bench: steps per run")
    ap.add_argument("--lanes", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--kv-dtype", default="fp8", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--draft", default="dflash2", choices=["dflash2", "dspine"],
                    help="draft model (check / kvfull with dspine: Q27_DRAFT_BLOCK=8 unless --widths, so every lane has 8 rows)")
    ap.add_argument("--rule", action="store_true",
                    help="check, dspine: the engine's rows per lane (Q27_SPEC_ADAPT_BATCH), replayed alone per step")
    ap.add_argument("--widths", default="", help="check, dspine: verify widths of the sequences (cyclic), e.g. 4,12,16,8")
    ap.add_argument("--prompt", default="", choices=["", *NAMES], help="bench: every lane this prompt (default: in turn)")
    ap.add_argument("--ctx", type=int, default=0, help="bench: tokens of a long document before each prompt")
    ap.add_argument("--profile", action="store_true", help="bench: per-op GPU time of the steps (Q27_PROFILE=1)")
    ap.add_argument("--out", default="", help="bench: result JSON (default bench/results/<UTC time>-custom-p6-10-batch.json)")
    args = ap.parse_args()
    lanes_max = max(int(x) for x in args.lanes.split(",")) if args.part == "bench" else args.seqs
    os.environ.setdefault("MAX_SEQS", str(8 if args.widths or args.rule else max(4, lanes_max)))
    if args.part == "kvfull":  # (a pool of 2,112 tokens: four sequences of --gen 600 do not fit; 512 per sequence)
        os.environ.setdefault("STATE_HOST_CACHE", "2")
        os.environ.setdefault("KV_POOL_TOKENS", str(512 * max(4, args.seqs)))
        args.max_tokens = 2048
    if args.profile:
        os.environ["Q27_PROFILE"] = "1"

    from transformers import AutoTokenizer

    from q27lib import Engine, KvFull
    from qwen27b.weights import DFLASH2, DSPINE, TARGET, engine_target, snapshot
    if args.draft == "dspine" and args.part != "bench" and not args.widths and not args.rule:
        os.environ.setdefault("Q27_DRAFT_BLOCK", "8")

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    t0 = time.time()
    eng = Engine(engine_target(args.weights), max_tokens=args.max_tokens, kv_dtype=args.kv_dtype, draft=snapshot(*(DSPINE if args.draft == "dspine" else DFLASH2)))
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
