"""Several sequences on one engine (Phase 6, P6-1): sequences interleaved by parking / swapping their state
must produce exactly the greedy tokens each one produces alone, and a sequence preempted to the prefix
cache (seq_suspend) must continue with the same tokens after it resumes from its checkpoint.

Each sequence's prompt is split into prefill() calls of one chunk (other sequences run in between), with
different sampling settings per sequence (cached graphs), plain decode and speculative steps, steps with
and without the next one launched ahead. A second part fills a small KV pool (KV_POOL_TOKENS) until the
engine raises KvFull, preempts the other sequence and retries (the idle parking areas, KV pages since P7-13, are
filled with short parked sequences first).

    docker compose --profile dev run --rm dev python scripts/multiseq_test.py
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--kv-dtype", default="fp8", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--part", default="switch,swaptime,kvfull")
    args = ap.parse_args()
    os.environ.setdefault("STATE_HOST_CACHE", "2")
    os.environ.setdefault("MAX_SEQS", "4")
    os.environ.setdefault("KV_POOL_TOKENS", "24576")  # (part 2 fills it; part 1 fits)

    from q27lib import Engine, KvFull
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot

    eng = Engine(engine_target(args.weights), max_tokens=16384, kv_dtype=args.kv_dtype, draft=snapshot(*DFLASH2))
    chunk = 2048
    prompts = [json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl")]
    by_len = sorted(prompts, key=len)
    fails = 0

    def check(name, got, ref):
        nonlocal fails
        ok = got == ref
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got {got[:40] if isinstance(got, list) else got}\n     ref {ref[:40] if isinstance(ref, list) else ref}"),
              flush=True)

    # Sequences: (prompt, speculative, top_k). Greedy everywhere (temperature 0); top_k changes the graph key.
    cases = [(by_len[-1][:5000], True, 1), (by_len[-3][:3100], True, 20), (by_len[2], False, 1), (by_len[4], True, 1)]

    steps = {}

    def solo(ids, spec, top_k, n):
        cached = eng.prefix_begin(ids, spec=spec)
        rest = ids[cached:]
        out = [eng.spec_begin(rest, 0.0, top_k, 1.0, seed=7) if spec else eng.forward_sample(rest, 0.0, top_k, 1.0)]
        k = 0
        while len(out) < n:
            out += eng.spec_step() if spec else [eng.forward_sample([out[-1]], 0.0, top_k, 1.0)]
            k += 1
        steps[len(ids)] = k
        return out  # (a speculative step may give a few more than n)

    if "switch" in args.part:
        print(f"park slots {eng.park_slots}", flush=True)
        refs = [solo(ids, spec, k, args.gen)[:args.gen] for ids, spec, k in cases]

        # Interleaved: a random sequence takes a turn of 1..3 operations (a prefill chunk, a step) at a time.
        rng = random.Random(args.seed)
        seqs = [dict(ids=ids, spec=spec, k=k, out=[], fed=None, area=None, steps=0) for ids, spec, k in cases]
        cur = None
        swaps, swap_s = 0, 0.0
        t0 = time.time()
        while any(len(s["out"]) < args.gen for s in seqs):
            s = rng.choice([s for s in seqs if len(s["out"]) < args.gen])
            if s is not cur:
                if cur is not None and len(cur["out"]) < args.gen:
                    if s["area"] is not None:
                        t1 = time.time()
                        eng.seq_swap(s["area"])
                        swap_s += time.time() - t1
                        cur["area"], s["area"] = s["area"], None
                        swaps += 1
                    else:
                        cur["area"] = eng.seq_park()
                elif s["area"] is not None:
                    eng.seq_unpark(s["area"])
                    s["area"] = None
                cur = s
                if s["fed"] is None:  # first turn: restore the cached prefix (a fresh sequence otherwise)
                    s["fed"] = eng.prefix_begin(s["ids"], spec=s["spec"])
            for _ in range(rng.randint(1, 3)):
                if len(s["out"]) >= args.gen:
                    break
                rest = s["ids"][s["fed"]:]
                if not s["out"] and len(rest) > chunk:
                    s["fed"] += eng.prefill(rest, chunk, spec=s["spec"])
                elif not s["out"]:
                    s["out"] = [eng.spec_begin(rest, 0.0, s["k"], 1.0, seed=7) if s["spec"]
                                else eng.forward_sample(rest, 0.0, s["k"], 1.0)]
                elif s["spec"]:
                    s["out"] += eng.spec_step(ahead=rng.random() < 0.5)
                    s["steps"] += 1
                else:
                    s["out"].append(eng.forward_sample([s["out"][-1]], 0.0, s["k"], 1.0))
        print(f"interleaved in {time.time() - t0:.1f} s ({swaps} swaps, {1000 * swap_s / max(swaps, 1):.2f} ms each)",
              flush=True)
        for i, (s, ref) in enumerate(zip(seqs, refs)):
            check(f"sequence {i} ({len(s['ids'])} tokens, {'spec' if s['spec'] else 'decode'}, top_k {s['k']}) "
                  f"interleaved = alone", s["out"][:args.gen], ref)
            if s["spec"]:  # (the draft's proposals depend on its context ring: same steps if it was kept)
                n_ref = steps[len(s["ids"])]
                check(f"sequence {i}: same number of verify steps (draft state kept)", s["steps"], n_ref)
        for s in seqs:
            if s["area"] is not None:
                eng.seq_drop(s["area"])

        # Preemption: park a sequence mid-generation, suspend it to the cache, resume it from its tokens.
        # The resumed sequence runs its last token alone (M = 1) where the uninterrupted one had it in a
        # verify block (M = 8): the skinny GEMMs only agree bit for bit with Q27_SKINNY_EXACT.
        eng.configure(Q27_SKINNY_EXACT=1)
        ids, spec, k = cases[0]
        refs[0] = solo(ids, spec, k, args.gen)[:args.gen]
        half = args.gen // 2
        out = solo(ids, spec, k, half)
        area = eng.seq_park()
        solo(cases[1][0], True, 20, 16)  # (another sequence runs meanwhile)
        eng.seq_suspend(area)
        full = ids + out
        cached = eng.prefix_begin(full, spec=spec)
        check("suspended sequence resumes from its checkpoint", cached, len(full) - 1)
        more = [eng.spec_begin(full[cached:], 0.0, k, 1.0, seed=7)]
        while len(out) + len(more) < args.gen:
            more += eng.spec_step()
        check("suspended sequence continues with the same tokens", (out + more)[:args.gen], refs[0])
        eng.configure(Q27_SKINNY_EXACT=None)

    if "swaptime" in args.part:
        # Switching cost without a step in flight (the scheduler ends a turn with spec_step(ahead=False)).
        solo(cases[0][0], True, 1, 16)
        area = eng.seq_park()
        solo(cases[1][0], True, 20, 16)
        eng.spec_step(ahead=False)
        for label, fn in (("seq_swap", lambda: eng.seq_swap(area)), ("spec_step(ahead=False)", lambda: eng.spec_step(ahead=False))):
            t1, n = time.time(), 20
            for _ in range(n):
                fn()
            print(f"info {label}: {1000 * (time.time() - t1) / n:.2f} ms", flush=True)
        eng.seq_drop(area)

    if "kvfull" in args.part:
        # The KV pool (KV_POOL_TOKENS) cannot hold two long sequences: the second one hits KvFull, the first
        # is suspended to the cache and the second retries; then the first resumes.
        eng.reset()
        a = (by_len[-1] * 8)[:15000]
        b = (by_len[-2] * 8)[:15000]
        ref_a = solo(a, True, 1, 32)[:32]
        eng.reset()
        ref_b = solo(b, True, 1, 32)[:32]
        eng.reset()
        out_a = solo(a, True, 1, 16)
        area = eng.seq_park()
        # Idle parking areas are KV pages too (P7-13): fill the others with short sequences, so the pool is
        # KV_POOL_TOKENS as before and cannot hold both long ones.
        fillers = []
        for i in range(eng.park_slots - 1):
            eng.reset()
            eng.forward_sample(by_len[i][:64], 0.0, 1, 1.0)
            fillers.append(eng.seq_park())
        hit = False
        fed = eng.prefix_begin(b, spec=True)
        while True:
            try:
                rest = b[fed:]
                if len(rest) > chunk:
                    fed += eng.prefill(rest, chunk, spec=True)
                    continue
                out_b = [eng.spec_begin(rest, 0.0, 1, 1.0, seed=7)]
                break
            except KvFull:
                hit = True
                fed = eng.pos
                print(f"KvFull at {fed}: suspending the other sequence", flush=True)
                eng.seq_suspend(area)
        check("second long sequence hits KvFull", hit, True)
        while len(out_b) < 32:
            out_b += eng.spec_step()
        check("sequence after KvFull = alone", out_b[:32], ref_b)
        full = a + out_a
        cached = eng.prefix_begin(full, spec=True)
        rest = full[cached:]
        while len(rest) > chunk:
            cached += eng.prefill(rest, chunk, spec=True)
            rest = full[cached:]
        more = [eng.spec_begin(rest, 0.0, 1, 1.0, seed=7)]
        while len(out_a) + len(more) < 32:
            more += eng.spec_step()
        # Its checkpoint was evicted for the other sequence's pages, so its generated tokens ran again as a
        # prompt chunk (prefill GEMMs, not the decode ones): the continuation may differ in the last bits.
        same = (out_a + more)[:32] == ref_a
        print(f"info preempted sequence resumed from {cached} of {len(full)} tokens; continuation "
              f"{'identical' if same else 'differs (recomputed as a prompt)'}", flush=True)
        for f in fillers:
            eng.seq_drop(f)

    print("PASS" if fails == 0 else f"{fails} FAILED", flush=True)
    eng.close()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
