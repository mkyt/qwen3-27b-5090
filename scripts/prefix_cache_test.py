"""Prefix cache (P4-3 .. P4-6): a request resumed from a checkpoint must produce exactly the tokens of the
same computation without the cache (greedy, plain decode and speculative).

Each case runs a conversation twice: with the cache (prefix_begin restores the deepest checkpoint) and
as a reference on a fresh sequence that feeds the same tokens through the same paths (prompt prefill,
the generation token by token / by speculative steps, then the rest of the next prompt). Resumed chunks
are kept above 64 tokens: shorter chunks take the decode GEMMs and differ in the last bits.
Checkpoints are saved at multiples of 64 below the requested positions (the GDN chunks).

    STATE_HOST_CACHE=1 docker compose --profile dev run --rm dev python scripts/prefix_cache_test.py
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
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--kv-fp8", action="store_true")
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8", "nvfp4"], help="overrides --kv-fp8")
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ.setdefault("STATE_HOST_CACHE", "1")
    os.environ.setdefault("CHECKPOINT_INTERVAL", "1024")

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, engine_target, snapshot

    eng = Engine(engine_target(args.weights), max_tokens=8192, kv_fp8=args.kv_fp8, kv_dtype=args.kv_dtype, draft=snapshot(*DFLASH2))
    prompts = [json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl")]
    base = max(prompts, key=len)
    # The longest prompt that differs from base in its first 64 tokens (several share a document).
    other = max((p for p in prompts if p[:64] != base[:64]), key=len)
    fails = 0

    def generate(spec, n):
        """Greedy continuation (at least n tokens) of the prompt run by begin(); the first comes from it."""
        out = [state["first"]]
        while len(out) < n:
            out += eng.spec_step() if spec else [eng.forward_sample([out[-1]], 0.0, 1, 1.0)]
        return out

    state = {}

    def run_rest(rest, spec):
        state["first"] = eng.spec_begin(rest, 0.0, 1, 1.0, seed=0) if spec else eng.forward_sample(rest, 0.0, 1, 1.0)

    def begin(ids, spec, at=(), cache=True):
        cached = eng.prefix_begin(ids, at=at, spec=spec) if cache else (eng.reset() or 0)
        run_rest(ids[cached:], spec)
        return cached

    def check(name, got, ref):
        nonlocal fails
        ok = got == ref
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n     got {got}\n     ref {ref}"), flush=True)

    n = args.gen
    for spec in (False, True):
        mode = "spec" if spec else "decode"
        # 1) First turn: a 2,000-token prompt with "message" checkpoints at 900 and 1,500 (saved at 896 and
        #    1,472: multiples of 64) and the interval 1,024, then one at the end of the generation.
        p1 = base[:2000]
        c = begin(p1, spec, at=(900, 1500))
        g1 = generate(spec, n)
        n_fed = eng.pos
        check(f"{mode}: fed tokens = prompt + generation - 1", n_fed, len(p1) + len(g1) - 1)
        eng.checkpoint()
        check(f"{mode}: cold prompt not cached", c, 0)

        # 2) Next turn = prompt + generation + 300 new tokens: resumes at the end of the generation
        #    (a partial page), runs the generation's last token + the new ones.
        p2 = p1 + g1 + other[:300]
        c = begin(p2, spec)
        check(f"{mode}: next turn resumes at the end of the generation", c, n_fed)
        g2 = generate(spec, n)[:n]
        begin(p1, spec, cache=False)  # reference: the same computation on a fresh sequence
        generate(spec, n)
        run_rest(p2[n_fed:], spec)
        check(f"{mode}: greedy after the resumed turn matches the reference", generate(spec, n)[:n], g2)

        # 3) Another continuation of the first 1,500 tokens: resumes at the checkpoint 1,472.
        p3 = base[:1500] + other[:400]
        c = begin(p3, spec)
        check(f"{mode}: shared prefix resumes at the checkpoint 1472", c, 1472)
        g3 = generate(spec, n)[:n]
        begin(p3, spec, cache=False)
        check(f"{mode}: greedy after the shared prefix matches a full prefill", generate(spec, n)[:n], g3)

        # 4) The interval checkpoint (1,024) under a prompt that leaves the others.
        p4 = base[:1030] + other[:200]
        c = begin(p4, spec)
        check(f"{mode}: interval checkpoint 1024", c, 1024)
        g4 = generate(spec, n)[:n]
        begin(p4, spec, cache=False)
        check(f"{mode}: greedy after the interval checkpoint matches a full prefill", generate(spec, n)[:n], g4)
        print(f"     {eng.cache_stats()}", flush=True)

    # 5) A speculative generation stopped by max_tokens: the steps output exactly max_new tokens in all,
    #    so the checkpoint at its end is a prefix of the next turn.
    p5 = other[:1200]
    eng.prefix_begin(p5, spec=True)
    out = [eng.spec_begin(p5, 0.0, 1, 1.0, seed=0, max_new=13)]
    while len(out) < 13:
        out += eng.spec_step()
    check("max_new: the steps stop at the token budget", (len(out), eng.pos), (13, len(p5) + 12))
    eng.checkpoint()
    check("max_new: the next turn resumes at the end", eng.prefix_begin(p5 + out + other[1200:1400], spec=True),
          len(p5) + 12)

    # 6) A plain-decode checkpoint has no draft context: speculative requests skip it, plain ones use it.
    q = other[2000:2900]
    begin(q, False, at=(800,))
    check("plain checkpoint not used by a speculative request", eng.prefix_begin(q + [11], spec=True), 0)
    check("plain checkpoint used by a plain request", eng.prefix_begin(q + [11], spec=False), 768)

    print(f"\n{'all passed' if not fails else f'{fails} FAILED'}; {eng.cache_stats()}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
