"""Logprobs end to end (OpenAI's logprobs / top_logprobs: the raw log-softmax of the logits, before the temperature,
the penalties and the cuts).

reference: plain decoding (forward_sample) with logprobs, greedy and with the penalties, against the decode path's
log-softmax one token at a time (forward_topk, its own kernel): the top entries (ids and values) and the token's own
value must agree within --tol (a near tie at the end of the list may swap entries).

spec: speculative decoding (spec_begin / spec_step; pipelined, parked and unparked, which drops the step launched
ahead) must give the tokens and logprobs of plain decoding (with Q27_SKINNY_EXACT=1 the verify rows are computed as
decode rows).

same: logprobs must not change the sampled tokens (temperature 0.7, plain and speculative, with and without
penalties), and a sampled token's value must be the one in its top entries.

    docker compose --profile dev run --rm dev python scripts/logprobs_test.py
"""

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

PROMPTS = {
    "list": "List twenty kinds of fruit, one per line, then say which one you like best.",
    "ja": "日本の都道府県を北から順に十個挙げて、それぞれ一文で説明してください。",
}
TOP = 20


def compare(a, b, tol):
    """Largest difference of two (logprob, top entries) lists of the same tokens; inf when the top ids differ beyond
    a near tie at the end."""
    worst = 0.0
    for (la, ta), (lb, tb) in zip(a, b):
        worst = max(worst, abs(la - lb))
        da, db = dict(ta), dict(tb)
        floor = min([*da.values(), *db.values()])
        for x in da.keys() | db.keys():
            if x in da and x in db:
                worst = max(worst, abs(da[x] - db[x]))
            elif max(da.get(x, floor), db.get(x, floor)) - floor > tol:
                return float("inf")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--presence", type=float, default=1.5)
    ap.add_argument("--frequency", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--kv-dtype", default="nvfp4", choices=["bf16", "fp8", "nvfp4"],
                    help="KV cache (the server's default; with bf16 / fp8 a verify block over 8 rows, DSpine's, is not bit-identical to decode)")
    ap.add_argument("--draft", choices=["dspine", "dflash2"], help="draft model (default $SPEC, else dspine as the server)")
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ.setdefault("Q27_SKINNY_EXACT", "1")
    # Tree verification (Q27_SPEC_PATHS > 1) runs its attention on CUDA cores, decode and chains on tensor cores: a
    # different summation order, which W4A4 carries to visibly different distributions. Both on CUDA cores here.
    if int(os.environ.get("Q27_SPEC_PATHS", "1")) > 1:
        os.environ.setdefault("Q27_ATTN_TC", "0")
    os.environ.setdefault("MAX_SEQS", "2")  # (a parking area)

    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import TARGET, draft_snapshot, engine_target, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(engine_target(args.weights), max_tokens=4096, kv_dtype=args.kv_dtype, draft=draft_snapshot(args.draft))
    ok = True

    def report(good, what):
        nonlocal ok
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {what}", flush=True)

    def plain(ids, t, k, p, pen, lp, seed=0):
        import random
        rng = random.Random(seed)
        eng.reset()
        eng.penalty_init([])
        out, lps = [eng.forward_sample(ids, t, k, p, rng.random(), **pen, logprobs=lp)], []
        lps += eng.logprobs()
        while len(out) < args.gen:
            out.append(eng.forward_sample([out[-1]], t, k, p, rng.random(), **pen, logprobs=lp))
            lps += eng.logprobs()
        return out, lps

    def spec(ids, t, k, p, pen, lp, ahead, seed=1):
        eng.reset()
        eng.penalty_init([])
        out = [eng.spec_begin(ids, t, k, p, seed=seed, uniform=0.5, **pen, logprobs=lp)]
        lps = eng.logprobs()
        steps = 0
        while len(out) < args.gen:
            out += eng.spec_step(ahead=ahead)
            lps += eng.logprobs()
            steps += 1
            if ahead and steps % 7 == 3:  # park and unpark: the step launched ahead is dropped
                eng.seq_unpark(eng.seq_park())
        return out[:args.gen], lps[:args.gen], steps

    for name, text in PROMPTS.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
        for label, pen in (("greedy", {}), ("penalties", dict(presence=args.presence, frequency=args.frequency))):
            toks, lps = plain(ids, 0.0, 1, 1.0, pen, TOP)
            report(len(lps) == len(toks), f"{name} {label}: plain gives {len(lps)} logprobs for {len(toks)} tokens")
            # reference: the decode path's log-softmax of the same tokens
            eng.reset()
            row = eng.forward_topk(ids, TOP)[-1]
            ref = []
            for t in toks:
                d = dict(row)
                ref.append((d.get(t), row))
                row = eng.forward_topk([t], TOP)[0]
            own = [(a, b) for (a, _), (b, _) in zip(lps, ref) if b is not None]
            d_top = compare([(0.0, a) for _, a in lps], [(0.0, b) for _, b in ref], args.tol)
            d_own = max((abs(a - b) for a, b in own), default=0.0)
            outside = len(toks) - len(own)
            report(d_top <= args.tol and d_own <= args.tol and (outside == 0 or pen),
                   f"{name} {label}: reference top {TOP} max diff {d_top:.2e}, token's own {d_own:.2e}"
                   f" ({outside} tokens outside the raw top {TOP})")
            if not pen:  # greedy: the token is the top entry
                report(all(abs(a - t[0][1]) <= 1e-6 and t[0][0] == x for x, (a, t) in zip(toks, lps)),
                       f"{name} {label}: the token is the first top entry")
            for ahead in (False, True):
                s, slp, steps = spec(ids, 0.0, 1, 1.0, pen, TOP, ahead)
                diff = next((i for i, (a, b) in enumerate(zip(toks, s)) if a != b), None)
                d = compare(lps, slp, args.tol) if diff is None and len(slp) == len(lps) else float("inf")
                report(diff is None and d <= args.tol,
                       f"{name} {label}: spec (ahead {ahead}) {steps} steps, tokens "
                       + ("the same" if diff is None else f"differ at {diff}") + f", logprobs max diff {d:.2e}")
        # sampling: logprobs do not change the tokens
        for label, pen in (("sampled", {}), ("sampled + penalties", dict(presence=args.presence, frequency=args.frequency))):
            p0, _ = plain(ids, 0.7, 20, 0.8, pen, -1, seed=3)
            p1, l1 = plain(ids, 0.7, 20, 0.8, pen, 3, seed=3)
            s0, _, _ = spec(ids, 0.7, 20, 0.8, pen, -1, True, seed=5)
            s1, sl1, _ = spec(ids, 0.7, 20, 0.8, pen, 3, True, seed=5)
            report(p0 == p1 and s0 == s1, f"{name} {label}: the same tokens with logprobs (plain {p0 == p1}, spec {s0 == s1})")
            rows = [(x, a, dict(t), t[0][1]) for x, (a, t) in zip(p1 + s1, l1 + sl1)]
            found = sum(x in d for x, _, d, _ in rows)
            report(len(rows) == len(p1) + len(s1) and
                   all(a <= top + 1e-6 and (x not in d or abs(a - d[x]) <= 1e-6) for x, a, d, top in rows),
                   f"{name} {label}: the token's value is its top entry's ({found} of {len(rows)} in the top 3)")
    eng.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
