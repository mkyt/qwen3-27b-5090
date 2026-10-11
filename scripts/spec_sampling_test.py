"""Losslessness of speculative sampling (docs/architecture.md §13): the tokens of a speculative step must follow
the target distribution, i.e. the same distribution as plain sampling.

For N seeds, both modes run the same prompt; the first token is sampled by the same (plain) sampler
with the same uniform in both, so runs are paired on it. Then plain decoding samples tokens 2 and 3,
and speculative decoding takes them from its first step(s) (drafted and accepted, or resampled from
the residual). The empirical distributions of token 2 given the most frequent first tokens, and of
token 3, are compared with a chi-square test of homogeneity (categories with small counts merged).
A wrong acceptance rule or residual shows up as tiny p-values; a lossless sampler gives p-values that
are uniform on (0, 1) across tests.

    docker compose --profile dev run --rm dev python scripts/spec_sampling_test.py --n 2000
"""

import argparse
import collections
import json
import math
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

PROMPTS = {
    "story": "Write the opening sentence of a short story about a cat who learns to fly.",
    "ja": "猫についての短い詩を書いてください。",
}


def chi2_homogeneity(a: collections.Counter, b: collections.Counter, min_expected=5.0):
    """Two-sample chi-square on the categories of a and b; rare categories merged into one."""
    na, nb = sum(a.values()), sum(b.values())
    cats = sorted(set(a) | set(b), key=lambda c: -(a[c] + b[c]))
    rows, rest = [], [0, 0]
    for c in cats:
        tot = a[c] + b[c]
        if tot * min(na, nb) / (na + nb) >= min_expected:
            rows.append((a[c], b[c]))
        else:
            rest[0] += a[c]
            rest[1] += b[c]
    if sum(rest) > 0:
        rows.append(tuple(rest))
    if len(rows) < 2:
        return 0.0, 0, 1.0
    stat = 0.0
    for x, y in rows:
        t = x + y
        ex, ey = t * na / (na + nb), t * nb / (na + nb)
        stat += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey
    dof = len(rows) - 1
    import torch  # (no scipy in the image) chi2 survival function = regularized upper incomplete gamma
    return stat, dof, float(torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64),
                                                    torch.tensor(stat / 2, dtype=torch.float64)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    # Verify rows must be computed like decode rows: by default the skinny GEMMs split K differently for M = 1
    # and M > 1, and with W4A4 that shifts the sampled distributions measurably (e.g. P(first token 'The')
    # 0.35 vs 0.41 for the story prompt), which this test would report as a lossy sampler (2026-10-10).
    os.environ.setdefault("Q27_SKINNY_EXACT", "1")

    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(snapshot(*TARGET), max_tokens=1024, draft=snapshot(*DFLASH2))
    smp = (args.temperature, args.top_k, args.top_p)
    report = {"args": vars(args), "tests": []}
    for name in args.prompts.split(","):
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPTS[name]}], add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
        plain, spec, accepted = [], [], 0
        t0 = time.perf_counter()
        for s in range(args.n):
            rng = random.Random(1000003 * s + 7)
            u1, u2, u3 = rng.random(), rng.random(), rng.random()
            eng.reset()
            a1 = eng.forward_sample(ids, *smp, u1)
            a2 = eng.forward_sample([a1], *smp, u2)
            a3 = eng.forward_sample([a2], *smp, u3)
            plain.append((a1, a2, a3))
            eng.reset()
            b = [eng.spec_begin(ids, *smp, seed=s * 7919 + 1, uniform=u1)]
            first = eng.spec_step()
            accepted += len(first) - 1
            b += first
            while len(b) < 3:
                b += eng.spec_step()
            spec.append(tuple(b[:3]))
            if (s + 1) % 500 == 0:
                print(f"{name}: {s + 1}/{args.n} ({time.perf_counter() - t0:.0f} s)", flush=True)
        assert all(p[0] == q[0] for p, q in zip(plain, spec)), "first tokens differ (pairing broken)"
        firsts = collections.Counter(p[0] for p in plain)
        tests = []
        for t1, cnt in firsts.most_common(3):
            if cnt < 200:
                break
            a = collections.Counter(p[1] for p in plain if p[0] == t1)
            b = collections.Counter(q[1] for q in spec if q[0] == t1)
            tests.append((f"token 2 | token 1 = {tok.decode([t1])!r} ({cnt})", a, b))
        tests.append(("token 2", collections.Counter(p[1] for p in plain), collections.Counter(q[1] for q in spec)))
        tests.append(("token 3", collections.Counter(p[2] for p in plain), collections.Counter(q[2] for q in spec)))
        print(f"--- {name}: mean accepted drafts in the first step {accepted / args.n:.2f}")
        for label, a, b in tests:
            stat, dof, p = chi2_homogeneity(a, b)
            print(f"{label:45s} chi2 {stat:8.2f}  dof {dof:3d}  p {p:.3f}")
            report["tests"].append({"prompt": name, "test": label, "chi2": stat, "dof": dof, "p": p})
    eng.close()
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    ps = [t["p"] for t in report["tests"]]
    print(f"min p {min(ps):.3f} over {len(ps)} tests (Bonferroni 0.01 / {len(ps)} = {0.01 / len(ps):.4f})")
    sys.exit(0 if min(ps) > 0.01 / len(ps) else 1)


if __name__ == "__main__":
    main()
