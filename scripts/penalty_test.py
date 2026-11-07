"""Presence / frequency penalties end to end (OpenAI's: logit -= presence * (c > 0) + frequency * c, c the token's
count in the output so far).

reference: greedy decoding with penalties in the engine (forward_sample) against the same on the host: one token at a
time through forward_topk (the decode path), the penalized argmax of its top-64 log-probs fed next. The tokens must
be the same (a near tie, within --tol after the penalties, may go either way and ends the comparison), and the
penalties must change some of the picks (positions where the plain argmax is a penalized token).

spec: greedy with penalties, speculative (spec_begin / spec_step; pipelined, parked and unparked, which drops the
step launched ahead) must output exactly the tokens of plain decoding (with Q27_SKINNY_EXACT=1 the verify rows are
computed as decode rows).

    docker compose --profile dev run --rm dev python scripts/penalty_test.py
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--presence", type=float, default=1.5)
    ap.add_argument("--frequency", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-3, help="reference: a near tie (log-prob difference)")
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ.setdefault("Q27_SKINNY_EXACT", "1")
    os.environ.setdefault("MAX_SEQS", "2")  # (a parking area)

    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(engine_target(args.weights), max_tokens=4096, draft=snapshot(*DFLASH2))
    pen = dict(presence=args.presence, frequency=args.frequency)
    ok = True
    for name, text in PROMPTS.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
        # plain greedy with penalties
        eng.reset()
        eng.penalty_init([])
        plain = [eng.forward_sample(ids, 0.0, 1, 1.0, **pen)]
        while len(plain) < args.gen:
            plain.append(eng.forward_sample([plain[-1]], 0.0, 1, 1.0, **pen))
        # reference: the same on the host, from the decode path's top 64 per step
        eng.reset()
        row = eng.forward_topk(ids, 64)[-1]
        ref, counts, changed, tie = [], {}, 0, None
        while len(ref) < args.gen:
            vals = sorted(((lp - (args.presence * (counts.get(x, 0) > 0) + args.frequency * counts.get(x, 0)), x)
                           for x, lp in row), reverse=True)
            t = vals[0][1]
            changed += t != row[0][0]
            if len(ref) < len(plain) and plain[len(ref)] != t:
                if vals[0][0] - dict((x, v) for v, x in vals).get(plain[len(ref)], float("-inf")) <= args.tol:
                    tie = len(ref)
                break
            ref.append(t)
            counts[t] = counts.get(t, 0) + 1
            row = eng.forward_topk([t], 64)[0]
        same = len(ref) if len(ref) < args.gen else args.gen
        good = (same == args.gen or tie is not None) and changed > 0
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} reference {name}: {same} of {args.gen} tokens the same"
              + (f" (a near tie at {tie})" if tie is not None else
                 "" if same == args.gen else f", differs at {same}: {plain[same:same + 6]} vs {ref[same:] + [t]}")
              + f", the penalties changed the argmax at {changed}", flush=True)
        # speculative greedy with penalties: the same tokens
        for ahead in (False, True):
            eng.reset()
            eng.penalty_init([])
            spec = [eng.spec_begin(ids, 0.0, 1, 1.0, seed=1, **pen)]
            steps = 0
            while len(spec) < args.gen:
                spec += eng.spec_step(ahead=ahead)
                steps += 1
                if ahead and steps % 7 == 3:  # park and unpark: the step launched ahead is dropped (counts undone)
                    eng.seq_unpark(eng.seq_park())
            spec = spec[:args.gen]
            diff = next((k for k, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
            ok &= diff is None
            print(f"{'ok  ' if diff is None else 'FAIL'} spec {name} (ahead {ahead}): {steps} steps"
                  + ("" if diff is None else f", differs at {diff}: {plain[diff:diff + 6]} vs {spec[diff:diff + 6]}"),
                  flush=True)
    eng.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
