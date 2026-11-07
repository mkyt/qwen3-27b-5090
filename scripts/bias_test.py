"""Logit bias end to end (OpenAI's logit_bias: a value added to a token's logit before the sampling).

reference: greedy decoding with a bias in the engine (forward_sample) against the same on the host: one token at a
time through forward_topk (the decode path), the biased argmax of its top-64 log-probs fed next. The bias bans two
tokens of the plain output (-100) and favours one (+3): the picks must change and be the same (a near tie, within
--tol, may go either way and ends the comparison).

boost: +100 on a token the model would not write: greedy decoding (plain and speculative) outputs only that token
(it is outside the top 64, so this is the whole-vocabulary path).

spec: greedy with the bias, speculative (spec_begin / spec_step; pipelined, parked and unparked, which drops the step
launched ahead) must output exactly the tokens of plain decoding (Q27_SKINNY_EXACT=1).

sampled: speculative sampling with banned tokens never outputs them, and the draft does not propose them (its
acceptance stays close to that of the same sampling without the bias).

batch: biased and unbiased sequences in batched steps give the tokens they give alone.

min_tokens: with +100 on the end-of-sequence ids and eos_before = the prompt's length + m, the first m tokens are
the model's and token m is the end of the sequence: plain, speculative (greedy and sampled) and in batched steps.

ignore_eos: a short answer decoded past its end (the end-of-sequence id fed like any token): speculative = plain.

logprobs: a banned token is not among the top entries, and a -5 bias moves a token's logprob as the renormalized
distribution says (the logprobs include the bias, as OpenAI's do).

    docker compose --profile dev run --rm dev python scripts/bias_test.py
"""

import argparse
import math
import os
import pathlib
import random
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
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--tol", type=float, default=1e-3, help="reference: a near tie (log-prob difference)")
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ.setdefault("Q27_SKINNY_EXACT", "1")
    os.environ.setdefault("MAX_SEQS", "4")  # (parking areas, batched steps)

    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(engine_target(args.weights), max_tokens=4096, draft=snapshot(*DFLASH2))
    eos = set(tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{'ok  ' if good else 'FAIL'} {msg}", flush=True)

    def chat(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False)
        return list(ids["input_ids"]) if not isinstance(ids, list) else ids

    def plain(ids, bias, n, eos_before=0):
        eng.reset()
        eng.bias_set(bias, eos_before)
        on = bool(bias) or eos_before > 0
        out = [eng.forward_sample(ids, 0.0, 1, 1.0, biased=on)]
        while len(out) < n:
            out.append(eng.forward_sample([out[-1]], 0.0, 1, 1.0, biased=on))
        return out

    def spec(ids, bias, n, ahead=False, park=False, smp=(0.0, 1, 1.0), seed=1, eos_before=0, ignore_eos=False):
        eng.reset()
        eng.bias_set(bias, eos_before)
        on = bool(bias) or eos_before > 0
        out, steps = [eng.spec_begin(ids, *smp, seed=seed, biased=on, ignore_eos=ignore_eos)], 0
        while len(out) < n and not ((smp[0] > 0 or eos_before) and not ignore_eos and any(t in eos for t in out)):
            out += eng.spec_step(ahead=ahead)
            steps += 1
            if park and steps % 7 == 3:  # park and unpark: the step launched ahead is dropped
                eng.seq_unpark(eng.seq_park())
        return out[:n], steps

    biases = {}
    for name, text in PROMPTS.items():
        ids = chat(text)
        base = plain(ids, {}, args.gen)
        # ban two tokens of the plain output (the first two distinct ones after the first 4), favour a third
        seen = []
        for t in base[4:]:
            if t not in seen and t not in eos:
                seen.append(t)
        bias = {seen[0]: -100.0, seen[1]: -100.0, seen[5]: 3.0}
        biases[name] = (ids, bias)
        out = plain(ids, bias, args.gen)
        # reference on the host
        eng.reset()
        eng.bias_set({})
        row = eng.forward_topk(ids, 64)[-1]
        ref, changed, tie, t = [], 0, None, None
        while len(ref) < args.gen:
            vals = sorted(((lp + bias.get(x, 0.0), x) for x, lp in row), reverse=True)
            t = vals[0][1]
            changed += t != row[0][0]
            if out[len(ref)] != t:
                if vals[0][0] - dict((x, v) for v, x in vals).get(out[len(ref)], float("-inf")) <= args.tol:
                    tie = len(ref)
                break
            ref.append(t)
            row = eng.forward_topk([t], 64)[0]
        same = len(ref)
        banned = sum(t in bias and bias[t] < 0 for t in out)
        report((same == args.gen or tie is not None) and changed > 0 and banned == 0,
               f"reference {name}: {same} of {args.gen} tokens the same"
               + (f" (a near tie at {tie})" if tie is not None else
                  "" if same == args.gen else f", differs at {same}: {out[same:same + 6]} vs {ref[same:] + [t]}")
               + f", the bias changed the argmax at {changed}, banned tokens out: {banned}")
        for ahead, park in ((False, False), (True, True)):
            s, steps = spec(ids, bias, args.gen, ahead, park)
            diff = next((k for k, (a, b) in enumerate(zip(out, s)) if a != b), None)
            report(diff is None, f"spec {name} (ahead {ahead}, park {park}): {steps} steps"
                   + ("" if diff is None else f", differs at {diff}: {out[diff:diff + 6]} vs {s[diff:diff + 6]}"))

    # boost: a token far down the list
    ids = chat("Say hello.")
    eng.reset()
    eng.bias_set({})
    row = eng.forward_topk(ids, 64)[-1]
    low = next(x for x in range(1000, 2000) if x not in {i for i, _ in row})
    p = plain(ids, {low: 100.0}, 24)
    s, _ = spec(ids, {low: 100.0}, 24, ahead=True)
    report(set(p) == {low} and set(s) == {low}, f"boost +100 on token {low}: plain {sorted(set(p))[:4]}, "
           f"spec {sorted(set(s))[:4]}")

    # min_tokens: EOS favoured, banned below eos_before
    ids = chat("Say hello.")
    up = {t: 100.0 for t in eos}
    for m in (1, 5, 13):
        at = len(ids) + m
        p = plain(ids, up, m + 1, at)
        s1, _ = spec(ids, up, m + 8, eos_before=at)
        s2, _ = spec(ids, up, m + 8, ahead=True, eos_before=at, smp=(1.0, 20, 0.95), seed=m)
        good = [len(o) > m and not any(t in eos for t in o[:m]) and o[m] in eos for o in (p, s1, s2)]
        report(all(good) and p[:m] == s1[:m], f"min_tokens {m}: end of sequence at token {m} (plain, spec, sampled "
               f"{good}), plain = spec {p[:m] == s1[:m]}")

    # ignore_eos: past the end
    p = plain(ids, {}, 48)
    s1, steps = spec(ids, {}, 48, ahead=True, ignore_eos=True)
    report(any(t in eos for t in p[:-1]) and p == s1, f"ignore_eos: end of sequence at {[i for i, t in enumerate(p) if t in eos][:3]}, "
           f"spec = plain {p == s1} ({steps} steps)")

    # sampled: banned tokens never out; the acceptance stays
    ids, bias = biases["list"]
    ban = {t for t, v in bias.items() if v < 0}
    n_out = n_steps = n_out0 = n_steps0 = hits = 0
    for seed in range(6):
        s, steps = spec(ids, {t: -100.0 for t in ban}, args.gen, ahead=True, smp=(1.0, 20, 0.95), seed=seed)
        hits += sum(t in ban for t in s)
        n_out, n_steps = n_out + len(s), n_steps + steps
        s0, steps0 = spec(ids, {}, args.gen, ahead=True, smp=(1.0, 20, 0.95), seed=seed)
        n_out0, n_steps0 = n_out0 + len(s0), n_steps0 + steps0
    report(hits == 0 and n_out / n_steps > 0.8 * n_out0 / n_steps0,
           f"sampled with banned tokens: {hits} banned tokens out, {n_out / n_steps:.2f} tokens per step "
           f"(without the bias {n_out0 / n_steps0:.2f})")

    # logprobs: a banned token's logprob includes the bias
    ids, bias = biases["list"]
    eng.reset()
    eng.bias_set({})
    first = eng.forward_sample(ids, 0.0, 1, 1.0, logprobs=0)
    lp0 = eng.logprobs()[0][0]
    eng.reset()
    eng.bias_set({first: -100.0})
    eng.forward_sample(ids, 0.0, 1, 1.0, logprobs=20, biased=True)
    tops = [i for i, _ in eng.logprobs()[0][1]]
    eng.reset()
    eng.bias_set({first: -5.0})
    t5 = eng.forward_sample(ids, 0.0, 1, 1.0, logprobs=20, biased=True)
    lp5 = dict(eng.logprobs()[0][1]).get(first)
    want = lp0 - 5.0 - math.log(1.0 - math.exp(lp0) + math.exp(lp0 - 5.0))  # (the distribution renormalized)
    report(first not in tops and lp5 is not None and abs(lp5 - want) < 0.02,
           f"logprobs: banned token {first} not among the top 20; with -5 its logprob {lp0:.3f} -> "
           f"{lp5 if lp5 is None else round(lp5, 3)} (expected {want:.3f}; next token {t5})")

    # batch: biased and unbiased lanes = alone
    # (the last lane: min_tokens 40 with EOS favoured: its EOS at token 40, as alone)
    seqs = [(chat(PROMPTS["list"]), biases["list"][1], 0), (chat("Write a haiku about rain."), {}, 0),
            (chat(PROMPTS["ja"]), biases["ja"][1], 0), (chat("Say hello."), up, 40)]
    n = 96
    alone = [spec(i, b, n, eos_before=len(i) + m if m else 0)[0] for i, b, m in seqs]
    eng.reset()
    outs, where = [], []
    for i, (ids, b, m) in enumerate(seqs):
        if i:
            where[-1] = eng.seq_park()
        eng.bias_set(b, len(ids) + m if m else 0)
        eng.prefix_begin(ids, spec=True)
        outs.append([eng.spec_begin(ids, 0.0, 1, 1.0, seed=1, uniform=0.5, biased=bool(b))])
        where.append(None)
    rng = random.Random(5)
    batched = 0
    while any(len(o) < n for o in outs):
        live = [i for i in range(len(seqs)) if len(outs[i]) < n and not any(t in eos for t in outs[i])]
        if len(live) < 2:
            break
        lanes = rng.sample(live, rng.randint(2, len(live)))
        for i, toks in zip(lanes, eng.spec_step_lanes([where[i] for i in lanes])):
            outs[i] += toks
        batched += 1
    for i, ((ids, b, m), a) in enumerate(zip(seqs, alone)):
        k = min(len(a), len(outs[i]), n)
        if m:
            k = min(len(a), len(outs[i]))
            report(k == m + 1 and a[:k] == outs[i][:k] and a[m] in eos, f"batch lane {i} (min_tokens {m}): "
                   f"{len(outs[i])} tokens, the end at {[j for j, t in enumerate(outs[i]) if t in eos][:1]}, as alone "
                   f"{a[:k] == outs[i][:k]} ({batched} batched steps)")
            continue
        report(k > 0 and a[:k] == outs[i][:k], f"batch lane {i} ({'biased' if b else 'free'}): {k} tokens "
               + ("the same" if a[:k] == outs[i][:k] else "differ") + f" ({batched} batched steps)")
    eng.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
