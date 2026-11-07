"""Structured output end to end (grammar masks, docs/architecture.md §12): the engine with the frontend's grammars
(frontend/structured.py) and mask writers (frontend/worker.py).

same: with the masks, speculative decoding (spec_draft + masks + spec_step) must give the tokens of plain decoding
(forward_sample with row 0 masked; Q27_SKINNY_EXACT=1 computes the verify rows as decode rows), greedy, for a JSON
schema, any JSON object, a regex and a choice; the output must match the grammar and end with EOS.

late: thinking on, the grammar starting after </think> (spec_begin's until ends the step there, Engine.spec_set_masked
masks from the next one; also with the step launched ahead, which it drops): the same tokens as plain decoding masked
from there on.

sampled: temperature 1.0 / top-p 0.95 / top-k 20 (and with presence 1.5, logprobs 5): every output matches.

batch: masked and unmasked sequences in batched steps (spec_draft_lanes + masks + spec_step_lanes) give the tokens
each gives alone.

speed (--speed): ms per step and tokens per step with and without the masks (one prompt, greedy and sampled).

    docker compose --profile dev run --rm dev python scripts/structured_test.py [--speed]
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
sys.path.insert(0, str(ROOT / "frontend"))

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer", "minimum": 0},
    "languages": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "address": {"type": "object", "properties": {"city": {"type": "string"},
                                                 "zip": {"type": "string", "pattern": "^[0-9]{3}-[0-9]{4}$"}},
                "required": ["city", "zip"], "additionalProperties": False},
    "status": {"enum": ["active", "inactive"]}},
    "required": ["name", "age", "languages", "address", "status"], "additionalProperties": False}
CASES = [  # (name, prompt, spec)
    ("schema", "Make up a person who lives in Japan and describe them.", ("json", SCHEMA)),
    ("object", "東京の観光地を三つ、JSON で紹介してください。", ("json", None)),
    ("regex", "What is the postal code of Tokyo Station? Reply with the code only.", ("regex", r"[0-9]{3}-[0-9]{4}")),
    ("choice", "Is the sky green? Answer in a full sentence.", ("choice", ["yes", "no", "it depends"])),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--weights", default="nvidia", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    ap.add_argument("--speed", action="store_true", help="also time the steps with and without the masks")
    args = ap.parse_args()
    os.environ.setdefault("Q27_SKINNY_EXACT", "1")
    os.environ.setdefault("MAX_SEQS", "4")  # (parking areas and batched steps)

    import llguidance
    from transformers import AutoTokenizer

    import structured
    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot
    from worker import write_mask, write_masks

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(engine_target(args.weights), max_tokens=4096, draft=snapshot(*DFLASH2))
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    think_end = tok.convert_tokens_to_ids("</think>")
    lltok = llguidance.LLTokenizer(str(pathlib.Path(snapshot(*TARGET)) / "tokenizer.json"),
                                   n_vocab=eng.mask_row(0)[1] * 8, eos_token=sorted(eos))
    ok = True

    def report(good, what):
        nonlocal ok
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {what}", flush=True)

    def chat(text, thinking=False):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=thinking)
        return list(ids["input_ids"]) if not isinstance(ids, list) else ids

    def matcher(spec, lead_ws=False):
        return llguidance.LLMatcher(lltok, structured.grammar(spec, lead_ws=lead_ws), log_level=0)

    def take(m, out, toks):
        """Appends a step's tokens up to EOS (True: EOS came) through matcher m (None: not masked yet)."""
        for t in toks:
            if t in eos:
                if m is not None:
                    assert m.is_accepting(), "EOS where the grammar does not allow it"
                return True
            if m is not None:
                assert m.consume_token(t), f"token {t} outside the grammar: {m.get_error()}"
            out.append(t)
        return False

    def plain(ids, spec, smp=(0.0, 1, 1.0), seed=0, after=None, pen=None):
        """Plain decoding; the grammar from the start, or after token `after`."""
        rng = random.Random(seed)
        eng.reset()
        eng.penalty_init([])
        m = matcher(spec, lead_ws=after is not None)
        live = m if after is None else None
        out, feed = [], ids
        while len(out) < args.gen:
            if live is not None:
                write_mask(eng, live, 0)
            t = eng.forward_sample(feed, *smp, rng.random(), masked=live is not None, **(pen or {}))
            if take(live, out, [t]):
                return out, True
            if live is None and t == after:
                live = m
            feed = [t]
        return out, False

    def spec(ids, spec_, smp=(0.0, 1, 1.0), seed=1, after=None, ahead=False, pen=None, lp=-1):
        rng = random.Random(seed)
        eng.reset()
        eng.penalty_init([])
        m = matcher(spec_, lead_ws=after is not None)
        live = m if after is None else None
        if live is not None:
            write_mask(eng, live, 0)
        out, steps = [], 0
        t = eng.spec_begin(ids, *smp, seed=seed, uniform=rng.random(), masked=live is not None, logprobs=lp,
                           until=after, **(pen or {}))
        toks = [t]
        while True:
            for i, t in enumerate(toks):
                if take(live, out, [t]):
                    return out, True, steps
                if live is None and t == after:  # (until: the step ends there)
                    if toks[i + 1:]:
                        return None, None, steps
                    live = m
                    eng.spec_set_masked()
            if len(out) >= args.gen:
                return out, False, steps
            if live is not None:
                write_masks(eng, live, eng.spec_draft(), 0, eos)
            toks = eng.spec_step(ahead=ahead)
            steps += 1

    def valid(name, spec_, out, done, cut_ok=False):
        """Whether the output matches (its tokens went through the matcher already); cut_ok: an output max_tokens cut
        (not done) counts."""
        text = tok.decode(out)
        if not done:
            return cut_ok, text + " [cut]"
        kind, v = spec_
        if kind == "json":
            try:
                obj = json.loads(text)
            except ValueError:
                return False, text
            good = isinstance(obj, dict) and (v is None or (set(obj) == set(v["required"]) and
                                                             obj["status"] in ("active", "inactive")))
            return good, text
        if kind == "choice":
            return text in v, text
        import re
        return re.fullmatch(v, text) is not None, text

    print("== same (greedy: speculative = plain, masked)", flush=True)
    for name, prompt, sp in CASES:
        ids = chat(prompt)
        p, pd = plain(ids, sp)
        s, sd, steps = spec(ids, sp)
        good, text = valid(name, sp, s, sd)
        diff = next((i for i, (a, b) in enumerate(zip(p, s)) if a != b), None)
        report(p == s and pd == sd, f"{name}: spec {len(s)} tokens in {steps} steps, plain {len(p)}: "
               + ("the same" if p == s else f"differ at {diff}"))
        report(good, f"{name}: matches the format: {text[:100]!r}")

    print("== late (thinking on: the grammar after </think>)", flush=True)
    ids = chat("Name one fruit and its color, as JSON. Think briefly.", thinking=True)
    sp = ("json", {"type": "object", "properties": {"fruit": {"type": "string"}, "color": {"type": "string"}},
                   "required": ["fruit", "color"], "additionalProperties": False})
    args_gen, args.gen = args.gen, 1500
    p, pd = plain(ids, sp, after=think_end)
    for ahead in (False, True):
        s, sd, steps = spec(ids, sp, after=think_end, ahead=ahead)
        if s is None:
            report(False, f"late (ahead {ahead}): the step with </think> went on past it")
            continue
        report(p == s and pd and sd, f"late (ahead {ahead}): spec {len(s)} tokens in {steps} steps, plain {len(p)}, "
               + ("the same" if p == s else "differ") + f"; answer {tok.decode(s[s.index(think_end) + 1:])!r}"
               if think_end in s else "late: no </think>")
    args.gen = args_gen

    print("== sampled", flush=True)
    for name, prompt, sp in CASES:
        ids = chat(prompt)
        goods = []
        for seed, pen, lp in ((11, None, -1), (12, dict(presence=1.5), 5)):
            s, sd, steps = spec(ids, sp, smp=(1.0, 20, 0.95), seed=seed, pen=pen, lp=lp)
            goods.append(valid(name, sp, s, sd, cut_ok=True))
        report(all(g for g, _ in goods), f"{name}: sampled outputs match: {[t[:60] for _, t in goods]}")

    print("== batch (masked and unmasked lanes = alone)", flush=True)
    seqs = [(chat(CASES[0][1]), CASES[0][2]), (chat("Write a haiku about rain."), None), (chat(CASES[3][1]), CASES[3][2]),
            (chat(CASES[1][1]), CASES[1][2])]
    n = 96
    alone = []
    for ids, sp in seqs:
        if sp is None:
            eng.reset()
            out = [eng.spec_begin(ids, 0.0, 1, 1.0, seed=1, uniform=0.5)]
            while len(out) < n and not any(t in eos for t in out):
                out += eng.spec_step(ahead=False)
            alone.append(out[:n])
        else:
            s, _, _ = spec(ids, sp)
            alone.append(s[:n])
    eng.reset()
    ms, outs, where = [], [], []
    for i, (ids, sp) in enumerate(seqs):
        if i:
            where[-1] = eng.seq_park()
        m = matcher(sp) if sp else None
        if m:
            write_mask(eng, m, 0)
        eng.prefix_begin(ids, spec=True)
        outs.append([eng.spec_begin(ids, 0.0, 1, 1.0, seed=1, uniform=0.5, masked=m is not None)])
        if m:
            m.consume_token(outs[-1][0])
        ms.append(m)
        where.append(None)
    done = [False] * len(seqs)
    rng = random.Random(5)
    batched = 0
    while not all(done):
        live = [i for i in range(len(seqs)) if not done[i]]
        lanes = rng.sample(live, rng.randint(min(2, len(live)), len(live))) if len(live) >= 2 else live
        if len(lanes) < 2:
            break
        areas = [where[i] for i in lanes]
        if any(ms[i] for i in lanes):
            for z, (i, chain) in enumerate(zip(lanes, eng.spec_draft_lanes(areas))):
                if ms[i]:
                    write_masks(eng, ms[i], chain, 8 * z, eos)
        for i, toks in zip(lanes, eng.spec_step_lanes(areas)):
            for t in toks:
                if t in eos or len(outs[i]) >= n:
                    outs[i].append(t)
                    done[i] = True
                    break
                outs[i].append(t)
                if ms[i]:
                    ms[i].consume_token(t)
        batched += 1
    for i, ((ids, sp), a) in enumerate(zip(seqs, alone)):
        b = [t for t in outs[i] if t not in eos][:n]
        a = [t for t in a if t not in eos][:n]
        k = min(len(a), len(b))
        report(a[:k] == b[:k] and k > 0, f"batch lane {i} ({'masked' if sp else 'free'}): {k} tokens "
               + ("the same" if a[:k] == b[:k] else "differ") + f" ({batched} batched steps)")
    eng.reset()

    if args.speed:
        print("== speed", flush=True)
        os.environ.pop("Q27_SKINNY_EXACT", None)
        eng.configure(Q27_SKINNY_EXACT=None)
        ids = chat(CASES[1][1])
        for smp in ((0.0, 1, 1.0), (1.0, 20, 0.95)):
            for rep in range(2):
                for masked in (False, True):
                    eng.reset()
                    m = matcher(CASES[1][2]) if masked else None
                    if m:
                        write_mask(eng, m, 0)
                    first = eng.spec_begin(ids, *smp, seed=7, uniform=0.5, masked=masked)
                    if m:
                        m.consume_token(first)
                    steps = toks = 0
                    t0 = None
                    while toks < 400:
                        if steps == 10:
                            t0, toks0 = time.perf_counter(), toks
                        if m:
                            write_masks(eng, m, eng.spec_draft(), 0, eos)
                        new = eng.spec_step(ahead=True)
                        stop = any(t in eos for t in new)
                        if m:
                            for t in new:
                                if t in eos:
                                    break
                                m.consume_token(t)
                        steps += 1
                        toks += len(new)
                        if stop:
                            break
                    if t0 is None or steps < 30:
                        print(f"  {smp} masked {masked}: too short ({steps} steps)")
                        continue
                    dt = time.perf_counter() - t0
                    print(f"  temp {smp[0]} masked {masked!s:5}: {dt * 1e3 / (steps - 10):.3f} ms/step, "
                          f"{(toks - toks0) / (steps - 10):.2f} tokens/step ({steps - 10} steps)", flush=True)
    eng.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
