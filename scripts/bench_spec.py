"""Speculative decoding (DFlash2) of the engine, in-process: decode speed and acceptance length.

Uses the decode prompts of bench_agentic.py (code / json / prose, chat template, thinking off) and the
checkpoint's default sampling (temperature 1.0, top-k 20, top-p 0.95) unless --temperature is given.
Decode tok/s counts like bench_agentic.py: (tokens - 1) / (time of the last token - time of the first).
The acceptance length is the generated tokens per verify step (the first token comes from the prompt).

--check-greedy N: greedy, N tokens per prompt, speculative vs plain decoding; the token sequences must
be identical (the verify rows are bit-identical to decode steps; the run sets Q27_SKINNY_EXACT=1, the
split-K policy that does not depend on the number of rows).

--ab CFG [CFG ...]: A/B of runtime switches in one process (like ab_decode.py): each config ("-" or
KEY=VALUE,...) runs greedy speculative decoding of the prompts in turn for --rounds rounds; greedy makes
the acceptance identical across configs, so the median ms per step compares the step time alone.

    docker compose --profile dev run --rm dev python scripts/bench_spec.py
    docker compose --profile dev run --rm dev python scripts/bench_spec.py --check-greedy 256
"""

import argparse
import datetime
import json
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

PROMPTS = {  # scripts/bench_agentic.py scenario_decode
    "code": "Write a complete, well-documented Python module implementing a thread-safe LRU cache "
            "with TTL expiry, plus pytest unit tests. Output only code.",
    "json": "Generate a JSON array of 25 fictional employee records with fields id, name, email, "
            "department, salary, start_date, skills (array). Output only JSON.",
    "prose": "Explain in detail how a CPU executes a program, from fetching instructions to "
             "branch prediction and caches.",
    # (not in bench_agentic.py; select with --prompts)
    "ja": "大規模言語モデルの推論を高速化する手法（量子化、投機デコード、KV キャッシュ）について、日本語で詳しく説明してください。",
    "ja-code": "Pythonで、CSVファイルを読み込んで列ごとの統計を計算し、結果をMarkdownの表で出力するスクリプトを書いてください。"
               "コメントは日本語で書いてください。",
    # copy-heavy (lookup drafting): reproduce a file with a small edit
    "edit": "Here is a Python script:\n\n```python\n" + (pathlib.Path(__file__).parent / "ab_decode.py").read_text() +
            "```\n\nRename the function `parse` to `parse_config` everywhere and output the complete updated file "
            "in one code block, with no other text.",
}
EOS = {248046, 248044}


def chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  enable_thinking=False)
    return list(ids["input_ids"]) if not isinstance(ids, list) else ids


def run_spec(eng, ids, max_tokens, temperature, top_k, top_p, seed):
    t0 = time.perf_counter()
    out = [eng.spec_begin(ids, temperature, top_k, top_p, seed=seed, uniform=random.Random(seed).random())]
    t_first = time.perf_counter()
    steps = 0
    while len(out) < max_tokens and out[-1] not in EOS:
        new = eng.spec_step()
        steps += 1
        for t in new:
            out.append(t)
            if t in EOS or len(out) >= max_tokens:
                break
    t_last = time.perf_counter()
    return out, steps, t_first - t0, t_last - t_first


def run_plain(eng, ids, n):
    eng.reset()
    out = [eng.forward_sample(ids)]
    while len(out) < n and out[-1] not in EOS:
        out.append(eng.forward_sample([out[-1]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prompts", default="code,json,prose")
    ap.add_argument("--check-greedy", type=int, default=0, metavar="N")
    ap.add_argument("--label", default="spec")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--profile", action="store_true", help="per-op GPU time of the steps (Q27_PROFILE=1)")
    ap.add_argument("--ab", nargs="+", metavar="CFG", help="A/B of Q27_* switches (greedy, ms per step)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kv-fp8", action="store_true", help="FP8 KV cache")
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8", "nvfp4"], default=None,
                    help="KV cache dtype; overrides --kv-fp8 when supplied")
    ap.add_argument("--weights", default="nvidia", help="weight profile (Phase Q): nvidia | quasar | a model directory")
    args = ap.parse_args()
    if args.profile:
        import os
        os.environ["Q27_PROFILE"] = "1"
        args.no_save = True

    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, engine_target, snapshot

    if args.check_greedy:
        import os
        os.environ["Q27_SKINNY_EXACT"] = "1"
    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    t0 = time.perf_counter()
    eng = Engine(engine_target(args.weights), max_tokens=8192,
                 kv_dtype=args.kv_dtype or ("fp8" if args.kv_fp8 else "bf16"), draft=snapshot(*DFLASH2))
    print(f"load {time.perf_counter() - t0:.1f} s", flush=True)
    names = args.prompts.split(",")

    if args.ab:
        from ab_decode import parse
        envs = [parse(c) for c in args.ab]
        keys = sorted({k for e in envs for k in e})
        res = {c: [] for c in args.ab}
        ids = {n: chat_ids(tok, PROMPTS[n]) for n in names}
        for r in range(args.rounds):
            for cfg, env in zip(args.ab, envs):
                eng.configure(**{k: env.get(k) for k in keys})
                ms = []
                for n in names:
                    eng.reset()
                    out, steps, _, dt = run_spec(eng, ids[n], min(args.max_tokens, 400), 0.0, 1, 1.0, 0)
                    ms.append(1000 * dt / max(steps, 1))
                res[cfg].append(statistics.mean(ms))
                print(f"round {r} {cfg:50s} {res[cfg][-1]:.3f} ms/step", flush=True)
        ref = statistics.median(res[args.ab[0]])
        print("--- greedy, ms per step (median of the rounds)")
        for cfg in args.ab:
            m = statistics.median(res[cfg])
            print(f"{cfg:50s} {m:7.3f}  {m - ref:+.3f}  [{' '.join(f'{x:.3f}' for x in res[cfg])}]")
        eng.close()
        return

    if args.check_greedy:
        ok = True
        for name in names:
            ids = chat_ids(tok, PROMPTS[name])
            plain = run_plain(eng, ids, args.check_greedy)
            eng.reset()
            spec, steps, _, _ = run_spec(eng, ids, len(plain), 0.0, 1, 1.0, 0)
            spec = spec[:len(plain)]
            diff = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
            ok &= diff is None and len(spec) == len(plain)
            acc = (len(spec) - 1) / max(steps, 1)
            st = eng.spec_stats()
            print(f"{name}: {len(plain)} tokens, {'identical' if diff is None else f'differs at {diff}'}, "
                  f"acceptance {acc:.2f}, lookup steps {st['lookup_steps']}/{st['steps']}", flush=True)
            if diff is not None:
                print("  plain:", tok.decode(plain[max(0, diff - 10):diff + 10]))
                print("  spec: ", tok.decode(spec[max(0, diff - 10):diff + 10]))
        eng.close()
        sys.exit(0 if ok else 1)

    rows = []
    for name in names:
        ids = chat_ids(tok, PROMPTS[name])
        for rep in range(args.reps):
            eng.reset()
            if args.profile:
                eng.profile_dump()  # drop the prompt
            out, steps, ttft, dt = run_spec(eng, ids, args.max_tokens, args.temperature, args.top_k, args.top_p, rep)
            if args.profile:
                print(f"--- profile {name}#{rep}", file=sys.stderr, flush=True)
                eng.profile_dump()
            tps = (len(out) - 1) / dt if dt > 0 else 0.0
            acc = (len(out) - 1) / max(steps, 1)
            st = eng.spec_stats()
            row = {"prompt": name, "rep": rep, "tokens": len(out), "steps": steps, "accept_len": round(acc, 3),
                   "decode_tps": round(tps, 2), "ms_per_step": round(1000 * dt / max(steps, 1), 3),
                   "ttft": round(ttft, 3), "lookup_steps": st["lookup_steps"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    eng.close()
    summary = {}
    for name in names:
        r = [x for x in rows if x["prompt"] == name]
        summary[name] = {"decode_tps": round(statistics.mean(x["decode_tps"] for x in r), 2),
                         "accept_len": round(statistics.mean(x["accept_len"] for x in r), 3),
                         "ms_per_step": round(statistics.mean(x["ms_per_step"] for x in r), 3)}
    print(json.dumps(summary), flush=True)
    if not args.no_save:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        out = ROOT / f"bench/results/{stamp}-{args.label}.json"
        out.write_text(json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
