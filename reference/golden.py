"""Golden next-token distributions: make prompts, record them from a model, compare two recordings.

A recording is JSONL, one line per prompt: {"name", "ids", "topk": [[[token, logprob], ...] per position]},
where topk[i] is the distribution of the token after ids[:i+1] (so the last entry predicts the first
generated token). Sources: this reference model, a running SGLang / vLLM server (same checkpoint), and
later the custom engine.

    python -m golden prompts --out bench/golden/prompts.jsonl
    python -m golden ref     --prompts bench/golden/prompts.jsonl --out bench/golden/ref-cpu.jsonl.gz [--device cuda]
    python -m golden server  --prompts ... --out bench/golden/sglang.jsonl.gz --api sglang --base-url http://127.0.0.1:8000
    python -m golden engine  --prompts ... --out bench/golden/engine.jsonl.gz   # the custom engine (libq27.so)
    python -m golden engine  ... --decode-tail 128 ; python -m golden compare --tail 128 ...  # decode path
    python -m golden compare bench/golden/sglang.jsonl.gz bench/golden/ref-cpu.jsonl.gz
"""

import argparse
import gzip
import json
import math
import os
import pathlib
import sys

TOPK = 20
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Fixed public texts (bench/golden/sources/README.md): Japanese prose, English prose, Python and C++ code.
# The prompts are prefixes of 128 / 512 / 4,096 / 16,384 tokens (as far as each text is long).
SOURCES = {name: ROOT / f"bench/golden/sources/{name}.txt" for name in ("ja-prose", "en-prose", "code-python", "code-cpp")}


def cmd_prompts(args):
    from transformers import AutoTokenizer

    from qwen27b.weights import TARGET, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, path in SOURCES.items():
            ids = tok(path.read_text()).input_ids
            for n in args.lengths:
                if len(ids) >= n:
                    f.write(json.dumps({"name": f"{name}-{n}", "ids": ids[:n]}) + "\n")
    print(f"wrote {out}")


def open_text(path, mode="r"):
    """Recordings are stored gzipped (`.jsonl.gz`, ~20 MB each uncompressed); plain `.jsonl` also works."""
    return gzip.open(path, mode + "t") if str(path).endswith(".gz") else open(path, mode)


def read_jsonl(path):
    with open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_ref(args):
    import torch

    from qwen27b.model import Qwen38Reference, SeqState
    from qwen27b.weights import BF16, QUASAR, TARGET, Checkpoint, snapshot

    if args.model == "quasar":
        ckpt = Checkpoint(snapshot(*QUASAR), lm_head_from=Checkpoint(snapshot(*TARGET)))
    else:
        ckpt = Checkpoint(snapshot(*(BF16 if args.model == "bf16" else TARGET)))
    ref = Qwen38Reference(ckpt, device=args.device, act_quant=not args.no_act_quant,
                          kv_dtype=args.kv_dtype)
    head = ref.lm_head()
    with open_text(args.out, "w") as f:
        for p in read_jsonl(args.prompts):
            if args.max_len and len(p["ids"]) > args.max_len:
                continue
            topk = []
            with torch.no_grad():
                h = ref.forward(torch.tensor(p["ids"]), SeqState(), all_logits=True, return_hidden=True)
                for s in range(0, h.shape[0], 1024):  # all-position logits in slices
                    lp = torch.log_softmax(head(h[s:s + 1024]).double(), -1)
                    v, i = lp.topk(TOPK, dim=-1)
                    topk += [[[int(t), float(l)] for t, l in zip(ii, vv)] for ii, vv in zip(i.tolist(), v.tolist())]
            f.write(json.dumps({"name": p["name"], "ids": p["ids"], "topk": topk}) + "\n")
            f.flush()
            print(p["name"], "next:", topk[-1][0], file=sys.stderr)


def cmd_engine(args):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine/python"))
    from q27lib import Engine

    from qwen27b.weights import engine_target

    prompts = [p for p in read_jsonl(args.prompts) if not args.max_len or len(p["ids"]) <= args.max_len]
    eng = Engine(engine_target(args.model), max_tokens=max(len(p["ids"]) for p in prompts),
                 prefill_chunk=args.prefill_chunk, kv_dtype=args.kv_dtype)
    with open_text(args.out, "w") as f:
        for p in prompts:
            eng.reset()
            ids, tail = p["ids"], min(args.decode_tail, len(p["ids"]) - 1)
            rows = eng.forward_topk(ids[:len(ids) - tail], TOPK)
            for tok in ids[len(ids) - tail:]:  # the decode path: one token per forward
                rows += eng.forward_topk([tok], TOPK)
            topk = [[[int(t), float(l)] for t, l in row] for row in rows]
            f.write(json.dumps({"name": p["name"], "ids": p["ids"], "topk": topk}) + "\n")
            f.flush()
            print(p["name"], "next:", topk[-1][0], file=sys.stderr)
    eng.close()


def cmd_server(args):
    import httpx

    client = httpx.Client(base_url=args.base_url, timeout=3600)
    with open_text(args.out, "w") as f:
        for p in read_jsonl(args.prompts):
            ids = p["ids"]
            if args.api == "sglang":
                r = client.post("/generate", json={
                    "input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": TOPK,
                }).json()
                meta = r["meta_info"]
                rows = meta["input_top_logprobs"][1:] + meta["output_top_logprobs"][:1]
                topk = [[[int(e[1]), float(e[0])] for e in row] for row in rows]
            else:  # vLLM-style /v1/completions with prompt_logprobs
                r = client.post("/v1/completions", json={
                    "model": args.model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                    "prompt_logprobs": TOPK, "logprobs": TOPK,
                }).json()
                ch = r["choices"][0]
                rows = ch["prompt_logprobs"][1:]
                topk = [sorted(([int(t), float(d["logprob"])] for t, d in row.items()), key=lambda e: -e[1])[:TOPK]
                        for row in rows]
                last = ch["logprobs"]["top_logprobs"][0]  # token text -> logprob; ids not available here
                topk.append([[-1, float(l)] for l in sorted(last.values(), reverse=True)])
            f.write(json.dumps({"name": p["name"], "ids": ids, "topk": topk}) + "\n")
            f.flush()
            print(p["name"], file=sys.stderr)


def compare_rows(a, b):
    """(top1 agree, KL(a||b) over a's top-k with b's missing mass floored, |logp_a - logp_b| of a's top-1)."""
    da, db = dict(map(tuple, a)), dict(map(tuple, b))
    floor = min(db.values()) - math.log(2)  # anything outside b's top-k is at most this likely
    kl = sum(math.exp(la) * (la - db.get(t, floor)) for t, la in da.items())
    t1 = a[0][0]
    return a[0][0] == b[0][0], kl, abs(a[0][1] - db.get(t1, floor))


def cmd_compare(args):
    A = {r["name"]: r for r in read_jsonl(args.a)}
    B = {r["name"]: r for r in read_jsonl(args.b)}
    print(f"{'prompt':<22}{'positions':>10}{'top1 agree':>12}{'mean KL':>10}{'max KL':>10}{'mean |dlogp|':>14}"
          f"{'KL>0.5':>9}")
    every = []

    def line(name, rows):
        kls = [r[1] for r in rows]
        print(f"{name:<22}{len(rows):>10}{sum(r[0] for r in rows) / len(rows):>12.4f}{sum(kls) / len(kls):>10.4f}"
              f"{max(kls):>10.3f}{sum(r[2] for r in rows) / len(rows):>14.4f}"
              f"{sum(k > 0.5 for k in kls) / len(kls):>9.2%}")

    for name in sorted(A.keys() & B.keys()):
        if A[name]["ids"] != B[name]["ids"]:
            print(f"{name}: different token ids, skipped")
            continue
        pairs = list(zip(A[name]["topk"], B[name]["topk"]))[-args.tail if args.tail else 0:]
        rows = [compare_rows(x, y) for x, y in pairs if x[0][0] >= 0 and y[0][0] >= 0]
        every += rows
        line(name, rows)
    if every:
        line("all", every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", default="bench/golden/prompts.jsonl")
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=[128, 512, 4096, 16384])
    p = sub.add_parser("ref")
    p.add_argument("--prompts", default="bench/golden/prompts.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-len", type=int, default=0, help="skip longer prompts (CPU runs)")
    p.add_argument("--kv-dtype", default="bf16", choices=["fp32", "bf16", "fp8", "nvfp4"])
    p.add_argument("--model", default="nvidia", choices=["nvidia", "quasar", "bf16"],
                   help="checkpoint (quasar: allfp4 candidate, bf16: the original)")
    p.add_argument("--no-act-quant", action="store_true")
    p = sub.add_parser("engine")
    p.add_argument("--prompts", default="bench/golden/prompts.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=0, help="skip longer prompts")
    p.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    p.add_argument("--model", default="nvidia",
                   help="nvidia | quasar | a model directory (allfp4 candidates, scripts/quantize_proj.py)")
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--decode-tail", type=int, default=0, help="run the last N tokens one at a time (decode path)")
    p = sub.add_parser("server")
    p.add_argument("--prompts", default="bench/golden/prompts.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--api", choices=["sglang", "openai"], default="sglang")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3.8-27b")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tail", type=int, default=0, help="compare only the last N positions of each prompt")
    args = ap.parse_args()
    {"prompts": cmd_prompts, "ref": cmd_ref, "engine": cmd_engine, "server": cmd_server, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
