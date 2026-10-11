#!/usr/bin/env python3
"""Long-context retrieval check (stdlib only): hide N key/value facts at spread-out depths in a
long filler document, then ask for all of them (greedy, thinking off). Scores how many come back.

Used to compare settings that can change long-sequence accuracy (e.g. --mamba-ssm-dtype bfloat16
vs float32 for the Gated DeltaNet state) at context sizes near the serving limit.

    python3 scripts/needle_test.py --tokens 32000,100000,180000
"""

import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = (
    "self value index result config request buffer token cache layer state return "
    "import class async await yield tensor batch schedule prefix block kernel worker "
    "engine server client stream parse render error retry path file module handler "
    "session context manager queue event loop thread offset length window"
).split()
NAMES = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor",
         "indigo", "juniper", "krypton", "lagoon", "marble", "nimbus", "onyx", "pumice"]


def post(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build(rng, approx_tokens, needles):
    """Filler lines (~3 tokens per word, same generator as bench_agentic.py) with needles spread over depth."""
    lines, n = [], 0
    while n < approx_tokens:
        w = [rng.choice(WORDS) for _ in range(rng.randint(5, 9))]
        lines.append(f"{'    ' * rng.randint(0, 2)}{w[0]}_{w[1]} = {w[2]}.{w[3]}({', '.join(w[4:])})")
        n += 3 * len(w)
    for i, (name, code) in enumerate(needles):
        pos = int(len(lines) * (i + 0.5) / len(needles))
        lines.insert(pos, f"# NOTE: the access code for project {name} is {code}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokens", default="32000,100000,180000", help="approximate prompt sizes in tokens")
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per request (a 262K prefill can take ~40 min)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    base = args.base_url.rstrip("/")
    model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]

    total = found = 0
    for size in (int(x) for x in args.tokens.split(",")):
        rng = random.Random(f"{args.seed}-{size}")
        needles = [(name, f"{rng.randint(0, 999999):06d}") for name in rng.sample(NAMES, args.needles)]
        # The filler generator undercounts real tokens by ~1.65x; scale so the prompt lands near `size`.
        doc = build(rng, int(size * 1.65), needles)
        question = ("List the access code of every project mentioned in the NOTE comments above, "
                    "one per line as `project: code`, in the order they appear. Output nothing else.")
        t0 = time.perf_counter()
        r = post(base, {"model": model, "temperature": 0, "max_tokens": 400,
                        "messages": [{"role": "user", "content": f"{doc}\n\n{question}"}],
                        "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
        dt = time.perf_counter() - t0
        text = r["choices"][0]["message"]["content"] or ""
        ok = sum(1 for name, code in needles if f"{name}: {code}" in text or
                 any(name in line and code in line for line in text.splitlines()))
        total += len(needles)
        found += ok
        print(f"{r['usage']['prompt_tokens']:>7} tokens: {ok}/{len(needles)} found ({dt:.1f}s)")
        if ok < len(needles):
            print("   expected: " + ", ".join(f"{n}={c}" for n, c in needles))
            print("   got:      " + " | ".join(text.strip().splitlines()))
    print(f"total: {found}/{total}")


if __name__ == "__main__":
    main()
