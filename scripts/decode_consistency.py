"""Greedy generations of the engine under several runtime settings must be identical token for token.

Runs a short English and a Japanese chat prompt (thinking off, 200 greedy tokens) once per setting,
each in a fresh process: the CUDA graph + PDL default, eager (Q27_NO_GRAPH=1), graph without PDL
(Q27_NO_PDL=1), and eager without PDL. A difference means a race or a stale-state bug in the decode
path, not numerics (the settings run the same kernels in the same order).

    docker compose --profile dev run --rm dev python scripts/decode_consistency.py
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = {"en": "Write a short story about a lighthouse keeper.", "ja": "日本の四季について説明してください。"}
SETTINGS = {"graph+pdl": {}, "eager": {"Q27_NO_GRAPH": "1"}, "graph": {"Q27_NO_PDL": "1"},
            "eager-nopdl": {"Q27_NO_GRAPH": "1", "Q27_NO_PDL": "1"}}


def generate(tokens):
    sys.path.insert(0, str(ROOT / "engine/python"))
    sys.path.insert(0, str(ROOT / "reference"))
    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import TARGET, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(snapshot(*TARGET), max_tokens=4096)
    out = {}
    for name, text in PROMPTS.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
        eng.reset()
        seq = [eng.forward_sample(ids)]
        while len(seq) < tokens:
            seq.append(eng.forward_sample([seq[-1]]))
        out[name] = seq
    print(json.dumps(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return generate(int(sys.argv[2]))
    runs = {}
    for name, env in SETTINGS.items():
        r = subprocess.run([sys.executable, __file__, "--child", "200"], env={**os.environ, **env},
                           capture_output=True, text=True, check=True)
        runs[name] = json.loads(r.stdout.strip().splitlines()[-1])
    base = runs["graph+pdl"]
    ok = True
    for name, seqs in runs.items():
        for p in PROMPTS:
            diff = next((i for i, (a, b) in enumerate(zip(seqs[p], base[p])) if a != b), None)
            ok &= diff is None
            print(f"{name:12s} {p}: {'identical' if diff is None else f'differs at token {diff}'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
