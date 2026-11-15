"""KV pool memory lent for video decoding (Engine.kv_lend / kv_unlend): the pages of whole blocks below the parking
areas move elsewhere and their memory goes back to the device while NVDEC decodes, then comes back. A small pool
with the prefix cache (KV_POOL_TOKENS 40,960, MAX_SEQS=1: no parking area) handing out its highest pages first (Q27_PAGES_HIGH), so
the lent range holds the running sequence's KV, plain greedy decoding:

  1. A sequence (9,000 tokens) runs 16 tokens; 300 MiB are lent (its pages move out), device memory in use drops by
     that much, a video is decoded meanwhile; it continues and its 48 tokens equal those it produces alone. The loan
     is returned and the memory is in use again.
  2. A loan larger than the pool can spare is refused (-1) and changes nothing: the sequence still matches.
  3. A checkpoint saved before a loan (its pages in the lent range) restores after it: the cached prefix runs on to
     the same tokens.
  4. Two loans at once (two videos decoding) lie one below the other, are returned in the other order, and the
     sequence running meanwhile still matches.

    docker compose --profile dev run --rm dev python scripts/lend_test.py
"""

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "frontend"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kv-dtype", default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--weights", default="gdnfp4", help="weight profile: nvidia | gdnfp4 | quasar | a model directory")
    args = ap.parse_args()
    os.environ.update({"MAX_SEQS": "1", "STATE_HOST_CACHE": "1", "KV_POOL_TOKENS": "40960", "Q27_PAGES_HIGH": "1"})

    from q27lib import Engine
    from qwen27b.weights import engine_target

    import videoproc

    eng = Engine(engine_target(args.weights), max_tokens=16384, kv_dtype=args.kv_dtype)
    prompts = sorted((json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl")), key=len)
    p = (prompts[-1] * 4)[:9000]
    fails = 0

    def check(name, ok):
        nonlocal fails
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)

    def more(out, n):
        while len(out) < n:
            out.append(eng.forward_sample([out[-1]], 0.0, 1, 1.0))
        return out

    def free():
        time.sleep(0.2)
        return eng.device_memory()["free"]

    eng.reset()
    ref = more([eng.forward_sample(p, 0.0, 1, 1.0)], 48)

    # 1. A loan while the sequence's pages are in the range
    eng.reset()
    out = more([eng.forward_sample(p, 0.0, 1, 1.0)], 16)
    f0 = free()
    loan = eng.kv_lend(300 << 20)
    lent = eng.kv_lent()
    f1 = free()
    print(f"loan {loan}: {lent >> 20} MiB lent, free device memory {f0 >> 20} -> {f1 >> 20} MiB", flush=True)
    check("a loan of 300 MiB is made", loan >= 0 and lent >= 300 << 20)
    check("the lent memory is free on the device", f1 - f0 >= lent - (64 << 20))
    raw = (ROOT / "bench/vision/colors.mp4").read_bytes()
    n, fps, w, h = videoproc.probe(raw)
    plan = videoproc.VideoProcessor(engine_target(args.weights)).plan(n, fps, w, h)
    need, sessions = videoproc.memory(raw, plan)
    frames = videoproc.frames(raw, plan, sessions)
    print(f"decoded {len(frames)} frames during the loan (estimate {need >> 20} MiB, {sessions} decoders)", flush=True)
    more(out, 48)
    check("the sequence continues as alone across the loan (its pages moved)", out == ref)
    eng.kv_unlend(loan)
    f2 = free()
    check("the returned memory is in use again", abs(f2 - f0) < 64 << 20 and eng.kv_lent() == 0)

    # 2. A loan the pool cannot spare
    eng.reset()
    out = more([eng.forward_sample(p, 0.0, 1, 1.0)], 16)
    check("a loan larger than the pool is refused", eng.kv_lend(64 << 30) == -1 and eng.kv_lent() == 0)
    more(out, 48)
    check("the sequence continues as alone after the refused loan", out == ref)

    # 3. A checkpoint across a loan
    eng.reset()
    cached = eng.prefix_begin(p, at=(8192,))
    out = more([eng.forward_sample(p[cached:], 0.0, 1, 1.0)], 4)
    eng.reset()
    loan = eng.kv_lend(300 << 20)
    check("a loan with a checkpoint in the range is made", loan >= 0)
    videoproc.frames(raw, plan, sessions)
    eng.kv_unlend(loan)
    cached = eng.prefix_begin(p, at=())
    out = more([eng.forward_sample(p[cached:], 0.0, 1, 1.0)], 48)
    check(f"the checkpoint restores after the loan ({cached} tokens cached) and runs on to the same tokens",
          cached == 8192 and out == ref)

    # 4. Two loans at once
    eng.reset()
    out = more([eng.forward_sample(p, 0.0, 1, 1.0)], 16)
    f0 = free()
    a, b = eng.kv_lend(150 << 20), eng.kv_lend(150 << 20)
    both = eng.kv_lent()
    f1 = free()
    check("two loans at once are made", a >= 0 and b >= 0 and a != b and both >= 300 << 20 and f1 - f0 >= both - (64 << 20))
    eng.kv_unlend(a)
    more(out, 32)
    c = eng.kv_lend(150 << 20)
    check("a loan after the upper one is returned takes its place", c >= 0 and eng.kv_lent() == both)
    eng.kv_unlend(b)
    eng.kv_unlend(c)
    more(out, 48)
    check("the sequence continues as alone across them", out == ref and eng.kv_lent() == 0 and abs(free() - f0) < 64 << 20)

    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
