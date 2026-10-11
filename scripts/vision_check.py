"""Image prompts through the reference model and the engine (P4-12): golden top-k and greedy continuations.

Each case is a chat prompt (thinking off) with one image from bench/vision/. `ref` runs the reference model
(vision.py + model.py, GPU) and writes a golden recording ({"name", "ids", "topk"} per prompt, as
reference/golden.py, plus the image spans and a greedy continuation); `engine` writes the same from the
custom engine. `python -m golden compare` (PYTHONPATH=reference) compares two recordings. The reference
also saves its image embeddings (cache/vision/<case>.pt) for the engine's vision tower check.

    docker compose --profile dev run --rm dev python scripts/vision_check.py ref --out bench/golden/vision-ref.jsonl.gz
    docker compose --profile dev run --rm dev python scripts/vision_check.py engine --out bench/golden/vision-engine.jsonl.gz
    docker compose --profile dev run --rm dev python scripts/vision_check.py spec --tokens 128
    docker compose --profile dev run --rm dev python scripts/vision_check.py bench --out bench/results/<stamp>-vision.json

`spec` checks that greedy speculative decoding (DFlash2) after an image prompt gives the plain greedy tokens
(the verify rows rotate at index + the M-RoPE delta). `bench` times the engine's vision tower per image size (random pixels; median of 5 after a warm-up).
"""

import argparse
import gzip
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

CASES = {
    "astronaut": ("astronaut.jpg", "Describe this image in two sentences."),
    "chart": ("chart.png", "Which month has the highest sales, and what is its value? Answer briefly."),
    "chart-ja": ("chart.png", "このグラフの2月の値はいくつですか。"),
}
TOPK = 20


def build(name, tok, proc):
    """(ids with image pads, spans, pixel_values, grid) of a case."""
    from PIL import Image

    from qwen27b.vision import expand_image_pads

    file, text = CASES[name]
    img = Image.open(ROOT / "bench/vision" / file).convert("RGB")
    batch = proc(images=[img], return_tensors="pt")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    ids, spans = expand_image_pads(ids, batch["image_grid_thw"].tolist(), pad)
    return ids, spans, batch["pixel_values"], batch["image_grid_thw"].tolist()


def cmd_ref(args, tok, proc):
    import torch

    from qwen27b.model import Qwen38Reference, SeqState
    from qwen27b.vision import VisionReference, mrope_positions
    from qwen27b.weights import TARGET, Checkpoint, snapshot

    ckpt = Checkpoint(snapshot(*TARGET))
    vis = VisionReference(ckpt, args.device)
    ref = Qwen38Reference(ckpt, args.device, kv_dtype=args.kv_dtype)
    (ROOT / "cache/vision").mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        for name in args.cases:
            t0 = time.time()
            ids, spans, pv, grid = build(name, tok, proc)
            with torch.no_grad():
                if args.engine_embeds:  # the engine's vision output (isolates the language model path)
                    emb = torch.load(ROOT / f"cache/vision/{name}-engine.pt")["embeds"]
                else:
                    emb = vis.encode(pv, grid)
                    torch.save({"pixel_values": pv, "grid": grid, "embeds": emb.cpu()}, ROOT / f"cache/vision/{name}.pt")
                rows = torch.cat([torch.arange(s.start, s.start + s.length) for s in spans])
                pos3, delta = mrope_positions(len(ids), spans)
                st = SeqState()
                hid = ref.forward(torch.tensor(ids), st, all_logits=True, return_hidden=True, mm_rows=rows,
                                  mm_embeds=emb, pos3=pos3)
                head = ref.lm_head()
                topk = []
                for r in range(0, hid.shape[0], 256):
                    lp = head(hid[r:r + 256]).float().log_softmax(-1)
                    v, i = lp.topk(TOPK, -1)
                    topk += [[[int(a), round(float(b), 4)] for a, b in zip(ii, vv)] for ii, vv in zip(i.tolist(), v.tolist())]
                gen = [topk[-1][0][0]]
                while len(gen) < args.tokens and gen[-1] not in (248044, 248046):
                    p = len(ids) + len(gen) - 1 + delta
                    out = ref.forward(torch.tensor(gen[-1:]), st, pos3=torch.full((3, 1), p))
                    gen.append(int(out[0].argmax()))
            rec = {"name": name, "ids": ids, "topk": topk, "spans": [[s.start, s.gh, s.gw] for s in spans],
                   "grid": grid, "gen": gen}
            f.write(json.dumps(rec) + "\n")
            print(f"{name}: {len(ids)} tokens ({sum(s.length for s in spans)} image), {time.time() - t0:.0f} s: "
                  f"{tok.decode(gen)!r}", flush=True)


def cmd_engine(args, tok, proc):
    import os

    import numpy as np
    import torch

    from q27lib import Engine
    from qwen27b.vision import image_token_ids
    from qwen27b.weights import TARGET, snapshot

    os.environ.setdefault("VISION", "bf16")
    eng = Engine(snapshot(*TARGET), max_tokens=8192, kv_dtype=args.kv_dtype, prefill_chunk=args.prefill_chunk)
    with gzip.open(args.out, "wt") as f:
        for name in args.cases:
            ids, spans, pv, grid = build(name, tok, proc)
            # Image rows as the frontend sends them: ids derived from the pixels.
            eids = list(ids)
            for sp in spans:
                eids[sp.start:sp.start + sp.length] = image_token_ids(pv.numpy().tobytes(), sp.length)
            eng.reset()
            eng.mm_set([(s.start, s.gh, s.gw) for s in spans])
            t0 = time.time()
            eng.mm_encode(0, pv.numpy())
            t_vis = time.time() - t0
            ref_path = ROOT / f"cache/vision/{name}.pt"
            if ref_path.exists():
                ref = torch.load(ref_path)["embeds"].float()
                ours = torch.from_numpy(eng.mm_embeds(0, spans[0].length).astype(np.int32) << 16).view(torch.float32)
                torch.save({"embeds": ours}, ROOT / f"cache/vision/{name}-engine.pt")
                d = ours - ref
                cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
                print(f"{name}: embeddings vs reference: max |d| / max |ref| {d.abs().max() / ref.abs().max():.2e}, "
                      f"rms d / rms ref {d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt():.2e}, "
                      f"min cosine {cos.min():.5f}", flush=True)
            t0 = time.time()
            topk = [[[int(a), round(float(b), 4)] for a, b in row] for row in eng.forward_topk(eids, TOPK)]
            t_pf = time.time() - t0
            gen = [topk[-1][0][0]]
            while len(gen) < args.tokens and gen[-1] not in (248044, 248046):
                gen.append(eng.forward_sample(gen[-1:]))
            rec = {"name": name, "ids": ids, "topk": topk, "spans": [[s.start, s.gh, s.gw] for s in spans],
                   "grid": grid, "gen": gen}
            f.write(json.dumps(rec) + "\n")
            print(f"{name}: {len(ids)} tokens, vision {t_vis * 1000:.0f} ms (first call includes setup), prefill "
                  f"{t_pf * 1000:.0f} ms: {tok.decode(gen)!r}", flush=True)
    eng.close()


def cmd_spec(args, tok, proc):
    import os

    from q27lib import Engine
    from qwen27b.vision import image_token_ids
    from qwen27b.weights import DFLASH2, TARGET, snapshot

    os.environ.setdefault("VISION", "bf16")
    os.environ["Q27_SKINNY_EXACT"] = "1"  # (verify-sized GEMMs with the decode summation order, as bench_spec.py)
    eng = Engine(snapshot(*TARGET), max_tokens=8192, kv_dtype=args.kv_dtype, draft=snapshot(*DFLASH2))
    ok = True
    for name in args.cases:
        ids, spans, pv, grid = build(name, tok, proc)
        for sp in spans:
            ids[sp.start:sp.start + sp.length] = image_token_ids(pv.numpy().tobytes(), sp.length)
        runs = []
        for spec in (False, True):
            eng.reset()
            eng.mm_set([(s.start, s.gh, s.gw) for s in spans])
            eng.mm_encode(0, pv.numpy())
            if spec:
                out = [eng.spec_begin(ids, 0.0)]
                while len(out) < args.tokens:
                    out += eng.spec_step()
            else:
                out = [eng.forward_sample(ids)]
                while len(out) < args.tokens:
                    out.append(eng.forward_sample(out[-1:]))
            out = out[:args.tokens]
            stop = next((i for i, t in enumerate(out) if t in (248044, 248046)), None)
            runs.append(out[:stop + 1] if stop is not None else out)  # (speculative steps end at EOS)
        same = runs[0] == runs[1]
        ok &= same
        at = next((i for i, (a, b) in enumerate(zip(*runs)) if a != b), min(map(len, runs)))
        print(f"{name}: {len(runs[0])} / {len(runs[1])} tokens, speculative "
              f"{'identical' if same else f'DIFFERENT from token {at}'} ({tok.decode(runs[0])[:80]!r})", flush=True)
        if not same:
            print(f"  plain: {tok.decode(runs[0][at:at + 12])!r}\n  spec:  {tok.decode(runs[1][at:at + 12])!r}")
    sys.exit(0 if ok else 1)


def cmd_bench(args, tok, proc):
    import os
    import statistics

    import numpy as np

    from q27lib import Engine
    from qwen27b.weights import TARGET, snapshot

    os.environ.setdefault("VISION", "bf16")
    eng = Engine(snapshot(*TARGET), max_tokens=8192, kv_dtype=args.kv_dtype)
    rng = np.random.default_rng(0)
    rows = []
    for side in (256, 512, 1024, 1536, 2048):  # square images, side x side pixels
        g = side // 16
        pv = rng.standard_normal((g * g, 1536), dtype=np.float32)
        eng.reset()
        eng.mm_set([(0, g // 2, g // 2)])
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            eng.mm_encode(0, pv)
            times.append(time.perf_counter() - t0)
        ms = statistics.median(times[1:]) * 1000
        rows.append({"pixels": side * side, "patches": g * g, "tokens": g * g // 4, "ms": round(ms, 2)})
        print(f"{side}x{side}: {g * g} patches, {g * g // 4} tokens, {ms:.1f} ms", flush=True)
    pathlib.Path(args.out).write_text(json.dumps({"vision": os.environ["VISION"], "encode": rows}, indent=1) + "\n")


def main():
    from transformers import AutoImageProcessor, AutoTokenizer

    from qwen27b.weights import TARGET, snapshot

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "engine", "spec", "bench"])
    ap.add_argument("--out", help="ref / engine / bench: output file")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--tokens", type=int, default=32, help="greedy tokens to generate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="engine: prefill chunk (chunks cut images)")
    ap.add_argument("--engine-embeds", action="store_true",
                    help="ref: take the image embeddings the engine mode saved (cache/vision/<case>-engine.pt)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    proc = AutoImageProcessor.from_pretrained(snapshot(*TARGET))
    {"ref": cmd_ref, "engine": cmd_engine, "spec": cmd_spec, "bench": cmd_bench}[args.mode](args, tok, proc)


if __name__ == "__main__":
    main()
