"""Videos through the engine's vision tower (frame pairs encoded together, q27_mm_encode_frames) against the
reference (vision.py, fp32 on the GPU) and against one pair per call; times the encode per video length. `ref` and
`engine` record video prompts end to end as golden recordings (as vision_check.py: top-k at every position and a
greedy continuation; `python -m golden compare` compares them): the reference reads the frames with PyAV and
transformers' processor, the engine as the server does (its demuxer, NVDEC, the GPU resize, server.Images).

The video is a pan over bench/vision/astronaut.jpg (or --video FILE; PyAV encodes the pan, the dev image has it),
planned and decoded as the server does (frontend/videoproc.py: the engine's demuxer, NVDEC, the GPU resize).

    docker compose --profile dev run --rm dev python scripts/video_check.py embeds
    docker compose --profile dev run --rm dev python scripts/video_check.py bench --out bench/results/<stamp>-video.json
    docker compose --profile dev run --rm dev python scripts/video_check.py ref --cases a.mp4,b.mkv --out bench/golden/video-ref.jsonl.gz
    docker compose --profile dev run --rm dev python scripts/video_check.py engine --cases a.mp4,b.mkv --out bench/golden/video-engine.jsonl.gz
"""

import argparse
import io
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "frontend"))


def pan(seconds, fps=30, size=(1280, 720)):
    """An H.264 video panning over the astronaut photo."""
    import av
    from PIL import Image

    img = Image.open(ROOT / "bench/vision/astronaut.jpg").convert("RGB").resize((size[0] * 2, size[1] * 2))
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        s.options = {"crf": "20", "preset": "ultrafast"}
        n = int(seconds * fps)
        for i in range(n):
            x, y = int(size[0] * i / max(n - 1, 1)), int(size[1] * i / max(n - 1, 1))
            for p in s.encode(av.VideoFrame.from_image(img.crop((x, y, x + size[0], y + size[1])))):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def load(args, seconds):
    import videoproc

    raw = pathlib.Path(args.video).read_bytes() if args.video else pan(seconds)
    proc = videoproc.VideoProcessor(args.model)
    plan = proc.plan(*videoproc.probe(raw), frame_pixels=int(os.environ.get("IMAGE_MAX_PIXELS", 4194304)))
    t0 = time.time()
    frames = videoproc.frames(raw, plan, args.sessions)
    return raw, proc, plan, frames, time.time() - t0


def spans_of(plan):
    t, gh, gw = plan.grid
    n = gh * gw // 4
    return [(g * (n + 8), gh // 2, gw // 2) for g in range(t)]  # (any ascending, non-overlapping starts)


def embeds(eng, plan, rows):
    import numpy as np
    import torch

    return torch.cat([torch.from_numpy(eng.mm_embeds(g, rows).astype(np.int32) << 16).view(torch.float32)
                      for g in range(plan.grid[0])])


def cmd_embeds(args):
    import torch

    from q27lib import Engine
    from qwen27b.vision import VisionReference
    from qwen27b.weights import Checkpoint, snapshot, TARGET

    raw, proc, plan, frames, t_dec = load(args, args.seconds)
    t, gh, gw = plan.grid
    rows = gh * gw // 4
    print(f"video {len(raw) / 1e6:.1f} MB: {len(plan.indices)} frames, {t} pairs of {gh}x{gw} patches "
          f"({t * rows} tokens), decoded in {t_dec:.2f} s", flush=True)
    eng = Engine(args.model, max_tokens=4096, kv_dtype="fp8")
    eng.reset()
    eng.mm_set(spans_of(plan))
    t0 = time.time()
    eng.mm_encode_frames(0, frames, proc.norm)
    print(f"engine ({os.environ['VISION']}): {time.time() - t0:.3f} s for {t} pairs together", flush=True)
    together = embeds(eng, plan, rows)
    for g in range(t):
        eng.mm_encode_frames(g, frames[2 * g:2 * g + 2], proc.norm)
    alone = embeds(eng, plan, rows)
    print(f"together vs one pair per call: max |d| {(together - alone).abs().max():.3e}", flush=True)
    eng.close()

    rw, rh = plan.size
    x = torch.stack([torch.frombuffer(bytearray(b), dtype=torch.uint8).view(rh, rw, 3) for b in frames]).float()
    x = (x.permute(0, 3, 1, 2) - torch.tensor(proc.norm[:3])[:, None, None]) / torch.tensor(proc.norm[3:])[:, None, None]
    p, m = proc.patch, proc.merge
    pv = x.view(t, 2, 3, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(t * gh * gw, -1)
    vis = VisionReference(Checkpoint(snapshot(*TARGET)), "cuda")
    with torch.no_grad():
        ref = torch.cat([vis.encode(pv[g * gh * gw:(g + 1) * gh * gw], [(1, gh, gw)]).float().cpu()
                         for g in range(t)])
    d = together - ref
    cos = torch.nn.functional.cosine_similarity(together, ref, dim=-1)
    print(f"engine vs reference: rms d / rms ref {d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt():.2e}, "
          f"min cosine {cos.min():.5f}", flush=True)
    # (a frame pair is not an image of its first frame twice: the second frame matters)
    if t > 1:
        print(f"pair 0 vs pair 1 (reference): rms d / rms {(ref[:rows] - ref[rows:2 * rows]).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt():.2e}")


TOPK = 20
QUESTIONS = ["Describe what happens in this video in three sentences.", "この動画で何が起きているか、2文で説明してください。"]


def prompt_of(q):
    return (f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{q}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def cases(args):
    """(name, video path, question) per case: each video with each question."""
    return [(f"{pathlib.Path(v).stem}-{k}", v, q) for v in args.cases.split(",") for k, q in enumerate(QUESTIONS)]


def chunked_head(ckpt, device="cuda", rows=8192):
    """The reference's lm_head (Checkpoint.linear, fp32, activation quantization on) dequantized a block of vocab rows
    at a time: the same numbers without the 20 GB of temporaries of a whole dequantization."""
    import torch

    from qwen27b.quant import QLinear, dequant_nvfp4

    c = ckpt
    if c.lm_head_from is not None and c.get("lm_head.weight").dtype == torch.bfloat16:
        c = c.lm_head_from
    w = c.get("lm_head.weight")
    if w.dtype != torch.uint8:  # (fp8 / bf16 heads are small enough)
        return c.linear("lm_head", device)
    scale, s2 = c.get("lm_head.weight_scale"), c.get("lm_head.weight_scale_2", device)
    out = torch.empty(w.shape[0], 2 * w.shape[1], dtype=torch.float32, device=device)
    for r in range(0, w.shape[0], rows):
        out[r:r + rows] = dequant_nvfp4(w[r:r + rows].to(device), scale[r:r + rows].to(device), s2).float()
    ins = c.get("lm_head.input_scale", device) if "lm_head.input_scale" in c else None
    return QLinear("nvfp4", out, ins, True)


def cmd_ref(args):
    """Phase 1: the reference vision tower on every case (embeddings kept on the CPU), then freed; phase 2: the
    reference language model (the two do not fit the GPU together with long prompts)."""
    import gc
    import gzip

    import av
    import numpy as np
    import torch
    import transformers
    from transformers.video_utils import VideoMetadata

    import videoproc
    from qwen27b.image_tokens import ImageSpan
    from qwen27b.model import Qwen38Reference, SeqState
    from qwen27b.vision import VisionReference, mrope_positions
    from qwen27b.weights import Checkpoint, snapshot, TARGET

    proc = transformers.AutoProcessor.from_pretrained(args.model)
    mine = videoproc.VideoProcessor(args.model)
    pad = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    ckpt = Checkpoint(snapshot(*TARGET))
    vis = VisionReference(ckpt, "cuda")
    prepared = []
    for name, path, q in cases(args):
        t0 = time.time()
        raw = pathlib.Path(path).read_bytes()
        total, rate, w, h = videoproc.probe(raw)
        # (only the frames transformers samples: the plan's, without the last one repeated to an even count)
        n = min(max(int(total / rate * mine.fps), mine.min_frames), mine.max_frames, total)
        idx = videoproc.linspace_round(total - 1, n)
        want, frames = set(idx), {}
        with av.open(path) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    frames[i] = fr.to_ndarray(format="rgb24")
                if i >= idx[-1]:
                    break
        md = VideoMetadata(total_num_frames=total, fps=rate, duration=total / rate, width=w, height=h,
                           frames_indices=idx)
        b = proc(text=[prompt_of(q)], videos=[np.stack([frames[i] for i in idx])], video_metadata=[md],
                 do_sample_frames=False, cap_pixels_per_frame=True, return_tensors="pt")
        ids = b["input_ids"][0].tolist()
        t, gh, gw = b["video_grid_thw"][0].tolist()
        k = gh * gw // 4
        starts = [i for i in range(len(ids)) if ids[i] == pad and (i == 0 or ids[i - 1] != pad)]
        spans = [ImageSpan(s0, gh // 2, gw // 2) for s0 in starts]
        assert len(spans) == t and all(ids[s0:s0 + k] == [pad] * k for s0 in starts)
        pv = b["pixel_values_videos"]
        with torch.no_grad():
            if args.engine_embeds:  # (the engine's vision output: isolates the language model path)
                emb = torch.load(ROOT / f"cache/video/{name}-engine.pt")["embeds"]  # (bf16 values in fp32, as vision_check)
            else:
                emb = torch.cat([vis.encode(pv[g * gh * gw:(g + 1) * gh * gw], [(1, gh, gw)]).cpu() for g in range(t)])
        prepared.append((name, ids, spans, (t, gh, gw), emb))
        print(f"{name}: {len(ids)} tokens ({t} pairs of {k}), vision {time.time() - t0:.0f} s", flush=True)
        del frames, b, pv
    del vis
    gc.collect()
    torch.cuda.empty_cache()

    ref = Qwen38Reference(ckpt, "cuda", kv_dtype=args.kv_dtype)
    # (the head once, while the GPU is emptiest: its dequantization takes a temporary of over 10 GB)
    with torch.no_grad():
        head = chunked_head(ckpt)
    torch.cuda.empty_cache()
    with gzip.open(args.out, "wt") as f:
        for name, ids, spans, grid, emb in prepared:
            t0 = time.time()
            k = grid[1] * grid[2] // 4
            with torch.no_grad():
                rows = torch.cat([torch.arange(s.start, s.start + k) for s in spans])
                pos3, delta = mrope_positions(len(ids), spans)
                st = SeqState()
                hid = ref.forward(torch.tensor(ids), st, all_logits=True, return_hidden=True, mm_rows=rows,
                                  mm_embeds=emb.cuda(), pos3=pos3)
                topk = []
                for r in range(0, hid.shape[0], 256):
                    lp = head(hid[r:r + 256]).float().log_softmax(-1)
                    v, i = lp.topk(TOPK, -1)
                    topk += [[[int(a), round(float(c), 4)] for a, c in zip(ii, vv)] for ii, vv in zip(i.tolist(), v.tolist())]
                gen = [topk[-1][0][0]]
                while len(gen) < args.tokens and gen[-1] not in (248044, 248046):
                    p = len(ids) + len(gen) - 1 + delta
                    h1 = ref.forward(torch.tensor(gen[-1:]), st, pos3=torch.full((3, 1), p), return_hidden=True)
                    gen.append(int(head(h1[-1:]).float()[0].argmax()))
                del hid, st
            torch.cuda.empty_cache()
            f.write(json.dumps({"name": name, "ids": ids, "topk": topk, "grid": list(grid), "gen": gen}) + "\n")
            f.flush()
            print(f"{name}: language model {time.time() - t0:.0f} s: {proc.tokenizer.decode(gen)!r}", flush=True)


def cmd_engine(args):
    import base64
    import gzip

    import server
    import videoproc
    from q27lib import Engine
    from tokenizer import Tokenizer

    os.environ.setdefault("VISION", "bf16")
    tok = Tokenizer(args.model)
    imgs = server.Images(args.model, int(os.environ.get("IMAGE_MAX_PIXELS", 4194304)))
    pads = tuple(tok.token_id(x) for x in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"))
    eng = Engine(args.model, max_tokens=args.max_tokens, kv_dtype=args.kv_dtype)
    proc = videoproc.VideoProcessor(args.model)
    with gzip.open(args.out, "wt") as f:
        for name, path, q in cases(args):
            raw = pathlib.Path(path).read_bytes()
            url = "data:video/mp4;base64," + base64.b64encode(raw).decode()
            ids, spans, srcs, videos = imgs.prepare(tok.encode(prompt_of(q)), [("video", url)], pads, tok.encode)
            plan = videos[0][1]
            t0 = time.time()
            frames = videoproc.frames(raw, plan)
            t_dec = time.time() - t0
            eng.reset()
            eng.mm_set(spans)
            t0 = time.time()
            eng.mm_encode_frames(0, frames, proc.norm)
            t_vis = time.time() - t0
            if args.save_embeds:  # (for ref --engine-embeds: the reference language model on these embeddings)
                import torch

                emb = embeds(eng, plan, plan.grid[1] * plan.grid[2] // 4)
                (ROOT / "cache/video").mkdir(parents=True, exist_ok=True)
                torch.save({"embeds": emb}, ROOT / f"cache/video/{name}-engine.pt")
            t0 = time.time()
            topk = [[[int(a), round(float(c), 4)] for a, c in row] for row in eng.forward_topk(ids, TOPK)]
            t_pf = time.time() - t0
            gen = [topk[-1][0][0]]
            while len(gen) < args.tokens and gen[-1] not in (248044, 248046):
                gen.append(eng.forward_sample(gen[-1:]))
            canon = [pads[1] if i >= 248320 else i for i in ids]  # (the pads as the processor has them)
            f.write(json.dumps({"name": name, "ids": canon, "topk": topk, "grid": list(plan.grid), "gen": gen}) + "\n")
            print(f"{name}: {len(ids)} tokens, decode {t_dec:.2f} s, vision {t_vis:.2f} s, prefill {t_pf:.2f} s: "
                  f"{tok.decode(gen)!r}", flush=True)
    eng.close()


def cmd_bench(args):
    from q27lib import Engine

    eng, rows_out = None, []
    for seconds in (10, 60, 300):
        raw, proc, plan, frames, t_dec = load(args, seconds)
        if eng is None:
            eng = Engine(args.model, max_tokens=4096, kv_dtype="fp8")
        t, gh, gw = plan.grid
        times = []
        for _ in range(3):
            eng.reset()
            eng.mm_set(spans_of(plan))
            t0 = time.perf_counter()
            eng.mm_encode_frames(0, frames, proc.norm)
            times.append(time.perf_counter() - t0)
        r = {"seconds": seconds, "frames": len(plan.indices), "pairs": t, "grid": [gh, gw], "tokens": t * gh * gw // 4,
             "decode_s": round(t_dec, 3), "encode_s": round(min(times[1:]), 3)}
        rows_out.append(r)
        print(r, flush=True)
    eng.close()
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"vision": os.environ["VISION"], "sessions": args.sessions,
                                                      "videos": rows_out}, indent=1) + "\n")


def main():
    from qwen27b.weights import TARGET, snapshot

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["embeds", "bench", "ref", "engine"])
    ap.add_argument("--cases", help="ref / engine: video files (comma separated), each asked QUESTIONS")
    ap.add_argument("--tokens", type=int, default=64, help="ref / engine: greedy continuation length")
    ap.add_argument("--kv-dtype", default="bf16", help="ref / engine: KV cache type")
    ap.add_argument("--max-tokens", type=int, default=16384, help="engine: context")
    ap.add_argument("--save-embeds", action="store_true", help="engine: save the vision embeddings (cache/video/)")
    ap.add_argument("--engine-embeds", action="store_true", help="ref: use the engine's saved vision embeddings")
    ap.add_argument("--video", help="a video file (default: a pan over the astronaut photo)")
    ap.add_argument("--seconds", type=float, default=8, help="embeds: the generated video's length")
    ap.add_argument("--sessions", type=int, default=0, help="parallel NVDEC decoders (0: one per engine)")
    ap.add_argument("--out", help="bench: output file")
    args = ap.parse_args()
    args.model = snapshot(*TARGET)
    os.environ.setdefault("VISION", "host")
    {"embeds": cmd_embeds, "bench": cmd_bench, "ref": cmd_ref, "engine": cmd_engine}[args.mode](args)


if __name__ == "__main__":
    main()
