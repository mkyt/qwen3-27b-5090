"""Videos through the engine's vision tower (frame pairs encoded together, q27_mm_encode_frames) against the
reference (vision.py, fp32 on the GPU) and against one pair per call; times the encode per video length.

The video is a pan over bench/vision/astronaut.jpg (or --video FILE; PyAV encodes the pan, the dev image has it),
planned and decoded as the server does (frontend/videoproc.py: the engine's demuxer, NVDEC, the GPU resize).

    docker compose --profile dev run --rm dev python scripts/video_check.py embeds
    docker compose --profile dev run --rm dev python scripts/video_check.py bench --out bench/results/<stamp>-video.json
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
    ap.add_argument("mode", choices=["embeds", "bench"])
    ap.add_argument("--video", help="a video file (default: a pan over the astronaut photo)")
    ap.add_argument("--seconds", type=float, default=8, help="embeds: the generated video's length")
    ap.add_argument("--sessions", type=int, default=0, help="parallel NVDEC decoders (0: one per engine)")
    ap.add_argument("--out", help="bench: output file")
    args = ap.parse_args()
    args.model = snapshot(*TARGET)
    os.environ.setdefault("VISION", "host")
    {"embeds": cmd_embeds, "bench": cmd_bench}[args.mode](args)


if __name__ == "__main__":
    main()
