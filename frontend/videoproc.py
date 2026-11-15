"""The checkpoint's video processor without transformers / PyTorch, as transformers 5.18 `Qwen3VLVideoProcessor`
with `cap_pixels_per_frame=True` (qwen-vl-utils' per-frame cap, transformers' default from v5.22): frames sampled
at `fps` (2) between `min_frames` and `max_frames`, an even count (the last frame repeated), all resized to one
size, the pixels of a frame capped so that a video costs tokens by its length.

The language model sees each pair of frames (the temporal patch) as an image of the (1, gh, gw) grid, preceded by
its time: `<{t:.1f} seconds><|vision_start|><|video_pad|> x gh * gw / 4<|vision_end|>`, inside the template's own
vision_start / vision_end (the processor's prompt has both).

The engine library reads and decodes the videos (engine/src/video_*.cc, *.cu): MP4 / MOV (fragmented too) or
Matroska / WebM with H.264, HEVC, AV1 or VP9, decoded on NVDEC, converted to RGB and resized on the GPU (PIL's
bicubic, the same bytes as imageproc.resize). The plan (the frames to take, the size, the grid, the times) needs
only the container's header (probe, no GPU), which the server reads; the engine process decodes (frames) only the
videos it has to encode.
"""

import ctypes
import json
import math
import os
import pathlib
from dataclasses import dataclass

from imageproc import _f32

DEFAULTS = {"fps": 2, "min_frames": 4, "max_frames": 768, "max_video_tokens": 768}  # (the processor class's)


@dataclass(frozen=True)
class Plan:
    """What to take from a video: the source frame of each sampled frame (an even count), the resized
    (width, height), the patch grid (t pairs, gh, gw) and the time of each pair in seconds."""

    indices: tuple
    size: tuple
    grid: tuple
    times: tuple


def smart_resize(num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels):
    """transformers' qwen3_vl video smart_resize: (height, width) divisible by factor, with the pixels of all
    frames together within min_pixels..max_pixels."""
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def linspace_round(stop, n):
    """numpy's linspace(0, stop, n).round() as ints (the same float steps; round half to even)."""
    if n == 1:
        return [0]
    step = stop / (n - 1)
    return [round(i * step) for i in range(n - 1)] + [round(float(stop))]


def _check(r, what):
    """Raises ValueError (the client's video: HTTP 400) or RuntimeError for a failed engine call."""
    if r == 0:
        return
    from q27lib import lib

    msg = lib().q27_last_error().decode(errors="replace")
    raise (ValueError if r == -3 else RuntimeError)(f"{msg}" if r == -3 else f"{what}: {msg}")


def probe(raw):
    """(frame count, frames per second, width, height) from the container's header (the engine library's
    demuxer; no GPU). The count is the track's samples (Matroska: its frames), as transformers' PyAV reader has it
    for MP4 (it has 0 for Matroska)."""
    from q27lib import lib

    out = (ctypes.c_double * 6)()
    _check(lib().q27_video_probe(raw, len(raw), out), "q27_video_probe")
    _, width, height, total, fps, _ = out
    if total <= 0 or fps <= 0:
        raise ValueError("cannot tell the video's frame count or rate")
    return int(total), float(fps), int(width), int(height)


class VideoProcessor:
    """video_preprocessor_config.json of a model directory (the processor class's defaults where it has none)."""

    def __init__(self, path):
        d = pathlib.Path(path)
        f = d / "video_preprocessor_config.json"
        c = json.loads((f if f.exists() else d / "preprocessor_config.json").read_text(encoding="utf-8"))
        c = {**DEFAULTS, **c}
        self.patch, self.merge, self.temporal = c["patch_size"], c["merge_size"], c["temporal_patch_size"]
        self.min_pixels, self.max_pixels = c["size"]["shortest_edge"], c["size"]["longest_edge"]
        self.fps, self.min_frames, self.max_frames = c["fps"], c["min_frames"], c["max_frames"]
        self.max_video_tokens = c["max_video_tokens"]  # (of a pair of frames)
        # VIDEO_MAX_TOKENS / VIDEO_MAX_FRAMES: the server's own bounds of a video in place of the checkpoint's
        # (12,288 tokens = its size.longest_edge, 768 frames); a request may ask up to VIDEO_MAX_FRAMES frames.
        if tokens := os.environ.get("VIDEO_MAX_TOKENS"):
            self.max_pixels = _positive("VIDEO_MAX_TOKENS", tokens) * self.pixels_per_token
        if frames := os.environ.get("VIDEO_MAX_FRAMES"):
            self.max_frames = _positive("VIDEO_MAX_FRAMES", frames)
        rescale = c.get("rescale_factor", 1 / 255)
        mean = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c.get("image_mean", [0.5] * 3)]
        std = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c.get("image_std", [0.5] * 3)]
        self.norm = tuple(mean + std)

    @property
    def pixels_per_token(self):
        """Pixels of all frames together per token (32 x 32 pixels of each frame of a pair: 2,048)."""
        return (self.patch * self.merge) ** 2 * self.temporal

    @property
    def max_tokens(self):
        """The tokens of a video at most (without a request's size), its pairs' time stamps aside."""
        return self.max_pixels // self.pixels_per_token

    def plan(self, total, fps, width, height, frame_pixels=None, options=None):
        """The Plan of a video of `total` frames at `fps` of width x height (frame_pixels: a further cap on the
        pixels of a frame, the vision tower's IMAGE_MAX_PIXELS). options: a request's overrides (options_of):
        "fps" (frames sampled per second) or "num_frames" (that many, as transformers' sample_frames takes them),
        "min_frames" / "max_frames" (the bounds of the fps count), "max_pixels" (size.longest_edge: the pixels of all
        frames together)."""
        o = options or {}
        if "num_frames" in o:
            n = o["num_frames"]
        else:
            n = int(total / fps * o.get("fps", self.fps))
            n = min(max(n, o.get("min_frames", self.min_frames)), o.get("max_frames", self.max_frames), total)
        indices = linspace_round(total - 1, n)
        factor = self.patch * self.merge
        max_pixels = o.get("max_pixels", self.max_pixels)
        per_frame = max(min(self.max_video_tokens * factor * factor, max_pixels // n), int(self.min_pixels * 1.05))
        if frame_pixels:
            per_frame = min(per_frame, frame_pixels)
        rh, rw = smart_resize(n, height, width, self.temporal, factor, self.min_pixels, per_frame * n)
        if pad := -n % self.temporal:
            indices += [indices[-1]] * pad
        t = self.temporal
        stamps = [i / fps for i in indices]
        times = tuple((stamps[k] + stamps[k + t - 1]) / 2 for k in range(0, len(stamps), t))
        return Plan(tuple(indices), (rw, rh), (len(indices) // t, rh // self.patch, rw // self.patch), times)


MAX_FRAMES = 768  # (the most frames a request may ask for by default: the processor's own max_frames)
MAX_REQUEST_PIXELS = 1 << 40  # (size.longest_edge of a request at most; IMAGE_TOKENS bounds the tokens)


def _positive(name, v):
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")
    return n


def options_of(kw, max_frames=MAX_FRAMES):
    """A request's video options (vLLM's `mm_processor_kwargs`: fps, num_frames, min_frames, max_frames, and
    transformers' size.longest_edge, the pixels of all frames together, as "max_pixels"), checked; None without
    any. max_frames: the most frames a request may ask for (VideoProcessor.max_frames). Raises ValueError (HTTP
    400) for a bad value; other keys are ignored."""
    if kw is None:
        return None
    if not isinstance(kw, dict):
        raise ValueError("mm_processor_kwargs must be an object")
    out = {}
    if "fps" in kw:
        v = kw["fps"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 60:
            raise ValueError("mm_processor_kwargs.fps must be a number in (0, 60]")
        out["fps"] = float(v)
    for k in ("num_frames", "min_frames", "max_frames"):
        if k in kw:
            v = kw[k]
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= max_frames:
                raise ValueError(f"mm_processor_kwargs.{k} must be an integer in [1, {max_frames}] (VIDEO_MAX_FRAMES)")
            out[k] = v
    if "size" in kw:
        size = kw["size"]
        if not isinstance(size, dict):
            raise ValueError('mm_processor_kwargs.size must be an object, e.g. {"longest_edge": 50331648}')
        if "longest_edge" in size:
            v = size["longest_edge"]
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_REQUEST_PIXELS:
                raise ValueError("mm_processor_kwargs.size.longest_edge (the pixels of all frames together) must be a "
                                 "positive integer")
            out["max_pixels"] = v
    if "fps" in out and "num_frames" in out:
        raise ValueError("mm_processor_kwargs: fps and num_frames are mutually exclusive")
    if out.get("min_frames", 1) > out.get("max_frames", max_frames):
        raise ValueError("mm_processor_kwargs: min_frames is above max_frames")
    return out or None


def tail(plan, g):
    """The Plan of pairs g.. of a plan (the frames a request still has to encode when its first g pairs are in the
    prefix cache)."""
    if g == 0:
        return plan
    t = len(plan.indices) // plan.grid[0]
    return Plan(plan.indices[t * g:], plan.size, (plan.grid[0] - g, *plan.grid[1:]), plan.times[g:])


def frames(raw, plan, sessions=0):
    """The resized frames of a Plan, [height][width][3] uint8 bytes each, in the plan's order: decoded on NVDEC
    and resized on the GPU (sessions: parallel decoders, 0: one per NVDEC engine). A frame the stream turns out
    not to have (past its end, after its edit list) is the last decoded one."""
    from q27lib import lib

    want = sorted(set(plan.indices))
    w, h = plan.size
    frame = w * h * 3
    buf = ctypes.create_string_buffer(len(want) * frame)
    st = (ctypes.c_double * 5)()
    _check(lib().q27_video_frames(raw, len(raw), (ctypes.c_int32 * len(want))(*want), len(want), w, h, sessions, buf, st),
           "q27_video_frames")
    shown = int(st[0])
    got = [i for i in want if i < shown]
    if not got:
        raise ValueError("no frames decoded")
    data = buf.raw
    slot = {i: data[k * frame:(k + 1) * frame] for k, i in enumerate(want) if i < shown}
    last = slot[got[-1]]
    return [slot.get(i, last) for i in plan.indices]


def memory(raw, plan, sessions=0):
    """(bytes, sessions): the device memory frames(raw, plan, sessions) takes while it runs, outside the engine's
    pools (NVDEC's surfaces for the stream's first sequence header, the driver's state, the resize buffers; read by
    the parser on the CPU, no decoder), and the parallel decoders it would run (from the closed key frame before the
    plan's first frame: a tail takes fewer). The worker lends that much of the KV
    pool for the decode (Engine.kv_lend)."""
    from q27lib import lib

    out = (ctypes.c_double * 4)()
    w, h = plan.size
    _check(lib().q27_video_memory(raw, len(raw), w, h, sessions, min(plan.indices), out), "q27_video_memory")
    return int(out[0]), int(out[1])
