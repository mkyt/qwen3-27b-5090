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
        self.max_video_tokens = c["max_video_tokens"]
        rescale = c.get("rescale_factor", 1 / 255)
        mean = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c.get("image_mean", [0.5] * 3)]
        std = [_f32(_f32(v) * _f32(1.0 / rescale)) for v in c.get("image_std", [0.5] * 3)]
        self.norm = tuple(mean + std)

    def plan(self, total, fps, width, height, frame_pixels=None):
        """The Plan of a video of `total` frames at `fps` of width x height (frame_pixels: a further cap on the
        pixels of a frame, the vision tower's IMAGE_MAX_PIXELS)."""
        n = int(total / fps * self.fps)
        n = min(max(n, self.min_frames), self.max_frames, total)
        indices = linspace_round(total - 1, n)
        factor = self.patch * self.merge
        per_frame = max(min(self.max_video_tokens * factor * factor, self.max_pixels // n), int(self.min_pixels * 1.05))
        if frame_pixels:
            per_frame = min(per_frame, frame_pixels)
        rh, rw = smart_resize(n, height, width, self.temporal, factor, self.min_pixels, per_frame * n)
        if pad := -n % self.temporal:
            indices += [indices[-1]] * pad
        t = self.temporal
        stamps = [i / fps for i in indices]
        times = tuple((stamps[k] + stamps[k + t - 1]) / 2 for k in range(0, len(stamps), t))
        return Plan(tuple(indices), (rw, rh), (len(indices) // t, rh // self.patch, rw // self.patch), times)


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
