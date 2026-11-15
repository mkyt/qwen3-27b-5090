"""videoproc.py, the engine's video demuxer and decoder, and the server's video prompts, on videos encoded here
(PyAV, the dev image only):

- the plan, the prompt ids and the M-RoPE positions are transformers' Qwen3VLVideoProcessor's
  (cap_pixels_per_frame=True), the probe is PyAV's (frames, rate, size), MP4 and Matroska / WebM;
- with a GPU: the frames are those of PyAV within the YUV-to-RGB conversion's rounding, the GPU resize is
  PIL's bit for bit, parallel decoders give the frames of one, a Matroska remux the frames of its MP4;
- damaged files are refused (ValueError), not crashed on.

Skipped where PyAV, the engine library or the checkpoint is missing (transformers and the GPU for their parts).
"""

import base64
import ctypes
import io
import pathlib
import random
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "engine/python"))

import videoproc  # noqa: E402

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

# (frames, fps, width, height): an odd count of samples, upscaling below the patch factor, 720p, a long one
# capped at max_frames, a short one with fewer frames than min_frames would ask for, a tall one.
VIDEOS = [(30, 10, 160, 120), (35, 5, 96, 200), (91, 30, 1280, 720), (1700, 2, 64, 48), (3, 30, 320, 240),
          (240, 24, 32, 400)]


def encode(frames, fps, width, height, seed=0, codec="libx264", options=None, container=None, fmt="mp4"):
    """A video of smooth random frames moving sideways (the codec's 4:2:0, so the decode is not the identity)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (height // 8 + 2, width // 8 + 2, 3), dtype=np.uint8))
    base = np.asarray(small.resize((small.width * 8, small.height * 8), Image.BICUBIC)).astype(np.float32)
    buf = io.BytesIO()
    with av.open(buf, "w", format=fmt, options=container or {}) as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = width + width % 2, height + height % 2, "yuv420p"
        s.options = {"crf": "18", "preset": "ultrafast", **(options or {})} if codec == "libx264" else (options or {})
        for i in range(frames):
            img = np.roll(base, i, axis=1)[:s.height, :s.width]
            f = av.VideoFrame.from_ndarray(np.clip(img + 3 * np.sin(i), 0, 255).astype(np.uint8), format="rgb24")
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def remux(raw, fmt="matroska"):
    """The same packets in another container."""
    buf = io.BytesIO()
    with av.open(io.BytesIO(raw)) as src, av.open(buf, "w", format=fmt) as dst:
        s = dst.add_stream_from_template(src.streams.video[0], opaque=True)
        for p in src.demux(video=0):
            if p.size:
                p.stream = s
                dst.mux(p)
    return buf.getvalue()


def decode_all(raw):
    """All frames as transformers' PyAV reader has them (rgb24)."""
    with av.open(io.BytesIO(raw)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def pyav_probe(raw):
    """(frames, rate, width, height); the frames counted by packet where the container does not say
    (Matroska, fragmented MP4: PyAV has 0)."""
    with av.open(io.BytesIO(raw)) as c:
        s = c.streams.video[0]
        info = [s.frames, float(s.average_rate), s.width, s.height]
        if not info[0]:
            info[0] = sum(1 for p in c.demux(s) if p.size)
        return tuple(info)


@pytest.fixture(scope="module")
def engine_lib():
    try:
        from q27lib import lib

        return lib()
    except OSError as e:
        pytest.skip(f"no engine library: {e}")


@pytest.fixture(scope="module")
def gpu(engine_lib):
    raw = encode(8, 10, 64, 64)
    try:
        videoproc.frames(raw, _plan([0, 1], (64, 64)))
    except RuntimeError as e:
        pytest.skip(f"no NVDEC: {e}")
    return True


@pytest.fixture(scope="module")
def model():
    from qwen27b.paths import engine_target

    try:
        return engine_target("gdnfp4")
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _plan(indices, size):
    return videoproc.Plan(tuple(indices), size, (len(indices) // 2, size[1] // 16, size[0] // 16), ())


def test_linspace_round():
    for stop in (0, 1, 2, 7, 29, 299, 1799, 10000):
        for n in (1, 2, 3, 4, 7, 120, 768):
            assert videoproc.linspace_round(stop, n) == np.linspace(0, stop, n).round().astype(int).tolist()


CODECS = {"h264": ("libx264", None), "bframes": ("libx264", {"bf": "3"}),
          "hevc": ("libx265", {"x265-params": "log-level=error"}), "av1": ("libsvtav1", {"preset": "12"}),
          "vp9": ("libvpx-vp9", {"deadline": "realtime", "cpu-used": "8"})}


@pytest.mark.parametrize("kind", ["h264", "bframes", "hevc", "av1", "fragmented", "mkv-h264", "mkv-bframes",
                                  "mkv-hevc", "mkv-av1", "webm-av1", "webm-vp9", "mkv-ntsc"])
def test_probe_matches_pyav(engine_lib, kind):
    fmt = {"mkv": "matroska", "webm": "webm"}.get(kind.split("-")[0], "mp4")
    codec, opts = CODECS[kind.split("-")[-1].replace("fragmented", "h264").replace("ntsc", "h264")]
    container = {"movflags": "frag_keyframe+empty_moov"} if kind == "fragmented" else None
    rate = Fraction(30000, 1001) if kind == "mkv-ntsc" else 15  # (DefaultDuration 33366667 ns: 30000 / 1001)
    raw = encode(45, rate, 176, 144, codec=codec, options=opts, container=container, fmt=fmt)
    assert videoproc.probe(raw) == pyav_probe(raw) == (45, float(rate), 176, 144)


def test_mkv_without_default_duration(engine_lib):
    """The rate from the frames' timestamps (libavformat guesses 30.3 from its first packets here)."""
    raw = encode(37, Fraction(30000, 1001), 176, 144, codec="libx264", fmt="matroska")
    at = raw.find(b"\x23\xe3\x83")  # DefaultDuration -> a Void element of the same length
    n = 4 + (raw[at + 3] & 0x7f)
    raw = raw[:at] + bytes([0xEC, 0x80 | (n - 2)]) + bytes(n - 2) + raw[at + n:]
    total, fps, _, _ = videoproc.probe(raw)
    assert total == 37 and abs(fps - 30000 / 1001) < 0.01


@pytest.mark.parametrize("video", VIDEOS)
def test_video_matches_transformers(engine_lib, model, video):
    """Plan and prompt ids are those of transformers, the language model's positions those of its
    get_rope_index; with a GPU the pixels are within the YUV conversion's differences of its pixel values."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from transformers.video_utils import VideoMetadata

    import server
    from tokenizer import Tokenizer

    raw = encode(*video)
    total, rate, w, h = videoproc.probe(raw)
    assert (total, rate) == (video[0], video[1])
    mine = videoproc.VideoProcessor(model)
    plan = mine.plan(total, rate, w, h)

    proc = transformers.AutoProcessor.from_pretrained(model)
    md = VideoMetadata(total_num_frames=total, fps=rate, duration=total / rate, width=w, height=h)
    prompt = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>What happens?<|im_end|>\n"
    ref = proc(text=[prompt], videos=[decode_all(raw)], video_metadata=[md], do_sample_frames=True,
               cap_pixels_per_frame=True, return_tensors="np", return_metadata=True)
    sampled = ref["video_metadata"][0].frames_indices.tolist()
    assert sampled == list(plan.indices[:len(sampled)])
    assert ref["video_grid_thw"].tolist() == [list(plan.grid)]

    # The server's prompt: the processor's ids, with each pair's ids derived from the video in place of the pads.
    tok = Tokenizer(model)
    imgs = server.Images(model, 4194304)
    pads = tuple(tok.token_id(s) for s in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"))
    url = "data:video/mp4;base64," + base64.b64encode(raw).decode()
    ids, spans, srcs, videos = imgs.prepare(tok.encode(prompt), [("video", url)], pads, tok.encode)
    t, gh, gw = plan.grid
    assert [pads[1] if i >= 248320 else i for i in ids] == ref["input_ids"][0].tolist()
    assert len(spans) == t and all((g, w2) == (gh // 2, gw // 2) for _, g, w2 in spans)
    assert [s[:3] for s in srcs] == [("video", 0, g) for g in range(t)]
    assert videos[0][1] == plan

    from qwen27b.image_tokens import ImageSpan
    from qwen27b.vision import mrope_positions
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    with torch.device("meta"):
        lm = Qwen3_5Model(transformers.AutoConfig.from_pretrained(model))
    pos, delta = lm.get_rope_index(torch.from_numpy(ref["input_ids"]), torch.from_numpy(ref["mm_token_type_ids"]),
                                   video_grid_thw=torch.from_numpy(ref["video_grid_thw"]))
    mine_pos, mine_delta = mrope_positions(len(ids), [ImageSpan(*sp) for sp in spans])
    assert torch.equal(pos[:, 0].cpu(), mine_pos) and int(delta) == mine_delta

    try:
        rgb = videoproc.frames(raw, plan)
    except RuntimeError as e:
        pytest.skip(f"no NVDEC for the pixels: {e}")
    except ValueError as e:  # (NVDEC's smallest picture is larger than some of these)
        assert "smaller than NVDEC decodes" in str(e)
        return
    rw, rh = plan.size
    x = torch.stack([torch.frombuffer(bytearray(b), dtype=torch.uint8).view(rh, rw, 3) for b in rgb]).float()
    x = (x.permute(0, 3, 1, 2) - torch.tensor(mine.norm[:3])[:, None, None]) / torch.tensor(mine.norm[3:])[:, None, None]
    p, m = mine.patch, mine.merge
    x = x.view(t, 2, 3, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    d = np.abs(x.reshape(t * gh * gw, -1).numpy() - ref["pixel_values_videos"]) * 127.5  # (in 8-bit steps)
    print(f"pixels vs transformers: mean {d.mean():.3f}, 99.9% {np.quantile(d, 0.999):.1f}, max {d.max():.1f} steps")
    # (FFmpeg's YUV-to-RGB on x86 rounds its own way, about 0.6 steps on average; transformers resizes with
    # torchvision, this with PIL: about 0.5 more)
    assert d.mean() < 1.5 and np.quantile(d, 0.999) <= 4 and d.max() <= 8


@pytest.mark.parametrize("size", [(1280, 720, 608, 320), (640, 360, 1152, 640), (720, 1280, 352, 640), (96, 64, 96, 32)])
def test_gpu_resize_is_pil(gpu, size):
    from PIL import Image

    w, h, ow, oh = size
    raw = encode(12, 10, w, h)
    idx = [0, 5, 6, 11]
    full = videoproc.frames(raw, _plan(idx, (w, h)))
    small = videoproc.frames(raw, _plan(idx, (ow, oh)))
    for a, b in zip(full, small):
        assert Image.frombytes("RGB", (w, h), a).resize((ow, oh), Image.BICUBIC).tobytes() == b


def test_tail_plan():
    plan = videoproc.Plan(tuple(range(0, 40, 2)), (64, 32), (10, 2, 4), tuple(float(t) for t in range(10)))
    assert videoproc.tail(plan, 0) is plan
    t = videoproc.tail(plan, 7)
    assert t.indices == plan.indices[14:] and t.times == plan.times[7:] and t.grid == (3, 2, 4) and t.size == plan.size


def _decoded(raw, plan, sessions):
    """(frames, pictures NVDEC decoded) of videoproc.frames' call."""
    from q27lib import lib

    want = sorted(set(plan.indices))
    w, h = plan.size
    buf = ctypes.create_string_buffer(len(want) * w * h * 3)
    st = (ctypes.c_double * 5)()
    assert lib().q27_video_frames(raw, len(raw), (ctypes.c_int32 * len(want))(*want), len(want), w, h, sessions, buf,
                                  st) == 0
    return buf.raw, int(st[1])


@pytest.mark.parametrize("kind", ["bframes", "hevc", "av1"])
def test_tail_decodes_from_its_key_frame(gpu, kind):
    """The frames of a plan's tail (the pairs past a cached prefix) are those of the whole plan, decoded from the
    closed key frame before them (HEVC's CRA pictures are open: from the start, or an IDR)."""
    codec, opts = CODECS[kind]
    opts = {**(opts or {}), "g": "48"} if kind != "av1" else {**opts, "g": "48"}
    raw = encode(400, 30, 320, 240, codec=codec, options=opts)
    plan = _plan(videoproc.linspace_round(399, 100), (160, 128))
    full = videoproc.frames(raw, plan, sessions=1)
    _, all_pictures = _decoded(raw, plan, 1)
    for g in (1, 17, 30, 49):
        for sessions in (1, 2):
            assert videoproc.frames(raw, videoproc.tail(plan, g), sessions=sessions) == full[2 * g:], (g, sessions)
    _, pictures = _decoded(raw, videoproc.tail(plan, 45), 1)
    if kind != "hevc":  # (x265's key frames are open CRA pictures: no closed one to start at)
        assert pictures < all_pictures // 2, (pictures, all_pictures)


def test_parallel_decoders_match_one(gpu):
    raw = encode(400, 30, 640, 360, options={"g": "32", "bf": "2"})
    plan = _plan(list(range(0, 400, 7)) + [399], (320, 192))
    assert videoproc.frames(raw, plan, sessions=2) == videoproc.frames(raw, plan, sessions=1)


@pytest.mark.parametrize("kind", ["bframes", "hevc", "av1", "vp9"])
def test_matroska_frames_are_mp4s(gpu, kind):
    codec, opts = CODECS[kind]
    raw = encode(60, 24, 320, 240, codec=codec, options={**(opts or {}), "g": "16"} if kind != "av1" else opts)
    plan = _plan(list(range(0, 60, 3)), (160, 128))
    mkv = remux(raw)
    (n, fps, *size), (n2, fps2, *size2) = videoproc.probe(mkv), videoproc.probe(raw)
    assert (n, size) == (n2, size2) and abs(fps - fps2) < 0.01  # (without DefaultDuration: millisecond stamps)
    assert videoproc.frames(mkv, plan, sessions=2) == videoproc.frames(raw, plan, sessions=1)


@pytest.mark.parametrize("fmt", ["mp4", "matroska"])
def test_damaged_files_are_refused(engine_lib, fmt):
    raw = encode(30, 10, 160, 120, fmt=fmt)
    with pytest.raises(ValueError):
        videoproc.probe(b"not a video at all" * 10)
    with pytest.raises(ValueError):
        videoproc.probe(b"\x1a\x45\xdf\xa3" + b"not a video at all" * 10)
    with pytest.raises(ValueError):
        videoproc.probe(raw[:len(raw) // 3])
    rng = random.Random(0)
    # (mostly the headers: the movie box, or the Matroska elements up to the first cluster's blocks)
    lo, hi = (max(0, raw.find(b"moov") - 4), len(raw)) if fmt == "mp4" else (0, raw.find(b"\x1f\x43\xb6\x75") + 256)
    for _ in range(300 if fmt == "mp4" else 1000):  # (the demuxer either reads the file or refuses it: never a crash)
        b = bytearray(raw)
        for _ in range(rng.randint(1, 8)):
            at = rng.randrange(lo, hi) if rng.random() < 0.7 else rng.randrange(len(b))
            b[at] = rng.randrange(256)
        try:
            videoproc.probe(bytes(b))
        except ValueError:
            pass
