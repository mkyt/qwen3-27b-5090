// Video decoding on NVDEC (the driver's libnvcuvid / nvcuvid.dll, loaded at run time) with its bitstream
// parser: the container's samples go to the parser, which calls back to decode each picture (decode order)
// and to show it (presentation order). Only the frames asked for are mapped and handed on; the rest stay
// on the decoder. The parser keeps up to kDisplayDelay pictures queued, so decoding overlaps the mapping.
// One session keeps one NVDEC engine busy: a long video is cut at closed key frames into parts decoded at
// once, one session per engine. The frames are converted to RGB and resized (PIL's bicubic, bit for bit)
// on the GPU; only the resized frames leave it.
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common.h"
#include "third_party/nvcodec/dynlink_nvcuvid.h"
#include "video.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace q27 {
namespace {

constexpr int kDisplayDelay = 4;  // pictures the parser queues before showing (the SDK's 2..4)
constexpr int kExtraSurfaces = 4;  // decode surfaces beyond the stream's minimum, for kDisplayDelay
// video_memory: per decoder, besides its surfaces (the driver's state, VP9's buffers, the resize; measured 12 to 57 MiB
// over the surfaces from 360p to 4K, VP9 the most) and in all (CUDA's stream-ordered pool, the resize weights).
constexpr size_t kSessionOverhead = size_t(72) << 20;
constexpr size_t kDecodeOverhead = size_t(16) << 20;

struct Cuvid {
  tcuvidGetDecoderCaps* GetDecoderCaps = nullptr;
  tcuvidCreateDecoder* CreateDecoder = nullptr;
  tcuvidDestroyDecoder* DestroyDecoder = nullptr;
  tcuvidDecodePicture* DecodePicture = nullptr;
  tcuvidMapVideoFrame64* MapVideoFrame = nullptr;
  tcuvidUnmapVideoFrame64* UnmapVideoFrame = nullptr;
  tcuvidCreateVideoParser* CreateVideoParser = nullptr;
  tcuvidParseVideoData* ParseVideoData = nullptr;
  tcuvidDestroyVideoParser* DestroyVideoParser = nullptr;
  std::string error;
};

const Cuvid& cuvid() {
  static Cuvid c;
  static std::once_flag once;
  std::call_once(once, [] {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvcuvid.dll");
    auto sym = [&](const char* n) { return h ? reinterpret_cast<void*>(GetProcAddress(h, n)) : nullptr; };
#else
    void* h = dlopen("libnvcuvid.so.1", RTLD_NOW | RTLD_LOCAL);
    auto sym = [&](const char* n) { return h ? dlsym(h, n) : nullptr; };
#endif
    if (!h) {
      c.error = "NVDEC (libnvcuvid) is not available: the NVIDIA driver's video decoder library is missing";
      return;
    }
#define Q27_SYM(field, name) c.field = reinterpret_cast<decltype(c.field)>(sym(name)), ok &= c.field != nullptr
    bool ok = true;
    Q27_SYM(GetDecoderCaps, "cuvidGetDecoderCaps");
    Q27_SYM(CreateDecoder, "cuvidCreateDecoder");
    Q27_SYM(DestroyDecoder, "cuvidDestroyDecoder");
    Q27_SYM(DecodePicture, "cuvidDecodePicture");
    Q27_SYM(MapVideoFrame, "cuvidMapVideoFrame64");
    Q27_SYM(UnmapVideoFrame, "cuvidUnmapVideoFrame64");
    Q27_SYM(CreateVideoParser, "cuvidCreateVideoParser");
    Q27_SYM(ParseVideoData, "cuvidParseVideoData");
    Q27_SYM(DestroyVideoParser, "cuvidDestroyVideoParser");
#undef Q27_SYM
    if (!ok) c.error = "NVDEC (libnvcuvid) lacks entry points: the NVIDIA driver is too old";
  });
  if (!c.error.empty()) throw std::runtime_error(c.error);
  return c;
}

cudaVideoCodec codec_of(VideoCodec c) {
  switch (c) {
    case VideoCodec::kH264: return cudaVideoCodec_H264;
    case VideoCodec::kHevc: return cudaVideoCodec_HEVC;
    case VideoCodec::kAv1: return cudaVideoCodec_AV1;
    case VideoCodec::kVp9: return cudaVideoCodec_VP9;
    default: throw VideoError("unsupported video codec");
  }
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A part's frames were not where a closed key frame puts them (an open GOP after all): decode it whole.
struct SplitMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The decoder of samples [a, b): their frames are presented at a .. b - 1 (a closed key frame at a).
struct Session {
  const Cuvid& api;
  const VideoTrack& track;
  int a, b;
  std::vector<int> want;  // (frames after the edit list, of this part)
  cudaStream_t stream = nullptr;
  const VideoFrameFn& on_frame;
  CUvideodecoder dec = nullptr;
  CUVIDEOFORMAT fmt{};
  int width = 0, height = 0, surfaces = 0;
  int shown = 0, next = 0, decoded = 0;
  bool done = false;
  double setup_ms = 0, t0 = 0;
  std::string error;  // (callbacks cannot throw across the driver)
  bool client_error = false;

  ~Session() {
    if (dec) api.DestroyDecoder(dec);
  }

  int sequence(CUVIDEOFORMAT* f) {
    if (dec) {  // (a new sequence header: fine while the format stays)
      if (f->coded_width == fmt.coded_width && f->coded_height == fmt.coded_height && f->chroma_format == fmt.chroma_format &&
          f->bit_depth_luma_minus8 == fmt.bit_depth_luma_minus8)
        return surfaces;
      return fail("the video changes its size or format midway", true);
    }
    if (f->chroma_format != cudaVideoChromaFormat_420 && f->chroma_format != cudaVideoChromaFormat_422 &&
        f->chroma_format != cudaVideoChromaFormat_444)
      return fail("monochrome video is not supported", true);
    CUVIDDECODECAPS caps{};
    caps.eCodecType = f->codec, caps.eChromaFormat = f->chroma_format, caps.nBitDepthMinus8 = f->bit_depth_luma_minus8;
    if (api.GetDecoderCaps(&caps) != CUDA_SUCCESS) return fail("cuvidGetDecoderCaps failed", false);
    if (!caps.bIsSupported)
      return fail(std::string("NVDEC does not decode this ") + video_codec_name(track.codec) + " stream (" +
                      std::to_string(8 + f->bit_depth_luma_minus8) + "-bit " +
                      (f->chroma_format == cudaVideoChromaFormat_420   ? "4:2:0"
                       : f->chroma_format == cudaVideoChromaFormat_422 ? "4:2:2"
                                                                       : "4:4:4") + ")", true);
    if (f->coded_width < caps.nMinWidth || f->coded_height < caps.nMinHeight)
      return fail("the video is smaller than NVDEC decodes (" + std::to_string(caps.nMinWidth) + "x" +
                      std::to_string(caps.nMinHeight) + ")", true);
    if (f->coded_width > caps.nMaxWidth || f->coded_height > caps.nMaxHeight)
      return fail("the video is larger than NVDEC decodes (" + std::to_string(caps.nMaxWidth) + "x" +
                      std::to_string(caps.nMaxHeight) + ")", true);
    fmt = *f;
    width = f->display_area.right - f->display_area.left;
    height = f->display_area.bottom - f->display_area.top;
    if (width <= 0 || height <= 0) return fail("the video has an empty picture", true);
    surfaces = f->min_num_decode_surfaces + kExtraSurfaces;
    CUVIDDECODECREATEINFO ci{};
    ci.CodecType = f->codec;
    ci.ChromaFormat = f->chroma_format;
    const bool wide = f->bit_depth_luma_minus8 > 0;
    ci.OutputFormat = f->chroma_format == cudaVideoChromaFormat_444 ? (wide ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444)
                      : f->chroma_format == cudaVideoChromaFormat_422 ? (wide ? cudaVideoSurfaceFormat_P216 : cudaVideoSurfaceFormat_NV16)
                                                                      : (wide ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12);
    ci.bitDepthMinus8 = f->bit_depth_luma_minus8;
    ci.DeinterlaceMode = f->progressive_sequence ? cudaVideoDeinterlaceMode_Weave : cudaVideoDeinterlaceMode_Adaptive;
    ci.ulNumOutputSurfaces = 2;
    ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    ci.ulNumDecodeSurfaces = surfaces;
    ci.ulWidth = ci.ulMaxWidth = f->coded_width;
    ci.ulHeight = ci.ulMaxHeight = f->coded_height;
    ci.display_area.left = short(f->display_area.left), ci.display_area.top = short(f->display_area.top);
    ci.display_area.right = short(f->display_area.right), ci.display_area.bottom = short(f->display_area.bottom);
    ci.ulTargetWidth = width, ci.ulTargetHeight = height;
    if (api.CreateDecoder(&dec, &ci) != CUDA_SUCCESS) {
      dec = nullptr;
      return fail("cuvidCreateDecoder failed", false);
    }
    setup_ms = now_ms() - t0;
    return surfaces;
  }

  int decode(CUVIDPICPARAMS* p) {
    if (!dec) return fail("a picture before the sequence header", true);
    if (api.DecodePicture(dec, p) != CUDA_SUCCESS) return fail("cuvidDecodePicture failed", false);
    decoded++;
    return 1;
  }

  int display(CUVIDPARSERDISPINFO* d) {
    if (!d || done) return 1;
    const int i = a + shown++ - track.skip_front;  // (the frame's index after the edit list)
    if (i < 0 || next >= int(want.size()) || i != want[next]) return 1;
    CUVIDPROCPARAMS pp{};
    pp.progressive_frame = d->progressive_frame;
    pp.top_field_first = d->top_field_first;
    pp.second_field = d->repeat_first_field + 1;
    pp.unpaired_field = d->repeat_first_field < 0;
    pp.output_stream = reinterpret_cast<CUstream>(stream);
    unsigned long long ptr = 0;
    unsigned int pitch = 0;
    if (api.MapVideoFrame(dec, d->picture_index, &ptr, &pitch, &pp) != CUDA_SUCCESS)
      return fail("cuvidMapVideoFrame failed", false);
    VideoFrame f;
    f.luma = reinterpret_cast<const uint8_t*>(ptr);
    f.chroma = f.luma + size_t(pitch) * ((height + 1) & ~1);
    f.pitch = pitch, f.width = width, f.height = height, f.bits = 8 + fmt.bit_depth_luma_minus8;
    f.chroma_format = fmt.chroma_format == cudaVideoChromaFormat_444 ? 2 : fmt.chroma_format == cudaVideoChromaFormat_422 ? 1 : 0;
    f.matrix = fmt.video_signal_description.matrix_coefficients;
    f.full_range = fmt.video_signal_description.video_full_range_flag;
    try {
      on_frame(i, f, stream);
    } catch (const VideoError& e) {
      fail(e.what(), true);
    } catch (const std::exception& e) {
      fail(e.what(), false);
    }
    cudaError_t e = cudaStreamSynchronize(stream);  // (the frame is the decoder's again after the unmap)
    api.UnmapVideoFrame(dec, ptr);
    if (e != cudaSuccess) return fail(cudaGetErrorString(e), false);
    if (!error.empty()) return 0;
    if (++next == int(want.size())) done = true;
    return 1;
  }

  int fail(const std::string& what, bool client) {
    if (error.empty()) error = what, client_error = client;
    return 0;
  }

  void run(const uint8_t* data) {
    t0 = now_ms();
    if (want.empty()) return;
    // (the engine's primary context current on this thread: NVDEC uses it. The decoding threads use only
    // their own non-blocking streams and stream-ordered allocations: the engine may be capturing a graph.)
    Q27_CUDA(cudaSetDevice(0));
    Q27_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    struct StreamGuard {
      cudaStream_t s;
      ~StreamGuard() { cudaStreamDestroy(s); }
    } sg{stream};
    CUVIDPARSERPARAMS pp{};
    pp.CodecType = codec_of(track.codec);
    pp.ulMaxNumDecodeSurfaces = 1;  // (the sequence callback returns the real count)
    pp.ulMaxDisplayDelay = kDisplayDelay;
    pp.pUserData = this;
    pp.pfnSequenceCallback = [](void* u, CUVIDEOFORMAT* f) { return static_cast<Session*>(u)->sequence(f); };
    pp.pfnDecodePicture = [](void* u, CUVIDPICPARAMS* p) { return static_cast<Session*>(u)->decode(p); };
    pp.pfnDisplayPicture = [](void* u, CUVIDPARSERDISPINFO* d) { return static_cast<Session*>(u)->display(d); };
    CUvideoparser parser = nullptr;
    if (api.CreateVideoParser(&parser, &pp) != CUDA_SUCCESS) throw std::runtime_error("cuvidCreateVideoParser failed");
    struct ParserGuard {
      const Cuvid& api;
      CUvideoparser p;
      ~ParserGuard() { api.DestroyVideoParser(p); }
    } guard{api, parser};
    std::vector<uint8_t> buf;
    auto feed = [&](const uint8_t* p, size_t len, unsigned long flags) {
      CUVIDSOURCEDATAPACKET pk{};
      pk.flags = flags;
      pk.payload_size = tcu_ulong(len);
      pk.payload = p;
      if (api.ParseVideoData(parser, &pk) != CUDA_SUCCESS) fail("the video's bitstream cannot be parsed", true);
    };
    for (int k = a; k < b && !done && error.empty(); k++) {
      const VideoSample& smp = track.samples[k];
      const uint8_t* p = data + smp.offset;
      buf.clear();
      if (k == a) buf.insert(buf.end(), track.prefix.begin(), track.prefix.end());
      if (track.nal_length) {  // (length-prefixed NAL units -> Annex B)
        size_t at = 0;
        while (at < smp.size) {
          if (smp.size - at < size_t(track.nal_length)) throw VideoError("a NAL unit's length is cut off");
          size_t len = 0;
          for (int q = 0; q < track.nal_length; q++) len = len << 8 | p[at + q];
          at += track.nal_length;
          if (len > smp.size - at) throw VideoError("a NAL unit runs past its sample");
          static const uint8_t sc[4] = {0, 0, 0, 1};
          buf.insert(buf.end(), sc, sc + 4);
          buf.insert(buf.end(), p + at, p + at + len);
          at += len;
        }
        feed(buf.data(), buf.size(), 0);
      } else if (!buf.empty()) {
        buf.insert(buf.end(), p, p + smp.size);
        feed(buf.data(), buf.size(), 0);
      } else {
        feed(p, smp.size, 0);
      }
    }
    if (!done && error.empty()) feed(nullptr, 0, CUVID_PKT_ENDOFSTREAM);
  }
};

// Whether sample k starts a closed group of pictures (no frame after it refers to one before): an IDR / BLA
// picture for H.264 / HEVC (not a CRA, whose leading pictures may), a sync sample for AV1 and VP9.
bool closed_key(const uint8_t* data, const VideoTrack& t, int k) {
  const VideoSample& s = t.samples[k];
  if (!s.sync) return false;
  if (t.codec != VideoCodec::kH264 && t.codec != VideoCodec::kHevc) return true;
  if (!t.nal_length) return false;
  const uint8_t* p = data + s.offset;
  for (size_t at = 0; at + t.nal_length < s.size;) {
    size_t len = 0;
    for (int q = 0; q < t.nal_length; q++) len = len << 8 | p[at + q];
    at += t.nal_length;
    if (len == 0 || len > s.size - at) return false;
    if (t.codec == VideoCodec::kH264) {
      const int type = p[at] & 31;
      if (type == 5) return true;
      if (type >= 1 && type <= 4) return false;
    } else {
      const int type = (p[at] >> 1) & 63;
      if (type >= 16 && type <= 20) return true;
      if (type < 32) return false;
    }
    at += len;
  }
  return false;
}

// The sample decoding starts at for frames from `first` on: the closed key frame at or before it.
int start_sample(const uint8_t* data, const VideoTrack& track, int first) {
  int k0 = std::min(track.frames() - 1, std::max(0, first + track.skip_front));
  while (k0 > 0 && !closed_key(data, track, k0)) k0--;
  return k0;
}

int nvdec_engines(const Cuvid& api, VideoCodec c) {
  static int n = [&] {
    Q27_CUDA(cudaSetDevice(0));
    CUVIDDECODECAPS caps{};
    caps.eCodecType = codec_of(c), caps.eChromaFormat = cudaVideoChromaFormat_420;
    return api.GetDecoderCaps(&caps) == CUDA_SUCCESS && caps.nNumNVDECs ? int(caps.nNumNVDECs) : 1;
  }();
  return n;
}

constexpr int kMinPartSamples = 64;  // (shorter parts are not worth a session's setup, about 60 ms)

VideoDecodeStats decode_parts(const uint8_t* data, const VideoTrack& track, const std::vector<int>& want, int parts,
                              const VideoFrameFn& on_frame) {
  const Cuvid& api = cuvid();
  const int n = track.frames();
  // The decode starts at the closed key frame at or before the first wanted frame (the samples before it are not
  // needed), and is cut at the closed key frames nearest to the j / parts points of the rest, at least
  // kMinPartSamples apart.
  const int k0 = start_sample(data, track, want.front());
  std::vector<int> cuts{k0};
  for (int j = 1; j < parts; j++) {
    const int target = k0 + int(int64_t(n - k0) * j / parts);
    int best = -1;
    for (int d = 0; d < n && best < 0; d++)
      for (int k : {target + d, target - d})
        if (best < 0 && k > cuts.back() + kMinPartSamples && k < n - kMinPartSamples && closed_key(data, track, k)) best = k;
    if (best > cuts.back()) cuts.push_back(best);
  }
  cuts.push_back(n);
  std::vector<std::unique_ptr<Session>> ses;
  for (size_t j = 0; j + 1 < cuts.size(); j++) {
    std::unique_ptr<Session> s(new Session{api, track, cuts[j], cuts[j + 1], {}, nullptr, on_frame});
    for (int i : want)
      if (i + track.skip_front >= s->a && i + track.skip_front < s->b) s->want.push_back(i);
    ses.push_back(std::move(s));
  }
  std::vector<std::exception_ptr> errs(ses.size());
  auto go = [&](size_t j) {
    try {
      ses[j]->run(data);
    } catch (...) {
      errs[j] = std::current_exception();
    }
  };
  std::vector<std::thread> th;
  for (size_t j = 1; j < ses.size(); j++) th.emplace_back(go, j);
  go(0);
  for (auto& t : th) t.join();
  for (auto& e : errs)
    if (e) std::rethrow_exception(e);
  VideoDecodeStats st;
  st.sessions = int(ses.size());
  for (size_t j = 0; j < ses.size(); j++) {
    Session& s = *ses[j];
    if (!s.error.empty()) {
      if (s.client_error) throw VideoError(s.error);
      throw std::runtime_error("NVDEC: " + s.error);
    }
    // (a part decoded to its end must have shown exactly its samples' frames)
    if (j + 1 < ses.size() && !s.done && s.shown != s.b - s.a)
      throw SplitMismatch("part " + std::to_string(j) + " showed " + std::to_string(s.shown) + " frames for " +
                          std::to_string(s.b - s.a) + " samples");
    st.decoded += s.decoded;
    st.setup_ms = std::max(st.setup_ms, s.setup_ms);
    if (j + 1 == ses.size()) st.presented = std::max(0, s.a + s.shown - track.skip_front);
  }
  return st;
}

// The sessions video_decode runs from sample k0: one per NVDEC engine (or `sessions`), each at least
// 2 kMinPartSamples long.
int decode_sessions(const Cuvid& api, const VideoTrack& track, int sessions, int k0) {
  const int parts = sessions > 0 ? sessions : std::min(4, nvdec_engines(api, track.codec));
  return std::max(1, std::min(parts, (track.frames() - k0) / (2 * kMinPartSamples)));
}

}  // namespace

VideoMemory video_memory(const uint8_t* data, size_t n, int out_w, int out_h, int sessions, int first) {
  const VideoTrack track = video_demux(data, n);
  const Cuvid& api = cuvid();
  Q27_CUDA(cudaSetDevice(0));
  // The parser alone (on the CPU, no decoder) up to the first sequence header.
  struct Probe {
    CUVIDEOFORMAT fmt{};
    bool seen = false;
  } pr;
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = codec_of(track.codec);
  pp.ulMaxNumDecodeSurfaces = 1;
  pp.pUserData = &pr;
  pp.pfnSequenceCallback = [](void* u, CUVIDEOFORMAT* f) {
    auto* p = static_cast<Probe*>(u);
    if (!p->seen) p->fmt = *f, p->seen = true;
    return int(f->min_num_decode_surfaces);
  };
  pp.pfnDecodePicture = [](void*, CUVIDPICPARAMS*) { return 1; };
  pp.pfnDisplayPicture = [](void*, CUVIDPARSERDISPINFO*) { return 1; };
  CUvideoparser parser = nullptr;
  if (api.CreateVideoParser(&parser, &pp) != CUDA_SUCCESS) throw std::runtime_error("cuvidCreateVideoParser failed");
  std::vector<uint8_t> buf;
  for (int k = 0; k < std::min(track.frames(), 8) && !pr.seen; k++) {
    const VideoSample& smp = track.samples[k];
    const uint8_t* p = data + smp.offset;
    buf.clear();
    if (k == 0) buf.insert(buf.end(), track.prefix.begin(), track.prefix.end());
    if (track.nal_length) {
      for (size_t at = 0; at + track.nal_length <= smp.size;) {
        size_t len = 0;
        for (int q = 0; q < track.nal_length; q++) len = len << 8 | p[at + q];
        at += track.nal_length;
        if (len > smp.size - at) break;
        static const uint8_t sc[4] = {0, 0, 0, 1};
        buf.insert(buf.end(), sc, sc + 4);
        buf.insert(buf.end(), p + at, p + at + len);
        at += len;
      }
    } else {
      buf.insert(buf.end(), p, p + smp.size);
    }
    CUVIDSOURCEDATAPACKET pk{};
    pk.payload_size = tcu_ulong(buf.size());
    pk.payload = buf.data();
    if (api.ParseVideoData(parser, &pk) != CUDA_SUCCESS) break;
  }
  api.DestroyVideoParser(parser);
  if (!pr.seen) throw VideoError("the video's first pictures have no sequence header NVDEC can read");
  const CUVIDEOFORMAT& f = pr.fmt;
  VideoMemory m;
  m.sessions = decode_sessions(api, track, sessions, start_sample(data, track, first));
  m.surfaces = int(f.min_num_decode_surfaces) + kExtraSurfaces;
  // A surface: luma and half-height chroma, pitch-aligned, 1 or 2 bytes a sample (NV12 / P016).
  const size_t bytes = f.bit_depth_luma_minus8 ? 2 : 1;
  const size_t planes2 = f.chroma_format == cudaVideoChromaFormat_444 ? 6 : f.chroma_format == cudaVideoChromaFormat_422 ? 4 : 3;
  const size_t surface = size_t(round_up(int64_t(f.coded_width), 256)) * round_up(int64_t(f.coded_height), 64) * planes2 / 2 * bytes;
  const int w = f.display_area.right - f.display_area.left, h = f.display_area.bottom - f.display_area.top;
  const size_t ow = out_w ? out_w : w, oh = out_h ? out_h : h;
  // Each session: its decode and output surfaces, the driver's own state, and the resize buffers (one frame of RGB
  // at most each, video_frames).
  const size_t own = kSessionOverhead + size_t(w) * h * 3 + ow * oh * 3 * (track.rotation ? 2 : 1);
  m.per_session = size_t(m.surfaces + 2) * surface + own;
  m.bytes = size_t(m.sessions) * m.per_session + kDecodeOverhead;
  return m;
}

VideoDecodeStats video_decode(const uint8_t* data, size_t n, const VideoTrack& track, const std::vector<int>& want,
                              int sessions, const VideoFrameFn& on_frame) {
  const double t0 = now_ms();
  if (want.empty() || n == 0) return {};
  const Cuvid& api = cuvid();
  const int parts = decode_sessions(api, track, sessions, start_sample(data, track, want.front()));
  VideoDecodeStats st;
  try {
    st = decode_parts(data, track, want, parts, on_frame);
  } catch (const SplitMismatch&) {
    st = decode_parts(data, track, want, 1, on_frame);
  }
  st.total_ms = now_ms() - t0;
  return st;
}

// ---------------------------------------------------------------- color conversion and resizing

namespace {

struct Yuv2Rgb {
  float ry, rv, gu, gv, bu;  // R = ry (Y - y0) + rv (V - c0), G = ry (Y - y0) + gu (U - c0) + gv (V - c0), ...
  float y0, c0;
};

Yuv2Rgb coefficients(int matrix, int full, int bits) {
  float kr = 0.299f, kb = 0.114f;  // BT.601 (5, 6) and unspecified (2), FFmpeg's default
  if (matrix == 1) kr = 0.2126f, kb = 0.0722f;                       // BT.709
  else if (matrix == 9 || matrix == 10) kr = 0.2627f, kb = 0.0593f;  // BT.2020
  else if (matrix == 4) kr = 0.30f, kb = 0.11f;                      // FCC
  else if (matrix == 7) kr = 0.212f, kb = 0.087f;                    // SMPTE 240M
  const float kg = 1.f - kr - kb;
  const float m = float(1 << (bits - 8));
  const float ys = full ? 1.f : 255.f / 219.f, cs = full ? 1.f : 255.f / 224.f;
  Yuv2Rgb c;
  c.ry = ys / m;
  c.rv = 2.f * (1.f - kr) * cs / m;
  c.gu = -2.f * (1.f - kb) * kb / kg * cs / m;
  c.gv = -2.f * (1.f - kr) * kr / kg * cs / m;
  c.bu = 2.f * (1.f - kb) * cs / m;
  c.y0 = full ? 0.f : 16.f * m;
  c.c0 = 128.f * m;
  return c;
}

struct Src {
  const uint8_t *luma, *chroma, *chroma_v;  // (chroma_v: 4:4:4's V plane; the others interleave U and V)
  size_t pitch;
  int w, h, shift;  // shift: 16-bit samples down to their bits
  bool wide;        // P016 / P216 / YUV444_16Bit
  int cx, cy;       // chroma subsampling: 4:2:0 1, 1; 4:2:2 1, 0; 4:4:4 0, 0
  Yuv2Rgb c;
};

Src src_of(const VideoFrame& f) {
  const int cf = f.chroma_format;
  const uint8_t* v = cf == 2 ? f.chroma + f.pitch * size_t((f.height + 1) & ~1) : nullptr;
  return {f.luma, f.chroma, v, f.pitch, f.width, f.height, 16 - f.bits, f.bits > 8, cf < 2 ? 1 : 0, cf == 0 ? 1 : 0,
          coefficients(f.matrix, f.full_range, f.bits)};
}

__device__ __forceinline__ float luma_at(const Src& s, int x, int y) {
  return s.wide ? float(reinterpret_cast<const uint16_t*>(s.luma + y * s.pitch)[x] >> s.shift) : float(s.luma[y * s.pitch + x]);
}
__device__ __forceinline__ void chroma_at(const Src& s, int x, int y, float& u, float& v) {
  if (s.chroma_v) {  // (4:4:4, planar)
    if (s.wide) {
      u = float(reinterpret_cast<const uint16_t*>(s.chroma + y * s.pitch)[x] >> s.shift);
      v = float(reinterpret_cast<const uint16_t*>(s.chroma_v + y * s.pitch)[x] >> s.shift);
    } else {
      u = s.chroma[y * s.pitch + x], v = s.chroma_v[y * s.pitch + x];
    }
    return;
  }
  if (s.wide) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(s.chroma + y * s.pitch) + 2 * x;
    u = float(p[0] >> s.shift), v = float(p[1] >> s.shift);
  } else {
    const uint8_t* p = s.chroma + y * s.pitch + 2 * x;
    u = p[0], v = p[1];
  }
}

// RGB of pixel (x, y), its chroma the sample of its 2 x 2 block (4:2:2: 2 x 1): what FFmpeg's default conversion of 4:2:0 to
// RGB does (swscale), so what the reference pipelines (transformers' and qwen-vl-utils' readers) give the model.
// (Interpolating the chroma, the more faithful picture, moved colorful frames by 5 to 7 steps on average.)
__device__ __forceinline__ void rgb_at(const Src& s, int x, int y, uint8_t out[3]) {
  float u, v;
  chroma_at(s, x >> s.cx, y >> s.cy, u, v);
  const Yuv2Rgb& c = s.c;
  const float yy = (luma_at(s, x, y) - c.y0) * c.ry, uu = u - c.c0, vv = v - c.c0;
  out[0] = uint8_t(fminf(fmaxf(rintf(yy + c.rv * vv), 0.f), 255.f));
  out[1] = uint8_t(fminf(fmaxf(rintf(yy + c.gu * uu + c.gv * vv), 0.f), 255.f));
  out[2] = uint8_t(fminf(fmaxf(rintf(yy + c.bu * uu), 0.f), 255.f));
}

// RGB [h][w][3] turned k quarter turns counter-clockwise (numpy's rot90(src, k)) into dst ([w][h][3] for odd k).
__global__ void turn_kernel(const uint8_t* src, int w, int h, int k, uint8_t* dst) {
  const int dw = k % 2 ? h : w, dh = k % 2 ? w : h;
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= dw) return;
  int sx, sy;
  if (k == 1) sx = w - 1 - y, sy = x;
  else if (k == 2) sx = w - 1 - x, sy = h - 1 - y;
  else sx = y, sy = h - 1 - x;
  const uint8_t* a = src + (size_t(sy) * w + sx) * 3;
  uint8_t* o = dst + (size_t(y) * dw + x) * 3;
  o[0] = a[0], o[1] = a[1], o[2] = a[2];
}

__global__ void rgb_kernel(Src s, uint8_t* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x < s.w) rgb_at(s, x, y, out + (int64_t(y) * s.w + x) * 3);
}

// PIL's resampling (libImaging/Resample.c, 8 bits per channel): per output pixel the taps bounds[2 i] ..
// + bounds[2 i + 1] with fixed-point weights k[i * ksize ..] of 22 fractional bits, the sum rounded and clamped.
constexpr int kPrecision = 22;

__device__ __forceinline__ uint8_t clip8(int v) {
  v >>= kPrecision;
  return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

// Horizontal pass from the frame (converted to RGB on the fly) for source rows y0 .. y0 + rows - 1 into
// tmp [rows][ow][3]; with ksize 0, the conversion alone (ow = the frame's width).
__global__ void hpass_kernel(Src s, int y0, int ow, const int* bounds, const int* kk, int ksize, uint8_t* tmp) {
  const int xx = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (xx >= ow) return;
  uint8_t* o = tmp + (int64_t(r) * ow + xx) * 3;
  if (ksize == 0) {
    rgb_at(s, xx, y0 + r, o);
    return;
  }
  const int xmin = bounds[2 * xx], xmax = bounds[2 * xx + 1];
  const int* k = kk + int64_t(xx) * ksize;
  int a0 = 1 << (kPrecision - 1), a1 = a0, a2 = a0;
  for (int x = 0; x < xmax; x++) {
    uint8_t p[3];
    rgb_at(s, xmin + x, y0 + r, p);
    a0 += p[0] * k[x], a1 += p[1] * k[x], a2 += p[2] * k[x];
  }
  o[0] = clip8(a0), o[1] = clip8(a1), o[2] = clip8(a2);
}

// Vertical pass: tmp [..][ow][3] -> out [oh][ow][3].
__global__ void vpass_kernel(const uint8_t* tmp, int ow, int oh, const int* bounds, const int* kk, int ksize, uint8_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, yy = blockIdx.y;
  if (i >= ow * 3) return;
  const int ymin = bounds[2 * yy], ymax = bounds[2 * yy + 1];
  const int* k = kk + int64_t(yy) * ksize;
  int acc = 1 << (kPrecision - 1);
  for (int y = 0; y < ymax; y++) acc += tmp[(int64_t(ymin + y) * ow) * 3 + i] * k[y];
  out[int64_t(yy) * ow * 3 + i] = clip8(acc);
}

double bicubic(double x) {
  constexpr double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

// PIL's precompute_coeffs + normalize_coeffs_8bpc for in_size -> out_size: bounds [out][2], weights [out][ksize].
int coefficients(int in_size, int out_size, std::vector<int>& bounds, std::vector<int>& kk) {
  double filterscale = double(float(in_size) - 0.f) / out_size;
  const double scale = filterscale;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = 2.0 * filterscale;
  const int ksize = int(std::ceil(support)) * 2 + 1;
  std::vector<double> pre(size_t(out_size) * ksize, 0.0);
  bounds.assign(size_t(out_size) * 2, 0);
  const double inv = 1.0 / filterscale;
  for (int xx = 0; xx < out_size; xx++) {
    const double center = 0.0 + (xx + 0.5) * scale;
    double ww = 0.0;
    int xmin = int(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &pre[size_t(xx) * ksize];
    for (int x = 0; x < xmax; x++) {
      const double w = bicubic((x + xmin - center + 0.5) * inv);
      k[x] = w;
      ww += w;
    }
    if (ww != 0.0)
      for (int x = 0; x < xmax; x++) k[x] /= ww;
    bounds[2 * xx] = xmin, bounds[2 * xx + 1] = xmax;
  }
  kk.resize(pre.size());
  for (size_t i = 0; i < pre.size(); i++)
    kk[i] = pre[i] < 0 ? int(-0.5 + pre[i] * (1 << kPrecision)) : int(0.5 + pre[i] * (1 << kPrecision));
  return ksize;
}

template <typename T>
T* upload(const std::vector<T>& v, cudaStream_t s) {
  T* d = nullptr;
  Q27_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d), std::max<size_t>(1, v.size()) * sizeof(T), s));
  Q27_CUDA(cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
  Q27_CUDA(cudaStreamSynchronize(s));  // (v is pageable and goes away; the tables serve other streams)
  return d;
}

// The resize of w x h frames to ow x oh, as PIL's ImagingResampleInner: a horizontal pass (if the width
// changes) over the rows the vertical pass reads, then a vertical pass (if the height changes).
struct Resize {
  int w, h, ow, oh;
  bool need_h, need_v;
  int ksize_h = 0, ksize_v = 0, y_first = 0, rows = 0;
  int *bh = nullptr, *kh = nullptr, *bv = nullptr, *kv = nullptr;
  cudaStream_t xs;  // (allocations)

  Resize(int w_, int h_, int ow_, int oh_, cudaStream_t s)
      : w(w_), h(h_), ow(ow_), oh(oh_), need_h(ow_ != w_), need_v(oh_ != h_), xs(s) {
    std::vector<int> b, k;
    ksize_h = coefficients(w, ow, b, k);
    bh = upload(b, xs), kh = upload(k, xs);
    ksize_v = coefficients(h, oh, b, k);
    y_first = b[0];
    const int y_last = b[size_t(oh) * 2 - 2] + b[size_t(oh) * 2 - 1];
    rows = h;
    if (need_h) {  // (the horizontal pass makes only the rows the vertical one reads)
      for (int i = 0; i < oh; i++) b[2 * i] -= y_first;
      rows = y_last - y_first;
    } else {
      y_first = 0;
    }
    bv = upload(b, xs), kv = upload(k, xs);
    if (!need_h) ksize_h = 0;
  }
  ~Resize() {
    for (int* p : {bh, kh, bv, kv}) cudaFreeAsync(p, xs);
  }
  size_t tmp_bytes() const { return size_t(rows) * (need_h ? ow : w) * 3; }

  // f -> out [oh][ow][3] (device) through tmp (tmp_bytes) on s.
  void run(const VideoFrame& f, uint8_t* tmp, uint8_t* out, cudaStream_t s) const {
    const Src src = src_of(f);
    const int tw = need_h ? ow : w;
    hpass_kernel<<<dim3((tw + 127) / 128, rows), 128, 0, s>>>(src, y_first, tw, bh, kh, ksize_h, need_v ? tmp : out);
    if (need_v) vpass_kernel<<<dim3((ow * 3 + 127) / 128, oh), 128, 0, s>>>(tmp, ow, oh, bv, kv, ksize_v, out);
    Q27_CUDA(cudaGetLastError());
  }
};

}  // namespace

void video_frame_rgb(const VideoFrame& f, uint8_t* out, cudaStream_t s) {
  rgb_kernel<<<dim3((f.width + 127) / 128, f.height), 128, 0, s>>>(src_of(f), out);
  Q27_CUDA(cudaGetLastError());
}

VideoDecodeStats video_frames(const uint8_t* data, size_t n, const std::vector<int>& want, int out_w, int out_h,
                              uint8_t* out, int sessions) {
  const VideoTrack track = video_demux(data, n);
  Q27_CUDA(cudaSetDevice(0));
  cudaStream_t xs;  // (the allocations, stream-ordered: no device-wide synchronization)
  Q27_CUDA(cudaStreamCreateWithFlags(&xs, cudaStreamNonBlocking));
  std::unique_ptr<Resize> rs;
  std::mutex mu;
  struct Bufs {
    uint8_t *tmp = nullptr, *dev = nullptr, *turned = nullptr;
  };
  const int k_turn = track.rotation;
  int rw = 0, rh = 0;  // (the resize's output: the upright size turned back for odd k_turn)
  std::deque<std::pair<cudaStream_t, Bufs>> bufs;  // (per session's stream; a deque keeps the others in place)
  int fw = 0, fh = 0;
  size_t frame = 0;
  auto cleanup = [&] {  // (every session's work is done: they synchronize each frame)
    for (auto& [s, b] : bufs) {
      cudaFreeAsync(b.tmp, xs), cudaFreeAsync(b.dev, xs);
      if (b.turned) cudaFreeAsync(b.turned, xs);
    }
    rs.reset();
    cudaStreamSynchronize(xs);
    cudaStreamDestroy(xs);
  };
  try {
    VideoDecodeStats st = video_decode(data, n, track, want, sessions, [&](int i, const VideoFrame& f, cudaStream_t s) {
      Bufs* b = nullptr;
      {
        std::lock_guard<std::mutex> lk(mu);
        if (!rs) {
          fw = f.width, fh = f.height;
          const int uw = k_turn % 2 ? fh : fw, uh = k_turn % 2 ? fw : fh;  // (upright)
          const int ow = out_w ? out_w : uw, oh = out_h ? out_h : uh;
          rw = k_turn % 2 ? oh : ow, rh = k_turn % 2 ? ow : oh;
          rs = std::make_unique<Resize>(fw, fh, rw, rh, xs);
          frame = size_t(ow) * oh * 3;
        }
        if (f.width != fw || f.height != fh) throw VideoError("the video's frames change size");
        for (auto& [bs, bb] : bufs)
          if (bs == s) b = &bb;
        if (!b) {
          bufs.push_back({s, {}});
          b = &bufs.back().second;
          Q27_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&b->tmp), std::max<size_t>(1, rs->tmp_bytes()), xs));
          Q27_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&b->dev), frame, xs));
          if (k_turn) Q27_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&b->turned), frame, xs));
          Q27_CUDA(cudaStreamSynchronize(xs));
        }
      }
      const size_t k = size_t(std::lower_bound(want.begin(), want.end(), i) - want.begin());
      rs->run(f, b->tmp, b->dev, s);
      const uint8_t* res = b->dev;
      if (k_turn) {
        const int dw = k_turn % 2 ? rh : rw;
        turn_kernel<<<dim3((dw + 127) / 128, k_turn % 2 ? rw : rh), 128, 0, s>>>(b->dev, rw, rh, k_turn, b->turned);
        Q27_CUDA(cudaGetLastError());
        res = b->turned;
      }
      Q27_CUDA(cudaMemcpyAsync(out + k * frame, res, frame, cudaMemcpyDeviceToHost, s));
    });
    cleanup();
    return st;
  } catch (...) {
    cleanup();
    throw;
  }
}

}  // namespace q27
