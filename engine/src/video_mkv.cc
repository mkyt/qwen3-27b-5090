// Matroska / WebM demuxer for video input, on libwebm's mkvparser (third_party/libwebm): the first video track's
// frames (SimpleBlocks and BlockGroups, laced ones too) as offsets into the file, its CodecPrivate as the
// parameter sets to send first, and the frame rate libavformat reports (from DefaultDuration; without it, from
// the frames' timestamps). mkvparser checks the elements against the file's length; the frames are checked
// again here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "third_party/libwebm/mkvparser/mkvparser.h"
#include "video.h"

namespace q27 {
namespace {

constexpr size_t kMaxFrames = 4u << 20;  // (as the MP4 demuxer's samples)

[[noreturn]] void bad(const std::string& what) { throw VideoError("Matroska: " + what); }

// The file in memory as mkvparser's reader.
class MemReader : public mkvparser::IMkvReader {
 public:
  MemReader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  int Read(long long pos, long len, unsigned char* buf) override {
    if (pos < 0 || len < 0 || uint64_t(pos) > n_ || uint64_t(len) > n_ - uint64_t(pos)) return -1;
    if (len) std::memcpy(buf, d_ + pos, size_t(len));
    return 0;
  }
  int Length(long long* total, long long* available) override {
    if (total) *total = (long long)n_;
    if (available) *available = (long long)n_;
    return 0;
  }

 private:
  const uint8_t* d_;
  size_t n_;
};

VideoCodec codec_of(const char* id) {
  if (!id) return VideoCodec::kNone;
  if (!std::strcmp(id, "V_MPEG4/ISO/AVC")) return VideoCodec::kH264;
  if (!std::strcmp(id, "V_MPEGH/ISO/HEVC")) return VideoCodec::kHevc;
  if (!std::strcmp(id, "V_AV1")) return VideoCodec::kAv1;
  if (!std::strcmp(id, "V_VP9")) return VideoCodec::kVp9;
  return VideoCodec::kNone;
}

// libavutil's av_reduce: num / den as the nearest fraction with both terms at most max (continued fractions).
double reduce(int64_t num, int64_t den, int64_t max) {
  auto gcd = [](int64_t a, int64_t b) {
    while (b) a %= b, std::swap(a, b);
    return a;
  };
  if (int64_t g = gcd(num, den)) num /= g, den /= g;
  int64_t a0n = 0, a0d = 1, a1n = 1, a1d = 0;
  if (num <= max && den <= max) a1n = num, a1d = den, den = 0;
  while (den) {
    int64_t x = num / den, next = num - den * x;
    const int64_t a2n = x * a1n + a0n, a2d = x * a1d + a0d;
    if (a2n > max || a2d > max) {
      if (a1n) x = (max - a0n) / a1n;
      if (a1d) x = std::min(x, (max - a0d) / a1d);
      if (den * (2 * x * a1d + a0d) > num * a1d) a1n = x * a1n + a0n, a1d = x * a1d + a0d;
      break;
    }
    a0n = a1n, a0d = a1d, a1n = a2n, a1d = a2d, num = den, den = next;
  }
  return a1d ? double(a1n) / double(a1d) : 0;
}

}  // namespace

VideoTrack mkv_demux(const uint8_t* d, size_t n) {
  if (n > size_t(std::numeric_limits<long long>::max())) bad("file too large");
  MemReader reader(d, n);
  long long pos = 0;
  mkvparser::EBMLHeader ebml;
  if (ebml.Parse(&reader, pos) < 0) bad("bad EBML header");
  if (!ebml.m_docType || (std::strcmp(ebml.m_docType, "matroska") && std::strcmp(ebml.m_docType, "webm")))
    bad("not a Matroska or WebM file");
  mkvparser::Segment* seg_raw = nullptr;
  if (mkvparser::Segment::CreateInstance(&reader, pos, seg_raw) != 0 || !seg_raw) bad("no segment");
  std::unique_ptr<mkvparser::Segment> seg(seg_raw);
  if (seg->Load() < 0) bad("damaged or truncated file");

  const mkvparser::Tracks* tracks = seg->GetTracks();
  const mkvparser::VideoTrack* vt = nullptr;
  VideoCodec codec = VideoCodec::kNone;
  std::string seen;
  for (unsigned long i = 0; tracks && i < tracks->GetTracksCount(); i++) {
    const mkvparser::Track* t = tracks->GetTrackByIndex(i);
    if (!t || t->GetType() != mkvparser::Track::kVideo) continue;
    if (seen.empty()) seen = t->GetCodecId() ? t->GetCodecId() : "?";
    if (VideoCodec c = codec_of(t->GetCodecId()); c != VideoCodec::kNone) {
      vt = static_cast<const mkvparser::VideoTrack*>(t), codec = c;
      break;
    }
  }
  if (!vt) {
    if (seen.empty()) bad("no video track");
    for (char& ch : seen) ch = (ch >= 32 && ch < 127) ? ch : '?';
    bad("unsupported video codec '" + seen.substr(0, 40) + "' (H.264, HEVC, AV1 and VP9 are)");
  }
  // (header stripping or zlib would make the frames other bytes than the file's: rare for video)
  if (vt->GetContentEncodingCount()) bad("compressed or encrypted frames (ContentEncoding)");

  VideoTrack v;
  v.codec = codec;
  const long long w = vt->GetWidth(), h = vt->GetHeight();
  if (w <= 0 || h <= 0 || w > 65535 || h > 65535) bad("bad frame size");
  v.width = int(w), v.height = int(h);
  // The rotation FFmpeg writes as ProjectionPoseRoll (degrees counter-clockwise, as its display matrix).
  if (const mkvparser::Projection* pj = vt->GetProjection(); pj && pj->pose_roll != mkvparser::Projection::kValueNotPresent) {
    const long q = std::lround(pj->pose_roll / 90.0);
    if (std::fabs(pj->pose_roll - 90.0 * q) < 1.0) v.rotation = int(((q % 4) + 4) % 4);
  }
  size_t priv_n = 0;
  const unsigned char* priv = vt->GetCodecPrivate(priv_n);
  try {
    video_configure(priv && priv_n ? std::vector<uint8_t>(priv, priv + priv_n) : std::vector<uint8_t>{}, v);
  } catch (const VideoError& e) {
    bad(std::string("the codec configuration (CodecPrivate) is damaged: ") + e.what());
  }

  // The track's frames in file (decode) order, and the span of their timestamps.
  const long number = vt->GetNumber();
  long long t_min = std::numeric_limits<long long>::max(), t_max = std::numeric_limits<long long>::min();
  for (const mkvparser::Cluster* cl = seg->GetFirst(); cl && !cl->EOS(); cl = seg->GetNext(cl)) {
    const mkvparser::BlockEntry* e = nullptr;
    if (cl->GetFirst(e) < 0) bad("damaged cluster");
    while (e && !e->EOS()) {
      const mkvparser::Block* b = e->GetBlock();
      if (b && b->GetTrackNumber() == number) {
        const long long t = b->GetTime(cl);
        t_min = std::min(t_min, t), t_max = std::max(t_max, t);
        for (int k = 0; k < b->GetFrameCount(); k++) {
          const mkvparser::Block::Frame& f = b->GetFrame(k);
          if (f.pos < 0 || f.len < 0 || uint64_t(f.pos) > n || uint64_t(f.len) > n - uint64_t(f.pos))
            bad("frame outside the file");
          if (v.samples.size() >= kMaxFrames) bad("too many frames");
          v.samples.push_back({uint64_t(f.pos), uint32_t(f.len), b->IsKey() && k == 0});
        }
      }
      if (cl->GetNext(e, e) < 0) bad("damaged cluster");
    }
  }
  if (v.samples.empty()) bad("the video track has no frames");
  // libavformat's average frame rate: 1 s / DefaultDuration (ns) as a fraction of terms up to 30000 (30000 / 1001
  // for 33366667). Without it, libavformat's guess from the first packets' millisecond timestamps is off by
  // a few percent (30.3 for 29.97); the spacing of all the frames' timestamps is used instead.
  if (const unsigned long long dd = vt->GetDefaultDuration(); dd > 0 && dd <= uint64_t(INT64_MAX))
    v.fps = reduce(1000000000, int64_t(dd), 30000);
  else if (v.samples.size() > 1 && t_max > t_min) v.fps = double(v.samples.size() - 1) * 1e9 / double(t_max - t_min);
  return v;
}

}  // namespace q27
