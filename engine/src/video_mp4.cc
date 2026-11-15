// MP4 / MOV (ISO/IEC 14496-12) demuxer for video input: the first video track's samples (the sample table,
// or the movie fragments of a fragmented file), its codec configuration as the parameter sets to send
// first, the frame rate libavformat reports (samples / summed sample durations) and the frames an edit list
// leaves out at the start. Every count and offset is checked against the box and the file.
#include <algorithm>
#include <cstring>
#include <limits>

#include "video.h"

namespace q27 {
namespace {

constexpr uint32_t fourcc(const char (&s)[5]) {
  return uint32_t(uint8_t(s[0])) << 24 | uint32_t(uint8_t(s[1])) << 16 | uint32_t(uint8_t(s[2])) << 8 | uint8_t(s[3]);
}

constexpr size_t kMaxSamples = 4u << 20;  // (about 39 hours at 30 fps)

[[noreturn]] void bad(const std::string& what) { throw VideoError("MP4: " + what); }

// A bounds-checked big-endian reader over [at, end) of the file.
struct Cur {
  const uint8_t* d;
  size_t at, end;
  void need(size_t k) const {
    if (k > end - at) bad("truncated box");
  }
  uint8_t u8() { need(1); return d[at++]; }
  uint16_t u16() { need(2); uint16_t v = uint16_t(d[at] << 8 | d[at + 1]); at += 2; return v; }
  uint32_t u32() {
    need(4);
    uint32_t v = uint32_t(d[at]) << 24 | uint32_t(d[at + 1]) << 16 | uint32_t(d[at + 2]) << 8 | d[at + 3];
    at += 4;
    return v;
  }
  uint64_t u64() { uint64_t hi = u32(); return hi << 32 | u32(); }
  void skip(size_t k) { need(k); at += k; }
  size_t left() const { return end - at; }
};

struct Box {
  uint32_t type = 0;
  size_t start = 0, body = 0, end = 0;  // the box, its payload, its end
};

// The boxes in [at, end), one at a time.
bool next_box(const uint8_t* d, size_t& at, size_t end, Box& b) {
  if (end - at < 8) return false;
  Cur c{d, at, end};
  uint64_t size = c.u32();
  b.type = c.u32();
  if (size == 1) size = c.u64();
  else if (size == 0) size = end - at;
  if (size < c.at - at || size > end - at) bad("box size out of range");
  b.start = at, b.body = c.at, b.end = at + size_t(size);
  at = b.end;
  return true;
}

template <typename F>
void each_box(const uint8_t* d, size_t at, size_t end, F&& f) {
  Box b;
  while (next_box(d, at, end, b)) f(b);
}

// A full box's body after its version and flags.
Cur full(const uint8_t* d, const Box& b, int& version, uint32_t& flags) {
  Cur c{d, b.body, b.end};
  uint32_t vf = c.u32();
  version = int(vf >> 24), flags = vf & 0xffffff;
  return c;
}

struct Trex {
  uint32_t duration = 0, size = 0, flags = 0;
};

struct Track {
  uint32_t id = 0, timescale = 0, handler = 0, entry = 0;
  int rotation = 0;
  int width = 0, height = 0;
  std::vector<uint8_t> config;  // avcC / hvcC / av1C body
  // sample table
  std::vector<uint32_t> sizes, durations;
  std::vector<int64_t> cto;   // composition offsets (empty: none)
  std::vector<uint64_t> chunk_offsets;
  std::vector<uint32_t> stsc;  // (first chunk, samples per chunk) pairs
  std::vector<uint32_t> sync;  // 1-based sample numbers (empty and !has_stss: all)
  bool has_stss = false;
  uint32_t constant_size = 0;
  int64_t edit_media_time = 0;  // first non-empty edit's media time
  // built samples (sample table or fragments)
  std::vector<VideoSample> samples;
  std::vector<uint32_t> frag_durations;
  std::vector<int64_t> frag_cto;
};

size_t checked_count(uint32_t count, size_t entry_bytes, const Cur& c) {
  if (count > kMaxSamples || size_t(count) * entry_bytes > c.left()) bad("table larger than its box");
  return count;
}

void parse_stsd(const uint8_t* d, const Box& b, Track& t) {
  int v;
  uint32_t fl;
  Cur c = full(d, b, v, fl);
  if (c.u32() < 1) bad("no sample description");
  Box e;
  size_t at = c.at;
  if (!next_box(d, at, b.end, e)) bad("no sample description");
  t.entry = e.type;
  // (only the video sample entries decoded here have the visual layout; an audio track's mp4a, a text track's tx3g
  // and the like are not read)
  if (e.type != fourcc("avc1") && e.type != fourcc("avc3") && e.type != fourcc("hvc1") && e.type != fourcc("hev1") &&
      e.type != fourcc("av01") && e.type != fourcc("vp09"))
    return;
  Cur s{d, e.body, e.end};
  s.skip(6 + 2 + 16);  // reserved, data reference index, pre-defined / reserved
  t.width = s.u16(), t.height = s.u16();
  s.skip(4 + 4 + 4 + 2 + 32 + 2 + 2);  // resolutions, reserved, frame count, compressor name, depth, pre-defined
  each_box(d, s.at, e.end, [&](const Box& k) {
    if (k.type == fourcc("avcC") || k.type == fourcc("hvcC") || k.type == fourcc("av1C"))
      t.config.assign(d + k.body, d + k.end);
  });
}

void parse_stbl(const uint8_t* d, const Box& stbl, Track& t) {
  each_box(d, stbl.body, stbl.end, [&](const Box& b) {
    int v;
    uint32_t fl;
    if (b.type == fourcc("stsd")) {
      parse_stsd(d, b, t);
    } else if (b.type == fourcc("stts")) {
      Cur c = full(d, b, v, fl);
      size_t n = checked_count(c.u32(), 8, c);
      for (size_t i = 0; i < n; i++) {
        uint32_t count = c.u32(), delta = c.u32();
        if (t.durations.size() + count > kMaxSamples) bad("too many samples");
        t.durations.insert(t.durations.end(), count, delta);
      }
    } else if (b.type == fourcc("ctts")) {
      Cur c = full(d, b, v, fl);
      size_t n = checked_count(c.u32(), 8, c);
      for (size_t i = 0; i < n; i++) {
        uint32_t count = c.u32(), off = c.u32();
        if (t.cto.size() + count > kMaxSamples) bad("too many samples");
        t.cto.insert(t.cto.end(), count, v == 0 ? int64_t(off) : int64_t(int32_t(off)));
      }
    } else if (b.type == fourcc("stsz")) {
      Cur c = full(d, b, v, fl);
      t.constant_size = c.u32();
      uint32_t count = c.u32();
      if (t.constant_size) {
        if (count > kMaxSamples) bad("too many samples");
        t.sizes.assign(count, t.constant_size);
      } else {
        size_t n = checked_count(count, 4, c);
        t.sizes.resize(n);
        for (auto& s : t.sizes) s = c.u32();
      }
    } else if (b.type == fourcc("stz2")) {
      Cur c = full(d, b, v, fl);
      c.skip(3);
      int bits = c.u8();
      uint32_t count = c.u32();
      if (bits != 4 && bits != 8 && bits != 16) bad("bad stz2 field size");
      if (count > kMaxSamples || (size_t(count) * bits + 7) / 8 > c.left()) bad("table larger than its box");
      t.sizes.resize(count);
      for (uint32_t i = 0; i < count; i++) {
        if (bits == 16) t.sizes[i] = c.u16();
        else if (bits == 8) t.sizes[i] = c.u8();
        else t.sizes[i] = (i % 2 == 0) ? d[c.at] >> 4 : (d[c.at++] & 15);
      }
    } else if (b.type == fourcc("stsc")) {
      Cur c = full(d, b, v, fl);
      size_t n = checked_count(c.u32(), 12, c);
      for (size_t i = 0; i < n; i++) {
        uint32_t first = c.u32(), per = c.u32();
        c.u32();  // (sample description index)
        t.stsc.push_back(first), t.stsc.push_back(per);
      }
    } else if (b.type == fourcc("stco") || b.type == fourcc("co64")) {
      Cur c = full(d, b, v, fl);
      const bool wide = b.type == fourcc("co64");
      size_t n = checked_count(c.u32(), wide ? 8 : 4, c);
      t.chunk_offsets.resize(n);
      for (auto& o : t.chunk_offsets) o = wide ? c.u64() : c.u32();
    } else if (b.type == fourcc("stss")) {
      Cur c = full(d, b, v, fl);
      size_t n = checked_count(c.u32(), 4, c);
      t.has_stss = true;
      t.sync.resize(n);
      for (auto& s : t.sync) s = c.u32();
    }
  });
}

void parse_trak(const uint8_t* d, const Box& trak, std::vector<Track>& out) {
  Track t;
  each_box(d, trak.body, trak.end, [&](const Box& b) {
    int v;
    uint32_t fl;
    if (b.type == fourcc("tkhd")) {
      Cur c = full(d, b, v, fl);
      c.skip(v == 1 ? 16 : 8);
      t.id = c.u32();
      // The display matrix's rotation (a b / c d in 16.16; FFmpeg's av_display_rotation_get): quarter turns only.
      c.skip(4 + (v == 1 ? 8 : 4) + 8 + 2 + 2 + 2 + 2);  // reserved, duration, reserved, layer, group, volume
      const int32_t a = int32_t(c.u32()), bb = int32_t(c.u32());
      c.u32();
      const int32_t cc = int32_t(c.u32()), dd = int32_t(c.u32());
      const int32_t one = 1 << 16;
      if (a == one && dd == one && !bb && !cc) t.rotation = 0;
      else if (!a && !dd && bb == -one && cc == one) t.rotation = 1;
      else if (a == -one && dd == -one && !bb && !cc) t.rotation = 2;
      else if (!a && !dd && bb == one && cc == -one) t.rotation = 3;
    } else if (b.type == fourcc("edts")) {
      each_box(d, b.body, b.end, [&](const Box& e) {
        if (e.type != fourcc("elst")) return;
        Cur c = full(d, e, v, fl);
        size_t n = checked_count(c.u32(), v == 1 ? 20 : 12, c);
        for (size_t i = 0; i < n; i++) {
          if (v == 1) c.u64();
          else c.u32();  // (segment duration)
          int64_t mt = v == 1 ? int64_t(c.u64()) : int64_t(int32_t(c.u32()));
          c.u32();  // (media rate)
          if (mt >= 0) {  // (-1: an empty edit, a delay)
            t.edit_media_time = mt;
            break;
          }
        }
      });
    } else if (b.type == fourcc("mdia")) {
      each_box(d, b.body, b.end, [&](const Box& m) {
        if (m.type == fourcc("mdhd")) {
          Cur c = full(d, m, v, fl);
          c.skip(v == 1 ? 16 : 8);
          t.timescale = c.u32();
        } else if (m.type == fourcc("hdlr")) {
          Cur c = full(d, m, v, fl);
          c.skip(4);
          t.handler = c.u32();
        } else if (m.type == fourcc("minf")) {
          each_box(d, m.body, m.end, [&](const Box& s) {
            if (s.type == fourcc("stbl")) parse_stbl(d, s, t);
          });
        }
      });
    }
  });
  out.push_back(std::move(t));
}

// Samples of the sample table, checked against the file.
void build_samples(Track& t, size_t file_size) {
  const size_t n = t.sizes.size();
  if (t.durations.size() < n) t.durations.resize(n, t.durations.empty() ? 1 : t.durations.back());
  t.samples.resize(n);
  size_t s = 0;
  for (size_t e = 0; e < t.stsc.size() / 2 && s < n; e++) {
    const uint64_t first = t.stsc[2 * e], per = t.stsc[2 * e + 1];
    const uint64_t last = e + 1 < t.stsc.size() / 2 ? t.stsc[2 * e + 2] : t.chunk_offsets.size() + 1;
    if (first < 1 || last < first || last > t.chunk_offsets.size() + 1) bad("bad sample-to-chunk table");
    for (uint64_t ch = first; ch < last && s < n; ch++) {
      uint64_t off = t.chunk_offsets[ch - 1];
      for (uint64_t k = 0; k < per && s < n; k++, s++) {
        if (off > file_size || t.sizes[s] > file_size - off) bad("sample outside the file");
        t.samples[s] = {off, t.sizes[s], !t.has_stss};
        off += t.sizes[s];
      }
    }
  }
  if (s != n) bad("the chunks hold fewer samples than the size table");
  for (uint32_t k : t.sync)
    if (k >= 1 && k <= n) t.samples[k - 1].sync = true;
}

// One movie fragment's runs of the tracks (trex: defaults by track id).
void parse_moof(const uint8_t* d, const Box& moof, size_t file_size, std::vector<Track>& tracks,
                const std::vector<std::pair<uint32_t, Trex>>& trex) {
  each_box(d, moof.body, moof.end, [&](const Box& traf) {
    if (traf.type != fourcc("traf")) return;
    Track* t = nullptr;
    Trex def;
    uint64_t base = moof.start;
    uint64_t cursor = 0;
    bool have_cursor = false;
    each_box(d, traf.body, traf.end, [&](const Box& b) {
      int v;
      uint32_t fl;
      if (b.type == fourcc("tfhd")) {
        Cur c = full(d, b, v, fl);
        uint32_t id = c.u32();
        for (auto& x : tracks)
          if (x.id == id) t = &x;
        for (auto& [tid, tx] : trex)
          if (tid == id) def = tx;
        if (fl & 0x1) base = c.u64();
        if (fl & 0x2) c.u32();
        if (fl & 0x8) def.duration = c.u32();
        if (fl & 0x10) def.size = c.u32();
        if (fl & 0x20) def.flags = c.u32();
      } else if (b.type == fourcc("trun") && t) {
        Cur c = full(d, b, v, fl);
        uint32_t count = c.u32();
        const size_t per = 4 * (!!(fl & 0x100) + !!(fl & 0x200) + !!(fl & 0x400) + !!(fl & 0x800));
        uint64_t at = have_cursor ? cursor : base;
        if (fl & 0x1) at = base + int64_t(int32_t(c.u32()));
        uint32_t first_flags = (fl & 0x4) ? c.u32() : 0;
        checked_count(count, per, c);
        if (t->samples.size() + count > kMaxSamples) bad("too many samples");
        for (uint32_t i = 0; i < count; i++) {
          uint32_t dur = (fl & 0x100) ? c.u32() : def.duration;
          uint32_t size = (fl & 0x200) ? c.u32() : def.size;
          uint32_t sflags = (fl & 0x400) ? c.u32() : (i == 0 && (fl & 0x4)) ? first_flags : def.flags;
          int64_t cto = 0;
          if (fl & 0x800) cto = v == 0 ? int64_t(c.u32()) : int64_t(int32_t(c.u32()));
          if (at > file_size || size > file_size - at) bad("sample outside the file");
          t->samples.push_back({at, size, !(sflags & 0x10000)});
          t->frag_durations.push_back(dur);
          t->frag_cto.push_back(cto);
          at += size;
        }
        cursor = at, have_cursor = true;
      }
    });
  });
}

void annexb(std::vector<uint8_t>& out, const uint8_t* p, size_t n) {
  static const uint8_t sc[4] = {0, 0, 0, 1};
  out.insert(out.end(), sc, sc + 4);
  out.insert(out.end(), p, p + n);
}

}  // namespace

const char* video_codec_name(VideoCodec c) {
  switch (c) {
    case VideoCodec::kH264: return "H.264";
    case VideoCodec::kHevc: return "HEVC";
    case VideoCodec::kAv1: return "AV1";
    case VideoCodec::kVp9: return "VP9";
    default: return "none";
  }
}

void video_configure(const std::vector<uint8_t>& config, VideoTrack& v) {
  Cur c{config.data(), 0, config.size()};
  if (v.codec == VideoCodec::kH264) {
    if (config.empty()) return;  // (avc3: the parameter sets are in the stream)
    c.skip(4);
    v.nal_length = (c.u8() & 3) + 1;
    for (int kind = 0; kind < 2; kind++) {
      int count = kind == 0 ? (c.u8() & 31) : c.u8();
      for (int i = 0; i < count; i++) {
        uint16_t len = c.u16();
        c.need(len);
        annexb(v.prefix, config.data() + c.at, len);
        c.skip(len);
      }
    }
  } else if (v.codec == VideoCodec::kHevc) {
    if (config.empty()) throw VideoError("HEVC without its configuration (hvcC)");
    c.skip(21);
    v.nal_length = (c.u8() & 3) + 1;
    int arrays = c.u8();
    for (int a = 0; a < arrays; a++) {
      c.u8();
      int count = c.u16();
      for (int i = 0; i < count; i++) {
        uint16_t len = c.u16();
        c.need(len);
        annexb(v.prefix, config.data() + c.at, len);
        c.skip(len);
      }
    }
  } else if (v.codec == VideoCodec::kAv1) {
    if (config.size() > 4) v.prefix.assign(config.begin() + 4, config.end());  // (configOBUs)
  }
  if (v.nal_length == 3) throw VideoError("3-byte NAL lengths");
  if (v.codec == VideoCodec::kH264 && config.empty()) v.nal_length = 4;
}

VideoTrack mp4_demux(const uint8_t* d, size_t n) {
  std::vector<Track> tracks;
  std::vector<std::pair<uint32_t, Trex>> trex;
  std::vector<Box> moofs;
  bool moov = false;
  each_box(d, 0, n, [&](const Box& b) {
    if (b.type == fourcc("moov")) {
      moov = true;
      each_box(d, b.body, b.end, [&](const Box& m) {
        if (m.type == fourcc("trak")) parse_trak(d, m, tracks);
        if (m.type == fourcc("mvex"))
          each_box(d, m.body, m.end, [&](const Box& x) {
            if (x.type != fourcc("trex")) return;
            int v;
            uint32_t fl;
            Cur c = full(d, x, v, fl);
            uint32_t id = c.u32();
            c.u32();
            Trex t;
            t.duration = c.u32(), t.size = c.u32(), t.flags = c.u32();
            trex.push_back({id, t});
          });
      });
    } else if (b.type == fourcc("moof")) {
      moofs.push_back(b);
    }
  });
  if (!moov) bad("no movie box (moov)");
  Track* t = nullptr;  // (tracks does not change size from here: parse_moof appends to a track's samples)
  VideoCodec codec = VideoCodec::kNone;
  uint32_t seen = 0;
  for (auto& x : tracks) {
    if (x.handler != fourcc("vide")) continue;
    seen = seen ? seen : x.entry;
    VideoCodec c = VideoCodec::kNone;
    if (x.entry == fourcc("avc1") || x.entry == fourcc("avc3")) c = VideoCodec::kH264;
    else if (x.entry == fourcc("hvc1") || x.entry == fourcc("hev1")) c = VideoCodec::kHevc;
    else if (x.entry == fourcc("av01")) c = VideoCodec::kAv1;
    else if (x.entry == fourcc("vp09")) c = VideoCodec::kVp9;
    if (c != VideoCodec::kNone) {
      t = &x, codec = c;
      break;
    }
  }
  if (!t) {
    if (!seen) bad("no video track");
    char s[5] = {char(seen >> 24), char(seen >> 16), char(seen >> 8), char(seen), 0};
    for (int i = 0; i < 4; i++) s[i] = (s[i] >= 32 && s[i] < 127) ? s[i] : '?';
    bad(std::string("unsupported video codec '") + s + "' (H.264, HEVC, AV1 and VP9 are)");
  }
  build_samples(*t, n);
  for (const Box& m : moofs) parse_moof(d, m, n, tracks, trex);

  VideoTrack v;
  v.codec = codec;
  v.width = t->width, v.height = t->height;
  v.rotation = t->rotation;
  video_configure(t->config, v);
  // Durations and composition offsets in decode order: the sample table's, then the fragments'.
  std::vector<uint32_t> dur(t->durations.begin(), t->durations.begin() + std::min(t->durations.size(), t->sizes.size()));
  dur.insert(dur.end(), t->frag_durations.begin(), t->frag_durations.end());
  std::vector<int64_t> cto(t->sizes.size(), 0);
  for (size_t i = 0; i < std::min(cto.size(), t->cto.size()); i++) cto[i] = t->cto[i];
  cto.insert(cto.end(), t->frag_cto.begin(), t->frag_cto.end());
  v.samples = std::move(t->samples);
  if (v.samples.empty()) bad("the video track has no samples");
  uint64_t total = 0;
  for (uint32_t x : dur) total += x;
  if (t->timescale && total) v.fps = double(v.samples.size()) * t->timescale / double(total);
  // Frames the edit list starts after (presentation time before its media time) are not shown.
  if (t->edit_media_time > 0) {
    int64_t dts = 0;
    for (size_t i = 0; i < v.samples.size(); i++) {
      if (dts + cto[i] < t->edit_media_time) v.skip_front++;
      dts += dur[i];
    }
  }
  return v;
}

VideoTrack video_demux(const uint8_t* data, size_t n) {
  if (n >= 4 && data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3) return mkv_demux(data, n);
  if (n < 8) throw VideoError("not a video file");
  return mp4_demux(data, n);
}

}  // namespace q27
