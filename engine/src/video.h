// Video input: the container's video track (video_mp4.cc, video_mkv.cc) and its decoding on NVDEC
// (video_nvdec.cu). The demuxers take the whole file in memory and check every offset and count against it:
// the bytes come from the client.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace q27 {

// What the client sent cannot be used (the message is for the client: HTTP 400).
struct VideoError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class VideoCodec { kNone = 0, kH264 = 1, kHevc = 2, kAv1 = 3, kVp9 = 4 };
const char* video_codec_name(VideoCodec c);

struct VideoSample {
  uint64_t offset = 0;  // in the file
  uint32_t size = 0;
  bool sync = false;
};

struct VideoTrack {
  VideoCodec codec = VideoCodec::kNone;
  int width = 0, height = 0;          // as the container states them (the decoder's display area decides)
  int nal_length = 0;                 // H.264 / HEVC: bytes of each NAL unit's length prefix (1, 2 or 4)
  std::vector<uint8_t> prefix;        // sent before the first sample: Annex B parameter sets, AV1 config OBUs
  std::vector<VideoSample> samples;   // in decode order
  double fps = 0;                     // samples / duration (libavformat's average frame rate)
  int skip_front = 0;                 // frames an edit list leaves out at the start (in presentation order)
  int rotation = 0;                   // quarter turns counter-clockwise that show a frame upright (the container's
                                      // display matrix / ProjectionPoseRoll, as FFmpeg applies them; 0 .. 3)
  int display_width() const { return rotation % 2 ? height : width; }
  int display_height() const { return rotation % 2 ? width : height; }
  int frames() const { return int(samples.size()); }
};

// The first video track of an MP4 / MOV (ISO BMFF, fragmented too) or Matroska / WebM file.
VideoTrack video_demux(const uint8_t* data, size_t n);
VideoTrack mp4_demux(const uint8_t* data, size_t n);
VideoTrack mkv_demux(const uint8_t* data, size_t n);

// v.prefix and v.nal_length from the codec configuration record (ISO BMFF avcC / hvcC / av1C, which Matroska's
// CodecPrivate holds as well) for v.codec.
void video_configure(const std::vector<uint8_t>& config, VideoTrack& v);

// A decoded frame on the device: NV12 / NV16 / YUV444 (8-bit) or P016 / P216 / YUV444_16Bit (more bits, in the high
// bits of 16), pitch-linear, luma then the chroma at chroma (interleaved, or 4:4:4's U and V planes); colors per matrix / full_range (H.273 matrix_coefficients).
struct VideoFrame {
  const uint8_t* luma = nullptr;
  const uint8_t* chroma = nullptr;
  size_t pitch = 0;
  int width = 0, height = 0, bits = 8;
  int chroma_format = 0;  // 0: 4:2:0 (NV12 / P016), 1: 4:2:2 (NV16 / P216, chroma rows as tall as luma),
                          // 2: 4:4:4 (planar U then V, each as large as luma)
  int matrix = 2, full_range = 0;
};

// Decodes the track on NVDEC in the engine's CUDA context and calls on_frame(i, frame, stream) for each frame i
// of `want` (presentation order after the edit list; ascending, no repeats), the frame valid on that stream
// until on_frame returns. With sessions > 1 (0: one per NVDEC engine, at most 4) the samples are cut at
// closed key frames (IDR) into that many parts decoded at once, so on_frame runs on several threads, each
// with its own stream. Frames past the stream's end are not called.
struct VideoDecodeStats {
  int presented = 0, decoded = 0, sessions = 0;
  double setup_ms = 0, total_ms = 0;
};
using VideoFrameFn = std::function<void(int, const VideoFrame&, cudaStream_t)>;
VideoDecodeStats video_decode(const uint8_t* data, size_t n, const VideoTrack& track, const std::vector<int>& want,
                              int sessions, const VideoFrameFn& on_frame);

// Device memory a decode of the video takes while it runs (outside the engine's pools): NVDEC's surfaces for the
// stream's first sequence header (the parser alone, on the CPU), the driver's state and video_frames' buffers, for
// the sessions video_decode would run for frames from `first` on (sessions as there). An estimate on the safe side,
// to lend (Model::kv_lend).
struct VideoMemory {
  size_t bytes = 0, per_session = 0;
  int sessions = 0, surfaces = 0;
};
VideoMemory video_memory(const uint8_t* data, size_t n, int out_w, int out_h, int sessions = 0, int first = 0);

// The frame as RGB [height][width][3] uint8 into out (device): BT.601 / 709 / 2020 per its matrix
// (unspecified: BT.601), limited or full range, each pixel taking its 2 x 2 block's chroma (FFmpeg's default).
void video_frame_rgb(const VideoFrame& f, uint8_t* out, cudaStream_t s);

// Frames want[] of a video resized to out_w x out_h (0: their own size) as RGB [h][w][3] uint8 into host
// memory out, one after another: PIL's bicubic resize (Image.resize, the same bytes) of video_frame_rgb. A rotated
// video's frames come upright (out_w x out_h is the upright size): resized to the turned size, then turned.
// The frames past the stream's end stay as they are (stats.presented tells). sessions as video_decode.
VideoDecodeStats video_frames(const uint8_t* data, size_t n, const std::vector<int>& want, int out_w, int out_h,
                              uint8_t* out, int sessions = 0);

}  // namespace q27
