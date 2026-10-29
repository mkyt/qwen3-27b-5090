// Shared helpers: error checking, logging, model dimensions.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace q27 {

[[noreturn]] void fail(const std::string& msg);
// The KV page pool cannot hold the running sequence's next positions (every cached checkpoint already
// evicted). Thrown before anything is launched, so the sequence is unchanged and the call can be
// retried once other sequences gave pages back (the C API returns -2).
struct KvFull : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#if defined(__GNUC__)
void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#else
void log(const char* fmt, ...);
#endif

#define Q27_CUDA(call)                                                                        \
  do {                                                                                        \
    cudaError_t err_ = (call);                                                                \
    if (err_ != cudaSuccess)                                                                  \
      ::q27::fail(std::string(#call " failed: ") + cudaGetErrorString(err_) + " at " __FILE__ \
                  ":" + std::to_string(__LINE__));                                            \
  } while (0)

#define Q27_CHECK(cond, msg)                                         \
  do {                                                               \
    if (!(cond)) ::q27::fail(std::string("check failed: ") + (msg)); \
  } while (0)

// Dimensions of Qwen3.8-27B (text). The runtime is specialized for this model; the loader checks
// config.json against these values.
namespace dims {
constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kLayers = 64;
constexpr int kInter = 17408;     // MLP intermediate
constexpr int kHeads = 24;        // attention query heads
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;    // partial RoPE: the first 64 dims of each head
constexpr float kRopeTheta = 1e7f;
constexpr float kEps = 1e-6f;
constexpr int kKHeads = 16;       // GDN key heads
constexpr int kVHeads = 48;       // GDN value heads
constexpr int kKDim = 128;
constexpr int kVDim = 128;
constexpr int kConvK = 4;
constexpr int kConvDim = 2 * kKHeads * kKDim + kVHeads * kVDim;  // 10240 = q | k | v
constexpr int kAttnEvery = 4;     // layer i is full attention when i % 4 == 3
constexpr bool is_attn(int layer) { return layer % kAttnEvery == kAttnEvery - 1; }
constexpr int kAttnLayers = kLayers / kAttnEvery;  // 16
constexpr int kGdnLayers = kLayers - kAttnLayers;  // 48
constexpr int kImagePad = 248056;  // <|image_pad|> (config.json image_token_id)
}  // namespace dims

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

}  // namespace q27
