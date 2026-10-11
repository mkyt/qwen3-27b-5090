// Vision tower of Qwen3.8-27B (P4-12): one image's patches -> one 5120-dim embedding per merged token.
// Numerics as reference/qwen27b/vision.py: BF16 weights and GEMM operands, fp32 residual stream, LayerNorm
// and softmax in fp32.
#pragma once

#include "kernels.h"

namespace q27 {

namespace vdims {
constexpr int kHidden = 1152;
constexpr int kHeads = 16;
constexpr int kHeadDim = 72;
constexpr int kDepth = 27;
constexpr int kInter = 4304;
constexpr int kOut = 5120;                        // the language model's hidden size
constexpr int kPatchDim = 3 * 2 * 16 * 16;        // channels x temporal patch x 16 x 16
constexpr int kMerge = 2;                         // 2x2 patches per output token
constexpr int kMergedDim = kHidden * kMerge * kMerge;  // 4608
constexpr int kPosSide = 48;                      // learned position table 48 x 48
constexpr int kPatchPixels = 16;
}  // namespace vdims

// Where the weight matrices live (VISION): BF16 on the device, BF16 in pinned host memory (copied to the
// device a stage at a time, overlapped with the previous stage), or e4m3 on the device with one fp32 scale
// per output row (dequantized to BF16 a stage at a time). Biases, LayerNorms and the position table are
// BF16 on the device in every mode.
enum class VisionMode { kDevice, kHost, kFp8 };

struct VisionMat {
  const bf16* w = nullptr;       // [out][in] BF16 (device, or pinned host for kHost)
  const uint8_t* q = nullptr;    // [out][in] e4m3 (kFp8)
  const float* scale = nullptr;  // [out] (kFp8)
  int out = 0, in = 0;
  size_t bf16_bytes() const { return size_t(out) * in * sizeof(bf16); }
};

struct VisionBlock {
  const bf16 *ln1_w = nullptr, *ln1_b = nullptr, *qkv_b = nullptr, *proj_b = nullptr;
  const bf16 *ln2_w = nullptr, *ln2_b = nullptr, *fc1_b = nullptr, *fc2_b = nullptr;
  VisionMat qkv, proj, fc1, fc2;
};

struct VisionWeights {
  VisionMode mode = VisionMode::kDevice;
  VisionMat patch;
  const bf16 *patch_b = nullptr, *pos = nullptr;
  VisionBlock blocks[vdims::kDepth];
  const bf16 *norm_w = nullptr, *norm_b = nullptr, *fc1_b = nullptr, *fc2_b = nullptr;
  VisionMat fc1, fc2;
};

// Device workspace for images of up to max_patches patches (with the BF16 staging buffers in the kHost /
// kFp8 modes).
size_t vision_ws_bytes(int max_patches, VisionMode mode);

// kFp8 at load: w [rows][cols] BF16 (device) -> e4m3 q and the per-row scale amax / 448.
void vision_quant_fp8(const bf16* w, int rows, int cols, uint8_t* q, float* scale, cudaStream_t s);

// Encodes one image: pixels [gh * gw][1536] fp32 in host memory (the image processor's pixel_values,
// patches in 2x2 merge-block order), gh x gw the patch grid (both even). Writes [gh * gw / 4][5120] BF16
// to out (device memory or a UVA pointer to pinned host memory). Synchronous with respect to `pixels`.
void vision_encode(const VisionWeights& w, const float* pixels, int gh, int gw, bf16* out, void* ws,
                   size_t ws_bytes, cudaStream_t s);

}  // namespace q27
