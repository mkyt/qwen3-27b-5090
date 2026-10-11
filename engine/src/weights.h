// Target model weights on the GPU (checkpoint layout, NVFP4 scale factors swizzled for CUTLASS) and
// the embedding table in pinned host memory (read through UVA).
#pragma once

#include <vector>

#include "kernels.h"
#include "memory.h"
#include "safetensors.h"
#include "vision.h"

namespace q27 {

struct LinearFp8 {
  const uint8_t* w = nullptr;  // e4m3 [out][in]
  float w_scale = 0.f, in_scale = 0.f;
  int out = 0, in = 0;
  AlphaSegments alpha;  // in_scale * w_scale per output range (several for a fused projection)
};

struct LinearFp4 {
  const uint8_t* w = nullptr;   // packed e2m1 [out][in/2]
  const uint8_t* sf = nullptr;  // e4m3 scale factors, swizzled
  float w_scale2 = 0.f, in_scale = 0.f;
  int out = 0, in = 0;
  AlphaSegments alpha;  // in_scale * w_scale2 per output range
};

// A projection of the GDN / attention layers: FP8 (the nvidia checkpoint) or NVFP4 (allfp4 checkpoints).
struct LinearQ {
  LinearFp8 f8;
  LinearFp4 f4;
  bool fp4 = false;
  float in_scale() const { return fp4 ? f4.in_scale : f8.in_scale; }
  int out() const { return fp4 ? f4.out : f8.out; }
};

struct GdnWeights {
  LinearQ qkv, z, out;
  LinearQ qkvz;  // qkv | z as one GEMM (adjacent in memory, same input scale)
  const bf16 *a = nullptr, *b = nullptr;  // [48][5120]
  const bf16 *A_log = nullptr, *dt_bias = nullptr, *conv = nullptr, *norm = nullptr;
};

struct AttnWeights {
  LinearQ q, k, v, o;
  LinearQ qkv;  // q | k | v as one GEMM
  const bf16 *q_norm = nullptr, *k_norm = nullptr;
};

struct LayerWeights {
  const bf16 *in_norm = nullptr, *post_norm = nullptr;
  GdnWeights gdn;
  AttnWeights attn;
  LinearFp4 gate, up, down;
  LinearFp4 gate_up;  // gate | up as one GEMM
};

class Weights {
 public:
  // Reserves device and host regions for every tensor (no I/O yet). NVFP4 layers may be in the modelopt
  // layout (weight, weight_scale, weight_scale_2, input_scale) or the compressed-tensors one
  // (weight_packed, weight_scale, weight_global_scale = 1 / weight_scale_2, input_global_scale =
  // 1 / input_scale). lm_ckpt (may be null): where lm_head comes from when the model's is BF16.
  void plan(const SafeTensors& ckpt, const SafeTensors* lm_ckpt, Pool& dev, Pool& host);
  // The vision tower (BF16, `model.visual.*`; P4-12; call after plan, before the pools commit). Weight
  // matrices: BF16 in dev (kDevice), BF16 in host (kHost), or e4m3 + row scales in dev quantized at load
  // (kFp8); the small tensors in dev.
  void plan_vision(const SafeTensors& ckpt, Pool& dev, Pool& host, VisionMode mode);
  // Reads the checkpoints into the committed pools.
  void load(cudaStream_t s);
  bool fp4_proj = false;  // the GDN / attention projections are NVFP4

  std::vector<LayerWeights> layers;
  const bf16* final_norm = nullptr;
  LinearFp4 lm_head;
  const bf16* embed = nullptr;  // pinned host [vocab][hidden], UVA pointer
  VisionWeights vision;         // (planned by plan_vision)

 private:
  struct Job {
    const SafeTensors* src;
    std::string name;
    Pool* pool;
    size_t offset;
    int sf_rows = 0, sf_kblocks = 0;  // > 0: swizzle as NVFP4 scale factors
    std::string dq;  // non-empty: an NVFP4 layer (this prefix, compressed-tensors) dequantized to BF16
    int q8_rows = 0;      // > 0: a BF16 matrix quantized to e4m3 with row scales (vision_quant_fp8)
    size_t q8_scale = 0;  // the scales' offset in pool
  };
  std::vector<Job> jobs_;
  struct ScalarJob {
    const SafeTensors* src;
    std::string name;
    float* dst;
    bool inv = false;  // store 1 / value
  };
  std::vector<ScalarJob> scalars_;
  // Fix-ups from (pool, offset) to pointers once the pools are committed.
  std::vector<std::pair<const void**, std::pair<Pool*, size_t>>> ptrs_;

  void tensor(const SafeTensors& ckpt, const std::string& name, Pool& pool, const void** dst, int sf_rows = 0,
              int sf_kblocks = 0);
  void fp8(const SafeTensors& ckpt, const std::string& prefix, Pool& dev, LinearFp8& l);
  // NVFP4 layers planned as a group: all weights adjacent, then all scale factors adjacent.
  void fp4(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearFp4*>>& group, Pool& dev);
  // Projections in FP8 or NVFP4 (as the checkpoint has them), planned as a group.
  void proj(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearQ*>>& group, Pool& dev);
  // A small BF16 weight [out][in] that may be stored as NVFP4 (dequantized at load).
  void bf16_weight(const SafeTensors& ckpt, const std::string& prefix, Pool& dev, const bf16** dst);
  void fuse();  // after load: alphas and the fused projections
};

}  // namespace q27
