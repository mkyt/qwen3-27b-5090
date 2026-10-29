// Target model weights on the GPU (checkpoint layout, NVFP4 scale factors in the CUTLASS SM1xx swizzled layout) and
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
  // Per-output-channel weight scales [out] (compressed-tensors FP8 "channel" weights, e.g. unsloth's and Red Hat's
  // checkpoints): the activations are then quantized per token at run time (QuantOut::kFp8Row) and w_scale,
  // in_scale are unused (alpha 1).
  const float* w_scales = nullptr;
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
  bool dyn = false;  // FP8 with per-channel scales: the input is quantized per token (f8.w_scales)
  float in_scale() const { return fp4 ? f4.in_scale : f8.in_scale; }
  int out() const { return fp4 ? f4.out : f8.out; }
  bool static8() const { return !fp4 && !dyn; }  // FP8 with a calibrated input scale (the producer may quantize)
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
  LinearFp4 gate, up, down;  // (gate / up: the alphas and scales; their weights are only in gate_up)
  LinearFp4 gate_up;  // gate | up as one GEMM, interleaved in blocks of 64 rows (kernels.h glu_row), alpha 1
  // An FP8 MLP (mlp8; per-channel scales, the last layers of unsloth's / Red Hat's checkpoints) in place of the
  // NVFP4 one: gate_up8 is gate | up concatenated (not interleaved).
  bool mlp8 = false;
  LinearFp8 gate8, up8, down8, gate_up8;
};

class Weights {
 public:
  // Reserves device and host regions for every tensor (no I/O yet). NVFP4 layers may be in the modelopt
  // layout (weight, weight_scale, weight_scale_2, input_scale) or the compressed-tensors one
  // (weight_packed, weight_scale, weight_global_scale = 1 / weight_scale_2, input_global_scale =
  // 1 / input_scale). A BF16 lm_head comes from lm_ckpt if given (Q27_LM_HEAD_CKPT: the NVFP4 one of the
  // model it was quantized from), else it is quantized to NVFP4 at load (fp4_from_bf16).
  // An FP8 lm_head with per-channel scales stays FP8 (lm_head8, the target's logits); with fp4_head (the draft model
  // reads lm_head) its NVFP4 copy is made at load as well (fp4_from_fp8), in head_pool if given (load-time memory:
  // the draft model only gathers its vocabulary head from it) instead of dev.
  void plan(const SafeTensors& ckpt, const SafeTensors* lm_ckpt, Pool& dev, Pool& host, bool fp4_head = true,
            Pool* head_pool = nullptr);
  // The vision tower (BF16, `model.visual.*`; P4-12; call after plan, before the pools commit). Weight
  // matrices: BF16 in dev (kDevice), BF16 in host (kHost), or e4m3 + row scales in dev quantized at load
  // (kFp8); the small tensors in dev.
  void plan_vision(const SafeTensors& ckpt, Pool& dev, Pool& host, VisionMode mode);
  // Reads the checkpoints into the committed pools.
  void load(cudaStream_t s);
  bool fp4_proj = false;  // the GDN / attention projections are NVFP4
  bool dyn8 = false;      // some layers are FP8 with per-channel scales (activations quantized per token)
  bool mlp8 = false;      // some MLPs are FP8 (LayerWeights::mlp8)

  std::vector<LayerWeights> layers;
  const bf16* final_norm = nullptr;
  LinearFp4 lm_head;   // (with lm_head8: its NVFP4 copy for the draft model, or empty)
  LinearFp8 lm_head8;  // an FP8 lm_head with per-channel scales (w null otherwise)
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
    size_t q8_scale = 0;  // the scales' offset in pool (q8_rows, q4)
    int glu = 0;          // 1 / 2: the MLP's gate / up, stored interleaved from offset (glu_row, fp4_glu)
    LinearFp4* q4 = nullptr;  // non-null: a BF16 matrix quantized to NVFP4 at load into this layer (fp4_from_bf16)
    const LinearFp8* from8 = nullptr;  // (q4) the source is this FP8 layer on the device, not BF16 (fp4_from_fp8)
    bool f32 = false;  // per-channel scales (BF16 or F32, [out] or [out][1]) stored as f32
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
  void scalar(const SafeTensors& ckpt, const std::string& name, float* dst, bool inv = false);
  // FP8 layers planned as a group: all weights adjacent, then (per-channel) all scales adjacent. True if the scales
  // are per channel.
  bool fp8(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearFp8*>>& group, Pool& dev);
  // NVFP4 layers planned as a group: all weights adjacent, then all scale factors adjacent.
  void fp4(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearFp4*>>& group, Pool& dev);
  // A BF16 layer (`prefix.weight`) quantized to NVFP4 at load: the weights' global scale from their amax, the
  // input scale kBf16InScale.
  void fp4_from_bf16(const SafeTensors& ckpt, const std::string& prefix, Pool& dev, LinearFp4& l);
  // The NVFP4 copy of an FP8 layer with per-channel scales (planned before it), quantized at load from its BF16
  // dequantization as fp4_from_bf16 does; the global scale from the largest channel scale (448 * scale).
  void fp4_from_fp8(const SafeTensors& ckpt, const std::string& prefix, const LinearFp8& src, Pool& dev, LinearFp4& l);
  // The MLP's gate and up as one interleaved [2 I][K] gate | up (kernels.h glu_row): gate.w / gate.sf are its
  // start; fuse() makes it gate_up and clears the parts' pointers.
  void fp4_glu(const SafeTensors& ckpt, const std::string& gate, const std::string& up, LinearFp4& g, LinearFp4& u,
               Pool& dev);
  // Projections in FP8 or NVFP4 (as the checkpoint has them), planned as a group.
  void proj(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearQ*>>& group, Pool& dev);
  // A small BF16 weight [out][in] that may be stored as NVFP4 (dequantized at load).
  void bf16_weight(const SafeTensors& ckpt, const std::string& prefix, Pool& dev, const bf16** dst);
  void fuse();  // after load: alphas and the fused projections
};

}  // namespace q27
