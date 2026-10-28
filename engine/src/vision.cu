// Vision tower (P4-12, see vision.h and reference/qwen27b/vision.py).
//
// Per image of P patches: patch embed (a GEMM on the flattened patches) + the bilinearly resampled position
// table, 27 pre-LN blocks, the 2x2 merger. GEMMs are gemm_bf16 (fp32 out); rows are processed in
// chunks of kRows where a GEMM needs a wide temporary. Attention is non-causal over the image's patches:
// a flash-attention kernel with BF16 mma.sync, head_dim 72 padded to 80 in shared memory.
//
// VISION=host / fp8: the weight matrices go through two BF16 staging buffers at the end of the workspace,
// one stage (the patch embedding, a block, merger fc1, merger fc2) each. host copies stage i + 2 from
// pinned memory on a side stream while stage i computes; fp8 dequantizes each stage just before it.
#include "vision.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>

#include "common.h"
#include "mma.cuh"

namespace q27 {
namespace {

using namespace vdims;

constexpr int kRows = 2048;  // patches per GEMM chunk (multiple of 4)
constexpr int kThreeH = 3 * kHidden;

// Workspace layout for P patches (offsets in bytes, 256-aligned).
struct Ws {
  float* x;      // [P][1152] residual stream
  bf16* ln;      // [P][1152] LayerNorm output (GEMM operand)
  bf16* heads;   // [3][16][P][72] q, k, v per head; the BF16 pixels [P][1536] before the first block
  bf16* o;       // [P][1152] attention output
  float* tmp;    // [kRows][4608]
  bf16* tmp2;    // [kRows][4304]
  char* stage[2];  // BF16 weight staging (kHost / kFp8)
  size_t bytes;
};

constexpr size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// Stages of weight matrices: 0 the patch embedding, 1..27 the blocks, 28 / 29 merger fc1 / fc2.
constexpr int kStages = kDepth + 3;
constexpr size_t kBlockStageBytes =
    align256(size_t(kThreeH) * kHidden * 2) + align256(size_t(kHidden) * kHidden * 2) + 2 * align256(size_t(kInter) * kHidden * 2);
constexpr size_t kStageBytes = std::max({align256(size_t(kHidden) * kPatchDim * 2), kBlockStageBytes,
                                         align256(size_t(kMergedDim) * kMergedDim * 2), align256(size_t(kOut) * kMergedDim * 2)});

Ws carve(void* base, int P, VisionMode mode) {
  Ws w{};
  size_t off = 0;
  auto take = [&](size_t n) {
    size_t at = off;
    off += (n + 255) / 256 * 256;
    return static_cast<char*>(base) + at;
  };
  w.x = reinterpret_cast<float*>(take(size_t(P) * kHidden * 4));
  w.ln = reinterpret_cast<bf16*>(take(size_t(P) * kHidden * 2));
  w.heads = reinterpret_cast<bf16*>(take(std::max(size_t(3) * P * kHidden, size_t(P) * kPatchDim) * 2));
  w.o = reinterpret_cast<bf16*>(take(size_t(P) * kHidden * 2));
  w.tmp = reinterpret_cast<float*>(take(size_t(kRows) * kMergedDim * 4));
  w.tmp2 = reinterpret_cast<bf16*>(take(size_t(kRows) * kInter * 2));
  for (int i = 0; i < 2; i++) w.stage[i] = mode == VisionMode::kDevice ? nullptr : take(kStageBytes);
  w.bytes = off;
  return w;
}

// Patch (row, col) of patch index p in the processor's order (2x2 merge blocks row-major, row-major inside).
__device__ __forceinline__ void patch_rc(int p, int gw, int& row, int& col) {
  const int in_col = p % kMerge, in_row = (p / kMerge) % kMerge, bw = gw / kMerge;
  const int bcol = (p / (kMerge * kMerge)) % bw, brow = p / (kMerge * kMerge * bw);
  row = brow * kMerge + in_row;
  col = bcol * kMerge + in_col;
}

__global__ void f32_to_bf16_kernel(const float* x, int64_t n, bf16* out) {
  for (int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4; i < n; i += int64_t(gridDim.x) * blockDim.x * 4) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    *reinterpret_cast<__nv_bfloat162*>(out + i) = __floats2bfloat162_rn(v.x, v.y);
    *reinterpret_cast<__nv_bfloat162*>(out + i + 2) = __floats2bfloat162_rn(v.z, v.w);
  }
}

// x[p] += patch bias + the position table resampled to the gh x gw grid (bilinear, align_corners; the sum
// rounded to BF16 as the reference). Block per patch.
__global__ void pos_embed_kernel(float* x, const bf16* bias, const bf16* table, int gh, int gw) {
  const int p = blockIdx.x;
  int row, col;
  patch_rc(p, gw, row, col);
  auto taps = [](int i, int n, int* t, float* w) {
    const float src = float(i) * float(kPosSide - 1) / float(max(n - 1, 1));
    const float f = floorf(src);
    t[0] = min(int(f), kPosSide - 1), t[1] = min(int(f) + 1, kPosSide - 1);
    w[0] = 1.f - (src - f), w[1] = src - f;
  };
  int ht[2], wt[2];
  float hw[2], ww[2];
  taps(row, gh, ht, hw);
  taps(col, gw, wt, ww);
  float* xr = x + int64_t(p) * kHidden;
  for (int c = threadIdx.x; c < kHidden; c += blockDim.x) {
    float pe = 0.f;
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
      for (int b = 0; b < 2; b++)
        pe += __bfloat162float(table[(ht[a] * kPosSide + wt[b]) * kHidden + c]) * (hw[a] * ww[b]);
    xr[c] += __bfloat162float(bias[c]) + __bfloat162float(__float2bfloat16(pe));
  }
}

// LayerNorm (eps 1e-6) of rows of 1152 fp32 -> BF16. 128 threads, 9 values each.
__global__ void __launch_bounds__(128) layernorm_kernel(const float* x, const bf16* w, const bf16* b, bf16* out) {
  constexpr int kPer = kHidden / 128;
  const float* xr = x + int64_t(blockIdx.x) * kHidden;
  float v[kPer], s = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; i++) v[i] = xr[threadIdx.x + i * 128], s += v[i];
  __shared__ float red[4];
  auto block_sum = [&](float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = a;
    __syncthreads();
    return red[0] + red[1] + red[2] + red[3];
  };
  const float mean = block_sum(s) / kHidden;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; i++) v[i] -= mean, q += v[i] * v[i];
  const float rstd = rsqrtf(block_sum(q) / kHidden + 1e-6f);
  bf16* o = out + int64_t(blockIdx.x) * kHidden;
#pragma unroll
  for (int i = 0; i < kPer; i++) {
    const int c = threadIdx.x + i * 128;
    o[c] = __float2bfloat16(v[i] * rstd * __bfloat162float(w[c]) + __bfloat162float(b[c]));
  }
}

// qkv rows r0..r0+rows-1 ([rows][3][16][72] fp32, no bias) -> + bias, 2D RoPE on q and k (rotate_half over
// the 72 dims: frequency j < 18 takes the patch row, 18..35 the column), BF16 into heads [3][16][P][72].
__global__ void qkv_rope_kernel(const float* qkv, const bf16* bias, int r0, int P, int gw, bf16* heads) {
  const int r = blockIdx.x, p = r0 + r;
  int row, col;
  patch_rc(p, gw, row, col);
  const float* src = qkv + int64_t(r) * kThreeH;
  for (int e = threadIdx.x; e < kThreeH; e += blockDim.x) {
    const int which = e / kHidden, hd = (e % kHidden) / kHeadDim, d = e % kHeadDim;
    float x = src[e] + __bfloat162float(bias[e]);
    if (which < 2) {
      constexpr int half = kHeadDim / 2, quarter = kHeadDim / 4;
      const int pe = d < half ? e + half : e - half;
      const float y = src[pe] + __bfloat162float(bias[pe]);
      const int j = d % half;
      const float inv = exp2f(-float(2 * (j % quarter)) / float(half) * log2f(10000.f));
      float sn, cs;
      sincosf(float(j < quarter ? row : col) * inv, &sn, &cs);
      x = x * cs + (d < half ? -y : y) * sn;
    }
    heads[((int64_t(which) * kHeads + hd) * P + p) * kHeadDim + d] = __float2bfloat16(x);
  }
}

// x[r][c] += bias[c] ([n] rows of N).
__global__ void add_bias_kernel(float* x, const bf16* bias, int64_t n, int N) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n * N; i += int64_t(gridDim.x) * blockDim.x)
    x[i] += __bfloat162float(bias[i % N]);
}

// out = bf16(act(x + bias)): act 0 = none, 1 = GELU (tanh), 2 = GELU (erf).
template <int kAct>
__global__ void bias_act_kernel(const float* x, const bf16* bias, int64_t n, int N, bf16* out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n * N; i += int64_t(gridDim.x) * blockDim.x) {
    float v = x[i] + __bfloat162float(bias[i % N]);
    if constexpr (kAct == 1) v = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    if constexpr (kAct == 2) v = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
    out[i] = __float2bfloat16(v);
  }
}

// ---------------------------------------------------------------- attention

constexpr int kBm = 64, kBn = 64;  // queries per CTA (16 per warp), keys per tile
constexpr int kStr = 88;           // shared row stride in elements (176 bytes: conflict-free ldmatrix)
constexpr int kKSteps = 5;         // head_dim 72 padded to 80 for Q K^T
constexpr int kDTiles = 10;        // n8 tiles of the padded head dim for P V
constexpr size_t kAttnSmem = size_t(kBm + 4 * kBn) * kStr * sizeof(bf16);

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }

// Grid (ceil(P / 64), 16 heads), 128 threads. q, k, v: [16][P][72] BF16; out [P][16 * 72] BF16.
__global__ void __launch_bounds__(128) vis_attn_kernel(const bf16* q, const bf16* k, const bf16* v, int P,
                                                       float scale_log2, bf16* out) {
  extern __shared__ __align__(16) bf16 sm[];
  bf16* sQ = sm;
  bf16* sK = sQ + kBm * kStr;   // [2][kBn][kStr]
  bf16* sV = sK + 2 * kBn * kStr;
  const int h = blockIdx.y, m0 = blockIdx.x * kBm;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const bf16* qh = q + int64_t(h) * P * kHeadDim;
  const bf16* kh = k + int64_t(h) * P * kHeadDim;
  const bf16* vh = v + int64_t(h) * P * kHeadDim;

  // Columns 72..87 of every row stay zero (cp.async fills 0..71).
  for (int r = tid; r < kBm + 4 * kBn; r += 128) {
    *reinterpret_cast<uint4*>(sm + r * kStr + kHeadDim) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(sm + r * kStr + kHeadDim + 8) = make_uint4(0, 0, 0, 0);
  }
  auto load = [&](bf16* dst, const bf16* src, int r0) {
    constexpr int kChunks = kHeadDim / 8;  // 16-byte pieces per row
    for (int c = tid; c < kBn * kChunks; c += 128) {
      const int r = c / kChunks, ch = c % kChunks;
      const bool ok = r0 + r < P;
      cp_async16z(smem_u32(dst + r * kStr + ch * 8), src + int64_t(ok ? r0 + r : 0) * kHeadDim + ch * 8, ok);
    }
  };
  load(sQ, qh, m0);
  load(sK, kh, 0);
  load(sV, vh, 0);
  cp_commit();

  const int nblk = (P + kBn - 1) / kBn;
  float o[kDTiles][4] = {};
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  uint32_t qf[kKSteps][4];
  for (int j = 0; j < nblk; j++) {
    const int buf = j & 1;
    if (j + 1 < nblk) {
      load(sK + (buf ^ 1) * kBn * kStr, kh, (j + 1) * kBn);
      load(sV + (buf ^ 1) * kBn * kStr, vh, (j + 1) * kBn);
    }
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (j == 0)
#pragma unroll
      for (int kk = 0; kk < kKSteps; kk++)
        ldsm_x4(qf[kk], smem_u32(sQ + (warp * 16 + lane % 16) * kStr + kk * 16 + (lane / 16) * 8));
    const bf16* tK = sK + buf * kBn * kStr;
    const bf16* tV = sV + buf * kBn * kStr;
    float s[8][4] = {};
#pragma unroll
    for (int kk = 0; kk < kKSteps; kk++)
#pragma unroll
      for (int np = 0; np < 4; np++) {
        uint32_t b[4];
        ldsm_x4(b, smem_u32(tK + (np * 16 + lane % 8 + (lane / 16) * 8) * kStr + kk * 16 + ((lane / 8) % 2) * 8));
        mma_bf16_16816(s[2 * np], qf[kk], b);
        mma_bf16_16816(s[2 * np + 1], qf[kk], b + 2);
      }
    // Online softmax over this tile (rows lane/4 and lane/4 + 8 of the warp's 16).
    const int kbase = j * kBn + 2 * (lane % 4);
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float& sv = s[nt][hf * 2 + e];
          sv = kbase + nt * 8 + e < P ? sv * scale_log2 : -INFINITY;
          mx = fmaxf(mx, sv);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mnew = fmaxf(mrow[hf], mx);
      const float alpha = exp2f(mrow[hf] - mnew);
      mrow[hf] = mnew;
      float sum = 0.f;
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float& sv = s[nt][hf * 2 + e];
          sv = exp2f(sv - mnew);
          sum += sv;
        }
      lrow[hf] = lrow[hf] * alpha + sum;
#pragma unroll
      for (int dt = 0; dt < kDTiles; dt++) o[dt][hf * 2] *= alpha, o[dt][hf * 2 + 1] *= alpha;
    }
#pragma unroll
    for (int kk = 0; kk < kBn / 16; kk++) {
      const uint32_t a[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]),
                             pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dp = 0; dp < kDTiles / 2; dp++) {
        uint32_t b[4];
        ldsm_x4_trans(b, smem_u32(tV + (kk * 16 + lane % 8 + ((lane / 8) % 2) * 8) * kStr + dp * 16 + (lane / 16) * 8));
        mma_bf16_16816(o[2 * dp], a, b);
        mma_bf16_16816(o[2 * dp + 1], a, b + 2);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int hf = 0; hf < 2; hf++) {
    float l = lrow[hf];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const float inv = 1.f / l;
    const int row = m0 + warp * 16 + lane / 4 + hf * 8;
    if (row >= P) continue;
    bf16* orow = out + int64_t(row) * kHidden + h * kHeadDim;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; dt++)
      *reinterpret_cast<__nv_bfloat162*>(orow + dt * 8 + 2 * (lane % 4)) =
          __floats2bfloat162_rn(o[dt][hf * 2] * inv, o[dt][hf * 2 + 1] * inv);
  }
}

int grid_for(int64_t n) { return int(std::min<int64_t>((n + 255) / 256, 4096)); }

// ---------------------------------------------------------------- FP8 weights

// Row per block, 256 threads: scale = amax / 448, q = e4m3(w / scale).
__global__ void __launch_bounds__(256) quant_rows_kernel(const bf16* w, int cols, uint8_t* q, float* scale) {
  const bf16* wr = w + int64_t(blockIdx.x) * cols;
  float a = 0.f;
  for (int c = threadIdx.x; c < cols; c += 256) a = fmaxf(a, fabsf(__bfloat162float(wr[c])));
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  __shared__ float red[8];
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = a;
  __syncthreads();
  a = 0.f;
  for (int i = 0; i < 8; i++) a = fmaxf(a, red[i]);
  const float sc = a > 0.f ? a / 448.f : 1.f;
  if (threadIdx.x == 0) scale[blockIdx.x] = sc;
  uint8_t* qr = q + int64_t(blockIdx.x) * cols;
  for (int c = threadIdx.x; c < cols; c += 256)
    qr[c] = __nv_cvt_float_to_fp8(__bfloat162float(wr[c]) / sc, __NV_SATFINITE, __NV_E4M3);
}

// out = bf16(e4m3 q * scale[row]), 8 values per thread step (cols a multiple of 8).
__global__ void dequant_rows_kernel(const uint8_t* q, const float* scale, int64_t n, int cols, bf16* out) {
  for (int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8; i < n; i += int64_t(gridDim.x) * blockDim.x * 8) {
    const uint2 v = *reinterpret_cast<const uint2*>(q + i);
    const float sc = scale[i / cols];
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint32_t word = k < 2 ? v.x : v.y;
      const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(word >> (16 * (k % 2))), __NV_E4M3);
      const float2 f = __half22float2(__half2(h2));
      o[k] = pack_bf16(f.x * sc, f.y * sc);
    }
    *reinterpret_cast<uint4*>(out + i) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// The weight matrices of each stage as BF16 device pointers (kDevice: the weights themselves).
class Stager {
 public:
  Stager(const VisionWeights& W, const Ws& ws, cudaStream_t s) : W_(W), s_(s) {
    mats_[0][0] = &W.patch;
    for (int i = 0; i < kDepth; i++) {
      const VisionBlock& B = W.blocks[i];
      mats_[1 + i][0] = &B.qkv, mats_[1 + i][1] = &B.proj, mats_[1 + i][2] = &B.fc1, mats_[1 + i][3] = &B.fc2;
    }
    mats_[kDepth + 1][0] = &W.fc1;
    mats_[kDepth + 2][0] = &W.fc2;
    for (int i = 0; i < 2; i++) buf_[i] = ws.stage[i];
    if (W.mode == VisionMode::kHost) {
      static cudaStream_t cs = nullptr;
      static cudaEvent_t ready[2], freed[2];
      if (!cs) {
        Q27_CUDA(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        for (int i = 0; i < 2; i++) {
          Q27_CUDA(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
          Q27_CUDA(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming));
        }
      }
      cs_ = cs;
      for (int i = 0; i < 2; i++) ready_[i] = ready[i], freed_[i] = freed[i];
      // The copy stream starts after the work already queued on s (the workspace may overlay it).
      Q27_CUDA(cudaEventRecord(freed_[0], s_));
      Q27_CUDA(cudaStreamWaitEvent(cs_, freed_[0]));
    }
    issue(0);
    issue(1);
  }

  // Stage st's matrix k, ready for use on s (waits for its copy).
  const bf16* get(int st, int k) {
    if (W_.mode == VisionMode::kHost && k == 0) Q27_CUDA(cudaStreamWaitEvent(s_, ready_[st % 2]));
    return ptr_[st][k];
  }
  // Stage st's work is queued: its buffer goes to stage st + 2.
  void done(int st) {
    if (W_.mode == VisionMode::kHost) Q27_CUDA(cudaEventRecord(freed_[st % 2], s_));
    if (st + 2 < kStages) issue(st + 2);
  }

 private:
  void issue(int st) {
    char* at = buf_[st % 2];
    if (W_.mode == VisionMode::kHost && st >= 2) Q27_CUDA(cudaStreamWaitEvent(cs_, freed_[st % 2]));
    for (int k = 0; k < 4 && mats_[st][k]; k++) {
      const VisionMat& m = *mats_[st][k];
      if (W_.mode == VisionMode::kDevice) {
        ptr_[st][k] = m.w;
        continue;
      }
      bf16* dst = reinterpret_cast<bf16*>(at);
      at += align256(m.bf16_bytes());
      ptr_[st][k] = dst;
      if (W_.mode == VisionMode::kHost) {
        Q27_CUDA(cudaMemcpyAsync(dst, m.w, m.bf16_bytes(), cudaMemcpyHostToDevice, cs_));
      } else {
        const int64_t n = int64_t(m.out) * m.in;
        dequant_rows_kernel<<<grid_for(n / 8), 256, 0, s_>>>(m.q, m.scale, n, m.in, dst);
      }
    }
    if (W_.mode == VisionMode::kHost) Q27_CUDA(cudaEventRecord(ready_[st % 2], cs_));
  }

  const VisionWeights& W_;
  cudaStream_t s_, cs_ = nullptr;
  cudaEvent_t ready_[2] = {}, freed_[2] = {};
  char* buf_[2] = {};
  const VisionMat* mats_[kStages][4] = {};
  const bf16* ptr_[kStages][4] = {};
};

}  // namespace

size_t vision_ws_bytes(int max_patches, VisionMode mode) { return carve(nullptr, max_patches, mode).bytes; }

void vision_quant_fp8(const bf16* w, int rows, int cols, uint8_t* q, float* scale, cudaStream_t s) {
  Q27_CHECK(cols % 8 == 0, "vision_quant_fp8: cols must be a multiple of 8");
  quant_rows_kernel<<<rows, 256, 0, s>>>(w, cols, q, scale);
  Q27_CUDA(cudaGetLastError());
}

void vision_encode(const VisionWeights& W, const float* pixels, int gh, int gw, bf16* out, void* ws, size_t ws_bytes,
                   cudaStream_t s) {
  Q27_CHECK(gh > 0 && gw > 0 && gh % kMerge == 0 && gw % kMerge == 0, "vision: the patch grid must be even");
  const int P = gh * gw;
  const Ws w = carve(ws, P, W.mode);
  Q27_CHECK(w.bytes <= ws_bytes, "vision: image larger than the workspace (IMAGE_MAX_PIXELS)");
  static bool attr = false;
  if (!attr) {
    Q27_CUDA(cudaFuncSetAttribute(vis_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttnSmem)));
    attr = true;
  }

  Stager st(W, w, s);  // (the first two stages start copying now)
  // Pixels (host fp32) -> BF16 in heads, through tmp in chunks.
  bf16* px = w.heads;
  const int64_t per = int64_t(kRows) * kMergedDim / kPatchDim;  // patches per staging chunk
  for (int64_t p0 = 0; p0 < P; p0 += per) {
    const int64_t n = std::min<int64_t>(per, P - p0) * kPatchDim;
    Q27_CUDA(cudaMemcpyAsync(w.tmp, pixels + p0 * kPatchDim, n * 4, cudaMemcpyHostToDevice, s));
    f32_to_bf16_kernel<<<grid_for(n / 4), 256, 0, s>>>(w.tmp, n, px + p0 * kPatchDim);
  }
  gemm_bf16(P, kHidden, kPatchDim, px, st.get(0, 0), 1.f, nullptr, 0.f, w.x, s);
  st.done(0);
  pos_embed_kernel<<<P, 256, 0, s>>>(w.x, W.patch_b, W.pos, gh, gw);
  Q27_CUDA(cudaGetLastError());

  const int64_t hs = int64_t(kHeads) * P * kHeadDim;  // one of q / k / v
  const float scale_log2 = 1.4426950408889634f / sqrtf(float(kHeadDim));
  for (int i = 0; i < kDepth; i++) {
    const VisionBlock& B = W.blocks[i];
    const bf16 *qkv_w = st.get(1 + i, 0), *proj_w = st.get(1 + i, 1);
    const bf16 *fc1_w = st.get(1 + i, 2), *fc2_w = st.get(1 + i, 3);
    layernorm_kernel<<<P, 128, 0, s>>>(w.x, B.ln1_w, B.ln1_b, w.ln);
    for (int r0 = 0; r0 < P; r0 += kRows) {
      const int rows = std::min(kRows, P - r0);
      gemm_bf16(rows, kThreeH, kHidden, w.ln + int64_t(r0) * kHidden, qkv_w, 1.f, nullptr, 0.f, w.tmp, s);
      qkv_rope_kernel<<<rows, 256, 0, s>>>(w.tmp, B.qkv_b, r0, P, gw, w.heads);
    }
    vis_attn_kernel<<<dim3((P + kBm - 1) / kBm, kHeads), 128, kAttnSmem, s>>>(w.heads, w.heads + hs, w.heads + 2 * hs,
                                                                              P, scale_log2, w.o);
    Q27_CUDA(cudaGetLastError());
    gemm_bf16(P, kHidden, kHidden, w.o, proj_w, 1.f, w.x, 1.f, w.x, s);
    add_bias_kernel<<<grid_for(int64_t(P) * kHidden), 256, 0, s>>>(w.x, B.proj_b, P, kHidden);
    layernorm_kernel<<<P, 128, 0, s>>>(w.x, B.ln2_w, B.ln2_b, w.ln);
    for (int r0 = 0; r0 < P; r0 += kRows) {
      const int rows = std::min(kRows, P - r0);
      gemm_bf16(rows, kInter, kHidden, w.ln + int64_t(r0) * kHidden, fc1_w, 1.f, nullptr, 0.f, w.tmp, s);
      bias_act_kernel<1><<<grid_for(int64_t(rows) * kInter), 256, 0, s>>>(w.tmp, B.fc1_b, rows, kInter, w.tmp2);
      float* xr = w.x + int64_t(r0) * kHidden;
      gemm_bf16(rows, kHidden, kInter, w.tmp2, fc2_w, 1.f, xr, 1.f, xr, s);
    }
    add_bias_kernel<<<grid_for(int64_t(P) * kHidden), 256, 0, s>>>(w.x, B.fc2_b, P, kHidden);
    Q27_CUDA(cudaGetLastError());
    st.done(1 + i);
  }

  // Merger: LayerNorm per patch, 4 patches (one 2x2 block) per row of 4608 -> fc1 -> GELU (erf) -> fc2.
  layernorm_kernel<<<P, 128, 0, s>>>(w.x, W.norm_w, W.norm_b, w.ln);
  const bf16 *mfc1_w = st.get(kDepth + 1, 0), *mfc2_w = st.get(kDepth + 2, 0);
  for (int r0 = 0; r0 < P; r0 += kRows) {
    const int m = std::min(kRows, P - r0) / (kMerge * kMerge);
    gemm_bf16(m, kMergedDim, kMergedDim, w.ln + int64_t(r0) * kHidden, mfc1_w, 1.f, nullptr, 0.f, w.tmp, s);
    bias_act_kernel<2><<<grid_for(int64_t(m) * kMergedDim), 256, 0, s>>>(w.tmp, W.fc1_b, m, kMergedDim, w.tmp2);
    gemm_bf16(m, kOut, kMergedDim, w.tmp2, mfc2_w, 1.f, nullptr, 0.f, w.tmp, s);
    bias_act_kernel<0><<<grid_for(int64_t(m) * kOut), 256, 0, s>>>(w.tmp, W.fc2_b, m, kOut,
                                                                   out + int64_t(r0 / (kMerge * kMerge)) * kOut);
  }
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
