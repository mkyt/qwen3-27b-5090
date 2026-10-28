// Kernels of the DFlash2 draft model and of the speculative step (selector walk, accept, advance).
// The draft runs one 8-row block per step, so these are small latency-bound kernels; the weights
// stream through the skinny BF16 GEMMs (skinny.cu).
#include <cuda_fp8.h>

#include "common.h"
#include "draft.h"
#include "launch.cuh"
#include "mma.cuh"
#include "spec.h"

namespace q27 {
namespace {

using namespace ddims;

constexpr int kNormThreads = kHidden / 16;  // 320: one 16-channel group per thread

__device__ float block_sum(float v) {
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  int nw = blockDim.x / 32;
  v = lane < nw ? red[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ float block_max(float v) {
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  int nw = blockDim.x / 32;
  v = lane < nw ? red[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Two e2m1 codes in one byte, `even` in the low nibble (round to nearest even, saturating).
__device__ __forceinline__ uint8_t cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(odd), "f"(even));
  return uint8_t(r);
}

// NVFP4 of one 16-element block (row m, k-block kb of [M][K]) with global scale s: block scale
// e4m3(amax / 6 / s), values e2m1(x / (sf * s)); dequantized x = q * sf * s (as kernels.cu).
__device__ __forceinline__ void quant_block16(const float* v, float s, int m, int kb, int K, uint8_t* q, uint8_t* sf) {
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
  const float gs = 1.f / s;
  __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
  float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
  float inv = sff != 0.f ? gs / sff : 0.f;
  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; j++) packed[j / 4] |= uint32_t(cvt_e2m1x2(v[2 * j] * inv, v[2 * j + 1] * inv)) << (8 * (j % 4));
  *reinterpret_cast<uint2*>(q + int64_t(m) * (K / 2) + kb * 8) = make_uint2(packed[0], packed[1]);
  sf[sf_offset(m, kb, K / 16)] = sfq;
}

// Per-row scale of the dynamic quantization from the row's absolute maximum.
__device__ __forceinline__ float row_scale(int fmt, float amax) {
  if (!(amax > 0.f)) return 1.f;
  return fmt == int(DraftQuant::kFp8) ? amax / 448.f : amax / (6.f * 448.f);
}

// Writes group g (16 values) of row `row` of a [T][K] GEMM input with row scale rs.
__device__ __forceinline__ void act_group(const ActOut& o, int row, int g, int K, const float* v, float rs) {
  if (o.fmt == int(DraftQuant::kBf16)) {
    uint32_t packed[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      __nv_bfloat162 b2 = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
      packed[j] = *reinterpret_cast<uint32_t*>(&b2);
    }
    uint4* dst = reinterpret_cast<uint4*>(static_cast<bf16*>(o.q) + int64_t(row) * K + g * 16);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  } else if (o.fmt == int(DraftQuant::kFp8)) {
    uint32_t packed[4] = {0, 0, 0, 0};
    const float inv = 1.f / rs;
#pragma unroll
    for (int j = 0; j < 16; j++)
      packed[j / 4] |= uint32_t(__nv_cvt_float_to_fp8(v[j] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * (j % 4));
    *reinterpret_cast<uint4*>(static_cast<uint8_t*>(o.q) + int64_t(row) * K + g * 16) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
  } else {
    quant_block16(v, rs, row, g, K, static_cast<uint8_t*>(o.q), o.sf);
  }
}

// Row-wise output where each thread holds one 16-value group of the row (K = 16 * blockDim.x).
__device__ __forceinline__ void act_row(const ActOut& o, int row, int K, const float* v) {
  float rs = 1.f;
  if (o.fmt != int(DraftQuant::kBf16)) {
    float am = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) am = fmaxf(am, fabsf(v[j]));
    rs = row_scale(o.fmt, block_max(am));
    if (threadIdx.x == 0) o.scale[row] = rs;
  }
  act_group(o, row, threadIdx.x, K, v, rs);
}

__device__ __forceinline__ void load16(const float* p, float* v) {
  const float4* src = reinterpret_cast<const float4*>(p);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float4 t = src[j];
    v[4 * j] = t.x, v[4 * j + 1] = t.y, v[4 * j + 2] = t.z, v[4 * j + 3] = t.w;
  }
}
__device__ __forceinline__ void load16(const bf16* p, float* v) {
  uint4 raw[2];
  raw[0] = __ldg(reinterpret_cast<const uint4*>(p));
  raw[1] = __ldg(reinterpret_cast<const uint4*>(p + 8));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(raw);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float2 f = __bfloat1622float2(h[j]);
    v[2 * j] = f.x, v[2 * j + 1] = f.y;
  }
}
__device__ __forceinline__ void store16(float* p, const float* v) {
  float4* o = reinterpret_cast<float4*>(p);
#pragma unroll
  for (int j = 0; j < 4; j++) o[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
__device__ __forceinline__ void store16(bf16* p, const float* v) {
  uint32_t packed[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    __nv_bfloat162 b2 = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
    packed[j] = *reinterpret_cast<uint32_t*>(&b2);
  }
  uint4* o = reinterpret_cast<uint4*>(p);
  o[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  o[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

// Block per row, thread per 16-channel group.
__global__ void __launch_bounds__(kNormThreads) norm_kernel(const float* x, const bf16* w, float* out, ActOut ao,
                                                            bf16* hb) {
  pdl_trigger();
  const int t = blockIdx.x, g = threadIdx.x;
  float wv[16];
  load16(w + g * 16, wv);
  pdl_wait();
  float v[16];
  load16(x + int64_t(t) * kHidden + g * 16, v);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) ss += v[j] * v[j];
  const float r = rsqrtf(block_sum(ss) / kHidden + kEps);
#pragma unroll
  for (int j = 0; j < 16; j++) v[j] = v[j] * r * wv[j];
  if (out) store16(out + int64_t(t) * kHidden + g * 16, v);
  if (hb) store16(hb + int64_t(t) * kHidden + g * 16, v);
  act_row(ao, t, kHidden, v);
}

// Coefficients of the 16 channels of group g (one thread) for row n and conv side.
struct ConvCoef {
  float c0[16], c1[16];
};
__device__ __forceinline__ ConvCoef conv_coef(const float* coef, const bf16* base, int n, int side, int g) {
  ConvCoef cc;
  load16(base + (side * kTaps + 0) * kHidden + g * 16, cc.c0);
  load16(base + (side * kTaps + 1) * kHidden + g * 16, cc.c1);
  const float d0 = coef[int64_t(n) * kCoef + (side * kTaps + 0) * kGroups + g];
  const float d1 = coef[int64_t(n) * kCoef + (side * kTaps + 1) * kGroups + g];
#pragma unroll
  for (int j = 0; j < 16; j++) cc.c0[j] += d0, cc.c1[j] += d1;
  return cc;
}

__global__ void __launch_bounds__(kNormThreads) conv_in_kernel(const float* h, const float* coef, const bf16* base,
                                                               ActOut ao) {
  pdl_trigger();
  pdl_wait();
  const int n = blockIdx.x, g = threadIdx.x, pos = n % kBlock;
  const ConvCoef cc = conv_coef(coef, base, n, 0, g);
  float v[16], p[16];
  load16(h + int64_t(n) * kHidden + g * 16, v);
  if (pos > 0) {
    load16(h + int64_t(n - 1) * kHidden + g * 16, p);
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = cc.c0[j] * v[j] + cc.c1[j] * p[j];
  } else {
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = cc.c0[j] * v[j];
  }
  act_row(ao, n, kHidden, v);
}

__global__ void __launch_bounds__(kNormThreads) conv_out_norm_kernel(const float* y, const float* coef,
                                                                     const bf16* base, float* x, const bf16* w,
                                                                     float* h, ActOut ao, bf16* hb) {
  pdl_trigger();
  const int n = blockIdx.x, g = threadIdx.x, pos = n % kBlock;
  float wv[16];
  load16(w + g * 16, wv);
  pdl_wait();
  const ConvCoef cc = conv_coef(coef, base, n, 1, g);
  float v[16], p[16], xr[16];
  load16(y + int64_t(n) * kHidden + g * 16, v);
  load16(x + int64_t(n) * kHidden + g * 16, xr);
  if (pos > 0) {
    load16(y + int64_t(n - 1) * kHidden + g * 16, p);
#pragma unroll
    for (int j = 0; j < 16; j++) xr[j] += cc.c0[j] * v[j] + cc.c1[j] * p[j];
  } else {
#pragma unroll
    for (int j = 0; j < 16; j++) xr[j] += cc.c0[j] * v[j];
  }
  store16(x + int64_t(n) * kHidden + g * 16, xr);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) ss += xr[j] * xr[j];
  const float r = rsqrtf(block_sum(ss) / kHidden + kEps);
#pragma unroll
  for (int j = 0; j < 16; j++) xr[j] = xr[j] * r * wv[j];
  if (h) store16(h + int64_t(n) * kHidden + g * 16, xr);
  if (hb) store16(hb + int64_t(n) * kHidden + g * 16, xr);
  act_row(ao, n, kHidden, xr);
}

// RMSNorm over 128 + RoPE (rotate_half over the full head) of one head row, thread d.
__device__ __forceinline__ float head_norm_rope(float x, const bf16* w, int pos, float* xs, float* red) {
  const int d = threadIdx.x;
  float ss = x * x;
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (d % 32 == 0) red[d / 32] = ss;
  __syncthreads();
  ss = red[0] + red[1] + red[2] + red[3];
  x = x * rsqrtf(ss / kHeadDim + kEps) * __bfloat162float(w[d]);
  xs[d] = x;
  __syncthreads();
  constexpr int half = kHeadDim / 2;
  const int fi = d % half;
  const float inv_freq = 1.f / powf(kRopeTheta, float(2 * fi) / float(kHeadDim));
  const float f = float(pos) * inv_freq;
  const float rot = d < half ? -xs[d + half] : xs[d - half];
  return x * cosf(f) + rot * sinf(f);
}

// Block per (token, head slot): 0..31 query heads, 32..39 key heads (+ their values).
// kBatch (P6-10): token t is row t % 8 of the block of lane t / 8 (pos_dev: the lanes' positions).
template <bool kBatch = false>
__global__ void __launch_bounds__(kHeadDim) qk_prep_kernel(const float* qkv, const bf16* q_norm, const bf16* k_norm,
                                                           const int32_t* pos_dev, float* q, float* kb, float* vb) {
  pdl_trigger();
  __shared__ float xs[kHeadDim], red[4];
  const int t = blockIdx.x, slot = blockIdx.y, d = threadIdx.x;
  const int pos = kBatch ? pos_dev[t / kBlock] + t % kBlock : *pos_dev + t;
  pdl_wait();
  const bool is_q = slot < kHeads;
  const int head = is_q ? slot : slot - kHeads;
  const float* row = qkv + int64_t(t) * kQkv;
  float x = row[(is_q ? head * kHeadDim : kHeads * kHeadDim + head * kHeadDim) + d];
  x = head_norm_rope(x, is_q ? q_norm : k_norm, pos, xs, red);
  if (is_q) {
    q[(int64_t(t) * kHeads + head) * kHeadDim + d] = x;
  } else {
    kb[(int64_t(t) * kKvHeads + head) * kHeadDim + d] = x;
    vb[(int64_t(t) * kKvHeads + head) * kHeadDim + d] = row[(kHeads + kKvHeads) * kHeadDim + head * kHeadDim + d];
  }
}

// Draft attention, split over the keys: CTA = (KV head g, split); warp w = block row (token) w, its
// 4 query heads 4g..4g+3; a lane owns 4 of the 128 dims. Keys: the context ring (positions
// pos - nctx .. pos - 1), then the 8 block keys. Row t sees context positions >= pos + t - 2047.
constexpr int kAttnSplits = 16;
constexpr int kGqa = kHeads / kKvHeads;  // 4
constexpr int kPart = kHeadDim + 4;      // acc, m, l (padded for float4 stores)

// kBatch (P6-10): lane blockIdx.z, its block's rows from 8 z and its context ring (lanes; one layer).
template <bool kBatch = false>
__global__ void __launch_bounds__(kBlock * 32) attn_kernel(const float* q, const float* kb, const float* vb,
                                                           const bf16* ring_k, const bf16* ring_v,
                                                           const int32_t* pos_dev, float* part, DraftLanes lanes) {
  pdl_trigger();
  if constexpr (kBatch) {
    const int z = blockIdx.z;
    const int64_t r0 = int64_t(kBlock) * z;
    q += r0 * kHeads * kHeadDim, kb += r0 * kKvHeads * kHeadDim, vb += r0 * kKvHeads * kHeadDim;
    part += r0 * kHeads * kAttnSplits * kPart;
    ring_k = lane_sel(lanes.ring_k, z), ring_v = lane_sel(lanes.ring_v, z), pos_dev = lanes.pos + z;
  }
  const int g = blockIdx.x, split = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, d0 = lane * 4;
  const int pos = *pos_dev;
  const int nctx = min(pos, kWindow - 1), nk = nctx + kBlock;
  const int kps = (nk + kAttnSplits - 1) / kAttnSplits;
  const int k0 = split * kps, k1 = min(nk, k0 + kps);
  const int t = warp, lo = pos + t - (kWindow - 1);  // lowest visible context position of this row
  pdl_wait();
  const float scale = rsqrtf(float(kHeadDim));
  float qr[kGqa][4], acc[kGqa][4], m[kGqa], l[kGqa];
#pragma unroll
  for (int r = 0; r < kGqa; r++) {
    float4 a = *reinterpret_cast<const float4*>(q + (int64_t(t) * kHeads + g * kGqa + r) * kHeadDim + d0);
    qr[r][0] = a.x * scale, qr[r][1] = a.y * scale, qr[r][2] = a.z * scale, qr[r][3] = a.w * scale;
    m[r] = -INFINITY, l[r] = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) acc[r][i] = 0.f;
  }
  for (int j = k0; j < k1; j++) {
    float kx[4], vx[4];
    if (j < nctx) {
      const int p = pos - nctx + j;
      if (p < lo) continue;
      const int64_t idx = (int64_t(p % kWindow) * kKvHeads + g) * kHeadDim + d0;
      uint2 kr = *reinterpret_cast<const uint2*>(ring_k + idx), vr = *reinterpret_cast<const uint2*>(ring_v + idx);
      const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kr);
      const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vr);
      float2 a = __bfloat1622float2(kh[0]), b = __bfloat1622float2(kh[1]);
      kx[0] = a.x, kx[1] = a.y, kx[2] = b.x, kx[3] = b.y;
      a = __bfloat1622float2(vh[0]), b = __bfloat1622float2(vh[1]);
      vx[0] = a.x, vx[1] = a.y, vx[2] = b.x, vx[3] = b.y;
    } else {
      const int64_t idx = (int64_t(j - nctx) * kKvHeads + g) * kHeadDim + d0;
      float4 a = *reinterpret_cast<const float4*>(kb + idx), b = *reinterpret_cast<const float4*>(vb + idx);
      kx[0] = a.x, kx[1] = a.y, kx[2] = a.z, kx[3] = a.w;
      vx[0] = b.x, vx[1] = b.y, vx[2] = b.z, vx[3] = b.w;
    }
    float sc[kGqa];
#pragma unroll
    for (int r = 0; r < kGqa; r++) sc[r] = qr[r][0] * kx[0] + qr[r][1] * kx[1] + qr[r][2] * kx[2] + qr[r][3] * kx[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int r = 0; r < kGqa; r++) sc[r] += __shfl_xor_sync(0xffffffffu, sc[r], o);
#pragma unroll
    for (int r = 0; r < kGqa; r++) {
      const float mn = fmaxf(m[r], sc[r]);
      const float corr = __expf(m[r] - mn), p = __expf(sc[r] - mn);
      l[r] = l[r] * corr + p;
      m[r] = mn;
#pragma unroll
      for (int i = 0; i < 4; i++) acc[r][i] = acc[r][i] * corr + p * vx[i];
    }
  }
#pragma unroll
  for (int r = 0; r < kGqa; r++) {
    float* pp = part + ((int64_t(t) * kHeads + g * kGqa + r) * kAttnSplits + split) * kPart;
    *reinterpret_cast<float4*>(pp + d0) = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    if (lane == 0) pp[kHeadDim] = m[r], pp[kHeadDim + 1] = l[r];
  }
}

// Tensor-core draft attention (P7-4): the same splits and partials as attn_kernel, on bf16 MMAs.
// CTA = (KV head g, split[, lane]), 2 warps; warp w has query rows R = 16 w .. 16 w + 15 (R = 4 t + h:
// token t, query head 4 g + h). Keys in chunks of 16 (K / V rows staged in shared memory, two slots,
// rows padded to 272 bytes): S = Q K^T (Q scaled, both rounded to bf16), the masks of attn_kernel
// (context positions >= pos + t - 2047, keys of the split), an online softmax in natural-log units,
// and O += P V with P rounded to bf16 (the row sums l of the rounded P). attn_kernel walked the keys one
// at a time on CUDA cores (20 shuffles and 8 exponentials per key and warp): 214 us per layer for 8
// lanes at a full window. The drafted tokens change (lossless: the target's distribution does not).
constexpr int kTcKeys = 16;
constexpr int kTcRow = kHeadDim + 8;  // bf16 per staged row (272 bytes: ldmatrix and 32-bit loads conflict-free)

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <bool kBatch = false>
__global__ void __launch_bounds__(64) attn_tc_kernel(const float* q, const float* kb, const float* vb,
                                                     const bf16* ring_k, const bf16* ring_v, const int32_t* pos_dev,
                                                     float* part, DraftLanes lanes) {
  pdl_trigger();
  if constexpr (kBatch) {
    const int z = blockIdx.z;
    const int64_t r0 = int64_t(kBlock) * z;
    q += r0 * kHeads * kHeadDim, kb += r0 * kKvHeads * kHeadDim, vb += r0 * kKvHeads * kHeadDim;
    part += r0 * kHeads * kAttnSplits * kPart;
    ring_k = lane_sel(lanes.ring_k, z), ring_v = lane_sel(lanes.ring_v, z), pos_dev = lanes.pos + z;
  }
  __shared__ __align__(16) bf16 sk[2][kTcKeys][kTcRow], sv[2][kTcKeys][kTcRow];
  const int g = blockIdx.x, split = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int pos = *pos_dev;
  const int nctx = min(pos, kWindow - 1), nk = nctx + kBlock;
  const int kps = (nk + kAttnSplits - 1) / kAttnSplits;
  const int k0 = split * kps, k1 = min(nk, k0 + kps);
  const int nch = k1 > k0 ? (k1 - k0 + kTcKeys - 1) / kTcKeys : 0;
  // The lane's rows: R0 = 16 w + r0 and R1 = R0 + 8 (tokens t0, t1).
  const int R0 = 16 * warp + r0, R1 = R0 + 8, t0 = R0 / 4, t1 = R1 / 4;
  const int lo0 = pos + t0 - (kWindow - 1), lo1 = pos + t1 - (kWindow - 1);
  pdl_wait();
  const float scale = rsqrtf(float(kHeadDim));
  uint32_t qa[kHeadDim / 16][4];
#pragma unroll
  for (int kk = 0; kk < kHeadDim / 16; kk++)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int R = (i & 1) ? R1 : R0, col = kk * 16 + c2 + (i >> 1) * 8;
      const float* qp = q + (int64_t(R / 4) * kHeads + g * kGqa + R % 4) * kHeadDim + col;
      qa[kk][i] = pack_bf16x2(qp[0] * scale, qp[1] * scale);
    }
  auto stage = [&](int c) {
    if (c < nch) {
      const int b = c & 1;
      for (int u = tid; u < kTcKeys * (kHeadDim / 8); u += 64) {
        const int kr = u / (kHeadDim / 8), col = (u % (kHeadDim / 8)) * 8, j = k0 + c * kTcKeys + kr;
        const uint32_t dk = uint32_t(__cvta_generic_to_shared(&sk[b][kr][col]));
        const uint32_t dv = uint32_t(__cvta_generic_to_shared(&sv[b][kr][col]));
        if (j < k1 && j < nctx) {
          const int64_t idx = (int64_t((pos - nctx + j) % kWindow) * kKvHeads + g) * kHeadDim + col;
          cp_async16z(dk, ring_k + idx, true);
          cp_async16z(dv, ring_v + idx, true);
        } else if (j < k1) {  // a block key (fp32)
          const int64_t idx = (int64_t(j - nctx) * kKvHeads + g) * kHeadDim + col;
          const float4 a = *reinterpret_cast<const float4*>(kb + idx), a2 = *reinterpret_cast<const float4*>(kb + idx + 4);
          const float4 v = *reinterpret_cast<const float4*>(vb + idx), v2 = *reinterpret_cast<const float4*>(vb + idx + 4);
          *reinterpret_cast<uint4*>(&sk[b][kr][col]) = make_uint4(pack_bf16x2(a.x, a.y), pack_bf16x2(a.z, a.w),
                                                                 pack_bf16x2(a2.x, a2.y), pack_bf16x2(a2.z, a2.w));
          *reinterpret_cast<uint4*>(&sv[b][kr][col]) = make_uint4(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w),
                                                                 pack_bf16x2(v2.x, v2.y), pack_bf16x2(v2.z, v2.w));
        } else {  // past the split: zeros (P is 0 there, V must not be garbage)
          cp_async16z(dk, ring_k, false);
          cp_async16z(dv, ring_v, false);
        }
      }
    }
    cp_commit();
  };
  float o[kHeadDim / 8][4];
#pragma unroll
  for (int n = 0; n < kHeadDim / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  stage(0);
  for (int c = 0; c < nch; c++) {
    stage(c + 1);
    cp_wait<1>();
    __syncthreads();
    const int b = c & 1, jb = k0 + c * kTcKeys;
    float sc[2][4] = {};
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 16; kk++)
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const bf16* kr = &sk[b][nt * 8 + r0][kk * 16 + c2];
        const uint32_t bf[2] = {*reinterpret_cast<const uint32_t*>(kr), *reinterpret_cast<const uint32_t*>(kr + 8)};
        mma_bf16_16816(sc[nt], qa[kk], bf);
      }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int j = jb + nt * 8 + c2 + (e & 1);
        const int lo = e < 2 ? lo0 : lo1;
        const bool vis = j < k1 && (j >= nctx || pos - nctx + j >= lo);
        if (!vis) sc[nt][e] = -INFINITY;
        if (e < 2)
          mx0 = fmaxf(mx0, sc[nt][e]);
        else
          mx1 = fmaxf(mx1, sc[nt][e]);
      }
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float corr0 = mn0 == -INFINITY ? 1.f : __expf(m0 - mn0), corr1 = mn1 == -INFINITY ? 1.f : __expf(m1 - mn1);
    m0 = mn0, m1 = mn1;
    uint32_t pa[4];
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      float p[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float mn = e < 2 ? mn0 : mn1;
        p[e] = mn == -INFINITY ? 0.f : __expf(sc[nt][e] - mn);
      }
      pa[2 * nt] = pack_bf16x2(p[0], p[1]), pa[2 * nt + 1] = pack_bf16x2(p[2], p[3]);
      const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(&pa[2 * nt]);
      const float2 x0 = __bfloat1622float2(r[0]), x1 = __bfloat1622float2(r[1]);
      ps0 += x0.x + x0.y, ps1 += x1.x + x1.y;
    }
    // (A fragment of P: a0 / a1 = tile 0 rows r0 / r0 + 8, a2 / a3 = tile 1)
    const uint32_t pfrag[4] = {pa[0], pa[1], pa[2], pa[3]};
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
      ps0 += __shfl_xor_sync(0xffffffffu, ps0, x);
      ps1 += __shfl_xor_sync(0xffffffffu, ps1, x);
    }
    l0 = l0 * corr0 + ps0, l1 = l1 * corr1 + ps1;
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; n++) o[n][0] *= corr0, o[n][1] *= corr0, o[n][2] *= corr1, o[n][3] *= corr1;
    const int mat = lane / 8, rr = lane % 8;
#pragma unroll
    for (int nd = 0; nd < kHeadDim / 8; nd += 2) {
      uint32_t r[4];
      ldsm_x4_trans(r, uint32_t(__cvta_generic_to_shared(&sv[b][(mat & 1) * 8 + rr][(nd + (mat >> 1)) * 8])));
      mma_bf16_16816(o[nd], pfrag, r);
      mma_bf16_16816(o[nd + 1], pfrag, r + 2);
    }
    __syncthreads();  // the slot is staged again two chunks on
  }
  cp_wait<0>();
  float* p0 = part + ((int64_t(t0) * kHeads + g * kGqa + R0 % 4) * kAttnSplits + split) * kPart;
  float* p1 = part + ((int64_t(t1) * kHeads + g * kGqa + R1 % 4) * kAttnSplits + split) * kPart;
#pragma unroll
  for (int n = 0; n < kHeadDim / 8; n++) {
    *reinterpret_cast<float2*>(p0 + n * 8 + c2) = make_float2(o[n][0], o[n][1]);
    *reinterpret_cast<float2*>(p1 + n * 8 + c2) = make_float2(o[n][2], o[n][3]);
  }
  if (lane % 4 == 0) p0[kHeadDim] = m0, p0[kHeadDim + 1] = l0, p1[kHeadDim] = m1, p1[kHeadDim + 1] = l1;
}

bool draft_attn_tc() {
  static const bool on = !getenv("Q27_DRAFT_ATTN_TC") || atoi(getenv("Q27_DRAFT_ATTN_TC")) != 0;
  return on;
}

// Block per (row, head): merge the splits; BF16 output for o_proj.
__global__ void __launch_bounds__(kHeadDim) attn_combine_kernel(const float* part, float* out) {
  pdl_trigger();
  pdl_wait();
  const int64_t row = blockIdx.x;  // t * 32 + h
  const int d = threadIdx.x;
  const float* pp = part + row * kAttnSplits * kPart;
  float M = -INFINITY;
  for (int s = 0; s < kAttnSplits; s++) M = fmaxf(M, pp[s * kPart + kHeadDim]);
  float a = 0.f, L = 0.f;
  for (int s = 0; s < kAttnSplits; s++) {
    const float ms = pp[s * kPart + kHeadDim];
    const float c = ms == -INFINITY ? 0.f : __expf(ms - M);
    a += c * pp[s * kPart + d];
    L += c * pp[s * kPart + kHeadDim + 1];
  }
  out[row * kHeadDim + d] = a / L;
}

// Rows of a [T][K] GEMM input from fp32 / BF16 values, or from silu(gate) * up (gu rows ld apart,
// up at +K): block per row; the row maximum first (second pass re-reads from L1 / L2).
// rowmap: row r is source row rowmap[r]; kLanes (P6-10): row r of lane r / 8 is its block's row rowmap[r].
template <class In, bool kSilu, bool kLanes = false>
__global__ void __launch_bounds__(256) quant_rows_kernel(const In* x, int64_t ld, int K, ActOut ao,
                                                         const int32_t* rowmap) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.x, groups = K / 16;
  const int src = rowmap ? (kLanes ? row & ~(kBlock - 1) : 0) + rowmap[row] : row;
  auto load = [&](int g, float* v) {
    load16(x + src * ld + g * 16, v);
    if constexpr (kSilu) {
      float u[16];
      load16(x + src * ld + K + g * 16, u);
#pragma unroll
      for (int j = 0; j < 16; j++) v[j] = v[j] / (1.f + expf(-v[j])) * u[j];
    }
  };
  float rs = 1.f;
  if (ao.fmt != int(DraftQuant::kBf16)) {
    float am = 0.f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
      float v[16];
      load(g, v);
#pragma unroll
      for (int j = 0; j < 16; j++) am = fmaxf(am, fabsf(v[j]));
    }
    rs = row_scale(ao.fmt, block_max(am));
    if (threadIdx.x == 0) ao.scale[row] = rs;
  }
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    float v[16];
    load(g, v);
    act_group(ao, row, g, K, v, rs);
  }
}

// Weight quantization (load time): block per row, the row's maximum gives the per-channel scale.
__global__ void __launch_bounds__(256) quant_weight_kernel(const bf16* w, int K, int fmt, DLinear l, int row0) {
  const int r = blockIdx.x, row = row0 + r, groups = K / 16;
  float am = 0.f;
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    float v[16];
    load16(w + int64_t(r) * K + g * 16, v);
#pragma unroll
    for (int j = 0; j < 16; j++) am = fmaxf(am, fabsf(v[j]));
  }
  const float rs = row_scale(fmt, block_max(am));
  if (threadIdx.x == 0) const_cast<float*>(l.scale)[row] = rs;
  ActOut o{fmt, const_cast<void*>(l.w), const_cast<uint8_t*>(l.sf), nullptr};
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    float v[16];
    load16(w + int64_t(r) * K + g * 16, v);
    act_group(o, row, g, K, v, rs);
  }
}

// Compact NVFP4 head of the draft vocabulary: row i <- row ids[i] of the target head (packed values and
// the 16-element block scales in the swizzled layout). Thread per (row, 16-byte chunk of values).
__global__ void gather_head_kernel(const uint8_t* w, const uint8_t* sf, const int32_t* ids, int n, int K, uint8_t* dw,
                                   uint8_t* dsf) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int chunks = K / 32;  // 16 bytes = 32 values
  if (i >= int64_t(n) * chunks) return;
  const int r = int(i / chunks), c = int(i % chunks), src = ids[r];
  reinterpret_cast<uint4*>(dw + int64_t(r) * (K / 2))[c] = reinterpret_cast<const uint4*>(w + int64_t(src) * (K / 2))[c];
  const int kblocks = K / 16;
  for (int kb = 2 * c; kb < 2 * c + 2; kb++) dsf[sf_offset(r, kb, kblocks)] = sf[sf_offset(src, kb, kblocks)];
}

__device__ __forceinline__ float e2m1_value(uint32_t code) {
  constexpr float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float m = lut[code & 7];
  return code & 8 ? -m : m;
}

// Dequantization of weight rows to BF16 (prompt context through the BF16 GEMM): thread per 16 values.
__global__ void dequant_weight_kernel(DLinear l, int fmt, int row0, int n, bf16* out) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int groups = l.K / 16;
  if (i >= int64_t(n) * groups) return;
  const int r = int(i / groups), g = int(i % groups), row = row0 + r;
  const float rs = l.scale[row];
  float v[16];
  if (fmt == int(DraftQuant::kFp8)) {
    const uint8_t* q = static_cast<const uint8_t*>(l.w) + int64_t(row) * l.K + g * 16;
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = float(__half(__nv_cvt_fp8_to_halfraw(q[j], __NV_E4M3))) * rs;
  } else {
    const uint8_t* q = static_cast<const uint8_t*>(l.w) + int64_t(row) * (l.K / 2) + g * 8;
    const float sff = float(__half(__nv_cvt_fp8_to_halfraw(l.sf[sf_offset(row, g, groups)], __NV_E4M3)));
#pragma unroll
    for (int j = 0; j < 8; j++) {
      v[2 * j] = e2m1_value(q[j] & 15) * sff * rs;
      v[2 * j + 1] = e2m1_value(q[j] >> 4) * sff * rs;
    }
  }
  store16(out + int64_t(r) * l.K + g * 16, v);
}

// Block per (row, KV head), thread per dim.
// kBatch (P6-10): row r is row r % 8 of lane r / 8 (its position, count n_dev + lane and ring).
template <bool kBatch = false>
__global__ void __launch_bounds__(kHeadDim) ctx_write_kernel(const float* kv, const bf16* k_norm,
                                                             const int32_t* pos_dev, int pos_add,
                                                             const int32_t* n_dev, bf16* ring_k, bf16* ring_v,
                                                             DraftLanes lanes) {
  pdl_trigger();
  __shared__ float xs[kHeadDim], red[4];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  int rl = r;  // the row within its lane's block
  if constexpr (kBatch) {
    const int z = r / kBlock;
    rl = r % kBlock;
    pos_dev = lanes.pos + z, n_dev += z;
    ring_k = const_cast<bf16*>(lane_sel(lanes.ring_k, z)), ring_v = const_cast<bf16*>(lane_sel(lanes.ring_v, z));
  }
  pdl_wait();
  if (n_dev && rl >= *n_dev) return;
  const int pos = *pos_dev + pos_add + rl;
  const float* row = kv + int64_t(r) * 2 * kKvHeads * kHeadDim;
  const float x = head_norm_rope(row[h * kHeadDim + d], k_norm, pos, xs, red);
  const int64_t idx = (int64_t(pos % kWindow) * kKvHeads + h) * kHeadDim + d;
  ring_k[idx] = __float2bfloat16(x);
  ring_v[idx] = __float2bfloat16(row[kKvHeads * kHeadDim + h * kHeadDim + d]);
}

// Selector: CTA e computes the lattice row block score[e][p][c] (p over the previous position's
// candidates, or the anchor for e = 0); the last CTA to finish walks the path (warp 0).
// kBatch (P6-10): lane blockIdx.y with its candidates (rows 8 z + 1 .. of all lanes), anchor, sampling,
// RNG, tree, lattice and counter; it writes the 7 drafted ids only (a chain: its block's rows).
template <bool kBatch = false>
__global__ void __launch_bounds__(256) select_kernel(const float* cand_v, const int32_t* cand_i, const float* hp,
                                                     const bf16* pred_cb, const bf16* succ_cb, const int32_t* anchor,
                                                     SpecSampling smp, const uint64_t* rng, int32_t* ids_out,
                                                     DraftChoice choice, float* lattice, int32_t* counter, Lookup lk,
                                                     SelectLanes lanes) {
  pdl_trigger();
  if constexpr (kBatch) {
    const int z = blockIdx.y;
    const int64_t r0 = int64_t(kBlock) * z + 1;
    cand_v += r0 * kSelK, cand_i += r0 * kSelK, hp += r0 * kSelRank;
    anchor += kBlock * z, ids_out += kBlock * z, rng += 2 * z;
    smp = lane_sel(lanes.smp, z);
    choice.tree += z, choice.src += z;
    lattice += int64_t(z) * kDraft * kSelK * kSelK, counter += z;
  }
  __shared__ float s_succ[kSelK][kSelRank], s_pred[kSelK][kSelRank], s_hp[kSelRank];
  __shared__ bool last;
  const int e = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  pdl_wait();
  const int np = e == 0 ? 1 : kSelK;
  s_hp[tid] = hp[e * kSelRank + tid];
  for (int i = tid; i < kSelK * kSelRank / 8; i += blockDim.x) {
    const int c = i / (kSelRank / 8), r8 = (i % (kSelRank / 8)) * 8;
    const int succ_id = cand_i[e * kSelK + c];
    float v[16];
    uint4 raw = __ldg(reinterpret_cast<const uint4*>(succ_cb + int64_t(succ_id) * kSelRank + r8));
    const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __bfloat1622float2(hh[j]);
      v[2 * j] = f.x, v[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) s_succ[c][r8 + j] = v[j];
    if (c < np) {
      const int pred_id = e == 0 ? *anchor : cand_i[(e - 1) * kSelK + c];
      raw = __ldg(reinterpret_cast<const uint4*>(pred_cb + int64_t(pred_id) * kSelRank + r8));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 f = __bfloat1622float2(hh[j]);
        v[2 * j] = f.x, v[2 * j + 1] = f.y;
      }
#pragma unroll
      for (int j = 0; j < 8; j++) s_pred[c][r8 + j] = v[j];
    }
  }
  __syncthreads();
  // score[p][c] = unary[c] + sum_r pred[p][r] * hp[r] * succ[c][r]; warp per pair, lane per 8 r.
  for (int pc = warp; pc < np * kSelK; pc += blockDim.x / 32) {
    const int p = pc / kSelK, c = pc % kSelK;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kSelRank / 32; j++) {
      const int r = lane + 32 * j;
      acc += s_pred[p][r] * s_hp[r] * s_succ[c][r];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) lattice[(e * kSelK + p) * kSelK + c] = cand_v[e * kSelK + c] + acc;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(counter, 1) == kDraft - 1;
  __syncthreads();
  if (!last || warp != 0) return;
  __threadfence();
  // Draft tree (warp 0): smp.paths walks through the lattice merged into a trie; lane 0 keeps the trie.
  // Sampling: walk k draws position e from softmax(score / T) with its own uniform, so the draws made
  // from a node are i.i.d. from that node's q (recursive rejection sampling stays lossless). Greedy:
  // the paths are the smp.paths best of a beam search (one path: the argmax walk).
  SpecTree* tr = choice.tree;
  const int K = max(1, min(smp.paths, kTreePaths));
  __shared__ int8_t child[kTreeMax][kSelK];
  __shared__ int n_nodes;
  if (lane == 0) {
    n_nodes = 1;
    tr->tok[0] = *anchor;
    tr->depth[0] = 0;
    tr->path[0][0] = 0;
    tr->cidx[0] = 0;
    for (int i = 0; i < kTreeMax; i++) tr->ndraws[i] = 0, tr->owner[i] = -1;
  }
  for (int i = lane; i < kTreeMax * kSelK; i += 32) child[i / kSelK][i % kSelK] = -1;
  __syncwarp();
  // Node of `parent` for candidate idx at position e (created when new); lane 0 only.
  auto node_of = [&](int parent, int idx, int e, float q_lane0_unused) {
    int c = child[parent][idx];
    if (c < 0) {
      c = n_nodes++;
      child[parent][idx] = int8_t(c);
      tr->tok[c] = cand_i[e * kSelK + idx];
      tr->depth[c] = int8_t(e + 1);
      tr->cidx[c] = int8_t(idx);
      for (int i = 0; i <= e; i++) tr->path[c][i] = tr->path[parent][i];
      tr->path[c][e + 1] = int8_t(c);
    }
    (void)q_lane0_unused;
    return c;
  };
  // Row of scores for drawing from a node whose token is candidate prev at position e - 1 (or the anchor).
  auto row = [&](int e, int prev) {
    return lane < kSelK ? __ldcg(&lattice[(e * kSelK + prev) * kSelK + lane]) : -INFINITY;
  };
  if (smp.temperature > 0.f || K == 1) {
    for (int k = 0; k < K; k++) {
      int node = 0, prev = 0;
      for (int e = 0; e < kDraft; e++) {
        const float sc = row(e, prev);
        int idx;
        float prob;
        if (smp.temperature <= 0.f) {  // greedy chain: the argmax walk (ties: the lower candidate)
          float best = sc;
          for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
          const unsigned hit = __ballot_sync(0xffffffffu, lane < kSelK && sc == best);
          idx = __ffs(hit) - 1;
          prob = lane == idx ? 1.f : 0.f;
        } else {
          const float x = sc / smp.temperature;
          float mx = x;
          for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
          const float ex = lane < kSelK ? expf(x - mx) : 0.f;
          float sum = ex;
          for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
          prob = ex / sum;
          float cum = prob;  // inclusive scan over the lanes
          for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, cum, o);
            if (lane >= o) cum += y;
          }
          const float u = rng_uniform(rng, kRngWalk + 8 * k + e);
          const unsigned below = __ballot_sync(0xffffffffu, lane < kSelK && u >= cum);
          idx = min(__popc(below), kSelK - 1);
        }
        // Every walk through the node draws from the same q (same position, same predecessor).
        if (tr->ndraws[node] == 0 && lane < kSelK) tr->q[node][lane] = prob;
        __syncwarp();
        int c = 0;
        if (lane == 0) {
          c = node_of(node, idx, e, 0.f);
          tr->draws[node][tr->ndraws[node]++] = int8_t(c);
        }
        node = __shfl_sync(0xffffffffu, c, 0);
        prev = idx;
        __syncwarp();
      }
      if (lane == 0) tr->leaf[k] = int8_t(node);
    }
  } else {
    // Beam search over log-softmax scores on candidate-index sequences (beams: at most K distinct
    // paths); the trie is built from the K final paths only (no dead branches whose rows the verify
    // would not compute).
    __shared__ int8_t b_seq[kTreePaths][kDraft], n_seq[kTreePaths][kDraft];
    __shared__ float b_score[kTreePaths], c_score[kTreePaths * kSelK];
    int nb = 1;
    if (lane == 0) b_score[0] = 0.f;
    __syncwarp();
    for (int e = 0; e < kDraft; e++) {
      for (int b = 0; b < nb; b++) {
        const float x = row(e, e == 0 ? 0 : b_seq[b][e - 1]);
        float mx = x;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float ex = lane < kSelK ? expf(x - mx) : 0.f;
        float sum = ex;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane < kSelK) c_score[b * kSelK + lane] = b_score[b] + (x - mx) - logf(sum);
      }
      __syncwarp();
      if (lane == 0) {
        float ns[kTreePaths];
        const int keep = min(K, nb * kSelK);
        for (int r = 0; r < keep; r++) {  // r-th best (ties: lower beam, then lower candidate)
          int best = -1;
          for (int i = 0; i < nb * kSelK; i++)
            if (c_score[i] > -INFINITY && (best < 0 || c_score[i] > c_score[best])) best = i;
          const int bb = best / kSelK;
          for (int i = 0; i < e; i++) n_seq[r][i] = b_seq[bb][i];
          n_seq[r][e] = int8_t(best % kSelK);
          ns[r] = c_score[best];
          c_score[best] = -INFINITY;
        }
        for (int r = 0; r < keep; r++) {
          for (int i = 0; i <= e; i++) b_seq[r][i] = n_seq[r][i];
          b_score[r] = ns[r];
        }
        nb = keep;
      }
      nb = __shfl_sync(0xffffffffu, nb, 0);
      __syncwarp();
    }
    if (lane == 0) {
      for (int k = 0; k < K; k++) {
        const int bk = min(k, nb - 1);
        int node = 0;
        for (int e = 0; e < kDraft; e++) {
          const int idx = b_seq[bk][e];
          const bool fresh = child[node][idx] < 0;
          const int c = node_of(node, idx, e, 0.f);
          if (fresh) tr->draws[node][tr->ndraws[node]++] = int8_t(c);
          node = c;
        }
        tr->leaf[k] = int8_t(node);
      }
    }
    for (int i = lane; i < kTreeMax * kSelK; i += 32) tr->q[i / kSelK][i % kSelK] = 0.f;  // (unused when greedy)
  }
  __syncwarp();
  if (lane == 0) {
    const int n = n_nodes;
    tr->n = n;
    tr->npaths = K;
    for (int p = 0; p < K; p++)  // owner: the first path through the node
      for (int i = 0; i <= tr->depth[tr->leaf[p]]; i++) {
        const int nd = tr->path[tr->leaf[p]][i];
        if (tr->owner[nd] < 0) tr->owner[nd] = int8_t(p);
      }
    for (int t = n; t < kTreeMax; t++) {  // padding rows: depth-0 copies of the anchor
      tr->tok[t] = tr->tok[0];
      tr->depth[t] = 0;
      tr->path[t][0] = int8_t(t);
      tr->cidx[t] = 0;
    }
    for (int t = 1; t < (kBatch ? kBlock : kTreeMax); t++) ids_out[t] = tr->tok[t];
    *choice.src = 0;
    if (lk.on) *lk.best = 0;  // (lookup chains are not merged into the tree yet)
  }
  if (lane == 0) *counter = 0;
}

}  // namespace

void draft_norm(const float* x, const bf16* w, int T, float* out, ActOut ao, bf16* hb, cudaStream_t s) {
  launch_pdl(norm_kernel, dim3(T), dim3(kNormThreads), 0, s, x, w, out, ao, hb);
  Q27_CUDA(cudaGetLastError());
}

void draft_conv_in(const float* h, const float* coef, const bf16* base, int T, ActOut ao, cudaStream_t s) {
  launch_pdl(conv_in_kernel, dim3(T), dim3(kNormThreads), 0, s, h, coef, base, ao);
  Q27_CUDA(cudaGetLastError());
}

void draft_conv_out_norm(const float* y, const float* coef, const bf16* base, int T, float* x, const bf16* w,
                         float* h, ActOut ao, bf16* hb, cudaStream_t s) {
  launch_pdl(conv_out_norm_kernel, dim3(T), dim3(kNormThreads), 0, s, y, coef, base, x, w, h, ao, hb);
  Q27_CUDA(cudaGetLastError());
}

void draft_qk_prep(const float* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos, float* q, float* kb,
                   float* vb, cudaStream_t s) {
  launch_pdl(qk_prep_kernel<false>, dim3(kBlock, kHeads + kKvHeads), dim3(kHeadDim), 0, s, qkv, q_norm, k_norm, pos,
             q, kb, vb);
  Q27_CUDA(cudaGetLastError());
}

size_t draft_attention_part_floats() { return size_t(kBlock) * kHeads * kAttnSplits * kPart; }

void draft_attention(const float* q, const float* kb, const float* vb, const bf16* ring_k, const bf16* ring_v,
                     const int32_t* pos, float* part, float* out, cudaStream_t s) {
  if (draft_attn_tc())
    launch_pdl(attn_tc_kernel<false>, dim3(kKvHeads, kAttnSplits), dim3(64), 0, s, q, kb, vb, ring_k, ring_v, pos, part,
               DraftLanes{});
  else
    launch_pdl(attn_kernel<false>, dim3(kKvHeads, kAttnSplits), dim3(kBlock * 32), 0, s, q, kb, vb, ring_k, ring_v, pos,
               part, DraftLanes{});
  Q27_CUDA(cudaGetLastError());
  launch_pdl(attn_combine_kernel, dim3(kBlock * kHeads), dim3(kHeadDim), 0, s, static_cast<const float*>(part), out);
  Q27_CUDA(cudaGetLastError());
}

void draft_quant_rows(const float* x, int T, int K, ActOut ao, cudaStream_t s) {
  launch_pdl(quant_rows_kernel<float, false>, dim3(T), dim3(256), 0, s, x, int64_t(K), K, ao,
             static_cast<const int32_t*>(nullptr));
  Q27_CUDA(cudaGetLastError());
}

void draft_quant_rows(const bf16* x, int T, int K, ActOut ao, cudaStream_t s, const int32_t* rowmap) {
  launch_pdl(quant_rows_kernel<bf16, false>, dim3(T), dim3(256), 0, s, x, int64_t(K), K, ao, rowmap);
  Q27_CUDA(cudaGetLastError());
}

void draft_quant_rows_lanes(const bf16* x, int T, int K, ActOut ao, cudaStream_t s, const int32_t* rowmap) {
  launch_pdl(quant_rows_kernel<bf16, false, true>, dim3(T), dim3(256), 0, s, x, int64_t(K), K, ao, rowmap);
  Q27_CUDA(cudaGetLastError());
}

void draft_silu_mul(const float* gu, int64_t ld, int T, int I, ActOut ao, cudaStream_t s) {
  launch_pdl(quant_rows_kernel<float, true>, dim3(T), dim3(256), 0, s, gu, ld, I, ao,
             static_cast<const int32_t*>(nullptr));
  Q27_CUDA(cudaGetLastError());
}

void draft_quant_weight(const bf16* w, int n, int K, DraftQuant fmt, const DLinear& l, int row0, cudaStream_t s) {
  quant_weight_kernel<<<n, 256, 0, s>>>(w, K, int(fmt), l, row0);
  Q27_CUDA(cudaGetLastError());
}

void draft_gather_head(const uint8_t* w, const uint8_t* sf, const int32_t* ids, int n, int K, uint8_t* dw, uint8_t* dsf,
                       cudaStream_t s) {
  const int64_t threads = int64_t(n) * (K / 32);
  gather_head_kernel<<<unsigned((threads + 255) / 256), 256, 0, s>>>(w, sf, ids, n, K, dw, dsf);
  Q27_CUDA(cudaGetLastError());
}

void draft_dequant_weight(const DLinear& l, DraftQuant fmt, int row0, int n, bf16* out, cudaStream_t s) {
  const int64_t threads = int64_t(n) * (l.K / 16);
  dequant_weight_kernel<<<unsigned((threads + 255) / 256), 256, 0, s>>>(l, int(fmt), row0, n, out);
  Q27_CUDA(cudaGetLastError());
}

void draft_ctx_write(const float* kv, int rows, const bf16* k_norm, const int32_t* pos, int pos_add,
                     const int32_t* n_dev, bf16* ring_k, bf16* ring_v, cudaStream_t s) {
  if (rows == 0) return;
  launch_pdl(ctx_write_kernel<false>, dim3(rows, kKvHeads), dim3(kHeadDim), 0, s, kv, k_norm, pos, pos_add, n_dev,
             ring_k, ring_v, DraftLanes{});
  Q27_CUDA(cudaGetLastError());
}

void draft_select(const float* cand_v, const int32_t* cand_i, const float* hp, const bf16* pred_cb,
                  const bf16* succ_cb, const int32_t* anchor, SpecSampling smp, const uint64_t* rng,
                  int32_t* ids_out, DraftChoice choice, float* lattice, int32_t* counter, const Lookup& lk,
                  cudaStream_t s) {
  launch_pdl(select_kernel<false>, dim3(kDraft), dim3(256), 0, s, cand_v, cand_i, hp, pred_cb, succ_cb, anchor, smp,
             rng, ids_out, choice, lattice, counter, lk, SelectLanes{});
  Q27_CUDA(cudaGetLastError());
}

void draft_qk_prep_lanes(int lanes, const float* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos,
                         float* q, float* kb, float* vb, cudaStream_t s) {
  launch_pdl(qk_prep_kernel<true>, dim3(kBlock * lanes, kHeads + kKvHeads), dim3(kHeadDim), 0, s, qkv, q_norm, k_norm,
             pos, q, kb, vb);
  Q27_CUDA(cudaGetLastError());
}

void draft_attention_lanes(const DraftLanes& lanes, const float* q, const float* kb, const float* vb, float* part,
                           float* out, cudaStream_t s) {
  if (draft_attn_tc())
    launch_pdl(attn_tc_kernel<true>, dim3(kKvHeads, kAttnSplits, lanes.n), dim3(64), 0, s, q, kb, vb, lanes.ring_k[0],
               lanes.ring_v[0], lanes.pos, part, lanes);
  else
    launch_pdl(attn_kernel<true>, dim3(kKvHeads, kAttnSplits, lanes.n), dim3(kBlock * 32), 0, s, q, kb, vb,
               lanes.ring_k[0], lanes.ring_v[0], lanes.pos, part, lanes);
  Q27_CUDA(cudaGetLastError());
  launch_pdl(attn_combine_kernel, dim3(kBlock * kHeads * lanes.n), dim3(kHeadDim), 0, s, static_cast<const float*>(part),
             out);
  Q27_CUDA(cudaGetLastError());
}

void draft_ctx_write_lanes(const DraftLanes& lanes, const float* kv, const bf16* k_norm, const int32_t* n_dev,
                           cudaStream_t s) {
  launch_pdl(ctx_write_kernel<true>, dim3(kBlock * lanes.n, kKvHeads), dim3(kHeadDim), 0, s, kv, k_norm, lanes.pos, 0,
             n_dev, const_cast<bf16*>(lanes.ring_k[0]), const_cast<bf16*>(lanes.ring_v[0]), lanes);
  Q27_CUDA(cudaGetLastError());
}

void draft_select_lanes(int lanes, const SelectLanes& sl, const float* cand_v, const int32_t* cand_i, const float* hp,
                        const bf16* pred_cb, const bf16* succ_cb, const int32_t* anchor, const uint64_t* rng,
                        int32_t* ids_out, DraftChoice choice, float* lattice, int32_t* counter, cudaStream_t s) {
  launch_pdl(select_kernel<true>, dim3(kDraft, lanes), dim3(256), 0, s, cand_v, cand_i, hp, pred_cb, succ_cb, anchor,
             sl.smp[0], rng, ids_out, choice, lattice, counter, Lookup{}, sl);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
