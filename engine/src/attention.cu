// Full-attention layers, Phase 1 versions: q/k norm + partial RoPE + paged KV write, and a plain
// flash-attention style kernel on CUDA cores (fp32 math, online softmax), no split-KV.
#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include <cuda_fp8.h>

#include "common.h"
#include "kernels.h"
#include "launch.cuh"
#include "mma.cuh"
#include "spec.h"

namespace q27 {
namespace {

using namespace dims;

template <class T>
__device__ T to_kv(float x);
template <>
__device__ bf16 to_kv<bf16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ uint8_t to_kv<uint8_t>(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ float from_kv(bf16 x) { return __bfloat162float(x); }
__device__ float from_kv(uint8_t x) { return float(__half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3))); }

__device__ __forceinline__ int64_t kv_row(const int32_t* page_table, int pos, int head) {
  return (int64_t(page_table[pos / kPageTokens]) * kPageTokens + pos % kPageTokens) * kKvHeads + head;
}
__device__ __forceinline__ int64_t kv_index(const int32_t* page_table, int pos, int head) {
  return kv_row(page_table, pos, head) * kHeadDim;
}

// NVFP4 KV: per (token, KV head) one 144-byte row, 128 bytes of e2m1 codes (dim 2i in the low nibble of
// byte i) and 16 e4m3 scales (one per 16 dims, x = e2m1 * scale; global scale 1 as the FP8 cache).
// Written with the numerics of quant_fp4 (scale = e4m3(amax / 6), codes = rne_e2m1(x / scale)).
struct Fp4Kv {
  uint8_t x;
};
template <class KV>
constexpr bool kKvFp4 = std::is_same_v<KV, Fp4Kv>;
template <class KV>
constexpr bool kKvFp8 = std::is_same_v<KV, uint8_t>;
template <class KV>
constexpr int kKvRowBytes = kKvFp4<KV> ? kHeadDim / 2 + kHeadDim / 16 : kHeadDim * int(sizeof(KV));
template <class KV>
__device__ __forceinline__ const uint8_t* kv_row_ptr(const KV* base, int64_t row) {
  return reinterpret_cast<const uint8_t*>(base) + row * kKvRowBytes<KV>;
}

__device__ __forceinline__ uint8_t cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(odd), "f"(even));
  return uint8_t(r);
}
// Writes dim d (thread d of 256) of one NVFP4 row: the 16 dims of a scale block are 16 adjacent lanes.
__device__ __forceinline__ void store_fp4(uint8_t* row, float x, int d) {
  float a = fabsf(x);
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(a / 6.f, __NV_SATFINITE, __NV_E4M3);
  const float sf = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
  const float inv = sf != 0.f ? 1.f / sf : 0.f;
  const float odd = __shfl_xor_sync(0xffffffffu, x, 1);
  if (d % 2 == 0) row[d / 2] = cvt_e2m1x2(x * inv, odd * inv);
  if (d % 16 == 0) row[kHeadDim / 2 + d / 16] = sfq;
}
// 32 NVFP4 values (16 bytes of codes w) with the scales of their two 16-dim blocks -> 16 f16 pairs.
// e2m1 * e4m3 has at most 5 significant bits and fits f16: exact (the NVFP4 cache feeds f16 MMAs).
__device__ __forceinline__ void fp4x32_to_f16(const uint32_t* w, uint32_t scales, uint32_t* out) {
  __half2 sc[2];
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const __half h = __half(__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(scales >> (8 * b)), __NV_E4M3));
    sc[b] = __halves2half2(h, h);
  }
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint32_t h;
    asm("{\n .reg .b8 t;\n cvt.u8.u32 t, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(h) : "r"((w[i / 4] >> (8 * (i % 4))) & 0xffu));
    const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&h), sc[i / 8]);
    out[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}
// 8 dims d0 .. d0 + 7 (d0 % 8 == 0) of an NVFP4 row as floats.
__device__ __forceinline__ void load8_fp4(const uint8_t* row, int d0, float* x) {
  const uint32_t w = __ldg(reinterpret_cast<const uint32_t*>(row + d0 / 2));
  const float sf = float(__half(__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(__ldg(row + kHeadDim / 2 + d0 / 16)), __NV_E4M3)));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t h;
    asm("{\n .reg .b8 t;\n cvt.u8.u32 t, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(h) : "r"((w >> (8 * i)) & 0xffu));
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    x[2 * i] = f.x * sf, x[2 * i + 1] = f.y * sf;
  }
}

// Block per (token, head slot): slots 0..23 are query heads, 24..27 key heads (and their values).
// kBatch (P6-10): row t belongs to lane t / kLaneRows (its position, page table and rotary delta).
template <class KV, bool kBatch = false>
__global__ void attn_prep_kernel(const float* qg, int ldq, const float* k, const float* v, int ldkv,
                                 const bf16* q_norm, const bf16* k_norm, const int32_t* pos_dev, KV* kc, KV* vc,
                                 const int32_t* page_table, float* q_out, const SpecTree* tree, RopePos rope,
                                 SeqLanes lanes) {
  pdl_trigger();
  int t = blockIdx.x, slot = blockIdx.y, d = threadIdx.x;
  int tl = t;  // the row's index in its block of new tokens
  if constexpr (kBatch) {
    const int z = t / kLaneRows;
    tl = t % kLaneRows;
    pos_dev = lanes.pos + z, page_table = lane_sel(lanes.page_table, z), rope.delta = lanes.rope_delta + z;
  }
  bool is_q = slot < kHeads;
  int head = is_q ? slot : slot - kHeads;
  const float* src = is_q ? qg + int64_t(t) * ldq + head * (2 * kHeadDim) : k + int64_t(t) * ldkv + head * kHeadDim;
  // Before the PDL wait: the norm weight, the position (advanced only at the end of a step / chunk,
  // which the host synchronizes, see Model::forward_sample) and the rotation.
  // A tree node (speculative verify) is stored at slot pos0 + t but rotated at its depth.
  const float wd = 1.f + __bfloat162float((is_q ? q_norm : k_norm)[d]);
  // (The rope delta and positions are set before the request's first chunk, like the position.)
  const int pos = *pos_dev + tl;
  const int delta = rope.delta ? *rope.delta : 0;
  float rc = 1.f, rs = 0.f;
  auto rotation = [&](int p) {
    if (d < kRotaryDim) {
      constexpr int half = kRotaryDim / 2;
      int fi = d % half;
      if (rope.mpos) p = rope.mpos[3 * t + fi % 3];  // M-RoPE: (t, h, w) interleaved
      float inv_freq = 1.f / powf(kRopeTheta, float(2 * fi) / float(kRotaryDim));
      float f = float(p) * inv_freq;
      rc = cosf(f), rs = sinf(f);
    }
  };
  if (!tree) rotation(pos + delta);
  pdl_wait();
  if (tree) rotation(*pos_dev + tree->depth[t] + delta);
  __shared__ float xs[kHeadDim];
  __shared__ float red[kHeadDim / 32];
  float x = src[d];
  float ss = x * x;
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (d % 32 == 0) red[d / 32] = ss;
  __syncthreads();
  ss = 0.f;
#pragma unroll
  for (int i = 0; i < kHeadDim / 32; i++) ss += red[i];
  x = x * rsqrtf(ss / kHeadDim + kEps) * wd;
  xs[d] = x;
  __syncthreads();
  if (d < kRotaryDim) {
    constexpr int half = kRotaryDim / 2;
    float rot = d < half ? -xs[d + half] : xs[d - half];
    x = x * rc + rot * rs;
  }
  if (is_q) {
    q_out[(int64_t(t) * kHeads + head) * kHeadDim + d] = x;
  } else if constexpr (kKvFp4<KV>) {
    const int64_t row = kv_row(page_table, pos, head);
    store_fp4(const_cast<uint8_t*>(kv_row_ptr(kc, row)), x, d);
    store_fp4(const_cast<uint8_t*>(kv_row_ptr(vc, row)), v[int64_t(t) * ldkv + head * kHeadDim + d], d);
  } else {
    int64_t idx = kv_index(page_table, pos, head) + d;
    kc[idx] = to_kv<KV>(x);
    vc[idx] = to_kv<KV>(v[int64_t(t) * ldkv + head * kHeadDim + d]);
  }
}

constexpr int kBq = 16, kBk = 32, kPad = kHeadDim + 1;
constexpr size_t kAttnSmem = (kBq * kPad + kBk * kPad + kBq * kBk + 2 * kBq) * sizeof(float);

// Block per (query head, 16-query tile), 256 threads. Thread t: scores for row t/16, keys t%16 and
// t%16 + 16; output column t for all 16 rows.
template <class KV>
__global__ void __launch_bounds__(256) attention_kernel(const float* q, const float* qg, int T, int pos0, const KV* kc,
                                                        const KV* vc, const int32_t* page_table, float* out) {
  extern __shared__ float smem[];
  float* sQ = smem;               // [kBq][kPad]
  float* sKV = sQ + kBq * kPad;   // [kBk][kPad]
  float* sP = sKV + kBk * kPad;   // [kBq][kBk]
  float* sAlpha = sP + kBq * kBk; // [kBq]
  float* sL = sAlpha + kBq;       // [kBq]
  int h = blockIdx.x, q0 = blockIdx.y * kBq, tid = threadIdx.x;
  int g = h / (kHeads / kKvHeads);
  int nq = min(kBq, T - q0);
  for (int i = tid; i < kBq * kHeadDim; i += blockDim.x) {
    int r = i / kHeadDim, d = i % kHeadDim;
    sQ[r * kPad + d] = r < nq ? q[(int64_t(q0 + r) * kHeads + h) * kHeadDim + d] : 0.f;
  }
  int row = tid / 16, kj = tid % 16;
  int qpos = row < nq ? pos0 + q0 + row : -1;
  float m = -INFINITY, l = 0.f;
  float acc[kBq];
#pragma unroll
  for (int r = 0; r < kBq; r++) acc[r] = 0.f;
  int nkeys = pos0 + q0 + nq;  // keys 0 .. nkeys-1 are visible to some row of this tile
  const float scale = rsqrtf(float(kHeadDim));
  int lj = tid / 8, ld = (tid % 8) * 32;  // loader: key lj, dims ld .. ld+31
  for (int k0 = 0; k0 < nkeys; k0 += kBk) {
    __syncthreads();  // previous tile's V reads are done (and sQ is visible on the first pass)
    if (k0 + lj < nkeys) {
      const KV* src = kc + kv_index(page_table, k0 + lj, g) + ld;
#pragma unroll
      for (int i = 0; i < 32; i++) sKV[lj * kPad + ld + i] = from_kv(src[i]);
    }
    __syncthreads();
    float s[2];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int j = kj + 16 * e, kpos = k0 + j;
      float dot = 0.f;
      for (int d = 0; d < kHeadDim; d++) dot += sQ[row * kPad + d] * sKV[j * kPad + d];
      s[e] = (kpos <= qpos && kpos < nkeys) ? dot * scale : -INFINITY;
    }
    float tmax = fmaxf(s[0], s[1]);
    for (int o = 8; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    float mnew = fmaxf(m, tmax);
    float alpha = mnew == -INFINITY ? 1.f : expf(m - mnew);
    float p0 = mnew == -INFINITY ? 0.f : expf(s[0] - mnew);
    float p1 = mnew == -INFINITY ? 0.f : expf(s[1] - mnew);
    float psum = p0 + p1;
    for (int o = 8; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
    l = l * alpha + psum;
    m = mnew;
    sP[row * kBk + kj] = p0;
    sP[row * kBk + kj + 16] = p1;
    if (kj == 0) sAlpha[row] = alpha;
    __syncthreads();  // scores done with sKV; P and alpha visible
    if (k0 + lj < nkeys) {
      const KV* src = vc + kv_index(page_table, k0 + lj, g) + ld;
#pragma unroll
      for (int i = 0; i < 32; i++) sKV[lj * kPad + ld + i] = from_kv(src[i]);
    } else {
#pragma unroll
      for (int i = 0; i < 32; i++) sKV[lj * kPad + ld + i] = 0.f;
    }
    __syncthreads();
    int d = tid;
#pragma unroll
    for (int r = 0; r < kBq; r++) {
      float a = acc[r] * sAlpha[r];
#pragma unroll 8
      for (int j = 0; j < kBk; j++) a += sP[r * kBk + j] * sKV[j * kPad + d];
      acc[r] = a;
    }
  }
  if (kj == 0) sL[row] = l;
  __syncthreads();
  int d = tid;
  for (int r = 0; r < nq; r++) {
    int t = q0 + r;
    float gate = qg[(int64_t(t) * kHeads + h) * (2 * kHeadDim) + kHeadDim + d];
    out[(int64_t(t) * kHeads + h) * kHeadDim + d] = acc[r] / sL[r] * (1.f / (1.f + expf(-gate)));
  }
}


// ---------------------------------------------------------------- decode (split-KV)
//
// For a few new tokens (T <= 8): CTA = (KV head g, token t, key range); its 6 rows are the query
// heads 6g..6g+5 of token t. Each warp walks keys warp, warp + 8, ... of the range; a lane owns 8 of
// the 256 dims, so one key row (K or V) is a single coalesced 512-byte (bf16) / 256-byte (fp8)
// warp load. Per-warp online softmax, then the CTA merges its warps and writes an unnormalized
// partial (acc[256], m, l) per (row, split); attention_combine merges the splits and applies the gate.

constexpr int kGqa = kHeads / kKvHeads;  // 6
constexpr int kDecWarps = 8;
constexpr int kDecUnroll = 2;            // keys in flight per warp

__device__ __forceinline__ void load8(const bf16* p, float* x) {
  uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __bfloat1622float2(h[i]);
    x[2 * i] = f.x, x[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void load8(const uint8_t* p, float* x) {
  uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = from_kv(b[i]);
}

struct DecPartial {
  float* acc;  // [rows][splits][256]
  float* ml;   // [rows][splits][2]
};

// Keys per split for `keys` keys (the split count is fixed so that the launch can live in a CUDA
// graph; the ranges follow the length). Splits from ceil(keys / keys_per_split) on are empty.
__device__ __forceinline__ int keys_per_split(int keys, int splits, int min_keys) {
  return max(min_keys, (keys + splits - 1) / splits + 63) & ~63;
}
constexpr int kDecPrefetchKeys = 4;  // per warp, prefetched to L2 before the PDL wait (Q27_ATTN_PREFETCH)

template <class KV>
__global__ void __launch_bounds__(kDecWarps * 32) attention_decode_kernel(const float* q, const int32_t* pos_dev, int T,
                                                                          const KV* kc, const KV* vc,
                                                                          const int32_t* page_table, int splits,
                                                                          int min_keys, int prefetch_keys,
                                                                          DecPartial part, const SpecTree* tree) {
  pdl_trigger();
  __shared__ float s_acc[kDecWarps][kGqa][kHeadDim];
  __shared__ float s_m[kDecWarps][kGqa], s_l[kDecWarps][kGqa];
  const int g = blockIdx.x, cta = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, d0 = lane * 8;
  // Work items (split, token), split-major, strided over the CTAs: each token splits its own keys
  // 0..qpos (as if decoded alone, whatever T is), and the launch stays at `splits` CTAs per head.
  // The position is advanced only at the end of a step / chunk (host-synchronized): read before the wait.
  // A tree node t (speculative verify) sees the committed keys and then its ancestors, as if its path
  // were decoded alone: logical key pos0 + i is slot pos0 + path[t][i]. Before the PDL wait the depths
  // are unknown (this step's tree), so the chain bound (depth <= t) picks the CTA's first item.
  const int pos0 = *pos_dev, items = T * splits;
  const int8_t* depth = nullptr;
  auto range = [&](int item, int& t, int& split, int& k_begin, int& k_end) {
    split = item / T, t = item % T;
    const int keys = pos0 + (depth ? depth[t] : t) + 1, kps = keys_per_split(keys, splits, min_keys);
    k_begin = split * kps, k_end = min(keys, k_begin + kps);
    return k_begin < keys;  // empty splits: attention_combine does not read them
  };
  int first = -1;
  for (int i = cta; i < items && first < 0; i += splits) {
    int t, sp, kb, ke;
    if (range(i, t, sp, kb, ke)) first = i;
  }
  if (first < 0) return;
  {
    // Keys before pos0 were written in earlier steps: pull this warp's first ones into L2 while the
    // previous kernels finish (the new keys pos0 + t come from attn_prep, after the wait).
    int t, sp, k_begin, k_end;
    range(first, t, sp, k_begin, k_end);
    if (lane < (kKvRowBytes<KV> + 127) / 128) {
      for (int i = 0, j = k_begin + warp; i < prefetch_keys && j < min(k_end, pos0); i++, j += kDecWarps) {
        const int64_t row = kv_row(page_table, j, g);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(kv_row_ptr(kc, row) + lane * 128));
        asm volatile("prefetch.global.L2 [%0];" ::"l"(kv_row_ptr(vc, row) + lane * 128));
      }
    }
  }
  pdl_wait();
  if (tree) depth = tree->depth;
  const float scale = rsqrtf(float(kHeadDim));
  for (int item = first; item < items; item += splits) {
    int t, split, k_begin, k_end;
    if (!range(item, t, split, k_begin, k_end)) continue;
    const int8_t* tpath = tree ? tree->path[t] : nullptr;
    auto slot = [&](int j) { return j < pos0 || !tpath ? j : pos0 + tpath[j - pos0]; };
    __syncthreads();  // the previous item's merge has read s_acc
    float qr[kGqa][8], acc[kGqa][8], m[kGqa], l[kGqa];
#pragma unroll
    for (int r = 0; r < kGqa; r++) {
      const float* src = q + (int64_t(t) * kHeads + g * kGqa + r) * kHeadDim + d0;
      float4 a = *reinterpret_cast<const float4*>(src), b = *reinterpret_cast<const float4*>(src + 4);
      qr[r][0] = a.x * scale, qr[r][1] = a.y * scale, qr[r][2] = a.z * scale, qr[r][3] = a.w * scale;
      qr[r][4] = b.x * scale, qr[r][5] = b.y * scale, qr[r][6] = b.z * scale, qr[r][7] = b.w * scale;
      m[r] = -INFINITY, l[r] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) acc[r][i] = 0.f;
    }

    for (int j0 = k_begin + warp; j0 < k_end; j0 += kDecWarps * kDecUnroll) {
      float kx[kDecUnroll][8], vx[kDecUnroll][8];
#pragma unroll
      for (int u = 0; u < kDecUnroll; u++) {
        int j = j0 + u * kDecWarps;
        if (j < k_end) {
          if constexpr (kKvFp4<KV>) {
            const int64_t row = kv_row(page_table, slot(j), g);
            load8_fp4(kv_row_ptr(kc, row), d0, kx[u]);
            load8_fp4(kv_row_ptr(vc, row), d0, vx[u]);
          } else {
            int64_t idx = kv_index(page_table, slot(j), g) + d0;
            load8(kc + idx, kx[u]);
            load8(vc + idx, vx[u]);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kDecUnroll; u++) {
        if (j0 + u * kDecWarps >= k_end) break;
        float sc[kGqa];
#pragma unroll
        for (int r = 0; r < kGqa; r++) {
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < 8; i++) dot += qr[r][i] * kx[u][i];
          sc[r] = dot;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
          for (int r = 0; r < kGqa; r++) sc[r] += __shfl_xor_sync(0xffffffffu, sc[r], o);
#pragma unroll
        for (int r = 0; r < kGqa; r++) {
          float mn = fmaxf(m[r], sc[r]);
          float corr = __expf(m[r] - mn), p = __expf(sc[r] - mn);
          l[r] = l[r] * corr + p;
          m[r] = mn;
#pragma unroll
          for (int i = 0; i < 8; i++) acc[r][i] = acc[r][i] * corr + p * vx[u][i];
        }
      }
    }

    // Merge the warps.
#pragma unroll
    for (int r = 0; r < kGqa; r++) {
#pragma unroll
      for (int i = 0; i < 8; i++) s_acc[warp][r][d0 + i] = acc[r][i];
      if (lane == 0) s_m[warp][r] = m[r], s_l[warp][r] = l[r];
    }
    __syncthreads();
    const int d = threadIdx.x;  // 256 threads = 256 dims
#pragma unroll
    for (int r = 0; r < kGqa; r++) {
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < kDecWarps; w++) M = fmaxf(M, s_m[w][r]);
      float a = 0.f, L = 0.f;
#pragma unroll
      for (int w = 0; w < kDecWarps; w++) {
        float c = s_m[w][r] == -INFINITY ? 0.f : __expf(s_m[w][r] - M);
        a += c * s_acc[w][r][d];
        L += c * s_l[w][r];
      }
      int64_t row = int64_t(t) * kHeads + g * kGqa + r;
      part.acc[(row * splits + split) * kHeadDim + d] = a;
      if (d == 0) part.ml[(row * splits + split) * 2] = M, part.ml[(row * splits + split) * 2 + 1] = L;
    }
  }
}

// Tensor-core version for a chain of T <= 8 new tokens (decode and the speculative verify, BF16 KV).
// The keys are cut into fixed 32-key chunks (within a 64-token page); chunk c belongs to split
// c % splits, and a split walks its chunks in ascending order. CTA = (KV head, split): each chunk is
// staged in shared memory once for all tokens (cp.async by producer warps, below; 16-byte columns XOR-swizzled
// by row % 8 so that ldmatrix is conflict-free): S^T = K Q^T (mma m16n8k16 with the keys as rows and the
// (token, query head) pairs as columns, 8 to an n8 tile), the online softmax over the chunk per column,
// P^T into the B fragment with movmatrix, O^T += V^T P^T. A column's arithmetic does not depend on T,
// its tile or the other columns, so a verified row is bit-identical to the decode of the same token.
// Q and P are rounded to BF16 (as in flash attention).
// FP8 KV: the chunk is staged as is (half the bytes) and converted to the BF16 layout in shared memory
// before the MMAs (E4M3 -> BF16 is exact), so the arithmetic is the BF16 kernel's on the same values.
// NVFP4 KV: the same with the 144-byte rows, converted to f16 (exact) and f16 MMAs (Q and P in f16).
constexpr int kTcChunk = 32;
constexpr int kTcMaxT = 8;
constexpr int kTcRowBytes = kHeadDim * 2;
constexpr int kTcStageBytes = 2 * kTcChunk * kTcRowBytes;  // K, then V

__device__ __forceinline__ uint32_t tc_swz(int row, int unit) { return row * kTcRowBytes + ((unit ^ (row & 7)) << 4); }

// A staged NVFP4 chunk (raw rows of 144 bytes, kRows of K then kRows of V, at src) -> E4M3 rows of 256
// bytes in the sw8 layout (K then V, at dst), rounded to nearest from the exact f16 values (FP8 MMAs).
template <int kRows, int kThreads>
__device__ __forceinline__ void fp4_chunk_to_e4m3(uint32_t src, uint32_t dst, int tid);

// A staged NVFP4 chunk (raw rows of 144 bytes, kRows of K then kRows of V, at src) -> the swizzled
// 16-bit layout of the BF16 chunk (tc_swz, K then V, at dst) in f16. Thread = 16 bytes of codes (32 dims).
template <int kRows>
__device__ __forceinline__ void fp4_unit_to_f16(uint32_t src, uint32_t dst, int u) {
  constexpr int kRow = kKvRowBytes<Fp4Kv>;
  const int kv = u / (kRows * 8), r = u / 8 % kRows, col = u % 8;
  const uint32_t row = src + (kv * kRows + r) * kRow;
  uint32_t w[4], sc;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
               : "r"(row + col * 16));
  asm volatile("ld.shared.u16 %0, [%1];" : "=r"(sc) : "r"(row + kHeadDim / 2 + 2 * col));
  uint32_t b[16];
  fp4x32_to_f16(w, sc, b);
  const uint32_t d = dst + kv * kRows * kTcRowBytes;
#pragma unroll
  for (int x = 0; x < 4; x++)
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(d + tc_swz(r, 4 * col + x)), "r"(b[4 * x]),
                 "r"(b[4 * x + 1]), "r"(b[4 * x + 2]), "r"(b[4 * x + 3]));
}
template <int kRows, int kThreads>
__device__ __forceinline__ void fp4_chunk_to_f16(uint32_t src, uint32_t dst, int tid) {
  static_assert(2 * kRows * 8 % kThreads == 0, "whole passes");
#pragma unroll
  for (int j = 0; j < 2 * kRows * 8 / kThreads; j++) fp4_unit_to_f16<kRows>(src, dst, tid + j * kThreads);
}
// A staged FP8 chunk (raw rows of 256 bytes, kRows of K then kRows of V, at src) -> the BF16 chunk
// (tc_swz, K then V, at dst): unit u = 16 FP8 values -> two swizzled 16-byte BF16 units (exact).
template <int kRows>
__device__ __forceinline__ void fp8_unit_to_bf16(uint32_t src, uint32_t dst, int u) {
  const int kv = u / (kRows * 16), r = u / 16 % kRows, col = u % 16;
  uint4 raw;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
               : "r"(src + (kv * kRows + r) * kHeadDim + col * 16));
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
  uint32_t b[8];
#pragma unroll
  for (int x = 0; x < 4; x++) {
#pragma unroll
    for (int hpart = 0; hpart < 2; hpart++) {
      const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(w[x] >> (16 * hpart)), __NV_E4M3);
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
      b[2 * x + hpart] = pack_bf16(f.x, f.y);
    }
  }
  const uint32_t d = dst + kv * kRows * kTcRowBytes;
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(d + tc_swz(r, 2 * col)), "r"(b[0]), "r"(b[1]),
               "r"(b[2]), "r"(b[3]));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(d + tc_swz(r, 2 * col + 1)), "r"(b[4]), "r"(b[5]),
               "r"(b[6]), "r"(b[7]));
}

// The CTA is compute warps (n8 tiles of columns: below) and kTcProdWarps
// producer warps. Producers stage the raw chunks with cp.async (kTcRaw<KV> slots, the next
// kTcAhead<KV> chunks in flight, page ids read a chunk ahead) and, for FP8 / NVFP4, convert chunk i
// into one of two 16-bit slots while the compute warps work on chunk i - 1: one CTA barrier per chunk.
// Before (P4-13) one CTA ran wait -> convert -> compute per chunk in series, about 1.8 us per chunk.
// Every token processes its chunks in the same order with the same arithmetic as before.
constexpr int kTcProdWarps = 4, kTcProd = kTcProdWarps * 32;
constexpr int kTcGrpWarps = 4;  // warps sharing the last 1 or 2 tiles (a decode's: Q27_ATTN_TC_DEC=0 gives it 1 warp)
template <class KV>
constexpr int kTcRaw = kKvFp8<KV> ? 2 : 3;
template <class KV>
constexpr int kTcAhead = kKvFp8<KV> || kKvFp4<KV> ? kTcRaw<KV> - 1 : kTcRaw<KV> - 2;
template <class KV>  // raw slots, then (FP8 / NVFP4) two 16-bit chunks: 96 / 96 / 91 KiB (BF16 / FP8 / NVFP4)
constexpr int kTcSmemOf = kTcRaw<KV> * 2 * kTcChunk * kKvRowBytes<KV> + (std::is_same_v<KV, bf16> ? 0 : 2 * kTcStageBytes);

// kBatch (P6-10): lane blockIdx.z, T rows per lane from row kLaneRows * z (its position and page table).
template <class KV, bool kBatch = false>
__global__ void __launch_bounds__(kTcMaxT * 32 + kTcProd, 1) attention_tc_kernel(const float* q, const int32_t* pos_dev, int T,
                                                                             const KV* kc, const KV* vc,
                                                                             const int32_t* page_table, int splits,
                                                                             DecPartial part, SeqLanes lanes) {
  pdl_trigger();
  if constexpr (kBatch) {
    const int z = blockIdx.z;
    const int64_t rows = int64_t(kLaneRows) * z * kHeads;
    q += rows * kHeadDim, pos_dev = lanes.pos + z, page_table = lane_sel(lanes.page_table, z);
    part.acc += rows * splits * kHeadDim, part.ml += rows * splits * 2;
  }
  constexpr bool kFp8 = kKvFp8<KV>, kFp4 = kKvFp4<KV>, kRaw = kFp8 || kFp4;
  constexpr bool kHalf = kFp4;  // NVFP4 KV: f16 operands (its values are exact in f16)
  constexpr int kRowBytes = kKvRowBytes<KV>;                         // staged row
  constexpr int kUnits = kTcChunk * (kRowBytes / 16);                // 16-byte units of K (or V) per chunk
  constexpr int kStageBytes = 2 * kTcChunk * kRowBytes;               // K, then V
  constexpr int kR = kTcRaw<KV>, kAhead = kTcAhead<KV>;
  extern __shared__ __align__(128) uint8_t tc_smem[];
  const uint32_t sbase = uint32_t(__cvta_generic_to_shared(tc_smem));
  // FP8 / NVFP4: the two 16-bit chunks the MMAs read, after the raw slots.
  const uint32_t cbase = sbase + kR * kStageBytes;
  const int g = blockIdx.x, split = blockIdx.y;
  const int ncw = blockDim.x / 32 - kTcProdWarps;  // compute warps
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  // The position is advanced only at the end of a step / chunk (host-synchronized): read before the wait.
  const int pos0 = *pos_dev, kmax = pos0 + T;
  const int nchunks = (kmax + kTcChunk - 1) / kTcChunk;
  if (split >= nchunks) return;
  const int nmine = (nchunks - split + splits - 1) / splits;  // chunks split, split + splits, ...

  if (warp >= ncw) {
    // ---- producers
    const int ptid = threadIdx.x - ncw * 32;
    // Stage chunk i of this split. Phase 0: keys < pos0 (written in earlier steps: before the PDL wait),
    // 1: the others, 2: all. Keys >= pos0 + T are zero-filled (P = 0 must not meet garbage).
    // A chunk (32 keys, 32-aligned) lies in one 64-token page: its page id is read once, ahead of time.
    auto page_of = [&](int i) { return i < nmine ? __ldg(page_table + (split + i * splits) * kTcChunk / kPageTokens) : 0; };
    auto issue = [&](int i, int phase, int page) {
      if (i >= nmine) return;
#ifdef Q27_X_TCNOLOAD  // (experiment builds: leave out a stage; results wrong)
      return;
#endif
      const int c = split + i * splits;
      const uint32_t st = sbase + (i % kR) * kStageBytes;
#pragma unroll
      for (int j = 0; j < (kUnits + kTcProd - 1) / kTcProd; j++) {
        const int u = ptid + j * kTcProd, r = u / (kRowBytes / 16), col = u % (kRowBytes / 16);
        if (kUnits % kTcProd != 0 && u >= kUnits) break;
        const int key = c * kTcChunk + r;
        if (phase == 0 && key >= pos0) continue;
        if (phase == 1 && key < pos0) continue;
        const bool ok = key < kmax;
        const int64_t row = (int64_t(page) * kPageTokens + key % kPageTokens) * kKvHeads + g;
        const uint32_t off = kRaw ? r * kRowBytes + col * 16 : tc_swz(r, col);  // (FP8 / NVFP4: swizzled when converted)
        cp_async16z(st + off, kv_row_ptr(kc, row) + col * 16, ok);
        cp_async16z(st + kTcChunk * kRowBytes + off, kv_row_ptr(vc, row) + col * 16, ok);
      }
    };
    // One commit group per chunk; the group of the last prologue chunk also holds the phase-1 rows of
    // the split's last chunk when that is in the prologue (only the last chunk can hold keys >= pos0).
    for (int i = 0; i < kAhead; i++) {
      issue(i, 0, page_of(i));
      if (i + 1 < kAhead) cp_commit();
    }
    int page = page_of(kAhead);  // the next chunk to issue
    pdl_wait();
    if (nmine - 1 < kAhead) issue(nmine - 1, 1, page_of(nmine - 1));
    cp_commit();
    for (int i = 0; i <= nmine; i++) {
      if (i < nmine) {
        issue(i + kAhead, 2, page);  // slot of chunk i - 1 (converted) or i - 2 (read in place)
        cp_commit();
        page = page_of(i + kAhead + 1);
        if (i == nmine - 1)
          cp_wait<0>();
        else
          cp_wait<kAhead>();
#ifdef Q27_X_TCNOCONV
        if constexpr (false) {
#else
        if constexpr (kRaw) {
#endif
          asm volatile("bar.sync 1, %0;" ::"n"(kTcProd));  // every producer's copies of chunk i landed
          const uint32_t src = sbase + (i % kR) * kStageBytes, dst = cbase + (i % 2) * kTcStageBytes;
#pragma unroll
          for (int j = 0; j < 2 * kUnits / kTcProd; j++) {
            if constexpr (kFp8)
              fp8_unit_to_bf16<kTcChunk>(src, dst, ptid + j * kTcProd);
            else
              fp4_unit_to_f16<kTcChunk>(src, dst, ptid + j * kTcProd);
          }
        }
      }
      __syncthreads();  // chunk i ready for the compute warps; chunk i - 1 consumed
    }
    return;
  }

  // ---- compute warps
  // Columns of S^T / O^T: (token t, head h) = column kGqa t + h, packed 8 to an n8 tile (6 tiles for
  // T = 8; before P4-14 one tile per token with 6 of its 8 columns used). Tiles 0 .. nfull - 1 have a
  // warp each; the rest (1 or 2) go to a group of kTcGrpWarps warps (the score tiles split by key
  // tile, P V by dims), so that every SMSP gets the same number of MMAs. A column sees keys
  // 0 .. pos0 + t; in a chunk past its last key it is left as is (corr 1, P 0: exact), which is what
  // skipping the chunk gave. Every column goes through the same operations in the same order whatever
  // its tile and warp, so a verified row stays bit-identical to the decode of the same token.
  pdl_wait();
  const int lm = lane / 8, lr = lane % 8;  // ldmatrix: matrix, row
  const int qcol = lane / 4, c2 = 2 * (lane % 4);  // fragment row / column pair
  const int ncols = T * kGqa, ntile = (ncols + 7) / 8;
  const int nfull = ncw == ntile ? ntile : ncw - kTcGrpWarps;
  auto col_kt = [&](int col) { return col < ncols ? pos0 + col / kGqa + 1 : 0; };  // keys column col sees
  // B fragments of Q^T for tile j (k = dims, n = 8 columns), pre-scaled.
  auto load_q = [&](int j, uint32_t (&qf)[kHeadDim / 16][2]) {
    const float scale = rsqrtf(float(kHeadDim));
    const int col = 8 * j + qcol;
    const bool ok = col < ncols;
    const float* src = q + (int64_t(col / kGqa) * kHeads + g * kGqa + col % kGqa) * kHeadDim + c2;
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 16; kk++) {
      float2 a = ok ? *reinterpret_cast<const float2*>(src + kk * 16) : make_float2(0.f, 0.f);
      float2 b = ok ? *reinterpret_cast<const float2*>(src + kk * 16 + 8) : make_float2(0.f, 0.f);
      qf[kk][0] = pack2<kHalf>(a.x * scale, a.y * scale);
      qf[kk][1] = pack2<kHalf>(b.x * scale, b.y * scale);
    }
  };
  // Scores of key tile mt of the staged chunk (keys 16 mt + qcol, + 8) for the lane's columns c2, c2 + 1,
  // masked by the keys each column sees; mx = their max over the tile's keys (all lanes).
  auto scores = [&](uint32_t ks, int c, int mt, const uint32_t (&qf)[kHeadDim / 16][2], const int (&ktc)[2],
                    float (&s)[4]) {
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 16; kk++) {
      uint32_t a[4];
      ldsm_x4(a, ks + tc_swz(mt * 16 + lr + 8 * (lm % 2), kk * 2 + lm / 2));
      mma16816<kHalf>(s, a, qf[kk]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int e = 0; e < 2; e++)
        if (c * kTcChunk + mt * 16 + h * 8 + qcol >= ktc[e]) s[2 * h + e] = -INFINITY;
  };
  // m := max(m, chunk max); ms = the shift of the exponentials (0 while the column has seen no key).
  auto rescale = [](float& m, float mx, float& corr, float& ms) {
    const float mn = fmaxf(m, mx);
    ms = mn == -INFINITY ? 0.f : mn;
    corr = __expf(m - ms);
    m = mn;
  };
  // The partial results of a column: written when the split holds keys it sees.
  auto store = [&](int col, int kt, const float* od, int stride, int nd, int dm0, float m_, float l_, bool ml) {
    if (split * kTcChunk >= kt) return;  // empty split of this column (kt = 0: no column): combine skips it
    const int64_t row = int64_t(col / kGqa) * kHeads + g * kGqa + col % kGqa;
    float* dst = part.acc + (row * splits + split) * kHeadDim + qcol;
#pragma unroll
    for (int d = 0; d < nd; d++) dst[(dm0 + d) * 16] = od[d * stride], dst[(dm0 + d) * 16 + 8] = od[d * stride + 2];
    if (ml && qcol == 0) part.ml[(row * splits + split) * 2] = m_, part.ml[(row * splits + split) * 2 + 1] = l_;
  };

  if (warp >= nfull) {
    // Group: tiles nfull .. ntile - 1 (ng = 1 or 2; a decode's one tile). Warp gw computes the scores
    // of key tile gw % 2 of tile gw / 2 and the P V of a quarter of the dims of every group tile; maxima
    // and P go through shared memory, and the row sum of a tile is added by its key-tile-0 warp in the
    // order of the full-tile path (P read back from x_pt: movmatrix is its own inverse).
    __shared__ float x_mx[2][2][32][2];
    __shared__ uint32_t x_pt[2][2][2][32];
    // ng (1 or 2) as a constant: a decode compiles to straight-line code.
    auto group = [&](auto ng_c) {
      constexpr int ng = decltype(ng_c)::value;
      constexpr int kDq = kHeadDim / 16 / kTcGrpWarps;  // dims tiles per warp
      const int gw = warp - nfull, jq = gw / 2, mt = gw % 2, dm0 = gw * kDq;
      const bool qk = jq < ng;
      const int kgrp = pos0 + T;  // the last tile holds the last token
      int ktc[2][2];
#pragma unroll
      for (int jj = 0; jj < 2; jj++)
#pragma unroll
        for (int e = 0; e < 2; e++) ktc[jj][e] = jj < ng ? col_kt(8 * (nfull + jj) + c2 + e) : 0;
      // ktc of tile jq (register arrays indexed by jq would go to local memory; also ms, corr below)
      const int ktq[2] = {col_kt(8 * (nfull + jq) + c2), col_kt(8 * (nfull + jq) + c2 + 1)};
      uint32_t qf[kHeadDim / 16][2];
      if (qk) load_q(nfull + jq, qf);
      float o[2][kDq][4] = {};
      float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, l[2] = {0.f, 0.f};  // l: tile jq
      __syncthreads();  // chunk 0 staged
      for (int i = 0; i < nmine; i++) {
        const int c = split + i * splits;
        const uint32_t ks = kRaw ? cbase + (i % 2) * kTcStageBytes : sbase + (i % kR) * kStageBytes;
        const uint32_t vs = ks + kTcChunk * kTcRowBytes;
#if defined(Q27_X_TCNOCOMP) || defined(Q27_X_TCNOCONV)
        if (false) {
#else
        if (c * kTcChunk < kgrp) {
#endif
          float s[4] = {};
          if (qk) {
            scores(ks, c, mt, qf, ktq, s);
#pragma unroll
            for (int e = 0; e < 2; e++) {
              float mx = fmaxf(s[e], s[2 + e]);
#pragma unroll
              for (int x = 4; x < 32; x <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, x));
              x_mx[jq][mt][lane][e] = mx;
            }
          }
          asm volatile("bar.sync 2, %0;" ::"n"(kTcGrpWarps * 32));
          float corr[2][2], ms[2][2];
#pragma unroll
          for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int e = 0; e < 2; e++)
              if (jj < ng) rescale(m[jj][e], fmaxf(x_mx[jj][0][lane][e], x_mx[jj][1][lane][e]), corr[jj][e], ms[jj][e]);
          if (qk) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const float p0 = __expf(s[2 * h] - (jq ? ms[1][0] : ms[0][0]));
              const float p1 = __expf(s[2 * h + 1] - (jq ? ms[1][1] : ms[0][1]));
              x_pt[jq][mt][h][lane] = movmatrix_trans(pack2<kHalf>(p0, p1));
            }
          }
          asm volatile("bar.sync 2, %0;" ::"n"(kTcGrpWarps * 32));
          if (qk && mt == 0) {
            float ps[2] = {0.f, 0.f};
#pragma unroll
            for (int t2 = 0; t2 < 2; t2++)
#pragma unroll
              for (int h = 0; h < 2; h++) {
                const float2 pr = unpack2<kHalf>(movmatrix_trans(x_pt[jq][t2][h][lane]));
                ps[0] += pr.x, ps[1] += pr.y;
              }
#pragma unroll
            for (int e = 0; e < 2; e++) {
#pragma unroll
              for (int x = 4; x < 32; x <<= 1) ps[e] += __shfl_xor_sync(0xffffffffu, ps[e], x);
              l[e] = l[e] * (jq ? corr[1][e] : corr[0][e]) + ps[e];
            }
          }
#pragma unroll
          for (int jj = 0; jj < 2; jj++) {
            if (jj >= ng) continue;
#pragma unroll
            for (int d = 0; d < kDq; d++) {
              o[jj][d][0] *= corr[jj][0], o[jj][d][1] *= corr[jj][1];
              o[jj][d][2] *= corr[jj][0], o[jj][d][3] *= corr[jj][1];
            }
          }
#pragma unroll
          for (int t2 = 0; t2 < 2; t2++) {
            uint32_t pt[2][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) pt[jj][0] = x_pt[jj][t2][0][lane], pt[jj][1] = x_pt[jj][t2][1][lane];
#pragma unroll
            for (int d = 0; d < kDq; d++) {
              uint32_t a[4];
              ldsm_x4_trans(a, vs + tc_swz(t2 * 16 + lr + 8 * (lm / 2), (dm0 + d) * 2 + lm % 2));
              mma16816<kHalf>(o[0][d], a, pt[0]);
              if (ng > 1) mma16816<kHalf>(o[1][d], a, pt[1]);
            }
          }
        }
        __syncthreads();  // chunk i consumed, chunk i + 1 staged
      }
#pragma unroll
      for (int jj = 0; jj < 2; jj++)
#pragma unroll
        for (int e = 0; e < 2; e++)
          if (jj < ng)
            store(8 * (nfull + jj) + c2 + e, ktc[jj][e], &o[jj][0][e], 4, kDq, dm0, m[jj][e], l[e],
                  qk && mt == 0 && jj == jq);
    };
    if (ntile - nfull == 1)
      group(std::integral_constant<int, 1>{});
    else
      group(std::integral_constant<int, 2>{});
    return;
  }

  // A tile on one warp.
  const int j = warp;
  int ktc[2];
#pragma unroll
  for (int e = 0; e < 2; e++) ktc[e] = col_kt(8 * j + c2 + e);
  const int ktile = col_kt(min(8 * j + 7, ncols - 1));
  uint32_t qf[kHeadDim / 16][2];
  load_q(j, qf);
  float o[kHeadDim / 16][4];
#pragma unroll
  for (int i = 0; i < kHeadDim / 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};  // columns c2, c2 + 1

  __syncthreads();  // chunk 0 staged
  for (int i = 0; i < nmine; i++) {
    const int c = split + i * splits;
    const uint32_t ks = kRaw ? cbase + (i % 2) * kTcStageBytes : sbase + (i % kR) * kStageBytes;
    const uint32_t vs = ks + kTcChunk * kTcRowBytes;
#if defined(Q27_X_TCNOCOMP) || defined(Q27_X_TCNOCONV)
    if (false) {
#else
    if (c * kTcChunk < ktile) {
#endif
      // Scores: s[mt][0..1] = key 16 mt + qcol, columns c2, c2 + 1; s[mt][2..3] = key + 8.
      float s[2][4] = {};
#pragma unroll
      for (int mt = 0; mt < 2; mt++) scores(ks, c, mt, qf, ktc, s[mt]);
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int x = 0; x < 4; x++) mx[x % 2] = fmaxf(mx[x % 2], s[mt][x]);
      float corr[2], ms[2], ps[2] = {0.f, 0.f};
#pragma unroll
      for (int e = 0; e < 2; e++) {
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], x));
        rescale(m[e], mx[e], corr[e], ms[e]);
      }
      uint32_t pt[2][2];  // B fragments of P^T per key tile
#pragma unroll
      for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float p0 = __expf(s[mt][2 * h] - ms[0]), p1 = __expf(s[mt][2 * h + 1] - ms[1]);
          const uint32_t pb = pack2<kHalf>(p0, p1);
          const float2 pr = unpack2<kHalf>(pb);
          ps[0] += pr.x, ps[1] += pr.y;
          pt[mt][h] = movmatrix_trans(pb);
        }
#pragma unroll
      for (int e = 0; e < 2; e++) {
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) ps[e] += __shfl_xor_sync(0xffffffffu, ps[e], x);
        l[e] = l[e] * corr[e] + ps[e];
      }
#pragma unroll
      for (int dm = 0; dm < kHeadDim / 16; dm++) {
        o[dm][0] *= corr[0], o[dm][1] *= corr[1], o[dm][2] *= corr[0], o[dm][3] *= corr[1];
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int dm = 0; dm < kHeadDim / 16; dm++) {
          uint32_t a[4];
          ldsm_x4_trans(a, vs + tc_swz(mt * 16 + lr + 8 * (lm / 2), dm * 2 + lm % 2));
          mma16816<kHalf>(o[dm], a, pt[mt]);
        }
      }
    }
    __syncthreads();  // chunk i consumed, chunk i + 1 staged
  }
#pragma unroll
  for (int e = 0; e < 2; e++) store(8 * j + c2 + e, ktc[e], &o[0][e], 4, kHeadDim / 16, 0, m[e], l[e], true);
}

// ---------------------------------------------------------------- prefill (tensor cores)
//
// Flash attention for a prefill chunk: CTA = (64-token query tile, kFaHeads query heads of one KV
// group, key split), 4 warps x 16 tokens per head; the heads share the staged keys.
// Keys go in fixed 32-key chunks (two cp.async stages, rows XOR-swizzled as in attention_tc_kernel;
// FP8 KV is converted to BF16 in shared memory). Per chunk each warp computes S = Q K^T (mma m16n8k16,
// Q in registers as A fragments, K rows through ldmatrix as B), the online softmax in registers, then
// O += P V with P re-used from the S accumulators as A fragments and V through ldmatrix.trans. Q and P
// are rounded to BF16.
// Tiles are aligned to absolute 64-token blocks. A block ending past kFaSplitMin splits its keys in two
// at half the block end (one CTA each, partials merged by attention_fa_combine_kernel): a 2,048-token
// chunk is otherwise 384 CTAs, 2.3 waves of one CTA per SM. The split depends only on the block, and a
// row's arithmetic does not depend on its tile (chunks past its position are fully masked and leave it
// unchanged), so a prefill cut anywhere gives the same rows as an uncut one.
constexpr int kFaQ = 64, kFaHeads = 2, kFaWarps = kFaHeads * kFaQ / 16, kFaK = 32;
constexpr int kFaSplits = 2, kFaSplitMin = 1024;
static_assert(kGqa % kFaHeads == 0, "heads of a CTA share one KV group");
static_assert(kFaSplitMin % (2 * kFaK) == 0 && kFaQ % kFaK == 0, "split points on key chunks");
constexpr int kFaStageBytes = 2 * kFaK * kTcRowBytes;  // K, then V (BF16)
constexpr int kFaSmem = 2 * kFaStageBytes;
constexpr int kFaSmemFp8 = 2 * kFaStageBytes / 2 + kFaStageBytes;  // raw FP8 stages + one BF16 chunk
constexpr int kFaSmemFp4 = 2 * 2 * kFaK * kKvRowBytes<Fp4Kv> + kFaStageBytes;  // raw NVFP4 stages + one BF16 chunk
// NVFP4 KV with FP8 MMAs: raw NVFP4 stages, the chunk in E4M3 (K, V), V^T.
constexpr int kFaSmemFp4F8 = 2 * 2 * kFaK * kKvRowBytes<Fp4Kv> + 2 * kFaK * kHeadDim + kHeadDim * kFaK;  // (V^T: kFaVtBytes)

__device__ __forceinline__ uint32_t sw8(int row, int unit);
template <int kRows, int kThreads>
__device__ __forceinline__ void fp4_chunk_to_e4m3(uint32_t src, uint32_t dst, int tid) {
  constexpr int kRow = kKvRowBytes<Fp4Kv>;
  static_assert(2 * kRows * 8 % kThreads == 0, "whole passes");
#pragma unroll
  for (int j = 0; j < 2 * kRows * 8 / kThreads; j++) {
    const int u = tid + j * kThreads, kv = u / (kRows * 8), r = u / 8 % kRows, col = u % 8;
    const uint32_t row = src + (kv * kRows + r) * kRow;
    uint32_t w[4], sc;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "r"(row + col * 16));
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(sc) : "r"(row + kHeadDim / 2 + 2 * col));
    uint32_t h[16], e[8];
    fp4x32_to_f16(w, sc, h);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint16_t lo, hi;
      asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(h[2 * i]));
      asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(h[2 * i + 1]));
      e[i] = uint32_t(lo) | uint32_t(hi) << 16;
    }
    const uint32_t d = dst + kv * kRows * kHeadDim;
#pragma unroll
    for (int x = 0; x < 2; x++)
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(d + sw8(r, 2 * col + x)), "r"(e[4 * x]),
                   "r"(e[4 * x + 1]), "r"(e[4 * x + 2]), "r"(e[4 * x + 3]));
  }
}
// NVFP4 KV: mode 0 converts the chunk to f16 (exact; f16 MMAs), mode 2 rounds it to E4M3 (sw8 rows) and
// runs the FP8 MMAs below (about 1.6x faster; the extra rounding is small next to FP4's own).
// FP8 MMAs (FP8 KV; Q27_ATTN_FP8): mode 1 runs P V in E4M3 (K still converted to BF16), mode 2 also
// Q K^T. K and V are the cache's E4M3 values as they are; Q is quantized per row (scale amax / 448) and
// P as 256 p. P V needs V with the keys along k, so each chunk's V is transposed in shared memory into
// V^T (256 dims x 32 keys) with the keys permuted to the order in which a thread holds P in the S
// accumulators (k 4j .. 4j + 3 = keys 2j, 2j + 1, 2j + 8, 2j + 9, and the same + 16): the C fragments
// pack into the A fragment without shuffles. The P V of a chunk starts from zero and is added to the
// rescaled O in FP32. The softmax runs in base 2 with the exponentials in f16x2 (P is E4M3).
// The raw stages are XOR-swizzled by 16-byte unit (sw8).
constexpr int kFaVtBytes = kHeadDim * kFaK;  // V^T of one chunk, E4M3
// Mode 3 (P7-8): two stages of K raw NVFP4 + V E4M3, V^T.
constexpr int kFaSmemFp4Q4 = 2 * kFaK * (kKvRowBytes<Fp4Kv> + kHeadDim) + kFaVtBytes;
constexpr int kFaSmemF8[3] = {0, 2 * kFaStageBytes / 2 + kFaK * kTcRowBytes + kFaVtBytes,
                              2 * kFaStageBytes / 2 + kFaVtBytes};

constexpr float kLog2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;

__device__ __forceinline__ uint32_t sw8(int row, int unit) { return row * kHeadDim + ((unit ^ (row & 7)) << 4); }
// 16-byte segment (dim d, keys 16 hh .. 16 hh + 15) of V^T: conflict-free for the transposing stores
// (8 consecutive dim groups of 4) and for ldmatrix (8 consecutive dims).
__device__ __forceinline__ uint32_t vt_off(int d, int hh) { return uint32_t((2 * d + hh) ^ ((d >> 2) & 7)) << 4; }

// Splits of the block ending at bend (absolute, a multiple of kFaQ); the second starts at bend / 2.
__host__ __device__ constexpr int fa_splits(int bend) { return bend > kFaSplitMin ? kFaSplits : 1; }

template <class KV, int kF8 = 0>
__global__ void __launch_bounds__(kFaWarps * 32, 1) attention_fa_kernel(const float* q, const float* qg, int T, int pos0,
                                                                        const KV* kc, const KV* vc,
                                                                        const int32_t* page_table, float* out,
                                                                        float* part, const uint8_t* v8 = nullptr,
                                                                        const int32_t* vpt = nullptr) {
  constexpr bool kFp8 = kKvFp8<KV>, kFp4 = kKvFp4<KV>;
  constexpr bool kHalf = kFp4 && kF8 == 0;  // NVFP4 KV: f16 operands (its values are exact in f16)
  static_assert((kF8 <= 2 && (kF8 == 0 || kFp8 || (kFp4 && kF8 == 2))) || (kF8 == 3 && kFp4),
                "FP8 MMAs need FP8 KV (or NVFP4 rounded to E4M3, mode 2); mode 3 needs NVFP4 KV");
  // Mode 3 (P7-8, NVFP4 KV): Q K^T on the native FP4 MMA (K's rows as they are in the cache, Q in NVFP4),
  // V read in E4M3 from the stage (v8 behind vpt) and P V as mode 2. A stage: K raw (linear), V E4M3 (sw8).
  constexpr bool kQk4 = kF8 == 3;
  constexpr int kRowBytes = kKvRowBytes<KV>;
  constexpr int kStageBytes = kQk4 ? kFaK * (kRowBytes + kHeadDim) : 2 * kFaK * kRowBytes;
  constexpr int kUnits = kFaK * (kRowBytes / 16);  // 16-byte units of K (or V) per chunk
  extern __shared__ __align__(128) uint8_t fa_smem[];
  const uint32_t sbase = uint32_t(__cvta_generic_to_shared(fa_smem));
  const uint32_t cbase = sbase + 2 * kStageBytes;  // FP8 / NVFP4: the converted chunk
  // FP8 MMAs: V^T (NVFP4 KV: after the chunk rounded to E4M3, K then V)
  const uint32_t vtbase = cbase + (kF8 == 1 ? kFaK * kTcRowBytes : kFp4 && kF8 == 2 ? 2 * kFaK * kHeadDim : 0);
  const int lead = pos0 % kFaQ;  // rows of the first block before this call
  const int ntiles = (lead + T + kFaQ - 1) / kFaQ;
  const int tile = ntiles - 1 - int(blockIdx.x) / kFaSplits, split = int(blockIdx.x) % kFaSplits;  // longest first
  const int q0 = tile * kFaQ - lead;  // index (in this call) of the tile's first row; negative: rows before pos0
  const int bend = pos0 + q0 + kFaQ, nsplit = fa_splits(bend);
  if (split >= nsplit) return;
  const int warp = threadIdx.x / 32 % (kFaQ / 16), lane = threadIdx.x % 32;
  const int h = blockIdx.y * kFaHeads + threadIdx.x / (kFaQ / 16 * 32), g = h / kGqa;
  const int rend = min(bend, pos0 + T);  // keys 0 .. rend - 1 are visible to some row of the tile
  const int kbeg = split ? bend / 2 : 0, kend = nsplit > 1 && !split ? bend / 2 : rend;
  const int cbeg = kbeg / kFaK, cend = kend > kbeg ? (kend + kFaK - 1) / kFaK : cbeg;

  auto issue = [&](int c) {
    const uint32_t st = sbase + (c % 2) * kStageBytes;
    if constexpr (kQk4) {
      // K: 32 x 9 units (the last 32 on the first warp), V: 32 x 16 (two per thread).
      constexpr int kKu = kFaK * (kRowBytes / 16);
      static_assert(kKu > kFaWarps * 32 && kKu <= 2 * kFaWarps * 32 && kFaK * kHeadDim / 16 == 2 * kFaWarps * 32);
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const int u = threadIdx.x + j * kFaWarps * 32;
        if (j == 0 || u < kKu) {
          const int r = u / (kRowBytes / 16), col = u % (kRowBytes / 16), key = c * kFaK + r;
          const bool ok = key < kend;
          cp_async16z(st + r * kRowBytes + col * 16, kv_row_ptr(kc, ok ? kv_row(page_table, key, g) : 0) + col * 16, ok);
        }
        const int r = u / (kHeadDim / 16), col = u % (kHeadDim / 16), key = c * kFaK + r;
        const bool ok = key < kend;
        cp_async16z(st + kFaK * kRowBytes + sw8(r, col), v8 + (ok ? kv_row(vpt, key, g) : 0) * kHeadDim + col * 16, ok);
      }
    } else {
#pragma unroll
    for (int j = 0; j < (kUnits + kFaWarps * 32 - 1) / (kFaWarps * 32); j++) {
      const int u = threadIdx.x + j * kFaWarps * 32, r = u / (kRowBytes / 16), col = u % (kRowBytes / 16);
      if (kUnits % (kFaWarps * 32) != 0 && u >= kUnits) break;
      const int key = c * kFaK + r;
      const bool ok = key < kend;
      const int64_t row = ok ? kv_row(page_table, key, g) : 0;
      const uint32_t off = kF8 && kFp8 ? sw8(r, col) : kFp8 || kFp4 ? r * kRowBytes + col * 16 : tc_swz(r, col);
      const uint8_t *ksrc = kv_row_ptr(kc, row) + col * 16, *vsrc = kv_row_ptr(vc, row) + col * 16;
#ifndef Q27_X_NOLOAD
      cp_async16z(st + off, ksrc, ok);
      cp_async16z(st + kFaK * kRowBytes + off, vsrc, ok);
#else
      if (c < 2) cp_async16z(st + off, ksrc, ok), cp_async16z(st + kFaK * kRowBytes + off, vsrc, ok);
#endif
    }
    }
  };
  issue(cbeg);
  cp_commit();
  if (cend - cbeg > 1) issue(cbeg + 1);
  cp_commit();

  // Q of this warp's 16 tokens as A fragments (rows = tokens, k = dims), pre-scaled by 1/16 (exact).
  // FP8 Q K^T: E4M3 with a scale per row (sq, applied to S).
  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;  // rows of this thread within the tile
  constexpr bool kQk8 = kF8 == 2;
  uint32_t qa[kQk8 || kQk4 ? kHeadDim / 32 : kHeadDim / 16][4];
  float sq[2] = {1.f, 1.f};  // rows ta, tb
  uint32_t sfa[kQk4 ? kHeadDim / 64 : 1];  // mode 3: the 4 block scales of k-step s (row ta: even t, tb: odd)
  if constexpr (kQk4) {
    // NVFP4 Q: per row a global scale gs = amax / (6 x 448) (applied to S with the 1/16), per 16 dims
    // sf = e4m3(amax_block / 6 / gs), codes rne_e2m1(x / (gs sf)). k-step s: a[0] / a[1] = rows ta / tb, dims
    // 64 s + 8 t .. + 7, a[2] / a[3] the same + 32 (low nibble = lower dim).
    const int t = lane % 4;
    float gsr[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int tt = q0 + (r ? tb : ta);
      const float* row = q + (int64_t(tt) * kHeads + h) * kHeadDim;
      const bool in = tt >= 0 && tt < T;
      float mx = 0.f;
      for (int d = 0; d < kHeadDim && in; d += 4) {
        const float4 v = *reinterpret_cast<const float4*>(row + d);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
      }
      gsr[r] = mx * (1.f / (6.f * 448.f));
      sq[r] = gsr[r] * (1.f / 16.f);
      const float ig = gsr[r] > 0.f ? 1.f / gsr[r] : 0.f;
#pragma unroll
      for (int st = 0; st < kHeadDim / 64; st++) {
        uint32_t sfw = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          float bm = 0.f;
#pragma unroll
          for (int d = 0; d < 16 && true; d += 4) {
            const float4 v = in ? *reinterpret_cast<const float4*>(row + 64 * st + 16 * b + d) : make_float4(0.f, 0.f, 0.f, 0.f);
            bm = fmaxf(bm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
          }
          const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(bm * ig * (1.f / 6.f), __NV_SATFINITE, __NV_E4M3);
          sfw |= uint32_t(sfq) << (8 * b);
          // this lane's 8 dims of block b: half hb = b / 2 when t / 2 == b % 2
          if (t / 2 == b % 2) {
            const float sf = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
            const float inv = sf > 0.f ? ig / sf : 0.f;
            const int d0 = 64 * st + 32 * (b / 2) + 8 * t;
            const float4 v0 = in ? *reinterpret_cast<const float4*>(row + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 v1 = in ? *reinterpret_cast<const float4*>(row + d0 + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
            qa[st][2 * (b / 2) + r] = uint32_t(cvt_e2m1x2(v0.x * inv, v0.y * inv)) |
                                      uint32_t(cvt_e2m1x2(v0.z * inv, v0.w * inv)) << 8 |
                                      uint32_t(cvt_e2m1x2(v1.x * inv, v1.y * inv)) << 16 |
                                      uint32_t(cvt_e2m1x2(v1.z * inv, v1.w * inv)) << 24;
          }
        }
        if (r == t % 2) sfa[st] = sfw;
      }
    }
  } else if constexpr (kQk8) {
    auto ld4 = [&](int t, int d) {
      if (q0 + t < 0 || q0 + t >= T) return make_float4(0.f, 0.f, 0.f, 0.f);
      return *reinterpret_cast<const float4*>(q + (int64_t(q0 + t) * kHeads + h) * kHeadDim + d);
    };
    float inv[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      float mx = 0.f;
#pragma unroll
      for (int kk = 0; kk < kHeadDim / 16; kk++) {
        const float4 v = ld4(r ? tb : ta, kk * 16 + 2 * c2);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      inv[r] = mx > 0.f ? 448.f / mx : 0.f;
      sq[r] = mx * (1.f / (448.f * 16.f));
    }
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 32; kk++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int r = i % 2;
        const float4 v = ld4(r ? tb : ta, kk * 32 + 16 * (i / 2) + 2 * c2);
        qa[kk][i] = pack_e4m3x4(v.x * inv[r], v.y * inv[r], v.z * inv[r], v.w * inv[r]);
      }
  } else {
    const float scale = rsqrtf(float(kHeadDim));
    auto ld = [&](int t, int d) {
      if (q0 + t < 0 || q0 + t >= T) return make_float2(0.f, 0.f);
      const float2 v = *reinterpret_cast<const float2*>(q + (int64_t(q0 + t) * kHeads + h) * kHeadDim + d);
      return make_float2(v.x * scale, v.y * scale);
    };
#pragma unroll
    for (int kk = 0; kk < kHeadDim / 16; kk++) {
      float2 a = ld(ta, kk * 16 + c2), b = ld(tb, kk * 16 + c2), c = ld(ta, kk * 16 + c2 + 8),
             d = ld(tb, kk * 16 + c2 + 8);
      qa[kk][0] = pack2<kHalf>(a.x, a.y), qa[kk][1] = pack2<kHalf>(b.x, b.y);
      qa[kk][2] = pack2<kHalf>(c.x, c.y), qa[kk][3] = pack2<kHalf>(d.x, d.y);
    }
  }
  const int pa = pos0 + q0 + ta, pb = pos0 + q0 + tb;  // positions of the two rows
  const int wend = min(pos0 + q0 + warp * 16 + 16, kend);  // keys this warp needs
  float o[kHeadDim / 8][4];
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};  // rows ta, tb
  const int lm = lane / 8, lr = lane % 8;

  for (int c = cbeg; c < cend; c++) {
    cp_wait<1>();
    __syncthreads();
    uint32_t ks = sbase + (c % 2) * kStageBytes;
    if constexpr (kFp4 && kF8 == 2) {  // NVFP4 -> E4M3 rows (sw8) at cbase: K is read from there, V transposed
      fp4_chunk_to_e4m3<kFaK, kFaWarps * 32>(ks, cbase, threadIdx.x);
      __syncthreads();
      ks = cbase;
    }
#ifndef Q27_X_NOVT  // (experiment builds: -DQ27_X_* leave out a stage to measure its cost; results wrong)
    if constexpr (kF8 > 0) {
#else
    if constexpr (false) {
#endif
      // V^T: thread = (4 dims 4 dg .. 4 dg + 3, key group j, half hh); 4 words of V -> 4 words of V^T.
      const uint32_t vraw = ks + kFaK * (kFp4 && !kQk4 ? kHeadDim : kRowBytes);
#pragma unroll
      for (int x = 0; x < kHeadDim / 4 * 4 * 2 / (kFaWarps * 32); x++) {
        const int u = threadIdx.x + x * kFaWarps * 32, j = u & 3, rest = u >> 5;
        const int dg = (rest & 7) * 8 + (u >> 2 & 7), hh = rest >> 3;
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int key = 16 * hh + 2 * j + (i & 1) + 8 * (i >> 1);
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w[i]) : "r"(vraw + sw8(key, dg / 4) + 4 * (dg % 4)));
        }
        const uint32_t t0 = __byte_perm(w[0], w[1], 0x5140), t1 = __byte_perm(w[2], w[3], 0x5140);
        const uint32_t t2 = __byte_perm(w[0], w[1], 0x7362), t3 = __byte_perm(w[2], w[3], 0x7362);
        const uint32_t y[4] = {__byte_perm(t0, t1, 0x5410), __byte_perm(t0, t1, 0x7632), __byte_perm(t2, t3, 0x5410),
                               __byte_perm(t2, t3, 0x7632)};
#pragma unroll
        for (int i = 0; i < 4; i++)
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(vtbase + vt_off(4 * dg + i, hh) + 4 * j), "r"(y[i]));
      }
    }
    if constexpr (kFp8 && !kQk8) {
      // K (and V unless P V runs in FP8) to BF16.
      constexpr int kConv = kF8 ? 1 : 2;
#pragma unroll
      for (int j = 0; j < kConv * kFaK * (kRowBytes / 16) / (kFaWarps * 32); j++) {
        const int u = threadIdx.x + j * kFaWarps * 32, kv = u / (kFaK * kRowBytes / 16);
        const int r = u / (kRowBytes / 16) % kFaK, col = u % (kRowBytes / 16);
        uint4 raw;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                     : "r"(ks + kv * kFaK * kRowBytes + (kF8 ? sw8(r, col) : r * kRowBytes + col * 16)));
        const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        uint32_t b[8];
#pragma unroll
        for (int x = 0; x < 4; x++)
#pragma unroll
          for (int hp = 0; hp < 2; hp++) {
            const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(w[x] >> (16 * hp)), __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            b[2 * x + hp] = pack_bf16(f.x, f.y);
          }
        const uint32_t dst = cbase + kv * kFaK * kTcRowBytes;
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(dst + tc_swz(r, 2 * col)), "r"(b[0]), "r"(b[1]),
                     "r"(b[2]), "r"(b[3]));
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(dst + tc_swz(r, 2 * col + 1)), "r"(b[4]),
                     "r"(b[5]), "r"(b[6]), "r"(b[7]));
      }
    }
    if constexpr (kFp4 && kF8 == 0) fp4_chunk_to_f16<kFaK, kFaWarps * 32>(ks, cbase, threadIdx.x);
    if constexpr (kFp8 || kFp4) {
      __syncthreads();
      if constexpr (!kQk8 && !kQk4) ks = cbase;
    }
    const uint32_t vs = ks + kFaK * kTcRowBytes;
    if (c * kFaK < wend) {
      // S: 16 tokens x 32 keys = 4 n8 tiles.
      float sc[4][4] = {};
      if constexpr (kQk4) {
        // B: key 8 nt + lane / 4, word t of the 16-byte pieces 2 s, 2 s + 1 of its 128 code bytes (ldmatrix: the
        // 144-byte rows hit distinct banks), the key's 4 block scales of step s.
        const uint32_t kq = ks + (lane / 4) * kRowBytes + kHeadDim / 2;
        const uint32_t kl = ks + (lr + 8 * (lm / 2)) * kRowBytes + 16 * (lm % 2);
        uint4 sk[4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(sk[nt].x), "=r"(sk[nt].y), "=r"(sk[nt].z), "=r"(sk[nt].w)
                       : "r"(kq + 8 * nt * kRowBytes));
#pragma unroll
        for (int st = 0; st < kHeadDim / 64; st++)
#pragma unroll
          for (int p = 0; p < 2; p++) {
            uint32_t b[4];  // (nt 2p: b[0], b[1]; nt 2p + 1: b[2], b[3])
            ldsm_x4(b, kl + 16 * p * kRowBytes + 32 * st);
            const uint32_t s0[4] = {sk[2 * p].x, sk[2 * p].y, sk[2 * p].z, sk[2 * p].w};
            const uint32_t s1[4] = {sk[2 * p + 1].x, sk[2 * p + 1].y, sk[2 * p + 1].z, sk[2 * p + 1].w};
            mma_fp4_16864(sc[2 * p], qa[st], b, sfa[st], s0[st]);
            mma_fp4_16864(sc[2 * p + 1], qa[st], b + 2, sfa[st], s1[st]);
          }
      } else if constexpr (kQk8) {
#ifndef Q27_X_NOQK
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 32; kk++)
#pragma unroll
          for (int p = 0; p < 2; p++) {
#ifndef Q27_X_NOLDSM
            uint32_t b[4];
            ldsm_x4(b, ks + sw8(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
#else
            uint32_t b[4] = {qa[kk][0] ^ uint32_t(p + c), qa[kk][1], qa[kk][2], qa[kk][3]};
#endif
            mma_e4m3_16832(sc[2 * p], qa[kk], b);
            mma_e4m3_16832(sc[2 * p + 1], qa[kk], b + 2);
          }
#endif
      } else {
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; kk++)
#pragma unroll
          for (int p = 0; p < 2; p++) {
            uint32_t b[4];
            ldsm_x4(b, ks + tc_swz(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma16816<kHalf>(sc[2 * p], qa[kk], b);
            mma16816<kHalf>(sc[2 * p + 1], qa[kk], b + 2);
          }
      }
      // FP8 MMAs: the softmax in base 2 (S scaled by sq log2 e; m is then in log2 units).
      if constexpr (kF8 > 0) {
        const float f[2] = {sq[0] * kLog2e, sq[1] * kLog2e};
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
          for (int e = 0; e < 4; e++) sc[nt][e] *= f[e / 2];
      }
      // Mask (only chunks reaching past the warp's first row) and the online softmax (each row is
      // spread over the 4 lanes of a quad).
      const bool diag = c * kFaK + kFaK - 1 > pos0 + q0 + warp * 16;
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int key = c * kFaK + nt * 8 + c2 + (e & 1);
          if (diag && key > (e < 2 ? pa : pb)) sc[nt][e] = -INFINITY;
          mx[e / 2] = fmaxf(mx[e / 2], sc[nt][e]);
        }
      float corr[2];
#pragma unroll
      for (int r = 0; r < 2; r++) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -INFINITY ? 1.f : kF8 > 0 ? ex2_approx(m[r] - mn) : __expf(m[r] - mn);
        m[r] = mn;
      }
      float ps[2] = {0.f, 0.f};
      if constexpr (kF8 > 0) {
        // P x 256 = 2^(s - m + 8) as one E4M3 A fragment over the 32 keys in the permuted order of V^T.
        uint32_t pf[4];
#pragma unroll
        for (int r = 0; r < 2; r++) {
          const float b = 8.f - (m[r] == -INFINITY ? 0.f : m[r]);  // (a row with no key yet: all -inf -> 0)
          uint32_t h[4];
#pragma unroll
          for (int nt = 0; nt < 4; nt++)
#ifndef Q27_X_NOEXP
            h[nt] = exp2_e4m3x2(sc[nt][2 * r] + b, sc[nt][2 * r + 1] + b);
#else
            h[nt] = pack_e4m3x4(fminf(fabsf(sc[nt][2 * r]), 256.f), fminf(fabsf(sc[nt][2 * r + 1]), 256.f), 0.f, 0.f) & 0xffff;
#endif
          pf[r] = h[0] | h[1] << 16;
          pf[2 + r] = h[2] | h[3] << 16;
          ps[r] = sum_e4m3x4(pf[r]) + sum_e4m3x4(pf[2 + r]);
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
          l[r] = l[r] * corr[r] + ps[r];
        }
        // O = O corr + P V: n = dims (32 n8 tiles, two per ldmatrix.x4 of V^T), k = the 32 keys. (Accumulating
        // in the mma and rescaling only when a max moves saves no energy: P5-10.)
#pragma unroll
        for (int dp = 0; dp < kHeadDim / 16; dp++) {
          float t[2][4] = {};
#ifndef Q27_X_NOPV
#ifndef Q27_X_NOLDSM
          uint32_t b[4];
          ldsm_x4(b, vtbase + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
#else
          uint32_t b[4] = {qa[dp % 8][0] ^ uint32_t(dp), qa[dp % 8][1], qa[dp % 8][2], qa[dp % 8][3]};
#endif
          mma_e4m3_16832(t[0], pf, b);
          mma_e4m3_16832(t[1], pf, b + 2);
#else
          t[0][0] = __uint_as_float(pf[0] & 0x3f00ff);
#endif
#pragma unroll
          for (int x = 0; x < 2; x++)
#pragma unroll
            for (int e = 0; e < 4; e++) o[2 * dp + x][e] = o[2 * dp + x][e] * corr[e / 2] + t[x][e];
        }
      } else {
        uint32_t pf[2][4];  // P as A fragments, keys 0-15 and 16-31
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const float p0 = m[r] == -INFINITY ? 0.f : __expf(sc[nt][2 * r] - m[r]);
            const float p1 = m[r] == -INFINITY ? 0.f : __expf(sc[nt][2 * r + 1] - m[r]);
            const uint32_t pb2 = pack2<kHalf>(p0, p1);
            const float2 pr = unpack2<kHalf>(pb2);
            ps[r] += pr.x + pr.y;
            pf[nt / 2][(nt % 2) * 2 + r] = pb2;
          }
#pragma unroll
        for (int r = 0; r < 2; r++) {
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
          l[r] = l[r] * corr[r] + ps[r];
        }
#pragma unroll
        for (int dt = 0; dt < kHeadDim / 8; dt++) o[dt][0] *= corr[0], o[dt][1] *= corr[0], o[dt][2] *= corr[1], o[dt][3] *= corr[1];
        // O += P V: k = keys (two k16 steps), n = dims (32 n8 tiles, two per ldmatrix.x4.trans).
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
          for (int dp = 0; dp < kHeadDim / 16; dp++) {
            uint32_t b[4];
            ldsm_x4_trans(b, vs + tc_swz(16 * j + lr + 8 * (lm % 2), 2 * dp + lm / 2));
            mma16816<kHalf>(o[2 * dp], pf[j], b);
            mma16816<kHalf>(o[2 * dp + 1], pf[j], b + 2);
          }
      }
    }
    __syncthreads();  // every warp is done with the stage (and the converted chunk)
    if (c + 2 < cend) issue(c + 2);
    cp_commit();
  }
  // One split: normalize, gate, write the two rows. Several: the unnormalized partials (o, m, l).
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= T) continue;
    const int64_t row = int64_t(t) * kHeads + h;
    if (nsplit > 1) {
      float* po = part + (row * kFaSplits + split) * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; dt++)
        *reinterpret_cast<float2*>(po + dt * 8 + c2) = make_float2(o[dt][2 * r], o[dt][2 * r + 1]);
      if (c2 == 0)
        *reinterpret_cast<float2*>(part + int64_t(T) * kHeads * kFaSplits * kHeadDim + (row * kFaSplits + split) * 2) =
            make_float2(kF8 > 0 ? m[r] * kLn2 : m[r], l[r]);  // (natural units for the combine)
      continue;
    }
    const float inv = 1.f / l[r];
    const float* gate = qg + row * (2 * kHeadDim) + kHeadDim;
    float* dst = out + row * kHeadDim;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; dt++) {
      const int d = dt * 8 + c2;
      const float2 gv = *reinterpret_cast<const float2*>(gate + d);
      float2 v;
      v.x = o[dt][2 * r] * inv * (1.f / (1.f + expf(-gv.x)));
      v.y = o[dt][2 * r + 1] * inv * (1.f / (1.f + expf(-gv.y)));
      *reinterpret_cast<float2*>(dst + d) = v;
    }
  }
}

// Block per (token, 4 heads), 4 dims per thread: merge the key splits of the rows whose
// block was split, normalize, apply sigmoid(gate).
constexpr int kFaCombineHeads = 4;
__global__ void __launch_bounds__(kFaCombineHeads * kHeadDim / 4) attention_fa_combine_kernel(const float* qg, int T, int pos0,
                                                                                    const float* part, float* out) {
  const int t = blockIdx.x, p = pos0 + t;
  if (fa_splits((p | (kFaQ - 1)) + 1) == 1) return;
  const int h = blockIdx.y * kFaCombineHeads + threadIdx.x / (kHeadDim / 4), d = threadIdx.x % (kHeadDim / 4) * 4;
  const int64_t row = int64_t(t) * kHeads + h;
  const float* ml = part + int64_t(T) * kHeads * kFaSplits * kHeadDim + row * kFaSplits * 2;
  float mx = -INFINITY;
#pragma unroll
  for (int s = 0; s < kFaSplits; s++) mx = fmaxf(mx, ml[2 * s]);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float l = 0.f;
#pragma unroll
  for (int s = 0; s < kFaSplits; s++) {
    const float w = __expf(ml[2 * s] - mx);  // an empty split: m = -inf, o = l = 0
    const float4 o = *reinterpret_cast<const float4*>(part + (row * kFaSplits + s) * kHeadDim + d);
    acc.x += o.x * w, acc.y += o.y * w, acc.z += o.z * w, acc.w += o.w * w;
    l += ml[2 * s + 1] * w;
  }
  const float inv = 1.f / l;
  const float4 gv = *reinterpret_cast<const float4*>(qg + row * (2 * kHeadDim) + kHeadDim + d);
  float4 v;
  v.x = acc.x * inv * (1.f / (1.f + expf(-gv.x)));
  v.y = acc.y * inv * (1.f / (1.f + expf(-gv.y)));
  v.z = acc.z * inv * (1.f / (1.f + expf(-gv.z)));
  v.w = acc.w * inv * (1.f / (1.f + expf(-gv.w)));
  *reinterpret_cast<float4*>(out + row * kHeadDim + d) = v;
}

// P7-8: the NVFP4 cache of keys 0 .. n - 1 rounded to E4M3 once per prefill call (the values of
// fp4_chunk_to_e4m3), in the FP8 cache layout behind an identity page table, so attention_fa_kernel
// <uint8_t, 2> reads it: the same rows as attention_fa_kernel<Fp4Kv, 2>, which converts every chunk in
// every CTA (each key (position / 64) x 3 times per layer). Thread = (K or V, key, head, 32 dims).
// Stage: [identity page table, 256-byte aligned][K: n x 4 x 256][V]; mode 3 (Q K^T on FP4 MMAs, K read from
// the cache): [page table][V] only.
__host__ __device__ constexpr int64_t fa_stage_pt_bytes(int64_t cap) {  // cap: keys, a multiple of kPageTokens
  return (cap / kPageTokens * 4 + 255) / 256 * 256;
}
// first = 1: V only (mode 3 reads K as it is), right after the page table.
__global__ void kv_fp4_to_e4m3_kernel(const Fp4Kv* kc, const Fp4Kv* vc, const int32_t* page_table, int n,
                                      uint8_t* stage, int first) {
  const int64_t per = int64_t(n) * kKvHeads * 8;
  const int64_t u = int64_t(blockIdx.x) * blockDim.x + threadIdx.x + first * per;
  const int64_t cap = int64_t(n + kPageTokens - 1) / kPageTokens * kPageTokens;  // keys of the K region
  int32_t* ident = reinterpret_cast<int32_t*>(stage);
  if (u - first * per < cap / kPageTokens) ident[u - first * per] = int32_t(u - first * per);
  if (u >= 2 * per) return;
  const int kv = int(u / per), key = int(u % per / (kKvHeads * 8)), head = int(u / 8 % kKvHeads), col = int(u % 8);
  const uint8_t* row = kv_row_ptr(kv ? vc : kc, kv_row(page_table, key, head));
  const uint4 w4 = __ldg(reinterpret_cast<const uint4*>(row + col * 16));
  const uint32_t w[4] = {w4.x, w4.y, w4.z, w4.w};
  const uint32_t sc = __ldg(reinterpret_cast<const uint16_t*>(row + kHeadDim / 2 + 2 * col));
  uint32_t h[16], e[8];
  fp4x32_to_f16(w, sc, h);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(h[2 * i]));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(h[2 * i + 1]));
    e[i] = uint32_t(lo) | uint32_t(hi) << 16;
  }
  uint8_t* dst = stage + fa_stage_pt_bytes(cap) + (kv - first) * cap * kKvHeads * kHeadDim +
                 (int64_t(key) * kKvHeads + head) * kHeadDim + col * 32;
  reinterpret_cast<uint4*>(dst)[0] = make_uint4(e[0], e[1], e[2], e[3]);
  reinterpret_cast<uint4*>(dst)[1] = make_uint4(e[4], e[5], e[6], e[7]);
}

// ---------------------------------------------------------------- decode / verify with FP8 MMAs (P7-7)
//
// attention_tc_kernel's chunks, splits and producer pipeline with the inner loop of the FP8 prefill
// (attention_fa_kernel, mode 2): rows = the 6 T (token, head) pairs, 16 to a compute warp (A fragments:
// Q in E4M3 with a scale per row), S = Q K^T with K's E4M3 rows (FP8 KV: the staged rows, sw8; NVFP4:
// rounded to E4M3 by the producers), the softmax in base 2, P x 256 in E4M3 and O += P V with V^T
// (transposed by the producers, keys in the order of the P fragments). Default for NVFP4 KV (tc_f8).
constexpr int kT8Rows = 16;
#ifndef Q27_X_T8RAW
#define Q27_X_T8RAW 3
#endif
template <class KV>
constexpr int kT8Ahead = kKvFp8<KV> ? Q27_X_T8RAW - 2 : Q27_X_T8RAW - 1;  // FP8: K is read from its raw slot
constexpr int kT8Raw = Q27_X_T8RAW;
constexpr int kT8Grp = 4;  // compute warps of a single row tile (Q27_ATTN_TC8_GRP=0: 1)
// Per converted slot: (NVFP4: K and V in E4M3 rows) then V^T.
template <class KV>
constexpr int kT8Conv = (kKvFp4<KV> ? 2 * kTcChunk * kHeadDim : 0) + kFaVtBytes;
template <class KV>
constexpr int kT8Smem = kT8Raw * 2 * kTcChunk * kKvRowBytes<KV> + 2 * kT8Conv<KV>;

template <class KV, bool kBatch = false>
__global__ void __launch_bounds__(kT8Grp * 32 + kTcProd, 1) attention_tc8_kernel(const float* q, const int32_t* pos_dev, int T,
                                                                          const KV* kc, const KV* vc,
                                                                          const int32_t* page_table, int splits,
                                                                          DecPartial part, SeqLanes lanes) {
  static_assert(kKvFp8<KV> || kKvFp4<KV>, "FP8 or NVFP4 KV");
  pdl_trigger();
  if constexpr (kBatch) {
    const int z = blockIdx.z;
    const int64_t rows = int64_t(kLaneRows) * z * kHeads;
    q += rows * kHeadDim, pos_dev = lanes.pos + z, page_table = lane_sel(lanes.page_table, z);
    part.acc += rows * splits * kHeadDim, part.ml += rows * splits * 2;
  }
  constexpr bool kFp4 = kKvFp4<KV>;
  constexpr int kRowBytes = kKvRowBytes<KV>;
  constexpr int kUnits = kTcChunk * (kRowBytes / 16);
  constexpr int kStageBytes = 2 * kTcChunk * kRowBytes;
  constexpr int kR = kT8Raw, kAhead = kT8Ahead<KV>;
  extern __shared__ __align__(128) uint8_t tc_smem[];
  const uint32_t sbase = uint32_t(__cvta_generic_to_shared(tc_smem));
  const uint32_t cbase = sbase + kR * kStageBytes;
  const int g = blockIdx.x, split = blockIdx.y;
  const int ncw = blockDim.x / 32 - kTcProdWarps;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int pos0 = *pos_dev, kmax = pos0 + T;
  const int nchunks = (kmax + kTcChunk - 1) / kTcChunk;
  if (split >= nchunks) return;
  const int nmine = (nchunks - split + splits - 1) / splits;

  if (warp >= ncw) {
    // ---- producers (as in attention_tc_kernel; FP8 rows staged in the sw8 layout, read in place as K)
    const int ptid = threadIdx.x - ncw * 32;
    auto page_of = [&](int i) { return i < nmine ? __ldg(page_table + (split + i * splits) * kTcChunk / kPageTokens) : 0; };
    auto issue = [&](int i, int phase, int page) {
      if (i >= nmine) return;
      const int c = split + i * splits;
      const uint32_t st = sbase + (i % kR) * kStageBytes;
#pragma unroll
      for (int j = 0; j < (kUnits + kTcProd - 1) / kTcProd; j++) {
        const int u = ptid + j * kTcProd, r = u / (kRowBytes / 16), col = u % (kRowBytes / 16);
        if (kUnits % kTcProd != 0 && u >= kUnits) break;
        const int key = c * kTcChunk + r;
        if (phase == 0 && key >= pos0) continue;
        if (phase == 1 && key < pos0) continue;
        const bool ok = key < kmax;
        const int64_t row = (int64_t(page) * kPageTokens + key % kPageTokens) * kKvHeads + g;
        const uint32_t off = kFp4 ? r * kRowBytes + col * 16 : sw8(r, col);
        cp_async16z(st + off, kv_row_ptr(kc, row) + col * 16, ok);
        cp_async16z(st + kTcChunk * kRowBytes + off, kv_row_ptr(vc, row) + col * 16, ok);
      }
    };
    for (int i = 0; i < kAhead; i++) {
      issue(i, 0, page_of(i));
      if (i + 1 < kAhead) cp_commit();
    }
    int page = page_of(kAhead);
    pdl_wait();
    if (nmine - 1 < kAhead) issue(nmine - 1, 1, page_of(nmine - 1));
    cp_commit();
    for (int i = 0; i <= nmine; i++) {
      if (i < nmine) {
        issue(i + kAhead, 2, page);
        cp_commit();
        page = page_of(i + kAhead + 1);
        if (i == nmine - 1)
          cp_wait<0>();
        else
          cp_wait<kAhead>();
        asm volatile("bar.sync 1, %0;" ::"n"(kTcProd));  // every producer's copies of chunk i landed
        const uint32_t src = sbase + (i % kR) * kStageBytes, slot = cbase + (i % 2) * kT8Conv<KV>;
        uint32_t vraw = src + kTcChunk * kRowBytes;
        if constexpr (kFp4) {
          fp4_chunk_to_e4m3<kTcChunk, kTcProd>(src, slot, ptid);
          asm volatile("bar.sync 1, %0;" ::"n"(kTcProd));
          vraw = slot + kTcChunk * kHeadDim;
        }
        const uint32_t vtbase = slot + kT8Conv<KV> - kFaVtBytes;
#ifdef Q27_X_T8NOVT
        if (false)
#endif
        // V^T as in attention_fa_kernel: thread = (4 dims, key group j, half hh).
#pragma unroll
        for (int x = 0; x < kHeadDim / 4 * 4 * 2 / kTcProd; x++) {
          const int u = ptid + x * kTcProd, j = u & 3, rest = u >> 5;
          const int dg = (rest & 7) * 8 + (u >> 2 & 7), hh = rest >> 3;
          uint32_t w[4];
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int key = 16 * hh + 2 * j + (k & 1) + 8 * (k >> 1);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w[k]) : "r"(vraw + sw8(key, dg / 4) + 4 * (dg % 4)));
          }
          const uint32_t t0 = __byte_perm(w[0], w[1], 0x5140), t1 = __byte_perm(w[2], w[3], 0x5140);
          const uint32_t t2 = __byte_perm(w[0], w[1], 0x7362), t3 = __byte_perm(w[2], w[3], 0x7362);
          const uint32_t y[4] = {__byte_perm(t0, t1, 0x5410), __byte_perm(t0, t1, 0x7632), __byte_perm(t2, t3, 0x5410),
                                 __byte_perm(t2, t3, 0x7632)};
#pragma unroll
          for (int k = 0; k < 4; k++)
            asm volatile("st.shared.u32 [%0], %1;" ::"r"(vtbase + vt_off(4 * dg + k, hh) + 4 * j), "r"(y[k]));
        }
      }
      __syncthreads();  // chunk i ready for the compute warps; chunk i - 1 consumed
    }
    return;
  }

  // ---- compute warps: rows 16 tile .. 16 tile + 15 of the 6 T (token, head) rows. One row tile (T <= 2,
  // a decode) runs on kT8Grp warps: warp w computes the scores of key tile w (8 keys) and the P V of a
  // quarter of the dims; the row maxima and P go through shared memory and every warp then holds the
  // whole P fragment, so the row sums and all element operations are those of a tile on one warp
  // (a decode row is bit-identical to the same row in a verify).
  pdl_wait();
  __shared__ float x_mx[kT8Grp][32][2];
  __shared__ uint32_t x_p[kT8Grp][32];
  auto compute = [&](auto grp_c) {
    constexpr bool kGrp = decltype(grp_c)::value;
    constexpr int kDp = kHeadDim / 16 / (kGrp ? kT8Grp : 1);  // 16-dim tiles of P V per warp
    const int tile = kGrp ? 0 : warp, gw = kGrp ? warp : 0, dp0 = gw * kDp;
    const int lm = lane / 8, lr = lane % 8;
    const int r0 = lane / 4, c2 = 2 * (lane % 4);
    const int nrows = T * kGqa;
    const int ra = tile * kT8Rows + r0, rb = ra + 8;
    auto row_kt = [&](int r) { return r < nrows ? pos0 + r / kGqa + 1 : 0; };  // keys row r sees
    auto row_q = [&](int r) { return q + (int64_t(r / kGqa) * kHeads + g * kGqa + r % kGqa) * kHeadDim; };
    const int kta = row_kt(ra), ktb = row_kt(rb);
    const int ktile = row_kt(min(tile * kT8Rows + kT8Rows - 1, nrows - 1));
    const int kmin = row_kt(tile * kT8Rows);  // chunks below it need no mask
    uint32_t qa[kHeadDim / 32][4];
    float sq[2];
    {
      auto ld4 = [&](int r, int d) {
        if (r >= nrows) return make_float4(0.f, 0.f, 0.f, 0.f);
        return *reinterpret_cast<const float4*>(row_q(r) + d);
      };
      float inv[2];
#pragma unroll
      for (int x = 0; x < 2; x++) {
        float mx = 0.f;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; kk++) {
          const float4 v = ld4(x ? rb : ra, kk * 16 + 2 * c2);
          mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        inv[x] = mx > 0.f ? 448.f / mx : 0.f;
        sq[x] = mx * (1.f / (448.f * 16.f)) * kLog2e;  // 1 / sqrt(256), base 2
      }
#pragma unroll
      for (int kk = 0; kk < kHeadDim / 32; kk++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int x = i % 2;
          const float4 v = ld4(x ? rb : ra, kk * 32 + 16 * (i / 2) + 2 * c2);
          qa[kk][i] = pack_e4m3x4(v.x * inv[x], v.y * inv[x], v.z * inv[x], v.w * inv[x]);
        }
    }
    float o[2 * kDp][4];
#pragma unroll
    for (int i = 0; i < 2 * kDp; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};  // rows ra, rb (m in log2 units)

    __syncthreads();  // chunk 0 staged
    for (int i = 0; i < nmine; i++) {
      const int c = split + i * splits;
      const uint32_t slot = cbase + (i % 2) * kT8Conv<KV>;
      const uint32_t ks = kFp4 ? slot : sbase + (i % kR) * kStageBytes;
      const uint32_t vt = slot + kT8Conv<KV> - kFaVtBytes;
#ifdef Q27_X_T8NOCOMP  // (experiment builds: leave out a stage; results wrong)
      if (false) {
#else
      if (c * kTcChunk < ktile) {
#endif
        // S: key tiles nt (8 keys) 0 .. 3, or the warp's own one (kGrp).
        constexpr int kNt = kGrp ? 1 : 4;
        float sc[kNt][4] = {};
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 32; kk++)
#pragma unroll
          for (int p = 0; p < (kGrp ? 1 : 2); p++) {
            uint32_t b[4];
            ldsm_x4(b, ks + sw8(16 * (kGrp ? gw / 2 : p) + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            if constexpr (kGrp) {
              const bool hi = gw % 2;  // (a register array indexed at run time would go to local memory)
              const uint32_t bk[2] = {hi ? b[2] : b[0], hi ? b[3] : b[1]};
              mma_e4m3_16832(sc[0], qa[kk], bk);
            } else {
              mma_e4m3_16832(sc[2 * p], qa[kk], b);
              mma_e4m3_16832(sc[2 * p + 1], qa[kk], b + 2);
            }
          }
        const bool diag = c * kTcChunk + kTcChunk > kmin;
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < kNt; n++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            sc[n][e] *= sq[e / 2];
            const int key = c * kTcChunk + (kGrp ? gw : n) * 8 + c2 + (e & 1);
            if (diag && key >= (e < 2 ? kta : ktb)) sc[n][e] = -INFINITY;
            mx[e / 2] = fmaxf(mx[e / 2], sc[n][e]);
          }
#pragma unroll
        for (int x = 0; x < 2; x++) {
          mx[x] = fmaxf(mx[x], __shfl_xor_sync(0xffffffffu, mx[x], 1));
          mx[x] = fmaxf(mx[x], __shfl_xor_sync(0xffffffffu, mx[x], 2));
        }
#ifdef Q27_X_T8NOXCH
        if constexpr (false) {
#else
        if constexpr (kGrp) {
#endif
          x_mx[gw][lane][0] = mx[0], x_mx[gw][lane][1] = mx[1];
          asm volatile("bar.sync 2, %0;" ::"n"(kT8Grp * 32));
#pragma unroll
          for (int x = 0; x < 2; x++)
            mx[x] = fmaxf(fmaxf(x_mx[0][lane][x], x_mx[1][lane][x]), fmaxf(x_mx[2][lane][x], x_mx[3][lane][x]));
        }
        float corr[2];
#pragma unroll
        for (int x = 0; x < 2; x++) {
          const float mn = fmaxf(m[x], mx[x]);
          corr[x] = mn == -INFINITY ? 1.f : ex2_approx(m[x] - mn);
          m[x] = mn;
        }
        // P x 256 = 2^(s - m + 8) as one E4M3 A fragment over the 32 keys in the permuted order of V^T.
        uint32_t h[2][4];
#pragma unroll
        for (int x = 0; x < 2; x++) {
          const float b = 8.f - (m[x] == -INFINITY ? 0.f : m[x]);
#pragma unroll
          for (int n = 0; n < kNt; n++) h[x][n] = exp2_e4m3x2(sc[n][2 * x] + b, sc[n][2 * x + 1] + b);
        }
#ifdef Q27_X_T8NOXCH
        if constexpr (false) {
#else
        if constexpr (kGrp) {
#endif
          x_p[gw][lane] = h[0][0] | h[1][0] << 16;
          asm volatile("bar.sync 2, %0;" ::"n"(kT8Grp * 32));
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const uint32_t w = x_p[n][lane];
            h[0][n] = w & 0xffffu, h[1][n] = w >> 16;
          }
        }
        uint32_t pf[4];
        float ps[2];
#pragma unroll
        for (int x = 0; x < 2; x++) {
          pf[x] = h[x][0] | h[x][1] << 16;
          pf[2 + x] = h[x][2] | h[x][3] << 16;
          ps[x] = sum_e4m3x4(pf[x]) + sum_e4m3x4(pf[2 + x]);
        }
#pragma unroll
        for (int x = 0; x < 2; x++) {
          ps[x] += __shfl_xor_sync(0xffffffffu, ps[x], 1);
          ps[x] += __shfl_xor_sync(0xffffffffu, ps[x], 2);
          l[x] = l[x] * corr[x] + ps[x];
        }
#pragma unroll
        for (int dp = 0; dp < kDp; dp++) {
          float t[2][4] = {};
          uint32_t b[4];
          ldsm_x4(b, vt + vt_off(16 * (dp0 + dp) + 8 * (lm / 2) + lr, lm % 2));
          mma_e4m3_16832(t[0], pf, b);
          mma_e4m3_16832(t[1], pf, b + 2);
#pragma unroll
          for (int x = 0; x < 2; x++)
#pragma unroll
            for (int e = 0; e < 4; e++) o[2 * dp + x][e] = o[2 * dp + x][e] * corr[e / 2] + t[x][e];
        }
      }
      __syncthreads();  // chunk i consumed, chunk i + 1 staged
    }
    // Partials of the rows with keys in this split (o and l carry the factor 256; m in natural units).
#pragma unroll
    for (int x = 0; x < 2; x++) {
      const int r = x ? rb : ra, kt = x ? ktb : kta;
      if (r >= nrows || split * kTcChunk >= kt) continue;
      const int64_t row = int64_t(r / kGqa) * kHeads + g * kGqa + r % kGqa;
      float* dst = part.acc + (row * splits + split) * kHeadDim + dp0 * 16;
#pragma unroll
      for (int dt = 0; dt < 2 * kDp; dt++)
        *reinterpret_cast<float2*>(dst + dt * 8 + c2) = make_float2(o[dt][2 * x], o[dt][2 * x + 1]);
      if (c2 == 0 && gw == 0)
        part.ml[(row * splits + split) * 2] = m[x] * kLn2, part.ml[(row * splits + split) * 2 + 1] = l[x];
    }
  };
  if (T * kGqa <= kT8Rows && ncw == kT8Grp)
    compute(std::true_type{});
  else
    compute(std::false_type{});
}

// Block per (token, head): merge the splits, normalize, apply sigmoid(gate).
// chunk > 0: the splits of attention_tc_kernel (chunk c in split c % splits).
// kBatch (P6-10): row t belongs to lane t / kLaneRows.
template <bool kBatch = false>
__global__ void attention_combine_kernel(const float* qg, int ldq, const int32_t* pos_dev, int splits, int min_keys,
                                         int chunk, DecPartial part, float* out, uint8_t* q8, float in_scale,
                                         const SpecTree* tree, SeqLanes lanes) {
  pdl_trigger();
  const int64_t row = blockIdx.x;  // t * 24 + h
  const int d = threadIdx.x, t = blockIdx.x / kHeads, h = blockIdx.x % kHeads;
  int tl = t;
  if constexpr (kBatch) pos_dev = lanes.pos + t / kLaneRows, tl = t % kLaneRows;
  const int pos0 = *pos_dev;
  pdl_wait();
  const int keys = pos0 + (tree ? tree->depth[t] : tl) + 1;
  const int per = chunk > 0 ? chunk : keys_per_split(keys, splits, min_keys);
  const int used = min(splits, (keys + per - 1) / per);
  float M = -INFINITY;
  for (int s = 0; s < used; s++) M = fmaxf(M, part.ml[(row * splits + s) * 2]);
  float a = 0.f, L = 0.f;
  for (int s = 0; s < used; s++) {
    float ms = part.ml[(row * splits + s) * 2];
    float c = ms == -INFINITY ? 0.f : __expf(ms - M);
    a += c * part.acc[(row * splits + s) * kHeadDim + d];
    L += c * part.ml[(row * splits + s) * 2 + 1];
  }
  float gate = qg[int64_t(t) * ldq + h * (2 * kHeadDim) + kHeadDim + d];
  float y = a / L * (1.f / (1.f + __expf(-gate)));
  out[row * kHeadDim + d] = y;
  if (q8) q8[row * kHeadDim + d] = to_kv<uint8_t>(y / in_scale);
}

// Moves the KV of the accepted tree path (rows cpath[i], at slots pos + cpath[i]) to pos + i. Block
// per (layer, KV head, K or V): all reads before the writes (a row's source may be another's target).
template <class KV>
__global__ void kv_commit_kernel(KvLayer* layers, const int32_t* page_table, const int32_t* pos_dev,
                                 const int32_t* ncommit, const int32_t* cpath) {
  pdl_trigger();
  pdl_wait();
  const int layer = blockIdx.x, head = blockIdx.y / 2, d = threadIdx.x;
  KV* base = static_cast<KV*>(blockIdx.y % 2 ? layers[layer].v : layers[layer].k);
  const int pos = *pos_dev, n = *ncommit;
  constexpr int kElems = kKvFp4<KV> ? kKvRowBytes<KV> : kHeadDim;  // elements of a row (NVFP4: bytes)
  const bool on = d < kElems;
  KV val[kSpecBlock];
#pragma unroll
  for (int i = 0; i < kSpecBlock; i++)
    if (on && i < n && cpath[i] != i) val[i] = base[kv_row(page_table, pos + cpath[i], head) * kElems + d];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kSpecBlock; i++)
    if (on && i < n && cpath[i] != i) base[kv_row(page_table, pos + i, head) * kElems + d] = val[i];
}

int g_attn_sms = 0;
}  // namespace

void attn_prep(const float* qg, int ldq, const float* k, const float* v, int ldkv, const bf16* q_norm,
               const bf16* k_norm, int T, const int32_t* pos_dev, KvLayer kv, KvType kv_type,
               const int32_t* page_table, float* q_out, cudaStream_t s, const SpecTree* tree, RopePos rope) {
  if (T == 0) return;
  dim3 grid(T, kHeads + kKvHeads);
  if (kv_type == KvType::BF16)
    launch_pdl(attn_prep_kernel<bf16>, dim3(grid), dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm, pos_dev,
               static_cast<bf16*>(kv.k), static_cast<bf16*>(kv.v), page_table, q_out, tree, rope, SeqLanes{});
  else if (kv_type == KvType::NVFP4)
    launch_pdl(attn_prep_kernel<Fp4Kv>, dim3(grid), dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm,
               pos_dev, static_cast<Fp4Kv*>(kv.k), static_cast<Fp4Kv*>(kv.v), page_table, q_out, tree, rope, SeqLanes{});
  else
    launch_pdl(attn_prep_kernel<uint8_t>, dim3(grid), dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm,
               pos_dev, static_cast<uint8_t*>(kv.k), static_cast<uint8_t*>(kv.v), page_table, q_out, tree, rope,
               SeqLanes{});
  Q27_CUDA(cudaGetLastError());
}

size_t attention_prefill_ws_bytes(int T) { return size_t(T) * kHeads * kFaSplits * (kHeadDim + 2) * sizeof(float); }
size_t attention_prefill_stage_bytes(int keys, bool v_only) {
  const int64_t cap = round_up(keys, kPageTokens);
  return size_t(fa_stage_pt_bytes(cap) + (v_only ? 1 : 2) * cap * kKvHeads * kHeadDim);
}
bool attention_prefill_stage_on() {
  const char* e = getenv("Q27_ATTN_FP4_STAGE");
  return !e || !*e || atoi(e) != 0;
}
int attention_prefill_mode(KvType kv_type, bool staged) {
  const char* e = getenv("Q27_ATTN_FP8");
  if (e && *e) return atoi(e);
  return kv_type == KvType::NVFP4 && staged && attention_prefill_stage_on() ? 3 : 2;
}

void attention(const float* q, const float* qg, int T, int pos0, KvLayer kv, KvType kv_type,
               const int32_t* page_table, float* out, void* ws, size_t ws_bytes, cudaStream_t s, void* stage,
               size_t stage_bytes) {
  if (T == 0) return;
  // Tensor-core flash attention (Q27_ATTN_PREFILL_TC=0: the CUDA-core kernel below).
  static const bool fa = !getenv("Q27_ATTN_PREFILL_TC") || atoi(getenv("Q27_ATTN_PREFILL_TC")) != 0;
  if (fa) {
    static bool init = [] {
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFaSmem));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<uint8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemFp8));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<uint8_t, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemF8[1]));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<uint8_t, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemF8[2]));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<Fp4Kv>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemFp4));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<Fp4Kv, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemFp4F8));
      Q27_CUDA(cudaFuncSetAttribute(attention_fa_kernel<Fp4Kv, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kFaSmemFp4Q4));
      return true;
    }();
    (void)init;
    Q27_CHECK(ws_bytes >= attention_prefill_ws_bytes(T), "attention: workspace too small");
    float* part = static_cast<float*>(ws);
    const dim3 grid(unsigned(ceil_div(pos0 % kFaQ + T, kFaQ) * kFaSplits), kHeads / kFaHeads);
    if (kv_type == KvType::BF16)
      attention_fa_kernel<bf16><<<grid, kFaWarps * 32, kFaSmem, s>>>(
          q, qg, T, pos0, static_cast<const bf16*>(kv.k), static_cast<const bf16*>(kv.v), page_table, out, part);
    else if (kv_type == KvType::NVFP4) {
      // Q27_ATTN_FP8 (attention_prefill_mode; default 3 with a stage, else 2): 3 = Q K^T on the FP4 MMA with K
      // from the cache, V staged in E4M3; 2 = the values rounded to E4M3 and the FP8 MMAs (staged when the
      // stage holds K and V: the same rows as per CTA); otherwise exact f16 and f16 MMAs.
      const int f8 = attention_prefill_mode(kv_type, stage != nullptr);
      const auto* kc = static_cast<const Fp4Kv*>(kv.k);
      const auto* vc = static_cast<const Fp4Kv*>(kv.v);
      const int n = pos0 + T;
      const bool staged = stage && attention_prefill_stage_on() &&
                          stage_bytes >= attention_prefill_stage_bytes(n, f8 == 3);
      Q27_CHECK(f8 != 3 || staged, "attention: Q27_ATTN_FP8=3 needs the NVFP4 stage");
      if ((f8 == 2 || f8 == 3) && staged) {
        // P7-8: the context rounded to E4M3 once into the stage (mode 3: V only), then the FP8 KV kernel
        // (mode 2, the same rows) or the FP4 Q K^T (mode 3).
        const int first = f8 == 3;
        const int64_t cap = round_up(n, kPageTokens);
        const int64_t threads = std::max<int64_t>((2 - first) * int64_t(n) * kKvHeads * 8, cap / kPageTokens);
        auto* st = static_cast<uint8_t*>(stage);
        kv_fp4_to_e4m3_kernel<<<unsigned(ceil_div(threads, 256)), 256, 0, s>>>(kc, vc, page_table, n, st, first);
        const uint8_t* k8 = st + fa_stage_pt_bytes(cap);
        const auto* ident = reinterpret_cast<const int32_t*>(st);
        const uint8_t* v8 = k8 + (first ? 0 : cap * kKvHeads * kHeadDim);
        if (f8 == 3)
          attention_fa_kernel<Fp4Kv, 3><<<grid, kFaWarps * 32, kFaSmemFp4Q4, s>>>(
              q, qg, T, pos0, kc, vc, page_table, out, part, v8, ident);
        else
          attention_fa_kernel<uint8_t, 2><<<grid, kFaWarps * 32, kFaSmemF8[2], s>>>(q, qg, T, pos0, k8, v8, ident, out, part);
      } else if (f8 == 2)
        attention_fa_kernel<Fp4Kv, 2><<<grid, kFaWarps * 32, kFaSmemFp4F8, s>>>(q, qg, T, pos0, kc, vc, page_table, out, part);
      else
        attention_fa_kernel<Fp4Kv><<<grid, kFaWarps * 32, kFaSmemFp4, s>>>(q, qg, T, pos0, kc, vc, page_table, out, part);
    }
    else {
      // FP8 MMAs (FP8 KV): Q27_ATTN_FP8 = 2 (Q K^T and P V, the default), 1 (P V) or 0 (BF16), read per call.
      const int f8 = attention_prefill_mode(kv_type, false);
      const auto* kc = static_cast<const uint8_t*>(kv.k);
      const auto* vc = static_cast<const uint8_t*>(kv.v);
      if (f8 == 2)
        attention_fa_kernel<uint8_t, 2><<<grid, kFaWarps * 32, kFaSmemF8[2], s>>>(q, qg, T, pos0, kc, vc, page_table, out, part);
      else if (f8 == 1)
        attention_fa_kernel<uint8_t, 1><<<grid, kFaWarps * 32, kFaSmemF8[1], s>>>(q, qg, T, pos0, kc, vc, page_table, out, part);
      else
        attention_fa_kernel<uint8_t><<<grid, kFaWarps * 32, kFaSmemFp8, s>>>(q, qg, T, pos0, kc, vc, page_table, out, part);
    }
    Q27_CUDA(cudaGetLastError());
    if (fa_splits(((pos0 + T - 1) | (kFaQ - 1)) + 1) > 1)
      attention_fa_combine_kernel<<<dim3(T, kHeads / kFaCombineHeads), kFaCombineHeads * kHeadDim / 4, 0, s>>>(qg, T, pos0, part, out);
    Q27_CUDA(cudaGetLastError());
    return;
  }
  Q27_CHECK(kv_type != KvType::NVFP4, "attention: the CUDA-core prefill kernel has no NVFP4 KV");
  dim3 grid(kHeads, unsigned(ceil_div(T, kBq)));
  if (kv_type == KvType::BF16) {
    static bool init = [] {
      Q27_CUDA(cudaFuncSetAttribute(attention_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttnSmem)));
      return true;
    }();
    (void)init;
    attention_kernel<bf16><<<grid, 256, kAttnSmem, s>>>(q, qg, T, pos0, static_cast<const bf16*>(kv.k),
                                                        static_cast<const bf16*>(kv.v), page_table, out);
  } else {
    static bool init = [] {
      Q27_CUDA(cudaFuncSetAttribute(attention_kernel<uint8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(kAttnSmem)));
      return true;
    }();
    (void)init;
    attention_kernel<uint8_t><<<grid, 256, kAttnSmem, s>>>(q, qg, T, pos0, static_cast<const uint8_t*>(kv.k),
                                                           static_cast<const uint8_t*>(kv.v), page_table, out);
  }
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27

namespace q27 {

// Compute warps of attention_tc8_kernel: one per 16-row tile, a single tile (T <= 2) on kT8Grp.
static int tc8_warps(int T) {
  static const bool grp = !getenv("Q27_ATTN_TC8_GRP") || atoi(getenv("Q27_ATTN_TC8_GRP")) != 0;
  const int tiles = ceil_div(T * kGqa, kT8Rows);
  return tiles == 1 && grp ? kT8Grp : tiles;
}
// Q27_ATTN_TC_F8 (read per launch): unset or empty = on for NVFP4 KV only (P7-7: the verify is MMA-bound
// there; FP8 KV is near its read bound), 1 = on for FP8 and NVFP4 KV, 0 = off.
static bool tc_f8(KvType kv_type) {
  static const bool init = [] {
    Q27_CUDA(cudaFuncSetAttribute(attention_tc8_kernel<uint8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kT8Smem<uint8_t>));
    Q27_CUDA(cudaFuncSetAttribute(attention_tc8_kernel<Fp4Kv>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kT8Smem<Fp4Kv>));
    Q27_CUDA(cudaFuncSetAttribute(attention_tc8_kernel<uint8_t, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kT8Smem<uint8_t>));
    Q27_CUDA(cudaFuncSetAttribute(attention_tc8_kernel<Fp4Kv, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kT8Smem<Fp4Kv>));
    return true;
  }();
  (void)init;
  if (kv_type == KvType::BF16) return false;
  const char* e = getenv("Q27_ATTN_TC_F8");
  if (!e || !*e) return kv_type == KvType::NVFP4;
  return atoi(e) != 0;
}

void attention_decode(const float* q, const float* qg, int ldq, int T, const int32_t* pos_dev, KvLayer kv,
                      KvType kv_type, const int32_t* page_table, float* out, void* ws, size_t ws_bytes,
                      cudaStream_t s, uint8_t* q8, float in_scale, const SpecTree* tree) {
  if (T == 0) return;
  Q27_CHECK(T <= kDecodeMaxT, "attention_decode: T must be <= 32");
  if (!g_attn_sms) Q27_CUDA(cudaDeviceGetAttribute(&g_attn_sms, cudaDevAttrMultiProcessorCount, 0));
  // About 2 CTAs per SM; short contexts leave most splits empty (they exit at once).
  const int waves = getenv("Q27_ATTN_WAVES") ? atoi(getenv("Q27_ATTN_WAVES")) : 2;  // read per launch (configure)
  const int min_keys = getenv("Q27_ATTN_MIN_KEYS") ? atoi(getenv("Q27_ATTN_MIN_KEYS")) : 64;
  const int prefetch = getenv("Q27_ATTN_PREFETCH") ? atoi(getenv("Q27_ATTN_PREFETCH")) : kDecPrefetchKeys;
  // The split count of a single token for every T (bit-identical rows); short contexts leave most empty.
  // Chains of up to 8 tokens (decode, the chain verify; BF16 or FP8 KV): tensor cores, every chunk read once.
  // Q27_ATTN_TC=0: the CUDA-core kernel. The split count is fixed per process (bit-identical rows).
  const bool tc_on = !getenv("Q27_ATTN_TC") || atoi(getenv("Q27_ATTN_TC")) != 0;
  const bool tc = tc_on && !tree && T <= kTcMaxT;
  const int tc_splits = getenv("Q27_ATTN_TC_SPLITS") ? atoi(getenv("Q27_ATTN_TC_SPLITS"))
                                                     : std::max(1, g_attn_sms / kKvHeads);
  const int splits = std::min(kDecodeMaxSplits,
                              tc ? tc_splits : std::max(1, int(ceil_div(waves * g_attn_sms, kKvHeads))));
  const int64_t rows = int64_t(T) * kHeads;
  DecPartial part;
  part.acc = static_cast<float*>(ws);
  part.ml = part.acc + rows * splits * kHeadDim;
  Q27_CHECK(size_t(rows * splits * (kHeadDim + 2)) * 4 <= ws_bytes, "attention_decode: workspace too small");
  dim3 grid(kKvHeads, splits);
  if (tc) {
    static bool init = [] {
      Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kTcSmemOf<bf16>));
      Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<uint8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kTcSmemOf<uint8_t>));
      Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<Fp4Kv>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kTcSmemOf<Fp4Kv>));
      return true;
    }();
    (void)init;
    if (tc_f8(kv_type)) {
      const dim3 block8(tc8_warps(T) * 32 + kTcProd);
      if (kv_type == KvType::NVFP4)
        launch_pdl(attention_tc8_kernel<Fp4Kv>, grid, block8, kT8Smem<Fp4Kv>, s, q, pos_dev, T,
                   static_cast<const Fp4Kv*>(kv.k), static_cast<const Fp4Kv*>(kv.v), page_table, splits, part,
                   SeqLanes{});
      else
        launch_pdl(attention_tc8_kernel<uint8_t>, grid, block8, kT8Smem<uint8_t>, s, q, pos_dev, T,
                   static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v), page_table, splits, part,
                   SeqLanes{});
    } else {
    const bool dec4 = T == 1 && (!getenv("Q27_ATTN_TC_DEC") || atoi(getenv("Q27_ATTN_TC_DEC")) != 0);
    // Compute warps: one per n8 tile of the 6 T (token, head) columns while there are at most 4 tiles,
    // else 4 + a group of 4 for the last 1 or 2 (T = 6..8); a decode's tile on a group. Then the producers.
    const int ntile = ceil_div(T * kGqa, 8);
    const int ncw = dec4 ? kTcGrpWarps : ntile <= 4 ? ntile : 4 + kTcGrpWarps;
    const dim3 block(ncw * 32 + kTcProd);
    if (kv_type == KvType::BF16)
      launch_pdl(attention_tc_kernel<bf16>, grid, block, kTcSmemOf<bf16>, s, q, pos_dev, T,
                 static_cast<const bf16*>(kv.k), static_cast<const bf16*>(kv.v), page_table, splits, part, SeqLanes{});
    else if (kv_type == KvType::NVFP4)
      launch_pdl(attention_tc_kernel<Fp4Kv>, grid, block, kTcSmemOf<Fp4Kv>, s, q, pos_dev, T,
                 static_cast<const Fp4Kv*>(kv.k), static_cast<const Fp4Kv*>(kv.v), page_table, splits, part, SeqLanes{});
    else
      launch_pdl(attention_tc_kernel<uint8_t>, grid, block, kTcSmemOf<uint8_t>, s, q, pos_dev, T,
                 static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v), page_table, splits, part, SeqLanes{});
    }
  } else if (kv_type == KvType::BF16)
    launch_pdl(attention_decode_kernel<bf16>, dim3(grid), dim3(kDecWarps * 32), 0, s, q, pos_dev, T,
               static_cast<const bf16*>(kv.k), static_cast<const bf16*>(kv.v), page_table, splits, min_keys, prefetch,
               part, tree);
  else if (kv_type == KvType::NVFP4)
    launch_pdl(attention_decode_kernel<Fp4Kv>, dim3(grid), dim3(kDecWarps * 32), 0, s, q, pos_dev, T,
               static_cast<const Fp4Kv*>(kv.k), static_cast<const Fp4Kv*>(kv.v), page_table, splits, min_keys,
               prefetch, part, tree);
  else
    launch_pdl(attention_decode_kernel<uint8_t>, dim3(grid), dim3(kDecWarps * 32), 0, s, q, pos_dev, T,
               static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v), page_table, splits, min_keys,
               prefetch, part, tree);
  Q27_CUDA(cudaGetLastError());
  launch_pdl(attention_combine_kernel<false>, dim3(unsigned(rows)), dim3(kHeadDim), 0, s, qg, ldq, pos_dev, splits,
             min_keys, tc ? kTcChunk : 0, part, out, q8, in_scale, tree, SeqLanes{});
  Q27_CUDA(cudaGetLastError());
}

void attn_prep_lanes(const float* qg, int ldq, const float* k, const float* v, int ldkv, const bf16* q_norm,
                     const bf16* k_norm, const SeqLanes& lanes, KvLayer kv, KvType kv_type, float* q_out,
                     cudaStream_t s) {
  Q27_CHECK(lanes.n >= 1 && lanes.n <= kMaxLanes, "attn_prep_lanes: 1..8 lanes");
  const dim3 grid(kLaneRows * lanes.n, kHeads + kKvHeads);
  if (kv_type == KvType::BF16)
    launch_pdl(attn_prep_kernel<bf16, true>, grid, dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm,
               lanes.pos, static_cast<bf16*>(kv.k), static_cast<bf16*>(kv.v), lanes.page_table[0], q_out,
               static_cast<const SpecTree*>(nullptr), RopePos{}, lanes);
  else if (kv_type == KvType::NVFP4)
    launch_pdl(attn_prep_kernel<Fp4Kv, true>, grid, dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm,
               lanes.pos, static_cast<Fp4Kv*>(kv.k), static_cast<Fp4Kv*>(kv.v), lanes.page_table[0], q_out,
               static_cast<const SpecTree*>(nullptr), RopePos{}, lanes);
  else
    launch_pdl(attn_prep_kernel<uint8_t, true>, grid, dim3(kHeadDim), 0, s, qg, ldq, k, v, ldkv, q_norm, k_norm,
               lanes.pos, static_cast<uint8_t*>(kv.k), static_cast<uint8_t*>(kv.v), lanes.page_table[0], q_out,
               static_cast<const SpecTree*>(nullptr), RopePos{}, lanes);
  Q27_CUDA(cudaGetLastError());
}

// The tensor-core kernel with the lanes on blockIdx.z, then the combine (rows of all lanes). A lane's
// launch geometry (splits, warps) is that of attention_decode for T = kLaneRows, so its rows are the same.
void attention_decode_lanes(const float* q, const float* qg, int ldq, const SeqLanes& lanes, KvLayer kv,
                            KvType kv_type, float* out, void* ws, size_t ws_bytes, cudaStream_t s, uint8_t* q8,
                            float in_scale) {
  Q27_CHECK(lanes.n >= 1 && lanes.n <= kMaxLanes, "attention_decode_lanes: 1..8 lanes");
  if (!g_attn_sms) Q27_CUDA(cudaDeviceGetAttribute(&g_attn_sms, cudaDevAttrMultiProcessorCount, 0));
  constexpr int T = kLaneRows;
  const int splits = std::min(kDecodeMaxSplits, getenv("Q27_ATTN_TC_SPLITS") ? atoi(getenv("Q27_ATTN_TC_SPLITS"))
                                                                              : std::max(1, g_attn_sms / kKvHeads));
  const int64_t rows = int64_t(T) * lanes.n * kHeads;
  DecPartial part;
  part.acc = static_cast<float*>(ws);
  part.ml = part.acc + rows * splits * kHeadDim;
  Q27_CHECK(size_t(rows * splits * (kHeadDim + 2)) * 4 <= ws_bytes, "attention_decode_lanes: workspace too small");
  static bool init = [] {
    Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<bf16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kTcSmemOf<bf16>));
    Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<uint8_t, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kTcSmemOf<uint8_t>));
    Q27_CUDA(cudaFuncSetAttribute(attention_tc_kernel<Fp4Kv, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kTcSmemOf<Fp4Kv>));
    return true;
  }();
  (void)init;
  const int ntile = ceil_div(T * kGqa, 8);
  const dim3 grid(kKvHeads, splits, lanes.n), block((ntile <= 4 ? ntile : 4 + kTcGrpWarps) * 32 + kTcProd);
  const dim3 block8(tc8_warps(T) * 32 + kTcProd);
  if (tc_f8(kv_type) && kv_type == KvType::NVFP4)
    launch_pdl(attention_tc8_kernel<Fp4Kv, true>, grid, block8, kT8Smem<Fp4Kv>, s, q, lanes.pos, T,
               static_cast<const Fp4Kv*>(kv.k), static_cast<const Fp4Kv*>(kv.v), lanes.page_table[0], splits, part,
               lanes);
  else if (tc_f8(kv_type) && kv_type == KvType::FP8)
    launch_pdl(attention_tc8_kernel<uint8_t, true>, grid, block8, kT8Smem<uint8_t>, s, q, lanes.pos, T,
               static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v), lanes.page_table[0], splits,
               part, lanes);
  else if (kv_type == KvType::BF16)
    launch_pdl(attention_tc_kernel<bf16, true>, grid, block, kTcSmemOf<bf16>, s, q, lanes.pos, T,
               static_cast<const bf16*>(kv.k), static_cast<const bf16*>(kv.v), lanes.page_table[0], splits, part,
               lanes);
  else if (kv_type == KvType::NVFP4)
    launch_pdl(attention_tc_kernel<Fp4Kv, true>, grid, block, kTcSmemOf<Fp4Kv>, s, q, lanes.pos, T,
               static_cast<const Fp4Kv*>(kv.k), static_cast<const Fp4Kv*>(kv.v), lanes.page_table[0], splits, part,
               lanes);
  else
    launch_pdl(attention_tc_kernel<uint8_t, true>, grid, block, kTcSmemOf<uint8_t>, s, q, lanes.pos, T,
               static_cast<const uint8_t*>(kv.k), static_cast<const uint8_t*>(kv.v), lanes.page_table[0], splits,
               part, lanes);
  Q27_CUDA(cudaGetLastError());
  launch_pdl(attention_combine_kernel<true>, dim3(unsigned(rows)), dim3(kHeadDim), 0, s, qg, ldq, lanes.pos, splits, 0,
             kTcChunk, part, out, q8, in_scale, static_cast<const SpecTree*>(nullptr), lanes);
  Q27_CUDA(cudaGetLastError());
}

void kv_commit(KvLayer* layers_dev, int nlayers, KvType kv_type, const int32_t* page_table, const int32_t* pos_dev,
               const int32_t* ncommit, const int32_t* cpath, cudaStream_t s) {
  const dim3 grid(nlayers, 2 * kKvHeads);
  if (kv_type == KvType::BF16)
    launch_pdl(kv_commit_kernel<bf16>, grid, dim3(kHeadDim), 0, s, layers_dev, page_table, pos_dev, ncommit, cpath);
  else if (kv_type == KvType::NVFP4)
    launch_pdl(kv_commit_kernel<Fp4Kv>, grid, dim3(kHeadDim), 0, s, layers_dev, page_table, pos_dev, ncommit, cpath);
  else
    launch_pdl(kv_commit_kernel<uint8_t>, grid, dim3(kHeadDim), 0, s, layers_dev, page_table, pos_dev, ncommit, cpath);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
