// Gated DeltaNet (linear attention) layers, Phase 1 versions: the delta rule runs token by token
// (prefill and decode share it). See reference/qwen27b/model.py `gdn_layer` for the definition.
#include <cooperative_groups.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "common.h"
#include "kernels.h"
#include "launch.cuh"
#include "mma.cuh"
#include "spec.h"

namespace q27 {
namespace {

using namespace dims;

constexpr int kQ = kKHeads * kKDim;  // 2048: q (and k) channels of the conv output

// New conv state = the last 3 raw inputs of [state | qkv]. One thread per channel (in place).
__global__ void conv_state_kernel(const float* qkv, int ld, float* state, int T) {
  pdl_trigger();
  pdl_wait();
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= kConvDim) return;
  float* st = state + c * (kConvK - 1);
#pragma unroll
  for (int i = 0; i < kConvK - 1; i++) {
    int s = T + i;  // reads only indices >= i, so increasing i is safe in place
    st[i] = s < kConvK - 1 ? st[s] : qkv[int64_t(s - (kConvK - 1)) * ld + c];
  }
}

// Block per (value head, 32 dv columns); thread (r, c) holds S[16r .. 16r+15][c] in registers.
// Per token: S *= exp(g); kv = S^T k; sq = S^T q; delta = (v - kv) * beta; o = sq + delta (k.q);
// S += k delta^T.
constexpr int kCols = 32, kGroups = 8, kRows = kKDim / kGroups;
__global__ void __launch_bounds__(256) gdn_recurrent_kernel(const float* q, const float* k, const float* v,
                                                            const float* g, const float* beta, int T, float* state,
                                                            float* o) {
  pdl_trigger();
  pdl_wait();
  int h = blockIdx.x, c0 = blockIdx.y * kCols;
  int c = threadIdx.x % kCols, r = threadIdx.x / kCols;
  int kh = h / (kVHeads / kKHeads);
  __shared__ float sk[kKDim], sq[kKDim], pk[kGroups][kCols], pq[kGroups][kCols], sdelta[kCols];
  float S[kRows];
  float* st = state + int64_t(h) * kKDim * kVDim;
#pragma unroll
  for (int i = 0; i < kRows; i++) S[i] = st[(r * kRows + i) * kVDim + c0 + c];
  for (int t = 0; t < T; t++) {
    __syncthreads();
    if (threadIdx.x < kKDim)
      sk[threadIdx.x] = k[(int64_t(t) * kKHeads + kh) * kKDim + threadIdx.x];
    else
      sq[threadIdx.x - kKDim] = q[(int64_t(t) * kKHeads + kh) * kKDim + threadIdx.x - kKDim];
    __syncthreads();
    float decay = expf(g[int64_t(t) * kVHeads + h]);
    float ak = 0.f, aq = 0.f;
#pragma unroll
    for (int i = 0; i < kRows; i++) {
      S[i] *= decay;
      ak += S[i] * sk[r * kRows + i];
      aq += S[i] * sq[r * kRows + i];
    }
    pk[r][c] = ak;
    pq[r][c] = aq;
    __syncthreads();
    if (r == 0) {
      float kv = 0.f, qv = 0.f, kq = 0.f;
#pragma unroll
      for (int i = 0; i < kGroups; i++) kv += pk[i][c], qv += pq[i][c];
#pragma unroll
      for (int i = 0; i < kKDim / 32; i++) kq += sk[c + 32 * i] * sq[c + 32 * i];
      for (int s = 16; s > 0; s >>= 1) kq += __shfl_xor_sync(0xffffffffu, kq, s);
      int64_t vi = (int64_t(t) * kVHeads + h) * kVDim + c0 + c;
      float d = (v[vi] - kv) * beta[int64_t(t) * kVHeads + h];
      sdelta[c] = d;
      o[vi] = qv + d * kq;
    }
    __syncthreads();
    float d = sdelta[c];
#pragma unroll
    for (int i = 0; i < kRows; i++) S[i] += sk[r * kRows + i] * d;
  }
#pragma unroll
  for (int i = 0; i < kRows; i++) st[(r * kRows + i) * kVDim + c0 + c] = S[i];
}

// Prefill prep in one pass: block (token, y) of 256 threads. y < 16: key head y, threads 0-127 its q
// channels and 128-255 its k channels: causal conv + SiLU, then the l2norm in registers (the same sums
// in the same order as the former separate conv and l2norm kernels: identical results); block y = 0 also
// computes g and beta. y >= 16: 256 v channels each.
constexpr int kPrepVBlocks = kVHeads * kVDim / 256;
__global__ void __launch_bounds__(256) conv_norm_kernel(const float* qkv, int ld, const float* state, const bf16* w,
                                                        float* q, float* k, float* v, const float* a, const float* b,
                                                        const bf16* A_log, const bf16* dt_bias, float* g,
                                                        float* beta) {
  pdl_trigger();
  pdl_wait();
  const int t = blockIdx.x, y = blockIdx.y, j = threadIdx.x;
  const int c = y < kKHeads ? (j < kKDim ? y * kKDim + j : kQ + y * kKDim + j - kKDim)
                            : 2 * kQ + (y - kKHeads) * 256 + j;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < kConvK; i++) {
    int s = t + i;  // index into [state (3) | qkv (T)]
    float x = s < kConvK - 1 ? state[c * (kConvK - 1) + s] : qkv[int64_t(s - (kConvK - 1)) * ld + c];
    acc += __bfloat162float(w[c * kConvK + i]) * x;
  }
  acc = acc / (1.f + expf(-acc));
  if (y >= kKHeads) {
    v[int64_t(t) * (kVHeads * kVDim) + c - 2 * kQ] = acc;
    return;
  }
  __shared__ float red[8];
  float sq = acc * acc;
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  if (j % 32 == 0) red[j / 32] = sq;
  __syncthreads();
  const float* r = red + (j < kKDim ? 0 : 4);
  sq = r[0] + r[1] + r[2] + r[3];
  if (j < kKDim)
    q[(int64_t(t) * kKHeads + y) * kKDim + j] = acc * rsqrtf(sq + 1e-6f) * rsqrtf(float(kKDim));
  else
    k[(int64_t(t) * kKHeads + y) * kKDim + j - kKDim] = acc * rsqrtf(sq + 1e-6f);
  if (y == 0 && j < kVHeads) {
    int64_t idx = int64_t(t) * kVHeads + j;
    float x = a[idx] + __bfloat162float(dt_bias[j]);
    float sp = x > 20.f ? x : log1pf(expf(x));
    g[idx] = -expf(__bfloat162float(A_log[j])) * sp;
    beta[idx] = 1.f / (1.f + expf(-b[idx]));
  }
}

// Warp per (token, value head), 4 dims per lane (8 heads per block). out (fp32 copy) may be null.
constexpr int kGnHeads = 8;
__global__ void __launch_bounds__(kGnHeads * 32) gated_norm_kernel(const float* o, const float* z, int ldz,
                                                                   const bf16* w, float eps, float* out, uint8_t* q8,
                                                                   float in_scale) {
  pdl_trigger();
  pdl_wait();
  const int t = blockIdx.x, h = blockIdx.y * kGnHeads + threadIdx.x / 32, d = threadIdx.x % 32 * 4;
  const int64_t base = (int64_t(t) * kVHeads + h) * kVDim + d;
  const float4 x = *reinterpret_cast<const float4*>(o + base);
  float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
  const float r = rsqrtf(ss / kVDim + eps);
  const float4 zz = *reinterpret_cast<const float4*>(z + int64_t(t) * ldz + h * kVDim + d);
  const float xs[4] = {x.x, x.y, x.z, x.w}, zs[4] = {zz.x, zz.y, zz.z, zz.w};
  float y[4];
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    y[i] = xs[i] * r * __bfloat162float(w[d + i]) * (zs[i] / (1.f + expf(-zs[i])));
    packed |= uint32_t(__nv_cvt_float_to_fp8(y[i] / in_scale, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
  }
  if (out) *reinterpret_cast<float4*>(out + base) = make_float4(y[0], y[1], y[2], y[3]);
  if (q8) *reinterpret_cast<uint32_t*>(q8 + base) = packed;
}


// ---------------------------------------------------------------- chunked prefill (P5-2)
//
// The gated delta rule in chunks of 64 tokens. With gamma the cumulative log decay inside a chunk
// (from its start), the chunk's deltas solve (I + L) delta = beta (v - e^gamma S0^T k) with
// L[t][s] = beta_t e^(gamma_t - gamma_s) k_t.k_s (s < t), so with A = (I + L)^-1:
//   delta = U - W S0,             U = A (beta V),  W = A (beta e^gamma K)
//   O = e^gamma Q S0 + M delta,   M[t][s] = e^(gamma_t - gamma_s) q_t.k_s (s <= t)
//   S = e^gamma_last S0 + K^T diag(e^(gamma_last - gamma)) delta.
// Chunks sit at fixed multiples of 64 positions (pos0: position of token 0 of the call); a call that
// starts or ends inside one runs its part with the rest as zero rows (k = q = 0, g = 0, beta = 0 leave
// the state exactly alone). A prefill cut at multiples of 64 therefore gives bit-identical results
// to an uncut one; a cut elsewhere changes the chunks around it.
//
// gdn_chunk_prep_kernel, block per (chunk, value head): gamma, the dots k_t.k_s and q_t.k_s (mma, bf16
// k / q), then thread j solves column j of (I + L) [W | U] = [beta e^gamma K | beta V] in fp32 by
// forward substitution in registers. U overwrites v; W, M, the decays and (block of the first value
// head of a key head) the chunk's q / k go to the workspace as bf16.
// gdn_chunk_kernel, block per (value head, kChWarps x 16 state columns): warp w keeps its 16 columns
// of S^T (m16 x dk 128) in mma accumulators across the chunks; per chunk, with bf16 operands,
//   delta^T = U^T - S^T W^T,  O^T = (S^T Q^T) diag(e^gamma) + delta^T M^T,
//   S^T = e^gamma_last S^T + (delta^T diag(e^(gamma_last - gamma))) K.
// One shared-memory stage (W, Q, K, M are 56 KB of the 99): each buffer is refilled for the next
// chunk as soon as this chunk is done with it.
constexpr int kCh = kGdnChunk;
constexpr int kChWarps = 4;
constexpr int kChPrepThreads = 256;
constexpr int kChTile = kCh * kKDim * 2;  // bf16 [64][128]
constexpr int kChPrepSmem = 2 * kChTile + (kCh * kVDim + 3 * kCh) * 4;  // k, q (then L), v
constexpr int kChSmem = 3 * kChTile + kCh * kCh * 2 + 2 * kCh * 4;

struct ChunkWs {  // per call of nc chunks
  bf16 *w, *m, *q, *k;  // [nc][48][64][128], [nc][48][64][64], [nc][16][64][128] x 2
  float* dec;           // [nc][48][2][64]: e^gamma, e^(gamma_last - gamma)
};
ChunkWs chunk_ws(void* ws, int nc) {
  ChunkWs c;
  c.w = static_cast<bf16*>(ws);
  c.m = c.w + size_t(nc) * kVHeads * kCh * kKDim;
  c.q = c.m + size_t(nc) * kVHeads * kCh * kCh;
  c.k = c.q + size_t(nc) * kKHeads * kCh * kKDim;
  c.dec = reinterpret_cast<float*>(c.k + size_t(nc) * kKHeads * kCh * kKDim);
  return c;
}
int chunk_count(int T, int pos0) { return (pos0 % kCh + T + kCh - 1) / kCh; }

// Byte offset of 16-byte unit `unit` of row `row` in a [rows][128 bf16] (256 B) or [rows][64 bf16]
// tile, units XOR-swizzled by row % 8 (conflict-free ldmatrix).
__device__ __forceinline__ uint32_t swz256(int row, int unit) { return row * 256 + ((unit ^ (row & 7)) << 4); }
__device__ __forceinline__ uint32_t swz128(int row, int unit) { return row * 128 + ((unit ^ (row & 7)) << 4); }

__global__ void __launch_bounds__(kChPrepThreads) gdn_chunk_prep_kernel(const float* __restrict__ q,
                                                                        const float* __restrict__ k,
                                                                        float* __restrict__ v, const float* g,
                                                                        const float* beta, int T, int off, ChunkWs ws) {
  pdl_trigger();
  pdl_wait();
  extern __shared__ __align__(128) uint8_t chp_smem[];
  const uint32_t skq = uint32_t(__cvta_generic_to_shared(chp_smem)), sqq = skq + kChTile;  // bf16 k, q
  const uint32_t sv = sqq + kChTile;                                // fp32 v [64][128]
  float* Ls = reinterpret_cast<float*>(chp_smem + kChTile);        // [64][64] over q, once Q K^T is done
  const float* vs = reinterpret_cast<const float*>(chp_smem + 2 * kChTile);
  float* gam = reinterpret_cast<float*>(chp_smem + 2 * kChTile + kCh * kVDim * 4);  // [64]
  float* bs = gam + kCh;                                                            // [64]
  float* eg = bs + kCh;                                                             // [64]
  const int c = blockIdx.x, h = blockIdx.y, kh = h / (kVHeads / kKHeads), tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32, r0 = lane / 4, c2 = 2 * (lane % 4), lm = lane / 8, lr = lane % 8;
  const int t0 = c * kCh - off;  // token of row 0
  const bool qk_out = h % (kVHeads / kKHeads) == 0;
  bf16* qb = ws.q + (size_t(c) * kKHeads + kh) * kCh * kKDim;
  bf16* kb = ws.k + (size_t(c) * kKHeads + kh) * kCh * kKDim;
  // v rows of the head (zero outside the call), for the solve.
  for (int i = tid; i < kCh * kVDim / 4; i += kChPrepThreads) {
    const int r = i / (kVDim / 4), t = t0 + r;
    const bool ok = t >= 0 && t < T;
    cp_async16z(sv + i * 16, v + (ok ? (int64_t(t) * kVHeads + h) * kVDim + i % (kVDim / 4) * 4 : 0), ok);
  }
  cp_commit();
  {  // k and q rows to bf16 (shared memory, and the workspace once per key head); loads first
    constexpr int kN = kCh * kKDim / 8 / kChPrepThreads;  // 16-byte bf16 units per thread
    float4 kv[2 * kN], qv[2 * kN];
#pragma unroll
    for (int n = 0; n < kN; n++) {
      const int i = tid + n * kChPrepThreads, r = i / 16, d = i % 16 * 8, t = t0 + r;
      const bool ok = t >= 0 && t < T;
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      const int64_t idx = (int64_t(t) * kKHeads + kh) * kKDim + d;
      kv[2 * n] = ok ? *reinterpret_cast<const float4*>(k + idx) : z;
      kv[2 * n + 1] = ok ? *reinterpret_cast<const float4*>(k + idx + 4) : z;
      qv[2 * n] = ok ? *reinterpret_cast<const float4*>(q + idx) : z;
      qv[2 * n + 1] = ok ? *reinterpret_cast<const float4*>(q + idx + 4) : z;
    }
#pragma unroll
    for (int n = 0; n < kN; n++) {
      const int i = tid + n * kChPrepThreads, r = i / 16, un = i % 16;
      const float4 a = kv[2 * n], b = kv[2 * n + 1], x = qv[2 * n], y = qv[2 * n + 1];
      const uint4 ku = make_uint4(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w), pack_bf16(b.x, b.y), pack_bf16(b.z, b.w));
      const uint4 qu = make_uint4(pack_bf16(x.x, x.y), pack_bf16(x.z, x.w), pack_bf16(y.x, y.y), pack_bf16(y.z, y.w));
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(skq + swz256(r, un)), "r"(ku.x), "r"(ku.y),
                   "r"(ku.z), "r"(ku.w));
      asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(sqq + swz256(r, un)), "r"(qu.x), "r"(qu.y),
                   "r"(qu.z), "r"(qu.w));
      if (qk_out) reinterpret_cast<uint4*>(kb)[i] = ku, reinterpret_cast<uint4*>(qb)[i] = qu;
    }
  }
  if (tid < kCh) {
    const int t = t0 + tid;
    const bool ok = t >= 0 && t < T;
    gam[tid] = ok ? g[int64_t(t) * kVHeads + h] : 0.f;
    bs[tid] = ok ? beta[int64_t(t) * kVHeads + h] : 0.f;
  }
  __syncthreads();
  if (tid < 32) {  // inclusive scan of g: lane l holds rows 2l, 2l + 1
    const float a = gam[2 * tid], b = gam[2 * tid + 1];
    float x = a + b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, x, o);
      if (tid >= o) x += y;
    }
    const float ex = x - (a + b);
    gam[2 * tid] = ex + a, gam[2 * tid + 1] = x;
  }
  __syncthreads();
  if (tid < kCh) {
    eg[tid] = expf(gam[tid]);
    float* dec = ws.dec + (size_t(c) * kVHeads + h) * 2 * kCh;
    dec[tid] = eg[tid];
    dec[kCh + tid] = expf(gam[kCh - 1] - gam[tid]);
  }
  // Q K^T (into M), then K K^T (into L, over q). Warp w: rows 16 (w % 4) .., 16-key tiles 2 (w / 4) + {0, 1}
  // (tiles past the row tile are skipped: zero, and never read).
  const int mt = warp % 4, p = warp / 4;
  {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int kt = 2 * p + half;  // 16-key tile
      float a2[2][4] = {};
      if (kt <= mt) {
#pragma unroll
        for (int kk = 0; kk < kKDim / 16; kk++) {
          uint32_t a[4], b[4];
          ldsm_x4(a, sqq + swz256(16 * mt + (lm % 2) * 8 + lr, 2 * kk + lm / 2));
          ldsm_x4(b, skq + swz256(16 * kt + lr + 8 * (lm / 2), 2 * kk + lm % 2));
          mma_bf16_16816(a2[0], a, b);
          mma_bf16_16816(a2[1], a, b + 2);
        }
        bf16* m = ws.m + (size_t(c) * kVHeads + h) * kCh * kCh;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int t = 16 * mt + r0 + 8 * (e / 2), s = 16 * kt + nt * 8 + c2 + (e & 1);
            m[t * kCh + s] = __float2bfloat16(s <= t ? expf(gam[t] - gam[s]) * a2[nt][e] : 0.f);
          }
      }
    }
    __syncthreads();  // q is no longer read: L goes over it
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int kt = 2 * p + half;
      if (kt > mt) continue;
      float a2[2][4] = {};
#pragma unroll
      for (int kk = 0; kk < kKDim / 16; kk++) {
        uint32_t a[4], b[4];
        ldsm_x4(a, skq + swz256(16 * mt + (lm % 2) * 8 + lr, 2 * kk + lm / 2));
        ldsm_x4(b, skq + swz256(16 * kt + lr + 8 * (lm / 2), 2 * kk + lm % 2));
        mma_bf16_16816(a2[0], a, b);
        mma_bf16_16816(a2[1], a, b + 2);
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int t = 16 * mt + r0 + 8 * (e / 2), s = 16 * kt + nt * 8 + c2 + (e & 1);
          Ls[t * kCh + s] = s < t ? bs[t] * expf(gam[t] - gam[s]) * a2[nt][e] : 0.f;
        }
    }
  }
  cp_wait<0>();
  __syncthreads();
  // Column j of [W | U]: (I + L) x = r. Right-looking: once x_m is final, x_i -= L[i][m] x_m for all
  // i > m (independent updates; the dependency chain is 64 steps, not the 2,016 of the row sums).
  const int j = tid;
  float x[kCh];
#pragma unroll
  for (int i = 0; i < kCh; i++) {
    if (j < kKDim) {
      uint16_t raw;
      asm volatile("ld.shared.u16 %0, [%1];" : "=h"(raw) : "r"(skq + swz256(i, j / 8) + (j % 8) * 2));
      x[i] = bs[i] * eg[i] * __bfloat162float(__ushort_as_bfloat16(raw));
    } else {
      x[i] = bs[i] * vs[i * kVDim + j - kKDim];
    }
  }
#pragma unroll
  for (int mm = 0; mm < kCh - 1; mm++)
#pragma unroll
    for (int i = mm + 1; i < kCh; i++) x[i] -= Ls[i * kCh + mm] * x[mm];
  if (j < kKDim) {
    bf16* w = ws.w + (size_t(c) * kVHeads + h) * kCh * kKDim + j;
#pragma unroll
    for (int i = 0; i < kCh; i++) w[i * kKDim] = __float2bfloat16(x[i]);
  } else {
#pragma unroll
    for (int i = 0; i < kCh; i++) {
      const int t = t0 + i;
      if (t >= 0 && t < T) v[(int64_t(t) * kVHeads + h) * kVDim + j - kKDim] = x[i];
    }
  }
}

__global__ void __launch_bounds__(kChWarps * 32, 1) gdn_chunk_kernel(const float* u, int T, int off, int nc,
                                                                     ChunkWs ws, float* state, float* o) {
  pdl_trigger();
  pdl_wait();
  extern __shared__ __align__(128) uint8_t chk_smem[];
  const uint32_t sw = uint32_t(__cvta_generic_to_shared(chk_smem)), sq = sw + kChTile, sk = sq + kChTile,
                 sm = sk + kChTile, sd = sm + kCh * kCh * 2;  // W, Q, K, M, decays
  const float* sdec = reinterpret_cast<const float*>(chk_smem + 3 * kChTile + kCh * kCh * 2);
  const int h = blockIdx.x, kh = h / (kVHeads / kKHeads), tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32, r0 = lane / 4, c2 = 2 * (lane % 4), lm = lane / 8, lr = lane % 8;
  const int n0 = (blockIdx.y * kChWarps + warp) * 16;  // this warp's state columns (dv)
  constexpr int kThreads = kChWarps * 32;

  // Copies of chunk c: W + Q + e^gamma, M, K + e^(gamma_last - gamma) (one cp.async group each).
  auto issue_wq = [&](int c) {
    if (c < nc) {
      const char* w = reinterpret_cast<const char*>(ws.w + (size_t(c) * kVHeads + h) * kCh * kKDim);
      const char* q = reinterpret_cast<const char*>(ws.q + (size_t(c) * kKHeads + kh) * kCh * kKDim);
      for (int i = tid; i < kChTile / 16; i += kThreads) {
        const int r = i / 16, un = i % 16;
        cp_async16z(sw + swz256(r, un), w + i * 16, true);
        cp_async16z(sq + swz256(r, un), q + i * 16, true);
      }
      if (tid < kCh / 4)
        cp_async16z(sd + tid * 16, ws.dec + (size_t(c) * kVHeads + h) * 2 * kCh + tid * 4, true);
    }
    cp_commit();
  };
  auto issue_m = [&](int c) {
    if (c < nc) {
      const char* m = reinterpret_cast<const char*>(ws.m + (size_t(c) * kVHeads + h) * kCh * kCh);
      for (int i = tid; i < kCh * kCh * 2 / 16; i += kThreads) cp_async16z(sm + swz128(i / 8, i % 8), m + i * 16, true);
    }
    cp_commit();
  };
  auto issue_k = [&](int c) {
    if (c < nc) {
      const char* k = reinterpret_cast<const char*>(ws.k + (size_t(c) * kKHeads + kh) * kCh * kKDim);
      for (int i = tid; i < kChTile / 16; i += kThreads) cp_async16z(sk + swz256(i / 16, i % 16), k + i * 16, true);
      if (tid < kCh / 4)
        cp_async16z(sd + kCh * 4 + tid * 16, ws.dec + (size_t(c) * kVHeads + h) * 2 * kCh + kCh + tid * 4, true);
    }
    cp_commit();
  };
  issue_wq(0);
  issue_m(0);
  issue_k(0);

  // S^T fragments: rows = state columns n0 + r0 (+8), cols = dk.
  float s[kKDim / 8][4];
  float* st = state + int64_t(h) * kKDim * kVDim;
#pragma unroll
  for (int nt = 0; nt < kKDim / 8; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) s[nt][e] = st[(nt * 8 + c2 + (e & 1)) * kVDim + n0 + r0 + 8 * (e / 2)];

  for (int c = 0; c < nc; c++) {
    const int t0 = c * kCh - off;
    // U^T of this chunk (rows of the call only), loaded early.
    float uu[kCh / 8][4];
#pragma unroll
    for (int nt = 0; nt < kCh / 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int t = t0 + nt * 8 + c2 + (e & 1);
        uu[nt][e] = t >= 0 && t < T ? u[(int64_t(t) * kVHeads + h) * kVDim + n0 + r0 + 8 * (e / 2)] : 0.f;
      }
    uint32_t sa[kKDim / 16][4];
#pragma unroll
    for (int kk = 0; kk < kKDim / 16; kk++) {
      sa[kk][0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]), sa[kk][1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
      sa[kk][2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      sa[kk][3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
    }
    cp_wait<2>();
    __syncthreads();
    // S^T W^T and S^T Q^T (n = the chunk's 64 tokens).
    float d[kCh / 8][4] = {}, ot[kCh / 8][4] = {};
#pragma unroll
    for (int kk = 0; kk < kKDim / 16; kk++)
#pragma unroll
      for (int p = 0; p < kCh / 16; p++) {
        uint32_t b[4];
        ldsm_x4(b, sw + swz256(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
        mma_bf16_16816(d[2 * p], sa[kk], b);
        mma_bf16_16816(d[2 * p + 1], sa[kk], b + 2);
        ldsm_x4(b, sq + swz256(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
        mma_bf16_16816(ot[2 * p], sa[kk], b);
        mma_bf16_16816(ot[2 * p + 1], sa[kk], b + 2);
      }
    const float egl = sdec[kCh - 1];
#pragma unroll
    for (int nt = 0; nt < kCh / 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        d[nt][e] = uu[nt][e] - d[nt][e];  // delta^T
        ot[nt][e] *= sdec[nt * 8 + c2 + (e & 1)];
      }
    __syncthreads();  // W, Q and the e^gamma are free
    issue_wq(c + 1);
    // O^T += delta^T M^T (M lower triangular: key tiles up to the token tile).
    uint32_t da[kCh / 16][4];
#pragma unroll
    for (int kt = 0; kt < kCh / 16; kt++) {
      da[kt][0] = pack_bf16(d[2 * kt][0], d[2 * kt][1]), da[kt][1] = pack_bf16(d[2 * kt][2], d[2 * kt][3]);
      da[kt][2] = pack_bf16(d[2 * kt + 1][0], d[2 * kt + 1][1]);
      da[kt][3] = pack_bf16(d[2 * kt + 1][2], d[2 * kt + 1][3]);
    }
    cp_wait<2>();
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kCh / 16; p++)
#pragma unroll
      for (int kt = 0; kt <= p; kt++) {
        uint32_t b[4];
        ldsm_x4(b, sm + swz128(16 * p + lr + 8 * (lm / 2), 2 * kt + lm % 2));
        mma_bf16_16816(ot[2 * p], da[kt], b);
        mma_bf16_16816(ot[2 * p + 1], da[kt], b + 2);
      }
    __syncthreads();  // M is free
    issue_m(c + 1);
#pragma unroll
    for (int nt = 0; nt < kCh / 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int t = t0 + nt * 8 + c2 + (e & 1);
        if (t >= 0 && t < T) o[(int64_t(t) * kVHeads + h) * kVDim + n0 + r0 + 8 * (e / 2)] = ot[nt][e];
      }
    // S^T = e^gamma_last S^T + (delta^T diag(e^(gamma_last - gamma))) K.
    cp_wait<2>();
    __syncthreads();
    const float* ed = sdec + kCh;
#pragma unroll
    for (int kt = 0; kt < kCh / 16; kt++) {
      const float e0 = ed[16 * kt + c2], e1 = ed[16 * kt + c2 + 1], e2 = ed[16 * kt + 8 + c2],
                  e3 = ed[16 * kt + 8 + c2 + 1];
      da[kt][0] = pack_bf16(d[2 * kt][0] * e0, d[2 * kt][1] * e1);
      da[kt][1] = pack_bf16(d[2 * kt][2] * e0, d[2 * kt][3] * e1);
      da[kt][2] = pack_bf16(d[2 * kt + 1][0] * e2, d[2 * kt + 1][1] * e3);
      da[kt][3] = pack_bf16(d[2 * kt + 1][2] * e2, d[2 * kt + 1][3] * e3);
    }
#pragma unroll
    for (int nt = 0; nt < kKDim / 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) s[nt][e] *= egl;
#pragma unroll
    for (int kt = 0; kt < kCh / 16; kt++)
#pragma unroll
      for (int dp = 0; dp < kKDim / 16; dp++) {
        uint32_t b[4];
        ldsm_x4_trans(b, sk + swz256(16 * kt + lr + 8 * (lm % 2), 2 * dp + lm / 2));
        mma_bf16_16816(s[2 * dp], da[kt], b);
        mma_bf16_16816(s[2 * dp + 1], da[kt], b + 2);
      }
    __syncthreads();  // K and the e^(gamma_last - gamma) are free
    issue_k(c + 1);
  }
#pragma unroll
  for (int nt = 0; nt < kKDim / 8; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) st[(nt * 8 + c2 + (e & 1)) * kVDim + n0 + r0 + 8 * (e / 2)] = s[nt][e];
}


// ---------------------------------------------------------------- decode (T = 1) and verify (T <= 8)
//
// Decode runs one token and advances the states; the speculative verify runs the T tokens of a draft
// block without advancing them past the block (which tokens survive is known only after the accept
// step). Instead, a verify keeps its per-token inputs (raw conv input, normalized k, v, a / b) in one
// of two buffers (the step parity), and the next verify first replays the tokens the previous step
// committed (*ncommit, from the accept kernel), writes the states, then runs its own block. The
// replay reads only what the previous step wrote, so it runs before the PDL wait.
// Both modes share every per-token expression (gdn_prep_token, gdn_step), so a verified token's result
// is bit-identical to decoding it alone, and the replayed state to the decoded one.
//
// gdn_prep_kernel<mode>, prep_blocks(T) blocks of 256 threads; block x:
//   [0, 16)   key head kh: conv + SiLU of its q (threads 0-127) and k (128-255) channels, l2norm
//   [16, 40)  256 v channels each: conv + SiLU
//   [40, ...) in_proj_a / in_proj_b on the normed hidden state, 4 outputs x 2 tokens per block (a latency-bound
//             GEMV: a warp per output needed 20 dependent loads and took 12 us; one block per output
//             re-read the T rows of h 96 times)
// gdn_rec_kernel<mode>, one block of 1024 threads per value head: warp w owns the state columns
// 4w..4w+3, lane l the rows l, l + 32, l + 64, l + 96 (16 values in registers). A token costs no
// __syncthreads: the row sums are a reduce-scatter over the lanes. The gated RMSNorm of the outputs
// runs after the token loop for all tokens at once.

constexpr int kPrepThreads = 256;
constexpr int kPrepQK = kKHeads;
constexpr int kPrepV = kVHeads * kVDim / kPrepThreads;
constexpr int kPrepABPer = 4;                            // a / b outputs per block
constexpr int kPrepAB = 2 * kVHeads / kPrepABPer;        // 24
// Tokens per a / b block: the a / b blocks of a verify are split over the tokens (a token's dots are
// those of T = 1 whichever block computes them; 205 registers allow one block per SM, and 24 blocks
// doing all 8 tokens took 5 of the 5.3 us at T = 8; in the step, 4 tokens per block beat 2). Decode keeps 8: with
// 2, the code the compiler emits made T = 1 0.4 us slower.
constexpr int kPrepGroup = 8;  // tokens per batch of per-token reductions (see below)
template <bool kVerify>
constexpr int kPrepABTok = kVerify ? 4 : kPrepGroup;
template <bool kVerify>
int prep_blocks(int T) {
  return kPrepQK + kPrepV + kPrepAB * std::max(1, (T + kPrepABTok<kVerify> - 1) / kPrepABTok<kVerify>);
}

// Conv window of channel c before token 0 of the call: the decode state, or the verify's committed
// state after replaying the previous block's committed raw inputs (written back by the t == 0 block).
struct ConvIn {
  float w[kConvK], x[kConvK - 1];
};
__device__ __forceinline__ ConvIn conv_load(const float* state, const bf16* w, int c) {
  ConvIn in;
#pragma unroll
  for (int i = 0; i < kConvK; i++) in.w[i] = __bfloat162float(w[c * kConvK + i]);
#pragma unroll
  for (int i = 0; i < kConvK - 1; i++) in.x[i] = state[c * (kConvK - 1) + i];
  return in;
}
__device__ __forceinline__ void conv_replay(ConvIn& in, int c, const float* raw_prev, int ncommit, const int32_t* cpath) {
  for (int j = 0; j < ncommit; j++)
    in.x[0] = in.x[1], in.x[1] = in.x[2], in.x[2] = raw_prev[int64_t(cpath[j]) * kConvDim + c];
}
// Where the prep kernel writes (decode: scratch q / k / v / a / b; verify: q and the replay buffers).
struct PrepOut {
  float *q, *k, *v, *ab;  // rows: q / k [T][2048], v [T][6144], ab [T][96] (a then b)
};

// One block per channel group (or 4 a / b outputs) handles all T tokens, in groups of 8 whose
// per-token reductions are batched behind one __syncthreads. Every token's expressions are those of
// T = 1 (decode). A verify reads the rows' tree paths (conv windows) from shared memory.

// Conv + SiLU of a row at depth d whose path rows are prow[0..d] (s_path): the inputs along the path
// from qkv rows ld apart, the window from before the call below depth 0.
__device__ __forceinline__ float conv_path(const ConvIn& in, const float* qkv, int ld, int c, int d,
                                           const int8_t* prow) {
  float x[kConvK];
#pragma unroll
  for (int i = 0; i < kConvK; i++) {
    const int s = d - (kConvK - 1) + i;  // < 0: the window from before the call
    x[i] = s < 0 ? in.x[s + kConvK - 1] : qkv[int64_t(prow[s]) * ld + c];
  }
  float acc = 0.f;
  acc += in.w[0] * x[0];
  acc += in.w[1] * x[1];
  acc += in.w[2] * x[2];
  acc += in.w[3] * x[3];
  return acc / (1.f + expf(-acc));
}

template <bool kVerify, bool kTree, bool kBatch = false>
__global__ void __launch_bounds__(kPrepThreads) gdn_prep_kernel(const float* qkv, int ld, int T, float* conv_state,
                                                                const bf16* conv_w, const float* h, const bf16* wa,
                                                                const bf16* wb, PrepOut po, GdnSpec sp) {
  // Before the PDL wait: everything that does not come from the previous kernels (weights, the conv
  // state of this layer, the previous verify's inputs); after it: qkv and the normed hidden state h.
  pdl_trigger();
  // Batched verify (P6-10): lane ln = blockIdx.y, its rows from r0 in every row buffer, its scalars at
  // offset ln and its conv state (0 / the call's otherwise). (Locals, not edits of sp: those spill it.)
  int ln = 0;
  if constexpr (kBatch) {
    ln = blockIdx.y;
    conv_state = lane_sel(sp.lconv, ln);
    qkv += int64_t(kLaneRows) * ln * ld, h += int64_t(kLaneRows) * ln * kHidden, po.q += int64_t(kLaneRows) * ln * kQ;
  }
  const int64_t r0 = int64_t(kLaneRows) * ln;
  const int blk = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  __shared__ float red[kPrepGroup][kPrepThreads / 32];
  __shared__ int8_t s_depth[kVerifyMaxT], s_path[kVerifyMaxT][kSpecBlock];
  int cur = 0, prev = 0, nc = 0;
  if constexpr (kVerify) {
    cur = sp.par[ln], prev = cur ^ 1, nc = sp.ncommit[ln];
    // (selects instead of indexing: a dynamic index into the parameter struct goes through the stack)
    po.k = (cur ? sp.k[1] : sp.k[0]) + r0 * kQ, po.v = (cur ? sp.v[1] : sp.v[0]) + r0 * kVHeads * kVDim;
    po.ab = (cur ? sp.ab[1] : sp.ab[0]) + r0 * 2 * kVHeads;
  }
  if (blk < kPrepQK + kPrepV) {
    const bool is_qk = blk < kPrepQK;
    const int half = tid / kKDim, j = tid % kKDim;  // qk blocks: half 0 = q, 1 = k
    const int c = is_qk ? half * kQ + blk * kKDim + j : 2 * kQ + (blk - kPrepQK) * kPrepThreads + tid;
    ConvIn in = conv_load(conv_state, conv_w, c);
    // (the committed window is written back by gdn_rec_kernel once every block has read the old one)
    if constexpr (kVerify) conv_replay(in, c, (prev ? sp.raw[1] : sp.raw[0]) + r0 * kConvDim, nc, sp.cpath + r0);
    if (T == 0) return;  // flush: replay only
    pdl_wait();
    // Paths of the rows: the tree (written by this step's selector, after the wait), or the chain 0..t
    // (row t at depth t: computed in place, no staging).
    if constexpr (kTree) {
      if (tid < T) {
        const int d = sp.tree[ln].depth[tid];
        s_depth[tid] = int8_t(d);
        for (int i = 0; i <= d; i++) s_path[tid][i] = sp.tree[ln].path[tid][i];
      }
      __syncthreads();
    }
    for (int t0 = 0; t0 < T; t0 += kPrepGroup) {
      const int tn = min(kPrepGroup, T - t0);
      float x[kPrepGroup];
#pragma unroll
      for (int u = 0; u < kPrepGroup; u++) {
        if (u >= tn) break;
        if constexpr (kTree) {
          x[u] = conv_path(in, qkv, ld, c, s_depth[t0 + u], s_path[t0 + u]);
        } else {  // chain: row t's inputs are rows t - 3 .. t (same expression as conv_path)
          const int t = t0 + u;
          float xin[kConvK];
#pragma unroll
          for (int i = 0; i < kConvK; i++) {
            const int s = t - (kConvK - 1) + i;
            xin[i] = s < 0 ? in.x[s + kConvK - 1] : qkv[int64_t(s) * ld + c];
          }
          float acc = 0.f;
          acc += in.w[0] * xin[0];
          acc += in.w[1] * xin[1];
          acc += in.w[2] * xin[2];
          acc += in.w[3] * xin[3];
          x[u] = acc / (1.f + expf(-acc));
        }
      }
      if constexpr (kVerify) {
        float* raw = (cur ? sp.raw[1] : sp.raw[0]) + r0 * kConvDim;
        for (int u = 0; u < tn; u++) raw[int64_t(t0 + u) * kConvDim + c] = qkv[int64_t(t0 + u) * ld + c];
      }
      if (!is_qk) {
        for (int u = 0; u < tn; u++) po.v[int64_t(t0 + u) * (kVHeads * kVDim) + c - 2 * kQ] = x[u];
        continue;
      }
#pragma unroll
      for (int u = 0; u < kPrepGroup; u++) {
        if (u >= tn) break;
        float ss = x[u] * x[u];
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) red[u][warp] = ss;
      }
      __syncthreads();
#pragma unroll
      for (int u = 0; u < kPrepGroup; u++) {
        if (u >= tn) break;
        const float ss = red[u][4 * half] + red[u][4 * half + 1] + red[u][4 * half + 2] + red[u][4 * half + 3];
        const int t = t0 + u;
        if (half == 0)
          po.q[int64_t(t) * kQ + blk * kKDim + j] = x[u] * rsqrtf(ss + 1e-6f) * rsqrtf(float(kKDim));
        else
          po.k[int64_t(t) * kQ + blk * kKDim + j] = x[u] * rsqrtf(ss + 1e-6f);
      }
      __syncthreads();  // red is refilled by the next group
    }
    if constexpr (!kVerify) {
      float* st = conv_state + c * (kConvK - 1);  // decode: slide the state by the token
      st[0] = in.x[1], st[1] = in.x[2], st[2] = qkv[c];
    }
  } else {
    __shared__ float red_ab[kPrepABPer][kPrepGroup][kPrepThreads / 32];
    const int ab = blk - kPrepQK - kPrepV;
    const int n0 = (ab % kPrepAB) * kPrepABPer;  // outputs n0 .. n0 + 3 of 0 .. 95: a then b
    const int tbeg = (ab / kPrepAB) * kPrepABTok<kVerify>, tend = min(T, tbeg + kPrepABTok<kVerify>);  // this block's tokens
    constexpr int kIters = (kHidden + kPrepThreads * 8 - 1) / (kPrepThreads * 8);  // 3
    uint4 wraw[kPrepABPer][kIters];
#pragma unroll
    for (int u = 0; u < kPrepABPer; u++) {
      const int n = n0 + u;
      const bf16* wr = (n < kVHeads ? wa : wb) + int64_t(n % kVHeads) * kHidden;
#pragma unroll
      for (int it = 0; it < kIters; it++) {
        const int i = tid * 8 + it * kPrepThreads * 8;
        if (i < kHidden) wraw[u][it] = __ldg(reinterpret_cast<const uint4*>(wr + i));
      }
    }
    if (tbeg >= tend) return;
    pdl_wait();
    for (int t0 = tbeg; t0 < tend; t0 += kPrepGroup) {
      const int tn = min(kPrepGroup, tend - t0);
#pragma unroll
      for (int g = 0; g < kPrepGroup; g++) {  // unrolled: the dots are independent (ILP)
        if (g >= tn) break;
        const float* ht = h + int64_t(t0 + g) * kHidden;
        float4 hx[kIters][2];
#pragma unroll
        for (int it = 0; it < kIters; it++) {
          const int i = tid * 8 + it * kPrepThreads * 8;
          if (i < kHidden)
            hx[it][0] = *reinterpret_cast<const float4*>(ht + i), hx[it][1] = *reinterpret_cast<const float4*>(ht + i + 4);
        }
#pragma unroll
        for (int u = 0; u < kPrepABPer; u++) {
          float acc = 0.f;  // (the expression of the one-output version: bit-identical per output)
#pragma unroll
          for (int it = 0; it < kIters; it++) {
            const int i = tid * 8 + it * kPrepThreads * 8;
            if (i >= kHidden) break;
            const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&wraw[u][it]);
            const float4 x0 = hx[it][0], x1 = hx[it][1];
            float2 w0 = __bfloat1622float2(hw[0]), w1 = __bfloat1622float2(hw[1]), w2 = __bfloat1622float2(hw[2]),
                   w3 = __bfloat1622float2(hw[3]);
            acc += x0.x * w0.x + x0.y * w0.y + x0.z * w1.x + x0.w * w1.y + x1.x * w2.x + x1.y * w2.y + x1.z * w3.x +
                   x1.w * w3.y;
          }
          for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
          if (lane == 0) red_ab[u][g][warp] = acc;
        }
      }
      __syncthreads();
      if (tid < kPrepABPer * tn) {
        const int u = tid / tn, g = tid % tn;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < kPrepThreads / 32; w++) sum += red_ab[u][g][w];
        po.ab[int64_t(t0 + g) * 2 * kVHeads + n0 + u] = sum;
      }
      __syncthreads();
    }
  }
}

// Recurrent part: a cluster of kRecSplit CTAs per value head, each owning 32 of the 128 state
// columns. In a warp, lane group c = lane / 8 owns column c of the warp's 4, and lane j = lane % 8 of
// the group the rows j, j + 8, ..., j + 120 (16 values in registers). A token costs no
// __syncthreads: the row sums are 3 shuffle levels within the 8-lane group. The gated RMSNorm of the
// outputs runs after the token loop; its sum over the head's columns goes through distributed shared
// memory.
#ifdef Q27_GDN_PROBE
__device__ long long g_gdn_probe[16];  // %globaltimer of CTA (0, y), y < 2, thread 0 at points of gdn_rec_kernel
#define GDN_PROBE(i)                                                    \
  if (blockIdx.x == 0 && blockIdx.y < 2 && threadIdx.x == 0) {          \
    long long t_;                                                       \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));              \
    g_gdn_probe[blockIdx.y * 8 + i] = t_;                               \
  }
#else
#define GDN_PROBE(i)
#endif
constexpr int kRecSplit = 4;                       // CTAs (cluster) per head
constexpr int kRecWarps = 8;                       // 256 threads
constexpr int kRecCols = 4;                        // columns per warp (lane groups of 8)
constexpr int kRecRows = kKDim / 8;                // 16 rows per lane
constexpr int kRecCtaCols = kVDim / kRecSplit;     // 32

// Decay and beta of value head h from the raw a / b.
__device__ __forceinline__ void decay_beta(float a, float b, float A, float dtb, float& decay, float& beta) {
  float x = a + dtb;
  float sp = x > 20.f ? x : log1pf(expf(x));
  decay = expf(-expf(A) * sp);
  beta = 1.f / (1.f + expf(-b));
}

// One token of the gated delta rule on the lane's column: S *= decay; kv = S^T k, sq = S^T q;
// d = (v - kv) beta; o = sq + d (k . q); S += k d^T. k / q: this token's [128] in shared memory, v
// the value of the lane's column. Returns o on the lanes j >= 4 of the group.
__device__ __forceinline__ float gdn_step(float (&S)[kRecRows], const float* k, const float* q, float v, float decay,
                                          float beta) {
  const int j = threadIdx.x % 8;
  float kr[kRecRows];
  float akv = 0.f, aq = 0.f, kq = 0.f;
#pragma unroll
  for (int i = 0; i < kRecRows; i++) {
    kr[i] = k[j + 8 * i];
    const float qi = q[j + 8 * i];
    kq += kr[i] * qi;
    S[i] *= decay;
    akv += S[i] * kr[i];
    aq += S[i] * qi;
  }
  // Sums over the 8 lanes of the group: kv / sq reduce-scattered (lanes j < 4 end with kv, j >= 4
  // with sq), k . q all-reduced.
  const bool hi = j & 4;
  float s = (hi ? aq : akv) + __shfl_xor_sync(0xffffffffu, hi ? akv : aq, 4);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  kq += __shfl_xor_sync(0xffffffffu, kq, 4);
  kq += __shfl_xor_sync(0xffffffffu, kq, 2);
  kq += __shfl_xor_sync(0xffffffffu, kq, 1);
  const float dl = (v - s) * beta;  // d on the kv lanes
  const float dp = __shfl_xor_sync(0xffffffffu, dl, 4);
  const float d = hi ? dp : dl;
#pragma unroll
  for (int i = 0; i < kRecRows; i++) S[i] += kr[i] * d;
  return s + d * kq;
}

struct RecIn {  // a token's inputs: k / q rows [2048] (16 key heads), v rows [6144], ab rows [96]
  const float *k, *q, *v, *ab;
};

// Decode (kVerify false): T = 1, the state advanced in place. Verify of a chain (kTree false): the
// CTAs run the rows 0..T-1 in order. Verify of a tree (kTree true): the CTAs stage every node and run
// the tree's root-to-leaf paths one after the other; a path continues from the state after its branch
// node (saved in registers when an earlier path computed that node), so every node is computed once,
// with exactly the steps of decoding its path alone. (One CTA per path, launched as a second grid row,
// was placed after the next GEMM and cost +14 us per layer for 2 paths.) Before the PDL wait (inputs
// of the previous step only), every CTA replays the previous verify's committed rows onto the
// committed state in registers and writes it to the other state buffer (other CTAs may still be
// reading `state`).
template <bool kVerify, bool kTree, bool kBatch = false>
__global__ void __launch_bounds__(kRecWarps * 32) gdn_rec_kernel(RecIn in, int T, const bf16* A_log,
                                                                 const bf16* dt_bias, float* state, const float* z,
                                                                 int ldz, const bf16* norm_w, float eps, float* out,
                                                                 uint8_t* q8, float in_scale, GdnSpec sp) {
  pdl_trigger();
  GDN_PROBE(0);
  // Batched verify (P6-10): lane ln = blockIdx.y (clusters run along x), its rows from r0, its scalars
  // at offset ln, its states lconv / lst (replayed in place); 0 / the call's otherwise.
  int ln = 0;
  if constexpr (kBatch) {
    ln = blockIdx.y;
    const int64_t o = int64_t(kLaneRows) * ln;
    in.q += o * kQ, z += o * ldz, out += o * kVHeads * kVDim;
    if (q8) q8 += o * kVHeads * kVDim;
  }
  const int64_t r0 = int64_t(kLaneRows) * ln;
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  constexpr int kRows = kTree ? kTreeMax : kSpecBlock;  // rows staged at once
  const int h = blockIdx.x / kRecSplit, part = blockIdx.x % kRecSplit, kh = h / (kVHeads / kKHeads);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, j = lane % 8;
  const int lcol = kRecCols * warp + lane / 8;  // the lane's column in the CTA
  const int col = part * kRecCtaCols + lcol;    // ... and in the head
  __shared__ float sk[kRows][kKDim], sq[kRows][kKDim], sv[kRows][kRecCtaCols];
  __shared__ float so[kRows][kRecCtaCols], s_part[kRows];
  __shared__ float s_decay[kRows], s_beta[kRows];
  __shared__ int s_rows[kRows];
  // The state (16 KiB per CTA) and the weights do not depend on the previous kernels: the state of
  // this layer is written only by this kernel, and every step ends with a host synchronization.
  int sb = 0;
  if constexpr (kVerify) {
    if constexpr (kBatch)
      state = lane_sel(sp.lst, ln);
    else
      sb = *sp.sbuf, state = sb ? sp.st[1] : sp.st[0];
  }
  const float* st = state + int64_t(h) * kKDim * kVDim;
  float S[kRecRows];
#pragma unroll
  for (int i = 0; i < kRecRows; i++) S[i] = st[(j + 8 * i) * kVDim + col];
  const float A = __bfloat162float(A_log[h]), dtb = __bfloat162float(dt_bias[h]);
  // Stages the inputs of n tokens (rows rows[0..n)) into shared memory (k, q of head kh, this CTA's v
  // columns, decay / beta).
  auto stage = [&](RecIn src, int n, const int* rows) {
    constexpr int kPer = kSpecBlock * kKDim / (kRecWarps * 32);  // 4 per 8 rows: unrolled, the loads overlap
    for (int r0 = 0; r0 < n; r0 += kSpecBlock) {
#pragma unroll
      for (int u = 0; u < kPer; u++) {
        const int i = tid + u * kRecWarps * 32;
        const int t = r0 + i / kKDim, jj = i % kKDim;
        if (t >= n) break;
        const int r = rows[t];
        sk[t][jj] = src.k[int64_t(r) * kQ + kh * kKDim + jj];
        if (src.q) sq[t][jj] = src.q[int64_t(r) * kQ + kh * kKDim + jj];
        if (jj < kRecCtaCols) sv[t][jj] = src.v[int64_t(r) * (kVHeads * kVDim) + h * kVDim + part * kRecCtaCols + jj];
      }
    }
    for (int t = tid; t < n; t += kRecWarps * 32)
      decay_beta(src.ab[rows[t] * 2 * kVHeads + h], src.ab[rows[t] * 2 * kVHeads + kVHeads + h], A, dtb, s_decay[t],
                 s_beta[t]);
    __syncthreads();
  };
  int n = T;  // rows computed (the chain's rows / the tree's nodes), row t = s_rows[t]
  if constexpr (kVerify) {
    // Replay of the previous verify's committed rows (inputs final: before the PDL wait).
    const int cur = sp.par[ln], prev = cur ^ 1, nc = sp.ncommit[ln];
    if (nc > 0) {
      if (tid < nc) s_rows[tid] = sp.cpath[r0 + tid];
      __syncthreads();
      const float* kp = (prev ? sp.k[1] : sp.k[0]) + r0 * kQ;
      stage({kp, nullptr, (prev ? sp.v[1] : sp.v[0]) + r0 * kVHeads * kVDim, (prev ? sp.ab[1] : sp.ab[0]) + r0 * 2 * kVHeads},
            nc, s_rows);
      for (int t = 0; t < nc; t++) gdn_step(S, sk[t], sk[t], sv[t][lcol], s_decay[t], s_beta[t]);
      __syncthreads();  // the staging buffers are refilled below
    }
    {  // the committed state moves to the other buffer (also when nothing was replayed)
      float* so_ = (kBatch ? state : sb ? sp.st[0] : sp.st[1]) + int64_t(h) * kKDim * kVDim;
#pragma unroll
      for (int i = 0; i < kRecRows; i++) so_[(j + 8 * i) * kVDim + col] = S[i];
    }
    in.k = (cur ? sp.k[1] : sp.k[0]) + r0 * kQ, in.v = (cur ? sp.v[1] : sp.v[0]) + r0 * kVHeads * kVDim;
    in.ab = (cur ? sp.ab[1] : sp.ab[0]) + r0 * 2 * kVHeads;
    pdl_wait();
    // Conv state of the layer: the committed window (the prep kernel, which reads the old one, is done).
    const int c = blockIdx.x * blockDim.x + tid;
    if (nc > 0 && c < kConvDim) {
      const float* rp = (prev ? sp.raw[1] : sp.raw[0]) + r0 * kConvDim;
      float* cs = (kBatch ? lane_sel(sp.lconv, ln) : sp.conv_state) + c * (kConvK - 1);
      float x0 = cs[0], x1 = cs[1], x2 = cs[2];
      for (int jj = 0; jj < nc; jj++) x0 = x1, x1 = x2, x2 = rp[int64_t(sp.cpath[r0 + jj]) * kConvDim + c];
      cs[0] = x0, cs[1] = x1, cs[2] = x2;
    }
    if (T == 0) return;  // (uniform over the cluster: no cluster barrier is reached)
    const SpecTree* tree = sp.tree + ln;
    if constexpr (kTree) {
      n = tree->n;
      for (int t = tid; t < n; t += kRecWarps * 32) s_rows[t] = t;
    } else {
      const int leaf = tree->leaf[0];
      n = tree->depth[leaf] + 1;
      if (tid < n) s_rows[tid] = tree->path[leaf][tid];
    }
  } else {
    pdl_wait();
    if (tid < T) s_rows[tid] = tid;
  }
  __syncthreads();
  GDN_PROBE(1);
  stage(in, n, s_rows);
  GDN_PROBE(2);
  if constexpr (kTree) {
    // Path k > 0 starts at its first own node (depth beg) from the state after its parent node par,
    // which an earlier path computed: that path saves S into snap[k - 1] right after computing par.
    const SpecTree* tree = sp.tree + ln;
    const int np = min(tree->npaths, kTreePaths);
    int par[kTreePaths], beg[kTreePaths];
#pragma unroll
    for (int k = 0; k < kTreePaths; k++) {
      par[k] = -1, beg[k] = 0;
      if (k > 0 && k < np) {
        const int leaf = tree->leaf[k], dl = tree->depth[leaf];
        int b = dl + 1;  // (a duplicate path owns no node: nothing to run)
        for (int d = dl; d >= 1; d--)
          if (tree->owner[tree->path[leaf][d]] == k) b = d;
        beg[k] = b, par[k] = b <= dl ? tree->path[leaf][b - 1] : -1;
      }
    }
    float snap[kTreePaths - 1][kRecRows];
#pragma unroll
    for (int k = 0; k < kTreePaths; k++) {
      if (k >= np) break;
      if (k > 0) {
        if (par[k] < 0) continue;
#pragma unroll
        for (int i = 0; i < kRecRows; i++) S[i] = snap[k - 1][i];
      }
      const int leaf = tree->leaf[k], dl = tree->depth[leaf];
      for (int d = beg[k]; d <= dl; d++) {
        const int t = tree->path[leaf][d];
        const float o = gdn_step(S, sk[t], sq[t], sv[t][lcol], s_decay[t], s_beta[t]);
        if (j == 4) so[t][lcol] = o;
#pragma unroll
        for (int kk = 1; kk < kTreePaths; kk++)
          if (kk > k && par[kk] == t) {
#pragma unroll
            for (int i = 0; i < kRecRows; i++) snap[kk - 1][i] = S[i];
          }
      }
    }
  } else {
    for (int t = 0; t < n; t++) {
      const float o = gdn_step(S, sk[t], sq[t], sv[t][lcol], s_decay[t], s_beta[t]);
      if (j == 4) so[t][lcol] = o;
    }
  }
  GDN_PROBE(3);
  if constexpr (!kVerify) {
    float* sw = state + int64_t(h) * kKDim * kVDim;
#pragma unroll
    for (int i = 0; i < kRecRows; i++) sw[(j + 8 * i) * kVDim + col] = S[i];
  }
  __syncthreads();
  // Gated RMSNorm: warp t sums o^2 of row t over this CTA's 32 columns, the cluster adds the 4 parts
  // (DSMEM).
  for (int t = warp; t < (kTree ? n : kRecWarps); t += kRecWarps) {
    const float o = t < n ? so[t][lane] : 0.f;
    float ss = o * o;
    for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    if (lane == 0) s_part[t] = ss;
  }
  GDN_PROBE(4);
  cluster.sync();
  GDN_PROBE(5);
  for (int t = warp; t < n; t += kRecWarps) {
    const float o = so[t][lane];
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < kRecSplit; r++) ss += cluster.map_shared_rank(s_part, r)[t];
    const int c = part * kRecCtaCols + lane;  // (thread (t, lane) of the norm, not the lane's state column)
    const int row = s_rows[t];
    const float zz = z[int64_t(row) * ldz + h * kVDim + c];
    const float y = o * rsqrtf(ss / kVDim + eps) * __bfloat162float(norm_w[c]) * (zz / (1.f + expf(-zz)));
    const int64_t oi = int64_t(row) * (kVHeads * kVDim) + h * kVDim + c;
    out[oi] = y;
    if (q8) q8[oi] = __nv_cvt_float_to_fp8(y / in_scale, __NV_SATFINITE, __NV_E4M3);
  }
  GDN_PROBE(6);
  cluster.sync();  // the other CTAs may still read s_part
  GDN_PROBE(7);
}

}  // namespace

void gdn_prep(const float* qkv, int ld, float* conv_state, const bf16* conv_w, const float* a, const float* b,
              const bf16* A_log, const bf16* dt_bias, int T, float* q, float* k, float* v, float* g, float* beta,
              cudaStream_t s) {
  if (T == 0) return;
  launch_pdl(conv_norm_kernel, dim3(T, kKHeads + kPrepVBlocks), dim3(256), 0, s, qkv, ld,
             static_cast<const float*>(conv_state), conv_w, q, k, v, a, b, A_log, dt_bias, g, beta);
  Q27_CUDA(cudaGetLastError());
  launch_pdl(conv_state_kernel, dim3((kConvDim + 255) / 256), dim3(256), 0, s, qkv, ld, conv_state, T);
  Q27_CUDA(cudaGetLastError());
}

void gdn_recurrent(const float* q, const float* k, const float* v, const float* g, const float* beta, int T,
                   float* state, float* o, cudaStream_t s) {
  if (T == 0) return;
  launch_pdl(gdn_recurrent_kernel, dim3(dim3(kVHeads, kVDim / kCols)), dim3(256), 0, s, q, k, v, g, beta, T, state, o);
  Q27_CUDA(cudaGetLastError());
}

size_t gdn_chunk_ws_bytes(int T) {
  const size_t nc = (T + 2 * kCh - 2) / kCh;  // any pos0
  return nc * ((size_t(kVHeads) * kCh * (kKDim + kCh) + 2 * size_t(kKHeads) * kCh * kKDim) * 2 +
               size_t(kVHeads) * 2 * kCh * 4);
}

void gdn_chunked(const float* q, const float* k, float* v, const float* g, const float* beta, int T, int pos0,
                 float* state, float* o, void* ws, cudaStream_t s) {
  if (T == 0) return;
  static bool init = [] {
    Q27_CUDA(cudaFuncSetAttribute(gdn_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kChPrepSmem));
    Q27_CUDA(cudaFuncSetAttribute(gdn_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kChSmem));
    return true;
  }();
  (void)init;
  const int off = pos0 % kCh, nc = chunk_count(T, pos0);
  const ChunkWs w = chunk_ws(ws, nc);
  launch_pdl(gdn_chunk_prep_kernel, dim3(nc, kVHeads), dim3(kChPrepThreads), kChPrepSmem, s, q, k, v, g, beta, T, off,
             w);
  Q27_CUDA(cudaGetLastError());
  launch_pdl(gdn_chunk_kernel, dim3(kVHeads, kVDim / (16 * kChWarps)), dim3(kChWarps * 32), kChSmem, s,
             static_cast<const float*>(v), T, off, nc, w, state, o);
  Q27_CUDA(cudaGetLastError());
}

void gdn_gated_norm(const float* o, const float* z, int ldz, const bf16* w, int T, float eps, float* out,
                    cudaStream_t s, uint8_t* q8, float in_scale) {
  if (T == 0) return;
  Q27_CHECK(ldz % 4 == 0, "gdn_gated_norm: z rows must be 16-byte aligned");
  launch_pdl(gated_norm_kernel, dim3(T, kVHeads / kGnHeads), dim3(kGnHeads * 32), 0, s, o, z, ldz, w, eps, out, q8,
             in_scale);
  Q27_CUDA(cudaGetLastError());
}

void gdn_decode_prep(const float* qkv, float* conv_state, const bf16* conv_w, const float* h, const bf16* wa,
                     const bf16* wb, float* q, float* k, float* v, float* a, float* b, cudaStream_t s) {
  // a and b are adjacent in the caller's scratch (a_ then b_ is not guaranteed): write ab rows to a.
  (void)b;
  launch_pdl(gdn_prep_kernel<false, false>, dim3(prep_blocks<false>(1)), dim3(kPrepThreads), 0, s, qkv, kConvDim, 1, conv_state,
             conv_w,
             h, wa, wb, PrepOut{q, k, v, a}, GdnSpec{});
  Q27_CUDA(cudaGetLastError());
}

void gdn_decode(const float* q, const float* k, const float* v, const float* a, const float* b, const bf16* A_log,
                const bf16* dt_bias, float* state, const float* z, const bf16* norm_w, float eps, float* out,
                uint8_t* q8, float in_scale, cudaStream_t s) {
  (void)b;
  launch_pdl_cluster(gdn_rec_kernel<false, false>, dim3(kVHeads * kRecSplit), dim3(kRecWarps * 32), 0, s, kRecSplit,
                     RecIn{k, q, v, a}, 1, A_log, dt_bias, state, z, kVHeads * kVDim, norm_w, eps, out, q8, in_scale,
                     GdnSpec{});
  Q27_CUDA(cudaGetLastError());
}

void gdn_verify_prep(const float* qkv, int ld, int T, float* conv_state, const bf16* conv_w, const float* h,
                     const bf16* wa, const bf16* wb, float* q, const GdnSpec& sp, cudaStream_t s) {
  Q27_CHECK(T <= kVerifyMaxT, "gdn_verify_prep: T must be <= 32");
  // T = 0 (flush): one column of blocks that only replays the conv state.
  // A chain (one path) needs no tree lookups: row t is at depth t.
  if (sp.lanes > 0) {  // batched chain verify: T rows per lane (P6-10)
    Q27_CHECK(sp.paths <= 1 && T <= kLaneRows && sp.lanes <= kMaxLanes, "gdn_verify_prep: batched chains only");
    launch_pdl(gdn_prep_kernel<true, false, true>, dim3(prep_blocks<true>(T), sp.lanes), dim3(kPrepThreads), 0, s, qkv,
               ld, T, conv_state, conv_w, h, wa, wb, PrepOut{q, nullptr, nullptr, nullptr}, sp);
  } else if (sp.paths > 1)
    launch_pdl(gdn_prep_kernel<true, true>, dim3(prep_blocks<true>(T)), dim3(kPrepThreads), 0, s, qkv, ld, T, conv_state, conv_w,
               h, wa, wb, PrepOut{q, nullptr, nullptr, nullptr}, sp);
  else
    launch_pdl(gdn_prep_kernel<true, false>, dim3(prep_blocks<true>(T)), dim3(kPrepThreads), 0, s, qkv, ld, T, conv_state,
               conv_w, h, wa, wb, PrepOut{q, nullptr, nullptr, nullptr}, sp);
  Q27_CUDA(cudaGetLastError());
}

void gdn_verify(const float* q, int T, const bf16* A_log, const bf16* dt_bias, float* state, const float* z, int ldz,
                const bf16* norm_w, float eps, float* out, uint8_t* q8, float in_scale, const GdnSpec& sp,
                cudaStream_t s) {
  Q27_CHECK(T <= kVerifyMaxT, "gdn_verify: T must be <= 32");
  if (sp.lanes > 0) {
    Q27_CHECK(sp.paths <= 1 && T <= kLaneRows && sp.lanes <= kMaxLanes, "gdn_verify: batched chains only");
    launch_pdl_cluster(gdn_rec_kernel<true, false, true>, dim3(kVHeads * kRecSplit, sp.lanes), dim3(kRecWarps * 32), 0,
                       s, kRecSplit, RecIn{nullptr, q, nullptr, nullptr}, T, A_log, dt_bias, state, z, ldz, norm_w,
                       eps, out, q8, in_scale, sp);
  } else if (sp.paths > 1)
    launch_pdl_cluster(gdn_rec_kernel<true, true>, dim3(kVHeads * kRecSplit), dim3(kRecWarps * 32), 0, s, kRecSplit,
                       RecIn{nullptr, q, nullptr, nullptr}, T, A_log, dt_bias, state, z, ldz, norm_w, eps, out, q8,
                       in_scale, sp);
  else
    launch_pdl_cluster(gdn_rec_kernel<true, false>, dim3(kVHeads * kRecSplit), dim3(kRecWarps * 32), 0, s, kRecSplit,
                       RecIn{nullptr, q, nullptr, nullptr}, T, A_log, dt_bias, state, z, ldz, norm_w, eps, out, q8,
                       in_scale, sp);
  Q27_CUDA(cudaGetLastError());
}

#ifdef Q27_GDN_PROBE
void gdn_probe_read(long long* out) { Q27_CUDA(cudaMemcpyFromSymbol(out, g_gdn_probe, sizeof(g_gdn_probe))); }
#else
void gdn_probe_read(long long* out) {
  for (int i = 0; i < 16; i++) out[i] = 0;
}
#endif

}  // namespace q27
