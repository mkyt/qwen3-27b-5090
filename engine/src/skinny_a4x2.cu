// Skinny W4A4x2 GEMM for the lm_head: D[M][N] = alpha * (X0 + X1)[M][K] . W[N][K]^T, M = 1..64, where X0 is the
// NVFP4 quantization of the activations and X1 the NVFP4 quantization of what X0 left (same global scale, a block
// scale of its own). The two terms hold the activations to about 1% (one: 9%), which on the lm_head gives the
// distributions of BF16 activations, and both run on block-scaled FP4 MMAs with the checkpoint's weights as they
// are: no expansion, the weights read once, twice the MMAs of W4A4.
//
// Operands: q [2 M][K / 2] packed e2m1 and sf [2 M][K / 16] ue4m3 (row-major, not swizzled), row 2 tok + term
// (quant_fp4x2). The weights are in skinny_fp4's layout.
//
// BF16 / FP16 activations on MMAs of their own were tried first: the same distributions, but the MMAs accumulate
// in fp32 at a fraction of the FP4 rate (+0.48 ms at 64 rows, against +0.08 here).
//
// A CTA owns 16 output channels per warp and all of K: the lm_head has over 1,000 tiles, for which skinny.cu uses
// no split-K either. Hence no partial sums, and a token's row does not depend on M. Both operands stream through
// shared memory (cp.async, ldmatrix): a ring per warp for the weights, a ring per CTA for the activations.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "common.h"
#include "kernels.h"
#include "launch.cuh"

namespace q27 {
namespace {

constexpr int kMaxTokens = 64;
constexpr int kTerms = 2;
constexpr int kChunkSteps = 4;                // k-steps of 64 elements, as skinny.cu
constexpr int kRowBytes = 32 * kChunkSteps;   // of a row of either operand per chunk
constexpr int kSfBytes = 4 * kChunkSteps;     // of an activation row's scales per chunk
constexpr int kWeightChunk = 16 * kRowBytes;  // per warp and stage
constexpr int kL2Chunks = 8;

struct X2Params {
  int M, N, chunks, steps, groups;
  const uint8_t* q;     // [2 M][K / 2]
  const uint8_t* q_sf;  // [2 M][K / 16]
  const uint8_t* w;     // [N][K / 2]
  const uint8_t* w_sf;  // swizzled scales
  float alpha;
  float* d;
};

__device__ __forceinline__ void cp_async16(uint32_t smem, const uint8_t* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_async16z(uint32_t smem, const uint8_t* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}
__device__ __forceinline__ void mma_fp4(float* acc, const uint32_t* a, const uint32_t* b, uint32_t sfa,
                                        uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%11, %12};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(uint16_t(0)),
        "h"(uint16_t(0)), "r"(sfb));
}

// Shared-memory offset of piece c (16 bytes) of row r of a chunk: 8 consecutive rows (an ldmatrix matrix) fall
// on 8 different 16-byte banks.
__device__ __forceinline__ int piece(int r, int c) { return r * kRowBytes + ((c ^ (r % 8)) * 16); }

__device__ __forceinline__ int64_t sf_step_offset(int row, int step, int steps) {
  return (int64_t(row / 128) * steps + step) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4;
}

// kG = ceil(M / 8) exactly: no branch on M between the MMAs, so the compiler schedules the loads ahead of them.
// The activations and their scales stream through a ring of the CTA like the weights (its warps share one read
// of each chunk): 4-byte loads of them through L1, as skinny.cu has up to 32 rows, cost 40-800 us a call here.
// The order of a token's MMAs does not depend on M or the layout: k ascending in pairs of steps, the first term
// before the second.
template <int kG, int kStages, int kWarps>
__global__ void __launch_bounds__(kWarps * 32) skinny_a4x2_kernel(X2Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int wrow = (blockIdx.x * kWarps + warp) * 16;
  const int row0 = wrow + g;
  const int64_t bytes = int64_t(p.steps) * 32;  // of a row of either operand

  const uint32_t sbase = uint32_t(__cvta_generic_to_shared(smem));
  const uint32_t ring = sbase + warp * kStages * kWeightChunk;
  constexpr int kM8 = 8 * kG, kARows = kTerms * kM8, kAStage = kARows * (kRowBytes + kSfBytes);
  const int aoff = kWarps * kStages * kWeightChunk;
  const uint32_t aring = sbase + aoff;

  const uint8_t* src[4];
  uint32_t dst[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = lane + 32 * j, r = i / 8, c = i % 8;
    src[j] = p.w + int64_t(wrow + r) * bytes + c * 16;
    dst[j] = ring + piece(r, c);
  }
  auto issue_w = [&](int ch) {
    if (ch >= p.chunks) return;
#pragma unroll
    for (int j = 0; j < 4; j++) cp_async16(dst[j] + (ch % kStages) * kWeightChunk, src[j] + int64_t(ch) * kRowBytes);
  };
  // Rows term * kM8 + tok of a stage, 8 pieces of codes each, then one piece of scales each.
  auto issue_x = [&](int ch) {
    if (ch >= p.chunks) return;
    const uint32_t base = aring + (ch % kStages) * kAStage;
    for (int i = threadIdx.x; i < 9 * kARows; i += kWarps * 32) {
      const int r = i / 9, c = i % 9, term = r / kM8, tok = r % kM8;
      const bool ok = tok < p.M;
      const int64_t from = 2 * int64_t(ok ? tok : 0) + term;
      if (c < 8)
        cp_async16z(base + piece(r, c), p.q + from * bytes + int64_t(ch) * kRowBytes + c * 16, ok);
      else
        cp_async16z(base + kARows * kRowBytes + r * 16, p.q_sf + from * p.steps * 4 + int64_t(ch) * kSfBytes, ok);
    }
  };

#pragma unroll
  for (int i = 0; i < kStages - 1; i++) {
    issue_w(i);
    cp_async_commit();
  }
  if (lane < 16) {
    const uint8_t* row = p.w + int64_t(wrow + lane) * bytes;
    for (int c = kStages - 1; c < min(p.chunks, kStages - 1 + kL2Chunks); c++)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(row + int64_t(c) * kRowBytes));
  }
  // The weights' scales are loaded one chunk ahead.
  const uint8_t* wsf = p.w_sf + sf_step_offset(t % 2 == 0 ? row0 : row0 + 8, 0, p.steps);
  uint32_t SA[kChunkSteps], SA_next[kChunkSteps];
  auto load_sa = [&](int ch) {
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++)
      SA_next[u] = __ldg(reinterpret_cast<const uint32_t*>(wsf + int64_t(ch * kChunkSteps + u) * 512));
  };
  load_sa(0);
  pdl_trigger();
  pdl_wait();  // the activations and their scales come from the previous kernel
#pragma unroll
  for (int i = 0; i < kStages - 1; i++) {
    issue_x(i);
    cp_async_commit();
  }

  const int lm_row = lane % 8 + 8 * ((lane / 8) % 2), lm_half = lane / 16;

  float acc[kG][4];
#pragma unroll
  for (int gi = 0; gi < kG; gi++) acc[gi][0] = acc[gi][1] = acc[gi][2] = acc[gi][3] = 0.f;

  for (int ch = 0; ch < p.chunks; ch++) {
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++) SA[u] = SA_next[u];
    load_sa(min(ch + 1, p.chunks - 1));
    cp_async_wait<kStages - 2>();
    const uint32_t wb = ring + (ch % kStages) * kWeightChunk;
    const uint32_t ab = aring + (ch % kStages) * kAStage;
    uint32_t A[kChunkSteps][4];
    __syncthreads();  // every thread's pieces of the chunk are in; the previous chunk's slot was read
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++) ldmatrix_x4(A[u], wb + piece(lm_row, 2 * u + lm_half));
    __syncwarp();  // the slot is refilled below
    issue_w(ch + kStages - 1);
    issue_x(ch + kStages - 1);
    cp_async_commit();

#pragma unroll
    for (int h = 0; h < kChunkSteps / 2; h++)
#pragma unroll
      for (int term = 0; term < kTerms; term++)
#pragma unroll
        for (int gi = 0; gi < kG; gi++) {
          const int r = term * kM8 + 8 * gi;
          uint32_t B[4];
          ldmatrix_x4(B, ab + piece(r + lane % 8, 4 * h + lane / 8));
          const uint2 sb = *reinterpret_cast<const uint2*>(smem + aoff + (ch % kStages) * kAStage + kARows * kRowBytes +
                                                          (r + g) * 16 + 8 * h);
          mma_fp4(acc[gi], A[2 * h], B, SA[2 * h], sb.x);
          mma_fp4(acc[gi], A[2 * h + 1], B + 2, SA[2 * h + 1], sb.y);
        }
  }
  cp_async_wait<0>();
  trace_point(2);

#pragma unroll
  for (int gi = 0; gi < kG; gi++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int row = row0 + (j / 2) * 8, tok = 8 * gi + 2 * t + j % 2;
      if (tok < p.M) p.d[int64_t(tok) * p.N + row] = __fmul_rn(p.alpha, acc[gi][j]);
    }
}

template <int kG, int kStages, int kWarps>
void launch_x2(const X2Params& p, cudaStream_t s) {
  constexpr int kSmem = kStages * (kWarps * kWeightChunk + kTerms * 8 * kG * (kRowBytes + kSfBytes));
  static_assert(kSmem <= 99 * 1024, "more shared memory than a CTA gets (RTX 5090)");
  static const bool attr = [] {
    Q27_CUDA(cudaFuncSetAttribute(skinny_a4x2_kernel<kG, kStages, kWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSmem));
    return true;
  }();
  (void)attr;
  launch_pdl(skinny_a4x2_kernel<kG, kStages, kWarps>, dim3(p.N / (16 * kWarps)), dim3(kWarps * 32), kSmem, s, p);
}

// The CTA and the rings by M, as swept on the lm_head (248,320 x 5,120): up to 32 rows 8 warps and 3 stages, up
// to 56 a CTA of 16 warps (half the reads of the activations from L2), 64 rows 10 warps (16 do not fit the shared
// memory). Only the speed depends on it. An N that such a CTA does not divide gets 8 warps.
template <int kG>
void launch_rows(const X2Params& p, cudaStream_t s) {
  if constexpr (kG <= 4) {
    launch_x2<kG, 3, 8>(p, s);
  } else if constexpr (kG < 8) {
    p.N % 256 ? launch_x2<kG, 2, 8>(p, s) : launch_x2<kG, 2, 16>(p, s);
  } else {
    p.N % 160 ? launch_x2<kG, 2, 8>(p, s) : launch_x2<kG, 2, 10>(p, s);
  }
}

// ---------------------------------------------------------------- the two-term quantization

// (as kernels.cu's: the same order of the sums, so that the norm equals rmsnorm_quant's bit for bit)
__device__ float block_sum(float v) {
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < int(blockDim.x / 32) ? red[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ uint16_t cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(odd), "f"(even));
  return r;
}

__device__ __forceinline__ float2 e2m1x2_values(uint16_t codes) {
  uint32_t r;
  asm("{ .reg .b8 __$lo, __$hi;\n"
      "  mov.b16 {__$lo, __$hi}, %1;\n"
      "  cvt.rn.f16x2.e2m1x2 %0, __$lo; }\n"
      : "=r"(r)
      : "h"(codes));
  return __half22float2(*reinterpret_cast<const __half2*>(&r));
}

// Thread k takes the 16 elements of block k. x: the residual stream (w: the final norm's weight, RMSNorm first)
// or, with w null, the activations themselves.
__global__ void __launch_bounds__(320) quant_fp4x2_kernel(const float* x, const bf16* w, int K, float eps,
                                                          float input_scale, uint8_t* q, uint8_t* sf) {
  pdl_trigger();
  const int m = blockIdx.x, kb = threadIdx.x;
  const float* xr = x + int64_t(m) * K + kb * 16;
  pdl_wait();
  float v[16];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 f = reinterpret_cast<const float4*>(xr)[i];
    v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
  }
  if (w) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) ss += v[i] * v[i];
    const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
    for (int i = 0; i < 16; i++) v[i] = v[i] * r * (1.f + __bfloat162float(__ldg(w + kb * 16 + i)));
  }
  const float gs = 1.f / input_scale;
#pragma unroll
  for (int term = 0; term < kTerms; term++) {  // (the first as kernels.cu's quant_block16)
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
    const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
    const float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
    const float inv = sff != 0.f ? gs / sff : 0.f;
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint16_t codes = cvt_e2m1x2(v[2 * j] * inv, v[2 * j + 1] * inv) & 0xff;
      packed[j / 4] |= uint32_t(codes) << (8 * (j % 4));
      if (term + 1 < kTerms) {  // what this term leaves
        const float2 c = e2m1x2_values(codes);
        v[2 * j] -= c.x * sff / gs;
        v[2 * j + 1] -= c.y * sff / gs;
      }
    }
    const int64_t row = 2 * int64_t(m) + term;
    *reinterpret_cast<uint2*>(q + row * (K / 2) + kb * 8) = make_uint2(packed[0], packed[1]);
    sf[row * (K / 16) + kb] = sfq;
  }
}

}  // namespace

void skinny_fp4_a4x2(int M, int N, int K, const uint8_t* q, const uint8_t* q_sf, const uint8_t* w,
                     const uint8_t* w_sf, float alpha, float* d, cudaStream_t s) {
  Q27_CHECK(M >= 1 && M <= kMaxTokens, "W4A4x2 gemm: M must be 1..64");
  Q27_CHECK(N % 128 == 0, "W4A4x2 gemm: N must be a multiple of 128");
  Q27_CHECK(K % (64 * kChunkSteps) == 0, "W4A4x2 gemm: K must be a multiple of 256");
  X2Params p{};
  p.M = M, p.N = N, p.steps = K / 64, p.chunks = p.steps / kChunkSteps, p.groups = int(ceil_div(M, 8));
  p.q = q, p.q_sf = q_sf, p.w = w, p.w_sf = w_sf, p.alpha = alpha, p.d = d;
  switch (p.groups) {
    case 1: launch_rows<1>(p, s); break;
    case 2: launch_rows<2>(p, s); break;
    case 3: launch_rows<3>(p, s); break;
    case 4: launch_rows<4>(p, s); break;
    case 5: launch_rows<5>(p, s); break;
    case 6: launch_rows<6>(p, s); break;
    case 7: launch_rows<7>(p, s); break;
    default: launch_rows<8>(p, s); break;
  }
  Q27_CUDA(cudaGetLastError());
}

void quant_fp4x2(const float* x, const bf16* w, int M, int K, float eps, float input_scale, uint8_t* q, uint8_t* sf,
                 cudaStream_t s) {
  Q27_CHECK(K % 16 == 0 && K / 16 <= 320, "quant_fp4x2: K must be a multiple of 16, up to 5120");
  Q27_CHECK(!w || K % 512 == 0, "quant_fp4x2: the norm needs whole warps");
  if (M == 0) return;
  launch_pdl(quant_fp4x2_kernel, dim3(M), dim3(K / 16), 0, s, x, w, K, eps, input_scale, q, sf);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
