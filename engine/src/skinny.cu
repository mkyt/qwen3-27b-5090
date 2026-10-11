// Skinny GEMMs for decode (M = 1..8 tokens): D[M][N] = alpha(n) * A[M][K] . W[N][K]^T (+ beta * C).
// Weights are the mma A operand (16 output channels per warp), the tokens the B operand (n8).
// NVFP4: mma.sync m16n8k64 block-scaled (ue4m3 scales, 16 elements per scale), FP8: m16n8k32,
// BF16 (the draft model): m16n8k16.
// One k-step is 32 bytes of a weight row in all formats, so the addressing is shared:
//   lane = 4 * g + t; A regs: rows g / g + 8, bytes 4t..4t+3 and 16 + 4t..; B regs: token g, same bytes.
// A CTA (8 warps) owns 128 output channels, which is also one row block of the CUTLASS scale-factor
// layout, so the CTA reads whole 512-byte scale atoms. K is split across CTAs (gridDim.y); the last
// CTA of a tile adds the partials in split order (deterministic) and applies the epilogue.
#include <algorithm>
#include <cstdlib>

#include "common.h"
#include "kernels.h"
#include "launch.cuh"

namespace q27 {
namespace {

#ifndef Q27_SKINNY_WARPS
#define Q27_SKINNY_WARPS 8
#endif
#ifndef Q27_SKINNY_STAGES
#define Q27_SKINNY_STAGES 3  // 48 KiB per CTA: two CTAs per SM (swept 2..8 stages, 4 / 8 warps)
#endif
constexpr int kWarps = Q27_SKINNY_WARPS;
constexpr int kRowsPerCta = 16 * kWarps;  // 128
constexpr int kMaxTokens = 64;  // M <= 64: up to 8 groups of 8 tokens (n8 MMA tiles) per warp
constexpr int kMaxTiles = 4096;  // split-K counters (N <= 512K)
constexpr int kWideMaxTiles = 1024;

struct Params {
  int M, N, kbytes, steps, steps_per_split, splits;
  const uint8_t* a;     // activations [M][kbytes]
  const uint8_t* a_sf;  // swizzled scales of the activations (NVFP4)
  const uint8_t* w;     // weights [N][kbytes]
  const uint8_t* w_sf;  // swizzled scales of the weights (NVFP4)
  AlphaSegments alpha;
  const float* c;
  float beta;
  float* d;
  const float* wscale;  // optional per-output-channel scale [N] (draft weights)
  const float* ascale;  // optional per-token scale [M] (dynamically quantized activations)
  int l2_chunks;        // chunks per warp prefetched to L2 before the PDL wait
  int ext_reduce;       // split-K partials summed by skinny_reduce_kernel (the next launch), not the last CTA
  float* part;          // [splits][8][N]
  int* counters;        // [N / 128], zero between calls
};

__device__ __forceinline__ uint32_t ld_cached(const uint8_t* p) {
  return __ldg(reinterpret_cast<const uint32_t*>(p));
}

// 4x4 transpose of 32-bit words across the 4 lanes of a quad (lane t ends with word t of every lane:
// w[i] = lane i's w[t]), two butterfly stages of shuffles.
__device__ __forceinline__ void quad_transpose(uint32_t* w, int t) {
#pragma unroll
  for (int m = 1; m <= 2; m *= 2) {
    const bool hi = t & m;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if (i & m) continue;
      const uint32_t r = __shfl_xor_sync(0xffffffffu, hi ? w[i] : w[i + m], m);
      if (hi)
        w[i] = r;
      else
        w[i + m] = r;
    }
  }
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

__device__ __forceinline__ void mma_fp8(float* acc, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_bf16(float* acc, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

enum class Fmt { kFp4, kFp8, kBf16 };

__device__ __forceinline__ float alpha_of(const AlphaSegments& s, int n) {
  float a = s.alpha[0];
#pragma unroll
  for (int i = 1; i < AlphaSegments::kMax; i++)
    if (i < s.count && n >= s.begin[i]) a = s.alpha[i];
  return a;
}

// Byte offset of the 4 scales of k-step `step` (k-blocks 4 * step .. + 3) of `row` in the CUTLASS
// swizzled layout (see sf_offset): contiguous, and one 512-byte atom per 128 rows and k-step.
__device__ __forceinline__ int64_t sf_step_offset(int row, int step, int steps) {
  return (int64_t(row / 128) * steps + step) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4;
}

__device__ __forceinline__ void cp_async16(uint32_t smem, const uint8_t* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}
// (zero-fills the 16 bytes when !ok)
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

// Each warp streams its 16 weight rows through a private ring of kStages chunks in shared memory
// (chunk = 4 k-steps = 16 rows x 128 bytes; 16-byte columns XOR-swizzled by row % 8 so that
// ldmatrix is conflict-free). ldmatrix.x4 hands out exactly the mma A fragment:
// matrices (rows 0-7 | 8-15) x (bytes 0-15 | 16-31) of a k-step -> a0, a1, a2, a3.
constexpr int kChunkSteps = 4;
constexpr int kChunkBytes = 16 * 32 * kChunkSteps;  // 2 KiB per warp and stage
// Chunks per warp prefetched to L2 before the PDL wait (Q27_SKINNY_L2_CHUNKS, default 8; swept
// 0 / 4 / 8 / 16: 0 is ~0.2 ms slower per decode step, the rest equal).
constexpr int kL2ChunksDefault = 8;

// kG groups of 8 tokens (M <= 8 kG): every group has its own accumulators and MMAs over the same A
// fragments, so a token's result does not depend on M (speculative verify rows equal decode rows).
// kWideB (M > 16, P6-11): the activations come in 16-byte loads and a quad transpose instead of 4-byte
// loads (8 cache lines per load instruction: at M = 32 they took most of the time over M = 8). The MMA
// inputs are the same.
// kG = 8 (M > 32, batched steps of 5..8 sequences, Phase 7): the CTA copies each chunk of the
// activations (64 tokens x 128 bytes) once into a ring of kStages slots after the weight rings (16-byte
// pieces XOR-swizzled by token % 8), and every warp takes its B fragments from there with ldmatrix
// (matrix = 8 tokens x one 16-byte piece: b0 / b1 of k-steps 2 per x4). kG = 8 runs one CTA per SM
// (170-225 registers), so the extra 24 KiB of shared memory are free; the 8 warps no longer load and
// transpose the same activations each. The MMA inputs are the same.
constexpr int kActChunkBytes = kMaxTokens * 128;  // one chunk of the activations (kG = 8)
template <Fmt kFmt, int kStages, int kG, bool kWideB = false>
__global__ void __launch_bounds__(kWarps * 32) skinny_kernel(Params p) {
  constexpr bool kFp4 = kFmt == Fmt::kFp4;
  constexpr bool kSmemB = kG == 8;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int tile = blockIdx.x, split = blockIdx.y;
  const int wrow = tile * kRowsPerCta + warp * 16;  // first row of this warp
  const int row0 = wrow + g;
  const int s0 = split * p.steps_per_split, s1 = min(p.steps, s0 + p.steps_per_split);
  const int chunks = (s1 - s0) / kChunkSteps;
  bool tok_ok[kG];
#pragma unroll
  for (int gi = 0; gi < kG; gi++) tok_ok[gi] = 8 * gi + g < p.M;

  const uint32_t ring = uint32_t(__cvta_generic_to_shared(smem)) + warp * kStages * kChunkBytes;
  // cp.async: lane copies 16-byte columns i = lane + 32 j (row i / 8, column i % 8) of a chunk.
  const uint8_t* src[4];
  uint32_t dst[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int i = lane + 32 * j, r = i / 8, c = i % 8;
    src[j] = p.w + int64_t(wrow + r) * p.kbytes + int64_t(s0) * 32 + c * 16;
    dst[j] = r * 128 + ((c ^ (r % 8)) * 16);
  }
  // kSmemB: thread i copies pieces (token i / 8 + 32 j, piece i % 8), j = 0, 1, of the tokens < 8 groups.
  const uint32_t aring = uint32_t(__cvta_generic_to_shared(smem)) + kWarps * kStages * kChunkBytes;
  const uint8_t* asrc[2];
  uint32_t adst[2];
  bool aok[2], acopy[2];
  if constexpr (kSmemB) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int tok = threadIdx.x / 8 + 32 * j, c = threadIdx.x % 8;
      aok[j] = tok < p.M, acopy[j] = tok < 8 * ((p.M + 7) / 8);
      asrc[j] = p.a + int64_t(aok[j] ? tok : 0) * p.kbytes + int64_t(s0) * 32 + c * 16;
      adst[j] = tok * 128 + ((c ^ (tok % 8)) * 16);
    }
  }
  // (weights / activations: the activations come from the previous kernel, after the PDL wait; the
  // prologue's weight groups come first, then those of its activations, then one group per chunk.)
  auto issue = [&](int chunk, bool wts = true, bool acts = true) {
    if (chunk < chunks) {
      uint32_t base = ring + (chunk % kStages) * kChunkBytes;
      if (wts) {
#pragma unroll
        for (int j = 0; j < 4; j++) cp_async16(base + dst[j], src[j] + int64_t(chunk) * 128);
      }
      if (kSmemB && acts) {
        const uint32_t abase = aring + (chunk % kStages) * kActChunkBytes;
#pragma unroll
        for (int j = 0; j < 2; j++)
          if (acopy[j]) cp_async16z(abase + adst[j], asrc[j] + (aok[j] ? int64_t(chunk) * 128 : 0), aok[j]);
      }
    }
    cp_async_commit();
  };
  // ldmatrix: lane gives the address of row (lane % 8) + 8 * (matrix % 2), half (matrix / 2).
  const int lm_row = lane % 8 + 8 * ((lane / 8) % 2), lm_half = lane / 16;

  const uint8_t* xa[kG];
  const uint8_t* xsf[kG];
  const uint8_t* wsf = nullptr;
#pragma unroll
  for (int gi = 0; gi < kG; gi++) {
    const int tok = tok_ok[gi] ? 8 * gi + g : 0;
    xa[gi] = p.a + int64_t(tok) * p.kbytes + 4 * t;
    xsf[gi] = kFp4 ? p.a_sf + sf_step_offset(tok, 0, p.steps) : nullptr;
  }
  if constexpr (kFp4) wsf = p.w_sf + sf_step_offset(t % 2 == 0 ? row0 : row0 + 8, 0, p.steps);

#pragma unroll
  for (int i = 0; i < kStages - 1; i++) issue(i, true, false);
  // While the previous kernels finish, pull the next chunks of this warp's weights into L2 as well
  // (the shared-memory ring holds only kStages - 1 chunks): DRAM keeps streaming through the short
  // kernels between GEMMs. Lane l < 16 covers row l of the warp's 16 rows, 128 bytes per chunk.
  if (lane < 16) {
    const uint8_t* row = p.w + int64_t(wrow + lane) * p.kbytes + int64_t(s0) * 32;
    for (int c = kStages - 1; c < min(chunks, kStages - 1 + p.l2_chunks); c++)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(row + int64_t(c) * 128));
  }
  // Everything above reads only weights; the activations come from the previous kernel.
  pdl_trigger();
  pdl_wait();
  if constexpr (kSmemB) {
#pragma unroll
    for (int i = 0; i < kStages - 1; i++) issue(i, false, true);
  }
  float acc[kG][4];
#pragma unroll
  for (int gi = 0; gi < kG; gi++) acc[gi][0] = acc[gi][1] = acc[gi][2] = acc[gi][3] = 0.f;
  for (int ch = 0; ch < chunks; ch++) {
    const int st0 = s0 + ch * kChunkSteps;
    uint32_t B[kChunkSteps][kG][2], SA[kChunkSteps], SB[kChunkSteps][kG];
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++) {
      const int64_t o = int64_t(st0 + u) * 32;
#pragma unroll
      for (int gi = 0; gi < kG; gi++) {
        if constexpr (!kWideB && !kSmemB) {
          B[u][gi][0] = tok_ok[gi] ? ld_cached(xa[gi] + o) : 0u;
          B[u][gi][1] = tok_ok[gi] ? ld_cached(xa[gi] + o + 16) : 0u;
        }
        if constexpr (kFp4) SB[u][gi] = tok_ok[gi] ? ld_cached(xsf[gi] + int64_t(st0 + u) * 512) : 0u;
      }
      if constexpr (kFp4) SA[u] = ld_cached(wsf + int64_t(st0 + u) * 512);
    }
    if constexpr (kWideB && !kSmemB) {
      // A token's chunk is 128 contiguous bytes = 8 pieces (k-step u, half h) of 16 bytes, piece 2u + h.
      // Lane t loads pieces t and 4 + t whole; the quad transpose hands lane t word t of every piece,
      // which is the B fragment (b0 = half 0, b1 = half 1).
#pragma unroll
      for (int gi = 0; gi < kG; gi++) {
        uint4 v[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (tok_ok[gi]) {
          const uint8_t* x = xa[gi] - 4 * t + int64_t(st0) * 32 + 16 * t;
          v[0] = __ldg(reinterpret_cast<const uint4*>(x));
          v[1] = __ldg(reinterpret_cast<const uint4*>(x + 64));
        }
#pragma unroll
        for (int j = 0; j < 2; j++) {
          uint32_t w[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
          quad_transpose(w, t);
#pragma unroll
          for (int i = 0; i < 4; i++) B[2 * j + i / 2][gi][i % 2] = w[i];
        }
      }
    }
    cp_async_wait<kStages - 2>();
    // kSmemB: every thread's pieces of the chunk are in; the slot refilled below was read by all warps in
    // the previous chunk (before this barrier).
    if constexpr (kSmemB)
      __syncthreads();
    else
      __syncwarp();
    const uint32_t base = ring + (ch % kStages) * kChunkBytes;
    if constexpr (kSmemB) {
      // B fragments of the 8 groups: token 8 gi + lane % 8, pieces 4 h + lane / 8 (k-steps 2 h, 2 h + 1).
      const uint32_t abase = aring + (ch % kStages) * kActChunkBytes;
#pragma unroll
      for (int gi = 0; gi < kG; gi++) {
        if (8 * gi >= p.M) break;  // (uniform)
        const int tok = 8 * gi + lane % 8;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          uint32_t r[4];
          ldmatrix_x4(r, abase + tok * 128 + (((4 * h + lane / 8) ^ (tok % 8)) * 16));
          B[2 * h][gi][0] = r[0], B[2 * h][gi][1] = r[1], B[2 * h + 1][gi][0] = r[2], B[2 * h + 1][gi][1] = r[3];
        }
      }
    }
    uint32_t A[kChunkSteps][4];
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++) {
      int c = 2 * u + lm_half;
      ldmatrix_x4(A[u], base + lm_row * 128 + ((c ^ (lm_row % 8)) * 16));
    }
    __syncwarp();  // the slot is refilled below
    issue(ch + kStages - 1);
#pragma unroll
    for (int u = 0; u < kChunkSteps; u++) {
#pragma unroll
      for (int gi = 0; gi < kG; gi++) {
        if (8 * gi >= p.M) break;  // (uniform; groups past M carry zeros)
        if constexpr (kFmt == Fmt::kFp4)
          mma_fp4(acc[gi], A[u], B[u][gi], SA[u], SB[u][gi]);
        else if constexpr (kFmt == Fmt::kFp8)
          mma_fp8(acc[gi], A[u], B[u][gi]);
        else
          mma_bf16(acc[gi], A[u], B[u][gi]);
      }
    }
  }
  cp_async_wait<0>();
  trace_point(2);

  // acc: (row0, token 2t), (row0, 2t + 1), (row0 + 8, 2t), (row0 + 8, 2t + 1).
  auto epilogue = [&](int row, int tok, float v) {
    int64_t i = int64_t(tok) * p.N + row;
    float out = alpha_of(p.alpha, row) * v;
    if (p.wscale) out *= p.wscale[row] * p.ascale[tok];
    if (p.c) out += p.beta * p.c[i];
    p.d[i] = out;
  };
  if (p.splits == 1) {
#pragma unroll
    for (int gi = 0; gi < kG; gi++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int row = row0 + (j / 2) * 8, tok = 8 * gi + 2 * t + j % 2;
        if (tok < p.M) epilogue(row, tok, acc[gi][j]);
      }
    return;
  }
#pragma unroll
  for (int gi = 0; gi < kG; gi++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int row = row0 + (j / 2) * 8, tok = 8 * gi + 2 * t + j % 2;
      if (tok < p.M) p.part[(int64_t(split) * p.M + tok) * p.N + row] = acc[gi][j];
    }
  if (p.ext_reduce) return;
  __threadfence();
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) last = atomicAdd(&p.counters[tile], 1) == p.splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  // The partials of a split batch are loaded together (one L2 round trip per 8 splits instead of one
  // per split: the memory queues are busy with the next GEMM's prefetch), then added in split order.
  if constexpr (kG >= 4) {
    // Batched steps (P6-10, M > 16): 4 outputs per thread at once, their loads in flight together
    // (M = 32 is 16 outputs per thread: one round trip each made the tail 4x as long as M = 8's).
    // The sums are those of the loop below.
    constexpr int kItems = 4;
    const int items = kRowsPerCta * p.M;
    const int64_t stride = int64_t(p.M) * p.N;
    for (int i0 = threadIdx.x; i0 < items; i0 += blockDim.x * kItems) {
      const float* src[kItems];
      float v[kItems];
#pragma unroll
      for (int it = 0; it < kItems; it++) {
        const int i = min(i0 + it * int(blockDim.x), items - 1);
        src[it] = p.part + int64_t(i / kRowsPerCta) * p.N + tile * kRowsPerCta + i % kRowsPerCta;
        v[it] = 0.f;
      }
      for (int sp0 = 0; sp0 < p.splits; sp0 += 8) {
        float x[kItems][8];
#pragma unroll
        for (int it = 0; it < kItems; it++)
#pragma unroll
          for (int u = 0; u < 8; u++) x[it][u] = sp0 + u < p.splits ? __ldcg(src[it] + (sp0 + u) * stride) : 0.f;
#pragma unroll
        for (int it = 0; it < kItems; it++)
#pragma unroll
          for (int u = 0; u < 8; u++)
            if (sp0 + u < p.splits) v[it] += x[it][u];
      }
      // The epilogue with its rounding spelled out: inlined here the compiler contracted it differently
      // from the loop below (one ULP off with a residual), and the rows must match those of M <= 8.
#pragma unroll
      for (int it = 0; it < kItems; it++) {
        const int i = i0 + it * int(blockDim.x);
        if (i >= items) continue;
        const int row = tile * kRowsPerCta + i % kRowsPerCta, tok = i / kRowsPerCta;
        const int64_t o = int64_t(tok) * p.N + row;
        float out = __fmul_rn(alpha_of(p.alpha, row), v[it]);
        if (p.wscale) out = __fmul_rn(out, __fmul_rn(p.wscale[row], p.ascale[tok]));
        if (p.c) out = __fmaf_rn(p.beta, p.c[o], out);
        p.d[o] = out;
      }
    }
  } else {
    for (int i = threadIdx.x; i < kRowsPerCta * p.M; i += blockDim.x) {
      int row = tile * kRowsPerCta + i % kRowsPerCta, tok = i / kRowsPerCta;
      const float* src = p.part + int64_t(tok) * p.N + row;
      const int64_t stride = int64_t(p.M) * p.N;
      float v = 0.f;
      for (int sp0 = 0; sp0 < p.splits; sp0 += 8) {
        float x[8];
#pragma unroll
        for (int u = 0; u < 8; u++) x[u] = sp0 + u < p.splits ? __ldcg(src + (sp0 + u) * stride) : 0.f;
#pragma unroll
        for (int u = 0; u < 8; u++)
          if (sp0 + u < p.splits) v += x[u];
      }
      epilogue(row, tok, v);
    }
  }
  if (threadIdx.x == 0) p.counters[tile] = 0;
  trace_point(3);
}

int g_sms = 0;

// The split-K sum of a wide call (P7-4): thread i sums outputs 4 i .. 4 i + 3 (4 rows of one token) over
// the splits in split order and applies the epilogue as the kG >= 4 path of skinny_kernel does (the same
// roundings), so the rows equal those of a narrower call. The last CTA of a tile took 17 us for M = 64,
// N = 5120 (128 x 64 outputs x 8 splits on 40 CTAs); here every SM takes a share.
__global__ void __launch_bounds__(256) skinny_reduce_kernel(Params p) {
  pdl_trigger();
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n4 = int64_t(p.M) * p.N / 4;
  pdl_wait();
  if (i >= n4) return;
  const int tok = int(i * 4 / p.N), row = int(i * 4 % p.N);
  const int64_t o = int64_t(tok) * p.N + row, stride = int64_t(p.M) * p.N;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sp0 = 0; sp0 < p.splits; sp0 += 8) {
    float4 x[8];
#pragma unroll
    for (int u = 0; u < 8; u++)
      x[u] = sp0 + u < p.splits ? __ldcg(reinterpret_cast<const float4*>(p.part + (sp0 + u) * stride + o))
                                : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (sp0 + u < p.splits) v[0] += x[u].x, v[1] += x[u].y, v[2] += x[u].z, v[3] += x[u].w;
  }
#pragma unroll
  for (int e = 0; e < 4; e++) {
    float out = __fmul_rn(alpha_of(p.alpha, row + e), v[e]);
    if (p.wscale) out = __fmul_rn(out, __fmul_rn(p.wscale[row + e], p.ascale[tok]));
    if (p.c) out = __fmaf_rn(p.beta, p.c[o + e], out);
    p.d[o + e] = out;
  }
}

template <Fmt kFmt>
void launch(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* w, const uint8_t* w_sf,
            const AlphaSegments& alpha, const float* c, float beta, float* d, void* ws, size_t ws_bytes,
            cudaStream_t s, const float* wscale = nullptr, const float* ascale = nullptr) {
  Q27_CHECK(M >= 1 && M <= kMaxTokens, "skinny gemm: M must be 1..64");
  Q27_CHECK(N % kRowsPerCta == 0, "skinny gemm: N must be a multiple of 128");
  constexpr bool kFp4 = kFmt == Fmt::kFp4;
  const int kbytes = kFp4 ? K / 2 : kFmt == Fmt::kFp8 ? K : 2 * K;
  Q27_CHECK(kbytes % 32 == 0, "skinny gemm: K must be a multiple of the k-step");
  if (!g_sms) Q27_CUDA(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  constexpr int kStages = Q27_SKINNY_STAGES;
  constexpr int kSmem = kWarps * kStages * kChunkBytes;
  constexpr int kSmem8 = kSmem + kStages * kActChunkBytes;  // kG = 8: the activations' ring as well
  static bool attr = false;
  if (!attr) {
    Q27_CUDA(cudaFuncSetAttribute(skinny_kernel<kFmt, kStages, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    Q27_CUDA(cudaFuncSetAttribute(skinny_kernel<kFmt, kStages, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    Q27_CUDA(cudaFuncSetAttribute(skinny_kernel<kFmt, kStages, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    Q27_CUDA(cudaFuncSetAttribute(skinny_kernel<kFmt, kStages, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem8));
    Q27_CUDA(cudaFuncSetAttribute(skinny_kernel<kFmt, kStages, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr = true;
  }
  Params p{};
  p.M = M, p.N = N, p.kbytes = kbytes, p.steps = kbytes / 32;
  const int tiles = N / kRowsPerCta;
  Q27_CHECK(p.steps % kChunkSteps == 0, "skinny gemm: K must be a multiple of 4 k-steps");
  // Split K so that there are about `waves` CTAs per SM, with at least 2 chunks per split. The split
  // partials grow with M, so wider calls split less (in-pipeline A/B, 2026-10-05: M = 1 is best at
  // 2 waves rounded up, M = 8 and M = 15 at 2 waves rounded down; one wave left qkvz at 128 CTAs). A
  // token's result then depends on M through the summation order; Q27_SKINNY_EXACT=1 uses the M = 1
  // policy for every M, which makes speculative verify rows bit-identical to decode (the greedy test).
  // (Tunables are read at every launch, see pdl_enabled.)
  const char* e_exact = getenv("Q27_SKINNY_EXACT");
  const bool exact = e_exact && *e_exact && *e_exact != '0';
  const int mclass = exact || M == 1 ? 0 : 1;
  const char* e_waves = getenv("Q27_SKINNY_WAVES");
  const int waves = e_waves ? atoi(e_waves) : 2;
  const int chunks = p.steps / kChunkSteps;
  const char* e_fit = getenv("Q27_SKINNY_FIT");
  const int fit = e_fit ? atoi(e_fit) : mclass > 0;
  const char* e_l2 = getenv("Q27_SKINNY_L2_CHUNKS");
  p.l2_chunks = e_l2 ? atoi(e_l2) : kL2ChunksDefault;
  int want = fit ? waves * g_sms / tiles : int(ceil_div(waves * g_sms, tiles));
  int splits = std::clamp(want, 1, std::max(1, chunks / 2));
  p.steps_per_split = int(ceil_div(chunks, splits)) * kChunkSteps;
  p.splits = splits = int(ceil_div(p.steps, p.steps_per_split));
  // Fixed-size counter area: every shape must see zeroed counters, whatever ran before it.
  constexpr size_t cnt_bytes = kMaxTiles * 4;
  Q27_CHECK(tiles <= kMaxTiles, "skinny gemm: N too large");
  if (splits > 1)
    Q27_CHECK(cnt_bytes + size_t(splits) * M * N * 4 <= ws_bytes, "skinny gemm: workspace too small");
  p.a = a, p.a_sf = a_sf, p.w = w, p.w_sf = w_sf, p.alpha = alpha, p.c = c, p.beta = beta, p.d = d;
  p.wscale = wscale, p.ascale = ascale;
  // Split-K sums in their own kernel above Q27_SKINNY_EXT_REDUCE_M rows (default 1: every verify and batched
  // step; P7-4. At M = 1 the last CTA is faster: decode 11.47 vs 11.74 ms).
  const char* e_ext = getenv("Q27_SKINNY_EXT_REDUCE_M");
  p.ext_reduce = splits > 1 && M > (e_ext ? atoi(e_ext) : 1);
  p.counters = static_cast<int*>(ws);
  p.part = reinterpret_cast<float*>(static_cast<uint8_t*>(ws) + cnt_bytes);
  const dim3 grid(tiles, splits), block(kWarps * 32);
  // M > 16: 16-byte activation loads (kWideB), except for calls of >= 1024 tiles (the lm_head: 250 -> 261 us
  // at M = 32 with them; the other shapes 3-8% faster; P6-11).
  const bool wide = tiles < kWideMaxTiles;
  if (M <= 8)
    launch_pdl(skinny_kernel<kFmt, kStages, 1>, grid, block, kSmem, s, p);
  else if (M <= 16)
    launch_pdl(skinny_kernel<kFmt, kStages, 2>, grid, block, kSmem, s, p);
  else if (M <= 32 && wide)
    launch_pdl(skinny_kernel<kFmt, kStages, 4, true>, grid, block, kSmem, s, p);
  else if (M <= 32)
    launch_pdl(skinny_kernel<kFmt, kStages, 4>, grid, block, kSmem, s, p);
  else  // (kG = 8 reads the activations from shared memory: kWideB is not used)
    launch_pdl(skinny_kernel<kFmt, kStages, 8>, grid, block, kSmem8, s, p);
  if (p.ext_reduce)
    launch_pdl(skinny_reduce_kernel, dim3(unsigned(ceil_div(int64_t(M) * N / 4, 256))), dim3(256), 0, s, p);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace

void skinny_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* w, const uint8_t* w_sf,
                const AlphaSegments& alpha, const float* c, float beta, float* d, void* ws, size_t ws_bytes,
                cudaStream_t s) {
  launch<Fmt::kFp4>(M, N, K, a, a_sf, w, w_sf, alpha, c, beta, d, ws, ws_bytes, s);
}

void skinny_fp8(int M, int N, int K, const uint8_t* a, const uint8_t* w, const AlphaSegments& alpha, const float* c,
                float beta, float* d, void* ws, size_t ws_bytes, cudaStream_t s) {
  launch<Fmt::kFp8>(M, N, K, a, nullptr, w, nullptr, alpha, c, beta, d, ws, ws_bytes, s);
}

void skinny_fp8_scaled(int M, int N, int K, const uint8_t* a, const float* ascale, const uint8_t* w, const float* wscale,
                       float* d, void* ws, size_t ws_bytes, cudaStream_t s) {
  launch<Fmt::kFp8>(M, N, K, a, nullptr, w, nullptr, AlphaSegments::one(1.f), nullptr, 0.f, d, ws, ws_bytes, s, wscale,
                    ascale);
}

void skinny_fp4_scaled(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const float* ascale, const uint8_t* w,
                       const uint8_t* w_sf, const float* wscale, float* d, void* ws, size_t ws_bytes, cudaStream_t s) {
  launch<Fmt::kFp4>(M, N, K, a, a_sf, w, w_sf, AlphaSegments::one(1.f), nullptr, 0.f, d, ws, ws_bytes, s, wscale,
                    ascale);
}

void skinny_bf16(int M, int N, int K, const bf16* a, const bf16* w, float alpha, const float* c, float beta, float* d,
                 void* ws, size_t ws_bytes, cudaStream_t s) {
  launch<Fmt::kBf16>(M, N, K, reinterpret_cast<const uint8_t*>(a), nullptr, reinterpret_cast<const uint8_t*>(w),
                     nullptr, AlphaSegments::one(alpha), c, beta, d, ws, ws_bytes, s);
}

}  // namespace q27
