// Prefill GEMMs on warp-level mma.sync (no CUTLASS): the same contract as gemm.cu.
// A = activations [M][K] row-major, B = weights [N][K] (K contiguous), D fp32 or BF16 row-major [M][N].
// NVFP4: mma m16n8k64 kind::mxf4nvf4 with the ue4m3 scales in the CUTLASS swizzled layout (sf_offset),
// read in place: for one row and one k64 step the four block scales are one 32-bit word of a 512-byte
// atom (128 rows), and the atoms of consecutive k64 steps are contiguous.
// A CTA owns a 128 x 128 tile of D: 8 MMA warps (2 x 4, 64 x 32 each: 4 m16 x 4 n8 MMA tiles) and one
// producer warp. K moves in steps of 128 elements through a ring of kStages stages: the producer fills a
// stage with TMA (A and B tiles: 64 bytes per row, 64-byte swizzle = 16-byte piece c of row r at
// c ^ ((r / 2) % 4), so that ldmatrix is conflict-free; rows past M read as zeros) and two bulk copies
// (the scale atoms of the two k64 steps), signalled on a "full" mbarrier; the MMA warps release it on
// an "empty" one. (A first version with cp.async from all threads and __syncthreads per stage ran at
// 62% tensor-pipe activity, stalled on the MIO queue and the barrier.)
#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <cstdlib>

#include "common.h"
#include "kernels.h"
#include "mma.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace q27 {
namespace {

constexpr int kBM = 128, kBN = 128;
constexpr int kBK = 128;              // elements per stage (two k64 MMA steps)
constexpr int kRowBytes = kBK / 2;    // 64
constexpr int kMmaWarps = 8;
// (+ the producer warp and, for kOutGlu, a store warp: registers are allocated per 4 warps, so the 10th
// warp costs none)
constexpr int kThreads = 32 * (kMmaWarps + 2);
constexpr int kRingStages = 3;   // (3 to 5 measure the same; 2 is 36% slower)
constexpr int kEpiBytes = 4096;  // per MMA warp: the output staging buffer of the TMA stores
constexpr int kGluEpiBytes = 128 * 32 + 512;  // kOutGlu: per tile (two of them in the CTA's staging space)
constexpr int kTileBytes = kBM * kRowBytes;    // 8 KiB (A; B the same)
constexpr int kSfBytes = 2 * 512;              // two scale atoms per operand
// kOut: the epilogue. kGlu (the MLP gate | up, stored interleaved: kernels.h glu_row): the B tile of 128 rows
// is 64 gate rows and the same 64 up rows, with one scale atom as any tile; the epilogue writes silu(gate) * up
// NVFP4-quantized (the next GEMM's activations). (With gate and up stored apart, the tile took half of a gate
// and half of an up atom: 50% more B-scale bytes per stage, ~3% of the GEMM time.)
enum { kOutF32, kOutBf16, kOutGlu };
constexpr int kStageBytes = 2 * kTileBytes + 2 * kSfBytes;

struct Fp4Params {
  int M, N, K;
  const uint8_t* a_sf;
  const uint8_t* b_sf;
  float alpha, beta;
  const float* c;
  float* d;
  bf16* d16;
  int epi;  // kEpiStore, kEpiAdd (D += alpha A.B^T by TMA reduce: C is D and beta 1), kEpiLoadC
  // kOutGlu: N = the intermediate size I (B = gate | up, [2 I][K]); alpha is the gate's, alpha_up the up's;
  // the output is q [M][I / 2] and its swizzled scales sf, quantized with the global scale out_scale.
  float alpha_up, out_scale;
  uint8_t* q;
  uint8_t* sf;
};
enum { kEpiStore, kEpiAdd, kEpiLoadC };

// Shared-memory offset of 16-byte piece `c` (0..3) of tile row `r` (the TMA 64-byte swizzle).
__device__ __forceinline__ uint32_t tile_off(int r, int c) { return r * kRowBytes + ((c ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  while (!done)
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(map), "r"(x), "r"(y), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void bulk_copy(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// mma_fp4_16864 with the B scales taken from lane (l / 4) * 4 + tid_b of each quad.
__device__ __forceinline__ void mma_fp4_sel(float* acc, const uint32_t* a, const uint32_t* b, uint32_t sfa,
                                            uint32_t sfb, uint16_t tid_b) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %11}, {%12}, {%11, %13};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(uint16_t(0)), "r"(sfb),
        "h"(tid_b));
}

// The operands of one k64 step of an MMA warp.
struct Frags {
  uint32_t a[4][4], b[4][2], sa[4], sb[4];
};

// Per-thread shared-memory offsets of the fragments of k64 step 0 (stage-relative; step 1: see load_frags).
// ldmatrix: A x4 of m16 tile mt: matrix q = lane / 8 is rows (q % 2) * 8 .. + 7, piece 2 ks + q / 2; B x4
// of n8 tiles 2 np, 2 np + 1: rows (q / 2) * 8 .. + 7, piece 2 ks + q % 2. Rows 16 apart share the
// swizzle, so the other m16 tiles (32 rows apart) are 2 KiB further, the other n16 tiles 1 KiB, and piece
// 2 + c is piece c with the 32-byte bit flipped. A scales: a 16-byte line of the atom holds rows r, r + 32,
// r + 64, r + 96, which are the warp's four m16 tiles: one 128-bit load of line r0 (t == 0, 2) or r0 + 8
// (t == 1, 3) gives all four, and the warp's 16 lines are 256 contiguous bytes (no bank conflicts; with
// 64 contiguous rows per warp it took two 64-bit loads at twice the shared-memory wavefronts). B scales: ldmatrix
// over the lines nt * 8 .. + 7 of the atom gives lane (g, t) word t of line nt * 8 + g; the column's word
// is kWc = wn / 32, so the MMA takes it from lane t = kWc.
struct FragOffsets {
  uint32_t a, b, sa, sb;
  __device__ FragOffsets(int wm, int wn, int lane) {
    const int q = lane / 8, lr = lane % 8, g = lane / 4, t = lane % 4;
    a = tile_off(wm + (q % 2) * 8 + lr, q / 2);  // (wm: 0 or 16; m16 tile mt at rows wm + 32 mt)
    b = kTileBytes + tile_off(wn + (q / 2) * 8 + lr, q % 2);
    sa = 2 * kTileBytes + (wm + g + (t % 2) * 8) * 16;
    sb = 2 * kTileBytes + kSfBytes + (q * 8 + lr) * 16;
  }
};

template <int kKs, bool kGlu = false>
__device__ __forceinline__ void load_frags(Frags& f, uint32_t st, const FragOffsets& o) {
  const uint32_t a = st + (o.a ^ (kKs * 32)), b = st + (o.b ^ (kKs * 32));
#pragma unroll
  for (int mt = 0; mt < 4; mt++) ldsm_x4(f.a[mt], a + mt * 2048);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(f.sa[0]), "=r"(f.sa[1]), "=r"(f.sa[2]), "=r"(f.sa[3])
               : "r"(st + o.sa + kKs * 512));
#pragma unroll
  for (int np = 0; np < 2; np++) {
    uint32_t x[4];
    ldsm_x4(x, b + np * (kGlu ? 4096 : 1024));  // (kGlu: gate rows, then the up rows 64 further)
    f.b[2 * np][0] = x[0], f.b[2 * np][1] = x[1], f.b[2 * np + 1][0] = x[2], f.b[2 * np + 1][1] = x[3];
  }
  ldsm_x4(f.sb, st + o.sb + kKs * 512);
}

// Two e2m1 codes in one byte, `even` in the low nibble (as kernels.cu).
__device__ __forceinline__ uint8_t cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(odd), "f"(even));
  return uint8_t(r);
}

template <int kWc, bool kGlu = false>
__device__ __forceinline__ void mma_step(float (&acc)[4][4][4], const Frags& f) {
  // (n outer: consecutive MMAs share the B operand and its scales, which the operand reuse cache keeps)
  // kGlu: gate n8 tile j (nt = j: rows 16 kWc + 8 j of the tile) is in lines (2 kWc + j) % 4 * 8 .. + 7 of the
  // atom, word kWc / 2; up tile j (nt = 2 + j: 64 rows further) in the same lines, word 2 + kWc / 2.
#pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    const int line = kGlu ? (2 * kWc + nt % 2) % 4 : nt;
    const uint16_t sel = kGlu ? (nt < 2 ? kWc / 2 : 2 + kWc / 2) : kWc;
#pragma unroll
    for (int mt = 0; mt < 4; mt++) mma_fp4_sel(acc[mt][nt], f.a[mt], f.b[nt], f.sa[mt], f.sb[line], sel);
  }
}

// The K loop of an MMA warp for one tile (rows wm + 32 mt .. + 15, columns kWc * 32 .. + 31 of the CTA tile); the
// ring position (stage s, its phase) carries over to the next tile. The fragments of the next k64 step
// (the next stage's first one at the end of a stage) load before the MMAs of this one.
template <int kStages, int kWc, int kOut>
__device__ __forceinline__ void mainloop(float (&acc)[4][4][4], uint32_t base, uint32_t full, uint32_t empty,
                                         int ktiles, FragOffsets o, int lane, int& s, uint32_t& phase) {
  constexpr bool kGlu = kOut == kOutGlu;
  Frags f0, f1;
  mbar_wait(full + 8 * s, phase);
  load_frags<0, kGlu>(f0, base + s * kStageBytes, o);
  for (int kt = 0; kt < ktiles; kt++) {
    // (opaque: otherwise the compiler recomputes the offsets from the thread index in every iteration,
    // ~45 integer instructions per 32 MMAs)
    asm volatile("" : "+r"(o.a), "+r"(o.b), "+r"(o.sa), "+r"(o.sb), "+r"(base), "+r"(full), "+r"(empty));
    const uint32_t st = base + s * kStageBytes;
    load_frags<1, kGlu>(f1, st, o);
    mma_step<kWc, kGlu>(acc, f0);
    const int s1 = s + 1 == kStages ? 0 : s + 1;
    const uint32_t phase1 = s1 == 0 ? phase ^ 1 : phase;
    if (kt + 1 < ktiles) {
      mbar_wait(full + 8 * s1, phase1);
      load_frags<0, kGlu>(f0, base + s1 * kStageBytes, o);
    }
    mma_step<kWc, kGlu>(acc, f1);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * s);
    s = s1, phase = phase1;
  }
}

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(map), "r"(x), "r"(y),
               "r"(src)
               : "memory");
}
__device__ __forceinline__ void tma_add_2d(const CUtensorMap* map, uint32_t src, int x, int y) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%1, %2}], [%3];" ::"l"(map),
               "r"(x), "r"(y), "r"(src)
               : "memory");
}
__device__ __forceinline__ void st_shared_u32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_shared_u8(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u8 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_shared_f32x2(uint32_t a, float x, float y) {
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(a), "f"(x), "f"(y) : "memory");
}

// Before a warp rewrites its staging buffer: the previous TMA store has read it.
__device__ __forceinline__ void epi_wait(int lane) {
  if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  __syncwarp();
}
// After the warp wrote its staging buffer: make the writes visible to the TMA unit and store (or add) it.
// kBoxes boxes of 16 rows, box i (box_bytes each in the buffer) at rows y + 32 i (the warp's m16 tiles).
template <int kBoxes>
__device__ __forceinline__ void epi_issue(const CUtensorMap& map, uint32_t buf, int box_bytes, int x, int y, bool add,
                                          int lane) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < kBoxes; i++) {
      if (add)
        tma_add_2d(&map, buf + i * box_bytes, x, y + 32 * i);
      else
        tma_store_2d(&map, buf + i * box_bytes, x, y + 32 * i);
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
}

// An MMA warp: its part of every tile of the CTA, then the epilogue (c[0..1] = row r0, columns 2t, 2t + 1
// of an n8 tile; c[2..3] = row r0 + 8), while the producer already fills the ring for the next tile.
template <int kStages, int kWc, int kOut>
__device__ __forceinline__ void consumer(const Fp4Params& p, const CUtensorMap& map_d, uint32_t base, uint32_t full,
                                         uint32_t empty, uint32_t epi, uint32_t epi_bar, int mtiles, int ntiles, int wm,
                                         int lane) {
  constexpr bool kBf16 = kOut == kOutBf16, kGlu = kOut == kOutGlu;
  constexpr int wn = kGlu ? kWc * 16 : kWc * 32;  // (kGlu: gate rows wn .. + 15, up rows 64 further)
  const FragOffsets o(wm, wn, lane);
  const int g = lane / 4, t = lane % 4, ktiles = p.K / kBK;
  int s = 0;
  uint32_t phase = 0;
  for (int tile = blockIdx.x, it = 0; tile < ntiles; tile += gridDim.x, it++) {
    const int m0 = (tile % mtiles) * kBM, n0 = (tile / mtiles) * (kGlu ? kBN / 2 : kBN);
    float acc[4][4][4] = {};
    mainloop<kStages, kWc, kOut>(acc, base, full, empty, ktiles, o, lane, s, phase);
    if constexpr (kGlu) {
      // silu(gate) * up of row r0 (+ 8), columns n0 + wn + 8 j + 2t, + 1 (j = 0, 1: one 16-element NVFP4
      // block over the quad), quantized as quant_block16 does (gate and up in fp32: not rounded to BF16).
      // Fast divisions: the IEEE ones went through a slow-path call each (~1,600 instructions per warp and
      // tile in this epilogue, the fused GEMM 18% slower than two plain ones).
      // The CTA's output is 128 rows x 32 bytes of q and exactly one 512-byte scale atom (rows m0 .. + 127,
      // blocks n0 / 16 .. + 3): the MMA warps write both into a staging buffer (two, alternating by tile),
      // and the store warp stores them with a TMA store and a bulk copy (handed over on mbarriers: the MMA
      // warps do not wait for each other; a CTA barrier and one MMA thread's stores measured the same).
      // (Each quad's 8-byte and 1-byte global stores to q and the scales took ~5% of the GEMM time.)
      const uint32_t qb = epi + (it & 1) * kGluEpiBytes, sb = qb + kBM * 32;
      mbar_wait(epi_bar + 16 + 8 * (it & 1), ((it >> 1) & 1) ^ 1);  // epi empty: the store of tile it - 2 read it
      const float gs = 1.f / p.out_scale;
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float v[4];
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
              const float gt = p.alpha * acc[mt][j][2 * h + e], up = p.alpha_up * acc[mt][2 + j][2 * h + e];
              v[2 * j + e] = __fdividef(gt, 1.f + __expf(-gt)) * up;
            }
          float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
          const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
          const float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
          const float inv = sff != 0.f ? __fdividef(gs, sff) : 0.f;
          // byte t of the block's 8 = columns 2t, 2t + 1 (j = 0); byte 4 + t for j = 1. Staged rows are 32
          // bytes with the TMA 32-byte swizzle (16-byte half c of row r at c ^ ((r / 4) % 2): the quad rows
          // g and g + 4 in different banks).
          const int r = wm + mt * 32 + g + 8 * h;
          const uint32_t row = qb + r * 32 + (((kWc / 2) ^ ((r >> 2) & 1)) << 4) + (kWc % 2) * 8;
          st_shared_u8(row + t, cvt_e2m1x2(v[0] * inv, v[1] * inv));
          st_shared_u8(row + 4 + t, cvt_e2m1x2(v[2] * inv, v[3] * inv));
          // the atom: row r at line r % 32, word r / 32 (= mt), byte = the block (kWc)
          if (t == 0) st_shared_u8(sb + (r % 32) * 16 + mt * 4 + kWc, sfq);
        }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) mbar_arrive(epi_bar + 8 * (it & 1));  // epi full (the store warp waits on it)
      continue;
    }
    // Epilogue as CUTLASS's: the warp writes its tile into its staging buffer (TMA swizzle: conflict-free)
    // and one lane hands it to a TMA store, or for the in-place residual a TMA reduce-add (no read of C
    // by the SM). The store reads the buffer asynchronously; the next tile waits for that before writing.
    // (Plain 4 / 8 / 16-byte stores from every lane took 11-15% of the GEMM time.)
    if constexpr (kBf16) {
      // 4 boxes (m16 tiles) of 16 rows x 32 columns of BF16: 64-byte rows, 16-byte piece c of row r at
      // c ^ ((r / 2) % 4).
      epi_wait(lane);
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int r = mt * 16 + g + 8 * h;
#pragma unroll
          for (int nt = 0; nt < 4; nt++) {
            const __nv_bfloat162 v =
                __floats2bfloat162_rn(p.alpha * acc[mt][nt][2 * h], p.alpha * acc[mt][nt][2 * h + 1]);
            st_shared_u32(epi + r * 64 + ((nt ^ ((r >> 1) & 3)) << 4) + 4 * t, *reinterpret_cast<const uint32_t*>(&v));
          }
        }
      epi_issue<4>(map_d, epi, 1024, n0 + wn, m0 + wm, false, lane);
    } else {
      // Two halves of 2 boxes (m16 tiles) of 16 rows x 32 columns of fp32: 128-byte rows, 16-byte piece c of
      // row r at c ^ (r % 8).
#pragma unroll
      for (int half = 0; half < 2; half++) {
        epi_wait(lane);
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int mt = 2 * half + mi, r = mi * 16 + g + 8 * h;
            const int row = m0 + wm + mt * 32 + g + 8 * h;
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
              float x = p.alpha * acc[mt][nt][2 * h], y = p.alpha * acc[mt][nt][2 * h + 1];
              if (p.epi == kEpiLoadC && row < p.M) {
                const float2 c = *reinterpret_cast<const float2*>(p.c + int64_t(row) * p.N + n0 + wn + nt * 8 + 2 * t);
                x += p.beta * c.x, y += p.beta * c.y;
              }
              st_shared_f32x2(epi + r * 128 + (((2 * nt + t / 2) ^ (r & 7)) << 4) + 8 * (t % 2), x, y);
            }
          }
        epi_issue<2>(map_d, epi, 2048, n0 + wn, m0 + wm + 64 * half, p.epi == kEpiAdd, lane);
      }
    }
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Persistent: CTA i takes tiles i, i + gridDim.x, ... (M-tile fastest: the CTAs running at the same time
// share the weight tiles in L2).
template <int kStages, int kOut>
__global__ void __launch_bounds__(kThreads) gemm_fp4_kernel(const __grid_constant__ CUtensorMap map_a,
                                                            const __grid_constant__ CUtensorMap map_b,
                                                            const __grid_constant__ CUtensorMap map_d, Fp4Params p) {
  extern __shared__ uint8_t smem_raw[];
  // (1 KiB alignment: the swizzle pattern follows the address bits)
  const uint32_t base = (uint32_t(__cvta_generic_to_shared(smem_raw)) + 1023) & ~1023u;
  const uint32_t epi = base + kStages * kStageBytes, full = epi + kMmaWarps * kEpiBytes, empty = full + 8 * kStages;
  const uint32_t epi_bar = empty + 8 * kStages;  // kOutGlu: epi full [2] (the MMA warps arrive), epi empty [2]
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  constexpr bool kGlu = kOut == kOutGlu;
  const int mtiles = (p.M + kBM - 1) / kBM, ntiles = mtiles * (p.N / (kGlu ? kBN / 2 : kBN));
  const int ktiles = p.K / kBK, steps = p.K / 64;

  if (tid == 0) {
    for (int s = 0; s < kStages; s++) mbar_init(full + 8 * s, 1), mbar_init(empty + 8 * s, kMmaWarps);
    for (int b = 0; b < 2; b++) mbar_init(epi_bar + 8 * b, kMmaWarps), mbar_init(epi_bar + 16 + 8 * b, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == kMmaWarps + 1) {  // kOutGlu: the store warp
    if (kGlu && lane == 0) {
      for (int tile = blockIdx.x, it = 0; tile < ntiles; tile += gridDim.x, it++) {
        const int m0 = (tile % mtiles) * kBM, n0 = (tile / mtiles) * (kBN / 2), b = it & 1;
        const uint32_t qb = epi + b * kGluEpiBytes;
        mbar_wait(epi_bar + 8 * b, (it >> 1) & 1);
        tma_store_2d(&map_d, qb, n0 / 2, m0);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 512;" ::"l"(
                         p.sf + sf_offset(m0, n0 / 16, p.N / 16)),
                     "r"(qb + kBM * 32)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        mbar_arrive(epi_bar + 16 + 8 * b);
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    return;
  }
  if (warp == kMmaWarps) {  // producer
    if (lane == 0) {
      int s = 0;
      uint32_t phase = 0;
      for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int mi = tile % mtiles, ni = tile / mtiles;
        const uint8_t* sfa = p.a_sf + int64_t(mi) * steps * 512;
        const uint8_t* sfb = p.b_sf + int64_t(ni) * steps * 512;
        for (int kt = 0; kt < ktiles; kt++) {
          mbar_wait(empty + 8 * s, phase ^ 1);
          const uint32_t st = base + s * kStageBytes, bar = full + 8 * s;
          mbar_expect_tx(bar, kStageBytes);
          tma_2d(st, &map_a, kt * kRowBytes, mi * kBM, bar);
          bulk_copy(st + 2 * kTileBytes, sfa + kt * kSfBytes, kSfBytes, bar);
          tma_2d(st + kTileBytes, &map_b, kt * kRowBytes, ni * kBN, bar);  // (kGlu: rows 128 ni = gate | up)
          bulk_copy(st + 2 * kTileBytes + kSfBytes, sfb + kt * kSfBytes, kSfBytes, bar);
          if (++s == kStages) s = 0, phase ^= 1;
        }
      }
    }
    return;
  }

  const int wm = (warp / 4) * 16;  // (m16 tiles at wm + 32 mt: see FragOffsets)
  // (the B scale selector is an immediate: a warp-varying value makes the compiler loop over its values)
  switch (warp % 4) {
    case 0: consumer<kStages, 0, kOut>(p, map_d, base, full, empty, epi + (kGlu ? 0 : warp * kEpiBytes), epi_bar, mtiles, ntiles, wm, lane); break;
    case 1: consumer<kStages, 1, kOut>(p, map_d, base, full, empty, epi + (kGlu ? 0 : warp * kEpiBytes), epi_bar, mtiles, ntiles, wm, lane); break;
    case 2: consumer<kStages, 2, kOut>(p, map_d, base, full, empty, epi + (kGlu ? 0 : warp * kEpiBytes), epi_bar, mtiles, ntiles, wm, lane); break;
    default: consumer<kStages, 3, kOut>(p, map_d, base, full, empty, epi + (kGlu ? 0 : warp * kEpiBytes), epi_bar, mtiles, ntiles, wm, lane); break;
  }
}

// A 2D TMA descriptor: rows x cols elements of `bytes` each, row-major, boxes of box_rows x box_cols.
CUtensorMap map_2d(const void* ptr, CUtensorMapDataType type, int bytes, int rows, int cols, int box_rows, int box_cols,
                   CUtensorMapSwizzle swizzle) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    Q27_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q));
    Q27_CHECK(f && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(f);
  }();
  CUtensorMap m;
  const cuuint64_t dim[2] = {cuuint64_t(cols), cuuint64_t(rows)}, stride[1] = {cuuint64_t(cols) * bytes};
  const cuuint32_t box[2] = {cuuint32_t(box_cols), cuuint32_t(box_rows)}, estride[2] = {1, 1};
  const CUresult r = encode(&m, type, 2, const_cast<void*>(ptr), dim, stride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  Q27_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return m;
}

// Operand tiles: [rows][K / 2] bytes, boxes of box_rows x 64 bytes with the 64-byte swizzle.
CUtensorMap tile_map(const uint8_t* ptr, int rows, int K, int box_rows) {
  return map_2d(ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, K / 2, box_rows, kRowBytes, CU_TENSOR_MAP_SWIZZLE_64B);
}

template <int kStages, int kOut>
void launch_fp4(const CUtensorMap& ma, const CUtensorMap& mb, const CUtensorMap& md, const Fp4Params& p, cudaStream_t s) {
  constexpr int smem = kStages * (kStageBytes + 16) + 32 + kMmaWarps * kEpiBytes + 1024;
  static bool init = false;
  if (!init) {
    Q27_CUDA(cudaFuncSetAttribute(gemm_fp4_kernel<kStages, kOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    init = true;
  }
  static int sms = [] {
    int dev = 0, n = 0;
    Q27_CUDA(cudaGetDevice(&dev));
    Q27_CUDA(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
  }();
  const int tiles = (p.M + kBM - 1) / kBM * (p.N / (kOut == kOutGlu ? kBN / 2 : kBN));
  const int grid = std::min(tiles, sms);
  gemm_fp4_kernel<kStages, kOut><<<grid, kThreads, smem, s>>>(ma, mb, md, p);
  Q27_CUDA(cudaGetLastError());
}

template <int kOut>
void run_fp4(const uint8_t* a, const uint8_t* b, const Fp4Params& p, cudaStream_t s) {
  if (p.M == 0) return;
  constexpr bool kBf16 = kOut == kOutBf16, kGlu = kOut == kOutGlu;
  Q27_CHECK(p.N % kBN == 0 && p.K % kBK == 0, "gemm_fp4_mma: N and K must be multiples of 128");
  // (kGlu: B is the interleaved [2 N][K])
  const CUtensorMap ma = tile_map(a, p.M, p.K, kBM), mb = tile_map(b, kGlu ? 2 * p.N : p.N, p.K, kBN);
  // Output: boxes of 16 rows (one m16 tile) x 32 columns: BF16 64-byte rows, fp32 128-byte rows.
  const CUtensorMap md = kGlu ? map_2d(p.q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, p.M, p.N / 2, kBM, 32, CU_TENSOR_MAP_SWIZZLE_32B)
                       : kBf16 ? map_2d(p.d16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p.M, p.N, 16, 32, CU_TENSOR_MAP_SWIZZLE_64B)
                               : map_2d(p.d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, p.M, p.N, 16, 32, CU_TENSOR_MAP_SWIZZLE_128B);
  Fp4Params q = p;
  q.epi = !p.c || p.beta == 0.f ? kEpiStore : p.c == p.d && p.beta == 1.f ? kEpiAdd : kEpiLoadC;
  launch_fp4<kRingStages, kOut>(ma, mb, md, q, s);
}

}  // namespace

void gemm_fp4_mma(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                  float alpha, const float* c, float beta, float* d, cudaStream_t s) {
  run_fp4<kOutF32>(a, b, {M, N, K, a_sf, b_sf, alpha, beta, c, d, nullptr}, s);
}

void gemm_fp4_bf16_mma(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b,
                       const uint8_t* b_sf, float alpha, bf16* d, cudaStream_t s) {
  run_fp4<kOutBf16>(a, b, {M, N, K, a_sf, b_sf, alpha, 0.f, nullptr, nullptr, d}, s);
}

void gemm_fp4_glu_mma(int M, int I, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* gate_up,
                      const uint8_t* gate_up_sf, float alpha_gate, float alpha_up, float out_scale, uint8_t* q,
                      uint8_t* q_sf, cudaStream_t s) {
  Fp4Params p{M, I, K, a_sf, gate_up_sf, alpha_gate, 0.f, nullptr, nullptr, nullptr};
  p.alpha_up = alpha_up, p.out_scale = out_scale, p.q = q, p.sf = q_sf;
  run_fp4<kOutGlu>(a, gate_up, p, s);
}

}  // namespace q27
