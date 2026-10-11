// Warp-level tensor core helpers (mma.sync m16n8k16 BF16 / f16 and m16n8k32 E4M3, ldmatrix, cp.async) shared by the
// attention and GDN kernels. Fragment layout (lane l, r0 = l / 4, c2 = 2 * (l % 4)): A a[0] = row r0,
// cols c2..c2+1, a[1] = row r0 + 8, a[2] / a[3] the same at cols + 8; B b[0] = k c2..c2+1 of col r0,
// b[1] = k + 8; C c[0..1] = row r0, cols c2..c2+1, c[2..3] = row r0 + 8. The C fragments of two
// adjacent n8 tiles are the A fragment of one k16 tile.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace q27 {
namespace {

__device__ __forceinline__ void cp_async16z(uint32_t smem, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t* r, uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ void mma_bf16_16816(float* acc, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// FP8 m16n8k32: A a[0] = row r0, k 4 * (l % 4) .. + 3 (byte i = k + i), a[1] = row r0 + 8, a[2] / a[3] the
// same at k + 16; B b[0] = k 4 * (l % 4) .. + 3 of col r0, b[1] = k + 16; C as above.
// Issued as the block-scaled kind::mxf8f6f4 with ue8m0 scales of 1.0: on the RTX 5090 the plain form
// accumulates FP32 at half rate (micro/mma_peak: 507 vs 1,007 TFLOPS) and the results are bit-identical
// (micro/mma_fp8_blockscale_eq). The trick is from NInfer (src/ops/common/mma.cuh). -DQ27_FP8_PLAIN_MMA: plain.
__device__ __forceinline__ void mma_e4m3_16832(float* acc, const uint32_t* a, const uint32_t* b) {
#ifdef Q27_FP8_PLAIN_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%10}, {%11, %12};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x7fu), "h"(uint16_t(0)),
        "h"(uint16_t(0)));
#endif
}
// NVFP4 m16n8k64 (kind::mxf4nvf4, ue4m3 scale per 16 k): A a[0] = row r0, k 8 * (l % 4) .. + 7 (low nibble
// first), a[1] = row r0 + 8, a[2] / a[3] the same at k + 32; B b[0] = k 8 * (l % 4) .. + 7 of col r0, b[1] = k + 32.
// Scales (thread-id 0): sfa = the 4 block scales of row r0 in lanes l % 4 == 0 and of row r0 + 8 in l % 4 == 1,
// sfb = those of col r0 in l % 4 == 0 (byte j = k 16 j .. + 15). As in skinny.cu.
__device__ __forceinline__ void mma_fp4_16864(float* acc, const uint32_t* a, const uint32_t* b, uint32_t sfa,
                                              uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%11, %12};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(uint16_t(0)),
        "h"(uint16_t(0)), "r"(sfb));
}
// Four floats to E4M3 (round to nearest, saturating), a in byte 0.
__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n}"
      : "=r"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// E4M3 of 2^a, 2^b (a in byte 0) through an f16x2 exp2: enough for the 3-bit mantissa of the result.
__device__ __forceinline__ uint32_t exp2_e4m3x2(float a, float b) {
  uint32_t h;
  uint16_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(b), "f"(a));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h));
  return r;
}
// Sum of the four E4M3 values of x.
__device__ __forceinline__ float sum_e4m3x4(uint32_t x) {
  uint32_t lo, hi;
  asm("{\n.reg .b16 a, b;\n"
      "mov.b32 {a, b}, %2;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi)
      : "r"(x));
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 g = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  return (f.x + f.y) + (g.x + g.y);
}
__device__ __forceinline__ void mma_f16_16816(float* acc, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi);
// BF16 or (kHalf) f16 operands of the m16n8k16 MMAs: pack, MMA, and the two rounded values of a pack.
template <bool kHalf>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  return kHalf ? pack_f16(lo, hi) : pack_bf16(lo, hi);
}
template <bool kHalf>
__device__ __forceinline__ void mma16816(float* acc, const uint32_t* a, const uint32_t* b) {
  if constexpr (kHalf) mma_f16_16816(acc, a, b); else mma_bf16_16816(acc, a, b);
}
template <bool kHalf>
__device__ __forceinline__ float2 unpack2(uint32_t x) {
  if constexpr (kHalf) return __half22float2(*reinterpret_cast<const __half2*>(&x));
  else return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
}  // namespace
}  // namespace q27
