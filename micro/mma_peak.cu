// Sustained tensor-core throughput of the SM120 warp-level MMAs the runtime uses, under the
// card's power limit (run long enough for clocks to settle; watch `nvidia-smi dmon` alongside).
//
//   bf16   mma.sync m16n8k16  bf16 x bf16 -> f32              (FA-style BF16 attention, fallback GEMMs)
//   fp8    mma.sync m16n8k32  kind::f8f6f4 e4m3 x e4m3 -> f32 (GDN / attention projections, FP8 attention)
//   mxfp8  mma.sync m16n8k32  kind::mxf8f6f4 block_scale 1X e4m3 x e4m3, ue8m0 scales of 1 -> f32
//          (same products as fp8; NInfer's note: the block-scaled form avoids the half-rate FP32 accumulation)
//   nvfp4  mma.sync m16n8k64  kind::mxf4nvf4 block_scale 4X e2m1 x e2m1, ue4m3 scales -> f32 (MLP, lm_head)
//
// Register-only loops with independent accumulators, so this is the instruction ceiling, not a GEMM.
// usage: mma_peak [--seconds 3]

#include "common.cuh"

#include <cstdint>
#include <cstring>

constexpr int kAcc = 8;  // independent accumulator sets per warp (ILP)

template <int kKind>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sf) {
  if constexpr (kKind == 0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (kKind == 1) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (kKind == 3) {
    const uint16_t zero = 0;
    const uint32_t unit = 0x7f;  // ue8m0 1.0
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(unit), "h"(zero), "h"(zero));
  } else {
    const uint16_t zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(zero), "h"(zero));
  }
}

template <int kKind>
__global__ void __launch_bounds__(256) mma_loop(uint32_t seed, long iters, float* out) {
  // Operand bits encode 1.0 (bf16 0x3f80, e4m3 0x38, e2m1 0x2) so nothing overflows to inf/nan.
  const uint32_t one = kKind == 0 ? 0x3f803f80u : (kKind == 1 || kKind == 3) ? 0x38383838u : 0x22222222u;
  uint32_t a[4] = {one, one ^ (seed & 1), one, one};
  uint32_t b[2] = {one, one};
  const uint32_t sf = 0x38383838u;  // four ue4m3 scales of 1.0
  float d[kAcc][4] = {};
  for (long i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < kAcc; ++k) mma<kKind>(d[k], a, b, sf);
  }
  float s = 0;
#pragma unroll
  for (int k = 0; k < kAcc; ++k) s += d[k][0] + d[k][1] + d[k][2] + d[k][3];
  if (s == 1234.5f) out[threadIdx.x] = s;
}

template <int kKind>
static void run(const char* name, double flops_per_mma, double seconds) {
  const int sms = sm_count();
  const int blocks = sms * 4, threads = 256;  // 32 warps per SM
  float* out = nullptr;
  CHECK(cudaMalloc(&out, threads * sizeof(float)));
  Timer t;
  // Calibrate the iteration count to ~0.2 s, then run for `seconds`.
  long iters = 1 << 12;
  t.begin();
  mma_loop<kKind><<<blocks, threads>>>(1, iters, out);
  CHECK(cudaGetLastError());
  double ms = t.end();
  iters = long(iters * 200.0 / ms);
  const int reps = int(seconds / 0.2 + 0.5);
  t.begin();
  for (int r = 0; r < reps; ++r) mma_loop<kKind><<<blocks, threads>>>(r, iters, out);
  ms = t.end();
  const double mmas = double(blocks) * (threads / 32) * kAcc * iters * reps;
  std::printf("%-6s %8.1f TFLOPS  (%.2f s)\n", name, mmas * flops_per_mma / (ms * 1e-3) / 1e12, ms / 1e3);
  CHECK(cudaFree(out));
}

int main(int argc, char** argv) {
  double seconds = 3;
  for (int a = 1; a < argc; ++a)
    if (!std::strcmp(argv[a], "--seconds") && a + 1 < argc) seconds = std::atof(argv[++a]);
  print_device();
  run<0>("bf16", 2.0 * 16 * 8 * 16, seconds);
  run<1>("fp8", 2.0 * 16 * 8 * 32, seconds);
  run<3>("mxfp8", 2.0 * 16 * 8 * 32, seconds);
  run<2>("nvfp4", 2.0 * 16 * 8 * 64, seconds);
  return 0;
}
