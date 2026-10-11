// Is mma.sync kind::mxf8f6f4 block_scale (ue8m0 scales of 1.0) bit-identical to the plain
// mma.sync m16n8k32 E4M3 -> f32? The block-scaled form runs at twice the rate on the RTX 5090
// (micro/mma_peak), so the FP8 GEMMs and the FP8 prefill attention could swap it in.
// Random E4M3 A / B (no NaN) and FP32 C over a wide range of magnitudes; counts unequal results.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
__global__ void k(const uint32_t* A, const uint32_t* B, const float* C, float* D0, float* D1) {
  const int l = threadIdx.x;
  uint32_t a[4] = {A[l * 4], A[l * 4 + 1], A[l * 4 + 2], A[l * 4 + 3]}, b[2] = {B[l * 2], B[l * 2 + 1]};
  float p[4] = {C[l * 4], C[l * 4 + 1], C[l * 4 + 2], C[l * 4 + 3]};
  float q[4] = {p[0], p[1], p[2], p[3]};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(p[0]), "+f"(p[1]), "+f"(p[2]), "+f"(p[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  const uint32_t unit = 0x7f;
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};"
      : "+f"(q[0]), "+f"(q[1]), "+f"(q[2]), "+f"(q[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(unit), "h"(zero), "h"(zero));
  for (int i = 0; i < 4; i++) D0[l * 4 + i] = p[i], D1[l * 4 + i] = q[i];
}
int main() {
  uint32_t *A, *B;
  float *C, *D0, *D1;
  cudaMallocManaged(&A, 512); cudaMallocManaged(&B, 256); cudaMallocManaged(&C, 512);
  cudaMallocManaged(&D0, 512); cudaMallocManaged(&D1, 512);
  srand(1);
  long n = 0, diff = 0, nonfinite = 0;
  for (int trial = 0; trial < 20000; trial++) {
    for (int i = 0; i < 128; i++) { uint32_t w = 0; for (int j = 0; j < 4; j++) { uint8_t x; do x = rand() & 0xff; while ((x & 0x7f) == 0x7f); w |= uint32_t(x) << (8 * j); } A[i] = w; }
    for (int i = 0; i < 64; i++) { uint32_t w = 0; for (int j = 0; j < 4; j++) { uint8_t x; do x = rand() & 0xff; while ((x & 0x7f) == 0x7f); w |= uint32_t(x) << (8 * j); } B[i] = w; }
    const float scale = std::ldexp(1.f, rand() % 60 - 30);
    for (int i = 0; i < 128; i++) C[i] = trial % 4 == 0 ? 0.f : (rand() / float(RAND_MAX) - 0.5f) * scale;
    k<<<1, 32>>>(A, B, C, D0, D1);
    cudaDeviceSynchronize();
    for (int i = 0; i < 128; i++) {
      n++;
      if (!std::isfinite(D0[i])) nonfinite++;
      if (__builtin_memcmp(&D0[i], &D1[i], 4) != 0) {
        if (diff < 5) printf("differ: plain %.9g block-scaled %.9g\n", D0[i], D1[i]);
        diff++;
      }
    }
  }
  printf("%ld results, %ld differ bitwise (%ld non-finite)\n", n, diff, nonfinite);
  return diff != 0;
}
