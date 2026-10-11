// Is the accumulation of mma.sync m16n8k32 E4M3 (f32 accumulators) as exact as FP32? Random E4M3 A / B
// and FP32 C; compares D with round-to-nearest FP32 of C + the exact dot product (double on the host).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp8.h>
__global__ void k(const uint32_t* A, const uint32_t* B, const float* C, float* D) {
  const int l = threadIdx.x;
  uint32_t a[4] = {A[l * 4], A[l * 4 + 1], A[l * 4 + 2], A[l * 4 + 3]}, b[2] = {B[l * 2], B[l * 2 + 1]};
  float acc[4] = {C[l * 4], C[l * 4 + 1], C[l * 4 + 2], C[l * 4 + 3]};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  for (int i = 0; i < 4; i++) D[l * 4 + i] = acc[i];
}
static float e4(uint8_t x) { __nv_fp8_e4m3 v; v.__x = x; return float(v); }
int main() {
  uint32_t *A, *B; float *C, *D;
  cudaMallocManaged(&A, 512); cudaMallocManaged(&B, 256); cudaMallocManaged(&C, 512); cudaMallocManaged(&D, 512);
  srand(1);
  long n = 0, exact = 0; double maxulp = 0;
  for (int trial = 0; trial < 2000; trial++) {
    for (int i = 0; i < 128; i++) { uint32_t w = 0; for (int j = 0; j < 4; j++) { uint8_t x; do x = rand() & 0xff; while ((x & 0x7f) == 0x7f); w |= uint32_t(x) << (8 * j); } A[i] = w; }
    for (int i = 0; i < 64; i++) { uint32_t w = 0; for (int j = 0; j < 4; j++) { uint8_t x; do x = rand() & 0xff; while ((x & 0x7f) == 0x7f); w |= uint32_t(x) << (8 * j); } B[i] = w; }
    const float scale = std::ldexp(1.f, rand() % 40 - 10);
    for (int i = 0; i < 128; i++) C[i] = (rand() / float(RAND_MAX) - 0.5f) * scale;
    k<<<1, 32>>>(A, B, C, D); cudaDeviceSynchronize();
    // A[row][k], B[k][col] from the fragment layout.
    float Am[16][32], Bm[32][8];
    for (int l = 0; l < 32; l++) for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) {
      const int row = l / 4 + 8 * (r % 2), kk = 4 * (l % 4) + j + 16 * (r / 2);
      Am[row][kk] = e4(A[l * 4 + r] >> (8 * j));
    }
    for (int l = 0; l < 32; l++) for (int r = 0; r < 2; r++) for (int j = 0; j < 4; j++)
      Bm[4 * (l % 4) + j + 16 * r][l / 4] = e4(B[l * 2 + r] >> (8 * j));
    for (int l = 0; l < 32; l++) for (int i = 0; i < 4; i++) {
      const int row = l / 4 + 8 * (i / 2), col = 2 * (l % 4) + i % 2;
      double dot = C[l * 4 + i], mag = std::fabs(C[l * 4 + i]);
      for (int kk = 0; kk < 32; kk++) dot += double(Am[row][kk]) * Bm[kk][col], mag = std::fmax(mag, std::fabs(double(Am[row][kk]) * Bm[kk][col]));
      const float ref = float(dot), got = D[l * 4 + i];
      const double ulp = std::ldexp(1.0, std::ilogb(float(mag)) - 23);  // FP32 ulp of the largest term
      maxulp = std::fmax(maxulp, std::fabs(double(got) - ref) / ulp);
      n++, exact += got == ref;
    }
  }
  printf("%ld results, %ld equal to RN(C + exact dot), max %.1f ulp of the largest term (FP32 ulp: 2^-23 relative)\n", n, exact, maxulp);
}
