// Dependent-chain latencies on SM120 (cycles per instruction, one warp): SHFL, LDS, FFMA, FMUL,
// MUFU.EX2, LDG (L1 hit), BAR.SYNC (1 warp). Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a latency.cu
#include <cstdio>

constexpr int N = 4096;

__global__ void lat(long long* out, float* buf, float seed) {
  __shared__ float sm[1024];
  const int lane = threadIdx.x;
  for (int i = lane; i < 1024; i += 32) sm[i] = (i + 1) % 1024;
  __syncwarp();
  float x = seed + lane;
  long long t0, t1;
  // SHFL
  t0 = clock64();
#pragma unroll 64
  for (int i = 0; i < N; i++) x = __shfl_xor_sync(0xffffffffu, x, 1);
  t1 = clock64();
  if (lane == 0) out[0] = t1 - t0;
  // FFMA
  t0 = clock64();
#pragma unroll 64
  for (int i = 0; i < N; i++) x = fmaf(x, 1.0001f, 0.5f);
  t1 = clock64();
  if (lane == 0) out[1] = t1 - t0;
  // LDS (pointer chase)
  int p = lane;
  t0 = clock64();
#pragma unroll 64
  for (int i = 0; i < N; i++) p = int(sm[p]);
  t1 = clock64();
  if (lane == 0) out[2] = t1 - t0;
  // MUFU.EX2
  t0 = clock64();
#pragma unroll 64
  for (int i = 0; i < N; i++) x = exp2f(x) * 1e-30f;
  t1 = clock64();
  if (lane == 0) out[3] = t1 - t0;
  // LDG pointer chase in L1
  int q = lane;
  t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < N / 16; i++) q = int(__ldca(buf + q));
  t1 = clock64();
  if (lane == 0) out[4] = (t1 - t0) * 16;
  // SHFL + FADD (a reduction level)
  t0 = clock64();
#pragma unroll 64
  for (int i = 0; i < N; i++) x += __shfl_xor_sync(0xffffffffu, x, 1);
  t1 = clock64();
  if (lane == 0) out[5] = t1 - t0;
  // __syncthreads with one warp
  t0 = clock64();
  for (int i = 0; i < N; i++) __syncthreads();
  t1 = clock64();
  if (lane == 0) out[6] = t1 - t0;
  buf[1024 + lane] = x + p + q;
}

int main() {
  long long* d;
  float* buf;
  cudaMalloc(&d, 64);
  cudaMalloc(&buf, 8192 * 4);
  float h[2048];
  for (int i = 0; i < 1024; i++) h[i] = float((i + 33) % 1024);
  cudaMemcpy(buf, h, 4096, cudaMemcpyHostToDevice);
  lat<<<1, 32>>>(d, buf, 1.f);
  lat<<<1, 32>>>(d, buf, 1.f);
  long long r[8];
  cudaMemcpy(r, d, 64, cudaMemcpyDeviceToHost);
  const char* names[] = {"SHFL", "FFMA", "LDS", "MUFU.EX2+FMUL", "LDG L1 hit", "SHFL+FADD", "BAR.SYNC 1 warp"};
  for (int i = 0; i < 7; i++) printf("%-18s %6.1f cycles\n", names[i], double(r[i]) / N);
  return 0;
}
