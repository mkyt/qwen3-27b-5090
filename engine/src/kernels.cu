// Elementwise and small kernels: embedding gather, RMSNorm, activation quantization, SiLU-mul,
// BF16 GEMV, scale-factor swizzle. Straightforward Phase 1 versions (one launch per op).
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "kernels.h"
#include "launch.cuh"
#include "mma.cuh"

namespace q27 {
namespace {

constexpr int kThreads = 256;

__device__ float block_sum(float v) {
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();  // red may still be read by a previous call
  if (lane == 0) red[warp] = v;
  __syncthreads();
  int nw = blockDim.x / 32;
  v = lane < nw ? red[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void embed_gather_kernel(const bf16* table, const int32_t* ids, int D, float* out, const bf16* mm,
                                    int vocab) {
  pdl_trigger();
  pdl_wait();
  const int32_t id = ids[blockIdx.x];
  const bf16* row = mm && id >= vocab ? mm + int64_t(id - vocab) * D : table + int64_t(id) * D;
  float* o = out + int64_t(blockIdx.x) * D;
  // 8 bf16 (16 bytes) per load: UVA reads over PCIe want wide transactions. Every load of the row is
  // issued before the first use (one PCIe round trip instead of one per loop iteration).
  constexpr int kPer = 4;
  uint4 raw[kPer];
#pragma unroll
  for (int u = 0; u < kPer; u++) {
    const int i = (threadIdx.x + u * blockDim.x) * 8;
    if (i < D) raw[u] = *reinterpret_cast<const uint4*>(row + i);
  }
#pragma unroll
  for (int u = 0; u < kPer; u++) {
    const int i = (threadIdx.x + u * blockDim.x) * 8;
    if (i >= D) break;
    const bf16* h = reinterpret_cast<const bf16*>(&raw[u]);
    float4 lo = make_float4(__bfloat162float(h[0]), __bfloat162float(h[1]), __bfloat162float(h[2]), __bfloat162float(h[3]));
    float4 hi = make_float4(__bfloat162float(h[4]), __bfloat162float(h[5]), __bfloat162float(h[6]), __bfloat162float(h[7]));
    *reinterpret_cast<float4*>(o + i) = lo;
    *reinterpret_cast<float4*>(o + i + 4) = hi;
  }
}

__global__ void rmsnorm_kernel(const float* x, const bf16* w, int D, float eps, bool zc, float* out) {
  const float* xr = x + int64_t(blockIdx.x) * D;
  float* o = out + int64_t(blockIdx.x) * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
  float r = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float wi = __bfloat162float(w[i]);
    o[i] = xr[i] * r * (zc ? 1.f + wi : wi);
  }
}

// Two e2m1 codes in one byte, `even` in the low nibble (round to nearest even, saturating).
__device__ uint8_t cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}" : "=h"(r) : "f"(odd), "f"(even));
  return uint8_t(r);
}


// NVFP4 quantization of one 16-element block (row m, k-block kb) of a [M][K] activation.
__device__ void quant_block16(const float* v, float input_scale, int m, int kb, int K, uint8_t* q, uint8_t* sf) {
  int kblocks = K / 16;
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
  float gs = 1.f / input_scale;
  __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
  float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
  float inv = sff != 0.f ? gs / sff : 0.f;
  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; j++) packed[j / 4] |= uint32_t(cvt_e2m1x2(v[2 * j] * inv, v[2 * j + 1] * inv)) << (8 * (j % 4));
  *reinterpret_cast<uint2*>(q + int64_t(m) * (K / 2) + kb * 8) = make_uint2(packed[0], packed[1]);
  sf[sf_offset(m, kb, kblocks)] = sfq;
}

__device__ __forceinline__ void load16(const bf16* p, float* v) {
  const uint4* src = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const uint4 t = src[j];
    const uint32_t w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[k]));
      v[8 * j + 2 * k] = f.x, v[8 * j + 2 * k + 1] = f.y;
    }
  }
}

__device__ __forceinline__ void load16(const float* p, float* v) {
  const float4* src = reinterpret_cast<const float4*>(p);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float4 t = src[j];
    v[4 * j] = t.x, v[4 * j + 1] = t.y, v[4 * j + 2] = t.z, v[4 * j + 3] = t.w;
  }
}

__device__ __forceinline__ uint8_t to_fp8(float x, float s) {
  return __nv_cvt_float_to_fp8(x / s, __NV_SATFINITE, __NV_E4M3);
}

// One thread per 16-element block.
__global__ void quant_fp4_kernel(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf) {
  int kblocks = K / 16;
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(M) * kblocks) return;
  int m = int(idx / kblocks), kb = int(idx % kblocks);
  float v[16];
  load16(x + int64_t(m) * K + kb * 16, v);
  quant_block16(v, input_scale, m, kb, K, q, sf);
}

// Block per token, one thread per 16 elements of the hidden state: zero-centered RMSNorm, then
// optionally the fp32 result and the quantized activation for the next GEMM.
__global__ void rmsnorm_quant_kernel(const float* x, const bf16* w, float eps, float* out, QuantOut qo, bf16* cap,
                                     int cap_ld) {
  pdl_trigger();
  constexpr int D = dims::kHidden;
  const int t = blockIdx.x, kb = threadIdx.x;
  uint4 wraw[2];  // the weights (16 bf16) are loaded before the PDL wait
  wraw[0] = __ldg(reinterpret_cast<const uint4*>(w + kb * 16));
  wraw[1] = __ldg(reinterpret_cast<const uint4*>(w + kb * 16 + 8));
  const bf16* wv = reinterpret_cast<const bf16*>(wraw);
  pdl_wait();
  float v[16];
  load16(x + int64_t(t) * D + kb * 16, v);
  if (cap) {
    uint32_t packed[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      __nv_bfloat162 b2 = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
      packed[j] = *reinterpret_cast<uint32_t*>(&b2);
    }
    uint4* o = reinterpret_cast<uint4*>(cap + int64_t(t) * cap_ld + kb * 16);
    o[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    o[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) ss += v[j] * v[j];
  float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
  for (int j = 0; j < 16; j++) v[j] = v[j] * r * (1.f + __bfloat162float(wv[j]));
  if (out) {
    float4* o = reinterpret_cast<float4*>(out + int64_t(t) * D + kb * 16);
#pragma unroll
    for (int j = 0; j < 4; j++) o[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
  }
  if (qo.kind == QuantOut::kFp4) {
    quant_block16(v, qo.in_scale, t, kb, D, qo.q, qo.sf);
  } else if (qo.kind == QuantOut::kFp8) {
    uint32_t packed[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 16; j++) packed[j / 4] |= uint32_t(to_fp8(v[j], qo.in_scale)) << (8 * (j % 4));
    *reinterpret_cast<uint4*>(qo.q + int64_t(t) * D + kb * 16) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

// silu(gate) * up, NVFP4-quantized; one thread per 16-element block. gate / up rows are ld apart.
template <class E>
__global__ void silu_mul_quant_fp4_kernel(const E* gate, const E* up, int64_t ld, int T, int I,
                                          float input_scale, uint8_t* q, uint8_t* sf) {
  pdl_trigger();
  pdl_wait();
  int kblocks = I / 16;
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= int64_t(T) * kblocks) return;
  int t = int(idx / kblocks), kb = int(idx % kblocks);
  float g[16], u[16];
  load16(gate + t * ld + kb * 16, g);
  load16(up + t * ld + kb * 16, u);
#pragma unroll
  for (int j = 0; j < 16; j++) g[j] = g[j] / (1.f + expf(-g[j])) * u[j];
  quant_block16(g, input_scale, t, kb, I, q, sf);
}

__global__ void quant_fp8_kernel(const float* x, int64_t n, float s, uint8_t* q) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) q[i] = __nv_cvt_float_to_fp8(x[i] / s, __NV_SATFINITE, __NV_E4M3);
}

__global__ void silu_mul_kernel(const float* g, const float* u, int64_t n, float* out) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

// One warp per output element.
__global__ void linear_bf16_kernel(const float* x, const bf16* w, int T, int N, int K, float* out) {
  int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= int64_t(T) * N) return;
  int t = int(warp / N), n = int(warp % N);
  const float* xr = x + int64_t(t) * K;
  const bf16* wr = w + int64_t(n) * K;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) acc += xr[k] * __bfloat162float(wr[k]);
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[warp] = acc;
}

// Two small BF16 linear layers on the same input (GDN in_proj_a / in_proj_b), one warp per output,
// 16-byte weight loads.
__global__ void linear_bf16_pair_kernel(const float* x, const bf16* wa, const bf16* wb, int T, int N, int K,
                                        float* outa, float* outb) {
  pdl_trigger();
  pdl_wait();
  int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= int64_t(T) * 2 * N) return;
  int t = int(warp / (2 * N)), n = int(warp % (2 * N));
  const bf16* wr = (n < N ? wa : wb) + int64_t(n % N) * K;
  const float* xr = x + int64_t(t) * K;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 32 * 8) {
    uint4 raw = __ldg(reinterpret_cast<const uint4*>(wr + k));
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float4 x0 = *reinterpret_cast<const float4*>(xr + k), x1 = *reinterpret_cast<const float4*>(xr + k + 4);
    float2 w0 = __bfloat1622float2(h[0]), w1 = __bfloat1622float2(h[1]), w2 = __bfloat1622float2(h[2]),
           w3 = __bfloat1622float2(h[3]);
    acc += x0.x * w0.x + x0.y * w0.y + x0.z * w1.x + x0.w * w1.y + x1.x * w2.x + x1.y * w2.y + x1.z * w3.x +
           x1.w * w3.y;
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) (n < N ? outa : outb)[int64_t(t) * N + n % N] = acc;
}

// Prefill version (T > 8): tensor cores with x split into bf16 hi + lo (x - hi), two MMAs per tile and
// fp32 accumulation (products exact, x kept to ~16 mantissa bits). Block = 32 tokens x the 96
// outputs (a | b), 12 warps = 2 token tiles x 6 pairs of n8 tiles, K in 64-wide cp.async stages.
// Each output accumulates over k in order regardless of where its row sits, so rows do not depend on
// T or on the other rows.
constexpr int kAbTok = 32, kAbN = 96, kAbK = 64, kAbWarps = 12;
constexpr int kAbXBytes = kAbTok * kAbK * 4, kAbWBytes = kAbN * kAbK * 2, kAbStage = kAbXBytes + kAbWBytes;

__global__ void __launch_bounds__(kAbWarps * 32) linear_bf16_pair_tc_kernel(const float* x, const bf16* wa,
                                                                            const bf16* wb, int T, int K,
                                                                            float* outa, float* outb) {
  pdl_trigger();
  __shared__ __align__(128) uint8_t ab_smem[2 * kAbStage];
  const uint32_t sbase = uint32_t(__cvta_generic_to_shared(ab_smem));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r0 = lane / 4, c2 = 2 * (lane % 4), lm = lane / 8, lr = lane % 8;
  const int t0 = blockIdx.x * kAbTok, mt = warp % 2, p = warp / 2;  // tokens 16 mt.., n8 tiles 2p, 2p + 1
  constexpr int kHalf = kAbN / 2;
  // x rows: 16 units of 16 B (unit ^ row % 8); w rows: 8 units (unit ^ row % 8).
  auto issue = [&](int kt) {
    const uint32_t st = sbase + (kt % 2) * kAbStage;
    const int k0 = kt * kAbK;
    for (int i = tid; i < kAbTok * 16 + kAbN * 8; i += kAbWarps * 32) {
      if (i < kAbTok * 16) {
        const int r = i / 16, u = i % 16, t = t0 + r;
        const bool ok = t < T;
        cp_async16z(st + r * 256 + ((u ^ (r & 7)) << 4), x + (ok ? int64_t(t) * K + k0 + u * 4 : 0), ok);
      } else {
        const int j = i - kAbTok * 16, n = j / 8, u = j % 8;
        const bf16* w = (n < kHalf ? wa : wb) + int64_t(n % kHalf) * K + k0 + u * 8;
        cp_async16z(st + kAbXBytes + n * 128 + ((u ^ (n & 7)) << 4), w, true);
      }
    }
    cp_commit();
  };
  const int nk = K / kAbK;
  pdl_wait();
  issue(0);
  float acc[2][4] = {};
  for (int kt = 0; kt < nk; kt++) {
    if (kt + 1 < nk) issue(kt + 1); else cp_commit();
    cp_wait<1>();
    __syncthreads();
    const uint32_t st = sbase + (kt % 2) * kAbStage;
    const float* xs = reinterpret_cast<const float*>(ab_smem + (kt % 2) * kAbStage);
#pragma unroll
    for (int kk = 0; kk < kAbK / 16; kk++) {
      uint32_t hi[4], lo[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {  // a[q]: row r0 + 8 (q % 2), cols c2 + 8 (q / 2)
        const int r = 16 * mt + r0 + 8 * (q % 2), c = kk * 16 + c2 + 8 * (q / 2);
        const float2 v = *reinterpret_cast<const float2*>(xs + r * 64 + (((c / 4) ^ (r & 7)) * 4) + c % 4);
        const __nv_bfloat162 h = __floats2bfloat162_rn(v.x, v.y);
        const float2 hf = __bfloat1622float2(h);
        hi[q] = *reinterpret_cast<const uint32_t*>(&h);
        lo[q] = pack_bf16(v.x - hf.x, v.y - hf.y);
      }
      uint32_t b[4];
      ldsm_x4(b, st + kAbXBytes + (16 * p + lr + 8 * (lm / 2)) * 128 + (((2 * kk + lm % 2) ^ lr) << 4));
      mma_bf16_16816(acc[0], hi, b);
      mma_bf16_16816(acc[1], hi, b + 2);
      mma_bf16_16816(acc[0], lo, b);
      mma_bf16_16816(acc[1], lo, b + 2);
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt = 0; nt < 2; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int t = t0 + 16 * mt + r0 + 8 * (e / 2), n = 16 * p + 8 * nt + c2 + (e & 1);
      if (t < T) (n < kHalf ? outa : outb)[int64_t(t) * kHalf + n % kHalf] = acc[nt][e];
    }
}

__global__ void add_pos_kernel(int32_t* pos, int T) {
  pdl_trigger();
  pdl_wait(); *pos += T; }

__global__ void add_pos_dev_kernel(int32_t* pos, const int32_t* n) {
  pdl_trigger();
  pdl_wait();
  *pos += *n;
}

__global__ void swizzle_sf_kernel(const uint8_t* src, int rows, int kblocks, uint8_t* dst) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(rows) * kblocks) return;
  int r = int(i / kblocks), kb = int(i % kblocks);
  dst[sf_offset(r, kb, kblocks)] = src[i];
}

unsigned blocks(int64_t n, int threads = kThreads) { return unsigned((n + threads - 1) / threads); }

}  // namespace

// Read at every launch (graph replays run no host code), so that Engine.configure can switch it.
bool pdl_enabled() {
  const char* e = getenv("Q27_NO_PDL");
  return !(e && *e && *e != '0');
}

#ifdef Q27_TRACE
namespace {
std::vector<TraceSetter>& trace_setters() {
  static std::vector<TraceSetter> v;
  return v;
}
std::vector<std::string> g_launches;
unsigned long long* g_trace_dev = nullptr;
}  // namespace
int trace_register(TraceSetter f) {
  trace_setters().push_back(f);
  return 0;
}
void trace_launch(const void* kernel, dim3 grid) {
  const char* name = "?";
  cudaFuncGetName(&name, kernel);
  g_launches.push_back(std::string(name) + " " + std::to_string(grid.x * grid.y * grid.z));
}
void trace_clear_launches() { g_launches.clear(); }
void trace_step_begin(cudaStream_t s) {
  if (!g_trace_dev) {
    Q27_CUDA(cudaMalloc(&g_trace_dev, (8 + size_t(kTraceMax) * 4) * 8));
    for (auto f : trace_setters()) f(g_trace_dev);
  }
  Q27_CUDA(cudaMemsetAsync(g_trace_dev, 0, 8, s));
}
// Writes the records of the last step ("gid cta kind t" lines) and the launch log of the last capture.
void trace_step_end(const char* path) {
  unsigned long long n = 0;
  Q27_CUDA(cudaMemcpy(&n, g_trace_dev, 8, cudaMemcpyDeviceToHost));
  n = std::min<unsigned long long>(n, kTraceMax);
  std::vector<unsigned long long> r(n * 4);
  Q27_CUDA(cudaMemcpy(r.data(), g_trace_dev + 8, n * 32, cudaMemcpyDeviceToHost));
  FILE* f = fopen(path, "w");
  for (auto& l : g_launches) fprintf(f, "L %s\n", l.c_str());
  for (size_t i = 0; i < n; i++) fprintf(f, "%llu %llu %llu %llu\n", r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
  fclose(f);
}
#else
void trace_clear_launches() {}
void trace_step_begin(cudaStream_t) {}
void trace_step_end(const char*) {}
#endif

void embed_gather(const bf16* table, const int32_t* ids, int T, int D, float* out, cudaStream_t s, const bf16* mm,
                  int vocab) {
  if (T == 0) return;
  Q27_CHECK(D % 8 == 0 && D <= 4 * 8 * 1024, "embed_gather: D must be a multiple of 8, at most 32768");
  const int threads = std::min(1024, ((D / 8 + 3) / 4 + 31) / 32 * 32);  // 4 loads per thread
  launch_pdl(embed_gather_kernel, dim3(T), dim3(threads), 0, s, table, ids, D, out, mm, vocab);
  Q27_CUDA(cudaGetLastError());
}

void rmsnorm(const float* x, const bf16* w, int T, int D, float eps, bool zero_centered, float* out, cudaStream_t s) {
  if (T == 0) return;
  rmsnorm_kernel<<<T, D >= 1024 ? 512 : 128, 0, s>>>(x, w, D, eps, zero_centered, out);
  Q27_CUDA(cudaGetLastError());
}

void quant_fp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  Q27_CHECK(K % 64 == 0, "quant_fp4: K must be a multiple of 64");
  if (M == 0) return;
  quant_fp4_kernel<<<blocks(int64_t(M) * (K / 16)), kThreads, 0, s>>>(x, M, K, input_scale, q, sf);
  Q27_CUDA(cudaGetLastError());
}

void quant_fp8(const float* x, int64_t n, float input_scale, uint8_t* q, cudaStream_t s) {
  if (n == 0) return;
  quant_fp8_kernel<<<blocks(n), kThreads, 0, s>>>(x, n, input_scale, q);
  Q27_CUDA(cudaGetLastError());
}

void silu_mul(const float* gate, const float* up, int64_t n, float* out, cudaStream_t s) {
  if (n == 0) return;
  silu_mul_kernel<<<blocks(n), kThreads, 0, s>>>(gate, up, n, out);
  Q27_CUDA(cudaGetLastError());
}

void linear_bf16(const float* x, const bf16* w, int T, int N, int K, float* out, cudaStream_t s) {
  if (T == 0) return;
  linear_bf16_kernel<<<blocks(int64_t(T) * N * 32), kThreads, 0, s>>>(x, w, T, N, K, out);
  Q27_CUDA(cudaGetLastError());
}

void rmsnorm_quant(const float* x, const bf16* w, int T, float eps, float* out, QuantOut q, cudaStream_t s, bf16* cap,
                   int cap_ld) {
  if (T == 0) return;
  launch_pdl(rmsnorm_quant_kernel, dim3(T), dim3(dims::kHidden / 16), 0, s, x, w, eps, out, q, cap, cap_ld);
  Q27_CUDA(cudaGetLastError());
}

void silu_mul_quant_fp4(const float* gate, const float* up, int64_t ld, int T, int I, float input_scale, uint8_t* q,
                        uint8_t* sf, cudaStream_t s) {
  Q27_CHECK(I % 64 == 0, "silu_mul_quant_fp4: I must be a multiple of 64");
  if (T == 0) return;
  launch_pdl(silu_mul_quant_fp4_kernel<float>, dim3(blocks(int64_t(T) * (I / 16))), dim3(kThreads), 0, s, gate, up, ld, T, I, input_scale, q, sf);
  Q27_CUDA(cudaGetLastError());
}

void silu_mul_quant_fp4(const bf16* gate, const bf16* up, int64_t ld, int T, int I, float input_scale, uint8_t* q,
                        uint8_t* sf, cudaStream_t s) {
  Q27_CHECK(I % 64 == 0, "silu_mul_quant_fp4: I must be a multiple of 64");
  if (T == 0) return;
  launch_pdl(silu_mul_quant_fp4_kernel<bf16>, dim3(blocks(int64_t(T) * (I / 16))), dim3(kThreads), 0, s, gate, up, ld, T, I, input_scale, q, sf);
  Q27_CUDA(cudaGetLastError());
}

void linear_bf16_pair(const float* x, const bf16* wa, const bf16* wb, int T, int N, int K, float* outa, float* outb,
                      cudaStream_t s) {
  Q27_CHECK(K % 256 == 0, "linear_bf16_pair: K must be a multiple of 256");
  if (T == 0) return;
  if (T > 8) {
    Q27_CHECK(2 * N == kAbN, "linear_bf16_pair: N must be 48 for T > 8");
    launch_pdl(linear_bf16_pair_tc_kernel, dim3((T + kAbTok - 1) / kAbTok), dim3(kAbWarps * 32), 0, s, x, wa, wb, T, K,
               outa, outb);
    Q27_CUDA(cudaGetLastError());
    return;
  }
  launch_pdl(linear_bf16_pair_kernel, dim3(blocks(int64_t(T) * 2 * N * 32)), dim3(kThreads), 0, s, x, wa, wb, T, N, K, outa, outb);
  Q27_CUDA(cudaGetLastError());
}

void add_pos(int32_t* pos, int T, cudaStream_t s) {
  launch_pdl(add_pos_kernel, dim3(1), dim3(1), 0, s, pos, T);
  Q27_CUDA(cudaGetLastError());
}

void add_pos_dev(int32_t* pos, const int32_t* n, cudaStream_t s) {
  launch_pdl(add_pos_dev_kernel, dim3(1), dim3(1), 0, s, pos, n);
  Q27_CUDA(cudaGetLastError());
}

__global__ void swap_bytes_kernel(uint4* a, uint4* b, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint4 x = a[i], y = b[i];
    a[i] = y;
    b[i] = x;
  }
}

void swap_bytes(void* a, void* b, size_t bytes, cudaStream_t s) {
  Q27_CHECK(bytes % 16 == 0 && reinterpret_cast<uintptr_t>(a) % 16 == 0 && reinterpret_cast<uintptr_t>(b) % 16 == 0,
            "swap_bytes: unaligned");
  swap_bytes_kernel<<<170 * 4, 256, 0, s>>>(static_cast<uint4*>(a), static_cast<uint4*>(b), bytes / 16);
  Q27_CUDA(cudaGetLastError());
}

void swizzle_sf(const uint8_t* src, int rows, int kblocks, uint8_t* dst, cudaStream_t s) {
  swizzle_sf_kernel<<<blocks(int64_t(rows) * kblocks), kThreads, 0, s>>>(src, rows, kblocks, dst);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
