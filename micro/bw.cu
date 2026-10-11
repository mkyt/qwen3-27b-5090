// DRAM read bandwidth: the ceiling for decode.
//
// 1. flat:   one streaming-read kernel over N bytes, for a range of sizes (L2-resident sizes included,
//            to show where L2 stops helping).
// 2. step:   the byte sizes of one Qwen3.8-27B decode step (nvidia checkpoint: 4 weight reads per layer
//            x 64 layers + lm_head = 17.56 GB), each read by its own kernel, captured in one CUDA graph.
//            Run without and with Programmatic Dependent Launch (PDL), where every kernel lets the next
//            one start streaming its weights early and only waits before its (tiny) dependent write --
//            the upper bound of what PDL can hide between GEMV kernels.
//
// usage: bw [--step-scale 1.0] [--iters 20]

#include "common.cuh"

#include <cstdint>
#include <cstring>
#include <vector>

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// Reads n16 16-byte words; folds them into one word per block so the loads are not dead code.
template <bool kPdl>
__global__ void __launch_bounds__(512) read_kernel(const uint4* __restrict__ src, size_t n16, uint32_t* sink) {
  if constexpr (kPdl) {
    // Allow the next kernel in the stream to launch now: its weight reads do not depend on us.
    asm volatile("griddepcontrol.launch_dependents;");
  }
  uint32_t acc = 0;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  constexpr int U = 4;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld_stream(src + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = ld_stream(src + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if constexpr (kPdl) {
    // The dependent part of a real GEMV (reading the previous layer's activations / writing ours)
    // must wait for the previous kernel to finish.
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;  // practically never taken
}

static void launch(bool pdl, int grid, const uint4* src, size_t n16, uint32_t* sink, cudaStream_t s) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = 512;
  cfg.stream = s;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  if (pdl)
    CHECK(cudaLaunchKernelEx(&cfg, read_kernel<true>, src, n16, sink));
  else
    CHECK(cudaLaunchKernelEx(&cfg, read_kernel<false>, src, n16, sink));
}

int main(int argc, char** argv) {
  double step_scale = 1.0;
  int iters = 20;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--step-scale") && a + 1 < argc) step_scale = std::atof(argv[++a]);
    if (!std::strcmp(argv[a], "--iters") && a + 1 < argc) iters = std::atoi(argv[++a]);
  }
  print_device();
  const int sms = sm_count();
  const int grid = sms * 4;  // 4 x 512 threads per SM

  // Decode-step read sizes (bytes), nvidia checkpoint, text only, embedding excluded.
  const double gdn_layer[] = {(10240 + 6144) * 5120 * 1.0, 6144 * 5120 * 1.0,  // in_proj qkv|z, out_proj (FP8)
                              2 * 17408 * 5120 * 0.5625, 17408 * 5120 * 0.5625};  // gate|up, down (NVFP4)
  const double attn_layer[] = {(12288 + 2048) * 5120 * 1.0, 6144 * 5120 * 1.0, 2 * 17408 * 5120 * 0.5625,
                               17408 * 5120 * 0.5625};
  const double lm_head = 248320.0 * 5120 * 0.5625;
  std::vector<size_t> sizes;
  for (int l = 0; l < 64; ++l) {
    const double* src = (l % 4 == 3) ? attn_layer : gdn_layer;
    for (int k = 0; k < 4; ++k) sizes.push_back(size_t(src[k] * step_scale) / 16 * 16);
  }
  sizes.push_back(size_t(lm_head * step_scale) / 16 * 16);
  size_t step_bytes = 0;
  for (size_t s : sizes) step_bytes += s;

  const size_t flat_max = size_t(4) << 30;
  const size_t alloc = std::max(step_bytes, flat_max);
  void* buf = nullptr;
  CHECK(cudaMalloc(&buf, alloc));
  CHECK(cudaMemset(buf, 0x5a, alloc));
  uint32_t* sink = nullptr;
  CHECK(cudaMalloc(&sink, grid * sizeof(uint32_t)));
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Timer t(s);

  std::printf("\n## flat read (one kernel)\n%12s %10s %10s\n", "bytes", "us", "GB/s");
  for (size_t bytes = size_t(16) << 20; bytes <= flat_max; bytes *= 2) {
    // Rotate the start offset so small sizes are not served from L2 by the previous iteration.
    const size_t slots = alloc / bytes;
    for (int w = 0; w < 3; ++w) launch(false, grid, (const uint4*)buf, bytes / 16, sink, s);
    t.begin();
    for (int it = 0; it < iters; ++it) {
      const char* p = (const char*)buf + (it % slots) * bytes;
      launch(false, grid, (const uint4*)p, bytes / 16, sink, s);
    }
    const double us = t.end() * 1e3 / iters;
    std::printf("%12zu %10.1f %10.1f\n", bytes, us, bytes / us / 1e3);
  }

  std::printf("\n## decode step: %zu kernels, %.2f GB (scale %.2f)\n", sizes.size(), step_bytes / 1e9, step_scale);
  for (int pdl = 0; pdl < 2; ++pdl) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    size_t off = 0;
    for (size_t sz : sizes) {
      launch(pdl, grid, (const uint4*)((const char*)buf + off), sz / 16, sink, s);
      off += sz;
    }
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) CHECK(cudaGraphLaunch(ge, s));
    t.begin();
    for (int it = 0; it < iters; ++it) CHECK(cudaGraphLaunch(ge, s));
    const double ms = t.end() / iters;
    std::printf("%-10s %8.3f ms/step %8.1f GB/s\n", pdl ? "graph+pdl" : "graph", ms, step_bytes / ms / 1e6);
    CHECK(cudaGraphExecDestroy(ge));
    CHECK(cudaGraphDestroy(g));
  }
  // Same bytes in one kernel: the floor with no kernel boundaries at all.
  t.begin();
  for (int it = 0; it < iters; ++it) launch(false, grid, (const uint4*)buf, step_bytes / 16, sink, s);
  const double ms = t.end() / iters;
  std::printf("%-10s %8.3f ms/step %8.1f GB/s\n", "one-kernel", ms, step_bytes / ms / 1e6);
  return 0;
}
