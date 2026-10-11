// Shared helpers for the Phase 0 microbenchmarks.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                                                  \
  do {                                                                                               \
    cudaError_t err_ = (call);                                                                       \
    if (err_ != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err_)); \
      std::exit(1);                                                                                  \
    }                                                                                                \
  } while (0)

struct Timer {
  cudaEvent_t start, stop;
  cudaStream_t stream;
  explicit Timer(cudaStream_t s = nullptr) : stream(s) {
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin() { CHECK(cudaEventRecord(start, stream)); }
  // Milliseconds since begin().
  float end() {
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

inline int sm_count() {
  int dev = 0, n = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n;
}

inline void print_device() {
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, 0));
  size_t free_b = 0, total_b = 0;
  CHECK(cudaMemGetInfo(&free_b, &total_b));
  std::printf("# device: %s sm_%d%d, %d SMs, L2 %.0f MiB, smem/block opt-in %zu KiB, mem free/total %.2f/%.2f GiB\n",
              p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1048576.0,
              p.sharedMemPerBlockOptin / 1024, free_b / 1073741824.0, total_b / 1073741824.0);
}
