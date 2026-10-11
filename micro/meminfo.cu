// How much VRAM can the process really use before WDDM starts paging to shared (host) memory?
//
// Allocates and touches device memory in steps, printing cudaMemGetInfo and the time of a full read of
// everything allocated so far. When an allocation lands in shared memory the read slows down sharply
// (decode dropped ~7x in the vLLM environment). Run scripts/vram.sh / gpu_monitor.sh alongside to see
// the Windows counters for the same steps.
//
// The GPU drops to idle clocks (memory 405-810 MHz, ~40 GB/s) during the pause between steps, which looks
// exactly like paging. So each step first reads everything for --warmup-ms to bring the clocks back up,
// then reports the best of 3 timed reads.
//
// usage: meminfo [--step-mib 256] [--pause 2] [--warmup-ms 300] [--stop-gib 31.5]

#include "common.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

__global__ void touch(uint4* p, size_t n16, uint32_t v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_uint4(v, v, v, v);
}

__global__ void sum(const uint4* __restrict__ p, size_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    acc ^= p[i].x;
  if (acc == 0x12345678u) *sink = acc;
}

int main(int argc, char** argv) {
  size_t step_mib = 256;
  double pause = 2, warmup_ms = 300, stop_gib = 31.5;
  for (int a = 1; a + 1 < argc; ++a) {
    if (!std::strcmp(argv[a], "--step-mib")) step_mib = std::atol(argv[++a]);
    else if (!std::strcmp(argv[a], "--pause")) pause = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "--warmup-ms")) warmup_ms = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "--stop-gib")) stop_gib = std::atof(argv[++a]);
  }
  print_device();
  const int grid = sm_count() * 4;
  uint32_t* sink = nullptr;
  CHECK(cudaMalloc(&sink, 4));
  std::vector<std::pair<uint4*, size_t>> chunks;
  size_t total = 0;
  Timer t;
  std::printf("%10s %10s %12s %12s\n", "alloc GiB", "free GiB", "read ms", "read GB/s");
  while (total < size_t(stop_gib * (1ull << 30))) {
    const size_t bytes = step_mib << 20;
    uint4* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      std::printf("# cudaMalloc failed after %.2f GiB\n", total / 1073741824.0);
      break;
    }
    touch<<<grid, 512>>>(p, bytes / 16, uint32_t(chunks.size()));
    chunks.emplace_back(p, bytes);
    total += bytes;
    CHECK(cudaDeviceSynchronize());
    auto read_all = [&] {
      for (auto& [q, n] : chunks) sum<<<grid, 512>>>(q, n / 16, sink);
    };
    const auto w0 = std::chrono::steady_clock::now();
    do {
      read_all();
      CHECK(cudaDeviceSynchronize());
    } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < warmup_ms);
    float ms = 1e30f;
    for (int r = 0; r < 3; ++r) {
      t.begin();
      read_all();
      ms = std::min(ms, t.end());
    }
    size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("%10.2f %10.2f %12.2f %12.1f\n", total / 1073741824.0, free_b / 1073741824.0, ms,
                total / (ms * 1e6));
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(pause));
  }
  return 0;
}
