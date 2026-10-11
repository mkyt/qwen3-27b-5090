// Host <-> device paths the runtime depends on (WSL2 / WDDM):
//   1. pinned cudaMemcpyAsync H2D / D2H (GDN state checkpoints ~115 MB, vision weights)
//   2. zero-copy UVA gather of embedding rows from a pinned host table (the host-embedding design):
//      latency of a decode step (1-64 rows) and throughput of a prefill chunk (2K-8K rows)
//
// usage: pcie [--table-gib 2.37]

#include "common.cuh"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

constexpr int kHidden = 5120;  // bf16 row = 10 KiB

__global__ void gather_rows(const uint4* __restrict__ table, const int* __restrict__ ids, uint4* __restrict__ out) {
  constexpr int kRow16 = kHidden * 2 / 16;  // 640 x 16 B per row
  const int row = blockIdx.x;
  const uint4* src = table + size_t(ids[row]) * kRow16;
  for (int i = threadIdx.x; i < kRow16; i += blockDim.x) out[size_t(row) * kRow16 + i] = src[i];
}

int main(int argc, char** argv) {
  double table_gib = 2.37;
  for (int a = 1; a < argc; ++a)
    if (!std::strcmp(argv[a], "--table-gib") && a + 1 < argc) table_gib = std::atof(argv[++a]);
  print_device();
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Timer t(s);

  std::printf("\n## pinned memcpy\n%10s %10s %10s %10s %10s\n", "MiB", "H2D us", "H2D GB/s", "D2H us", "D2H GB/s");
  const size_t max_bytes = size_t(256) << 20;
  void *h = nullptr, *d = nullptr;
  CHECK(cudaHostAlloc(&h, max_bytes, cudaHostAllocDefault));
  CHECK(cudaMalloc(&d, max_bytes));
  std::memset(h, 1, max_bytes);
  for (size_t bytes = size_t(1) << 20; bytes <= max_bytes; bytes *= 4) {
    double us[2];
    for (int dir = 0; dir < 2; ++dir) {
      const int iters = 10;
      auto kind = dir ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
      CHECK(cudaMemcpyAsync(dir ? h : d, dir ? d : h, bytes, kind, s));
      t.begin();
      for (int i = 0; i < iters; ++i) CHECK(cudaMemcpyAsync(dir ? h : d, dir ? d : h, bytes, kind, s));
      us[dir] = t.end() * 1e3 / iters;
    }
    std::printf("%10zu %10.1f %10.1f %10.1f %10.1f\n", bytes >> 20, us[0], bytes / us[0] / 1e3, us[1],
                bytes / us[1] / 1e3);
  }

  const size_t rows_total = size_t(table_gib * (1ull << 30)) / (kHidden * 2);
  std::printf("\n## UVA gather from a %.2f GiB pinned table (%zu rows of %d B)\n", table_gib, rows_total,
              kHidden * 2);
  void* table = nullptr;
  CHECK(cudaHostAlloc(&table, rows_total * kHidden * 2, cudaHostAllocDefault));
  std::memset(table, 0, rows_total * kHidden * 2);
  const int max_rows = 8192;
  int* ids = nullptr;
  uint4* out = nullptr;
  CHECK(cudaMalloc(&ids, max_rows * sizeof(int)));
  CHECK(cudaMalloc(&out, size_t(max_rows) * kHidden * 2));
  std::vector<int> hid(max_rows);
  std::mt19937 rng(0);
  for (int& x : hid) x = int(rng() % rows_total);
  CHECK(cudaMemcpy(ids, hid.data(), max_rows * sizeof(int), cudaMemcpyHostToDevice));
  std::printf("%8s %10s %10s\n", "rows", "us", "GB/s");
  for (int rows : {1, 8, 64, 512, 2048, 8192}) {
    const int iters = rows <= 64 ? 200 : 20;
    gather_rows<<<rows, 128, 0, s>>>((const uint4*)table, ids, out);
    t.begin();
    for (int i = 0; i < iters; ++i) gather_rows<<<rows, 128, 0, s>>>((const uint4*)table, ids, out);
    const double us = t.end() * 1e3 / iters;
    std::printf("%8d %10.1f %10.2f\n", rows, us, double(rows) * kHidden * 2 / us / 1e3);
  }
  CHECK(cudaGetLastError());
  return 0;
}
