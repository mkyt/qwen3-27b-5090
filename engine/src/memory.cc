#include "memory.h"

#include <algorithm>

#include "common.h"

namespace q27 {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

Pool::~Pool() {
  if (!base_ || !owned_) return;
  if (host_)
    cudaFreeHost(base_);
  else
    cudaFree(base_);
}

size_t Pool::add(const std::string& what, size_t bytes) {
  Q27_CHECK(!base_, "pool " + name_ + " already committed");
  size_t off = size_t(round_up(int64_t(size_), 256));
  size_ = off + bytes;
  items_.emplace_back(what, bytes);
  return off;
}

void Pool::commit() {
  if (base_ || size_ == 0) return;
  if (host_)
    Q27_CUDA(cudaHostAlloc(&base_, size_, cudaHostAllocMapped | cudaHostAllocPortable));
  else
    Q27_CUDA(cudaMalloc(&base_, size_));
}

void Pool::commit_at(void* base) {
  Q27_CHECK(!base_ && base, "pool " + name_ + ": commit_at");
  base_ = base, owned_ = false;
}

Budget make_budget() {
  size_t free = 0, total = 0;
  Q27_CUDA(cudaFree(nullptr));  // create the context first so its own allocation is not counted as ours
  Q27_CUDA(cudaMemGetInfo(&free, &total));
  // cudaMemGetInfo does not count other processes' memory (measured on WSL2 and Windows): the server passes what NVML sees.
  if (const char* e = getenv("Q27_DEVICE_FREE_BYTES"); e && *e) {
    const size_t seen = size_t(strtoull(e, nullptr, 10));
    if (seen + (size_t(256) << 20) < free) log("free VRAM: %.2f GiB by cudaMemGetInfo, %.2f GiB with other programs", free / kGiB, seen / kGiB);
    free = std::min(free, seen);
  }
  return budget_for(free, total);
}

Budget budget_for(size_t free, size_t total) {
  Budget b;
  b.free_at_start = free, b.total = total;
  double margin = 0.6, cap = 30.3;
  if (const char* m = getenv("WDDM_MARGIN_GIB")) margin = atof(m);
  // cudaMemGetInfo's free reaches 0 well before WDDM starts to evict (measured on WSL2), so take the
  // smaller of "free minus margin" and the measured safe process total minus what is already used.
  double used = double(b.total - b.free_at_start) / kGiB;
  double by_free = double(b.free_at_start) / kGiB - margin;
  b.budget = size_t(std::max(0.0, std::min(by_free, cap - used)) * kGiB);
  // VRAM_BUDGET_GIB only lowers it: above this, WDDM would move memory to the host without an error.
  if (const char* v = getenv("VRAM_BUDGET_GIB")) {
    const size_t asked = size_t(atof(v) * kGiB);
    if (asked > b.budget) b.asked = asked;
    b.budget = std::min(asked, b.budget);
  }
  return b;
}

void check_and_log(const Budget& b, const std::vector<Pool*>& pools) {
  size_t dev = 0, host = 0;
  log("memory budget: free at start %.2f GiB of %.2f GiB, budget %.2f GiB", b.free_at_start / kGiB, b.total / kGiB,
      b.budget / kGiB);
  if (b.asked) log("VRAM_BUDGET_GIB %.2f GiB is above what is free: using %.2f GiB", b.asked / kGiB, b.budget / kGiB);
  for (auto* p : pools) {
    (p->host() ? host : dev) += p->size();
    log("  %-12s %-6s %8.3f GiB", p->name().c_str(), p->host() ? "host" : "device", p->size() / kGiB);
    auto items = p->items();
    std::sort(items.begin(), items.end(), [](auto& x, auto& y) { return x.second > y.second; });
    for (size_t i = 0; i < items.size() && i < 4; i++)
      if (items[i].second >= (64 << 20)) log("      %-28s %8.3f GiB", items[i].first.c_str(), items[i].second / kGiB);
  }
  log("  total: device %.3f GiB, host pinned %.3f GiB", dev / kGiB, host / kGiB);
  if (dev > b.budget)
    fail("not enough VRAM: device pools (" + std::to_string(dev / kGiB) + " GiB) exceed the VRAM budget (" +
         std::to_string(b.budget / kGiB) + " GiB); lower MAX_TOKENS / PREFILL_CHUNK");
}

}  // namespace q27
