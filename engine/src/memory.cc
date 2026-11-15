#include "memory.h"

#include <algorithm>
#include <cstring>
#include <type_traits>

#include <cuda.h>

#include "common.h"

namespace q27 {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

namespace {

// The driver's VMM entry points, through the runtime (no link to the driver library, as for cuTensorMapEncodeTiled).
struct Vmm {
  decltype(&cuMemAddressReserve) reserve = nullptr;
  decltype(&cuMemAddressFree) addr_free = nullptr;
  decltype(&cuMemCreate) create = nullptr;
  decltype(&cuMemRelease) release = nullptr;
  decltype(&cuMemMap) map = nullptr;
  decltype(&cuMemUnmap) unmap = nullptr;
  decltype(&cuMemSetAccess) access = nullptr;
  decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
  decltype(&cuDeviceGetAttribute) attribute = nullptr;
  bool ok = false;
};

const Vmm& vmm_api() {
  static const Vmm v = [] {
    Vmm a;
    auto get = [](const char* name, auto& fn) {
      void* f = nullptr;
      cudaDriverEntryPointQueryResult q;
      if (cudaGetDriverEntryPointByVersion(name, &f, 12000, cudaEnableDefault, &q) != cudaSuccess ||
          q != cudaDriverEntryPointSuccess)
        f = nullptr;
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(f);
      return f != nullptr;
    };
    a.ok = get("cuMemAddressReserve", a.reserve) && get("cuMemAddressFree", a.addr_free) &&
           get("cuMemCreate", a.create) && get("cuMemRelease", a.release) && get("cuMemMap", a.map) &&
           get("cuMemUnmap", a.unmap) && get("cuMemSetAccess", a.access) &&
           get("cuMemGetAllocationGranularity", a.granularity) && get("cuDeviceGetAttribute", a.attribute);
    return a;
  }();
  return v;
}

CUmemAllocationProp vmm_prop(int device) {
  CUmemAllocationProp p;
  std::memset(&p, 0, sizeof p);
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = device;
  return p;
}

}  // namespace

Pool::~Pool() {
  if (!base_ || !owned_) return;
  if (host_) {
    cudaFreeHost(base_);
  } else if (!chunks_.empty()) {
    const Vmm& v = vmm_api();
    const CUdeviceptr b = CUdeviceptr(base_);
    for (size_t i = 0; i < chunks_.size(); i++)
      if (chunks_[i]) v.unmap(b + i * kVmmChunk, kVmmChunk), v.release(CUmemGenericAllocationHandle(chunks_[i]));
    v.addr_free(b, chunks_.size() * kVmmChunk);
  } else {
    cudaFree(base_);
  }
}

void Pool::map_chunk(size_t i) {
  const Vmm& v = vmm_api();
  int dev = 0;
  Q27_CUDA(cudaGetDevice(&dev));
  const CUmemAllocationProp prop = vmm_prop(dev);
  CUmemGenericAllocationHandle h;
  if (v.create(&h, kVmmChunk, &prop, 0) != CUDA_SUCCESS) fail("pool " + name_ + ": cuMemCreate failed (out of VRAM?)");
  const CUdeviceptr at = CUdeviceptr(base_) + i * kVmmChunk;
  CUmemAccessDesc acc;
  std::memset(&acc, 0, sizeof acc);
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (v.map(at, kVmmChunk, 0, h, 0) != CUDA_SUCCESS || v.access(at, kVmmChunk, &acc, 1) != CUDA_SUCCESS) {
    v.unmap(at, kVmmChunk);
    v.release(h);
    fail("pool " + name_ + ": cuMemMap / cuMemSetAccess failed");
  }
  chunks_[i] = h;
}

bool Pool::commit_vmm() {
  if (const char* e = getenv("Q27_KV_VMM"); e && !atoi(e)) return false;
  const Vmm& v = vmm_api();
  if (!v.ok) return false;
  int dev = 0, supported = 0;
  Q27_CUDA(cudaGetDevice(&dev));
  if (v.attribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !supported)
    return false;
  const CUmemAllocationProp prop = vmm_prop(dev);
  size_t g = 0;
  if (v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || !g || kVmmChunk % g) return false;
  const size_t n = (size_ + kVmmChunk - 1) / kVmmChunk;
  CUdeviceptr b = 0;
  if (v.reserve(&b, n * kVmmChunk, kVmmChunk, 0, 0) != CUDA_SUCCESS) return false;
  base_ = reinterpret_cast<void*>(b);
  chunks_.assign(n, 0);
  for (size_t i = 0; i < n; i++) map_chunk(i);
  return true;
}

std::pair<size_t, size_t> Pool::release(size_t offset, size_t bytes) {
  if (chunks_.empty()) return {0, 0};
  const size_t lo = (offset + kVmmChunk - 1) / kVmmChunk, hi = std::min(chunks_.size(), (offset + bytes) / kVmmChunk);
  if (hi <= lo) return {0, 0};
  const Vmm& v = vmm_api();
  for (size_t i = lo; i < hi; i++) {
    Q27_CHECK(chunks_[i], "pool " + name_ + ": a chunk released twice");
    const CUdeviceptr at = CUdeviceptr(base_) + i * kVmmChunk;
    if (v.unmap(at, kVmmChunk) != CUDA_SUCCESS || v.release(CUmemGenericAllocationHandle(chunks_[i])) != CUDA_SUCCESS)
      fail("pool " + name_ + ": cuMemUnmap / cuMemRelease failed");
    chunks_[i] = 0;
  }
  return {lo * kVmmChunk, (hi - lo) * kVmmChunk};
}

void Pool::restore(std::pair<size_t, size_t> range) {
  for (size_t i = range.first / kVmmChunk; i < (range.first + range.second) / kVmmChunk; i++)
    if (!chunks_[i]) map_chunk(i);
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
  else if (!vmm_wanted_ || !commit_vmm())
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
