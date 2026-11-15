// Memory planner: every device buffer lives in a named pool that is sized up front and allocated
// once (no cudaMalloc while serving). Pools are planned first (add), then committed together after
// the total has been checked against the VRAM budget.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace q27 {

class Pool {
 public:
  explicit Pool(std::string name, bool host_pinned = false) : name_(std::move(name)), host_(host_pinned) {}
  // A device pool committed through CUDA's virtual memory management, in chunks of kVmmChunk: parts of it can give
  // their memory back and take it again at the same addresses (release / restore). Where VMM is missing (or
  // Q27_KV_VMM=0) the pool is one cudaMalloc and releases nothing.
  static constexpr size_t kVmmChunk = size_t(32) << 20;
  void use_vmm() { vmm_wanted_ = !host_; }
  bool vmm() const { return !chunks_.empty(); }
  // Gives back the memory of the whole chunks inside [offset, offset + bytes); returns the range released (empty
  // without VMM). Nothing may touch it until restore. The caller makes sure no work in flight uses it.
  std::pair<size_t, size_t> release(size_t offset, size_t bytes);
  // Maps fresh memory under a range release returned (its contents are undefined).
  void restore(std::pair<size_t, size_t> range);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  // Planning: reserve `bytes` (256-byte aligned) and return the offset of the region.
  size_t add(const std::string& what, size_t bytes);
  void commit();
  // Places the planned regions at `base` instead: memory of another committed pool that is unused while these are
  // (not owned; e.g. load-time scratch in the KV pool before the first request).
  void commit_at(void* base);
  void* base() const { return base_; }
  template <class T>
  T* at(size_t offset) const {
    return reinterpret_cast<T*>(static_cast<char*>(base_) + offset);
  }

  const std::string& name() const { return name_; }
  size_t size() const { return size_; }
  bool host() const { return host_; }
  bool committed() const { return base_ != nullptr; }
  // Largest reservations, for the breakdown log.
  std::vector<std::pair<std::string, size_t>> items() const { return items_; }

 private:
  std::string name_;
  bool host_;
  size_t size_ = 0;
  void* base_ = nullptr;
  bool owned_ = true;
  std::vector<std::pair<std::string, size_t>> items_;
  bool vmm_wanted_ = false;
  std::vector<unsigned long long> chunks_;  // (VMM) the physical allocation of each chunk, 0 while released
  bool commit_vmm();
  void map_chunk(size_t i);
};

struct Budget {
  size_t free_at_start = 0;   // cudaMemGetInfo free before any allocation
  size_t total = 0;
  size_t budget = 0;          // device bytes this process may allocate in pools
  size_t asked = 0;           // VRAM_BUDGET_GIB, if it was above what is free (and so not used)
};

// min(free - WDDM_MARGIN_GIB (0.6), 30.3 GiB minus the CUDA context), or VRAM_BUDGET_GIB if that is lower.
Budget make_budget();
// The same for a device with `free` of `total` bytes free (after the CUDA context), without CUDA (Options::plan_only).
Budget budget_for(size_t free, size_t total);

// Prints a breakdown of all pools and fails if the device pools exceed the budget.
void check_and_log(const Budget& b, const std::vector<Pool*>& pools);

}  // namespace q27
