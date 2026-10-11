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
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  // Planning: reserve `bytes` (256-byte aligned) and return the offset of the region.
  size_t add(const std::string& what, size_t bytes);
  void commit();
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
  std::vector<std::pair<std::string, size_t>> items_;
};

struct Budget {
  size_t free_at_start = 0;   // cudaMemGetInfo free before any allocation
  size_t total = 0;
  size_t budget = 0;          // device bytes this process may allocate in pools
};

// VRAM_BUDGET_GIB (if set) or min(free - WDDM_MARGIN_GIB (0.6), 30.3 GiB minus the CUDA context).
Budget make_budget();

// Prints a breakdown of all pools and fails if the device pools exceed the budget.
void check_and_log(const Budget& b, const std::vector<Pool*>& pools);

}  // namespace q27
