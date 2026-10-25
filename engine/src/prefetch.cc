#include "prefetch.h"

#include <algorithm>

namespace q27 {

int load_threads() {
  const char* e = getenv("Q27_LOAD_THREADS");
  return std::clamp(e && *e ? atoi(e) : 8, 1, 64);
}

Prefetcher::Prefetcher(std::vector<Read> reads) : reads_(std::move(reads)) {
  const int threads = load_threads();
  size_t nslot_reads = 0;
  slot_of_.assign(reads_.size(), -1);
  slot_seq_.assign(reads_.size(), 0);
  done_.assign(reads_.size(), 0);
  nslots_ = std::max(2, 2 * threads);
  for (size_t i = 0; i < reads_.size(); i++) {
    if (reads_[i].direct) continue;
    Q27_CHECK(fits(reads_[i].len), "prefetch: a read larger than a slot: " + reads_[i].name);
    slot_seq_[i] = nslot_reads;
    slot_of_[i] = int(nslot_reads++ % size_t(nslots_));
  }
  nslots_ = int(std::min<size_t>(size_t(nslots_), std::max<size_t>(nslot_reads, 1)));
  for (size_t i = 0; i < reads_.size(); i++)
    if (slot_of_[i] >= 0) slot_of_[i] = int(slot_seq_[i] % size_t(nslots_));
  Q27_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&slots_), size_t(nslots_) * kSlotBytes, cudaHostAllocMapped));
  ev_.resize(nslots_);
  for (auto& e : ev_) Q27_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  for (int t = 0; t < threads; t++) threads_.emplace_back([this] { worker(); });
}

Prefetcher::~Prefetcher() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
  for (auto& e : ev_) cudaEventDestroy(e);
  if (slots_) cudaFreeHost(slots_);
}

void Prefetcher::worker() {
  try {
    for (;;) {
      size_t i;
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_ || next_ >= reads_.size()) return;
        i = next_++;
      }
      const Read& r = reads_[i];
      void* dst = r.direct;
      if (const int slot = slot_of_[i]; slot >= 0) {
        // The slot's previous read must have been handed back and its uploads finished.
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || released_ + size_t(nslots_) > slot_seq_[i]; });
        if (stop_) return;
        lk.unlock();
        Q27_CUDA(cudaEventSynchronize(ev_[slot]));
        dst = slots_ + size_t(slot) * kSlotBytes;
      }
      r.src->read_range(r.name, r.begin, r.len, dst);
      {
        std::lock_guard<std::mutex> lk(mu_);
        done_[i] = 1;
      }
      cv_.notify_all();
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!error_) error_ = std::current_exception();
      stop_ = true;
    }
    cv_.notify_all();
  }
}

void* Prefetcher::get(size_t i, const std::string& name, int64_t begin) {
  Q27_CHECK(i < reads_.size() && reads_[i].name == name && reads_[i].begin == begin,
            "prefetch: the loader left its plan at " + name);
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return done_[i] || error_; });
  if (error_) std::rethrow_exception(error_);
  return slot_of_[i] >= 0 ? slots_ + size_t(slot_of_[i]) * kSlotBytes : nullptr;
}

void Prefetcher::release(size_t i, cudaStream_t s) {
  const int slot = slot_of_[i];
  Q27_CHECK(slot >= 0 && slot_seq_[i] == released_, "prefetch: slots are released in order");
  Q27_CUDA(cudaEventRecord(ev_[slot], s));
  {
    std::lock_guard<std::mutex> lk(mu_);
    released_++;
  }
  cv_.notify_all();
}

}  // namespace q27
