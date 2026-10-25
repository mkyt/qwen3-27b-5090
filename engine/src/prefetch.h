// Weight loading: reads the byte ranges a loader will ask for, in its order, ahead with several threads.
// One read stream from WSL2's virtiofs (/mnt/<drive>) gets about 1.2-1.5 GB/s and parallel reads scale almost
// linearly, so the loaders plan their reads, the threads fill pinned slots (or read straight into pinned host
// destinations), and the loaders take them in order and upload as before: the bytes in the pools do not change.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "safetensors.h"

namespace q27 {

class Prefetcher {
 public:
  struct Read {
    const SafeTensors* src;
    std::string name;
    int64_t begin, len;
    void* direct;  // pinned host destination (read straight there), or nullptr: into a slot
  };
  static constexpr size_t kSlotBytes = 32 << 20;

  // Q27_LOAD_THREADS threads (default 8; 1: one thread, still ahead of the uploads).
  explicit Prefetcher(std::vector<Read> reads);
  ~Prefetcher();
  Prefetcher(const Prefetcher&) = delete;
  Prefetcher& operator=(const Prefetcher&) = delete;

  // Waits for read i (the loader asks in plan order; name and begin are checked against the plan) and returns
  // its slot (nullptr for a direct read). A slot read must be handed back with release(i, s) once the work that
  // reads it is queued on s.
  void* get(size_t i, const std::string& name, int64_t begin);
  void release(size_t i, cudaStream_t s);
  // Reads of a slot's size at most: a larger range goes straight to the loader's own buffer.
  static bool fits(int64_t len) { return len <= int64_t(kSlotBytes); }

 private:
  void worker();

  std::vector<Read> reads_;
  std::vector<int> slot_of_;       // read -> slot (-1: direct)
  std::vector<size_t> slot_seq_;   // read -> its number among the slot reads
  std::vector<char> done_;         // read i landed
  size_t released_ = 0;            // slot reads handed back (in order)
  size_t next_ = 0;                // next read a thread takes
  int nslots_ = 0;
  char* slots_ = nullptr;
  std::vector<cudaEvent_t> ev_;    // per slot: the uploads from it
  std::mutex mu_;
  std::condition_variable cv_;
  std::exception_ptr error_;
  bool stop_ = false;
  std::vector<std::thread> threads_;
};

// Q27_LOAD_THREADS (default 8).
int load_threads();

}  // namespace q27
