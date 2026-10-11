// Reads tensors from a (sharded) safetensors checkpoint with pread, into caller-provided host memory.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "json.h"

namespace q27 {

struct TensorInfo {
  std::string dtype;            // "BF16", "F32", "F8_E4M3", "U8", ...
  std::vector<int64_t> shape;
  int fd = -1;                  // shard file
  int64_t offset = 0;           // absolute byte offset in the shard file
  int64_t nbytes = 0;

  int64_t numel() const;
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& dir);
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  const Json& config() const { return config_; }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const;
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }

  // Reads the whole tensor into dst (info(name).nbytes bytes).
  void read(const std::string& name, void* dst) const;
  // Reads a byte range of the tensor.
  void read_range(const std::string& name, int64_t begin, int64_t len, void* dst) const;

 private:
  std::string dir_;
  Json config_;
  std::vector<int> fds_;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace q27
