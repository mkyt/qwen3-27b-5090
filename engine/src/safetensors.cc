#include "safetensors.h"

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "common.h"
#include "fs.h"

namespace q27 {
namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(fs_path(path), std::ios::binary);
  if (!f) fail("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

#ifdef _WIN32
// pread for a CRT descriptor: a positioned ReadFile on its OS handle (does not move the file pointer
// for other readers; the handle is not opened for overlapped I/O, so the call is synchronous).
int64_t pread_once(int fd, char* p, int64_t len, int64_t off) {
  OVERLAPPED ov{};
  ov.Offset = DWORD(uint64_t(off));
  ov.OffsetHigh = DWORD(uint64_t(off) >> 32);
  DWORD n = 0;
  if (!ReadFile(HANDLE(_get_osfhandle(fd)), p, DWORD(len), &n, &ov)) return -1;
  return int64_t(n);
}
int open_ro(const std::string& path) {
  return ::_wopen(fs_path(path).c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
}
void close_fd(int fd) { ::_close(fd); }
#else
int64_t pread_once(int fd, char* p, int64_t len, int64_t off) { return ::pread(fd, p, size_t(len), off); }
int open_ro(const std::string& path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
void close_fd(int fd) { ::close(fd); }
#endif

void pread_all(int fd, void* dst, int64_t len, int64_t off) {
  char* p = static_cast<char*>(dst);
  while (len > 0) {
    // Bounded requests: very large single reads are slow on virtiofs / 9p.
    int64_t n = pread_once(fd, p, std::min<int64_t>(len, 64 << 20), off);
    if (n <= 0) fail("pread failed: " + std::string(strerror(errno)));
    p += n;
    off += n;
    len -= n;
  }
}

}  // namespace

int64_t TensorInfo::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

SafeTensors::SafeTensors(const std::string& dir) : dir_(dir) {
  config_ = Json::parse(slurp(dir + "/config.json"));
  std::vector<std::string> shards;
  std::ifstream idx(fs_path(dir + "/model.safetensors.index.json"));
  if (idx) {
    std::set<std::string> uniq;
    Json index = Json::parse(slurp(dir + "/model.safetensors.index.json"));
    for (auto& [name, file] : index["weight_map"].obj) uniq.insert(file.as_str());
    shards.assign(uniq.begin(), uniq.end());
  } else {
    shards.push_back("model.safetensors");
  }
  for (auto& shard : shards) {
    std::string path = dir + "/" + shard;
    int fd = open_ro(path);
    if (fd < 0) fail("cannot open " + path);
    fds_.push_back(fd);
    uint64_t hlen = 0;
    pread_all(fd, &hlen, 8, 0);
    std::string header(hlen, '\0');
    pread_all(fd, header.data(), int64_t(hlen), 8);
    Json h = Json::parse(header);
    for (auto& [name, t] : h.obj) {
      if (name == "__metadata__") continue;
      TensorInfo ti;
      ti.dtype = t["dtype"].as_str();
      for (auto& d : t["shape"].arr) ti.shape.push_back(d.as_int());
      ti.fd = fd;
      int64_t b = t["data_offsets"][0].as_int(), e = t["data_offsets"][1].as_int();
      ti.offset = 8 + int64_t(hlen) + b;
      ti.nbytes = e - b;
      tensors_.emplace(name, std::move(ti));
    }
  }
}

SafeTensors::~SafeTensors() {
  for (int fd : fds_) close_fd(fd);
}

const TensorInfo& SafeTensors::info(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) fail("tensor not in checkpoint: " + name);
  return it->second;
}

void SafeTensors::read(const std::string& name, void* dst) const {
  const auto& t = info(name);
  pread_all(t.fd, dst, t.nbytes, t.offset);
}

void SafeTensors::read_range(const std::string& name, int64_t begin, int64_t len, void* dst) const {
  const auto& t = info(name);
  Q27_CHECK(begin >= 0 && begin + len <= t.nbytes, "read_range out of bounds: " + name);
  pread_all(t.fd, dst, len, t.offset + begin);
}

}  // namespace q27
