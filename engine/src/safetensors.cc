#include "safetensors.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstring>
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include "check.h"
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
int64_t file_size(int fd) {
  struct _stat64 s{};
  Q27_CHECK(::_fstat64(fd, &s) == 0, "cannot stat checkpoint");
  return s.st_size;
}
#else
int64_t pread_once(int fd, char* p, int64_t len, int64_t off) { return ::pread(fd, p, size_t(len), off); }
int open_ro(const std::string& path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
void close_fd(int fd) { ::close(fd); }
int64_t file_size(int fd) {
  struct stat s{};
  Q27_CHECK(::fstat(fd, &s) == 0, "cannot stat checkpoint");
  return s.st_size;
}
#endif

int dtype_bytes(const std::string& dtype) {
  if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
  if (dtype == "BF16" || dtype == "F16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  fail("unsupported safetensors dtype: " + dtype);
}

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
  // Check nonzero dimensions too when an empty tensor has a zero dimension.
  for (auto d : shape) {
    Q27_CHECK(d >= 0 && (!d || n <= std::numeric_limits<int64_t>::max() / d), "invalid tensor shape");
    if (d) n *= d;
  }
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  return n;
}

SafeTensors::SafeTensors(const std::string& dir) : dir_(dir) {
  try {
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
      const auto relative = fs_path(shard);
      Q27_CHECK(!shard.empty() && !relative.is_absolute() && shard.find_first_of("\\:") == std::string::npos,
                "invalid shard path");
      for (const auto& component : relative)
        Q27_CHECK(component != ".." && component != ".", "invalid shard path");
      std::string path = dir + "/" + shard;
      int fd = open_ro(path);
      if (fd < 0) fail("cannot open " + path);
      fds_.push_back(fd);
      const int64_t size = file_size(fd);
      Q27_CHECK(size >= 8, "truncated safetensors header");
      uint64_t hlen = 0;
      pread_all(fd, &hlen, 8, 0);
      Q27_CHECK(hlen <= 100000000 && hlen <= uint64_t(size - 8), "invalid safetensors header length");
      std::string header(hlen, '\0');
      pread_all(fd, header.data(), int64_t(hlen), 8);
      Json h = Json::parse(header);
      Q27_CHECK(h.kind == Json::Object, "safetensors header must be an object");
      std::vector<std::pair<int64_t, int64_t>> ranges;
      for (auto& [name, t] : h.obj) {
        if (name == "__metadata__") continue;
        TensorInfo ti;
        ti.dtype = t["dtype"].as_str();
        Q27_CHECK(t["shape"].kind == Json::Array && t["data_offsets"].kind == Json::Array &&
                      t["data_offsets"].arr.size() == 2, "invalid tensor metadata: " + name);
        for (auto& d : t["shape"].arr) ti.shape.push_back(d.as_int());
        ti.fd = fd;
        int64_t b = t["data_offsets"][0].as_int(), e = t["data_offsets"][1].as_int();
        Q27_CHECK(b >= 0 && e >= b && e <= std::numeric_limits<int64_t>::max() - 8 - int64_t(hlen),
                  "invalid tensor offsets: " + name);
        const int64_t elements = ti.numel(), item_bytes = dtype_bytes(ti.dtype);
        Q27_CHECK(elements <= std::numeric_limits<int64_t>::max() / item_bytes &&
                      e - b == elements * item_bytes, "tensor shape / dtype / byte size mismatch: " + name);
        ti.offset = 8 + int64_t(hlen) + b;
        ti.nbytes = e - b;
        ti.file_size = size;
        Q27_CHECK(tensors_.emplace(name, std::move(ti)).second, "duplicate tensor: " + name);
        ranges.emplace_back(b, e);
      }
      std::sort(ranges.begin(), ranges.end());
      int64_t end = 0;
      for (auto [b, e] : ranges) {
        Q27_CHECK(b == end, "overlapping or noncontiguous tensors");
        end = e;
      }
    }
  } catch (...) {
    for (int fd : fds_) close_fd(fd);
    throw;
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
  read_range(name, 0, t.nbytes, dst);
}

void SafeTensors::read_range(const std::string& name, int64_t begin, int64_t len, void* dst) const {
  const auto& t = info(name);
  Q27_CHECK(begin >= 0 && len >= 0 && begin <= t.nbytes && len <= t.nbytes - begin,
            "read_range out of bounds: " + name);
  Q27_CHECK(t.offset <= t.file_size && begin <= t.file_size - t.offset && len <= t.file_size - t.offset - begin,
            "truncated tensor data: " + name);
  pread_all(t.fd, dst, len, t.offset + begin);
}

}  // namespace q27
