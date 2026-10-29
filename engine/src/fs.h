// Paths in UTF-8: the C API takes them from Python's str.encode(), and env_path reads the environment as UTF-8.
// On Windows the narrow file APIs (fopen, std::ifstream of a char string, _open) and getenv use the ANSI code
// page instead, so a path with e.g. Japanese characters would not open; fs_path makes the wide path there.
#pragma once

#include <filesystem>
#include <string>
#ifdef _WIN32
#include <cstdlib>
#endif

namespace q27 {

inline std::filesystem::path fs_path(const std::string& utf8) {
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

// A path-valued environment variable in UTF-8; empty when unset.
inline std::string env_path(const char* name) {
#ifdef _WIN32
  const std::wstring wname(name, name + std::char_traits<char>::length(name));  // (ASCII names)
  const wchar_t* v = _wgetenv(wname.c_str());
  if (!v) return {};
  const std::u8string u = std::filesystem::path(v).u8string();
  return std::string(u.begin(), u.end());
#else
  const char* v = getenv(name);
  return v ? v : "";
#endif
}

}  // namespace q27
