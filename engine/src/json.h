// Minimal JSON reader for config.json and safetensors headers (objects, arrays, strings, numbers,
// true/false/null). Not a general-purpose parser: no \u surrogate pairs beyond the BMP.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace q27 {

struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  static Json parse(const std::string& text);

  bool has(const std::string& key) const { return kind == Object && obj.count(key); }
  const Json& operator[](const std::string& key) const;
  const Json& operator[](size_t i) const;
  int64_t as_int() const;
  double as_num() const;
  const std::string& as_str() const;
};

}  // namespace q27
