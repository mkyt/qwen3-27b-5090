#include "json.h"

#include <cstring>

#include "common.h"

namespace q27 {
namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;

  [[noreturn]] void error(const char* what) { fail(std::string("json: ") + what + " at offset " + std::to_string(i)); }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++;
  }
  char peek() {
    ws();
    if (i >= s.size()) error("unexpected end");
    return s[i];
  }
  void expect(char c) {
    if (peek() != c) error("unexpected character");
    i++;
  }
  bool literal(const char* lit) {
    size_t n = strlen(lit);
    if (s.compare(i, n, lit) == 0) {
      i += n;
      return true;
    }
    return false;
  }

  static void utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += char(cp);
    } else if (cp < 0x800) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }

  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) error("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      char e = s[i++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': utf8(out, unsigned(std::stoul(s.substr(i, 4), nullptr, 16))); i += 4; break;
        default: out += e;
      }
    }
    return out;
  }

  Json value() {
    Json v;
    char c = peek();
    if (c == '{') {
      v.kind = Json::Object;
      i++;
      if (peek() == '}') {
        i++;
        return v;
      }
      while (true) {
        std::string k = string();
        expect(':');
        v.obj.emplace(std::move(k), value());
        if (peek() == ',') {
          i++;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.kind = Json::Array;
      i++;
      if (peek() == ']') {
        i++;
        return v;
      }
      while (true) {
        v.arr.push_back(value());
        if (peek() == ',') {
          i++;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind = Json::String;
      v.str = string();
      return v;
    }
    if (literal("true")) {
      v.kind = Json::Bool;
      v.b = true;
      return v;
    }
    if (literal("false")) {
      v.kind = Json::Bool;
      return v;
    }
    if (literal("null")) return v;
    size_t end = 0;
    v.kind = Json::Number;
    v.num = std::stod(s.substr(i, 64), &end);
    if (end == 0) error("bad value");
    i += end;
    return v;
  }
};

const Json& null_json() {
  static Json n;
  return n;
}

}  // namespace

Json Json::parse(const std::string& text) {
  Parser p{text};
  return p.value();
}

const Json& Json::operator[](const std::string& key) const {
  if (kind != Object) fail("json: not an object (key " + key + ")");
  auto it = obj.find(key);
  if (it == obj.end()) fail("json: missing key " + key);
  return it->second;
}

const Json& Json::operator[](size_t idx) const {
  if (kind != Array || idx >= arr.size()) fail("json: bad index");
  return arr[idx];
}

int64_t Json::as_int() const {
  if (kind != Number) fail("json: not a number");
  return int64_t(num);
}

double Json::as_num() const {
  if (kind != Number) fail("json: not a number");
  return num;
}

const std::string& Json::as_str() const {
  if (kind != String) fail("json: not a string");
  return str;
}

}  // namespace q27
