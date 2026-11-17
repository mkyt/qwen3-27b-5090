// Error checks shared by the CPU file readers and the CUDA engine.
#pragma once

#include <string>

namespace q27 {
[[noreturn]] void fail(const std::string& msg);
}

#define Q27_CHECK(cond, msg)                                         \
  do {                                                               \
    if (!(cond)) ::q27::fail(std::string("check failed: ") + (msg)); \
  } while (0)
