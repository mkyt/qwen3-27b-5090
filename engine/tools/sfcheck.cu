// Checks q27::sf_offset (the swizzled NVFP4 scale-factor layout used by quant_fp4 and the loader)
// against CUTLASS's own Sm1xxBlockScaledConfig layout for the shapes this model uses. Host only.
#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "kernels.h"

int main() {
  using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;
  struct Shape {
    int rows, k;
  } shapes[] = {{1, 5120}, {130, 5120}, {2048, 17408}, {5120, 17408}, {17408, 5120}, {248320, 5120}, {200, 64}};
  int bad = 0;
  for (auto [rows, K] : shapes) {
    auto layout = Cfg::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, K, 1));
    int kb = K / 16;
    for (int r = 0; r < rows; r += (rows > 4096 ? 7 : 1))
      for (int b = 0; b < kb; b++) {
        int64_t want = layout(r, b * 16, 0);
        int64_t got = q27::sf_offset(r, b, kb);
        if (want != got && bad++ < 10) printf("mismatch rows=%d K=%d at (%d, %d): cute %ld, ours %ld\n", rows, K, r, b, (long)want, (long)got);
      }
    int64_t size = cute::cosize(cute::filter_zeros(layout));
    if (size > q27::sf_bytes(rows, kb) && bad++ < 10)
      printf("size rows=%d K=%d: cute %ld > ours %ld\n", rows, K, (long)size, (long)q27::sf_bytes(rows, kb));
  }
  printf(bad ? "FAIL (%d)\n" : "OK\n", bad);
  return bad != 0;
}
