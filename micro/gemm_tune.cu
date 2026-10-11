// P5-3: CUTLASS SM120 GEMM configurations (tile, schedule, output type) on the prefill shapes.
// Times only; the data is random (FP4 nibbles, FP8 bytes without NaN, scale factors 1.0).
//   gemm_tune [M ...]   (default M = 2048)
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template <class ElementA, class OpClass, int AlignAB, class Tile, class Schedule, class ElementD, class Scheduler>
struct Cfg {
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, Tile, Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto, float,
      float, ElementD, cutlass::layout::RowMajor, AlignC, ElementD, cutlass::layout::RowMajor, AlignC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, ElementA, cutlass::layout::RowMajor, AlignAB, ElementA,
      cutlass::layout::ColumnMajor, AlignAB, float, Tile, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Nvf4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using E4 = cutlass::float_e4m3_t;
using Auto = cutlass::gemm::collective::KernelScheduleAuto;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using SK = cutlass::gemm::StreamKScheduler;
using bf16 = cutlass::bfloat16_t;

static uint8_t *g_a, *g_b, *g_sfa, *g_sfb, *g_d, *g_ws;
static const size_t kWs = 64 << 20;

template <class G>
float run_fp4(int M, int N, int K, int iters) {
  using K_ = typename G::GemmKernel;
  using C = typename K_::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using DA = cutlass::float_e2m1_t;
  using ED = typename K_::ElementD;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  auto lA = C::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lB = C::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, 1},
                             {reinterpret_cast<DA*>(g_a), sA, reinterpret_cast<DA*>(g_b), sB,
                              reinterpret_cast<cutlass::float_ue4m3_t*>(g_sfa), lA,
                              reinterpret_cast<cutlass::float_ue4m3_t*>(g_sfb), lB},
                             {{1.f, 0.f}, reinterpret_cast<ED*>(g_d), sC, reinterpret_cast<ED*>(g_d), sD}};
  G g;
  if (G::get_workspace_size(args) > kWs || g.can_implement(args) != cutlass::Status::kSuccess) return -1;
  Timer t;
  float best = 1e9;
  for (int r = 0; r < 3; r++) {
    t.begin();
    for (int i = 0; i < iters; i++) {
      g.initialize(args, g_ws);
      g.run();
    }
    best = std::min(best, t.end() / iters);
  }
  CHECK(cudaGetLastError());
  return best;
}

template <class G>
float run_fp8(int M, int N, int K, int iters) {
  using K_ = typename G::GemmKernel;
  using ED = typename K_::ElementD;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, 1},
                             {reinterpret_cast<E4*>(g_a), sA, reinterpret_cast<E4*>(g_b), sB},
                             {{1.f, 0.f}, reinterpret_cast<ED*>(g_d), sC, reinterpret_cast<ED*>(g_d), sD}};
  G g;
  if (G::get_workspace_size(args) > kWs || g.can_implement(args) != cutlass::Status::kSuccess) return -1;
  Timer t;
  float best = 1e9;
  for (int r = 0; r < 3; r++) {
    t.begin();
    for (int i = 0; i < iters; i++) {
      g.initialize(args, g_ws);
      g.run();
    }
    best = std::min(best, t.end() / iters);
  }
  CHECK(cudaGetLastError());
  return best;
}

// MXFP8 (block-scaled E4M3, ue8m0 scales 1.0 = 0x7f): full-rate FP8 MMA on the RTX 5090 (P5-11).
template <class G>
float run_mx8(int M, int N, int K, int iters) {
  using K_ = typename G::GemmKernel;
  using C = typename K_::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ED = typename K_::ElementD;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  auto lA = C::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lB = C::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, 1},
                             {reinterpret_cast<E4*>(g_a), sA, reinterpret_cast<E4*>(g_b), sB,
                              reinterpret_cast<cutlass::float_ue8m0_t*>(g_sfa), lA,
                              reinterpret_cast<cutlass::float_ue8m0_t*>(g_sfb), lB},
                             {{1.f, 0.f}, reinterpret_cast<ED*>(g_d), sC, reinterpret_cast<ED*>(g_d), sD}};
  G g;
  if (G::get_workspace_size(args) > kWs || g.can_implement(args) != cutlass::Status::kSuccess) return -1;
  Timer t;
  float best = 1e9;
  for (int r = 0; r < 3; r++) {
    t.begin();
    for (int i = 0; i < iters; i++) {
      g.initialize(args, g_ws);
      g.run();
    }
    best = std::min(best, t.end() / iters);
  }
  CHECK(cudaGetLastError());
  return best;
}

struct Shp {
  const char* name;
  int N, K;
};

template <class G>
void fp4(const char* cfg, int M, const std::vector<Shp>& shapes) {
  for (auto& s : shapes) {
    float ms = run_fp4<G>(M, s.N, s.K, 20);
    double tf = ms > 0 ? 2.0 * M * s.N * s.K / (ms * 1e9) : 0;
    std::printf("fp4 %-28s M=%-5d %-9s %8.1f us %7.0f TFLOPS\n", cfg, M, s.name, ms * 1e3, tf);
  }
}

template <class G>
void fp8(const char* cfg, int M, const std::vector<Shp>& shapes) {
  for (auto& s : shapes) {
    float ms = run_fp8<G>(M, s.N, s.K, 20);
    double tf = ms > 0 ? 2.0 * M * s.N * s.K / (ms * 1e9) : 0;
    std::printf("fp8 %-28s M=%-5d %-9s %8.1f us %7.0f TFLOPS\n", cfg, M, s.name, ms * 1e3, tf);
  }
}

template <class G>
void mx8(const char* cfg, int M, const std::vector<Shp>& shapes) {
  for (auto& s : shapes) {
    float ms = run_mx8<G>(M, s.N, s.K, 20);
    double tf = ms > 0 ? 2.0 * M * s.N * s.K / (ms * 1e9) : 0;
    std::printf("mx8 %-28s M=%-5d %-9s %8.1f us %7.0f TFLOPS\n", cfg, M, s.name, ms * 1e3, tf);
  }
}

__global__ void fill(uint8_t* p, size_t n, uint32_t seed, uint8_t mask, uint8_t fixed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t x = uint32_t(i) * 2654435761u ^ seed;
    x ^= x >> 15, x *= 2246822519u, x ^= x >> 13;
    p[i] = fixed ? fixed : uint8_t(x) & mask;
  }
}

int main(int argc, char** argv) {
  std::vector<int> ms;
  for (int i = 1; i < argc; i++) ms.push_back(atoi(argv[i]));
  if (ms.empty()) ms.push_back(2048);
  const size_t big = size_t(16384) * 17408;  // largest M x K (fp8 bytes) or N x K
  CHECK(cudaMalloc(&g_a, big));
  CHECK(cudaMalloc(&g_b, size_t(17408) * 17408));
  CHECK(cudaMalloc(&g_sfa, big / 16));
  CHECK(cudaMalloc(&g_sfb, size_t(17408) * 17408 / 16));
  CHECK(cudaMalloc(&g_d, size_t(16384) * 34816 * 4));
  CHECK(cudaMalloc(&g_ws, kWs));
  // FP8: clear bit 6 so no byte is NaN; FP4 nibbles may be anything; scale factors 1.0 (0x38).
  fill<<<1024, 256>>>(g_a, big, 1, 0xBF, 0);
  fill<<<1024, 256>>>(g_b, size_t(17408) * 17408, 2, 0xBF, 0);
  fill<<<1024, 256>>>(g_sfa, big / 16, 0, 0, 0x38);
  fill<<<1024, 256>>>(g_sfb, size_t(17408) * 17408 / 16, 0, 0, 0x38);
  CHECK(cudaDeviceSynchronize());

  std::vector<Shp> f4 = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}, {"gate|up", 34816, 5120}};
  std::vector<Shp> f8 = {{"gdn qkv", 10240, 5120}, {"gdn z", 6144, 5120}, {"gdn qkvz", 16384, 5120},
                         {"gdn out", 5120, 6144},  {"attn q", 12288, 5120}, {"attn kv", 1024, 5120},
                         {"attn o", 5120, 6144}};
  const bool only_mx8 = getenv("ONLY_MX8") != nullptr;
  for (int M : ms) {
    using BS = cutlass::arch::OpClassBlockScaledTensorOp;
    using Mx8 = cutlass::mx_float8_t<E4>;
    // (the scale buffers hold 0x38 = ue4m3 1.0; refill them with ue8m0 1.0 for the MXFP8 runs)
    fill<<<1024, 256>>>(g_sfa, big / 16, 0, 0, 0x7f);
    fill<<<1024, 256>>>(g_sfb, size_t(17408) * 17408 / 16, 0, 0, 0x7f);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _128>, Auto, float, void>::Gemm>(
        "plain auto 128x128x128 f32", M, f8);
    mx8<Cfg<Mx8, BS, 16, Shape<_128, _128, _128>, Auto, float, void>::Gemm>("auto 128x128x128 f32", M, f8);
    mx8<Cfg<Mx8, BS, 16, Shape<_128, _128, _128>, Ping, float, void>::Gemm>("ping 128x128x128 f32", M, f8);
    mx8<Cfg<Mx8, BS, 16, Shape<_128, _128, _128>, Coop, float, void>::Gemm>("coop 128x128x128 f32", M, f8);
    mx8<Cfg<Mx8, BS, 16, Shape<_128, _128, _128>, Coop, float, SK>::Gemm>("coop 128x128x128 f32 sk", M, f8);
    mx8<Cfg<Mx8, BS, 16, Shape<_128, _128, _128>, Auto, bf16, void>::Gemm>("auto 128x128x128 bf16", M, f8);
    fill<<<1024, 256>>>(g_sfa, big / 16, 0, 0, 0x38);
    fill<<<1024, 256>>>(g_sfb, size_t(17408) * 17408 / 16, 0, 0, 0x38);
    if (only_mx8) continue;
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>, Auto, float, void>::Gemm>(
        "auto 128x128x128 f32", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>, Auto, bf16, void>::Gemm>(
        "auto 128x128x128 bf16", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>, Ping, float, void>::Gemm>(
        "ping 128x128x128 f32", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _256>, Coop, float, void>::Gemm>(
        "coop 128x128x256 f32", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _256>, Ping, float, void>::Gemm>(
        "ping 128x128x256 f32", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>, Coop, float, SK>::Gemm>(
        "coop 128x128x128 f32 sk", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _256, _128>, Coop, float, void>::Gemm>(
        "coop 128x256x128 f32", M, f4);
    fp4<Cfg<Nvf4, cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _256, _128>, Coop, bf16, void>::Gemm>(
        "coop 128x256x128 bf16", M, f4);

    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _128>, Auto, float, void>::Gemm>(
        "auto 128x128x128 f32", M, f8);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _128>, Auto, bf16, void>::Gemm>(
        "auto 128x128x128 bf16", M, f8);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _128>, Ping, float, void>::Gemm>(
        "ping 128x128x128 f32", M, f8);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _64>, Ping, float, void>::Gemm>(
        "ping 128x128x64 f32", M, f8);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _256, _64>, Coop, float, void>::Gemm>(
        "coop 128x256x64 f32", M, f8);
    fp8<Cfg<E4, cutlass::arch::OpClassTensorOp, 16, Shape<_128, _128, _128>, Coop, float, SK>::Gemm>(
        "coop 128x128x128 f32 sk", M, f8);
  }
  return 0;
}
