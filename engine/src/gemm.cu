// Phase 1 GEMMs on CUTLASS SM120 kernels (one tile config for every M; the skinny decode GEMMs
// come in Phase 2). A = activations [M][K] row-major, B = checkpoint weights [N][K] (K contiguous),
// C / D fp32 row-major [M][N]. Since P5-12 the prefill runs gemm_mma.cu's kernels; these remain for comparison
// (Q27_GEMM_CUTLASS=1) in builds with Q27_CUTLASS_GEMM (CUTLASS 3.x / CuTe do not compile with MSVC).
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "common.h"
#include "kernels.h"

namespace q27 {
namespace {

using namespace cute;

template <class ElementA, class ElementB, class OpClass, int AlignAB, class Tile, class ElementOut = float>
struct GemmConfig {
  using ElementC = ElementOut;
  using ElementD = ElementOut;
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, Tile, Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto, float,
      float, ElementC, cutlass::layout::RowMajor, AlignC, ElementD, cutlass::layout::RowMajor, AlignC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, ElementA, cutlass::layout::RowMajor, AlignAB, ElementB,
      cutlass::layout::ColumnMajor, AlignAB, float, Tile, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Fp4 = GemmConfig<cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                       cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>>;
// BF16 output (the MLP gate / up feed only silu * up: half the bytes written and read back).
using Fp4Bf16 = GemmConfig<cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                           cutlass::arch::OpClassBlockScaledTensorOp, 32, Shape<_128, _128, _128>, cutlass::bfloat16_t>;
using Fp8 = GemmConfig<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::arch::OpClassTensorOp, 16,
                       Shape<_128, _128, _128>>;
// The same E4M3 GEMM as block-scaled MXFP8 with every ue8m0 scale 1.0: the RTX 5090 runs the plain FP8
// MMA with FP32 accumulation at half rate and the block-scaled one at full rate, with bit-identical
// products (micro/mma_peak, micro/mma_fp8_blockscale_eq; the trick is NInfer's). The unit scales live
// at the end of the GEMM workspace (gemm_cutlass_workspace_init); Q27_GEMM_MXFP8=0 keeps the plain kernel.
using Mxfp8 = GemmConfig<cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::mx_float8_t<cutlass::float_e4m3_t>,
                         cutlass::arch::OpClassBlockScaledTensorOp, 16, Shape<_128, _128, _128>>;

constexpr size_t kCutlassWsBytes = 4 << 20;           // CUTLASS's own scratch at the start of the workspace
constexpr size_t kUnitSfBytes = size_t(16384) * 256;  // unit scales: rows (M or N) <= 16K, K <= 8K
const void* g_unit_ws = nullptr;                      // the workspace whose unit scales are filled

template <class Gemm>
void run(Gemm& gemm, typename Gemm::Arguments& args, void* ws, size_t ws_bytes, cudaStream_t s, const char* what) {
  size_t need = Gemm::get_workspace_size(args);
  Q27_CHECK(need <= ws_bytes, std::string(what) + ": workspace too small");
  auto st = gemm.can_implement(args);
  Q27_CHECK(st == cutlass::Status::kSuccess, std::string(what) + ": can_implement failed");
  st = gemm.initialize(args, ws, s);
  Q27_CHECK(st == cutlass::Status::kSuccess, std::string(what) + ": initialize failed");
  st = gemm.run(s);
  Q27_CHECK(st == cutlass::Status::kSuccess, std::string(what) + ": run failed");
  Q27_CUDA(cudaGetLastError());
}

template <class G, class E>
void run_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
             float alpha, const E* c, float beta, E* d, void* workspace, size_t ws_bytes, cudaStream_t s,
             const char* what) {
  if (M == 0) return;
  using K_ = typename G::GemmKernel;
  using Cfg = typename K_::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using DataA = typename cutlass::float_e2m1_t;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  auto lSFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lSFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename G::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const DataA*>(a), sA, reinterpret_cast<const DataA*>(b), sB,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(a_sf), lSFA,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(b_sf), lSFB},
      {{alpha, beta}, c ? c : d, sC, d, sD}};
  G gemm;
  run(gemm, args, workspace, ws_bytes, s, what);
}

}  // namespace

void gemm_fp4_cutlass(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
              float alpha, const float* c, float beta, float* d, void* workspace, size_t ws_bytes, cudaStream_t s) {
  run_fp4<Fp4::Gemm>(M, N, K, a, a_sf, b, b_sf, alpha, c, beta, d, workspace, ws_bytes, s, "gemm_fp4");
}

void gemm_fp4_bf16_cutlass(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                   float alpha, bf16* d, void* workspace, size_t ws_bytes, cudaStream_t s) {
  auto* dd = reinterpret_cast<cutlass::bfloat16_t*>(d);
  run_fp4<Fp4Bf16::Gemm>(M, N, K, a, a_sf, b, b_sf, alpha, static_cast<const cutlass::bfloat16_t*>(nullptr), 0.f, dd,
                         workspace, ws_bytes, s, "gemm_fp4_bf16");
}

namespace {

// MXFP8 when the workspace carries the unit scales and they cover the shape; returns false otherwise.
bool gemm_mxfp8(int M, int N, int K, const uint8_t* a, const uint8_t* b, float alpha, const float* c, float beta,
                float* d, void* workspace, size_t ws_bytes, cudaStream_t s) {
  const char* e = getenv("Q27_GEMM_MXFP8");  // (read per call: A/B in one process)
  const bool on = !e || atoi(e) != 0;
  if (!on || workspace != g_unit_ws || ws_bytes < kCutlassWsBytes + kUnitSfBytes) return false;
  using G = Mxfp8::Gemm;
  using K_ = typename G::GemmKernel;
  using Cfg = typename K_::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto lSFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lSFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  if (size_t(cosize(lSFA)) > kUnitSfBytes || size_t(cosize(lSFB)) > kUnitSfBytes) return false;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  auto* sf = reinterpret_cast<const cutlass::float_ue8m0_t*>(static_cast<const uint8_t*>(workspace) + kCutlassWsBytes);
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, 1},
                             {reinterpret_cast<const cutlass::float_e4m3_t*>(a), sA,
                              reinterpret_cast<const cutlass::float_e4m3_t*>(b), sB, sf, lSFA, sf, lSFB},
                             {{alpha, beta}, c ? c : d, sC, d, sD}};
  G gemm;
  run(gemm, args, workspace, kCutlassWsBytes, s, "gemm_fp8 (mxfp8)");
  return true;
}

}  // namespace

void gemm_fp8_cutlass(int M, int N, int K, const uint8_t* a, const uint8_t* b, float alpha, const float* c, float beta,
              float* d, void* workspace, size_t ws_bytes, cudaStream_t s) {
  if (M == 0) return;
  if (gemm_mxfp8(M, N, K, a, b, alpha, c, beta, d, workspace, ws_bytes, s)) return;
  using G = Fp8::Gemm;
  using K_ = typename G::GemmKernel;
  auto sA = cutlass::make_cute_packed_stride(typename K_::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, {M, N, 1});
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, 1},
                             {reinterpret_cast<const cutlass::float_e4m3_t*>(a), sA,
                              reinterpret_cast<const cutlass::float_e4m3_t*>(b), sB},
                             {{alpha, beta}, c ? c : d, sC, d, sD}};
  G gemm;
  run(gemm, args, workspace, ws_bytes, s, "gemm_fp8");
}

size_t gemm_cutlass_workspace_bytes(int max_m) {
  // The default (data-parallel) tile scheduler needs little or no workspace; reserve a fixed slab and
  // let run() check the actual requirement. The MXFP8 unit scales follow it.
  (void)max_m;
  return kCutlassWsBytes + kUnitSfBytes;
}

void gemm_cutlass_workspace_init(void* workspace, size_t ws_bytes, cudaStream_t s) {
  Q27_CHECK(ws_bytes >= kCutlassWsBytes + kUnitSfBytes, "gemm_cutlass_workspace_init: workspace too small");
  Q27_CUDA(cudaMemsetAsync(static_cast<uint8_t*>(workspace) + kCutlassWsBytes, 0x7f, kUnitSfBytes, s));  // ue8m0 1.0
  g_unit_ws = workspace;
}

}  // namespace q27
