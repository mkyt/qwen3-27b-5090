// BF16 GEMM for many rows (the draft's prompt context, up to a prefill chunk): CUTLASS 2.x SM80
// tensor-op kernel (mma.sync m16n8k16, also native on SM120). A = activations [M][K] row-major,
// W = weights [N][K] (K contiguous, i.e. column-major B), D fp32 [M][N].
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "common.h"
#include "kernels.h"

namespace q27 {
namespace {

using Bf16Gemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

}  // namespace

void gemm_bf16_cutlass(int M, int N, int K, const bf16* a, const bf16* w, float alpha, const float* c, float beta, float* d,
               cudaStream_t s) {
  if (M == 0) return;
  using E = cutlass::bfloat16_t;
  Bf16Gemm::Arguments args({M, N, K}, {reinterpret_cast<const E*>(a), K}, {reinterpret_cast<const E*>(w), K},
                           {c ? c : d, N}, {d, N}, {alpha, beta});
  Bf16Gemm gemm;
  auto st = gemm.can_implement(args);
  Q27_CHECK(st == cutlass::Status::kSuccess, "gemm_bf16: can_implement failed");
  st = gemm(args, nullptr, s);
  Q27_CHECK(st == cutlass::Status::kSuccess, "gemm_bf16: run failed");
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
