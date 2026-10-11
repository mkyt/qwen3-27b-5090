// Programmatic dependent launch (PDL) for the decode-path kernels: a kernel launched with
// launch_pdl may start while its predecessor is still running. Every such kernel must call
// pdl_wait() before it touches anything the predecessor writes, and calls pdl_trigger() to let its
// own successor start early. Weight prefetches (independent of the predecessor) go before the wait.
// Kernels trigger first thing: the successor may then launch while this kernel still waits, so a
// GEMM two launches ahead can stream its weights during the tail of the running one. Correctness is
// transitive (each wait covers the whole predecessor, which itself waited), and no CTA can starve:
// a successor launches only once every CTA of its predecessor is resident and has triggered.
#pragma once

#include "common.h"

namespace q27 {

bool pdl_enabled();  // Q27_NO_PDL=1 turns the launch attribute off (kernels.cu)

#ifdef Q27_TRACE
// Debug timeline (build with -DQ27_TRACE, see trace_step_end in kernels.cu): thread 0 of every CTA
// records (grid id, CTA, kind, %globaltimer) at the PDL trigger (kind 0), after the PDL wait (1) and
// at points a kernel marks itself (trace_point). Each translation unit has its own device pointer.
constexpr int kTraceMax = 1 << 20;
using TraceSetter = void (*)(unsigned long long*);
int trace_register(TraceSetter f);
void trace_launch(const void* kernel, dim3 grid);
namespace {
__device__ unsigned long long* g_trace;  // [0]: record count, records of 4 from [8] on
__device__ __forceinline__ void trace_point(int kind) {
  if (threadIdx.x == 0 && g_trace) {
    unsigned long long t, gid;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gid));
    unsigned long long i = atomicAdd(g_trace, 1ull);
    if (i < kTraceMax) {
      unsigned long long* r = g_trace + 8 + i * 4;
      r[0] = gid, r[1] = blockIdx.x + uint64_t(blockIdx.y) * gridDim.x, r[2] = kind, r[3] = t;
    }
  }
}
void trace_set(unsigned long long* p) { Q27_CUDA(cudaMemcpyToSymbol(g_trace, &p, sizeof p)); }
[[maybe_unused]] const int trace_reg = trace_register(trace_set);
}  // namespace
#else
__device__ __forceinline__ void trace_point(int) {}
#endif

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  trace_point(1);
}
__device__ __forceinline__ void pdl_trigger() {
  trace_point(0);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// a[z] of a small parameter array (z < 8) by a tree of selects: a dynamic index into a kernel parameter
// goes through the stack.
template <class T, int N>
__device__ __forceinline__ T lane_sel(const T (&a)[N], int z) {
  static_assert(N == 8, "lane arrays have kMaxLanes entries");
  const T lo = z < 2 ? (z == 0 ? a[0] : a[1]) : (z == 2 ? a[2] : a[3]);
  const T hi = z < 6 ? (z == 4 ? a[4] : a[5]) : (z == 6 ? a[6] : a[7]);
  return z < 4 ? lo : hi;
}

template <class... KArgs, class... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = s != nullptr && pdl_enabled();  // not on the legacy stream
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
#ifdef Q27_TRACE
  trace_launch(reinterpret_cast<const void*>(kernel), grid);
#endif
  Q27_CUDA(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

// launch_pdl with thread-block clusters of `cluster` CTAs along x.
template <class... KArgs, class... Args>
void launch_pdl_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, int cluster,
                        Args... args) {
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = s != nullptr && pdl_enabled();
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = cluster;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
#ifdef Q27_TRACE
  trace_launch(reinterpret_cast<const void*>(kernel), grid);
#endif
  Q27_CUDA(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

}  // namespace q27
