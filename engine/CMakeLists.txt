cmake_minimum_required(VERSION 3.28)
project(qwen38_engine LANGUAGES CXX CUDA)

# sm_120a: arch-specific features (block-scaled FP4 MMA) are only available on the "a" target.
set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")

set(CUTLASS_DIR $ENV{CUTLASS_DIR} CACHE PATH "CUTLASS checkout")
# The CUTLASS 3.x prefill GEMMs (src/gemm.cu), kept for comparison (Q27_GEMM_CUTLASS=1): CuTe does not compile with
# MSVC, so a Windows build turns them off. The engine's own GEMMs (src/gemm_mma.cu) run either way.
option(Q27_CUTLASS_GEMM "Build the CUTLASS 3.x prefill GEMMs (gemm.cu)" ON)
find_package(CUDAToolkit REQUIRED)

# One shared library with a C ABI: used by the Python frontend and by the kernel tests (ctypes).
add_library(q27 SHARED
  src/json.cc
  src/safetensors.cc
  src/prefetch.cc
  src/memory.cc
  src/weights.cc
  src/kernels.cu
  src/gemm_mma.cu
  src/skinny.cu
  src/gdn.cu
  src/attention.cu
  src/sampling.cu
  src/draft.cu
  src/draft.cc
  src/vision.cu
  src/model.cc
  src/capi.cc
)
target_include_directories(q27 PRIVATE src)
# SYSTEM: no warnings from CUTLASS's own headers (e.g. #2908-D, implicit capture of `this`, in the SM100 / SM103 /
# SM107 collectives the builder header includes); the generated code is the same.
target_include_directories(q27 SYSTEM PRIVATE ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)
target_link_libraries(q27 PRIVATE CUDA::cudart)
if(Q27_CUTLASS_GEMM)
  target_sources(q27 PRIVATE src/gemm.cu src/gemm_bf16.cu)
  target_compile_definitions(q27 PRIVATE Q27_CUTLASS_GEMM)
  set_source_files_properties(src/gemm.cu PROPERTIES COMPILE_OPTIONS "-DCUTLASS_ENABLE_GDC_FOR_SM100=0")
endif()

if(Q27_CUTLASS_GEMM)  # (checks sf_offset against CuTe's layout)
  add_executable(q27-sfcheck tools/sfcheck.cu)
  target_link_libraries(q27-sfcheck PRIVATE CUDA::cudart)
  target_include_directories(q27-sfcheck PRIVATE src)
  target_include_directories(q27-sfcheck SYSTEM PRIVATE ${CUTLASS_DIR}/include)
endif()
