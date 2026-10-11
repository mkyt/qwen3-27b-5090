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
find_package(CUDAToolkit REQUIRED)

# One shared library with a C ABI: used by the Python frontend and by the kernel tests (ctypes).
add_library(q27 SHARED
  src/json.cc
  src/safetensors.cc
  src/memory.cc
  src/weights.cc
  src/kernels.cu
  src/gemm.cu
  src/gemm_bf16.cu
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
target_include_directories(q27 PRIVATE src ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)
target_link_libraries(q27 PRIVATE CUDA::cudart)
# CUTLASS kernels need the relaxed constexpr flag and generate a lot of template code.
set_source_files_properties(src/gemm.cu PROPERTIES COMPILE_OPTIONS "-DCUTLASS_ENABLE_GDC_FOR_SM100=0")

add_executable(q27-sfcheck tools/sfcheck.cu)
target_link_libraries(q27-sfcheck PRIVATE CUDA::cudart)
target_include_directories(q27-sfcheck PRIVATE src ${CUTLASS_DIR}/include)
