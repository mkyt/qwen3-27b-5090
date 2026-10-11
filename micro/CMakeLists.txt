cmake_minimum_required(VERSION 3.28)
project(q27_micro LANGUAGES CXX CUDA)

# sm_120a: arch-specific features (block-scaled FP4 MMA) are only available on the "a" target.
set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -Xptxas=-v")

foreach(name bw mma_peak pcie meminfo mma_fp8_blockscale_eq)
  add_executable(${name} ${name}.cu)
endforeach()
