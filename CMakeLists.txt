cmake_minimum_required(VERSION 3.18)
project(scout_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU architecture; default to sm_80 (A100) if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Core attention library
add_library(attention_kernels STATIC
    src/naive_attention.cu
    src/tiled_attention.cu
    src/scout_score.cu
    src/scout_attention.cu
)
set_target_properties(attention_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE attention_kernels)

# Benchmark
add_executable(bench_attention benchmarks/bench_attention.cu)
target_link_libraries(bench_attention PRIVATE attention_kernels)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
