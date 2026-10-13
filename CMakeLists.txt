
cmake_minimum_required(VERSION 3.28)

# Default GPU architecture: NVIDIA Blackwell SM 120
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING
    "CUDA GPU architectures")

project(
    llm_kernel_lab
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Toolkit
find_package(CUDAToolkit 12.8 REQUIRED)

# Shared include directories
add_library(kernel_lab_config INTERFACE)

target_include_directories(
    kernel_lab_config INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

# CTest support
enable_testing()

message(STATUS "==================================")
message(STATUS "LLM Kernel Performance Lab")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==================================")

# Operator targets will be added incrementally.
# Example after GEMM V00 implementation:
#
add_subdirectory(operators/gemm)
# ------------------------------------------------------------
# Common CUDA utility tests
# ------------------------------------------------------------

add_executable(
    cuda_utils_test
    tests/test_cuda_utils.cu
)

target_link_libraries(
    cuda_utils_test
    PRIVATE
        kernel_lab_config
        CUDA::cudart
)

add_test(
    NAME cuda_utils_test
    COMMAND cuda_utils_test
)

# ------------------------------------------------------------
# Shared benchmarks
# ------------------------------------------------------------

add_subdirectory(benchmarks)