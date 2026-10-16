
# ============================================================
# GEMM - General Matrix Multiplication
# LLM Kernel Performance Lab
#
# Each CUDA implementation is preserved as an independent
# version. All versions share the same build configuration.
# ============================================================

# ------------------------------------------------------------
# 1. Shared GEMM build configuration
# ------------------------------------------------------------

add_library(gemm_build_config INTERFACE)

target_link_libraries(
    gemm_build_config
    INTERFACE
        kernel_lab_config
        CUDA::cudart
)

# ------------------------------------------------------------
# 2. CUDA profiling configuration
# ------------------------------------------------------------

# Preserve source line information for Nsight Compute
# while keeping Release optimization enabled.

target_compile_options(
    gemm_build_config
    INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ------------------------------------------------------------
# 3. GEMM version targets
# ------------------------------------------------------------

# Version targets will be added after the shared
# correctness and benchmark infrastructure is ready.
#
# Example:
#
# add_executable(gemm_v00 cuda/v00_naive.cu)
#
# target_link_libraries(
#     gemm_v00
#     PRIVATE
#         gemm_build_config
# )


# ------------------------------------------------------------
# GEMM V00 - Naive FP32
# ------------------------------------------------------------

add_library(
    gemm_v00
    STATIC
    cuda/v00_naive.cu
)

target_link_libraries(
    gemm_v00
    PUBLIC
        gemm_build_config
)

# ------------------------------------------------------------
# GEMM V00 Correctness Runner
# ------------------------------------------------------------

add_executable(
    gemm_v00_test
    tests/test_v00.cu
)

target_include_directories(
    gemm_v00_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_link_libraries(
    gemm_v00_test
    PRIVATE
        gemm_v00
)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME gemm_v00_smoke
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_v00_suite.py
        --runner
        $<TARGET_FILE:gemm_v00_test>
        --suite smoke
)
# ------------------------------------------------------------
# 4. GEMM correctness tests
# ------------------------------------------------------------

# CTest targets will be registered incrementally.
