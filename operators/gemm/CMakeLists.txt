
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
# 4. GEMM correctness tests
# ------------------------------------------------------------

# CTest targets will be registered incrementally.
