
# ============================================================
# Shared Benchmark Infrastructure
# ============================================================

add_executable(
    benchmark_smoke
    benchmark_smoke.cu
)

target_link_libraries(
    benchmark_smoke
    PRIVATE
        kernel_lab_config
        CUDA::cudart
)

target_compile_options(
    benchmark_smoke
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ------------------------------------------------------------
# GEMM V00 Benchmark
# ------------------------------------------------------------

add_executable(
    gemm_v00_benchmark
    gemm_v00_benchmark.cu
)

target_include_directories(
    gemm_v00_benchmark
    PRIVATE
        ${PROJECT_SOURCE_DIR}/operators/gemm/cuda
)

target_link_libraries(
    gemm_v00_benchmark
    PRIVATE
        gemm_v00
)

target_compile_options(
    gemm_v00_benchmark
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
