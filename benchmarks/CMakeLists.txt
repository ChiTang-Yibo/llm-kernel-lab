
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
