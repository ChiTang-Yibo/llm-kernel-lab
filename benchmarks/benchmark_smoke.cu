
#include "benchmark_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

// ------------------------------------------------------------
// Vector Add Kernel
// Used only for benchmark infrastructure validation.
// ------------------------------------------------------------

__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int THREADS = 256;

    const int blocks =
        (N + THREADS - 1) / THREADS;

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(float);

    // --------------------------------------------------------
    // Host input
    // --------------------------------------------------------

    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // --------------------------------------------------------
    // Device allocation
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A), bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B), bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C), bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes,
        cudaMemcpyHostToDevice
    ));

    // --------------------------------------------------------
    // Kernel launch callback
    // --------------------------------------------------------

    auto launch = [&]() {
        vector_add_kernel<<<blocks, THREADS>>>(
            d_A, d_B, d_C, N
        );
    };

    // --------------------------------------------------------
    // Correctness verification (outside timing)
    // --------------------------------------------------------

    launch();

    CUDA_CHECK_KERNEL();
    CUDA_CHECK_SYNC();

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes,
        cudaMemcpyDeviceToHost
    ));

    const bool correct = std::all_of(
        h_C.begin(),
        h_C.end(),
        [](float x) {
            return x == 3.0f;
        }
    );

    if (!correct) {
        std::fprintf(
            stderr, "Benchmark smoke correctness FAILED\n"
        );

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        return 1;
    }

    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    kernel_lab::BenchmarkConfig config;

    config.warmup = 10;
    config.samples = 30;
    config.iterations_per_sample = 20;

    const auto result =
        kernel_lab::benchmark_cuda(launch, config);

    // One floating-point addition per element
    const double flops = static_cast<double>(N);

    // Logical memory traffic:
    // 2 FP32 input loads + 1 FP32 output store
    const double logical_bytes =
        3.0 * static_cast<double>(bytes);

    const double gflops =
        kernel_lab::calculate_gflops(
            flops, result.median_us
        );

    const double effective_gbps =
        kernel_lab::calculate_bandwidth_gbps(
            logical_bytes, result.median_us
        );

    // --------------------------------------------------------
    // Device metadata
    // --------------------------------------------------------

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // --------------------------------------------------------
    // Console output
    // --------------------------------------------------------

    std::printf("GPU: %s\n", prop.name);
    std::printf("Vector size: %d\n", N);
    std::printf("Median latency: %.3f us\n",
                result.median_us);
    std::printf("Min latency: %.3f us\n",
                result.min_us);
    std::printf("Max latency: %.3f us\n",
                result.max_us);
    std::printf("GFLOP/s: %.3f\n", gflops);
    std::printf("Effective bandwidth: %.3f GB/s\n",
                effective_gbps);

    // --------------------------------------------------------
    // CSV output
    // --------------------------------------------------------

    kernel_lab::BenchmarkCsvRow row{
        "vector_add",
        "smoke",
        "N=1048576",
        "fp32",
        prop.name,
        config.warmup,
        config.samples,
        config.iterations_per_sample,
        result.median_us,
        result.min_us,
        result.max_us,
        gflops,
        effective_gbps
    };

    kernel_lab::append_benchmark_csv(
        "results/tables/benchmark_smoke.csv",
        row
    );

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::printf("Benchmark Smoke Test PASSED\n");

    return 0;
}
