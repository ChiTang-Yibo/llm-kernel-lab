
#include "benchmark_utils.cuh"
#include "v00_naive.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {

    const bool profile_once =
        argc == 5 && std::string(argv[4]) == "--profile-once";

    if (argc != 4 && !profile_once) {
        std::fprintf(
            stderr,
            "Usage: %s M N K\n",
            argv[0]
        );
        return 1;
    }

    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);

    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "Invalid dimensions\n");
        return 1;
    }

    // --------------------------------------------------------
    // Matrix sizes
    // --------------------------------------------------------

    const std::size_t count_A =
        static_cast<std::size_t>(M) * K;

    const std::size_t count_B =
        static_cast<std::size_t>(K) * N;

    const std::size_t count_C =
        static_cast<std::size_t>(M) * N;

    const std::size_t bytes_A =
        count_A * sizeof(float);

    const std::size_t bytes_B =
        count_B * sizeof(float);

    const std::size_t bytes_C =
        count_C * sizeof(float);

    // --------------------------------------------------------
    // Host input: deterministic FP32 values
    // --------------------------------------------------------

    std::vector<float> h_A(count_A);
    std::vector<float> h_B(count_B);

    for (std::size_t i = 0; i < count_A; ++i) {
        h_A[i] =
            static_cast<float>(
                static_cast<int>(i % 23) - 11
            ) / 11.0f;
    }

    for (std::size_t i = 0; i < count_B; ++i) {
        h_B[i] =
            static_cast<float>(
                static_cast<int>(i % 19) - 9
            ) / 9.0f;
    }

    // --------------------------------------------------------
    // GPU allocation
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A), bytes_A
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B), bytes_B
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C), bytes_C
    ));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes_A,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes_B,
        cudaMemcpyHostToDevice
    ));

    // --------------------------------------------------------
    // CUDA stream
    // --------------------------------------------------------

    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking
    ));

    // --------------------------------------------------------
    // Kernel launch configuration
    // --------------------------------------------------------

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    auto launch = [&]() {
        gemm_v00_kernel<<<grid, block, 0, stream>>>(
            d_A, d_B, d_C, M, N, K
        );
    };

    // --------------------------------------------------------
    // Nsight Compute profiling mode
    // --------------------------------------------------------

    if (profile_once) {

        // Warm up the GPU.
        for (int i = 0; i < 10; ++i) {
            launch();
        }

        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // One additional kernel launch for profiling.
        launch();

        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        std::printf(
            "GEMM V00 Profiling Run Completed\n"
        );

        return 0;
    }


    // --------------------------------------------------------
    // Benchmark configuration
    // --------------------------------------------------------

    kernel_lab::BenchmarkConfig config;

    config.warmup = 10;
    config.samples = 30;
    config.iterations_per_sample = 20;

    const auto result =
        kernel_lab::benchmark_cuda(
            launch,
            config,
            stream
        );

    // --------------------------------------------------------
    // GEMM performance metrics
    // --------------------------------------------------------

    const double flops =
        2.0 * M * N * K;

    const double gflops =
        kernel_lab::calculate_gflops(
            flops,
            result.median_us
        );

    // Minimum input/output traffic:
    // read A once, read B once, write C once.
    //
    // This is a model-based metric, not measured DRAM traffic.

    const double minimum_bytes =
        static_cast<double>(bytes_A) +
        static_cast<double>(bytes_B) +
        static_cast<double>(bytes_C);

    const double effective_gbps =
        kernel_lab::calculate_bandwidth_gbps(
            minimum_bytes,
            result.median_us
        );

    // --------------------------------------------------------
    // GPU metadata
    // --------------------------------------------------------

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // --------------------------------------------------------
    // Console output
    // --------------------------------------------------------

    std::printf("\nGEMM V00 Benchmark\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("Shape: M=%d N=%d K=%d\n", M, N, K);
    std::printf("Block: (%u, %u)\n",
                block.x, block.y);

    std::printf(
        "Median latency: %.3f us\n",
        result.median_us
    );

    std::printf(
        "Min latency: %.3f us\n",
        result.min_us
    );

    std::printf(
        "Max latency: %.3f us\n",
        result.max_us
    );

    std::printf(
        "Performance: %.3f GFLOP/s\n",
        gflops
    );

    std::printf(
        "Minimum-I/O effective rate: %.3f GB/s\n",
        effective_gbps
    );

    // --------------------------------------------------------
    // CSV output
    // --------------------------------------------------------

    const std::string shape =
        "M=" + std::to_string(M) +
        ",N=" + std::to_string(N) +
        ",K=" + std::to_string(K);

    kernel_lab::BenchmarkCsvRow row{
        "gemm",
        "v00_naive",
        shape,
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
        "results/tables/gemm_v00.csv",
        row
    );

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::printf("GEMM V00 Benchmark PASSED\n");

    return 0;
}
