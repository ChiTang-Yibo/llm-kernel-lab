
#include "cuda_utils.cuh"
#include "v00_naive.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// Read raw FP32 binary data exported by PyTorch.
// ------------------------------------------------------------

std::vector<float> load_fp32(
    const fs::path& path,
    std::size_t count
) {
    const std::size_t expected_bytes =
        count * sizeof(float);

    if (!fs::exists(path) ||
        fs::file_size(path) != expected_bytes) {
        throw std::runtime_error(
            "Missing or invalid fixture: " + path.string()
        );
    }

    std::vector<float> data(count);

    std::ifstream file(path, std::ios::binary);

    if (!file.read(
            reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(expected_bytes)
        )) {
        throw std::runtime_error(
            "Failed to read: " + path.string()
        );
    }

    return data;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "Usage: %s FIXTURE_DIR M N K ATOL RTOL\n",
            argv[0]
        );
        return 1;
    }

    try {
        const fs::path fixture_dir = argv[1];

        const int M = std::stoi(argv[2]);
        const int N = std::stoi(argv[3]);
        const int K = std::stoi(argv[4]);

        const double atol = std::stod(argv[5]);
        const double rtol = std::stod(argv[6]);

        if (M <= 0 || N <= 0 || K <= 0 ||
            atol < 0 || rtol < 0) {
            throw std::invalid_argument(
                "Invalid dimensions or tolerances"
            );
        }

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

        // ----------------------------------------------------
        // Load PyTorch fixtures
        // ----------------------------------------------------

        auto h_A = load_fp32(
            fixture_dir / "A.bin", count_A
        );

        auto h_B = load_fp32(
            fixture_dir / "B.bin", count_B
        );

        auto h_ref = load_fp32(
            fixture_dir / "C_ref.bin", count_C
        );

        std::vector<float> h_C(count_C);

        // ----------------------------------------------------
        // GPU memory allocation
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // Host to Device
        // ----------------------------------------------------

        CUDA_CHECK(cudaMemcpy(
            d_A, h_A.data(),
            bytes_A, cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_B, h_B.data(),
            bytes_B, cudaMemcpyHostToDevice
        ));

        // Initialize C with a recognizable byte pattern.
        CUDA_CHECK(cudaMemset(d_C, 0x7f, bytes_C));

        // ----------------------------------------------------
        // GEMM V00 Kernel Launch
        // ----------------------------------------------------

        dim3 block(16, 16);

        dim3 grid(
            (N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y
        );

        gemm_v00_kernel<<<grid, block>>>(
            d_A, d_B, d_C, M, N, K
        );

        CUDA_CHECK_KERNEL();
        CUDA_CHECK_SYNC();

        // ----------------------------------------------------
        // Device to Host
        // ----------------------------------------------------

        CUDA_CHECK(cudaMemcpy(
            h_C.data(), d_C,
            bytes_C, cudaMemcpyDeviceToHost
        ));

        // ----------------------------------------------------
        // Compare against PyTorch reference
        // ----------------------------------------------------

        std::size_t failures = 0;

        double max_abs_error = 0.0;
        double max_rel_error = 0.0;

        for (std::size_t idx = 0; idx < count_C; ++idx) {

            const double actual = h_C[idx];
            const double expected = h_ref[idx];

            const double abs_error =
                std::abs(actual - expected);

            const double rel_error =
                abs_error /
                std::max(std::abs(expected), 1.0e-12);

            max_abs_error =
                std::max(max_abs_error, abs_error);

            max_rel_error =
                std::max(max_rel_error, rel_error);

            const double tolerance =
                atol + rtol * std::abs(expected);

            if (!std::isfinite(actual) ||
                abs_error > tolerance) {

                ++failures;

                if (failures <= 5) {
                    std::printf(
                        "Mismatch at C[%zu,%zu]: "
                        "actual=%.8f expected=%.8f "
                        "abs_error=%.8e\n",
                        idx / static_cast<std::size_t>(N),
                        idx % static_cast<std::size_t>(N),
                        actual,
                        expected,
                        abs_error
                    );
                }
            }
        }

        // ----------------------------------------------------
        // Report
        // ----------------------------------------------------

        std::printf("\nGEMM V00 Correctness\n");
        std::printf("Shape: M=%d N=%d K=%d\n", M, N, K);
        std::printf("Elements checked: %zu\n", count_C);
        std::printf("Max absolute error: %.8e\n",
                    max_abs_error);
        std::printf("Max relative error: %.8e\n",
                    max_rel_error);
        std::printf("Failed elements: %zu\n", failures);

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        if (failures != 0) {
            std::printf("GEMM V00 FAILED\n");
            return 1;
        }

        std::printf("GEMM V00 PASSED\n");
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
