
#pragma once

#include "cuda_utils.cuh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_lab {

// ------------------------------------------------------------
// Benchmark configuration
// ------------------------------------------------------------

struct BenchmarkConfig {
    int warmup = 10;
    int samples = 30;
    int iterations_per_sample = 20;
};

// ------------------------------------------------------------
// Benchmark result
// ------------------------------------------------------------

struct BenchmarkResult {
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

// ------------------------------------------------------------
// CUDA Event benchmarking
//
// Requirements:
// - Memory allocations are completed before timing.
// - The launch callback uses the specified CUDA stream.
// - Each iteration executes the same computational workload.
// - No per-launch synchronization inside the callback.
// ------------------------------------------------------------

template <typename LaunchFn>
BenchmarkResult benchmark_cuda(
    LaunchFn&& launch,
    const BenchmarkConfig& config,
    cudaStream_t stream = 0
) {
    if (config.warmup < 0 ||
        config.samples <= 0 ||
        config.iterations_per_sample <= 0) {
        throw std::invalid_argument(
            "Invalid benchmark configuration"
        );
    }

    // Warmup
    for (int i = 0; i < config.warmup; ++i) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> timings_us;
    timings_us.reserve(config.samples);

    for (int sample = 0; sample < config.samples; ++sample) {

        CUDA_CHECK(cudaEventRecord(start, stream));

        for (int i = 0; i < config.iterations_per_sample; ++i) {
            launch();
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, start, stop
        ));

        const double latency_us =
            static_cast<double>(elapsed_ms) * 1000.0
            / config.iterations_per_sample;

        timings_us.push_back(latency_us);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(timings_us.begin(), timings_us.end());

    const std::size_t n = timings_us.size();

    BenchmarkResult result;
    result.min_us = timings_us.front();
    result.max_us = timings_us.back();

    if (n % 2 == 1) {
        result.median_us = timings_us[n / 2];
    } else {
        result.median_us =
            (timings_us[n / 2 - 1] + timings_us[n / 2]) / 2.0;
    }

    return result;
}

// ------------------------------------------------------------
// Performance metrics
// ------------------------------------------------------------

// FLOPs / second, expressed as GFLOP/s
inline double calculate_gflops(
    double flops,
    double latency_us
) {
    return flops / (latency_us * 1.0e3);
}

// Bytes / second, expressed as decimal GB/s
inline double calculate_bandwidth_gbps(
    double bytes,
    double latency_us
) {
    return bytes / (latency_us * 1.0e3);
}

// ------------------------------------------------------------
// Benchmark CSV record
// ------------------------------------------------------------

struct BenchmarkCsvRow {
    std::string op;
    std::string version;
    std::string shape;
    std::string dtype;
    std::string gpu;

    int warmup;
    int samples;
    int iterations_per_sample;

    double median_us;
    double min_us;
    double max_us;
    double gflops;
    double effective_gbps;
};

// Escape strings for CSV formatting.
inline std::string csv_quote(const std::string& value) {
    std::string result = "\"";

    for (char c : value) {
        if (c == '"') {
            result += "\"\"";
        } else {
            result += c;
        }
    }

    result += '"';
    return result;
}

// ------------------------------------------------------------
// Append benchmark result to CSV
// ------------------------------------------------------------

inline void append_benchmark_csv(
    const std::filesystem::path& path,
    const BenchmarkCsvRow& row
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(
            path.parent_path()
        );
    }

    const bool has_content =
        std::filesystem::exists(path) &&
        std::filesystem::file_size(path) > 0;

    std::ofstream file(path, std::ios::app);

    if (!file) {
        throw std::runtime_error(
            "Failed to open benchmark CSV"
        );
    }

    if (!has_content) {
        file
            << "operator,version,shape,dtype,gpu,"
            << "warmup,samples,iterations_per_sample,"
            << "median_us,min_us,max_us,"
            << "gflops,effective_gbps\n";
    }

    file << std::fixed << std::setprecision(6);

    file
        << csv_quote(row.op) << ','
        << csv_quote(row.version) << ','
        << csv_quote(row.shape) << ','
        << csv_quote(row.dtype) << ','
        << csv_quote(row.gpu) << ','
        << row.warmup << ','
        << row.samples << ','
        << row.iterations_per_sample << ','
        << row.median_us << ','
        << row.min_us << ','
        << row.max_us << ','
        << row.gflops << ','
        << row.effective_gbps << '\n';
}

}  // namespace kernel_lab
