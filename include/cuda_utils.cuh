
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace kernel_lab {

inline void check_cuda(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line
) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "[CUDA ERROR]\n"
            "  Expression: %s\n"
            "  File: %s\n"
            "  Line: %d\n"
            "  Error: %s\n",
            expression,
            file,
            line,
            cudaGetErrorString(error)
        );

        std::exit(EXIT_FAILURE);
    }
}

}  // namespace kernel_lab

// CUDA Runtime API error checking
#define CUDA_CHECK(call)                              \
    do {                                              \
        ::kernel_lab::check_cuda(                     \
            (call), #call, __FILE__, __LINE__);       \
    } while (0)

// Check kernel launch errors without synchronization
#define CUDA_CHECK_KERNEL()                           \
    CUDA_CHECK(cudaGetLastError())

// Synchronize and check asynchronous execution errors
// Use in correctness tests, not inside timed regions.
#define CUDA_CHECK_SYNC()                             \
    CUDA_CHECK(cudaDeviceSynchronize())
