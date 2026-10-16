
#pragma once

#include <cuda_runtime.h>

// GEMM V00 - Naive FP32
// A: [M, K]
// B: [K, N]
// C: [M, N]
// All matrices are contiguous and row-major.

__global__ void gemm_v00_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
