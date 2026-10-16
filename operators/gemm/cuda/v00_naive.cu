
#include "v00_naive.cuh"

// ============================================================
// GEMM V00 - Naive FP32
//
// A: [M, K], row-major
// B: [K, N], row-major
// C: [M, N], row-major
//
// One CUDA thread computes one output element.
// ============================================================

__global__ void gemm_v00_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // TODO 1: Derive the output row and column.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float value = 0.0f;
        for(int k =0; k < K; ++k){
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }

}
