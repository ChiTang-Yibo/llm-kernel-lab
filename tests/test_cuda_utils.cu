
#include "cuda_utils.cuh"
#include <cstdio>

int main() {
    int device_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device found.\n");
        return 1;
    }

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);
    std::printf(
        "Compute Capability: %d.%d\n",
        prop.major,
        prop.minor
    );

    CUDA_CHECK_SYNC();

    std::printf("CUDA Utils Test PASSED\n");

    return 0;
}
