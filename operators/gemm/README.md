# GEMM — CUDA Kernel Performance Lab

This module investigates how mathematical GEMM maps to CUDA threads, memory access patterns, and GPU microarchitecture. Versions are preserved independently and evaluated using the same correctness and benchmark methodology.

## Mathematical definition

For contiguous row-major FP32 inputs $A \in \mathbb{R}^{M\times K}$ and $B \in \mathbb{R}^{K\times N}$:

$$
C_{i,j} = \sum_{k=0}^{K-1} A_{i,k}B_{k,j}
$$

## Versions

| Version | Approach                                       | Status                                                       | Technical note                                       |
| ------- | ---------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **V00** | Naive FP32 SIMT; one thread per output element | Correctness **7/7**, benchmark and Nsight profiling completed | [V00 technical report](notes/GEMM_V00_Naive_FP32.md) |
| V01     | Thread / memory-access mapping comparison      | Planned; not yet measured                                    | —                                                    |
| V02     | Shared-memory tiled GEMM                       | Planned; not yet measured                                    | —                                                    |

## V00 results (RTX 5070 Laptop, SM120)

CUDA Events benchmark, `block(16,16)`, warmup 10, 30 samples × 20 launches:

| Matrix shape (M, N, K) | Median kernel latency |      Throughput |
| ---------------------- | --------------------: | --------------: |
| (128, 128, 128)        |              9.119 µs |  459.94 GFLOP/s |
| (256, 256, 256)        |             33.285 µs | 1008.10 GFLOP/s |
| (257, 255, 259)        |             32.300 µs | 1050.99 GFLOP/s |

These are kernel-only measurements on a particular laptop GPU and are not comparisons against PyTorch or Triton. Consult the [V00 report](notes/GEMM_V00_Naive_FP32.md) for measurement constraints and profiling evidence.

## Build and validate

Run from the repository root:

```bash
cmake --preset release-sm120
cmake --build --preset release-sm120
```

With exported PyTorch fixtures present under `results/fixtures/gemm/`:

```bash
source ~/.venvs/kernel-lab/bin/activate
python operators/gemm/tests/run_v00_suite.py \
  --runner build/release-sm120/operators/gemm/gemm_v00_test \
  --suite smoke
ctest --test-dir build/release-sm120 --output-on-failure
```

Benchmark V00 on a supported NVIDIA GPU:

```bash
./build/release-sm120/benchmarks/gemm_v00_benchmark 256 256 256
```

Note: `release-sm120` targets compute capability 12.0. Other architectures require an appropriate CMake configuration. Generated fixtures and build artifacts may be excluded from Git; the reference exporter is [`pytorch/reference.py`](pytorch/reference.py).