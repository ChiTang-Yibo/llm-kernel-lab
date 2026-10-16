
# LLM Kernel Performance Lab

A reproducible GPU kernel engineering project focused on
CUDA C++, Triton, PyTorch, and NVIDIA Nsight profiling.

## Project Goals

This project aims to develop practical GPU kernel
optimization skills through independent implementation,
correctness verification, benchmarking, and profiling.

The primary objectives are:

- Translate mathematical algorithms into GPU execution models.
- Implement CUDA kernels from first principles.
- Understand GPU memory access and data reuse.
- Analyze performance using NVIDIA Nsight tools.
- Optimize kernels through measurable hypotheses.
- Compare CUDA, Triton, and PyTorch implementations.
- Build reusable kernels for future LLM inference projects.

## Development Environment

| Component | Configuration |
|---|---|
| OS | Windows 11 + WSL2 |
| Linux | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 5070 Laptop |
| Compute Capability | 12.0 |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Nsight Compute | 2026.2.1 |
| Nsight Systems | 2026.1.3 |
| Build System | CMake + Ninja |
| Languages | CUDA C++, Python |

## Kernel Development Workflow

Every CUDA kernel follows the same workflow:

1. Mathematical Model
2. Parallel Decomposition and GPU Mapping
3. Memory Mapping and Index Derivation
4. Independent CUDA Implementation
5. Correctness Verification
6. Standardized Benchmarking
7. NVIDIA Nsight Profiling
8. Bottleneck Analysis
9. Hypothesis-Driven Optimization
10. CUDA / Triton / PyTorch Comparison

Each optimization version preserves its own
CUDA source code and technical notes.

## Operators

| Operator | Main Topics | Status |
|---|---|---|
| GEMM | Coalescing, shared-memory tiling, register tiling | Planned |
| GEMV | Memory bandwidth, warp reduction | Planned |
| Linear | GEMM, bias fusion | Planned |
| Reduction | Parallel reduction, divergence | Planned |
| Softmax | Stable reduction, fusion | Planned |
| RMSNorm | Reduction, memory bandwidth | Planned |
| RoPE | Memory layout, vectorization | Planned |
| SwiGLU | Elementwise fusion | Planned |
| Attention | Tiling, memory reuse | Planned |

## First Project: GEMM

The first optimization series focuses on FP32 GEMM:

C = A @ B

Planned CUDA implementations:

- V00: Naive GEMM
- V01: Memory Access and Coalescing
- V02: Shared Memory Tiling
- V03: Thread Coarsening
- V04: Register Tiling
- V05: Vectorized Memory Access

The benchmark suite will include square,
rectangular, irregular, tall, wide, and
skinny matrices.

Correctness testing uses PyTorch as a reference
from V00 onward.

## Performance Methodology

Performance evaluation includes:

- Correctness across multiple matrix shapes
- Warmup and repeated measurements
- Median kernel latency
- GFLOP/s and effective memory bandwidth
- Nsight Compute hardware metrics
- Nsight Systems CUDA timeline analysis
- Comparisons against PyTorch and Triton

Optimizations are retained even when they fail
to improve performance, with documented evidence.

## Repository Structure

- `operators/` — Kernel implementations and version notes
- `benchmarks/` — Shared benchmark infrastructure
- `include/` — Common CUDA utilities
- `profiling/` — Profiling scripts and analysis reports
- `results/` — Benchmark tables and visualizations
- `docs/` — Engineering standards and documentation
- `scripts/` — Automation and reproducibility tools

## Project Status

**Active Development — GEMM V00 Completed**

The first FP32 CUDA GEMM baseline has been implemented,
correctness-validated, benchmarked, and profiled.

- Correctness: 7/7 tests passed.
- CTest: 2/2 passed.
- Benchmark: CUDA Events measurements completed.
- Profiling: Nsight Compute Basic and Detailed reports collected.

See the [GEMM project](operators/gemm/README.md)
and the [V00 technical report](operators/gemm/notes/GEMM_V00_Naive_FP32.md).

Next milestone: GEMM V01 — Thread Mapping and Memory Access.