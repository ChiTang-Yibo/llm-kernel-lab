# GEMM V00 — Naive FP32 CUDA Baseline

> - **Project:** `llm-kernel-lab`
> - **Operator:** GEMM · **Version:** V00
> - **Implementation:** one CUDA thread per output element; scalar FP32 accumulation
> - **Device:** NVIDIA GeForce RTX 5070 Laptop GPU (compute capability 12.0)
> **Status:** CUDA compilation passed; 7/7 smoke correctness cases passed; benchmark and two Nsight Compute reports collected.

## 1. Mathematical Model

For contiguous row-major matrices,

$$
A \in \mathbb{R}^{M \times K},\qquad
B \in \mathbb{R}^{K \times N},\qquad
C \in \mathbb{R}^{M \times N},
$$

$$
C_{i,j} = \sum_{k=0}^{K-1} A_{i,k}B_{k,j},
\quad 0 \le i < M,\quad 0 \le j < N.
$$

The conventional GEMM operation count used for throughput reporting is

$$
F \approx 2MNK \quad \text{FLOPs}.
$$

Each multiplication-addition contributes two operations in this convention. V00 performs FP32 scalar arithmetic; it does not explicitly use Tensor Cores.

## 2. Parallel Decomposition and GPU Mapping

Each CUDA thread computes **one output element** `C[row, col]`. The launch configuration is:

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x,
          (M + block.y - 1) / block.y);
```

The global coordinates are:

$$
\mathrm{row} = \mathrm{blockIdx.y}\,\mathrm{blockDim.y} + \mathrm{threadIdx.y}
$$

$$
\mathrm{col} = \mathrm{blockIdx.x}\,\mathrm{blockDim.x} + \mathrm{threadIdx.x}
$$

CUDA `x` maps to the **output column**, and CUDA `y` maps to the **output row**. Out-of-range threads do no work. Threads compute distinct output elements and do not need inter-thread synchronization in this baseline.

## 3. Memory Mapping, Index Derivation, and Timeline

The arrays are contiguous and row-major:

| Element   | Linear FP32 index | Row stride |
| --------- | ----------------- | ---------- |
| $A_{i,k}$ | `i * K + k`       | `K`        |
| $B_{k,j}$ | `k * N + j`       | `N`        |
| $C_{i,j}$ | `i * N + j`       | `N`        |

The per-thread execution order is: compute `(row, col)` → check output bounds → initialize `value = 0` → loop over `k` and accumulate the product → store one `C` element.

- **Global memory:** input arrays A and B and output array C.
- **Registers:** the thread-private accumulator is a local scalar, normally register-allocated. Nsight reported **40 registers/thread** for the compiled kernel.
- **Shared memory:** not explicitly used in V00; Nsight reported zero static and dynamic shared memory per block.
- **Source-level logical accesses per output:** `2K` FP32 loads and one FP32 store, or `8K + 4` logical bytes. This is **not** measured DRAM traffic: hardware caching, memory coalescing, and compiler optimizations affect actual transactions.

## 4. Independent CUDA Implementation

The kernel was written independently from the mathematical and index derivations. Its essential computation is:

```cpp
__global__ void gemm_v00_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
```

Full source: [`../cuda/v00_naive.cu`](../cuda/v00_naive.cu) and declaration: [`../cuda/v00_naive.cuh`](../cuda/v00_naive.cuh).

The kernel is built into the static library `gemm_v00`. The correctness and benchmark executables link against the same library; the host-side allocation, copies, launch, and synchronization are implemented in the respective runners rather than inside the kernel source.

Build commands (from repository root):

```bash
cmake --preset release-sm120
cmake --build --preset release-sm120
```

## 5. Correctness Gate

The [PyTorch reference](../pytorch/reference.py) generates FP32 binary fixtures. The [CUDA correctness runner](../tests/test_v00.cu) loads them and applies the elementwise criterion:

$$
|C_{\mathrm{CUDA}}-C_{\mathrm{ref}}| \le
\mathrm{atol} + \mathrm{rtol}|C_{\mathrm{ref}}|,
$$

with `atol = rtol = 1e-4` in the smoke suite.

**All seven smoke cases passed**, including a one-element test, regular square matrices, a non-divisible matrix, and a rectangular matrix.

| Case          | Shape (M, N, K) | Result |
| ------------- | --------------- | ------ |
| `tiny`        | (4, 4, 4)       | PASS   |
| `single`      | (1, 1, 1)       | PASS   |
| `small`       | (16, 16, 16)    | PASS   |
| `regular`     | (128, 128, 128) | PASS   |
| `divisible`   | (256, 256, 256) | PASS   |
| `irregular`   | (257, 255, 259) | PASS   |
| `rectangular` | (128, 256, 64)  | PASS   |

Observed diagnostics for two individually inspected cases:

| Case        | Max absolute error | Max relative error | Failed elements |
| ----------- | -----------------: | -----------------: | --------------: |
| `tiny`      |     2.38418579e-07 |     1.46163190e-06 |               0 |
| `irregular` |     4.76837158e-05 |     8.60745324e-03 |               0 |

A high maximum relative error for a reference value near zero does not imply failure of the combined absolute-plus-relative tolerance criterion. CTest results: **2/2 PASS** (`cuda_utils_test`, `gemm_v00_smoke`).

Reproduce:

```bash
source ~/.venvs/kernel-lab/bin/activate
python operators/gemm/tests/run_v00_suite.py \
  --runner build/release-sm120/operators/gemm/gemm_v00_test \
  --suite smoke
ctest --test-dir build/release-sm120 --output-on-failure
```

## 6. CUDA Events Benchmark

**Timing scope:** kernel-only; excludes host input initialization, allocations, H2D copies, and CSV output. The runner uses the same V00 kernel as the correctness executable.

- Block: `(16, 16)` (256 threads/block).
- Warmup: 10 launches.
- Samples: 30.
- Iterations per sample: 20.
- Primary statistic: median sample-averaged kernel latency.
- Throughput: `GFLOP/s = 2*M*N*K / (median_us*1000)`.

| Shape (M, N, K) | Median (µs) | Min (µs) | Max (µs) |  GFLOP/s |
| --------------- | ----------: | -------: | -------: | -------: |
| (128, 128, 128) |      9.1192 |   8.1904 |  13.5504 |  459.942 |
| (256, 256, 256) |     33.2848 |  32.5984 |  43.5824 | 1008.101 |
| (257, 255, 259) |     32.3000 |  31.5216 |  36.5072 | 1050.995 |

The performance is higher in GFLOP/s for the larger tested shapes, despite a longer kernel duration. The slightly lower measured latency of the irregular case compared with `(256,256,256)` is a single benchmark observation and does **not** establish that irregular dimensions are inherently faster.

The logged `effective_gbps` values derive from the minimum one-read-per-input / one-write-per-output I/O model. They must **not** be interpreted as measured DRAM bandwidth or total kernel memory traffic.

Benchmark command:

```bash
./build/release-sm120/benchmarks/gemm_v00_benchmark 256 256 256
```

The CSV is stored in [`../../../results/tables/gemm_v00.csv`](../../../results/tables/gemm_v00.csv). Power limit, temperatures, clock stability, and independent-trial dispersion were **not fully recorded** in this first run; they should be controlled more rigorously for future cross-version comparisons.

## 7. Nsight Compute Evidence

Two profiling collections were made for `(M,N,K) = (256,256,256)` using a dedicated `--profile-once` launch path. Nsight Compute 2026.2.1 ran on a GPU reporting **36 SMs**. Profiling duration should not be compared directly to the CUDA Events median because profiling and replay conditions differ.

### 7.1 Basic: utilization, launch, and residency

| Metric                           |    Measured value | Interpretation                                               |
| -------------------------------- | ----------------: | ------------------------------------------------------------ |
| Nsight kernel duration           |          56.29 µs | Profiled-run duration, not the primary benchmark timing      |
| SM throughput                    |            73.07% | Aggregate SM throughput metric; **not** 73% of FP32 peak arithmetic |
| SM instruction-executed LSU pipe |            73.07% | High relative load/store-related execution activity          |
| L1/TEX cache throughput          |            76.94% | Substantial near-SM memory path utilization                  |
| L2 throughput                    |            15.31% | Lower normalized throughput at this level                    |
| DRAM throughput                  |            12.50% | No evidence of external DRAM peak-bandwidth saturation       |
| Registers/thread                 |                40 | Compiled kernel resource usage                               |
| Static/dynamic shared memory     | 0 / 0 B per block | No explicit shared-memory allocation                         |
| Theoretical / achieved occupancy |     100% / 76.54% | Average active warp residency below the theoretical limit    |
| Waves per SM                     |              1.19 | Partial final block wave is possible                         |

For `(256,256,256)` and `block(16,16)`, there are 256 blocks, each with 8 warps. At 48 theoretical active warps per SM, six blocks/SM are possible in this configuration. The 36 SMs can therefore accommodate 216 blocks in one ideal full wave, followed by the remaining 40 blocks. This **may** contribute to a tail effect; it does not alone explain all achieved-occupancy loss.

### 7.2 Detailed: scheduler, stalls, and memory

| Metric                                       | Measured value | Interpretation                                               |
| -------------------------------------------- | -------------: | ------------------------------------------------------------ |
| Active warps / scheduler                     |          10.02 | Warps resident and active on average                         |
| Eligible warps / scheduler                   |           0.81 | Few warps ready to issue a next instruction                  |
| Issued warp instructions / scheduler / cycle |           0.26 | Low average issue rate                                       |
| No eligible warp                             |         73.54% | Scheduler often has no ready warp                            |
| Warp cycles / issued instruction             |          37.87 | Warp-state statistical normalization scale                   |
| Long Scoreboard                              |   22.04 cycles | Prominent dependency stalls waiting on L1TEX-related data    |
| LG Throttle                                  |    9.54 cycles | Pressure on global/local memory request admission/resources  |
| L1/TEX hit rate                              |         89.12% | High hit rate does not imply zero access latency             |
| L2 hit rate                                  |         67.81% | L2 hit statistics have a different request population/denominator |
| Memory pipes busy                            |         66.32% | Substantial memory-pipeline activity                         |
| Local/shared spilling requests               |          0 / 0 | No spilling requests reported by these metrics               |

Long Scoreboard accounts for approximately `22.04/37.87 = 58.2%` of the reported warp-state cycle decomposition; LG Throttle for `9.54/37.87 = 25.2%`. **These are not independent shares of total kernel wall-clock duration.**

### 7.3 Evidence-based interpretation

**Direct observations:** load/store-related throughput is elevated, DRAM throughput is relatively low, schedulers frequently have no eligible warp, and memory-related stall categories dominate the sampled warp-state breakdown.

**Interpretation:** this naive kernel has substantial memory-operation dependencies and global/local memory request pressure. A high cache hit rate does not eliminate nonzero access latency, dependency chains, or request-issue constraints. The current results do not support describing the kernel simply as **DRAM-bandwidth-bound**.

**Limits:** Basic and detailed values were gathered in separate profiling collections. Their counters support a coherent hypothesis but do not quantify a unique causal contribution to latency, prove that a particular A or B load is responsible, or guarantee a speedup from any specific code change.

## 8. Conclusion and Next Experiment

**V00 decision: KEEP as a correctness-validated FP32 naive baseline.** It is an intentionally simple version; no performance claim relative to PyTorch or Triton is made at this stage.

**V01 primary hypothesis:** test an alternative thread-block mapping while preserving one output per thread, the same mathematical algorithm, FP32 dtype, and the same benchmark protocol. Candidate: `block(32,8)` versus the V00 `block(16,16)`. Both have 256 threads/block, but change warp-level accesses to A and B.

**Prediction (not yet measured):** the mapping change may alter memory request organization and, consequently, LSU activity, Long Scoreboard, LG Throttle, eligible-warps behavior, and kernel latency. Neither the sign nor the size of any change is assumed in advance. Note that the total block count at `(256,256,256)` stays 256 for these two shapes, so the theoretical block-wave count need not improve.

**Acceptance protocol:** run the exact same correctness suite, obtain repeated CUDA Events benchmark measurements under comparable GPU operating conditions, then profile the best-controlled runs and distinguish observations from hypotheses. Keep V00 unchanged regardless of V01 outcome.

---

### Reproducibility notes

- Environment: Windows/WSL2 Ubuntu 24.04 LTS, CUDA Toolkit 12.8.93, SM120 release target, Nsight Compute 2026.2.1.
- Commands: `cmake --preset release-sm120`, `cmake --build --preset release-sm120`, and `ctest --test-dir build/release-sm120 --output-on-failure`.
- Nsight report locations in the local repository: `profiling/reports/gemm_v00_256_basic.ncu-rep` and `profiling/reports/gemm_v00_256_detailed.ncu-rep` (may be excluded from Git if too large).
- **Provenance:** all numeric measurements in this note are from the V00 terminal output and the two saved Nsight report exports; predicted V01 effects are explicitly hypotheses.
