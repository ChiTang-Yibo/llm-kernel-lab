
"""
GEMM Correctness Reference

Mathematical definition:
    C = A @ B

Input:
    A: [M, K], FP32, row-major
    B: [K, N], FP32, row-major

Output:
    C: [M, N], FP32, row-major

No benchmarking is performed in this script.
"""

import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT / "operators/gemm/configs/shapes.json"
)


def make_inputs(M, N, K, seed):
    """Generate reproducible CPU FP32 inputs."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    A = torch.randn(
        (M, K),
        generator=generator,
        dtype=torch.float32,
    )

    B = torch.randn(
        (K, N),
        generator=generator,
        dtype=torch.float32,
    )

    return A.contiguous(), B.contiguous()


def gemm_reference(A, B):
    """Compute the PyTorch GEMM reference."""

    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[0]

    return torch.mm(A, B)


def run_suite(suite, export=False):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["dtype"] == "float32"

    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    seed = config["seed"]
    atol = config["atol"]
    rtol = config["rtol"]

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision:", torch.get_float32_matmul_precision())
    print("Suite:", suite)

    cases = config[suite]

    for case in cases:
        M = case["M"]
        N = case["N"]
        K = case["K"]
        name = case["name"]

        A_cpu, B_cpu = make_inputs(M, N, K, seed)

        A = A_cpu.to(device)
        B = B_cpu.to(device)

        with torch.no_grad():
            C = gemm_reference(A, B)

        torch.cuda.synchronize()

        assert C.shape == (M, N)
        assert C.dtype == torch.float32
        assert C.is_contiguous()
        assert torch.isfinite(C).all().item()

        # Independent higher-precision numerical sanity check
        # for the small correctness suite.
        if suite == "smoke":
            C_fp64 = (
                A_cpu.double() @ B_cpu.double()
            ).float()

            torch.testing.assert_close(
                C.cpu(), C_fp64,
                atol=atol,
                rtol=rtol,
            )

        if export:
            output_dir = (
                PROJECT_ROOT / "results/fixtures/gemm" / name
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            A_cpu.numpy().tofile(output_dir / "A.bin")
            B_cpu.numpy().tofile(output_dir / "B.bin")
            C.cpu().numpy().tofile(output_dir / "C_ref.bin")

            metadata = {
                "name": name,
                "M": M,
                "N": N,
                "K": K,
                "dtype": "float32",
                "layout": "row-major",
                "seed": seed,
                "atol": atol,
                "rtol": rtol,
            }

            with open(
                output_dir / "metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, indent=2)

        print(
            f"[PASS] {name:12s} "
            f"M={M:5d} N={N:5d} K={K:5d}"
        )

    print(f"\nGEMM Reference {suite.upper()} PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        choices=["smoke", "extended"],
        default="smoke",
    )

    parser.add_argument(
        "--export",
        action="store_true",
    )

    args = parser.parse_args()
    run_suite(args.suite, args.export)


if __name__ == "__main__":
    main()
