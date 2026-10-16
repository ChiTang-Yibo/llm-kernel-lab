
"""
GEMM V00 Correctness Suite

Read test shapes from configs/shapes.json
and execute the CUDA correctness runner.

Python is used only for test orchestration.
The GEMM computation is performed by CUDA.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = (
    PROJECT_ROOT
    / "operators/gemm/configs/shapes.json"
)

FIXTURE_ROOT = (
    PROJECT_ROOT
    / "results/fixtures/gemm"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runner",
        type=Path,
        required=True,
        help="Path to the compiled CUDA test executable",
    )

    parser.add_argument(
        "--suite",
        choices=["smoke", "extended"],
        default="smoke",
    )

    args = parser.parse_args()

    runner = args.runner.resolve()

    if not runner.is_file():
        print(f"ERROR: Runner not found: {runner}")
        return 1

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    cases = config[args.suite]

    atol = config["atol"]
    rtol = config["rtol"]

    passed = 0
    failed = 0

    print("=" * 65)
    print("GEMM V00 - Correctness Suite")
    print("=" * 65)
    print(f"Suite: {args.suite}")
    print(f"Total cases: {len(cases)}")
    print(f"ATOL: {atol}")
    print(f"RTOL: {rtol}")
    print("=" * 65, flush=True)

    for case in cases:
        name = case["name"]

        M = case["M"]
        N = case["N"]
        K = case["K"]

        fixture_dir = FIXTURE_ROOT / name

        command = [
            str(runner),
            str(fixture_dir),
            str(M),
            str(N),
            str(K),
            str(atol),
            str(rtol),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(
            f"[{status}] {name:12s} "
            f"M={M:5d} N={N:5d} K={K:5d}",
            flush=True,
        )

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)

    print("=" * 65)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 65)

    if failed:
        print("GEMM V00 CORRECTNESS FAILED")
        return 1

    print("GEMM V00 CORRECTNESS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
