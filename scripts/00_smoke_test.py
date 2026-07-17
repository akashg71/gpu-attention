"""
Phase 0 gate. Run this FIRST on the GPU box:

    python scripts/00_smoke_test.py

Builds the Triton fused-attention kernel, runs it on one shape, checks it
against attention_reference with torch.allclose, prints PASS/FAIL + max abs
error. If this fails, that's expected on the very first run (see the warning
at the top of triton_attention.py) — fix the kernel against your installed
Triton's own tutorial before doing anything else. Nothing downstream
(benchmarking, profiling) is meaningful until this passes.
"""
import sys

import torch

sys.path.insert(0, "src")

from gpu_attention.env import print_env, get_device
from gpu_attention.correctness import check_one


def main():
    print("=== environment ===")
    print_env()
    device = get_device()

    print("\n=== smoke test: single shape, fp16, non-causal ===")
    result = check_one(
        batch=2,
        heads=4,
        seq_len=512,
        head_dim=64,
        causal=False,
        dtype=torch.float16,
        device=device,
    )

    status = "PASS" if result.passed else "FAIL"
    print(f"shape={result.shape} dtype={result.dtype} causal={result.causal}")
    print(f"max_abs_err={result.max_abs_err:.6f} max_rel_err={result.max_rel_err:.6f}")
    print(f"{status}")

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
