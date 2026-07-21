"""
Phase 1: correctness sweep across seq_len, head_dim, batch, causal/non-causal,
fp16/bf16 (80 combinations — see SEQ_LENS/HEAD_DIMS/BATCHES/CAUSALS/DTYPES in
gpu_attention.correctness). Expect this to take a few minutes: each new
seq_len triggers a fresh Triton autotuning search (see triton_attention.py),
and that's a one-time cost per seq_len, not per combination.

    python scripts/01_correctness.py
"""
import sys

sys.path.insert(0, "src")

from gpu_attention.env import get_device
from gpu_attention.correctness import run_sweep


def main():
    device = get_device()
    print("Running Phase 1 correctness sweep (this will take a few minutes)...\n")
    results = run_sweep(device)

    header = f"{'shape (B,H,N,D)':<20} {'dtype':<10} {'causal':<7} {'max_abs_err':>12}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        dtype_name = str(r.dtype).replace("torch.", "")
        if r.error:
            print(f"{str(r.shape):<20} {dtype_name:<10} {str(r.causal):<7} {'—':>12}  ERROR: {r.error}")
        else:
            status = "PASS" if r.passed else "FAIL"
            print(f"{str(r.shape):<20} {dtype_name:<10} {str(r.causal):<7} {r.max_abs_err:>12.6f}  {status}")

    failures = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for r in failures:
            dtype_name = str(r.dtype).replace("torch.", "")
            if r.error:
                print(f"  shape={r.shape} dtype={dtype_name} causal={r.causal} ERROR: {r.error}")
            else:
                print(f"  shape={r.shape} dtype={dtype_name} causal={r.causal} max_abs_err={r.max_abs_err:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
