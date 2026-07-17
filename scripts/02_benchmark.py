"""
Phase 0: single-shape benchmark (naive / SDPA / Triton latency, TFLOP/s, peak
memory). Phase 2 extends this to a full sequence-length sweep with plots.

    python scripts/02_benchmark.py
"""
import sys

sys.path.insert(0, "src")

from gpu_attention.benchmark import run_all
from gpu_attention.env import get_device


def main():
    device = get_device()
    results = run_all(device=device)
    print(f"{'impl':<8} {'latency (ms)':>14} {'TFLOP/s':>10} {'peak mem (GB)':>15}")
    for r in results:
        print(f"{r.name:<8} {r.latency_ms:>14.3f} {r.tflops:>10.2f} {r.peak_mem_gb:>15.3f}")


if __name__ == "__main__":
    main()
