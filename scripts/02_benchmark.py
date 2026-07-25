"""
Phase 2: sequence-length sweep (naive / SDPA / Triton), latency + TFLOP/s +
peak memory. Writes results/benchmarks.md and two plots to results/figures/.

    python scripts/02_benchmark.py
"""
import sys

sys.path.insert(0, "src")

from gpu_attention.benchmark import (
    run_seqlen_sweep,
    write_results_markdown,
    plot_latency_vs_seqlen,
    plot_peakmem_vs_seqlen,
)
from gpu_attention.env import get_device


def main():
    device = get_device()
    print("Running Phase 2 sequence-length sweep...\n")
    results = run_seqlen_sweep(device=device)

    print(f"{'seq_len':>8} {'impl':<8} {'latency (ms)':>14} {'TFLOP/s':>10} {'peak mem (GB)':>15}")
    for r in results:
        if r.error:
            print(f"{r.seq_len:>8} {r.name:<8} {r.error:>14} {r.error:>10} {r.error:>15}")
        else:
            print(f"{r.seq_len:>8} {r.name:<8} {r.latency_ms:>14.3f} {r.tflops:>10.2f} {r.peak_mem_gb:>15.3f}")

    write_results_markdown(results, "results/benchmarks.md")
    plot_latency_vs_seqlen(results, "results/figures/latency_vs_seqlen.png")
    plot_peakmem_vs_seqlen(results, "results/figures/peak_mem_vs_seqlen.png")

    print("\nWrote results/benchmarks.md")
    print("Wrote results/figures/latency_vs_seqlen.png")
    print("Wrote results/figures/peak_mem_vs_seqlen.png")


if __name__ == "__main__":
    main()
