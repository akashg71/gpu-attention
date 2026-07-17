"""
Phase 3 stub — not implemented. Will emit the exact ncu/nsys command lines
for profiling the Triton kernel vs naive, and parse the results (HBM bytes
moved, memory throughput, occupancy, warp stalls) into results/benchmarks.md
and the roofline plot.

Reminder from instructions.md: ncu commonly needs elevated privileges that
hosted notebooks (Colab/Kaggle) block. Expect to run this on a rented box
with sudo, not the free-tier T4 used for Phases 0-2.

Planned commands (fill in once a target kernel launch is isolated):
    ncu --set full --metrics dram__bytes.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        -o results/traces/triton_attn python scripts/02_benchmark.py
    nsys profile -o results/traces/timeline python scripts/02_benchmark.py
"""
raise NotImplementedError("Phase 3 — needs a rented GPU box with sudo")
