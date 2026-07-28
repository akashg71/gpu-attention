"""
Phase 3: Nsight Compute profiling — HBM bytes moved, memory/compute
throughput, roofline placement for naive vs Triton.

*** FIRST RUN ON THIS BOX. Exact ncu CLI/metric names not yet verified
against the installed version here — see instructions.md's own warning that
API knowledge can be stale, same lesson as the Triton kernel itself. This
script deliberately does NOT try to cherry-pick named metric columns; it
dumps the raw CSV and prints every column generically, specifically so a
wrong guess about metric names surfaces as visible real output to fix,
rather than a silent wrong number or an opaque crash. ***

Two modes:
  --target {naive,sdpa,triton}  — runs a few warmup calls then ONE real call
      of a single implementation at a fixed shape. This is what ncu wraps;
      not meant to be run directly for its own sake.
  (no args) — driver mode: checks ncu/nsys are present, shells out to ncu
      for each implementation, saves raw CSV to results/traces/, prints
      every captured column, and (if dram-bytes-like columns are found)
      computes arithmetic intensity and plots the roofline.

Prerequisites — run these checks first if unsure:
    which ncu nsys
    sudo -v                          # cache sudo credentials so the
                                      # subprocess sudo call below doesn't
                                      # hang waiting for a password prompt
    ncu --query-metrics | grep -iE "dram__bytes|throughput|warps_active"
                                      # confirms real metric names on this
                                      # box if the roofline set's output
                                      # doesn't have what's expected

    python scripts/03_profile.py
"""
import argparse
import csv
import io
import subprocess
import sys

sys.path.insert(0, "src")

SHAPE = dict(batch=2, heads=8, seq_len=2048, head_dim=64)
DTYPE_NAME = "float16"


def run_target(name: str) -> None:
    import torch

    from gpu_attention.env import get_device
    from gpu_attention.reference import attention_naive, attention_sdpa
    from gpu_attention.triton_attention import triton_attention, _fwd_kernel

    device = get_device()
    q = torch.randn(SHAPE["batch"], SHAPE["heads"], SHAPE["seq_len"], SHAPE["head_dim"],
                     device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    impls = {
        "naive": lambda: attention_naive(q, k, v, causal=False),
        "sdpa": lambda: attention_sdpa(q, k, v, causal=False),
        "triton": lambda: triton_attention(q, k, v, causal=False),
    }
    fn = impls[name]

    # Settle JIT compilation / autotuning search / cuBLAS algo selection
    # BEFORE the launch(es) ncu actually captures in detail below.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    fn()  # the call actually meant to be profiled
    torch.cuda.synchronize()

    if name == "triton":
        # To stderr, not stdout — ncu's CSV report is on stdout, and mixing
        # print() output into that stream would corrupt parsing again (same
        # class of bug as the ==PROF== noise). Asks the autotuner directly
        # which config it picked, rather than inferring it from ncu's replay
        # order — several _CONFIGS entries share identical grid/block
        # dimensions (same BLOCK_M and num_warps, differing only in BLOCK_N/
        # num_stages, neither visible in ncu's report), so grid/block alone
        # can't always disambiguate which config actually won.
        print(f"[diagnostic] autotuner cache: {getattr(_fwd_kernel, 'cache', 'NO .cache ATTR')}",
              file=sys.stderr)
        print(f"[diagnostic] autotuner best_config: {getattr(_fwd_kernel, 'best_config', 'NO .best_config ATTR')}",
              file=sys.stderr)


def _find_tools() -> dict:
    """Resolves absolute paths for ncu/nsys in the CURRENT user's PATH, and
    uses those absolute paths everywhere below instead of relying on sudo to
    find them — sudo's secure_path typically doesn't include
    /usr/local/cuda/bin even when the invoking user's own PATH does, so
    `sudo ncu` can fail with "command not found" while plain `ncu` works
    fine. Absolute paths sidestep that PATH-resolution difference entirely.
    """
    import shutil

    paths = {}
    for tool in ("ncu", "nsys"):
        resolved = shutil.which(tool)
        if resolved is None:
            print(f"'{tool}' not found on PATH. Install: "
                  f"sudo apt install -y nsight-compute nsight-systems")
        else:
            print(f"{tool}: {resolved}")
            paths[tool] = resolved
    return paths


def _run_ncu_for(target: str, ncu_path: str) -> str:
    """Returns raw stdout (CSV) from ncu, or '' on failure (with a message
    printed explaining what to check).
    """
    cmd = [
        "sudo", ncu_path, "--set", "roofline", "--csv",
        sys.executable, __file__, "--target", target,
    ]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"ncu exited {result.returncode} for target={target}.")
        print("--- stderr (last 40 lines) ---")
        print("\n".join(result.stderr.splitlines()[-40:]))
        if "ERR_NVGPUCTRPERM" in result.stderr:
            print("\nPermission error — GPU performance counters need root.")
            print("Try: sudo -v   (cache credentials) then re-run this script.")
            print("If that doesn't fix it, the driver's NVreg_RestrictProfilingToAdminUsers")
            print("flag may need clearing — see RUNBOOK.md, or search that exact flag name.")
        elif "unknown metric" in result.stderr.lower() or "--set" in result.stderr.lower():
            print("\nPossible unrecognized metric/set name for this ncu version.")
            print("Run: ncu --list-sets    and    ncu --query-metrics | head -50")
            print("to see what's actually available, then adjust this script.")
        return ""

    diagnostic_lines = [l for l in result.stderr.splitlines() if l.startswith("[diagnostic]")]
    for line in diagnostic_lines:
        print(line)

    return result.stdout


def _strip_ncu_noise(raw_text: str) -> str:
    """ncu's --csv stdout isn't pure CSV — it's interleaved with "==PROF=="
    connection/profiling-progress status lines printed as profiling happens
    (one or more per kernel launch). Strip those and blank lines so only the
    actual CSV table (header + data rows) remains.
    """
    lines = [line for line in raw_text.splitlines()
             if line.strip() and not line.startswith("==PROF==")]
    return "\n".join(lines)


# Kernels that are test-setup noise, not part of the attention computation
# itself — torch.randn() generating Q/K/V, and (for Triton) what's very
# likely the autotuner's internal cache-flush fill between candidate-config
# timing trials. Excluded from every summary below.
_NOISE_MARKERS = ("distribution_elementwise", "distribution_nullary", "FillFunctor")


def _parse_launches(csv_text: str, label: str) -> list[dict]:
    """ncu's CSV is long-format: one row per (kernel launch, single metric),
    not one row per launch with metrics as columns — 'Metric Name' and
    'Metric Value' are themselves columns. Groups by launch ID and pivots
    each launch's metrics into a dict: {kernel_name, grid_size, block_size,
    metrics: {name: value}}. Filters out known noise kernels.
    """
    csv_text = _strip_ncu_noise(csv_text)
    if not csv_text.strip():
        print(f"{label}: no output captured (or it was entirely ==PROF== noise).")
        return []

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        print(f"{label}: CSV had no data rows after stripping ==PROF== lines.")
        print(csv_text[:500])
        return []

    launches = {}
    for r in rows:
        lid = int(r.get("ID", -1))
        if lid not in launches:
            launches[lid] = {
                "id": lid,
                "kernel_name": r.get("Kernel Name", ""),
                "grid_size": r.get("Grid Size", ""),
                "block_size": r.get("Block Size", ""),
                "metrics": {},  # name -> (value, unit)
            }
        metric_name = r.get("Metric Name", "")
        if metric_name:
            launches[lid]["metrics"][metric_name] = (r.get("Metric Value", ""), r.get("Metric Unit", ""))

    all_launches = sorted(launches.values(), key=lambda l: l["id"])
    real_launches = [l for l in all_launches
                      if not any(m in l["kernel_name"] for m in _NOISE_MARKERS)]

    print(f"\n{label}: {len(rows)} metric-rows -> {len(all_launches)} distinct kernel launches "
          f"({len(all_launches) - len(real_launches)} excluded as RNG/fill noise).")

    from collections import Counter
    counts = Counter((l["kernel_name"][:70], l["grid_size"], l["block_size"]) for l in real_launches)
    print("Attention-relevant kernels (name, grid size, block size):")
    for (name, grid, block), count in counts.most_common():
        print(f"  {count:>4}x  grid={grid:<15} block={block:<12} {name}")

    # Per-launch detail, in launch order (ID ascending) — this is what lets us
    # SEE the transition from Triton's autotuning search (many different grid
    # shapes early on) to steady state (the same grid repeating at the end),
    # instead of guessing which launches are "the real one" from counts alone.
    key_metrics = ["Duration", "DRAM Throughput", "Compute (SM) Throughput", "Memory Throughput"]
    print(f"\nPer-launch detail, in launch order ({key_metrics}):")
    for l in real_launches:
        short_name = l["kernel_name"][:28]
        vals = []
        for m in key_metrics:
            if m in l["metrics"]:
                value, unit = l["metrics"][m]
                vals.append(f"{m}={value}{unit}")
        print(f"  id={l['id']:>4}  grid={l['grid_size']:<15} {short_name:<28}  {'  '.join(vals)}")

    return real_launches


_TIME_UNIT_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def _representative_naive_launches(launches: list[dict]) -> list[dict]:
    """One representative per distinct kernel name. Naive's kernels showed
    near-identical metrics across all 4 occurrences in the real run (no
    autotuning, no warmup-dependent variance) — any occurrence works;
    taking the last (chronologically) avoids relying on that assumption.
    """
    by_name = {}
    for l in launches:
        by_name[l["kernel_name"]] = l
    return list(by_name.values())


def _representative_triton_launches(launches: list[dict], tolerance: float = 0.05) -> list[dict]:
    """The tail cluster of launches sharing the same grid/block with Duration
    within `tolerance` of the very last launch. Triton's autotune search
    (many launches, early in ID order, several different grid/block shapes)
    should end and settle into a repeated steady-state shape by the end of
    the launch sequence — this walks backward from the last launch and stops
    at the first one that doesn't match, rather than assuming a specific
    launch count (which depends on exactly how many configs/trials the
    search needed, not something to hardcode).
    """
    if not launches:
        return []
    last = launches[-1]
    last_val = float(last["metrics"]["Duration"][0])

    cluster = []
    for l in reversed(launches):
        if l["kernel_name"] != last["kernel_name"] or l["grid_size"] != last["grid_size"]:
            break
        val = float(l["metrics"]["Duration"][0])
        if abs(val - last_val) / last_val > tolerance:
            break
        cluster.append(l)
    return list(reversed(cluster))


def _bytes_per_launch(launches: list[dict], peak_bandwidth_gbps: float) -> list[float]:
    """bytes ≈ (DRAM Throughput % of peak) × peak_bandwidth × Duration, one
    value per launch. Deliberately uses ncu's throughput PERCENTAGE, not its
    absolute Duration alone — the percentage is normalized to whatever the
    (possibly profiling-inflated) duration actually was, so profiling
    overhead cancels out of the bytes estimate even though it would corrupt
    a direct latency comparison.

    Returns a list rather than a total — summing is only correct across
    launches that are DIFFERENT sequential kernels forming one pipeline
    (naive's 6 kernels: GEMM, scale, cast, softmax, cast, GEMM = one call).
    Summing across REPEATED launches of the SAME kernel (Triton's
    steady-state cluster — several launches of _fwd_kernel, all doing
    independent, complete calls) would count one call's bytes N times over.
    Caller sums or averages depending on which situation applies.
    """
    values = []
    for l in launches:
        if "Duration" not in l["metrics"] or "DRAM Throughput" not in l["metrics"]:
            continue
        dur_val, dur_unit = l["metrics"]["Duration"]
        dram_val, dram_unit = l["metrics"]["DRAM Throughput"]
        if dram_unit != "%":
            continue  # unexpected unit — skip rather than silently miscompute
        duration_s = float(dur_val) * _TIME_UNIT_TO_SECONDS.get(dur_unit, 1e-9)
        dram_fraction = float(dram_val) / 100.0
        values.append(dram_fraction * peak_bandwidth_gbps * 1e9 * duration_s)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["naive", "sdpa", "triton"], default=None)
    args = parser.parse_args()

    if args.target:
        run_target(args.target)
        return

    print(f"=== Phase 3 profiling driver — shape {SHAPE}, dtype {DTYPE_NAME} ===\n")
    tools = _find_tools()
    if "ncu" not in tools or "nsys" not in tools:
        print("\nFix the above before continuing — see RUNBOOK.md section 6 (Nsight tools).")
        sys.exit(1)

    import os
    os.makedirs("results/traces", exist_ok=True)

    csv_outputs = {}
    for target in ("naive", "triton"):
        csv_text = _run_ncu_for(target, tools["ncu"])
        csv_outputs[target] = csv_text
        if csv_text:
            with open(f"results/traces/{target}_ncu.csv", "w") as f:
                f.write(_strip_ncu_noise(csv_text))
            print(f"Wrote results/traces/{target}_ncu.csv (==PROF== noise stripped)")

    parsed = {}
    for target, csv_text in csv_outputs.items():
        parsed[target] = _parse_launches(csv_text, target)

    print("\n=== Computing bytes moved, arithmetic intensity, roofline ===")
    import torch
    from gpu_attention import roofline
    from gpu_attention.env import get_device
    from gpu_attention.reference import attention_naive
    from gpu_attention.triton_attention import triton_attention
    from gpu_attention.benchmark import bench_one, attention_flops

    device = get_device()
    peak_bw = roofline.get_peak_hbm_bandwidth_gbps(device)

    naive_reps = _representative_naive_launches(parsed.get("naive", []))
    triton_reps = _representative_triton_launches(parsed.get("triton", []))
    print(f"naive: {len(naive_reps)} representative kernels (1 per distinct kernel in the pipeline)")
    print(f"triton: {len(triton_reps)} representative launches (tail steady-state cluster)")

    # naive: SUM across its 6 representative launches -- different sequential
    # kernels forming one pipeline, so the sum is "bytes for one full call".
    naive_bytes = sum(_bytes_per_launch(naive_reps, peak_bw))
    # triton: AVERAGE across the steady-state cluster -- repeated launches of
    # the SAME single-kernel pipeline; summing would count one call's bytes
    # once per cluster member instead of representing "bytes for one call".
    triton_per_launch = _bytes_per_launch(triton_reps, peak_bw)
    triton_bytes = sum(triton_per_launch) / len(triton_per_launch)
    print(f"naive total HBM traffic (one call):  {naive_bytes / 1e9:.3f} GB")
    print(f"triton total HBM traffic (one call): {triton_bytes / 1e9:.3f} GB "
          f"(averaged over {len(triton_per_launch)} steady-state launches)")

    # Fresh, UNPROFILED latency at the same shape for achieved FLOP/s — not
    # ncu's Duration, which we already saw is inflated ~3-4x for Triton by
    # profiling instrumentation overhead. This mirrors exactly how Phase 2
    # measured latency (bench_one, CUDA events, no profiler attached).
    q = torch.randn(SHAPE["batch"], SHAPE["heads"], SHAPE["seq_len"], SHAPE["head_dim"],
                     device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    naive_bench = bench_one("naive", lambda: attention_naive(q, k, v, causal=False),
                             causal=False, device=device, **SHAPE)
    triton_bench = bench_one("triton", lambda: triton_attention(q, k, v, causal=False),
                              causal=False, device=device, **SHAPE)
    print(f"naive latency (fresh, unprofiled):  {naive_bench.latency_ms:.3f}ms")
    print(f"triton latency (fresh, unprofiled): {triton_bench.latency_ms:.3f}ms")

    flops = attention_flops(SHAPE["batch"], SHAPE["heads"], SHAPE["seq_len"], SHAPE["head_dim"], causal=False)
    naive_achieved_flops = flops / (naive_bench.latency_ms / 1000)
    triton_achieved_flops = flops / (triton_bench.latency_ms / 1000)

    naive_ai = roofline.arithmetic_intensity(flops, naive_bytes)
    triton_ai = roofline.arithmetic_intensity(flops, triton_bytes)
    print(f"naive arithmetic intensity:  {naive_ai:.3f} FLOPs/byte, {naive_achieved_flops / 1e12:.3f} TFLOP/s achieved")
    print(f"triton arithmetic intensity: {triton_ai:.3f} FLOPs/byte, {triton_achieved_flops / 1e12:.3f} TFLOP/s achieved")

    import os
    os.makedirs("results/figures", exist_ok=True)
    roofline.plot_roofline(
        device, torch.float16,
        [("naive", naive_ai, naive_achieved_flops), ("triton", triton_ai, triton_achieved_flops)],
        "results/figures/roofline.png",
    )
    print("\nWrote results/figures/roofline.png")

    print("\n=== For manual/qualitative review (warp-stall reasons, full detail) ===")
    print(f"sudo {tools['ncu']} --set full -o results/traces/naive_full {sys.executable} {__file__} --target naive")
    print(f"sudo {tools['ncu']} --set full -o results/traces/triton_full {sys.executable} {__file__} --target triton")
    print(f"Then: {tools['ncu']} -i results/traces/triton_full.ncu-rep   (opens the text report)")
    print("\n=== Timeline (Nsight Systems) ===")
    print(f"sudo {tools['nsys']} profile -o results/traces/timeline {sys.executable} {__file__} --target triton")


if __name__ == "__main__":
    main()
