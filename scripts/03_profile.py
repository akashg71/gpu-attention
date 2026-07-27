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
    from gpu_attention.triton_attention import triton_attention

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
        lid = r.get("ID")
        if lid not in launches:
            launches[lid] = {
                "kernel_name": r.get("Kernel Name", ""),
                "grid_size": r.get("Grid Size", ""),
                "block_size": r.get("Block Size", ""),
                "metrics": {},
            }
        metric_name = r.get("Metric Name", "")
        if metric_name:
            launches[lid]["metrics"][metric_name] = r.get("Metric Value", "")

    all_launches = list(launches.values())
    real_launches = [l for l in all_launches
                      if not any(m in l["kernel_name"] for m in _NOISE_MARKERS)]

    print(f"\n{label}: {len(rows)} metric-rows -> {len(all_launches)} distinct kernel launches "
          f"({len(all_launches) - len(real_launches)} excluded as RNG/fill noise).")

    from collections import Counter
    counts = Counter((l["kernel_name"][:70], l["grid_size"], l["block_size"]) for l in real_launches)
    print("Attention-relevant kernels (name, grid size, block size):")
    for (name, grid, block), count in counts.most_common():
        print(f"  {count:>4}x  grid={grid:<15} block={block:<12} {name}")

    if real_launches:
        available_metrics = sorted(real_launches[0]["metrics"].keys())
        print(f"\nMetric names available per launch ({len(available_metrics)} total):")
        for m in available_metrics:
            print(f"  {m}")

    return real_launches


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

    for target, csv_text in csv_outputs.items():
        _parse_launches(csv_text, target)

    print("\n=== For manual/qualitative review (warp-stall reasons, full detail) ===")
    print(f"sudo {tools['ncu']} --set full -o results/traces/naive_full {sys.executable} {__file__} --target naive")
    print(f"sudo {tools['ncu']} --set full -o results/traces/triton_full {sys.executable} {__file__} --target triton")
    print(f"Then: {tools['ncu']} -i results/traces/triton_full.ncu-rep   (opens the text report)")
    print("\n=== Timeline (Nsight Systems) ===")
    print(f"sudo {tools['nsys']} profile -o results/traces/timeline {sys.executable} {__file__} --target triton")

    print("\nNext: once real column names are confirmed above, tell me what you see —")
    print("the roofline plot (roofline.plot_roofline) needs actual bytes-moved and")
    print("achieved-FLOP/s numbers from this output, not guessed column names.")


if __name__ == "__main__":
    main()
