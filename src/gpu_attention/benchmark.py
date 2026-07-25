"""
Timing harness: naive vs SDPA vs Triton. CUDA-event timing, median of N,
latency + TFLOP/s + peak memory. Phase 0 needs this runnable on one shape
(see __main__ below / scripts/02_benchmark.py); Phase 2 sweeps sequence length.
"""
from dataclasses import dataclass
from typing import Callable

import torch

from .reference import attention_naive, attention_sdpa
from .triton_attention import triton_attention


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int, causal: bool) -> float:
    """FLOP count for attention: QK^T and P@V are each a (N,N,d) matmul, each
    costing 2*N^2*d FLOPs (multiply+add), so 4*B*H*N^2*d total. Causal attention
    only computes the lower triangle, which is ~half the work — approximated
    here as an exact half rather than accounting for the diagonal blocks
    separately, since that's a second-order correction.
    """
    flops = 4 * batch * heads * (seq_len ** 2) * head_dim
    if causal:
        flops *= 0.5
    return flops


@dataclass
class BenchResult:
    name: str
    seq_len: int
    latency_ms: float
    tflops: float
    peak_mem_gb: float
    error: str = ""


def _warmup(fn: Callable[[], torch.Tensor], iters: int = 10) -> None:
    """Runs fn() `iters` times and synchronizes, with no timing or memory
    tracking. Split out from _time_cuda as its own step so one-time costs —
    Triton JIT compilation, its autotuning search over _CONFIGS, cuDNN
    algorithm search — complete and are discarded *before* peak-memory
    tracking starts in bench_one(). Autotuning in particular allocates its
    own internal scratch buffer to benchmark candidate configs fairly; if
    peak-memory tracking were active during that search, it would count a
    one-time setup allocation as if it were the kernel's steady-state memory
    footprint (this is exactly what inflated Triton's reported peak memory
    from 0.016GB to 0.266GB after autotuning was added — the reset was
    happening before this warmup, not after).
    """
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def _time_cuda(fn: Callable[[], torch.Tensor], iters: int = 50) -> float:
    """Median latency in ms over `iters` runs, timed with CUDA events.
    Assumes warmup (see _warmup) has already happened — does not warm up
    itself, so it can be called after peak-memory tracking has been reset.
    """
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def bench_one(
    name: str,
    fn: Callable[[], torch.Tensor],
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    device: torch.device,
) -> BenchResult:
    """Runs and times one implementation. OOM is caught here rather than left
    to crash the caller: the brief explicitly expects naive to OOM or fall
    off a cliff at large seq_len (Section 4.3) — that's a result to record
    (BenchResult.error), not a failure that should take down a sweep over
    many shapes.
    """
    try:
        _warmup(fn)
        torch.cuda.reset_peak_memory_stats(device)
        latency_ms = _time_cuda(fn)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchResult(
            name=name, seq_len=seq_len,
            latency_ms=float("nan"), tflops=float("nan"), peak_mem_gb=float("nan"),
            error="OOM",
        )

    flops = attention_flops(batch, heads, seq_len, head_dim, causal)
    tflops = flops / (latency_ms / 1000) / 1e12

    return BenchResult(name=name, seq_len=seq_len, latency_ms=latency_ms, tflops=tflops, peak_mem_gb=peak_mem_gb)


def run_all(
    batch: int = 2,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    causal: bool = False,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> list[BenchResult]:
    device = device or torch.device("cuda")
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    impls = {
        "naive": lambda: attention_naive(q, k, v, causal=causal),
        "sdpa": lambda: attention_sdpa(q, k, v, causal=causal),
        "triton": lambda: triton_attention(q, k, v, causal=causal),
    }

    results = []
    for name, fn in impls.items():
        results.append(bench_one(name, fn, batch, heads, seq_len, head_dim, causal, device))
    return results


# Phase 2 sweep. Fixed batch/heads/head_dim per the brief (Section 4.3);
# only seq_len varies. Powers of 2 512->8192 — wide enough that naive's
# O(N^2) intermediate is expected to OOM before the top end on a 16GB-class
# GPU (2*8*8192*8192*2 bytes ~ 2.1GB for the score matrix alone, per batch
# element considered — the point is to find where it breaks, not to avoid it).
SEQ_LEN_SWEEP = (512, 1024, 2048, 4096, 8192)


def run_seqlen_sweep(
    batch: int = 2,
    heads: int = 8,
    head_dim: int = 64,
    causal: bool = False,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> list[BenchResult]:
    device = device or torch.device("cuda")
    results = []

    for seq_len in SEQ_LEN_SWEEP:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

        impls = {
            "naive": lambda: attention_naive(q, k, v, causal=causal),
            "sdpa": lambda: attention_sdpa(q, k, v, causal=causal),
            "triton": lambda: triton_attention(q, k, v, causal=causal),
        }
        for name, fn in impls.items():
            results.append(bench_one(name, fn, batch, heads, seq_len, head_dim, causal, device))

        # Free this seq_len's tensors before allocating the next, larger set —
        # otherwise peak-memory readings at seq_len N could be contaminated by
        # still-resident tensors from seq_len < N.
        del q, k, v
        torch.cuda.empty_cache()

    return results


def write_results_markdown(results: list[BenchResult], path: str) -> None:
    lines = [
        "# Benchmark results",
        "",
        "Phase 2 sequence-length sweep. batch=2, heads=8, head_dim=64, fp16, "
        "non-causal (see SEQ_LEN_SWEEP in benchmark.py). Generated by "
        "scripts/02_benchmark.py — not hand-edited.",
        "",
        "| seq_len | impl | latency (ms) | TFLOP/s | peak mem (GB) |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.seq_len} | {r.name} | {r.error} | {r.error} | {r.error} |")
        else:
            lines.append(
                f"| {r.seq_len} | {r.name} | {r.latency_ms:.3f} | {r.tflops:.2f} | {r.peak_mem_gb:.3f} |"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_latency_vs_seqlen(results: list[BenchResult], path: str) -> None:
    """Log-log latency plot. Log-log is deliberate, not decorative: it makes
    polynomial scaling visible as a straight line whose slope is the
    exponent, so naive's O(N^2) vs the O(N)-memory kernels should visibly
    separate in slope, not just in height.
    """
    import matplotlib.pyplot as plt

    # Fixed categorical order (dataviz skill palette, light-mode slots 1-3),
    # kept consistent with whatever other plots this project adds later.
    colors = {"naive": "#2a78d6", "sdpa": "#008300", "triton": "#e87ba4"}

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name in ("naive", "sdpa", "triton"):
        xs = [r.seq_len for r in results if r.name == name]
        ys = [r.latency_ms for r in results if r.name == name]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                 color=colors[name], label=name)

        # Direct end-label — required mitigation for the magenta slot's
        # light-surface contrast, and just generally clearer than legend-only.
        valid = [(x, y) for x, y in zip(xs, ys) if y == y]  # drop NaNs (OOM)
        if valid:
            last_x, last_y = valid[-1]
            ax.annotate(name, (last_x, last_y), textcoords="offset points",
                        xytext=(6, 0), fontsize=9, color=colors[name], va="center")

        # Explicitly mark where a series stops (OOM), rather than letting it
        # silently vanish from the line.
        if len(valid) < len(xs):
            ax.annotate("OOM beyond this point", (last_x, last_y),
                        textcoords="offset points", xytext=(6, -14),
                        fontsize=8, color="#898781", style="italic")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(SEQ_LEN_SWEEP))
    ax.set_xticklabels([str(s) for s in SEQ_LEN_SWEEP])
    ax.set_xlabel("sequence length", color="#52514e")
    ax.set_ylabel("latency (ms, log scale)", color="#52514e")
    ax.set_title("Attention latency vs sequence length", color="#0b0b0b", fontsize=13)
    ax.tick_params(colors="#898781")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.legend(frameon=False, labelcolor="#52514e")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_peakmem_vs_seqlen(results: list[BenchResult], path: str) -> None:
    """The clearest single visual for this project's actual thesis: naive's
    O(N^2) materialised score matrix vs Triton/SDPA's O(N) memory footprint.
    """
    import matplotlib.pyplot as plt

    colors = {"naive": "#2a78d6", "sdpa": "#008300", "triton": "#e87ba4"}

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name in ("naive", "sdpa", "triton"):
        xs = [r.seq_len for r in results if r.name == name]
        ys = [r.peak_mem_gb for r in results if r.name == name]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                 color=colors[name], label=name)
        valid = [(x, y) for x, y in zip(xs, ys) if y == y]
        if valid:
            last_x, last_y = valid[-1]
            ax.annotate(name, (last_x, last_y), textcoords="offset points",
                        xytext=(6, 0), fontsize=9, color=colors[name], va="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(SEQ_LEN_SWEEP))
    ax.set_xticklabels([str(s) for s in SEQ_LEN_SWEEP])
    ax.set_xlabel("sequence length", color="#52514e")
    ax.set_ylabel("peak memory (GB)", color="#52514e")
    ax.set_title("Peak memory vs sequence length: O(N²) vs O(N)", color="#0b0b0b", fontsize=13)
    ax.tick_params(colors="#898781")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.legend(frameon=False, labelcolor="#52514e")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from .env import get_device

    device = get_device()
    results = run_all(device=device)
    print(f"{'impl':<8} {'latency (ms)':>14} {'TFLOP/s':>10} {'peak mem (GB)':>15}")
    for r in results:
        print(f"{r.name:<8} {r.latency_ms:>14.3f} {r.tflops:>10.2f} {r.peak_mem_gb:>15.3f}")
