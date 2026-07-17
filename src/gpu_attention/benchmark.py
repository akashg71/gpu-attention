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
    latency_ms: float
    tflops: float
    peak_mem_gb: float


def _time_cuda(fn: Callable[[], torch.Tensor], warmup: int = 10, iters: int = 50) -> float:
    """Median latency in ms over `iters` runs, timed with CUDA events.
    Warms up first so the first-call kernel-compile/cache cost (Triton
    autotuning, cuDNN algo search, etc.) doesn't pollute the measurement.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

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
    torch.cuda.reset_peak_memory_stats(device)
    latency_ms = _time_cuda(fn)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    flops = attention_flops(batch, heads, seq_len, head_dim, causal)
    tflops = flops / (latency_ms / 1000) / 1e12

    return BenchResult(name=name, latency_ms=latency_ms, tflops=tflops, peak_mem_gb=peak_mem_gb)


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


if __name__ == "__main__":
    from .env import get_device

    device = get_device()
    results = run_all(device=device)
    print(f"{'impl':<8} {'latency (ms)':>14} {'TFLOP/s':>10} {'peak mem (GB)':>15}")
    for r in results:
        print(f"{r.name:<8} {r.latency_ms:>14.3f} {r.tflops:>10.2f} {r.peak_mem_gb:>15.3f}")
