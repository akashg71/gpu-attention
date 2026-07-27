"""
Diagnostic for the open SDPA anomaly in instructions.md's Progress Log
(Phase 2): SDPA measured ~6-7.7 TFLOP/s in Phase 0's isolated single-shape
runs vs ~12.6 TFLOP/s in Phase 2's sweep, at the identical shape
(batch=2, heads=8, seq_len=1024, head_dim=64, fp16, non-causal). Naive and
Triton did not show this gap between the same two contexts — only SDPA did.

"Cold start" turns out not to be quite right as a framing: naive always runs
immediately before SDPA at the same shape in both scripts, so SDPA is never
literally the first GPU op in either. The sharper hypothesis: SDPA
auto-selects among multiple backends (flash / memory-efficient / math), and
something about running after prior CUDA activity in the same process (a
full round at seq_len=512, in the sweep's case) changes which one gets
picked. This script checks that directly by forcing each backend explicitly.

Uses torch.nn.attention.sdpa_kernel / SDPBackend, the torch 2.1+ API for
this. If this import fails, the installed torch version's backend-selection
API has moved — check `python -c "import torch; help(torch.nn.attention)"`
and adjust (this project targets torch 2.13.0, ahead of general knowledge
of the exact API at any fixed point in time — verify, don't assume).

    python scripts/sdpa_diagnostic.py
"""
import sys

sys.path.insert(0, "src")

import torch

from gpu_attention.env import get_device
from gpu_attention.reference import attention_sdpa
from gpu_attention.benchmark import bench_one

SHAPE = dict(batch=2, heads=8, seq_len=1024, head_dim=64, causal=False)


def _prime(device: torch.device) -> None:
    """Throwaway round at a different shape, mimicking what run_seqlen_sweep()
    does at seq_len=512 before it ever reaches seq_len=1024.
    """
    pq = torch.randn(2, 8, 512, 64, device=device, dtype=torch.float16)
    pk = torch.randn(2, 8, 512, 64, device=device, dtype=torch.float16)
    pv = torch.randn(2, 8, 512, 64, device=device, dtype=torch.float16)
    for _ in range(10):
        attention_sdpa(pq, pk, pv, causal=False)
    torch.cuda.synchronize()
    del pq, pk, pv
    torch.cuda.empty_cache()


def measure(device: torch.device, prime_first: bool, backend=None):
    if prime_first:
        _prime(device)

    q = torch.randn(SHAPE["batch"], SHAPE["heads"], SHAPE["seq_len"], SHAPE["head_dim"],
                     device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    fn = lambda: attention_sdpa(q, k, v, causal=SHAPE["causal"])

    if backend is not None:
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel([backend]):
            return bench_one("sdpa", fn, **SHAPE, device=device)
    return bench_one("sdpa", fn, **SHAPE, device=device)


def main():
    device = get_device()

    print("=== Part 1: does isolated-vs-primed reproduce the Phase 0/2 gap? ===")
    cold = measure(device, prime_first=False)
    primed = measure(device, prime_first=True)
    print(f"isolated (no priming): {cold.latency_ms:.3f}ms  {cold.tflops:.2f} TFLOP/s")
    print(f"primed (512 first):    {primed.latency_ms:.3f}ms  {primed.tflops:.2f} TFLOP/s")
    print(f"ratio: {primed.tflops / cold.tflops:.2f}x\n")

    print("=== Part 2: does forcing one backend make isolated == primed? ===")
    print("(if yes: backend selection explains the gap. if a gap remains")
    print(" even with the backend pinned: it's something else, not this.)\n")
    try:
        from torch.nn.attention import SDPBackend
    except ImportError:
        print("torch.nn.attention.SDPBackend not importable on this torch build.")
        print("API has likely moved — check `help(torch.nn.attention)` and adjust this script.")
        return

    for backend_name, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]:
        try:
            c = measure(device, prime_first=False, backend=backend)
            p = measure(device, prime_first=True, backend=backend)
            print(f"{backend_name:<20} isolated={c.tflops:>6.2f} TFLOP/s   "
                  f"primed={p.tflops:>6.2f} TFLOP/s   ratio={p.tflops / c.tflops:.2f}x")
        except RuntimeError as e:
            print(f"{backend_name:<20} not usable for this shape/hardware: {e}")


if __name__ == "__main__":
    main()
