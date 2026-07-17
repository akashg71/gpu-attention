# gpu-attention

A FlashAttention-style fused attention kernel written in Triton, benchmarked
against a naive PyTorch baseline and `torch.nn.functional.scaled_dot_product_attention`
(SDPA), and profiled to explain the speedup in hardware terms — measured HBM
bytes moved, memory throughput, and roofline position.

**Thesis:** transformer inference attention is largely memory-bound, not
compute-bound. The craft is minimising HBM traffic by keeping intermediate
results in on-chip SRAM. This repo builds that kernel, then proves the
memory-bound claim with a profiler trace rather than asserting it.

**Not the claim being made:** that this kernel beats PyTorch's SDPA. SDPA is
backed by FlashAttention-2/3 and is extremely well optimised — beating it
isn't realistic and isn't the point. The deliverable is the analysis: naive
vs SDPA vs this kernel, with `ncu` explaining exactly where the naive
implementation's HBM traffic goes and why the fused kernel avoids it.

## Requirements

**A real NVIDIA GPU with CUDA. Nothing in this repo runs on macOS, including
this scaffold's own kernel and correctness tests.** Triton does not ship
macOS wheels — it compiles to PTX and needs an NVIDIA driver present at
compile time, full stop. If you're reading this on a Mac, that's expected:
this repo was scaffolded here, but every script that touches `torch.cuda` or
`triton` needs to run on a GPU box. See RUNBOOK.md for where to get one.

## Status (as of this scaffold)

Phase 0 scaffolding is done, but **the kernel is unverified** — it was
written without a GPU to compile or run it against. The very first thing to
do on a GPU box is `python scripts/00_smoke_test.py` and expect to debug it.
See the warning docstring at the top of
[src/gpu_attention/triton_attention.py](src/gpu_attention/triton_attention.py)
for exactly what's unverified and what to check against your installed
Triton version's own tutorial.

## Repo layout

```
src/gpu_attention/
    env.py             — print GPU/CUDA/torch/triton versions, get device
    reference.py        — plain PyTorch attention (correctness oracle) + naive baseline + SDPA wrapper
    triton_attention.py — the fused-attention Triton kernel (forward only) — UNVERIFIED, see file header
    correctness.py       — allclose checks vs reference, fp16/bf16 tolerances
    benchmark.py          — CUDA-event timing: latency, TFLOP/s, peak memory
    roofline.py            — Phase 3 stub: peak FLOP/s + HBM BW, arithmetic intensity, roofline plot
    kvcache.py               — Phase 4a stub: KV-cache decode loop
    quant.py                  — Phase 4b stub: weight-only quantization study
scripts/
    00_smoke_test.py — Phase 0 gate: kernel + allclose vs reference, run this first
    01_correctness.py — Phase 1: correctness sweep across shapes (not implemented)
    02_benchmark.py    — single-shape benchmark; Phase 2 extends to a seq-len sweep
    03_profile.py       — Phase 3 stub: emits ncu/nsys commands (not implemented)
    04_extension.py      — Phase 4 stub: pick kvcache.py or quant.py (not implemented)
results/
    benchmarks.md — numbers table, populated as phases complete
    figures/        — plots (gitignored dumps excluded, committed figures kept)
    traces/           — ncu/nsys output (gitignored — large, machine-specific)
```

## Quickstart (on a GPU box)

See RUNBOOK.md for the full walkthrough. Short version:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m gpu_attention.env          # confirm GPU/CUDA/torch/triton versions
python scripts/00_smoke_test.py      # Phase 0 gate — kernel vs reference
python scripts/02_benchmark.py       # single-shape latency/TFLOP/s/memory
```

## Project phases

- **Phase 0 — Env + smoke** (this scaffold): GPU env confirmed, kernel
  written, smoke test wired up. **Not yet run on a GPU.**
- **Phase 1 — Correct kernel:** validate across seq len / head dim / batch /
  causal+non-causal, fp16/bf16 tolerances.
- **Phase 2 — Benchmark:** naive vs SDPA vs Triton, sweep sequence length,
  latency + TFLOP/s + peak memory, plots.
- **Phase 3 — Profile (the differentiator):** Nsight Compute — HBM bytes
  moved, memory throughput, occupancy, warp stalls; roofline plot tying the
  speedup to reduced HBM traffic.
- **Phase 4 — Extension:** KV-cache decode with int8 KV-cache, *or* a
  weight-only int8/int4 quantization study (pick one).
- **Phase 5 — Writeup:** roofline, benchmark plots, ncu traces, the
  memory-bound explanation, honest "didn't beat SDPA, here's why."

## What to run next

**Phase 1**, on an actual GPU machine:
1. `python -m gpu_attention.env` — confirm what you're actually running on.
2. `python scripts/00_smoke_test.py` — this is the real starting point. It
   will very likely need fixes against your installed Triton's API (see the
   header of `triton_attention.py`). Debugging this file *is* the first unit
   of work, not a sign the scaffold failed.
3. Once smoke test passes, flesh out `scripts/01_correctness.py` to sweep
   seq_len ∈ {128, 512, 1024, 2048, 4096}, head_dim ∈ {64, 128}, batch ∈ {1,
   4}, causal ∈ {True, False}, dtype ∈ {fp16, bf16}.
4. Report back with the smoke test output (PASS/FAIL + max abs error) and
   anything you had to change in the kernel — that tells us whether to keep
   going or debug further before Phase 2.
