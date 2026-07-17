# RUNBOOK

Short and operable. This is the "what do I actually type" doc — README.md has
the project narrative.

## 0. Get a GPU machine

Your Mac cannot run any part of this repo past editing text files. Pick one:

- **Free, fastest to start:** Google Colab or Kaggle, free T4 GPU. Fine for
  Phases 0–2 (kernel correctness + benchmarking). Not fine for Phase 3 (`ncu`
  needs elevated privileges Colab/Kaggle block — see below).
- **Rented, for Phase 3 profiling and anything cross-generation:** RunPod,
  Lambda Labs, or Vast.ai. Get a box with sudo (needed for `ncu` counter
  access) — an A100, L4, or H100 instance.

## 1. Get the code onto the GPU box

```bash
git clone <your-repo-url> gpu-attention
cd gpu-attention
```

(If you're working in a Colab/Kaggle notebook instead of a real shell, `!git
clone ...` in a cell, then `%cd gpu-attention`.)

## 2. Environment setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Install torch FIRST, matching the box's CUDA version — do NOT just
# `pip install torch`, that can silently grab a CPU-only or mismatched-CUDA
# build. Check https://pytorch.org/get-started/locally/ for the exact
# command for your CUDA version, e.g. for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Then confirm what you actually got:

```bash
python -m gpu_attention.env
```

Expected output shape:
```
Python:  3.11.x
torch:   2.x.x+cuXXX
CUDA:    12.x
GPU:     NVIDIA <name>
SM count: NN, compute capability: N.N
VRAM:    NN.N GB
triton:  3.x.x
```

If `triton` is missing after installing a CUDA torch build on Linux, install
it separately: `pip install triton`.

## 3. Phase 0 gate: smoke test

```bash
python scripts/00_smoke_test.py
```

**This is expected to fail or need edits on the first run.** The kernel in
`src/gpu_attention/triton_attention.py` was written without a GPU to test
against, and the file's header docstring says exactly that. Before touching
the kernel code:

1. Check your installed Triton version: `python -c "import triton; print(triton.__version__)"`.
2. Open that version's own fused-attention tutorial on GitHub
   (`triton-lang/triton`, path `python/tutorials/06-fused-attention.py`,
   checked out at the tag matching your installed version) and diff the
   kernel API against what's in `triton_attention.py` — `tl.dot` signature,
   `tl.exp` vs `tl.math.exp`, whether block pointers are expected.
3. Fix `triton_attention.py` to match, re-run the smoke test.

When it passes you'll see:
```
PASS
max_abs_err=0.00XX max_rel_err=0.00XX
```

## 4. Phase 0 benchmark stub

```bash
python scripts/02_benchmark.py
```

Prints a latency / TFLOP/s / peak-memory table for naive / SDPA / Triton on
one fixed shape (batch=2, heads=8, seq_len=1024, head_dim=64, fp16). Don't
read anything into the numbers yet — this is the harness working, not a
result to report. Phase 2 does the real sweep.

## 5. Nsight tools (Phase 3 only — not needed yet)

```bash
which ncu nsys   # check if already on the box (common on cloud GPU images)
```

If missing and you have sudo:
```bash
# Ubuntu/Debian example — exact package name/version varies by CUDA toolkit version
sudo apt update && sudo apt install -y nsight-compute nsight-systems
```

`ncu` needs GPU performance-counter access that Colab/Kaggle block outright —
don't burn time trying to get Phase 3 working on the free tier, move to a
rented box with sudo for this phase.

## 6. Common gotchas

- **`torch.cuda.is_available()` is False**: wrong torch build (CPU-only) or
  no GPU attached to the instance. Re-check step 2's install command.
- **OOM on the naive baseline at large seq_len**: expected and part of the
  Phase 2 story (naive materialises O(N²) memory, Triton doesn't) — capture
  it, don't "fix" it by lowering seq_len silently.
- **`torch.allclose` fails by a small margin in fp16/bf16**: check
  `TOLERANCES` in `src/gpu_attention/correctness.py` — fp16/bf16 need looser
  tolerances than fp32, this is expected numerically, not necessarily a bug.
  If the error is large (not a small margin), that's a real correctness bug.
