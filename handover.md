# Handover Brief — gpu-attention project, state after Phase 2

**For a fresh AI agent (or the human) picking this up without prior context.**
Repo: `github.com/akashg71/gpu-attention` (public). This doc is a snapshot —
if it's been a while, check `instructions.md`'s "Progress Log" section and
recent commits for anything that's moved on since this was written.

## What this project is

A portfolio project demonstrating GPU inference-systems skill: a
FlashAttention-style fused attention kernel written in Triton, benchmarked
against a naive PyTorch baseline and PyTorch's built-in
`scaled_dot_product_attention` (SDPA), with the goal of *explaining* the
result in hardware terms (memory traffic, roofline position) rather than
claiming to beat SDPA — SDPA is production FlashAttention-2/3 and beating it
isn't the point. Full original spec is `instructions.md`'s main body (the
"Handover Brief" sections before "Progress Log"); that's the plan, this file
and the Progress Log are what actually happened.

## Where this runs

**Not on the author's Mac** — no CUDA there. Everything GPU-dependent runs on
a rented GCP box:
- Project `project-9d8f69e0-e809-4fab-b37`, instance `instance-20260718-143824`,
  zone `europe-west2-b`, **Spot** provisioning, Tesla T4, Deep Learning VM
  image (Ubuntu 22.04, CUDA 12.9, driver reports CUDA 13.0).
- Connect: `gcloud compute ssh akashg7171_com@instance-20260718-143824` — that
  exact username (differs from the local Mac's own gcloud-derived username).
- Cost control: `gcloud compute instances stop/start instance-20260718-143824`
  between sessions. Spot billing is per-second while running only; disk
  persists regardless of running state.
- Being Spot, restarting can occasionally hit a stockout (no T4 capacity in
  that zone at that moment) — if so, retry, or fall back to Standard
  provisioning / a different zone.
- GPU clocks need re-locking every session (driver-level lock doesn't survive
  stop/start): see RUNBOOK.md's "Reducing benchmark variance" section —
  `sudo nvidia-smi -lgc 1590` (1590MHz is this T4's max graphics clock).

## What's actually been done

**Phase 0 (env + smoke test) — done.** Kernel (`src/gpu_attention/triton_attention.py`)
written blind (no GPU access while writing it), then verified for real:
passes `torch.allclose` against a plain-PyTorch reference on a single shape.
Needed zero kernel-API changes against the installed Triton (3.7.1) despite
being written speculatively — the only fix needed was an environment gap
(missing `python3-dev`, required for Triton's JIT C-compile step).

**Phase 1 (correctness sweep) — 80/80 passed**, then a gap was found in the
sweep's *own* design and one more case added. Grid: seq_len × head_dim ×
batch × causal × dtype (fp16/bf16). Two notable findings:
- Expected bf16 might fail outright — T4 is Turing (sm75), which lacks
  native bf16 tensor-core support (that's Ampere+). It didn't fail; all bf16
  cases passed. This shows bf16 is *correct* here, says nothing about
  whether it's *efficient* (untested, performance is a different question).
- Every original seq_len (128/512/1024/2048/4096) was a power of 2, so the
  kernel's boundary-masking code (for when seq_len isn't a multiple of the
  block size) was never actually exercised despite 80/80 passing. Added
  `seq_len=1000` to close the gap (commit `a5e93a5`) — **confirmed by re-run:
  96/96 passed**, including all 16 `seq_len=1000` rows. Phase 1 is fully
  closed, no open threads.

**Phase 2 (benchmark sweep) — done, no open threads.**
Swept seq_len 512→8192 (batch=2, heads=8, head_dim=64, fp16, non-causal).
Key results (`results/benchmarks.md`, `results/figures/*.png`):
- **Triton loses to naive at every seq_len tested, and the gap widens with
  N** (~1.3x slower at 512, ~1.5x slower at 8192) — this refuted an earlier
  hypothesis that Triton might catch up once naive's O(N²) cost got
  expensive enough. It doesn't. SDPA's TFLOP/s triples across the sweep
  (4.23 → ~13.6) as small-N launch overhead washes out; Triton's stays flat
  (~1.14–1.39) the whole time — it isn't getting more efficient at scale
  the way SDPA does. **Root cause not yet known — that's what Phase 3's
  `ncu` profiling is for.** Don't speculate further without it.
- Naive never actually OOM'd in this range (10GB peak at seq_len=8192, T4
  has 14.6GB) — the brief expected a memory cliff; it exists past this
  range, just not hit yet. Memory story is otherwise the cleanest result in
  the whole project: naive's O(N²) growth vs Triton/SDPA's flat O(N) is
  visually unambiguous in `peak_mem_vs_seqlen.png`.
- **SDPA anomaly (Phase 0's 6.0–7.7 TFLOP/s vs Phase 2's 12.6–12.65 TFLOP/s
  at the identical seq_len=1024 shape) — resolved**, via a dedicated
  diagnostic (`scripts/sdpa_diagnostic.py`). The original write-up here
  guessed "cold start" / backend dispatch setup cost; that turned out
  wrong. What actually explains it: **the same clock-variance issue already
  fixed via clock-locking**, not a separate problem. Reproducing "isolated"
  conditions with clocks locked gave 13.92 TFLOP/s — nowhere near Phase 0's
  slow numbers — which rules out cold-start as the mechanism (isolated
  should have reproduced the slow number if that were true; it didn't).
  Phase 0's original three runs already showed a 22% spread *among
  themselves* with zero context difference, all pre-clock-lock — sufficient
  on its own to explain swings this size. Backend selection was checked
  explicitly (not the mechanism — pinning a backend still shows a small
  gap, in the *opposite* direction from the original anomaly, consistent
  with ordinary noise) but confirmed something real regardless: **Flash
  Attention is hardware-incompatible with this T4** (PyTorch's own error:
  "Flash attention only supports gpu architectures in the range [sm80,
  sm121]. Attempting to run on a sm 7.5 gpu.") — SDPA runs via the
  memory-efficient backend on this box, confirmed fact now.
- Also fixed along the way: a peak-memory measurement bug (autotuning's
  internal search allocated a scratch buffer that got miscounted as
  steady-state memory — fixed by moving `reset_peak_memory_stats()` to
  after warmup), a kernel performance bug (missing `@triton.autotune`, fixed,
  ~2.5x speedup), and a plot label-collision bug (two lines converging to
  near-identical values rendered overlapping text — fixed with pixel-space
  collision detection in `_place_end_labels()`).

## Repo structure quick reference

```
src/gpu_attention/
    env.py              — GPU/CUDA/torch/triton version check
    reference.py        — correctness oracle + naive + SDPA wrapper
    triton_attention.py — the kernel (verified correct; performance gap open)
    correctness.py       — Phase 1 sweep logic + tolerances
    benchmark.py          — Phase 2 sweep logic, plotting, OOM handling
    roofline.py, kvcache.py, quant.py — Phase 3/4 stubs, not implemented
scripts/
    00_smoke_test.py — Phase 0 gate (passing)
    01_correctness.py — Phase 1 sweep runner (80/80 last confirmed run)
    02_benchmark.py    — Phase 2 sweep runner
    03_profile.py, 04_extension.py — Phase 3/4 stubs
```

Full narrative + checkboxed history: `instructions.md`'s "Progress Log"
section. Concept reference notes (kernel theory + cloud/GPU-infra mechanics
learned while deploying): `concepts.md`. Operational "what do I type":
`RUNBOOK.md`.

## What's next: Phase 3 — Profile

Not started. This is meant to be the differentiator of the whole project:
Nsight Compute (`ncu`) on the naive vs Triton kernels, pulling HBM bytes
moved, memory throughput, occupancy, warp-stall reasons — then a roofline
plot placing each kernel by arithmetic intensity. This is also where the
Phase 2 open question (why is Triton's TFLOP/s flat instead of improving
with scale?) should actually get answered, instead of guessed at.

Prerequisites: `ncu`/`nsys` need to be present or installed
(`sudo apt install -y nsight-compute nsight-systems`) — this needs real GPU
performance-counter access, which requires root on the box (have it here,
this being a self-managed VM, not a hosted notebook) but may still need the
driver's `NVreg_RestrictProfilingToAdminUsers` flag checked/cleared if `ncu`
throws a permissions error (`ERR_NVGPUCTRPERM`) — not yet tested on this box.

## Notes for whichever agent picks this up

- **Verify before recommending.** This doc is a snapshot; file paths,
  function names, and results may have moved on. Check `git log` and the
  actual files before telling the user something exists or works.
- The user is a senior backend engineer, new to CUDA/Triton/GPU profiling —
  comment/explain generously, don't assume familiarity with GPU-specific
  vocabulary (see `concepts.md` for what's already been explained).
- Phases 0-2 have no open threads as of this writing — both loose ends noted
  in earlier drafts of this doc (the Phase 1 extended-grid re-run, and the
  SDPA anomaly) were resolved and are recorded above. Phase 3 is next.
