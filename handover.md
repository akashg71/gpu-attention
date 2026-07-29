# Handover Brief — gpu-attention project, state mid-Phase 3

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

## Phase 3 — Profile (DONE)

The differentiator phase, and it delivered: Nsight Compute (`ncu`) on naive
vs Triton, real HBM bytes moved, the roofline plot, and — the actual payoff —
a hardware-grounded answer to Phase 2's open question (why was Triton's
TFLOP/s flat across the seq_len sweep instead of improving like SDPA's did?).

**The finding, i.e. the point of the whole project:**
- Triton moves **~1.01GB** of HBM traffic for one call at seq_len=2048;
  naive moves **~2.10GB** — almost exactly half, confirming the O(N) vs
  O(N²) memory thesis with real measured data, not just theory. Arithmetic
  intensity: naive 8.19 FLOPs/byte, Triton 16.96 (~2x, consistent with half
  the bytes at equal FLOPs). Both sit far below the T4's roofline ridge
  point (~217 FLOPs/byte for fp16) — attention is memory-bound on this
  hardware regardless of implementation. See `results/figures/roofline.png`.
- **But Triton is still slower in wall-clock time** (12.49ms vs naive's
  8.59ms) despite moving half the bytes. Nsight Compute's `--set full`
  (occupancy + warp-stall data) explains why: **Theoretical Occupancy is
  capped at 25%** for Triton's steady-state kernel launches — not a
  failure to reach potential, the kernel sits right at that ceiling
  (achieved 24.7%). Most likely cause: register pressure from holding the
  online-softmax accumulators (running max/sum/output) largely in
  registers, limiting how many concurrent thread-blocks fit per SM.
- That occupancy ceiling is the single root cause of the earlier "low DRAM%
  and low Compute% at the same time" pattern — not two problems, one
  problem (not enough concurrent warps to hide memory latency *or* keep
  compute pipelines fed) with two symptoms. Dominant stall reason: ~53% of
  stall cycles are warps waiting on the MIO queue (shared-memory/special-
  instruction pipeline) being full; Nsight's own analysis estimates up to a
  **50.95% speedup** if that specific stall were eliminated.
- **Full causal chain**: small tile config → high per-thread register usage
  → few concurrent thread-blocks per SM → 25% occupancy ceiling →
  insufficient parallelism to hide latency → low utilization on both memory
  and compute → despite moving half the bytes naive does, still loses on
  wall-clock time. Plain-English version + kitchen analogy: `concepts.md`.
- **Confirmed via `ncu`'s own Occupancy section** (not inferred): the
  steady-state launch (213 registers/thread, 16.38KB shared memory/block)
  hits `Block Limit Registers = 2` AND `Block Limit Shared Mem = 2` — tied,
  both independently capping at 2 blocks/SM. Register math checks out
  exactly: 65,536 registers/SM ÷ (213 × 128 threads/block) ≈ 2.4, floors to
  2. Registers are the harder constraint (would still cap at 2 even if
  shared memory weren't limiting), which motivated the experiment below.
- **Experiment tried, negative result**: added 3 `maxnreg=128` configs
  (commit `1ce23a3`) to let autotuning test whether capping registers
  (more concurrent blocks, less scratch space each) actually helps. Re-ran
  the full sweep — no meaningful change anywhere (seq_len=2048, the
  profiled shape, went from ~12.49ms to 12.528ms, within existing noise;
  seq_len=8192 got slightly *worse*). Autotuning almost certainly tried and
  rejected the new configs. Likely reason: the dominant stall (MIO pipeline
  pressure, ~53% of stalls) is a *different* bottleneck than occupancy —
  more warps competing for the same shared-memory-access pipeline doesn't
  relieve contention on that pipeline, and can worsen it. Real fix would be
  restructuring the kernel's inner loop for fewer, wider memory loads
  (Nsight's own suggestion) — genuine kernel engineering, still correctly
  out of scope. Configs left in place as an honest documented negative
  result, not reverted.

**Bugs found and fixed getting here** (real ones, not guesses — each
confirmed by actually running on this box): no `ERR_NVGPUCTRPERM`
permission wall (this GCP box's real root + PCIe-passthrough GPU sidesteps
the Colab/Kaggle restriction); a `sudo` PATH gotcha (`ncu`/`nsys` live at
`/usr/local/cuda/bin/`, on the user's PATH but not `sudo`'s `secure_path` —
fixed via `shutil.which()` + absolute paths); `ncu`'s CSV is long-format
(one row per kernel-launch-×-single-metric, not one row per launch with
metrics as columns); and a bytes-aggregation bug caught *before* trusting
the result — the first real computation showed Triton moving 2x *more*
bytes than naive (the opposite of the point of the kernel), traced to
wrongly summing across repeated launches of the same kernel (correct for
naive's 6 *different* sequential kernels, wrong for Triton's repeated
launches of *one* kernel — fixed to average instead).

Full narrative with all intermediate (broken) numbers, in the order they
actually happened: `instructions.md`'s Progress Log, Phase 3 section. Raw
terminal logs: `results/logs/phase3_profile_run1.md`, `phase3_profile_run2.md`,
`phase_3_profile_full_run1.md`.

Reference for next session: `ncu`/`nsys` at `/usr/local/cuda/bin/`. Re-lock
clocks each session (`sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1590` —
doesn't survive stop/start).

## Notes for whichever agent picks this up

- **Verify before recommending.** This doc is a snapshot; file paths,
  function names, and results may have moved on. Check `git log` and the
  actual files before telling the user something exists or works.
- The user is a senior backend engineer, new to CUDA/Triton/GPU profiling —
  comment/explain generously, don't assume familiarity with GPU-specific
  vocabulary (see `concepts.md` for what's already been explained).
- Phases 0-2 have no open threads as of this writing — both loose ends noted
  in earlier drafts of this doc (the Phase 1 extended-grid re-run, and the
  SDPA anomaly) were resolved and are recorded above.
- Phase 3 is actively in progress, not "next" — three real bugs found and
  fixed getting the profiling harness working at all (see above), but the
  actual payoff (real bytes-moved numbers, the roofline plot) hasn't
  happened yet. Don't report Phase 3 as unstarted or as done — it's
  mid-flight. The concrete next action is spelled out above; do that
  before anything else in this phase.
