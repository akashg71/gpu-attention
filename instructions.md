# Handover Brief — Project 2: GPU Inference Optimization (Triton Fused-Attention Kernel)

**Paste this whole document into Claude Code (agent mode) on a GPU machine, in an empty project folder.** Read it fully before acting.

**Hard requirement up front:** this project needs a real **NVIDIA GPU** (CUDA). Triton does not run on a Mac / Apple MPS / CPU. Do everything here on a GPU box — Colab or Kaggle (free T4) to start; a rented A100 / L4 / H100 (RunPod, Lambda, Vast) for the profiling and cross-generation work later.

---

## Progress Log

*Added by Claude Code to track actual progress against this brief. Checkboxes
reflect what's actually been done and verified, not what's planned — see each
Result line for specifics.*

### Phase 0 — Env + smoke

- [x] **Confirm GPU environment** (Section 0, step 1)
  - Result: Tesla T4, driver reports CUDA 13.0, toolkit 12.9, Python 3.10.12,
    torch 2.13.0+cu130, triton 3.7.1. Runs on a rented GCP Compute Engine box
    (local dev machine is a Mac, no CUDA — see infra notes at the bottom).
- [x] **Initialise git repo + create structure** (Section 3)
  - Result: done, matches the spec. Pushed to
    `github.com/akashg71/gpu-attention` (public).
- [x] **requirements.txt + install deps**
  - Result: installed in a venv (Python 3.10) on the GPU box. `pip install
    torch` alone picked up a CUDA-matched build (cu130) — no manual index URL
    needed on this box.
- [x] **One fused-attention Triton kernel running, validated vs PyTorch
      reference on one shape**
  - Result: **PASS**. `max_abs_err=0.000610` (fp16, shape
    batch=2/heads=4/seq_len=512/head_dim=64, non-causal). The kernel needed
    zero API changes against Triton 3.7.1 despite being written blind
    (no GPU available while writing it) — the only fix needed was an
    environment issue (missing `python3-dev`/`Python.h`, required for
    Triton's JIT C-compile step), not a kernel bug.
- [x] **Stub benchmark harness (naive vs SDPA vs Triton) runs on one shape**
  - Result: runs. See "Issues found & fixed" below — the first run's numbers
    were misleading and needed two follow-up fixes before they were
    trustworthy.
- [x] **README.md + RUNBOOK.md written**
  - Result: done. Also added `concepts.md` (not originally scoped) — running
    reference notes on kernel theory + cloud/GPU-infra mechanics learned
    while deploying.

### Issues found & fixed during Phase 0

- [x] **Bug: Triton kernel was slower than naive** (8.11ms vs 2.54ms,
      batch=2/heads=8/seq_len=1024/head_dim=64, fp16)
  - Root cause: kernel had zero autotuning — hardcoded `BLOCK_M=BLOCK_N=64`,
    default `num_warps`/`num_stages`, never tuned for the actual GPU.
  - Fix: added `@triton.autotune` over 7 configs to `_fwd_kernel`.
  - Result: 3.22ms (~2.5x faster). Still behind naive at this specific shape
    — open question, see Phase 1/2 below.
- [x] **Bug: peak memory reading contaminated by autotuning warmup**
  - Symptom: after adding autotuning, Triton's reported peak memory jumped
    0.016GB → 0.266GB — worse than naive, contradicting the whole point of
    the kernel.
  - Root cause: `torch.cuda.reset_peak_memory_stats()` ran *before* warmup in
    `benchmark.py`, and warmup is when the autotuning search happens —
    Triton's autotuner allocates its own scratch buffer to benchmark
    candidate configs fairly, and that one-time allocation was getting
    counted as steady-state memory.
  - Fix: split warmup out of `_time_cuda()` into its own `_warmup()` step so
    the reset happens after one-time costs settle.
  - Result: confirmed — peak memory returned to 0.016GB, exactly matching
    SDPA, after the fix.
- [ ] **Known gap, not yet fixed: GPU clocks not locked**
  - ~20% run-to-run variance observed in SDPA's measured TFLOP/s across 3
    separate runs (7.71 → 6.92 → 6.01). Brief explicitly calls for locking
    clocks to reduce this (Section 4.3). Flagged, not addressed — should be
    fixed before Phase 2's sweep numbers are treated as final.

### Latest single-shape benchmark

batch=2, heads=8, seq_len=1024, head_dim=64, fp16, non-causal — after both
fixes above:

| impl   | latency (ms) | TFLOP/s | peak mem (GB) |
|--------|-------------:|--------:|--------------:|
| naive  | 2.566        | 1.67    | 0.170         |
| sdpa   | 0.715        | 6.01    | 0.016         |
| triton | 3.224        | 1.33    | 0.016         |

Memory story now matches the thesis cleanly (naive ~10x Triton/SDPA — O(N²)
vs O(N) intermediate). Latency story doesn't yet — Triton still behind naive,
open question for the Phase 2 sweep (hypothesis: at seq_len=1024, naive's
O(N²) intermediate is still cheap enough that FlashAttention's advantage
hasn't kicked in — untested).

**Phase 0 status: DONE** — every item in Section 6's definition of done is
met.

### Phase 1 — Correct kernel
BankLogin!3akash
- [x] **Implement and run the correctness sweep** (Section 4.2: seq_len,
      head_dim, batch, causal on/off, fp16/bf16)
  - Grid: `seq_len ∈ {128, 512, 1024, 2048, 4096}`, `head_dim ∈ {64, 128}`,
    `batch ∈ {1, 4}`, `causal ∈ {False, True}`, `dtype ∈ {fp16, bf16}` — 80
    combinations, each isolated in try/except so one bad combo can't crash
    the whole sweep.
  - Result: **80/80 PASS** on the GPU box.
  - Correction to a concern raised while writing this: expected the 40 bf16
    combinations might fail to compile, since T4 (Turing/sm75) lacks native
    bf16 tensor-core support (that arrived with Ampere). They didn't fail —
    all passed. This only shows bf16 is *correct* here, not that it's
    *efficient* — untested, and not assumed either way.
  - Error pattern, and why it's informative rather than just numbers: fp16
    non-causal error shrinks as seq_len grows (0.000977 at N=128 → 0.000244
    at N=4096) because softmax over more keys spreads weight thinner,
    shrinking output magnitude, and absolute error tracks magnitude. Causal
    error instead sits at an exact constant (`0.001953125` fp16 = 2⁻⁹,
    `0.015625` bf16 = 2⁻⁶) almost everywhere — the first row in a causal
    block attends to ~1 key, so softmax there degenerates to copying a V
    value through dtype rounding, a fixed floor independent of seq_len.
- [x] **Found and closed a gap in the sweep's own coverage**
  - Every seq_len in the original grid was a power of 2, dividing evenly
    into every autotune block size (32/64/128) — meaning `_fwd_kernel`'s
    boundary-masking code (the `mask=...` args for when seq_len isn't a
    multiple of the block size) had never actually been exercised, despite
    80/80 passing.
  - Fix: added `seq_len=1000` to the grid (deliberately not a multiple of
    32, 64, or 128).
  - Result: pending re-run on the GPU box.

### Phase 2 — Benchmark

- [x] **Implement and run the seq_len sweep** (Section 4.3: naive vs SDPA vs
      Triton, latency + TFLOP/s + peak memory, 512→8192)
  - batch=2, heads=8, head_dim=64, fp16, non-causal. `bench_one()` now
    catches `torch.cuda.OutOfMemoryError` per-combination rather than
    letting it crash the sweep, since the brief expects naive to hit a
    memory wall at some point.
  - Result: naive never actually OOM'd in this range — 10.055GB peak at
    seq_len=8192, under the T4's 14.6GB. The expected memory cliff exists
    past this range (seq_len=8192 estimate ~2.1GB *per batch element* for
    the score matrix alone scales to ~17GB total by seq_len=16384), just
    not hit by 8192 — noted, not chased further for now.
  - **Correction to a hypothesis from Phase 0/1**: expected Triton might
    close the gap with naive at larger seq_len, since naive's O(N²)
    intermediate should get more expensive as N grows. **The data refutes
    this** — Triton loses to naive at every seq_len tested, and the gap
    *widens* with N (~1.3x slower at 512, ~1.5x slower at 8192). TFLOP/s
    tells the sharper story: SDPA's efficiency triples (4.23 → ~13.6) as N
    grows past small-N launch-overhead territory, while Triton's stays flat
    (~1.14–1.39) the entire sweep — it isn't getting more efficient at
    scale the way SDPA does. Root cause not chased further here — that's
    what Phase 3's `ncu` profiling is for (occupancy, warp stalls, actual
    memory throughput), not something to guess at from latency numbers
    alone.
  - Memory story remains clean: naive's O(N²) growth vs Triton/SDPA's
    O(N) is the clearest, least ambiguous result of this sweep — see
    `results/figures/peak_mem_vs_seqlen.png`.
- [x] **Reduce benchmark variance**: GPU clocks locked to 1590MHz (T4's max
      graphics clock, paired automatically with the 5001MHz memory clock)
      via `nvidia-smi -lgc` per the new RUNBOOK.md section. Needs redoing
      each session — this is a Spot VM, and losing the instance resets any
      driver-level clock lock.
- [x] **Found and fixed a plotting bug**: `sdpa` and `triton` converge to
      nearly identical peak memory at seq_len=8192 (0.070GB both), so their
      direct end-labels landed on the same pixel position in
      `peak_mem_vs_seqlen.png` and rendered as illegible overlapping text.
      Fixed with a shared label-placement helper that nudges colliding
      labels apart in rendered pixel space. Verified locally against the
      actual sweep data before pushing (matplotlib doesn't need the GPU box).
- [x] **Found and fixed a git-push gap on the GPU VM**: pushing from the box
      failed with `Password authentication is not supported` — the VM had
      never authenticated with GitHub (cloning a public repo needs no auth;
      pushing always does). Fixed by installing `gh` on the VM (Linux build)
      and running `gh auth login` there, same device-code flow as the local
      Mac setup. Will persist across stop/start (not tied to the Spot
      instance's ephemeral state the way the clock lock is), since it's
      stored in the VM's home directory on the persistent disk.
  - Confirmed: label fix verified against the real regenerated
    `peak_mem_vs_seqlen.png` (commit `f2dd570`), not just the local
    synthetic test — `sdpa`/`triton` labels render cleanly stacked.
- [x] **SDPA anomaly — resolved via `scripts/sdpa_diagnostic.py`.** The
      original "cold-start" framing didn't hold up: naive always runs
      immediately before SDPA at the same shape in both `run_all()` and
      `run_seqlen_sweep()`, so SDPA is never literally the first GPU op in
      either — the hypothesis as originally stated was wrong.
  - Actual root cause: **the same clock-variance issue already identified
    and fixed**, not a separate problem. Ran the diagnostic's "isolated"
    (no priming) case with clocks locked at 1590MHz — it measured 13.92
    TFLOP/s, nowhere near Phase 0's original 6.0–7.7 range and much closer
    to Phase 2's sweep number. If cold-start/priming were the mechanism,
    isolated should have reproduced the slow number; it didn't. Combined
    with Phase 0's own three original runs already showing a 22% spread
    *among themselves* with zero context difference (all fresh `run_all()`
    calls, all pre-clock-lock) — that alone is sufficient evidence that
    unlocked clock variance produces swings this large, without needing any
    process-context explanation at all.
  - Backend selection was checked explicitly and is not the mechanism (the
    residual 8–15% isolated-vs-primed gap under a pinned backend runs in
    the *opposite* direction from the original anomaly — consistent with
    ordinary measurement noise, not a repeatable effect) — but confirmed
    something real regardless: **Flash Attention is hardware-incompatible
    with this T4.** PyTorch's own error is explicit: *"Flash attention only
    supports gpu architectures in the range [sm80, sm121]. Attempting to
    run on a sm 7.5 gpu."* SDPA on this box runs via the memory-efficient
    backend — confirmed fact now, not an assumption. MATH backend also
    confirmed available as a (much slower, ~1.1 TFLOP/s) fallback.
- [x] **Phase 1 extended grid — confirmed.** Re-ran `01_correctness.py`:
      **96/96 passed**, including all 16 `seq_len=1000` rows exercising the
      boundary-masking path. Error magnitudes consistent with neighboring
      seq_lens, nothing anomalous. Phase 1 is genuinely closed now.

**Phase 2 status: DONE.** No open threads remain — sweep runs, plots and
table are correct, benchmark variance understood and addressed, and the
SDPA anomaly traced to the same clock-variance root cause rather than left
as an unresolved mystery.

### Phase 3 — Profile

**DONE.** Goal: Nsight Compute (`ncu`) on naive vs Triton, pulling real HBM
bytes moved + throughput, to finally answer Phase 2's open question (why is
Triton's TFLOP/s flat across the seq_len sweep instead of improving like
SDPA's does?) and produce the roofline plot. Answered — see the finding at
the end of this section, arrived at only after fixing several real bugs in
the profiling harness itself along the way (documented below in the order
they were actually hit, not cleaned up in hindsight).

- [x] **`roofline.py` implemented**: T4 peak specs from the published
      datasheet (65 TFLOPS fp16 tensor-core, 8.1 TFLOPS fp32, 300 GB/s HBM
      bandwidth), keyed by compute capability so it extends to other GPUs
      rather than being hardcoded to only this card. bf16 deliberately
      omitted — Turing (sm75) has no native bf16 tensor-core path, and its
      real achieved ceiling here is unmeasured, not something to guess at.
- [x] **No `ERR_NVGPUCTRPERM` permission error** — confirmed by just trying
      it. This GCP box (real root, PCIe-passthrough GPU, not a shared/hosted
      notebook) profiles fine; the permission wall the brief warned about
      (Section 4.4) is specifically a Colab/Kaggle problem, doesn't apply
      here. One less thing to worry about for the rest of this phase.
- [x] **Found and fixed: `sudo` PATH gotcha.** `ncu`/`nsys` live at
      `/usr/local/cuda/bin/`, on the invoking user's PATH but not on sudo's
      restricted `secure_path` — `sudo ncu ...` failed with "command not
      found" even though plain `ncu` resolved fine. Fixed by resolving
      absolute paths via `shutil.which()` once and using those everywhere
      instead of relying on sudo's own PATH resolution (commit `aca2434`).
- [x] **Found and fixed: ncu's CSV is long-format, not wide.** Wrongly
      assumed one row per kernel launch with each metric as a column. It's
      actually one row per **(kernel launch, single metric)** — "Metric
      Name"/"Metric Value" are themselves columns. This is what produced
      the nonsensical first real output ("606/2361 launches, 1 column
      each" — actually counting `==PROF==` status-line noise, a separate,
      also-fixed issue). `_parse_launches()` now groups by launch ID and
      pivots each launch's metrics into a proper dict (commit `845023c`).
- [x] **Confirmed real kernel breakdown** at seq_len=2048, batch=2, heads=8,
      head_dim=64, fp16: naive decomposes into two GEMM kernels
      (`turing_fp16_s1688gemm_fp16_128x128...` and `...256x128...` — cuBLAS
      picks a different algorithm for QKᵀ vs P@V, different tile shapes),
      one softmax kernel, one scaling multiply, and copy/cast kernels
      (matches `reference.py`'s fp32-softmax-then-cast-back). Triton's
      `_fwd_kernel` confirmed captured. A `distribution_elementwise...`
      kernel (torch.randn() generating test Q/K/V, not part of attention)
      and a `FillFunctor` kernel (likely the autotuner's internal
      cache-flush buffer between candidate-config timing trials) identified
      as noise and excluded from `_parse_launches()`'s output.
- [ ] **Not yet done — the actual next step**: the parser fix
      (`_parse_launches`, commit `845023c`) has been pushed but **never
      run** — its output (the list of available Metric Names per launch,
      needed to confirm the real name for HBM bytes/throughput on this ncu
      version) hasn't been seen yet. After that: pick the right launch(es)
      per kernel (Triton's autotuning means several launches share the name
      `_fwd_kernel` but have different grid/block sizes from different
      configs — need the one matching the actual winning config, likely the
      last one), extract `dram__bytes.sum`-equivalent + achieved throughput,
      compute arithmetic intensity via `roofline.arithmetic_intensity()`,
      and call `roofline.plot_roofline()` — none of this is wired up yet.
- [x] **Bytes-moved aggregation bug, caught before trusting the result.**
      First real computation showed Triton moving ~4.05GB vs naive's
      ~2.11GB — the opposite of the entire point of the kernel. Root cause
      was in the analysis code, not the kernel: naive's 6 representative
      launches are DIFFERENT sequential kernels forming one pipeline, so
      summing them is correct ("bytes for one call"). Triton's steady-state
      cluster is the SAME kernel launched 4 independent times — summing
      counted one call's bytes four times over. Fixed to average Triton's
      cluster instead of summing it (naive still sums, correctly).
- [x] **Corrected result: Triton moves ~1.01GB vs naive's ~2.10GB** — almost
      exactly half, confirming the O(N) vs O(N²) memory thesis with real
      measured data. Arithmetic intensity: naive 8.19 FLOPs/byte, Triton
      16.96 FLOPs/byte (~2x, consistent with ~half the bytes at the same
      FLOP count). Both sit far below the T4's roofline ridge point (~217
      FLOPs/byte for fp16) — confirms attention is memory-bound on this
      hardware regardless of implementation. See
      `results/figures/roofline.png`.
- [x] **Warp-stall reasons + achieved occupancy — the actual "why".** Ran
      `--set full` (a much heavier collection — 105 kernel launches × 31
      replay passes each) and read the text report. Found:
  - **Theoretical Occupancy capped at 25%** for the steady-state `_fwd_kernel`
    launches — not a failure to reach potential, the kernel is *at* its
    ceiling (achieved 24.7%, right at the theoretical cap). Most likely
    cause: register pressure from holding the online-softmax accumulators
    (running max/sum/output) largely in registers, limiting how many
    concurrent thread-blocks fit per SM.
  - This is the direct cause of the "low DRAM% and low Compute% at the same
    time" pattern found earlier — not two separate problems, one problem
    (insufficient occupancy) with two symptoms: not enough concurrent warps
    to hide memory latency *or* keep compute pipelines fed.
  - Dominant stall reason: ~53% of stall cycles are warps waiting on the MIO
    queue (shared-memory/special-instruction pipeline) being full. Nsight's
    own analysis estimates up to a **50.95% speedup** if this specific
    stall were eliminated — a quantified number, not a guess.
  - **Full causal chain**: small tile config → high per-thread register
    usage → few concurrent thread-blocks per SM → 25% occupancy ceiling →
    insufficient parallelism to hide memory or compute latency → both
    utilization numbers stay low → despite moving half the bytes naive
    does, wall-clock time still loses. Plain-English walkthrough + kitchen
    analogy for this whole chain: `concepts.md`, "Warp stalls / hiding
    latency."
  - **Confirmed via `ncu`'s own Occupancy section** (grepped directly from
    `results/traces/triton_full_report.txt`, not inferred): the steady-state
    launch (213 registers/thread, 16.38KB dynamic shared memory/block) hits
    `Block Limit Registers = 2` **and** `Block Limit Shared Mem = 2` —
    tied, both independently capping at 2 blocks/SM. Verified the register
    math directly: 65,536 registers/SM ÷ (213 × 128 threads/block) ≈ 2.4,
    floors to 2 — matches exactly. Registers are the harder of the two
    (would still cap at 2 blocks even if shared memory weren't a factor),
    which is what motivated the experiment below.
  - Caveat: this specific `--set full` run's autotuner picked a *different*
    near-tied config (`BLOCK_N=32, num_stages=2`) than the two lighter
    `--set roofline` runs did (`BLOCK_N=64, num_stages=3`) — both share
    `BLOCK_M=32`/`num_warps=4` so are indistinguishable by grid/block size
    alone. Likely explanation: heavier profiling overhead perturbed the
    autotuner's own timing-based decision between two very close
    candidates. The occupancy/stall story should still be representative
    of the small-tile config family, but isn't a perfect match to the
    exact config behind the roofline numbers above.
- [x] **Experiment: `maxnreg`-capped configs to test the register-limited
      finding directly — negative result.** Added 3 `maxnreg=128` config
      candidates (commit `1ce23a3`) to let autotuning empirically test
      whether capping registers (trading per-thread scratch space for more
      concurrent blocks) improves speed. Re-ran the full seq_len sweep:
      **no meaningful change anywhere** — seq_len=2048 (the profiled shape)
      went from 12.487-12.498ms to 12.528ms, well within existing run-to-run
      noise, and seq_len=8192 got slightly *worse* (205ms vs ~197ms).
      Autotuning almost certainly tried and rejected the new configs.
  - **Why it likely didn't help**: occupancy and the dominant stall reason
    found earlier (MIO pipeline pressure, ~53% of stall cycles — see above)
    are *different* bottlenecks that produced a similar-looking symptom.
    MIO pressure is about contention on the shared-memory-access pipeline
    itself, not how many warps are available to use it — more resident
    warps competing for that same limited pipeline can leave things flat or
    even slightly worse (consistent with the 8192 regression), not better.
  - **Refined conclusion**: the real fix is restructuring the kernel's
    inner loop to do fewer, wider memory loads (Nsight's own suggestion),
    not a tuning-level change — genuine kernel engineering, correctly left
    out of scope. Configs left in `_CONFIGS` as a documented, honest
    negative result rather than reverted.

Raw logs from each run: `results/logs/phase3_profile_run1.md`,
`phase3_profile_run2.md`, `phase_3_profile_full_run1.md`.

### Phase 4 — Extension

- [ ] Not started. Not yet decided: KV-cache (4a) vs quantization (4b).

### Phase 5 — Writeup

- [ ] Not started.

### Infra notes (for reference)

- Local dev machine: macOS (Apple Silicon), no CUDA — all GPU work runs on a
  rented box, never locally.
- GPU box: GCP Compute Engine, project `project-9d8f69e0-e809-4fab-b37`,
  instance `instance-20260718-143824`, zone `europe-west2-b`, **Spot**
  provisioning, Tesla T4, Deep Learning VM image (Ubuntu 22.04 + CUDA 12.9).
- Repo: `github.com/akashg71/gpu-attention` (public).
- Connect: `gcloud compute ssh akashg7171_com@instance-20260718-143824` — this
  exact username, which differs from the local Mac's gcloud-derived username
  (`akashgupta`) and owns the actual clone/venv on the box.
- Cost control: `gcloud compute instances stop/start instance-20260718-143824`
  between sessions — Spot billing is per-second while running only; the disk
  (and everything on it) persists regardless of running state.

## 0. TL;DR — what I want from you in THIS first session
The folder is empty. Do this and stop:
1. Confirm the GPU environment (print GPU name, CUDA version, torch version, Triton version).
2. Initialise a git repo and create the structure in Section 3.
3. Set up `requirements.txt` and install deps.
4. Get **one** fused-attention Triton kernel running (adapt from Triton's official fused-attention tutorial) and **validate it against a plain PyTorch reference** with `torch.allclose` on a single shape — this is the smoke test / gate.
5. Stub the benchmark harness (naive vs `F.scaled_dot_product_attention` vs Triton) so it runs on one shape.
6. Write `README.md` and `RUNBOOK.md`, and tell me exactly what to run next.

Do **not** attempt the full project now. Correct kernel + smoke test + harness skeleton is the goal.

**Two standing notes:**
- I'm a senior backend engineer, strong at systems, but **new to CUDA/Triton/GPU profiling** — comment generously and keep the README operable.
- This brief was written from early-2026 knowledge. **Triton's API changes a lot between versions.** Before writing kernel code, check the *installed* Triton version and its own fused-attention tutorial for the current kernel API (block pointers / `make_block_ptr` / newer constructs). Verify `torch.nn.functional.scaled_dot_product_attention` backend selection too. Don't trust my exact API calls blindly — verify them.

---

## 1. Project context & goal
This is a portfolio project to demonstrate **GPU inference-systems** ability for ML-infra / performance roles. The thesis: transformer inference is largely **memory-bound**, and the craft is exploiting the memory hierarchy (keep work in on-chip SRAM, minimise HBM traffic). The project implements a **FlashAttention-style fused attention kernel in Triton**, benchmarks it rigorously against a naive baseline and PyTorch's optimised SDPA, and — the part that matters most — **profiles it to explain the speedup in hardware terms** (measured HBM bytes moved, memory throughput, roofline position).

The bar to clear is "this person can write a GPU kernel, benchmark it honestly, and reason about inference performance at the hardware level." The headline evidence is not "I'm faster than everyone" — I almost certainly won't beat PyTorch's SDPA, and that's fine and expected — it's **"X faster than naive, here's the profiler trace showing HBM traffic dropped from N to M, and here's where each kernel sits on the roofline."** You can't fake an ncu trace.

---

## 2. Overall plan (phases) — so you scaffold for all of it
Execute Phase 0 now; scaffold for the rest.
- **Phase 0 — Env + smoke:** GPU env works; adapt the Triton fused-attention tutorial (forward only); validate against a PyTorch reference on one shape.
- **Phase 1 — Correct kernel:** forward-only fused attention Triton kernel, validated against the reference across shapes (varying seq len, head dim, batch, causal + non-causal), with sensible fp16/bf16 tolerances.
- **Phase 2 — Benchmark:** naive PyTorch attention vs `F.scaled_dot_product_attention` vs the Triton kernel; sweep sequence length; report latency, throughput (TFLOP/s), and peak memory; plots.
- **Phase 3 — Profile (the differentiator):** Nsight Compute on the kernels — HBM bytes moved, memory throughput, achieved occupancy, warp-stall reasons; a **roofline plot** placing each kernel; tie the speedup to reduced HBM traffic (O(N) vs O(N²) intermediate).
- **Phase 4 — Extension (pick ONE):**
  - (a) **KV-cache decode:** a simple autoregressive decode loop with a KV cache; measure tokens/sec and memory vs context length; add an **int8 KV-cache** and measure the memory/bandwidth win and any quality delta; **or**
  - (b) **Quantization study:** weight-only int8/int4 on a small model; measure perplexity vs latency vs memory.
- **Phase 5 — Writeup:** clean repo + a blog post: the roofline, benchmark plots, ncu traces, the memory-bound explanation, and an honest "didn't beat SDPA, here's why."

---

## 3. Repository structure to create
```
gpu-attention/
├── README.md                  # overview + setup + how to run each phase
├── RUNBOOK.md                 # short, operable, "how I run this on a GPU box"
├── requirements.txt
├── .gitignore                 # venv, __pycache__, *.ncu-rep, *.nsys-rep, results caches
├── src/gpu_attention/
│   ├── __init__.py
│   ├── env.py                  # print GPU name, CUDA, torch, triton versions; get device
│   ├── reference.py            # plain PyTorch attention (correctness oracle) + naive baseline
│   ├── triton_attention.py     # the fused-attention Triton kernel (forward only)
│   ├── correctness.py          # allclose checks across shapes, fp16/bf16 tolerances
│   ├── benchmark.py            # time naive vs SDPA vs Triton; latency, TFLOP/s, peak mem
│   ├── roofline.py             # GPU peak FLOP/s + peak HBM BW; arithmetic intensity; plot
│   ├── kvcache.py              # (Phase 4a) decode loop + KV cache + int8 KV variant
│   └── quant.py                # (Phase 4b) int8/int4 weight-only + perplexity/latency/mem
├── scripts/
│   ├── 00_smoke_test.py        # Phase 0 gate: run kernel + allclose vs reference
│   ├── 01_correctness.py
│   ├── 02_benchmark.py
│   ├── 03_profile.py           # emits ncu/nsys commands + parses results
│   └── 04_extension.py
├── notebooks/
│   └── results_figures.ipynb
└── results/
    ├── figures/                # latency_vs_seqlen.png, roofline.png, hbm_bytes.png
    ├── traces/                 # ncu/nsys outputs (gitignored)
    └── benchmarks.md           # the numbers table
```

---

## 4. Technical specification

### 4.1 The kernel
- Implement **forward-only** FlashAttention-style fused attention in Triton. **Start from Triton's official fused-attention tutorial** (commonly `06-fused-attention`) and adapt — don't reinvent from zero, but understand every part and comment it.
- Core idea to preserve: **tile Q/K/V into blocks and use an online (streaming) softmax** (running max `m`, running denominator `l`, rescale on each block) so the full N×N score matrix is **never materialised in HBM** — O(N) memory instead of O(N²). Support causal and non-causal.
- Skip the backward pass entirely (this is inference-focused). Don't chase every FlashAttention variant.

### 4.2 Correctness (do this before any benchmarking)
- `reference.py`: a plain, obviously-correct PyTorch attention (`softmax(QKᵀ/√d + mask) @ V`).
- Validate the Triton kernel against it with `torch.allclose` using tolerances appropriate to the dtype (fp16/bf16 need looser `atol/rtol` than fp32). Test across seq lengths, head dims, batch sizes, and both causal/non-causal. **A kernel that isn't verified correct is worthless — gate benchmarking on this.**

### 4.3 Benchmarking (rigor is the point)
- Compare three implementations: **naive** (manual PyTorch), **SDPA** (`torch.nn.functional.scaled_dot_product_attention`), **Triton**.
- Timing: warm up, then time N iterations with **CUDA events** (`torch.cuda.Event`), report the **median**; call `torch.cuda.synchronize()` correctly. Lock GPU clocks if the box allows (reduces variance).
- Metrics per config: **latency (ms)**, **throughput (TFLOP/s)** using the attention FLOP count (≈ `4·B·H·N²·d`; causal ≈ halves it — state your formula), and **peak memory** (`torch.cuda.max_memory_allocated`).
- Sweep sequence length (e.g. 512 → 8192) at fixed batch/heads/head_dim; the naive baseline will OOM or fall off a cliff at large N — that contrast is a result, capture it. Triton's `triton.testing.perf_report` / `do_bench` are handy.

### 4.4 Profiling (Phase 3 — the differentiator)
- Use **Nsight Compute (`ncu`)** on the kernels and pull: **HBM/DRAM bytes moved** (e.g. `dram__bytes.sum`), **memory throughput** (`gpu__dram_throughput` / `sm__throughput`), **achieved occupancy**, and **warp stall reasons**. Use **Nsight Systems (`nsys`)** for the timeline. `scripts/03_profile.py` should emit the exact `ncu`/`nsys` command lines and parse the output.
- The story to land: the fused kernel moves far fewer HBM bytes than naive (no N×N intermediate), and inference attention is **memory-bound**, so bytes-moved ≈ the thing that matters. Put measured bytes-moved next to the roofline.
- **Permissions caveat:** `ncu` frequently needs elevated privileges / GPU counter access that hosted notebooks (Colab/Kaggle) **block**. Expect to run the profiling on a **rented box with sudo**. Note this clearly in the README. PyTorch profiler (`torch.profiler` with CUDA activities) is a lighter fallback for a first look.

### 4.5 Roofline (`roofline.py`)
- For the target GPU, get peak FLOP/s (for the dtype) and peak HBM bandwidth. Compute each kernel's **arithmetic intensity** (FLOPs / bytes moved) and plot it against the roofline; show attention sitting in the **memory-bound** region. This is the visual that ties theory to your measurements.

### 4.6 Phase 4 extension — implement ONE
- **(a) KV-cache decode:** a minimal decode loop over a small model (GPT-2 or a small Llama/Qwen-class) with a KV cache; measure **tokens/sec** and **memory** vs context length; show decode is **memory-bandwidth-bound** (re-reads the whole cache each step). Then add an **int8 KV-cache**, measure the memory/bandwidth improvement and any output-quality delta.
- **(b) Quantization study:** weight-only **int8 / int4** on a small model; measure **perplexity** (on a held-out slice) vs **latency** vs **memory**, and plot the tradeoff.

### 4.7 Honesty framing (bake into the writeup)
- PyTorch's SDPA is backed by FlashAttention-2/3 and is brutally optimised; **you probably won't beat it, and you shouldn't claim to.** The value is the analysis: your kernel vs naive vs SOTA, with the profiler explaining the gap. Same principle as my SAE project — "here's what I built, measured, and understood" reads as competence; overclaiming gets punctured in interviews.

---

## 5. Environment setup
1. On the GPU box: `python3.11 -m venv .venv && source .venv/bin/activate`.
2. `requirements.txt` (let pip resolve, then pin with `pip freeze`):
   ```
   torch            # CUDA build; on Linux+CUDA this typically brings a compatible triton
   triton           # only if not bundled with the torch build
   transformers
   datasets
   einops
   numpy
   pandas
   matplotlib
   tqdm
   # optional
   wandb
   ```
3. Nsight tools for Phase 3: `nsight-compute` (`ncu`) and `nsight-systems` (`nsys`) — often preinstalled on cloud GPU images; may need `apt install` and sudo. Not needed for Phase 0.
4. Python **3.10/3.11**. After install, print and record `torch.__version__`, `triton.__version__`, `torch.version.cuda`, and `torch.cuda.get_device_name()`.

---

## 6. Definition of done for THIS session
- GPU env confirmed (versions + device name printed).
- Git repo + full structure from Section 3; `.gitignore` in place.
- Deps installed on the GPU box.
- `scripts/00_smoke_test.py` runs and: builds the Triton fused-attention kernel, runs it on one shape, and **passes `torch.allclose` against the PyTorch reference** (print PASS + max abs error).
- `benchmark.py` runs on a single shape and prints latency for naive / SDPA / Triton.
- `README.md` + `RUNBOOK.md` written; end by telling me what to run for Phase 1.
- If the installed Triton API differs from this doc, **fix it against the installed version** and note what changed in the README.

---

## 7. Things I (the human) need to get or do
Flag anything I'm missing.
1. **An NVIDIA GPU.** Colab or Kaggle (free T4) for Phases 0–2; a **rented A100 / L4 / H100** (RunPod / Lambda / Vast) for Phase 3 profiling (ncu needs privileges hosted notebooks lack) and for any cross-generation comparison. **My Mac cannot run any of this.**
2. **Nsight Compute + Nsight Systems** available on the profiling box (may need sudo).
3. For Phase 4: a **small model** (GPT-2, or a small Llama/Qwen/Gemma-class) via Hugging Face, plus a **HF token** (`huggingface-cli login`).
4. **Disk:** ~10 GB (env + model + traces).
5. Optional: **Weights & Biases** for logging.

---

## 8. Notes & constraints for you (Claude Code)
- **Forward-only, inference-focused.** No backward pass; don't chase every attention variant.
- **Correctness before speed.** Validate against the PyTorch reference before benchmarking or optimising.
- **Benchmark rigorously:** warmup, CUDA-event timing, median of N, correct `synchronize()`, report latency + TFLOP/s + peak memory, lock clocks if possible. My systems background means sloppy benchmarking is the thing I'll be judged on — get it right.
- **Verify APIs against installed versions** (Triton kernel API, SDPA backend). My identifiers may be stale.
- **Arch-aware:** read the GPU's peak FLOP/s and HBM BW at runtime; don't hardcode a specific card.
- **Don't claim to beat SDPA.** The analysis is the deliverable.
- Keep it modular, seeded where relevant, reproducible. Small and clean beats big and fancy.
- Comment well and write a real README/RUNBOOK — I'm new to this tooling.
- Add a sensible `.gitignore` (venv, caches, `*.ncu-rep`, `*.nsys-rep`).
- End the session by telling me exactly what to run for Phase 1.

---

## 9. Reference — sketch to pin the approach (you'll flesh this out)
```python
# reference.py — the correctness oracle (obviously-correct, slow)
import torch, torch.nn.functional as F

def attention_reference(q, k, v, causal=False):
    # q,k,v: (batch, heads, seq, head_dim)
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)          # (B,H,N,N) — materialised on purpose
    if causal:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```
```
# FlashAttention forward, conceptually (what the Triton kernel implements):
# for each query block Q_i:
#   m_i = -inf; l_i = 0; acc = 0
#   for each key/value block (K_j, V_j):
#       S_ij   = (Q_i @ K_j^T) / sqrt(d)            # small tile, lives in SRAM
#       m_new  = max(m_i, rowmax(S_ij))
#       P_ij   = exp(S_ij - m_new)
#       l_i    = exp(m_i - m_new) * l_i + rowsum(P_ij)
#       acc    = exp(m_i - m_new) * acc + P_ij @ V_j
#       m_i    = m_new
#   O_i = acc / l_i                                  # never materialised the full N×N matrix in HBM
```
Validate the Triton output against `attention_reference` before doing anything else.

---

*End of brief. Confirm the GPU env, get the kernel correct, run the smoke test, and tell me what to run next.*
