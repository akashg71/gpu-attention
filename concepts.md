# Concepts reference

Running notes on what this project actually requires understanding, split into
the ML/kernel theory (needed before writing/reading the kernel) and the
cloud/infra mechanics (learned by actually deploying Phase 0). Add to this as
the project progresses — it's meant to be a personal reference, not polished
prose.

---

## Part 1 — ML/GPU kernel concepts

### Memory hierarchy (the whole point of the project)
- **HBM** (High Bandwidth Memory) — the GPU's main memory. What
  `torch.randn(...).cuda()` allocates into. Large (tens of GB), but slow
  relative to on-chip memory.
- **SRAM** — on-chip memory (registers + "shared memory" in CUDA terms). Tiny
  (tens of KB per SM) but ~100x faster than HBM. FlashAttention's entire trick
  is keeping Q/K/V tiles here instead of writing the full N×N score matrix out
  to HBM.
- **Memory-bound vs compute-bound** — whether a kernel's runtime is dominated
  by moving bytes or by doing arithmetic.
- **Arithmetic intensity** = FLOPs ÷ bytes moved. The number that determines
  which side of the above a kernel is on.
- **Roofline model** — a plot of achievable FLOP/s vs arithmetic intensity,
  with a memory-bound "ramp" and a compute-bound "ceiling." Where a kernel
  lands on it is Phase 3's headline chart — the visual proof that attention is
  memory-bound.

### GPU execution model
- **SM** (Streaming Multiprocessor) — a GPU's "core." A T4 has 40 of them
  (confirmed via `env.py` on the actual box); bigger GPUs (A100/H100) have
  more.
- **Thread / warp / block** — a **warp** is 32 threads executing in lockstep;
  a **block** (CTA) is a group of warps scheduled onto one SM.
- **Occupancy** — how many warps are actually resident on an SM vs its max
  capacity. Low occupancy often means the SM sits idle waiting on memory.
- **Warp stalls** — the reason a warp isn't issuing an instruction on a given
  cycle (commonly: waiting on a memory load). `ncu` reports these by category
  in Phase 3.

### Triton's programming model
- Triton is a Python-embedded DSL that JIT-compiles to PTX (GPU assembly).
  You write kernels in terms of **blocks/tiles** (`BLOCK_M`, `BLOCK_N` in
  `triton_attention.py`), not individual threads — Triton handles the
  thread/warp mapping for you, unlike raw CUDA C++.
- `tl.load` / `tl.store` with a **mask** — how a kernel reads/writes a tile
  while safely ignoring out-of-bounds elements (e.g. when seq_len isn't a
  multiple of the block size).
- `@triton.jit`, `constexpr` — compile-time constants (block sizes) baked
  into the compiled kernel, as opposed to runtime values (tensor pointers,
  strides).
- Triton needs a working C toolchain **and OS-level Python dev headers**
  (`Python.h`) at first run, because it JIT-compiles a small internal C
  helper (`cuda_utils.c`) — this is independent of anything inside a Python
  venv. Missing `python3-dev` is what actually broke the first real run on
  the GPU box, not a bug in the kernel.

### The core algorithmic trick: online (streaming) softmax
The reason FlashAttention-style kernels avoid ever materializing the full
N×N attention matrix: instead of computing an entire row before normalizing,
keep a running max (`m`) and running sum (`l`) per row, and *rescale* the
partial output as each new key/value block is streamed in. This is what
allows attention's memory traffic to be O(N) instead of O(N²). See
`instructions.md` Section 9 for the reference pseudocode — worth tracing
through by hand on a tiny example (N=4, block size=2) until the rescaling
step is intuitive.

### Precision
- **fp16 / bf16 / fp32** — bf16 keeps fp32's exponent range but has less
  mantissa precision than fp16; both run faster than fp32 on **tensor
  cores** (specialized matmul hardware), which is why inference kernels
  default to them. This is also why correctness checks need looser
  atol/rtol tolerances in fp16/bf16 than fp32 (see `TOLERANCES` in
  `correctness.py`).

### Benchmarking mechanics
- **CUDA events** vs Python `time.time()` — GPU work is asynchronous, so
  wall-clock timing without `torch.cuda.synchronize()` measures "how fast
  Python could launch kernels," not "how fast the GPU ran them."
- **Warmup** — first calls pay one-time costs (Triton JIT compilation, cuDNN
  algorithm search) that aren't representative of steady-state performance.

### Profiling tools (Phase 3)
- **Nsight Compute (`ncu`)** — per-kernel deep profiler: bytes moved,
  memory throughput, occupancy, warp-stall reasons.
- **Nsight Systems (`nsys`)** — timeline view across the whole program, good
  for seeing gaps between kernel launches.

---

## Part 2 — Cloud/GPU-infra concepts (learned deploying Phase 0)

### Quota vs stockout — two different failure modes
- **Quota** = are you *allowed* to use N GPUs of a given type in a given
  region. A permissions/allocation question, resolved in the console or via
  request.
- **Stockout** = physical GPUs of that type are all rented out in that zone
  *right now*. Unrelated to quota — you can be fully within quota and still
  get a stockout error. Transient; often resolved by trying a different zone
  in the same region, or waiting.

### Quota is split into separate buckets
Even for the same GPU type + region, GCP tracks these independently:
- **Committed** — reserved-capacity purchases (1-3 year contracts). Not
  relevant for on-demand usage.
- **On-demand** (plain "NVIDIA T4 GPUs") — the one that gates a normal
  `Create Instance` call.
- **Preemptible / Spot** — separate quota again, for interruptible instances.

### Spot / preemptible instances
Cheaper and often easier to get capacity for than standard on-demand — but
Google can reclaim the instance at any time it needs the capacity back. Fine
for short interactive work (a smoke test, a quick benchmark run); riskier for
anything long-running you don't want interrupted (e.g. a long Phase 3
profiling session). Stopping/starting a Spot instance yourself is safe for
your data (disk persists), but restarting can re-trigger a stockout since
Spot capacity isn't guaranteed.

### Trial vs activated billing accounts
- **Free trial**: spend is hard-capped at the free credit — cannot be
  charged past it even accidentally. GPUs are usually quota-locked to 0 in
  this state specifically to prevent abuse (crypto mining etc.).
- **Activated (paid) account**: credit still applies and is used first, but
  once/if it runs out, the card on file *can* be charged for real. Worth
  setting a billing budget/alert at the same time you activate.
- Account/billing-account **age and history** measurably affects whether a
  GPU quota request gets auto-approved vs denied for "insufficient account
  history" — an old, previously-used billing account can already have
  default quota a brand-new one gets denied for.

### Deep Learning VM images
GCP (and equivalents on AWS/Azure) offer marketplace boot-disk images with
NVIDIA driver + CUDA + Python pre-installed and version-matched, so you skip
manually installing GPU drivers on a fresh Linux box.

### SSH username provisioning differs by access method
The GCP Console's browser-based SSH button and the `gcloud compute ssh` CLI
can provision **different Linux usernames on the same VM** — the console
derives one from your Google account email (e.g. `akashg7171_com`), while
`gcloud compute ssh` defaults to deriving one from your local machine's OS
username (e.g. `akashgupta`). These are two separate home directories with
no access to each other by default. Fix: explicitly specify the desired
remote username, e.g. `gcloud compute ssh akashg7171_com@<instance-name>`.

### Tooling without Homebrew/sudo
Several tools were installed on the local Mac without Homebrew or root, all
via user-space installers:
- **`gh`** (GitHub CLI) — direct binary download from GitHub releases.
- **`uv`** — a standalone Python version/package manager; used to install a
  newer Python (3.11) in user space to fix `gcloud`'s dependency on a modern
  Python, without touching the system Python or needing `sudo`.
- **`gcloud`** — Google Cloud SDK, once pointed at the `uv`-managed Python
  via the `CLOUDSDK_PYTHON` environment variable.
