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
