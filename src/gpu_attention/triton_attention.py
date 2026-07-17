"""
Fused (FlashAttention-style) attention forward pass, in Triton.

*** STATUS: UNVERIFIED. This file has never been run. ***
It was written on a Mac with no CUDA GPU and no Triton installed, so it could
not be compiled, executed, or checked against the reference. Triton's kernel
API also changes between versions (this is called out explicitly in
instructions.md), and there was no installed version here to check against.

Before trusting ANYTHING below:
1. On the GPU box, run `python -c "import triton; print(triton.__version__)"`
   and open that version's own tutorial (`06-fused-attention` in
   github.com/triton-lang/triton, tag matched to your version).
2. Run scripts/00_smoke_test.py. Expect this to fail or need fixes on first
   try — that's the actual Phase 0 work, not a sign anything is broken.
3. Common breakage points to check against the real tutorial: `tl.dot`
   signature/accumulator dtype, whether `tl.make_block_ptr` is expected instead
   of raw pointer arithmetic (used below for stability across versions), and
   `libdevice`/`tl.math` exp namespace (`tl.exp` vs `tl.math.exp` vs
   `triton.language.extra.libdevice.exp`).

Deliberately uses raw pointer + stride arithmetic instead of `make_block_ptr`,
since that calling convention has been more stable across Triton releases and
is easier to read line-by-line if you're new to Triton (each `tl.load` is an
explicit address computation, nothing implicit).

Algorithm (see instructions.md Section 9 for the reference pseudocode this
mirrors): tile Q into row-blocks of size BLOCK_M, tile K/V into column-blocks
of size BLOCK_N. For each Q row-block, stream over K/V column-blocks and
maintain a running (online) softmax — running max `m_i`, running denominator
`l_i`, running weighted output `acc` — so the full (N, N) score matrix is
never materialised. This is what keeps HBM traffic at O(N) instead of O(N^2);
that reduction is the entire point of Phase 3's profiling story.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, N_CTX,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Grid: axis 0 tiles the query sequence dim, axis 1 tiles (batch * heads).
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    # Base pointers for this (batch, head) slice.
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query row indices this block owns
    offs_d = tl.arange(0, BLOCK_D)                        # head_dim indices (BLOCK_D == head_dim, no tiling)

    # Load the Q block once — it's reused against every K/V block below, which
    # is the whole SRAM-reuse point (Q block stays on-chip for the full inner loop).
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax running state.
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)  # running row max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                       # running row denominator
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)               # running weighted-V accumulator

    # Causal: a query block never needs to look past its own last row, so we
    # can stop the K/V loop early instead of masking-and-wasting-work on
    # fully-future blocks.
    end_n = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S_ij = scale * Q_i @ K_j^T — small (BLOCK_M, BLOCK_N) tile, lives in SRAM/registers.
        qk = tl.dot(q, tl.trans(k)) * scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # Mask out-of-range K columns (sequence not a multiple of BLOCK_N).
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))

        # --- online softmax update ---
        m_ij = tl.max(qk, axis=1)                    # row max of this tile
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)                   # rescale factor for old accumulator
        p = tl.exp(qk - m_new[:, None])                # unnormalised probs for this tile

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Final normalisation — divide by the running denominator once, at the end,
    # instead of after every block (that's the "online" part of online softmax).
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Fused attention forward. q, k, v: (batch, heads, seq, head_dim), same
    dtype/device, head_dim a power of 2 (required so BLOCK_D covers it exactly
    with no tiling — this kernel does not tile the head_dim, only Q's seq and
    K/V's seq, which matches the tutorial's default and is fine for the
    head_dim=64/128 shapes used here).
    """
    assert q.shape == k.shape == v.shape, "q, k, v must have matching shapes"
    batch, heads, seq_len, head_dim = q.shape
    assert head_dim in (16, 32, 64, 128, 256), (
        f"head_dim={head_dim}: this kernel loads the whole head_dim per block "
        "(BLOCK_D=head_dim) and expects a power of 2 tl.dot can use efficiently."
    )

    out = torch.empty_like(q)
    scale = 1.0 / (head_dim ** 0.5)

    grid = (triton.cdiv(seq_len, block_m), batch * heads)

    _fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        heads, seq_len,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        CAUSAL=causal,
    )
    return out
