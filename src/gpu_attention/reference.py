"""
The correctness oracle: plain, obviously-correct PyTorch attention.

This is deliberately the "naive" implementation too — it materialises the full
(B, H, N, N) score matrix in HBM, which is exactly the O(N^2) memory traffic
the Triton kernel is built to avoid. Everything else in this repo is checked
against this function, so it must stay simple enough to trust by inspection.
"""
import torch
import torch.nn.functional as F


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """softmax(QK^T / sqrt(d) + causal_mask) @ V.

    q, k, v: (batch, heads, seq, head_dim), same dtype/device.
    Returns: (batch, heads, seq, head_dim).
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # (B, H, N, N) — materialised on purpose

    if causal:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax in fp32 regardless of input dtype: fp16/bf16 softmax over long rows
    # is a real source of numerical drift that would otherwise show up as a
    # correctness failure that's actually just a precision artifact.
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return attn @ v


def attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Alias kept distinct from attention_reference for benchmark.py's three-way
    comparison (naive / SDPA / Triton), even though today they're the same
    computation. If the naive baseline ever needs to diverge (e.g. to match a
    specific "what a beginner would write" baseline), split the implementations
    here instead of at every call site.
    """
    return attention_reference(q, k, v, causal=causal)


def attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch's built-in fused attention (FlashAttention-2/3 or memory-efficient
    backend, chosen automatically). This is the bar we're not expecting to beat.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
