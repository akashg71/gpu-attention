"""
Correctness gate: Triton kernel output vs attention_reference, via torch.allclose.

Phase 0 only needs check_one() to pass on a single shape (that's the smoke
test). Phase 1 sweeps shapes/dtypes/causal — scripts/01_correctness.py drives
that using the same check_one() so the tolerance logic lives in exactly one
place.
"""
from dataclasses import dataclass

import torch

from .reference import attention_reference
from .triton_attention import triton_attention

# fp16/bf16 accumulate error over the softmax + matmul chain in ways fp32
# doesn't; these tolerances are starting points from common FlashAttention
# correctness tests, not derived — tighten them once you have real numbers
# from Phase 1 and see how much slack is actually needed.
TOLERANCES = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=3e-2, rtol=3e-2),
    torch.float32: dict(atol=1e-4, rtol=1e-4),
}


@dataclass
class CorrectnessResult:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    shape: tuple
    dtype: torch.dtype
    causal: bool


def check_one(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> CorrectnessResult:
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    ref = attention_reference(q, k, v, causal=causal)
    out = triton_attention(q, k, v, causal=causal)

    diff = (out - ref).abs()
    max_abs_err = diff.max().item()
    max_rel_err = (diff / (ref.abs() + 1e-6)).max().item()

    tol = TOLERANCES[dtype]
    passed = torch.allclose(out, ref, atol=tol["atol"], rtol=tol["rtol"])

    return CorrectnessResult(
        passed=passed,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        shape=(batch, heads, seq_len, head_dim),
        dtype=dtype,
        causal=causal,
    )
