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
    error: str = ""


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


# Sweep grid for Phase 1 (instructions.md Section 4.2: seq len, head dim,
# batch, causal on/off, fp16/bf16). Heads is fixed, not swept — the brief
# doesn't list it as an axis and it doesn't interact with tile/tolerance
# logic the way these five do.
SEQ_LENS = (128, 512, 1024, 2048, 4096)
HEAD_DIMS = (64, 128)
BATCHES = (1, 4)
CAUSALS = (False, True)
DTYPES = (torch.float16, torch.bfloat16)
SWEEP_HEADS = 8


def run_sweep(device: torch.device) -> list[CorrectnessResult]:
    """Runs check_one() over the full grid above. Each combination is
    isolated in its own try/except: a single shape/dtype combo raising
    (e.g. bf16 matmul on a GPU without bf16 tensor-core support) is recorded
    as a failure with the error message, not allowed to crash the whole
    sweep and lose every result gathered so far.
    """
    results = []
    for seq_len in SEQ_LENS:
        for head_dim in HEAD_DIMS:
            for batch in BATCHES:
                for causal in CAUSALS:
                    for dtype in DTYPES:
                        try:
                            result = check_one(
                                batch=batch,
                                heads=SWEEP_HEADS,
                                seq_len=seq_len,
                                head_dim=head_dim,
                                causal=causal,
                                dtype=dtype,
                                device=device,
                            )
                        except Exception as e:
                            result = CorrectnessResult(
                                passed=False,
                                max_abs_err=float("nan"),
                                max_rel_err=float("nan"),
                                shape=(batch, SWEEP_HEADS, seq_len, head_dim),
                                dtype=dtype,
                                causal=causal,
                                error=str(e).splitlines()[0][:200],
                            )
                        results.append(result)
    return results
