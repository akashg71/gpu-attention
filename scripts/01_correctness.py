"""
Phase 1: correctness sweep across seq_len, head_dim, batch, causal/non-causal,
fp16/bf16. Not implemented yet — Phase 0 (00_smoke_test.py) must pass first.
Will call gpu_attention.correctness.check_one() over a grid of shapes, same
function the smoke test uses, so tolerance logic isn't duplicated.
"""
raise NotImplementedError("Phase 1 — run after 00_smoke_test.py passes")
