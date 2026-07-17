"""
Phase 4a stub (pick ONE of kvcache.py / quant.py — not both).

Plan: minimal autoregressive decode loop over a small HF model (GPT-2 or a
small Llama/Qwen-class) with an explicit KV cache; measure tokens/sec and
memory vs context length to show decode is memory-bandwidth-bound (each step
re-reads the whole growing cache). Then add an int8 KV-cache variant and
measure the memory/bandwidth win and any output-quality delta.

Not implemented — out of scope until Phases 0-3 are done.
"""
