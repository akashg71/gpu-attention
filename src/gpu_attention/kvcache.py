"""
Phase 4a: autoregressive decode with an explicit KV cache — tokens/sec and
memory vs context length (to show decode is memory-bandwidth-bound: each
step re-reads the whole growing cache), plus an int8 KV-cache variant
measuring the real memory/bandwidth win and any output-quality delta.

Model: GPT-2 (small, 124M) via HuggingFace `transformers` — no gated access
or HF token needed, tiny download, and its `past_key_values` cache handling
is about as stable an API as HF has. Deliberately not our own Triton
kernel here — the point is showing the memory-bound *decode* story
generalizes past a single hand-written kernel, using a real model's real
cache, not re-deriving Phases 0-3 inside a bigger model.

*** UNVERIFIED against a real transformers install. *** Written without a
GPU/transformers to check against — same situation the Triton kernel was
in during Phase 0. The main risk: `past_key_values`'s exact type (a legacy
tuple-of-tuples vs a newer `Cache` object) differs across transformers
versions. Handled via `_cache_to_tuples`/`_tuples_to_cache`, which use
`to_legacy_cache()`/`from_legacy_cache()` when available and fall back to
raw tuples otherwise — but this is a best-effort guess at the installed
version's API, not a confirmed match. Run scripts/04_extension.py and
expect to fix this against whatever `transformers.__version__` actually is
here, same "verify on real hardware" pattern as everything before it.
"""
import torch

MODEL_NAME = "gpt2"

# A real (short) passage rather than random token IDs — so generated text is
# actually inspectable for the quality-delta comparison, not just numbers.
SAMPLE_TEXT = (
    "The history of artificial intelligence began with early philosophers "
    "who considered the question of whether machines could truly think. "
    "In the decades since, the field has moved through cycles of intense "
    "optimism and quiet retrenchment, each time building on the lessons "
    "of what came before."
)


def load_model(device: torch.device, dtype: torch.dtype = torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def make_prompt(tokenizer, device: torch.device, target_len: int) -> torch.Tensor:
    """Tokenizes SAMPLE_TEXT and repeats/truncates it to exactly target_len
    tokens, so the context-length sweep can hit specific lengths with real
    (if repetitive) text rather than random garbage.
    """
    ids = tokenizer(SAMPLE_TEXT, return_tensors="pt").input_ids[0]
    reps = (target_len // ids.shape[0]) + 1
    ids = ids.repeat(reps)[:target_len]
    return ids.unsqueeze(0).to(device)


def quantize_int8(tensor: torch.Tensor) -> tuple:
    """Per-tensor symmetric int8 quantization. Returns (int8_tensor, scale)."""
    scale = tensor.abs().max().clamp(min=1e-8) / 127.0
    q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return q.to(dtype) * scale


def _cache_to_tuples(past_key_values):
    """Normalizes whatever transformers hands back (legacy tuple-of-tuples,
    or a modern Cache object) into a plain list of (key, value) tensors —
    so the int8 logic doesn't depend on which one this installed version
    uses.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        raw = past_key_values.to_legacy_cache()
    else:
        raw = list(past_key_values)
    # Confirmed via diagnostic on transformers 5.14.1: iterating a
    # DynamicCache directly yields (key, value, None) per layer, not a
    # clean 2-tuple -- to_legacy_cache() doesn't exist on this version at
    # all. Keep only the first two elements regardless of how many trail
    # after them, rather than assuming exactly 2.
    return [(layer[0], layer[1]) for layer in raw]


def _tuples_to_cache(tuples, reference):
    """Reverses _cache_to_tuples — rebuilds whatever type the model expects.
    transformers 5.x removed both to_legacy_cache() and from_legacy_cache()
    (confirmed via diagnostic), so the primary path builds a fresh
    DynamicCache using its own .update() API rather than a legacy
    constructor. from_legacy_cache is kept as a fallback for older
    transformers installs this project might run on later.
    """
    if reference is not None and hasattr(type(reference), "from_legacy_cache"):
        return type(reference).from_legacy_cache(tuple(tuples))
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(tuples):
        cache.update(k, v, layer_idx)
    return cache


def cache_bytes(past_key_values, dtype_bytes: int) -> int:
    """Total bytes the KV cache occupies given tensor shapes — lets us state
    the fp16-vs-int8 cache size directly (2 bytes/elem vs 1), independent of
    whatever else is contributing to peak-memory noise.
    """
    tuples = _cache_to_tuples(past_key_values)
    if not tuples:
        return 0
    total_elems = sum(k.numel() + v.numel() for k, v in tuples)
    return total_elems * dtype_bytes


@torch.no_grad()
def decode_fp16(model, input_ids: torch.Tensor, num_new_tokens: int):
    """Standard decode loop — cache managed entirely by transformers
    internally, never touched. This is the baseline.
    """
    generated = input_ids
    past = None
    per_token_ms = []

    for _ in range(num_new_tokens):
        model_input = generated if past is None else generated[:, -1:]

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = model(model_input, past_key_values=past, use_cache=True)

        end.record()
        torch.cuda.synchronize()
        per_token_ms.append(start.elapsed_time(end))

        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated, per_token_ms, past


@torch.no_grad()
def decode_int8_kv(model, input_ids: torch.Tensor, num_new_tokens: int):
    """Same decode loop, but the cache is genuinely stored in int8 BETWEEN
    steps — real int8 tensors held in memory (so peak-memory numbers
    reflect actual savings), dequantized to the model's working dtype just
    before each forward call, requantized right after. This measures the
    storage/bandwidth tradeoff honestly, including the dequantization
    compute cost — it does NOT implement an int8-aware attention kernel
    (attention itself still runs in fp16), so any speed win here is net of
    that dequant overhead, not a best-case number.
    """
    generated = input_ids
    past_raw = None   # last raw past_key_values from the model, only kept to know its type
    q_cache = None     # our int8 cache: list of (q_k, scale_k, q_v, scale_v) per layer
    per_token_ms = []
    working_dtype = next(model.parameters()).dtype

    for _ in range(num_new_tokens):
        model_input = generated if q_cache is None else generated[:, -1:]

        if q_cache is not None:
            deq_tuples = [
                (dequantize_int8(qk, sk, working_dtype), dequantize_int8(qv, sv, working_dtype))
                for (qk, sk, qv, sv) in q_cache
            ]
            past_in = _tuples_to_cache(deq_tuples, past_raw)
        else:
            past_in = None

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = model(model_input, past_key_values=past_in, use_cache=True)

        end.record()
        torch.cuda.synchronize()
        per_token_ms.append(start.elapsed_time(end))

        past_raw = outputs.past_key_values
        tuples = _cache_to_tuples(past_raw)
        q_cache = []
        for k, v in tuples:
            qk, sk = quantize_int8(k)
            qv, sv = quantize_int8(v)
            q_cache.append((qk, sk, qv, sv))

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated, per_token_ms, q_cache


def theoretical_min_step_ms(cache_bytes_total: int, peak_bandwidth_gbps: float) -> float:
    """If a decode step were perfectly memory-bandwidth-bound (all it had to
    do was re-read the cache, at peak bandwidth, nothing else), this is the
    fastest that step could possibly be. Achieved time will be higher —
    the gap is everything else (compute, overhead, imperfect bandwidth
    utilization) on top of the bandwidth floor.
    """
    return (cache_bytes_total / (peak_bandwidth_gbps * 1e9)) * 1000


def run_context_sweep(device: torch.device, prompt_lengths=(32, 128, 512, 1024), num_new_tokens: int = 20):
    """For each starting prompt length: prefill, then decode num_new_tokens,
    reporting median per-token latency and peak memory. Rising per-token
    latency as prompt_lengths grows is the direct evidence decode is
    memory-bandwidth-bound (each step re-reads the whole cache).
    """
    model, tokenizer = load_model(device)
    results = []
    for prompt_len in prompt_lengths:
        input_ids = make_prompt(tokenizer, device, prompt_len)

        torch.cuda.reset_peak_memory_stats(device)
        _, per_token_ms, past = decode_fp16(model, input_ids, num_new_tokens)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        # Skip the first step (prefill processes the whole prompt at once —
        # not comparable to the single-new-token steps that follow).
        decode_steps = sorted(per_token_ms[1:])
        median_ms = decode_steps[len(decode_steps) // 2]

        results.append(dict(
            prompt_len=prompt_len,
            median_step_ms=median_ms,
            tokens_per_sec=1000.0 / median_ms,
            peak_mem_gb=peak_mem_gb,
            cache_bytes_fp16=cache_bytes(past, dtype_bytes=2),
        ))
    return results


def compare_fp16_vs_int8(device: torch.device, prompt_len: int = 512, num_new_tokens: int = 20):
    model, tokenizer = load_model(device)
    input_ids = make_prompt(tokenizer, device, prompt_len)

    torch.cuda.reset_peak_memory_stats(device)
    gen_fp16, ms_fp16, past_fp16 = decode_fp16(model, input_ids.clone(), num_new_tokens)
    mem_fp16_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    torch.cuda.reset_peak_memory_stats(device)
    gen_int8, ms_int8, q_cache = decode_int8_kv(model, input_ids.clone(), num_new_tokens)
    mem_int8_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    new_fp16 = gen_fp16[:, prompt_len:]
    new_int8 = gen_int8[:, prompt_len:]
    token_match_fraction = (new_fp16 == new_int8).float().mean().item()

    int8_cache_bytes_val = sum(qk.numel() + qv.numel() for qk, _, qv, _ in q_cache) if q_cache else 0

    def _median(ms_list, skip_first=True):
        vals = sorted(ms_list[1:] if skip_first else ms_list)
        return vals[len(vals) // 2]

    return dict(
        fp16=dict(median_step_ms=_median(ms_fp16), peak_mem_gb=mem_fp16_gb,
                   cache_bytes=cache_bytes(past_fp16, dtype_bytes=2)),
        int8=dict(median_step_ms=_median(ms_int8), peak_mem_gb=mem_int8_gb,
                   cache_bytes=int8_cache_bytes_val),
        token_match_fraction=token_match_fraction,
        text_fp16=tokenizer.decode(new_fp16[0], skip_special_tokens=True),
        text_int8=tokenizer.decode(new_int8[0], skip_special_tokens=True),
    )
