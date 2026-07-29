"""
Phase 4: KV-cache decode. Sweeps context length to show decode is
memory-bandwidth-bound (rising per-token latency as the cache grows), then
compares fp16 vs int8 KV-cache at a fixed context length (memory, speed,
and output-token match).

    python scripts/04_extension.py
"""
import sys

sys.path.insert(0, "src")


def main():
    from gpu_attention.env import get_device
    from gpu_attention.kvcache import compare_fp16_vs_int8, run_context_sweep, theoretical_min_step_ms
    from gpu_attention.roofline import get_peak_hbm_bandwidth_gbps

    device = get_device()
    peak_bw = get_peak_hbm_bandwidth_gbps(device)

    print("=== Context-length sweep (fp16 cache) ===")
    print(f"{'prompt_len':>10} {'median step (ms)':>18} {'tokens/s':>10} {'peak mem (GB)':>15} {'theoretical min (ms)':>22}")
    sweep_results = run_context_sweep(device)
    for r in sweep_results:
        theo_min = theoretical_min_step_ms(r["cache_bytes_fp16"], peak_bw)
        print(f"{r['prompt_len']:>10} {r['median_step_ms']:>18.3f} {r['tokens_per_sec']:>10.1f} "
              f"{r['peak_mem_gb']:>15.3f} {theo_min:>22.3f}")

    print("\n=== fp16 vs int8 KV-cache, prompt_len=512 ===")
    cmp = compare_fp16_vs_int8(device, prompt_len=512, num_new_tokens=20)
    for name in ("fp16", "int8"):
        r = cmp[name]
        print(f"{name:<6} median_step_ms={r['median_step_ms']:.3f}  "
              f"peak_mem_gb={r['peak_mem_gb']:.3f}  cache_bytes={r['cache_bytes']:,}")

    print(f"\ncache size ratio (int8/fp16): {cmp['int8']['cache_bytes'] / cmp['fp16']['cache_bytes']:.2f}")
    print(f"token match fraction (int8 vs fp16 greedy decode): {cmp['token_match_fraction']:.2f}")
    print(f"\nfp16 generated: {cmp['text_fp16']!r}")
    print(f"int8 generated: {cmp['text_int8']!r}")


if __name__ == "__main__":
    main()
