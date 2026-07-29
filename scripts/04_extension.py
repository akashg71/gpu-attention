"""
Phase 4: KV-cache decode. Sweeps context length to show decode is
memory-bandwidth-bound (rising per-token latency as the cache grows), then
compares the baseline (bf16 — see kvcache.py's module docstring for why not
fp16) vs int8 KV-cache at a fixed context length (memory, speed, and
output-token match).

    python scripts/04_extension.py
"""
import sys

sys.path.insert(0, "src")


def main():
    from gpu_attention.env import get_device
    from gpu_attention.kvcache import compare_baseline_vs_int8, run_context_sweep, theoretical_min_step_ms
    from gpu_attention.roofline import get_peak_hbm_bandwidth_gbps

    device = get_device()
    peak_bw = get_peak_hbm_bandwidth_gbps(device)

    print("=== Context-length sweep (baseline dtype cache) ===")
    print(f"{'prompt_len':>10} {'median step (ms)':>18} {'tokens/s':>10} {'peak mem (GB)':>15} {'theoretical min (ms)':>22}")
    sweep_results = run_context_sweep(device)
    for r in sweep_results:
        theo_min = theoretical_min_step_ms(r["cache_bytes_baseline"], peak_bw)
        print(f"{r['prompt_len']:>10} {r['median_step_ms']:>18.3f} {r['tokens_per_sec']:>10.1f} "
              f"{r['peak_mem_gb']:>15.3f} {theo_min:>22.3f}")

    print("\n=== baseline vs int8 KV-cache, prompt_len=512 ===")
    cmp = compare_baseline_vs_int8(device, prompt_len=512, num_new_tokens=20)
    for name in ("baseline", "int8"):
        r = cmp[name]
        print(f"{name:<9} median_step_ms={r['median_step_ms']:.3f}  "
              f"peak_mem_gb={r['peak_mem_gb']:.3f}  cache_bytes={r['cache_bytes']:,}")

    print(f"\ncache size ratio (int8/baseline): {cmp['int8']['cache_bytes'] / cmp['baseline']['cache_bytes']:.2f}")
    print(f"token match fraction (int8 vs baseline greedy decode): {cmp['token_match_fraction']:.2f}")
    print(f"\nbaseline generated: {cmp['text_baseline']!r}")
    print(f"int8 generated: {cmp['text_int8']!r}")


if __name__ == "__main__":
    main()
