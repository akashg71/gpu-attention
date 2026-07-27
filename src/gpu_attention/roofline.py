"""
Roofline model: peak FLOP/s and peak HBM bandwidth for the GPU actually in
use, each kernel's arithmetic intensity (FLOPs moved / bytes moved), and a
plot placing measured kernels against the theoretical roofline.

No CUDA API exposes theoretical peak FLOP/s or memory bandwidth — those are
spec-sheet constants, not runtime-queryable, so *some* lookup table is
unavoidable (torch.cuda.get_device_properties() gives clock rate/SM count,
but converting that to peak FLOP/s needs an arch-specific FLOPs-per-cycle
constant anyway, which is the same problem restated). Keyed by compute
capability rather than device name string, since that's what's actually
stable/queryable. Deliberately raises for anything not listed rather than
guessing — add an entry when profiling moves to different hardware.
"""
import torch

# Bytes moved / FLOPs are theoretical (vendor datasheet) peaks, not something
# ever fully achieved in practice — the roofline's value is showing how far
# *below* these ceilings a measured kernel sits, not hitting them.
_PEAK_SPECS = {
    (7, 5): {  # Turing — T4
        "name": "Turing (e.g. T4)",
        "hbm_bandwidth_gbps": 300.0,  # 16GB GDDR6, published spec
        "peak_flops": {
            torch.float32: 8.1e12,
            torch.float16: 65e12,  # tensor-core, dense
            # bf16 deliberately omitted: Turing (sm75) has no native bf16
            # tensor-core path (that's Ampere/sm80+). Phase 1 showed bf16
            # runs *correctly* on this GPU regardless, but its achieved
            # ceiling is a different, unmeasured question — don't guess a
            # peak number for it here.
        },
    },
    # (8, 0): {...},  # Ampere / A100 — add if profiling ever moves hardware
}


def _specs_for(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    key = (props.major, props.minor)
    if key not in _PEAK_SPECS:
        raise NotImplementedError(
            f"No peak-spec entry for compute capability {key} ({props.name}). "
            f"Add one to _PEAK_SPECS in roofline.py — check the vendor datasheet, "
            f"don't guess."
        )
    return _PEAK_SPECS[key]


def get_peak_hbm_bandwidth_gbps(device: torch.device) -> float:
    return _specs_for(device)["hbm_bandwidth_gbps"]


def get_peak_flops(device: torch.device, dtype: torch.dtype) -> float:
    specs = _specs_for(device)
    if dtype not in specs["peak_flops"]:
        raise NotImplementedError(
            f"No peak FLOP/s entry for {dtype} on {specs['name']}. "
            f"Add one to _PEAK_SPECS in roofline.py — check the vendor datasheet."
        )
    return specs["peak_flops"][dtype]


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte moved. The x-axis of a roofline plot — where a kernel
    sits relative to peak_flops/hbm_bandwidth (the ridge point) is what
    determines whether it's memory-bound (left of the ridge) or
    compute-bound (right of it).
    """
    return flops / bytes_moved


def plot_roofline(
    device: torch.device,
    dtype: torch.dtype,
    points: list[tuple],
    path: str,
) -> None:
    """points: list of (label, arithmetic_intensity, achieved_flops_per_sec).
    Draws the theoretical roofline (memory-bound ramp + compute-bound
    ceiling) for this GPU/dtype and places each measured kernel against it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    peak_flops = get_peak_flops(device, dtype)
    peak_bw_bytes = get_peak_hbm_bandwidth_gbps(device) * 1e9
    ridge_ai = peak_flops / peak_bw_bytes  # AI where the roofline bends

    colors = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#4a3aa7"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ai_values = [p[1] for p in points]
    ai_min = min(ai_values + [ridge_ai]) / 4
    ai_max = max(ai_values + [ridge_ai]) * 4

    # Roofline itself: memory-bound ramp (FLOP/s = AI * peak_bandwidth) up to
    # the ridge point, then a flat compute-bound ceiling at peak_flops.
    ramp_ai = np.array([ai_min, ridge_ai])
    ax.plot(ramp_ai, ramp_ai * peak_bw_bytes, color="#898781", linewidth=1.5, linestyle="--")
    ax.plot([ridge_ai, ai_max], [peak_flops, peak_flops], color="#898781", linewidth=1.5, linestyle="--")
    ax.annotate(f"peak {peak_flops / 1e12:.0f} TFLOP/s (compute-bound)",
                (ai_max, peak_flops), textcoords="offset points", xytext=(-6, 6),
                ha="right", fontsize=8, color="#898781")
    ax.annotate(f"peak {peak_bw_bytes / 1e9:.0f} GB/s (memory-bound)",
                (ridge_ai, ridge_ai * peak_bw_bytes), textcoords="offset points",
                xytext=(-40, -14), fontsize=8, color="#898781", rotation=38)

    for i, (label, ai, achieved_flops) in enumerate(points):
        color = colors[i % len(colors)]
        ax.scatter([ai], [achieved_flops], color=color, s=60, zorder=5, label=label)
        ax.annotate(label, (ai, achieved_flops), textcoords="offset points",
                    xytext=(8, 0), fontsize=9, color=color, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs/byte, log scale)", color="#52514e")
    ax.set_ylabel("achieved FLOP/s (log scale)", color="#52514e")
    ax.set_title(f"Roofline — {_specs_for(device)['name']}, {str(dtype).replace('torch.', '')}",
                 color="#0b0b0b", fontsize=13)
    ax.tick_params(colors="#898781")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
