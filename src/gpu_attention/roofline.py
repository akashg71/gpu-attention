"""
Phase 3 stub. Not wired up yet — needs real ncu-measured bytes-moved numbers
per kernel before this is meaningful, so it's scaffolded but not implemented.

Plan: read peak FLOP/s (for the benchmark dtype) and peak HBM bandwidth from
the device at runtime (torch.cuda.get_device_properties + a per-arch peak-FLOPs
table, since torch doesn't expose peak FLOP/s directly), compute each kernel's
arithmetic intensity (FLOPs / bytes moved, bytes from ncu's dram__bytes.sum),
and plot against the roofline to show attention sitting in the memory-bound
region.
"""
import torch


def get_peak_hbm_bandwidth_gbps(device: torch.device) -> float:
    """Placeholder — torch doesn't expose memory bus width/clock in a portable
    way. Phase 3: either hardcode a small arch->peak-BW table (checked against
    the vendor datasheet for whatever GPU the profiling box actually has) or
    measure it empirically with a bandwidth-bound memcpy kernel.
    """
    raise NotImplementedError("Phase 3: implement once profiling box GPU is known")


def get_peak_flops(device: torch.device, dtype: torch.dtype) -> float:
    """Same issue as above — peak FLOP/s depends on GPU arch and dtype
    (tensor core fp16/bf16 vs fp32) and isn't queryable from torch directly.
    """
    raise NotImplementedError("Phase 3: implement once profiling box GPU is known")
