"""
Environment/version check. Run this FIRST on any new machine (`python -m gpu_attention.env`)
before touching the kernel — if this doesn't show a CUDA GPU, nothing else in this repo will run.

Triton has no macOS build: it JIT-compiles to PTX and needs an NVIDIA driver + CUDA runtime
present at import/compile time. This whole project only runs on Linux + NVIDIA GPU
(Colab/Kaggle T4, or a rented A100/L4/H100).
"""
import sys


def get_device() -> "torch.device":
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible to torch. This repo requires an NVIDIA GPU + CUDA "
            "(Colab/Kaggle T4 to start, a rented A100/L4/H100 for profiling). "
            "See RUNBOOK.md."
        )
    return torch.device("cuda")


def print_env() -> None:
    print(f"Python:  {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("torch:   NOT INSTALLED — run `pip install -r requirements.txt` on a GPU box")
        return
    print(f"torch:   {torch.__version__}")

    if not torch.cuda.is_available():
        print("CUDA:    NOT AVAILABLE — torch.cuda.is_available() is False.")
        print("         Either you're not on a GPU machine, or the driver/CUDA build is wrong.")
        return

    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"SM count: {props.multi_processor_count}, compute capability: {props.major}.{props.minor}")
    total_mem_gb = props.total_memory / (1024 ** 3)
    print(f"VRAM:    {total_mem_gb:.1f} GB")

    try:
        import triton
        print(f"triton:  {triton.__version__}")
    except ImportError:
        print("triton:  NOT INSTALLED — expected to ship with a CUDA-enabled torch on Linux; "
              "if missing, `pip install triton` separately.")


if __name__ == "__main__":
    print_env()
