"""GPU / environment check for the garbage detection project.

Reports the PyTorch build, CUDA availability, GPU model, VRAM, and a small
allocation test so training readiness can be confirmed before launching
train_yolo.py.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("Python       :", sys.version.split()[0])
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed in this environment.")
        print("Install with: pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    print("PyTorch      :", torch.__version__)
    print("CUDA build   :", torch.version.cuda or "none (CPU-only build)")
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if not cuda_available:
        print("\nNo usable CUDA device. Training will be extremely slow on CPU.")
        print("Fix: install the CUDA build of PyTorch into .venv (see README).")
        return 2

    count = torch.cuda.device_count()
    print("GPU count    :", count)
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"GPU {i}        : {props.name}")
        print(f"  Compute capability : {props.major}.{props.minor}")
        print(f"  Total VRAM         : {total_gb:.2f} GB")
        print(f"  Multiprocessors    : {props.multi_processor_count}")

    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    print(f"VRAM reserved: {reserved_gb:.2f} GB (before test)")

    # Allocation test: a tensor block similar to a small training batch.
    try:
        test = torch.zeros(64, 3, 640, 640, device="cuda")
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Alloc test   : OK (64x3x640x640 tensor, peak {peak_gb:.2f} GB)")
        del test
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        print("Alloc test   : FAILED ->", exc)
        return 3

    print("\nREADY: GPU training supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
