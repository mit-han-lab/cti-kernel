# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Device-scalar GEMM epilogue for trainable vocabulary heads."""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from pathlib import Path

import torch

from .utils import load_aot

WORKSPACE_BYTES = 32 * 1024**2


def _library_dirs() -> list[Path]:
    """Candidate cuBLASLt directories: nvidia wheels (cu12 cublas/lib or cu13 lib), toolkit."""
    roots = [Path(torch.__file__).resolve().parent.parent / "nvidia"]
    try:
        spec = importlib.util.find_spec("nvidia")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        roots.extend(Path(location) for location in spec.submodule_search_locations or ())
    directories = [
        path.parent for root in roots for path in sorted(root.glob("*/lib/libcublasLt.so.*"))
    ]
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    directories.extend(Path(home) / name for name in ("lib64", "lib"))
    return list(dict.fromkeys(directories))


@cache
def _load_module():
    # Link the SONAME that torch's own cuBLAS uses when present (same library at runtime).
    for directory in _library_dirs():
        for library in sorted(directory.glob("libcublasLt.so.*")):
            major = library.name.removeprefix("libcublasLt.so.").split(".")[0]
            soname = f"libcublasLt.so.{major}"
            if major.isdigit() and (directory / soname).is_file():
                return load_aot(
                    "alpha_gemm",
                    cuda_files=["alpha_gemm.cu"],
                    extra_ldflags=[f"-L{directory}", f"-Wl,-rpath,{directory}", f"-l:{soname}"],
                )
    raise RuntimeError(
        "alpha_gemm requires libcublasLt.so.<major> (nvidia cublas wheel or CUDA toolkit)"
    )


@cache
def _device_zero(device: torch.device) -> torch.Tensor:
    # Initialize during eager warmup, before capturing the first backward; the
    # one-time sync lets any later stream (or captured graph) read it.
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up alpha_gemm before CUDA graph capture")
        zero = torch.zeros((), dtype=torch.float32, device=device)
        torch.cuda.current_stream().synchronize()
        return zero


def alpha_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write alpha * a.T @ b, scaling FP32 accumulation before the output cast.

    Inputs are [K,M] and [K,N] with unit column stride; out is contiguous
    [M,N]. CUDA requires caller-owned uint8 workspace of at least 32 MiB.
    CPU is the explicit FP32 reference implementation. CUDA build errors propagate.
    """
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("alpha_gemm inputs and out must be matrices")
    if a.shape[0] != b.shape[0] or out.shape != (a.shape[1], b.shape[1]):
        raise ValueError("alpha_gemm requires A[K,M], B[K,N], out[M,N]")
    if a.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not (
        a.dtype == b.dtype == out.dtype
    ):
        raise TypeError("alpha_gemm matrices must share BF16, FP16 or FP32 dtype")
    if alpha.dtype != torch.float32 or alpha.ndim > 1 or alpha.numel() != 1:
        raise TypeError("alpha_gemm alpha must be a scalar or one-element FP32 tensor")
    if any(t.device != a.device for t in (b, alpha, out)):
        raise ValueError("alpha_gemm tensors must share a device")
    if any(t.stride(1) != 1 or t.stride(0) < t.shape[1] for t in (a, b)):
        raise ValueError("alpha_gemm inputs require unit column stride and nonoverlapping rows")
    if not out.is_contiguous():
        raise ValueError("alpha_gemm out must be contiguous")
    if a.device.type == "cpu":
        torch.mul(a.float().t() @ b.float(), alpha.reshape(()), out=out)
        return out
    if not a.is_cuda:
        raise ValueError("alpha_gemm requires CPU or CUDA tensors")
    if (
        workspace is None
        or workspace.device != a.device
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < WORKSPACE_BYTES
        or workspace.data_ptr() % 256 != 0
    ):
        raise ValueError(
            "alpha_gemm requires at least 32 MiB of contiguous, 256-byte-aligned CUDA uint8 workspace"
        )
    module = _load_module()
    zero = _device_zero(a.device)
    with torch.cuda.device(a.device):
        module.alpha_gemm(a, b, alpha, zero, out, workspace)
    return out
