# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Compilation cache only: mutable workspaces belong to individual operation objects."""

import importlib.metadata
import threading
from functools import lru_cache

VERSION = "4.8.0"
_LOCK = threading.RLock()
_COMPILED = {}


@lru_cache(maxsize=1)
def available():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def compile_kernel(n, k, config, capability, mtiles=1, gate_stride=0, *, fp32=False, out_stride=0):
    """Fake tensors allow tracing without allocating CUDA tensors (also useful on CPU).

    Nonzero ``out_stride`` compiles a row-strided [M,N] output (unit last stride).
    """
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32
    from cutlass.runtime import make_fake_stream

    from ._kernel import Kernel

    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int()
    x = fake(BFloat16, (m, k), (k, 1), assumed_align=16)
    w = fake(BFloat16, (n, k), (k, 1), assumed_align=16)
    y = fake(Float32 if fp32 else BFloat16, (m, n), (out_stride or n, 1), assumed_align=16)
    partial = (
        fake(Float32, (config.splits, m, n), (m * n, n, 1), assumed_align=16)
        if config.scratch_splits
        else fake(Float32, (1, 1, 1), (1, 1, 1), assumed_align=16)
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    arch = f"sm_{capability[0]}{capability[1]}a"
    args = (x, w, y, partial, stream)
    if gate_stride:
        args += (fake(BFloat16, (m, k), (gate_stride, 1), assumed_align=16),)
    return cute.compile(
        Kernel(
            n,
            k,
            config,
            streaming=capability == (12, 1),
            mtiles=mtiles,
            gate_stride=gate_stride,
            fp32=fp32,
            out_stride=out_stride,
        ),
        *args,
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


def mtiles_for(m):
    """M bucket: 8-row x tiles, 1..8 (M <= 64). Runtime M guards rows inside a bucket."""
    return (m + 7) // 8


def get_compiled(
    device,
    n,
    k,
    config,
    capability,
    capturing,
    mtiles=1,
    gate_stride=0,
    *,
    fp32=False,
    out_stride=0,
):
    # Runtime M guards input/output inside the bucket, with zero-filled MMA columns.
    key = (device, n, k, 8 * mtiles, config, capability, gate_stride)
    if fp32:
        key += ("fp32",)
    if out_stride:
        key += ("out_stride", out_stride)
    compiled = _COMPILED.get(key)
    if compiled is None and not capturing:
        with _LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                compiled = compile_kernel(
                    n,
                    k,
                    config,
                    capability,
                    mtiles,
                    gate_stride,
                    **({"fp32": True} if fp32 else {}),
                    **({"out_stride": out_stride} if out_stride else {}),
                )
                _COMPILED[key] = compiled
    return compiled
