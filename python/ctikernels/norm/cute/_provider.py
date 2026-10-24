# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Dynamic-M tvm-ffi compilation cache, including unavailable compiler results."""

import importlib.metadata
import threading
from functools import lru_cache

_LOCK = threading.RLock()
_COMPILED = {}
_FAILURES = {}


@lru_cache(maxsize=1)
def available():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except importlib.metadata.PackageNotFoundError:
        return False


def compile_kernel(
    dtype,
    weight_dtype,
    hidden,
    zero_centered,
    config,
    capability,
    store_residual=True,
    has_sum=False,
    residual_dtype=None,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._kernel import Kernel

    scalars = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}
    scalar = scalars[dtype]
    residual_scalar = scalars[residual_dtype or dtype]
    # The ABI and every address calculation remain 64-bit, including input strides * M.
    m = cute.sym_int64()
    fake = cute.runtime.make_fake_tensor
    x = fake(scalar, (m, hidden), (hidden, 1), assumed_align=16)
    residual = fake(residual_scalar, (m, hidden), (hidden, 1), assumed_align=16)
    weight = fake(scalars[weight_dtype], (hidden,), (1,), assumed_align=16)
    summed = fake(residual_scalar, (m, hidden), (cute.sym_int64(), 1), assumed_align=16)
    return cute.compile(
        Kernel(hidden, zero_centered, config, store_residual, has_sum),
        x,
        residual,
        weight,
        summed,
        cutlass.Int64(0),
        cutlass.Float32(1e-6),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --ptxas-options=--fmad=false --gpu-arch sm_{capability[0]}{capability[1]}a",
    )


def get_compiled(
    device,
    dtype,
    weight_dtype,
    hidden,
    zero_centered,
    config,
    capability,
    capturing=False,
    strict=False,
    store_residual=True,
    has_sum=False,
    residual_dtype=None,
    *,
    capture_miss="fallback",
):
    residual_dtype = residual_dtype or dtype
    # Config includes the cluster split; graph replay reuses the exact compiled launch.
    key = (
        device,
        residual_dtype,
        dtype,
        weight_dtype,
        hidden,
        zero_centered,
        config,
        capability,
        store_residual,
        has_sum,
    )
    if key in _COMPILED:
        return _COMPILED[key]
    if key in _FAILURES:
        if strict:
            raise RuntimeError("CuTe residual norm compilation unavailable") from _FAILURES[key]
        return None
    if capturing:
        if strict:
            raise RuntimeError("warm CuTe residual norm before CUDA graph capture")
        if capture_miss == "raise":
            raise RuntimeError(
                "CuTe residual norm config is not compiled; call warm_up(max_m) "
                "(or one eager call of this shape) before CUDA graph capture"
            )
        return None
    with _LOCK:
        if key not in _COMPILED and key not in _FAILURES:
            try:
                _COMPILED[key] = compile_kernel(
                    dtype,
                    weight_dtype,
                    hidden,
                    zero_centered,
                    config,
                    capability,
                    store_residual,
                    has_sum,
                    residual_dtype,
                )
            except Exception as error:  # noqa: BLE001 -- optional compiler boundary only
                _FAILURES[key] = error
        if key in _FAILURES:
            if strict:
                raise RuntimeError("CuTe residual norm compilation unavailable") from _FAILURES[key]
            return None
        return _COMPILED[key]
