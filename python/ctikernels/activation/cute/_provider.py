# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Dynamic-M tvm-ffi compilation cache, including unavailable compiler results."""

import importlib.metadata
import threading
from functools import lru_cache

_LOCK = threading.RLock()
_COMPILED = {}
_FAILURES = {}
_GENERAL_COMPILED = {}


@lru_cache(maxsize=1)
def available():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except importlib.metadata.PackageNotFoundError:
        return False


def compile_kernel(dtype, width, config, capability):
    _validate_width(width, config)
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._kernel import Kernel

    scalar = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16}[dtype]
    # The ABI and every address calculation remain 64-bit, including input strides * M.
    m = cute.sym_int64()
    fake = cute.runtime.make_fake_tensor
    x = fake(scalar, (m, 2 * width), (2 * width, 1), assumed_align=16)
    y = fake(scalar, (m, width), (width, 1), assumed_align=16)
    return cute.compile(
        Kernel(width, config),
        x,
        y,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=(
            f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a"
            " --ptxas-options=--fmad=false"
        ),
    )


def _validate_width(width, config):
    if width <= 0 or width % config.elems:
        raise ValueError(
            f"CuTe SiLU elems={config.elems} requires positive width divisible by {config.elems}"
        )


def get_compiled(
    device,
    dtype,
    width,
    config,
    capability,
    capturing=False,
    strict=False,
    *,
    capture_miss="fallback",
):
    _validate_width(width, config)
    key = (device, dtype, width, config, capability)
    if key in _COMPILED:
        return _COMPILED[key]

    if key in _FAILURES:
        if strict:
            raise RuntimeError("CuTe SiLU compilation unavailable") from _FAILURES[key]
        return None
    if capturing:
        if strict:
            raise RuntimeError("warm CuTe SiLU before CUDA graph capture")
        if capture_miss == "raise":
            raise RuntimeError(
                "CuTe SiLU config is not compiled; call warm_up(max_m) "
                "(or one eager call of this shape) before CUDA graph capture"
            )
        return None
    with _LOCK:
        if key not in _COMPILED and key not in _FAILURES:
            try:
                _COMPILED[key] = compile_kernel(dtype, width, config, capability)
            except Exception as error:  # noqa: BLE001 -- optional compiler boundary only
                _FAILURES[key] = error
        if key in _FAILURES:
            if strict:
                raise RuntimeError("CuTe SiLU compilation unavailable") from _FAILURES[key]
            return None
        return _COMPILED[key]


def get_general_compiled(device, dtype, width, kind, lanes, capability, *, capturing=False):
    """Required native path; no device whitelist or torch arithmetic fallback."""
    key = (device, dtype, width, kind, lanes, capability)
    if key in _GENERAL_COMPILED:
        return _GENERAL_COMPILED[key]
    if capturing:
        raise RuntimeError("warm CuTe activation before CUDA graph capture")
    if not available():
        raise RuntimeError("CuTe activation requires nvidia-cutlass-dsl==4.8.0")
    with _LOCK:
        if key not in _GENERAL_COMPILED:
            _GENERAL_COMPILED[key] = compile_general(dtype, width, kind, lanes, capability)
    return _GENERAL_COMPILED[key]


def compile_general(dtype, width, kind, lanes, capability):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from .._dispatch import Config
    from ._kernel import Kernel

    scalar = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
    }[dtype]
    m = cute.sym_int64()
    input_width = width if kind == "relu2" else 2 * width
    alignment = lanes * scalar.width // 8
    fake = cute.runtime.make_fake_tensor
    x = fake(
        scalar, (m, input_width), (cute.sym_int64(divisibility=lanes), 1), assumed_align=alignment
    )
    y = fake(scalar, (m, width), (cute.sym_int64(divisibility=lanes), 1), assumed_align=alignment)
    return cute.compile(
        Kernel(width, Config(), kind=kind, general=True, lanes=lanes),
        x,
        y,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=(
            f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}"
            + ("a" if capability[0] >= 9 else "")
            + " --ptxas-options=--fmad=false"
        ),
    )
