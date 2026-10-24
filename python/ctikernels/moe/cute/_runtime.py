# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Shared lazy loader and compilation cache for all MoE CuTe kernels.

Experts may use ``rt = load(device)`` and ``rt.compile_cached(key, build)``;
``build()`` returns a compiled callable and runs under the shared compilation
lock. Include a kernel-family prefix and every specialization in ``key``.
Alternatively ``rt.compile(key, kernel, tensors, scalars=())`` builds contiguous
fake tensors with symbolic leading dimensions and a TVM-FFI environment stream.
Call the result with the tensors/scalars, WITHOUT an explicit stream argument.
Warm each specialization before graph capture. Failures compiling an available
provider propagate; they must not silently turn into a fallback or test skip.
"""

import importlib.metadata
import threading

import torch

VERSION = "4.8.0"
DISTRIBUTION = "nvidia-cutlass-dsl"
_loading = threading.Lock()
_compiling = threading.Lock()
_runtimes = {}


def arch_flag(capability):
    """Metadata-only architecture choice, also usable in CPU dispatch tests."""
    return {(12, 0): "sm_120a", (12, 1): "sm_121a"}.get(tuple(capability))


def load(device):
    """Return a device Runtime, or None for missing DSL/unsupported devices."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    try:
        if importlib.metadata.version(DISTRIBUTION) != VERSION:
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with _loading:
        if device not in _runtimes:
            arch = arch_flag(torch.cuda.get_device_capability(device))
            if arch is None:
                return None
            try:
                import cutlass.cute  # noqa: F401
                from cutlass.runtime import make_fake_stream  # noqa: F401
            except ImportError:
                return None
            _runtimes[device] = Runtime(device, arch)
        return _runtimes[device]


def require(device):
    runtime = load(device)
    if runtime is None:
        raise RuntimeError("MoE CuTe requires nvidia-cutlass-dsl 4.8.0 and SM120/SM121")
    return runtime


class Runtime:
    def __init__(self, device, arch):
        self.device = device
        self.arch = arch
        self.options = f"--enable-tvm-ffi --gpu-arch {arch}"
        self._compiled = {}

    def compile_cached(self, key, build):
        compiled = self._compiled.get(key)
        if compiled is None:
            with _compiling:
                compiled = self._compiled.get(key)
                if compiled is None:
                    with torch.cuda.device(self.device):
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError(
                                "warm CuTe kernel specializations before graph capture"
                            )
                        compiled = build()
                    self._compiled[key] = compiled
        return compiled

    def compile(self, key, kernel, tensors, scalars=()):
        def build():
            import cutlass
            import cutlass.cute as cute
            from cutlass.runtime import make_fake_stream

            types = {
                torch.bfloat16: cutlass.BFloat16,
                torch.float32: cutlass.Float32,
                torch.int32: cutlass.Int32,
                torch.int64: cutlass.Int64,
                torch.uint8: cutlass.Uint8,
                torch.uint16: cutlass.Uint16,
                torch.bool: cutlass.Boolean,
            }
            fake = []
            for tensor in tensors:
                if not tensor.is_contiguous():
                    raise ValueError("CuTe MoE requires contiguous tensors")
                shape = (cute.sym_int(), *tensor.shape[1:])
                # Views may be only element-aligned; do not overpromise to FFI.
                alignment = min(16, tensor.element_size())
                fake.append(
                    cute.runtime.make_fake_tensor(
                        types[tensor.dtype], shape, tuple(tensor.stride()), assumed_align=alignment
                    )
                )
            stream = make_fake_stream(use_tvm_ffi_env_stream=True)
            return cute.compile(kernel, *fake, *scalars, stream, options=self.options)

        # Include metadata because optional arguments are represented by dummies.
        signature = tuple((t.dtype, tuple(t.shape[1:]), tuple(t.stride())) for t in tensors)
        return self.compile_cached((key, signature), build)


def compile_cached(device, key, build):
    return require(device).compile_cached(key, build)
