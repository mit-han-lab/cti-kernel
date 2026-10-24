# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Cached TVM-FFI entry points using torch's current stream, including during capture."""

from __future__ import annotations

import threading

import cutlass
import cutlass.cute as cute
import torch
from cutlass.runtime import make_fake_stream

from . import _early_trigger, _geometry, _pdl
from .kernel import Kernel

_compiling = threading.Lock()
_OPTIONAL = (
    "weight_scale", "up_scale", "alpha", "output_scale", "reciprocal", "column_scale",
    "gate_factor", "up_factor", "clear_scale",
)  # fmt: skip


class Launcher:
    def __init__(self, device):
        self.device = device
        major, minor = torch.cuda.get_device_capability(device)
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.wide = (major, minor) == (12, 0)
        self.name = torch.cuda.get_device_name(device)
        self.options = f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a"
        self.compiled = {}

    def quantize(self, x, global_scale, *, gated, data, scale, gated_row_blocks=False, **kwargs):
        for name in ("weight_scale", "up_scale", "gate_factor", "up_factor"):
            value = kwargs.get(name)
            if value is not None and value.requires_grad:
                raise ValueError(f"NVFP4 quantize: {name} is frozen and must not require grad")
        if not x.shape[0]:
            return
        flags = tuple(kwargs.get(name) is not None for name in _OPTIONAL)
        columns = kwargs["column_scale"].numel() if flags[5] else 0
        clear = flags[8] and kwargs["clear_scale"].numel() > 0
        aligned_x = x.data_ptr() % 16 == 0 and x.stride(0) % 8 == 0
        aligned_out = data.data_ptr() % 4 == 0 and data.stride(0) % 4 == 0
        geometry = _geometry(data.shape[1] * 2, x.shape[0], self.wide)
        # Alignment is a runtime ABI argument, so offset views share cache keys.
        # Only whole 16-byte chunks use vector stores; other views fall back.
        scale_alignment = 16 if scale.data_ptr() % 16 == 0 else 4
        if scale.data_ptr() % 4:
            scale_alignment = 1
        key = (
            x.dtype, data.shape[1] * 2, gated, flags, columns, clear,
            aligned_x, aligned_out, geometry, gated_row_blocks,
            _early_trigger(data.shape[1] * 2, x.shape[0]),
            _pdl(data.shape[1] * 2, x.shape[0], self.name),
        )  # fmt: skip
        compiled = self.compiled.get(key)
        if compiled is None:
            with _compiling, torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prewarm this CuTe NVFP4 quantizer shape/dtype/alignment/geometry "
                        "outside CUDA graph capture"
                    )
                compiled = self.compiled.get(key)
                if compiled is None:
                    compiled = self._compile(key)
                    self.compiled[key] = compiled
        # Absent arguments alias a live scalar but are eliminated by constexpr branches.
        optional = tuple(
            kwargs[name].view(-1) if flags[i] else global_scale for i, name in enumerate(_OPTIONAL)
        )
        compiled(x, data, scale.view(-1), global_scale, *optional, scale_alignment, self.sm_count)

    def _compile(self, key):
        (
            dtype,
            cols,
            gated,
            flags,
            columns,
            clear,
            aligned_x,
            aligned_out,
            geometry,
            row_blocks,
            early,
            pdl,
        ) = key
        dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        rows = cute.sym_int64()
        fake = cute.runtime.make_fake_tensor
        x = fake(
            dtype,
            (rows, cols * (2 if gated else 1)),
            (cute.sym_int64(divisibility=8 if aligned_x else 1), 1),
            assumed_align=16 if aligned_x else 2,
        )
        data = fake(
            cutlass.Uint8,
            (rows, cols // 2),
            (cute.sym_int64(divisibility=4 if aligned_out else 1), 1),
            assumed_align=4 if aligned_out else 1,
        )
        scale = fake(cutlass.Uint8, (cute.sym_int64(divisibility=4),), (1,), assumed_align=1)
        scalar = fake(cutlass.Float32, (1,), (1,), assumed_align=4)
        optional = tuple(
            fake(
                cutlass.Uint8 if i == 8 else cutlass.Float32,
                (cute.sym_int64() if i == 8 else columns,), (1,), assumed_align=1 if i == 8 else 4,
            )
            if flags[i] and i in (5, 8)
            else scalar
            for i in range(len(_OPTIONAL))
        )  # fmt: skip
        kernel = Kernel(
            cols,
            gated,
            flags,
            columns,
            clear,
            aligned_x,
            aligned_out,
            geometry,
            gated_row_blocks=row_blocks,
            early_trigger=early,
            pdl=pdl,
        )
        return cute.compile(
            kernel,
            x,
            data,
            scale,
            scalar,
            *optional,
            cutlass.Int32(16),
            cutlass.Int32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=self.options,
        )
