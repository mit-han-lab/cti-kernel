# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Owned FP16 attention staging, with one FP32 scale / BF16 rounding launch.

Compile and reserve outside capture. TVM-FFI borrows tensors and the current
stream; sizes/scales/PDL are host arguments, never read back from the device.
"""

import math
import threading
from functools import lru_cache

import torch

from .._storage import reject_grad

_compiling = threading.Lock()


@lru_cache(maxsize=None)
def _compile(source_dtype, output_dtype, arch):
    if (source_dtype, output_dtype) != (torch.float16, torch.bfloat16):
        raise ValueError("scale-cast supports FP16 to BF16 only")
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._scale_cast_kernel import ScaleCast

    size = cute.sym_int(64, divisibility=8)
    fake = cute.runtime.make_fake_tensor
    return cute.compile(
        ScaleCast(int(arch.removeprefix("sm_").rstrip("a")) >= 90),
        fake(cutlass.Float16, (size,), (1,), assumed_align=16),
        fake(cutlass.BFloat16, (size,), (1,), assumed_align=2),
        cutlass.Float32(1),
        cutlass.Boolean(False),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


def _scale(value):
    if value is None:
        return 1.0
    # Reject tensors before __float__ can synchronize a device scalar. Allow
    # NaN/inf: the kernel uses IEEE multiplication, including 0 * inf -> NaN.
    if type(value) not in (int, float):
        raise ValueError("v_scale must be a host float or None")
    value = float(value)
    # A finite host float outside FP32 range rounds to infinity in the ABI.
    return math.copysign(math.inf, value) if abs(value) > 3.4028234663852886e38 else value


class FP16StagedOutput:
    """Own 2 * capacity_elements bytes, separate from borrowed attention scratch.

    reserve() grows storage only at plan time; run() allocates only when out=None.
    Graph callers keep geometry fixed and serialize access on one stream.
    For |v_scale| <= 1 (None means 1), stage in FP16 then scale in FP32.
    The stage's 2^-24 quantum gives an absolute floor ~2^-24 * |v_scale|.
    Otherwise use native BF16 output with the same plan, including inf/NaN.
    FlashInfer has no FP32 output. Graph capture bakes the host scale and route.
    """

    def __init__(self, shape, device):
        self.device = torch.device(device)
        major, minor = torch.cuda.get_device_capability(self.device)
        self.supports_pdl = major >= 9
        with _compiling:
            self._kernel = _compile(torch.float16, torch.bfloat16, f"sm_{major}{minor}")
        self._storage = None
        self._bf16_output = None
        self.last_output_dtype = None
        self.reserve(shape)

    def prepare(self, native, *, head_dim, idx_dtype, decode):
        """Load the matching BF16 module after every native plan, outside capture."""
        from ._flashinfer import BF16OutputModule

        self._bf16_output = BF16OutputModule(
            native, head_dim=head_dim, idx_dtype=idx_dtype, decode=decode
        )

    def reserve(self, shape):
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("staged output requires positive [tokens, heads, dim]")
        count = math.prod(shape)
        if count % 8:
            raise ValueError("staged output element count must be divisible by eight")
        if self._storage is None or self._storage.numel() < count:
            self._storage = torch.empty(count, dtype=torch.float16, device=self.device)
        self.stage = self._storage[:count].view(shape)

    def run(self, native, *, q, paged_kv_cache, out, k_scale, v_scale, pdl=None):
        reject_grad(q, *paged_kv_cache, *((out,) if out is not None else ()))
        if (
            q.shape != self.stage.shape
            or q.dtype != torch.bfloat16
            or q.device != self.stage.device
        ):
            raise ValueError("query must match the staged plan shape, BF16 dtype and device")
        if out is not None and (
            out.shape != q.shape
            or out.dtype != torch.bfloat16
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 with the query shape and device")
        scale = _scale(v_scale)
        pdl = self.supports_pdl if pdl is None else pdl and self.supports_pdl
        if not abs(scale) <= 1:
            if self._bf16_output is None:
                raise RuntimeError("prepare the BF16 output module before running")
            self.last_output_dtype = torch.bfloat16
            return self._bf16_output.run(
                native,
                q=q,
                paged_kv_cache=paged_kv_cache,
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
                enable_pdl=pdl,
            )
        self.last_output_dtype = torch.float16
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        native.run(
            q=q,
            paged_kv_cache=paged_kv_cache,
            out=self.stage,
            k_scale=k_scale,
            v_scale=None,
            enable_pdl=pdl,
        )
        with torch.cuda.device(q.device):
            self._kernel(self.stage.view(-1), out.view(-1), scale, pdl)
        return out
