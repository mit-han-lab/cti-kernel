# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Recipe follows FlashInfer 0.7.0.post1 quantization_utils.cuh, design reference only.
"""Test-only scalar-block CuTe PTX oracle, imported only inside GPU tests.

One thread owns sixteen values. No production quantizer, tiling, lane reduction,
or shared scale/factor routine: only the public PTX arithmetic primitives overlap.
"""

from functools import cache
from threading import Lock

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float8E4M3FN, Float32, Int64, Uint8, Uint32, cute
from cutlass.runtime import make_fake_stream

from ctikernels.linear import _nvfp4_recipe as ptx
from ctikernels.linear._nvfp4_cute_recipe import (
    amax_nonzero,
    maximum,
    mul,
    pack,
    rcp,
    reciprocal,
)

_E4M3 = ptx.cute_asm(ptx.E4M3, 1)
_compiling = Lock()


@cute.kernel
def _kernel(x: cute.Tensor, input_scale: cute.Tensor, data: cute.Tensor, scales: cute.Tensor):
    block = Int64(cute.arch.block_idx()[0]) * Int64(128) + Int64(cute.arch.thread_idx()[0])
    if block < scales.shape[0]:
        start = block * Int64(16)
        # Seed from the input so an all-NaN block stays NaN. Widen before abs/max.
        amax = cute.math.abs(Float32(x[start]))
        for i in cutlass.range(1, 16):
            amax = maximum(amax, cute.math.abs(Float32(x[start + Int64(i)])))

        inverse = reciprocal(Float32(input_scale[0]))
        scaled = mul(inverse, mul(amax, rcp(Float32(6.0))))
        bits = cute.arch.inline_ptx(_E4M3, write_only_types=[Uint32], read_only_args=[scaled])
        sf = Uint8(bits)
        decoded = sf.bitcast(Float8E4M3FN).to(Float32)
        factor = rcp(mul(decoded, rcp(inverse)))
        factor = Float32(cutlass.select_(amax_nonzero(amax) != Uint32(0), factor, Float32(0.0)))

        for pair in cutlass.range(8):
            low = Float32(x[start + Int64(pair) * Int64(2)])
            high = Float32(x[start + Int64(pair) * Int64(2) + Int64(1)])
            data[block * Int64(8) + Int64(pair)] = pack(mul(low, factor), mul(high, factor))
        scales[block] = sf


@cute.jit
def _launch(
    x: cute.Tensor,
    input_scale: cute.Tensor,
    data: cute.Tensor,
    scales: cute.Tensor,
    stream: cuda.CUstream,
):
    _kernel(x, input_scale, data, scales).launch(
        grid=[(scales.shape[0] + 127) // 128, 1, 1],
        block=[128, 1, 1],
        stream=stream,
    )


@cache
def _compile(dtype, capability):
    """One shape-independent compilation per input dtype and target architecture."""
    element = {
        torch.float32: Float32,
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
    }[dtype]
    fake = cute.runtime.make_fake_tensor
    x = fake(element, (cute.sym_int64(divisibility=16),), (1,), assumed_align=element.width // 8)
    scalar = fake(Float32, (1,), (1,), assumed_align=4)
    data = fake(Uint8, (cute.sym_int64(divisibility=8),), (1,), assumed_align=1)
    scales = fake(Uint8, (cute.sym_int64(),), (1,), assumed_align=1)
    major, minor = capability
    return cute.compile(
        _launch,
        x,
        scalar,
        data,
        scales,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a",
    )


def quantize(x, input_scale, data, scales):
    """Launch on torch's current stream; runtime tensors use the TVM-FFI DLPack ABI."""
    if not x.numel():
        return
    capability = torch.cuda.get_device_capability(x.device)
    if not ptx.native_e2m1(capability):
        raise RuntimeError("the CuTe oracle needs hardware E2M1 conversion (SM100+)")
    with _compiling, torch.cuda.device(x.device):
        compiled = _compile(x.dtype, capability)
    compiled(x.view(-1), input_scale.view(1), data.view(-1), scales.view(-1))
