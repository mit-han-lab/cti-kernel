# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from SGLang 2428d0f531602b882892f9f29521cd07ddd3c959
# (python/sglang/kernels/ops/quantization/fp8_kernel.py) and vLLM
# f4b8d43a681a908ce22b63ece3a86ab5b198e760
# (vllm/model_executor/layers/fusion/fused_act_quant.py), via our pre-merge
# CuTe activation variant. Independently written DSL; see THIRD_PARTY_NOTICES.

"""PDL activation quantization, matching the package Triton quantizer bit for bit.

DSL imports are deferred until compilation. Eight row-groups per CTA, sixteen
lanes per group, eight contiguous elements per lane (128-bit aligned loads).
Unaligned row strides/pointers use scalar loads. M and the row stride are symbolic.
Reference design: SGLang 72730d6 clamped activation and UE8M0 quantizer.
Reference: public PTX ISA https://docs.nvidia.com/cuda/parallel-thread-execution/
"""

import math
import threading

import torch

_compiled: dict[tuple, object] = {}
_compiling = threading.Lock()
_ready: set[tuple] = set()


def _compile(
    k,
    dtype,
    vectorized,
    capability,
    *,
    gated=False,
    swiglu_limit=None,
    scale_ue8m0=False,
    scale_layout="k",
):
    groups = (k + 127) // 128
    max_op = "max.f32" if scale_ue8m0 else "max.NaN.f32"
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float8E4M3FN, Float16, Float32, Int64, Uint16
    from cutlass.runtime import make_fake_stream

    from ._block_fp8_quant_math import fp32_op, pack_pair, quant_scale, quant_value, silu_product

    class Quantizer:
        @cute.jit
        def __call__(
            self, x: cute.Tensor, q: cute.Tensor, scale: cute.Tensor, stream: cuda.CUstream
        ):
            self.run(x, q, scale).launch(
                grid=(cute.ceil_div(x.shape[0] * groups, 8), 1, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def run(self, x: cute.Tensor, q: cute.Tensor, scale: cute.Tensor):
            cute.arch.griddepcontrol_wait()
            # A dependent must wait for this entire grid before reading q/scale.
            # Signal even for inactive lanes in the last CTA.
            cute.arch.griddepcontrol_launch_dependents()
            bid, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            group = Int64(bid) * 8 + tid // 16
            row, kg = group // groups, group % groups
            col = kg * 128 + (tid % 16) * 8
            values = cute.make_rmem_tensor(8, x.element_type)
            values.fill(0)
            if row < x.shape[0]:
                offset = cute.assume(row * x.stride[0] + col, divby=8 if vectorized else 1)
                source = cute.make_tensor(x.iterator + offset, cute.make_layout(8))
                if cutlass.const_expr(vectorized):
                    load = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128
                    )
                    cute.copy(load, source, values)
                else:
                    for i in cutlass.range_constexpr(8):
                        if col + i < k:
                            values[i] = source[i]
            v = values.load().to(Float32)
            if cutlass.const_expr(gated):
                up = cute.make_rmem_tensor(8, x.element_type)
                up.fill(0)
                if row < x.shape[0]:
                    offset = cute.assume(row * x.stride[0] + col + k, divby=8 if vectorized else 1)
                    source_up = cute.make_tensor(x.iterator + offset, cute.make_layout(8))
                    if cutlass.const_expr(vectorized):
                        load_up = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128
                        )
                        cute.copy(load_up, source_up, up)
                    else:
                        for i in cutlass.range_constexpr(8):
                            if col + i < k:
                                up[i] = source_up[i]
                act = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(8):
                    g, u = v[i], Float32(up[i])
                    value = silu_product(g, u, swiglu_limit)
                    # Match the materialized BF16/FP16 product of the two-kernel path.
                    act[i] = Float32(value.to(x.element_type))
                v = act.load()
            # triton/_block_fp8.py:57-60: NaN-propagating abs-max, eps floor,
            # then canonical NaN. Max is exact, so reduction topology is immaterial.
            amax = Float32(0)
            for i in cutlass.range_constexpr(8):
                amax = fp32_op(max_op, amax, abs(v[i]))
            for i in cutlass.range_constexpr(4):
                amax = fp32_op(max_op, amax, cute.arch.shuffle_sync_bfly(amax, offset=8 >> i))
            stored, multiplier = quant_scale(amax, scale_ue8m0)
            rounded = cute.make_rmem_tensor(8, Float32)
            for i in cutlass.range_constexpr(8):
                # triton/_block_fp8.py:65-66,119: multiply, then propagating clamp.
                # Inf amax yields +0 multiplier: finite entries become signed zero,
                # infinite entries become NaN; a NaN amax makes the whole group NaN.
                rounded[i] = quant_value(v[i], multiplier, scale_ue8m0)
            packed = cute.make_rmem_tensor(4, Uint16)
            for i in cutlass.range_constexpr(4):
                # triton/_block_fp8.py:122: E4M3 RN ties-to-even, finite saturation.
                # PTX's first scalar occupies the high byte; store in input order.
                packed[i] = pack_pair(rounded[2 * i], rounded[2 * i + 1])
            if row < x.shape[0]:
                if cutlass.const_expr(k % 128 == 0):
                    target = cute.make_tensor(
                        cute.recast_ptr(
                            q.iterator + cute.assume(row * k + col, divby=8), dtype=Uint16
                        ),
                        cute.make_layout(4),
                    )
                    store = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), Uint16, num_bits_per_copy=64
                    )
                    cute.copy(store, packed, target)
                else:
                    for i in cutlass.range_constexpr(8):
                        if col + i < k:
                            q[row, col + i] = rounded[i].to(Float8E4M3FN)
                if tid % 16 == 0:
                    scale[row, kg] = stored

    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int64()
    input_type = BFloat16 if dtype == torch.bfloat16 else Float16
    x = fake(
        input_type,
        (m, k * (2 if gated else 1)),
        (cute.sym_int64(divisibility=8 if vectorized else 1), 1),
        assumed_align=16 if vectorized else 2,
    )
    q = fake(Float8E4M3FN, (m, k), (k, 1), assumed_align=16)
    scale = fake(
        Float32,
        (m, groups),
        (groups, 1) if scale_layout == "k" else (1, cute.sym_int64(divisibility=4)),
        assumed_align=4,
    )
    major, minor = capability
    return cute.compile(
        Quantizer(),
        x,
        q,
        scale,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a",
    )


def quantize(x, *, scale_ue8m0=False, outputs=None):
    """Return contiguous E4M3 [M,K] and FP32 [M,K/128]; warm up before capture."""
    from ..block_fp8 import _check_activation, _require_inference
    from ._block_fp8_provider import available

    check_options(None, scale_ue8m0, "k")
    if not isinstance(x, torch.Tensor):
        raise TypeError("cute-fp8 quantizer: x must be a tensor")
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 128:
        raise ValueError("cute-fp8 quantizer: need [M,K] with positive K divisible by 128")
    _check_activation("cute-fp8 quantizer", x, x.device, x.shape[1])
    _require_inference("cute-fp8 quantizer", x)
    if not x.is_cuda:
        raise ValueError("cute-fp8 quantizer: x must be on CUDA")
    m, k = x.shape
    # Rows are only read, so broadcast or overlapping rows (stride(0) < K) are valid.
    # block_fp8.py:258-259: no MN-major padding; both public outputs are contiguous.
    if outputs is None:
        q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    else:
        q, scale = outputs
    if not m:
        return q, scale
    vectorized = x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    key = (x.device, k, x.dtype, vectorized, scale_ue8m0)
    with torch.cuda.device(x.device):
        # A first eager launch loads the module for the context, whatever the stream.
        ready_key = key
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and ready_key not in _ready:
            raise RuntimeError(
                "cute-fp8 quantizer: warmup this dtype/row-stride class before capture"
            )
        fn = _compiled.get(key)
        if fn is None:
            with _compiling:
                fn = _compiled.get(key)
                if fn is None:
                    capability = torch.cuda.get_device_capability(x.device)
                    if not available(capability):
                        raise RuntimeError(
                            "cute-fp8 quantizer: need sm120/sm121 and CuTe DSL 4.8.0"
                        )
                    fn = _compile(k, x.dtype, vectorized, capability, scale_ue8m0=scale_ue8m0)
                    _compiled[key] = fn
        fn(x, q, scale)
        _ready.add(ready_key)
    return q, scale


def check_options(swiglu_limit, scale_ue8m0, scale_layout):
    if swiglu_limit is not None and (
        not isinstance(swiglu_limit, (int, float))
        or not math.isfinite(swiglu_limit)
        or swiglu_limit <= 0
        or swiglu_limit > 3.38e38
    ):
        raise ValueError("swiglu_limit must be a positive finite BF16-representable limit")
    if not isinstance(scale_ue8m0, bool):
        raise TypeError("scale_ue8m0 must be bool")
    if scale_layout not in ("k", "mn"):
        raise ValueError("scale_layout must be 'k' or 'mn'")


def allocate_outputs(m, k, device, scale_layout):
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    groups = (k + 127) // 128
    if scale_layout == "mn":
        scale = torch.empty((groups, (m + 3) // 4 * 4), dtype=torch.float32, device=device).t()[:m]
    else:
        scale = torch.empty((m, groups), dtype=torch.float32, device=device)
    return q, scale


def silu_quantize(x, *, swiglu_limit=None, scale_ue8m0=False, scale_layout="k", outputs=None):
    """Gate-first input, rounded product, one PDL launch; validated by the public wrapper.

    Reference design: SGLang 72730d6 silu_and_mul_clamp and per-token-group v2 quantization.
    The plain arithmetic is rebuilt from our earlier CuTe SiLU kernel (revision 950aa9e).
    """
    from ._block_fp8_provider import available

    check_options(swiglu_limit, scale_ue8m0, scale_layout)
    m, two_k = x.shape
    k = two_k // 2
    q, scale = allocate_outputs(m, k, x.device, scale_layout) if outputs is None else outputs
    if not m or not k:
        return q, scale
    vectorized = k % 128 == 0 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    key = ("silu", x.device, k, x.dtype, vectorized, swiglu_limit, scale_ue8m0, scale_layout)
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing() and key not in _ready:
            raise RuntimeError("cute-fp8 SiLU quantizer: warmup this signature before capture")
        fn = _compiled.get(key)
        if fn is None:
            with _compiling:
                fn = _compiled.get(key)
                if fn is None:
                    capability = torch.cuda.get_device_capability(x.device)
                    if not available(capability):
                        raise RuntimeError(
                            "cute-fp8 SiLU quantizer: need sm120/sm121 and DSL 4.8.0"
                        )
                    fn = _compile(
                        k,
                        x.dtype,
                        vectorized,
                        capability,
                        gated=True,
                        swiglu_limit=swiglu_limit,
                        scale_ue8m0=scale_ue8m0,
                        scale_layout=scale_layout,
                    )
                    _compiled[key] = fn
        fn(x, q, scale)
        _ready.add(key)
    return q, scale
