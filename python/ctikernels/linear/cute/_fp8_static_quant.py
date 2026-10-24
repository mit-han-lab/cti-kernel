# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Static E4M3 activation quantization; DSL imports deferred until compilation.

Each thread converts eight adjacent elements. Aligned rows use 128-bit loads
and 64-bit stores; other rows and K tails use scalar accesses. M, K and row
stride are symbolic. Division and conversion use public PTX ISA instructions.
"""

import threading

import torch

_compiled: dict[tuple, object] = {}
_ready: set[tuple] = set()
_compiling = threading.Lock()


def _compile(dtype, vectorized, write_scale, capability, pdl=False):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float8E4M3FN, Float16, Float32, Int64, Uint8, Uint16
    from cutlass.runtime import make_fake_stream

    from ._block_fp8_quant_math import fp32_op, pack_pair

    class Quantizer:
        @cute.jit
        def __call__(
            self, x: cute.Tensor, s: cute.Tensor, q: cute.Tensor, scale, stream: cuda.CUstream
        ):
            self.run(x, s, q, scale).launch(
                grid=(cute.ceil_div(x.shape[0] * cute.ceil_div(x.shape[1], 8), 128), 1, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=pdl,
            )

        @cute.kernel
        def run(self, x: cute.Tensor, s: cute.Tensor, q: cute.Tensor, scale):
            if cutlass.const_expr(pdl):
                cute.arch.griddepcontrol_wait()
                # Match the block quantizer: signal all lanes, including inactive
                # ones, before loads. Dependents wait for the entire producer grid.
                cute.arch.griddepcontrol_launch_dependents()
            bid, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            chunk = Int64(bid) * 128 + tid
            chunks = cute.ceil_div(x.shape[1], 8)
            row, col = chunk // chunks, (chunk % chunks) * 8
            if row < x.shape[0]:
                values = cute.make_rmem_tensor(8, x.element_type)
                values.fill(0)
                if cutlass.const_expr(vectorized):
                    source = cute.make_tensor(
                        x.iterator + cute.assume(row * x.stride[0] + col, divby=8),
                        cute.make_layout(8),
                    )
                    load = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128
                    )
                    cute.copy(load, source, values)
                else:
                    for i in cutlass.range_constexpr(8):
                        if col + i < x.shape[1]:
                            values[i] = x[row, col + i]
                divisor = s[0]
                converted = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(8):
                    value = fp32_op("div.rn.f32", Float32(values[i]), divisor)
                    value = fp32_op("max.NaN.f32", value, Float32(-448.0))
                    converted[i] = fp32_op("min.NaN.f32", value, Float32(448.0))
                packed = cute.make_rmem_tensor(4, Uint16)
                for i in cutlass.range_constexpr(4):
                    packed[i] = pack_pair(converted[2 * i], converted[2 * i + 1])
                if cutlass.const_expr(vectorized):
                    target = cute.make_tensor(
                        cute.recast_ptr(
                            q.iterator + cute.assume(row * q.shape[1] + col, divby=8),
                            dtype=Uint16,
                        ),
                        cute.make_layout(4),
                    )
                    store = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), Uint16, num_bits_per_copy=64
                    )
                    cute.copy(store, packed, target)
                else:
                    bits = cute.make_tensor(cute.recast_ptr(q.iterator, dtype=Uint8), q.layout)
                    for i in cutlass.range_constexpr(8):
                        if col + i < x.shape[1]:
                            bits[row, col + i] = Uint8(packed[i // 2] >> (8 * (i % 2)))
                if cutlass.const_expr(write_scale == "row"):
                    if col == 0:
                        scale[row, 0] = divisor
                elif cutlass.const_expr(write_scale):
                    if col % 128 == 0:
                        scale[row, col // 128] = divisor

    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int64()
    k = cute.sym_int64(divisibility=8 if vectorized else 1)
    groups = cute.sym_int64()
    input_type = BFloat16 if dtype == torch.bfloat16 else Float16
    x = fake(
        input_type,
        (m, k),
        (cute.sym_int64(divisibility=8 if vectorized else 1), 1),
        assumed_align=16 if vectorized else 2,
    )
    q = fake(Float8E4M3FN, (m, k), (k, 1), assumed_align=16)
    s = fake(Float32, (1,), (1,), assumed_align=4)
    scale = fake(Float32, (m, groups), (groups, 1), assumed_align=4) if write_scale else None
    major, minor = capability
    return cute.compile(
        Quantizer(),
        x,
        s,
        q,
        scale,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a",
    )


def quantize(x, input_scale, *, block_scale=False, row_scale=False, pdl=False, outputs=None):
    """Validated 2-D input; return q or (q, scales), with no host reads."""
    from ._block_fp8_provider import available

    if block_scale and row_scale:
        raise ValueError("block_scale and row_scale are mutually exclusive")
    mode = "row" if row_scale else block_scale
    m, k = x.shape
    if outputs is None:
        q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
        scale = (
            torch.empty(
                (m, 1 if row_scale else (k + 127) // 128), dtype=torch.float32, device=x.device
            )
            if mode
            else None
        )
    else:
        q, scale = outputs
    if not m:
        return (q, scale) if mode else q
    vectorized = k % 8 == 0 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    key = (x.device, x.dtype, vectorized, mode, pdl)
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing() and key not in _ready:
            raise RuntimeError(
                "static FP8 quantizer: warm up this dtype/stride class before capture"
            )
        fn = _compiled.get(key)
        if fn is None:
            with _compiling:
                fn = _compiled.get(key)
                if fn is None:
                    capability = torch.cuda.get_device_capability(x.device)
                    if not available(capability):
                        raise RuntimeError(
                            "static FP8 quantizer needs sm120/sm121 and CuTe DSL 4.8.0"
                        )
                    fn = _compile(x.dtype, vectorized, mode, capability, pdl)
                    _compiled[key] = fn
        # A singleton may retain an arbitrary stride after reshape. Its sole
        # address is unchanged by this view, which matches the compiled layout.
        fn(x, input_scale.as_strided((1,), (1,)), q, scale)
        _ready.add(key)
    return (q, scale) if mode else q
