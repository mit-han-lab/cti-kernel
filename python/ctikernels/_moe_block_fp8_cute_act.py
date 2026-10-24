# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""One-launch BF16 SiLU/mul and 1x128 E4M3 quantization.

Four warps per CTA, one group per warp, four adjacent columns per lane. R is
symbolic, so decode and prefill share a callable cached by (I, architecture).
SiLU is SGLang's g / (1 + expf(-g)): accurate exponential and div.rn.f32. The
FP32 product is rounded to BF16 exactly once.

Quantization follows linear.block_fp8: max(abs(h), 1e-10), stored scale
amax * fp32(1/448), multiplier div.rn(448, amax), then multiply, clamp and
E4M3 round-to-nearest-even with saturation. In particular, never invert the
stored scale. NaN propagates through the group; Inf follows the quantizer's
IEEE multiply (finite h * 0 -> zero, Inf * 0 -> NaN).
"""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32
from cutlass.runtime import make_fake_stream

_CACHE = {}
_LOCK = threading.Lock()


@cute.jit
def _divide_rn(a: Float32, b: Float32):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


class SiluMulQuantKernel:
    @cute.jit
    def __call__(
        self, gu: cute.Tensor, q: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream, act=None
    ):
        self.run(gu, q, scales, act).launch(
            grid=(cute.ceil_div(q.shape[0] * scales.shape[1], 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def run(self, gu: cute.Tensor, q: cute.Tensor, scales: cute.Tensor, act):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        group = bid * 4 + tid // 32
        lane = tid % 32
        groups = scales.shape[1]
        if group < q.shape[0] * groups:
            row, col = group // groups, (group % groups) * 128 + lane * 4
            gate = cute.make_rmem_tensor((4,), BFloat16)
            up = cute.make_rmem_tensor((4,), BFloat16)
            h = cute.make_rmem_tensor((4,), Float32)
            packed = cute.make_rmem_tensor((4,), Float8E4M3FN)
            # Contiguous views may have a BF16 storage offset. Keep one compiled
            # signature while vectorizing the normal allocator-aligned case.
            if gu.iterator.toint() % 8 == 0:
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=64
                )
                g = cute.make_tensor(
                    (gu.iterator + gu.layout((row, col))).align(8), cute.make_layout(4)
                )
                u = cute.make_tensor(
                    (gu.iterator + gu.layout((row, col + q.shape[1]))).align(8),
                    cute.make_layout(4),
                )
                cute.copy(atom, g, gate)
                cute.copy(atom, u, up)
            else:
                for j in cutlass.range_constexpr(4):
                    gate[j] = gu[row, col + j]
                    up[j] = gu[row, col + q.shape[1] + j]
            peak, bad = Float32(1e-10), Int32(0)
            for j in cutlass.range_constexpr(4):
                g = Float32(gate[j])
                value = _divide_rn(g, Float32(1) + cute.math.exp(-g, fastmath=False))
                value = Float32((value * Float32(up[j])).to(BFloat16))
                h[j] = value
                peak = cutlass.max(peak, cute.math.abs(value))
                bad |= Int32(value != value)
            if cutlass.const_expr(act is not None):
                for j in cutlass.range_constexpr(4):
                    act[row, col + j] = h[j].to(BFloat16)
            for shift in cutlass.range_constexpr(5):
                peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, offset=1 << shift))
                bad |= cute.arch.shuffle_sync_bfly(bad, offset=1 << shift)
            if bad != 0:
                peak = Float32(float("nan"))
            multiplier = _divide_rn(Float32(448), peak)
            if lane == 0:
                scales[row, group % groups] = peak * Float32(1.0 / 448.0)
            for j in cutlass.range_constexpr(4):
                value = h[j] * multiplier
                bounded = cutlass.min(Float32(448), cutlass.max(Float32(-448), value))
                if value != value:
                    bounded = value
                packed[j] = bounded.to(Float8E4M3FN)
            if q.iterator.toint() % 4 == 0:
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float8E4M3FN, num_bits_per_copy=32
                )
                target = cute.make_tensor(
                    (q.iterator + q.layout((row, col))).align(4), cute.make_layout(4)
                )
                cute.copy(atom, packed, target)
            else:
                for j in cutlass.range_constexpr(4):
                    q[row, col + j] = packed[j]


def compile_kernel(i, *, arch="sm_120a", options="", return_act=False):
    """Offline compile with symbolic R; no device allocation or invocation."""
    if i < 128 or i % 128:
        raise ValueError("I must be a positive multiple of 128")
    fake = cute.runtime.make_fake_tensor
    gu = fake(BFloat16, (cute.sym_int64(), 2 * i), (2 * i, 1), assumed_align=2)
    q = fake(Float8E4M3FN, (cute.sym_int64(), i), (i, 1), assumed_align=1)
    scales = fake(Float32, (cute.sym_int64(), i // 128), (i // 128, 1), assumed_align=4)
    act = fake(BFloat16, (cute.sym_int64(), i), (i, 1), assumed_align=2) if return_act else None
    args = (act,) if return_act else ()
    return cute.compile(
        SiluMulQuantKernel(),
        gu,
        q,
        scales,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        *args,
        options=f"--enable-tvm-ffi --gpu-arch {arch} {options}",
    )


def silu_mul_quant_fp8(gate_up, q_out, scale_out, act_out=None):
    """Write E4M3 [R,I] and FP32 [R,I/128] on the current torch stream.

    Optional BF16 act_out [R,I] receives the rounded value being quantized.
    Eager first use compiles; BlockFP8MoE.prepare warms this outside capture.
    Arguments must be disjoint contiguous inference tensors on one CUDA device.
    """
    from .moe_block_fp8 import _tensor

    _tensor("gate_up", gate_up, torch.bfloat16)
    if gate_up.ndim != 2 or gate_up.shape[1] < 256 or gate_up.shape[1] % 256:
        raise ValueError("gate_up must be [R, 2I] with I a positive multiple of 128")
    r, twice_i = gate_up.shape
    i = twice_i // 2
    _tensor("q_out", q_out, torch.float8_e4m3fn, (r, i))
    _tensor("scale_out", scale_out, torch.float32, (r, i // 128))
    outputs = (q_out, scale_out)
    if act_out is not None:
        _tensor("act_out", act_out, torch.bfloat16, (r, i))
        outputs += (act_out,)
    if not gate_up.is_cuda or any(t.device != gate_up.device for t in outputs):
        raise ValueError("activation tensors must share a CUDA device")
    if not r:
        return
    with torch.cuda.device(gate_up.device):
        major, minor = torch.cuda.get_device_capability(gate_up.device)
        if (major, minor) not in ((12, 0), (12, 1)):
            raise RuntimeError("silu_mul_quant_fp8 requires sm120/121")
        arch = f"sm_{major}{minor}a"
        key = (i, arch) if act_out is None else (i, arch, True)
        if key not in _CACHE:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("silu_mul_quant_fp8: call prepare outside capture")
            with _LOCK:
                if key not in _CACHE:
                    _CACHE[key] = compile_kernel(i, arch=arch, return_act=act_out is not None)
        if act_out is None:
            _CACHE[key](gate_up, q_out, scale_out)
        else:
            _CACHE[key](gate_up, q_out, scale_out, act_out)
