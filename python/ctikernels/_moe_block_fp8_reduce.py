# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Ordered route combine as FlashInfer's finalize (vLLM's default), optionally adding a LoRA delta.

Both MoE providers store unweighted BF16 FC2 rows y in down. A live route's row, with the
optional BF16 LoRA delta added and rounded once (y' = BF16(y + delta), as vLLM adds the delta to
the unweighted down output), enters ``acc = fma(w, y', acc)`` in route order from +0; the sum is
rounded once. Masked rows are never loaded. Eight adjacent BF16 elements per thread give 16-byte
loads/stores.
"""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.runtime import make_fake_stream

_CACHE = {}
_LOCK = threading.Lock()


class MaskedReduceKernel:
    @cute.jit
    def __call__(self, down, ids, out, expert_map, delta, weights, stream: cuda.CUstream):
        self.run(down, ids, out, expert_map, delta, weights).launch(
            grid=(out.shape[0], cute.ceil_div(out.shape[1], 1024), 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def run(self, down, ids, out, expert_map, delta, weights):
        tid, _, _ = cute.arch.thread_idx()
        token, tile, _ = cute.arch.block_idx()
        col = tile * 1024 + tid * 8
        if col < out.shape[1]:
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            )
            total = cute.make_rmem_tensor((8,), Float32)
            value = cute.make_rmem_tensor((8,), BFloat16)
            extra = cute.make_rmem_tensor((8,), BFloat16)
            total.fill(0)
            for slot in cutlass.range_constexpr(ids.shape[1]):
                expert = ids[token, slot]
                live = expert >= 0
                if cutlass.const_expr(expert_map is not None):
                    if live:
                        live = expert_map[expert] >= 0
                if live:
                    route = token * ids.shape[1] + slot
                    source = cute.make_tensor(
                        (down.iterator + down.layout((route, col))).align(16),
                        cute.make_layout(8),
                    )
                    cute.copy(atom, source, value)
                    if cutlass.const_expr(delta is not None):
                        source_delta = cute.make_tensor(
                            (delta.iterator + delta.layout((route, col))).align(16),
                            cute.make_layout(8),
                        )
                        cute.copy(atom, source_delta, extra)
                    weight = weights[token, slot]
                    for j in cutlass.range_constexpr(8):
                        v = Float32(value[j])
                        if cutlass.const_expr(delta is not None):
                            v = Float32((v + Float32(extra[j])).to(BFloat16))
                        # One explicit FMA per route keeps the chain's single roundings.
                        total[j] = cute.arch.inline_ptx(
                            "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                            write_only_types=[Float32],
                            read_only_args=[weight, v, total[j]],
                        )
            for j in cutlass.range_constexpr(8):
                value[j] = total[j].to(BFloat16)
            target = cute.make_tensor(
                (out.iterator + out.layout((token, col))).align(16), cute.make_layout(8)
            )
            cute.copy(atom, value, target)


def compile_kernel(h, top_k, *, has_map=False, has_delta=False, arch="sm_120a", options=""):
    """Compile symbolic M without allocating device tensors or launching."""
    if h < 128 or h % 128 or top_k < 1:
        raise ValueError("H must be a positive multiple of 128 and top_k positive")
    fake = cute.runtime.make_fake_tensor
    down = fake(BFloat16, (cute.sym_int64(), h), (h, 1), assumed_align=16)
    ids = fake(Int32, (cute.sym_int64(), top_k), (top_k, 1), assumed_align=4)
    out = fake(BFloat16, (cute.sym_int64(), h), (h, 1), assumed_align=16)
    mapping = fake(Int32, (cute.sym_int64(),), (1,), assumed_align=4) if has_map else None
    delta = fake(BFloat16, (cute.sym_int64(), h), (h, 1), assumed_align=16) if has_delta else None
    weights = fake(Float32, (cute.sym_int64(), top_k), (top_k, 1), assumed_align=4)
    return cute.compile(
        MaskedReduceKernel(),
        down,
        ids,
        out,
        mapping,
        delta,
        weights,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch} {options}",
    )


def masked_reduce(down, ids, out, weights, *, expert_map=None, delta=None):
    """Combine on the current torch stream; inputs and out must be disjoint.

    IDs in [-1,E) and finite routing weights are caller preconditions. No value
    validation or host synchronization is performed. First eager use compiles.
    """
    from .moe_block_fp8 import _tensor

    _tensor("out", out, torch.bfloat16)
    if out.ndim != 2 or out.shape[1] < 128 or out.shape[1] % 128:
        raise ValueError("out must be [M,H] with H a positive multiple of 128")
    m, h = out.shape
    _tensor("ids", ids, torch.int32)
    if ids.ndim != 2 or ids.shape[0] != m or ids.shape[1] < 1:
        raise ValueError("ids must be [M,top_k] with top_k positive")
    top_k = ids.shape[1]
    _tensor("down", down, torch.bfloat16, (m * top_k, h))
    _tensor("weights", weights, torch.float32, ids.shape)
    tensors = [down, ids, out, weights]
    aligned = [down, out]
    if expert_map is not None:
        _tensor("expert_map", expert_map, torch.int32)
        if expert_map.ndim != 1 or not expert_map.numel():
            raise ValueError("expert_map must be a nonempty vector")
        tensors.append(expert_map)
    if delta is not None:
        _tensor("delta", delta, torch.bfloat16, down.shape)
        tensors.append(delta)
        aligned.append(delta)
    if any(t.data_ptr() % 16 for t in aligned):
        raise ValueError("down, out and delta must be 16-byte aligned")
    if not down.is_cuda or any(t.device != down.device for t in tensors):
        raise ValueError("reduction tensors must share a CUDA device")
    if not m:
        return
    with torch.cuda.device(down.device):
        major, minor = torch.cuda.get_device_capability(down.device)
        if (major, minor) not in ((12, 0), (12, 1)):
            raise RuntimeError("masked_reduce requires sm120/121")
        arch = f"sm_{major}{minor}a"
        key = (h, top_k, expert_map is not None, delta is not None, arch)
        if key not in _CACHE:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("masked_reduce: call prepare outside capture")
            with _LOCK:
                if key not in _CACHE:
                    _CACHE[key] = compile_kernel(
                        h,
                        top_k,
                        has_map=expert_map is not None,
                        has_delta=delta is not None,
                        arch=arch,
                    )
        _CACHE[key](down, ids, out, expert_map, delta, weights)
