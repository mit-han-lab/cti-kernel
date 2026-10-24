# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton MoE LoRA backward steps (the ``TorchSteps`` interface of ``moe._lora_backward``).

Deterministic: rows are grouped by a stable expert sort (``align_block_size``), every output
row or factor-gradient tile is owned by one program, and every sum runs in a fixed order (no
atomics), so gradients are bitwise equal run to run. MMA operands are BF16 with FP32
accumulation; row gradients are kept in FP32 between steps.
"""

import torch
import triton
import triton.language as tl

from ...moe_impl import moe_align_block_size
from ...triton.fused_moe import DX_BLOCK_M, block_fp8_moe_dgrad
from ...triton.silu import sigmoid, silu
from .._lora_backward import _base_backward
from . import _lora_kernels as K
from ._slot_layout import align_block_size

_BM = 32


def layout(ids, experts):
    """Alignment dict for the steps: expert-sorted pair slots in increasing pair order."""
    # Ids outside [0, experts) get no slot; unused rows hold the sentinel ids.numel() and
    # blocks past the last expert have expert -1.
    sorted_ids, block_experts, _ = align_block_size(ids, _BM, experts)
    ordered = torch.where(block_experts < 0, experts, block_experts)
    starts = torch.searchsorted(
        ordered, torch.arange(experts + 1, device=ids.device, dtype=ordered.dtype)
    ).to(torch.int32)
    npad = torch.full((1,), sorted_ids.numel(), dtype=torch.int32, device=ids.device)
    return {"sorted": sorted_ids, "block_experts": block_experts, "npad": npad, "starts": starts}


def _bf16(t):
    return t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)


@triton.jit
def _act_backward_kernel(
    c1_ptr, dc2_ptr, rw_ptr, ids_ptr, dc1_ptr, dw_ptr, c2_ptr, E, INTER, c1_stride, dc2_stride,
    NEED_DC1: tl.constexpr, NEED_DW: tl.constexpr, NEED_C2: tl.constexpr, HAS_RW: tl.constexpr,
    ROUND_SILU: tl.constexpr, BI: tl.constexpr,
):  # fmt: skip
    """One pair row: weighted SwiGLU backward, routing-weight gradient sum(dc2 * c2).
    ROUND_SILU follows Unsloth's forward, c2 = BF16(BF16(silu(g)) * u), through its roundings."""
    p = tl.program_id(0).to(tl.int64)
    e = tl.load(ids_ptr + p).to(tl.int64)
    live = (e >= 0) & (e < E)
    w = tl.load(rw_ptr + p) if HAS_RW else 1.0
    dw = tl.zeros((BI,), dtype=tl.float32)
    for i0 in range(0, INTER, BI):
        offs = i0 + tl.arange(0, BI)
        m = (offs < INTER) & live
        g = tl.load(c1_ptr + p * c1_stride + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(c1_ptr + p * c1_stride + INTER + offs, mask=m, other=0.0).to(tl.float32)
        d = tl.load(dc2_ptr + p * dc2_stride + offs, mask=m, other=0.0).to(tl.float32)
        # torch's F.silu and its backward (sigmoid(g) (1 + g (1 - sigmoid(g)))), bit for bit.
        sig = sigmoid(g)
        act = silu(g)
        dsilu = sig * (1.0 + g * (1.0 - sig))
        if ROUND_SILU:
            act = act.to(tl.bfloat16).to(tl.float32)
            c2 = (act * u).to(tl.bfloat16).to(tl.float32)
        else:
            c2 = act * u
        if NEED_DW:
            dw += d * c2
        if NEED_C2:
            tl.store(c2_ptr + p * INTER + offs, c2, mask=offs < INTER)
        if NEED_DC1:
            wd = d * w
            tl.store(
                dc1_ptr + p * 2 * INTER + offs,
                wd * u * dsilu,
                mask=offs < INTER,
            )
            tl.store(dc1_ptr + p * 2 * INTER + INTER + offs, wd * act, mask=offs < INTER)
    if NEED_DW:
        tl.store(dw_ptr + p, tl.sum(dw, axis=0))


@triton.jit
def _outer_kernel(
    left_ptr, right_ptr, ps_ptr, scale_ptr, out_ptr, sorted_ptr, starts_ptr, P, L, R,
    l_stride, r_stride, L_DIV: tl.constexpr, R_DIV: tl.constexpr, HAS_PS: tl.constexpr,
    BM: tl.constexpr, BL: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """out[e] = scale * sum over e's pairs (fixed order) of (ps * left[p // L_DIV]) x right[p // R_DIV]."""
    e = tl.program_id(0)
    offs_l = tl.program_id(1) * BL + tl.arange(0, BL)
    offs_r = tl.program_id(2) * BR + tl.arange(0, BR)
    first = tl.load(starts_ptr + e)
    last = tl.load(starts_ptr + e + 1)
    acc = tl.zeros((BL, BR), dtype=tl.float32)
    for b in range(first, last):
        slot = tl.load(sorted_ptr + b * BM + tl.arange(0, BM))
        live = slot < P
        slot = tl.where(live, slot, 0).to(tl.int64)
        lt = tl.load(
            left_ptr + (slot // L_DIV)[:, None] * l_stride + offs_l[None, :],
            mask=live[:, None] & (offs_l[None, :] < L), other=0.0,
        ).to(tl.float32)  # fmt: skip
        if HAS_PS:
            lt = lt * tl.load(ps_ptr + slot, mask=live, other=0.0)[:, None]
        rt = tl.load(
            right_ptr + (slot // R_DIV)[:, None] * r_stride + offs_r[None, :],
            mask=live[:, None] & (offs_r[None, :] < R), other=0.0,
        )  # fmt: skip
        acc = tl.dot(tl.trans(lt.to(tl.bfloat16)), rt.to(tl.bfloat16), acc)
    acc = acc * tl.load(scale_ptr)
    dst = out_ptr + e.to(tl.int64) * L * R + offs_l[:, None] * R + offs_r[None, :]
    tl.store(dst, acc, mask=(offs_l[:, None] < L) & (offs_r[None, :] < R))


class TritonSteps:
    """The backward steps on the Triton kernels; ``alignment`` is :func:`layout`'s dict.

    dc1 comes back in the cache dtype (the base dX GEMM's operand), like the CuTe steps."""

    @staticmethod
    def frozen_base(inp, weight, ids, rw, state, divisor, *, reduce_choices=False):
        """Frozen-base dgrad rows; block-FP8 bases (STE through the activation quantizers) go
        through the grouped FP8 dgrad, BF16 ones through the base grouped GEMM. FP32 row
        gradients (BF16 bases keep FC1 rows in FP32) round once into the BF16 GEMM operand."""
        inp = _bf16(inp)
        if "w13_scale" not in state:
            return _base_backward(
                inp, weight, ids, rw, state, divisor, reduce_choices=reduce_choices
            )
        routes = state["ids"]
        scale = state["w13_scale"] if reduce_choices else state["w2_scale"]
        layout = moe_align_block_size(routes, DX_BLOCK_M, weight.shape[0])
        k = weight.shape[1]
        # Remote rows are not written: zero them where a reduction will read them.
        out = (torch.zeros if reduce_choices else torch.empty)(
            (routes.numel(), k), dtype=inp.dtype, device=inp.device
        )
        block_fp8_moe_dgrad(
            inp, weight.transpose(1, 2), scale, routes, layout, row_divisor=divisor, out=out
        )
        return out.view(*routes.shape, k) if reduce_choices else out

    @staticmethod
    def shrink(x, a, ids, alignment, divisor, **_):
        """[P, r] FP32 rows x[p // divisor] . a[e]^T (zero for invalid pairs)."""
        p, (e, r, k) = ids.numel(), a.shape
        z = torch.zeros((p, r), dtype=torch.float32, device=x.device)
        if p == 0:
            return z
        x = _bf16(x) if x.stride(-1) == 1 else _bf16(x.contiguous())
        a = _bf16(a).contiguous()
        sorted_ids = alignment["sorted"]
        blocks = sorted_ids.numel() // _BM
        br = 64
        K.shrink_kernel[(blocks, triton.cdiv(r, br), 1)](
            x, a, z, sorted_ids, alignment["block_experts"], alignment["npad"], sorted_ids.numel(), p,
            e, k, 0, x.stride(0), k, ROW_DIV=divisor, S=1, R=r, BM=_BM, BR=br, BK=64, KS=1,
            num_warps=4, num_stages=2,
        )  # fmt: skip
        return z

    @staticmethod
    def expand(out, z, b, ids, alignment, scale):
        """out += scale * z . b[e]^T on valid rows, in place (rows contiguous); returns out."""
        result = out
        if result.stride(-1) != 1 or (result.shape[0] > 1 and result.stride(0) != result.shape[1]):
            raise ValueError("expand needs contiguous output rows")
        p, (e, n, r) = ids.numel(), b.shape
        if p == 0:
            return result
        b = _bf16(b).contiguous()
        sorted_ids = alignment["sorted"]
        bn, br = 64, 64
        K.expand_kernel[(sorted_ids.numel() // _BM, triton.cdiv(n, bn))](
            z.contiguous(), b, result, result, scale, scale, sorted_ids, alignment["block_experts"],
            alignment["npad"], sorted_ids.numel(), p, e, n, 0, result.stride(0), 0, S=1, R=r,
            BM=_BM, BR=br, BN=bn, WEIGHTED=False, ACT=False, KS=1, num_warps=4, num_stages=2,
        )  # fmt: skip
        return result

    @staticmethod
    def act(
        c1, dc2, rw, ids, experts, *, need_dc1=True, need_dw=True, need_c2=True, round_silu=False
    ):
        p, i = c1.shape[0], c1.shape[1] // 2
        device = c1.device
        dc1 = torch.empty((p, 2 * i), dtype=c1.dtype, device=device) if need_dc1 else None
        dw = torch.empty((p,), dtype=torch.float32, device=device) if need_dw else None
        c2 = torch.empty((p, i), dtype=torch.float32, device=device) if need_c2 else None
        if p:
            flat = ids.reshape(-1).contiguous()
            dummy = c1
            _act_backward_kernel[(p,)](
                c1, dc2, rw.reshape(-1) if rw is not None else dummy, flat,
                dc1 if need_dc1 else dummy, dw if need_dw else dummy, c2 if need_c2 else dummy,
                experts, i, c1.stride(0), dc2.stride(0), NEED_DC1=need_dc1, NEED_DW=need_dw,
                NEED_C2=need_c2, HAS_RW=rw is not None, ROUND_SILU=round_silu, BI=256, num_warps=4,
            )  # fmt: skip
        return dc1, dw, c2

    @staticmethod
    def outer(
        left, right, ids, experts, scale, *, left_divisor=1, right_divisor=1, pair_scale=None,
        out=None, **alignment,
    ):  # fmt: skip
        l_cols, r_cols = left.shape[-1], right.shape[-1]
        if out is None:
            out = torch.empty((experts, l_cols, r_cols), dtype=torch.float32, device=left.device)
        out = out.view(experts, l_cols, r_cols)
        if left.stride(-1) != 1:
            left = left.contiguous()
        if right.stride(-1) != 1:
            right = right.contiguous()
        bl, br = 64, 64
        _outer_kernel[(experts, triton.cdiv(l_cols, bl), triton.cdiv(r_cols, br))](
            left, right, pair_scale.reshape(-1) if pair_scale is not None else left, scale, out,
            alignment["sorted"], alignment["starts"], ids.numel(), l_cols, r_cols,
            left.stride(0), right.stride(0), L_DIV=left_divisor, R_DIV=right_divisor,
            HAS_PS=pair_scale is not None, BM=_BM, BL=bl, BR=br, num_warps=4, num_stages=2,
        )  # fmt: skip
        return out

    @staticmethod
    def outer_buffer(factor, ids):
        return torch.empty(factor.shape, dtype=torch.float32, device=factor.device)
