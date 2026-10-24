# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""MoE LoRA backward orchestration (B1..B6, W1..W4) over a steps interface.

The Triton steps run on CUDA; the torch steps accept strided matrices and FP32 inputs
so the decomposition can also be checked without a GPU or the DSL package.
"""

import torch

from ..moe_impl import fused_moe_kernel_triton, moe_sum_reduce_triton
from ._lora_weights import MoELoRAWeights, check_tensor, select_adapter, validate_rank
from .torch._lora_reference import TorchSteps


def training_factors(lora, x, w13, w2, *, shared=False, adapter_id=0):
    """Routed: (gate_up_a, gate_up_b, down_a, down_b); shared: (as13, bs13, as2, bs2)."""
    if lora is None:
        return (None,) * 4
    if isinstance(lora, MoELoRAWeights):
        factors = lora.shared if shared else lora.leaves
    else:
        factors = tuple(lora)
    if factors is None or len(factors) != 4:
        raise ValueError("provide 4 LoRA factors")
    dims = (3, 3, 2, 2) if shared else (3, 4, 3, 3)
    factors = tuple(select_adapter(t, d, adapter_id) for t, d in zip(factors, dims, strict=True))
    rank = validate_rank(factors[0].shape[-2])
    h, i = x.shape[1], w2.shape[-1]
    if shared:
        shapes = ((2, rank, h), (2, i, rank), (rank, i), (h, rank))
    else:
        e = w13.shape[0]
        shapes = ((e, rank, h), (2, e, i, rank), (e, rank, i), (e, h, rank))
    dtype = getattr(factors[0], "dtype", None)
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("LoRA factors must be BF16 or FP32 (masters) tensors")
    for tensor, shape in zip(factors, shapes, strict=True):
        check_tensor(tensor, "LoRA factor", dtype=dtype, device=x.device)
        if tensor.shape != shape:
            raise ValueError(f"LoRA factor must have shape {shape}")
    return factors


def check_routed(lora, x, w13, w2):
    """Routed factor shapes against the base, either layout (adapter already selected)."""
    if lora is None:
        return
    e, h, i = w13.shape[0], x.shape[1], w2.shape[-1]
    r = lora.rank
    shapes = ((e, r, h),) * 2 + ((e, i, r),) * 2 + ((e, r, i), (e, h, r))
    for tensor, shape in zip(lora.routed, shapes, strict=True):
        if tensor.device != x.device or tensor.shape != shape:
            raise ValueError(f"LoRA factor must have shape {shape} on {x.device}")


def _base_backward(inp, weight, ids, rw, state, divisor, *, reduce_choices=False):
    if not inp.is_cuda:
        return TorchSteps.base(inp, weight, ids, row_divisor=divisor)

    shape = (*ids.shape, weight.shape[1]) if reduce_choices else (ids.numel(), 1, weight.shape[1])
    # Only dx enters an unmasked reduction; alignment may skip remote rows.
    allocate = torch.zeros if reduce_choices and state["base_skipped"] else torch.empty
    out = allocate(shape, device=inp.device, dtype=inp.dtype)
    sorted_ids, experts, padded = state["base_alignment"]
    fused_moe_kernel_triton(
        inp,
        weight,
        out,
        rw,
        state["ids"],
        sorted_ids,
        experts,
        padded,
        False,
        divisor,
        state["config"],
        inp.dtype,
    )
    return out if reduce_choices else out[:, 0]


def backward_steps(dy, x, w13, w2, rw, ids, factors, scale, state, needs, *, steps=None):
    """B1..B6 and W1..W4, skipping branches without requested derivatives.

    ``factors`` are (gate_up_a [E,r,H], gate_up_b [2,E,I,r], down_a, down_b): gate and up share
    A, so dz13 is one shrink over [dc1_gate, dc1_up] . [B_gate; B_up] (K = 2I) and dA one outer
    product. ``needs`` = (x, routing weights, *factors); the gradients come back in that order.
    """
    steps = TorchSteps if steps is None else steps
    base = getattr(steps, "frozen_base", _base_backward)  # block-FP8: the steps' dgrads
    nx, nw, na13, nb13, na2, nb2 = needs
    a13, b13, a2, b2 = factors
    if x.shape[0] == 0:
        return tuple(
            torch.zeros_like(t) if need and t is not None else None
            for t, need in zip((x, rw, *factors), needs, strict=True)
        )
    top_k, e = ids.shape[1], w13.shape[0]
    i = w2.shape[-1]
    alignment = state.get("alignment", {})
    written = state.get("local_only", False)
    c1, z13, z2 = (state[k] for k in ("c1", "z13", "z2"))
    dx = dw = da13 = db13 = da2 = db2 = dz13 = None
    need_dc1 = nx or na13 or nb13
    need_act = need_dc1 or nw or na2
    need_dz2 = a2 is not None and (need_act or na2)
    dz2 = None
    if need_dz2:
        b2t = b2.transpose(-1, -2).contiguous()
        dz2 = steps.shrink(dy, b2t, ids, alignment, top_k, fully_written=written)
    dc1 = c2 = None
    if need_act:
        dc2 = base(dy, w2.transpose(1, 2), ids, rw, state, top_k)
        if a2 is not None:
            dc2 = steps.expand(dc2, dz2, a2.transpose(-1, -2).contiguous(), ids, alignment, scale)
        rounding = {"round_silu": True} if state.get("round_silu") else {}
        dc1, dw, c2 = steps.act(
            c1, dc2, rw, ids, e, need_dc1=need_dc1, need_dw=nw, need_c2=na2, **rounding
        )
    outer = lambda left, right, **kw: steps.outer(left, right, ids, e, scale, **alignment, **kw)
    if nb2:
        db2 = outer(dy, z2[:, 0], left_divisor=top_k, pair_scale=rw)
    if na2:
        da2 = outer(dz2, c2, pair_scale=rw)
    if a13 is not None and (nx or na13):
        bt = b13.permute(1, 3, 0, 2).reshape(e, b13.shape[-1], 2 * i).contiguous()
        dz13 = steps.shrink(dc1, bt, ids, alignment, 1, fully_written=written)
    if nb13:
        db13 = steps.outer_buffer(b13, ids)
        for s in range(2):
            outer(dc1[:, s * i : (s + 1) * i], z13[:, 0], out=db13[s])
    if na13:
        da13 = outer(dz13, x, right_divisor=top_k, out=steps.outer_buffer(a13, ids))
    if nx:
        dx_rows = base(
            dc1, w13.transpose(1, 2), ids, rw, state, 1, reduce_choices=x.is_cuda
        )  # fmt: skip
        dxp = dx_rows.view(-1, x.shape[1])
        if a13 is not None:
            dxp = steps.expand(dxp, dz13, a13.transpose(-1, -2).contiguous(), ids, alignment, scale)
        if x.is_cuda:
            dx = torch.empty_like(x)
            moe_sum_reduce_triton(dx_rows, dx)
        else:
            valid = ((ids >= 0) & (ids < e)).reshape(-1, 1)
            dx = torch.where(valid, dxp, 0).reshape(x.shape[0], top_k, x.shape[1]).float().sum(1)
    values = (dx, None if dw is None else dw.view_as(rw), da13, db13, da2, db2)
    return tuple(
        None if value is None else value.to(t.dtype)
        for value, t in zip(values, (x, rw, *factors), strict=True)
    )
