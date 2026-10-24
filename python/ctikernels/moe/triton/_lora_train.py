# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""MoE LoRA training forward (BF16 base) in Unsloth 2026.10.7's order (unsloth_zoo's
grouped-GEMM experts; the shared expert as its compiled HF MLP, ``SharedMLP``), on the Triton
kernels; the routed backward is ``TritonSteps``.

Each LoRA is added as PEFT adds it, BF16(base + BF16(BF16(BF16(x . A^T) . B^T) * scale));
SiLU and SiLU * up are each rounded to BF16; FC2 rows are unweighted BF16; routes combine as
moe_block's (BF16(w * y) per route with BF16 weights, four FP32 partials in a fixed order,
one rounding). No atomics. Inference keeps its own composition (``_lora_ops``).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ...moe_impl import fused_moe_kernel_triton, moe_align_block_size
from ...triton.silu import silu
from .._align import align_allocations
from .._block import expert_tiles
from . import _kernels as MK
from . import _lora_kernels as K
from ._lora_backward import _BM, layout
from ._lora_ops import _routed, project_rounded
from ._slot_layout import align_allocations as layout_allocations


@triton.jit
def _act_kernel(c1_ptr, c2_ptr, INTER, BN: tl.constexpr):
    # c2 = BF16(BF16(silu(g)) * u), torch's F.silu(gate) * up on BF16 tensors, bit for bit.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = offs < INTER
    g = tl.load(c1_ptr + row * 2 * INTER + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(c1_ptr + row * 2 * INTER + INTER + offs, mask=mask, other=0.0).to(tl.float32)
    s = silu(g).to(tl.bfloat16).to(tl.float32)
    tl.store(c2_ptr + row * INTER + offs, (s * u).to(tl.bfloat16), mask=mask)


def routed_forward(x, w13, w2, rw, ids_local, lora, scale, *, local_only, expert_config):
    """(out [T,H] BF16, backward state). Ids outside [0, E)
    (remote routes) add nothing."""
    t, h = x.shape
    e, twice_i, _ = w13.shape
    inter, top_k = twice_i // 2, ids_local.shape[1]
    p, rank = t * top_k, 0 if lora is None else lora.rank
    up, down = expert_tiles(t, e, inter, e, x.device, expert_config)
    if local_only and ids_local.dtype == torch.int32 and ids_local.is_contiguous():
        ids = ids_local
    else:  # remote routes onto the sentinel E: no-work blocks
        ids = torch.empty((t, top_k), dtype=torch.int32, device=x.device)
        K.route_prep_kernel[(triton.cdiv(p, 1024),)](ids_local, ids, p, e, SENTINEL=e, BLOCK=1024)
    aligned = moe_align_block_size(ids, up["BLOCK_SIZE_M"], e)
    aligned_down = aligned
    if down["BLOCK_SIZE_M"] != up["BLOCK_SIZE_M"]:
        aligned_down = moe_align_block_size(ids, down["BLOCK_SIZE_M"], e)
    c1 = torch.empty((p, twice_i), dtype=x.dtype, device=x.device)
    y = torch.empty((p, h), dtype=x.dtype, device=x.device)
    z13 = torch.zeros((p, 1, rank), dtype=x.dtype, device=x.device)  # shared A: one shrink
    z2 = torch.zeros((p, 1, rank), dtype=x.dtype, device=x.device)
    fused_moe_kernel_triton(
        x, w13, c1.view(t, top_k, twice_i), rw, ids, *aligned, False, top_k, up, x.dtype
    )  # fmt: skip
    if lora is not None:
        a13, b13 = _routed(lora, down=False)
        align = (*aligned, up["BLOCK_SIZE_M"])
        project_rounded(c1, x, a13, b13, scale, z13, top_k, align, e, top_k)
    c2 = torch.empty((p, inter), dtype=x.dtype, device=x.device)
    if p:
        _act_kernel[(p, triton.cdiv(inter, 1024))](c1, c2, inter, BN=1024, num_warps=4)
    fused_moe_kernel_triton(
        c2, w2, y.view(t, top_k, h), rw, ids, *aligned_down, False, 1, down, x.dtype
    )  # fmt: skip
    if lora is not None:
        a2, b2 = _routed(lora, down=True)
        align = (*aligned_down, down["BLOCK_SIZE_M"])
        project_rounded(y, c2, a2, b2, scale, z2, 1, align, e, top_k)
    live = ((ids_local >= 0) & (ids_local < e)).reshape(-1)
    inv = torch.where(live, torch.arange(p, dtype=torch.int32, device=x.device), -1)
    weights = rw.to(torch.bfloat16).float().contiguous()
    out = torch.empty_like(x)
    columns = min(256, h & -h)
    MK._combine_kernel[(t, h // columns)](
        y, inv, weights, out, inv, t, h, top_k, False, False, columns, num_warps=4
    )  # fmt: skip
    state = {
        "c1": c1,
        "z13": z13,
        "z2": z2,
        "ids": ids,
        "alignment": layout(ids_local, e),
        "base_alignment": aligned[:3],
        "config": up,
        "base_skipped": True,  # remote rows of the base dX are zeroed, never reduced stale
        "local_only": local_only,
        "round_silu": True,
    }
    return out, state


def training_bytes(tokens, h, i, e, top_k, rank, device, *, local_only, expert_config,
                   fp8_block_m=None):  # fmt: skip
    """``fused_experts_lora_training_bytes``: ``(saved, backward)``. A block-FP8 base
    (``fp8_block_m``, its tile rows) trains on ``_lora_ops``' composition, whose state has
    the same rows (BF16 c1, z13, z2) with its own ids copy and base alignment."""
    held = lambda n: -(-n // 512) * 512
    p = tokens * top_k
    inputs = held(2 * tokens * h) + 2 * held(4 * p)  # x, FP32 routing weights, int32 ids
    if tokens == 0:  # the backward returns zero factor gradients
        return inputs, sum(held(2 * e * rank * n) for n in (h, 2 * i, i, h))
    if fp8_block_m is None:
        up, _ = expert_tiles(tokens, e, i, e, device, expert_config)
        base = align_allocations(
            p, up["BLOCK_SIZE_M"], e, capacity_experts=e + 1, invalid_bucket=True
        )
        routes = 0 if local_only else held(4 * p)  # remote routes remapped
    elif p * 4 <= e:  # per-pair blocks: a cached layout and a copy of the experts
        base, routes = (4 * p,), held(4 * p)
    else:
        base = align_allocations(p, fp8_block_m, e, capacity_experts=e + 1, invalid_bucket=True)
        routes = held(4 * p)  # the ids copy
    saved = (
        inputs
        + routes
        + sum(held(n) for n in base)
        + held(4 * p * i) + 2 * held(2 * p * rank)  # c1, z13, z2
        + sum(held(n) for n in layout_allocations(p, _BM, e)) + held(4 * (e + 1)) + held(4)
    )  # fmt: skip
    # Everything backward_steps keeps lives until its BF16 conversions at the end.
    fp32_factors = held(4 * e * h * rank) + held(4 * e * rank * i) + held(8 * e * i * rank)
    fp32_factors += held(4 * e * rank * h)  # dB2, dA2, dB13, dA13
    backward = (
        (0 if fp8_block_m else held(4 * p))  # BF16-rounded routing weights in FP32
        + held(2 * e * h * rank) + held(4 * e * i * rank)  # B2^T, [B_gate; B_up]^T
        + 2 * held(4 * p * rank)  # dz2, dz13
        + held(2 * p * i) + held(4 * p * i) + held(4 * p) + held(4 * p * i)  # dc2, dc1, dw, c2
        + fp32_factors
        + held(2 * p * h) + held(2 * tokens * h)  # dX rows, dX
        + held(2 * e * rank * h) + held(4 * e * i * rank) + held(2 * e * rank * i)
        + held(2 * e * h * rank)  # the factor gradients in BF16
    )  # fmt: skip
    return saved, backward


def _lora_linear(inp, w, z, b, scale):
    # unsloth_zoo's lora_forward, addmm(xW, z, B^T, alpha=scale): the B product accumulated in
    # FP32, added to the BF16 base and rounded once.
    return (F.linear(inp, w).float() + F.linear(z.float(), b.float()) * scale).to(inp.dtype)


class SharedMLP(torch.autograd.Function):
    """The shared expert's MLP with LoRA as Unsloth 2026.10.7 trains it: HF's MLP with PEFT
    LoRA Linears inside one torch.compile frame (unsloth_compiled_cache), so the rounding points
    are that frame's Inductor kernels (dn_288 / dn_359 output code of that version): forward
    h = BF16(silu(gate) * up) with SiLU in FP32 (one rounding); backward LoRA input grads
    BF16(BF16(g . B) * s), the down input grad summed from its two BF16 GEMMs in FP32 and never
    rounded, fused into the FP32 SiLU * up backward, and dx accumulated by an addmm chain.
    SiLU and sigmoid are torch's accurate FP32 ones, not Inductor's approximate divide."""

    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, a13, b13, a2, b2, scale):
        # Factors (FP32 masters too) are saved as given and cast to x's dtype at each use.
        factors = (a13, b13, a2, b2)
        a13, b13, a2, b2 = (f.to(x.dtype) for f in factors)
        i = w_down.shape[1]
        z_gate, z_up = F.linear(x, a13[0]), F.linear(x, a13[1])
        gate = _lora_linear(x, w_gate_up[:i], z_gate, b13[0], scale)
        up = _lora_linear(x, w_gate_up[i:], z_up, b13[1], scale)
        h = (F.silu(gate.float()) * up.float()).to(x.dtype)
        z_down = F.linear(h, a2)
        ctx.scale = scale
        ctx.save_for_backward(x, w_gate_up, w_down, *factors, gate, up, h, z_gate, z_up,
                              z_down)  # fmt: skip
        return _lora_linear(h, w_down, z_down, b2, scale)

    @staticmethod
    def backward(ctx, g):
        x, w_gate_up, w_down, a13, b13, a2, b2, gate, up, h, z_gate, z_up, z_down = (
            ctx.saved_tensors
        )
        a13, b13, a2, b2 = (f.to(x.dtype) for f in (a13, b13, a2, b2))
        i, scale = w_down.shape[1], ctx.scale
        scaled = lambda t: (t.float() * scale).to(t.dtype)
        dz_down = scaled(g @ b2)
        dh = (dz_down @ a2).float() + (g @ w_down).float()
        sig = torch.sigmoid(gate.float())
        d_up = (dh * F.silu(gate.float())).to(g.dtype)
        d_gate = ((dh * up.float()) * sig * (gate.float() * (1 - sig) + 1)).to(g.dtype)
        dz_gate, dz_up = scaled(d_gate @ b13[0]), scaled(d_up @ b13[1])
        dx = torch.addmm(dz_up @ a13[1], d_up, w_gate_up[i:])
        dx = torch.addmm(torch.addmm(dx, dz_gate, a13[0]), d_gate, w_gate_up[:i])
        da13 = torch.stack((dz_gate.t() @ x, dz_up.t() @ x))
        db13 = torch.stack((scaled(d_gate.t() @ z_gate), scaled(d_up.t() @ z_up)))
        da2, db2 = dz_down.t() @ h, scaled(g.t() @ z_down)
        return dx, None, None, da13, db13, da2, db2, None


def shared_mlp(x, w_gate_up, w_down, factors, scale):
    """The shared expert's MLP as Unsloth trains it (``SharedMLP``); without LoRA, HF's MLP in
    BF16 torch ops. ``factors`` are the shared (as13, bs13, as2, bs2) or all None."""
    if factors[0] is not None:
        return SharedMLP.apply(x, w_gate_up, w_down, *factors, scale)
    i = w_down.shape[1]
    return F.linear(F.silu(F.linear(x, w_gate_up[:i])) * F.linear(x, w_gate_up[i:]), w_down)


def shared_forward(x, w_gate_up, w_down, w_gate, factors, scale):
    """The gated shared expert, BF16(sigmoid(x . w_gate)) * shared_mlp, as the Unsloth-patched
    sparse block computes it; the caller adds it in BF16 after the routed sum."""
    return torch.sigmoid(F.linear(x, w_gate)) * shared_mlp(x, w_gate_up, w_down, factors, scale)
