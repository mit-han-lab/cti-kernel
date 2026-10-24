# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen block-FP8 expert dX and router dW: the provider-independent part (validation, the
served forward cache) and the dispatcher. CuTe (``_moe_block_fp8_dx_gemm``, sm120/sm121, CuTe DSL)
and our Triton (``triton.fused_moe.block_fp8_moe_dx``, compute capability 8.9+) providers
are optional imports: either may be absent from an installation."""

import torch

from .linear.block_fp8_gated import silu_and_mul_quantize_fp8
from .moe_impl import moe_align_block_size, moe_align_block_size_bytes

try:
    from . import _moe_block_fp8_dx_gemm as _cute
except ImportError:  # no (or an older) CuTe DSL
    _cute = None
try:
    from .triton import fused_moe as _triton
except ImportError:  # no Triton (CPU-only installs)
    _triton = None


def _config(op, m):
    if op.provider_for(m) != "triton-fp8":
        raise ValueError("block_fp8_moe_dx requires a triton-fp8 forward (BF16 gate_up contract)")
    return op.config_for(m)


def _quantize_activation(gu, act_kind):
    """Use the served activation path, shared with the accuracy oracle."""
    from .activation.inference import silu_and_mul
    from .linear.block_fp8 import quantize_per_token_group_fp8

    r, twice_i = gu.shape
    if act_kind == "triton-silu-quant":
        return silu_and_mul_quantize_fp8(gu)
    if act_kind == "cute-silu-quant":
        from ._moe_block_fp8_cute_act import silu_mul_quant_fp8

        q = torch.empty((r, twice_i // 2), dtype=torch.float8_e4m3fn, device=gu.device)
        qs = torch.empty((r, twice_i // 256), dtype=torch.float32, device=gu.device)
        silu_mul_quant_fp8(gu, q, qs)
        return q, qs
    act = torch.empty((r, twice_i // 2), dtype=gu.dtype, device=gu.device)
    silu_and_mul(gu, act)
    return quantize_per_token_group_fp8(act)


def _forward_cache(op, x, weights, ids, config, saved):
    """Repeat served FC1/activation; return gate/up, FP8 q/scales and route layout."""
    from .linear.block_fp8 import quantize_per_token_group_fp8
    from .moe_impl import fused_moe_kernel_triton, moe_align_block_size

    fc1, fc2 = (c.copy() for c in config)
    bm = fc1["BLOCK_SIZE_M"]
    route_blocks = fc1.pop("route_blocks", ids.numel() <= 64 and bm == 16)
    if fc2.pop("route_blocks", route_blocks) != route_blocks or fc2["BLOCK_SIZE_M"] != bm:
        raise ValueError("FC1 and FC2 must share the route layout")
    act_kind = fc2.pop("act", "split")
    layout = (
        op._route_layout(ids, bm)
        if route_blocks
        else moe_align_block_size(ids, bm, op.num_global_experts)
    )
    m, r = x.shape[0], ids.numel()
    gu = saved
    if gu is None:
        gu = torch.empty((m, op.top_k, 2 * op.i), dtype=x.dtype, device=x.device)
        q, qs = quantize_per_token_group_fp8(x)
        fused_moe_kernel_triton(
            q,
            op.w13,
            gu,
            weights,
            ids,
            *layout,
            False,
            op.top_k,
            fc1,
            x.dtype,
            a_scale=qs,
            b_scale=op.s13,
            block_fp8_provider="triton-fp8",
            expert_map=op.expert_map,
            num_global_experts=op.num_global_experts,
        )
    gu = gu.view(r, 2 * op.i)
    q, qs = _quantize_activation(gu, act_kind)
    return gu, (q, qs), layout if _cute is not None and bm == _cute.BM else None


def _validate(op, x, weights, ids, dy, saved):
    from .moe_block_fp8 import _check_inputs, _require_bf16_api, _tensor

    _require_bf16_api("dx()", x)
    _check_inputs(x, weights, ids, device=op.w13.device, hidden=op.h, top_k=op.top_k)
    _tensor("dy", dy, torch.bfloat16, x.shape)
    if dy.device != x.device:
        raise ValueError("dy must share x's device")
    if isinstance(saved, dict):
        if set(saved) != {"gate_up"}:
            raise ValueError("saved must contain only gate_up")
        saved = saved["gate_up"]
    if saved is not None:
        _tensor("saved gate_up", saved, torch.bfloat16)
        if tuple(saved.shape) not in ((x.shape[0], op.top_k, 2 * op.i), (ids.numel(), 2 * op.i)):
            raise ValueError("saved gate_up must be [M, top_k, 2I] or [M*top_k, 2I]")
        if saved.device != x.device:
            raise ValueError("saved gate_up must share x's device")
    return saved


def _held(nbytes):
    # The CUDA caching allocator hands out multiples of 512 bytes.
    return -(-nbytes // 512) * 512


def training_bytes(op, m):
    """``BlockFP8MoE.training_bytes``: ``(saved, backward)`` of the forward and ``dx`` here."""
    if m == 0:
        return 0, 0
    h, i, r = op.h, op.i, m * op.top_k
    saved = _held(2 * m * h) + 2 * _held(4 * r)  # x, FP32 weights, int32 ids

    def aligned(block):
        return sum(_held(n) for n in moe_align_block_size_bytes(r, block, op.num_global_experts))

    # Phase 1 recomputes the forward cache: its layout (route blocks are cached on the op),
    # gate/up, x's quantization and the FP8 activation. Phase 2 drops x's quantization and
    # adds the GEMM tail on dx's layout: CuTe reuses the forward's when its block matches;
    # Triton aligns by its own block and keeps the forward's alongside in that case.
    fc1, _ = _config(op, m)
    bm = fc1["BLOCK_SIZE_M"]
    forward = 0 if fc1.get("route_blocks", r <= 64 and bm == 16) else aligned(bm)
    reused = forward if _cute is not None and bm == _cute.BM else None
    if op.device_class[:2] in ((12, 0), (12, 1)) and _cute is not None:
        layout = aligned(_cute.BM) if reused is None else reused
    else:
        layout = (reused or 0) + aligned(_triton.DX_BLOCK_M)
    gate_up = _held(4 * r * i)
    act = _held(r * i) + _held(4 * r * (i // 128))  # FP8 activation and scales
    x_quant = _held(m * h) + _held(4 * m * (h // 128))
    tail = gate_up + _held(4 * r * h) + _held(4 * r * (i // 64))  # dgdu, route dX, dW partials
    tail += _held(2 * m * h) + _held(4 * r)  # dX, dW
    return saved, max(forward + gate_up + act + x_quant, layout + gate_up + act + tail)


DX_PROVIDERS = ("cute", "triton")


def block_fp8_moe_dx(op, x, topk_weights, topk_ids, dy, *, saved=None, provider=None):
    """Return BF16 dX [M,H], FP32 router dW [M,top_k]; frozen weights/scales.

    ``provider``: "cute" (sm120/sm121 only), "triton" (our Triton chain, every compute
    capability 8.9+) or None: CuTe on sm120/sm121, Triton elsewhere.

    saved is a contiguous BF16 gate_up tensor or {"gate_up": tensor} from
    the same triton-fp8 forward, input, routes, weights and config. It is
    caller-owned and must remain unchanged through capture/replay.
    """
    if provider not in (None, *DX_PROVIDERS):
        raise ValueError(f"block_fp8_moe_dx: provider must be one of {DX_PROVIDERS} or None")
    saved = _validate(op, x, topk_weights, topk_ids, dy, saved)
    m = x.shape[0]
    if not x.is_cuda:
        raise ValueError("block_fp8_moe_dx requires CUDA inputs")
    if not m:
        return torch.empty_like(x), torch.empty_like(topk_weights)
    config = _config(op, m)
    key = (m, tuple(tuple(sorted(c.items())) for c in config))
    with torch.cuda.device(x.device):
        cc = torch.cuda.get_device_capability(x.device)
        sm12x = cc in ((12, 0), (12, 1))
        if provider == "cute" and not sm12x:
            raise RuntimeError("block_fp8_moe_dx: the CuTe provider requires sm120/121")
        if provider == "triton" or not sm12x or _cute is None:
            if provider == "cute":
                raise RuntimeError("block_fp8_moe_dx: the CuTe provider needs CuTe DSL 4.8.0")
            # Our Triton provider (compute capability 8.9+).
            gu, (q, qs), _ = _forward_cache(op, x, topk_weights, topk_ids, config, saved)
            layout = moe_align_block_size(topk_ids, _triton.DX_BLOCK_M, op.num_global_experts)
            return _triton.block_fp8_moe_dx(op, dy, topk_weights, topk_ids, gu, q, qs, layout)
        gu, (q, qs), layout = _forward_cache(op, x, topk_weights, topk_ids, config, saved)
        if layout is None:
            layout = moe_align_block_size(topk_ids, _cute.BM, op.num_global_experts)
        return _cute.cute_dx(op, x, topk_weights, topk_ids, dy, gu, q, qs, layout, key, cc)


def prepare_dx(op, batch_sizes, *, provider=None):
    """Compile and warm every requested complete chain before graph capture."""
    with torch.cuda.device(op.w13.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_dx must run outside capture")
        for m in batch_sizes:
            if not isinstance(m, int) or m < 0:
                raise ValueError("batch sizes must be nonnegative integers")
            x = torch.zeros((m, op.h), dtype=torch.bfloat16, device=op.w13.device)
            weights = torch.ones((m, op.top_k), dtype=torch.float32, device=x.device)
            ids = torch.zeros((m, op.top_k), dtype=torch.int32, device=x.device)
            block_fp8_moe_dx(op, x, weights, ids, x, provider=provider)
