# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own frozen MoE-block dX tree p2c (2026-10-11); expert-parallel masks
# and the shared expert's own intermediate size added here.
"""Frozen MoE block, differentiable only in its input."""

from __future__ import annotations

import os
from importlib import import_module

import torch

_WEIGHT_NAMES = (
    "router_weight",
    "shared_gate_weight",
    "experts_w13",
    "experts_w2",
    "shared_w13",
    "shared_w2",
)
# Capability -> default GEMM backend. CuTe is qualified on sm_120/sm_121 (RTX 5090,
# RTX PRO 6000, GB10); Triton remains the frozen fallback elsewhere and for unsupported shapes.
_AUTO = {(9, 0): "triton", (10, 3): "triton", (12, 0): "cute", (12, 1): "cute"}


def _provider_for(device, ni, nis, h):
    requested = os.environ.get("CTI_MOE_PROVIDER", "auto")
    if requested not in ("auto", "triton", "cute"):
        raise ValueError("moe_block: CTI_MOE_PROVIDER must be auto, triton or cute")
    if device.type != "cuda":
        raise ValueError("moe_block: a supported CUDA device is required")
    name = requested
    if name == "auto":
        capability = torch.cuda.get_device_capability(device)
        name = _AUTO.get(capability)
        if name is None:
            raise ValueError(f"moe_block: unsupported CUDA device capability {capability}")
    try:
        provider = import_module(f".{name}._provider", __package__)
    except ImportError as exc:
        if name == "cute":
            raise ImportError(
                "moe_block: CTI_MOE_PROVIDER selected cute, but its backend could not be imported"
            ) from exc
        raise
    # CuTe serves sm_120/sm_121 and H, I, Is multiples of 64; other shapes use Triton.
    if name == "cute" and (
        torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1))
        or not provider.supports(h, ni, nis)
    ):
        provider = import_module(".triton._provider", __package__)
    backend = provider.backend(device, ni, nis, h)
    return backend


def _check(x, weights, top_k, expert_map, routing, valid_tokens, chunk_tokens):
    # Frozen checks precede even shape validation and provider loading, also under no_grad.
    for name, w in weights.items():
        if w is not None and w.requires_grad:
            raise ValueError(f"moe_block: {name} must be frozen (requires_grad=False)")
    r, sg, w13, w2, s13, s2 = (weights[n] for n in _WEIGHT_NAMES)
    shared = (sg, s13, s2)
    if any(w is None for w in shared) and not all(w is None for w in shared):
        raise ValueError(
            "moe_block: shared_gate_weight, shared_w13 and shared_w2 must all be None or all be tensors"
        )
    for name, w, ndim in (
        ("x", x, 2),
        ("router_weight", r, 2),
        ("experts_w13", w13, 3),
        ("experts_w2", w2, 3),
        ("shared_w2", s2, 2),
    ):
        if name == "shared_w2" and w is None:
            continue
        if w is None or w.ndim != ndim:
            raise ValueError(f"moe_block: {name} must have {ndim} dimensions")
    t, h = x.shape
    e, el, ni = r.shape[0], w13.shape[0], w2.shape[2]
    nis = s2.shape[1] if s2 is not None else ni
    shapes = ((e, h), (1, h), (el, 2 * ni, h), (el, h, ni), (2 * nis, h), (h, nis))
    for (name, w), shape in zip(weights.items(), shapes, strict=True):
        if w is None:
            continue
        if tuple(w.shape) != shape or w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise ValueError(f"moe_block: {name} must be contiguous BF16 {shape}")
        if w.device != x.device:
            raise ValueError(f"moe_block: {name} must be on x's device")
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("moe_block: x must be contiguous BF16")
    for name, d in (("H", h), ("I", ni), ("Is", nis)):
        b = min(64, d)
        if b < 16 or b & (b - 1) or d % b:
            raise ValueError(f"moe_block: {name} = {d} must be a multiple of 64, or 16 or 32")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 < top_k <= e <= 256:
        raise ValueError("moe_block: need integer 0 < top_k <= E <= 256 (uint8 record)")
    if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool) or chunk_tokens <= 0:
        raise ValueError("moe_block: chunk_tokens must be a positive integer")
    if expert_map is None:
        if el != e:
            raise ValueError("moe_block: E_local must equal E without expert_map")
    elif (
        expert_map.shape != (e,)
        or expert_map.dtype != torch.int32
        or expert_map.device != x.device
        or not expert_map.is_contiguous()
    ):
        raise ValueError("moe_block: expert_map must be contiguous int32 [E] on x's device")
    if routing is not None and (
        routing.shape != (t, top_k)
        or routing.dtype != torch.uint8
        or routing.device != x.device
        or not routing.is_contiguous()
    ):
        raise ValueError("moe_block: routing must be contiguous uint8 [T, top_k] on x's device")
    if valid_tokens is not None and (
        valid_tokens.numel() != 1
        or valid_tokens.ndim > 1
        or valid_tokens.dtype != torch.int32
        or valid_tokens.device != x.device
    ):
        raise ValueError("moe_block: valid_tokens must be an int32 scalar tensor on x's device")


def moe_block(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor | None,
    experts_w13: torch.Tensor,
    experts_w2: torch.Tensor,
    shared_w13: torch.Tensor | None,
    shared_w2: torch.Tensor | None,
    *,
    top_k: int,
    expert_map: torch.Tensor | None = None,
    routing: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
    save_activations: bool = False,
    chunk_tokens: int = 8192,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 output [T,H] and GLOBAL expert ids [T,top_k] uint8.

    Contiguous BF16 CUDA inputs: x [T,H], router [E,H], shared gate [1,H],
    experts_w13 [E_local,2I,H], experts_w2 [E_local,H,I], shared_w13 [2Is,H],
    shared_w2 [H,Is]. H, I and Is must be multiples of 64, or 16 or 32; E <= 256.
    All three shared weights may be None together to omit the shared expert
    and shared-gate term (e.g. Qwen3-MoE); there are then exactly top_k slots.
    Weights must be frozen, even in no_grad. Only x is differentiable (first order).
    CTI_MOE_PROVIDER selects auto (default), triton or cute per call. Auto uses
    CuTe on sm_120/121; cute falls back to Triton where unsupported. CuTe results
    are not bitwise equal to Triton; saved/recomputed backward may tile I differently.
    CuTe uses programmatic dependent launch on sm_120/121 to prefetch down/dX
    weights during the preceding GEMM's epilogue; CTI_MOE_CUTE_PDL=0 disables it.
    Warm up each PDL setting before CUDA graph capture.
    CuTe layout is on by default; CTI_MOE_CUTE_LAYOUT=0 restores the torch
    path. Atomic placement reorders rows within experts without changing row math.
    CuTe also runs the backward token kernel (CTI_MOE_CUTE_TOKEN=0 keeps Triton's).
    chunk_tokens defaults to 8192; each chunk rereads local expert weights.

    Numerics follow Unsloth's grouped-GEMM Qwen3-MoE training path: BF16 router
    logits, FP32 softmax over all E, top-k, FP32 renormalisation, BF16 route
    weights; BF16 SiLU, BF16 SiLU*up, BF16 unweighted down outputs; per slot
    bf16(w*y) summed in a fixed order into four FP32 partials, rounded once;
    shared output scaled by the BF16 sigmoid gate and added in BF16. No atomics,
    so results are run-to-run bitwise. The backward differentiates through those
    rounded values. Replay uses current logits. norm_topk_prob=False weights the
    selected experts by the unrenormalised softmax; the router derivative then
    reaches every expert's router row: dz_i = p_i (dw_i [i selected] - sum_j p_j dw_j).
    The unselected-expert term is a [T,E]x[E,H] FP32 torch matmul added after
    the token kernel's BF16 rounding (one extra rounding).
    valid_tokens is a device int32 scalar, clamped to [0,T]; later rows are zero
    in output, record and dX, including when their input or upstream dY is NaN.

    expert_map is contiguous device int32 [E], global id -> local row or -1.
    Values outside [-1,E_local) are undefined behaviour; values are not read
    back for validation. Replay ids must be distinct and in [0,E).
    The caller supplies a shared I-shard (gate rows then matching up rows,
    corresponding down columns). Is is independent of I.

    Output and dX are rank partials: the caller all-reduces output at the MoE
    exit and dX through its input-side collective. All k selected router rows
    remain in dz_j = w_j (dw_j - sum_i w_i dw_i), with remote dw_j = 0.
    Thus each rank contributes sum_local w_j dw_j (R_j - Rbar), where
    Rbar = sum_j w_j R_j; summing ranks recovers the full router derivative.
    The shared-gate derivative is linear in each shard's output as well.
    BF16 rounding means sums of partials need not be bitwise equal to EP1.
    """
    weights = dict(
        zip(
            _WEIGHT_NAMES,
            (router_weight, shared_gate_weight, experts_w13, experts_w2, shared_w13, shared_w2),
            strict=True,
        )
    )
    _check(x, weights, top_k, expert_map, routing, valid_tokens, chunk_tokens)
    if not isinstance(norm_topk_prob, bool):
        raise ValueError("moe_block: norm_topk_prob must be a bool")
    ni = experts_w2.shape[2]
    nis = shared_w2.shape[1] if shared_w2 is not None else ni
    backend = _provider_for(x.device, ni, nis, x.shape[1])
    from ._training_backend import _MoEBlock

    return _MoEBlock.apply(
        x,
        *weights.values(),
        top_k,
        expert_map,
        routing,
        valid_tokens,
        save_activations,
        chunk_tokens,
        backend,
        norm_topk_prob,
    )


def moe_block_training_bytes(
    tokens: int,
    *,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    shared_intermediate: int = 0,
    device: torch.device | str = "cuda",
    chunk_tokens: int = 8192,
    save_activations: bool = False,
    norm_topk_prob: bool = True,
) -> tuple[int, int]:
    """``(saved, backward)`` bytes of ``moe_block`` over ``tokens`` rows with ``experts``
    local experts of width ``intermediate`` and a shared expert of width
    ``shared_intermediate`` (0: none), as the CUDA caching allocator holds them. ``saved`` is
    what the forward keeps for the backward, the input included; ``backward`` the most the
    backward holds at once, the input gradient included (BF16 contiguous output gradient).
    Routing does not enter: slot buffers are sized for every expert's last block partial.
    Exact tensor bytes for ``norm_topk_prob=True``, an upper bound otherwise; the caching
    allocator may hold up to 1 MiB more per allocation of 1 MiB or more (an unsplit block)."""
    device = torch.device(device)
    nis = shared_intermediate or intermediate
    backend = _provider_for(device, intermediate, nis, hidden)
    from ._training_backend import training_bytes

    return training_bytes(
        tokens, hidden, intermediate, nis, experts, top_k, shared_intermediate > 0, backend,
        device, chunk_tokens=chunk_tokens, save_activations=save_activations,
        renormalize=norm_topk_prob,
    )  # fmt: skip


__all__ = ["moe_block", "moe_block_training_bytes"]
