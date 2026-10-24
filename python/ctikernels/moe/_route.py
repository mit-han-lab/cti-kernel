# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Row-count-invariant MoE routing: BF16 router logits, top-k ids and FP32 weights.

Each logit is the FP32 dot product of one token row and one router row, computed as
ROUTER_SPLIT partial dots over fixed hidden slices (``tl.dot`` with a zero accumulator)
summed in slice order, then rounded once to the input precision (BF16 for the models
here) -- the arithmetic of the block tree's router. Nothing depends on how many rows a launch holds (fixed tiles, no split-K that
varies with M), so a token gets the same experts and weights at M=1 and M=16384, and the
same as inside the fused block. Softmax scoring is vLLM 0.31's topk_softmax
(``ctikernels.triton.topk_softmax``): every expert's probability expf(z - max) * (1 / sum)
(nonfinite ones 0), top-k by probability with ties to the lower expert id, renormalised by
the reciprocal of the selected sum. A power-of-two E up to 512 sums as vLLM's warp does;
other E in this module's fixed order.

Triton: bitwise agreement with the fused Triton block needs the same dot lowering; a CuTe
DSL rewrite is pending with the block.
"""

from __future__ import annotations

import torch

ROUTER_SPLIT = 4  # the block tree's router split (triton_block / triton_block_gb10)
SCORE_GEMM_ROWS = 512  # sqrtsoftplus: cuBLAS gate GEMM above this many rows (GB300 crossover)
SCORE_TILE = (
    1,
    1,
)  # sqrtsoftplus top-k: rows per program, warps (GB300: SGLang's tile, measured best)
_ROUND = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 0}


def _kernels():
    # The Triton kernels, imported on first use (the module loads without Triton).
    from . import _route_kernels

    return _route_kernels


def _cdiv(a, b):
    return -(-a // b)


def _pow2(n):
    return 1 << max(0, n - 1).bit_length()


def _logit_tiles(t, e, bk, itemsize=2, kind=None):
    """(BT, BE, BKC, SPLIT, warps, stages, FLAT). Up to a few thousand rows the slices run
    on their own CTAs (FP32 partials, summed by the top-k kernel) for parallelism; prefill
    tiles reuse operands in one pipelined loop. Every choice gives the same logits (see the
    kernel); the table comes from per-device sweeps against cuBLAS + top-k."""
    bkc = min(128, max(16, _pow2(bk)))
    if itemsize > 2:
        bkc = min(bkc, 32)
    stages = 2 if itemsize > 2 else 3
    if t <= 64:
        return 16, min(_pow2(e), 16), bkc, True, 4, 2 if itemsize > 2 else 4, False
    if t <= 256:
        return 32, min(_pow2(e), 64), bkc, True, 4, stages, False
    if t <= (512 if kind == "gb10" else 2048):
        return 64, min(_pow2(e), 64), bkc, True, 4, stages, False
    bkc = min(bkc, 64)
    return 64, min(_pow2(e), 128), bkc, False, 8, stages, bk % bkc == 0


def route(
    x: torch.Tensor,
    router_w: torch.Tensor,
    top_k: int = 8,
    *,
    renormalize: bool = True,
    valid_tokens: torch.Tensor | None = None,
    selected: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
    scoring: str = "softmax",
    bias: torch.Tensor | None = None,
    scale: float = 1.0,
    batch_invariant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 ``weights`` and int32 global expert ``ids``, both ``[T, top_k]``.

    ``x`` is [T, H] and ``router_w`` [E, H], both BF16, FP16 or FP32 (the logit is rounded
    to that precision), on one CUDA device; the router is frozen. Any H, E and
    1 <= ``top_k`` <= E. With the int32 device scalar ``valid_tokens`` [1], rows at or past
    it get ids -1 and weights 0 (graph padding). Graph-capturable; no host synchronization.

    ``selected`` (contiguous integer [T, top_k], distinct ids in [0, E)) replays a selection:
    the weights are those experts' under the same arithmetic. Without ``renormalize``,
    ``probabilities`` (contiguous FP32 [T, E]) receives every expert's softmax probability
    (0 on padding rows).

    ``scoring="sqrtsoftplus"`` is DeepSeek-V4's router (checkpoint ``config.json``:
    scoring_func sqrtsoftplus, topk_method noaux_tc, norm_topk_prob, routed_scaling_factor):
    the logit stays FP32, s = sqrt(softplus(logit)), experts by s + ``bias`` (FP32 [E],
    selection only) descending with ties to the lower id, weights s over the selected s'
    slot-order sum (+ 1e-20) with ``renormalize``, all times ``scale``. Above SCORE_GEMM_ROWS
    rows the gate GEMM runs in cuBLAS (its algorithm may change with the row count);
    ``batch_invariant=True`` keeps the fused, row-count-invariant logits at every row count.
    Softmax scoring is row-count invariant always.
    """
    if router_w.requires_grad:
        raise ValueError("the router must be frozen (requires_grad=False)")
    if x.ndim != 2 or router_w.ndim != 2 or x.shape[1] != router_w.shape[1]:
        raise ValueError("expected x [T, H] and router_w [E, H]")
    t, h = x.shape
    e = router_w.shape[0]
    if not 1 <= top_k <= e:
        raise ValueError("top_k must be in [1, E]")
    if x.dtype not in _ROUND or router_w.dtype != x.dtype:
        raise ValueError("x and router_w must share a BF16, FP16 or FP32 dtype")
    if not x.is_cuda or router_w.device != x.device:
        raise ValueError("x and router_w must be CUDA tensors on the same device")
    if valid_tokens is not None and (
        valid_tokens.shape != (1,)
        or valid_tokens.dtype != torch.int32
        or valid_tokens.device != x.device
    ):
        raise ValueError("valid_tokens must be an int32 [1] tensor on x's device")
    if selected is not None and (
        selected.shape != (t, top_k)
        or selected.dtype not in (torch.uint8, torch.int32, torch.int64)
        or selected.device != x.device
        or not selected.is_contiguous()
    ):
        raise ValueError("selected must be a contiguous integer [T, top_k] tensor on x's device")
    if probabilities is not None and (
        renormalize
        or probabilities.shape != (t, e)
        or probabilities.dtype != torch.float32
        or probabilities.device != x.device
        or not probabilities.is_contiguous()
    ):
        raise ValueError(
            "probabilities needs renormalize=False and a contiguous FP32 [T, E] tensor on x's device"
        )
    score = scoring == "sqrtsoftplus"
    if scoring not in ("softmax", "sqrtsoftplus"):
        raise ValueError("scoring must be softmax or sqrtsoftplus")
    if not score and (bias is not None or scale != 1.0):
        raise ValueError("bias and scale belong to sqrtsoftplus scoring")
    if score and (e > 8192 or top_k > 64):
        raise ValueError("sqrtsoftplus scoring takes E <= 8192 and top_k <= 64")
    if score and probabilities is not None:
        raise ValueError("probabilities belong to softmax scoring")
    if bias is not None and (
        bias.shape != (e,) or bias.dtype != torch.float32 or bias.device != x.device
    ):
        raise ValueError("bias must be an FP32 [E] tensor on x's device")
    weights = torch.empty((t, top_k), dtype=torch.float32, device=x.device)
    ids = torch.empty((t, top_k), dtype=torch.int32, device=x.device)
    if score and (t == 0 or h == 0):
        if t:
            _score_empty_hidden(
                weights, ids, top_k, renormalize, valid_tokens, selected, bias, scale
            )
        return weights, ids
    if t == 0 or h == 0:
        if t:
            # An empty hidden dimension: every logit is zero, ties go to the lowest ids.
            if selected is not None:
                ids.copy_(selected)
            else:
                ids.copy_(torch.arange(top_k, dtype=torch.int32, device=x.device).expand(t, -1))
            weights.fill_(1.0 / (top_k if renormalize else e))
            if probabilities is not None:
                probabilities.fill_(1.0 / e)
            if valid_tokens is not None:
                live = torch.arange(t, device=x.device)[:, None] < valid_tokens.clamp(0, t)
                ids.masked_fill_(~live, -1)
                weights.masked_fill_(~live, 0.0)
                if probabilities is not None:
                    probabilities.masked_fill_(~live, 0.0)
        return weights, ids
    kernels = _kernels()

    x = x.contiguous()
    router_w = router_w.contiguous()
    bk = -(-h // ROUTER_SPLIT)
    from ._block import hardware

    bt, be, bkc, split, warps, stages, flat = _logit_tiles(
        t, e, bk, x.element_size(), hardware(x.device)
    )
    # sqrtsoftplus above SCORE_GEMM_ROWS: the gate GEMM in cuBLAS (FP32 out), our kernel only
    # for scoring, bias, top-k and scale; below it, one fused logit kernel (measured faster).
    gemm = score and t > SCORE_GEMM_ROWS and not batch_invariant
    if gemm:
        split = False
    # Softmax scoring is vLLM's topk_softmax on the gating output in x's dtype (a power-of-two
    # E up to 512 also in its warp's reduction order). Split partials stay FP32; fused logits
    # are stored rounded, exactly, in x's dtype.
    vllm = not score and e <= 512 and not e & (e - 1)
    logits = torch.empty(
        (ROUTER_SPLIT, t, e) if split else (t, e),
        dtype=torch.float32 if split or score else x.dtype,
        device=x.device,
    )
    # DeepSeek-V4 keeps the logit in FP32 (no rounding to x's precision).
    rnd = 0 if score else _ROUND[x.dtype]
    ep, kp = _pow2(e), _pow2(top_k)
    # Four rows per top-k program keep T / 4 programs in flight (far faster than 16 rows).
    sb = max(1, min(4, 1024 // ep))
    # Very many experts or a large k: one row per program, experts streamed (no tile of E).
    stream = not vllm and (ep > 4096 or top_k > 32)
    with torch.cuda.device(x.device):
        neb = _cdiv(e, be)
        if gemm:
            torch.mm(x, router_w.t(), out_dtype=torch.float32, out=logits)
        else:
            kernels._route_logits_kernel[(neb * _cdiv(t, bt), ROUTER_SPLIT if split else 1)](
                x,
                router_w,
                logits,
                t,
                h,
                e,
                ROUTER_SPLIT,
                bt,
                be,
                bkc,
                split,
                x.dtype == torch.float32,
                rnd,
                flat,
                neb,
                num_warps=warps,
                num_stages=stages,
            )
        if score:
            sb_score, score_warps = SCORE_TILE
            sb_score = min(sb_score, sb)
            kernels._route_score_kernel[(_cdiv(t, sb_score),)](
                logits,
                bias if bias is not None else weights,
                ids,
                weights,
                valid_tokens if valid_tokens is not None else ids,
                selected if selected is not None else ids,
                t,
                t * e,
                float(scale),
                e,
                ep,
                top_k,
                kp,
                sb_score,
                renormalize,
                valid_tokens is not None,
                ROUTER_SPLIT,
                split,
                bias is not None,
                selected is not None,
                num_warps=score_warps,
            )
        elif stream:
            kernels._route_topk_stream_kernel[(t,)](
                logits,
                ids,
                weights,
                valid_tokens if valid_tokens is not None else ids,
                selected if selected is not None else ids,
                probabilities if probabilities is not None else weights,
                t,
                t * e,  # partial stride, int64 when large
                e,
                top_k,
                renormalize,
                valid_tokens is not None,
                ROUTER_SPLIT,
                split,
                _ROUND[x.dtype],
                1024,
                SEL=selected is not None,
                PROBS=probabilities is not None,
                num_warps=4,
            )
        else:
            kernels._route_topk_kernel[(_cdiv(t, sb),)](
                logits,
                ids,
                weights,
                valid_tokens if valid_tokens is not None else ids,
                selected if selected is not None else ids,
                probabilities if probabilities is not None else weights,
                t,
                t * e,  # partial stride, int64 when large
                e,
                ep,
                top_k,
                kp,
                sb,
                renormalize,
                valid_tokens is not None,
                ROUTER_SPLIT,
                split,
                _ROUND[x.dtype],
                SEL=selected is not None,
                PROBS=probabilities is not None,
                VLLM=vllm,
                ELTS=min(16, x.element_size() * e) // x.element_size(),
                num_warps=4,
            )
    return weights, ids


def _score_empty_hidden(weights, ids, top_k, renormalize, valid_tokens, selected, bias, scale):
    # An empty hidden dimension: every logit is 0, so every s is sqrt(softplus(0)); the bias
    # alone ranks, ties to the lower id.
    t = weights.shape[0]
    if selected is not None:
        ids.copy_(selected)
    elif bias is not None:
        ids.copy_(torch.sort(bias, descending=True, stable=True).indices[:top_k].expand(t, -1))
    else:
        ids.copy_(torch.arange(top_k, dtype=torch.int32, device=ids.device).expand(t, -1))
    s = torch.nn.functional.softplus(torch.zeros((), device=ids.device)).sqrt()
    if renormalize:
        tot = torch.zeros((), device=ids.device)
        for _ in range(top_k):
            tot = tot + s
        s = s / (tot + 1e-20)
    weights.copy_((s * scale).expand_as(weights))
    if valid_tokens is not None:
        live = torch.arange(t, device=ids.device)[:, None] < valid_tokens.clamp(0, t)
        ids.masked_fill_(~live, -1)
        weights.masked_fill_(~live, 0.0)


def hash_route(
    x: torch.Tensor,
    router_w: torch.Tensor,
    token_ids: torch.Tensor,
    table: torch.Tensor,
    *,
    renormalize: bool = True,
    scale: float = 1.0,
    valid_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4's hash-routed layers: FP32 ``weights`` and int32 ``ids``, both [T, k].

    Row t's experts are ``table[token_ids[t]]`` (``table`` int32 [vocab, k], the checkpoint's
    ``tid2eid``); each weight is sqrt(softplus(x_t . w_e)) with an FP32 dot, normalised over
    the row's experts (slot order, + 1e-20) with ``renormalize``, times ``scale``
    (checkpoint inference/model.py Gate; SGLang 0.5.22 srt/layers/moe/hash_topk.py). Rows at
    or past ``valid_tokens`` get ids -1 and weights 0. Graph-capturable.
    """
    if router_w.requires_grad:
        raise ValueError("the router must be frozen (requires_grad=False)")
    if x.ndim != 2 or router_w.ndim != 2 or x.shape[1] != router_w.shape[1]:
        raise ValueError("expected x [T, H] and router_w [E, H]")
    t, h = x.shape
    if table.ndim != 2 or table.dtype != torch.int32 or not table.is_contiguous():
        raise ValueError("table must be a contiguous int32 [vocab, k] tensor")
    if token_ids.shape != (t,) or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be an integer [T] tensor")
    if x.dtype not in _ROUND or router_w.dtype != x.dtype:
        raise ValueError("x and router_w must share a BF16, FP16 or FP32 dtype")
    devices = {x.device, router_w.device, token_ids.device, table.device}
    if not x.is_cuda or len(devices) != 1:
        raise ValueError("all tensors must be CUDA tensors on one device")
    if valid_tokens is not None and (
        valid_tokens.shape != (1,)
        or valid_tokens.dtype != torch.int32
        or valid_tokens.device != x.device
    ):
        raise ValueError("valid_tokens must be an int32 [1] tensor on x's device")
    k = table.shape[1]
    weights = torch.empty((t, k), dtype=torch.float32, device=x.device)
    ids = torch.empty((t, k), dtype=torch.int32, device=x.device)
    if t == 0 or k == 0:
        return weights, ids
    kernels = _kernels()

    with torch.cuda.device(x.device):
        kernels._hash_route_kernel[(t,)](
            x.contiguous(),
            router_w.contiguous(),
            token_ids.contiguous(),
            table,
            ids,
            weights,
            valid_tokens if valid_tokens is not None else ids,
            t,
            float(scale),
            h,
            k,
            _pow2(k),
            min(512, max(16, _pow2(h))),
            renormalize,
            valid_tokens is not None,
            num_warps=4,
        )
    return weights, ids


def to_local_topk_ids(topk_ids: torch.Tensor, expert_map: torch.Tensor) -> torch.Tensor:
    """int32 local expert ids of ``topk_ids`` (int32/int64 global ids, any shape) through
    ``expert_map`` (int32 [E_global]: local index or -1): -1 for ids outside [0, E_global) and
    for experts the map does not hold. One launch on CUDA, graph-capturable, no host sync."""
    if (
        not isinstance(expert_map, torch.Tensor)
        or expert_map.dtype != torch.int32
        or expert_map.ndim != 1
        or not expert_map.is_contiguous()
        or expert_map.device != topk_ids.device
    ):
        raise ValueError(
            "expert_map must be a contiguous int32 [E_global] tensor on topk_ids' device"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32 or int64")
    g = expert_map.numel()
    if not topk_ids.is_cuda:
        # The contract in torch (CPU).
        if g == 0:
            return torch.full_like(topk_ids, -1, dtype=torch.int32)
        valid = (topk_ids >= 0) & (topk_ids < g)
        mapped = expert_map[topk_ids.clamp(0, g - 1).long()]
        return torch.where(valid, mapped, -1).to(torch.int32)
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    n = ids.numel()
    if n:
        with torch.cuda.device(ids.device):
            _kernels()._local_ids_kernel[(_cdiv(n, 1024),)](
                ids, expert_map, out, n, g, 1024, num_warps=4
            )
    return out
