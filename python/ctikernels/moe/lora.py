# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Per-expert MoE LoRA (gate, up and down of every routed expert, plus the shared expert).

Frozen base and router; factors are never merged into base weights.
The CuTe kernels bind on first CUDA use (``cute._lora_provider.kernels``).
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ..activation.inference import silu_and_mul
from ..moe_impl import (
    block_fp8_moe_config,
    fused_moe_kernel_triton,
    moe_align_block_size,
    moe_shared_join_,
    moe_sum_reduce_triton,
)
from . import _lora_workspace
from ._lora_backward import (
    backward_steps,
    check_routed,
    training_factors,
)
from ._lora_common import device_info, grid_capacity, overlap_stream, scale_tensor
from ._lora_weights import (
    FactorSlices,
    MoELoRAWeights,
    check_slices,
    check_tensor,
    routed_slices,
    select_adapter,
    validate_rank,
)
from .cute import _lora_provider
from .cute._lora_shared_table import shared_choice
from .torch import _lora_reference as reference

__all__ = [
    "MoELoRAWeights",
    "supported",
    "validate_rank",
    "moe_lora_w13_",
    "moe_lora_w13_act_",
    "moe_lora_w2_",
    "moe_lora_sorted_shrink_",
    "shared_lora_w13_",
    "shared_lora_w13_act_",
    "shared_lora_w2_",
    "fused_experts_lora",
    "fused_experts_lora_training_bytes",
    "shared_expert_lora",
]


PROVIDERS = ("auto", "cute", "triton", "torch")
_CUTE_CAPABILITIES = ((12, 0), (12, 1))


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """True when ``provider="auto"`` runs a kernel (CuTe or Triton, never the torch path) for
    every call on ``device`` and ``dtype`` at any row count: BF16 on CUDA with compute
    capability >= 8.9. Ranks outside a provider's range raise ValueError."""
    device = torch.device(device)
    return (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and _lora_provider.device_info(device)[0] >= (8, 9)
    )


def _resolve(provider, x, rank):
    """``auto``: torch on CPU; Triton on every CUDA device of compute capability >= 8.9, for
    inference and training alike (it keeps routes in FP32 to one rounding, as the MoE block
    tree, and its training forward is the inference composition; the CuTe composition rounds
    the FC1/FC2 caches). ``cute`` stays available explicitly. ``rank`` is unused by auto."""
    if provider not in PROVIDERS:
        raise ValueError("provider must be auto, cute, triton, or torch")
    if provider != "auto" or not x.is_cuda:
        return "torch" if provider == "auto" else provider
    capability = _lora_provider.device_info(x.device)[0]
    if capability < (8, 9):
        raise ValueError("MoE LoRA kernels need compute capability >= 8.9")
    return "triton"


def _triton_ops():
    # Imports Triton: bound on first CUDA use only.
    from .triton import _lora_ops

    return _lora_ops


def _lora_train():
    from .triton import _lora_train

    return _lora_train


def _factors(weights, down, shared, adapter_id):
    """One projection's (A, B) FactorSlices, each slice [E,R,C] (E = 1 for shared)."""
    if isinstance(weights, MoELoRAWeights):
        if shared:
            weights = weights.shared
            if weights is None:
                raise ValueError("shared factors were not supplied")
        else:
            weights = weights.routed[4:] if down else weights.routed[:4]
    if shared:
        if len(weights) == 4:
            weights = weights[2:] if down else weights[:2]
        if len(weights) != 2:
            raise ValueError(
                "shared weights must be MoELoRAWeights, four factors or an (A, B) pair"
            )
        a, b = (select_adapter(t, 2 if down else 3, adapter_id) for t in weights)
        if down:
            return FactorSlices((a[None],)), FactorSlices((b[None],))
        return FactorSlices((a[0][None], a[1][None])), FactorSlices((b[0][None], b[1][None]))
    if not down and len(weights) == 4:  # per-slice (gate_a, up_a, gate_b, up_b)
        f = [select_adapter(t, 3, adapter_id) for t in weights]
        check_slices(f[0], f[1], "A")
        check_slices(f[2], f[3], "B")
        return FactorSlices(f[:2]), FactorSlices(f[2:])
    if len(weights) != 2:
        raise ValueError(
            "routed weights must be MoELoRAWeights, (gate_up_a, gate_up_b), "
            "(gate_a, up_a, gate_b, up_b) or (down_a, down_b)"
        )
    a, b = (
        select_adapter(t, d, adapter_id)
        for t, d in zip(weights, (3, 3 if down else 4), strict=True)
    )
    if down:
        return FactorSlices((a,)), FactorSlices((b,))
    if b.ndim != 4 or b.shape[0] != 2:
        raise ValueError("gate_up_b must have shape [2, E, I, r]")
    a_gate, a_up, b_gate, b_up = routed_slices(a, b, None, None)[:4]
    return FactorSlices((a_gate, a_up)), FactorSlices((b_gate, b_up))


def _apply(
    out,
    x,
    weights,
    ids,
    *,
    scaling,
    top_k,
    down=False,
    shared=False,
    topk_weights=None,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
    activated=None,
    stage="full",
    chunks=None,
):
    _lora_provider.validate_chunks(chunks, stage=stage, shared=shared)
    if token_adapter_ids is not None:
        raise NotImplementedError("per-token adapters are not supported")
    if provider not in PROVIDERS:
        raise ValueError("provider must be auto, cute, triton, or torch")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    a, b = _factors(weights, down, shared, adapter_id)
    # The routed down cache may be FP32 (callers keeping weighted routes unrounded).
    fp32_cache = down and not shared and getattr(out, "dtype", None) == torch.float32
    check_tensor(out, "cache", dtype=out.dtype if fp32_cache else torch.bfloat16, device=x.device)
    for name, t in (("x", x), *(("A", t) for t in a), *(("B", t) for t in b)):
        check_tensor(t, name, device=x.device)
    if x.ndim != 2 or out.ndim != 2:
        raise ValueError("input and cache must be matrices")
    e, slices, rank, k = a.shape
    validate_rank(rank)
    n = b.shape[-2]
    if e <= 0 or k <= 0 or n <= 0 or k % 8 or n % 8:
        raise ValueError("E must be positive; K and N positive multiples of 8")
    if slices != (1 if down else 2) or b.shape != (e, slices, n, rank):
        raise ValueError("incompatible A/B factor shapes")
    p = out.shape[0]
    if x.shape != (p if down else p // top_k, k) or p % top_k or out.shape[1] != slices * n:
        raise ValueError("input/cache shapes disagree with factors and top_k")
    if not shared:
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk ids must be int32 or int64")
        check_tensor(ids, "topk_ids_local", dtype=ids.dtype, device=x.device)
        if ids.shape != (p // top_k, top_k):
            raise ValueError("topk_ids_local must have shape [T, top_k]")
        if down and topk_weights is not None:
            check_tensor(topk_weights, "topk_weights", dtype=torch.float32, device=x.device)
            if topk_weights.shape != ids.shape:
                raise ValueError("topk_weights must match topk_ids_local")
    if z_out is not None:
        if z_out.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("z_out must be BF16 or FP32")
        check_tensor(z_out, "z_out", dtype=z_out.dtype, device=x.device)
        if z_out.ndim != 3 or z_out.shape[0] != p or z_out.shape[2] != rank:
            raise ValueError("z_out must have shape [pairs, slices, rank]")
    if activated is not None:
        check_tensor(activated, "cache2", device=x.device)
        if activated.shape != (p, n):
            raise ValueError("cache2 must have shape [pairs, I]; cache1 must be [pairs, 2I]")
        # Contiguous views can share an arena while occupying disjoint ranges.
        if p and (
            activated.data_ptr() < out.data_ptr() + out.numel() * out.element_size()
            and out.data_ptr() < activated.data_ptr() + activated.numel() * activated.element_size()
        ):
            raise ValueError("cache2 must not alias cache1")
    alignment = (sorted_token_ids, expert_ids, num_tokens_post_padded)
    if any(t is not None for t in alignment):
        if any(t is None for t in alignment) or block_m not in (16, 32, 64):
            raise ValueError("alignment requires all three tensors and block_m in {16,32,64}")
        for name, t in zip(
            ("sorted_token_ids", "expert_ids", "num_tokens_post_padded"), alignment, strict=True
        ):
            check_tensor(t, name, dtype=torch.int32, device=x.device)
            if t.ndim != 1:
                raise ValueError(f"{name} must be one dimensional")
        if (
            num_tokens_post_padded.numel() != 1
            or expert_ids.numel() < (sorted_token_ids.numel() + block_m - 1) // block_m
        ):
            raise ValueError("alignment capacity is inconsistent")
    elif block_m is not None and block_m not in (16, 32, 64):
        raise ValueError("block_m must be 16, 32, or 64")
    scale = scale_tensor(scaling, x.device)
    if _trains(x, a, b, out, activated, topk_weights):
        # The low-level ops are inference kernels; on CUDA, training runs through
        # fused_experts_lora (deterministic Triton or CuTe autograd), never a silent torch path.
        if x.is_cuda and provider != "torch":
            raise ValueError(
                "MoE LoRA insertion ops are inference-only on CUDA: train the routed experts "
                "through fused_experts_lora, or pass provider='torch' for the reference"
            )
        if stage != "full":
            raise ValueError("split stages are inference-only")
        return reference.autograd_apply_(
            out,
            x,
            a.stacked(),
            b.stacked(),
            ids,
            scale,
            top_k,
            topk_weights if down else None,
            z_out,
            activated,
            down=down,
        )
    tensors = (x, *a, *b, out, z_out, activated)
    aligned_storage = all(t.data_ptr() % 16 == 0 for t in tensors if t is not None)
    if fp32_cache and provider == "cute":
        raise ValueError("cute takes a BF16 cache3; an FP32 cache3 runs on triton or torch")
    if fp32_cache and provider == "auto" and x.is_cuda:
        provider = "triton"
    provider = _resolve(provider, x, rank)
    # Shared-A gate/up: Triton runs the shrink once and keeps one z slice.
    z_slices = 1 if provider == "triton" and len(a) == 2 and a.gap == 0 else slices
    if z_out is not None and z_out.shape[1] != z_slices:
        raise ValueError(f"z_out must have shape [pairs, {z_slices}, rank]")
    if provider == "triton":
        if stage != "full":
            raise ValueError("split stages require the cute provider")
        if not x.is_cuda:
            raise ValueError("triton requires CUDA tensors")
        return _triton_ops().apply_(
            out,
            x,
            a,
            b,
            None if shared else ids,
            scale,
            top_k,
            topk_weights if down else None,
            z_out,
            down=down,
            activated=activated,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            block_m=block_m,
        )
    if provider == "cute":
        if not _lora_provider.supports_rank(rank):
            raise ValueError("cute ranks are multiples of 8 in [8, 128]")
        if (
            activated is not None
            and not shared
            and any(
                t is None for t in (sorted_token_ids, expert_ids, num_tokens_post_padded, block_m)
            )
        ):
            raise ValueError("fused activation requires alignment and block_m")
        if not x.is_cuda:
            raise ValueError("cute requires CUDA tensors")
        if not aligned_storage:
            raise ValueError(
                "cute input, factor, cache and z_out addresses must be 16-byte aligned"
            )
        mode, config = "cute", None
        if shared and stage == "full":
            if activated is not None:
                mode = _shared_activation_mode(x, p, rank)
            else:
                mode, config = _shared_choice(x, p, rank, k, n, slices)
        if mode == "dense":
            return _lora_provider.kernels().dense.dense_apply_(
                out,
                x,
                a.packed()[0].reshape(slices * rank, k),
                b.packed()[0],
                scale,
                rank=rank,
                slices=slices,
                z_out=z_out,
                config=config,
            )
        if mode != "cute":
            # Dense shared rows: one factor set serves every row, so the shrink is a
            # plain skinny library GEMM with FP32 accumulation and output.
            a2d = a.packed()[0].reshape(slices * rank, k).t()
            if mode == "split" and (z_out is None or z_out.dtype == torch.bfloat16):
                # BF16 output is exactly the BF16 z the expand stage consumes.
                z = torch.mm(x, a2d).view(p, slices, rank)
                if z_out is not None:
                    z_out.copy_(z)
                return _lora_provider.apply_(
                    out,
                    x,
                    a,
                    b,
                    ids,
                    scale,
                    top_k,
                    topk_weights,
                    z,
                    down=down,
                    activated=activated,
                    stage="expand",
                )
            z = torch.mm(x, a2d, out_dtype=torch.float32)
            if z_out is not None:
                z_out.copy_(z.view(p, slices, rank))
            if mode == "library" and activated is None:
                # Fold the live scale into z with one BF16 rounding, then add each
                # slice in place: FP32 accumulation, one rounding into the cache.
                zs = torch.empty((p, slices * rank), dtype=out.dtype, device=x.device)
                torch.mul(z, scale, out=zs)
                for s in range(slices):
                    out[:, s * n : (s + 1) * n].addmm_(
                        zs[:, s * rank : (s + 1) * rank], b[s][0].t()
                    )
                return out
            # Otherwise the CuTe expand stage consumes the BF16 z.
            z_out, stage = z.to(torch.bfloat16).view(p, slices, rank), "expand"
        return _lora_provider.apply_(
            out,
            x,
            a,
            b,
            ids,
            scale,
            top_k,
            topk_weights,
            z_out,
            down=down,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            block_m=block_m,
            activated=activated,
            stage=stage,
            chunks=chunks,
        )
    if stage != "full":
        raise ValueError("split stages require the cute provider")
    if activated is not None:
        return reference.w13_act(
            activated, out, x, a.stacked(), b.stacked(), ids, scale, top_k, z_out
        )
    return reference.apply_(
        out, x, a.stacked(), b.stacked(), ids, scale, top_k, topk_weights, z_out, down=down
    )


def _trains(*tensors):
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


def _shared_choice(x, rows, rank, k, n, slices):
    capability, sms = _lora_provider.device_info(x.device)
    return shared_choice(rows, rank, k, n, slices, capability, sms)


def _shared_activation_mode(x, rows, rank):
    """Dense shared rows, by measured winner on the RTX 5090: one CuTe launch at
    rank 8 up to 128 rows (and all decode rows); library GEMMs for rank 128, rank
    >= 64 from 1024 rows and every rank from 2048 rows; otherwise a library
    shrink feeding the CuTe expand stage."""
    _, sms = _lora_provider.device_info(x.device)
    if not _lora_provider.uses_block_path(rows, rank, sms, shared=True):
        return "cute"
    if rank >= 128 or rows >= 2048 or (rank >= 64 and rows >= 1024):
        return "library"
    if rank <= 8 and rows <= 128:
        return "cute"
    return "split"


def moe_lora_w13_act_(
    cache2,
    cache1,
    x,
    lora,
    topk_ids_local,
    *,
    scaling,
    top_k,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Write BF16 SwiGLU after FP32 LoRA addition; cache1 is read only.

    Alignment (required by CuTe, optional for Triton) must cover every local valid pair.
    Invalid/unhandled pairs leave cache2 and z_out untouched. cache2 [P,I] must not alias
    cache1 [P,2I].
    """
    check_tensor(cache2, "cache2", device=x.device)
    return _apply(
        cache1,
        x,
        lora,
        topk_ids_local,
        scaling=scaling,
        top_k=top_k,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_m=block_m,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
        activated=cache2,
    )


def shared_lora_w13_act_(
    cache2,
    cache1,
    x,
    shared_factors,
    *,
    scaling,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Dense shared SwiGLU with FP32 LoRA addition and one BF16 output rounding."""
    check_tensor(cache2, "cache2", device=x.device)
    return _apply(
        cache1,
        x,
        shared_factors,
        None,
        scaling=scaling,
        top_k=1,
        shared=True,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
        activated=cache2,
    )


def _uses_block_projection(x, cache1, weights, provider, *, shared=False, adapter_id=0, down=False):
    if not x.is_cuda:
        return False
    a, _ = _factors(weights, down, shared, adapter_id)
    resolved = _resolve(provider, x, a.shape[-2])
    if resolved == "triton":
        return True  # every Triton path fuses the activation
    _, sms = _lora_provider.device_info(x.device)
    if not _lora_provider.uses_block_path(cache1.shape[0], a.shape[-2], sms, shared):
        return False
    return resolved == "cute"


def _shrink_async(out, x, weights, ids, *, scaling, top_k, down=False, **alignment):
    """Fork after input production; the caller joins before consuming BF16 z.

    Allocate on the caller's stream and retain every side-stream input through
    its last use. Events are per invocation so independent callers cannot reuse
    a recorded event while another call is still enqueueing its join.
    """
    a, b = _factors(weights, down, False, 0)
    scale = scale_tensor(scaling, x.device)
    z = torch.empty((out.shape[0], a.shape[1], a.shape[-2]), dtype=torch.bfloat16, device=x.device)
    main = torch.cuda.current_stream(x.device)
    side = overlap_stream(x.device)
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record(main)
    with torch.cuda.stream(side):
        side.wait_event(ready)
        for tensor in (x, *a, *b, ids, scale, z, out, *alignment.values()):
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(side)
        _apply(
            out,
            x,
            weights,
            ids,
            scaling=scale,
            top_k=top_k,
            down=down,
            provider="cute",
            z_out=z,
            stage="shrink",
            **alignment,
        )
        done.record(side)
    return z, done


_SPLIT_ROUTED = None  # Private override: None uses policy, 0 disables, positive int sets chunks.


def _shrink_now(out, x, weights, ids, *, scaling, top_k, down=False, z=None, **alignment):
    """Shrink stage on the current stream; the expand stage reads its BF16 z."""
    if z is None:
        a, _ = _factors(weights, down, False, 0)
        z = torch.empty(
            (out.shape[0], a.shape[1], a.shape[-2]), dtype=torch.bfloat16, device=x.device
        )
    _apply(
        out,
        x,
        weights,
        ids,
        scaling=scaling,
        top_k=top_k,
        down=down,
        provider="cute",
        z_out=z,
        stage="shrink",
        **alignment,
    )
    return z, None


def moe_lora_w13_(
    cache1,
    x,
    weights_or_tensors,
    topk_ids_local,
    *,
    scaling,
    top_k,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Add gate/up LoRA to the BF16 cache [T*top_k, 2I], in pair order."""
    return _apply(
        cache1,
        x,
        weights_or_tensors,
        topk_ids_local,
        scaling=scaling,
        top_k=top_k,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_m=block_m,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
    )


def moe_lora_sorted_shrink_(z, x, a, row_token, row_pair, expert_offsets):
    """z[pair, s] = A_s[e] @ x[token] (FP32) over expert-sorted rows: rows
    [expert_offsets[e], expert_offsets[e+1]) belong to local expert e; row i reads token
    ``row_token[i]`` and writes entry ``row_pair[i]`` of z [P, slices, r], or entry i (sorted-row
    order) when ``row_pair`` is None. ``a`` is the
    projection's A slices, each [E, r, K] BF16 (e.g. (gate_a, up_a)). Entries of z without
    a row are left untouched. No padding, no allocation: capture-safe. Triton, CUDA only."""
    a = FactorSlices(a)
    for t in (*a, x):
        check_tensor(t, "x and A", device=x.device)
    check_slices(a[0], a[-1], "A")
    experts, rank, k = a[0].shape
    check_tensor(z, "z", dtype=torch.float32, device=x.device)
    if z.ndim != 3 or z.shape[1:] != (len(a), rank) or x.ndim != 2 or x.shape[1] != k:
        raise ValueError("z must be [P, slices, r] and x [T, K] for A [E, r, K]")
    order = row_token if row_pair is None else row_pair
    for name, t in (("row_token", row_token), ("row_pair", order)):
        check_tensor(t, name, dtype=torch.int32, device=x.device)
    check_tensor(expert_offsets, "expert_offsets", dtype=torch.int32, device=x.device)
    if row_token.shape != order.shape or expert_offsets.shape != (experts + 1,):
        raise ValueError("row_token and row_pair [rows]; expert_offsets [E + 1]")
    if not x.is_cuda:
        raise ValueError("moe_lora_sorted_shrink_ runs on CUDA (Triton)")
    if row_pair is None and z.shape[0] < row_token.shape[0]:
        raise ValueError("sorted-row order needs z with at least one entry per row")
    if row_token.numel() == 0:
        return z
    from .triton import _lora_sorted  # Imports Triton: bound on first CUDA use only.

    return _lora_sorted.sorted_shrink_(z, x, a, row_token, row_pair, expert_offsets)


def moe_lora_w2_(
    cache3_2d,
    cache2,
    weights_or_tensors,
    topk_ids_local,
    topk_weights,
    *,
    scaling,
    top_k,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Add routed-weighted down LoRA to BF16 or FP32 [T*top_k, H] (FP32: no BF16 rounding;
    triton or torch)."""
    return _apply(
        cache3_2d,
        cache2,
        weights_or_tensors,
        topk_ids_local,
        scaling=scaling,
        top_k=top_k,
        down=True,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_m=block_m,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
    )


def shared_lora_w13_(
    cache1,
    x,
    weights_or_tensors,
    *,
    scaling,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Shared gate/up factors [2,r,H]/[2,Is,r]; no routing tensor is allocated."""
    return _apply(
        cache1,
        x,
        weights_or_tensors,
        None,
        scaling=scaling,
        top_k=1,
        shared=True,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
    )


def shared_lora_w2_(
    cache3,
    cache2,
    weights_or_tensors,
    *,
    scaling,
    adapter_id=0,
    token_adapter_ids=None,
    provider="auto",
    z_out=None,
):
    """Shared down factors [r,Is]/[H,r], with no routed weight."""
    return _apply(
        cache3,
        cache2,
        weights_or_tensors,
        None,
        scaling=scaling,
        top_k=1,
        down=True,
        shared=True,
        adapter_id=adapter_id,
        token_adapter_ids=token_adapter_ids,
        provider=provider,
        z_out=z_out,
    )


_NAIVE = {}


def _naive_alignment(pairs, block_m, device):
    """Static per-shape metadata: block i holds routed pair i, padded with ``pairs``."""
    key = (device, pairs, block_m)
    cached = _NAIVE.get(key)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up this MoE batch size before CUDA graph capture")
        # Process-global: never an inference tensor of the first caller (training saves it).
        with torch.inference_mode(False):
            ids = torch.full((pairs, block_m), pairs, dtype=torch.int32, device=device)
            ids[:, 0] = torch.arange(pairs, dtype=torch.int32, device=device)
            padded = torch.tensor([pairs * block_m], dtype=torch.int32, device=device)
        cached = _NAIVE[key] = (ids.view(-1), padded)
    return cached


def base_config_for(tokens, *, shared_lora_tiles=False, gb10=False):
    """Base grouped-GEMM tiles, always GROUP_SIZE_M=1.

    BM/BN/BK, warps/stages: <=32 uses 16/64/128, 4/4; 33-256 uses
    32/64/128, 4/3; 257-512 uses 64/128/64, 8/3; >512 uses 128/128/64, 8/3.
    At 33-64 tokens, ``shared_lora_tiles`` (RTX 5090 with LoRA) uses
    16/64/128, 4/3; ``gb10`` takes precedence and uses 16/64/128, 8/4.
    The 257-512 specials collapsed under concentrated routing: removing them
    at Qwen3.5 rank 16, T=447 raised vLLM/ours narrow/skewed ratios from
    0.671/0.702 to 1.017/1.045 on RTX 5090 and 0.825/0.896 to 0.996/1.014 on GB10.
    """
    if gb10 and 32 < tokens <= 64:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        }
    if shared_lora_tiles and 32 < tokens <= 64:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    if tokens <= 32:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        }
    if tokens <= 256:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    if tokens <= 512:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 3,
        }
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 3,
    }


def _silu_and_mul(c1, out=None):
    if c1.is_cuda:
        try:
            return silu_and_mul(c1, out=out)
        except ImportError:  # FlashInfer is optional; fall back to the FP32 torch gate.
            pass
    gate, up = c1.float().chunk(2, dim=-1)
    result = (F.silu(gate) * up).to(c1.dtype)
    return result if out is None else out.copy_(result)


def _lora_tensors(lora):
    if lora is None:
        return ()
    if isinstance(lora, MoELoRAWeights):
        # Every present factor, routed and shared, decides the training path.
        return lora.routed + (lora.shared or ())
    return tuple(lora)


def fused_experts_lora(
    x,
    w13,
    w2,
    topk_weights,
    topk_ids_local,
    lora,
    *,
    scaling,
    provider="auto",
    base_config=None,
    local_only=False,
    overlap=None,
    num_global_experts=None,
    adapter_id=0,
    w13_scale=None,
    w2_scale=None,
    shared_expert=None,
    shared=None,
    shared_gate=None,
):
    """Frozen routed base MLP with LoRA inserted before activation and reduction.

    CUDA uses moe_impl's base Triton pipeline; CPU uses FP32 grouped matmuls. On the Triton
    provider a BF16 base is exactly ``moe.inference.fused_experts`` on the same local routes
    (EP: other ranks' routes parked on expert 0 at weight 0 there, -1 here) with its tiles for
    E = ``w13.shape[0]`` (``base_config`` is its ``config``), each LoRA delta added into its
    caches, so zero factors (or ``lora=None``) give its bits.
    Invalid local experts contribute zero to both the base and the LoRA branch.
    ``local_only=True`` promises every id is a valid local expert (EP1) and skips
    the masking pass. Small batches give each routed pair its own base block
    (no alignment kernel and no padded grid).
    ``overlap=True`` overlaps block shrinks with the base CUDA GEMMs on a side
    stream; ``None`` (default) or False keeps both projections on the current
    stream (the overlap measured no end-to-end gain on sm_120).
    ``num_global_experts`` (optional, EP) is the model's expert count; with it, the
    LoRA row tile assumes only the local share of the routes lands on this rank
    (EP2 Qwen3-30B-A3B at T=512, rank 32: 11% faster on the RTX 5090 than sizing
    tiles by every route). Without it every route is assumed local.
    Training on CUDA (auto, triton or cute) follows Unsloth 2026.10.7's order on the Triton
    kernels (``triton._lora_train``): each LoRA added as PEFT does in BF16, BF16 SiLU and
    SiLU * up, unweighted BF16 FC2 rows, moe_block's combine with BF16 route weights, and the
    shared expert as HF's MLP with PEFT LoRA added in BF16 after the routed sum. A block-FP8
    base trains on its inference composition (straight-through quantizers). The torch provider
    and CPU keep the differentiable FP32 reference. ``adapter_id`` selects the only adapter
    receiving factor gradients. Warm up forward and backward before capture.

    EP contract: a local id outside [0, E) (``to_local_topk_ids`` gives -1 for another
    rank's expert) adds exactly nothing and, on the Triton provider, costs no base or LoRA
    work; ids parked on expert 0 at routing weight 0 remain correct.
    Block-FP8 base (Qwen3.5-35B-A3B-FP8): pass E4M3 ``w13`` [E,2I,H] and ``w2`` [E,H,I]
    with their 128x128 block scales ``w13_scale`` [E,ceil(2I/128),ceil(H/128)] and
    ``w2_scale`` [E,ceil(H/128),ceil(I/128)] (FP32 or BF16). Activations are quantized 1x128
    per token; the LoRA branch stays BF16. Runs on the Triton provider; training uses the
    straight-through estimator for the activation quantizers (the FP8 base stays frozen).
    ``shared_expert`` = (w_gate_up [2Is, H], w_down [H, Is], w_gate [1, H]), BF16, frozen:
    adds the gated shared expert sigmoid(x . w_gate) * shared(x), with ``lora.shared`` factors
    when given, so one call is the MoE block minus the router. On Triton BF16 inference it
    runs on the grouped GEMM as one dense expert, in FP32 up to its one SwiGLU rounding, and
    joins the routes' combine as fused_experts' ``shared`` input; elsewhere it is
    shared_expert_lora's output joined in vLLM's BF16 order. TP: pass this rank's I-window (w_gate_up
    rows, w_down columns and the matching shared-LoRA slices) and the full w_gate; the output
    is this rank's partial for the block's one exit sum.
    ``shared`` [T, H] and ``shared_gate`` [T, 1] (x's dtype; inference only, in place of
    ``shared_expert``): the caller's shared expert output and gate logits, joined in
    the combine as ``moe_sum_reduce_triton`` joins them, r(r(r(sigmoid(gate)) * shared) +
    r(routed)) in x's dtype (vLLM's order).
    """
    if (shared is None) != (shared_gate is None):
        raise ValueError("pass both shared and shared_gate, or neither")
    if shared is not None:
        _check_joined_shared(x, shared, shared_gate, shared_expert)
        if _trains(x, topk_weights, *_lora_tensors(lora)):
            raise ValueError("shared and shared_gate are inference-only")
    if (
        isinstance(lora, MoELoRAWeights)
        and lora.dtype != torch.bfloat16
        and not _trains(x, topk_weights, *_lora_tensors(lora))
    ):
        lora = lora.to(torch.bfloat16)  # FP32 masters: inference reads BF16 factors
    options = {
        "scaling": scaling, "provider": provider, "base_config": base_config,
        "local_only": local_only, "overlap": overlap, "num_global_experts": num_global_experts,
        "adapter_id": adapter_id, "w13_scale": w13_scale, "w2_scale": w2_scale,
    }  # fmt: skip
    if shared_expert is None:
        joined = {} if shared is None else {"shared": shared, "shared_gate": shared_gate}
        return _routed_lora(x, w13, w2, topk_weights, topk_ids_local, lora, **options, **joined)
    kernel = None if not x.is_cuda else _resolve(provider, x, None if lora is None else lora.rank)
    fold = w13_scale is None and kernel == "triton"
    if (
        w13_scale is None
        and kernel in ("triton", "cute")
        and _trains(x, topk_weights, *_lora_tensors(lora))
    ):
        return _block_training(
            x, w13, w2, topk_weights, topk_ids_local, lora, shared_expert, scaling, base_config,
            local_only, adapter_id,
        )  # fmt: skip
    shared = _shared_expert(x, shared_expert, lora, scaling, provider, adapter_id, fold)
    if fold:
        gated = {"shared_expert": shared}
        return _routed_lora(x, w13, w2, topk_weights, topk_ids_local, lora, **options, gated=gated)
    shared, shared_gate = shared
    out = _routed_lora(x, w13, w2, topk_weights, topk_ids_local, lora, **options)
    if _trains(x, topk_weights, *_lora_tensors(lora)):
        # The differentiable FP32 composition's join (CPU and the torch provider).
        joined = out.float() + torch.sigmoid(shared_gate.float()) * shared.float()
        return joined.to(out.dtype)
    # vLLM's gated shared expert, each a BF16 op (qwen2_moe.py:121, moe_runner.py:786).
    return torch.sigmoid(shared_gate) * shared + out


def _block_training(
    x, w13, w2, topk_weights, ids, lora, shared_expert, scaling, base_config, local_only,
    adapter_id,
):  # fmt: skip
    """Trainable MoE block minus the router: Unsloth's routed experts (TritonBlockLoRA) plus
    the gated shared expert in torch, added in BF16 after the routed sum."""
    _validate_base(x, w13, w2, ids, topk_weights)
    w_gate_up, w_down, w_gate = _check_shared_weights(x, shared_expert)
    factors = training_factors(lora, x, w13, w2, adapter_id=adapter_id)
    shared = None if lora is None else lora.shared
    shared_factors = training_factors(
        shared, x, w_gate_up, w_down, shared=True, adapter_id=adapter_id
    )
    scale = scale_tensor(scaling, x.device).detach()
    options = {"local_only": local_only, "expert_config": base_config}
    routed = TritonBlockLoRA.apply(x, w13, w2, topk_weights, ids, scale, *factors, options)
    gated = _lora_train().shared_forward(x, w_gate_up, w_down, w_gate, shared_factors, scale)
    return routed + gated


def _check_shared_weights(x, weights):
    if not isinstance(weights, (tuple, list)) or len(weights) != 3:
        raise ValueError("shared_expert must be (w_gate_up, w_down, w_gate)")
    w_gate_up, w_down, w_gate = weights
    _check_frozen_base(("w_gate_up", w_gate_up), ("w_down", w_down), ("w_gate", w_gate))
    _validate_shared_base(x, w_gate_up, w_down)
    check_tensor(w_gate, "w_gate", device=x.device)
    if w_gate.shape != (1, x.shape[1]):
        raise ValueError(f"w_gate must have shape (1, {x.shape[1]})")
    return w_gate_up, w_down, w_gate


def _shared_expert(x, weights, lora, scaling, provider, adapter_id, fold):
    """Validated ``weights`` = (w_gate_up, w_down, w_gate): for the Triton fold, the weights
    with the shared factor slices (or None); else (shared_expert_lora output, BF16 logits)."""
    w_gate_up, w_down, w_gate = _check_shared_weights(x, weights)
    lora_shared = None if lora is None else lora.shared
    if not fold:
        out = shared_expert_lora(
            x, w_gate_up, w_down, lora_shared, scaling, provider=provider, adapter_id=adapter_id
        )
        return out, _dense(x, w_gate)
    factors = None
    if lora_shared is not None:
        factors = (
            *_factors(lora, False, True, adapter_id),
            *_factors(lora, True, True, adapter_id),
        )
    return w_gate_up, w_down, w_gate, factors


def _routed_lora(
    x, w13, w2, topk_weights, topk_ids_local, lora, *, scaling, provider, base_config,
    local_only, overlap, num_global_experts, adapter_id, w13_scale, w2_scale, gated=None,
    shared=None, shared_gate=None,
):  # fmt: skip
    if w13_scale is not None or w2_scale is not None:
        return _fused_experts_lora_block_fp8(
            x,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids_local,
            lora,
            scaling=scaling,
            provider=provider,
            base_config=base_config,
            local_only=local_only,
            adapter_id=adapter_id,
            shared=shared,
            shared_gate=shared_gate,
        )
    if num_global_experts is not None and (
        type(num_global_experts) is not int or num_global_experts < w13.shape[0]
    ):
        raise ValueError("num_global_experts must be an int >= the local expert count")
    if _trains(x, topk_weights, *_lora_tensors(lora)):
        if overlap is not None and type(overlap) is not bool:
            raise ValueError("overlap must be None, False, or True")
        if provider not in PROVIDERS:
            raise ValueError("provider must be auto, cute, triton, or torch")
        _validate_base(x, w13, w2, topk_ids_local, topk_weights)
        factors = training_factors(lora, x, w13, w2, adapter_id=adapter_id)
        if lora is not None:
            lora = MoELoRAWeights(*factors)
        scale = scale_tensor(scaling, x.device).detach()
        rank = None if factors[0] is None else factors[0].shape[-2]
        if x.is_cuda and _resolve(provider, x, rank) in ("triton", "cute"):
            options = {"local_only": local_only, "expert_config": base_config}
            return TritonBlockLoRA.apply(
                x, w13, w2, topk_weights, topk_ids_local, scale, *factors, options
            )
        return reference.fused_experts_autograd(
            x, w13, w2, topk_weights, topk_ids_local, None if lora is None else lora.routed, scale
        )
    if lora is not None:
        lora = lora.adapter(adapter_id)
    if _resolve(provider, x, None if lora is None else lora.rank) == "triton":
        if overlap:
            raise ValueError("overlap is a cute option")
        _validate_base(x, w13, w2, topk_ids_local, topk_weights)
        check_routed(lora, x, w13, w2)
        return _triton_ops().fused_experts_lora(
            x,
            w13,
            w2,
            topk_weights,
            topk_ids_local,
            lora,
            scaling=scaling,
            base_config={**base_config_for(x.shape[0]), **(base_config or {})},
            local_only=local_only,
            expert_config=base_config,
            shared=shared,
            shared_gate=shared_gate,
            **(gated or {}),
        )
    with torch.no_grad():
        out = _fused_experts_lora(
            x,
            w13,
            w2,
            topk_weights,
            topk_ids_local,
            lora,
            scaling=scaling,
            provider=provider,
            base_config=base_config,
            local_only=local_only,
            overlap=overlap,
            num_global_experts=num_global_experts,
        )
        if shared is None:
            return out
        if out.is_cuda:  # the reduce's join: one launch, bitwise
            return moe_shared_join_(out, shared, shared_gate)
        return torch.sigmoid(shared_gate) * shared + out  # its BF16 ops, each rounded


@torch.no_grad()
def _fused_experts_lora(
    x,
    w13,
    w2,
    topk_weights,
    topk_ids_local,
    lora,
    *,
    scaling,
    provider="auto",
    base_config=None,
    local_only=False,
    overlap=None,
    num_global_experts=None,
):
    """ML1 composition (inference)."""
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None, False, or True")
    _validate_base(x, w13, w2, topk_ids_local, topk_weights)
    t, h = x.shape
    e, twice_i, _ = w13.shape
    top_k = topk_ids_local.shape[1]
    rank = 0 if lora is None else lora.rank
    z13 = z2 = None
    if t == 0:
        return torch.empty_like(x)
    naive = x.is_cuda and t * top_k * 4 <= e
    masked = not (
        local_only and topk_ids_local.dtype == torch.int32 and topk_ids_local.is_contiguous()
    )
    skipped = masked and x.is_cuda and not naive and e >= 64
    resident = (
        _lora_workspace.ENABLED
        and x.device.type == "cuda"
        and overlap is not True
        and not any(v.requires_grad for v in (x, topk_weights, *_lora_tensors(lora)))
    )
    bm = None
    split_count = 0
    if x.is_cuda:
        capability, sms = _lora_provider.device_info(x.device)
        # RTX 5090 (sm_120, < 180 SMs); the RTX PRO 6000 and GB10 keep the defaults.
        rtx5090 = capability == (12, 0) and sms < 180
        cfg = base_config_for(
            t, shared_lora_tiles=lora is not None and rtx5090, gb10=capability == (12, 1)
        )
        if base_config is not None:
            cfg.update(base_config)
        if lora is not None and (not naive or _lora_provider.uses_block_path(t * top_k, rank, sms)):
            pairs = t * top_k
            if num_global_experts is not None and not local_only and not naive:
                pairs = -(-pairs * e // num_global_experts)  # expected local routes
            bm = _lora_provider.block_size(pairs, e, rank or 16)
            if not overlap:
                split_count = (
                    _lora_provider.split_chunks(pairs, rank, sms)
                    if _SPLIT_ROUTED is None
                    else _SPLIT_ROUTED
                )
    if resident:
        shapes = {
            "c13": ((t * top_k * max(twice_i, h),), x.dtype),
            "c2": ((t * top_k, twice_i // 2), x.dtype),
        }
        if masked:
            shapes.update(
                valid=(topk_ids_local.shape, torch.bool),
                below=(topk_ids_local.shape, torch.bool),
                masked_ids=(topk_ids_local.shape, topk_ids_local.dtype),
                id_fill=((), topk_ids_local.dtype),
                rw_fill=((), topk_weights.dtype),
                ids=(topk_ids_local.shape, torch.int32),
                rw=(topk_weights.shape, topk_weights.dtype),
            )
        if split_count:
            shapes.update(
                z13=((t * top_k, 2, rank), torch.bfloat16),
                z2=((t * top_k, 1, rank), torch.bfloat16),
            )
        scratch = dict(zip(shapes, _lora_workspace.carve(x.device, shapes.values()), strict=True))
        c1 = scratch["c13"][: t * top_k * twice_i].view(t * top_k, twice_i)
        c3 = scratch["c13"][: t * top_k * h].view(t, top_k, h)
        c2 = scratch["c2"]
    if resident and masked:
        valid = scratch["valid"]
        torch.ge(topk_ids_local, 0, out=valid)
        torch.lt(topk_ids_local, e, out=scratch["below"])
        torch.logical_and(valid, scratch["below"], out=valid)
        torch.where(
            valid,
            topk_ids_local,
            scratch["id_fill"].fill_(e if skipped else 0),
            out=scratch["masked_ids"],
        )
        ids = scratch["ids"].copy_(scratch["masked_ids"])
        rw = topk_weights.contiguous()
        if not skipped:
            rw = torch.where(valid, topk_weights, scratch["rw_fill"].zero_(), out=scratch["rw"])
    elif not masked:
        ids, rw = topk_ids_local, topk_weights.contiguous()
    else:
        valid = (topk_ids_local >= 0) & (topk_ids_local < e)
        # The alignment kernel's single-block path (few experts, small batches)
        # mishandles ids in its extra last bucket, so park remote pairs there only
        # with >= 64 local experts (the multi-block path, e.g. 256 or 128 under EP2).
        if naive or not x.is_cuda or e < 64:
            # Per-pair base blocks cannot skip a pair: compute it on expert 0, weight 0.
            ids = torch.where(valid, topk_ids_local, 0).to(torch.int32)
            rw = torch.where(valid, topk_weights, 0).contiguous()
        else:
            # Alignment sorts id E into its extra last bucket and excludes that bucket
            # from num_tokens_post_padded, so the base GEMMs never visit remote pairs.
            ids = torch.where(valid, topk_ids_local, e).to(torch.int32)
            rw = topk_weights.contiguous()
    # The base alignment leaves remote rows unwritten only on the skipped path.
    if not resident:
        c1 = torch.empty((t * top_k, twice_i), dtype=x.dtype, device=x.device)
        c3 = (torch.zeros if skipped else torch.empty)(
            (t, top_k, h), dtype=x.dtype, device=x.device
        )
    fuse_act = lora is not None and not naive and _uses_block_projection(x, c1, lora, provider)
    split_count = split_count if fuse_act else 0
    shrink13 = None
    alignment = {}
    if x.is_cuda:
        if naive:
            sorted_ids, padded = _naive_alignment(t * top_k, cfg["BLOCK_SIZE_M"], x.device)
            experts = ids.view(-1)
        else:
            sorted_ids, experts, padded = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], e)
            sorted_ids = grid_capacity(sorted_ids, t * top_k, cfg["BLOCK_SIZE_M"])
        if bm is not None:
            # LoRA tiles prefer few rows per expert block; a second alignment is far
            # cheaper than mostly-empty LoRA blocks.
            lora_sorted, lora_experts, lora_padded = sorted_ids, experts, padded
            if naive or bm != cfg["BLOCK_SIZE_M"]:
                lora_sorted, lora_experts, lora_padded = moe_align_block_size(ids, bm, e)
                lora_sorted = grid_capacity(lora_sorted, t * top_k, bm)
            alignment = {
                "sorted_token_ids": lora_sorted,
                "expert_ids": lora_experts,
                "num_tokens_post_padded": lora_padded,
                "block_m": bm,
            }
        if fuse_act and overlap:
            shrink13 = _shrink_async(
                c1, x, lora, topk_ids_local, scaling=scaling, top_k=top_k, **alignment
            )
        elif split_count:
            shrink13 = _shrink_now(
                c1,
                x,
                lora,
                topk_ids_local,
                scaling=scaling,
                top_k=top_k,
                z=scratch["z13"] if resident else None,
                **alignment,
            )
        fused_moe_kernel_triton(
            x,
            w13,
            c1.view(t, top_k, twice_i),
            rw,
            ids,
            sorted_ids,
            experts,
            padded,
            False,
            top_k,
            cfg,
            x.dtype,
        )
    else:
        for expert in range(e):
            rows = torch.where(ids.reshape(-1) == expert)[0]
            c1[rows] = (x[rows // top_k].float() @ w13[expert].float().T).to(x.dtype)
    if fuse_act:
        # Small-expert EP routes use expert zero with weight zero in base2.
        # Give those unread-by-LoRA rows defined inputs (NaN * 0 is not zero).
        if resident:
            if masked and not skipped:
                c2.zero_()
        else:
            c2 = (torch.zeros if masked and not skipped else torch.empty)(
                (t * top_k, twice_i // 2), dtype=x.dtype, device=x.device
            )
        if shrink13 is not None:
            z13, done13 = shrink13
            if done13 is not None:
                torch.cuda.current_stream(x.device).wait_event(done13)
            _apply(
                c1,
                x,
                lora,
                topk_ids_local,
                scaling=scaling,
                top_k=top_k,
                provider="cute",
                **alignment,
                z_out=z13,
                activated=c2,
                stage="expand",
                chunks=split_count or None,
            )
        else:
            moe_lora_w13_act_(
                c2,
                c1,
                x,
                lora,
                topk_ids_local,
                scaling=scaling,
                top_k=top_k,
                provider=provider,
                **alignment,
            )
    else:
        if lora is not None:
            moe_lora_w13_(
                c1,
                x,
                lora,
                topk_ids_local,
                scaling=scaling,
                top_k=top_k,
                provider=provider,
                **alignment,
                z_out=z13,
            )
        c2 = _silu_and_mul(c1, out=c2) if resident else _silu_and_mul(c1)
    # c1 is dead only after activation; c3 reuses its bytes on the resident path.
    if resident and skipped:
        c3.zero_()
    shrink2 = None
    if (
        (overlap or split_count)
        and lora is not None
        and not naive
        and _uses_block_projection(x, c3.view(-1, h), lora, provider, down=True)
    ):
        split = {} if overlap else {"z": scratch["z2"] if resident else None}
        shrink2 = (_shrink_async if overlap else _shrink_now)(
            c3.view(-1, h),
            c2,
            lora,
            topk_ids_local,
            scaling=scaling,
            top_k=top_k,
            down=True,
            **alignment,
            **split,
        )
    if x.is_cuda:
        fused_moe_kernel_triton(
            c2, w2, c3, rw, ids, sorted_ids, experts, padded, False, 1, cfg, x.dtype
        )
    else:
        for expert in range(e):
            rows = torch.where(ids.reshape(-1) == expert)[0]
            c3.view(-1, h)[rows] = (c2[rows].float() @ w2[expert].float().T).to(x.dtype)
    if shrink2 is not None:
        z2, done2 = shrink2
        if done2 is not None:
            torch.cuda.current_stream(x.device).wait_event(done2)
        _apply(
            c3.view(-1, h),
            c2,
            lora,
            topk_ids_local,
            scaling=scaling,
            top_k=top_k,
            down=True,
            provider="cute",
            **alignment,
            z_out=z2,
            stage="expand",
            chunks=split_count or None,
        )
    elif lora is not None:
        moe_lora_w2_(
            c3.view(-1, h),
            c2,
            lora,
            topk_ids_local,
            None,
            scaling=scaling,
            top_k=top_k,
            provider=provider,
            **alignment,
            z_out=z2,
        )
    # Unweighted BF16 routes (the down delta in each, one rounding), combined as vLLM's
    # finalize; parked remote routes weigh 0 and skipped ones are zero rows.
    if x.is_cuda:
        out = torch.empty_like(x)
        moe_sum_reduce_triton(c3, out, ids, topk_weights=rw)
    else:
        out = (c3.float() * rw[..., None].float()).sum(dim=1).to(x.dtype)
    return out


def _check_joined_shared(x, shared, shared_gate, shared_expert):
    if shared_expert is not None:
        raise ValueError("pass shared_expert or shared/shared_gate, not both")
    if (
        shared.shape != x.shape
        or shared.dtype != x.dtype
        or shared.device != x.device
        or shared.stride(1) != 1
        or shared_gate.shape != (x.shape[0], 1)
        or shared_gate.dtype != x.dtype
        or shared_gate.device != x.device
    ):
        raise ValueError("shared must be [T, H] and shared_gate [T, 1], both like x")


def _fused_experts_lora_block_fp8(
    x, w13, w2, w13_scale, w2_scale, topk_weights, ids, lora, *, scaling, provider, base_config,
    local_only, adapter_id, shared=None, shared_gate=None,
):  # fmt: skip
    if w13_scale is None or w2_scale is None:
        raise ValueError("pass both w13_scale and w2_scale for a block-FP8 base")
    if provider not in ("auto", "triton"):
        raise ValueError("block-FP8 bases run on the triton provider")
    _check_frozen_base(("w13", w13), ("w2", w2), ("w13_scale", w13_scale), ("w2_scale", w2_scale))
    if not supported(x.device, x.dtype):
        raise ValueError("block-FP8 MoE LoRA needs BF16 x on CUDA with compute capability >= 8.9")
    check_tensor(x, "x", device=x.device)
    for name, value in (("w13", w13), ("w2", w2)):
        check_tensor(value, name, dtype=torch.float8_e4m3fn, device=x.device)
    for name, value in (("w13_scale", w13_scale), ("w2_scale", w2_scale)):
        check_tensor(value, name, dtype=value.dtype, device=x.device)
        if value.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"{name} must be FP32 or BF16")
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("expected x [T,H], w13 [E,2I,H], w2 [E,H,I]")
    e, twice_i, h = w13.shape
    groups = lambda n: -(-n // 128)
    if (
        h != x.shape[1]
        or twice_i % 16
        or w2.shape != (e, h, twice_i // 2)
        or w13_scale.shape != (e, groups(twice_i), groups(h))
        or w2_scale.shape != (e, groups(h), groups(twice_i // 2))
    ):
        raise ValueError("incompatible block-FP8 base or scale shapes")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk ids must be int32 or int64")
    check_tensor(ids, "ids", dtype=ids.dtype, device=x.device)
    check_tensor(topk_weights, "topk_weights", dtype=torch.float32, device=x.device)
    if ids.ndim != 2 or ids.shape[0] != x.shape[0] or topk_weights.shape != ids.shape:
        raise ValueError("routing tensors must have shape [T,top_k]")
    if lora is not None:
        lora = lora.adapter(adapter_id)
        check_routed(lora, x, w13, w2)
    config = block_fp8_moe_config(
        x.shape[0],
        e,
        twice_i,
        h,
        ids.shape[1],
        capability=torch.cuda.get_device_capability(x.device),
    )
    config.update(base_config or {})
    if _trains(x, topk_weights, *_lora_tensors(lora)):
        # STE through the activation quantizers; the frozen FP8 base gets no gradient.
        factors = training_factors(lora, x, w13, w2)
        options = {
            "base_config": config,
            "local_only": local_only,
            "w13_scale": w13_scale,
            "w2_scale": w2_scale,
        }
        scale = scale_tensor(scaling, x.device).detach()
        return TritonBlockLoRA.apply(x, w13, w2, topk_weights, ids, scale, *factors, options)
    return _triton_ops().fused_experts_lora(
        x,
        w13,
        w2,
        topk_weights,
        ids,
        lora,
        scaling=scaling,
        base_config=config,
        local_only=local_only,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        shared=shared,
        shared_gate=shared_gate,
    )


def _validate_base(x, w13, w2, ids, weights):
    _check_frozen_base(("w13", w13), ("w2", w2))
    for name, value in (("x", x), ("w13", w13), ("w2", w2)):
        check_tensor(value, name, device=x.device)
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("expected x [T,H], w13 [E,2I,H], w2 [E,H,I]")
    if (
        w13.shape[0] <= 0
        or w13.shape[2] != x.shape[1]
        or w13.shape[1] % 16
        or w2.shape != (w13.shape[0], x.shape[1], w13.shape[1] // 2)
    ):
        raise ValueError("incompatible base factor shapes")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk ids must be int32 or int64")
    check_tensor(ids, "ids", dtype=ids.dtype, device=x.device)
    check_tensor(weights, "topk_weights", dtype=torch.float32, device=x.device)
    if (
        ids.ndim != 2
        or ids.shape[0] != x.shape[0]
        or ids.shape[1] <= 0
        or weights.shape != ids.shape
    ):
        raise ValueError("routing tensors must have shape [T,top_k]")


def _check_frozen_base(*weights):
    for name, weight in weights:
        if weight.requires_grad:
            raise ValueError(f"base expert weights are frozen; {name} requires_grad=True")


def fused_experts_lora_training_bytes(
    tokens: int,
    *,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    rank: int,
    local_only: bool = False,
    device: torch.device | str = "cuda",
    expert_config=None,
    block_fp8: bool = False,
    base_config=None,
) -> tuple[int, int]:
    """``(saved, backward)`` bytes of ``fused_experts_lora`` training on CUDA over ``tokens``
    rows: BF16 base (``block_fp8``: E4M3 base with ``base_config`` as passed to
    ``fused_experts_lora``) of ``experts`` local experts, every factor and the input
    trainable, no shared expert. ``saved`` is what the forward keeps for the backward, the
    input, routing weights and ids included; ``backward`` the most the backward holds at
    once, every gradient included. Bytes as the caching allocator holds the tensors (an
    unsplit block may add up to 1 MiB per allocation of 1 MiB or more); routing does not
    enter."""
    device = torch.device(device)
    fp8_block_m = None
    if block_fp8:
        config = block_fp8_moe_config(
            tokens, experts, 2 * intermediate, hidden, top_k,
            capability=device_info(device)[0],
        )  # fmt: skip
        fp8_block_m = {**config, **(base_config or {})}["BLOCK_SIZE_M"]
    return _lora_train().training_bytes(
        tokens, hidden, intermediate, experts, top_k, rank, device,
        local_only=local_only, expert_config=expert_config, fp8_block_m=fp8_block_m,
    )  # fmt: skip


def shared_expert_lora(
    x, w_gate_up, w_down, lora_shared, scaling, *, provider="auto", adapter_id=0
):
    """Shared MLP before the caller's shared-expert sigmoid gate. The base GEMMs are the BF16
    library GEMM on CUDA (FP32 accumulate, one rounding; no weight upcast), FP32 on CPU.

    Trainable inputs use ML2 autograd and BF16 base GEMMs when CUDA resolves to
    CuTe; CPU and the torch provider retain the differentiable FP32 reference.
    """
    _check_frozen_base(("w_gate_up", w_gate_up), ("w_down", w_down))
    _validate_shared_base(x, w_gate_up, w_down)
    factors = training_factors(
        lora_shared, x, w_gate_up, w_down, shared=True, adapter_id=adapter_id
    )
    if provider not in PROVIDERS:
        raise ValueError("provider must be auto, cute, triton, or torch")
    rank = None if factors[0] is None else factors[0].shape[-2]
    kernel = None if not x.is_cuda else _resolve(provider, x, rank)
    trains = _trains(x, *_lora_tensors(lora_shared))
    if not trains and factors[0] is not None:
        factors = tuple(f.to(x.dtype) for f in factors)  # FP32 masters: one cast for inference
    if trains and kernel in ("triton", "cute"):
        # Unsloth's shared expert: HF's MLP with PEFT LoRA Linears, BF16 torch ops (autograd).
        scale = scale_tensor(scaling, x.device).detach()
        return _lora_train().shared_mlp(x, w_gate_up, w_down, factors, scale)
    if kernel == "triton":
        # FP32 rows to one rounding.
        with torch.no_grad():
            out, _ = _triton_ops().shared_rows(
                x, w_gate_up, w_down, None, _shared_slices(*factors),
                scale_tensor(scaling, x.device), base_config_for(x.shape[0]),
            )  # fmt: skip
        return out.to(x.dtype)
    if trains:
        scale = scale_tensor(scaling, x.device).detach()
        return _shared_expert_lora(
            x, w_gate_up, w_down, lora_shared, scale, provider="torch", adapter_id=adapter_id
        )
    with torch.no_grad():
        return _shared_expert_lora(
            x, w_gate_up, w_down, lora_shared, scaling, provider=provider, adapter_id=adapter_id
        )


def _dense(x, w):
    """``x . w^T`` rounded once to x's dtype: the BF16 library GEMM (FP32 accumulate, no weight
    copy) on CUDA, FP32 on CPU (reference)."""
    if x.is_cuda:
        return torch.mm(x, w.t())
    return (x.float() @ w.float().T).to(x.dtype)


def _shared_expert_lora(x, w_gate_up, w_down, lora_shared, scaling, *, provider, adapter_id):
    _validate_shared_base(x, w_gate_up, w_down)
    c1 = _dense(x, w_gate_up)
    if lora_shared is not None and _uses_block_projection(
        x, c1, lora_shared, provider, shared=True, adapter_id=adapter_id
    ):
        c2 = torch.empty((x.shape[0], w_down.shape[1]), dtype=x.dtype, device=x.device)
        shared_lora_w13_act_(
            c2, c1, x, lora_shared, scaling=scaling, adapter_id=adapter_id, provider=provider
        )
    else:
        if lora_shared is not None:
            shared_lora_w13_(
                c1, x, lora_shared, scaling=scaling, adapter_id=adapter_id, provider=provider
            )
        gate, up = c1.float().chunk(2, dim=-1)
        c2 = (F.silu(gate) * up).to(x.dtype)
    out = _dense(c2, w_down)
    if lora_shared is not None:
        shared_lora_w2_(
            out, c2, lora_shared, scaling=scaling, adapter_id=adapter_id, provider=provider
        )
    return out


def _validate_shared_base(x, w_gate_up, w_down):
    for name, t in (("x", x), ("w_gate_up", w_gate_up), ("w_down", w_down)):
        check_tensor(t, name, device=x.device)
    if (
        x.ndim != 2
        or w_gate_up.ndim != 2
        or w_down.ndim != 2
        or w_gate_up.shape[1] != x.shape[1]
        or w_down.shape != (x.shape[1], w_gate_up.shape[0] // 2)
        or w_gate_up.shape[0] % 16
    ):
        raise ValueError("incompatible shared base shapes")


class TritonBlockLoRA(torch.autograd.Function):
    """Routed-expert MoE LoRA training on the Triton kernels. A BF16 base runs Unsloth's order
    (``triton._lora_train.routed_forward``); a block-FP8 base (``w13_scale`` in options) its
    inference composition. The forward's state (gate/up rows, low-rank inputs, alignments)
    is kept for the backward, which runs the deterministic Triton steps (no atomics,
    fixed-order sums)."""

    @staticmethod
    def forward(ctx, x, w13, w2, rw, ids, scale, *args):
        *factors, options = args
        out, state = _routed_training_forward(x, w13, w2, rw, ids, scale, factors, options)
        ctx.save_for_backward(x, w13, w2, rw, ids, scale, *factors)
        ctx.options, ctx.state = options, state
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, w13, w2, rw, ids, scale, *factors = ctx.saved_tensors
        need = ctx.needs_input_grad
        state, ctx.state = ctx.state, None
        weights = rw if "w13_scale" in ctx.options else rw.to(torch.bfloat16).float()
        dx, dw, *df = backward_steps(
            dy.contiguous(), x, w13, w2, weights, ids, factors, scale, state,
            (need[0], need[3], *need[6:10]), steps=_triton_ops().TritonSteps,
        )  # fmt: skip
        return dx, None, None, None if dw is None else dw.to(rw.dtype), None, None, *df, None


def _routed_training_forward(x, w13, w2, rw, ids, scale, factors, options):
    # FP32 master factors are cast to BF16 for this forward only (the Function keeps them).
    lora = None if factors[0] is None else MoELoRAWeights(*factors).to(x.dtype)
    if "w13_scale" in options:
        return _triton_ops().fused_experts_lora(
            x, w13, w2, rw, ids, lora, scaling=scale, return_state=True, **options
        )
    return _lora_train().routed_forward(x, w13, w2, rw, ids, lora, scale, **options)


def _shared_slices(a13, b13, a2, b2):
    """Shared factors as one-expert FactorSlices (a13, b13, a2, b2), or None."""
    if a13 is None:
        return None
    return (
        FactorSlices((a13[0][None], a13[1][None])),
        FactorSlices((b13[0][None], b13[1][None])),
        FactorSlices((a2[None],)),
        FactorSlices((b2[None],)),
    )
