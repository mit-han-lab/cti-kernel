# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kernel/moe_impl.py`),
# which builds on SGLang/vLLM's fused MoE launcher; see THIRD_PARTY_NOTICES.
# Block-FP8 operands contributed by Waterpine.

import functools
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Tuple

import torch

BLOCK_FP8_PROVIDERS = ("triton-fp8",)


@dataclass(frozen=True)
class GateUpLoRA:
    """A per-expert LoRA delta added to FC1's gate and up in FP32 before the fused activation
    (``fused_moe_kernel_triton(swiglu_quant_scale=..., gate_up_lora=...)``):
    ``gate[route, n] += scale * low[route, 0] . b_gate[e, n]``, up likewise with slot 1, e
    being the route's local expert. ``low`` FP32 ``[routes, 2, r]`` in route order (``ids``
    flattened); ``b_gate`` / ``b_up`` ``[E_local, I, r]`` in the compute dtype with equal
    strides and unit last stride; ``scale`` FP32 ``[1]`` on the device. r <= 128."""

    low: torch.Tensor
    b_gate: torch.Tensor
    b_up: torch.Tensor
    scale: torch.Tensor


def _check_swiglu_extras(C, compute_type, routes, num_experts, act, lora):
    width = C.shape[-1]
    if act is not None and (
        act.dtype != compute_type
        or act.ndim != 2
        or tuple(act.shape) != (routes, width)
        or act.stride(1) != 1
        or act.device != C.device
    ):
        raise ValueError(
            f"swiglu_act must be {compute_type} [{routes}, {width}], unit column stride"
        )
    if lora is None:
        return
    low, b_gate, b_up, scale = lora.low, lora.b_gate, lora.b_up, lora.scale
    r = low.shape[-1] if low.ndim == 3 else -1
    if (
        low.dtype != torch.float32
        or tuple(low.shape) != (routes, 2, r)
        or low.stride(2) != 1
        or not 1 <= r <= 128
        or any(b.dtype != compute_type for b in (b_gate, b_up))
        or any(tuple(b.shape) != (num_experts, width, r) for b in (b_gate, b_up))
        or b_gate.stride() != b_up.stride()
        or b_gate.stride(2) != 1
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or any(t.device != C.device for t in (low, b_gate, b_up, scale))
    ):
        raise ValueError(
            f"gate_up_lora: low FP32 [{routes}, 2, r<=128]; b_gate/b_up {compute_type} "
            f"[{num_experts}, {width}, r] with equal strides; scale FP32 [1]; one device"
        )


def block_fp8_moe_provider(device: torch.device | str | int) -> str:
    """The block-FP8 path ``fused_moe_kernel_triton`` takes on ``device``: E4M3 tensor-core
    dots, compute capability 8.9 and newer."""
    capability = torch.cuda.get_device_capability(device)
    if capability >= (8, 9):
        return "triton-fp8"
    raise RuntimeError(
        f"block-FP8 MoE needs E4M3 tensor cores (compute capability 8.9 or newer), not {capability}"
    )


def _check_block_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    a_scale: torch.Tensor | None,
    b_scale: torch.Tensor | None,
    config: Dict[str, Any],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swiglu_quant_scale: torch.Tensor | None = None,
) -> None:
    """Metadata checks of the block-FP8 operands and of every extent the kernel indexes (no
    device work, no host sync). Route ``r < R = topk_ids.numel()`` reads row ``r // top_k`` of
    ``A`` and ``a_scale`` and writes ``C`` row ``r`` of the ``[R, N]`` flattening of ``C``."""
    if a_scale is None or b_scale is None:
        raise ValueError("block-FP8 MoE needs both a_scale and b_scale")
    for name, tensor in (("A", A), ("B", B), ("a_scale", a_scale), ("b_scale", b_scale)):
        if tensor.requires_grad:
            raise ValueError(f"block-FP8 MoE is inference-only: {name} must not require grad")
    routing = (("topk_weights", topk_weights), ("topk_ids", topk_ids))
    routing += (("sorted_token_ids", sorted_token_ids), ("expert_ids", expert_ids))
    for name, tensor in (("A", A), ("B", B), ("a_scale", a_scale), ("b_scale", b_scale), *routing):
        if tensor.device != C.device:
            raise ValueError(f"block-FP8 MoE: {name} is on {tensor.device}, C on {C.device}")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError(f"block-FP8 MoE needs E4M3 A and B, not {A.dtype} and {B.dtype}")
    if a_scale.dtype != torch.float32 or b_scale.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("block-FP8 MoE needs FP32 a_scale and FP32 or BF16 b_scale")
    swiglu = swiglu_quant_scale is not None
    if C.dtype not in ((torch.float8_e4m3fn,) if swiglu else (torch.bfloat16, torch.float16)):
        kind = "E4M3" if swiglu else "BF16 or FP16"
        raise TypeError(f"block-FP8 MoE writes {kind} C, not {C.dtype}")
    if A.ndim != 2 or B.ndim != 3 or A.stride(1) != 1 or B.stride(2) != 1:
        raise ValueError("block-FP8 MoE needs A [rows, K] and B [E, N, K] with unit K stride")
    E, N, K = B.shape
    groups = -(-K // 128)
    if A.shape[1] != K:
        raise ValueError(f"block-FP8 MoE: A has {A.shape[1]} columns, B has K = {K}")
    if tuple(a_scale.shape) != (A.shape[0], groups):
        raise ValueError(f"block-FP8 MoE: a_scale must be {(A.shape[0], groups)}")
    if tuple(b_scale.shape) != (E, -(-N // 128), groups):
        raise ValueError(f"block-FP8 MoE: b_scale must be {(E, -(-N // 128), groups)}")
    if 128 % config["BLOCK_SIZE_K"]:
        raise ValueError("block-FP8 MoE needs BLOCK_SIZE_K dividing the 128-column group")
    routes = topk_ids.numel()
    if top_k < 1 or A.shape[0] * top_k != routes:
        raise ValueError(
            f"block-FP8 MoE: A must have {routes} / top_k rows ({routes} routes, top_k = "
            f"{top_k}), not {A.shape[0]}"
        )
    width = N // 2 if swiglu else N
    if C.ndim != 3 or C.shape[0] * C.shape[1] != routes or C.shape[2] != width:
        raise ValueError(
            f"block-FP8 MoE: C must be [M, top_k, {width}] holding {routes} routes, not "
            f"{tuple(C.shape)}"
        )
    if swiglu and (
        mul_routed_weight
        or config["BLOCK_SIZE_N"] != 256
        or N % 256
        or K % 128
        or swiglu_quant_scale.dtype != torch.float32
        or tuple(swiglu_quant_scale.shape) != (routes, width // 128)
        or swiglu_quant_scale.stride(1) != 1
        or C.stride(2) != 1
    ):
        raise ValueError(
            "block-FP8 MoE SwiGLU quantization: FC1 only (no routed weight), BLOCK_SIZE_N 256, "
            "I and K multiples of 128, FP32 scale [routes, I / 128] with unit column stride"
        )
    if C.shape[0] > 1 and C.stride(0) != C.shape[1] * C.stride(1):
        raise ValueError("block-FP8 MoE: C's first two dimensions must flatten to route rows")
    if mul_routed_weight and (topk_weights.numel() != routes or not topk_weights.is_contiguous()):
        raise ValueError(f"block-FP8 MoE: topk_weights must be contiguous with {routes} routes")
    if sorted_token_ids.ndim != 1 or expert_ids.ndim != 1:
        raise ValueError("block-FP8 MoE: sorted_token_ids and expert_ids must be vectors")
    for name, tensor in (("sorted_token_ids", sorted_token_ids), ("expert_ids", expert_ids)):
        if tensor.stride(0) != 1:
            raise ValueError(f"block-FP8 MoE: {name} must be contiguous (unit stride)")
    if expert_ids.numel() < -(-sorted_token_ids.numel() // config["BLOCK_SIZE_M"]):
        raise ValueError("block-FP8 MoE: expert_ids needs one entry per BLOCK_SIZE_M block")


def block_fp8_moe_config(
    num_tokens: int,
    num_experts: int,
    N: int,
    K: int,
    top_k: int,
    *,
    capability: tuple[int, int],
):
    """A launch configuration for the block-FP8 ``fused_moe_kernel_triton`` (BLOCK_SIZE_K is
    the 128-column group), for ``moe_align_block_size(..., config["BLOCK_SIZE_M"], ...)``.
    ``capability`` (CC major, minor) 12.1 (GB10) selects configurations from GB10 sweeps of the
    Qwen3.5-35B-A3B MoE forward: four stages for small batches, 64 x 64 tiles in two stages for
    large ones (64 x 128 above 1024 tokens)."""
    small = num_tokens * top_k <= 4 * num_experts
    gb10 = tuple(capability) == (12, 1)
    if small:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4 if gb10 else 3,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64 if gb10 and num_tokens <= 1024 else 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2 if gb10 else 3,
    }


_IDENTITY_MAPS = {}
_IDENTITY_MAP_LOCK = RLock()


def _identity_expert_map(device: torch.device, num_experts: int) -> torch.Tensor:
    """Reuse eager identity maps, never retaining tensors from a capture pool."""
    key = (device, num_experts)
    with torch.cuda.device(device), _IDENTITY_MAP_LOCK:
        mapping = _IDENTITY_MAPS.get(key)
        if mapping is None:
            mapping = torch.arange(num_experts, dtype=torch.int32, device=device)
            if not torch.cuda.is_current_stream_capturing():
                _IDENTITY_MAPS[key] = mapping
        return mapping


@functools.lru_cache(maxsize=None)
def _pdl(device: torch.device) -> Dict[str, Any]:
    """Programmatic dependent launch on sm_90+ (the kernels wait before their first load)."""
    if torch.cuda.get_device_capability(device)[0] >= 9:
        return {"USE_GDC": True, "launch_pdl": True}
    return {}


def _check_expert_map(expert_map, device, num_local_experts, num_global_experts=None):
    """Metadata only; map values and global route bounds are caller preconditions."""
    if num_global_experts is not None and (
        not isinstance(num_global_experts, int) or num_global_experts < num_local_experts
    ):
        raise ValueError("num_global_experts must be an integer >= the local expert count")
    if expert_map is None:
        if num_global_experts not in (None, num_local_experts):
            raise ValueError("num_global_experts requires expert_map for nonlocal experts")
        return num_local_experts
    if not isinstance(expert_map, torch.Tensor) or expert_map.dtype != torch.int32:
        raise TypeError("expert_map must be an int32 tensor")
    if expert_map.ndim != 1:
        raise ValueError("expert_map must be 1-D")
    if not expert_map.is_contiguous():
        raise ValueError("expert_map must be contiguous")
    if expert_map.device != device:
        raise ValueError("expert_map must share the weights' device")
    required = num_global_experts if num_global_experts is not None else num_local_experts
    if expert_map.numel() < required:
        raise ValueError("expert_map must cover all global experts (at least the local count)")
    return expert_map.numel()


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    *,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    block_fp8_provider: str | None = None,
    remote_experts: bool = False,
    expert_map: torch.Tensor | None = None,
    num_global_experts: int | None = None,
    swiglu_quant_scale: torch.Tensor | None = None,
    swiglu_act: torch.Tensor | None = None,
    gate_up_lora: GateUpLoRA | None = None,
) -> None:
    """Grouped expert GEMM ``C[route] = A[token] . B[expert]^T`` over the routes that
    ``moe_align_block_size`` sorted. With ``expert_map`` alignment every active block is
    local: remote routes get no block, their ``C`` rows are not written, and the masked
    ``moe_sum_reduce_triton`` never reads them. Without it, a block whose expert is -1
    (routes already mapped to local IDs, -1 for another rank's) writes zero rows.

    Alternatively, pass ``expert_map`` here for global block IDs from route blocks
    or ``moe_align_block_size(ids, block_size, num_global_experts)`` without an
    alignment-level map. It maps global IDs to local weight/scale rows; -1 skips
    the GEMM and writes zero rows. Do not map already-local alignment output again.
    Map values and route bounds are caller preconditions; ``num_global_experts``
    optionally checks the minimum map extent using metadata only.
    ``remote_experts`` is accepted for compatibility: the RC already handles -1
    for every dtype, regardless of this flag. No extra kernel constexpr is needed.

    With ``a_scale`` and ``b_scale`` the operands are block FP8: ``A`` E4M3 ``[rows, K]``
    with FP32 ``a_scale`` ``[rows, ceil(K / 128)]`` (``quantize_per_token_group_fp8``) and
    ``B`` E4M3 ``[E, N, K]`` with FP32 or BF16 ``b_scale`` ``[E, ceil(N / 128), ceil(K /
    128)]``, scales multiplying as in ``ctikernels.linear.block_fp8``. Each group's dot
    accumulates in FP32 and is scaled in FP32. ``block_fp8_provider`` forces a
    ``BLOCK_FP8_PROVIDERS`` path (default: ``block_fp8_moe_provider(C.device)``).

    ``swiglu_quant_scale`` (block FP8 FC1 only, ``config["BLOCK_SIZE_N"] == 256``) fuses the
    activation: ``C`` is then E4M3 ``[M, top_k, I]`` receiving the 1x128 quantization of
    SiLU(gate) * up, and ``swiglu_quant_scale`` FP32 ``[M * top_k, I / 128]`` its scales, the
    bits of the unfused FC1 + ``silu_and_mul_quantize_fp8``. With it, ``swiglu_act``
    (compute dtype ``[M * top_k, I]``) also receives the rounded product, and ``gate_up_lora``
    (``GateUpLoRA``) adds a per-expert LoRA delta to gate and up in FP32 before the activation.
    """
    if (swiglu_act is not None or gate_up_lora is not None) and swiglu_quant_scale is None:
        raise ValueError("swiglu_act and gate_up_lora need swiglu_quant_scale")
    if expert_map is not None or num_global_experts is not None:
        _check_expert_map(expert_map, B.device, B.shape[0], num_global_experts)
    block_fp8 = a_scale is not None or b_scale is not None
    if block_fp8:
        _check_block_fp8(
            A, B, C, a_scale, b_scale, config, topk_weights, topk_ids, sorted_token_ids,
            expert_ids, mul_routed_weight, top_k, swiglu_quant_scale,
        )  # fmt: skip
    if swiglu_quant_scale is not None:
        _check_swiglu_extras(
            C, compute_type, topk_ids.numel(), B.shape[0], swiglu_act, gate_up_lora
        )
    # topk_ids supplies only numel(); num_tokens_post_padded is a single scalar load.
    # Neither is indexed as a packed array by the kernel.
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        -(-sorted_token_ids.shape[0] // META["BLOCK_SIZE_M"])
        * -(-B.shape[1] // META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    if block_fp8:
        provider = block_fp8_provider or block_fp8_moe_provider(C.device)
        if provider not in BLOCK_FP8_PROVIDERS:
            raise ValueError(f"block_fp8_provider must be one of {BLOCK_FP8_PROVIDERS}")
    else:
        a_scale = b_scale = A
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        B,
        a_scale,
        b_scale,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0) if block_fp8 else 0,
        b_scale.stride(1),
        b_scale.stride(2) if block_fp8 else 0,
        C.stride(1),
        C.stride(2),
        BLOCK_FP8=block_fp8,  # type: ignore
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **({"expert_map_ptr": expert_map} if expert_map is not None else {}),
        **(
            {
                "c_scale_ptr": swiglu_quant_scale,
                "stride_csm": swiglu_quant_scale.stride(0),
                "SWIGLU_QUANT": True,
            }
            if swiglu_quant_scale is not None
            else {}
        ),
        **(
            {"act_ptr": swiglu_act, "stride_actm": swiglu_act.stride(0), "ACT_OUT": True}
            if swiglu_act is not None
            else {}
        ),
        **(_lora_kwargs(gate_up_lora) if gate_up_lora is not None else {}),
        **_pdl(A.device),
        GDC_EARLY=bool(_pdl(A.device)) and A.shape[0] <= 512,  # type: ignore
        **config,
    )


def _lora_kwargs(lora: GateUpLoRA) -> Dict[str, Any]:
    import triton

    rank = lora.low.shape[-1]
    return {
        "lora_low_ptr": lora.low,
        "lora_gate_ptr": lora.b_gate,
        "lora_up_ptr": lora.b_up,
        "lora_scale_ptr": lora.scale,
        "lora_rank": rank,
        "stride_llm": lora.low.stride(0),
        "stride_lls": lora.low.stride(1),
        "stride_lbe": lora.b_gate.stride(0),
        "stride_lbn": lora.b_gate.stride(1),
        "LORA_R": max(16, triton.next_power_of_2(rank)),  # tensor-core dot minimum
    }


def check_shared(output, shared, shared_gate) -> None:
    """The gated shared expert's contract: ``shared`` [T, H] (unit hidden stride) and its gate
    logits ``shared_gate`` [T, 1], both in ``output``'s dtype and device; or neither."""
    if (shared is None) != (shared_gate is None):
        raise ValueError("pass both shared and shared_gate, or neither")
    if shared is not None and (
        shared.shape != output.shape
        or shared.dtype != output.dtype
        or shared.stride(1) != 1
        or shared_gate.shape != (output.shape[0], 1)
        or shared_gate.dtype != output.dtype
        or shared.device != output.device
        or shared_gate.device != output.device
    ):
        raise ValueError("shared must be [T, H] and shared_gate [T, 1], both like output")


def moe_shared_join_(output: torch.Tensor, shared: torch.Tensor, shared_gate: torch.Tensor):
    """In place, ``output`` [T, H] (the routed block output, unit hidden stride) joins the gated
    shared expert as ``moe_sum_reduce_triton`` does: ``r(r(r(sigmoid(gate)) * shared) + routed)``
    in ``output``'s dtype, bit for bit. One launch; graph-capturable."""
    check_shared(output, shared, shared_gate)
    if output.ndim != 2 or output.stride(1) != 1 or not output.is_cuda:
        raise ValueError("output must be a CUDA [T, H] tensor with unit hidden stride")
    if output.shape[0] == 0:
        return output
    from .triton.fused_moe import moe_shared_join_kernel

    block = 1024
    moe_shared_join_kernel[(output.shape[0], -(-output.shape[1] // block))](
        output, output.stride(0), shared, shared.stride(0), shared_gate, shared_gate.stride(0),
        output.shape[1], BLOCK_DIM=block, num_warps=4,
    )  # fmt: skip
    return output


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    shared: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
    exchange=None,
    *,
    topk_weights: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Combine routes, never loading intermediate rows belonging to remote experts.

    ``input`` [T, top_k, H] is the output's dtype or FP32. With ``topk_weights`` [T, top_k]
    (FP32), its rows are unweighted and combine as FlashInfer's finalize, vLLM's default:
    ``acc = fma(w_k, y_k, acc)`` in route order from +0 and one rounding, skipping -1 (and,
    with ``expert_map``, remote) routes; this needs ``topk_ids``. Without weights the rows are
    summed in FP32 and rounded once.

    With ``shared`` [T, H] and its gate logits ``shared_gate`` [T, 1] (same dtype), the gated
    shared expert joins in vLLM's op order (Qwen2-MoE), each step rounded to the output dtype:
    ``out = r(r(r(sigmoid(gate)) * shared) + routed)``. With the int32 device scalar
    ``valid_tokens`` [1], rows at or past it are stored as +0.0.

    ``exchange`` (a ``collective.inference.Epilogue``) publishes the rounded output to the
    group's exchange. Returns the output tiling (rows, columns) that ``Exchange.reduce`` takes.
    """
    if expert_map is not None and topk_ids is None:
        raise ValueError("expert_map needs topk_ids")
    if topk_ids is not None and expert_map is None and topk_weights is None:
        raise ValueError("topk_ids without expert_map needs topk_weights")
    if topk_weights is not None and (
        topk_ids is None
        or topk_weights.shape != input.shape[:2]
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or topk_weights.device != input.device
    ):
        raise ValueError("topk_weights must be contiguous FP32 [T, top_k] with topk_ids")
    check_shared(output, shared, shared_gate)
    if valid_tokens is not None and (
        valid_tokens.shape != (1,)
        or valid_tokens.dtype != torch.int32
        or valid_tokens.device != output.device
    ):
        raise ValueError("valid_tokens must be an int32 [1] tensor on the output device")
    if input.ndim != 3 or output.shape != (input.shape[0], input.shape[2]):
        raise ValueError("expected input [T, top_k, H] and output [T, H]")
    if not input.is_contiguous() or not output.is_contiguous():
        raise ValueError("input and output must be contiguous")
    if input.dtype not in (output.dtype, torch.float32) or not output.is_floating_point():
        raise ValueError("input must be FP32 or the output's floating dtype")
    if topk_ids is not None:
        from .moe._align import _validate_routes

        if topk_ids.shape != input.shape[:2]:
            raise ValueError("topk_ids must match input's route dimensions")
        if expert_map is not None:
            _validate_routes(topk_ids, expert_map)
        elif topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if topk_ids.device != input.device:
            raise ValueError("routing tensors and input must share a device")
    if not input.is_cuda or output.device != input.device:
        raise ValueError("input and output must be CUDA tensors on the same device")
    BLOCK_DIM = 512 if input.shape[0] <= 16 else 2048
    if input.shape[0] == 0:
        return 1, BLOCK_DIM

    from .triton.fused_moe import moe_sum_reduce_kernel

    token_num, topk_num, hidden_dim = input.shape

    # Decode: more, narrower CTAs shorten the reduction's tail after the down projection.
    BLOCK_M = 1
    NUM_STAGE = 1
    num_warps = 4 if token_num <= 16 else 8

    grid = (
        -(-token_num // BLOCK_M),
        -(-hidden_dim // BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        topk_ids,
        expert_map,
        topk_weights,
        shared,
        0 if shared is None else shared.stride(0),
        shared_gate,
        0 if shared_gate is None else shared_gate.stride(0),
        valid_tokens,
        MASKED=expert_map is not None,
        WEIGHTED=topk_weights is not None,
        SHARED=shared is not None,
        VALID=valid_tokens is not None,
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
        **_pdl(input.device),
        GDC_EARLY=bool(_pdl(input.device)) and token_num <= 512,  # type: ignore
        **(
            {}
            if exchange is None
            else {
                "xch": exchange.desc,
                "EXCHANGE": True,
                "XCH_RANK": exchange.rank,
                "XCH_WORLD": exchange.world,
            }
        ),
    )
    return BLOCK_M, BLOCK_DIM


def _div_ceil(a: int, b: int) -> int:
    return -(-a // b)


def moe_align_block_size_bytes(n: int, block_size: int, num_experts: int) -> tuple[int, ...]:
    """Bytes of each allocation ``moe_align_block_size`` holds for ``n`` routes (no map)."""
    from .moe import align_allocations

    return align_allocations(
        n, block_size, num_experts, capacity_experts=num_experts + 1, invalid_bucket=True
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort and pad routed tokens per expert so every expert's tokens fill whole
    blocks of ``block_size`` for the grouped matrix multiplication.

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``: the
    flattened token indices ordered by expert with padding entries appended, the
    expert assigned to each block, and the padded total token count. With
    ``expert_map``, ``num_experts`` is the local weight count and remote routes
    receive no slots or blocks.
    """
    from .moe import moe_align_block_size_local

    # int64 ids (torch.topk indices) are read as int32, and ids are made contiguous, on both
    # paths.
    if topk_ids.dtype == torch.int64:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    if expert_map is not None:
        return moe_align_block_size_local(topk_ids, block_size, expert_map, num_experts)
    # -1 routes fill an extra bucket of expert -1 blocks (zero rows), so unmasked route sums
    # never read stale rows.
    return moe_align_block_size_local(
        topk_ids,
        block_size,
        _identity_expert_map(topk_ids.device, num_experts),
        num_experts,
        capacity_experts=num_experts + 1,
        identity=True,
        invalid_bucket=True,
    )
