# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Device-free metadata validation and hardware lookup for MoE providers."""

from functools import lru_cache

import torch

from ._configs import lookup


def admits_shape(
    kind, *, rows, hidden, experts, inter, local_experts, shared_inter, dtype, contiguous=True
):
    limit = {"rtx5090": 256, "pro6000": 256, "gb10": 1024}.get(kind, -1)
    return (
        0 <= rows <= limit
        and hidden == 2048
        and experts == 256
        and inter == 512
        and local_experts in (64, 128, 256)
        and shared_inter in (128, 256, 512)
        and dtype == torch.bfloat16
        and contiguous
    )


@lru_cache(maxsize=None)
def hardware(device):
    from ctikernels.linear._dispatch import device_class

    props = torch.cuda.get_device_properties(device)
    return device_class((props.major, props.minor), props.multi_processor_count)


def validate_block(
    x,
    router_w,
    w_gate_up,
    w_down,
    shared_gate_up,
    shared_down,
    shared_gate_w,
    *,
    expert_map=None,
    valid_tokens=None,
    out=None,
):
    weights = (router_w, w_gate_up, w_down, shared_gate_up, shared_down, shared_gate_w)
    if any(w.requires_grad for w in weights):
        raise ValueError("all MoE weights must be frozen (requires_grad=False)")
    if x.requires_grad:
        raise ValueError("MoEBlock is inference only; x must not require gradients")
    if x.ndim != 2 or router_w.ndim != 2 or w_gate_up.ndim != 3 or w_down.ndim != 3:
        raise ValueError(
            "expected x [T,H], router [E,H], expert weights [E_local,2I,H]/[E_local,H,I]"
        )
    t, h = x.shape
    e, rh = router_w.shape
    local, twice_i, wh = w_gate_up.shape
    i = twice_i // 2
    if h == 0 or e < 8 or rh != h or not local or not i or twice_i % 2 or wh != h:
        raise ValueError("incompatible router or expert weight shapes")
    if w_down.shape != (local, h, i):
        raise ValueError("incompatible down weight shape")
    if shared_down.ndim != 2 or shared_down.shape[0] != h or shared_down.shape[1] == 0:
        raise ValueError("shared_down must be [H, SI] with SI > 0")
    si = shared_down.shape[1]
    if shared_gate_up.shape != (2 * si, h) or shared_gate_w.shape != (1, h):
        raise ValueError("shared_gate_up must be [2SI,H] and shared_gate_w [1,H]")
    if expert_map is None and local != e:
        raise ValueError("expert_map is required unless all experts are local")
    tensors = [x, *weights]
    if out is not None:
        if out.shape != x.shape or out.requires_grad:
            raise ValueError("out must have shape [T,H] and not require gradients")
        tensors.append(out)
    if any(v.dtype != torch.bfloat16 for v in tensors):
        raise ValueError("x, weights and out must be BF16")
    for name, tensor, shape in (
        ("expert_map", expert_map, (e,)),
        ("valid_tokens", valid_tokens, (1,)),
    ):
        if tensor is not None:
            if tensor.shape != shape or tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32 with shape {shape}")
            tensors.append(tensor)
    if any(not v.is_contiguous() for v in tensors):
        raise ValueError("all tensors must be contiguous")
    if not x.is_cuda or any(v.device != x.device for v in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")
    if (
        out is not None
        and any(
            out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
            for v in (x, *weights)
        )
        and t
    ):
        raise ValueError("MoEBlock out must not alias x or weights")
    return {
        "rows": t,
        "hidden": h,
        "experts": e,
        "inter": i,
        "local_experts": local,
        "shared_inter": si,
        "dtype": x.dtype,
    }


# The BF16 experts' grouped-GEMM tiles: one rule for fused_experts and moe.lora's BF16 base.


def validate_tile(tile):
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        value = tile[key]
        if not isinstance(value, int) or value <= 0 or value & (value - 1):
            raise ValueError(f"{key} must be a positive power of two")
    if not 16 <= tile["BLOCK_SIZE_M"] <= 256:
        raise ValueError("BLOCK_SIZE_M must be in [16, 256]")
    if not isinstance(tile["GROUP_SIZE_M"], int) or tile["GROUP_SIZE_M"] <= 0:
        raise ValueError("GROUP_SIZE_M must be a positive integer")
    if "num_warps" in tile and tile["num_warps"] not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    if "num_stages" in tile and (not isinstance(tile["num_stages"], int) or tile["num_stages"] < 1):
        raise ValueError("num_stages must be positive")


def base_tiles(t, global_experts, config=None):
    """(up, down) at ``t`` tokens: the default rule with the caller's ``config`` (one dict,
    or "up"/"down" dicts) over it, validated; device-free."""
    small = t <= global_experts
    tile = {
        "BLOCK_SIZE_M": 16 if small else 64,
        "BLOCK_SIZE_N": 32 if small else 64,
        "BLOCK_SIZE_K": 64 if small else 32,
        "GROUP_SIZE_M": 1 if small else 8,
    }
    up, down = dict(tile), dict(tile)
    if config is not None:
        up.update(config.get("up", config) if "down" not in config else config.get("up", {}))
        down.update(config.get("down", config) if "up" not in config else config.get("down", {}))
    validate_tile(up)
    validate_tile(down)
    return up, down


def expert_tiles(t, e, i, global_experts, device, config=None):
    """``base_tiles``, or without a ``config`` the device's tuned table where it has the shape."""
    up, down = base_tiles(t, global_experts, config)
    if config is None:
        kind = hardware(device)
        up = lookup(kind, e, i, t, "up") or up
        down = lookup(kind, e, i, t, "down") or down
    return up, down
