# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Identical seeded norm pools for parent, FI qualification child and market worker."""

from dataclasses import dataclass

import torch

POOL_BASES = ("x", "residual", "weight", "h1", "saved_x", "saved_r", "saved_h1", "aux")


def clone_pool_views(value, storage=None):
    """Clone backing allocations once, preserving offsets, strides and aliases.

    Tensor.clone() can compact views (including singleton strides) and always
    resets storage_offset. Warmup must exercise exactly the capture tensor ABI.
    CUDA allocations have the same base alignment, so retaining the offset also
    retains the vector alignment used by admission and provider specialization.
    """
    storage = {} if storage is None else storage
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return type(value)(clone_pool_views(t, storage) for t in value)
    key = (value.device, value.untyped_storage().data_ptr(), value.dtype)
    if key not in storage:
        count = value.untyped_storage().nbytes() // value.element_size()
        storage[key] = value.as_strided((count,), (1,), 0).clone()
    result = storage[key].as_strided(value.shape, value.stride(), value.storage_offset())
    if result.data_ptr() % 16 != value.data_ptr() % 16:
        raise RuntimeError("private warmup storage changed tensor alignment")
    # Keep Dynamo's explicit dynamic-shape hints used by the activation markets.
    for dim in getattr(value, "_dynamo_dynamic_indices", ()):
        torch._dynamo.mark_dynamic(result, dim)
    return result


def clone_norm_tensors(tensors):
    """Private warm/profile storage with the pool views' exact tensor ABI."""
    return clone_pool_views(tensors)


def warm_norm_call(fn, tensors):
    if hasattr(fn, "call_index"):
        for index in (0, 1):
            fn.call_index(clone_norm_tensors(tensors), None, index)
    else:
        fn(clone_norm_tensors(tensors))


def pool_spec(args, m, regime, l2):
    return {
        "op": args.op,
        "m": m,
        "regime": regime,
        "width": args.width,
        "dtype": args.dtype,
        "layers": args.layers,
        "seed": 20261015 + m,
        "l2_bytes": l2,
        "static_weight": bool(args.static_weight),
    }


def pool_count(spec):
    m, width, layers = (spec[key] for key in ("m", "width", "layers"))
    if min(m, width, layers, spec["l2_bytes"]) <= 0:
        raise ValueError("pool dimensions and L2 must be positive")
    if spec["regime"] not in ("hot", "engine", "cold"):
        raise ValueError("unknown pool regime")
    rotating = (m + 1 + (m if spec["regime"] == "cold" else 0)) * width * 2
    return (
        layers
        if spec["regime"] == "hot"
        else max(layers, (2 * spec["l2_bytes"] // (rotating * layers) + 1) * layers)
    )


@dataclass
class NormPools:
    spec: dict
    count: int
    tensors: list
    inputs: list
    x: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    h1: torch.Tensor
    saved_x: torch.Tensor
    saved_r: torch.Tensor
    saved_h1: torch.Tensor
    aux: torch.Tensor | None

    def restore(self, unused=None):
        self.residual.copy_(self.saved_r)
        self.h1.copy_(self.saved_h1)
        self.x.copy_(self.saved_x)

    def previous_hidden(self, index):
        # vLLM's site B re-reads the o_proj output (h1) produced before the MLP. It is as
        # old as the residual, so it rotates (L2-cold) in the engine and cold regimes.
        return self.h1[0 if self.spec["regime"] == "hot" else index]

    @property
    def input_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.x, self.residual, self.weight))


def norm_pool_views(spec, *, x, residual, weight, h1, saved_x, saved_r, saved_h1, aux):
    """Build the same views over locally allocated or IPC-imported base tensors."""
    count = pool_count(spec)
    unique = 1 if spec["regime"] == "hot" else count
    width = spec["width"]
    tensors = []
    for i in range(unique):
        triple = (x[i if spec["regime"] == "cold" else 0], residual[i], weight[i])
        tensors.append(triple if aux is None else (*triple, aux[i, :, width : 2 * width]))
    inputs = tensors * count if spec["regime"] == "hot" else tensors
    return NormPools(
        spec, count, tensors, inputs, x, residual, weight, h1, saved_x, saved_r, saved_h1, aux
    )


def build_norm_pools(spec, device="cuda", *, need_h1=True):
    count = pool_count(spec)
    unique = 1 if spec["regime"] == "hot" else count
    unique_x = unique if spec["regime"] == "cold" else 1
    m, width = spec["m"], spec["width"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[spec["dtype"]]
    torch.manual_seed(spec["seed"])
    x = torch.randn((unique_x, m, width), device=device, dtype=dtype)
    residual = torch.randn((unique, m, width), device=device, dtype=dtype)
    weight = torch.randn((unique, width), device=device, dtype=dtype) * 0.125
    weight = -weight.abs() - (1.25 if spec["op"].startswith("gemma") else 0.25)
    # Generate after the original pools, preserving the --fi07 check-input sequence.
    h1 = torch.randn((unique if need_h1 else 0, m, width), device=device, dtype=dtype)
    aux = (
        torch.full((unique, m, 3 * width), 17.0, device=device, dtype=dtype)
        if spec["op"].endswith("_sum")
        else None
    )
    return norm_pool_views(
        spec,
        x=x,
        residual=residual,
        weight=weight,
        h1=h1,
        saved_x=x.clone(),
        saved_r=residual.clone(),
        saved_h1=h1.clone(),
        aux=aux,
    )


def fp32_residual_pools(market):
    """Reuse the market's activation/weight geometry with a separate FP32 stream.

    The market still owns BF16/FP16 residuals. Widen its seeded values outside
    timing, keeping identical initial values and a conservative cold pool size.
    """
    residual = market.saved_r.float()
    return norm_pool_views(
        market.spec | {"residual_dtype": "fp32"},
        x=market.x,
        residual=residual,
        weight=market.weight,
        h1=market.h1,
        saved_x=market.saved_x,
        saved_r=residual.clone(),
        saved_h1=market.saved_h1,
        aux=market.aux.float() if market.aux is not None else None,
    )


@dataclass
class RWCopyPools:
    inputs: list
    source: torch.Tensor
    destination: torch.Tensor
    saved_source: torch.Tensor

    def restore(self):
        # Restore outside timing, ending with the input just as NormPools does.
        self.destination.copy_(self.saved_source)
        self.source.copy_(self.saved_source)


def build_rw_copy_pools(norm):
    """One FP32 copy pair per norm input, with exactly the same bank rotation.

    Hot repeats one resident pair across L calls, like the norm's hot input.
    Engine/cold rotate through the norm's full pool, grouped into L-call graphs.
    Each call reads and writes ceil(6*M*H/4) FP32 elements (weight excluded).
    """
    elements = (3 * norm.spec["m"] * norm.spec["width"] + 1) // 2
    source = torch.ones((len(norm.tensors), elements), device=norm.x.device, dtype=torch.float32)
    destination = torch.empty_like(source)
    pairs = [(source[i], destination[i]) for i in range(len(norm.tensors))]
    inputs = pairs * norm.count if norm.spec["regime"] == "hot" else pairs
    return RWCopyPools(inputs, source, destination, source.clone())
