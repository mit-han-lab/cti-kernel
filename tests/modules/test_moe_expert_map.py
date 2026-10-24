# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contract checks, requiring neither NumPy nor GPU providers."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ctikernels.moe import fused_experts, moe_align_block_size_local  # noqa: E402
from ctikernels.moe_impl import (  # noqa: E402
    moe_align_block_size,
    moe_sum_reduce_triton,
)


def _reference_align(ids, block_size, mapping, local_experts):
    n = ids.numel()
    sorted_ids = torch.full((n + local_experts * (block_size - 1),), n, dtype=torch.int32)
    experts = torch.full(((len(sorted_ids) + block_size - 1) // block_size,), -1, dtype=torch.int32)
    cursor = 0
    for expert in range(local_experts):
        routes = [
            r for r, g in enumerate(ids.flatten().tolist()) if g >= 0 and mapping[g] == expert
        ]
        padded = (len(routes) + block_size - 1) // block_size * block_size
        sorted_ids[cursor : cursor + len(routes)] = torch.tensor(routes, dtype=torch.int32)
        experts[cursor // block_size : (cursor + padded) // block_size] = expert
        cursor += padded
    return sorted_ids, experts, torch.tensor([cursor], dtype=torch.int32)


def test_reference_align_hand_cases():
    ids = torch.tensor([[3, 0], [-1, 2], [1, 3]], dtype=torch.int32)
    mapping = torch.tensor([1, -1, 0, 1], dtype=torch.int32)
    sorted_ids, experts, total = _reference_align(ids, 4, mapping, 3)
    assert total.tolist() == [8]
    assert sorted_ids.tolist() == [3, 6, 6, 6, 0, 1, 5, 6] + [6] * 7
    assert experts.tolist() == [0, 1, -1, -1]
    remote = _reference_align(ids, 4, torch.full_like(mapping, -1), 2)
    assert remote[2].item() == 0
    assert (remote[0] == ids.numel()).all()
    assert (remote[1] == -1).all()
    empty = _reference_align(torch.empty((0, 2), dtype=torch.int32), 4, mapping, 3)
    assert empty[2].item() == 0
    assert empty[0].tolist() == [0] * 9
    assert empty[1].tolist() == [-1] * 3


@pytest.mark.parametrize("block,local", [(0, 2), (-16, 2), (16, 0), (16, -1)])
def test_align_scalar_validation(block, local):
    with pytest.raises(ValueError, match="block_size|num_local_experts"):
        moe_align_block_size_local(
            torch.zeros((1, 2), dtype=torch.int32), block, torch.arange(2, dtype=torch.int32), local
        )


@pytest.mark.parametrize(
    "kind",
    [
        "ids_dtype",
        "ids_rank",
        "ids_stride",
        "map_dtype",
        "map_rank",
        "map_stride",
        "empty_map",
        "cpu",
    ],
)
def test_align_tensor_validation(kind):
    ids, mapping = torch.zeros((2, 2), dtype=torch.int32), torch.arange(4, dtype=torch.int32)
    if kind == "ids_dtype":
        ids = ids.long()
    elif kind == "ids_rank":
        ids = ids.flatten()
    elif kind == "ids_stride":
        ids = ids.T
    elif kind == "map_dtype":
        mapping = mapping.long()
    elif kind == "map_rank":
        mapping = mapping[None]
    elif kind == "map_stride":
        mapping = mapping[::2]
    elif kind == "empty_map":
        mapping = mapping[:0]
    message = (
        "CUDA" if kind == "cpu" else "at least one" if kind == "empty_map" else "contiguous int32"
    )
    with pytest.raises(ValueError, match=message):
        moe_align_block_size_local(ids, 16, mapping, 2)
    if kind not in ("ids_dtype", "ids_stride"):  # the public entry casts and packs ids
        with pytest.raises(ValueError, match=message):
            moe_align_block_size(ids, 16, 2, expert_map=mapping)


def _inputs():
    return [
        torch.empty((2, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 32), dtype=torch.bfloat16),
        torch.ones((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "grad1",
        "grad2",
        "shape",
        "weight_shape",
        "dtype",
        "route_dtype",
        "route_shape",
        "weights_dtype",
        "activation",
        "out",
        "stride",
        "cpu",
    ],
)
def test_fused_experts_validation_without_providers(kind):
    args, kwargs = _inputs(), {}
    if kind.startswith("grad"):
        args[int(kind[-1])].requires_grad_(True)
    elif kind == "shape":
        args[0] = args[0].flatten()
    elif kind == "weight_shape":
        args[2] = args[2][:, :, :16].contiguous()
    elif kind == "dtype":
        args[0] = args[0].float()
    elif kind == "route_dtype":
        args[4] = args[4].long()
    elif kind == "route_shape":
        args[4] = args[4][:1]
    elif kind == "weights_dtype":
        args[3] = args[3].bfloat16()
    elif kind == "activation":
        kwargs["activation"] = "gelu"
    elif kind == "out":
        kwargs["out"] = torch.empty(1)
    elif kind == "stride":
        args[1] = args[1].transpose(1, 2)
    message = {
        "grad1": "frozen",
        "grad2": "frozen",
        "shape": "expected hidden_states",
        "weight_shape": "incompatible",
        "dtype": "BF16",
        "route_dtype": "int32",
        "route_shape": "topk_ids",
        "weights_dtype": "float32",
        "activation": "silu",
        "out": "out must",
        "stride": "contiguous",
        "cpu": "CUDA",
    }[kind]
    with pytest.raises(ValueError, match=message):
        fused_experts(*args, **kwargs)


def test_reduce_routing_arguments():
    x, out = torch.empty(2, 2, 64), torch.empty(2, 64)
    ids, mapping = torch.zeros(2, 2, dtype=torch.int32), torch.arange(2, dtype=torch.int32)
    with pytest.raises(ValueError, match="needs topk_weights"):
        moe_sum_reduce_triton(x, out, topk_ids=ids)
    with pytest.raises(ValueError, match="needs topk_ids"):
        moe_sum_reduce_triton(x, out, expert_map=mapping)
    for routes, weights in (
        (None, torch.ones(2, 2)),
        (ids, torch.ones(2, 3)),
        (ids, torch.ones(2, 2, dtype=torch.bfloat16)),
    ):
        with pytest.raises(ValueError, match="topk_weights must be"):
            moe_sum_reduce_triton(x, out, topk_ids=routes, topk_weights=weights)
    with pytest.raises(ValueError, match="CUDA"):
        moe_sum_reduce_triton(x, out)
    with pytest.raises(ValueError, match="CUDA"):
        moe_sum_reduce_triton(x, out, topk_ids=ids, topk_weights=torch.ones(2, 2))


@pytest.mark.parametrize(
    "config",
    [
        {"BLOCK_SIZE_M": 8},
        {"BLOCK_SIZE_M": 17},
        {"BLOCK_SIZE_N": 0},
        {"BLOCK_SIZE_K": 3},
        {"GROUP_SIZE_M": -1},
    ],
)
def test_invalid_config_before_cuda(config):
    with pytest.raises(ValueError, match="BLOCK_SIZE|GROUP_SIZE"):
        fused_experts(*_inputs(), config=config)


@pytest.mark.parametrize("n,experts", [(8, 128), (4097, 128), (8, 513)])
def test_align_ep1_uses_the_identity_map(monkeypatch, n, experts):
    import ctikernels.moe as moe
    import ctikernels.moe_impl as impl

    ids = torch.zeros((n, 1), dtype=torch.int32)
    identity = torch.arange(experts, dtype=torch.int32)
    result = object()

    def local(routes, block, mapping, count, **options):
        assert routes is ids and block == 16 and mapping is identity and count == experts
        assert options == {
            "capacity_experts": experts + 1,
            "identity": True,
            "invalid_bucket": True,
        }
        return result

    monkeypatch.setattr(impl, "_identity_expert_map", lambda device, count: identity)
    monkeypatch.setattr(moe, "moe_align_block_size_local", local)
    assert moe_align_block_size(ids, 16, experts) is result


@pytest.mark.parametrize(
    "n,block,experts,invalid",
    [
        (8, 16, 128, True),
        (10240, 256, 512, True),
        (10241, 16, 128, False),
        (8, 16, 513, False),
        (8, 24, 128, False),
        (8, 512, 128, False),
    ],
)
def test_align_is_triton_at_every_size(monkeypatch, n, block, experts, invalid):
    """Our Triton alignment is the only provider: every route count, block and expert count."""
    from ctikernels.moe import _align

    ids = torch.zeros((n, 1), dtype=torch.int32)
    mapping = torch.arange(experts, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(_align, "_validate_routes", lambda *args: None)

    def triton_align(routes, size, count, mapped, **options):
        assert routes is ids and (size, count) == (block, experts) and mapped is mapping
        calls.append(options)

    monkeypatch.setitem(
        sys.modules,
        "ctikernels.moe.triton._slot_layout",
        SimpleNamespace(align_block_size=triton_align),
    )
    extra = {"capacity_experts": experts + 1, "invalid_bucket": True} if invalid else {}
    moe_align_block_size_local(ids, block, mapping, experts, **extra)
    assert calls == [extra or {"capacity_experts": None, "invalid_bucket": False}]


@pytest.mark.parametrize("mapped", (False, True))
def test_align_casts_int64_ids(monkeypatch, mapped):
    """int64 ids (torch.topk indices) reach the dispatcher as contiguous int32, with or
    without an expert map."""
    import ctikernels.moe as moe
    import ctikernels.moe_impl as impl

    ids = torch.tensor([[3, 0], [-1, 1], [0, 2]], dtype=torch.int64).t()  # non-contiguous
    seen = []
    monkeypatch.setattr(impl, "_identity_expert_map", lambda device, count: None)
    monkeypatch.setattr(
        moe, "moe_align_block_size_local", lambda routes, *a, **k: seen.append(routes)
    )
    mapping = torch.arange(4, dtype=torch.int32) if mapped else None
    moe_align_block_size(ids, 16, 4, expert_map=mapping)
    assert seen[0].dtype == torch.int32 and seen[0].is_contiguous()
    assert torch.equal(seen[0], ids.to(torch.int32))


def test_align_identity_map_cache_and_capture(monkeypatch):
    import ctikernels.moe_impl as impl

    cache, allocations, contexts = {}, [], []
    capturing = True
    arange = torch.arange
    monkeypatch.setattr(impl, "_IDENTITY_MAPS", cache)

    def device_context(device):
        contexts.append(device)
        return nullcontext()

    def allocate(count, *, dtype, device):
        allocations.append((count, dtype, device))
        return arange(count, dtype=dtype)

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch, "arange", allocate)
    device = torch.device("cuda:0")
    first = impl._identity_expert_map(device, 8)
    second = impl._identity_expert_map(device, 8)
    assert first is not second and cache == {}
    capturing = False
    eager = impl._identity_expert_map(device, 8)
    assert eager.tolist() == list(range(8))
    assert cache[(device, 8)] is eager
    capturing = True
    assert impl._identity_expert_map(device, 8) is eager
    assert len(allocations) == 3
    capturing = False
    impl._identity_expert_map(device, 16)
    other = torch.device("cuda:1")
    impl._identity_expert_map(other, 8)
    assert set(cache) == {(device, 8), (device, 16), (other, 8)}
    assert all(dtype == torch.int32 for _, dtype, _ in allocations)
    assert contexts == [device] * 5 + [other]
