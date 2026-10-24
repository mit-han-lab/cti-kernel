# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Orchestrator-only CuTe qualification: exact Q0, maps, PDL and graph replay.

Available-provider compilation failures must fail, never turn into skips.
Every Q0 byte and valid routed row is checked, including duplicate slots.
"""

import pytest
import torch

from ctikernels.moe.cute._runtime import load
from ctikernels.moe.cute._w4a8_model import unpack_q0
from ctikernels.moe.cute._w4a8_prepare import Config, W4A8Workspace, prepare

from .._moe_w4a8_reference import prepare_reference

SIZES = (1, 2, 4, 8, 16, 17, 42, 43, 64, 256, 1024, 4096)


@pytest.fixture
def cute_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if load(torch.device("cuda")) is None:
        pytest.skip("CuTe 4.8.0 and SM120/SM121 required")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(old)


def inputs(m, pattern, seed=15):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, 4096, generator=gen).bfloat16()
    # One row includes group zeros, BF16 subnormals/extrema, RNE ties and
    # neighbors on either side of scale boundaries. Other rows remain random.
    groups = [
        torch.full((32,), v, dtype=torch.bfloat16)
        for v in (
            0.0,
            -0.0,
            2.0**-133,
            -(2.0**-133),
            2.0**-126,
            torch.finfo(torch.bfloat16).max,
            -torch.finfo(torch.bfloat16).max,
        )
    ]
    for exponent in (-127, -120, -8, 0, 100):
        center = torch.tensor(448 * 2.0**exponent, dtype=torch.bfloat16)
        for value in (
            torch.nextafter(center, torch.zeros_like(center)),
            center,
            torch.nextafter(center, torch.full_like(center, float("inf"))),
        ):
            group = value.expand(32).clone()
            group[1::2] *= -1
            groups.append(group)
    groups.append(
        torch.tensor(
            [448, -448, 0, -0.0, 1, 1.0625, 1.1875, 2.0**-9, 2.0**-10, 3 * 2.0**-10] + [0] * 22,
            dtype=torch.bfloat16,
        )
    )
    edge = torch.cat(groups)
    x[0, : len(edge)] = edge
    ids = torch.randint(-1, 128, (m, 6), dtype=torch.int32, generator=gen)
    if pattern == "skewed":
        ids[ids > 3] = 127
    elif pattern == "duplicate":
        ids[:] = torch.tensor([0, 127, 0, -1, 127, 0])
    elif pattern == "remote":
        ids.fill_(-1)
    weights = torch.randn(m, 6, generator=gen)
    return x, ids, weights


def assert_prepared(ws, out, x, ids, weights, config):
    expected = prepare_reference(x, ids, weights)
    for name in ("expert_offsets", "scale_offsets", "num_rows"):
        assert torch.equal(getattr(ws, name).cpu(), getattr(expected, name)), name
    length = int(expected.num_rows[0])
    pairs = ws.row_pair[:length].cpu().long()
    expected_pairs = expected.row_pair[:length].long()
    if ws.rows <= config.small_rows:
        assert torch.equal(pairs, expected_pairs)
    else:
        for expert in range(128):
            start, stop = expected.expert_offsets[expert : expert + 2].tolist()
            assert torch.equal(pairs[start:stop].sort().values, expected_pairs[start:stop])
    inverse = ws.slot_row.cpu().flatten()
    assert torch.equal(inverse[pairs], torch.arange(length, dtype=torch.int32))
    remote = (ids.flatten() < 0) | (ids.flatten() >= 128)
    assert (inverse[remote] == -1).all()
    assert torch.equal(ws.row_token[:length].cpu(), pairs // 6)
    assert torch.equal(ws.row_weight[:length].cpu(), weights.flatten()[pairs])
    assert all(torch.equal(a, b) for a, b in zip(unpack_q0(ws), unpack_q0(expected), strict=True))
    occupied = torch.zeros(ws.scale_rows, dtype=torch.bool)
    occupied[expected.pair_scale[expected.pair_scale >= 0].long()] = True
    assert (ws.q0_scales.cpu()[:, ~occupied] == 0x7F7F7F7F).all()
    assert (ws.q1_scales.cpu() == 0x7F7F7F7F).all()
    result = out.cpu()
    if ws.combine == "atomic":
        assert not result.any() and not torch.signbit(result).any()
    else:
        assert result.isnan().all()


@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("pattern", ["random", "skewed", "duplicate", "remote"])
@pytest.mark.parametrize("pdl", [False, True])
def test_prepare(cute_device, m, pattern, pdl):
    x, ids, weights = inputs(m, pattern)
    ws = W4A8Workspace(m, cute_device)
    out = torch.full((m, 4096), float("nan"), dtype=torch.bfloat16, device=cute_device)
    config = Config(pdl=pdl)
    prepare(x.to(cute_device), ids.to(cute_device), weights.to(cute_device), ws, out, config=config)
    assert_prepared(ws, out, x, ids, weights, config)


@pytest.mark.parametrize("m", [1, 43, 1024])
@pytest.mark.parametrize("pattern", ["random", "duplicate"])
def test_prepare_maps_global_ids(cute_device, m, pattern):
    # PREP maps global ids itself (small and general paths): the result equals preparing
    # the mapped local ids, which it also writes to ws.local_ids.
    x, local, weights = inputs(m, pattern)
    gen = torch.Generator().manual_seed(3)
    owner = torch.randperm(300, generator=gen)[:128]  # global id of each local expert
    expert_map = torch.full((300,), -1, dtype=torch.int32)
    expert_map[owner] = torch.arange(128, dtype=torch.int32)
    ids = torch.where(local >= 0, owner[local.clamp(min=0).long()].int(), -1)
    remote = torch.isin(torch.arange(300), owner, invert=True).nonzero().flatten()
    ids[0, :3] = torch.tensor([300, -2, int(remote[0])])  # out of range, padding, remote
    local[0, :3] = -1
    ws = W4A8Workspace(m, cute_device)
    out = torch.full((m, 4096), float("nan"), dtype=torch.bfloat16, device=cute_device)
    args = [t.to(cute_device) for t in (x, ids, weights)]
    prepare(*args, ws, out, expert_map=expert_map.to(cute_device))
    assert_prepared(ws, out, x, local, weights, Config())
    assert torch.equal(ws.local_ids.cpu(), local.flatten())


@pytest.mark.parametrize("m", [2, 43, 64])
@pytest.mark.parametrize("chunk", [256, 512, 1024])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_launch_variants(cute_device, m, chunk, fused, pdl):
    x, ids, weights = inputs(m, "duplicate")
    # Force the general path at small M as well; include out-of-contract IDs
    # to qualify the defensive bounds masks before any expert buffer access.
    ids[0, :3] = torch.tensor([-2, 128, 129])
    ws = W4A8Workspace(m, cute_device)
    # Element-aligned views must not inherit a false 16-byte alignment promise.
    xd = torch.empty(m * 4096 + 1, dtype=torch.bfloat16, device=cute_device)[1:].reshape(m, 4096)
    xd.copy_(x)
    out = torch.empty_like(xd)
    config = Config(small_rows=0, q0_chunk=chunk, fuse_assignment=fused, pdl=pdl)
    prepare(xd, ids.to(cute_device), weights.to(cute_device), ws, out, config=config)
    assert_prepared(ws, out, x, ids, weights, config)


@pytest.mark.parametrize("m", [1, 42, 43, 256, 4096])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("combine", ["atomic", "rows"])
def test_graph_replay_changed_routes_and_x(cute_device, m, pdl, fused, combine):
    original = inputs(m, "random")
    xd, ids_d, weights_d = [value.to(cute_device) for value in original]
    ws = W4A8Workspace(m, cute_device, combine=combine)
    out = torch.empty_like(xd)
    config = Config(pdl=pdl, fuse_assignment=fused)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            prepare(xd, ids_d, weights_d, ws, out, config=config)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare(xd, ids_d, weights_d, ws, out, config=config)
    addresses = [value.data_ptr() for value in (ws.q0, ws.q0_scales, ws.q1, out, ws.row_pair)]
    for seed, pattern in enumerate(("skewed", "remote", "duplicate", "random"), 51):
        x, ids, weights = inputs(m, pattern, seed)
        x.neg_()
        xd.copy_(x)
        ids_d.copy_(ids)
        weights_d.copy_(weights)
        out.fill_(float("nan"))
        ws.q0_scales.fill_(-1)
        ws.q1_scales.fill_(-1)
        graph.replay()
        assert_prepared(ws, out, x, ids, weights, config)
        assert addresses == [
            value.data_ptr() for value in (ws.q0, ws.q0_scales, ws.q1, out, ws.row_pair)
        ]


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("unaligned", [False, True])
def test_finalize_slot_order_rounding_and_positive_zero(cute_device, pdl, unaligned):
    from ctikernels.moe.cute._w4a8_finalize import arguments, compile_finalize

    from .._moe_w4a8_reference import rows_combine

    m = 7
    ws = W4A8Workspace(m, cute_device, combine="rows")
    rt = load(cute_device)
    gen = torch.Generator().manual_seed(734)
    slots = torch.randn((m, 6, 4096), generator=gen)
    slots[0].zero_()
    slots[1, :, :8] = torch.tensor([2.0**25, -(2.0**25), 1, 0, 0, 0])[:, None]
    slots[2, :, :8] = torch.tensor([1, 2.0**-8, 2.0**-8, 2.0**-8, 2.0**-8, 2.0**-8])[:, None]
    slots[3, :, :8] = torch.tensor([2.0**-133, -(2.0**-133), 0, 0, 0, 0])[:, None]
    slots[4, 1::2].zero_()
    live = torch.ones((m, 6), dtype=torch.bool)
    live[0] = False
    live[4, 1::2] = False
    expected = rows_combine(slots).view(torch.int16)
    base = torch.empty(m * 4096 + int(unaligned), device=cute_device, dtype=torch.bfloat16)
    out = base[int(unaligned) :].view(m, 4096)
    launch = compile_finalize(rt, ws, out, pdl=pdl)
    for permutation in (torch.arange(6 * m), torch.randperm(6 * m, generator=gen)):
        inverse = permutation.reshape(m, 6).int()
        inverse[~live] = -1
        y = torch.full((6 * m, 4096), float("nan"), dtype=torch.float32)
        y[inverse[live].long()] = slots[live]
        ws.y.copy_(y)
        ws.slot_row.copy_(inverse)
        for _ in range(3):
            out.fill_(float("nan"))
            launch(*arguments(ws, out))
            assert torch.equal(out.cpu().view(torch.int16), expected)
        assert not out[0].signbit().any()
