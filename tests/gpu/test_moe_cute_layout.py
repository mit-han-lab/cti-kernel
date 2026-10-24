# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe slot-layout qualification, including graph replay of changed device data."""

import pytest
import torch

from ctikernels.moe.cute._provider import available
from ctikernels.moe.cute._routing import build_layout

from ..modules._moe_layout_reference import assert_layout, build_slot_layout, old_route
from ..modules._moe_layout_reference import slot_expert as _slot_expert

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def capability():
    cap = torch.cuda.get_device_capability()
    if cap not in ((12, 0), (12, 1)) or not available():
        pytest.skip("sm_120/121 and CuTe DSL 4.8.0 required")
    return cap


def route(logits, s, k, routing=None, valid=None, *, has_shared):
    """Selections and live rows from the torch reference, in build_layout's dtypes."""
    sel, _, live, _ = old_route(logits, s, k, routing, valid, has_shared=has_shared)
    return sel.to(torch.int32), live


def check(logits, s, k, routing, valid, shared, mapping, e, bm, lay):
    want = old_route(logits, s, k, routing, valid, has_shared=shared)
    expected = build_slot_layout(_slot_expert(want[0], want[2], e, mapping, shared), e, bm)
    assert_layout(lay, expected)


@pytest.mark.parametrize("t", (1, 7, 64, 1000, 8192))
@pytest.mark.parametrize(
    "shared,replay,mapped,masked",
    (
        (True, False, False, False),
        (True, True, True, True),
        (False, False, True, True),
        (False, True, False, False),
    ),
)
def test_layout(capability, t, shared, replay, mapped, masked):
    torch.manual_seed(t)
    e, k, local = 256, 8, 64 if mapped else 256
    logits = torch.randn(t, e, device="cuda").bfloat16()
    # Both all-equal rows and partial ties at the top-k boundary.
    logits[::3] = 0
    logits[:, :12] = logits[:, :1]
    s = torch.randn(t, device="cuda")
    routing = torch.randint(e, (t, k), dtype=torch.uint8, device="cuda") if replay else None
    valid = torch.tensor(t // 2, dtype=torch.int32, device="cuda") if masked else None
    mapping = (torch.arange(e, device="cuda", dtype=torch.int32) % local) if mapped else None
    if mapped:
        mapping[::3] = -1
    got = route(logits, s, k, routing, valid, has_shared=shared)
    for bm in (16, 64):
        lay = build_layout(
            got[0], got[1], local, bm, mapping, has_shared=shared, capability=capability
        )
        check(logits, s, k, routing, valid, shared, mapping, local, bm, lay)
        # Offset chunk views must not assume 16-byte base alignment.
        if t > 1:
            start = 1
            chunk = tuple(x[start:] for x in got)
            chunk_valid = None if valid is None else (valid - start).clamp_min(0)
            lay = build_layout(
                chunk[0], chunk[1], local, bm, mapping, has_shared=shared, capability=capability
            )
            check(
                logits[start:],
                s[start:],
                k,
                None if routing is None else routing[start:],
                chunk_valid,
                shared,
                mapping,
                local,
                bm,
                lay,
            )


@pytest.mark.parametrize("t", (7, 2048))
@pytest.mark.parametrize("replay,shared", ((False, True), (True, False)))
def test_graph_replay(capability, t, replay, shared):
    e, k, local, bm = 33, 3, 7, 32
    logits = torch.randn(t, e, device="cuda").bfloat16()
    s = torch.randn(t, device="cuda")
    routing = torch.zeros((t, k), dtype=torch.uint8, device="cuda") if replay else None
    valid = torch.tensor(t, dtype=torch.int32, device="cuda")
    mapping = torch.arange(e, dtype=torch.int32, device="cuda") % local

    def run():
        sel, live = route(logits, s, k, routing, valid, has_shared=shared)
        return build_layout(sel, live, local, bm, mapping, has_shared=shared, capability=capability)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lay = run()
    torch.cuda.current_stream().wait_stream(stream)
    for n in (0, t // 2, t, -(2**31), 2**31 - 1):
        logits.copy_(torch.randint(-2, 3, logits.shape, device="cuda"))
        s.add_(0.125)
        valid.fill_(n)
        mapping.copy_(torch.arange(e, device="cuda", dtype=torch.int32) % local)
        mapping[::2] = -1
        if replay:
            routing.copy_(torch.randint(e, routing.shape, device="cuda", dtype=torch.uint8))
        graph.replay()
        check(logits, s, k, routing, valid, shared, mapping, local, bm, lay)


@pytest.mark.parametrize("e,k", ((1, 1), (17, 8), (33, 3), (256, 33)))
@pytest.mark.parametrize("shared", (False, True))
def test_empty_local_shard(capability, e, k, shared):
    logits = torch.zeros(7, e, dtype=torch.bfloat16, device="cuda")
    s = torch.zeros(7, device="cuda")
    mapping = torch.full((e,), -1, dtype=torch.int32, device="cuda")
    sel, live = route(logits, s, k, has_shared=shared)
    lay = build_layout(sel, live, 0, 16, mapping, has_shared=shared, capability=capability)
    check(logits, s, k, None, None, shared, mapping, 0, 16, lay)
