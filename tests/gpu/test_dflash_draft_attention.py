# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""K0 DRAFT masks, compact queries and exact-B changing-metadata graphs.

BF16 uses 2e-2 absolute/relative error, frozen before GPU execution: quantized
inputs and BF16 output rounding plus fused softmax accumulation differ from the
dense FP32 oracle. Exact mask membership is tested separately, without tolerating
leakage at excluded keys. FP32 oracle arithmetic is checked against FP64 on CPU.
"""

import os
from itertools import product

import pytest
import torch

from ._dflash import Fixture
from ._exhaustive import ci_cases

ATOL = 2e-2
RTOL = 2e-2
QUERIES = [4, 5, 8, 9, 12, 13, 16, 17] + (
    [2, 32] if os.environ.get("CTIK_DFLASH_EXTENDED") == "1" else []
)
DRAFT_MASKS = [(False, None), (True, 4096), (False, 2048)]
MASKS = DRAFT_MASKS + [(True, None)]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def provider_and_oracle_precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def capture(f):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        f.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        f.run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


# CI keeps B=1, Q=16 causal attention with 16-token pages and 32/8 heads.
# The graph still replays zero lengths, slot -1 and changing positions.
@pytest.mark.parametrize(
    "b,q,p,hq,hk,causal,window",
    ci_cases(
        [
            (b, q, p, hq, hk, causal, window)
            for b, q, p, (hq, hk), (causal, window) in product(
                [1, 2, 3, 4], QUERIES, [1, 16], [(16, 4), (32, 8)], DRAFT_MASKS
            )
        ]
        + [(b, q, p, 32, 8, True, None) for b, q, p in product([1, 3], [4, 16], [1, 16])],
        core=[(1, 16, 16, 32, 8, True, None)],
    ),
)
def test_exact_batch_graph_changes_lengths_slots_positions(
    b, q, p, hq, hk, causal, window, draft_provider
):
    f = Fixture(b, q, p, hq, hk, causal, window, provider=draft_provider)
    f.stage([128] * b, list(range(b)))
    graph = capture(f)
    saved_k, saved_v = f.k.clone(), f.v.clone()
    contexts = [
        0,
        1,
        128,
        512,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        p - 1,
        p,
        p + 1,
        f.c - q - 1,
        f.c - q,
    ]
    # Independent random Q fixture on every replay; every graph is an exact B.
    for turn in range(32):
        slots = list(range(b))[turn % b :] + list(range(b))[: turn % b]
        if turn % 5 in (1, 3):
            slots[(turn // 5) % b] = -1
        if turn % 5 == 2:
            slots = [-1] * b
        # Reuse a new generation of slot identities after inactivity.
        slots = [slot + turn * b if slot != -1 else -1 for slot in slots]
        lengths = [contexts[(turn + row) % len(contexts)] for row in range(b)]
        base = 0 if window is None else (0, 2**31 + 17, 2**63 - 1 - f.c)[turn % 3]
        if base:
            lengths = [max(length, window - 1) for length in lengths]
        f.stage(lengths, slots, turn, base)
        graph.replay()
        torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
        f.check_metadata()
        for row, slot in enumerate(slots):
            if slot == -1:
                assert torch.count_nonzero(f.out[row * q : (row + 1) * q]) == 0
    torch.testing.assert_close(f.k, saved_k, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(f.v, saved_v, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "q,p,causal,window",
    ci_cases(
        [
            (q, p, causal, window)
            for q, p, (causal, window) in product(QUERIES, [1, 16], DRAFT_MASKS)
        ]
        + [(q, p, True, None) for q, p in product([4, 16], [1, 16])],
        core=[
            (4, 1, False, None),
            (5, 16, True, 4096),
            (16, 16, False, 2048),
            (16, 16, True, None),
        ],
    ),
)
def test_exact_boundary_membership_and_later_block_keys(q, p, causal, window, draft_provider):
    f = Fixture(1, q, p, 16, 4, causal, window, c=4160, provider=draft_provider)
    w = window or 4096
    # First query left bound sits before/on/after a physical-page boundary.
    for cutoff in (p - 1, p, p + 1):
        context = w - 1 + cutoff
        f.stage([context], [0], turn=cutoff)
        f.query.zero_()
        f.k.zero_()
        for query_index in (0, q - 1):
            position = context + query_index
            for distance in (w - 1, w, w + 1, 0, -(q - 1 - query_index)):
                key = position - distance
                if key < 0 or key >= context + q:
                    continue
                f.v.zero_()
                physical = int(f.cpu_token_idx[key])
                f.v.view(-1, 4, 128)[physical].fill_(16384)
                f.run()
                expected = f.oracle()
                torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
                included = (window is None or abs(distance) < w) and (not causal or distance >= 0)
                if included:
                    assert (f.out[query_index] > 0).all()
                else:
                    assert torch.count_nonzero(f.out[query_index]) == 0


@pytest.mark.parametrize(
    "causal,window",
    [
        pytest.param(
            causal, window, marks=() if (causal, window) == (True, 4096) else pytest.mark.exhaustive
        )
        for causal, window in MASKS
    ],
)
def test_b4_three_active_matches_separately_captured_b3(causal, window, draft_provider):
    compact = Fixture(3, 16, 1, 16, 4, causal, window, c=544, provider=draft_provider)
    padded = Fixture(4, 16, 1, 16, 4, causal, window, c=544, provider=draft_provider)
    compact.stage([128, 512, 1], [2, 0, 1])
    padded.stage([128, 512, 1, 0], [2, 0, 1, -1])
    # Same physical mapping and identical local head slices for the first rows.
    padded.k[: compact.k.shape[0]].copy_(compact.k)
    padded.v[: compact.v.shape[0]].copy_(compact.v)
    padded.query[:48].copy_(compact.query)
    g3, g4 = capture(compact), capture(padded)
    g3.replay()
    g4.replay()
    torch.testing.assert_close(padded.out[:48], compact.out, atol=ATOL, rtol=RTOL)
    assert torch.count_nonzero(padded.out[48:]) == 0


def test_grad_and_alias_rejected_before_any_output_write(draft_provider):
    f = Fixture(1, 8, 1, 16, 4, False, None, c=128, provider=draft_provider)
    f.stage([1], [0])
    f.op.prepare()
    for value in (f.query, f.k, f.v):
        value.requires_grad_(True)
        with torch.no_grad(), pytest.raises(ValueError, match="gradients"):
            f.op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
        assert f.out.isnan().all()
        value.requires_grad_(False)
    with pytest.raises(ValueError, match="overlap"):
        f.op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.query)


@pytest.mark.parametrize("operand", ["q", "k", "v", "out"])
@pytest.mark.parametrize("offset", [1, 8])
def test_operand_base_pointer_alignment(operand, offset, draft_provider):
    f = Fixture(1, 8, 1, 16, 4, False, None, c=128, provider=draft_provider)
    f.stage([32], [0])
    f.run()
    expected = f.out.clone()
    operands = {"q": f.query, "k": f.k, "v": f.v, "out": f.out}
    original = operands[operand]
    view = torch.empty(original.numel() + offset, dtype=original.dtype, device=original.device)
    view = view[offset:].view_as(original)
    view.copy_(original)
    assert view.is_contiguous()
    assert view.data_ptr() % 16 == (offset * 2) % 16
    operands[operand] = view
    operands["out"].fill_(float("nan"))
    args = {
        "q": operands["q"],
        "paged_kv_cache": (operands["k"], operands["v"]),
        "out": operands["out"],
    }
    if offset == 1:
        with pytest.raises(ValueError, match="16-byte"):
            f.op.run(**args)
        assert operands["out"].isnan().all()
    else:
        f.op.run(**args)
        torch.testing.assert_close(operands["out"], expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "q,causal,window",
    ci_cases(
        QUERIES,
        MASKS,
        core=[(4, False, None)],
    ),
)
def test_tp_contiguous_head_shards_match_global_oracle(q, causal, window, draft_provider):
    full = Fixture(2, q, 16, 32, 8, causal, window, c=544, provider=draft_provider)
    full.stage([512, 128], [1, 0])
    expected = full.oracle()
    local = Fixture(2, q, 16, 16, 4, causal, window, c=544, provider=draft_provider)
    local.stage([512, 128], [1, 0])
    for rank in (0, 1):
        local.query.copy_(full.query[:, rank * 16 : (rank + 1) * 16])
        local.k.copy_(full.k[:, :, rank * 4 : (rank + 1) * 4])
        local.v.copy_(full.v[:, :, rank * 4 : (rank + 1) * 4])
        local.run()
        torch.testing.assert_close(
            local.out.float(), expected[:, rank * 16 : (rank + 1) * 16], atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize("draft_provider", ["flashinfer"], indirect=True)
def test_runtime_has_no_host_planning_readback_or_allocator_growth(monkeypatch, draft_provider):
    f = Fixture(4, 16, 1, 16, 4, False, 2048, provider=draft_provider)
    f.stage([2048, 1, 4096, 0], [0, -1, 2, -1])
    f.run()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime host planning/readback/synchronization")

    with monkeypatch.context() as checks:
        for name in ("item", "cpu", "numpy", "tolist"):
            checks.setattr(torch.Tensor, name, forbidden)
        checks.setattr(torch.cuda, "synchronize", forbidden)
        checks.setattr(f.op._implementation._native, "plan", forbidden)
        f.run()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "b,causal,window,p,splits",
    ci_cases(
        [1, 3],
        MASKS,
        [(16, None), (1, None), (1, "1")],
        core=[(3, False, 2048, 1, None)],
    ),
)
def test_prepare_and_run_matches_prepare_then_run(
    b, causal, window, p, splits, draft_provider, monkeypatch
):
    # One CuTe split leaves fewer publishing threads than P=1 pages (strided loop).
    if splits is not None:
        monkeypatch.setenv("CTI_DRAFT_SPLITS", splits)
    q = 9
    f = Fixture(b, q, p, 32, 8, causal, window, provider=draft_provider)
    f.stage([128] * b, list(range(b)))

    def fused():
        return f.op.prepare_and_run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fused()
    torch.cuda.current_stream().wait_stream(stream)
    w = window or 4096
    cases = [
        ([w - 1] + [w + 1] * (b - 1), list(range(b))),
        ([4160 - q] * b, list(reversed(range(b)))),
        ([0] + [w] * (b - 1), [-1] + list(range(1, b))),
        ([w] * b, [-1] * b),
    ]
    for turn, (contexts, slots) in enumerate(cases):
        f.stage(contexts, slots, turn)
        # Poison every output prepare must refresh; unused page-index tails stay untouched.
        for metadata in (
            f.query_ptr,
            f.page_ptr,
            f.last,
            f.page_idx[: int(f.expected_page_ptr[-1])],
        ):
            metadata.fill_(-7)
        f.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        # The fused call refreshed every metadata output before it returned.
        f.check_metadata()
        result = f.out.clone()
        f.out.fill_(float("nan"))
        f.run()
        torch.cuda.synchronize()
        assert torch.equal(result, f.out)
        torch.testing.assert_close(result.float(), f.oracle(), atol=ATOL, rtol=RTOL)
        for row, slot in enumerate(slots):
            if slot == -1:
                assert torch.count_nonzero(result[row * q : (row + 1) * q]) == 0
