# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""GPU qualification only: parity, scalar/vector tails and self-cleaning replay."""

from functools import cache

import pytest
import torch

from ctikernels.sampling import RowArgmax

from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
OUTPUTS = [torch.int32, torch.int64]


@cache
def operation():
    """One CUDA operation per process; tests patch its provider."""
    return RowArgmax("cuda")


def new_workspace(capacity, device="cuda"):
    return torch.zeros(RowArgmax.workspace_size(capacity), dtype=torch.uint8, device=device)


def check(logits, out, workspace):
    operation()(logits, out, workspace)
    assert torch.equal(out, torch.argmax(logits, dim=-1))
    assert not workspace.any().item()


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 128])
@pytest.mark.parametrize("vocab", [1, 7, 4097, 151936, 248320])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_random(rows, vocab, dtype, out_dtype):
    logits = torch.randn(rows, vocab, dtype=dtype, device="cuda")
    out = torch.empty(rows, dtype=out_dtype, device="cuda")
    work = new_workspace(rows + 3)
    assert operation().provider == "cute"
    check(logits, out, work)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
@pytest.mark.parametrize("layout", ["aligned", "padded", "offset", "alternating"])
def test_dflash_row_argmax_edges(dtype, out_dtype, layout):
    rows, vocab = 4, 4097
    stride = 4112 if layout != "alternating" else 4099
    offset = 1 if layout == "offset" else 0
    storage = torch.empty(rows, stride, dtype=dtype, device="cuda")
    x = storage[:, offset : offset + vocab]
    if layout == "aligned":
        # Still exercise a tail, but only the first row needs to be aligned.
        x = torch.empty(rows, vocab, dtype=dtype, device="cuda")
    out = torch.empty(rows, dtype=out_dtype, device="cuda")
    work = new_workspace(rows + 7)
    # Maxima span all possible splits, with a nonzero first winning index.
    x.fill_(-float("inf"))
    x[:, 7] = 9
    x[:, vocab // 2] = 9
    x[:, -1] = 9
    check(x, out, work)
    assert (out == 7).all().item()
    x.fill_(-float("inf"))
    check(x, out, work)
    assert (out == 0).all().item()
    x.fill_(-1)
    x[:, 3] = -0.0
    x[:, vocab // 2] = +0.0
    check(x, out, work)
    assert (out == 3).all().item()
    x[:, 3] = +0.0
    x[:, vocab // 2] = -0.0
    check(x, out, work)
    x[:, 0] = float("inf")
    x[:, 11] = float("nan")
    x[:, -1] = -float("nan")
    check(x, out, work)
    assert (out == 11).all().item()
    # Cover finite negatives, subnormals and both infinities without arithmetic.
    x.fill_(-float("inf"))
    x[:, 13] = torch.finfo(dtype).tiny / 2
    x[:, 25] = -torch.finfo(dtype).tiny / 2
    check(x, out, work)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_graph(dtype, out_dtype):
    x = torch.randn(16, 4097, device="cuda", dtype=dtype)
    out = torch.empty(16, device="cuda", dtype=out_dtype)
    work = new_workspace(20)
    operation().warm_up(x, out, work)
    check(x, out, work)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()(x, out, work)
        operation()(x, out, work)  # reuse inside a graph as well as across replays
    for i in range(100):
        x.normal_()
        if i % 10 == 0:
            x[:, (i * 37) % x.shape[1]] = float("nan")
        graph.replay()
        assert torch.equal(out, x.argmax(-1))
        assert not work.any().item()


def test_dflash_row_argmax_one_launch():
    x = torch.randn(4, 4097, device="cuda")
    out = torch.empty(4, dtype=torch.int64, device="cuda")
    work = new_workspace(4)
    operation().warm_up(x, out, work)
    from benchmark.dflash_sampling.bench import count_launches

    assert count_launches(lambda: operation()(x, out, work))["kernel_launches"] == 1


def chain_proposals(expected, proposals, first, vocab):
    """first == Q-1 means every proposal matches (including the bonus row)."""
    proposals.copy_(expected[:, :-1])
    for b, position in enumerate(first):
        if position < proposals.shape[1]:
            proposals[b, position] = (expected[b, position] + 1) % vocab


def check_chain(out, expected, first, workspace):
    actual = out.view_as(expected)
    prefix = (
        torch.arange(expected.shape[1], device=out.device)[None, :]
        <= torch.tensor(first, device=out.device)[:, None]
    )
    assert torch.equal(actual[prefix], expected[prefix])
    assert ((actual[~prefix] == expected[~prefix]) | (actual[~prefix] == -1)).all().item()
    assert not workspace.any().item()


@pytest.mark.parametrize(
    "batch,block,vocab",
    ci_cases(
        [1, 3, 4],
        [2, 5, 16, 32],
        [40, 4099, 151936],
        core=[(1, 2, 40), (3, 5, 4099), (4, 16, 151936), (3, 32, 40)],
    )
    + [(2, q, 40) for q in (33, 64, 65, 128)]
    + [(4, 64, 151936)],  # Q > 32 long-row chain mode
)
@pytest.mark.parametrize("proposal_dtype", OUTPUTS)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_chain(monkeypatch, batch, block, vocab, proposal_dtype, out_dtype):
    from ctikernels.sampling._dflash.cute import argmax

    monkeypatch.setattr(argmax, "CHAIN_MIN_BYTES", 0)
    gen = torch.Generator().manual_seed(batch + block + vocab)
    x = torch.randn(batch * block, vocab, generator=gen, dtype=torch.bfloat16).cuda()
    if block > 32 and vocab >= 151936:
        # Force the long-row chain mode (normally only for logits larger than L2).
        monkeypatch.setattr(operation()._provider, "l2_bytes", 0)
    expected = x.argmax(-1).view(batch, block)
    proposals = torch.empty(2 * batch, block + 2, device="cuda", dtype=proposal_dtype)[::2, 1:block]
    out = torch.empty(batch * block, device="cuda", dtype=out_dtype)
    work = new_workspace(batch * block + 7)
    chain = {"proposals": proposals, "block_size": block}
    operation().warm_up(x, out, work, **chain)
    # Include every possible first mismatch and no mismatch, plus random mixtures.
    positions = [[i] * batch for i in range(block)]
    positions += torch.randint(block, (3, batch), generator=gen).tolist()
    for first in positions:
        chain_proposals(expected, proposals, first, vocab)
        out.fill_(-99)
        operation()(x, out, work, **chain)
        check_chain(out, expected, first, work)
        check(x, out, work)  # Same oversized workspace alternates chain/plain.


@pytest.mark.parametrize(
    "batch,block,vocab",
    [(1, 33, 4099), (2, 40, 40961), (3, 64, 151936), (4, 33, 7)],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 1])
def test_dflash_row_argmax_chain_long_rows(monkeypatch, batch, block, vocab, dtype, offset):
    # Long-row chain mode (Q > 32 above L2 on GB10; forced here on every device):
    # split early positions, whole-row CTAs with mid-row stops, aligned or
    # unaligned rows, odd tails.
    from ctikernels.sampling._dflash.cute import argmax

    monkeypatch.setattr(argmax, "CHAIN_MIN_BYTES", 0)
    gen = torch.Generator().manual_seed(batch * block + vocab + offset)
    storage = torch.randn(batch * block, vocab + 9, generator=gen).to(dtype).cuda()
    x = storage[:, offset : offset + vocab]
    provider = operation()._provider
    monkeypatch.setattr(provider, "l2_bytes", 0)
    monkeypatch.setattr(provider, "is_integrated", True)
    expected = x.argmax(-1).view(batch, block)
    proposals = torch.empty(batch, block - 1, device="cuda", dtype=torch.int32)
    out = torch.empty(batch * block, device="cuda", dtype=torch.int64)
    work = new_workspace(batch * block + 5)
    chain = {"proposals": proposals, "block_size": block}
    operation().warm_up(x, out, work, **chain)
    early = argmax.LONG_EARLY_POSITIONS
    positions = [
        [i] * batch for i in sorted({0, early - 1, early, early + 1, block - 2, block - 1})
    ]
    positions += torch.randint(block, (4, batch), generator=gen).tolist()
    for first in positions:
        chain_proposals(expected, proposals, first, vocab)
        out.fill_(-99)
        operation()(x, out, work, **chain)
        check_chain(out, expected, first, work)


@pytest.mark.parametrize("proposal_dtype", OUTPUTS)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_chain_graph(monkeypatch, proposal_dtype, out_dtype):
    from ctikernels.sampling._dflash.cute import argmax

    monkeypatch.setattr(argmax, "CHAIN_MIN_BYTES", 0)
    batch, block, vocab = 4, 16, 4099
    gen = torch.Generator().manual_seed(32)
    x = torch.randn(batch * block, vocab, generator=gen, dtype=torch.bfloat16).cuda()
    expected = x.argmax(-1).view(batch, block)
    proposals = torch.empty(batch * 2, block + 3, dtype=proposal_dtype, device="cuda")[::2, 1:block]
    out = torch.empty(batch * block, dtype=out_dtype, device="cuda")
    work = new_workspace(batch * block + 3)
    chain = {"proposals": proposals, "block_size": block}
    first = [0, block - 2, block - 1, 3]
    chain_proposals(expected, proposals, first, vocab)
    operation().warm_up(x, out, work, **chain)
    operation()(x, out, work, **chain)
    check_chain(out, expected, first, work)
    check(x, out, work)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()(x, out, work, **chain)
    check_chain(out, expected, first, work)
    for i in range(20):
        first = torch.randint(block, (batch,), generator=gen).tolist()
        first[i % batch] = block - 1 if i % 2 else 0
        chain_proposals(expected, proposals, first, vocab)
        out.fill_(-99)
        graph.replay()
        check_chain(out, expected, first, work)
        check(x, out, work)


@pytest.mark.parametrize("out_dtype", OUTPUTS)
@pytest.mark.parametrize("offset", [0, 1])
def test_dflash_row_argmax_packed_bf16_edges(out_dtype, offset):
    from ctikernels.sampling._dflash.cute.argmax import split_count

    rows, vocab = 7, 4099  # Odd tail, with aligned or uniformly unaligned row starts.
    storage = torch.full((rows, 4112), -float("inf"), dtype=torch.bfloat16, device="cuda")
    x = storage[:, offset : offset + vocab]
    splits = split_count(
        rows,
        vocab,
        x.element_size(),
        torch.cuda.get_device_properties(x.device).multi_processor_count,
    )
    boundary = (((vocab + 7) // 8 + splits - 1) // splits) * 8
    assert 0 < boundary < vocab
    x[0, [3, 6, boundary, vocab - 1]] = float("nan")
    x[1, 3], x[1, 4] = -0.0, +0.0
    x[2, 3], x[2, 4] = +0.0, -0.0
    # Row 3 stays all -inf. Rows 4/5 tie across splits or within one packed vector.
    x[4, [boundary - 1, boundary, vocab - 1]] = 9
    x[5, [5, 7, 13]] = 9
    x[6, -1] = 9
    out = torch.empty(rows, dtype=out_dtype, device="cuda")
    work = new_workspace(rows)
    check(x, out, work)
    assert out.tolist() == [3, 3, 3, 0, boundary - 1, 5, vocab - 1]


@pytest.mark.parametrize("proposal_dtype", OUTPUTS)
@pytest.mark.parametrize("splits", [1, 3])
def test_dflash_row_argmax_chain_beyond_grid_y(monkeypatch, proposal_dtype, splits):
    # B=2048, Q=32: 65536 rows and 8 MiB of BF16 logits reach the chain size
    # threshold and exercise a second grid-stride iteration.
    from ctikernels.sampling._dflash.cute import argmax

    # Large row counts normally choose one split; exercise arrival counters and
    # shared-memory reuse with multiple splits as well.
    monkeypatch.setattr(argmax, "split_count", lambda *args: splits)
    batch, block, vocab = 2048, 32, 64
    assert batch * block > 65535
    assert batch * block * vocab * 2 >= argmax.CHAIN_MIN_BYTES
    gen = torch.Generator().manual_seed(65536)
    x = torch.randn(batch * block, vocab, generator=gen, dtype=torch.bfloat16).cuda()
    expected = x.argmax(-1).view(batch, block)
    proposals = torch.empty(batch, block - 1, dtype=proposal_dtype, device="cuda")
    first = torch.randint(block, (batch,), generator=gen).tolist()
    chain_proposals(expected, proposals, first, vocab)
    out = torch.full((batch * block,), -99, dtype=torch.int32, device="cuda")
    work = new_workspace(batch * block)
    chain = {"proposals": proposals, "block_size": block}
    assert argmax.Provider._chain(x, proposals) is proposals
    operation().warm_up(x, out, work, **chain)
    operation()(x, out, work, **chain)
    torch.cuda.synchronize()
    check_chain(out, expected, first, work)


@pytest.mark.parametrize("dtype", OUTPUTS)
@pytest.mark.parametrize("vocab", [4097, 600001])
def test_dflash_row_argmax_q1_capture(dtype, vocab):
    logits = torch.zeros(4, vocab, device="cuda")
    logits[:, 3] = logits[:, -1] = 2
    proposals = torch.empty_strided((4, 0), (17, 9), dtype=dtype, device="cuda")
    out = torch.empty(4, dtype=dtype, device="cuda")
    work = new_workspace(4)
    operation().warm_up(logits, out, work, proposals=proposals, block_size=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()(logits, out, work, proposals=proposals, block_size=1)
    for _ in range(3):
        graph.replay()
        assert out.cpu().tolist() == [3] * 4
        assert not work.any().item()


@pytest.mark.parametrize("batch,block", [(1024, 17), (4096, 17)])
@pytest.mark.parametrize("vocab", [16, 4095, 4097])
@pytest.mark.parametrize("dtype", DTYPES)
def test_dflash_row_argmax_many_chain_rows(monkeypatch, batch, block, vocab, dtype):
    from ctikernels.sampling._dflash.cute import argmax

    monkeypatch.setattr(argmax, "CHAIN_MIN_BYTES", 0)
    # Offset and padded rows exercise both alignment paths and scalar tails.
    x = torch.randn(batch * block, vocab + 3, dtype=dtype, device="cuda")[:, 1 : vocab + 1]
    expected = x.argmax(-1).view(batch, block)
    proposals = torch.empty(batch, block - 1, dtype=torch.int64, device="cuda")
    out = torch.empty(batch * block, dtype=torch.int32, device="cuda")
    work = new_workspace(batch * block + 7)
    assert argmax.Provider._chain(x, proposals) is proposals
    for first in ([block - 1] * batch, [i % block for i in range(batch)], [0] * batch):
        chain_proposals(expected, proposals, first, vocab)
        operation()(x, out, work, proposals=proposals, block_size=block)
        check_chain(out, expected, first, work)


@pytest.mark.parametrize("vocab", [16, 4095, 4097])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_many_plain_rows(vocab, dtype, out_dtype):
    rows = 70001
    x = torch.randn(rows, vocab + 3, dtype=dtype, device="cuda")[:, 1 : vocab + 1]
    out = torch.empty(rows, dtype=out_dtype, device="cuda")
    work = new_workspace(rows + 3)
    check(x, out, work)
    check(x, out, work)


@pytest.mark.parametrize(
    "mode,proposal_dtype",
    [("plain", None)]
    + [(mode, dtype) for mode in ("chain", "stream", "long") for dtype in OUTPUTS],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUTPUTS)
def test_dflash_row_argmax_runtime_shapes_capture(
    monkeypatch, mode, proposal_dtype, dtype, out_dtype
):
    from ctikernels.sampling._dflash.cute import argmax

    # Isolate the cache so another test cannot hide shape/split specializations.
    # The real chain threshold applies: small chain shapes run the plain kernel,
    # so the family compiles at most one variant per mode, never one per shape.
    monkeypatch.setattr(argmax, "_COMPILED", {})
    device = torch.device("cuda", torch.cuda.current_device())
    provider = operation()._provider
    monkeypatch.setattr(provider, "is_integrated", mode == "long")
    monkeypatch.setattr(provider, "l2_bytes", 0)
    if mode == "plain":
        shapes = [(1, 1, 151936), (70001, 1, 16), (17, 1, 4095), (4, 1, 4097)]
    elif mode == "chain":
        shapes = [(1, 17, 151936), (1024, 17, 4095), (4096, 17, 16), (3, 5, 4097)]
    else:
        shapes = [(1, 33, 151936), (2122, 33, 16), (2, 65, 4095), (3, 40, 4097)]
    splits_seen = set()
    modes = set()
    for index, (batch, block, vocab) in enumerate(shapes):
        # Both row and proposal strides vary independently of the compile key.
        offset = index % 2
        storage = torch.randn(batch * block, vocab + index + 8, dtype=dtype, device=device)
        x = storage[:, offset : offset + vocab]
        expected = x.argmax(-1).view(batch, block)
        out = torch.empty(batch * block, dtype=out_dtype, device=device)
        work = new_workspace(batch * block + index, device)
        proposals = None
        kwargs = {}
        if proposal_dtype is not None:
            proposals = torch.empty(batch * 2, block + index, dtype=proposal_dtype, device=device)[
                ::2, : block - 1
            ]
            proposals.copy_(expected[:, :-1])
            kwargs = {"proposals": proposals, "block_size": block}
        chain, stream, long_rows = provider._mode(
            x.numel() * x.element_size(), proposals is not None, block
        )
        key = (device, dtype, out_dtype, proposal_dtype if chain else None, stream, long_rows)
        if key not in argmax._COMPILED:
            operation().warm_up(x, out, work, **kwargs)
            operation()(x, out, work, **kwargs)
            modes.add(key)
            assert set(argmax._COMPILED) == modes
        waves = argmax.CHAIN_LONG_WAVES if mode == "stream" else 1
        splits_seen.add(
            argmax.split_count(batch * block, vocab, x.element_size(), provider.sm_count, waves)
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()(x, out, work, **kwargs)
            operation()(x, out, work, **kwargs)
        assert set(argmax._COMPILED) == modes
        for replay in range(3):
            first = (
                [block - 1] * batch if replay == 0 else [(i + replay) % block for i in range(batch)]
            )
            if proposals is not None:
                chain_proposals(expected, proposals, first, vocab)
            out.fill_(-99)
            graph.replay()
            if proposals is None:
                assert torch.equal(out, expected.flatten())
                assert not work.any().item()
            else:
                check_chain(out, expected, first, work)
        assert set(argmax._COMPILED) == modes
    # Each family reaches its own mode; the small chain shapes fall back to plain.
    assert (device, dtype, out_dtype, proposal_dtype, mode == "stream", mode == "long") in modes
    assert len(modes) <= 2
    if mode != "long":
        assert len(splits_seen) > 1


@pytest.mark.parametrize(
    "block,capacity,vocab",
    [
        (16, 4, 151936),  # crosses the chain threshold: B=1 plain, B>=2 chain
        (33, 12, 151936),  # crosses the L2 threshold: chain, then stream/long rows
        (5, 64, 4097),  # all below the chain threshold
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dflash_row_argmax_capacity_warm_up_covers_smaller_batches(
    monkeypatch, block, capacity, vocab, dtype
):
    from ctikernels.sampling._dflash.cute import argmax

    monkeypatch.setattr(argmax, "_COMPILED", {})
    device = torch.device("cuda", torch.cuda.current_device())
    provider = operation()._provider
    rows = capacity * block
    x = torch.randn(rows, vocab, dtype=dtype, device=device)
    out = torch.empty(rows, dtype=torch.int64, device=device)
    work = new_workspace(rows, device)
    proposals = torch.empty(capacity, block - 1, dtype=torch.int64, device=device)
    # One warm-up at capacity, then capture every smaller batch with no compile.
    operation().warm_up(x, out, work, proposals=proposals, block_size=block)
    warmed = set(argmax._COMPILED)
    reachable = {
        provider._mode(b * block * vocab * x.element_size(), True, block)
        for b in range(1, capacity + 1)
    }
    assert {key[3:] for key in warmed} == {
        (torch.int64 if chain else None, stream, long_rows)
        for chain, stream, long_rows in reachable
    }
    for batch in range(1, capacity + 1):
        n = batch * block
        xb, outb, pb = x[:n], out[:n], proposals[:batch]
        expected = xb.argmax(-1).view(batch, block)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()(xb, outb, work, proposals=pb, block_size=block)
        assert set(argmax._COMPILED) == warmed
        for first in ([block - 1] * batch, [i % block for i in range(batch)]):
            chain_proposals(expected, pb, first, vocab)
            outb.fill_(-99)
            graph.replay()
            check_chain(outb, expected, first, work)
