# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DN5's post-filter mask, frozen FP32 reduction, and borrowed-buffer contract."""

import pytest
import torch

from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.ordinary import RequestLocalSampling
from ctikernels.sampling._dflash.reference.ops import min_p_row, probability_row, probability_sum

from .test_dflash_probabilities import probability_data
from .test_dflash_sampling_reference import buffers, verify

HINTS = ("none", "top_k", "top_p", "both", "auto")


def sorted_min_p(row, minimum):
    """Independent FP32 sorted selection, then sum in original token order."""
    if minimum <= 0:
        return row.clone()
    ordered, ids = torch.sort(row, descending=True, stable=True)
    pivot = torch.tensor(minimum, dtype=torch.float32) * ordered[0]
    selected = ids[ordered >= pivot]
    masked = torch.zeros_like(row)
    masked[selected] = row[selected]
    result = torch.zeros_like(row)
    result[selected] = row[selected] / probability_sum(masked)
    return result


def assert_bits(actual, expected):
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def min_p_data(batch, length, vocab, dtype, hint):
    # Concentrated support keeps large top-p fixtures inexpensive. Distinct
    # positions change the maximum's location and catch row-index mistakes.
    logits = torch.full((batch, length, vocab), -12.0)
    head = torch.tensor([0, 0, -0.6931471824645996, -1.3862943649291992] * 4)
    logits[..., :16] = head[: min(16, vocab)]
    for i in range(length):
        logits[:, i] = logits[:, i].roll(i % vocab, dims=-1)
    data = probability_data(logits.to(dtype))
    if hint in ("top_k", "both", "auto"):
        data["top_k"].fill_(min(12, vocab))
        if batch >= 4:
            data["top_k"][2] = min(2048, vocab)  # Mix compact and large rows.
    if hint in ("top_p", "both", "auto"):
        data["top_p"].fill_(0.8)
    if batch >= 2:
        data["temperatures"][1] = 0
    if batch >= 3:
        data["request_indices"][-1] = -1
        data["logits"][-1].fill_(float("nan"))
        data["temperatures"][-1] = float("nan")
    return data


def tie_min_p(row):
    """Find an admitted FP32 ratio whose rounded product is an existing p."""
    maximum = row.max()
    values = row.unique(sorted=True)
    for value in values[values > 0]:
        ratio = value / maximum
        for candidate in (
            ratio,
            torch.nextafter(ratio, torch.tensor(0.0)),
            torch.nextafter(ratio, torch.tensor(1.0)),
        ):
            if 0 < candidate <= 1 and candidate * maximum == value:
                return candidate
    raise AssertionError("no FP32 tie pivot")


@pytest.mark.parametrize("vocab", [16, 4097])
@pytest.mark.parametrize("minimum", [0, 0.05, 0.5, 1])
@pytest.mark.parametrize("hint", HINTS)
def test_min_p_reference(vocab, minimum, hint):
    data = min_p_data(4, 2, vocab, torch.float32, hint)
    op = RequestLocalChainSampling("cpu", 2, vocab)
    op.probabilities(**data, filters=hint)
    original = data["out_probs"].clone()
    expected = original.clone()
    for b in (0, 2):
        for i in range(2):
            expected[b, i] = sorted_min_p(original[b, i], minimum)
    mins = torch.full((4,), float(minimum))
    mins[1] = mins[-1] = float("nan")  # Must not be read on greedy/padding.
    op.probabilities(**data, min_p=mins, filters=hint)
    assert_bits(data["out_probs"], expected)


def test_min_p_ties_and_maxima():
    row = torch.tensor([0.375, 0.375, 0.1875, 0.0625])
    assert_bits(min_p_row(row, 0.5), torch.tensor([0.4, 0.4, 0.2, 0]))
    assert_bits(min_p_row(row, 1), torch.tensor([0.5, 0.5, 0, 0]))
    assert_bits(min_p_row(row, 0), row)
    assert_bits(
        min_p_row(row, torch.nextafter(torch.tensor(0.5), torch.tensor(1.0))),
        torch.tensor([0.5, 0.5, 0, 0]),
    )


def test_min_p_chunk_tree_and_sequential_chunk_sum():
    gen = torch.Generator().manual_seed(508)
    row = torch.rand(3 * 4096 + 1, generator=gen)
    row /= probability_sum(row)
    assert_bits(min_p_row(row, 0.05), sorted_min_p(row, 0.05))


def test_min_p_after_top_k_top_p_order():
    logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    data = probability_data(
        logits.reshape(1, 1, 4),
        top_k=torch.tensor([3], dtype=torch.int32),
        top_p=torch.tensor([0.55]),
    )
    RequestLocalChainSampling("cpu", 1, 4).probabilities(**data, min_p=torch.tensor([0.6]))
    # Applying min-p first removes 0.2/0.1; top-p then keeps only the 0.4 token.
    before = logits.masked_fill(logits.exp() < torch.tensor(0.6) * logits.exp().max(), -torch.inf)
    early = probability_row(before, 1, 3, 0.55)
    assert torch.nonzero(early).flatten().tolist() == [0]
    assert torch.nonzero(data["out_probs"][0, 0]).flatten().tolist() == [0, 1]


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "device", "grad", "stride", "out_alias", "work_alias"]
)
def test_min_p_validation(bad):
    data = probability_data(torch.zeros(2, 1, 4))
    minimum = torch.zeros(2)
    if bad == "dtype":
        minimum = minimum.half()
    elif bad == "shape":
        minimum = minimum.reshape(2, 1)
    elif bad == "device":
        minimum = torch.empty(2, device="meta")
    elif bad == "grad":
        minimum.requires_grad_()
    elif bad == "stride":
        minimum = torch.zeros(4)[::2]
    elif bad == "out_alias":
        minimum = data["out_probs"].flatten()[:2]
    else:
        minimum = data["workspace"][:8].view(torch.float32)
    with pytest.raises((TypeError, ValueError)):
        RequestLocalChainSampling("cpu", 1, 4).probabilities(**data, min_p=minimum)


def test_min_p_layout_and_launch_dispatch(monkeypatch):
    op = RequestLocalChainSampling("cpu", 1, 4097)
    ordinary = RequestLocalSampling("cpu", 4097)
    chain = op.workspace_layout(3, "probabilities")
    layout = ordinary.workspace_layout(3)
    assert layout["regions"][: len(chain["regions"])] == chain["regions"]
    regions = {r["name"]: r for r in layout["regions"]}
    state, maximum = regions["row_states"], regions["min_p_chunk_maxima"]
    assert maximum["offset"] == state["offset"] + state["aligned_extent"]
    assert op.probability_launch_counts == dict(zip(HINTS, (3, 7, 8, 11, 11), strict=True))
    assert op.probability_min_p_launches == 2
    assert ordinary.launch_counts == dict(zip(HINTS, (4, 9, 9, 12, 12), strict=True))
    calls = []
    monkeypatch.setattr(op._provider, "run", lambda mode, *args: calls.append(mode))
    data = probability_data(torch.zeros(1, 1, 4097))
    for hint in HINTS:
        calls.clear()
        op.probabilities(**data, filters=hint)
        assert calls == ["probabilities"]
        calls.clear()
        op.probabilities(**data, min_p=torch.zeros(1), filters=hint)
        assert calls == ["probabilities_min_p", "min_p_renormalize"]


def test_min_p_empty_batch():
    data = probability_data(torch.empty(0, 1, 16))
    RequestLocalChainSampling("cpu", 1, 16).probabilities(**data, min_p=torch.empty(0))


def test_min_p_cute_dispatch_and_capture_without_allocations(monkeypatch):

    from .test_dflash_cute_boundary import BorrowedStorageOnly

    op = RequestLocalChainSampling("cpu", 3, 16)
    from .test_dflash_symb import stub_provider

    provider = stub_provider(monkeypatch)
    op._provider = provider
    capturing, compiled = False, []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    def compile_stub(kernel, tensors):
        assert not capturing
        compiled.append(kernel)
        return lambda *args: None

    monkeypatch.setattr(provider, "_compile", compile_stub)
    keys = op.warm_up(batch_capacity=2, dtypes=(torch.float32,), modes=())
    assert len(compiled) == len(keys) == 11  # L no longer specializes; renorm shared.
    assert ("min_p", (16,)) in compiled
    for hint in HINTS:
        assert ("probabilities", (16, hint, 1), {"output_maxima": True}) in compiled
    data = probability_data(torch.zeros(2, 3, 16))
    minimum = torch.zeros(2)
    capturing = True
    with BorrowedStorageOnly([*data.values(), minimum]):
        op.probabilities(**data, min_p=minimum)
        op.probabilities(**data)
    assert len(compiled) == 11


def point_min_p_data():
    op, data = buffers(2, 2, 4)
    probs = probability_data(torch.tensor([0.0, -4.0, -8.0, -12.0]).expand(2, 3, 4).contiguous())
    probs["out_probs"] = data["p"]
    op.probabilities(**probs, min_p=torch.tensor([0.5, 0.5]))
    data["drafts"][0].fill_(1)  # Below pivot: always reject at position zero.
    data["drafts"][1].fill_(0)  # Unit target: accept both and draw a bonus.
    return op, data


def test_point_mass_consumes_min_p_filtered_probabilities():
    op, data = point_min_p_data()
    verify(op, data, point=True)
    assert data["accepted_counts"].tolist() == [0, 2]
    assert data["out_tokens"].tolist() == [[0, -1, -1], [0, 0, 0]]


def test_probability_warm_up_variants_and_capture_guard(monkeypatch):
    op = RequestLocalChainSampling("cpu", 2, 4)
    assert op.warm_up(batch_capacity=2, modes=()) == ()
    assert (
        op.warm_up(
            batch_capacity=1,
            dtypes=(torch.float32,),
            filters=("none",),
            min_p=(True,),
            modes=(),
        )
        == ()
    )
    op.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        op.warm_up(batch_capacity=1, modes=())
