# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from ctikernels.sampling import CategoricalSampling, ChainSampling, RequestLocalCategorical
from ctikernels.sampling._dflash.reference import fixed_mass
from ctikernels.sampling._dflash.reference.ops import _uint64_to_float32
from ctikernels.sampling._request_local import filter_requests


@pytest.fixture
def sampling_provider(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    sampling = ModuleType("flashinfer.sampling")
    sampling.softmax = Mock(return_value=torch.empty(2, 3))
    sampling.sampling_from_probs = Mock(return_value=torch.tensor([2, 0]))
    sampling.top_k_sampling_from_probs = Mock(return_value=torch.tensor([2, 0]))
    sampling.top_p_sampling_from_probs = Mock(return_value=torch.tensor([2, 0]))
    sampling.top_k_top_p_sampling_from_probs = Mock(return_value=torch.tensor([2, 0]))
    sampling.top_k_renorm_probs = Mock()
    sampling.top_p_renorm_probs = Mock()
    sampling.sample_requests = Mock(return_value=torch.tensor([2, 0]))
    monkeypatch.setattr("ctikernels.sampling.inference.sample_requests", sampling.sample_requests)
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.sampling = sampling
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.sampling", sampling)
    return sampling


@pytest.mark.parametrize(
    ("use_top_k", "use_top_p", "provider_name"),
    [
        (False, False, "sampling_from_probs"),
        (True, False, "top_k_sampling_from_probs"),
        (False, True, "top_p_sampling_from_probs"),
        (True, True, "top_k_top_p_sampling_from_probs"),
    ],
)
def test_categorical_routes_filters_without_copying(
    sampling_provider: ModuleType,
    use_top_k: bool,
    use_top_p: bool,
    provider_name: str,
) -> None:
    operation = CategoricalSampling()
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    top_k = torch.tensor([2, 1], dtype=torch.int32) if use_top_k else None
    top_p = torch.tensor([0.8, 0.9]) if use_top_p else None

    result = operation.sample(probabilities, top_k=top_k, top_p=top_p)

    provider = getattr(sampling_provider, provider_name)
    assert result is provider.return_value
    provider.assert_called_once()
    expected = (probabilities, *(value for value in (top_k, top_p) if value is not None))
    assert len(provider.call_args.args) == len(expected)
    assert all(
        actual is argument
        for actual, argument in zip(provider.call_args.args, expected, strict=True)
    )
    assert provider.call_args.kwargs == {}
    for name in (
        "softmax",
        "sampling_from_probs",
        "top_k_sampling_from_probs",
        "top_p_sampling_from_probs",
        "top_k_top_p_sampling_from_probs",
    ):
        if name != provider_name:
            getattr(sampling_provider, name).assert_not_called()


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_categorical_forwards_temperature_softmax_and_pdl(
    sampling_provider: ModuleType, enable_pdl: bool
) -> None:
    operation = CategoricalSampling()
    logits = torch.tensor([[0.1, 0.7, -0.2], [1.2, 0.3, -0.4]])
    temperatures = torch.tensor([0.5, 0.7])

    result = operation.softmax(logits, temperatures, enable_pdl=enable_pdl)

    provider = sampling_provider.softmax
    assert result is provider.return_value
    provider.assert_called_once()
    assert len(provider.call_args.args) == 2
    assert provider.call_args.args[0] is logits
    assert provider.call_args.args[1] is temperatures
    assert provider.call_args.kwargs == {"enable_pdl": enable_pdl}


def test_chain_probability_filters_use_materialized_q_in_the_same_order(sampling_provider):
    provider = sampling_provider
    provider.top_k_renorm_probs = Mock(return_value=torch.full((2, 3), 1 / 3))
    provider.top_p_renorm_probs = Mock(return_value=torch.ones(2, 3))
    logits = torch.ones(2, 3, dtype=torch.bfloat16)
    temperatures = torch.tensor([0.7, 1.2])
    ks, ps = torch.tensor([2, 3]), torch.tensor([0.9, 1.0])
    result = ChainSampling().probabilities(logits, temperatures, ks, ps)
    assert result is provider.top_p_renorm_probs.return_value
    assert provider.softmax.call_args.args[0].dtype == torch.float32
    assert provider.softmax.call_args.args[1] is temperatures
    assert provider.top_k_renorm_probs.call_args.args[0] is provider.softmax.return_value
    assert provider.top_k_renorm_probs.call_args.args[1] is ks
    assert provider.top_p_renorm_probs.call_args.args[0] is provider.top_k_renorm_probs.return_value
    assert provider.top_p_renorm_probs.call_args.args[1] is ps


def test_chain_rejects_cpu_probabilities_before_calling_provider(sampling_provider):
    with pytest.raises(ValueError, match="CUDA FP32"):
        ChainSampling().sample(torch.ones(2, 3), seed=torch.tensor([1]), offset=torch.tensor([0]))
    sampling_provider.sampling_from_probs.assert_not_called()


def test_categorical_resolves_provider_module_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    sampling_provider: ModuleType,
) -> None:
    operation = CategoricalSampling()
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("blocked_flashinfer"))
    monkeypatch.delitem(sys.modules, "flashinfer.sampling")
    logits, temperatures = torch.ones(2, 3), torch.ones(2)

    for _ in range(2):
        probabilities = operation.softmax(logits, temperatures, enable_pdl=False)
        assert operation.sample(probabilities) is sampling_provider.sampling_from_probs.return_value

    assert sampling_provider.softmax.call_count == 2
    assert sampling_provider.sampling_from_probs.call_count == 2


def test_categorical_resolves_only_functions_used_by_each_call(
    monkeypatch: pytest.MonkeyPatch,
    sampling_provider: ModuleType,
) -> None:
    monkeypatch.delattr(sampling_provider, "softmax")
    monkeypatch.delattr(sampling_provider, "top_k_sampling_from_probs")
    monkeypatch.delattr(sampling_provider, "top_p_sampling_from_probs")
    monkeypatch.delattr(sampling_provider, "top_k_top_p_sampling_from_probs")
    operation = CategoricalSampling()
    probabilities = torch.full((2, 3), 1 / 3)
    top_k = torch.tensor([2, 1], dtype=torch.int32)

    assert operation.sample(probabilities) is sampling_provider.sampling_from_probs.return_value
    with pytest.raises(AttributeError, match="top_k_sampling_from_probs"):
        operation.sample(probabilities, top_k=top_k)

    provider = Mock(return_value=torch.tensor([1, 0]))
    monkeypatch.setattr(sampling_provider, "top_k_sampling_from_probs", provider, raising=False)
    assert operation.sample(probabilities, top_k=top_k) is provider.return_value
    provider.assert_called_once()
    assert provider.call_args.args[0] is probabilities
    assert provider.call_args.args[1] is top_k


def test_request_local_forwards_whole_batch_and_rng_in_one_call(
    sampling_provider: ModuleType,
) -> None:
    operation = RequestLocalCategorical()
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    seeds, positions = (17, 23), (0, 4)
    provider = sampling_provider.sample_requests

    result = operation.sample(probabilities, seeds, positions)

    assert result is provider.return_value
    provider.assert_called_once_with(probabilities, seeds, positions)
    assert provider.call_args.args[0] is probabilities
    sampling_provider.sampling_from_probs.assert_not_called()


def test_request_local_resolves_provider_callable_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    sampling_provider: ModuleType,
) -> None:
    operation = RequestLocalCategorical()
    provider = sampling_provider.sample_requests
    provider.return_value = torch.tensor([1])
    replacement = Mock(side_effect=AssertionError("provider must not be resolved again"))
    monkeypatch.setattr("ctikernels.sampling.inference.sample_requests", replacement)
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("blocked_flashinfer"))
    monkeypatch.delitem(sys.modules, "flashinfer.sampling")

    for position in (0, 1):
        result = operation.sample(torch.full((1, 3), 1 / 3), (17,), (position,))
        assert torch.equal(result, torch.tensor([1]))

    assert provider.call_count == 2
    replacement.assert_not_called()


@pytest.mark.parametrize("use_top_k", [False, True])
@pytest.mark.parametrize("use_top_p", [False, True])
def test_request_local_filters_before_drawing_with_unchanged_rng(
    sampling_provider: ModuleType, use_top_k: bool, use_top_p: bool
) -> None:
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    original = probabilities.clone()
    top_k = torch.tensor([2, 1], dtype=torch.int32) if use_top_k else None
    top_p = torch.tensor([0.4, 0.9]) if use_top_p else None
    seeds, positions = (17, 23), (0, 4)

    result = RequestLocalCategorical().sample(
        probabilities, seeds, positions, top_k=top_k, top_p=top_p
    )

    drawn, *rng = sampling_provider.sample_requests.call_args.args
    assert rng == [seeds, positions]
    if use_top_k or use_top_p:
        assert torch.equal(drawn, filter_requests(probabilities, top_k, top_p))
    else:
        assert drawn is probabilities
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)
    sampling_provider.top_k_renorm_probs.assert_not_called()
    sampling_provider.top_p_renorm_probs.assert_not_called()
    sampling_provider.sampling_from_probs.assert_not_called()
    assert result.tolist() == [2, 0]


def _renormalized(row: torch.Tensor, kept: list[int]) -> torch.Tensor:
    """K1's scale: the exact retained 2^-62 mass, one RNE rounding, FP32 div.rn."""
    scale = _uint64_to_float32(sum(fixed_mass(row[i]) for i in kept))
    out = torch.zeros_like(row)
    out[kept] = row[kept] / (scale * torch.tensor(2.0**-62))
    return out


def _filtered_row(row: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """Scalar K1 law (reference.probability_row's filters) on a probability row."""

    def descending() -> list[int]:
        return sorted(range(len(row)), key=lambda i: (-float(row[i]), i))

    if k < len(row):
        row = _renormalized(row, descending()[:k])
    if p != 1:
        threshold, preceding, kept = fixed_mass(p, ceiling=True), 0, []
        for token in descending():
            if preceding >= threshold:
                break
            kept.append(token)
            preceding += fixed_mass(row[token])
        row = _renormalized(row, kept)
    return row


def test_request_local_filters_match_scalar_k1_law_bitwise() -> None:
    generator = torch.Generator().manual_seed(2026)
    probabilities = (torch.randn(8, 300, generator=generator) * 3).softmax(-1)
    probabilities[1, 100:140] = probabilities[1, 7]  # ties across the top-k boundary
    probabilities[2] = 1 / 300  # every token tied
    probabilities[3, :4] = torch.tensor([0.5, 0.25, 0.25, 0.0])  # top-p lands exactly on mass
    probabilities[3, 4:] = 0
    probabilities[4, 50:] = 0
    probabilities[4] /= probabilities[4].sum()
    original = probabilities.clone()
    top_k = torch.tensor([300, 20, 3, 300, 1, 299, 64, 2], dtype=torch.int32)
    top_p = torch.tensor([0.9, 1.0, 0.5, 0.75, 1.0, 1e-6, 0.95, 0.999])

    for k, p in ((top_k, None), (None, top_p), (top_k, top_p)):
        actual = filter_requests(probabilities, k, p)
        for b in range(8):
            expected = _filtered_row(
                probabilities[b],
                300 if k is None else int(k[b]),
                1.0 if p is None else float(p[b]),
            )
            assert torch.equal(actual[b].view(torch.int32), expected.view(torch.int32)), b
        assert torch.equal(filter_requests(probabilities, k, p), actual)
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)


def test_request_local_filters_keep_exactly_k_smaller_ids_and_exact_mass() -> None:
    uniform = torch.full((1, 8), 1 / 8)
    kept = filter_requests(uniform, torch.tensor([3], dtype=torch.int32), None)
    assert torch.equal(kept, torch.tensor([[1 / 3] * 3 + [0.0] * 5]))
    # Exclusive mass 0.5 + 0.25 reaches p = 0.75 exactly, so the tied token 2 is dropped.
    row = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    kept = filter_requests(row, None, torch.tensor([0.75]))
    assert torch.equal(kept, torch.tensor([[2 / 3, 1 / 3, 0.0, 0.0]]))
    identity = filter_requests(
        torch.tensor([[0.1, 0.2, 0.3, 0.4000001]]),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([1.0]),
    )
    assert torch.equal(identity, torch.tensor([[0.1, 0.2, 0.3, 0.4000001]]))


@pytest.mark.parametrize(
    ("seeds", "positions"),
    [((17,), (0, 4)), ((17, 23), (0,)), ((), ())],
)
def test_request_local_rejects_mismatched_rng_batch_before_provider_call(
    sampling_provider: ModuleType,
    seeds: tuple[int, ...],
    positions: tuple[int, ...],
) -> None:
    operation = RequestLocalCategorical()

    with pytest.raises(ValueError, match="RNG state must match the probability batch"):
        operation.sample(torch.full((2, 3), 1 / 3), seeds, positions)

    sampling_provider.sampling_from_probs.assert_not_called()
    sampling_provider.sample_requests.assert_not_called()


@pytest.mark.parametrize("filtered", [False, True])
def test_request_local_zero_rows_is_empty_without_provider_call(
    sampling_provider: ModuleType, filtered: bool
) -> None:
    empty = torch.empty(0, dtype=torch.int32)
    result = RequestLocalCategorical().sample(
        torch.empty(0, 3),
        (),
        (),
        top_k=empty if filtered else None,
        top_p=torch.empty(0) if filtered else None,
    )

    assert result.shape == (0,) and result.dtype == torch.int32
    sampling_provider.top_k_renorm_probs.assert_not_called()
    sampling_provider.top_p_renorm_probs.assert_not_called()
    sampling_provider.sample_requests.assert_not_called()


@pytest.mark.parametrize("block", [2, 32, 33, 64, 65, 128])
def test_chain_large_q_reserves_whole_philox_blocks(sampling_provider, monkeypatch, block):
    # Exercise host shape checks/reservation without a CUDA launch.
    monkeypatch.setattr(ChainSampling, "_require_rng", staticmethod(lambda *args: None))
    expected = torch.zeros(1, block, dtype=torch.int32)
    sampling_provider.chain_speculative_sampling = Mock(return_value=(expected, None, None))
    op = ChainSampling()
    offset = torch.tensor([7])
    counts = torch.full((1,), 99, dtype=torch.int32)
    result = op.verify(
        torch.ones(1, block - 1, 1),
        torch.zeros(1, block - 1, dtype=torch.int32),
        torch.ones(1, block, 1),
        accepted_counts=counts,
        seed=torch.tensor([1]),
        offset=offset,
    )
    assert result is expected
    assert counts.tolist() == [0]
    assert offset.item() == 7 + 4 * ((block + 4) // 4)
    sampling_provider.chain_speculative_sampling.assert_called_once()


def test_chain_q1_bonus_only(sampling_provider, monkeypatch):
    monkeypatch.setattr(ChainSampling, "_require_rng", staticmethod(lambda *args: None))
    expected = torch.tensor([[2], [0]], dtype=torch.int32)
    sampling_provider.chain_speculative_sampling = Mock(return_value=(expected, None, None))
    counts = torch.full((2,), 99, dtype=torch.int32)
    offset = torch.tensor([7])
    result = ChainSampling().verify(
        torch.empty(2, 0, 3),
        torch.empty(2, 0, dtype=torch.int32),
        torch.full((2, 1, 3), 1 / 3),
        accepted_counts=counts,
        seed=torch.tensor([1]),
        offset=offset,
    )
    assert result is expected
    assert counts.tolist() == [0, 0]
    assert offset.item() == 11
    sampling_provider.chain_speculative_sampling.assert_called_once()


def test_chain_verify_zero_rows_draws_nothing(sampling_provider, monkeypatch):
    monkeypatch.setattr(ChainSampling, "_require_rng", staticmethod(lambda *args: None))
    sampling_provider.chain_speculative_sampling = Mock()
    offset = torch.tensor([7])
    result = ChainSampling().verify(
        torch.empty(0, 2, 3),
        torch.empty(0, 2, dtype=torch.int32),
        torch.empty(0, 3, 3),
        accepted_counts=torch.empty(0, dtype=torch.int32),
        seed=torch.tensor([1]),
        offset=offset,
    )
    assert result.shape == (0, 3) and result.dtype == torch.int32
    assert offset.item() == 7
    sampling_provider.chain_speculative_sampling.assert_not_called()


@pytest.mark.parametrize("call", ["softmax", "sample", "chain_probabilities", "chain_sample"])
def test_categorical_and_chain_zero_rows_skip_provider(sampling_provider, monkeypatch, call):
    monkeypatch.setattr(ChainSampling, "_require_rng", staticmethod(lambda *args: None))
    for name in ("softmax", "sampling_from_probs", "top_k_renorm_probs", "top_p_renorm_probs"):
        getattr(sampling_provider, name).side_effect = AssertionError("provider called")
    rows = torch.empty(0, 3)
    empty = torch.empty(0)
    offset = torch.tensor([7])
    if call == "softmax":
        result, expected = CategoricalSampling().softmax(rows, empty, enable_pdl=False), (0, 3)
    elif call == "sample":
        result, expected = CategoricalSampling().sample(rows), (0,)
    elif call == "chain_probabilities":
        k = torch.empty(0, dtype=torch.int32)
        result, expected = ChainSampling().probabilities(rows, empty, k, empty), (0, 3)
    else:
        sample = ChainSampling().sample
        result, expected = sample(rows, seed=torch.tensor([1]), offset=offset), (0,)
    assert result.shape == expected
    assert result.dtype == (torch.float32 if len(expected) == 2 else torch.int32)
    assert offset.item() == 7
