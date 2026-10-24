# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math

import pytest
import torch

from ctikernels.sampling import CategoricalSampling, RequestLocalCategorical
from ctikernels.sampling._request_local import filter_requests


@pytest.mark.parametrize(("batch_size", "vocab_size"), ((1, 257), (8, 128), (32, 4096)))
def test_temperature_softmax_matches_reference_with_supported_pdl(
    batch_size: int, vocab_size: int
) -> None:
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    temperatures = torch.linspace(0.5, 1.5, batch_size, device="cuda")
    original_logits, original_temperatures = logits.clone(), temperatures.clone()
    expected = torch.softmax(logits / temperatures[:, None], dim=-1)
    operation = CategoricalSampling()

    without_pdl = operation.softmax(logits, temperatures, enable_pdl=False)
    torch.testing.assert_close(without_pdl, expected)
    if torch.cuda.get_device_capability(logits.device) >= (9, 0):
        with_pdl = operation.softmax(logits, temperatures, enable_pdl=True)
        torch.testing.assert_close(with_pdl, without_pdl)
        torch.testing.assert_close(with_pdl, expected)
    torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
    torch.testing.assert_close(temperatures, original_temperatures, rtol=0, atol=0)


@pytest.mark.parametrize("filter_kind", ("none", "top_k", "top_p", "top_k_top_p"))
@pytest.mark.parametrize("seeded", [False, True])
def test_categorical_sampling_respects_single_token_support(filter_kind: str, seeded: bool) -> None:
    batch_size, vocab_size = 8, 257
    selected = (torch.arange(batch_size, device="cuda") * 31 + 7) % vocab_size
    probabilities = torch.zeros(batch_size, vocab_size, device="cuda")
    top_k, top_p = None, None
    if filter_kind == "none":
        probabilities.scatter_(1, selected[:, None], 1.0)
    else:
        # The unfiltered distribution has many possible outcomes; each filter
        # leaves only the unique maximum, so this is not a statistical assertion.
        probabilities.fill_(0.5 / (vocab_size - 1))
        probabilities.scatter_(1, selected[:, None], 0.5)
        if filter_kind in ("top_k", "top_k_top_p"):
            top_k = torch.ones(batch_size, device="cuda", dtype=torch.int32)
        if filter_kind in ("top_p", "top_k_top_p"):
            top_p = torch.full((batch_size,), 0.1, device="cuda")
    original = probabilities.clone()

    if seeded:
        sampled = RequestLocalCategorical().sample(
            probabilities, tuple(range(batch_size)), (3,) * batch_size, top_k=top_k, top_p=top_p
        )
    else:
        sampled = CategoricalSampling().sample(probabilities, top_k=top_k, top_p=top_p)

    assert sampled.shape == (batch_size,)
    torch.testing.assert_close(sampled.long(), selected, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)


def _check_frequencies(sampled: torch.Tensor, expected: torch.Tensor) -> None:
    counts = torch.bincount(sampled.long(), minlength=expected.numel()).cpu()
    assert counts.numel() == expected.numel()
    assert not bool(counts[expected == 0].any())
    frequencies = counts.double() / sampled.numel()
    # Hoeffding + union bound: false rejection probability <= 1e-9 per case.
    # This deliberately tests distribution, not provider-specific RNG bit patterns.
    tolerance = math.sqrt(math.log(2 * expected.numel() / 1e-9) / (2 * sampled.numel()))
    torch.testing.assert_close(frequencies, expected.double(), rtol=0, atol=tolerance)


@pytest.mark.parametrize("filter_kind", ("none", "top_k", "top_p", "top_k_top_p"))
@pytest.mark.parametrize("seeded", [False, True])
def test_categorical_sampling_matches_non_degenerate_distribution(
    filter_kind: str, seeded: bool
) -> None:
    torch.manual_seed(2026)
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda").repeat(8192, 1)
    original = probabilities.clone()
    top_k = (
        torch.full(
            (8192,), 3 if filter_kind == "top_k_top_p" else 2, device="cuda", dtype=torch.int32
        )
        if filter_kind in ("top_k", "top_k_top_p")
        else None
    )
    top_p = (
        torch.full((8192,), 0.6 if filter_kind == "top_p" else 0.75, device="cuda")
        if filter_kind in ("top_p", "top_k_top_p")
        else None
    )
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4] if filter_kind == "none" else [0, 0, 3 / 7, 4 / 7])
    # Combined filtering applies top-k first. Both filters are necessary: ignoring
    # either k=3 or p=.75 would incorrectly retain the token with probability .2.

    if seeded:
        sampled = RequestLocalCategorical().sample(
            probabilities, tuple(range(8192)), (7,) * 8192, top_k=top_k, top_p=top_p
        )
    else:
        sampled = CategoricalSampling().sample(probabilities, top_k=top_k, top_p=top_p)

    assert sampled.shape == (8192,)
    _check_frequencies(sampled, expected)
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)


def test_distribution_check_rejects_argmax_and_uniform_sampling() -> None:
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4])
    for incorrect in (torch.full((8192,), 3), torch.arange(8192) % 4):
        with pytest.raises(AssertionError):
            _check_frequencies(incorrect, expected)


def test_request_local_rng_uses_both_seed_and_position() -> None:
    batch_size = 32
    probabilities = torch.full((batch_size, 257), 1 / 257, device="cuda")
    seeds = tuple(range(batch_size))
    positions = (0,) * batch_size
    operation = RequestLocalCategorical()
    first = operation.sample(probabilities, seeds, positions)
    changed_seed = operation.sample(probabilities, tuple(seed + 1000 for seed in seeds), positions)
    changed_position = operation.sample(probabilities, seeds, (1,) * batch_size)

    torch.testing.assert_close(
        operation.sample(probabilities, seeds, positions), first, rtol=0, atol=0
    )
    # With 32 independent uniform streams, accidental whole-vector equality has
    # probability 257**-32; an ignored seed/offset would fail deterministically.
    assert not torch.equal(first, changed_seed)
    assert not torch.equal(first, changed_position)


@pytest.mark.parametrize("filter_kind", ("none", "top_k", "top_p", "top_k_top_p"))
@pytest.mark.parametrize("vocab_size", [257, 4096, 151936])
def test_request_local_rng_repeats_and_survives_permutation_and_partition(
    filter_kind: str, vocab_size: int
) -> None:
    torch.manual_seed(43)
    batch_size = 8
    probabilities = torch.randn(batch_size, vocab_size, device="cuda").softmax(dim=-1)
    original = probabilities.clone()
    seeds = tuple(range(1000, 1000 + batch_size))
    positions = tuple(index % 3 for index in range(batch_size))
    top_k = (
        torch.tensor([1, 2, 7, 31, 16, 4, 127, vocab_size], device="cuda", dtype=torch.int32)
        if filter_kind in ("top_k", "top_k_top_p")
        else None
    )
    top_p = (
        torch.tensor([0.01, 0.3, 0.7, 0.95, 0.5, 0.1, 0.99, 1.0], device="cuda")
        if filter_kind in ("top_p", "top_k_top_p")
        else None
    )
    original_k = top_k.clone() if top_k is not None else None
    original_p = top_p.clone() if top_p is not None else None
    operation = RequestLocalCategorical()
    expected = operation.sample(probabilities, seeds, positions, top_k=top_k, top_p=top_p)
    repeated = operation.sample(probabilities, seeds, positions, top_k=top_k, top_p=top_p)
    order = (7, 2, 0, 6, 5, 1, 3, 4)
    permutation = torch.tensor(order, device="cuda")
    permuted = operation.sample(
        probabilities[permutation],
        tuple(seeds[index] for index in order),
        tuple(positions[index] for index in order),
        top_k=top_k[permutation] if top_k is not None else None,
        top_p=top_p[permutation] if top_p is not None else None,
    )
    partitioned = torch.cat(
        tuple(
            operation.sample(
                probabilities[start:end],
                seeds[start:end],
                positions[start:end],
                top_k=top_k[start:end] if top_k is not None else None,
                top_p=top_p[start:end] if top_p is not None else None,
            )
            for start, end in ((0, 3), (3, batch_size))
        )
    )

    assert expected.shape == (batch_size,)
    assert bool(((0 <= expected) & (expected < vocab_size)).all())
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    torch.testing.assert_close(permuted[permutation.argsort()], expected, rtol=0, atol=0)
    torch.testing.assert_close(partitioned, expected, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)
    if top_k is not None:
        torch.testing.assert_close(top_k, original_k, rtol=0, atol=0)
    if top_p is not None:
        torch.testing.assert_close(top_p, original_p, rtol=0, atol=0)


def test_request_local_identity_filters_and_ties() -> None:
    probabilities = torch.full((8, 257), 1 / 257, device="cuda")
    seeds, positions = tuple(range(8)), (11,) * 8
    operation = RequestLocalCategorical()
    original = operation.sample(probabilities, seeds, positions)
    identity = operation.sample(
        probabilities,
        seeds,
        positions,
        top_k=torch.full((8,), 257, device="cuda", dtype=torch.int32),
        top_p=torch.ones(8, device="cuda"),
    )
    torch.testing.assert_close(identity, original, rtol=0, atol=0)
    # Top-k keeps exactly k tokens, ties to the smaller ID, as K1 and vLLM V2's exact-k.
    tied = operation.sample(
        probabilities,
        seeds,
        positions,
        top_k=torch.ones(8, device="cuda", dtype=torch.int32),
    )
    assert not bool(tied.any())


@pytest.mark.parametrize("batch_size", [1, 8])
def test_request_local_filters_are_bitwise_deterministic_and_exact(batch_size: int) -> None:
    vocab = 248320
    generator = torch.Generator(device="cuda").manual_seed(2027)
    logits = torch.randn(batch_size, vocab, device="cuda", generator=generator) * 4
    # 64 tied maxima: k < 64 and small p must keep the smaller IDs among them.
    logits[:, 1000:1064] = logits.amax(-1, keepdim=True) + 1
    probabilities = logits.softmax(-1)
    original = probabilities.clone()
    top_k = torch.tensor(
        [40, vocab, 1, 2000, 7, vocab, 64, 300][:batch_size], device="cuda", dtype=torch.int32
    )
    top_p = torch.tensor([0.9, 0.5, 1.0, 0.95, 0.3, 1.0, 0.99, 0.8][:batch_size], device="cuda")
    seeds = tuple(range(31, 31 + batch_size))
    positions = tuple(range(5, 5 + batch_size))
    operation = RequestLocalCategorical()

    filtered = filter_requests(probabilities, top_k, top_p)
    tokens = operation.sample(probabilities, seeds, positions, top_k=top_k, top_p=top_p)
    for _ in range(50):
        assert torch.equal(filter_requests(probabilities, top_k, top_p), filtered)
        repeated = operation.sample(probabilities, seeds, positions, top_k=top_k, top_p=top_p)
        assert torch.equal(repeated, tokens)
    # Sorts, integer sums and IEEE FP32 division: CUDA equals the CPU law bitwise.
    host = filter_requests(probabilities.cpu(), top_k.cpu(), top_p.cpu())
    assert torch.equal(filtered.cpu().view(torch.int32), host.view(torch.int32))
    torch.testing.assert_close(probabilities, original, rtol=0, atol=0)

    after_k = filter_requests(probabilities, top_k, None)
    for row in range(batch_size):
        k, p = int(top_k[row]), float(top_p[row])
        order = torch.sort(probabilities[row], descending=True, stable=True).indices
        expected = torch.zeros(vocab, dtype=torch.bool, device="cuda")
        expected[order[:k]] = True
        assert torch.equal(after_k[row] > 0, expected)  # exactly k, smaller-ID ties
        if k < 64:
            assert torch.equal(after_k[row].nonzero()[:, 0], torch.arange(1000, 1000 + k).cuda())
        values, order = torch.sort(after_k[row], descending=True, stable=True)
        masses = (values.double() * 2**62).long()
        preceding = masses.cumsum(0) - masses
        kept = (preceding < math.ceil(p * 2**62)) | (p == 1)
        assert torch.equal(filtered[row, order] > 0, kept & (values > 0))
        assert abs(float(filtered[row].double().sum()) - 1) < 1e-5
        assert bool(filtered[row, int(tokens[row])] > 0)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32, 64])
@pytest.mark.parametrize("vocab_size", [1, 2, 3, 128, 257, 4096, 151936])
def test_request_local_batch_matches_scalar_provider_exactly(batch_size, vocab_size) -> None:
    from flashinfer.sampling import sampling_from_probs

    torch.manual_seed(47)
    probabilities = torch.randn(batch_size, vocab_size, device="cuda").softmax(-1)
    seeds = tuple((0, 2**32 + 13, 2**62 + 7, 2**63 - 1)[i % 4] for i in range(batch_size))
    positions = tuple((0, 1, 3, 4, 127, 2**32 + 17)[i % 6] for i in range(batch_size))
    expected = torch.cat(
        [
            sampling_from_probs(probabilities[i : i + 1], seed=seeds[i], offset=positions[i])
            for i in range(batch_size)
        ]
    )
    actual = RequestLocalCategorical().sample(probabilities, seeds, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_request_local_overlapping_submissions_own_rng_staging_storage() -> None:
    from flashinfer.sampling import sampling_from_probs

    batch_size = 8
    probabilities = torch.full((batch_size, 4096), 1 / 4096, device="cuda")
    operation = RequestLocalCategorical()
    operation.sample(probabilities, tuple(range(batch_size)), (0,) * batch_size)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for position in range(32):
        seeds = tuple(position * 71 + i for i in range(batch_size))
        with torch.cuda.stream(streams[position % 2]):
            result = operation.sample(probabilities, seeds, (position,) * batch_size)
        results.append((result, seeds, position))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for result, seeds, position in results:
        expected = torch.cat(
            [
                sampling_from_probs(probabilities[i : i + 1], seed=seed, offset=position)
                for i, seed in enumerate(seeds)
            ]
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_request_local_device_draw_replays_with_current_rng_state() -> None:
    from ctikernels.sampling._request_local import _sampling_module

    probabilities = torch.full((8, 257), 1 / 257, device="cuda")
    rng = torch.tensor([list(range(8)), [0] * 8], device="cuda", dtype=torch.int64)
    output = torch.empty(8, device="cuda", dtype=torch.int32)
    module = _sampling_module()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.launch(probabilities, rng, output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        module.launch(probabilities, rng, output)
    for position in (0, 1, 2**32 + 17):
        rng[1].fill_(position)
        graph.replay()
        expected = RequestLocalCategorical().sample(probabilities, tuple(range(8)), (position,) * 8)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad_tensor", ["dtype", "stride", "rng_shape", "output_shape"])
def test_request_local_device_draw_rejects_invalid_tensor_metadata(bad_tensor) -> None:
    from ctikernels.sampling._request_local import _sampling_module

    probabilities = torch.full((8, 257), 1 / 257, device="cuda")
    rng = torch.zeros((2, 8), device="cuda", dtype=torch.int64)
    output = torch.empty(8, device="cuda", dtype=torch.int32)
    if bad_tensor == "dtype":
        probabilities = probabilities.half()
    elif bad_tensor == "stride":
        probabilities = probabilities.T.contiguous().T
    elif bad_tensor == "rng_shape":
        rng = rng[:, :4].contiguous()
    else:
        output = output[:4]
    with pytest.raises(RuntimeError, match="Tensor match failed"):
        _sampling_module().launch(probabilities, rng, output)


@pytest.mark.parametrize("filtered", [False, True])
def test_request_local_zero_rows_is_empty_and_capturable(filtered: bool) -> None:
    probabilities = torch.empty(0, 257, device="cuda")
    top_k = torch.empty(0, device="cuda", dtype=torch.int32) if filtered else None
    top_p = torch.empty(0, device="cuda") if filtered else None
    operation = RequestLocalCategorical()

    result = operation.sample(probabilities, (), (), top_k=top_k, top_p=top_p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = operation.sample(probabilities, (), (), top_k=top_k, top_p=top_p)
    graph.replay()

    for tensor in (result, captured):
        assert tensor.shape == (0,) and tensor.dtype == torch.int32 and tensor.is_cuda


@pytest.mark.parametrize("filter_kind", ("none", "top_k", "top_p", "top_k_top_p"))
def test_categorical_zero_rows_match_nonempty_dtypes(filter_kind: str) -> None:
    operation = CategoricalSampling()
    for rows in (1, 0):
        logits = torch.zeros(rows, 257, device="cuda")
        temperatures = torch.ones(rows, device="cuda")
        top_k = (
            torch.ones(rows, device="cuda", dtype=torch.int32) if "top_k" in filter_kind else None
        )
        top_p = torch.ones(rows, device="cuda") if filter_kind.endswith("top_p") else None
        probabilities = operation.softmax(logits, temperatures, enable_pdl=False)
        sampled = operation.sample(probabilities, top_k=top_k, top_p=top_p)
        assert probabilities.shape == (rows, 257) and sampled.shape == (rows,)
        if rows:
            dtypes = probabilities.dtype, sampled.dtype
    assert (probabilities.dtype, sampled.dtype) == dtypes
