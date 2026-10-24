# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Deterministic probability construction: frozen outputs and independent FP64 oracle."""

import pytest
import torch

from ctikernels.sampling._dflash import RequestLocalChainSampling, validate_filters
from ctikernels.sampling._dflash.reference import fixed_mass, probability_row

from .dflash_oracle import probabilities_fp64


def fp32(value):
    return torch.as_tensor(value, dtype=torch.float32)


# (logits, temperature, k, p, slot, literal FP32 output). A large temperature
# keeps distinct z keys for [3,2,...] while exp(z-max) rounds to one. These exact
# goldens therefore do not depend on CPU/device transcendental approximation.
INF = float("-inf")
GOLDEN_PROBABILITIES = [
    ([3, 2, 2, 2, 1, INF], 33554432, 2, 1, 0, [0.5, 0.5, 0, 0, 0, 0]),
    ([2, 3, 2, 2, 2, 1], 33554432, 4, 1, 0, [0.25, 0.25, 0.25, 0.25, 0, 0]),
    ([0, 0, 0, 0, INF, INF], 1, 6, 0.5, 0, [0.5, 0.5, 0, 0, 0, 0]),
    (
        [0, 0, 0, 0, INF, INF],
        1,
        6,
        0.75,
        0,
        [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0, 0, 0],
    ),
    ([0, 0, 0, 0, 0, 0], 1, 4, 0.5, 0, [0.5, 0.5, 0, 0, 0, 0]),
    ([0, 0, 0, 0, INF, INF], 1, 6, 1, 0, [0.25, 0.25, 0.25, 0.25, 0, 0]),
    ([1, 3, 3, 0, 3, INF], 0, 1, 0.01, 0, [0, 1, 0, 0, 0, 0]),
    ([float("nan")] * 6, float("nan"), -999, float("nan"), -1, [0, 0, 0, 0, 0, 0]),
    ([INF, 2, INF, 2, INF, INF], 1, 6, 1, 0, [0, 0.5, 0, 0.5, 0, 0]),
    ([1, 2, 2, 1, INF, 0], 1, 1, 1, 0, [0, 1, 0, 0, 0, 0]),
    ([-0.0, 0.0, -0.0, 0, INF, INF], 1, 2, 1, 0, [0.5, 0.5, 0, 0, 0, 0]),
    ([INF, 2, INF, INF, INF, INF], 1, 4, 0.5, 0, [0, 1, 0, 0, 0, 0]),
]


def probability_data(logits, temperatures=None, top_k=None, top_p=None, slots=None):
    b, length, v = logits.shape
    return {
        "logits": logits,
        "temperatures": torch.ones(b) if temperatures is None else temperatures,
        "top_k": torch.full((b,), v, dtype=torch.int32) if top_k is None else top_k,
        "top_p": torch.ones(b) if top_p is None else top_p,
        "request_indices": torch.arange(b, dtype=torch.int32) if slots is None else slots,
        "workspace": torch.empty(
            RequestLocalChainSampling("cpu", max(2, length), v).workspace_size(
                b, "probabilities", length=length
            ),
            dtype=torch.uint8,
        ),
        "out_probs": torch.empty_like(logits, dtype=torch.float32),
    }


def golden_probability_data(dtype=torch.float32):
    rows, temps, ks, ps, slots, expected = zip(*GOLDEN_PROBABILITIES, strict=True)
    data = probability_data(
        torch.tensor(rows, dtype=dtype).unsqueeze(1),
        torch.tensor(temps, dtype=torch.float32),
        torch.tensor(ks, dtype=torch.int32),
        torch.tensor(ps, dtype=torch.float32),
        torch.tensor(slots, dtype=torch.int32),
    )
    return data, torch.tensor(expected, dtype=torch.float32).unsqueeze(1)


def oracle(data):
    return probabilities_fp64(
        data["logits"],
        data["temperatures"],
        data["top_k"],
        data["top_p"],
        request_indices=data["request_indices"],
    )


def margin_probability_data(vocab, dtype=torch.float32):
    """Random finite tails and 16 prominent tokens; margins checked after dtype cast.

    Large-vocabulary flat softmax cannot have a >1e-4 CDF margin, so use a
    concentrated distribution while still exercising every token and radix pass.
    """
    gen = torch.Generator().manual_seed(919)
    logits = -22 + torch.rand(4, 1, vocab, generator=gen)
    # Same small leading support in all modes, with random token locations.
    ids = torch.randperm(vocab, generator=gen)[:16]
    head = torch.linspace(2, -2, 16) + torch.rand(16, generator=gen) * 0.05
    logits[:, 0, ids] = head
    data = probability_data(logits.to(dtype), torch.tensor([0.7, 1.2, 1.0, 0.9]))
    data["top_k"][[1, 3]] = 8
    for b in range(4):
        z = data["logits"][b, 0].double() / data["temperatures"][b].double()
        order = torch.argsort(z, descending=True, stable=True)
        k = int(data["top_k"][b])
        if k < vocab:
            assert z[order[k - 1]] - z[order[k]] > 1e-4
            z[order[k:]] = -torch.inf
        probs = torch.softmax(z, -1)
        if b >= 2:
            cumulative = probs.sort(descending=True, stable=True).values.cumsum(0)
            data["top_p"][b] = (cumulative[3] + cumulative[4]) / 2
            assert (cumulative - data["top_p"][b].double()).abs().min() > 1e-4
    return data


def assert_probability_tolerance(actual, expected):
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
    assert ((actual.double().sum(-1) - 1).abs() <= 2e-6).all()
    assert torch.equal(actual > 0, expected > 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_probability_goldens(dtype):
    data, expected = golden_probability_data(dtype)
    op = RequestLocalChainSampling("cpu", 2, 6)
    assert op.probabilities(**data) is None
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))
    assert op.launch_counts["probabilities"] == 11


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [32, 4097, 50000, 248320])
def test_probability_fp64_tolerances(vocab, dtype):
    data = margin_probability_data(vocab, dtype)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data)
    assert_probability_tolerance(data["out_probs"], oracle(data))


def test_probability_identity_is_unrenormalized_softmax():
    # Literal reference-FP32 values (correctly rounded exp) expose an otherwise
    # tempting extra normalization.
    result = probability_row([3, 2, 2, 2, 1, -9], 1, 6, 1)
    expected = torch.tensor(
        [
            0.4466320276260376,
            0.16430674493312836,
            0.16430674493312836,
            0.16430674493312836,
            0.06044507026672363,
            0.0000027442019927548245,
        ],
        dtype=torch.float32,
    )
    assert torch.equal(result, expected)


def test_fixed_mass_exact_bits_and_tiny_thresholds():
    for value in [0, 2**-149, 2**-100, 2**-63, 2**-62, 0.1, 0.5, 1]:
        numerator, denominator = float(fp32(value)).as_integer_ratio()
        scaled = numerator * 2**62
        assert fixed_mass(value) == scaled // denominator
        assert fixed_mass(value, ceiling=True) == (scaled + denominator - 1) // denominator
    assert torch.equal(
        probability_row([0, 0, 0, 0], 1, 4, fp32(2**-149)),
        torch.tensor([1, 0, 0, 0], dtype=torch.float32),
    )


@pytest.mark.parametrize("length", [1, 2, 7, 8])
@pytest.mark.parametrize("batch", [0, 2])
def test_probability_lengths_and_empty_batch(length, batch):
    data = probability_data(torch.zeros(batch, length, 4))
    op = RequestLocalChainSampling("cpu", 8, 4)
    op.probabilities(**data)
    assert (data["out_probs"] == 0.25).all()


@pytest.mark.parametrize("name", ["logits", "temperatures", "top_p", "out_probs"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_probability_requires_grad(name, grad_enabled):
    data = probability_data(torch.zeros(1, 1, 4))
    data[name].requires_grad_(True)
    op = RequestLocalChainSampling("cpu", 2, 4)
    op._provider.run = lambda *a, **kw: pytest.fail("provider reached with grad tensor")
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="requires grad"):
        op.probabilities(**data)


@pytest.mark.parametrize(
    "problem",
    [
        "dtype",
        "k_dtype",
        "p_dtype",
        "length",
        "vocab",
        "batch",
        "layout",
        "alias",
        "overlap",
        "device",
        "workspace",
    ],
)
def test_probability_validation(problem):
    data = probability_data(torch.zeros(2, 2, 4))
    op = RequestLocalChainSampling("cpu", 2, 4)
    op._provider.run = lambda *a, **kw: pytest.fail("provider reached with invalid tensor")
    if problem == "dtype":
        data["logits"] = data["logits"].double()
    elif problem == "k_dtype":
        data["top_k"] = data["top_k"].long()
    elif problem == "p_dtype":
        data["top_p"] = data["top_p"].half()
    elif problem == "length":
        data["logits"] = torch.zeros(2, 3, 4)
    elif problem == "vocab":
        data["logits"] = torch.zeros(2, 2, 3)
    elif problem == "batch":
        data["request_indices"] = data["request_indices"][:1]
    elif problem == "layout":
        data["logits"] = data["logits"].transpose(0, 1)
    elif problem == "alias":
        data["out_probs"] = data["logits"]
    elif problem == "overlap":
        backing = torch.empty(17)
        data["logits"], data["out_probs"] = backing[:16].view(2, 2, 4), backing[1:].view(2, 2, 4)
    elif problem == "device":
        data["logits"] = data["logits"].to("meta")
    elif problem == "workspace":
        data["workspace"] = torch.empty(0, dtype=torch.int32)
    with pytest.raises((TypeError, ValueError)):
        op.probabilities(**data)


@pytest.mark.parametrize("k", [True, 1.5])
def test_filter_invalid_k(k):
    with pytest.raises(TypeError):
        validate_filters(k, 1, 32)


@pytest.mark.parametrize("vocab", [0, -1, True, 1.5])
def test_filter_invalid_vocab(vocab):
    with pytest.raises(ValueError):
        validate_filters(None, None, vocab)


def test_probability_top_k_ordinary_temperature():
    expected = torch.tensor([0.7310585975646973, 0.2689414322376251, 0, 0, 0], dtype=torch.float32)
    assert torch.equal(probability_row([3, 2, 2, 2, 1], 1, 2, 1), expected)


def test_probability_total_fixed_mass_below_threshold():
    logits = torch.arange(6, dtype=torch.float32) / fp32(6)
    threshold = torch.nextafter(fp32(1), fp32(0))
    plain = probability_row(logits, 1, 6, 1)
    assert sum(fixed_mass(value) for value in plain) < fixed_mass(threshold, ceiling=True)
    expected = torch.tensor(
        [
            0.10554753988981247,
            0.12468968331813812,
            0.14730344712734222,
            0.1740184724330902,
            0.2055785357952118,
            0.24286234378814697,
        ],
        dtype=torch.float32,
    )
    result = probability_row(logits, 1, 6, threshold)
    assert torch.equal(result, expected)
    assert (result > 0).all()
    assert not torch.equal(result, plain)  # nonidentity top-p still renormalizes


def test_probability_split_ties_across_cta_tiles():
    data = probability_data(torch.zeros(1, 2, 4097))
    data["top_k"].fill_(1024)
    data["top_p"].fill_(0.75)
    RequestLocalChainSampling("cpu", 2, 4097).probabilities(**data)
    expected = torch.zeros_like(data["out_probs"])
    expected[:, :, :768] = 0.0013020833721384406
    assert torch.equal(data["out_probs"], expected)


def test_fixed_mass_random_fp32_encodings():
    rng = torch.Generator().manual_seed(53)
    for bits in torch.randint(0, 0x3F800001, (1024,), dtype=torch.int32, generator=rng):
        value = bits.view(torch.float32)
        numerator, denominator = float(value).as_integer_ratio()
        scaled = numerator << 62
        assert fixed_mass(value) == scaled // denominator
        assert fixed_mass(value, ceiling=True) == (scaled + denominator - 1) // denominator


@pytest.mark.parametrize("batch", [0, 1, 2, 8])
@pytest.mark.parametrize("length", [1, 7, 16])
@pytest.mark.parametrize("vocab", [1, 4096, 4097, 151936, 248320])
def test_probability_workspace_layout(batch, length, vocab):
    op = RequestLocalChainSampling("cpu", 16, vocab)
    chunks = (vocab + 4095) // 4096
    layout = op.workspace_layout(batch, "probabilities", length=length)
    assert layout["base_alignment"] == 8
    assert layout["total_bytes"] == (
        ((batch * length * (61440 + chunks * 20) + 255) // 256) * 256
        + ((batch * length * 8196 + 255) // 256) * 256
        + ((batch * length * 288 + 255) // 256) * 256
        + 2 * ((batch * length * chunks * 4 + 255) // 256) * 256
    )
    assert [r["name"] for r in layout["regions"]] == (
        [
            "top_k_histograms",
            "top_p_histograms",
            "probability_chunk_records",
            "compact_lists",
            "row_states",
            "min_p_chunk_maxima",
            "min_p_chunk_partials",
        ]
        if batch
        else []
    )
    offset = 0
    for region, payload in zip(
        layout["regions"],
        [
            batch * length * 20480,
            batch * length * 40960,
            batch * length * chunks * 20,
            batch * length * 8196,
            batch * length * 288,
            batch * length * chunks * 4,
            batch * length * chunks * 4,
        ],
        strict=bool(batch),
    ):
        assert region["offset"] == offset
        assert region["payload_bytes"] == payload
        assert region["offset"] % region["alignment"] == 0
        assert region["aligned_extent"] == ((payload + 255) // 256) * 256
        offset += region["aligned_extent"]
    assert offset == layout["total_bytes"]
    assert op.workspace_size(batch, "probabilities") == op.workspace_size(
        batch, "probabilities", length=16
    )


@pytest.mark.parametrize("length", [0, 17, True, 1.5])
def test_probability_invalid_workspace_length(length):
    with pytest.raises(ValueError, match="length"):
        RequestLocalChainSampling("cpu", 16, 4).workspace_layout(1, "probabilities", length=length)


@pytest.mark.parametrize("problem", ["small", "unaligned", "hint"])
def test_probability_workspace_and_hint_admission(problem):
    data = probability_data(torch.zeros(1, 1, 4))
    op = RequestLocalChainSampling("cpu", 2, 4)
    op._provider.run = lambda *a, **kw: pytest.fail("invalid call reached provider")
    kwargs = {}
    if problem == "small":
        data["workspace"] = data["workspace"][:-1]
    elif problem == "unaligned":
        # Four-byte alignment used to be sufficient; uint64 mass atomics need eight.
        data["workspace"] = torch.empty(data["workspace"].numel() + 4, dtype=torch.uint8)[4:]
    else:
        kwargs["filters"] = "invalid"
    with pytest.raises(ValueError):
        op.probabilities(**data, **kwargs)


def chunk_tie_probability_data(vocab):
    """Eight exactly uniform exponential weights, at IDs spanning chunk boundaries."""
    ids = torch.linspace(0, vocab - 1, 8).long()
    logits = torch.full((4, 2, vocab), -torch.inf)
    logits[:, :, ids] = 2
    logits[:, :, ids[-1]] = 3
    logits[3, :, ids[1]] = 3  # greedy max ties in distant chunks
    data = probability_data(
        logits,
        torch.tensor([33554432, 33554432, 33554432, 0], dtype=torch.float32),
        torch.tensor([4, vocab, 4, 1], dtype=torch.int32),
        torch.tensor([1, 0.75, 0.5, 0.1]),
    )
    expected = torch.zeros_like(data["out_probs"])
    expected[0, :, ids[:3]] = 0.25
    expected[0, :, ids[-1]] = 0.25
    expected[1, :, ids[:6]] = 0.1666666716337204
    expected[2, :, ids[:2]] = 0.5
    expected[3, :, ids[1]] = 1
    return data, expected


@pytest.mark.parametrize("vocab", [4097, 50000, 248320])
def test_probability_ties_in_multiple_chunks(vocab):
    data, expected = chunk_tie_probability_data(vocab)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data)
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))


def test_probability_chunk_order_and_top_k_tie_term():
    from ctikernels.sampling._dflash.reference import probability_sum
    from ctikernels.sampling._dflash.reference.ops import _exp

    # A chunk-order accumulation loses each later unit individually. A tree over
    # chunk totals would retain them; this freezes sequential chunk combination.
    values = torch.zeros(4096 * 3, dtype=torch.float32)
    values[[0, 4096, 8192]] = torch.tensor([2**24, 1, 1], dtype=torch.float32)
    assert probability_sum(values) == fp32(2**24)

    # Nonuniform weights expose rounding of one aggregated tie term, rather than
    # summing retained ties in their original chunk/lane positions.
    logits = torch.full((9000,), -1.125)
    logits[[0, 4096, 8999]] = torch.tensor([0, -0.25, -0.5])
    k = 5000
    weights = _exp(logits)
    above = logits > fp32(-1.125)
    total = fp32(
        probability_sum(torch.where(above, weights, fp32(0))) + fp32(fp32(k - 3) * weights[1])
    )
    expected = torch.zeros_like(weights)
    kept = torch.argsort(logits, descending=True, stable=True)[:k]
    expected[kept] = fp32(weights[kept] / total)
    actual = probability_row(logits, 1, k, 1)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    naive = torch.zeros_like(weights)
    naive[kept] = weights[kept]
    assert probability_sum(naive) != total


@pytest.mark.parametrize("hint", ["none", "top_k", "top_p", "both", "auto"])
def test_probability_filters_hint(hint):
    data = margin_probability_data(4097)
    if hint in ("none", "top_p"):
        data["top_k"].fill_(4097)
    if hint in ("none", "top_k"):
        data["top_p"].fill_(1)
    op = RequestLocalChainSampling("cpu", 2, 4097)
    op.probabilities(**data)
    expected = data["out_probs"].clone()
    op.probabilities(**data, filters=hint)
    assert torch.equal(data["out_probs"], expected)


def compact_boundary_probability_data(dtype=torch.float32):
    """Frozen arithmetic expectations: CAP, CAP+1, split ties across three chunks.

    Scaling keeps distinct z but rounds every retained exponential to one, so
    the expected bits do not depend on host/device exp approximations.
    """
    vocab = 9001
    ids = torch.arange(0, vocab, 7)  # 1286 finite weights, spanning three chunks
    logits = torch.full((7, 2, vocab), -torch.inf, dtype=dtype)
    logits[:, :, ids] = 2
    logits[:, :, ids[-3:]] = 3
    data = probability_data(
        logits,
        torch.tensor([33554432] * 5 + [0, float("nan")], dtype=torch.float32),
        torch.tensor([20, 1024, 1025, vocab, 1024, 1, -999], dtype=torch.int32),
        torch.tensor([0.95, 0.75, 0.75, 0.5, 1, 0.1, float("nan")]),
        torch.tensor([0, 1, 2, 3, 4, 5, -1], dtype=torch.int32),
    )
    logits[-1].fill_(float("nan"))
    expected = torch.zeros_like(data["out_probs"])
    # For k=20, FP32(.05)*19 exceeds FP32(.95): 19 retained tokens.
    # Three highest z IDs come last in the p ordering (all p bits are equal).
    expected[0, :, ids[:17]] = 0.05263157933950424
    expected[0, :, ids[-3:-1]] = 0.05263157933950424
    expected[1, :, ids[:768]] = 0.0013020833721384406
    expected[2, :, ids[:769]] = 0.0013003901112824678
    expected[3, :, ids[:643]] = 0.0015552099794149399
    expected[4, :, ids[:1021]] = 0.0009765625
    expected[4, :, ids[-3:]] = 0.0009765625
    expected[5, :, ids[-3]] = 1
    return data, expected


def test_probability_compact_boundary_frozen_bits():
    data, expected = compact_boundary_probability_data()
    RequestLocalChainSampling("cpu", 2, 9001).probabilities(**data)
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("exponent", range(54, 64))
@pytest.mark.parametrize("odd", [False, True])
def test_integer_to_fp32_avoids_double_rounding(exponent, odd):
    from ctikernels.sampling._dflash.reference.ops import _uint64_to_float32

    significand = 2**23 + int(odd)
    lower = significand << (exponent - 23)
    midpoint = lower + (1 << (exponent - 24))
    upper = lower + (1 << (exponent - 23))
    for delta, expected in [(-1, lower), (0, upper if odd else lower), (1, upper)]:
        result = _uint64_to_float32(midpoint + delta)
        assert result.dtype == torch.float32
        assert int(result) == expected


# Rows whose l/T overflows FP32 at tiny T: (l - max)/T is 0 for the maxima and far
# below exp's range otherwise, so the expected outputs are exact 0, 0.5 or 1.
OVERFLOW_TEMPERATURES = (2.0**-149, 2.0**-140, 2.0**-126)
OVERFLOW_ROWS = (
    ([1, 1, 0, 0, 0, 0], (0, 1)),
    ([0, 1, 0, 0, 0, 0], (1,)),
    ([3e38, -3e38, 3e38, 1e38, -1, 0], (0, 2)),
    ([-3e38, -3e38, -1e38, -3e38, -3e38, -3e38], (2,)),
)


def overflow_probability_data(filters, vocab):
    rows = [(row, top) for row, top in OVERFLOW_ROWS for _ in OVERFLOW_TEMPERATURES]
    logits = torch.full((len(rows), 1, vocab), -3e38)
    expected = torch.zeros(len(rows), 1, vocab)
    nucleus = filters in ("top_p", "both")
    for b, (row, top) in enumerate(rows):
        logits[b, 0, :6] = torch.tensor(row)
        # top-k=2 keeps both maxima; top-p=0.5 stops after the smaller-ID maximum.
        kept = top[:1] if nucleus else top
        expected[b, 0, list(kept)] = 1 / len(kept)
    b = len(rows)
    data = probability_data(
        logits,
        torch.tensor(OVERFLOW_TEMPERATURES * len(OVERFLOW_ROWS)),
        torch.full((b,), 2 if filters in ("top_k", "both") else vocab, dtype=torch.int32),
        torch.full((b,), 0.5 if nucleus else 1.0),
    )
    return data, expected


@pytest.mark.parametrize("filters", ["none", "top_k", "top_p", "both"])
@pytest.mark.parametrize("vocab", [6, 4100])
def test_probability_overflowing_scaled_logits(filters, vocab):
    data, expected = overflow_probability_data(filters, vocab)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=filters)
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))


# Logit spreads above FLT_MAX overflow (l - max) to -inf, so those elements fall
# back to scaling first; nearby large logits must subtract first, since scaling
# first rounds each one separately. Checked against the FP64 oracle, not the
# reference, together with the tiny-T rows and random extreme mixes.
SPREAD_TEMPERATURES = (2.0**127, 2.0**64, 1.0, 2.0**120, 2.0**120 * (1 - 2.0**-24))
NEARBY_TEMPERATURES = (3.0, 1.0, 0.5)
EXTREME_MAGNITUDES = (3e38, 2.0**127, 1e30, 2.0**64, 1.0, 2.0**-100, 0.0)
EXTREME_MIX_TEMPERATURES = (2.0**-149, 2.0**-126, 0.5, 1.0, 2.0, 2.0**64, 2.0**127, 3e38)
FILTER_HINTS = ("none", "top_k", "top_p", "both", "auto")
MASK_TEMPERATURES = (2.0**-149, 2.0**-126, 1.0, 2.0**127)
INF = float("inf")
MASK_ROWS = (
    (-1.0, -INF),
    (-INF, -1.0, -INF),
    (2.0**127, -INF, -(2.0**127)),
    (-INF, 3e38, -3e38, -INF, 0.0, 1e30),
    (1e7, -INF, 9999999.0),
    (1.0, 1.0, -INF),
)


def extreme_probability_data(filters, vocab, seed=127):
    """Rows of [2^127, -2^127], tiny-T rows, masked rows and random extreme mixes."""
    rows, temps = [], []
    for t in SPREAD_TEMPERATURES:
        rows.append([2.0**127, -(2.0**127)])
        temps.append(t)
    for t in NEARBY_TEMPERATURES:
        rows.append([1e7, 9999999.0])
        temps.append(t)
    for t in (2.0**-149, 2.0**-140, 2.0**-126):
        rows += [[1.0, 1.0], [0.0, 1.0]]
        temps += [t, t]
    gen = torch.Generator().manual_seed(seed)
    magnitudes = torch.tensor(EXTREME_MAGNITUDES)
    for t in EXTREME_MIX_TEMPERATURES:
        for _ in range(2):
            picks = torch.randint(len(EXTREME_MAGNITUDES), (6,), generator=gen)
            signs = torch.randint(2, (6,), generator=gen) * 2 - 1
            rows.append((magnitudes[picks] * signs).tolist())
            temps.append(t)
    # Masked logits (-inf) are not overflows: they keep probability 0 under every T.
    for t in MASK_TEMPERATURES:
        for row in MASK_ROWS:
            rows.append(list(row))
            temps.append(t)
    for t in EXTREME_MIX_TEMPERATURES:
        picks = torch.randint(len(EXTREME_MAGNITUDES), (6,), generator=gen)
        signs = torch.randint(2, (6,), generator=gen) * 2 - 1
        row = magnitudes[picks] * signs
        masked = torch.randint(2, (6,), generator=gen).bool()
        masked[0] = False
        rows.append(row.masked_fill(masked, -torch.inf).tolist())
        temps.append(t)
    logits = torch.full((len(rows), 1, vocab), -(2.0**127))
    for b, row in enumerate(rows):
        logits[b, 0, : min(len(row), vocab)] = torch.tensor(row[:vocab])
    b = len(rows)
    truncating_k = filters in ("top_k", "both", "auto")
    truncating_p = filters in ("top_p", "both", "auto")
    return probability_data(
        logits,
        torch.tensor(temps),
        torch.full((b,), 2 if truncating_k else vocab, dtype=torch.int32),
        torch.full((b,), 0.9 if truncating_p else 1.0),
    )


def assert_masked_probabilities(data, actual):
    """Masked logits get exactly 0; [-1, -inf] alone is exactly [1, 0] under every T."""
    logits = data["logits"]
    assert (actual[torch.isneginf(logits)] == 0).all()
    if logits.shape[-1] == 2:
        row = logits[:, 0]
        masks = (row[:, 0] == -1) & torch.isneginf(row[:, 1])
        assert masks.sum() == len(MASK_TEMPERATURES)
        assert torch.equal(actual[masks, 0], torch.tensor([[1.0, 0.0]] * len(MASK_TEMPERATURES)))


def assert_extreme_probabilities(actual, expected):
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
    assert ((actual.double().sum(-1) - 1).abs() <= 2e-6).all()


@pytest.mark.parametrize("filters", FILTER_HINTS)
@pytest.mark.parametrize("vocab", [2, 6, 4100])
def test_probability_extreme_spreads_match_fp64_oracle(filters, vocab):
    data = extreme_probability_data(filters, vocab)
    expected = oracle(data)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=filters)
    assert_extreme_probabilities(data["out_probs"], expected)
    assert_masked_probabilities(data, data["out_probs"])
    if vocab > 2:
        return
    # The review's example: [2^127, -2^127] at T=2^127 is [0.8808, 0.1192].
    torch.testing.assert_close(
        data["out_probs"][0, 0, :2].double(),
        torch.tensor([0.8807970779778823, 0.11920292202211755], dtype=torch.float64),
        atol=2e-7,
        rtol=0,
    )
    # [1e7, 9999999] at T=3 is [0.5826, 0.4174] (scale-first gave [0.5622, 0.4378]).
    nearby = len(SPREAD_TEMPERATURES)
    torch.testing.assert_close(
        data["out_probs"][nearby, 0, :2].double(),
        torch.tensor([0.5825702064623147, 0.41742979353768533], dtype=torch.float64),
        atol=2e-7,
        rtol=0,
    )


def nearby_top_p_boundary_data(vocab):
    """top_p=0.57 keeps only 1e7 (FP64 0.5826); scale-first rounding (0.5622) kept both."""
    logits = torch.full((1, 1, vocab), -(2.0**127))
    logits[0, 0, :2] = torch.tensor([1e7, 9999999.0])
    return probability_data(logits, torch.tensor([3.0]), top_p=torch.tensor([0.57]))


@pytest.mark.parametrize("filters", FILTER_HINTS)
@pytest.mark.parametrize("vocab", [2, 6, 4100])
def test_probability_nearby_top_p_boundary_matches_fp64_oracle(filters, vocab):
    data = nearby_top_p_boundary_data(vocab)
    if filters in ("none", "top_k"):
        data["top_p"].fill_(1)
    expected = oracle(data)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=filters)
    assert_extreme_probabilities(data["out_probs"], expected)
    assert (data["out_probs"][0, 0, 1] == 0) == (filters not in ("none", "top_k"))


def long_probability_data(block, filters):
    data = probability_data(torch.zeros(2, block, 4))
    if filters == "both":
        data["top_k"].fill_(2)
        data["top_p"].fill_(0.5)
    data["request_indices"][1] = -1
    data["logits"][1].fill_(float("nan"))
    return data


@pytest.mark.parametrize("block", [33, 64, 65, 128])
@pytest.mark.parametrize("filters", ["none", "both"])
def test_long_probability_reference(block, filters):
    data = long_probability_data(block, filters)
    RequestLocalChainSampling("cpu", block, 4).probabilities(**data, filters=filters)
    expected = torch.zeros_like(data["out_probs"])
    expected[0, :, 0] = 1
    if filters == "none":
        expected[0].fill_(0.25)
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))


def test_q1_probabilities_and_cursors():
    data, expected = golden_probability_data()
    op = RequestLocalChainSampling("cpu", 1, 6)
    op.probabilities(**data)
    assert torch.equal(data["out_probs"], expected)
    slots = torch.tensor([0, -1], dtype=torch.int32)
    cursors = torch.empty(2, dtype=torch.int64)
    op.prepare_cursors(
        torch.tensor([23, -999], dtype=torch.int64),
        torch.tensor([20], dtype=torch.int64),
        request_indices=slots,
        workspace=torch.empty(0, dtype=torch.uint8),
        out_cursors=cursors,
    )
    assert cursors.tolist() == [4, 0]
