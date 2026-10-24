# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordinary sampler GPU qualification; CUDA execution is required.

Distribution tests freeze N=8192, family alpha=.01, Bonferroni M=5*(8+9)=85
(small vocabulary bins plus eight peaks and one tail bin). No retries, and
exhaustive selection never reduces the reserved family size.
"""

import pytest
import torch

from ctikernels.sampling import ORDINARY
from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.reference import (
    categorical,
    probability_row,
    request_seed,
    uniform,
)

from ..modules._binomial import clopper_pearson
from ..modules.test_seeded_sampling import HINTS, data_for
from ._exhaustive import ci_cases
from .test_dflash_sampling import SYMB_CASES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU qualification only")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
VOCABS = (32, 4097, 151936, 248320)


def gpu_probabilities(data):
    b, v = data["logits"].shape
    chain = RequestLocalChainSampling("cuda", 2, v)
    probs = torch.empty((b, 1, v), device="cuda")
    work = torch.empty(
        chain.workspace_size(b, "probabilities", length=1), dtype=torch.uint8, device="cuda"
    )
    chain.probabilities(
        data["logits"].unsqueeze(1),
        data["temperatures"],
        data["top_k"],
        data["top_p"],
        request_indices=data["request_indices"],
        workspace=work,
        out_probs=probs,
        filters=data["filters"],
    )
    # The ordinary draws derive the min-p pivot from row state (no chunk-maxima scratch).
    return probs[:, 0].cpu()


def assert_reference(data, p):
    cpu = {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in data.items()
        if name != "workspace"
    }
    for b, actual in enumerate(cpu["out_tokens"].tolist()):
        slot = int(cpu["request_indices"][b])
        if slot == -1:
            assert actual == -1
            continue
        t = float(cpu["temperatures"][b])
        u = (
            uniform(request_seed(cpu["seeds"], slot), int(cpu["positions"][b]), ORDINARY, 0)
            if t
            else 0
        )
        weights = p[b]
        if t and cpu.get("min_p") is not None:
            pivot = cpu["min_p"][b] * weights.max()
            weights = torch.where(weights >= pivot, weights, 0.0)
        expected = categorical(weights, u) if t else int(torch.argmax(weights))
        assert actual == expected
        ref = probability_row(cpu["logits"][b], t, int(cpu["top_k"][b]), cpu["top_p"][b])
        # Provider exp may differ; identical FP32 probability bits require identical tokens.
        if torch.equal(p[b].view(torch.int32), ref.view(torch.int32)):
            if t and cpu.get("min_p") is not None:
                ref = torch.where(ref >= cpu["min_p"][b] * ref.max(), ref, 0.0)
            assert actual == (categorical(ref, u) if t else int(torch.argmax(ref)))


@pytest.mark.parametrize(
    "dtype,hint,v,b,case",
    ci_cases(
        DTYPES,
        HINTS,
        VOCABS,
        (1, 2, 3, 4, 8),
        ("random", "ties", "threshold"),
        core=[
            (
                dtype,
                hint,
                VOCABS[(i + j) % 4],
                (1, 2, 3, 4, 8)[i],
                ("random", "ties", "threshold")[j],
            )
            for i, hint in enumerate(HINTS)
            for j, dtype in enumerate(DTYPES)
        ],
    ),
)
@pytest.mark.parametrize("use_min_p", [False, True])
def test_exact_draw(dtype, hint, v, b, case, use_min_p):
    op, data = data_for(b, v, dtype, "cuda", hint)
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0] * ((b + 3) // 4), device="cuda")[:b]
    generator = torch.Generator(device="cuda").manual_seed(811)
    data["logits"].copy_(torch.randn(b, v, device="cuda", generator=generator) * 3)
    data["temperatures"].fill_(0.7)
    if case in ("ties", "threshold"):
        data["logits"].fill_(-torch.inf)
        data["logits"][:, :8] = 0
        if hint in ("top_k", "both", "auto"):
            data["top_k"].fill_(4)
        if hint in ("top_p", "both", "auto"):
            data["top_p"].fill_(0.5)
    if b > 1:
        data["temperatures"][0] = 0
    if b > 2:
        data["request_indices"][1] = -1
        data["logits"][1].fill_(torch.nan)
        data["temperatures"][1] = torch.nan
        data["positions"][1] = -1
    op.sample(**data)
    assert_reference(data, gpu_probabilities(data))
    expected = data["out_tokens"].clone()
    data["workspace"].fill_(255)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    op.sample(**data)
    assert torch.cuda.memory_allocated() == allocated
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize(
    "v,hint",
    ci_cases(VOCABS, HINTS, core=[(v, HINTS[i]) for i, v in enumerate(VOCABS)] + [(32, "auto")]),
)
@pytest.mark.parametrize("use_min_p", [False, True])
def test_batch_pool_graph_invariance(v, hint, use_min_p):
    op, data = data_for(4, v, torch.bfloat16, "cuda", hint)
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(911)
    data["logits"].copy_(torch.randn(4, v, device="cuda", generator=generator) * 3)
    data["temperatures"][0] = 0
    op.sample(**data)
    expected = data["out_tokens"].clone()
    for indices in ([0], [1], [2], [3], [3, 1, 0, 2], [2, 0, 3, 1, 1, 0]):
        other, packed = data_for(len(indices), v, torch.bfloat16, "cuda", hint)
        for name in ("logits", "temperatures", "top_k", "top_p", "positions", "request_indices"):
            packed[name] = data[name][indices].clone()
        if use_min_p:
            packed["min_p"] = data["min_p"][indices].clone()
        packed["seeds"] = data["seeds"]
        other.sample(**packed)
        assert torch.equal(packed["out_tokens"], expected[indices])
    data["seeds"] = torch.cat(
        [data["seeds"][i : i + 1] for i in [3, 0, 1, 2]]
        + [torch.zeros(3, 2, dtype=torch.uint32, device="cuda")]
    )
    data["request_indices"].copy_(torch.tensor([1, -1, 3, 0], dtype=torch.int32, device="cuda"))
    data["logits"][1].fill_(torch.nan)
    data["temperatures"][1] = torch.nan
    data["positions"][1] = -1
    expected[1] = -1
    op.sample(**data)  # warm the changed seed-pool shape
    assert torch.equal(data["out_tokens"], expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample(**data)
    graph.replay()
    assert torch.equal(data["out_tokens"], expected)
    for pos in (2**31 + 3, 2**32 - 1, 2**32, 2**40):
        data["positions"].fill_(pos)
        data["seeds"].copy_(
            torch.tensor([[17 + i, 912 + i] for i in range(7)], dtype=torch.uint32, device="cuda")
        )
        if use_min_p:
            data["min_p"].copy_(torch.tensor([1.0, 0.3, 0.05, 0], device="cuda"))
        graph.replay()
        replay = data["out_tokens"].clone()
        op.sample(**data)
        assert torch.equal(data["out_tokens"], replay)
        assert_reference(data, gpu_probabilities(data))


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("dependency", ["producer", "consumer"])
def test_pdl_stream_dependencies(hint, b, capture, dependency):
    """Delayed logits and immediate output consumers preserve stream/graph ordering."""
    v = 248320
    op, data = data_for(b, v, torch.float32, "cuda", hint)
    src = torch.full_like(data["logits"], -torch.inf)
    # Put the entire support near the end: stale logits select token zero instead.
    src[:, -8:] = torch.arange(8, dtype=torch.float32, device="cuda")
    if b > 1:
        data["temperatures"][0] = 0
        data["request_indices"][-1] = -1
    torch.mul(src, 1.0, out=data["logits"])
    torch.cuda.synchronize()  # Reference logits are ready before sampling begins.
    op.sample(**data)  # Also compile every specialization before graph capture.
    torch.cuda.synchronize()
    expected = data["out_tokens"].cpu()
    live = expected[:-1] if b > 1 else expected
    assert (live >= v - 8).all()
    if b > 1:
        assert expected[-1] == -1

    def sequence():
        torch.cuda._sleep(2_000_000)
        if dependency == "producer":
            torch.mul(src, 1.0, out=data["logits"])
        op.sample(**data)
        # No synchronization between sampling and its output consumer.
        return data["out_tokens"].clone()

    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            consumed = sequence()
    for _ in range(3):
        if dependency == "producer":
            data["logits"].fill_(-torch.inf)
            data["logits"][:, 0] = 0
        data["out_tokens"].fill_(-99)
        if capture:
            consumed.fill_(-99)
        torch.cuda.synchronize()
        if capture:
            graph.replay()
        else:
            consumed = sequence()
        assert torch.equal(consumed.cpu(), expected)
        assert torch.equal(data["out_tokens"].cpu(), expected)


@pytest.mark.parametrize("use_min_p", [False, True])
def test_empty_seed_pool_and_warm_up(use_min_p):
    op, data = data_for(3, 32, torch.float32, "cuda", "none")
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.3, 1.0], device="cuda")
    op.warm_up([3], [torch.float32], ["none"])
    # The helper must have compiled both stages for this signature.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample(**data)
    graph.replay()
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.empty((0, 2), dtype=torch.uint32, device="cuda")
    data["positions"].fill_(-1)
    data["temperatures"].fill_(torch.nan)
    data["logits"].fill_(torch.nan)
    op.sample(**data)
    assert data["out_tokens"].tolist() == [-1, -1, -1]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,v", [(1, 151936), (4, 248320), (300, 151936)])
def test_warm_up_covers_all_greedy_capture(dtype, b, v, monkeypatch):
    op, data = data_for(b, v, dtype, "cuda", "none")
    # Start from an empty compile cache so earlier tests cannot have warmed this key.
    monkeypatch.setattr(op._provider, "compiled", {})
    op.warm_up([b], [dtype], ["none"])
    generator = torch.Generator(device="cuda").manual_seed(b + v)
    data["logits"].copy_(torch.randn(b, v, device="cuda", generator=generator))
    data["temperatures"].zero_()
    # Capture compiles nothing: warm_up must have built the all_greedy specialization.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample(**data, all_greedy=True)
    graph.replay()
    torch.cuda.synchronize()
    live = data["request_indices"] >= 0
    reference = data["logits"].float().argmax(dim=1).to(torch.int32)
    assert torch.equal(data["out_tokens"][live], reference[live])


@pytest.mark.parametrize(
    "v,hint",
    ci_cases((8, 151936), HINTS, core=[(8, hint) for hint in HINTS]),
)
def test_frozen_distribution(v, hint):
    """N=8192 positions; alpha=.01/85 per bin, equal-tailed exact-binomial CI."""
    n, batch = 8192, 32
    op, data = data_for(batch, v, torch.float32, "cuda", hint)
    data["logits"].fill_(-30)
    data["logits"][:, :8] = torch.tensor(
        [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75], device="cuda"
    )
    data["seeds"].copy_(
        torch.tensor([[42, 917]], dtype=torch.uint32, device="cuda").repeat(batch, 1)
    )
    data["positions"].copy_(torch.arange(batch, device="cuda"))
    op.sample(**data)
    p = gpu_probabilities(data)[0].double()
    p /= p.sum()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample(**data)
    draws = torch.empty(n, dtype=torch.int32, device="cuda")
    for start in range(0, n, batch):
        graph.replay()
        draws[start : start + batch].copy_(data["out_tokens"])
        data["positions"].add_(batch)
    counts = torch.bincount(draws.cpu().long().clamp(max=8), minlength=9)
    masses = torch.cat((p[:8], p[8:].sum().reshape(1)))
    for count, mass in zip(counts.tolist(), masses.tolist(), strict=True):
        if mass == 0:
            assert count == 0
        lo, hi = clopper_pearson(count, n, 0.01 / 85)
        assert lo <= mass <= hi


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [3, 2048, 4097])
def test_min_p_mixed_rows_all_paths(hint, dtype, k):
    # All four min-p values must occur on live stochastic rows, including
    # the large filtered path (k>1024), compact scatter, and p=1 bypass.
    op, data = data_for(8, 4097, dtype, "cuda", hint)
    generator = torch.Generator(device="cuda").manual_seed(1201)
    data["logits"].copy_(torch.randn(8, 4097, device="cuda", generator=generator))
    data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0] * 2, device="cuda")
    if hint in ("top_k", "both", "auto"):
        data["top_k"].fill_(k)
    if hint in ("top_p", "both", "auto"):
        data["top_p"][4:] = 1
    op.sample(**data)
    assert_reference(data, gpu_probabilities(data))
    with_min_p = data["out_tokens"].clone()
    data["min_p"].zero_()
    op.sample(**data)
    zeros = data["out_tokens"].clone()
    data["min_p"] = None
    op.sample(**data)
    assert torch.equal(zeros, data["out_tokens"])
    assert with_min_p[0] == zeros[0]


@pytest.mark.parametrize("hint", ["top_p", "both", "auto"])
@pytest.mark.parametrize("k", [3, 4097])
def test_min_p_top_p_can_remove_raw_logit_argmax(hint, k):
    op, data = data_for(4, 4097, torch.float32, "cuda", hint)
    data["logits"].fill_(-torch.inf)
    # exp(-2^-26) rounds to 1: top-p breaks this probability tie by ID.
    data["logits"][:, :3] = torch.tensor([-(2**-26), 0.0, -1.0], device="cuda")
    data["top_k"].fill_(4097 if hint == "top_p" else k)
    data["top_p"].fill_(0.5)
    data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0], device="cuda")
    op.sample(**data)
    p = gpu_probabilities(data)
    assert_reference(data, p)
    # A more aggressive top-p removes the raw argmax at ID 1 entirely.
    data["top_p"].fill_(0.25)
    op.sample(**data)
    p = gpu_probabilities(data)
    assert p[:, 1].eq(0).all() and p[:, 0].eq(1).all()
    assert_reference(data, p)


@pytest.mark.parametrize("hint", ["top_k", "both"])
@pytest.mark.parametrize("k", [1, 5, 20, 1024])
@pytest.mark.parametrize("pattern", ["flat", "tied_kth"])
def test_compact_split_ties(hint, k, pattern):
    # Compact top-k rows whose k-th logit ties across chunks keep the smallest IDs; the
    # compact kernel fills them in order from P4's per-chunk tie counts.
    b, v = 4, 151936
    op, data = data_for(b, v, torch.bfloat16, "cuda", hint)
    generator = torch.Generator(device="cuda").manual_seed(1301 + k)
    logits = torch.randn(b, v, device="cuda", generator=generator)
    if pattern == "flat":
        logits.zero_()
    else:
        ties = torch.rand(b, v, device="cuda", generator=generator) < 64 / v
        logits = torch.where(ties, 2.0, logits.clamp(max=1.0))
        logits[:, :4] = 3.0
    data["logits"].copy_(logits)
    data["temperatures"].fill_(0.8)
    data["top_k"].fill_(k)
    if hint == "both":
        data["top_p"].copy_(torch.tensor([1.0, 0.9, 0.5, 1e-3], device="cuda"))
    data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0], device="cuda")
    for _ in range(3):
        op.sample(**data)
        assert_reference(data, gpu_probabilities(data))
        data["positions"].add_(1)


@pytest.mark.parametrize("hint", ["top_p", "auto"])
@pytest.mark.parametrize("pattern", ["flat", "two_level", "random"])
def test_top_p_histogram_rows(hint, pattern):
    # Flat rows put a whole warp in one mass bin (aggregated sliced adds); mixed rows
    # use per-lane sliced adds. Both must resolve top-p exactly.
    b, v = 4, 151936
    op, data = data_for(b, v, torch.bfloat16, "cuda", hint)
    generator = torch.Generator(device="cuda").manual_seed(1401)
    logits = torch.randn(b, v, device="cuda", generator=generator) * 2
    if pattern == "flat":
        logits.zero_()
    elif pattern == "two_level":
        logits = torch.where(torch.rand(b, v, device="cuda", generator=generator) < 0.5, 0.0, 0.25)
    data["logits"].copy_(logits)
    data["temperatures"].copy_(torch.tensor([1.0, 0.7, 1.3, 0.0], device="cuda"))
    data["top_k"].fill_(v)
    data["top_p"].copy_(torch.tensor([0.9, 0.5, 0.999, 0.9], device="cuda"))
    data["min_p"] = torch.tensor([0, 0.05, 0.3, 0.0], device="cuda")
    for _ in range(3):
        op.sample(**data)
        assert_reference(data, gpu_probabilities(data))
        data["positions"].add_(1)


@pytest.mark.parametrize("use_min_p", [False, True])
def test_identity_large_batch_loop(use_min_p):
    # Enough rows that P4/D1 CTAs each take several chunk groups.
    b, v = 192, 151936
    op, data = data_for(b, v, torch.bfloat16, "cuda", "none")
    generator = torch.Generator(device="cuda").manual_seed(1501)
    data["logits"].copy_(torch.randn(b, v, device="cuda", generator=generator) * 3)
    temperatures = torch.tensor([0.0, 0.7, 1.0, 1.3], device="cuda").repeat(b // 4)
    data["temperatures"].copy_(temperatures)
    data["request_indices"][5] = -1
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0], device="cuda").repeat(b // 4)
    op.sample(**data)
    assert_reference(data, gpu_probabilities(data))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("b,v", [(1, 32), (3, 4097), (4, 151936), (31, 248320), (300, 151936)])
def test_all_greedy_hint_matches_default_path(dtype, b, v):
    op, data = data_for(b, v, dtype, "cuda", "none")
    generator = torch.Generator(device="cuda").manual_seed(b * 7 + v)
    # Rounded logits make cross-chunk and in-chunk ties common.
    logits = (torch.randn(b, v, device="cuda", generator=generator) * 4).round()
    data["logits"].copy_(logits)
    if b > 1:
        data["request_indices"][b // 2] = -1
        data["logits"][b // 3].fill_(-torch.inf)
    data["temperatures"].zero_()
    op.sample(**data)
    expected = data["out_tokens"].clone()
    reference = data["logits"].float().argmax(dim=1).to(torch.int32)
    live = data["request_indices"] >= 0
    assert torch.equal(expected[live], reference[live])
    assert (expected[~live] == -1).all()
    # The hint reads none of the sampling inputs.
    data["temperatures"].fill_(torch.nan)
    data["top_k"].fill_(-999)
    data["top_p"].fill_(torch.nan)
    data["min_p"] = torch.full((b,), torch.nan, device="cuda")
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32, device="cuda")
    data["request_indices"].masked_fill_(live, 0)
    data["out_tokens"].fill_(7)
    for name in ("none", "top_k", "both"):
        data["filters"] = name
        op.sample(**data, all_greedy=True)
        assert torch.equal(data["out_tokens"], expected)
    data["out_tokens"].fill_(7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample(**data, all_greedy=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize(
    "batch,length,vocab,dtype,hint,use_min_p",
    SYMB_CASES,
)
def test_symb_ordinary_strided_grid(batch, length, vocab, dtype, hint, use_min_p):
    from ..modules.test_dflash_symb import strided_copy

    cpu_op, cpu = data_for(batch, vocab, dtype, hint=hint)
    cpu["logits"].fill_(-torch.inf)
    cpu["logits"][:, :8] = 0
    cpu["temperatures"][1::3] = 0
    cpu["request_indices"][2::4] = -1
    if hint in ("top_k", "both", "auto"):
        cpu["top_k"].fill_(4)
    if use_min_p:
        cpu["min_p"] = torch.full((batch,), 0.5)
    cpu_op.sample(**cpu)
    op, data = data_for(batch, vocab, dtype, "cuda", hint)
    data.update(
        {
            name: value.cuda() if isinstance(value, torch.Tensor) else value
            for name, value in cpu.items()
        }
    )
    data["logits"] = strided_copy(data["logits"], 3)
    op.sample(**data)
    assert torch.equal(data["out_tokens"].cpu(), cpu["out_tokens"])


@pytest.mark.parametrize(
    "batches",
    [
        (0, 1, 5),
        pytest.param(
            (0, 1, 2, 3, 4, 5, 8, 16, 32, 48, 191, 256, 1024), marks=pytest.mark.exhaustive
        ),
    ],
)
def test_symb_ordinary_capacity_capture(batches, monkeypatch):
    from ctikernels.sampling import RequestLocalSampling

    from ..modules.test_dflash_symb import strided_copy

    op = RequestLocalSampling("cuda", 16)
    monkeypatch.setattr(op._provider, "compiled", {})
    keys = op.warm_up(1024)
    count = len(op._provider.compiled)
    assert count == len(keys)
    for b in batches:
        for dtype in DTYPES:
            for hint in HINTS:
                cpu_op, cpu = data_for(b, 16, dtype, hint=hint)
                cpu["logits"].fill_(-torch.inf)
                cpu["logits"][:, :8] = 0
                cpu["temperatures"][1::3] = 0
                cpu["request_indices"][2::4] = -1
                for pool_rows in (0, max(1, b), b + 7):
                    cpu["seeds"] = torch.zeros(pool_rows, 2, dtype=torch.uint32)
                    cpu["request_indices"] = torch.arange(b, dtype=torch.int32)
                    cpu["request_indices"][2::4] = -1
                    if not pool_rows:
                        cpu["request_indices"].fill_(-1)
                    for use in (False, True):
                        cpu["min_p"] = torch.full((b,), 0.5) if use else None
                        for greedy in (False, True):
                            cpu_op.sample(**cpu, all_greedy=greedy)
                            data = {
                                name: value.cuda() if isinstance(value, torch.Tensor) else value
                                for name, value in cpu.items()
                            }
                            data["logits"] = strided_copy(data["logits"], 1 + b % 4)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                op.sample(**data, all_greedy=greedy)
                            graph.replay()
                            assert len(op._provider.compiled) == count
                            assert torch.equal(data["out_tokens"].cpu(), cpu["out_tokens"])


@pytest.mark.exhaustive
def test_symb_identity_capture_changes_group_and_loop(monkeypatch):
    from ctikernels.sampling import RequestLocalSampling

    from ..modules.test_dflash_symb import strided_copy

    vocab, dtype = 151936, torch.bfloat16
    op = RequestLocalSampling("cuda", vocab)
    monkeypatch.setattr(op._provider, "compiled", {})
    op.warm_up(191, dtypes=(dtype,), filters=("none",))
    count = len(op._provider.compiled)
    for b in (1, 4, 32, 191):
        # Eight equal leading probabilities and all-zero tails have identical
        # CDF bits at V=16 and V=151936. This keeps CPU oracle work bounded.
        ref, cpu = data_for(b, 16, dtype, hint="none")
        cpu["logits"].fill_(-torch.inf)
        cpu["logits"][:, :8] = 0
        cpu["temperatures"][1::3] = 0
        cpu["request_indices"][2::4] = -1
        for use in (False, True):
            cpu["min_p"] = torch.full((b,), 0.5) if use else None
            ref.sample(**cpu)
            data = {
                name: value.cuda() if isinstance(value, torch.Tensor) else value
                for name, value in cpu.items()
            }
            logits = torch.full((b, vocab), -torch.inf, dtype=dtype, device="cuda")
            logits[:, :8] = 0
            data["logits"] = strided_copy(logits, 1)
            data["top_k"].fill_(vocab)
            data["workspace"] = torch.empty(op.workspace_size(b), dtype=torch.uint8, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.sample(**data)
            graph.replay()
            assert len(op._provider.compiled) == count
            assert torch.equal(data["out_tokens"].cpu(), cpu["out_tokens"])
