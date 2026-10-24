# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordinary RNG, borrowed buffers, validation and frozen CPU qualification."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels.sampling import ORDINARY, RequestLocalSampling
from ctikernels.sampling._dflash import MAX_VOCAB, RequestLocalChainSampling
from ctikernels.sampling._dflash.reference import categorical, probability_row, rng

from ._binomial import clopper_pearson
from .test_dflash_cute_boundary import BorrowedStorageOnly

HINTS = ("none", "top_k", "top_p", "both", "auto")
GOLDEN = [
    # seed, response position, logits, T, k, p, frozen token
    (0, 0, [0, 0, 0, 0], 1, 4, 1, 0),
    (42, 2**31 + 7, [0, 1, 2, 3], 0.7, 4, 1, 3),
    (2**64 - 1, 2**32 - 1, [3, 2, 2, 2], 1, 2, 1, 0),
    (42, 2**32, [0, 0, 0, 0], 1, 4, 0.5, 0),
    (42, 2**32 + 1, [0, 0, 0, 0], 1, 4, 0.5, 1),
    (42, 2**63 - 1, [3, 2, 2, 2], 1, 2, 1, 1),
    (42, 2**32 + 1, [2, 3, 3, 0], 0, 1, 0.1, 1),
    (0x123456789ABCDEF0, 2**40, [0, 1, 2, 3], 0.6, 3, 0.8, 3),
    (42, 2**63 - 1, [-torch.inf, 0, 0, -torch.inf], 1, 4, 1, 2),
]


def data_for(b=3, v=4, dtype=torch.float32, device="cpu", hint="auto"):
    op = RequestLocalSampling(device, v)
    k = min(3, v) if hint in ("top_k", "both", "auto") else v
    p = 0.75 if hint in ("top_p", "both", "auto") else 1.0
    data = {
        "logits": torch.arange(v, dtype=torch.float32).repeat(b, 1).to(device=device, dtype=dtype),
        "temperatures": torch.ones(b, device=device),
        "top_k": torch.full((b,), k, dtype=torch.int32, device=device),
        "top_p": torch.full((b,), p, device=device),
        "request_indices": torch.arange(b, dtype=torch.int32, device=device),
        "seeds": torch.tensor(
            [[42 + i, i] for i in range(b)], dtype=torch.uint32, device=device
        ).reshape(b, 2),
        "positions": torch.arange(b, dtype=torch.int64, device=device) + 2**32,
        "workspace": torch.full((op.workspace_size(b),), 255, dtype=torch.uint8, device=device),
        "out_tokens": torch.empty(b, dtype=torch.int32, device=device),
        "filters": hint,
    }
    return op, data


@pytest.mark.parametrize("seed,pos,logits,t,k,p,expected", GOLDEN)
def test_golden(seed, pos, logits, t, k, p, expected):
    op, data = data_for(1)
    data["logits"][0] = torch.tensor(logits)
    data["temperatures"].fill_(t)
    data["top_k"].fill_(k)
    data["top_p"].fill_(p)
    data["seeds"][0] = torch.tensor([seed & rng.MASK, seed >> 32], dtype=torch.uint32)
    data["positions"].fill_(pos)
    # Recompute the counter/key mapping independently of uniform()/words().
    word = rng.philox4x32((pos & rng.MASK, pos >> 32, 0x100, 0), (seed & rng.MASK, seed >> 32))[0]
    u = (word >> 8) / 2**24
    assert ORDINARY == 0x100 and u == rng.uniform(seed, pos, ORDINARY, 0)
    row = probability_row(data["logits"][0], t, k, data["top_p"][0])
    assert categorical(row, u) == expected
    assert op.sample(**data) is None
    assert data["out_tokens"].tolist() == [expected]


def test_threshold_and_kth_ties_are_retained():
    row = probability_row(torch.zeros(4), 1, 4, 0.5)
    assert row.tolist() == [0.5, 0.5, 0, 0]
    row = probability_row(torch.tensor([3, 2, 2, 2]), 1, 2, 1)
    assert torch.nonzero(row).flatten().tolist() == [0, 1]


@pytest.mark.parametrize("b", [0, 1, 3, 8])
@pytest.mark.parametrize("v", [1, 32, 4097, 151936, 248320, MAX_VOCAB])
def test_workspace(b, v):
    op = RequestLocalSampling("cpu", v)
    layout = op.workspace_layout(b)
    sizes = [
        b * 5120 * 4,
        b * 5120 * 8,
        b * ((v + 4095) // 4096) * 20,
        b * 8196,
        b * 288,
        b * ((v + 4095) // 4096) * 4,
        b * ((v + 4095) // 4096) * 4,
        b * v * 4,
        b * ((v + 4095) // 4096) * 8,
    ]
    assert layout["base_alignment"] == 8
    assert (
        layout["total_bytes"] == op.workspace_size(b) == sum((s + 255) // 256 * 256 for s in sizes)
    )
    if not b:
        assert layout["regions"] == ()
        return
    assert [r["name"] for r in layout["regions"]] == [
        "top_k_histograms",
        "top_p_histograms",
        "probability_chunk_records",
        "compact_lists",
        "row_states",
        "min_p_chunk_maxima",
        "min_p_chunk_partials",
        "probabilities",
        "chunk_partials",
    ]
    offset = 0
    for region, size in zip(layout["regions"], sizes, strict=True):
        assert region["payload_bytes"] == size
        assert region["offset"] == offset and offset % 256 == 0
        offset += region["aligned_extent"]
    assert op.launch_counts == {"none": 4, "top_k": 9, "top_p": 9, "both": 12, "auto": 12}


@pytest.mark.parametrize("v", [0, -1, MAX_VOCAB + 1, True, 1.5])
def test_vocab_bounds(v):
    with pytest.raises(ValueError, match="vocab_size"):
        RequestLocalSampling("cpu", v)


@pytest.mark.parametrize("b", [-1, 2**31, True, 1.5])
def test_capacity_bounds(b):
    with pytest.raises(ValueError, match="capacity"):
        RequestLocalSampling("cpu", 4).workspace_size(b)


@pytest.mark.parametrize(
    "name",
    [
        "logits",
        "temperatures",
        "top_k",
        "top_p",
        "min_p",
        "request_indices",
        "seeds",
        "positions",
        "out_tokens",
    ],
)
@pytest.mark.parametrize("fault", ["dtype", "shape", "layout", "device", "not_tensor"])
def test_tensor_validation(name, fault):
    op, data = data_for()
    data["min_p"] = torch.full((3,), 0.3)
    t = data[name]
    if fault == "dtype":
        data[name] = t.double()
    elif fault == "shape":
        data[name] = t.unsqueeze(0)
    elif fault == "layout":
        data[name] = torch.empty((*t.shape, 2), dtype=t.dtype)[..., 0]
    elif fault == "device":
        data[name] = torch.empty_like(t, device="meta")
    else:
        data[name] = "not a tensor"
    with pytest.raises((ValueError, TypeError)):
        op.sample(**data)


@pytest.mark.parametrize("name", ["logits", "temperatures", "top_p", "min_p"])
def test_grad_rejected_before_provider(name, monkeypatch):
    op, data = data_for()
    monkeypatch.setattr(op._provider, "run", lambda *args: pytest.fail("provider reached"))
    data["min_p"] = torch.full((3,), 0.3)
    data[name].requires_grad_(True)
    with pytest.raises(ValueError, match="requires grad"):
        op.sample(**data)


@pytest.mark.parametrize("fault", ["small", "dtype", "shape", "layout", "alignment", "device"])
def test_bad_workspace(fault):
    op, data = data_for()
    n = op.workspace_size(3)
    data["workspace"] = {
        "small": lambda: torch.empty(n - 1, dtype=torch.uint8),
        "dtype": lambda: torch.empty(n),
        "shape": lambda: torch.empty(1, n, dtype=torch.uint8),
        "layout": lambda: torch.empty(n * 2, dtype=torch.uint8)[::2],
        "alignment": lambda: torch.empty(n + 1, dtype=torch.uint8)[1:],
        "device": lambda: torch.empty(n, dtype=torch.uint8, device="meta"),
    }[fault]()
    with pytest.raises(ValueError, match="workspace"):
        op.sample(**data)


@pytest.mark.parametrize(
    "name",
    [
        "logits",
        "temperatures",
        "top_k",
        "top_p",
        "min_p",
        "request_indices",
        "seeds",
        "positions",
        "out_tokens",
    ],
)
def test_workspace_aliases(name):
    op, data = data_for()
    data["min_p"] = torch.full((3,), 0.3)
    t = data[name]
    data[name] = data["workspace"][: t.numel() * t.element_size()].view(t.dtype).view(t.shape)
    with pytest.raises(ValueError, match="alias"):
        op.sample(**data)


@pytest.mark.parametrize("name", ["top_k", "request_indices", "positions", "min_p"])
def test_output_aliases(name):
    op, data = data_for()
    data["min_p"] = torch.full((3,), 0.3)
    data["out_tokens"] = data[name].view(torch.int32)[:3]
    with pytest.raises(ValueError, match="alias"):
        op.sample(**data)


@pytest.mark.parametrize("filters", ["bad", None, [], 1])
def test_bad_filters(filters):
    op, data = data_for()
    data["filters"] = filters
    with pytest.raises(ValueError, match="filters"):
        op.sample(**data)


def test_empty_and_greedy_skip_rng(monkeypatch):
    from ctikernels.sampling._dflash.reference import ops

    monkeypatch.setattr(ops, "uniform", lambda *args: pytest.fail("unexpected RNG"))
    op, data = data_for(0)
    op.sample(**data)
    op, data = data_for()
    data["temperatures"].zero_()
    data["positions"].fill_(-1)
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32)
    op.sample(**data)
    assert data["out_tokens"].tolist() == [3, 3, 3]
    data["request_indices"].fill_(-1)
    data["logits"].fill_(torch.nan)
    data["temperatures"].fill_(torch.nan)
    data["top_k"].fill_(-999)
    data["top_p"].fill_(torch.nan)
    op.sample(**data)
    assert data["out_tokens"].tolist() == [-1, -1, -1]


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("use_min_p", [False, True])
def test_batch_invariance(hint, use_min_p):
    op, data = data_for(4, hint=hint)
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.05, 0.3, 1.0])
    data["temperatures"][0] = 0
    op.sample(**data)
    expected = data["out_tokens"].clone()
    for indices in ([0], [1], [2], [3], [3, 1, 0, 2], [2, 0, 3, 1, 1, 0]):
        other, packed = data_for(len(indices), hint=hint)
        for name in ("logits", "temperatures", "top_k", "top_p", "positions", "request_indices"):
            packed[name] = data[name][indices].clone()
        if use_min_p:
            packed["min_p"] = data["min_p"][indices].clone()
        packed["seeds"] = data["seeds"]
        other.sample(**packed)
        assert torch.equal(packed["out_tokens"], expected[indices])
    # Reorder the pool, change S, and place a poisoned padding row between live rows.
    order = [3, 0, 1, 2]
    data["seeds"] = torch.cat(
        [data["seeds"][i : i + 1] for i in order] + [torch.zeros(1, 2, dtype=torch.uint32)]
    )
    data["request_indices"] = torch.tensor([1, -1, 3, 0], dtype=torch.int32)
    data["logits"][1].fill_(torch.nan)
    data["temperatures"][1] = torch.nan
    data["positions"][1] = -1
    op.sample(**data)
    expected[1] = -1
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize("hint", HINTS[:4])
@pytest.mark.parametrize("min_p", [None, 0.3])
def test_frozen_distribution(hint, min_p):
    """N=2048 positions, alpha=.01, Bonferroni M=4 modes * 2 min-p settings * 4 bins=32; no retries.

    Exact-binomial Clopper-Pearson intervals test the normalized filtered FP32 p.
    Zero-mass bins must have exactly zero observations.
    """
    n = 2048
    op, data = data_for(1, hint=hint)
    p = probability_row(data["logits"][0], 1, int(data["top_k"][0]), data["top_p"][0])
    if min_p is not None:
        data["min_p"] = torch.tensor([min_p])
        p = torch.where(p >= data["min_p"][0] * p.max(), p, 0.0)
    p = p.double()
    p /= p.sum()
    counts = torch.zeros(4, dtype=torch.int64)
    for pos in range(n):
        data["positions"].fill_(pos)
        op.sample(**data)
        counts[int(data["out_tokens"][0])] += 1
    for count, mass in zip(counts.tolist(), p.tolist(), strict=True):
        if mass == 0:
            assert count == 0
        lo, hi = clopper_pearson(count, n, 0.01 / 32)
        assert lo <= mass <= hi


def test_warm_up():
    op = RequestLocalSampling("cpu", 4)
    op.warm_up([0, 1, 3], [torch.float16, torch.bfloat16, torch.float32], HINTS)


@pytest.mark.parametrize("use_min_p", [False, True])
def test_cute_boundary_without_storage_or_readback(monkeypatch, use_min_p):
    from ctikernels.sampling._dflash.cute import Provider

    op, data = data_for(3, 4097)
    if use_min_p:
        data["min_p"] = torch.tensor([0, 0.3, 1.0])
    provider = Provider.__new__(Provider)
    provider.compiled = {}
    from .test_dflash_symb import KernelStub

    provider.kernels = SimpleNamespace(
        Probabilities=lambda *args, **kw: KernelStub(("probabilities", args, kw)),
        Chain=lambda *args, **kw: KernelStub(("chain", args, kw)),
        OrdinaryFiltered=lambda *args, **kw: KernelStub(
            (
                "filtered",
                args,
                {k: v for k, v in kw.items() if k not in ("compact_reserve", "strided")},
            )
        ),
        identity_group=lambda *args: 1,
    )
    op._provider = provider
    capturing, launches, compiled = False, [], []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=48)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    def compile_stub(kernel, tensors):
        assert not capturing
        compiled.append(kernel)
        return lambda *args: launches.append(kernel)

    monkeypatch.setattr(provider, "_compile", compile_stub)
    with BorrowedStorageOnly([x for x in data.values() if isinstance(x, torch.Tensor)]):
        capturing = True
        with pytest.raises(RuntimeError, match="warm .* before capture"):
            op.sample(**data)
        capturing = False
        op.sample(**data)
        capturing = True
        op.sample(**data)
    # One compiled operation: probabilities without large-row FP32 storage plus the draw.
    assert len(compiled) == 1 and len(launches) == 2
    assert compiled[0] == ("filtered", (4097, 3, "auto", ORDINARY), {"min_p": use_min_p})
    keys = [key[0] for key in provider.compiled]
    modes = ["ordinary_filtered"]
    assert keys == modes
    # Existing proposal construction retains defaults and a distinct cache key.
    chain = RequestLocalChainSampling("cpu", 2, 4097)
    chain._provider = provider
    capturing = False
    q = torch.empty(3, 1, 4097)
    work = torch.empty(chain.workspace_size(3), dtype=torch.uint8)
    chain.sample_dense_proposals(
        q,
        data["temperatures"],
        request_indices=data["request_indices"],
        seeds=data["seeds"],
        cursors=data["positions"],
        workspace=work,
        out_ids=data["out_tokens"].view(3, 1),
    )
    assert compiled[-1] == ("chain", (1, 4097, "sample"), {})
    assert [key[0] for key in provider.compiled] == [*modes, "sample"]
    assert chain.probability_launch_counts["auto"] == 11


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_min_p_zero_preserves_tokens(hint, dtype):
    op, data = data_for(8, 32, dtype, hint=hint)
    data["temperatures"][0] = 0
    op.sample(**data)
    expected = data["out_tokens"].clone()
    data["min_p"] = torch.zeros(8)
    op.sample(**data)
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize(
    "min_p,expected", [(0.0, [0, 1, 2, 2, 2]), (0.5, [0, 0, 1, 1, 1]), (1.0, [0, 0, 0, 0, 0])]
)
def test_min_p_exact_pivot_and_last_positive(min_p, expected, monkeypatch):
    from ctikernels.sampling._dflash.reference import ops

    # Exact pivot 0.5 * 0.5 = 0.25 must retain token 1. No renorm before CDF.
    q = torch.tensor([[[0.5, 0.25, 0.125, 0.0]]])
    op, data = data_for(1)
    # Include U=1 to force the defensive last-positive fallback over masked w.
    draws = iter([0.0, 0.6, 0.9, 1 - 2**-24, 1.0])
    monkeypatch.setattr(ops, "uniform", lambda *args: next(draws))
    actual = []
    for _ in range(5):
        op._provider.run(
            "ordinary_min_p",
            (
                q,
                data["temperatures"],
                data["request_indices"],
                data["seeds"],
                data["positions"],
                data["workspace"],
                data["out_tokens"].view(1, 1),
                torch.tensor([min_p]),
                torch.empty(1),
            ),
        )
        actual.append(int(data["out_tokens"][0]))
    assert actual == expected


@pytest.mark.parametrize("k,top_p", [(3, 0.6), (4, 0.55)])
def test_min_p_after_top_k_top_p(k, top_p):
    op, data = data_for(1)
    data["logits"][0] = torch.tensor([0.5, 0.3, 0.1, 0.1]).log()
    data["top_k"].fill_(k)
    data["top_p"].fill_(top_p)
    data["min_p"] = torch.tensor([0.25])
    row = probability_row(data["logits"][0], 1, k, data["top_p"][0])
    weights = torch.where(row >= data["min_p"][0] * row.max(), row, 0.0)
    assert torch.nonzero(weights).flatten().tolist() == [0, 1]
    # Applying min-p first renormalizes [0.5, 0.3] to [0.625, 0.375];
    # Either top-p threshold then keeps only ID 0, which is the wrong order.
    wrong = probability_row(torch.tensor([0.5, 0.3, 0, 0]).log(), 1, 4, top_p)
    assert torch.nonzero(wrong).flatten().tolist() == [0]
    tokens = []
    for pos in [0, 1, 2, 3, 4, 5, 2**32, 2**63 - 1]:
        data["positions"].fill_(pos)
        op.sample(**data)
        tokens.append(int(data["out_tokens"][0]))
    assert tokens == [0, 0, 0, 1, 0, 0, 0, 1]


def test_min_p_greedy_and_padding_ignore_poison():
    op, data = data_for(3)
    data["min_p"] = torch.full((3,), torch.nan)
    data["temperatures"].zero_()
    data["request_indices"][1] = -1
    data["logits"][1].fill_(torch.nan)
    data["positions"].fill_(-1)
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32)
    op.sample(**data)
    assert data["out_tokens"].tolist() == [3, -1, 3]


def test_min_p_range_is_admission_precondition(monkeypatch):
    op, data = data_for(3)
    data["min_p"] = torch.tensor([-1.0, 2.0, torch.nan])
    modes = []
    monkeypatch.setattr(op._provider, "run", lambda mode, *args: modes.append(mode))
    with BorrowedStorageOnly([x for x in data.values() if isinstance(x, torch.Tensor)]):
        op.sample(**data)
    assert modes == ["probabilities_min_p", "ordinary_min_p"]


def test_min_p_empty_batch(monkeypatch):
    op, data = data_for(0)
    data["min_p"] = torch.empty(0)
    monkeypatch.setattr(op._provider, "run", lambda *args: pytest.fail("unexpected launch"))
    op.sample(**data)


def test_min_p_pivot_rounds_once_to_fp32(monkeypatch):
    from ctikernels.sampling._dflash.reference import ops

    op, data = data_for(1)
    min_p = torch.tensor([0.3])
    maximum = torch.tensor(0.3)
    pivot = min_p[0] * maximum
    assert pivot.double() < min_p[0].double() * maximum.double()
    q = torch.stack([maximum, pivot, torch.tensor(0.01), torch.tensor(0.0)]).view(1, 1, 4)
    monkeypatch.setattr(ops, "uniform", lambda *args: 0.99)
    op._provider.run(
        "ordinary_min_p",
        (
            q,
            data["temperatures"],
            data["request_indices"],
            data["seeds"],
            data["positions"],
            data["workspace"],
            data["out_tokens"].view(1, 1),
            min_p,
            torch.empty(1),
        ),
    )
    assert data["out_tokens"].tolist() == [1]


@pytest.mark.parametrize("hint", HINTS)
def test_all_greedy_hint_matches_t0_and_reads_no_sampling_inputs(hint, monkeypatch):
    from ctikernels.sampling._dflash.reference import ops

    op, data = data_for(4, v=6, hint=hint)
    data["logits"] = torch.tensor(
        [[0, 5, 5, 1, 0, 0], [-torch.inf] * 6, [2, 1, 0, 7, 7, 7], [3, 3, 3, 3, 3, 3.0]]
    )
    data["request_indices"][1] = -1
    data["temperatures"].zero_()
    op.sample(**data)
    expected = data["out_tokens"].clone()
    assert expected.tolist() == [1, -1, 3, 0]
    monkeypatch.setattr(ops, "uniform", lambda *args: pytest.fail("unexpected RNG"))
    data["temperatures"].fill_(torch.nan)
    data["top_k"].fill_(-999)
    data["top_p"].fill_(torch.nan)
    data["min_p"] = torch.full((4,), torch.nan)
    data["out_tokens"].fill_(7)
    op.sample(**data, all_greedy=True)
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_all_greedy_hint_must_be_bool(value):
    op, data = data_for()
    with pytest.raises(TypeError, match="all_greedy"):
        op.sample(**data, all_greedy=value)
