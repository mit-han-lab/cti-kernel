# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU coverage of runtime sampling shapes, row strides and warm-up contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels.sampling import RequestLocalSampling
from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.cute.provider import Provider, compile_key

from .test_dflash_probabilities import probability_data
from .test_seeded_sampling import HINTS, data_for


class KernelStub(tuple):
    def config_key(self):
        return self[0], repr(self[1:])


def semantic(kw):
    # Grid/occupancy-only options (flat grid, compact CTA reservation) are not
    # what these stubs check.
    return {k: v for k, v in kw.items() if k not in ("flat", "compact_reserve", "strided")}


def stub_provider(monkeypatch):
    provider = Provider.__new__(Provider)
    provider.compiled = {}
    provider.kernels = SimpleNamespace(
        Probabilities=lambda length, vocab, hint, batch, group, **kw: KernelStub(
            ("probabilities", (vocab, hint, group), semantic(kw))
        ),
        MinPRenormalize=lambda length, vocab, batch, **kw: KernelStub(("min_p", (vocab,))),
        identity_group=lambda rows, vocab, sms: (
            2 if rows * ((vocab + 4095) // 4096) >= 4 * sms else 1
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=48)
    )
    monkeypatch.setattr(provider, "_compile", lambda *args: lambda *args: None)
    return provider


def strided_copy(logits, pad=3):
    shape = logits.shape
    if logits.ndim == 2:
        strides = (shape[-1] + pad, 1)
    else:
        strides = (shape[1] * (shape[-1] + pad) + pad, shape[-1] + pad, 1)
    result = torch.empty_strided(shape, strides, dtype=logits.dtype, device=logits.device)
    result.copy_(logits)
    return result


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("use_min_p", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_strided_probabilities_reference(hint, use_min_p, dtype):
    b, length, v = 5, 3, 16
    logits = torch.randn(b, length, v, generator=torch.Generator().manual_seed(91)).to(dtype)
    data = probability_data(logits)
    data["temperatures"][1] = 0
    data["request_indices"][-1] = -1
    if hint in ("top_k", "both", "auto"):
        data["top_k"].fill_(7)
    if hint in ("top_p", "both", "auto"):
        data["top_p"].fill_(0.75)
    if use_min_p:
        data["min_p"] = torch.tensor([0, 1, 0.2, 0.5, torch.nan])
    op = RequestLocalChainSampling("cpu", length, v)
    op.probabilities(**data, filters=hint)
    expected = data["out_probs"].clone()
    data["logits"] = strided_copy(logits)
    op.probabilities(**data, filters=hint)
    assert torch.equal(data["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("use_min_p", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_strided_ordinary_reference(hint, use_min_p, dtype):
    op, data = data_for(8, 16, dtype, hint=hint)
    data["temperatures"][1] = 0
    data["request_indices"][-1] = -1
    if use_min_p:
        data["min_p"] = torch.full((8,), 0.5)
    op.sample(**data)
    expected = data["out_tokens"].clone()
    data["logits"] = strided_copy(data["logits"])
    op.sample(**data)
    assert torch.equal(data["out_tokens"], expected)


@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("bad", ["inner", "row", "batch"])
def test_invalid_logits_strides(ordinary, bad):
    if ordinary and bad == "batch":
        pytest.skip("only rank-three logits have a separate batch stride")
    if ordinary:
        op, data = data_for(3, 16)
        strides = (32, 2) if bad == "inner" else (15, 1)
        call = op.sample
    else:
        op = RequestLocalChainSampling("cpu", 2, 16)
        data = probability_data(torch.zeros(3, 2, 16))
        strides = {"inner": (64, 32, 2), "row": (32, 15, 1), "batch": (31, 16, 1)}[bad]
        call = op.probabilities
    data["logits"] = torch.empty_strided(data["logits"].shape, strides)
    with pytest.raises(ValueError, match="layout"):
        call(**data)


@pytest.mark.parametrize("ordinary", [False, True])
def test_empty_strided_batch(ordinary, monkeypatch):
    if ordinary:
        op, data = data_for(0, 16)
        call = op.sample
    else:
        op = RequestLocalChainSampling("cpu", 2, 16)
        data = probability_data(torch.empty(0, 2, 16))
        call = op.probabilities
    data["logits"] = strided_copy(data["logits"])
    monkeypatch.setattr(op._provider, "run", lambda *args: pytest.fail("empty launch"))
    call(**data)


def test_logits_alias_uses_strided_span():
    op, data = data_for(2, 16)
    backing = torch.empty(128)
    data["logits"] = backing.as_strided((2, 16), (64, 1))
    # Beyond numel()*element_size(), inside the second row's actual storage.
    data["out_tokens"] = backing[64:66].view(torch.int32)
    with pytest.raises(ValueError, match="alias"):
        op.sample(**data)


def test_single_batch_unused_outer_stride():
    data = probability_data(torch.zeros(1, 2, 16))
    data["logits"] = torch.zeros(32).as_strided((1, 2, 16), (0, 16, 1))
    RequestLocalChainSampling("cpu", 2, 16).probabilities(**data)


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", [], [1, -1], [True], [[1]], [1, 1.5], 2**31])
@pytest.mark.parametrize("ordinary", [False, True])
def test_warm_up_invalid_capacity(value, ordinary):
    op = RequestLocalSampling("cpu", 16) if ordinary else RequestLocalChainSampling("cpu", 4, 16)
    with pytest.raises(ValueError, match="capacity|batch"):
        op.warm_up(batch_capacity=value)


@pytest.mark.parametrize("kw", [{"dtypes": [torch.int32]}, {"filters": ["bad"]}])
@pytest.mark.parametrize("ordinary", [False, True])
def test_warm_up_invalid_choices(kw, ordinary):
    op = RequestLocalSampling("cpu", 16) if ordinary else RequestLocalChainSampling("cpu", 4, 16)
    with pytest.raises(ValueError):
        op.warm_up(batch_capacity=1, **kw)


def test_warm_up_cpu_validation_and_compatibility():
    chain = RequestLocalChainSampling("cpu", 4, 16)
    ordinary = RequestLocalSampling("cpu", 16)
    assert chain.warm_up(batch_capacity=1024) == ()
    assert not hasattr(chain, "warm_up_probabilities")
    assert ordinary.warm_up([0, 1, 8]) == ()
    assert ordinary.warm_up(0) == ()
    for kw in ({"min_p": [1]}, {"modes": ["bad"]}):
        with pytest.raises(ValueError):
            chain.warm_up(batch_capacity=1, **kw)
    with pytest.raises(TypeError, match="all_greedy"):
        ordinary.warm_up(1, all_greedy=1)


def test_symbolic_cache_key_and_capacity_groups(monkeypatch):
    provider = stub_provider(monkeypatch)
    op = RequestLocalChainSampling("cpu", 64, 4097)
    op._provider = provider
    keys = op.warm_up(batch_capacity=1024, dtypes=(torch.float32,), modes=())
    assert len(keys) == len(provider.compiled) == 13  # 2 identity groups, 4 filtered hints, min-p.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for b, length in [(1, 1), (3, 9), (191, 2), (256, 1), (1024, 1)]:
        # Metadata verifies dispatch/key reuse without running the reference at large B/V.
        from ctikernels.sampling._dflash import _warmup

        for mode, inputs, hint in _warmup.probability_variants(
            RequestLocalChainSampling("cpu", length, 4097),
            b,
            (torch.float32,),
            HINTS,
            (False, True),
        ):
            kernel, tensors, position, _ = provider._prepare(mode, inputs, hint, 48)
            key = compile_key(mode, position, kernel, tensors, op.device)
            provider._get(key, kernel, tensors)
    assert len(provider.compiled) == 13  # the stub key omits the dense/strided flag


def test_ordinary_cache_ignores_loop_pool_and_row_stride(monkeypatch):
    from ctikernels.sampling._dflash import _warmup

    provider = stub_provider(monkeypatch)
    provider.kernels.greedy_group = provider.kernels.identity_group
    provider.kernels.OrdinaryIdentity = lambda vocab, batch, group, domain, **kw: KernelStub(
        ("identity", (vocab, group, domain), kw)
    )
    provider.kernels.OrdinaryFiltered = lambda vocab, batch, hint, domain, **kw: KernelStub(
        ("filtered", (vocab, hint, domain), kw)
    )
    op = RequestLocalSampling("cpu", 151936)
    op._provider = provider
    keys = op.warm_up(1024, dtypes=(torch.float32,))
    # 14 unlooped variants plus the looped identity draw (both min-p presences)
    # that large B selects; the loop count itself stays a runtime value. Dense and
    # row-strided logits double that (32), and top-k filtered draws above one row
    # per SM drop the compact reservation (+12: 3 hints x 2 min-p x 2 layouts).
    assert len(keys) == len(provider.compiled) == 44
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    loops = set()
    for b in (1, 4, 5, 8, 191, 1024):
        for mode, inputs, hint in _warmup.ordinary_variants(op, b, (torch.float32,), HINTS, True):
            kernel, tensors, position, scalars = provider._prepare(mode, inputs, hint, 48)
            tensors = list(tensors)
            tensors[0] = torch.empty_strided(
                (b, op.vocab_size), (op.vocab_size + b, 1), device="meta"
            )
            seed_index = 4 if mode == "ordinary_identity" else 6
            tensors[seed_index] = _warmup.tensor(torch.uint32, b + 71, 2)
            key = compile_key(mode, position, kernel, tensors, op.device)
            provider._get(key, kernel, tensors)
            loops.update(scalars)
    assert len(loops) > 1
    assert len(provider.compiled) == 44
