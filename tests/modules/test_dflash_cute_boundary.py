# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Exercise the real Python/CuTe boundary on CPU, without importing CUDA kernels."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.cute import Provider

from .test_dflash_sampling_reference import buffers, sparsify, verify


class BorrowedStorageOnly(TorchDispatchMode):
    def __init__(self, tensors):
        super().__init__()
        self.pointers = {t.untyped_storage().data_ptr() for t in tensors}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, "host readback"
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor):
                assert value.untyped_storage().data_ptr() in self.pointers, "new storage"
        return result


@pytest.mark.parametrize(
    "mode", ["dense", "sparse", "point", "sample", "candidates", "cursors", "probabilities"]
)
@pytest.mark.parametrize("gamma", [pytest.param(0, id="q1"), 2])
@pytest.mark.parametrize("batch", [0, 1, 3])
@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_arguments_and_capture_cache(monkeypatch, mode, batch, empty_first, gamma):
    op, data = buffers(batch, gamma, 3)
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.zeros(1, 2, dtype=torch.uint32)
    empty_seeds = torch.empty(0, 2, dtype=torch.uint32)
    prompts = torch.zeros(1, dtype=torch.int64)
    empty_prompts = torch.empty(0, dtype=torch.int64)
    anchors = torch.full((batch,), -999, dtype=torch.int64)
    if mode == "sparse":
        sparsify(data)
    if mode == "candidates":
        op = RequestLocalChainSampling("cpu", gamma + 1, 16)
        data = {
            **{k: data[k] for k in ("temperatures", "request_indices", "seeds", "cursors")},
            "scores": torch.full((batch, 16), float("nan")),
            "ids": torch.full((batch, 16), -999, dtype=torch.int32),
            "out_ids": torch.empty(batch, dtype=torch.int32),
            "out_q": torch.empty(batch, 16),
            "workspace": torch.empty(0, dtype=torch.uint8),
        }
    if mode == "probabilities":
        from .test_dflash_probabilities import probability_data

        data = probability_data(torch.full((batch, gamma + 1, 3), float("nan")))
        data["request_indices"].fill_(-1)
        data["seeds"] = torch.zeros(1, 2, dtype=torch.uint32)
    from .test_dflash_symb import KernelStub

    provider = Provider.__new__(Provider)
    provider.kernels = SimpleNamespace(
        Chain=lambda *args, **kwargs: KernelStub((mode, args)),
        Candidates=lambda *args, **kwargs: KernelStub((mode, args)),
        Cursors=lambda: KernelStub((mode, ())),
        Probabilities=lambda *args, **kwargs: KernelStub((mode, args)),
        identity_group=lambda rows, vocab, sms: 1,
    )
    provider.compiled = {}
    op._provider = provider
    capturing = False
    compiled_shapes, launches = [], []

    def compile_stub(kernel, tensors):
        assert not capturing, "compilation during capture"
        assert all(t.numel() > 0 for t in tensors), "zero extent reached CuTe compilation"
        shapes = [(t.shape, t.stride(), t.dtype) for t in tensors]
        compiled_shapes.append(shapes)

        def launch(*args):
            assert all(t.numel() > 0 for t in args), "zero extent reached CuTe launch"
            assert [(t.shape, t.stride(), t.dtype) for t in args] == shapes
            launches.append(kernel)

        return launch

    monkeypatch.setattr(provider, "_compile", compile_stub)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=48)
    )

    def call():
        if mode in ("dense", "sparse", "point"):
            verify(op, data, mode == "sparse", point=mode == "point")
        elif mode == "sample":
            op.sample_dense_proposals(
                data["q"],
                data["temperatures"],
                **{k: data[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
                out_ids=data["drafts"],
            )
        elif mode == "candidates":
            op.sample_candidates(position_i=0, **data)
        elif mode == "probabilities":
            op.probabilities(**{k: v for k, v in data.items() if k != "seeds"})
        else:
            op.prepare_cursors(
                anchors,
                prompts,
                request_indices=data["request_indices"],
                workspace=data["workspace"],
                out_cursors=data["cursors"],
            )

    full_seeds, full_prompts = data["seeds"], prompts
    borrowed = [*data.values(), empty_seeds, prompts, empty_prompts, anchors]
    if empty_first:
        data["seeds"], prompts = empty_seeds, empty_prompts
    with BorrowedStorageOnly(borrowed):
        call()
        # S=0 and S=1 share a positive dummy layout; no new cache entry or storage.
        data["seeds"] = full_seeds if empty_first else empty_seeds
        prompts = full_prompts if empty_first else empty_prompts
        capturing = True
        for _ in range(3):
            call()
    launches_expected = bool(batch) and (gamma > 0 or mode not in ("sample", "candidates"))
    assert len(compiled_shapes) == int(launches_expected)
    assert len(launches) == 4 * launches_expected


def test_probability_hints_have_separate_capture_cache(monkeypatch):
    from .test_dflash_probabilities import probability_data

    op = RequestLocalChainSampling("cpu", 2, 4097)
    data = probability_data(torch.zeros(1, 1, 4097))
    from .test_dflash_symb import KernelStub

    provider = Provider.__new__(Provider)
    provider.kernels = SimpleNamespace(
        Probabilities=lambda *args, **kw: KernelStub(args),
        identity_group=lambda rows, vocab, sms: 2,
    )
    provider.compiled = {}
    op._provider = provider
    capturing = False
    compiled_hints, launched_hints = [], []

    def compile_stub(kernel, tensors):
        assert not capturing
        length, vocab, hint, batch, group = kernel
        assert (length, vocab, batch, group) == (1, 4097, 1, 2 if hint == "none" else 1)
        assert tensors[-1] is data["workspace"]
        compiled_hints.append(hint)
        return lambda *args: launched_hints.append(hint)

    monkeypatch.setattr(provider, "_compile", compile_stub)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=48)
    )
    hints = ["none", "top_k", "top_p", "both", "auto"]
    with BorrowedStorageOnly(list(data.values())):
        for hint in hints:
            op.probabilities(**data, filters=hint)
        capturing = True
        for hint in hints:
            op.probabilities(**data, filters=hint)
    assert compiled_hints == hints
    assert launched_hints == hints * 2
