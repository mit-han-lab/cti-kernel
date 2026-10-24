# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only host plans, scratch ABI and cached tuple keys."""

import inspect
import pickle
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _cute_prefill as host
from ctikernels.gdn import _triton_prefill, inference
from ctikernels.gdn._contract import Geometry, HashedTuple
from ctikernels.gdn.triton.prefill import Prefill


@pytest.fixture
def launcher_factory(monkeypatch):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=48),
    )

    def make(geometry=(16, 32), cc=(12, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cc)
        return host.Launcher(torch.device("cpu"), object(), geometry)

    return make


@pytest.mark.parametrize("items", [(), (1, "cute", (16, 32)), (None, False, 64)])
def test_hashed_tuple_matches_tuple_and_pickles_as_plain_tuple(items):
    key = HashedTuple(iter(items))
    assert key == items and items == key
    assert hash(key) == hash(items)
    assert {key: "cached"}[items] == "cached"
    assert {items: "plain"}[key] == "plain"
    assert len({items, key, HashedTuple(items)}) == 1
    assert key in {items} and items in {key}
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(key, protocol=protocol))
        assert type(restored) is tuple
        assert restored == items


def _tune_cases():
    for cc, geometry in host.GEOMETRY_TUNES:
        modes = ["default", "aq_qk=0", "aq_qk=1"]
        modes += [f"row-{i}" for i in range(len(host.ROW_TUNES.get((cc, geometry), ())))]
        modes += [f"o_split=1,bvo={bvo}" for bvo in (32, 64, 128)]
        modes += ["v2=1", "v2=1,v2_snorm=1", "o_form=pre", "form=tk,t_store=fp32"]
        for mode in modes:
            yield pytest.param(cc, geometry, mode, id=f"sm{cc[0]}{cc[1]}-{geometry}-{mode}")


def _tuned_launcher(monkeypatch, factory, cc, geometry, mode):
    if mode.startswith("row-"):
        # Exercise EACH row's allocation contract at every T/N, even outside its band.
        row = host.ROW_TUNES[(cc, geometry)][int(mode[4:])]
        monkeypatch.setitem(host.ROW_TUNES, (cc, geometry), ((None, None, None, row[-1]),))
    elif mode == "default":
        monkeypatch.setitem(host.ROW_TUNES, (cc, geometry), ())
    else:
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", mode)
    return factory(geometry, cc)


def _arguments(geometry, tokens, count, stride=None):
    def tensor(*shape, dtype=torch.float32):
        return torch.empty(*shape, dtype=dtype, device="meta")

    projected = tensor(tokens, stride or geometry.proj_width, dtype=torch.bfloat16)
    return (
        projected[:, : geometry.proj_width],
        tensor(geometry.conv_dim, 4, dtype=torch.bfloat16),
        tensor(geometry.v_heads),
        tensor(geometry.v_heads),
        tensor(128),
        tensor(count + 1, dtype=torch.int32),
        tensor(count, dtype=torch.int32),
        tensor(count, dtype=torch.bool),
        tensor(count, geometry.v_heads, 128, 128),
        tensor(count, 3, geometry.conv_dim, dtype=torch.bfloat16),
    )


# Legacy allocation order differs from the executable's SCRATCH argument order.
_ALLOCATIONS = (
    "kk",
    "qk",
    "q",
    "k",
    "w",
    "u",
    "kl",
    "aq",
    "qp",
    "op",
    "g",
    "history_out",
    "raw_out",
    "partial",
    "hbuf",
    "vbuf",
    "y",
    "flags",
)
_ARGUMENTS = {
    "q": 10,
    "k": 11,
    "w": 12,
    "u": 13,
    "aq": 14,
    "g": 15,
    "history_out": 16,
    "raw_out": 17,
    "partial": 18,
    "y": 19,
    "kk": 20,
    "qk": 21,
    "kl": 22,
    "flags": 23,
    "qp": 24,
    "op": 25,
    "hbuf": 28,
    "vbuf": 29,
}


class _RecordingTorch:
    """Patch only host.torch: test tensor construction keeps the real torch.empty."""

    def __init__(self):
        self.allocations = []

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        tensor = torch.empty(*args, **(kwargs | {"device": "meta"}))
        self.allocations.append(tensor)
        return tensor


def _check_layout(plan, sizes):
    assert tuple(name for name, *_ in plan.layout) == host.SCRATCH
    end = 0
    for name, offset, numel, dtype in plan.layout:
        assert offset % 256 == 0
        assert offset >= end
        assert (numel, dtype) == sizes[name]
        end = offset + numel * dtype.itemsize
        assert end <= plan.nbytes
    assert plan.nbytes % 256 == 0
    assert plan.nbytes - end < 256


@pytest.mark.parametrize("cc,heads,mode", list(_tune_cases()))
@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 600, 4096])
@pytest.mark.parametrize("count", [1, 3])
def test_scratch_layout_matches_legacy_allocations_and_plan(
    monkeypatch, launcher_factory, cc, heads, mode, tokens, count
):
    launcher = _tuned_launcher(monkeypatch, launcher_factory, cc, heads, mode)
    geometry = Geometry(*heads)
    args = _arguments(geometry, tokens, count)
    key = launcher.compile_key(geometry, args[0], count)
    calls = []
    launcher.compiled[key] = lambda *a: calls.append(a)
    tune, _ = launcher._row_tune(tokens, count)
    sizes = dict(
        zip(host.SCRATCH, host.scratch_layout(geometry, tune, tokens, count, launcher), strict=True)
    )
    recorder = _RecordingTorch()
    with monkeypatch.context() as patch:
        patch.setattr(host, "torch", recorder)
        result = launcher(geometry, *args)
    expected = sizes | {"y": (tokens * geometry.value_width, torch.bfloat16)}
    assert [(t.numel(), t.dtype) for t in recorder.allocations] == [
        expected[n] for n in _ALLOCATIONS
    ]
    (launched,) = calls
    assert len(launched) == 30
    # Check both sizes and identity through the ABI, including empty placeholders.
    for name, allocated in zip(_ALLOCATIONS, recorder.allocations, strict=True):
        argument = launched[_ARGUMENTS[name]]
        assert (argument.numel(), argument.dtype) == expected[name]
        assert argument is allocated or argument._base is allocated
    assert result is recorder.allocations[_ALLOCATIONS.index("y")]
    plan = launcher.plan(geometry, tokens, count, args[0].stride(0))
    assert plan.key == key
    _check_layout(plan, sizes)


@pytest.mark.parametrize("output", ["aligned", "misaligned", "none"])
def test_legacy_output_allocation_order(monkeypatch, launcher_factory, output):
    geometry = Geometry(16, 32)
    launcher = launcher_factory()
    args = _arguments(geometry, 1, 1)
    storage = torch.empty(geometry.value_width + 1, dtype=torch.bfloat16)
    offset = int(output == "misaligned")
    out = None if output == "none" else storage[offset : offset + geometry.value_width].view(1, -1)
    calls = []
    launcher.compiled[launcher.compile_key(geometry, args[0], 1)] = lambda *a: calls.append(a)
    # The fake executable writes nothing; suppress only the meta-to-CPU copy-back.
    copies = []
    recorder = _RecordingTorch()
    with monkeypatch.context() as patch:
        patch.setattr(host, "torch", recorder)
        patch.setattr(torch.Tensor, "copy_", lambda dst, src: copies.append((dst, src)))
        result = launcher(geometry, *args, out=out)
    names = [n for n in _ALLOCATIONS if n != "y" or output != "aligned"]
    assert len(recorder.allocations) == len(names)
    for name, tensor in zip(names, recorder.allocations, strict=True):
        argument = calls[0][_ARGUMENTS[name]]
        assert argument is tensor or argument._base is tensor
    assert result is (calls[0][19]._base if out is None else out)
    assert len(copies) == int(output == "misaligned")
    if copies:
        assert copies[0][0] is out
        assert copies[0][1] is recorder.allocations[names.index("y")]


def test_plans_are_shared_across_launchers_and_keyed_by_layout(launcher_factory):
    geometry = Geometry(16, 32)
    first, second = launcher_factory(), launcher_factory()
    assert first is not second
    assert first._policy == second._policy
    plan = first.plan(geometry, 65, 3, geometry.proj_width)
    assert first.plan(geometry, 65, 3, geometry.proj_width) is plan
    assert second.plan(geometry, 65, 3, geometry.proj_width) is plan
    assert second.plan(geometry, 65, 3, geometry.proj_width + 8) is not plan
    assert second.plan(geometry, 64, 3, geometry.proj_width) is not plan
    assert second.plan(geometry, 65, 1, geometry.proj_width) is not plan
    # One plan serves both calls: the capturing executable is the key's twin.
    assert plan.key[-1] == 0 and plan.capture_key == (*plan.key[:-1], 1)


@pytest.mark.parametrize("invalid", ["captures", "grid-x", "grid-y"])
def test_plan_and_legacy_reject_the_same_invalid_calls(monkeypatch, launcher_factory, invalid):
    heads, tokens, count = (16, 32), 65, 1
    if invalid == "captures":
        # v2 supports captures; merged preparation is an actually unsupported variant.
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "merge_prepare=1")
        message = "capture requires"
    elif invalid == "grid-x":
        count, message = 2**31, "grid.x limit"
    else:
        heads, message = (1, 65536), "grid.y limit"
    geometry = Geometry(*heads)
    launcher = launcher_factory(heads)
    args = _arguments(geometry, tokens, count)
    rows = torch.empty(count, dtype=torch.int32, device="meta") if invalid == "captures" else None
    with pytest.raises(ValueError, match=message) as legacy:
        launcher(geometry, *args, capture_rows=rows)
    with pytest.raises(ValueError, match=message) as planned:
        plan = launcher.plan(geometry, tokens, count, args[0].stride(0))
        # Captures never change the plan: the capturing call raises when it runs.
        assert plan.capture_key is None
        launcher.run(plan, geometry, *args, None, rows, False)
    assert str(planned.value) == str(legacy.value)


def test_no_plan_takes_a_capture_argument():
    for plan in (Prefill.plan, _triton_prefill.TritonPrefill.plan, host.Launcher.plan):
        assert not any("captur" in name for name in inspect.signature(plan).parameters)


@pytest.mark.parametrize("capture", [False, True])
def test_fast_path_calls_the_triton_plan_with_its_signature(monkeypatch, capture):
    """GatedDeltaNetPrefill._call_fast against the real direct-Triton binding's plan()."""
    geometry = Geometry(1, 2)
    signature = inspect.signature(Prefill.plan)
    calls = []

    class Planned(Exception):
        pass

    def plan(self, *args, **kwargs):
        signature.bind(self, *args, **kwargs)  # TypeError on any arity mismatch
        calls.append((args, kwargs))
        raise Planned

    kernel = object.__new__(Prefill)
    monkeypatch.setattr(Prefill, "plan", plan)
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(inference, "_prefill_kernel", lambda *a, **k: kernel)
    op = inference.GatedDeltaNetPrefill(num_k_heads=1, num_v_heads=2, device="cpu")
    assert op._fast
    args = [
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype) for t in _arguments(geometry, 5, 2)
    ]
    rows = torch.full((2,), -1, dtype=torch.int32) if capture else None
    with pytest.raises(Planned):
        op(*args, capture_rows=rows)
    assert len(calls) == 1
    ((actual, keywords),) = calls
    assert len(actual) == 2 and actual[0] is args[0] and actual[1] == 2 and keywords == {}
