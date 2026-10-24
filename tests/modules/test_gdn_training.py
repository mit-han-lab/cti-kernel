# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn training on the CPU: the contract and autograd node with a recording kernel,
the provider composition with stand-in entry points and the PyTorch reference of its Triton
parts, and that reference against FP64 (the GPU tests hold the Triton parts to fla-core and
FP64)."""

from __future__ import annotations

import functools
import gc
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ctikernels.gdn import GatedDeltaNetTraining, _contract, _cute, _fla, training

from . import _gdn_training_reference as reference
from ._gdn_provider import BINDINGS, ENTRIES, fake_fla, fake_kernels

K = V = 128
WEIGHTS = ("conv_weight", "A_log", "dt_bias", "norm_weight")
GEOMETRY = _contract.Geometry(1, 2)  # conv_dim 512, P 772, Hv V 256
LONGEST = 64  # the longest sequence _arguments packs


class _Recorder:
    """A training kernel that records its calls; it saves two weakly watched tensors and returns
    y as a view, as the provider does, and a fixed gradient."""

    def __init__(self) -> None:
        self.forwards: list[tuple] = []
        self.backwards: list[tuple] = []
        self.saved: list[weakref.ref] = []

    def train_forward(self, geometry, projected, *rest):
        self.forwards.append((geometry, projected, *rest))
        saved = (projected, torch.ones(3), torch.arange(4))
        self.saved.extend(weakref.ref(tensor) for tensor in saved[1:])
        rows = projected.shape[0]
        y = torch.full((rows * geometry.v_heads, V), 0.5, dtype=torch.bfloat16)
        return y.view(rows, geometry.value_width), saved

    def train_backward(self, geometry, saved, dy):
        self.backwards.append((geometry, [tensor.data_ptr() for tensor in saved], dy))
        return torch.full(saved[0].shape, 2.0, dtype=torch.bfloat16)

    def alive(self) -> list[bool]:
        gc.collect()
        return [ref() is not None for ref in self.saved]


@pytest.fixture
def kernel(monkeypatch) -> _Recorder:
    """The operation bound on the CPU to a recording kernel, resolved once per construction."""
    recorder, resolutions = _Recorder(), []

    def resolve(geometry, device):
        resolutions.append((geometry, device))
        return recorder

    monkeypatch.setattr(training, "cuda_device", lambda operation, device: torch.device("cpu"))
    monkeypatch.setattr(training, "_training_kernel", resolve)
    recorder.resolutions = resolutions
    return recorder


def _op() -> GatedDeltaNetTraining:
    return GatedDeltaNetTraining(
        num_k_heads=GEOMETRY.k_heads, num_v_heads=GEOMETRY.v_heads, device="cuda:0"
    )


def _arguments(tokens: int = 70, lengths=(1, LONGEST, 5)) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    assert int(cu[-1]) == tokens
    return {
        "projected": torch.randn(tokens, GEOMETRY.proj_width, generator=generator).bfloat16(),
        "conv_weight": torch.randn(GEOMETRY.conv_dim, 4, generator=generator).bfloat16(),
        "A_log": torch.randn(GEOMETRY.v_heads, generator=generator),
        "dt_bias": torch.randn(GEOMETRY.v_heads, generator=generator),
        "norm_weight": torch.randn(V, generator=generator),
        "cu_seqlens": cu,
    }


def test_calls_reach_the_kernel_resolved_at_construction_with_checked_inputs(kernel) -> None:
    op = _op()
    arguments = _arguments()
    for _ in range(3):
        y = op(*arguments.values(), max_sequence_length=LONGEST)
        assert y.shape == (70, GEOMETRY.value_width) and y.grad_fn is None
    assert kernel.resolutions == [(GEOMETRY, torch.device("cpu"))]
    assert len(kernel.forwards) == 3 and kernel.backwards == []
    for call in kernel.forwards:
        assert call[0] == GEOMETRY
        assert all(sent is given for sent, given in zip(call[1:], arguments.values(), strict=True))
    assert (op.geometry, op.device) == (GEOMETRY, torch.device("cpu"))


@pytest.mark.parametrize("cute_device", [0, 1])
@pytest.mark.parametrize("triton", [True, False])
def test_training_selects_its_provider_per_device_at_construction(monkeypatch, cute_device, triton):
    """CuTe where it resolves, else Triton within its capabilities, else fla-core."""
    cute, ours, fla = _Recorder(), _Recorder(), _Recorder()
    cute.provider, ours.provider = "cute", "triton"
    resolutions, fallbacks = [], []

    def resolve(device):
        resolutions.append(device)
        return cute if device.index == cute_device else None

    def fallback(operation, device):
        fallbacks.append((operation, device))
        return fla

    monkeypatch.setattr(training, "cuda_device", lambda operation, device: torch.device(device))
    # A card without Triton measurements (not in TRITON_FASTER), where CuTe is the pick.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 5080")
    monkeypatch.setattr(_cute, "training", resolve)
    monkeypatch.setattr(training, "_triton", lambda device: ours if triton else None)
    monkeypatch.setattr(_fla, "load", fallback)
    devices = [torch.device("cuda", i) for i in range(2)]
    ops = [GatedDeltaNetTraining(num_k_heads=16, num_v_heads=32, device=d) for d in devices]
    assert resolutions == devices
    other = devices[1 - cute_device]
    assert fallbacks == ([] if triton else [("GatedDeltaNetTraining", other)])
    assert ops[cute_device].provider == "cute"
    assert ops[1 - cute_device].provider == ("triton" if triton else "fla-core")
    # Resolution is fixed on the operation.
    monkeypatch.setattr(_cute, "training", lambda device: None)
    assert ops[cute_device].provider == "cute"


@pytest.mark.parametrize(
    ("capability", "provider"),
    [((9, 0), "fla-core"), ((10, 0), "triton"), ((10, 3), "triton"), ((12, 1), "triton"),
     ((12, 2), "fla-core")],
)  # fmt: skip
def test_triton_covers_one_interval_of_compute_capabilities(monkeypatch, capability, provider):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn.triton.training", _FakeTriton)
    kernel = training._triton(torch.device("cuda", 0))
    assert (kernel is not None) == (provider == "triton")


class _FakeTriton:
    """Stands in for gdn.triton.training (Triton is absent on the CPU)."""

    class Training:
        provider = "triton"

        def __init__(self, capability):
            self.capability = capability


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA GeForce RTX 5080", "NVIDIA GB10", "NVIDIA RTX PRO 6000 Blackwell Server Edition"],
)
@pytest.mark.parametrize(
    ("k_heads", "v_heads"), [(16, 32), (16, 48), (8, 16), (8, 24), (4, 8), (4, 12)]
)
def test_training_selects_its_provider_per_geometry_and_card(
    monkeypatch, device_name, k_heads, v_heads
):
    """Where CuTe resolves, it runs at CUTE_GEOMETRIES except where Triton measured faster on
    the card (TRITON_FASTER); elsewhere Triton runs, without resolving CuTe."""
    cute, ours = _Recorder(), _Recorder()
    cute.provider, ours.provider = "cute", "triton"
    resolutions = []

    def resolve(device):
        resolutions.append(device)
        return cute

    monkeypatch.setattr(training, "cuda_device", lambda operation, device: torch.device(device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: device_name)
    monkeypatch.setattr(_cute, "training", resolve)
    monkeypatch.setattr(training, "_triton", lambda device: ours)
    geometry = _contract.Geometry(k_heads, v_heads)
    expected = geometry in training.CUTE_GEOMETRIES and geometry not in training.TRITON_FASTER.get(
        device_name, frozenset()
    )
    op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda:0")
    assert op.provider == ("cute" if expected else "triton")
    assert len(resolutions) == expected


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("grad_mode", [True, False])
@pytest.mark.parametrize("projected_requires_grad", [True, False])
def test_a_weight_that_requires_grad_is_rejected_before_any_other_check(
    weight, grad_mode, projected_requires_grad, kernel
) -> None:
    op = _op()
    arguments = _arguments()
    arguments["projected"].requires_grad_(projected_requires_grad)
    # Every other argument is wrong too: the frozen rule must be the first check.
    broken = dict.fromkeys(arguments)
    broken[weight] = torch.zeros(1, requires_grad=True)
    with torch.set_grad_enabled(grad_mode):
        with pytest.raises(
            ValueError, match=f"GatedDeltaNetTraining: {weight} is a frozen base weight"
        ):
            op(*broken.values(), max_sequence_length=None)
        arguments[weight] = arguments[weight].clone().requires_grad_(True)
        with pytest.raises(ValueError, match=f"{weight} is a frozen base weight"):
            op(*arguments.values(), max_sequence_length=LONGEST)
    assert kernel.forwards == [] and kernel.backwards == []


def _strided(shape, dtype) -> torch.Tensor:
    return torch.zeros(shape[1], shape[0], dtype=dtype).t()


P, DC, HV = GEOMETRY.proj_width, GEOMETRY.conv_dim, GEOMETRY.v_heads
VIOLATIONS = {
    "projected dtype": ("projected", lambda: torch.zeros(70, P, dtype=torch.float16), TypeError),
    "projected width": ("projected", lambda: torch.zeros(70, P - 1, dtype=torch.bfloat16), ValueError),
    "projected rank": ("projected", lambda: torch.zeros(70, 1, P, dtype=torch.bfloat16), ValueError),
    "projected column stride": ("projected", lambda: _strided((70, P), torch.bfloat16), ValueError),
    "projected not a tensor": ("projected", lambda: [0.0], TypeError),
    "projected device": ("projected", lambda: torch.zeros(70, P, dtype=torch.bfloat16, device="meta"), ValueError),
    "conv_weight dtype": ("conv_weight", lambda: torch.zeros(DC, 4), TypeError),
    "conv_weight checkpoint shape": ("conv_weight", lambda: torch.zeros(DC, 1, 4, dtype=torch.bfloat16), ValueError),
    "conv_weight strided": ("conv_weight", lambda: _strided((DC, 4), torch.bfloat16), ValueError),
    "A_log BF16": ("A_log", lambda: torch.zeros(HV, dtype=torch.bfloat16), TypeError),
    "A_log heads": ("A_log", lambda: torch.zeros(HV + 1), ValueError),
    "dt_bias BF16": ("dt_bias", lambda: torch.zeros(HV, dtype=torch.bfloat16), TypeError),
    "norm_weight width": ("norm_weight", lambda: torch.zeros(2 * V), ValueError),
    "norm_weight BF16": ("norm_weight", lambda: torch.zeros(V, dtype=torch.bfloat16), TypeError),
    "cu_seqlens dtype": ("cu_seqlens", lambda: torch.tensor([0, 1, 65, 70]), TypeError),
    "cu_seqlens rank": ("cu_seqlens", lambda: torch.zeros(4, 1, dtype=torch.int32), ValueError),
    "cu_seqlens without a sequence": ("cu_seqlens", lambda: torch.zeros(1, dtype=torch.int32), ValueError),
    "more sequences than rows": ("cu_seqlens", lambda: torch.zeros(72, dtype=torch.int32), ValueError),
    "cu_seqlens not a tensor": ("cu_seqlens", lambda: [0, 1, 65, 70], TypeError),
    "cu_seqlens device": ("cu_seqlens", lambda: torch.zeros(4, dtype=torch.int32, device="meta"), ValueError),
}  # fmt: skip


@pytest.mark.parametrize("case", list(VIOLATIONS))
def test_training_checks_its_contract_before_any_kernel(case, kernel) -> None:
    name, value, error = VIOLATIONS[case]
    op = _op()
    arguments = {**_arguments(), name: value()}
    with pytest.raises(error):
        op(*arguments.values(), max_sequence_length=LONGEST)
    assert kernel.forwards == []


@pytest.mark.parametrize(
    ("value", "error"),
    [(None, TypeError), (64.0, TypeError), (True, TypeError), (0, ValueError), (71, ValueError)],
)
def test_max_sequence_length_is_a_host_integer_within_the_tokens(value, error, kernel) -> None:
    op = _op()
    with pytest.raises(error, match="max_sequence_length"):
        op(*_arguments().values(), max_sequence_length=value)
    assert kernel.forwards == []


def test_training_only_bound_is_kept_until_the_provider_audit(kernel, monkeypatch) -> None:
    """The longest sequence times projected's row stride must stay below 2^31; meta tensors
    carry such a stride without its memory."""
    monkeypatch.setattr(training, "cuda_device", lambda operation, device: torch.device("meta"))
    op = _op()
    arguments = {name: value.to("meta") for name, value in _arguments().items()}
    for stride, accepted in ((2**31 // LONGEST - 1, True), (2**31 // LONGEST, False)):
        projected = torch.empty_strided((70, P), (stride, 1), dtype=torch.bfloat16, device="meta")
        call = {**arguments, "projected": projected}
        if accepted:
            op(*call.values(), max_sequence_length=LONGEST)
        else:
            with pytest.raises(
                ValueError, match="a training-only bound kept until the provider audit"
            ):
                op(*call.values(), max_sequence_length=LONGEST)
            # The bound is the longest sequence, not the token count.
            op(*call.values(), max_sequence_length=LONGEST - 1)
    assert len(kernel.forwards) == 2


def test_projected_may_be_a_row_strided_view(kernel) -> None:
    op = _op()
    arguments = _arguments()
    wide = torch.zeros(70, P + 64, dtype=torch.bfloat16)
    op(*{**arguments, "projected": wide[:, :P]}.values(), max_sequence_length=LONGEST)
    overlapping = torch.zeros(70 * P, dtype=torch.bfloat16).as_strided((70, P), (100, 1))
    with pytest.raises(ValueError, match="non-overlapping rows"):
        op(*{**arguments, "projected": overlapping}.values(), max_sequence_length=LONGEST)
    assert len(kernel.forwards) == 1


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode", "projected without grad"])
def test_a_call_without_autograd_runs_the_forward_alone_and_keeps_nothing(mode, kernel) -> None:
    op = _op()
    arguments = _arguments()
    if mode == "no_grad":
        arguments["projected"].requires_grad_(True)
        context = torch.no_grad()
    elif mode == "inference_mode":
        arguments["projected"].requires_grad_(True)
        context = torch.inference_mode()
    else:
        context = torch.enable_grad()
    with context:
        y = op(*arguments.values(), max_sequence_length=LONGEST)
    assert y.grad_fn is None and not y.requires_grad
    assert len(kernel.forwards) == 1 and kernel.backwards == []
    assert kernel.alive() == [False, False]


def test_the_autograd_node_returns_the_kernels_gradient_for_projected_only(kernel) -> None:
    op = _op()
    arguments = _arguments()
    projected = arguments["projected"].requires_grad_(True)
    y = op(*arguments.values(), max_sequence_length=LONGEST)
    assert y.grad_fn is not None and len(kernel.forwards) == 1
    assert kernel.alive() == [True, True]  # kept for the backward, through autograd
    saved = [ref() for ref in kernel.saved]
    expected = [projected.data_ptr(), *(tensor.data_ptr() for tensor in saved)]
    del saved
    dy = torch.randn(GEOMETRY.value_width, 70).bfloat16().t()  # not contiguous
    (grad,) = torch.autograd.grad(y, projected, dy)
    assert torch.equal(grad, torch.full((70, P), 2.0, dtype=torch.bfloat16))
    [(geometry, pointers, sent)] = kernel.backwards
    assert geometry == GEOMETRY and pointers == expected
    assert sent.is_contiguous() and torch.equal(sent, dy)
    assert kernel.alive() == [False, False]  # released by the backward
    for name in WEIGHTS:
        assert arguments[name].grad is None


def test_the_result_may_be_modified_in_place_under_autograd(kernel) -> None:
    """The kernel's y is a view; the node's result is not, so an in-place op on it is recorded
    like any other and its gradient reaches the kernel's backward."""
    op = _op()
    arguments = _arguments()
    projected = arguments["projected"].requires_grad_(True)
    y = op(*arguments.values(), max_sequence_length=LONGEST)
    assert not y._is_view()
    y.mul_(3)
    y.float().sum().backward()
    [(_, _, sent)] = kernel.backwards
    assert torch.equal(sent, torch.full((70, GEOMETRY.value_width), 3.0, dtype=torch.bfloat16))
    assert torch.equal(projected.grad, torch.full((70, P), 2.0, dtype=torch.bfloat16))


def test_the_backward_is_not_differentiable_again(kernel) -> None:
    op = _op()
    arguments = _arguments()
    projected = arguments["projected"].requires_grad_(True)
    y = op(*arguments.values(), max_sequence_length=LONGEST)
    dy = torch.ones_like(y, requires_grad=True)
    (grad,) = torch.autograd.grad(y, projected, dy, create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        grad.float().sum().backward()


@pytest.mark.parametrize("reentrant", [False, True])
def test_activation_checkpointing_recomputes_the_forward_for_the_backward(
    reentrant, kernel
) -> None:
    """The first forward keeps nothing (autograd's saved tensors are dropped under
    checkpointing); the backward recomputes it and reads the recomputed tensors."""
    op = _op()
    arguments = _arguments()
    projected = arguments["projected"].requires_grad_(True)
    bounded = functools.partial(op, max_sequence_length=LONGEST)
    y = checkpoint(bounded, *arguments.values(), use_reentrant=reentrant)
    assert len(kernel.forwards) == 1 and kernel.alive() == [False, False]
    y.backward(torch.ones_like(y))
    assert len(kernel.forwards) == 2 and len(kernel.backwards) == 1
    assert torch.equal(projected.grad, torch.full((70, P), 2.0, dtype=torch.bfloat16))


# -- the provider composition with stand-in entry points ----------------------------------


class _Entries:
    """fla-core's entry points as shape-correct stand-ins that record the varlen metadata
    (cu_seqlens, chunk table) every call receives."""

    def __init__(self, geometry: _contract.Geometry, tokens: int) -> None:
        self.geometry, self.tokens = geometry, tokens
        self.metadata: list[tuple[str, object, object]] = []
        self.generator = torch.Generator().manual_seed(1)

    def _random(self, *shape, dtype=torch.float32) -> torch.Tensor:
        return (0.1 * torch.randn(*shape, generator=self.generator)).to(dtype)

    def _record(self, name, kwargs) -> None:
        self.metadata.append((name, kwargs["cu_seqlens"], kwargs["chunk_indices"]))

    def bindings(self) -> dict[str, object]:
        t, hk, hv = self.tokens, self.geometry.k_heads, self.geometry.v_heads
        chunks = lambda kwargs: kwargs["chunk_indices"].shape[0]
        bf16 = torch.bfloat16

        def chunk(**kwargs):
            self._record("chunk", kwargs)
            return (
                -self._random(1, t, hv).abs(),
                self._random(1, t, hv, V, dtype=bf16),
                self._random(1, t, hv, 64, dtype=bf16),
                None,
                None,
                None,
            )

        def gated_norm_fwd(x, **kwargs):
            return x.clone(), None, 1 + self._random(x.shape[0]).abs(), x

        def recompute_w_u(**kwargs):
            self._record("recompute_w_u", kwargs)
            return self._random(1, t, hv, K, dtype=bf16), self._random(1, t, hv, V, dtype=bf16)

        def chunk_h(**kwargs):
            self._record("chunk_h", kwargs)
            h = self._random(1, chunks(kwargs), hv, K, V, dtype=bf16)
            return h, self._random(1, t, hv, V, dtype=bf16), None

        def dv_local(**kwargs):
            self._record("dv_local", kwargs)
            return self._random(1, t, hv, V, dtype=bf16)

        def dhu(**kwargs):
            self._record("dhu", kwargs)
            dh = self._random(1, chunks(kwargs), hv, K, V)
            return dh, None, self._random(1, t, hv, V, dtype=bf16)

        def dqkwg(**kwargs):
            self._record("dqkwg", kwargs)
            dq, dk = (self._random(1, t, hk, K) for _ in range(2))
            return dq, dk, self._random(1, t, hv, K), self._random(1, t, hv)

        def wy_bwd(**kwargs):
            self._record("wy_bwd", kwargs)
            dv = self._random(1, t, hv, V, dtype=bf16)
            return self._random(1, t, hk, K), dv, self._random(1, t, hv), self._random(1, t, hv)

        def local_cumsum(dg, *, chunk_size, reverse, **kwargs):
            self._record("local_cumsum", kwargs)
            assert chunk_size == _fla.CHUNK and reverse
            return dg.clone()

        def l2norm(x):
            return x, 1 + self._random(*x.shape[:-1]).abs()

        def norm_kernel(o, z, weight, rstd, dy, do, dz):
            assert o.shape == z.shape == dy.shape == do.shape == dz.shape
            do.zero_()
            dz.copy_(z)

        return {
            "conv_update": None,
            "recurrent": None,
            "gate_cumsum": None,
            "intra": None,
            "state_kernel": None,
            "output": None,
            "training_chunk": chunk,
            "l2norm": l2norm,
            "beta_sigmoid": lambda b: torch.sigmoid(b.float()),
            "gated_norm": None,
            "gated_norm_fwd": gated_norm_fwd,
            "recompute_w_u": recompute_w_u,
            "chunk_h": chunk_h,
            "dv_local": dv_local,
            "dhu": dhu,
            "dqkwg": dqkwg,
            "wy_bwd": wy_bwd,
            "l2norm_bwd": lambda x, rstd, dx: dx.to(x.dtype),
            "beta_sigmoid_bwd": lambda b, dbeta: dbeta,
            "local_cumsum": local_cumsum,
            "norm_kernel": norm_kernel,
        }


def _core(entries: dict[str, object], **replaced) -> _fla.FlaCore:
    """FlaCore over stand-in entry points, with the PyTorch reference of its Triton parts."""
    norm_kernel = entries.pop("norm_kernel")
    core = _fla.FlaCore(**{**entries, **replaced})
    core._gated_norm_bwd = norm_kernel

    core._sequence_index = reference.sequence_index
    core._conv_fwd, core._conv_bwd = reference.conv_forward, reference.conv_backward
    core._gate_gradient = reference.gate_gradient
    return core


def _cute_core(monkeypatch, stand_ins):
    """The real CuTe composition over fake imports and a CPU-only chunk stand-in."""
    fake_fla(monkeypatch, dispatch_disabled=True)
    entries = stand_ins.bindings()
    for module, names in ENTRIES.items():
        for name in names:
            monkeypatch.setattr(sys.modules[module], name, entries[BINDINGS[name][1:]])
    from ctikernels.gdn import _fla_fixed, _fla_training, _gated_norm, _l2norm

    monkeypatch.setattr(_fla_fixed, "layer_norm_gated_fwd", entries["gated_norm_fwd"])
    monkeypatch.setattr(_fla_fixed, "chunk_local_cumsum", entries["local_cumsum"])
    monkeypatch.setattr(_l2norm, "l2norm_forward", entries["l2norm"])
    monkeypatch.setattr(_l2norm, "l2norm_backward", entries["l2norm_bwd"])

    for name in ("sequence_index", "conv_forward", "conv_backward", "gate_gradient"):
        monkeypatch.setattr(_fla_training, name, getattr(reference, name))
    monkeypatch.setattr(_gated_norm, "gated_rms_norm_backward", entries["norm_kernel"])

    class Chunk:
        def __call__(self, **kwargs):
            return entries["training_chunk"](**kwargs)[:3]

        def backward(self, **kwargs):
            stand_ins._record("cute_backward", kwargs)
            return tuple(torch.zeros_like(kwargs[name]) for name in ("q", "k", "v", "beta", "g"))

    chunk = Chunk()
    monkeypatch.setattr(_cute, "load", lambda device: chunk)
    return _cute.training(torch.device("cpu"))


@pytest.mark.parametrize("provider", ["fla-core", "cute"])
def test_the_composition_hands_every_entry_one_fresh_cu_seqlens_and_its_device_table(
    monkeypatch,
    provider,
) -> None:
    """Every varlen entry point gets one fresh cu_seqlens and its device-built table, so
    fla-core's identity cache only returns this call's metadata (plumbing only)."""
    fake_kernels(monkeypatch)
    geometry, lengths = _contract.Geometry(1, 2), [1, 64, 65, 3]
    tokens = sum(lengths)
    stand_ins = _Entries(geometry, tokens)
    core = (
        _core(stand_ins.bindings())
        if provider == "fla-core"
        else _cute_core(monkeypatch, stand_ins)
    )
    generator = torch.Generator().manual_seed(2)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    projected = torch.randn(tokens, geometry.proj_width, generator=generator).bfloat16()
    weights = (
        torch.randn(geometry.conv_dim, 4, generator=generator).bfloat16(),
        torch.randn(geometry.v_heads, generator=generator),
        torch.randn(geometry.v_heads, generator=generator),
        torch.randn(V, generator=generator),
    )
    y, saved = core.train_forward(geometry, projected, *weights, cu)
    assert y.shape == (tokens, geometry.value_width) and len(saved) == 19
    assert all(isinstance(tensor, torch.Tensor) for tensor in saved)
    dy = torch.randn(tokens, geometry.value_width, generator=generator).bfloat16()
    grad = core.train_backward(geometry, saved, dy)
    assert grad.shape == projected.shape and grad.dtype == torch.bfloat16
    assert torch.isfinite(grad.float()).all()
    metadata = stand_ins.metadata
    expected_calls = [
        "chunk",
        "recompute_w_u",
        "chunk_h",
        "dv_local",
        "dhu",
        "dqkwg",
        "wy_bwd",
        "local_cumsum",
    ]
    if provider == "cute":
        expected_calls = ["chunk", "cute_backward", "local_cumsum"]
    assert [name for name, _, _ in metadata] == expected_calls
    fresh = {id(cu_seqlens) for _, cu_seqlens, _ in metadata}
    tables = {id(table) for _, _, table in metadata}
    assert len(fresh) == 1 and len(tables) == 1
    _, cu_seqlens, table = metadata[0]
    assert cu_seqlens is not cu and torch.equal(cu_seqlens, cu)
    assert torch.equal(table, _fla._chunk_table(cu, tokens))
    # The norm's dz section is the kernel's output, written back through the strided rows.
    z = projected[:, geometry.conv_dim : geometry.conv_dim + geometry.value_width]
    assert torch.equal(grad[:, geometry.conv_dim : geometry.conv_dim + geometry.value_width], z)


# -- the PyTorch reference of the Triton parts --------------------------------------------------


def _cu(lengths) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def test_sequence_ids_give_each_tokens_sequence_and_position() -> None:
    lengths = [1, 3, 64, 2]
    sequence, position = reference.sequence_ids(_cu(lengths), sum(lengths))
    expected = [(n, p) for n, length in enumerate(lengths) for p in range(length)]
    assert list(zip(sequence.tolist(), position.tolist(), strict=True)) == expected


def _sequence_conv(x: torch.Tensor, weight: torch.Tensor, lengths) -> torch.Tensor:
    """Per sequence, from three zero rows: the causal depthwise conv in FP32, taps in order."""
    out, start, taps = [], 0, weight.float()
    for length in lengths:
        window = torch.cat((torch.zeros(3, x.shape[1]), x[start : start + length].float()))
        total = torch.zeros(length, x.shape[1])
        for j in range(4):
            total += taps[:, j] * window[j : j + length]
        out.append(total)
        start += length
    return torch.cat(out)


def test_the_pre_activation_is_the_causal_conv_within_each_sequence() -> None:
    generator = torch.Generator().manual_seed(3)
    lengths = [1, 2, 3, 7, 1]
    x = torch.randn(sum(lengths), 24, generator=generator).bfloat16()
    weight = torch.randn(24, 4, generator=generator).bfloat16()
    pre = reference.pre_activation(x, weight, _cu(lengths))
    assert pre.dtype == torch.float32
    assert torch.equal(pre, _sequence_conv(x, weight, lengths))


def test_the_conv_backward_is_the_conv_input_gradient_within_each_sequence() -> None:
    """d x of sum(dc * silu(conv(x))) against FP64 autograd, per sequence (taps never cross a
    sequence start)."""
    dc = 6
    generator = torch.Generator().manual_seed(4)
    lengths = [1, 70, 2, 65]
    tokens = sum(lengths)
    x = torch.randn(tokens, dc, generator=generator).bfloat16()
    weight = (0.5 * torch.randn(dc, 4, generator=generator)).bfloat16()
    d_mixed = torch.randn(tokens, dc, generator=generator).bfloat16()
    cu = _cu(lengths)
    out = torch.empty(tokens, dc, dtype=torch.bfloat16)
    parts = (d_mixed[:, :2], d_mixed[:, 2:4], d_mixed[:, 4:])
    reference.conv_backward(x, weight, reference.sequence_index(cu, tokens), cu, parts, out)
    leaf = x.double().requires_grad_(True)
    expected, start = [], 0
    for length in lengths:
        rows = slice(start, start + length)
        window = torch.cat((torch.zeros(3, dc, dtype=torch.float64), leaf[rows]))
        conv = sum(weight.double()[:, j] * window[j : j + length] for j in range(4))
        expected.append(F.silu(conv))
        start += length
    (want,) = torch.autograd.grad(torch.cat(expected), leaf, d_mixed.double())
    error = (out.double() - want).abs().max() / want.abs().max()
    assert error < 2e-2, error.item()  # BF16 pre-activation and output
