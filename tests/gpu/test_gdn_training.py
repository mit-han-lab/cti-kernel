# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn training on the GPU, against an FP64 reference and fla-core's autograd."""

from __future__ import annotations

import functools
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from ctikernels.gdn import (
    GatedDeltaNetPrefill,
    GatedDeltaNetTraining,
    _cute,
    _fla,
    _fla_training,
    _gated_norm,
    training,
)

from . import _gdn_reference as reference
from ._gdn_l2norm import pinned_fla_reference
from ._gdn_precision import assert_means, no_worse, no_worse_section

K = V = 128
EPS = 1e-6
# Qwen3.5-9B / 35B-A3B and Qwen3.5-27B per rank at TP1.
GEOMETRIES = [(16, 32), (16, 48)]
# Per rank under tensor parallelism: Qwen3.5-9B / 35B-A3B and Qwen3.5-27B at TP2, then at TP4.
TP_GEOMETRIES = [(8, 16), (8, 24), (4, 8), (4, 12)]
# Several sequences, around fla-core's 64-token chunks.
LENGTHS = [1, 63, 64, 65, 130, 7]
WEIGHTS = ("conv_weight", "A_log", "dt_bias", "norm_weight")
DECAYS = {"ordinary": (-2.0, 2.0), "slow": (-8.0, -6.0)}

# The forward's tolerance is prefill's (tests/gpu/test_gdn.py). The gradients' allow BF16
# activations between fla-core's kernels; a wrong gate, decay, tap, head or section errs by ~1.
Y_TOLERANCE = {"rms": 1e-2, "peak": 3e-2}
GRAD_TOLERANCE = {"rms": 3e-2, "peak": 1e-1}
SEEDS = (1, 2)


@pytest.fixture(autouse=True)
def _fla_provider(monkeypatch):
    """These tests hold the fla-core provider; test_gdn_cute.py holds the CuTe provider and
    test_gdn_triton_training.py ours."""
    monkeypatch.setattr(_cute, "training", lambda device: None)
    monkeypatch.setattr(training, "_triton", lambda device: None)


def _widths(k_heads: int, v_heads: int) -> tuple[int, int]:
    conv_dim = 2 * k_heads * K + v_heads * V
    return conv_dim, conv_dim + v_heads * V + 2 * v_heads


def _sections(grad: torch.Tensor, k_heads: int, v_heads: int) -> dict[str, torch.Tensor]:
    """d projected by section: the conv input's q | k | v columns, then dz, db and da."""
    conv_dim, _ = _widths(k_heads, v_heads)
    key, value = k_heads * K, v_heads * V
    return {
        "dq": grad[:, :key],
        "dk": grad[:, key : 2 * key],
        "dv": grad[:, 2 * key : conv_dim],
        "dz": grad[:, conv_dim : conv_dim + value],
        "db": grad[:, conv_dim + value : conv_dim + value + v_heads],
        "da": grad[:, conv_dim + value + v_heads :],
    }


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """``actual``'s error against ``expected`` relative to its RMS and to its largest magnitude
    (FP64)."""
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    return {
        "rms": (error.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item(),
        "peak": (error.abs().max() / expected.abs().max()).item(),
    }


def _assert_close(actual, expected, *, rms: float, peak: float, label: str = "") -> None:
    assert torch.isfinite(actual.double()).all(), label
    found = _relative(actual, expected)
    assert found["rms"] <= rms and found["peak"] <= peak, (label, found)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The exact bit pattern, including the sign of zero."""
    return tensor.contiguous().view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _cu(lengths) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()


def _inputs(k_heads: int, v_heads: int, seed: int, *, lengths=LENGTHS, decay="ordinary"):
    """Weights, projected, cu_seqlens and dy (all on the GPU), and the longest length."""
    generator = torch.Generator().manual_seed(seed)
    conv_dim, width = _widths(k_heads, v_heads)
    low, high = DECAYS[decay]
    weights = reference.Weights(
        conv_weight=(0.5 * torch.randn(conv_dim, 4, generator=generator)).to(torch.bfloat16).cuda(),
        # A = exp(A_log) in about [0.5, 8], the checkpoints' range.
        A_log=(torch.rand(v_heads, generator=generator) * 2.8 - 0.7).cuda(),
        dt_bias=(low + (high - low) * torch.rand(v_heads, generator=generator)).cuda(),
        norm_weight=(1 + 0.1 * torch.randn(V, generator=generator)).cuda(),
    )
    tokens = sum(lengths)
    projected = torch.randn(tokens, width, generator=generator).to(torch.bfloat16).cuda()
    dy = torch.randn(tokens, v_heads * V, generator=generator).to(torch.bfloat16).cuda()
    return SimpleNamespace(
        weights=weights,
        projected=projected,
        cu_seqlens=_cu(lengths),
        dy=dy,
        max_sequence_length=max(lengths),
    )


@functools.cache
def _reference(k_heads: int, v_heads: int, seed: int, decay: str):
    """The inputs with the FP64 reference's y and d projected."""
    case = _inputs(k_heads, v_heads, seed, decay=decay)
    leaf = case.projected.double().requires_grad_(True)
    y = reference.training(leaf, case.weights, k_heads, case.cu_seqlens)
    (grad,) = torch.autograd.grad(y, leaf, case.dy.double())
    return case, y.detach(), grad


def _training_op(k_heads: int, v_heads: int) -> GatedDeltaNetTraining | None:
    """The operation, or None after asserting its refusal where fla-core computes it wrongly
    (``_fla.WRONG_ON``): the refusal is the caller's expected outcome there, not a skip."""
    name = GatedDeltaNetTraining.__name__
    reason = _fla.WRONG_ON.get((name, torch.cuda.get_device_capability()))
    if reason is None:
        op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
        assert op.provider == "fla-core"
        return op
    with pytest.raises(RuntimeError) as raised:
        GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    device = _fla.describe(torch.device("cuda", torch.cuda.current_device()))
    assert str(raised.value) == (
        f"{name} on {device} is not supported by fla-core {_fla.VERSION}: {reason}"
    )
    return None


def _run(op, case, *, cu_seqlens=None) -> tuple[torch.Tensor, torch.Tensor]:
    """y and d projected through the operation's autograd node."""
    projected = case.projected.clone().requires_grad_(True)
    cu = case.cu_seqlens if cu_seqlens is None else cu_seqlens
    y = op(projected, *case.weights.args(), cu, max_sequence_length=case.max_sequence_length)
    (grad,) = torch.autograd.grad(y, projected, case.dy)
    return y.detach(), grad


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
@pytest.mark.parametrize("decay", list(DECAYS))
def test_training_matches_the_fp64_reference_forward_and_backward(k_heads, v_heads, decay):
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case, expected_y, expected_grad = _reference(k_heads, v_heads, 1, decay)
    y, grad = _run(op, case)
    torch.cuda.synchronize()
    assert y.dtype == torch.bfloat16 and grad.dtype == torch.bfloat16
    assert grad.shape == case.projected.shape
    _assert_close(y, expected_y, **Y_TOLERANCE, label="y")
    expected = _sections(expected_grad, k_heads, v_heads)
    for name, actual in _sections(grad, k_heads, v_heads).items():
        _assert_close(actual, expected[name], **GRAD_TOLERANCE, label=name)


def _fla_autograd(case, k_heads: int, v_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fla-core's public chain under its own autograd (conv, chunked gated delta rule with the
    L2 norm, gate and beta sigmoid in kernel, gated RMSNorm): y and d projected."""
    from fla.modules.conv import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    conv_dim, _ = _widths(k_heads, v_heads)
    rows, value = case.projected.shape[0], v_heads * V
    x = case.projected.clone().requires_grad_(True)
    weights = case.weights
    mixed, _ = causal_conv1d(
        x[None, :, :conv_dim],
        weight=weights.conv_weight,
        activation="silu",
        cu_seqlens=case.cu_seqlens.clone(),
    )
    q, k, v = mixed[0].split([k_heads * K, k_heads * K, value], -1)
    tail = x[:, conv_dim + value :]
    o, _ = chunk_gated_delta_rule(
        q.reshape(1, rows, k_heads, K),
        k.reshape(1, rows, k_heads, K),
        v.reshape(1, rows, v_heads, V),
        g=tail[:, v_heads:].reshape(1, rows, v_heads),
        beta=tail[:, :v_heads].reshape(1, rows, v_heads),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=weights.A_log,
        dt_bias=weights.dt_bias,
        use_beta_sigmoid_in_kernel=True,
        cu_seqlens=case.cu_seqlens.clone(),
    )
    z = x[:, conv_dim : conv_dim + value]
    y = rms_norm_gated(
        o.reshape(-1, V), z.reshape(-1, V), weights.norm_weight, None, "silu", eps=EPS
    ).view(rows, value)
    (grad,) = torch.autograd.grad(y, x, case.dy)
    return y.detach(), grad


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
@pytest.mark.parametrize("decay", list(DECAYS))
def test_every_output_is_flas_with_the_same_pinned_configs(k_heads, v_heads, decay):
    """Algorithm identity only: compare to FLA with the same fixed launch configs."""
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 1, decay=decay)
    y, grad = _run(op, case)
    with pinned_fla_reference():
        fla_y, fla_grad = _fla_autograd(case, k_heads, v_heads)
    torch.cuda.synchronize()

    # Only the autograd-chain comparison folds signed zero: fla-core accumulates into
    # zero-filled input gradients (0.0 + -0.0 = +0.0); our kernels write their signed value.
    def chain_bits(tensor):
        return _bits(torch.where(tensor == 0, torch.zeros_like(tensor), tensor))

    assert torch.equal(chain_bits(y), chain_bits(fla_y))
    fla_sections = _sections(fla_grad, k_heads, v_heads)
    for name, actual in _sections(grad, k_heads, v_heads).items():
        differ = int((chain_bits(actual) != chain_bits(fla_sections[name])).sum())
        assert differ == 0, (name, differ, actual.numel())


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
def test_every_output_is_no_worse_than_unmodified_fla_against_fp64(k_heads, v_heads):
    """y and every section of d projected hold the bounds (``_gdn_precision``) against
    unmodified fla-core, autotuned L2 norm included, over the decays and seeds."""
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    ratios: dict[str, list[tuple[float, float]]] = {}
    for decay in DECAYS:
        for seed in SEEDS:
            case, expected_y, expected_grad = _reference(k_heads, v_heads, seed, decay)
            y, grad = _run(op, case)
            flas_y, flas_grad = _fla_autograd(case, k_heads, v_heads)
            label = (k_heads, v_heads, decay, seed)
            ratios.setdefault("y", []).append(no_worse(y, flas_y, expected_y, ("y", *label)))
            expected = _sections(expected_grad, k_heads, v_heads)
            flas = _sections(flas_grad, k_heads, v_heads)
            for name, actual in _sections(grad, k_heads, v_heads).items():
                ratio = no_worse_section(name, actual, flas[name], expected[name], (name, *label))
                ratios.setdefault(name, []).append(ratio)
    assert_means(ratios)


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_gate_backward_reads_projected_without_saving_a_copy(k_heads, v_heads, monkeypatch):
    """The saved projected tensor supplies strided a; da keeps the dense helper's bits."""
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 9)
    provider = op._kernel
    gates, gradient = provider._gates, provider._gate_gradient
    copies, calls = [], []

    def record_gates(geometry, projected):
        b, a = gates(geometry, projected)
        copies.append(a)
        return b, a

    def check_gradient(a, A_log, dt_bias, dyg, out):
        dc, vw = op.geometry.conv_dim, op.geometry.value_width
        expected = case.projected[:, dc + vw + v_heads :]
        assert a.data_ptr() == expected.data_ptr() and a.stride() == expected.stride()
        dense = torch.empty_like(out)
        gradient(a.contiguous(), A_log, dt_bias, dyg, dense)
        gradient(a, A_log, dt_bias, dyg, out)
        assert _same(out, dense)
        calls.append(True)

    monkeypatch.setattr(provider, "_gates", record_gates)
    monkeypatch.setattr(provider, "_gate_gradient", check_gradient)
    y, saved = provider.train_forward(
        op.geometry, case.projected, *case.weights.args(), case.cu_seqlens
    )
    assert len(copies) == 1
    assert all(t.data_ptr() != copies[0].data_ptr() for t in saved)
    grad = provider.train_backward(op.geometry, saved, case.dy)
    assert y.shape == case.dy.shape and grad.shape == case.projected.shape
    assert calls == [True]


def test_a_reused_cu_seqlens_buffer_is_read_afresh_on_every_call():
    """fla-core caches chunk metadata by cu_seqlens' identity: a buffer refilled in place gives
    a new buffer's results, forward and backward, bit for bit."""
    k_heads, v_heads = 16, 32
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 3, lengths=[64, 66, 200])
    cu = case.cu_seqlens
    first = _run(op, case, cu_seqlens=cu)
    cu.copy_(_cu([130, 1, 199]))  # same tensor object, new contents
    reused = _run(op, case, cu_seqlens=cu)
    fresh = _run(op, case, cu_seqlens=_cu([130, 1, 199]))
    torch.cuda.synchronize()
    assert all(_same(a, b) for a, b in zip(reused, fresh, strict=True))
    assert not _same(reused[0], first[0])


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_recompute_is_bitwise_and_checkpointing_changes_nothing(k_heads, v_heads):
    """The forward and backward are deterministic, so activation checkpointing (which runs the
    forward again for the backward) gives the plain call's gradient bit for bit."""
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 4)
    first, second = _run(op, case), _run(op, case)
    projected = case.projected.clone().requires_grad_(True)
    y = checkpoint(
        op,
        projected,
        *case.weights.args(),
        case.cu_seqlens,
        max_sequence_length=case.max_sequence_length,
        use_reentrant=False,
    )
    y.backward(case.dy)
    torch.cuda.synchronize()
    assert all(_same(a, b) for a, b in zip(first, second, strict=True))
    assert _same(y.detach(), first[0]) and _same(projected.grad, first[1])


@pytest.mark.parametrize("cute", ["auto", "0"])
def test_training_forward_matches_prefill_of_stateless_sequences(cute, monkeypatch):
    """The training forward and stateless prefill (slot -1) compute the same y: to rounding
    where both run fla-core's chain (bitwise equality is not part of the contract), within
    either's error against FP64 where prefill runs CuTe or our Triton kernels. With CuTe
    disabled, prefill runs our Triton prefill from compute capability 10.0."""
    if cute == "0":
        monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "0")
    k_heads, v_heads = 16, 32
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    prefill = GatedDeltaNetPrefill(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    case = _inputs(k_heads, v_heads, 5)
    count = len(LENGTHS)
    conv_dim, _ = _widths(k_heads, v_heads)
    state = torch.zeros(1, v_heads, V, K, device="cuda")
    conv_state = torch.zeros(1, 3, conv_dim, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        y = op(
            case.projected,
            *case.weights.args(),
            case.cu_seqlens,
            max_sequence_length=case.max_sequence_length,
        )
        expected = prefill(
            case.projected,
            *case.weights.args(),
            case.cu_seqlens,
            torch.full((count,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros(count, dtype=torch.bool, device="cuda"),
            state,
            conv_state,
        )
    torch.cuda.synchronize()
    if prefill.provider in ("triton", "cute"):
        _assert_close(y, expected, rms=1e-2, peak=2e-2)
    else:
        _assert_close(y, expected, rms=1e-3, peak=1e-2)


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("grad_mode", [True, False])
def test_a_weight_that_requires_grad_is_rejected_before_any_device_work(weight, grad_mode):
    k_heads, v_heads = 16, 32
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 6)
    weights = {
        name: value.clone().requires_grad_(name == weight)
        for name, value in vars(case.weights).items()
    }
    for projected in (case.projected, case.projected.clone().requires_grad_(True)):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        with (
            torch.set_grad_enabled(grad_mode),
            pytest.raises(ValueError, match=f"{weight} is a frozen base weight"),
        ):
            op(
                projected,
                *(weights[name] for name in WEIGHTS),
                case.cu_seqlens,
                max_sequence_length=case.max_sequence_length,
            )
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated


def test_forward_and_backward_never_synchronize_with_the_host():
    k_heads, v_heads = 16, 32
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    case = _inputs(k_heads, v_heads, 7)
    _run(op, case)  # compile and autotune first
    projected = case.projected.clone().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        y = op(
            projected,
            *case.weights.args(),
            case.cu_seqlens,
            max_sequence_length=case.max_sequence_length,
        )
        torch.autograd.grad(y, projected, case.dy)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


# fla-core 0.5.2's backward functions that form weight gradients (whatever requires grad), by
# module: the conv's dw, the gated norm's dw and the gate's dA_log and ddt_bias.
WEIGHT_GRADIENTS = {
    "fla.modules.conv.triton.ops": "causal_conv1d_bwd",
    "fla.modules.fused_norm_gate": "layer_norm_gated_bwd",
    "fla.ops.gated_delta_rule.chunk": "gdn_gate_bwd",
    "fla.ops.gated_delta_rule.gate": "gdn_gate_bwd",
}


def test_no_weight_gradient_path_of_fla_core_runs(monkeypatch):
    """The operation's forward and backward call none of fla-core's weight-gradient functions,
    neither through fla-core's modules nor through the provider's bound callables; fla-core's
    own autograd chain, the control, calls each."""
    k_heads, v_heads = 16, 32
    op = _training_op(k_heads, v_heads)
    if op is None:
        return
    called: list[str] = []
    provider = op._kernel
    for module_name, name in WEIGHT_GRADIENTS.items():
        module = importlib.import_module(module_name)
        original = getattr(module, name)

        def recorder(*args, _original=original, _name=name, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, recorder)
        # The provider binds its callables when built: record those too.
        for attribute, value in vars(provider).items():
            if value is original:
                monkeypatch.setattr(provider, attribute, recorder)
    case = _inputs(k_heads, v_heads, 8)
    _run(op, case)
    torch.cuda.synchronize()
    assert called == []
    _fla_autograd(case, k_heads, v_heads)
    torch.cuda.synchronize()
    assert set(called) == set(WEIGHT_GRADIENTS.values())


def test_sm12x_conv_backward_dispatches_to_gluon(monkeypatch):
    """The three tuned devices launch the Gluon callable for both model geometries."""
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    if name not in {
        "NVIDIA GeForce RTX 5090",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA GB10",
    }:
        pytest.skip("Gluon conv backward dispatch is specific to the three tuned sm12x devices")
    called = []

    class LaunchRecorder:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                called.append(self.kernel)
                return launch(*args, **kwargs)

            return run

    gluon_kernel = _fla_training._conv_bwd_gluon
    monkeypatch.setattr(_fla_training, "_conv_bwd_gluon", LaunchRecorder(gluon_kernel))
    monkeypatch.setattr(
        _fla_training, "_conv_bwd_kernel", LaunchRecorder(_fla_training._conv_bwd_kernel)
    )
    for k_heads, v_heads in GEOMETRIES:
        case = _inputs(k_heads, v_heads, 9, lengths=[1, 7])
        dc, _ = _widths(k_heads, v_heads)
        key = k_heads * K
        x = case.projected[:, :dc]
        parts = (x[:, :key], x[:, key : 2 * key], x[:, 2 * key :])
        _fla_training.conv_backward(
            x,
            case.weights.conv_weight,
            _fla_training.sequence_index(case.cu_seqlens, x.shape[0]),
            case.cu_seqlens,
            parts,
            torch.empty_like(x),
        )
    torch.cuda.synchronize()
    assert called == [gluon_kernel] * len(GEOMETRIES)


def test_the_conv_kernels_are_flas_bit_for_bit(monkeypatch):
    """conv_forward is fla-core's SiLU(causal_conv1d) split into q, k and v, and
    conv_backward its causal_conv1d_bwd dx (the Gluon kernel where the device has tiles, and
    the Triton fallback), bit for bit across sequence starts, into a strided output, and down
    to pre-activations of -400 (where fla-core's SiLU flushes)."""
    op = _training_op(1, 2)  # configures and binds fla-core
    if op is None:
        return
    from fla.modules.conv import causal_conv1d
    from fla.modules.conv.triton.ops import causal_conv1d_bwd

    generator = torch.Generator(device="cuda").manual_seed(9)
    lengths, dc, key = [1, 700, 5, 1294, 1000], 1024, 256
    rows = sum(lengths)
    cu = _cu(lengths)
    table = _fla._chunk_table(cu, rows)
    sequence = _fla_training.sequence_index(cu, rows)
    weight = (0.5 * torch.randn(dc, 4, generator=generator, device="cuda")).bfloat16()
    d_mixed = (64 * torch.randn(rows, dc, generator=generator, device="cuda")).bfloat16()
    parts = (d_mixed[:, :key], d_mixed[:, key : 2 * key], d_mixed[:, 2 * key :])
    x = (3 * torch.randn(rows, dc + 40, generator=generator, device="cuda")).bfloat16()
    x[:, :32] = -400.0  # deep below where fla-core's SiLU flushes
    x = x[:, :dc]  # strided rows, as the composition passes projected
    mixed, _ = causal_conv1d(
        x[None], weight=weight, activation="silu", cu_seqlens=cu.clone(), chunk_indices=table
    )
    q, k, v = _fla_training.conv_forward(x, weight, sequence, cu, key)
    assert all(t.is_contiguous() for t in (q, k, v))
    assert _same(torch.cat((q, k, v), 1), mixed[0])
    ours = torch.empty(rows, dc, dtype=torch.bfloat16, device="cuda")
    _fla_training.conv_backward(x, weight, sequence, cu, parts, ours)
    theirs = causal_conv1d_bwd(
        x=x[None], dy=d_mixed[None], dht=None, weight=weight, activation="silu", cu_seqlens=cu
    )[0][0]
    assert _same(ours, theirs)
    wide = torch.zeros(rows, dc + 64, dtype=torch.bfloat16, device="cuda")
    _fla_training.conv_backward(x, weight, sequence, cu, parts, wide[:, :dc])
    assert _same(wide[:, :dc], ours) and not bool(wide[:, dc:].any())
    monkeypatch.setattr(_fla_training, "_GLUON_TILES", {})  # the Triton fallback
    fallback = torch.empty_like(ours)
    _fla_training.conv_backward(x, weight, sequence, cu, parts, fallback)
    assert _same(fallback, theirs)


def _fla_norm_backward(dy, x, z, weight, rstd, tile=None):
    """fla-core's layer_norm_gated_bwd with the weight (as its autograd runs it), under its
    autotuner or forced to one tile (rows, warps)."""
    import fla.modules.fused_norm_gate as norm
    import triton

    tuner = norm.layer_norm_gated_bwd_kernel
    while not hasattr(tuner, "configs"):
        tuner = tuner.fn
    saved = tuner.configs, tuner.cache
    if tile is not None:
        tuner.configs, tuner.cache = [triton.Config({"BT": tile[0]}, num_warps=tile[1])], {}
    try:
        do, dz, dw, *_ = norm.layer_norm_gated_bwd(
            dy, x, z, weight, None, activation="silu", eps=EPS, rstd=rstd, is_rms_norm=True
        )
        assert dw is not None
        return do, dz
    finally:
        tuner.configs, tuner.cache = saved


def test_the_norm_input_gradients_are_flas_without_its_weight_gradient():
    """The norm backward kernel is fla-core's layer_norm_gated_bwd bit for bit at every tile of
    its autotune space; heads over strided token rows equal one head per row; both match its
    autotuned run to rounding."""
    op = _training_op(1, 2)  # configures and binds fla-core
    if op is None:
        return
    provider = op._kernel
    generator = torch.Generator(device="cuda").manual_seed(11)
    rows, hv = 3001, 8
    o = (2 * torch.randn(rows, hv, V, generator=generator, device="cuda")).bfloat16()
    z = (3 * torch.randn(rows, hv * V, generator=generator, device="cuda")).bfloat16()
    weight = 1 + 0.3 * torch.randn(V, generator=generator, device="cuda")
    scale = torch.ones(rows, 1, device="cuda")
    scale[::7], scale[3::11], scale[5::13] = 0.0, 2.0**10, 2.0**-10
    dy = (torch.randn(rows, hv * V, generator=generator, device="cuda") * scale).bfloat16()
    x, g, flat_dy = o.reshape(-1, V), z.reshape(-1, V), dy.reshape(-1, V)
    _, _, rstd, _ = provider._gated_norm_fwd(
        x=x, g=g, weight=weight, bias=None, activation="silu", eps=EPS, is_rms_norm=True
    )
    single = [t.unsqueeze(1) for t in (x, g, flat_dy)]  # [N, 1, D]: one head per row
    for tile in [(bt, warps) for bt in (16, 32, 64) for warps in (4, 8, 16)]:
        theirs = _fla_norm_backward(flat_dy, x, g, weight, rstd, tile)
        ours = tuple(torch.empty_like(t) for t in theirs)
        _gated_norm.gated_rms_norm_backward(
            single[0], single[1], weight, rstd, single[2], *(t.unsqueeze(1) for t in ours),
            rows_per_program=tile[0], warps=tile[1],
        )  # fmt: skip
        assert _same(ours[0], theirs[0]) and _same(ours[1], theirs[1]), tile
    # z and dz as column slices of wider rows, as the composition passes them.
    columns = slice(32, 32 + hv * V)
    wide_z = torch.zeros(rows, hv * V + 96, dtype=torch.bfloat16, device="cuda")
    wide_z[:, columns] = z
    wide_dz = torch.zeros_like(wide_z)
    do = torch.empty_like(o)
    heads = lambda t: t.unflatten(1, (hv, V))
    _gated_norm.gated_rms_norm_backward(
        o, heads(wide_z[:, columns]), weight, rstd, heads(dy), do, heads(wide_dz[:, columns])
    )
    one = torch.empty_like(x), torch.empty_like(g)
    _gated_norm.gated_rms_norm_backward(
        *single[:2], weight, rstd, single[2], *(t.unsqueeze(1) for t in one)
    )
    dz = wide_dz[:, columns]
    assert _same(do.view(-1, V), one[0]) and _same(dz.reshape(-1, V), one[1])
    assert not bool(wide_dz[:, :32].any() or wide_dz[:, columns.stop :].any())
    tuned = _fla_norm_backward(flat_dy, x, g, weight, rstd)
    _assert_close(do.view(-1, V), tuned[0], rms=1e-3, peak=1e-2)
    _assert_close(dz.reshape(-1, V), tuned[1], rms=1e-3, peak=1e-2)
