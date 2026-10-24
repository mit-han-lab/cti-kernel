# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""gdn.triton.training on the GPU: FP64 and unmodified fla-core bounds, determinism, CUDA graph
replay, no host synchronisation, prefill consistency, and the rule's state in and out against
the torch reference."""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from ctikernels.gdn import (
    GatedDeltaNetPrefill,
    GatedDeltaNetTraining,
    _cute,
    _fla,
    training,
)
from ctikernels.gdn.torch import training as reference_rule
from ctikernels.gdn.triton import training as triton_training

from ._gdn_precision import assert_means, no_worse, output_ulp
from .test_gdn_training import (
    DECAYS,
    GEOMETRIES,
    GRAD_TOLERANCE,
    LENGTHS,
    TP_GEOMETRIES,
    Y_TOLERANCE,
    _assert_close,
    _fla_autograd,
    _inputs,
    _reference,
    _same,
    _sections,
    _widths,
)

K = V = 128


@pytest.fixture(autouse=True)
def _triton_provider(monkeypatch):
    """These tests hold the Triton provider, on the devices it covers."""
    low, high = training.TRITON_CAPABILITIES
    if not low <= torch.cuda.get_device_capability() <= high:
        pytest.skip("the Triton training provider covers compute capabilities 10.0 to 12.1")
    monkeypatch.setattr(_cute, "training", lambda device: None)


def _op(k_heads: int, v_heads: int) -> GatedDeltaNetTraining:
    op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    assert op.provider == "triton"
    return op


def _run(op, case) -> tuple[torch.Tensor, torch.Tensor]:
    projected = case.projected.clone().requires_grad_(True)
    y = op(projected, *case.weights.args(), case.cu_seqlens,
           max_sequence_length=case.max_sequence_length)  # fmt: skip
    (grad,) = torch.autograd.grad(y, projected, case.dy)
    return y.detach(), grad


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
@pytest.mark.parametrize("decay", list(DECAYS))
def test_training_matches_the_fp64_reference(k_heads, v_heads, decay):
    case, expected_y, expected_grad = _reference(k_heads, v_heads, 1, decay)
    y, grad = _run(_op(k_heads, v_heads), case)
    _assert_close(y, expected_y, **Y_TOLERANCE, label="y")
    expected = _sections(expected_grad, k_heads, v_heads)
    for name, actual in _sections(grad, k_heads, v_heads).items():
        _assert_close(actual, expected[name], **GRAD_TOLERANCE, label=name)


# The precision gate against unmodified fla-core, both measured against FP64 over eight seeds
# per decay (16 cases):
# - The geometric means hold ``_gdn_precision``'s bounds, unchanged.
# - Per case, each section's RMS and largest error are within CAPS times fla-core's, plus, for
#   the largest error, one BF16 ulp of the output at the reference's largest magnitude.
#
# The caps come from an independent accepted pipeline: our CuTe training vs fla-core, the same
# 16 cases on the GB10. A cap is that control's largest per-case ratio pooled over the six
# geometries, plus 10%; noise does not depend on head count. dq takes dk's cap: CuTe's dq keeps
# fla-core's dqkwg arithmetic, so it has no spread of its own.
#
#   control max   y     dq    dk    dv    dz    db    da
#   peak          1.13  1.00  1.18  1.34  1.20  1.21  1.59
#   RMS           1.00  1.01  1.01  1.01  1.00  1.02  1.31
#
# The output ulp: d projected is BF16, so where both pipelines round the same value one ulp
# apart, the per-case ratio jumps with neither at fault. Worst element, ref / ours / fla-core:
# - (16, 32), seed 3: 100.1 / 99 / 99.5 (ulp 0.5);
# - (16, 48), slow decay, seed 8: -48.05 / -48.5 / -48.25 (ulp 0.25);
# - (4, 12), seed 13: -35.16 / -34.5 / -34.75 (ulp 0.25).
# FP32 do and v' (or FP32 dk terms) lowered the means but left these maxima, and cost 0.56 ms.
#
# Our kernels round where fla-core does (BF16 q, k, v, w, u, chunk states, v', do and ds; FP32
# accumulation), so the two errors share their dominant term. In a decomposed case
# (Qwen3.5-9B heads, slow decay, a sequence's fifth row) the BF16 inputs account for 0.42% of
# the largest dq and the kernel's own rounding for 0.18%.
GATE_SEEDS = tuple(range(1, 9))
CAPS = {  # section: (RMS, largest error) versus fla-core
    "y": (1.10, 1.24),
    "dq": (1.11, 1.30),
    "dk": (1.11, 1.30),
    "dv": (1.11, 1.47),
    "dz": (1.10, 1.32),
    "db": (1.12, 1.33),
    "da": (1.44, 1.75),
}


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
def test_every_output_is_no_worse_than_unmodified_fla_against_fp64(k_heads, v_heads):
    # fla-core imported through our loader first (its pinned configuration); a direct import
    # before it would leave fla-core's backend dispatch on for every later test in the process.
    try:
        _fla.load("GatedDeltaNetDecode", torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(f"the comparison needs fla-core (cti-kernels[gdn]): {error}")
    op = _op(k_heads, v_heads)
    ratios: dict[str, list[tuple[float, float]]] = {}
    for decay in DECAYS:
        for seed in GATE_SEEDS:
            case, expected_y, expected_grad = _reference(k_heads, v_heads, seed, decay)
            y, grad = _run(op, case)
            flas_y, flas_grad = _fla_autograd(case, k_heads, v_heads)
            expected = {"y": expected_y, **_sections(expected_grad, k_heads, v_heads)}
            flas = {"y": flas_y, **_sections(flas_grad, k_heads, v_heads)}
            ours = {"y": y, **_sections(grad, k_heads, v_heads)}
            for name, actual in ours.items():
                rms, peak = CAPS[name]
                ratio = no_worse(
                    actual, flas[name], expected[name], (name, k_heads, v_heads, decay, seed),
                    rms_versus_fla=rms, peak_versus_fla=peak, peak_floor=output_ulp(expected[name]),
                )  # fmt: skip
                ratios.setdefault(name, []).append(ratio)
    assert_means(ratios)


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_calls_are_bitwise_deterministic_and_replay_from_a_cuda_graph(k_heads, v_heads):
    """Two calls, a checkpointed call, and a step captured at another T than the eager warm-up
    (no compile inside the capture) give the same bits."""
    op = _op(k_heads, v_heads)
    case = _inputs(k_heads, v_heads, 4)
    first, second = _run(op, case), _run(op, case)
    assert all(_same(a, b) for a, b in zip(first, second, strict=True))
    projected = case.projected.clone().requires_grad_(True)
    y = torch.utils.checkpoint.checkpoint(
        op, projected, *case.weights.args(), case.cu_seqlens,
        max_sequence_length=case.max_sequence_length, use_reentrant=False,
    )  # fmt: skip
    y.backward(case.dy)
    assert _same(y.detach(), first[0]) and _same(projected.grad, first[1])
    other = _inputs(k_heads, v_heads, 5, lengths=[1, 97, 3])  # T = 101 against 330
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(op, other)
    graph.replay()
    torch.cuda.synchronize()
    assert all(_same(a, b) for a, b in zip(captured, _run(op, other), strict=True))


def test_forward_and_backward_never_synchronize_with_the_host():
    op = _op(16, 32)
    case = _inputs(16, 32, 7)
    _run(op, case)  # compile first
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _run(op, case)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_the_forward_matches_stateless_prefill():
    """The training forward and stateless prefill (slot -1) agree within either's error
    against FP64 (prefill's provider is whichever the device resolves)."""
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    try:
        prefill = GatedDeltaNetPrefill(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    except RuntimeError as refused:  # no prefill provider on this device yet
        pytest.skip(str(refused))
    case = _inputs(k_heads, v_heads, 5)
    count = len(LENGTHS)
    conv_dim, _ = _widths(k_heads, v_heads)
    with torch.inference_mode():
        y = op(case.projected, *case.weights.args(), case.cu_seqlens,
               max_sequence_length=case.max_sequence_length)  # fmt: skip
        expected = prefill(
            case.projected, *case.weights.args(), case.cu_seqlens,
            torch.full((count,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros(count, dtype=torch.bool, device="cuda"),
            torch.zeros(1, v_heads, V, K, device="cuda"),
            torch.zeros(1, 3, conv_dim, dtype=torch.bfloat16, device="cuda"),
        )  # fmt: skip
    _assert_close(y, expected, rms=1e-2, peak=2e-2)


@pytest.mark.parametrize(("k_heads", "v_heads"), [(16, 32), (4, 12)])
def test_the_rule_carries_states_in_and_out(k_heads, v_heads):
    """rule_forward / rule_backward with initial states and the final states' gradient
    against the FP64 torch reference of the same algorithm (with the gated RMSNorm)."""
    generator = torch.Generator(device="cuda").manual_seed(11)
    lengths = [5, 64, 130, 1]
    rows, count = sum(lengths), len(lengths)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    q, k = (torch.nn.functional.normalize(rand(rows, k_heads, K), dim=-1).bfloat16()
            for _ in range(2))  # fmt: skip
    v = rand(rows, v_heads, V).bfloat16()
    gates = rand(rows, 2 * v_heads).bfloat16()  # b | a
    z = rand(rows, v_heads * V).bfloat16()
    norm_weight = 1 + 0.1 * rand(V)
    A_log = torch.rand(v_heads, device="cuda", generator=generator) - 0.5
    dt_bias = torch.rand(v_heads, device="cuda", generator=generator) * 4 - 2
    h0 = 0.1 * rand(count, v_heads, V, K)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()
    b, a = gates[:, :v_heads], gates[:, v_heads:]
    beta = torch.sigmoid(b.float())
    chunks = triton_training.Chunks.of(cu, rows)
    configs = triton_training.configs(torch.cuda.get_device_capability())
    o, y, _, saved, ht = triton_training.rule_forward(
        q, k, v, a, beta, A_log, dt_bias, z, norm_weight, chunks, configs, h0, final=True
    )
    do, dht = rand(rows, v_heads, V).bfloat16(), rand(count, v_heads, V, K)
    dgates = torch.empty_like(gates)
    dq, dk, dk_terms, dv, dh0 = triton_training.rule_backward(
        q, k, v, a, b, beta, saved, A_log, dt_bias, do, chunks, dgates[:, v_heads:],
        dgates[:, :v_heads], configs, True, dht,
    )  # fmt: skip

    leaves = [x.double().requires_grad_() for x in (q, k, v, gates, h0)]
    rb, ra = leaves[3][:, :v_heads], leaves[3][:, v_heads:]
    g = -torch.exp(A_log.double()) * torch.nn.functional.softplus(ra + dt_bias.double())
    eo, eht = reference_rule.ChunkRule.apply(leaves[0], leaves[1], leaves[2], g,
                                             torch.sigmoid(rb), leaves[4], K**-0.5, cu)  # fmt: skip
    zz = z.double().view(rows, v_heads, V)
    ey = eo * torch.rsqrt(eo.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_weight.double()
    _assert_close(o, eo, rms=1e-2, peak=3e-2, label="o")
    _assert_close(
        y, (ey * zz * torch.sigmoid(zz)).reshape(rows, -1), rms=1e-2, peak=3e-2, label="y"
    )
    _assert_close(ht, eht, rms=1e-2, peak=3e-2, label="ht")
    grads = torch.autograd.grad((eo * do.double()).sum() + (eht * dht.double()).sum(), leaves)
    group = (rows, k_heads, v_heads // k_heads, K)
    dk = dk + dk_terms.view(group).sum(2)
    found = (dq, dk, dv, dgates.float(), dh0)
    for name, actual, expected in zip(
        ("dq", "dk", "dv", "dgates", "dh0"), found, grads, strict=True
    ):
        _assert_close(actual, expected, **GRAD_TOLERANCE, label=name)


@pytest.mark.parametrize(
    "lengths",
    [[1], [64], [65], [4096], [5, 130], [1] * 7, [300, 1, 64, 63, 129], [7] * 600, [2] * 256,
     [3] * 1000],
)  # fmt: skip
def test_the_chunk_index_matches_its_torch_definition(lengths):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()
    rows, count = int(cu[-1]), len(lengths)
    chunks = triton_training.Chunks.of(cu, rows)
    offsets = torch.nn.functional.pad(((cu.diff() + 63) // 64).cumsum(0), (1, 0)).int()
    entry = torch.arange((rows + 63 * count) // 64, device="cuda", dtype=torch.int32)
    sequence = torch.searchsorted(offsets[1:], entry, right=True).clamp(max=count - 1)
    table = torch.stack((sequence, entry - offsets[sequence]), 1).int()
    token = torch.arange(rows, device="cuda", dtype=torch.int32)
    expected = (cu, offsets, table, torch.searchsorted(cu[1:], token, right=True).int())
    found = (chunks.cu_seqlens, chunks.offsets, chunks.table, chunks.sequence)
    assert chunks.cu_seqlens.data_ptr() != cu.data_ptr()
    assert all(torch.equal(a, b) for a, b in zip(found, expected, strict=True))


def test_one_spill_free_chunk_index_kernel_serves_every_sequence_count():
    """A variant loaded on a later batch's sequence count would raise the CUDA stack limit after
    the caller planned its memory: one specialization serves 1 to 1000 sequences, spill-free."""
    from ctikernels.gdn.triton import _train_chunks

    for count in (1, 2, 256, 1000):
        cu = torch.arange(0, 2 * count + 1, 2, dtype=torch.int32, device="cuda")
        triton_training.Chunks.of(cu, 2 * count)
    kernels = list(
        _train_chunks._index_kernel.device_caches[torch.cuda.current_device()][0].values()
    )
    assert len(kernels) == 1 and kernels[0].n_spills == 0


@pytest.mark.parametrize("capability", [100, 103, 120, 121])
def test_every_kernel_compiles_for_each_covered_target(monkeypatch, capability):
    """A training step compiled (not launched) for each compute capability the provider
    covers: every Triton kernel compiles. Compile-only is the check for sm_100, where no device
    is validated."""
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    target = GPUTarget("cuda", capability, 32)
    compiled = {}
    run = JITFunction.run

    def compile_only(self, *args, grid, warmup, **kwargs):
        # Fresh caches bound to the patched target, restored at teardown: a kernel compiled
        # here for another target must not serve later launches (the shared gated norm's).
        monkeypatch.setattr(self, "device_caches", defaultdict(self.create_binder))
        kernel = run(self, *args, grid=grid, warmup=True, **kwargs)
        compiled[self.fn.__name__] = kernel
        return kernel

    monkeypatch.setattr(driver.active, "get_current_target", lambda: target)
    monkeypatch.setattr(JITFunction, "run", compile_only)
    case = _inputs(16, 48, 8)
    kernel = triton_training.Training((capability // 10, capability % 10))
    _, saved = kernel.train_forward(
        training.Geometry(16, 48), case.projected, *case.weights.args(), case.cu_seqlens
    )
    kernel.train_backward(training.Geometry(16, 48), saved, case.dy)
    assert len(compiled) == 10, sorted(compiled)
    assert all(kernel.asm["cubin"] for kernel in compiled.values())
