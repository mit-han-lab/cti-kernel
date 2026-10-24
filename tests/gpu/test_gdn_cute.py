# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/ops/gated_delta_rule/chunk.py`, backward comparison composition);
# see THIRD_PARTY_NOTICES.

"""ctikernels.gdn's CuTe DSL training forward and backward (``_cute``) on the GPU: no worse than
fla-core's chunked ones against an FP64 reference, per output, alone and inside the training
operation. On sm_120 / sm_121 (RTX 5090, RTX PRO 6000, GB10) they must run, as the provider
``cute``; on other devices ``load`` is None (the expected outcome, not a skip).
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from ctikernels.gdn import GatedDeltaNetTraining, _cute, _fla, _fla_fixed, training
from ctikernels.gdn._contract import Geometry

from ..support.child import run_child
from ._gdn_l2norm import unmodified_fla_reference_op
from ._gdn_precision import (
    GRAD_PEAK_VERSUS_FLA,
    PEAK_VERSUS_FLA,
    assert_means,
    no_worse,
    no_worse_section,
)
from .test_gdn_training import (
    DECAYS,
    GEOMETRIES,
    SEEDS,
    TP_GEOMETRIES,
    _reference,
    _run,
    _same,
    _sections,
    _training_op,
)

K = V = 128
CHUNK = 64
F64 = torch.float64


@pytest.fixture(autouse=True)
def _cute_at_every_gated_geometry(monkeypatch):
    """These tests hold CuTe at every gated geometry, wherever Triton is faster."""
    monkeypatch.setattr(training, "TRITON_FASTER", {})


RCP_LN2 = 1.4426950408889634
LENGTHS = {"short": [1, 63, 64, 65, 130, 7], "one": [1024], "mixed": [3000, 1, 700]}
# The precision gates' geometries: TP1 and the TP-local ones the training operation runs on
# CuTe (``training.CUTE_GEOMETRIES``: CuTe misses a per-case bound at the others).
# Small Hv can select other V slices (``_slice``, ``_grad_slice``).
CUTE_TP_GEOMETRIES = [g for g in TP_GEOMETRIES if Geometry(*g) in training.CUTE_GEOMETRIES]
GATE_GEOMETRIES = [*GEOMETRIES, *CUTE_TP_GEOMETRIES]


def _forward() -> _cute.ChunkKernels | None:
    """The CuTe forward on this GPU: required on sm_120 / sm_121 (the provider is ``cute``),
    None elsewhere."""
    device = torch.device("cuda", torch.cuda.current_device())
    forward = _cute.load(device)
    if torch.cuda.get_device_capability(device) not in _cute.CAPABILITIES:
        assert forward is None
        return None
    try:
        installed = importlib.metadata.version(_cute.DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert forward is not None, (
        f"the CuTe path must run here: it needs {_cute.DISTRIBUTION} {_cute.VERSION} (installed:"
        f" {installed}) and {_cute.VARIABLE} not 0"
    )
    op = GatedDeltaNetTraining(num_k_heads=16, num_v_heads=32, device=device)
    assert op.provider == "cute"
    return forward


def _flas_chunk():
    """fla-core's chunked forward, imported under the provider's configuration."""
    _fla.load(GatedDeltaNetTraining.__name__, torch.device("cuda", torch.cuda.current_device()))
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    return partial(
        chunk_gated_delta_rule_fwd,
        initial_state=None,
        output_final_state=False,
        use_gate_in_kernel=True,
    )


def _call(k_heads: int, v_heads: int, lengths, decay: str, seed: int) -> dict:
    """The training call of the chunked forward, as ``_fla``'s ``train_forward`` makes it."""
    generator = torch.Generator().manual_seed(seed)
    rows = sum(lengths)
    low, high = DECAYS[decay]

    def normal(*shape):
        return torch.randn(*shape, generator=generator)

    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return {
        "q": F.normalize(normal(1, rows, k_heads, K), dim=-1).to(torch.bfloat16).cuda(),
        "k": F.normalize(normal(1, rows, k_heads, K), dim=-1).to(torch.bfloat16).cuda(),
        "v": normal(1, rows, v_heads, V).to(torch.bfloat16).cuda(),
        "g": normal(1, rows, v_heads).to(torch.bfloat16).cuda(),
        "beta": torch.sigmoid(normal(1, rows, v_heads)).cuda(),
        "scale": K**-0.5,
        "cu_seqlens": cu,
        "chunk_indices": _fla._chunk_table(cu, rows),
        "A_log": (torch.rand(v_heads, generator=generator) * 2.8 - 0.7).cuda(),
        "dt_bias": (low + (high - low) * torch.rand(v_heads, generator=generator)).cuda(),
    }


def _chunk_reference(call: dict, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """o [T, Hv, V] and the WY matrix (I + L)^-1 [T, Hv, 64] in FP64 from the call and G."""
    q, k, v = (call[name][0].to(F64) for name in "qkv")
    return _chunk_ops(q, k, v, call["beta"][0].to(F64), G[0].to(F64), call)


def _chunk_ops(q, k, v, beta, G, call: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunked forward in FP64 on FP64 q, k [T, Hk, K], v [T, Hv, V], beta and G [T, Hv]
    (differentiable)."""
    rows, v_heads = beta.shape
    group = v_heads // q.shape[1]
    o, A = [], torch.zeros(rows, v_heads, CHUNK, dtype=F64, device="cuda")
    bounds = call["cu_seqlens"].tolist()
    for bos, eos in zip(bounds[:-1], bounds[1:], strict=True):
        S = torch.zeros(v_heads, K, V, dtype=F64, device="cuda")
        for start in range(bos, eos, CHUNK):
            end = min(start + CHUNK, eos)
            qc, kc = (x[start:end].repeat_interleave(group, 1).transpose(0, 1) for x in (q, k))
            vc, gc, bc = (x[start:end].transpose(0, 1) for x in (v, G, beta))
            # 2^(g_i - g_j) on and below the diagonal only (above it, it overflows and its
            # gradient would be NaN through the mask)
            lower = torch.ones(end - start, end - start, dtype=torch.bool, device="cuda").tril()
            gap = gc[:, :, None] - gc[:, None, :]
            decay = torch.exp2(torch.where(lower, gap, torch.full_like(gap, -torch.inf)))
            L = torch.tril(bc[:, :, None] * (kc @ kc.transpose(1, 2)) * decay, -1)
            eye = torch.eye(end - start, dtype=F64, device="cuda").expand_as(L)
            T = torch.linalg.solve_triangular(eye + L, eye, upper=False)
            W = T @ (kc * (bc * torch.exp2(gc))[:, :, None])
            new = T @ (vc * bc[:, :, None]) - W @ S
            local = torch.tril(qc @ kc.transpose(1, 2) * decay) @ new
            o.append(
                (call["scale"] * (torch.exp2(gc)[:, :, None] * (qc @ S) + local)).transpose(0, 1)
            )
            A[start:end, :, : end - start] = T.detach().transpose(0, 1)
            last = gc[:, -1:]
            S = torch.exp2(last)[:, :, None] * S
            S = S + kc.transpose(1, 2) @ (torch.exp2(last - gc)[:, :, None] * new)
    return torch.cat(o), A


def _reference_grads(call: dict, G: torch.Tensor, do: torch.Tensor) -> list[torch.Tensor]:
    """dq, dk, dv, dbeta, dg (fla-core's units: 1/ln 2 of the gradient of G) and dg
    reverse-cumulated per chunk, of sum(o do) in FP64."""
    leaves = [call[name][0].to(F64).requires_grad_(True) for name in ("q", "k", "v", "beta")]
    G = G[0].to(F64).requires_grad_(True)
    o, _ = _chunk_ops(*leaves, G, call)
    *grads, dG = torch.autograd.grad(o, [*leaves, G], do[0].to(F64))
    dg = dG * RCP_LN2
    cumulated = dg.clone()
    bounds = call["cu_seqlens"].tolist()
    for bos, eos in zip(bounds[:-1], bounds[1:], strict=True):
        for start in range(bos, eos, CHUNK):
            end = min(start + CHUNK, eos)
            cumulated[start:end] = dg[start:end].flip(0).cumsum(0).flip(0)
    return [*grads, dg, cumulated]


@pytest.mark.parametrize(
    ("k_heads", "v_heads"),
    [pytest.param(16, 32, marks=pytest.mark.exhaustive), (16, 48), *GATE_GEOMETRIES[2:]],
)
def test_the_forward_is_no_worse_than_flas_against_fp64(k_heads, v_heads):
    """G is fla-core's bit for bit; o and the WY matrix A hold the bounds against FP64, over
    the decays, lengths and seeds."""
    forward = _forward()
    if forward is None:
        return
    ratios = {"o": [], "A": []}
    for decay in DECAYS:
        for lengths in LENGTHS:
            for seed in SEEDS:
                label = (decay, lengths, seed)
                call = _call(k_heads, v_heads, LENGTHS[lengths], decay, seed)
                flas_G, flas_o, flas_A, *_ = _flas_chunk()(**call)
                G, o, A = forward(**call)
                assert o.shape == flas_o.shape and o.dtype == flas_o.dtype
                assert A.shape == flas_A.shape and A.dtype == flas_A.dtype
                assert _same(G, flas_G), label
                expected_o, expected_A = _chunk_reference(call, G)
                ratios["o"].append(no_worse(o[0], flas_o[0], expected_o, ("o", *label)))
                ratios["A"].append(no_worse(A[0], flas_A[0], expected_A, ("A", *label)))
    assert_means(ratios)


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES + TP_GEOMETRIES)
def test_the_training_provider_follows_the_precision_gate(k_heads, v_heads):
    """CuTe runs the gated geometries (where Triton is not faster on the card)."""
    forward = _forward()
    if forward is None:
        return
    op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    gated = (k_heads, v_heads) in GATE_GEOMETRIES
    assert gated == (Geometry(k_heads, v_heads) in training.CUTE_GEOMETRIES)
    picked = training._cute_geometry(Geometry(k_heads, v_heads), op.device)
    assert op.provider == ("cute" if picked else "triton")


@pytest.mark.parametrize(("k_heads", "v_heads"), GATE_GEOMETRIES)
def test_training_with_it_is_no_worse_than_flas_composition(k_heads, v_heads, monkeypatch):
    """The training operation runs its forward and backward where they run; y and every
    section of d projected hold the bounds against the FP64 reference, over the decays and
    seeds."""
    forward = _forward()
    if forward is None:
        return
    ratios: dict[str, list[float]] = {}
    op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    assert op.provider == "cute"
    with monkeypatch.context() as fallback:
        fallback.setattr(_cute, "training", lambda device: None)
        fallback.setattr(training, "_triton", lambda device: None)
        flas_op = _training_op(k_heads, v_heads)
    assert flas_op is not None and flas_op.provider == "fla-core"
    flas_op = unmodified_fla_reference_op(flas_op)
    for decay in DECAYS:
        for seed in SEEDS:
            case, expected_y, expected_grad = _reference(k_heads, v_heads, seed, decay)
            y, grad = _run(op, case)
            flas_y, flas_grad = _run(flas_op, case)
            label = (k_heads, v_heads, decay, seed)
            ratios.setdefault("y", []).append(no_worse(y, flas_y, expected_y, ("y", *label)))
            expected = _sections(expected_grad, k_heads, v_heads)
            flas = _sections(flas_grad, k_heads, v_heads)
            for name, actual in _sections(grad, k_heads, v_heads).items():
                ratio = no_worse_section(name, actual, flas[name], expected[name], (name, *label))
                ratios.setdefault(name, []).append(ratio)
    assert_means(ratios)


def _assert_streams_and_graph(function) -> None:
    first = function()
    again = function()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        on_side = function()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = function()
    graph.replay()
    torch.cuda.synchronize()
    for outputs in (again, on_side, captured):
        assert all(_same(a, b) for a, b in zip(first, outputs, strict=True))


def _assert_no_host_sync(function) -> None:
    function()  # compile first
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        function()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def _peak_memory(function) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = function()
    torch.cuda.synchronize()
    del outputs
    return torch.cuda.max_memory_allocated() - base


def test_the_forward_is_bitwise_on_any_stream_and_in_a_cuda_graph():
    forward = _forward()
    if forward is None:
        return
    call = _call(16, 48, LENGTHS["mixed"], "ordinary", 2)
    _assert_streams_and_graph(partial(forward, **call))


def test_the_forward_never_synchronizes_with_the_host():
    forward = _forward()
    if forward is None:
        return
    call = _call(16, 32, LENGTHS["short"], "ordinary", 3)
    _assert_no_host_sync(partial(forward, **call))


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_its_peak_memory_is_no_more_than_flas(k_heads, v_heads):
    forward = _forward()
    if forward is None:
        return
    call = _call(k_heads, v_heads, [4096], "ordinary", 4)
    flas = _flas_chunk()

    ours, reference = partial(forward, **call), partial(flas, **call)
    _peak_memory(ours), _peak_memory(reference)  # compile and autotune first
    assert _peak_memory(ours) <= _peak_memory(reference)


def test_resolving_it_first_leaves_fla_core_configured():
    """Resolved first, in a fresh process without fla-core's variables, it has the provider
    configure fla-core, so an operation constructed afterwards loads."""
    if _forward() is None:
        return
    code = """
import torch

from ctikernels.gdn import GatedDeltaNetTraining, _cute, training

assert _cute.load(torch.device("cuda", torch.cuda.current_device())) is not None
training.TRITON_FASTER.clear()  # CuTe at every gated geometry
op = GatedDeltaNetTraining(num_k_heads=16, num_v_heads=32, device="cuda")
assert op.provider == "cute"
"""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FLA_")}
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    run_child((sys.executable, "-c", code), env=environment, timeout=600, check=True)


def _do(call: dict, seed: int) -> torch.Tensor:
    rows, v_heads = call["beta"].shape[1:]
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, rows, v_heads, V, generator=generator).to(torch.bfloat16).cuda()


def _backward_args(call: dict, G, A, do) -> dict:
    names = ("q", "k", "v", "beta", "scale", "cu_seqlens", "chunk_indices")
    return {name: call[name] for name in names} | {"A": A, "g": G, "do": do}


def _flas_backward(call: dict, G, A, do) -> list[torch.Tensor]:
    """fla-core's chunked backward, chunk_gated_delta_rule_bwd's pieces: dq, dk, dv, dbeta and
    dg before its reverse cumulative sum."""
    _flas_chunk()
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd

    q, k, v, beta, scale = (call[name] for name in ("q", "k", "v", "beta", "scale"))
    index = {"cu_seqlens": call["cu_seqlens"], "chunk_indices": call["chunk_indices"]}
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=G, **index)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=G, initial_state=None, output_final_state=False, **index
    )
    du = chunk_bwd_dv_local(q=q, k=k, g=G, do=do, scale=scale, **index)
    dh, _, du = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, g=G, h0=None, dht=None, do=do, dv=du, scale=scale, **index
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, w=w, g=G, h=h, dv=du, do=do, dh=dh, scale=scale, **index
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(k=k, v=v, beta=beta, g=G, A=A, dw=dw, du=du, **index)
    return [dq, dk.add_(dk2), dv, db, dg.add_(dg2)]


@pytest.mark.parametrize(("k_heads", "v_heads"), GATE_GEOMETRIES)
def test_the_backward_is_no_worse_than_flas_against_fp64(k_heads, v_heads):
    """dq, dk, dv, dbeta and dg, as returned and reverse-cumulated as the training op uses it,
    hold the bounds against FP64, over the decays, lengths and seeds. fla-core keeps dq / dk in
    BF16 and forms dk as a BF16 add of two BF16 parts (dk.add_(dk2)), so one element's dk peak is
    rounding noise (no FP32 staging)."""
    kernels = _forward()
    if kernels is None:
        return
    from fla.ops.utils.cumsum import chunk_local_cumsum

    ratios: dict[str, list[float]] = {}
    for decay in DECAYS:
        for lengths in LENGTHS:
            for seed in SEEDS:
                call = _call(k_heads, v_heads, LENGTHS[lengths], decay, seed)
                G, _, A = kernels(**call)
                do = _do(call, seed)
                ours = list(kernels.backward(**_backward_args(call, G, A, do)))
                flas = _flas_backward(call, G, A, do)
                index = {"cu_seqlens": call["cu_seqlens"], "chunk_indices": call["chunk_indices"]}
                ours.append(_fla_fixed.chunk_local_cumsum(ours[-1], CHUNK, reverse=True, **index))
                flas.append(chunk_local_cumsum(flas[-1], CHUNK, reverse=True, **index))
                expected = _reference_grads(call, G, do)
                names = ("dq", "dk", "dv", "dbeta", "dg", "dg cumulated")
                for name, a, f, e in zip(names, ours, flas, expected, strict=True):
                    assert a.shape == f.shape and a.dtype == f.dtype, name
                    label = (name, decay, lengths, seed)
                    bound = PEAK_VERSUS_FLA if name == "dq" else GRAD_PEAK_VERSUS_FLA
                    ratio = no_worse(a[0], f[0], e, label, peak_versus_fla=bound)
                    ratios.setdefault(name, []).append(ratio)
    assert_means(ratios)


def test_the_backward_skips_the_chunk_tables_padding():
    """1024 sequences of 64 rows, so half the chunk table pads (entries past the last sequence,
    down to -64448 valid rows): every gradient is finite and agrees with fla-core's to rounding."""
    kernels = _forward()
    if kernels is None:
        return
    call = _call(16, 32, [CHUNK] * 1024, "ordinary", 3)
    G, _, A = kernels(**call)
    do = _do(call, 3)
    ours = kernels.backward(**_backward_args(call, G, A, do))
    flas = _flas_backward(call, G, A, do)
    for name, actual, flas_ in zip(("dq", "dk", "dv", "dbeta", "dg"), ours, flas, strict=True):
        a, f = actual.double(), flas_.double()
        assert torch.isfinite(a).all(), name
        assert (a - f).square().mean().sqrt() <= 0.01 * f.square().mean().sqrt(), name
        assert (a - f).abs().max() <= 0.02 * f.abs().max(), name


def test_the_backward_is_bitwise_on_any_stream_and_in_a_cuda_graph():
    kernels = _forward()
    if kernels is None:
        return
    call = _call(16, 48, LENGTHS["mixed"], "slow", 2)
    G, _, A = kernels(**call)
    args = _backward_args(call, G, A, _do(call, 2))
    _assert_streams_and_graph(partial(kernels.backward, **args))


def test_the_backward_never_synchronizes_with_the_host():
    kernels = _forward()
    if kernels is None:
        return
    call = _call(16, 32, LENGTHS["short"], "ordinary", 3)
    G, _, A = kernels(**call)
    args = _backward_args(call, G, A, _do(call, 3))
    _assert_no_host_sync(partial(kernels.backward, **args))


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_the_backward_peak_memory_is_no_more_than_flas(k_heads, v_heads):
    kernels = _forward()
    if kernels is None:
        return
    call = _call(k_heads, v_heads, [4096], "ordinary", 4)
    G, _, A = kernels(**call)
    do = _do(call, 4)

    args = _backward_args(call, G, A, do)
    ours, reference = partial(kernels.backward, **args), partial(_flas_backward, call, G, A, do)
    _peak_memory(ours), _peak_memory(reference)  # compile and autotune first
    assert _peak_memory(ours) <= _peak_memory(reference)
