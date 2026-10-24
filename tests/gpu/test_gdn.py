# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn on the GPU: the FP64 reference, fla-core called directly, slot-pool
semantics, chunked and varlen prefill, host-sync freedom and CUDA-graph replay."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetPrefill, _fla, _l2norm, inference
from ctikernels.gdn._contract import Geometry

from ..support.child import run_child
from . import _gdn_reference as reference
from ._gdn_l2norm import pinned_fla_reference

K = V = 128
SLOTS = 24
# Qwen3.5-9B / 35B-A3B and Qwen3.5-27B per rank at TP1.
GEOMETRIES = [(16, 32), (16, 48)]


def _widths(k_heads: int, v_heads: int) -> tuple[int, int]:
    conv_dim = 2 * k_heads * K + v_heads * V
    return conv_dim, conv_dim + v_heads * V + 2 * v_heads


def _weights(k_heads: int, v_heads: int, seed: int, *, dt_bias=(-2.0, 2.0)) -> reference.Weights:
    generator = torch.Generator().manual_seed(seed)
    conv_dim, _ = _widths(k_heads, v_heads)
    low, high = dt_bias
    return reference.Weights(
        conv_weight=(0.5 * torch.randn(conv_dim, 4, generator=generator)).to(torch.bfloat16).cuda(),
        # A = exp(A_log) in about [0.5, 8], the checkpoints' range.
        A_log=(torch.rand(v_heads, generator=generator) * 2.8 - 0.7).cuda(),
        dt_bias=(low + (high - low) * torch.rand(v_heads, generator=generator)).cuda(),
        norm_weight=(1 + 0.1 * torch.randn(V, generator=generator)).cuda(),
    )


def _projected(rows: int, k_heads: int, v_heads: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    _, width = _widths(k_heads, v_heads)
    return torch.randn(rows, width, generator=generator).to(torch.bfloat16).cuda()


def _pools(k_heads: int, v_heads: int, seed: int, slots: int = SLOTS):
    """Random state and history in every slot."""
    generator = torch.Generator().manual_seed(seed)
    conv_dim, _ = _widths(k_heads, v_heads)
    state = 0.2 * torch.randn(slots, v_heads, V, K, generator=generator)
    conv_state = torch.randn(slots, 3, conv_dim, generator=generator).to(torch.bfloat16)
    return state.cuda(), conv_state.cuda()


def _poison(pool: torch.Tensor, rows: set[int], value: str, seed: int) -> torch.Tensor:
    """A copy of ``pool`` with NaN or new random values in every row outside ``rows``."""
    generator = torch.Generator().manual_seed(seed)
    copy = pool.clone()
    for row in range(pool.shape[0]):
        if row not in rows:
            if value == "nan":
                copy[row] = float("nan")
            else:
                copy[row] = torch.randn(pool.shape[1:], generator=generator).to(pool.dtype)
    return copy


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rms: float, peak: float):
    """``actual`` within ``rms`` of ``expected``'s RMS overall and ``peak`` of its largest
    magnitude anywhere (relative), both computed in FP64."""
    actual, expected = actual.double(), expected.double()
    assert torch.isfinite(actual).all()
    scale = expected.pow(2).mean().sqrt()
    error = actual - expected
    assert error.pow(2).mean().sqrt() <= rms * scale, (error.pow(2).mean().sqrt() / scale).item()
    assert error.abs().max() <= peak * expected.abs().max(), (
        error.abs().max() / expected.abs().max()
    ).item()


# Measured with fla-core 0.5.2 on the RTX 5090 over the cases below: y at most 4.6e-3 RMS and
# 9.6e-3 peak, states 3.4e-3 RMS and 5.9e-3 peak (BF16 activations between fla-core's kernels).
# A wrong gate, decay, conv tap, history row or head mapping gives errors of order 1.
Y_TOLERANCE = {"rms": 1e-2, "peak": 3e-2}
STATE_TOLERANCE = {"rms": 1e-2, "peak": 2e-2}
# The fla-core adapter vs fla-core's public chain: the same kernels but for the L2 norm's and
# the RMSNorm's sums, whose rounding moves q, k and y by about one BF16 ulp.
ADAPTER_TOLERANCE = {"rms": 4e-3, "peak": 2e-2}


def _decode_slots(padding: list[int], batch: int = 8, seed: int = 0) -> torch.Tensor:
    """Unique slots in 1..SLOTS-1 (slot 0 and the rest stay outside the batch), -1 at
    ``padding``."""
    slots = torch.randperm(SLOTS - 1, generator=torch.Generator().manual_seed(seed))[:batch] + 1
    slots[padding] = -1
    return slots.to(torch.int32).cuda()


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_decode_matches_the_fp64_reference_and_touches_only_its_slots(k_heads, v_heads):
    weights = _weights(k_heads, v_heads, seed=1)
    projected = _projected(8, k_heads, v_heads, seed=2)
    slot_indices = _decode_slots(padding=[2, 5])
    active = set(slot_indices[slot_indices >= 0].tolist())
    state, conv_state = _pools(k_heads, v_heads, seed=3)
    expected_y, expected_state, expected_conv = reference.decode(
        projected, weights, k_heads, slot_indices, state, conv_state
    )
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    outcomes = []
    for value, seed in (("nan", 4), ("random", 5)):
        pools = [_poison(pool, active, value, seed) for pool in (state, conv_state)]
        before = [pool.clone() for pool in pools]
        with torch.inference_mode():
            y = op(projected, *weights.args(), slot_indices, *pools)
        torch.cuda.synchronize()
        untouched = [row for row in range(SLOTS) if row not in active]
        for pool, old in zip(pools, before, strict=True):
            assert _same(pool[untouched], old[untouched])
        outcomes.append((y, *pools))
    (y, state_out, conv_out), (other_y, other_state, other_conv) = outcomes
    # Rows outside the batch never reach the outputs (NaN there changes nothing).
    assert _same(y, other_y)
    rows = sorted(active)
    assert _same(state_out[rows], other_state[rows]) and _same(conv_out[rows], other_conv[rows])
    assert not y[[2, 5]].any()
    _assert_close(y, expected_y, **Y_TOLERANCE)
    _assert_close(state_out[rows], expected_state[rows], **STATE_TOLERANCE)
    assert _same(conv_out[rows], expected_conv[rows])


@pytest.fixture(params=["default", "fla-core", "triton"])
def prefill_provider(request, monkeypatch):
    """The device's own prefill provider, then fla-core's adapter and our portable Triton
    prefill bound on every device."""
    if request.param == "fla-core":
        monkeypatch.setattr(
            inference,
            "_prefill_kernel",
            lambda geometry, device, **options: (
                _fla.load(GatedDeltaNetPrefill.__name__, device).prefill
            ),
        )
    elif request.param == "triton":
        from ctikernels.gdn.triton.prefill import Prefill

        monkeypatch.setattr(
            inference,
            "_prefill_kernel",
            lambda geometry, device, **options: Prefill(device, geometry),
        )
    return request.param


def _prefill_op(
    k_heads: int, v_heads: int, provider: str = "default"
) -> GatedDeltaNetPrefill | None:
    """The prefill operation on this GPU, or, for ``provider == "fla-core"`` on a device where
    fla-core computes prefill wrongly (``_fla.WRONG_ON``), None after checking that binding it
    is refused there, naming the device and the reason. The caller's test ends there: the
    refusal is its expected outcome on that device, not a skip."""
    reason = _fla.WRONG_ON.get((GatedDeltaNetPrefill.__name__, torch.cuda.get_device_capability()))
    if reason is None or provider != "fla-core":
        return GatedDeltaNetPrefill(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    with pytest.raises(RuntimeError) as raised:
        _fla.load(GatedDeltaNetPrefill.__name__, torch.device("cuda"))
    device = _fla.describe(torch.device("cuda", torch.cuda.current_device()))
    assert str(raised.value) == (
        f"GatedDeltaNetPrefill on {device} is not supported by fla-core {_fla.VERSION}: {reason}"
    )
    return None


def _prefill_case(lengths: list[int], slots: list[int], initial: list[bool]):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return (
        cu,
        torch.tensor(slots, dtype=torch.int32).cuda(),
        torch.tensor(initial, dtype=torch.bool).cuda(),
    )


@pytest.mark.parametrize(
    ("k_heads", "v_heads"), [(16, 32), pytest.param(16, 48, marks=pytest.mark.exhaustive)]
)
@pytest.mark.parametrize("decay", [pytest.param("ordinary", marks=pytest.mark.exhaustive), "slow"])
def test_prefill_matches_the_fp64_reference_for_continued_fresh_and_stateless_sequences(
    k_heads, v_heads, decay, prefill_provider
):
    op = _prefill_op(k_heads, v_heads, prefill_provider)
    if op is None:
        return
    # Lengths around fla-core's 64-token chunks; continued, fresh and stateless sequences.
    lengths = [1, 63, 64, 65, 130, 7, 129]
    slots = [3, 7, -1, 11, 2, 20, -1]
    initial = [True, False, False, True, False, True, True]
    cu_seqlens, slot_indices, has_initial_state = _prefill_case(lengths, slots, initial)
    # dt_bias near -8 keeps decays above 0.999: state accumulates across every chunk.
    weights = _weights(
        k_heads, v_heads, seed=6, dt_bias=(-8.0, -6.0) if decay == "slow" else (-2.0, 2.0)
    )
    projected = _projected(sum(lengths), k_heads, v_heads, seed=7)
    state, conv_state = _pools(k_heads, v_heads, seed=8)
    read = {slot for slot, start in zip(slots, initial, strict=True) if slot >= 0 and start}
    written = {slot for slot in slots if slot >= 0}
    expected_y, expected_state, expected_conv = reference.prefill(
        projected, weights, k_heads, cu_seqlens, slot_indices, has_initial_state, state, conv_state
    )
    outcomes = []
    for value, seed in (("nan", 9), ("random", 10)):
        pools = [_poison(pool, read, value, seed) for pool in (state, conv_state)]
        before = [pool.clone() for pool in pools]
        with torch.inference_mode():
            y = op(projected, *weights.args(), cu_seqlens, slot_indices, has_initial_state, *pools)
        torch.cuda.synchronize()
        untouched = [row for row in range(SLOTS) if row not in written]
        for pool, old in zip(pools, before, strict=True):
            assert _same(pool[untouched], old[untouched])
        outcomes.append((y, *pools))
    (y, state_out, conv_out), (other_y, other_state, other_conv) = outcomes
    # Fresh sequences' rows (NaN in one run) are never read; stateless ones have none.
    assert _same(y, other_y)
    rows = sorted(written)
    assert _same(state_out[rows], other_state[rows]) and _same(conv_out[rows], other_conv[rows])
    _assert_close(y, expected_y, **Y_TOLERANCE)
    _assert_close(state_out[rows], expected_state[rows], **STATE_TOLERANCE)
    assert _same(conv_out[rows], expected_conv[rows])


@pytest.mark.parametrize(
    ("k_heads", "v_heads"), [(16, 32), pytest.param(16, 48, marks=pytest.mark.exhaustive)]
)
def test_chunked_prefill_then_decode_follows_one_sequence(k_heads, v_heads, prefill_provider):
    """A sequence prefilled in chunks of 1, 3, 4, 63, 64, 65 and 60 tokens, then decoded for
    four tokens, matches the reference run over the whole sequence."""
    prefill = _prefill_op(k_heads, v_heads, prefill_provider)
    if prefill is None:
        return
    chunks, decode_steps = [1, 3, 4, 63, 64, 65, 60], 4
    total = sum(chunks) + decode_steps
    weights = _weights(k_heads, v_heads, seed=11)
    projected = _projected(total, k_heads, v_heads, seed=12)
    state, conv_state = _pools(k_heads, v_heads, seed=13)
    slot = 5
    cu, slots, initial = _prefill_case([total], [slot], [False])
    expected_y, expected_state, expected_conv = reference.prefill(
        projected, weights, k_heads, cu, slots, initial, state, conv_state
    )
    state[slot] = float("nan")  # a recycled slot: never read before its first chunk
    conv_state[slot] = float("nan")
    decode = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    outputs, start = [], 0
    with torch.inference_mode():
        for index, length in enumerate(chunks):
            cu, slots, initial = _prefill_case([length], [slot], [index > 0])
            rows = projected[start : start + length]
            outputs.append(prefill(rows, *weights.args(), cu, slots, initial, state, conv_state))
            start += length
        slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
        for _ in range(decode_steps):
            outputs.append(
                decode(projected[start : start + 1], *weights.args(), slots, state, conv_state)
            )
            start += 1
    y = torch.cat(outputs)
    _assert_close(y, expected_y, **Y_TOLERANCE)
    _assert_close(state[slot], expected_state[slot], **STATE_TOLERANCE)
    assert _same(conv_state[slot], expected_conv[slot])


def _fla_core() -> SimpleNamespace:
    """fla-core's public entry points, imported after the provider configured it (constructing
    an operation need not load fla-core)."""
    _fla.load(GatedDeltaNetDecode.__name__, torch.device("cuda"))
    import fla.modules.conv.triton.ops as conv
    from fla.modules.conv import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    return SimpleNamespace(
        conv_update=conv.causal_conv1d_update,
        conv=causal_conv1d,
        gated_norm=rms_norm_gated,
        chunk=chunk_gated_delta_rule,
        recurrent=fused_recurrent_gated_delta_rule,
    )


def _split(projected, k_heads, v_heads):
    conv_dim, _ = _widths(k_heads, v_heads)
    value_width = v_heads * V
    x = projected[:, :conv_dim]
    z = projected[:, conv_dim : conv_dim + value_width]
    b = projected[:, conv_dim + value_width : conv_dim + value_width + v_heads]
    a = projected[:, conv_dim + value_width + v_heads :]
    return x, z, b.contiguous(), a.contiguous()


def test_decode_equals_fla_core_called_directly_on_dense_states():
    """The fla-core decode (the fallback where no fused kernel is accepted) adds nothing to
    fla-core: y and the pools equal its public decode chain run on dense copies of the batch's
    rows, bit for bit."""
    k_heads, v_heads, batch = 16, 32, 6
    fla = _fla_core()
    weights = _weights(k_heads, v_heads, seed=14)
    projected = _projected(batch, k_heads, v_heads, seed=15)
    slot_indices = torch.tensor([9, 1, 17, 4, 22, 13], dtype=torch.int32, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=16)
    rows = slot_indices.long()
    with torch.inference_mode():
        x, z, b, a = _split(projected, k_heads, v_heads)
        cache = torch.zeros(batch, x.shape[1], 4, dtype=torch.bfloat16, device="cuda")
        cache[:, :, 1:] = conv_state[rows].transpose(1, 2)
        mixed, _ = fla.conv_update(x, cache, weight=weights.conv_weight, activation="silu")
        q, k, v = mixed.split([k_heads * K, k_heads * K, v_heads * V], 1)
        o, final = fla.recurrent(
            q.reshape(batch, 1, k_heads, K),
            k.reshape(batch, 1, k_heads, K),
            v.reshape(batch, 1, v_heads, V),
            g=a.view(batch, 1, v_heads),
            beta=b.view(batch, 1, v_heads),
            initial_state=state[rows].clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=weights.A_log,
            dt_bias=weights.dt_bias,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
        )
        expected = fla.gated_norm(
            o.reshape(-1, V),
            z.reshape(-1, V),
            weights.norm_weight,
            None,
            activation="silu",
            eps=1e-6,
        ).view(batch, -1)
        decode = _fla.load(GatedDeltaNetDecode.__name__, torch.device("cuda")).decode
        history = conv_state[rows]
        y = decode(
            Geometry(k_heads, v_heads),
            projected,
            *weights.args(),
            slot_indices,
            state,
            conv_state,
        )
    torch.cuda.synchronize()
    assert _same(y, expected)
    assert _same(state[rows], final)
    assert _same(conv_state[rows], torch.cat((history[:, 1:], x[:, None]), 1))


def test_prefill_equals_fla_core_called_directly_on_dense_states():
    """The fla-core prefill adapter (our features, norm and slot glue around fla's chunk
    kernels) against fla-core's public chunked chain on dense states: y and the final states
    differ only by the rounding of the q/k L2 norm's and the RMSNorm's sums."""
    k_heads, v_heads = 16, 32
    if _prefill_op(k_heads, v_heads, "fla-core") is None:
        return
    fla = _fla_core()
    lengths = [1, 63, 64, 65, 130, 7]
    slots = [3, 7, 12, 11, 2, 20]
    cu_seqlens, slot_indices, has_initial_state = _prefill_case(lengths, slots, [True] * 6)
    weights = _weights(k_heads, v_heads, seed=17)
    rows_total = sum(lengths)
    projected = _projected(rows_total, k_heads, v_heads, seed=18)
    state, conv_state = _pools(k_heads, v_heads, seed=19)
    rows = slot_indices.long()
    with torch.inference_mode():
        x, z, b, a = _split(projected, k_heads, v_heads)
        cache = torch.zeros(len(lengths), x.shape[1], 4, dtype=torch.bfloat16, device="cuda")
        cache[:, :, 1:] = conv_state[rows].transpose(1, 2)
        mixed, _ = fla.conv(
            x[None],
            weight=weights.conv_weight,
            initial_state=cache,
            activation="silu",
            cu_seqlens=cu_seqlens.clone(),
        )
        q, k, v = (
            part.contiguous() for part in mixed[0].split([k_heads * K, k_heads * K, v_heads * V], 1)
        )
        with pinned_fla_reference(training=False):
            o, final = fla.chunk(
                q.view(1, -1, k_heads, K),
                k.view(1, -1, k_heads, K),
                v.view(1, -1, v_heads, V),
                g=a.view(1, -1, v_heads),
                beta=b.view(1, -1, v_heads),
                initial_state=state[rows].clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=True,
                cu_seqlens=cu_seqlens.clone(),
                use_gate_in_kernel=True,
                A_log=weights.A_log,
                dt_bias=weights.dt_bias,
            )
        expected = fla.gated_norm(
            o.reshape(-1, V),
            z.reshape(-1, V),
            weights.norm_weight,
            None,
            activation="silu",
            eps=1e-6,
        ).view(rows_total, -1)
        prefill = _fla.load(GatedDeltaNetPrefill.__name__, torch.device("cuda")).prefill
        y = prefill(
            Geometry(k_heads, v_heads),
            projected,
            *weights.args(),
            cu_seqlens,
            slot_indices,
            has_initial_state,
            state,
            conv_state,
        )
    torch.cuda.synchronize()
    _assert_close(y, expected, **ADAPTER_TOLERANCE)
    _assert_close(state[rows], final, **ADAPTER_TOLERANCE)


def _launches(call) -> int:
    """Kernels (and memsets/copies) one call puts on the GPU."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    return sum(event.device_type == torch.autograd.DeviceType.CUDA for event in trace.events())


@pytest.mark.parametrize(("provider", "captures"), [("fla-core", 0), ("triton", 0), ("triton", 1)])
def test_prefill_launches(provider, captures):
    """fla-core's adapter: our features, prepare, norm and commit launches around fla's gate,
    A zeroing, intra, w/u, state and output kernels (no captures). Our portable prefill:
    features, intra, state and norm, captures included."""
    k_heads, v_heads = 16, 32
    if provider == "fla-core":
        if _prefill_op(k_heads, v_heads, "fla-core") is None:
            return
        prefill = _fla.load(GatedDeltaNetPrefill.__name__, torch.device("cuda")).prefill
        expected = 10
    else:
        from ctikernels.gdn.triton.prefill import Prefill

        prefill, expected = Prefill(torch.device("cuda"), Geometry(k_heads, v_heads)), 4
    weights = _weights(k_heads, v_heads, seed=50)
    state, conv_state = _pools(k_heads, v_heads, seed=51)
    case = _prefill_case([300, 0, 64, 700], [3, 7, -1, 11], [True, False, True, False])
    projected = _projected(1064, k_heads, v_heads, seed=52)
    extra = {}
    if captures:
        extra = {"capture_rows": torch.tensor([12, -1, 15, 17], dtype=torch.int32).cuda()}

    def call():
        with torch.inference_mode():
            prefill(
                Geometry(k_heads, v_heads),
                projected,
                *weights.args(),
                *case,
                state,
                conv_state,
                **extra,
            )

    call()
    assert _launches(call) == expected


@pytest.mark.parametrize("operation", ["decode", "prefill"])
def test_calls_never_synchronize_with_the_host(operation, prefill_provider):
    k_heads, v_heads = 16, 32
    weights = _weights(k_heads, v_heads, seed=20)
    state, conv_state = _pools(k_heads, v_heads, seed=21)
    if operation == "decode":
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
        projected, slots = _projected(8, k_heads, v_heads, seed=22), _decode_slots(padding=[3])
        arguments = (projected, *weights.args(), slots, state, conv_state)
    else:
        op = _prefill_op(k_heads, v_heads, prefill_provider)
        if op is None:
            return
        cu, slots, initial = _prefill_case([5, 70, 64], [4, -1, 9], [True, False, False])
        projected = _projected(139, k_heads, v_heads, seed=22)
        arguments = (projected, *weights.args(), cu, slots, initial, state, conv_state)
    with torch.inference_mode():
        op(*arguments)  # compile and autotune first
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            op(*arguments)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def _capture(call):
    """A CUDA graph of ``call`` and its outputs, captured on a side stream."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = call()
    return graph, outputs


def test_decode_graph_follows_new_slots_padding_and_inputs_on_replay():
    k_heads, v_heads, batch = 16, 32, 8
    weights = _weights(k_heads, v_heads, seed=23)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=24)
    slot_indices = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=25)
    arguments = (projected, *weights.args(), slot_indices, state, conv_state)
    with torch.inference_mode():
        op(*arguments)  # the warm-up a capture needs; all padding: no pool row changes
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(*arguments))
    cases = [
        [3, 9, 1, 17, 4, 22, 13, 8],
        [-1] * batch,
        [5, -1, 6, -1, 7, -1, 2, 19],
        [21, 20, 19, 18, 17, 16, 15, 14],
    ]
    for case, slots in enumerate(cases):
        new_state, new_conv = _pools(k_heads, v_heads, seed=26 + case)
        new_projected = _projected(batch, k_heads, v_heads, seed=30 + case)
        new_slots = torch.tensor(slots, dtype=torch.int32, device="cuda")
        eager_state, eager_conv = new_state.clone(), new_conv.clone()
        with torch.inference_mode():
            expected = op(new_projected, *weights.args(), new_slots, eager_state, eager_conv)
        statics = (projected, slot_indices, state, conv_state)
        for static, value in zip(
            statics, (new_projected, new_slots, new_state, new_conv), strict=True
        ):
            static.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected), slots
        assert _same(state, eager_state) and _same(conv_state, eager_conv), slots


def test_prefill_graph_follows_new_lengths_slots_and_initial_states_on_replay(prefill_provider):
    k_heads, v_heads = 16, 32
    op = _prefill_op(k_heads, v_heads, prefill_provider)
    if op is None:
        return
    weights = _weights(k_heads, v_heads, seed=34)
    cu_seqlens, slot_indices, has_initial_state = _prefill_case(
        [64, 64, 64, 64], [1, 2, 3, 4], [False] * 4
    )
    projected = _projected(256, k_heads, v_heads, seed=35)
    state, conv_state = _pools(k_heads, v_heads, seed=36)
    arguments = (
        projected,
        *weights.args(),
        cu_seqlens,
        slot_indices,
        has_initial_state,
        state,
        conv_state,
    )
    with torch.inference_mode():
        op(*arguments)
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(*arguments))
    cases = [
        ([1, 63, 65, 127], [6, -1, 7, 8], [True, False, True, False]),
        ([200, 1, 1, 54], [10, 11, -1, 12], [False, True, False, True]),
        ([253, 1, 1, 1], [-1, -1, -1, -1], [True, True, False, False]),
    ]
    for case, (lengths, slots, initial) in enumerate(cases):
        new_cu, new_slots, new_initial = _prefill_case(lengths, slots, initial)
        new_state, new_conv = _pools(k_heads, v_heads, seed=37 + case)
        new_projected = _projected(256, k_heads, v_heads, seed=40 + case)
        eager_state, eager_conv = new_state.clone(), new_conv.clone()
        with torch.inference_mode():
            new = (new_projected, *weights.args(), new_cu, new_slots, new_initial)
            expected = op(*new, eager_state, eager_conv)
        statics = (projected, cu_seqlens, slot_indices, has_initial_state, state, conv_state)
        values = (new_projected, new_cu, new_slots, new_initial, new_state, new_conv)
        for static, value in zip(statics, values, strict=True):
            static.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected), lengths
        assert _same(state, eager_state) and _same(conv_state, eager_conv), lengths


def test_capturing_a_shape_that_never_ran_eagerly_is_refused():
    k_heads, v_heads = 16, 32
    weights = _weights(k_heads, v_heads, seed=43)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=44)
    slots = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    projected = _projected(4, k_heads, v_heads, seed=45)
    with torch.inference_mode():
        op(projected[:2], *weights.args(), slots[:2], state, conv_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match=r"shape \(4,\) once eagerly"):
            with torch.cuda.graph(graph):
                op(projected, *weights.args(), slots, state, conv_state)
    torch.cuda.synchronize()


@pytest.mark.parametrize("operation", ["decode", "prefill"])
@pytest.mark.parametrize("weight", ["conv_weight", "A_log", "dt_bias", "norm_weight"])
@pytest.mark.parametrize("grad_mode", [True, False])
def test_a_weight_that_requires_grad_is_rejected_before_any_device_work(
    operation, weight, grad_mode
):
    k_heads, v_heads = 16, 32
    weights = _weights(k_heads, v_heads, seed=47)
    weights = reference.Weights(
        **{
            name: value.clone().requires_grad_(name == weight)
            for name, value in vars(weights).items()
        }
    )
    state, conv_state = _pools(k_heads, v_heads, seed=48)
    before = state.clone(), conv_state.clone()
    if operation == "decode":
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
        slots = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        arguments = (
            _projected(4, k_heads, v_heads, seed=49),
            *weights.args(),
            slots,
            state,
            conv_state,
        )
    else:
        op = _prefill_op(k_heads, v_heads)
        if op is None:
            return
        cu, slots, initial = _prefill_case([4, 4], [1, 2], [False, True])
        projected = _projected(8, k_heads, v_heads, seed=49)
        arguments = (projected, *weights.args(), cu, slots, initial, state, conv_state)
    with (
        torch.set_grad_enabled(grad_mode),
        pytest.raises(ValueError, match=f"{weight} is a frozen"),
    ):
        op(*arguments)
    torch.cuda.synchronize()
    assert _same(state, before[0]) and _same(conv_state, before[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_row_copy_reads_and_writes_only_active_entries(dtype):
    from ctikernels.gdn._rows import gather_rows, write_rows

    generator = torch.Generator().manual_seed(50)
    pool = torch.randn(16, 3, 256, generator=generator).to(dtype).cuda()
    pool[9] = float("nan")  # an inactive entry's own slot
    slot_indices = torch.tensor([7, -1, 3, 9, -1, 12], dtype=torch.int32, device="cuda")
    active = torch.tensor([True, False, True, False, False, True], device="cuda")
    before = pool.clone()
    rows = gather_rows(pool, slot_indices, active)
    expected = torch.zeros_like(rows)
    expected[[0, 2, 5]] = before[[7, 3, 12]]
    assert _same(rows, expected) and _same(pool, before)
    new = torch.randn(6, 3, 256, generator=generator).to(dtype).cuda()
    write_rows(pool, slot_indices, new, active)
    expected = before.clone()
    expected[[7, 3, 12]] = new[[0, 2, 5]]
    assert _same(pool, expected)
    # Out-of-range rows are skipped like padding.
    far = torch.tensor([16, 40, -1, 2**30, 15, -7], dtype=torch.int32, device="cuda")
    everything = torch.ones(6, dtype=torch.bool, device="cuda")
    gathered = gather_rows(pool, far, everything)
    assert _same(gathered[[0, 1, 2, 3, 5]], torch.zeros_like(gathered[[0, 1, 2, 3, 5]]))
    assert _same(gathered[4], pool[15])
    write_rows(pool, far, new, everything)
    expected[15] = new[4]
    assert _same(pool, expected)


def _in_order(fixed_first: bool, path: str) -> None:
    """In a fresh process: one fixed-config L2 launch and the fla-core provider's decode and
    prefill, in the given order; saves every output to ``path``."""
    cuda = torch.device("cuda")
    k_heads, v_heads = 16, 32
    x = torch.randn(1, 64, 4, K, generator=torch.Generator().manual_seed(30)).bfloat16().cuda()

    def fixed():
        return {"l2norm": _l2norm.l2norm_forward(x)[0]}

    def provider():
        weights = _weights(k_heads, v_heads, seed=31)
        state, conv_state = _pools(k_heads, v_heads, seed=32)
        slots = torch.tensor([9, 1, 17, 4, 22, 13], dtype=torch.int32, device=cuda)
        y_decode = _fla.load(GatedDeltaNetDecode.__name__, cuda).decode(
            Geometry(k_heads, v_heads),
            _projected(6, k_heads, v_heads, seed=33),
            *weights.args(),
            slots,
            state,
            conv_state,
        )
        y_prefill = _fla.load(GatedDeltaNetPrefill.__name__, cuda).prefill(
            Geometry(k_heads, v_heads),
            _projected(75, k_heads, v_heads, seed=34),
            *weights.args(),
            *_prefill_case([5, 70], [3, 11], [True, False]),
            state,
            conv_state,
        )
        return {"decode": y_decode, "prefill": y_prefill, "state": state, "conv": conv_state}

    with torch.inference_mode():
        outputs = {**fixed(), **provider()} if fixed_first else {**provider(), **fixed()}
    torch.save({name: tensor.cpu() for name, tensor in outputs.items()}, path)


@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="fla-core is not installed")
def test_fixed_config_kernels_first_leave_the_fla_core_provider_loadable(tmp_path):
    """GEN-GDN's fixed-config kernels launched first in a fresh process import fla-core through
    the provider's setup point, so its decode and prefill load afterwards and give the same bits
    as in the other order."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FLA_")}
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), *sys.path]
    )
    outputs = {}
    for fixed_first in (True, False):
        path = tmp_path / f"{fixed_first}.pt"
        code = f"from tests.gpu.test_gdn import _in_order; _in_order({fixed_first}, {str(path)!r})"
        run = run_child([sys.executable, "-c", code], env=environment, timeout=600, check=False)
        assert run.returncode == 0, run.stderr[-3000:]
        outputs[fixed_first] = torch.load(path)
    for name, tensor in outputs[True].items():
        assert _same(tensor, outputs[False][name]), name
