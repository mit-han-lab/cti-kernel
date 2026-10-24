# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn's fused Triton decode on the devices it is accepted on (RTX 5090, RTX PRO
6000, GB10): its resolution there, every configuration against the FP64
reference and against fla-core's error, heavy and zero-key inputs, decode chains against prefill
and the training forward, the scratch protocols across calls, configurations, streams and CUDA
graphs (with a long replay stress per configuration class), and 64-bit pool addressing."""

from __future__ import annotations

import gc
import time
import weakref

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetTraining, _fla, _triton_decode
from ctikernels.gdn._contract import Geometry

from . import _gdn_reference as reference
from ._exhaustive import ci_cases
from .test_gdn import (
    STATE_TOLERANCE,
    Y_TOLERANCE,
    _assert_close,
    _bits,
    _poison,
    _pools,
    _prefill_case,
    _prefill_op,
    _projected,
    _same,
    _weights,
)

K = V = 128
GEOMETRIES = [(8, 16), (8, 24), (16, 32), (16, 48)]
# Batch sizes reaching every configuration of the RTX 5090 (170 SMs), the RTX PRO 6000 (188)
# and GB10 (48) at these geometries: quarters with finishers, the three-CTA split, 64-row flag
# and finisher tiles in one and several waves, whole heads and 64/8 tiles; 64 and 256 are
# engine batch sizes. Other devices' classes: test_every_configuration_class_of_this_device.
BATCHES = [1, 2, 3, 4, 6, 8, 12, 14, 16, 24, 40, 64, 256]
# The precision gate's hard floor on the conditioned error (BF16 ULP).
HARD_FLOOR_ULP = 4.0
# Errors within FP32's smallest normal count as exact (the gate's absolute floor).
ABSOLUTE_FLOOR = 2.0**-126
# The fused path on ordinary data: FP32 throughout, y rounded once to BF16.
ORDINARY_ULP = 1.0


fused_only = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < _triton_decode.MIN_CAPABILITY,
    reason="the fused decode serves compute capability 8.9 and newer",
)


def _launch_traits(device: torch.device) -> dict:
    """decode_plan's device arguments, as TritonDecode derives them."""
    return {
        "sms": torch.cuda.get_device_properties(device).multi_processor_count,
        "pdl": torch.cuda.get_device_capability(device) >= _triton_decode.PDL_CAPABILITY,
    }


def _config_classes() -> dict[tuple, tuple[int, int, int]]:
    """The first (Hk, Hv, batch), B <= 256, of each launch configuration on this device."""
    device = torch.device("cuda")
    traits = _launch_traits(device)
    sms = traits["sms"]
    classes = {}
    for k_heads, v_heads in GEOMETRIES:
        for batch in range(1, 257):
            plan = _triton_decode.decode_plan(Geometry(k_heads, v_heads), batch, 1, True, **traits)
            arguments = {k: v for k, v in plan.arguments.items() if k not in ("SP0", "HK", "HV")}
            key = (
                tuple(sorted((k, str(v)) for k, v in arguments.items() if k != "OCC")),
                arguments["OCC"] > 0,
                plan.grid[0] <= sms,
            )
            classes.setdefault(key, (k_heads, v_heads, batch))
    return classes


def _max_ulp(got: torch.Tensor, exact: torch.Tensor) -> float:
    """Largest error in BF16 ULP at max(|exact|, row RMS / 8), as the precision gate measures
    it without an op-specific conditioning (rows along the last dimension): the spacing is at
    least BF16's smallest subnormal, and an error within FP32's smallest normal counts as exact."""
    got, exact = got.double(), exact.double()
    assert torch.isfinite(got).all()
    rms = exact.pow(2).mean(-1, keepdim=True).sqrt()
    scale = torch.maximum(exact.abs(), rms / 8)
    _, exponent = torch.frexp(scale)
    ulp = torch.pow(2.0, (exponent - 8).double()).clamp_min(2.0**-133)
    error = (got - exact).abs()
    error = torch.where(error <= ABSOLUTE_FLOOR, 0.0, error)
    return float((error / ulp).max()) if got.numel() else 0.0


def _rms_error(got: torch.Tensor, exact: torch.Tensor) -> float:
    return float((got.double() - exact.double()).pow(2).mean().sqrt())


def _slots(batch: int, slots: int, seed: int, padding=()) -> torch.Tensor:
    chosen = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))[:batch] + 1
    chosen[list(padding)] = -1
    return chosen.to(torch.int32).cuda()


def _fla_decode(geometry: Geometry, *arguments) -> torch.Tensor:
    kernel = _fla.load(GatedDeltaNetDecode.__name__, torch.device("cuda")).decode
    return kernel(geometry, *arguments)


@fused_only
def test_the_fused_decode_serves_its_device_types_and_geometries():
    """From compute capability 8.9 every geometry resolves to the fused kernel (a missing Gluon
    fails here, not silently into fla-core)."""
    device = torch.device("cuda")
    for k_heads, v_heads in GEOMETRIES:
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device=device)
        assert op.provider == "triton", (k_heads, v_heads, torch.cuda.get_device_name())
    assert GatedDeltaNetDecode(num_k_heads=1, num_v_heads=1, device=device).provider == "triton"


@fused_only
# CI covers both full and TP geometries, B=1/3/4/8 and the largest batch.
@pytest.mark.parametrize(
    "k_heads,v_heads,batch",
    ci_cases(
        GEOMETRIES,
        BATCHES,
        core=[(8, 16, 1), (8, 24, 3), (16, 32, 4), (16, 32, 8), (16, 48, 1), (16, 48, 40)],
    ),
)
def test_every_configuration_matches_the_fp64_reference(k_heads, v_heads, batch):
    _check_fp64(k_heads, v_heads, batch)


@fused_only
def test_every_configuration_class_of_this_device_matches_the_fp64_reference():
    """The batch sizes above are chosen for the sm12x devices; on any device, every launch
    configuration it reaches at B <= 256 runs once."""
    for k_heads, v_heads, batch in _config_classes().values():
        _check_fp64(k_heads, v_heads, batch)


def _check_fp64(k_heads: int, v_heads: int, batch: int) -> None:
    """y and the new state within one BF16 ULP of FP64 (conditioned), the history bitwise, no
    worse than fla-core's error on the same inputs, padding rows zero, and no other pool row
    read (NaN there) or written."""
    geometry = Geometry(k_heads, v_heads)
    slots = batch + 8
    weights = _weights(k_heads, v_heads, seed=100 + batch)
    projected = _projected(batch, k_heads, v_heads, seed=200 + batch)
    # Padding first, inside and last; at least one live row.
    padding = sorted({0, batch // 2, batch - 1} - {1}) if batch > 2 else [1][: batch - 1]
    slot_indices = _slots(batch, slots, seed=300 + batch, padding=padding)
    active = sorted(set(slot_indices[slot_indices >= 0].tolist()))
    state, conv_state = _pools(k_heads, v_heads, seed=400 + batch, slots=slots)
    expected_y, expected_state, expected_conv = reference.decode(
        projected, weights, k_heads, slot_indices, state, conv_state
    )
    pools = [_poison(pool, set(active), "nan", 500 + batch) for pool in (state, conv_state)]
    before = [pool.clone() for pool in pools]
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    with torch.inference_mode():
        y = op(projected, *weights.args(), slot_indices, *pools)
    torch.cuda.synchronize()
    new_state, new_conv = pools
    untouched = [row for row in range(slots) if row not in active]
    for pool, old in zip(pools, before, strict=True):
        assert _same(pool[untouched], old[untouched])
    assert not y[padding].any()
    live = [row for row in range(batch) if row not in padding]
    y_ulp = _max_ulp(y[live].view(len(live), v_heads, V), expected_y[live].view(-1, v_heads, V))
    state_ulp = _max_ulp(new_state[active], expected_state[active])
    assert y_ulp <= ORDINARY_ULP and state_ulp <= ORDINARY_ULP, (y_ulp, state_ulp)
    assert _same(new_conv[active], expected_conv[active])

    fla_state, fla_conv = state.clone(), conv_state.clone()
    with torch.inference_mode():
        fla_y = _fla_decode(geometry, projected, *weights.args(), slot_indices, fla_state, fla_conv)
    torch.cuda.synchronize()
    for got, fla, exact in (
        (y[live], fla_y[live], expected_y[live]),
        (new_state[active], fla_state[active], expected_state[active]),
    ):
        assert _rms_error(got, exact) <= _rms_error(fla, exact) * 1.05 + 1e-9
        assert _max_ulp(got, exact) <= max(_max_ulp(fla, exact), ORDINARY_ULP)


def _heavy_case(k_heads: int, v_heads: int, batch: int, slots: int, kind: str, seed: int):
    """Inputs far from the ordinary ones: ``heavy`` scales projected rows, history and state by
    log-normal factors with decays from near 0 to near 1; ``zero_keys`` additionally zeroes the
    keys' columns of projected and of the history, so the keys are exactly zero."""
    generator = torch.Generator().manual_seed(seed)
    weights = _weights(k_heads, v_heads, seed=seed, dt_bias=(-6.0, 6.0))
    projected = _projected(batch, k_heads, v_heads, seed=seed + 1).float().cpu()
    projected *= torch.exp(1.5 * torch.randn(batch, 1, generator=generator))
    state, conv_state = (pool.cpu() for pool in _pools(k_heads, v_heads, seed + 2, slots))
    state *= torch.exp(2.0 * torch.randn(slots, v_heads, 1, 1, generator=generator))
    conv_state = conv_state.float() * torch.exp(torch.randn(slots, 3, 1, generator=generator))
    if kind == "zero_keys":
        keys = slice(k_heads * K, 2 * k_heads * K)
        projected[:, keys] = 0
        conv_state[:, :, keys] = 0
    return (
        weights,
        projected.to(torch.bfloat16).cuda(),
        state.cuda(),
        conv_state.to(torch.bfloat16).cuda(),
    )


@fused_only
# CI covers heavy/zero-key inputs at TP/full geometry and B=1/3/4/16.
@pytest.mark.parametrize(
    "k_heads,v_heads,batch,kind",
    ci_cases(
        [(8, 24), (16, 32)],
        [1, 2, 3, 4, 16],
        ["heavy", "zero_keys"],
        core=[
            (8, 24, 1, "heavy"),
            (8, 24, 3, "zero_keys"),
            (16, 32, 4, "heavy"),
            (16, 32, 16, "zero_keys"),
        ],
    ),
)
def test_heavy_and_zero_key_inputs_stay_within_the_hard_floor(k_heads, v_heads, batch, kind):
    slots = batch + 4
    weights, projected, state, conv_state = _heavy_case(
        k_heads, v_heads, batch, slots, kind, seed=600 + batch
    )
    slot_indices = _slots(batch, slots, seed=700 + batch)
    rows = slot_indices.long()
    expected_y, expected_state, expected_conv = reference.decode(
        projected, weights, k_heads, slot_indices, state, conv_state
    )
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    with torch.inference_mode():
        y = op(projected, *weights.args(), slot_indices, state, conv_state)
    torch.cuda.synchronize()
    y_ulp = _max_ulp(y.view(batch, v_heads, V), expected_y.view(batch, v_heads, V))
    state_ulp = _max_ulp(state[rows], expected_state[rows])
    assert y_ulp <= HARD_FLOOR_ULP and state_ulp <= HARD_FLOOR_ULP, (y_ulp, state_ulp)
    assert _same(conv_state[rows], expected_conv[rows])


def _self_cleaning_scratch_is_zero(state: torch.Tensor) -> None:
    """Under the wide rules (self-cleaning scratch) every word of the pool's scratch is zero
    between calls."""
    if _launch_traits(state.device)["sms"] < _triton_decode.WIDE_SMS:
        return
    for key, (scratch, _, _) in list(_triton_decode._POOLS._shared.items()):
        if key[1] == state.data_ptr():
            assert int(torch.count_nonzero(scratch)) == 0, key


@fused_only
@pytest.mark.parametrize(("k_heads", "v_heads"), [(8, 16), (16, 48)])
def test_consecutive_calls_across_configurations_keep_the_protocol(k_heads, v_heads):
    """Many calls on one pool, the batch size (so the configuration and the scratch set)
    changing between them and slots recurring: each step matches the FP64 step from the pool
    as the previous call left it."""
    slots = 48
    weights = _weights(k_heads, v_heads, seed=800)
    state, conv_state = _pools(k_heads, v_heads, seed=801, slots=slots)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    for step, batch in enumerate([1, 3, 1, 2, 16, 4, 1, 40, 2, 6, 1, 12, 3, 1, 24, 14, 1, 8]):
        projected = _projected(batch, k_heads, v_heads, seed=900 + step)
        slot_indices = _slots(batch, slots, seed=step % 3, padding=[0] if step % 4 == 1 else [])
        expected_y, expected_state, expected_conv = reference.decode(
            projected, weights, k_heads, slot_indices, state, conv_state
        )
        with torch.inference_mode():
            y = op(projected, *weights.args(), slot_indices, state, conv_state)
        torch.cuda.synchronize()
        rows = slot_indices[slot_indices >= 0].long()
        live = slot_indices >= 0
        y_ulp = _max_ulp(y[live].view(-1, v_heads, V), expected_y[live].view(-1, v_heads, V))
        state_ulp = _max_ulp(state[rows], expected_state[rows])
        assert y_ulp <= ORDINARY_ULP and state_ulp <= ORDINARY_ULP, (step, y_ulp, state_ulp)
        assert _same(conv_state[rows], expected_conv[rows]), step
        _self_cleaning_scratch_is_zero(state)


@fused_only
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8, 16, 64, 256])
def test_graphs_follow_new_slots_on_shared_and_graph_private_scratch(batch):
    """A graph captured on a pool no eager call has seen (graph-private scratch) and one
    captured on a pool eager calls use (shared scratch), each replayed on new slots, padding
    and data, interleaved with eager calls: every replay equals the eager call bitwise."""
    k_heads, v_heads, slots = 16, 32, batch + 8
    weights = _weights(k_heads, v_heads, seed=1000)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=1001)
    warm_state, warm_conv = _pools(k_heads, v_heads, seed=1002, slots=slots)
    slot_indices = _slots(batch, slots, seed=1003)
    with torch.inference_mode():
        op(projected, *weights.args(), slot_indices, warm_state, warm_conv)
    graphs = []
    for fresh in (True, False):
        state, conv_state = _pools(k_heads, v_heads, seed=1004, slots=slots)
        if not fresh:
            state, conv_state = warm_state, warm_conv
        arguments = (projected, *weights.args(), slot_indices, state, conv_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            y = op(*arguments)
        graphs.append((graph, y, state, conv_state))
    for case in range(6):
        padding = [case % batch] if case % 2 and batch > 1 else []
        new_slots = _slots(batch, slots, seed=1100 + case, padding=padding)
        new_projected = _projected(batch, k_heads, v_heads, seed=1200 + case)
        new_state, new_conv = _pools(k_heads, v_heads, seed=1300 + case, slots=slots)
        for graph, y, state, conv_state in graphs:
            eager_state, eager_conv = new_state.clone(), new_conv.clone()
            with torch.inference_mode():
                expected = op(new_projected, *weights.args(), new_slots, eager_state, eager_conv)
            projected.copy_(new_projected)
            slot_indices.copy_(new_slots)
            state.copy_(new_state)
            conv_state.copy_(new_conv)
            graph.replay()
            torch.cuda.synchronize()
            assert _same(y, expected), case
            assert _same(state, eager_state) and _same(conv_state, eager_conv), case
            # An eager call on the graph's pool between replays (shared scratch: the same
            # words the graph uses).
            with torch.inference_mode():
                op(projected, *weights.args(), slot_indices, state, conv_state)
            torch.cuda.synchronize()
            _self_cleaning_scratch_is_zero(state)


def _reuse_freed_memory() -> list[torch.Tensor]:
    """Allocations of the sizes scratch takes, filled with ones, so memory the allocator got
    back is handed out again and overwritten."""
    sizes = [1 << shift for shift in range(8, 21, 2)]
    return [torch.full((n,), -1, dtype=torch.int64, device="cuda") for n in sizes * 8]


@fused_only
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_a_graph_on_a_detached_pool_survives_the_first_tensor(batch):
    """The review's reproduction: warm with ``state1``, capture with ``state1.detach()``, delete
    ``state1`` and replay. The scratch follows the pool's storage, not the first tensor, so
    replays still equal eager calls bitwise after the allocator hands memory out again."""
    k_heads, v_heads, slots = 16, 32, 24
    weights = _weights(k_heads, v_heads, seed=1050)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=1051)
    slot_indices = _slots(batch, slots, seed=1052)
    state1, conv_state = _pools(k_heads, v_heads, seed=1053, slots=slots)
    with torch.inference_mode():
        op(projected, *weights.args(), slot_indices, state1, conv_state)
    state2 = state1.detach()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        y = op(projected, *weights.args(), slot_indices, state2, conv_state)
    del state1
    gc.collect()
    assert any(key[1] == state2.data_ptr() for key in _triton_decode._POOLS._shared)
    junk = _reuse_freed_memory()
    for case in range(4):
        new_slots = _slots(batch, slots, seed=1060 + case)
        new_projected = _projected(batch, k_heads, v_heads, seed=1070 + case)
        new_state, new_conv = _pools(k_heads, v_heads, seed=1080 + case, slots=slots)
        eager_state, eager_conv = new_state.clone(), new_conv.clone()
        with torch.inference_mode():
            expected = op(new_projected, *weights.args(), new_slots, eager_state, eager_conv)
        projected.copy_(new_projected)
        slot_indices.copy_(new_slots)
        state2.copy_(new_state)
        conv_state.copy_(new_conv)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected), case
        assert _same(state2, eager_state) and _same(conv_state, eager_conv), case
        _self_cleaning_scratch_is_zero(state2)
    del junk


@fused_only
def test_scratch_lives_while_the_graphs_that_captured_it_live():
    """A graph holds every scratch its kernels address, a fresh pool's graph-private scratch
    and a warm pool's shared one, after the pools' tensors are gone; destroying the graph
    releases them."""
    k_heads, v_heads, slots, batch = 16, 32, 8, 2
    weights = _weights(k_heads, v_heads, seed=1090)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=1091)
    slot_indices = _slots(batch, slots, seed=1092)
    warm = _pools(k_heads, v_heads, seed=1093, slots=slots)
    fresh = _pools(k_heads, v_heads, seed=1094, slots=slots)
    with torch.inference_mode():
        op(projected, *weights.args(), slot_indices, *warm)
    holds = _triton_decode._POOLS._graphs._graphs
    before = set(holds)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        op(projected, *weights.args(), slot_indices, *fresh)
        op(projected, *weights.args(), slot_indices, *warm)
    (capture,) = set(holds) - before
    held = holds[capture][1]
    shared = [
        scratch
        for key, (scratch, _, _) in _triton_decode._POOLS._shared.items()
        if key[1] == warm[0].data_ptr()
    ]
    assert shared and all(any(scratch is kept for kept in held) for scratch in shared)
    assert len(held) > len(shared)
    alive = [weakref.ref(scratch) for scratch in held]
    del warm, fresh, held, shared
    gc.collect()
    assert all(ref() is not None for ref in alive)
    del graph
    gc.collect()
    torch.cuda.synchronize()
    deadline = time.monotonic() + 10
    while capture in holds and time.monotonic() < deadline:
        with _triton_decode._POOLS._lock:
            _triton_decode._POOLS._graphs.drain()
        time.sleep(0.01)
    assert capture not in holds
    gc.collect()
    assert all(ref() is None for ref in alive)


@fused_only
def test_a_pool_used_on_another_stream_follows_its_scratch():
    k_heads, v_heads, slots, batch = 8, 24, 16, 2
    weights = _weights(k_heads, v_heads, seed=1400)
    state, conv_state = _pools(k_heads, v_heads, seed=1401, slots=slots)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    side = torch.cuda.Stream()
    for step in range(4):
        projected = _projected(batch, k_heads, v_heads, seed=1402 + step)
        slot_indices = _slots(batch, slots, seed=step)
        expected_y, expected_state, _ = reference.decode(
            projected, weights, k_heads, slot_indices, state, conv_state
        )
        stream = side if step % 2 else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            y = op(projected, *weights.args(), slot_indices, state, conv_state)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        rows = slot_indices.long()
        assert _max_ulp(y.view(batch, v_heads, V), expected_y.view(batch, v_heads, V)) <= 1.0
        assert _max_ulp(state[rows], expected_state[rows]) <= 1.0


@fused_only
def test_a_first_call_on_a_new_pool_never_synchronizes_with_the_host():
    k_heads, v_heads, batch = 16, 32, 1
    weights = _weights(k_heads, v_heads, seed=1500)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=1501)
    slot_indices = _slots(batch, 8, seed=1502)
    with torch.inference_mode():
        op(projected, *weights.args(), slot_indices, *_pools(k_heads, v_heads, 1503, 8))
        pools = _pools(k_heads, v_heads, 1504, 8)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            op(projected, *weights.args(), slot_indices, *pools)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


@fused_only
@pytest.mark.parametrize("batch", [1, 2, 6])
def test_slots_past_two_to_the_31_state_elements_are_addressed_in_64_bits(batch):
    """A pool of 8200 slots at (8, 16) holds more than 2^31 FP32 state elements: the last
    slots are reached exactly (an int32 offset would wrap)."""
    k_heads, v_heads, slots = 8, 16, 8200
    need = slots * v_heads * V * K * 4 + (2 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    weights = _weights(k_heads, v_heads, seed=1600)
    projected = _projected(batch, k_heads, v_heads, seed=1601)
    conv_dim = 2 * k_heads * K + v_heads * V
    state = torch.zeros(slots, v_heads, V, K, device="cuda")
    conv_state = torch.zeros(slots, 3, conv_dim, dtype=torch.bfloat16, device="cuda")
    slot_indices = torch.arange(slots - batch, slots, dtype=torch.int32, device="cuda")
    small_state, small_conv = _pools(k_heads, v_heads, seed=1602, slots=batch)
    state[slots - batch :] = small_state
    conv_state[slots - batch :] = small_conv
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    with torch.inference_mode():
        y = op(projected, *weights.args(), slot_indices, state, conv_state)
    torch.cuda.synchronize()
    local = torch.arange(batch, dtype=torch.int32, device="cuda")
    expected_y, expected_state, expected_conv = reference.decode(
        projected, weights, k_heads, local, small_state, small_conv
    )
    assert _max_ulp(y.view(batch, v_heads, V), expected_y.view(batch, v_heads, V)) <= 1.0
    assert _max_ulp(state[slots - batch :], expected_state) <= 1.0
    assert _same(conv_state[slots - batch :], expected_conv)
    for rows in (slice(0, 64), slice(slots - batch - 64, slots - batch)):
        assert not state[rows].any() and not conv_state[rows].any()


@fused_only
def test_capturing_a_layout_that_never_ran_eagerly_is_refused():
    """The fused decode compiles per row stride of ``projected``: after a warm-up on dense rows,
    capturing rows of a wider buffer is refused rather than compiled inside the capture."""
    k_heads, v_heads, batch = 16, 32, 2
    weights = _weights(k_heads, v_heads, seed=1700)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=1701, slots=8)
    slot_indices = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    projected = _projected(batch, k_heads, v_heads, seed=1702)
    wide = torch.zeros(batch, projected.shape[1] + 64, dtype=torch.bfloat16, device="cuda")
    wide[:, : projected.shape[1]] = projected
    strided = wide[:, : projected.shape[1]]
    with torch.inference_mode():
        op(projected, *weights.args(), slot_indices, state, conv_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match=r"shape \(2,\) once eagerly .*per input layout"):
            with torch.cuda.graph(graph):
                op(strided, *weights.args(), slot_indices, state, conv_state)
    torch.cuda.synchronize()


def _training_op(k_heads: int, v_heads: int) -> GatedDeltaNetTraining | None:
    """The training operation here, or None where fla-core is refused (asserted, as for
    prefill in ``_prefill_op``)."""
    capability = torch.cuda.get_device_capability()
    if (GatedDeltaNetTraining.__name__, capability) not in _fla.WRONG_ON:
        return GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    with pytest.raises(RuntimeError, match="is not supported by fla-core"):
        GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    return None


@fused_only
@pytest.mark.parametrize(("k_heads", "v_heads"), [(16, 32), (8, 24)])
@pytest.mark.parametrize("start", ["zero", "carried"])
def test_a_64_step_decode_chain_follows_prefill_and_the_training_forward(k_heads, v_heads, start):
    """Four sequences decoded for 64 steps from one start state. The chain stays within the hard
    floor of FP64 and is no worse than prefill (or, from a zero start, the training forward) on
    the same tokens; it differs from them by at most their own bound against FP64."""
    steps, batch, slots = 64, 4, 12
    weights = _weights(k_heads, v_heads, seed=1800)
    tokens = _projected(batch * steps, k_heads, v_heads, seed=1801)
    state, conv_state = _pools(k_heads, v_heads, seed=1802, slots=slots)
    rows = [3, 7, 1, 10]
    if start == "zero":
        state[rows] = 0
        conv_state[rows] = 0
    cu_seqlens, slot_indices, initial = _prefill_case([steps] * batch, rows, [True] * batch)
    expected_y, expected_state, _ = reference.prefill(
        tokens, weights, k_heads, cu_seqlens, slot_indices, initial, state, conv_state
    )
    decode = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    chain_state, chain_conv = state.clone(), conv_state.clone()
    per_step = tokens.view(batch, steps, -1)
    with torch.inference_mode():
        y = [
            decode(
                per_step[:, t].contiguous(), *weights.args(), slot_indices, chain_state, chain_conv
            )
            for t in range(steps)
        ]
    y = torch.stack(y, 1).flatten(0, 1)
    torch.cuda.synchronize()
    final = chain_state[rows]
    assert _max_ulp(y.view(-1, v_heads, V), expected_y.view(-1, v_heads, V)) <= HARD_FLOOR_ULP
    assert _max_ulp(final, expected_state[rows]) <= HARD_FLOOR_ULP

    others = {}
    prefill = _prefill_op(k_heads, v_heads)
    if prefill is not None:
        pools = state.clone(), conv_state.clone()
        with torch.inference_mode():
            others["prefill"] = (
                prefill(tokens, *weights.args(), cu_seqlens, slot_indices, initial, *pools),
                pools[0][rows],
            )
    training = _training_op(k_heads, v_heads) if start == "zero" else None
    if training is not None:
        with torch.inference_mode():
            y_train = training(tokens, *weights.args(), cu_seqlens, max_sequence_length=steps)
        others["training"] = (y_train, None)
    torch.cuda.synchronize()
    for name, (other_y, other_state) in others.items():
        assert _rms_error(y, expected_y) <= _rms_error(other_y, expected_y), name
        _assert_close(y, other_y, **Y_TOLERANCE)
        if other_state is not None:
            assert _rms_error(final, expected_state[rows]) <= _rms_error(
                other_state, expected_state[rows]
            ), name
            _assert_close(final, other_state, **STATE_TOLERANCE)


# Graph replays per configuration class in the stress run.
REPLAYS = 10_000


@fused_only
@pytest.mark.exhaustive
def test_long_graph_replay_runs_equal_eager_calls_in_every_configuration_class():
    """The cross-CTA exchange under load: per configuration class of this device, REPLAYS
    replays of a graph on a fresh pool (graph-private scratch) against as many eager calls on a
    copy (the pool's shared scratch), slots permuted and padded between steps. Every step's y
    and every touched pool row agree bitwise (checked on the device, read once at the end)."""
    for k_heads, v_heads, batch in _config_classes().values():
        slots = 2 * batch + 2
        weights = _weights(k_heads, v_heads, seed=1900)
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
        bank = 8
        inputs = _projected(bank * batch, k_heads, v_heads, seed=1901).view(bank, batch, -1)
        generator = torch.Generator().manual_seed(1902)
        orders = torch.stack(
            [torch.randperm(slots - 1, generator=generator)[:batch] + 1 for _ in range(bank)]
        )
        if batch > 1:
            orders[1::2, 0] = -1
        orders = orders.to(torch.int32).cuda()
        graph_pools = _pools(k_heads, v_heads, seed=1903, slots=slots)
        eager_pools = tuple(pool.clone() for pool in graph_pools)
        projected, slot_indices = inputs[0].clone(), orders[0].clone()
        with torch.inference_mode():
            # Warm the shape on a third pool so the graph's pool stays unseen.
            op(projected, *weights.args(), slot_indices, *_pools(k_heads, v_heads, 1904, slots))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = op(projected, *weights.args(), slot_indices, *graph_pools)
            mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
            for step in range(REPLAYS):
                projected.copy_(inputs[step % bank])
                slot_indices.copy_(orders[step % bank])
                graph.replay()
                expected = op(projected, *weights.args(), slot_indices, *eager_pools)
                touched = slot_indices.clamp_min(0).long()
                mismatches += (_bits(y) != _bits(expected)).sum()
                for got, want in zip(graph_pools, eager_pools, strict=True):
                    mismatches += (_bits(got[touched]) != _bits(want[touched])).sum()
        torch.cuda.synchronize()
        assert int(mismatches) == 0, (k_heads, v_heads, batch)
        assert all(_same(a, b) for a, b in zip(graph_pools, eager_pools, strict=True))


@fused_only
@pytest.mark.parametrize(("k_heads", "v_heads"), [(16, 32), (8, 24)])
def test_batch_invariant_rows_do_not_depend_on_their_batch(k_heads, v_heads):
    """batch_invariant=True: a row's y and new state are bitwise the same alone, in any batch
    up to 256 and next to padding, eager or replayed in a CUDA graph; and within one BF16 ULP of
    FP64 like every configuration."""
    full, slots = 256, 300
    weights = _weights(k_heads, v_heads, seed=2600)
    projected = _projected(full, k_heads, v_heads, seed=2601)
    slot_indices = _slots(full, slots, seed=2602)
    state, conv_state = _pools(k_heads, v_heads, seed=2603, slots=slots)
    op = GatedDeltaNetDecode(
        num_k_heads=k_heads, num_v_heads=v_heads, device="cuda", batch_invariant=True
    )

    def run(rows, padded=False):
        pools = state.clone(), conv_state.clone()
        x, s = projected[rows], slot_indices[rows]
        if padded:
            x = torch.cat((x, projected[:1]))
            s = torch.cat((s, s.new_full((1,), -1)))
        with torch.inference_mode():
            y = op(x, *weights.args(), s, *pools)
        return y[: len(rows)], pools[0]

    y_full, state_full = run(list(range(full)))
    for batch in (1, 2, 3, 4, 8, 16, 64):
        for rows in (list(range(batch)), list(range(full - batch, full))):
            for padded in (False, True):
                y, new_state = run(rows, padded)
                touched = slot_indices[rows].long()
                assert _same(y, y_full[rows]), (batch, padded)
                assert _same(new_state[touched], state_full[touched]), (batch, padded)
    rows = list(range(8))
    expected_y, _, _ = reference.decode(
        projected[rows], weights, k_heads, slot_indices[rows], state, conv_state
    )
    assert _max_ulp(y_full[rows].view(8, v_heads, V), expected_y.view(8, v_heads, V)) <= 1.0
    x, s = projected[:4].clone(), slot_indices[:4].clone()
    pools = state.clone(), conv_state.clone()
    with torch.inference_mode():
        op(x, *weights.args(), s, *_pools(k_heads, v_heads, 2604, slots))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = op(x, *weights.args(), s, *pools)
    graph.replay()
    torch.cuda.synchronize()
    assert _same(y, y_full[:4])


# Served compute capabilities without a qualifying device here, each with an SM count to plan at.
AOT_TARGETS = [(100, 148)]


@pytest.mark.parametrize(("arch", "sms"), AOT_TARGETS)
def test_the_decode_compiles_ahead_of_time_for_every_served_capability(arch, sms):
    """Compiles the Gluon decode to a cubin for sm100 without its device (served,
    never run here): batch sizes reaching quarters, halves and 64/8 tiles, and the
    batch-invariant whole heads."""
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.experimental.gluon._runtime import GluonASTSource

        from ctikernels.gdn._triton_decode_kernels import gdn_decode_kernel
    except ImportError as error:
        pytest.skip(f"Triton's Gluon is unavailable: {error}")
    hk, hv = 16, 32
    stride = (2 * hk + 2 * hv) * K + 2 * hv
    types = ("*bf16", "*i64", "*fp32", "*fp32", "*fp32", "*i32", "*fp32", "*bf16", "*bf16")
    signature = dict(
        zip(gdn_decode_kernel.arg_names[:12], (*types, "*i64", "*i64", "i32"), strict=True)
    )
    for batch, invariant in ((1, False), (4, False), (64, False), (8, True)):
        plan = _triton_decode.decode_plan(
            Geometry(hk, hv),
            batch,
            stride,
            True,
            sms=sms,
            pdl=arch >= 90,
            batch_invariant=invariant,
        )
        arguments = dict(plan.arguments)
        options = {
            k: arguments.pop(k) for k in ("num_warps", "launch_pdl", "maxnreg") if k in arguments
        }
        arguments["ROWIDX"] = False
        compiled = triton.compile(
            GluonASTSource(gdn_decode_kernel, signature, constexprs=arguments),
            target=GPUTarget("cuda", arch, 32),
            options=options,
        )
        assert compiled.asm["cubin"], (arch, batch, invariant)


@fused_only
def test_reset_scratch_outside_inference_mode_after_inference_mode_calls():
    """Scratch made by inference-mode calls is a normal tensor: reset_scratch outside inference
    mode (an engine switching back from training) succeeds, and later calls equal those on a
    fresh pool bitwise."""
    k_heads, v_heads, batch, slots = 16, 32, 4, 10
    weights = _weights(k_heads, v_heads, seed=2800)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=2801, slots=slots)
    with torch.inference_mode():
        for step in range(3):
            slot_indices = _slots(batch, slots, seed=2802 + step)
            op(
                _projected(batch, k_heads, v_heads, 2810 + step),
                *weights.args(),
                slot_indices,
                state,
                conv_state,
            )
    held = [
        s for key, (s, _, _) in _triton_decode._POOLS._shared.items() if key[1] == state.data_ptr()
    ]
    assert held and not any(s.is_inference() for s in held)
    op.reset_scratch(state)
    fresh = state.clone(), conv_state.clone()
    with torch.inference_mode():
        for step in range(3):
            slot_indices = _slots(batch, slots, seed=2820 + step)
            projected = _projected(batch, k_heads, v_heads, 2830 + step)
            y = op(projected, *weights.args(), slot_indices, state, conv_state)
            expected = op(projected, *weights.args(), slot_indices, *fresh)
            assert _same(y, expected), step
    torch.cuda.synchronize()
    assert _same(state, fresh[0]) and _same(conv_state, fresh[1])


@fused_only
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_reset_scratch_recovers_a_pool_whose_scratch_was_lost(batch):
    """A memory-saver pause/resume without backup loses the scratch's contents (simulated by
    filling it with garbage): after reset_scratch the next calls equal those on a fresh pool
    bitwise, for the epoch and the self-cleaning protocols alike."""
    k_heads, v_heads, slots = 16, 32, 2 * batch + 2
    weights = _weights(k_heads, v_heads, seed=2700)
    op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    state, conv_state = _pools(k_heads, v_heads, seed=2701, slots=slots)
    with torch.inference_mode():
        for step in range(3):
            slot_indices = _slots(batch, slots, seed=2702 + step)
            op(
                _projected(batch, k_heads, v_heads, 2710 + step),
                *weights.args(),
                slot_indices,
                state,
                conv_state,
            )
        torch.cuda.synchronize()
        held = [
            s
            for key, (s, _, _) in _triton_decode._POOLS._shared.items()
            if key[1] == state.data_ptr()
        ]
        assert held
        generator = torch.Generator(device="cuda").manual_seed(2720)
        for scratch in held:
            scratch.random_(generator=generator)
        op.reset_scratch(state)
        fresh = state.clone(), conv_state.clone()
        for step in range(3):
            slot_indices = _slots(batch, slots, seed=2730 + step)
            projected = _projected(batch, k_heads, v_heads, 2740 + step)
            y = op(projected, *weights.args(), slot_indices, state, conv_state)
            expected = op(projected, *weights.args(), slot_indices, *fresh)
            assert _same(y, expected), step
        torch.cuda.synchronize()
    assert _same(state, fresh[0]) and _same(conv_state, fresh[1])
