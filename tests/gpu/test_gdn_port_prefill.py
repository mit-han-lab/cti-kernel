# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CP2 contract and precision relative to CP1 glue and the independent FP64 oracle."""

from dataclasses import replace

import pytest
import torch

from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120
from ctikernels.gdn.chunk_ws.prefill import PortPrefill
from ctikernels.gdn.inference import GatedDeltaNetPrefill

from . import _gdn_reference as reference
from .test_gdn import Y_TOLERANCE, _assert_close
from .test_gdn_chunk_ws import _reason, _same_bits

_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
LENGTHS = ((16,), (64,), (100,), (1024,), (32, 96, 200, 184), (1, 2, 3, 1))


def make_case(hk, hv, lengths, seed=13, device="cuda"):
    from ctikernels.gdn._contract import Geometry

    geo = Geometry(hk, hv)
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, device=device, dtype=dtype)

    weights = reference.Weights(
        (rand(geo.conv_dim, 4) * 0.2).bfloat16(),
        rand(hv) * 0.3,
        rand(hv) * 0.2 - 2,
        rand(128) * 0.1 + 1,
    )
    x = rand(sum(lengths), geo.proj_width, dtype=torch.bfloat16)
    n = len(lengths)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    # Keep one padding row in varlen cases and unused slots at both ends.
    slots = torch.tensor(
        [i + 1 if i != 1 else -1 for i in range(n)], dtype=torch.int32, device=device
    )
    initial = torch.tensor([i % 2 == 0 for i in range(n)], device=device)
    pools = (rand(n + 2, hv, 128, 128) * 0.02, rand(n + 2, 3, geo.conv_dim, dtype=torch.bfloat16))
    return x, weights, (cu, slots, initial), pools


def cp1_glue(x, weights, hk, meta, pools):
    from benchmark.kernel_tuning.gdn_prefill_port.cp1_precision import finish, prepare

    q, k, v, g, beta = prepare(x, weights, hk, meta, pools[1])
    # Match fused numerics: round O to BF16 before the FP32 norm.
    o = torch.empty_like(v)
    ChunkPrefillSm120(x.device, hk, v.shape[1])(
        q, k, v, g, beta, o, pools[0], meta[1], meta[2], meta[0], 128**-0.5
    )
    return finish(o, x, weights)


def error(got, exact):
    return ((got.double() - exact).square().mean()).sqrt().item()


@pytest.fixture(scope="module")
def operators():
    return {}


@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("lengths", LENGTHS)
def test_port_reference(operators, hk, hv, lengths):
    key = hk, hv, "reference"
    if key not in operators:
        operators[key] = PortPrefill(hk, hv, "cuda", cp=False)
    x, weights, meta, base = make_case(hk, hv, lengths)
    cu, slots, initial = meta
    pools = tuple(t.clone() for t in base)
    cp1_pools = tuple(t.clone() for t in base)
    # Poison every initial state/history that must not be read.
    owned = slots[slots >= 0].tolist()
    for n, slot in enumerate(slots.tolist()):
        if slot >= 0 and not initial[n].item():
            pools[0][slot].fill_(float("nan"))
            pools[1][slot].fill_(float("nan"))
    out_storage = torch.full(
        (x.shape[0] + 64, hv * 128), 123, dtype=torch.bfloat16, device=x.device
    )
    out = out_storage[: x.shape[0]]
    assert operators[key](x, *weights.args(), *meta, *pools, out=out) is out
    assert (out_storage[x.shape[0] :] == 123).all()
    exact_y, exact_state, exact_conv = reference.prefill(x, weights, hk, *meta, *base)
    baseline = cp1_glue(x, weights, hk, meta, cp1_pools)
    for got, expected, cp1 in (
        (out, exact_y, baseline),
        (pools[0][owned], exact_state[owned], cp1_pools[0][owned]),
    ):
        assert torch.isfinite(got).all()
        assert error(got, expected) <= 1.05 * error(cp1, expected) + 1e-7
    _same_bits(pools[1], exact_conv)
    untouched = [slot for slot in range(base[0].shape[0]) if slot not in owned]
    _same_bits(pools[0][untouched], base[0][untouched])
    # Compare with the product's v1 path without making it the accuracy oracle.
    v1_pools = tuple(t.clone() for t in base)
    v1 = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=x.device)
    v1_y = v1(x, *weights.args(), *meta, *v1_pools)
    v1_rms = v1_y.double().square().mean().sqrt().item()
    assert error(out, v1_y.double()) <= 0.02 * v1_rms + 1e-4
    state_rms = v1_pools[0][owned].double().square().mean().sqrt().item()
    assert error(pools[0][owned], v1_pools[0][owned].double()) <= 0.02 * state_rms + 1e-5
    _same_bits(pools[1], v1_pools[1])


@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
def test_port_graph_new_metadata(hk, hv):
    lengths = (32, 96, 200, 184)
    x, weights, meta, base = make_case(hk, hv, lengths)
    op = PortPrefill(hk, hv, "cuda")
    pools = tuple(t.clone() for t in base)
    out = torch.empty(x.shape[0], hv * 128, dtype=torch.bfloat16, device=x.device)
    op(x, *weights.args(), *meta, *pools, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, *weights.args(), *meta, *pools, out=out)
    for new_lengths in ((1, 2, 100, 409), (128, 128, 128, 128)):
        cu, slots, initial = meta
        cu.copy_(torch.tensor([0, *torch.tensor(new_lengths).cumsum(0).tolist()], device=x.device))
        slots.copy_(torch.tensor([4, 2, -1, 1], device=x.device))
        initial.copy_(torch.tensor([True, False, True, True], device=x.device))
        eager_pools = tuple(t.clone() for t in base)
        eager = op(x, *weights.args(), *meta, *eager_pools)
        for dst, src in zip(pools, base, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        _same_bits(out, eager)
        for got, expected in zip(pools, eager_pools, strict=True):
            _same_bits(got, expected)
    assert len(op.compiled) == 1


@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("initial", [False, True])
def test_port_short_history_and_strided_rows(operators, length, initial):
    key = (16, 32, "short")
    if key not in operators:
        operators[key] = PortPrefill(16, 32, "cuda")
    op = operators[key]
    x, weights, meta, base = make_case(16, 32, (length,))
    backing = torch.empty(length * (x.shape[1] + 8) + 8, device=x.device, dtype=x.dtype)
    strided = backing[8:].view(length, x.shape[1] + 8)[:, : x.shape[1]]
    strided.copy_(x)
    meta[2].fill_(initial)
    pools = tuple(t.clone() for t in base)
    y = op(strided, *weights.args(), *meta, *pools)
    exact = reference.prefill(x, weights, 16, *meta, *base)
    _same_bits(pools[1], exact[2])
    assert torch.isfinite(y).all()
    meta[1].fill_(-1)
    untouched = tuple(t.clone() for t in pools)
    y = op(strided, *weights.args(), *meta, *pools)
    expected, _, _ = reference.prefill(strided, weights, 16, *meta, *pools)
    assert error(y, expected) < 0.02
    for got, before in zip(pools, untouched, strict=True):
        _same_bits(got, before)


def test_port_empty_pools_stateless(operators):
    key = (16, 32, "short")
    if key not in operators:
        operators[key] = PortPrefill(16, 32, "cuda")
    x, weights, meta, pools = make_case(16, 32, (2,))
    meta[1].fill_(-1)
    y = operators[key](x, *weights.args(), *meta, pools[0][:0], pools[1][:0])
    expected, _, _ = reference.prefill(x, weights, 16, *meta, pools[0][:0], pools[1][:0])
    assert error(y, expected) < 0.02


@pytest.mark.parametrize("prefetch", [2, 4, 8])
@pytest.mark.parametrize("producer_registers", [56, 64, 72, 80])
def test_port_tuning_knobs(operators, prefetch, producer_registers):
    # Cross warp-row, chunk and short-history boundaries, with a padded projected
    # row stride and both owned and padding slots. Every setting must preserve
    # the default path's output and pool updates exactly.
    hk, hv = 16, 48
    key = (hk, hv, "tuning-baseline")
    if key not in operators:
        operators[key] = PortPrefill(hk, hv, "cuda")
    x, weights, meta, base = make_case(hk, hv, (65, 3, 2, 97))
    storage = torch.empty(x.shape[0], x.shape[1] + 8, dtype=x.dtype, device=x.device)
    storage[:, :-8].copy_(x)
    x = storage[:, :-8]
    expected_pools = tuple(t.clone() for t in base)
    actual_pools = tuple(t.clone() for t in base)
    expected = operators[key](x, *weights.args(), *meta, *expected_pools)
    op = PortPrefill(hk, hv, "cuda", prefetch=prefetch, producer_registers=producer_registers)
    actual = op(x, *weights.args(), *meta, *actual_pools)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_pools, expected_pools, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("misalignment", ["base", "stride"])
def test_port_rejects_projected_tma_misalignment(operators, misalignment):
    key = (16, 32, "short")
    if key not in operators:
        operators[key] = PortPrefill(16, 32, "cuda")
    x, weights, meta, pools = make_case(16, 32, (65,))
    rows, width = x.shape
    if misalignment == "base":
        backing = torch.empty(rows * width + 1, dtype=x.dtype, device=x.device)
        invalid = backing[1:].view(rows, width)
        message = "projected TMA base must be 16-byte aligned"
    else:
        backing = torch.empty(rows, width + 1, dtype=x.dtype, device=x.device)
        invalid = backing[:, :width]
        message = "projected TMA row stride must be a multiple of 8"
    invalid.copy_(x)
    before = tuple(t.clone() for t in pools)
    with pytest.raises(ValueError, match=message):
        operators[key](invalid, *weights.args(), *meta, *pools)
    for got, want in zip(pools, before, strict=True):
        _same_bits(got, want)


@pytest.mark.parametrize("segments", [1, 4])
@pytest.mark.parametrize("remap_y", [False, True])
def test_port_q_lookahead(segments, remap_y):
    # A four-chunk tail catches cursor/phase errors, and CP exercises MN's omitted Q plus SEG.
    x, weights, meta, base = make_case(16, 32, (225,))
    expected_pools = tuple(t.clone() for t in base)
    actual_pools = tuple(t.clone() for t in base)
    baseline = PortPrefill(16, 32, "cuda", cp_segments=segments)
    ahead = PortPrefill(16, 32, "cuda", cp_segments=segments, q_lookahead=True, watchdog=True)
    expected = baseline(x, *weights.args(), *meta, *expected_pools)
    actual = ahead(x, *weights.args(), *meta, *actual_pools)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_pools, expected_pools, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("hv", [32, 48])
def test_port_remapped_y_varlen_and_misaligned_norm(hv):
    x, weights, meta, base = make_case(16, hv, (1, 3, 17, 65, 97))
    backing = torch.empty(129, dtype=torch.float32, device=x.device)
    backing[1:].copy_(weights.norm_weight)
    weights = replace(weights, norm_weight=backing[1:])
    expected_pools = tuple(t.clone() for t in base)
    actual_pools = tuple(t.clone() for t in base)
    baseline = PortPrefill(16, hv, "cuda", cp=False)
    remapped = PortPrefill(16, hv, "cuda", cp=False, remap_y=True)
    expected = baseline(x, *weights.args(), *meta, *expected_pools)
    actual = remapped(x, *weights.args(), *meta, *actual_pools)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_pools, expected_pools, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("initial", [False, True])
def test_stateless_current_tap(cp, initial):
    """Review reproduction: slot -1 must produce ~silu(1), even with empty pools."""
    x, weights, meta, base = make_case(16, 32, (128,))
    x.fill_(1)
    weights.conv_weight.zero_()
    weights.conv_weight[:, 3] = 1
    weights.norm_weight.fill_(1)
    meta[1].fill_(-1)
    meta[2].fill_(initial)
    pools = tuple(t[:0] for t in base)
    op = PortPrefill(16, 32, x.device, cp=cp, cp_segments=2)
    actual = op(x, *weights.args(), *meta, *pools)
    expected, _, _ = reference.prefill(x, weights, 16, *meta, *pools)
    torch.testing.assert_close(actual.double(), expected, atol=0.006, rtol=0.006)
    assert (actual > 0.7).all()


@pytest.mark.parametrize("lengths", [(0,), (0, 0, 0, 0), (0, 65, 0, 1, 0), (1, 0, 2)])
@pytest.mark.parametrize("initial", [False, True])
def test_empty_sequences(lengths, initial):
    x, weights, meta, base = make_case(8, 16, lengths)
    meta[2].fill_(initial)
    # A slot -1 empty extend and real empty slots coexist; unused rows are sentinels.
    if len(lengths) > 1:
        meta[1][0] = -1
        meta[1][1:] = torch.arange(2, len(lengths) + 1, device=x.device)
    pools = tuple(t.clone() for t in base)
    if not initial:
        for slot in meta[1].tolist():
            if slot >= 0:
                pools[0][slot].fill_(float("nan"))
                pools[1][slot].fill_(float("nan"))
    op = PortPrefill(8, 16, x.device)
    out = op(x, *weights.args(), *meta, *pools)
    exact_y, exact_state, exact_history = reference.prefill(x, weights, 8, *meta, *base)
    if out.numel():
        assert error(out, exact_y) < 0.02
    torch.testing.assert_close(pools[0].double(), exact_state, atol=0.002, rtol=0.02)
    _same_bits(pools[1], exact_history)
    cu = meta[0].tolist()
    for n, slot in enumerate(meta[1].tolist()):
        if slot >= 0 and cu[n] == cu[n + 1]:
            if initial:
                _same_bits(pools[0][slot], base[0][slot])
                _same_bits(pools[1][slot], base[1][slot])
            else:
                assert torch.count_nonzero(pools[0][slot]) == 0
                assert torch.count_nonzero(pools[1][slot]) == 0


def test_graph_lengths_change_to_and_from_zero():
    x, weights, meta, base = make_case(8, 16, (0, 65, 0, 63))
    meta[1].copy_(torch.tensor([0, 1, -1, 3], device=x.device))
    op = PortPrefill(8, 16, x.device)
    pools = tuple(t.clone() for t in base)
    out = op(x, *weights.args(), *meta, *pools)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, *weights.args(), *meta, *pools, out=out)
    for lengths in ((32, 0, 96, 0), (0, 128, 0, 0), (1, 0, 0, 127)):
        meta[0].copy_(torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=x.device))
        meta[2].logical_not_()
        x.normal_()
        eager_pools = tuple(t.clone() for t in base)
        eager = op(x, *weights.args(), *meta, *eager_pools)
        for dst, src in zip(pools, base, strict=True):
            dst.copy_(src)
        graph.replay()
        _same_bits(out, eager)
        for got, expected in zip(pools, eager_pools, strict=True):
            _same_bits(got, expected)


@pytest.mark.parametrize("cp", [False, True])
def test_concurrent_streams_match_serial(cp):
    # Uneven packed tails exercise mutable TMA descriptors; N=1 exercises CP W.
    lengths = (4096,) if cp else (65, 127, 1000, 2049)
    cases = [make_case(8, 16, lengths, seed=seed) for seed in (29, 71)]
    op = PortPrefill(8, 16, "cuda", cp=cp, cp_segments=4)
    serial = []
    for x, weights, meta, base in cases:
        pools = tuple(t.clone() for t in base)
        serial.append((op(x, *weights.args(), *meta, *pools), pools))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    # Keep launches queued independently, with no ordering between the two streams.
    for _ in range(3):
        runs = []
        for stream, (x, weights, meta, base) in zip(streams, cases, strict=True):
            with torch.cuda.stream(stream):
                pools = tuple(t.clone() for t in base)
                runs.append((op(x, *weights.args(), *meta, *pools), pools))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for (out, pools), (expected, expected_pools) in zip(runs, serial, strict=True):
            _same_bits(out, expected)
            for got, want in zip(pools, expected_pools, strict=True):
                _same_bits(got, want)


@pytest.mark.parametrize("length,segments", [(200, 1), (1024, 3)])
def test_large_slot_pool_addressing(length, segments):
    """Slot 2800 at Hv=48: slot * Hv * 16384 elements passes 2^31 (int32 overflow from 2731)."""
    hk, hv, slot = 16, 48, 2800
    free, _ = torch.cuda.mem_get_info()
    if free < (slot + 1) * hv * 128 * 128 * 4 * 1.3:
        pytest.skip("needs ~9 GB free device memory for the large pool")
    x, weights, _, small = make_case(hk, hv, (length,), seed=29)
    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    initial = torch.tensor([True], device="cuda")
    op = PortPrefill(hk, hv, "cuda", cp_segments=segments)
    # Reference: the same call on slot 0 of a one-row pool.
    ref_pools = (small[0][:1].clone(), small[1][:1].clone())
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    ref_y = op(x, *weights.args(), cu, zero, initial, *ref_pools)
    state = torch.full((slot + 1, hv, 128, 128), 7.0, device="cuda")
    history = torch.full((slot + 1, 3, small[1].shape[2]), 3.0, device="cuda", dtype=torch.bfloat16)
    state[slot].copy_(small[0][0])
    history[slot].copy_(small[1][0])
    slots = torch.full((1,), slot, dtype=torch.int32, device="cuda")
    y = op(x, *weights.args(), cu, slots, initial, state, history)
    _same_bits(y, ref_y)
    _same_bits(state[slot], ref_pools[0][0])
    _same_bits(history[slot], ref_pools[1][0])
    # Wrapped int32 offsets would land in low rows: they must stay untouched.
    assert bool((state[:slot] == 7.0).all()) and bool((history[:slot] == 3.0).all())
    del state, history


@pytest.mark.parametrize("length", [64, 200])
@pytest.mark.parametrize("initial", [False, True])
def test_port_default_scaled_dynamic_range(operators, length, initial):
    hk, hv = 8, 16
    x, weights, meta, base = make_case(hk, hv, (length,))
    gen = torch.Generator(device=x.device).manual_seed(19)
    exponents = torch.randint(-10, 11, (length, 1), generator=gen, device=x.device)
    # Only the pre-convolution Q/K/V columns vary over 2**[-10, 10].
    conv_dim = weights.conv_weight.shape[0]
    x[:, :conv_dim] = (x[:, :conv_dim].float() * 2.0**exponents).bfloat16()
    meta[2].fill_(initial)
    key = (hk, hv, "default-scaled")
    if key not in operators:
        operators[key] = PortPrefill(hk, hv, "cuda")
    pools = tuple(t.clone() for t in base)
    y = operators[key](x, *weights.args(), *meta, *pools)
    exact_y, exact_state, exact_history = reference.prefill(x, weights, hk, *meta, *base)
    owned = meta[1].long()
    got, expected = pools[0][owned].double(), exact_state[owned]
    assert torch.isfinite(got).all()
    # Pool layout is [slot, Hv, V, K]; each V row gets its own reference RMS.
    row_rms = expected.square().mean(-1).sqrt().clamp_min(1e-30)
    relative = (got - expected).abs().amax(-1) / row_rms
    assert relative.max().item() <= 0.25
    _assert_close(y, exact_y, **Y_TOLERANCE)
    _same_bits(pools[1], exact_history)
