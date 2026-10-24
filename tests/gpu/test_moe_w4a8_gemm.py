# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Orchestrator-only mixed-MMA, stage and graph qualification.

No compilation exceptions are converted to skips. The full-shape fixture is
sparse in channels so an equivalent compact declared FP64 oracle remains cheap.
"""

import importlib.util

import pytest
import torch

from ctikernels.moe._common import ExpertWeights, decode_e2m1, decode_mxfp8
from ctikernels.moe.cute import _w4a8_fc1 as fc1
from ctikernels.moe.cute import _w4a8_fc2 as fc2
from ctikernels.moe.cute import _w4a8_finalize as finalize
from ctikernels.moe.cute._runtime import load
from ctikernels.moe.cute._w4a8 import Config, W4A8Experts
from ctikernels.moe.cute._w4a8_layout import Tile, choose_tile
from ctikernels.moe.cute._w4a8_model import interleave_weight_scales
from ctikernels.moe.cute._w4a8_schedule import ORDERS
from ctikernels.moe.torch._w4a8_tiled_reference import quantize, routed

from .._moe_w4a8_reference import (
    activation_interval,
    assert_attainable,
    atomic_bound,
    pack_scales,
    prepare,
    q1_cells,
    row_quantized,
    rows_bound,
    rows_combine,
    workspace,
)
from .._moe_w4a8_validation import TINY, U32, gemm_bound


@pytest.fixture(scope="module")
def runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required; run this module on the orchestrator")
    rt = load(torch.device("cuda"))
    if rt is None:
        pytest.skip("CuTe 4.8.0 and SM120/SM121 required")
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield rt
    torch.backends.cuda.matmul.allow_tf32 = before


def native_weights(k, intermediate, experts=128):
    generator = torch.Generator(device="cuda").manual_seed(381)

    def codes(shape):
        return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=generator)

    def scales(shape):
        return torch.randint(120, 125, shape, device="cuda", dtype=torch.uint8, generator=generator)

    return ExpertWeights(
        codes((experts, intermediate * 2, k // 2)),
        scales((experts, intermediate * 2, k // 32)),
        codes((experts, k, intermediate // 2)),
        scales((experts, k, intermediate // 32)),
    )


def launch_stages(rt, ws, native, out, t1, t2, *, pdl=False, combine="atomic"):
    props = torch.cuda.get_device_properties(rt.device)
    traces = {
        "gate": torch.empty_like(ws.q1, dtype=torch.float32),
        "up": torch.empty_like(ws.q1, dtype=torch.float32),
        "h": torch.empty_like(ws.q1, dtype=torch.bfloat16),
        "fc2": torch.empty_like(ws.q0, dtype=torch.float32),
    }
    s13, s2 = (
        interleave_weight_scales(native.w13_scales),
        interleave_weight_scales(native.w2_scales),
    )
    first = fc1.compile_fc1(
        rt,
        t1,
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
        ws,
        native.w13,
        s13,
        pdl=pdl,
        trace=traces,
    )
    second = fc2.compile_fc2(
        rt,
        t2,
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
        ws,
        native.w2,
        s2,
        out,
        pdl=pdl,
        trace=traces,
        combine=combine,
    )
    first(*fc1.arguments(ws, native.w13, s13, traces))
    second(*fc2.arguments(ws, native.w2, s2, out, traces, combine=combine))
    if combine == "rows":
        last = finalize.compile_finalize(rt, ws, out, pdl=pdl)
        last(*finalize.arguments(ws, out))
    return traces


def check_stages(ws, native, traces, out, *, structural=False, combine="atomic"):
    k, intermediate = ws.q0.shape[1], ws.q1.shape[1]
    slots = torch.zeros((ws.m, 6, k), device=out.device, dtype=torch.float64)
    radii = torch.zeros_like(slots)
    observed_slots = torch.zeros_like(slots)
    baseline_slots = torch.zeros_like(slots)
    for e in range(128):
        first, last = ws.expert_offsets[e : e + 2].cpu().tolist()
        if first == last:
            continue
        pairs = ws.row_pair[first:last].long()
        a = decode_mxfp8(row_quantized(ws, 0, e), torch.float64)
        w13 = decode_e2m1(native.w13[e], native.w13_scales[e], torch.float64)
        w2 = decode_e2m1(native.w2[e], native.w2_scales[e], torch.float64)
        exact = a @ w13.T
        bound = gemm_bound(a, w13)
        baseline = a.float() @ w13.float().T
        observed = torch.cat((traces["gate"][first:last], traces["up"][first:last]), -1)
        assert ((baseline.double() - exact).abs() <= bound).all(), "FP32 FC1 baseline"
        assert ((observed.double() - exact).abs() <= bound).all(), "mixed FC1 accumulator"
        g, u = exact.chunk(2, -1)
        bg, bu = bound.chunk(2, -1)
        h, low, high = activation_interval(g, u, bg, bu, ws.row_weight[first:last])
        hb = traces["h"][first:last]
        assert ((hb.double() >= low) & (hb.double() <= high)).all(), "weighted BF16 h"
        qh = quantize(h.bfloat16())
        observed_q = row_quantized(ws, 1, e)
        exact_encoder = quantize(hb)
        assert torch.equal(
            observed_q.values.view(torch.uint8), exact_encoder.values.view(torch.uint8)
        )
        assert torch.equal(observed_q.scales, exact_encoder.scales)
        d = q1_cells(qh, low, high, observed_q)
        gb, ub = baseline.chunk(2, -1)
        base_h = gb.clamp_max(10) * torch.sigmoid(gb.clamp_max(10)) * ub.clamp(-10, 10)
        base_q = quantize((base_h * ws.row_weight[first:last, None]).bfloat16())
        q1_cells(qh, low, high, base_q)
        b = decode_mxfp8(qh, torch.float64)
        z = b @ w2.T
        b2 = d @ w2.abs().T + 2 * intermediate * U32 * ((b.abs() + d) @ w2.abs().T) + TINY
        got = traces["fc2"][first:last].double()
        base_z = decode_mxfp8(base_q).float() @ w2.float().T
        assert ((got - z).abs() <= b2).all(), "mixed FC2 accumulator"
        assert ((base_z.double() - z).abs() <= b2).all(), "FP32 FC2 baseline"
        slots[pairs // 6, pairs % 6] = z
        radii[pairs // 6, pairs % 6] = b2
        observed_slots[pairs // 6, pairs % 6] = got
        baseline_slots[pairs // 6, pairs % 6] = base_z.double()
    bound = (rows_bound if combine == "rows" else atomic_bound)(slots, radii)
    assert ((out.double() - slots.sum(1)).abs() <= bound).all(), "BF16 combine"
    base_out = torch.zeros_like(out)
    for slot in range(6):
        base_out = (base_out.float() + baseline_slots[:, slot].bfloat16().float()).bfloat16()
    if combine == "rows":
        base_out = rows_combine(baseline_slots)
        assert torch.equal(
            out.view(torch.int16), rows_combine(observed_slots.float()).view(torch.int16)
        )
    assert ((base_out.double() - slots.sum(1)).abs() <= bound).all()
    local = torch.zeros(ws.m, dtype=torch.bool, device=out.device)
    local[ws.row_token[: int(ws.num_rows)].long()] = True
    assert not out[~local].any() and not out[~local].signbit().any()
    if structural:
        assert_attainable(out, observed_slots)


@pytest.mark.parametrize("rows,stages", [(8, 1), (8, 2), (32, 1), (32, 2), (64, 1), (64, 2)])
@pytest.mark.parametrize("scale_edges", [False, True])
def test_mixed_k32_nibbles_exponent_ramp(runtime, rows, stages, scale_edges):
    k, intermediate, m = 512, 128, 3
    native = native_weights(k, intermediate)
    # All sixteen nibble values and both parity positions appear. Distinct
    # gate/up rows and nonuniform scales detect reorder and K128 broadcasts.
    data = torch.arange(2 * intermediate * (k // 2), device="cuda").remainder(256).byte()
    native.w13[0] = data.reshape(2 * intermediate, k // 2)
    ramp = torch.arange(k // 32, device="cuda").remainder(7) + 124
    native.w13_scales[0] = ramp[None, :]
    native.w13_scales[0, intermediate:] += 1
    x = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    for token in range(m):
        x[token, torch.arange(k // 32, device="cuda") * 32 + token] = 1
    ids = torch.full((m, 6), -1, device="cuda", dtype=torch.int32)
    ids[:, 0] = 0
    routes = torch.ones((m, 6), device="cuda")
    ws = workspace(m, k, intermediate, "cuda")
    out = torch.empty_like(x)
    prepare(x, ids, routes, ws, out)
    # Exercise nonuniform activation bytes independently of the quantizer.
    sf = (torch.arange(k // 32, device="cuda").remainder(5) + 125).byte()
    if scale_edges:
        sf[0], sf[1] = 0, 254
        # The product is finite even for UE8M0 byte 0/254. Qualify hardware
        # scale application without silently excluding extreme finite scales.
        native.w13_scales[0, :, :2] = (254 - sf[:2]).expand(2 * intermediate, -1)
    ws.q0_scales[:, :m] = pack_scales(sf.expand(1, m, -1))[0]
    traces = launch_stages(runtime, ws, native, out, Tile(1, rows, stages), Tile(2, 32, 1))
    a = decode_mxfp8(row_quantized(ws, 0, 0), torch.float64)
    w = decode_e2m1(native.w13[0], native.w13_scales[0], torch.float64)
    expected = a @ w.T
    got = torch.cat((traces["gate"][:m], traces["up"][:m]), -1)
    # These binary one-hot products and sums are exactly representable in FP32.
    torch.testing.assert_close(got.double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "r1,r2,s1,s2,pdl",
    [
        (8, 32, 1, 1, False),
        (8, 32, 2, 4, True),
        (32, 64, 2, 2, False),
        (64, 32, 2, 2, True),
        (64, 128, 2, 3, True),
    ],
)
def test_fc1_q1_fc2_and_atomic_components(runtime, r1, r2, s1, s2, pdl):
    torch.manual_seed(94)
    m, k, intermediate = 8, 512, 512
    native = native_weights(k, intermediate)
    x = torch.randn((m, k), device="cuda").bfloat16()
    x[0].zero_()
    ids = torch.tensor(
        [[127, 0, 127, -1, 0, 3], [-1] * 6] + [[0] * 6] * (m - 2), device="cuda", dtype=torch.int32
    )
    routes = (
        torch.tensor([0.1, -0.3, 0, 10, 0.8, 2.0**-130], device="cuda").expand(m, -1).contiguous()
    )
    ws = workspace(m, k, intermediate, "cuda")
    # Deliberately element-aligned, not 4/16-byte aligned.
    out = torch.empty(m * k + 1, device="cuda", dtype=torch.bfloat16)[1:].view(m, k)
    prepare(x, ids, routes, ws, out)
    traces = launch_stages(runtime, ws, native, out, Tile(1, r1, s1), Tile(2, r2, s2), pdl=pdl)
    check_stages(ws, native, traces, out, structural=True)
    q1, scales = ws.q1.clone(), ws.q1_scales.clone()
    ws.y.fill_(float("nan"))
    rows_trace = launch_stages(
        runtime, ws, native, out, Tile(1, r1, s1), Tile(2, r2, s2), pdl=pdl, combine="rows"
    )
    assert torch.equal(ws.q1, q1) and torch.equal(ws.q1_scales, scales)
    check_stages(ws, native, rows_trace, out, combine="rows")
    length = int(ws.num_rows)
    assert torch.equal(ws.y[:length], rows_trace["fc2"][:length].bfloat16())
    assert ws.y[length:].isnan().all()


@pytest.fixture(scope="module")
def full_weights(runtime):
    del runtime
    return full_weight_fixture()


def full_weight_fixture():
    # Full checkpoint ABI, compact nonzero subspace; all other output channels
    # have exactly zero oracle/slot values. Three experts include the last ID.
    compact = native_weights(128, 128, experts=3)
    full = ExpertWeights(
        torch.zeros((128, 4096, 2048), device="cuda", dtype=torch.uint8),
        torch.full((128, 4096, 128), 127, device="cuda", dtype=torch.uint8),
        torch.zeros((128, 4096, 1024), device="cuda", dtype=torch.uint8),
        torch.full((128, 4096, 64), 127, device="cuda", dtype=torch.uint8),
    )
    for j, e in enumerate((0, 3, 127)):
        for native_start, compact_start in ((0, 0), (2048, 128)):
            full.w13[e, native_start : native_start + 128, :64] = compact.w13[
                j, compact_start : compact_start + 128
            ]
            full.w13_scales[e, native_start : native_start + 128, :4] = compact.w13_scales[
                j, compact_start : compact_start + 128
            ]
        full.w2[e, :128, :64] = compact.w2[j]
        full.w2_scales[e, :128, :4] = compact.w2_scales[j]
    return full, compact


def reference_end_to_end(x, ids, routes, compact, actual, *, combine="atomic"):
    # Translate only expert IDs; the active K/N coordinates are unchanged.
    small_ids = torch.full_like(ids, -1)
    for j, e in enumerate((0, 3, 127)):
        small_ids[ids == e] = j
    small_x = x[:, :128].contiguous()
    declared = routed(
        small_x, small_ids, routes, compact, dtype=torch.float64, return_accumulator=True
    )
    slots = torch.zeros((len(x), 6, 128), device=x.device, dtype=torch.float64)
    radii = torch.zeros_like(slots)
    a = decode_mxfp8(quantize(small_x), torch.float64)
    for expert in range(3):
        w13 = decode_e2m1(compact.w13[expert], compact.w13_scales[expert], torch.float64)
        w2 = decode_e2m1(compact.w2[expert], compact.w2_scales[expert], torch.float64)
        exact = a @ w13.T
        # Production K is 4096/2048 even though padded channels contribute zero.
        b1 = 2 * 4096 * U32 * (a.abs() @ w13.abs().T) + TINY
        g, u = exact.chunk(2, -1)
        bg, bu = b1.chunk(2, -1)
        for slot in range(6):
            keep = small_ids[:, slot] == expert
            if not keep.any():
                continue
            h, low, high = activation_interval(
                g[keep], u[keep], bg[keep], bu[keep], routes[keep, slot]
            )
            q = quantize(h.bfloat16())
            d = q1_cells(q, low, high)
            b = decode_mxfp8(q, torch.float64)
            slots[keep, slot] = b @ w2.T
            radii[keep, slot] = (
                d @ w2.abs().T + 2 * 2048 * U32 * ((b.abs() + d) @ w2.abs().T) + TINY
            )
    torch.testing.assert_close(declared, slots.sum(1), rtol=1e-14, atol=1e-14)
    bound = (rows_bound if combine == "rows" else atomic_bound)(slots, radii)
    assert ((actual[:, :128].double() - declared).abs() <= bound).all()
    assert not actual[:, 128:].any()
    empty = (ids == -1).all(1)
    assert not actual[empty].any() and not actual[empty].signbit().any()


@pytest.mark.parametrize("m", [1, 8, 16, 64, 256, 1024, 2048, 4096])
def test_provider_full_geometry_and_changed_route_graph(runtime, full_weights, m):
    if importlib.util.find_spec("ctikernels.moe.cute._w4a8_prepare") is None:
        pytest.skip("PREP task must be merged for provider integration")
    full, compact = full_weights
    provider = W4A8Experts(full, config=Config(pdl=m % 2 == 0))
    provider.workspace(m)
    x = torch.randn((m, 4096), device="cuda").bfloat16()
    ids = torch.zeros((m, 6), device="cuda", dtype=torch.int32)
    routes = (
        torch.tensor([0.1, -0.2, 0.3, 0, 0.8, 2.0**-40], device="cuda").expand(m, -1).contiguous()
    )
    out = torch.empty(m * 4096 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(m, 4096)
    provider(x, ids, routes, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider(x, ids, routes, out=out)
    buffers = (
        "q0",
        "q0_scales",
        "q1",
        "q1_scales",
        "row_token",
        "row_weight",
        "row_pair",
        "expert_offsets",
        "scale_offsets",
        "num_rows",
    )
    pointers = tuple(getattr(provider.workspace(m), name).data_ptr() for name in buffers)
    # Replay one captured graph across duplicates, remote slots, all-remote,
    # last-expert skew, and moved experts. Contents change, addresses do not.
    for pattern in ((0, 3, 127, -1, 0, 127), (-1,) * 6, (127,) * 6, (3, -1, 0, 3, -1, 127)):
        ids.copy_(torch.tensor(pattern, device="cuda", dtype=torch.int32).expand_as(ids))
        x.mul_(0.75)
        routes.mul_(0.9)
        graph.replay()
        reference_end_to_end(x, ids, routes, compact, out, combine=provider.workspace(m).combine)
    assert pointers == tuple(getattr(provider.workspace(m), name).data_ptr() for name in buffers)


@pytest.mark.parametrize(
    "r1,s1,r2,s2",
    [
        (8, 1, 32, 1),
        (8, 2, 32, 4),
        (32, 1, 32, 2),
        (32, 2, 64, 3),
        (64, 1, 64, 1),
        (64, 2, 32, 3),
        (8, 2, 64, 2),
        (128, 2, 128, 3),
        (128, 1, 128, 2),
    ],
)
@pytest.mark.parametrize("pdl,watchdog", [(False, False), (True, True)])
def test_pipeline_matches_single_stage(runtime, r1, s1, r2, s2, pdl, watchdog):

    # One persistent CTA crosses many work records. Three K tiles exercise
    # partial rings, including stage counts larger than K and odd wrap phases.
    m, k, intermediate = 37, 384, 384
    native = native_weights(k, intermediate)
    weights = (native.w13, native.w2)
    scales = (
        interleave_weight_scales(native.w13_scales),
        interleave_weight_scales(native.w2_scales),
    )
    x = torch.randn((m, k), device="cuda").bfloat16()
    ids = torch.full((m, 6), -1, device="cuda", dtype=torch.int32)
    ids[:17, 0], ids[17:, 0] = 0, 127
    routes = torch.full((m, 6), 0.25, device="cuda")
    ws = workspace(m, k, intermediate, "cuda")
    out = torch.empty_like(x)
    limit = torch.cuda.get_device_properties(runtime.device).shared_memory_per_block_optin

    def run(w13, w2, stages1, stages2):
        prepare(x, ids, routes, ws, out)
        traces = {
            "gate": torch.zeros_like(ws.q1, dtype=torch.float32),
            "up": torch.zeros_like(ws.q1, dtype=torch.float32),
            "h": torch.zeros_like(ws.q1, dtype=torch.bfloat16),
            "fc2": torch.zeros_like(ws.q0, dtype=torch.float32),
        }
        first = fc1.compile_fc1(
            runtime,
            Tile(1, r1, stages1, watchdog=watchdog),
            1,
            limit,
            ws,
            w13,
            scales[0],
            pdl=pdl,
            trace=traces,
        )
        second = fc2.compile_fc2(
            runtime,
            Tile(2, r2, stages2, watchdog=watchdog),
            1,
            limit,
            ws,
            w2,
            scales[1],
            out,
            pdl=pdl,
            trace=traces,
        )
        first(*fc1.arguments(ws, w13, scales[0], traces))
        second(*fc2.arguments(ws, w2, scales[1], out, traces))
        return traces, out.clone(), ws.q1[:m].clone(), ws.q1_scales.clone()

    baseline, expected, q1, sf = run(native.w13, native.w2, 1, 1)
    traces, actual, got_q1, got_sf = run(*weights, s1, s2)
    for key in traces:
        assert torch.equal(traces[key], baseline[key]), key
    assert torch.equal(got_q1, q1)
    assert torch.equal(got_sf, sf)
    # One local slot/token avoids order-dependent BF16 atomic accumulation.
    assert torch.equal(actual, expected)
    check_stages(ws, native, traces, out)


@pytest.mark.parametrize("m", [1024, 2048, 4096])
@pytest.mark.parametrize("pattern", ["balanced", "skewed"])
def test_large_m_orders_and_rows_match_64(runtime, m, pattern):
    # Compact channels retain nonuniform K32 scales and multiple N tiles;
    # full M/routing exercises persistent strides, expert tails and empty CTAs.
    k = intermediate = 384
    native = native_weights(k, intermediate)
    w13, w2 = native.w13, native.w2
    s13, s2 = (
        interleave_weight_scales(native.w13_scales),
        interleave_weight_scales(native.w2_scales),
    )
    generator = torch.Generator(device="cuda").manual_seed(631 + m)
    x = torch.randn((m, k), device="cuda", generator=generator).bfloat16()
    ids = torch.full((m, 6), -1, device="cuda", dtype=torch.int32)
    if pattern == "balanced":
        ids[:, :3] = torch.arange(m * 3, device="cuda").reshape(m, 3) % 128
    else:
        ids.fill_(127)  # Duplicate slots, empty experts and a long M-tile run.
        ids[0, 0], ids[1:23, 0], ids[23:66, 0] = 0, 31, 32
        ids[-1] = -1
    routes = (
        torch.tensor([0.25, -0.5, 0.75, 0.125, -0.25, 0.5], device="cuda")
        .expand(m, -1)
        .contiguous()
    )
    ws = workspace(m, k, intermediate, "cuda")
    out = torch.empty_like(x)
    prepare(x, ids, routes, ws, out)
    props = torch.cuda.get_device_properties(runtime.device)
    traces = {
        "gate": torch.empty_like(ws.q1, dtype=torch.float32),
        "up": torch.empty_like(ws.q1, dtype=torch.float32),
        "h": torch.empty_like(ws.q1, dtype=torch.bfloat16),
        "fc2": torch.empty_like(ws.q0, dtype=torch.float32),
    }

    def run(r1, r2, order):
        out.zero_()
        ws.q1.zero_()
        ws.q1_scales.fill_(0x7F7F7F7F)
        for value in traces.values():
            value.zero_()
        limit = props.shared_memory_per_block_optin
        first = fc1.compile_fc1(
            runtime,
            choose_tile(1, r1, limit, order=order),
            props.multi_processor_count,
            limit,
            ws,
            w13,
            s13,
            pdl=True,
            trace=traces,
        )
        second = fc2.compile_fc2(
            runtime,
            choose_tile(2, r2, limit, order=order),
            props.multi_processor_count,
            limit,
            ws,
            w2,
            s2,
            out,
            pdl=True,
            trace=traces,
        )
        first(*fc1.arguments(ws, w13, s13, traces))
        second(*fc2.arguments(ws, w2, s2, out, traces))

    run(64, 64, ORDERS[0])
    expected = out.clone()
    q1, scales = ws.q1.clone(), ws.q1_scales.clone()
    baseline = {name: value.clone() for name, value in traces.items()}
    count = int(ws.num_rows)
    slots = torch.zeros((m * 6, k), device="cuda", dtype=torch.float64)
    slots[ws.row_pair[:count].long()] = baseline["fc2"][:count].bfloat16().double()
    slots = slots.reshape(m, 6, k)
    bound = atomic_bound(slots, torch.zeros_like(slots))
    center = slots.sum(1)
    assert ((expected.double() - center).abs() <= bound).all()
    for r1, r2 in ((64, 64), (128, 64), (64, 128), (128, 128)):
        for order in ORDERS:
            if (r1, r2, order) == (64, 64, ORDERS[0]):
                continue
            run(r1, r2, order)
            assert torch.equal(ws.q1, q1), (r1, r2, order, "Q1 codes")
            assert torch.equal(ws.q1_scales, scales), (r1, r2, order, "Q1 scales")
            for name, value in traces.items():
                assert torch.equal(value, baseline[name]), (r1, r2, order, name)
            assert ((out.double() - center).abs() <= bound).all()
            assert ((out.double() - expected.double()).abs() <= 2 * bound).all()


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_128_alias_short_k_and_partial_ring(runtime, stages):
    # K=1 is shorter than both automatic rings. Reusing an unfilled stage
    # before the preceding epilogue drains would corrupt the aliased buffer.
    m, k = 257, 128
    native = native_weights(k, k)
    ws = workspace(m, k, k, "cuda")
    x = torch.randn((m, k), device="cuda").bfloat16()
    ids = torch.full((m, 6), -1, device="cuda", dtype=torch.int32)
    ids[:, 0] = 127
    routes = torch.full((m, 6), 0.25, device="cuda")
    out = torch.empty_like(x)
    props = torch.cuda.get_device_properties(runtime.device)
    s13, s2 = (
        interleave_weight_scales(native.w13_scales),
        interleave_weight_scales(native.w2_scales),
    )
    w13, w2 = native.w13, native.w2

    def run(rows, s1, s2_count):
        prepare(x, ids, routes, ws, out)
        first = fc1.compile_fc1(
            runtime,
            Tile(1, rows, s1, watchdog=True, order=ORDERS[int(rows == 128)]),
            1,
            props.shared_memory_per_block_optin,
            ws,
            w13,
            s13,
            pdl=False,
        )
        second = fc2.compile_fc2(
            runtime,
            Tile(2, rows, s2_count, watchdog=True, order=ORDERS[int(rows == 128)]),
            1,
            props.shared_memory_per_block_optin,
            ws,
            w2,
            s2,
            out,
            pdl=False,
        )
        first(*fc1.arguments(ws, w13, s13))
        second(*fc2.arguments(ws, w2, s2, out))
        return out.clone(), ws.q1.clone(), ws.q1_scales.clone()

    baseline = run(64, 2, 3)
    actual = run(128, min(stages, 2), stages)
    assert all(torch.equal(a, b) for a, b in zip(actual, baseline, strict=True))


@pytest.mark.parametrize("m,pdl", [(1, False), (43, True), (1024, False), (4096, True)])
def test_rows_provider_deterministic_changed_routes(runtime, full_weights, m, pdl):
    full, compact = full_weights
    provider = W4A8Experts(full, config=Config(combine="rows", pdl=pdl))
    ws = provider.workspace(m)
    x = torch.randn((m, 4096), device="cuda").bfloat16()
    ids = torch.zeros((m, 6), device="cuda", dtype=torch.int32)
    routes = (
        torch.tensor([0.1, -0.2, 0.3, 0, 0.8, 2.0**-40], device="cuda").expand(m, -1).contiguous()
    )
    out = torch.empty_like(x)
    provider(x, ids, routes, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider(x, ids, routes, out=out)
    pointers = (ws.y.data_ptr(), ws.slot_row.data_ptr())
    for pattern in ([127] * 6, [-1] * 6, [0, 3, 127, -1, 0, 127], [3, -2, 128, 3, -1, 127]):
        ids.copy_(torch.tensor(pattern, device="cuda", dtype=torch.int32).expand_as(ids))
        x.mul_(0.75)
        routes.mul_(0.9)
        graph.replay()
        reference_end_to_end(x, ids, routes, compact, out, combine="rows")
        expected = out.view(torch.int16).clone()
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out.view(torch.int16), expected)
    assert pointers == (ws.y.data_ptr(), ws.slot_row.data_ptr())
