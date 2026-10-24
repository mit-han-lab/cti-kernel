# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Static FP8 qualification on compute capability 8.9 or newer; never run by the CPU suite.

The oracle decodes the same quantized operands in FP64. As in test_block_fp8,
compare against an FP32 GEMM rounded to the output dtype (TF32 disabled), with
test_block_fp8's RMS and elementwise allowances (see assert_precision).
"""

from functools import partial

import pytest
import torch

from ctikernels.linear.cute._block_fp8_provider import available
from ctikernels.linear.fp8_static import PROVIDERS as BASE_PROVIDERS
from ctikernels.linear.fp8_static import StaticFP8Linear, quantize_static_fp8

E4M3 = torch.float8_e4m3fn
DTYPES = (torch.bfloat16, torch.float16)
INNER_PROVIDERS = (
    "block-fp8:triton-fp8",
    "block-fp8:flashinfer-cutlass",
    "block-fp8:cuda-sm12x",
    "block-fp8:cute-fp8:swap128x32s3-mx",
    "block-fp8:cute-fp8:swap128x32s3-mx-pdl",
)
PROVIDERS = (*BASE_PROVIDERS, *INNER_PROVIDERS)
TP2_SHAPES = (
    ((3072, 512, 512, 3072), 5120),
    ((1024, 1024, 3072, 3072), 5120),
    ((5120,), 3072),
)


@pytest.fixture(autouse=True)
def hardware(request):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires CUDA compute capability 8.9 or newer")
    if not request.node.name.startswith("test_fp8pt_gen") and not available(
        torch.cuda.get_device_capability()
    ):
        pytest.skip("legacy provider matrix requires sm12x and CuTe DSL 4.8.0")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(173)
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def reference_quant(x, s):
    return (x.float() / s).clamp(-448, 448).to(E4M3).reshape(-1, x.shape[-1])


def assert_quant(q, ref):
    nan = torch.isnan(ref.float())
    assert torch.equal(torch.isnan(q.float()), nan)
    assert torch.all((q.view(torch.uint8)[nan] & 127) == 127)
    assert torch.equal(q.view(torch.uint8)[~nan], ref.view(torch.uint8)[~nan])


@pytest.mark.parametrize("k", [5120, 6144, 3072, 200, 1, 136])
@pytest.mark.parametrize("m", [0, 1, 3, 64, 1000])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["packed", "strided", "broadcast", "overlap", "unaligned"])
def test_quantizer(k, m, dtype, layout):
    s = torch.tensor(0.00713, device="cuda")
    stride = {
        "packed": k,
        "strided": k + 16,
        "broadcast": 0,
        "overlap": max(0, k // 2),
        "unaligned": k + 1,
    }[layout]
    storage = (torch.randn(max(1, m) * (k + 16) + 1, device="cuda") * 8).to(dtype)
    # Mix saturation, signed zero, infinities and NaN into the same reference.
    special = torch.tensor(
        [float("inf"), -float("inf"), float("nan"), 0.0, -0.0, 100.0, -100.0],
        device="cuda",
        dtype=dtype,
    )
    storage[:7] = special
    x = torch.as_strided(storage, (m, k), (stride, 1), storage_offset=int(layout == "unaligned"))
    ref = reference_quant(x, s)
    baseline = quantize_static_fp8(x, s)
    for block, row in ((False, False), (True, False), (False, True)):
        result = quantize_static_fp8(x, s, block_scale=block, row_scale=row)
        q, scale = result if block or row else (result, None)
        assert torch.equal(q.view(torch.uint8), baseline.view(torch.uint8))
        assert q.shape == (m, k) and q.is_contiguous()
        assert_quant(q, ref)
        if block or row:
            assert scale.shape == (m, 1 if row else (k + 127) // 128) and scale.is_contiguous()
            assert torch.equal(scale, s.expand_as(scale))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [0.0137, 1.0, 2.0**-120, 2.0**-140, 3.1e30])
def test_all_input_encodings(dtype, scale):
    # Includes rounding boundaries, subnormals, both infinities and all NaN encodings.
    x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(256, 256).cuda()
    s = torch.tensor(scale, device="cuda")
    assert_quant(quantize_static_fp8(x, s), reference_quant(x, s))


def make_op(slots, k, provider, *, equal=False, quantizer=None, bias=None):
    w = (torch.randn(sum(slots), k, device="cuda") * 4).clamp(-448, 448).to(E4M3)
    ws = torch.tensor([0.013, 0.027, 0.041, 0.059][: len(slots)], device="cuda")
    if equal:
        ws.fill_(0.03125)
    s = torch.tensor(0.0217, device="cuda")
    try:
        op = StaticFP8Linear(
            w, ws, s, slot_rows=slots, provider=provider, quantizer=quantizer, bias=bias
        )
    except ValueError as error:
        if "unavailable" in str(error):
            pytest.skip(str(error))
        raise
    return op, w, ws, s


def assert_precision(y, q, w, ws, s, slots, bias=None):
    """test_block_fp8's FP32-floor rule on the same quantized operands (FP64 oracle):
    RMS error <= 1.05x the FP32 floor's, and per element <= floor error + 1 output ulp
    + 2^-22 * sum|products| (FP32 accumulation order slack)."""
    scales = torch.repeat_interleave(ws.double(), torch.tensor(slots, device="cuda"))
    a64 = q.double() * s.double()
    b64 = w.double() * scales[:, None]
    oracle = a64 @ b64.T
    magnitude = a64.abs() @ b64.abs().T
    floor_error = ((a64.float() @ b64.float().T).to(y.dtype).double() - oracle).abs()
    bits, minimum = (7, -126) if y.dtype == torch.bfloat16 else (10, -14)
    ulp = torch.exp2(torch.floor(torch.log2(oracle.abs().clamp_min(2.0**minimum))) - bits)
    bias_ulp = torch.zeros_like(oracle)
    if bias is not None:
        oracle = oracle + bias.double()
        bias_ulp = torch.exp2(torch.floor(torch.log2(oracle.abs().clamp_min(2.0**minimum))) - bits)
    error = (y.double() - oracle).abs()

    def rms(e):
        return e.square().mean().sqrt().item()

    assert rms(error) <= 1.05 * rms(floor_error) + rms(bias_ulp) + 1e-30, (
        rms(error),
        rms(floor_error),
        rms(bias_ulp),
    )
    excess = error - floor_error - ulp - bias_ulp - 2.0**-22 * magnitude
    assert excess.max().item() <= 0, f"max excess: {excess.max().item()}"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "slots,k",
    [
        ((768, 128, 128, 768), 640),
        ((256, 256, 768, 768), 640),
        ((100, 36, 120), 200),
        ((100, 36, 60, 60), 200),
    ],
)
@pytest.mark.parametrize("m", [0, 1, 3, 64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_precision_and_quantized(provider, slots, k, m, dtype):
    op, w, ws, s = make_op(slots, k, provider)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    q = op.quantize(x)
    assert_quant(q, reference_quant(x, s))
    y = op(x)
    assert torch.equal(y, op.quantized(q, out_dtype=dtype))
    if m:
        assert_precision(y, q, w, ws, s, slots)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [0, 1, 3])
def test_strided_output(provider, dtype, m):
    op, _, _, _ = make_op((128, 256), 128, provider)
    x = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)
    buf = torch.full((m, 397), -19, dtype=dtype, device="cuda")
    out = buf[:, 3:387]
    eager = op(x, out_dtype=dtype)
    assert op(x, out=out) is out
    assert torch.equal(out, eager)
    assert bool(torch.all(buf[:, :3] == -19)) and bool(torch.all(buf[:, 387:] == -19))
    assert op.quantized(op.quantize(x), out=out) is out
    assert torch.equal(out, eager)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_broadcast_and_higher_rank(provider):
    op, _, _, _ = make_op((128, 128), 128, provider)
    x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16).expand(3, -1)
    assert torch.equal(op(x), op(x.contiguous()))
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    assert op(x).shape == (2, 3, 256)
    assert torch.equal(op(x).reshape(6, 256), op(x.reshape(6, 128)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_equal_scale_single_call_matches_slots(dtype):
    # Exact integer dot products isolate the fusion policy from GEMM reduction order.
    w = torch.randint(-4, 5, (384, 128), device="cuda").to(E4M3)
    ws = torch.full((2,), 0.03125, device="cuda")
    s = torch.tensor(0.125, device="cuda")
    op = StaticFP8Linear(w, ws, s, slot_rows=(128, 256), provider="torch-scaled-mm")
    q = torch.randint(-4, 5, (3, 128), device="cuda").to(E4M3)
    fused = op.quantized(q, out_dtype=dtype)
    assert len(op._libraries["torch-scaled-mm"]._slots) == 1
    pieces = []
    start = 0
    for i, rows in enumerate(op.slot_rows):
        one = StaticFP8Linear(w[start : start + rows], ws[i : i + 1], s, provider="torch-scaled-mm")
        pieces.append(one.quantized(q, out_dtype=dtype))
        start += rows
    assert torch.equal(fused, torch.cat(pieces, dim=1))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [3, 16, 17])
def test_graph_after_warm_up(provider, quantized, dtype, m):
    op, _, _, _ = make_op((128, 128), 128, provider)
    x = torch.randn(m, 128, device="cuda", dtype=dtype)
    q = op.quantize(x)
    out = torch.empty((m, 269), device="cuda", dtype=dtype)[:, 3:259]
    call, arg = (op.quantized, q) if quantized else (op, x)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # The outer stream context restores the caller even if capture_end raises.
    with torch.cuda.stream(stream):
        op.warm_up(max_m=m, dtype=dtype, out_dtype=dtype, quantized=quantized, out_row_stride=269)
        with torch.cuda.graph(graph, stream=stream):
            result = call(arg, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    x.normal_()
    q.copy_(op.quantize(x))
    expected = call(arg, out_dtype=dtype)
    graph.replay()
    assert result is out and torch.equal(out, expected)


def _capture_cold(op, call, arg, out, **warm):
    """Capture `call` right after warm_up, with no eager call of that size first."""
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(**warm)
        with torch.cuda.graph(graph, stream=stream):
            result = call(arg, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result


@pytest.mark.parametrize("provider", INNER_PROVIDERS)
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("m", [2, 5, 9, 13])
@pytest.mark.parametrize("use_workspace", [False, True], ids=["plain", "workspace"])
def test_forced_provider_warm_up_intermediate_m(provider, quantized, m, use_workspace):
    # A fresh instance per M: warm_up(16) must cover every specialisation up to 16,
    # including the packed scales the workspace entries pass to the block kernels.
    try:
        op, _, _, _ = make_op((128, 128), 256, provider)
    except ValueError as error:
        if provider in ("block-fp8:flashinfer-cutlass", "block-fp8:cuda-sm12x"):
            pytest.skip(str(error))
        raise
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    arg = op.quantize(x) if quantized else x
    call = op.quantized if quantized else op
    if use_workspace:
        workspace = torch.empty(op.workspace_bytes(16) // 4, device="cuda", dtype=torch.float32)
        call = partial(call, workspace=workspace)
    out = torch.empty((m, 256), device="cuda", dtype=torch.bfloat16)
    graph, result = _capture_cold(op, call, arg, out, max_m=16, quantized=quantized)
    expected = call(arg)
    graph.replay()
    assert result is out and torch.equal(out, expected)


@pytest.mark.parametrize(
    "provider", [p for p in INNER_PROVIDERS if p.startswith("block-fp8:cute-fp8")]
)
@pytest.mark.parametrize("parent_cols,offset", [(272, 0), (257, 1), (272, 1)])
def test_forced_cute_warm_up_output_variants(provider, parent_cols, offset):
    op, _, _, _ = make_op((256,), 128, provider)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    parent = torch.full((4, parent_cols), -19, device="cuda", dtype=torch.bfloat16)
    out = parent[:, offset : offset + 256]
    graph, result = _capture_cold(op, op, x, out, max_m=4, out_row_stride=272)
    expected = op(x)
    graph.replay()
    assert result is out and torch.equal(out, expected)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "slots,k", [((6144, 1024, 1024, 6144), 5120), ((2048, 2048, 6144, 6144), 5120), ((5120,), 6144)]
)
def test_real_shapes(provider, slots, k):
    op, _, _, _ = make_op(slots, k, provider)
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    y = op(x)
    assert y.shape == (1, sum(slots)) and bool(torch.isfinite(y).all())
    assert torch.equal(y, op.quantized(op.quantize(x)))


@pytest.mark.parametrize("provider", [None, *PROVIDERS])
@pytest.mark.parametrize("slots,k", TP2_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_workspace", [False, True], ids=["plain", "workspace"])
def test_quantized_matches_op_bitwise(provider, slots, k, dtype, use_workspace):
    try:
        op, _, _, s = make_op(slots, k, provider)
    except ValueError as error:
        if provider in ("block-fp8:flashinfer-cutlass", "block-fp8:cuda-sm12x"):
            pytest.skip(str(error))
        raise
    op.warm_up(max_m=64, dtype=dtype, out_dtype=dtype)
    op.warm_up(max_m=64, dtype=dtype, out_dtype=dtype, quantized=True)
    workspace = (
        torch.empty(op.workspace_bytes(64) // 4, device="cuda", dtype=torch.float32)
        if use_workspace
        else None
    )
    # Reuse the warmed instance across M to keep the full provider matrix inexpensive.
    for m in (1, 2, 3, 4, 16, 64):
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        q = quantize_static_fp8(x, s)
        assert torch.equal(
            op(x, workspace=workspace), op.quantized(q, out_dtype=dtype, workspace=workspace)
        ), (provider, m, dtype, use_workspace)


def gemm_launches(events, provider):
    """Our launches: torch 2.13's rowwise _scaled_mm adds its own DtoD memcpy on sm12x
    (with fresh contiguous scales and no out= too), and a Triton split-K config
    launches its reduction; anything else, including scale copies, counts. The split-K
    reduction belongs to that one GEMM: Q2 asks for one GEMM per call, not one per slot."""
    rowwise = provider == "torch-scaled-mm:rowwise"
    return [
        e
        for e in events
        if e != "_split_sum_kernel" and not (rowwise and e.startswith("Memcpy DtoD"))
    ]


def cuda_events(call):
    """Count all device work, including memcpy/memset; never filter by kernel name."""
    call()  # Prime the exact call outside the measured interval.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        call()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


@pytest.mark.parametrize("slots,k", TP2_SHAPES)
@pytest.mark.parametrize("use_workspace", [False, True], ids=["plain", "workspace"])
def test_quantized_one_launch(slots, k, use_workspace):
    op, _, _, s = make_op(slots, k, None)
    op.warm_up(max_m=64, quantized=True)
    workspace = (
        torch.empty(op.workspace_bytes(64) // 4, device="cuda", dtype=torch.float32)
        if use_workspace
        else None
    )
    for m in (1, 2, 3, 4, 16, 32, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        q = quantize_static_fp8(x, s)
        out = torch.empty(m, sum(slots), device="cuda", dtype=torch.bfloat16)
        # One GEMM launch; a Triton split-K config also launches its own reduction.
        gemm = cuda_events(partial(op.quantized, q, out=out, workspace=workspace))
        assert len(gemm_launches(gemm, op.provider_for(m))) == 1, (m, op.provider_for(m), gemm)
        events = cuda_events(partial(op, x, out=out, workspace=workspace))
        assert len(events) == len(gemm) + 1, (m, op.provider_for(m), events)


@pytest.mark.parametrize(
    "slots,k,ms", [((1008, 16), 512, range(1, 65)), ((1008, 4112), 3072, (1, 4, 64))]
)
def test_unaligned_slots_single_launch(slots, k, ms, monkeypatch):
    # Unequal scales and an unaligned partition take rowwise on every device, whatever
    # the table row (e.g. FlashInfer for (5120, 3072) on RTX PRO 6000).
    from ctikernels.linear import _fp8_static_dispatch as dispatch

    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((64, "block-fp8"),))
    op, w, ws, s = make_op(slots, k, None)
    op.warm_up(max_m=64)
    op.warm_up(max_m=64, quantized=True)
    for m in ms:
        assert op.provider_for(m) == "torch-scaled-mm:rowwise"
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        q = quantize_static_fp8(x, s)
        out = torch.empty(m, sum(slots), device="cuda", dtype=torch.bfloat16)
        events = cuda_events(partial(op.quantized, q, out=out))
        assert len(gemm_launches(events, op.provider_for(m))) == 1, (m, events)
        assert_precision(out, q, w, ws, s, slots)
        assert torch.equal(op(x), out)


@pytest.mark.parametrize("block", [False, True])
def test_quantizer_graph_reads_device_scale(block):
    x = torch.randn(3, 200, device="cuda", dtype=torch.bfloat16)
    s = torch.full((17,), 0.03, device="cuda")[::17]
    quantize_static_fp8(x, s, block_scale=block)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = quantize_static_fp8(x, s, block_scale=block)
    x.normal_()
    s.fill_(0.0173)
    graph.replay()
    q, scale = result if block else (result, None)
    assert_quant(q, reference_quant(x, s))
    if block:
        assert torch.equal(scale, s.expand_as(scale))


@pytest.mark.parametrize("provider", INNER_PROVIDERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_inner_precision_and_output_view(provider, dtype, m):
    try:
        op, w, ws, s = make_op((128, 128), 256, provider)
    except ValueError as error:
        if provider in ("block-fp8:flashinfer-cutlass", "block-fp8:cuda-sm12x"):
            pytest.skip(str(error))
        raise
    assert op.provider_for(m) == provider
    x = torch.randn(m, 256, device="cuda", dtype=dtype)
    q = op.quantize(x)
    y = op(x)
    assert_precision(y, q, w, ws, s, (128, 128))
    buf = torch.full((m, 269), -19, device="cuda", dtype=dtype)
    out = buf[:, 3:259]
    assert op(x, out=out) is out and torch.equal(y, out)
    # Force the prequantized alignment repair and stride-zero scale entry.
    padded_q = torch.empty((m, 257), device="cuda", dtype=E4M3)[:, 1:]
    padded_q.copy_(q)
    assert op.quantized(padded_q, out=out) is out and torch.equal(y, out)
    assert bool(torch.all(buf[:, :3] == -19)) and bool(torch.all(buf[:, 259:] == -19))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("layout", ["aligned", "unaligned", "tail"])
def test_pdl_quantizer_bitwise(dtype, block, layout):
    k = 200 if layout == "tail" else 256
    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(dtype)
    x = bits[: (65536 // k) * k].reshape(-1, k)
    if layout == "unaligned":
        padded = torch.empty((x.shape[0], k + 1), device="cuda", dtype=dtype)[:, 1:]
        padded.copy_(x)
        x = padded
    s = torch.tensor(0.0137, device="cuda")
    plain = quantize_static_fp8(x, s, block_scale=block)
    pdl = quantize_static_fp8(x, s, block_scale=block, pdl=True)
    plain_q, plain_s = plain if block else (plain, None)
    pdl_q, pdl_s = pdl if block else (pdl, None)
    assert torch.equal(plain_q.view(torch.uint8), pdl_q.view(torch.uint8))
    if block:
        assert torch.equal(plain_s.view(torch.int32), pdl_s.view(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rowwise_probe_and_fallback(dtype):
    from ctikernels.linear import _fp8_static_dispatch as dispatch
    from ctikernels.linear.torch import _fp8_static as market

    device = torch.device("cuda", torch.cuda.current_device())
    first = market.rowwise_probe(device, dtype)
    assert market.rowwise_probe(device, dtype) == first
    assert (device, dtype) in market._ROWWISE_PROBES
    w = torch.zeros((256, 128), device=device, dtype=E4M3)
    ws = torch.tensor([0.01, 0.02], device=device)
    s = torch.tensor(0.03, device=device)
    alias = StaticFP8Linear(w, ws, s, slot_rows=(128, 128), provider="torch-scaled-mm")
    # FP16 output never takes rowwise (it misses the precision gate); BF16 follows the probe.
    rowwise_ok = first is None and dtype == torch.bfloat16
    assert alias.provider_for(3, dtype) == (
        "torch-scaled-mm:rowwise" if rowwise_ok else "torch-scaled-mm:slots"
    )
    key = (
        *torch.cuda.get_device_capability(),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(dispatch.TABLE, key, {(256, 128): ((0, "torch-scaled-mm:rowwise"),)})
        automatic = StaticFP8Linear(w, ws, s, slot_rows=(128, 128))
        assert automatic.provider_for(3, dtype) == (
            "torch-scaled-mm:rowwise" if rowwise_ok else "block-fp8"
        )
    errors = [market.rowwise_probe(device, d) for d in DTYPES]
    if any(errors):
        with pytest.raises(ValueError) as caught:
            StaticFP8Linear(w, ws, s, slot_rows=(128, 128), provider="torch-scaled-mm:rowwise")
        assert next(e for e in errors if e) in str(caught.value)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["packed", "unaligned", "tail"])
@pytest.mark.parametrize("pdl", [False, True])
def test_row_scale_mode_bitwise_and_capture(dtype, layout, pdl):
    k = 200 if layout == "tail" else 256
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(dtype)
    x = bits[: (65536 // k) * k].reshape(-1, k)
    if layout == "unaligned":
        storage = torch.empty((x.shape[0], k + 1), device="cuda", dtype=dtype)
        storage[:, 1:].copy_(x)
        x = storage[:, 1:]
    s = torch.tensor(0.0137, device="cuda")
    plain = quantize_static_fp8(x, s, pdl=pdl)
    block, bs = quantize_static_fp8(x, s, block_scale=True, pdl=pdl)
    row, rs = quantize_static_fp8(x, s, row_scale=True, pdl=pdl)
    assert torch.equal(plain.view(torch.uint8), block.view(torch.uint8))
    assert torch.equal(plain.view(torch.uint8), row.view(torch.uint8))
    assert rs.shape == (x.shape[0], 1) and rs.dtype == torch.float32 and rs.is_contiguous()
    assert torch.equal(rs.expand_as(bs), bs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        row, rs = quantize_static_fp8(x, s, row_scale=True, pdl=pdl)
    s.fill_(0.0273)
    graph.replay()
    assert torch.equal(rs, s.expand_as(rs))
    expected = quantize_static_fp8(x, s, pdl=pdl)
    assert torch.equal(row.view(torch.uint8), expected.view(torch.uint8))


def capture_workspace(call, warm, *, library_allocations=0, prime=True):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        warm()
        if prime:
            # cuBLAS/cuBLASLt workspace is lazy and belongs to the handle + stream;
            # this eager call is for library routes only. Block-FP8 routes pass
            # prime=False and capture right after warm_up, so a cold specialisation
            # missed by warm_up fails the capture instead of compiling here.
            call()
        with torch.cuda.graph(graph, stream=stream):
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            call()
            after = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.current_stream().wait_stream(stream)
    # Never raise an allocation assertion while the stream is still capturing.
    # torch may re-create its cuBLASLt workspace inside a capture (graph-pool owned),
    # depending on earlier captures in the process.
    assert 0 <= after - before <= library_allocations
    return graph


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDERS)
@torch.inference_mode()
def test_engine_workspace_bitwise_and_allocation_free_capture(provider):
    op, w, ws, s = make_op((128, 256), 2048, provider)
    workspace = torch.empty(op.workspace_bytes(320) // 4, device="cuda", dtype=torch.float32)
    for dtype in DTYPES:
        for m in (1, 17, 320):
            x = torch.randn(m, 2049, device="cuda", dtype=dtype)[:, 1:]
            q = op.quantize(x)
            q_parent = torch.empty((m, 2049), dtype=E4M3, device="cuda")
            q_parent[:, 1:].copy_(q)
            q = q_parent[:, 1:]
            parent = torch.full((m, 387), -19, device="cuda", dtype=dtype)
            out = parent[:, 1:385]
            for quantized in (False, True):

                def call(buffer=None, target=None, *, quantized=quantized, q=q, x=x, dtype=dtype):
                    return (
                        op.quantized(q, out_dtype=dtype, out=target, workspace=buffer)
                        if quantized
                        else op(x, out_dtype=dtype, out=target, workspace=buffer)
                    )

                # Capture first, on a fresh instance with no eager call of any kind:
                # eager calls do not register warmed layouts, but could compile a
                # specialisation that warm_up should have prepared itself.
                cold = StaticFP8Linear(w, ws, s, slot_rows=(128, 256), provider=provider)

                def cold_call(*, cold=cold, quantized=quantized, q=q, x=x, dtype=dtype, out=out):
                    return (
                        cold.quantized(q, out_dtype=dtype, out=out, workspace=workspace)
                        if quantized
                        else cold(x, out_dtype=dtype, out=out, workspace=workspace)
                    )

                graph = capture_workspace(
                    cold_call,
                    partial(
                        cold.warm_up,
                        batch_sizes=[m],
                        dtype=dtype,
                        out_dtype=dtype,
                        quantized=quantized,
                        out_row_stride=out.stride(0),
                    ),
                    library_allocations=int(provider.startswith("torch-scaled-mm")),
                    prime=not provider.startswith("block-fp8"),
                )
                expected = call().clone()
                workspace.fill_(float("nan"))
                graph.replay()
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                assert call(workspace, out) is out
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                first = call(workspace)
                saved = first.clone()
                assert first.untyped_storage().data_ptr() != workspace.untyped_storage().data_ptr()
                assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
                # A different input must not overwrite the preceding layer's result.
                if quantized:
                    second = op.quantized(torch.zeros_like(q), out_dtype=dtype, workspace=workspace)
                else:
                    second = op(torch.zeros_like(x), out_dtype=dtype, workspace=workspace)
                assert second.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
                assert torch.equal(first.view(torch.uint8), saved.view(torch.uint8))
                assert torch.all(parent[:, 0] == -19) and torch.all(parent[:, 385:] == -19)


@pytest.mark.parametrize("backend", ["cublas", "cutlass", "auto"])
@torch.inference_mode()
def test_engine_workspace_flashinfer_padding(backend):
    op, _, _, _ = make_op((96, 160), 144, f"flashinfer-bmm:{backend}")
    workspace = torch.empty(op.workspace_bytes(17) // 4, device="cuda", dtype=torch.float32)
    for dtype in DTYPES:
        x = torch.randn(17, 144, dtype=dtype, device="cuda")
        expected = op(x)
        out = torch.empty_like(expected)
        op(x, out=out, workspace=workspace)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
        graph = capture_workspace(
            partial(op, x, out=out, workspace=workspace),
            partial(op.warm_up, batch_sizes=[17], dtype=dtype),
        )
        workspace.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("k", [1, 8, 100, 5120])
@pytest.mark.parametrize("m", [0, 1, 5, 64])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["packed", "padded", "misaligned", "broadcast"])
def test_fp8pt_gen_quantizers_bitwise(k, m, dtype, layout):
    if not available(torch.cuda.get_device_capability()):
        pytest.skip("CuTe quantizer unavailable")
    stride = {"packed": k, "padded": k + 16, "misaligned": k + 1, "broadcast": 0}[layout]
    offset = int(layout == "misaligned")
    storage = torch.randn(max(1, m) * (k + 16) + 32, dtype=dtype, device="cuda")
    special = torch.tensor(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            -0.0,
            0.0,
            2**-24,
            -(2**-24),
            2**-10,
            -(2**-10),
            1.0625,
            1.0625 - 2**-7,
            1.0625 + 2**-7,
        ],
        dtype=dtype,
        device="cuda",
    )
    storage[offset : offset + special.numel()].copy_(special)
    x = storage.as_strided((m, k), (stride, 1), storage_offset=offset)
    for s in (torch.tensor(1.0, device="cuda"), torch.full((17,), 0.0137, device="cuda")[::17]):
        ref = reference_quant(x, s)
        for block, row in ((False, False), (True, False), (False, True)):
            results = [
                quantize_static_fp8(
                    x, s, block_scale=block, row_scale=row, quantizer=choice, pdl=True
                )
                for choice in ("triton", "cute")
            ]
            tq, ts = results[0] if block or row else (results[0], None)
            cq, cs = results[1] if block or row else (results[1], None)
            assert_quant(tq, ref)
            assert torch.equal(tq.view(torch.uint8), cq.view(torch.uint8))
            if block or row:
                assert torch.equal(ts, cs) and torch.equal(ts, s.expand_as(ts))


GEN_SHAPES = tuple(
    (slots, k)
    for tp in (1, 2, 4)
    for slots, k in (
        (tuple(n // tp for n in (6144, 1024, 1024, 6144)), 5120),
        ((10240 // tp, 6144 // tp), 5120),
        ((5120,), 6144 // tp),
    )
) + (((504, 496), 128), ((128, 128), 1000), ((128, 128), 129))


def _precision_sample(y, q, w, ws, s, slots, bias):
    # Sample columns from every slot and rows across M; retain the full K reduction.
    rows = torch.linspace(0, y.shape[0] - 1, min(3, y.shape[0]), device="cuda").long()
    columns, start = [], 0
    for n in slots:
        columns.append(torch.linspace(start, start + n - 1, min(16, n), device="cuda").long())
        start += n
    indices = torch.cat(columns)
    assert_precision(
        y[rows][:, indices],
        q.view(torch.uint8)[rows].view(E4M3),
        w.view(torch.uint8)[indices].view(E4M3),
        ws,
        s,
        tuple(len(c) for c in columns),
        None if bias is None else bias[indices],
    )


@pytest.mark.parametrize("slots,k", GEN_SHAPES)
@pytest.mark.parametrize("quantizer", ["triton", "cute"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp8pt_gen_grid(slots, k, quantizer, dtype, with_bias):
    if quantizer == "cute" and not available(torch.cuda.get_device_capability()):
        pytest.skip("CuTe quantizer unavailable")
    n = sum(slots)
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.03 if with_bias else None
    op, w, ws, s = make_op(slots, k, None, quantizer=quantizer, bias=bias)
    assert op.quantizer == quantizer
    if n % 16 or k % 16:
        assert op.provider == "block-fp8" and getattr(op.provider, "reason", None) is None
    workspace = torch.empty(op.workspace_bytes(512) // 4, device="cuda", dtype=torch.float32)
    for index, m in enumerate((0, 1, 2, 3, 4, 17, 64, 65, 300, 512)):
        out_dtype = DTYPES[index % 2]
        x = torch.randn(m, k + 17, device="cuda", dtype=dtype)[:, 1 : k + 1]
        q = quantize_static_fp8(x, s, quantizer=quantizer)
        assert torch.equal(q.view(torch.uint8), op.quantize(x).view(torch.uint8))
        y = op(x, out_dtype=out_dtype)
        assert torch.equal(y, op.quantized(q, out_dtype=out_dtype))
        parent = torch.full((m, n + 7), -19, dtype=out_dtype, device="cuda")
        out = parent[:, 1 : n + 1]
        scratch = workspace if index % 2 else None
        assert op(x, out=out, workspace=scratch) is out and torch.equal(y, out)
        assert op.quantized(q, out=out, workspace=scratch) is out and torch.equal(y, out)
        packed = torch.empty_like(y)
        assert op(x, out=packed, workspace=scratch) is packed and torch.equal(y, packed)
        assert op.quantized(q, out=packed, workspace=scratch) is packed and torch.equal(y, packed)
        assert torch.all(parent[:, 0] == -19) and torch.all(parent[:, n + 1 :] == -19)
        if m:
            _precision_sample(y, q, w, ws, s, slots, bias)


@pytest.mark.parametrize(
    "slots,k,m,dtype",
    [
        ((128, 128), 129, 4096, torch.bfloat16),
        ((504, 496), 128, 16384, torch.float16),
        ((128, 128), 1000, 16384, torch.bfloat16),
    ],
)
def test_fp8pt_gen_large_m(slots, k, m, dtype):
    bias = torch.randn(sum(slots), device="cuda", dtype=dtype) * 0.03
    op, w, ws, s = make_op(slots, k, None, quantizer="triton", bias=bias)
    x = torch.randn(m, k + 1, device="cuda", dtype=dtype)[:, :k]
    q = quantize_static_fp8(x, s, quantizer="triton")
    y = op(x)
    out = torch.empty(m, sum(slots) + 1, dtype=dtype, device="cuda")[:, : sum(slots)]
    assert op.quantized(q, out=out) is out and torch.equal(y, out)
    _precision_sample(y, q, w, ws, s, slots, bias)


@pytest.mark.parametrize("m", [1, 3, 64])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
def test_fp8pt_gen_cold_capture(m, quantized, workspace):
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    op, _, _, s = make_op((128, 128), 129, None, quantizer="triton", bias=bias)
    x = torch.randn(m, 146, device="cuda", dtype=torch.bfloat16)[:, 1:130]
    q = reference_quant(x, s)
    arg, call = (q, op.quantized) if quantized else (x, op)
    if workspace:
        scratch = torch.empty(op.workspace_bytes(m) // 4, device="cuda", dtype=torch.float32)
        call = partial(call, workspace=scratch)
    out = torch.empty(m, 269, device="cuda", dtype=torch.bfloat16)[:, 1:257]
    graph, result = _capture_cold(
        op, call, arg, out, max_m=m, quantized=quantized, out_row_stride=269
    )
    x.normal_()
    q.copy_(reference_quant(x, s))
    expected = call(arg)
    graph.replay()
    assert result is out and torch.equal(out, expected)


def test_fp8pt_gen_visible_reroute(monkeypatch):
    from ctikernels import _reroute
    from ctikernels.linear import _fp8_static_dispatch as dispatch
    from ctikernels.linear import fp8_static as static

    def unavailable(*args, **kwargs):
        raise ImportError("test library unavailable")

    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((0, "flashinfer-bmm:auto"),))
    monkeypatch.setattr(static, "FlashInferBMM", unavailable)
    w = torch.zeros(256, 128, device="cuda", dtype=E4M3)
    ws = torch.ones(1, device="cuda")
    s = torch.ones(1, device="cuda")
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    op = StaticFP8Linear(w, ws, s, quantizer="triton")
    assert op.provider_for(1) == "block-fp8"
    assert "test library unavailable" in op.provider_for(1).reason
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(_reroute.RerouteError, match="test library unavailable"):
        StaticFP8Linear(w, ws, s, quantizer="triton")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [0.0137, 1.0, 2.0**-120, 2.0**-140, 3.1e30])
def test_fp8pt_gen_all_encodings(dtype, scale):
    x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(256, 256).cuda()
    s = torch.tensor(scale, device="cuda")
    q = quantize_static_fp8(x, s, quantizer="triton")
    assert_quant(q, reference_quant(x, s))
    if available(torch.cuda.get_device_capability()):
        cute = quantize_static_fp8(x, s, quantizer="cute")
        assert torch.equal(q.view(torch.uint8), cute.view(torch.uint8))
