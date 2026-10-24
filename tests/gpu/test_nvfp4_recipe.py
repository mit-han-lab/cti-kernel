# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Market-recipe qualification against FlashInfer 0.7.0.post1.

All requested rows cross the large and odd widths. Scales rotate across shapes
and focused cases to limit runtime; every tensor mixes block categories. Run
on Blackwell; CPU collection imports neither FlashInfer nor GPU DSL kernels.
Cold external-provider JIT time is separate from the <2 minute execution target.
"""

import pytest
import torch

from ctikernels.linear import _nvfp4_cute_quant, _nvfp4_cute_small
from ctikernels.linear.nvfp4 import NVFP4Weight, quantize_nvfp4, scale_shape

from .nvfp4_oracle import cute_quantize, flashinfer_quantize

CATEGORIES = (
    "normal",
    "zero",
    "subnormals",
    "one-subnormal",
    "positive-inf",
    "negative-inf",
    "one-nan",
    "all-nan",
    "e2m1-ties",
    "sf-448",
    "sf-min",
    "sf-half-min",
    "sf-midpoint",
    "sf-high-midpoint",
    "negative-zero",
)
SHAPES = tuple(
    (rows, cols)
    for rows in (1, 2, 3, 8, 17, 64, 65, 128, 129, 256, 512, 1024)
    for cols in (64, 96, 160, 544, 5120, 17408)
)

SCALES = (1 / 2688, 10 / 2688, 1000 / 2688, 1e-30, 1e-8, 1e8, 3e30)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("hardware E2M1 conversion requires Blackwell or newer")
    return torch.device("cuda")


def mixed_input(rows, cols, dtype, device, input_scale, focus="normal"):
    """Seeded, mixed blocks. No host reads of device scalar inputs."""
    generator = torch.Generator(device=device).manual_seed(1701 + rows + cols)
    x = torch.randn((rows * (cols // 16), 16), device=device, generator=generator)
    x *= 10
    templates = torch.zeros((len(CATEGORIES), 16), dtype=torch.float32)
    smallest = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    templates[2] = torch.tensor([smallest, -smallest] * 8)
    templates[3, 7] = smallest
    templates[4:7] = 1
    templates[4, 3] = float("inf")
    templates[5, 8] = -float("inf")
    templates[6, 11] = float("nan")
    templates[7] = float("nan")
    templates[8] = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6]
    )
    # Set amax = 6 * input_scale * target_sf, then round in the input dtype.
    # The focused cases use input_scale=1, making the specified ties exact.
    for i, sf in enumerate((448, 2**-9, 2**-10, 1.0625, 432), start=9):
        templates[i, ::2] = min(6 * input_scale * sf, torch.finfo(torch.float32).max)
        templates[i, 1::2] = -templates[i, ::2]
    templates[14] = -0.0
    category = (torch.arange(x.shape[0], device=device) + CATEGORIES.index(focus)) % len(CATEGORIES)
    x = torch.where((category == 0)[:, None], x, templates.to(device)[category])
    return x.to(dtype).reshape(rows, cols), category.reshape(rows, cols // 16)


def assert_bytes(actual, expected, x, category, label):
    """Report byte counts by category and first (row, block, ours, FI, amax)."""
    data, scales = (t.view(torch.uint8) for t in actual)
    ref_data, ref_scales = expected
    if torch.equal(data, ref_data) and torch.equal(scales, ref_scales):
        return
    rows, cols = x.shape
    blocks = x.float().cpu().reshape(rows, cols // 16, 16)
    amax = blocks.abs()[..., 0]
    for i in range(1, 16):
        amax = torch.fmax(amax, blocks.abs()[..., i])
    category = category.cpu()
    counts, examples = {}, {}
    for kind, ours_gpu, theirs_gpu in (("fp4", data, ref_data), ("sf", scales, ref_scales)):
        ours, theirs = ours_gpu.cpu(), theirs_gpu.cpu()
        if kind == "sf":
            ours = ours.permute(0, 3, 2, 1, 4).reshape(ours.shape[0] * 128, -1)
            theirs = theirs.permute(0, 3, 2, 1, 4).reshape(theirs.shape[0] * 128, -1)
        indices = (ours != theirs).nonzero()
        row_ids, col_ids = indices.unbind(1)
        block_ids = col_ids // 8 if kind == "fp4" else col_ids
        valid = (row_ids < rows) & (block_ids < cols // 16)
        ids = torch.full_like(row_ids, len(CATEGORIES))
        ids[valid] = category[row_ids[valid], block_ids[valid]]
        histogram = torch.bincount(ids, minlength=len(CATEGORIES) + 1)
        counts[kind] = {
            name: int(n) for name, n in zip((*CATEGORIES, "padding"), histogram, strict=True) if n
        }
        examples[kind] = []
        for row, col in indices[:12].tolist():
            block = col // 8 if kind == "fp4" else col
            valid_block = row < rows and block < cols // 16
            examples[kind].append(
                (
                    row,
                    block,
                    int(ours[row, col]),
                    int(theirs[row, col]),
                    float(amax[row, block]) if valid_block else 0.0,
                )
            )
    pytest.fail(
        f"{label}: mismatch byte counts/category={counts}; "
        f"first (row, block, ours, flashinfer, amax)={examples}; "
        "check H1 setp.ne.ftz and hardware NaN behavior"
    )


# (capability, device name): every capability NVFP4 admits; the names are the qualified
# sm12x devices and typical others.
ROUTES = (
    ((8, 9), "NVIDIA L40S"),
    ((9, 0), "NVIDIA H100 80GB HBM3"),
    ((10, 0), "NVIDIA B200"),
    ((10, 3), "NVIDIA B300 SXM6 AC"),
    ((12, 0), "NVIDIA GeForce RTX 5090"),
    ((12, 1), "NVIDIA GB10"),
)


@pytest.mark.parametrize("capability,name", ROUTES, ids=[f"sm{a}{b}" for (a, b), _ in ROUTES])
def test_cpu_routing_truth_table(monkeypatch, capability, name):
    """Hardware E2M1 from SM100, software E2M1 below; the quantizer and GEMM provider per
    capability; no capability >= 8.9 raises."""
    from ctikernels.linear import _nvfp4_flashinfer, _nvfp4_quack, _nvfp4_triton, nvfp4
    from ctikernels.linear._nvfp4_recipe import native_e2m1

    sm12x = capability >= (12, 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    pins = {"nvidia-cutlass-dsl": "4.8.0", "quack-kernels": "0.6.5"}
    monkeypatch.setattr(_nvfp4_quack.importlib.metadata, "version", lambda dist: pins[dist])
    monkeypatch.setattr(_nvfp4_quack, "Launcher", lambda: "quack")
    monkeypatch.setattr(_nvfp4_quack, "_launchers", {})
    monkeypatch.delenv(_nvfp4_quack.VARIABLE, raising=False)
    device = torch.device("cuda", 0)

    assert native_e2m1(capability) == (capability >= (10, 0))
    assert _nvfp4_cute_quant.available(device) == sm12x
    assert _nvfp4_flashinfer.qualified("0.7.0.post1", capability) == sm12x
    assert nvfp4._resolve("op", device, None)[0] == ("quack" if sm12x else "dequant")

    # The Triton quantizer (the only one off sm12x) launches with the matching E2M1 path.
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(kwargs)

    monkeypatch.setattr(_nvfp4_triton, "_quantize_kernel", Kernel())
    for gated in (False, True):
        rows, cols = 3, 64
        _nvfp4_triton.quantize(
            torch.zeros(rows, 2 * cols if gated else cols, dtype=torch.bfloat16),
            torch.ones(1),
            gated=gated,
            data=torch.zeros(rows, cols // 2, dtype=torch.uint8),
            scale=torch.zeros(scale_shape(rows, cols), dtype=torch.uint8),
        )
    assert [launch["NATIVE"] for launch in launches] == [capability >= (10, 0)] * 2


def test_cpu_recipe_diagnostics_cover_data_and_padded_scale_rows():
    x, category = mixed_input(3, 512, torch.bfloat16, torch.device("cpu"), 1.0)
    data = torch.zeros((3, 256), dtype=torch.uint8)
    scales = torch.zeros(scale_shape(3, 512), dtype=torch.uint8)
    expected = data.clone(), scales.clone()
    assert_bytes((data, scales), expected, x, category, "equal")
    data[0, 6 * 8] = 7  # one-nan block
    scales[0, 0, 3, 0, 0] = 1  # padded logical row 3, block 0
    with pytest.raises(pytest.fail.Exception) as failure:
        assert_bytes((data, scales), expected, x, category, "injected")
    message = str(failure.value)
    assert "'one-nan': 1" in message and "'padding': 1" in message
    assert "(0, 6, 7, 0, 1.0)" in message and "(3, 0, 1, 0, 0.0)" in message


def run_provider(provider, x, scale, monkeypatch):
    if provider == "cute-small":
        if x.shape[0] > 32:
            pytest.skip("fused small-M provider supports 1..32 rows")
        if x.shape[1] <= 64:
            pytest.skip("fused small-M provider needs K > 64 (dispatch never selects it below)")
        if not _nvfp4_cute_small.available(x.device):
            pytest.skip("requires CuTe DSL 4.8.0 on sm120/sm121")
        # Existing debug hook stores the actual GEMM input quantizer bytes.
        # No second public activation-quantization entry point is introduced.
        w = NVFP4Weight(
            torch.zeros((16, x.shape[1] // 2), dtype=torch.uint8, device=x.device),
            torch.ones((16, x.shape[1] // 16), device=x.device).to(torch.float8_e4m3fn),
            torch.ones(1, device=x.device),
        )
        prepared = _nvfp4_cute_small.prepare(w)
        config = _nvfp4_cute_small.Config(16, 128, 2, 1, 2)
        workspace = _nvfp4_cute_small.Workspace(prepared, x.shape[0], config, debug=True)
        _nvfp4_cute_small.linear(x, prepared, scale, config, workspace=workspace)
        return workspace.debug_data, workspace.debug_scale
    if provider == "cute" and not _nvfp4_cute_quant.available(x.device):
        pytest.skip("requires CuTe DSL 4.8.0 on sm120/sm121")
    monkeypatch.setenv("CTI_NVFP4_QUANT", provider)
    # Exercise the single public entry, including live input_scale reads.
    result = quantize_nvfp4(x, scale)
    return result.data, result.scale


@pytest.mark.parametrize("provider", ["cute", "triton", "cute-small"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows,cols", SHAPES)
def test_market_recipe_shape_grid(device, monkeypatch, provider, dtype, rows, cols):
    if provider == "cute-small" and rows > 32:
        pytest.skip("fused small-M provider supports 1..32 rows")
    scale_value = SCALES[SHAPES.index((rows, cols)) % len(SCALES)]
    x, category = mixed_input(rows, cols, dtype, device, scale_value)
    scale = torch.tensor([scale_value], device=device)
    expected = flashinfer_quantize(x, scale)
    actual = run_provider(provider, x, scale, monkeypatch)
    assert_bytes(actual, expected, x, category, provider)


@pytest.mark.parametrize("provider", ["cute", "triton", "cute-small"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("category", CATEGORIES)
def test_market_recipe_categories(device, monkeypatch, provider, dtype, category):
    # Fixed geometry amortizes compilation. All categories coexist in each input.
    # Reuse the same scalar storage, checking updates on every call.
    scale = torch.ones(1, device=device)
    for scale_value in (1.0, SCALES[CATEGORIES.index(category) % len(SCALES)]):
        scale.fill_(scale_value)
        x, categories = mixed_input(3, 512, dtype, device, scale_value, category)
        expected = flashinfer_quantize(x, scale)
        actual = run_provider(provider, x, scale, monkeypatch)
        assert_bytes(actual, expected, x, categories, f"{provider}/{category}/scale={scale_value}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("widen", [False, True], ids=["native", "fp32"])
def test_cute_gpu_oracle_agrees_with_flashinfer(device, dtype, widen):
    for scale_value in (1.0, *SCALES):
        x, category = mixed_input(3, 512, dtype, device, scale_value)
        scale = torch.tensor([scale_value], device=device)
        assert_bytes(
            cute_quantize(x.float() if widen else x, scale),
            flashinfer_quantize(x, scale),
            x,
            category,
            f"CuTe-helper/{dtype}/widen={widen}/scale={scale_value}",
        )


try:  # GPU-only helper; CPU collection works without Triton.
    import triton
    import triton.language as tl

    from ctikernels.linear._nvfp4_triton import _pack
except ImportError:
    _pack_kernel = None
else:

    @triton.jit
    def _pack_kernel(x, out, n, NATIVE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        low = tl.load(x + 2 * i, mask=i < n, other=0.0)
        high = tl.load(x + 2 * i + 1, mask=i < n, other=0.0)
        tl.store(out + i, _pack(low, high, NATIVE), mask=i < n)


def _special_bits():
    """FP32 bit patterns: zeros, subnormals, infinities, NaNs and every E2M1 tie and boundary
    with its neighbours, both signs."""
    import struct

    edges = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
    bits = {0, 1, 0x7FFFFF, 0x800000, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x7F7FFFFF}
    for edge in edges:
        center = struct.unpack("<I", struct.pack("<f", edge))[0]
        bits.update(center + d for d in (-2, -1, 0, 1, 2) if center + d >= 0)
    return [b | sign for b in sorted(bits) for sign in (0, 0x80000000)]


@pytest.mark.parametrize("native", [True, False], ids=["hardware", "software"])
def test_software_e2m1_matches_hardware_bit_patterns(device, native):
    """``_pack``'s software path equals ``cvt.rn.satfinite.e2m1x2.f32`` on special patterns,
    random bit patterns (every exponent) and dense values around the E2M1 range."""
    if _pack_kernel is None:
        pytest.skip("Triton unavailable")
    generator = torch.Generator(device=device).manual_seed(90)
    special = torch.tensor(_special_bits(), dtype=torch.int64).to(torch.int32)
    random_bits = torch.randint(
        -(2**31), 2**31 - 1, (1 << 22,), device=device, dtype=torch.int32, generator=generator
    )
    dense = torch.empty(1 << 22, device=device).uniform_(-7, 7, generator=generator)
    values = torch.cat(
        (special.to(device).view(torch.float32), random_bits.view(torch.float32), dense)
    )
    values = torch.cat((values, values.flip(0)))  # each value in both nibbles
    pairs = values.numel() // 2

    def run(native_path):
        out = torch.empty(pairs, dtype=torch.uint8, device=device)
        _pack_kernel[(-(-pairs // 1024),)](values, out, pairs, NATIVE=native_path, BLOCK=1024)
        return out

    expected, actual = run(True), run(native)
    mismatch = (expected != actual).nonzero().flatten()
    assert mismatch.numel() == 0, [
        (hex(int(values[2 * i].view(torch.int32)) & 0xFFFFFFFF),
         hex(int(values[2 * i + 1].view(torch.int32)) & 0xFFFFFFFF),
         int(expected[i]), int(actual[i]))
        for i in mismatch[:8].tolist()
    ]  # fmt: skip


@pytest.mark.parametrize("gated", [False, True], ids=["plain", "gated"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows,cols", SHAPES[::5])
def test_software_e2m1_quantizer_matches_hardware(device, monkeypatch, gated, dtype, rows, cols):
    """The Triton quantizer as compute capability 8.9 - 9.x runs it (software E2M1) writes the
    same bytes as with hardware E2M1, on the recipe grid's mixed blocks and scales."""
    from ctikernels.linear import _nvfp4_recipe, _nvfp4_triton

    scale_value = SCALES[SHAPES.index((rows, cols)) % len(SCALES)]
    x, _ = mixed_input(rows, 2 * cols if gated else cols, dtype, device, scale_value)
    scale = torch.tensor([scale_value], device=device)

    def run(native):
        monkeypatch.setattr(_nvfp4_recipe, "native_e2m1", lambda capability: native)
        data = torch.full((rows, cols // 2), 0xAB, dtype=torch.uint8, device=device)
        scales = torch.full(scale_shape(rows, cols), 0xAB, dtype=torch.uint8, device=device)
        _nvfp4_triton.quantize(x, scale, gated=gated, data=data, scale=scales)
        return data, scales

    (data, scales), (ref_data, ref_scales) = run(False), run(True)
    assert torch.equal(data, ref_data)
    assert torch.equal(scales, ref_scales)


@pytest.mark.parametrize("rows", [1, 5, 300])
def test_software_e2m1_dequant_ops_match_hardware(device, monkeypatch, rows):
    """NVFP4Linear and NVFP4GateUpSiLU on the path compute capability 8.9 - 9.x take
    (``dequant`` provider, Triton quantizer, software E2M1) give the same bits as with
    hardware E2M1."""
    from ctikernels.linear import _nvfp4_recipe
    from ctikernels.linear.nvfp4 import NVFP4GateUpSiLU, NVFP4Linear

    monkeypatch.setenv("CTI_NVFP4_QUANT", "triton")
    generator = torch.Generator(device=device).manual_seed(rows)

    def weight(n, k):
        data = torch.randint(0, 256, (n, k // 2), device=device, generator=generator)
        scale = torch.rand((n, k // 16), device=device, generator=generator) * 4
        return NVFP4Weight(
            data.to(torch.uint8), scale.to(torch.float8_e4m3fn), torch.tensor([0.7], device=device)
        )

    hidden, inter = 256, 192
    x, _ = mixed_input(rows, hidden, torch.bfloat16, device, 1 / 64)
    input_scale = torch.tensor([1 / 64], device=device)
    linear = NVFP4Linear(weight(320, hidden), input_scale, provider="dequant")
    gated = NVFP4GateUpSiLU(
        weight(inter, hidden),
        weight(inter, hidden),
        input_scale=input_scale,
        output_scale=torch.tensor([1 / 32], device=device),
        provider="dequant",
    )
    assert linear.provider == gated.provider == "dequant"

    def run(native):
        monkeypatch.setattr(_nvfp4_recipe, "native_e2m1", lambda capability: native)
        a = gated(x)
        return linear(x), a.data.clone(), a.scale.view(torch.uint8).clone()

    for ours, expected in zip(run(False), run(True), strict=True):
        assert torch.equal(ours.view(torch.uint8), expected.view(torch.uint8))
