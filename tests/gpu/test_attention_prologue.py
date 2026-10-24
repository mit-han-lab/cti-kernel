# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Prologue on the GPU: no worse than the norm, rotary and store composition against an FP64
reference, its bitwise invariants, determinism, CUDA-graph replay and paged varlen batches. On
the RTX 5090, RTX PRO 6000 and GB10 with the pinned DSL the kernel must run; on
other devices the composition is the expected outcome, not a skip."""

from __future__ import annotations

import importlib.metadata
import os

import pytest
import torch

from ctikernels.attention import _prologue
from ctikernels.attention.inference import Prologue

VARIANTS = [(256, 64, True), (128, 128, False)]
EPS = 1e-6
# Qwen3.5-9B TP1/2, Qwen3.8-27B TP2/1, Qwen3.5-35B-A3B TP1/2: (Q heads, KV heads) per rank;
# Qwen3-8B and the DFlash drafts TP1/2/4, Qwen3-30B-A3B TP1.
GEOMETRIES = {
    VARIANTS[0]: [(16, 4), (8, 2), (12, 2), (24, 4), (16, 2), (8, 1)],
    VARIANTS[1]: [(32, 8), (16, 4), (8, 2), (32, 4)],
}
# Model maximum context: Qwen3.5/3.8 262,144; Qwen3 131,072 (YaRN).
MAX_POSITION = {VARIANTS[0]: 262144, VARIANTS[1]: 131072}
FLOOR = 2.0**-24


@pytest.fixture(params=VARIANTS, ids=["hd256", "hd128"])
def variant(request):
    return request.param


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _cute_runs(variant) -> bool:
    """Whether the CuTe kernel runs here, after asserting why not where it does not."""
    device = _device()
    if _prologue.load(device, variant) is not None:
        return True
    try:
        installed = importlib.metadata.version(_prologue.DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    hardware = torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)
    assert (
        os.environ.get(_prologue.VARIABLE) == "0"
        or hardware not in _prologue.DEVICES[variant]
        or installed is None
    ), f"{_prologue.DISTRIBUTION} {installed} is installed; the kernel needs {_prologue.VERSION}"
    return False


def _table(variant, max_position: int = 8192) -> torch.Tensor:
    head_dim, rotary_dim, zero_centered = variant
    theta = 1e7 if zero_centered else 1e6
    inverse = theta ** -(torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    angles = torch.outer(torch.arange(max_position, dtype=torch.float64), inverse)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float().to(_device())


def _inputs(
    variant,
    tokens,
    q_heads,
    kv_heads,
    *,
    slots=4096,
    seed=0,
    positions=None,
    indices=None,
    max_position=8192,
):
    """Merged rows [Q | K | V | G], norm weights, positions, slots and caches."""
    head_dim, rotary_dim, zero_centered = variant
    generator = torch.Generator(device=_device()).manual_seed(seed)

    def normal(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=generator, device=_device())).bfloat16()

    if positions is None:
        positions = torch.randint(max_position, (tokens,), generator=generator, device=_device())
    if indices is None:
        indices = torch.randperm(slots, generator=generator, device=_device())[:tokens]
    return {
        "rows": normal(tokens, 2 * (q_heads + kv_heads) * head_dim),
        "q_weight": (normal(head_dim, scale=0.3).float() + (0 if zero_centered else 1)).bfloat16(),
        "k_weight": (normal(head_dim, scale=0.3).float() + (0 if zero_centered else 1)).bfloat16(),
        "positions": positions.long(),
        "indices": indices.to(torch.int32),
        "k_cache": normal(slots, kv_heads, head_dim),
        "v_cache": normal(slots, kv_heads, head_dim),
    }


def _call(variant, run, data, table, q_heads, kv_heads, **scales):
    """``run`` on copies of ``data``: (Q, K cache, V cache, rows afterwards)."""
    head_dim, rotary_dim, zero_centered = variant
    rows, k_cache, v_cache = data["rows"].clone(), data["k_cache"].clone(), data["v_cache"].clone()
    qkv = rows[:, : (q_heads + 2 * kv_heads) * head_dim]
    arguments = (qkv, data["q_weight"], data["k_weight"], table, data["positions"])
    arguments += (k_cache, v_cache, data["indices"])
    if run is None:
        _prologue._compose(*arguments, EPS, q_heads, kv_heads, head_dim, zero_centered)
        q = qkv[:, : q_heads * head_dim].view(-1, q_heads, head_dim)
    else:
        q = run(*arguments, **scales)
    return q, k_cache, v_cache, rows


def _reference(variant, data, table, q_heads, kv_heads):
    """Q and the written K rows in FP64."""
    head_dim, rotary_dim, zero_centered = variant
    rows = data["rows"].double()
    tokens = rows.shape[0]
    q = rows[:, : q_heads * head_dim].view(tokens, q_heads, head_dim)
    k = rows[:, q_heads * head_dim : (q_heads + kv_heads) * head_dim].view(tokens, kv_heads, -1)
    cos, sin = table[data["positions"].long()].double()[:, None].chunk(2, dim=-1)

    def prologue(x, weight):
        x = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
            * (int(zero_centered) + weight.double())
        )
        first, second = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
        rotated = (first * cos - second * sin, second * cos + first * sin)
        return torch.cat((*rotated, x[..., rotary_dim:]), dim=-1)

    return prologue(q, data["q_weight"]), prologue(k, data["k_weight"])


def _errors(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """The RMS error and the largest error in units of the reference's BF16 spacing."""
    error = (output.double() - reference).abs()
    spacing = torch.exp2(torch.floor(torch.log2(reference.abs().clamp_min(2.0**-126))) - 7)
    return error.pow(2).mean().sqrt().item(), (error / spacing).max().item()


def _operation(variant, q_heads, kv_heads, cache_dtype=torch.bfloat16, *, masked=False) -> Prologue:
    head_dim, rotary_dim, zero_centered = variant
    return Prologue(
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        eps=EPS,
        device=_device(),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        zero_centered=zero_centered,
        cache_dtype=cache_dtype,
        masked=masked,
    )


def _check(variant, data, table, q_heads, kv_heads, q, k_cache, v_cache, rows):
    """Against FP64 no worse than the composition (RMS and largest error in BF16 spacings);
    V, G and unwritten slots bit for bit."""
    head_dim, rotary_dim, zero_centered = variant
    composed = _call(variant, None, data, table, q_heads, kv_heads)
    reference_q, reference_k = _reference(variant, data, table, q_heads, kv_heads)
    valid = data["indices"] >= 0
    slots = data["indices"][valid].long()
    for output, baseline, reference in (
        (q, composed[0], reference_q),
        (k_cache[slots], composed[1][slots], reference_k[valid]),
    ):
        if not output.numel():
            continue
        rms, peak = _errors(output, reference)
        baseline_rms, baseline_peak = _errors(baseline, reference)
        assert rms <= baseline_rms + FLOOR, (rms, baseline_rms)
        assert peak <= baseline_peak, (peak, baseline_peak)
        if _cute_runs(variant):  # one rounding: nearly every element is the correctly rounded value
            exact = (output == reference.bfloat16()).double().mean().item()
            assert exact >= 0.999, exact
            # Cancellation near zero can amplify spacing error; allow one BF16 spacing
            # relative to the head's peak magnitude, as well as the compose comparison.
            spacing = torch.exp2(
                torch.floor(torch.log2(reference.abs().amax(-1, keepdim=True))) - 7
            )
            assert torch.all((output.double() - reference).abs() <= spacing)
    v_start = (q_heads + kv_heads) * head_dim
    v = data["rows"][:, v_start : v_start + kv_heads * head_dim].view(-1, kv_heads, head_dim)
    assert torch.equal(v_cache[slots], v[valid])
    assert torch.equal(rows[:, v_start:], data["rows"][:, v_start:])
    kept = torch.ones(k_cache.shape[0], dtype=torch.bool, device=k_cache.device)
    kept[slots] = False
    assert torch.equal(k_cache.view(torch.uint8)[kept], data["k_cache"].view(torch.uint8)[kept])
    assert torch.equal(v_cache.view(torch.uint8)[kept], data["v_cache"].view(torch.uint8)[kept])


def test_each_device_runs_its_implementation(variant, monkeypatch):
    """The CuTe kernel on qualified devices with the pinned package, the composition elsewhere."""
    composed, compose = [], _prologue._compose
    with monkeypatch.context() as patch:
        patch.setattr(_prologue, "_compose", lambda *args: composed.append(compose(*args)))
        data, table = _inputs(variant, 5, 16, 4), _table(variant)
        op = _operation(variant, 16, 4)
        assert op.provider == ("cute" if _cute_runs(variant) else "compose")
        outputs = _call(variant, op, data, table, 16, 4)
    assert len(composed) == (0 if _cute_runs(variant) else 1)
    _check(variant, data, table, 16, 4, *outputs)


@pytest.mark.parametrize(
    "tokens,variant,geometry",
    [
        pytest.param(
            tokens,
            variant,
            geometry,
            marks=()
            if (tokens, variant[0], geometry)
            in {
                (8192, 256, (8, 2)),
                (33, 128, (32, 8)),
                (4, 128, (16, 4)),
                (3, 128, (16, 4)),
                (8192, 128, (32, 8)),
                (4097, 256, (24, 4)),
                (8, 256, (16, 2)),
                (8, 256, (8, 1)),
            }
            else pytest.mark.exhaustive,
            id=f"{tokens}-hd{variant[0]}-{geometry[0]}x{geometry[1]}",
        )
        for tokens in [1, 2, 3, 4, 8, 16, 32, 33, 48, 64, 129, 1000, 4097, 8192]
        for variant, geometries in GEOMETRIES.items()
        for geometry in geometries
    ],
)
def test_no_worse_than_the_composition_against_fp64(variant, geometry, tokens):
    data, table = (
        _inputs(variant, tokens, *geometry, slots=max(4096, 2 * tokens), seed=tokens),
        _table(variant),
    )
    _check(
        variant,
        data,
        table,
        *geometry,
        *_call(variant, _operation(variant, *geometry), data, table, *geometry),
    )


def test_paged_varlen_batch_at_the_largest_positions(variant):
    """Packed sequences restarting their positions near the table's end, written to scattered
    pages of the engine's cache layout (K and V halves of one buffer, 16-token pages)."""
    head_dim, rotary_dim, zero_centered = variant
    q_heads, kv_heads, page = 16, 4, 16
    lengths = [1, 17, 300, 5]
    table = _table(variant, 262144)
    positions = torch.cat([262144 - length + torch.arange(length) for length in lengths])
    pages = torch.randperm(64, generator=torch.Generator().manual_seed(3))
    slots = torch.cat(
        [pages[i * 20 + torch.arange(length) // page] * page + torch.arange(length) % page
         for i, length in enumerate(lengths)]
    )  # fmt: skip
    data = _inputs(
        variant, sum(lengths), q_heads, kv_heads, slots=64 * page, positions=positions.cuda(),
        indices=slots.cuda(),
    )  # fmt: skip
    buffer = torch.stack((data["k_cache"], data["v_cache"])).view(2, 64, page, kv_heads, -1)
    rows = data["rows"].clone()
    q = _operation(variant, q_heads, kv_heads)(
        rows[:, : (q_heads + 2 * kv_heads) * head_dim], data["q_weight"], data["k_weight"],
        table, data["positions"], buffer[0].view(-1, kv_heads, head_dim),
        buffer[1].view(-1, kv_heads, head_dim), data["indices"],
    )  # fmt: skip
    _check(
        variant, data, table, q_heads, kv_heads, q, *buffer.view(2, -1, kv_heads, head_dim), rows
    )


def test_padded_slots_and_empty_calls(variant):
    head_dim, rotary_dim, zero_centered = variant
    q_heads, kv_heads = 8, 2
    data, table = _inputs(variant, 40, q_heads, kv_heads, slots=128), _table(variant)
    op = _operation(variant, q_heads, kv_heads)
    padded = torch.zeros(2, 128, kv_heads + 1, head_dim, dtype=torch.bfloat16, device=_device())
    padded[0, :, :kv_heads], padded[1, :, :kv_heads] = data["k_cache"], data["v_cache"]
    rows = data["rows"].clone()
    qkv = rows[:, : (q_heads + 2 * kv_heads) * head_dim]
    arguments = (data["q_weight"], data["k_weight"], table, data["positions"])
    k_cache, v_cache = padded[0, :, :kv_heads], padded[1, :, :kv_heads]
    q = op(qkv, *arguments, k_cache, v_cache, data["indices"])
    _check(variant, data, table, q_heads, kv_heads, q, k_cache, v_cache, rows)
    assert torch.equal(padded[:, :, kv_heads], torch.zeros_like(padded[:, :, kv_heads]))
    before = padded.clone()
    empty = op(
        qkv[:0], *arguments[:3], data["positions"][:0], k_cache, v_cache, data["indices"][:0]
    )
    assert empty.shape == (0, q_heads, head_dim)
    assert torch.equal(padded, before)


def test_shared_slots_leave_only_that_slot_undefined(variant):
    """Padding tokens sent to one sink slot: Q and every other slot exact."""
    head_dim, rotary_dim, zero_centered = variant
    geometry, tokens, valid = (16, 4), 64, 48
    data, table = _inputs(variant, tokens, *geometry), _table(variant)
    spare = torch.ones(data["k_cache"].shape[0], dtype=torch.bool, device=_device())
    spare[data["indices"][:valid].long()] = False
    spares = spare.nonzero()[:, 0].int()
    sink = spares[0].item()
    indices = data["indices"].clone()
    indices[valid:] = sink
    op = _operation(variant, *geometry)
    q, k_cache, v_cache, rows = _call(variant, op, dict(data, indices=indices), table, *geometry)
    # The same call with the padding tokens sent to distinct spare slots instead.
    unique = indices.clone()
    unique[valid:] = spares[1 : 1 + tokens - valid]
    expected = _call(variant, op, dict(data, indices=unique), table, *geometry)
    other = torch.ones_like(spare)
    other[sink] = False
    other[unique[valid:].long()] = False
    assert torch.equal(q, expected[0])
    assert torch.equal(rows, expected[3])
    assert torch.equal(k_cache[other], expected[1][other])
    assert torch.equal(v_cache[other], expected[2][other])
    # Each sink element is one of the writers' values (writes may interleave).
    for cache, rows_of in ((k_cache, expected[1]), (v_cache, expected[2])):
        written = rows_of[unique[valid:].long()]
        assert (cache[sink][None] == written).any(0).all()


def test_repeated_calls_agree_bitwise(variant):
    geometry = GEOMETRIES[variant][2]
    data, table = _inputs(variant, 300, *geometry), _table(variant)
    op = _operation(variant, *geometry)
    first = _call(variant, op, data, table, *geometry)
    second = _call(variant, op, data, table, *geometry)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16, 32, 48, 64, 500])
def test_graph_replay_on_new_data_matches_eager(variant, tokens):
    head_dim, rotary_dim, zero_centered = variant
    q_heads, kv_heads = 16, 4
    op, table = _operation(variant, q_heads, kv_heads), _table(variant)
    buffers = _inputs(variant, tokens, q_heads, kv_heads, seed=1)
    qkv = buffers["rows"][:, : (q_heads + 2 * kv_heads) * head_dim]
    arguments = (qkv, buffers["q_weight"], buffers["k_weight"], table, buffers["positions"])
    arguments += (buffers["k_cache"], buffers["v_cache"], buffers["indices"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*arguments)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q = op(*arguments)
    for seed in (2, 3):
        fresh = _inputs(variant, tokens, q_heads, kv_heads, seed=seed)
        for name, tensor in fresh.items():
            buffers[name].copy_(tensor)
        graph.replay()
        eager = _call(variant, op, fresh, table, q_heads, kv_heads)
        assert torch.equal(q, eager[0])
        assert torch.equal(buffers["k_cache"], eager[1])
        assert torch.equal(buffers["v_cache"], eager[2])


def test_calls_allocate_nothing(variant):
    head_dim, rotary_dim, zero_centered = variant
    data, table = _inputs(variant, 2048, 16, 4), _table(variant)
    op = _operation(variant, 16, 4)
    _call(variant, op, data, table, 16, 4)
    rows, k_cache, v_cache = data["rows"], data["k_cache"], data["v_cache"]
    qkv = rows[:, : 24 * head_dim]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    op(qkv, data["q_weight"], data["k_weight"], table, data["positions"], k_cache, v_cache,
       data["indices"])  # fmt: skip
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda a: a.update(qkv=a["qkv"].half()), TypeError),
        (lambda a: a.update(positions=a["positions"].int()), TypeError),
        (lambda a: a.update(qkv=a["rows"][:, 1 : 1 + a["qkv"].shape[1]]), ValueError),
        (lambda a: a.update(q_weight=torch.stack((a["q_weight"],) * 2, 1)[:, 0]), ValueError),
        (lambda a: a.update(cache_indices=a["cache_indices"][:-1]), ValueError),
        (lambda a: a.update(k_cache=a["k_cache"][:, :1]), ValueError),
        (lambda a: a.update(v_cache=a["v_cache"].transpose(0, 1)), ValueError),
        (lambda a: a.update(v_cache=torch.cat((a["v_cache"],) * 2, 1)[:, :2]), ValueError),
        (lambda a: a.update(cos_sin_cache=a["cos_sin_cache"][:, :32]), ValueError),
    ],
)
def test_contract_violations_raise_before_anything_runs(variant, change, error):
    head_dim, rotary_dim, zero_centered = variant
    data, table = _inputs(variant, 6, 8, 2), _table(variant)
    rows = data["rows"].clone()
    arguments = {
        "qkv": rows[:, : 12 * head_dim],
        "q_weight": data["q_weight"],
        "k_weight": data["k_weight"],
        "cos_sin_cache": table,
        "positions": data["positions"],
        "k_cache": data["k_cache"].clone(),
        "v_cache": data["v_cache"].clone(),
        "cache_indices": data["indices"],
    }
    op = _operation(variant, 8, 2)
    op(**arguments)
    before = rows.clone()
    arguments["rows"] = rows
    change(arguments)
    del arguments["rows"]
    with pytest.raises(error):
        op(**arguments)
    assert torch.equal(rows, before)


FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def _fp8_ulp(reference, dtype):
    info = torch.finfo(dtype)
    magnitude = reference.abs().clamp(min=info.tiny, max=info.max)
    return torch.exp2(torch.floor(torch.log2(magnitude))) * info.eps


def _fp8_inputs(
    variant, tokens, dtype, case="random", seed=0, geometry=None, band=None, max_position=None
):
    """``band`` = "long": positions in the last ``tokens`` positions of the model's context
    and slots at the top of a cache of model-max slots (capped extreme band)."""
    geometry = geometry or GEOMETRIES[variant][0]
    if band == "long":
        top = MAX_POSITION[variant]
        generator = torch.Generator(device=_device()).manual_seed(seed)
        positions = top - 1 - torch.randperm(tokens, generator=generator, device=_device())
        data = _inputs(
            variant, tokens, *geometry, slots=top, seed=seed, positions=positions, indices=positions
        )
    else:
        data = _inputs(
            variant,
            tokens,
            *geometry,
            slots=max(64, 2 * tokens),
            seed=seed,
            max_position=max_position or MAX_POSITION[variant],
        )
    if case == "large":
        # Norm removes input magnitude; large weights also force K saturation.
        data["rows"].mul_(100)
        data["k_weight"].mul_(1000)
    elif case == "tiny":
        # Include subnormal FP8 inputs along with values that saturate at tiny scales.
        start = sum(geometry) * variant[0]
        data["rows"][:, start : start + 8] = torch.tensor(
            [0, 2**-28, -(2**-28), 2**-20, -(2**-20), 2**-12, 1, -1],
            device=_device(),
            dtype=torch.bfloat16,
        )
    for name in ("k_cache", "v_cache"):
        data[name] = data[name].to(dtype)
    values = (1e-5, 3e-5) if case == "tiny" else (0.0478, 0.0213)
    scales = {
        name: None if case == "none" else torch.tensor([value], device=_device())
        for name, value in zip(("k_scale", "v_scale"), values, strict=True)
    }
    return data, scales


def _check_fp8(
    variant,
    data,
    table,
    dtype,
    scales,
    outputs,
    *,
    compose=False,
    geometry=None,
    check_q=False,
    masked=False,
):
    geometry = geometry or GEOMETRIES[variant][0]
    q_heads, kv_heads = geometry
    q, kc, vc, rows = outputs
    baseline = dict(data)
    for name in ("k_cache", "v_cache"):
        baseline[name] = data[name].bfloat16()
    bf16 = _call(variant, _operation(variant, *geometry, masked=masked), baseline, table, *geometry)
    assert torch.equal(q, bf16[0])
    if check_q:
        _check(variant, baseline, table, *geometry, *bf16)
    _, oracle = _reference(variant, data, table, *geometry)
    k_scale = 1 if scales["k_scale"] is None else scales["k_scale"].double()
    v_scale = 1 if scales["v_scale"] is None else scales["v_scale"]
    limit = torch.finfo(dtype).max
    expected = (oracle / k_scale).clamp(-limit, limit)
    valid = data["indices"] >= 0
    slots = data["indices"][valid].long()
    if compose:
        # compose quantizes its BF16 K, whose rounding near RoPE cancellation can exceed any
        # FP8 ulp bound of the oracle: it must match that quantization bitwise.
        bf16_k = bf16[1][slots].float()
        k_div = 1 if scales["k_scale"] is None else scales["k_scale"]
        exact = (bf16_k / k_div).clamp(-limit, limit).to(dtype).view(torch.uint8)
        assert torch.equal(kc.view(torch.uint8)[slots], exact)
    else:
        actual = kc.view(torch.uint8)[slots].view(dtype).double()
        ulps = (actual - expected[valid]).abs() / _fp8_ulp(expected[valid], dtype)
        assert bool((ulps <= 1).all()), ulps.max().item()
    start = (q_heads + kv_heads) * variant[0]
    v = data["rows"][:, start : start + kv_heads * variant[0]].reshape_as(oracle)
    expected_v = (v.float() / v_scale).clamp(-limit, limit).to(dtype).view(torch.uint8)
    assert torch.equal(vc.view(torch.uint8)[slots], expected_v[valid])
    assert torch.equal(rows[:, start:], data["rows"][:, start:])
    kept = torch.ones(kc.shape[0], device=kc.device, dtype=torch.bool)
    kept[slots] = False
    for cache, name in ((kc, "k_cache"), (vc, "v_cache")):
        assert torch.equal(cache.view(torch.uint8)[kept], data[name].view(torch.uint8)[kept])


def _guarded_caches(data):
    slots, kv_heads, head_dim = data["k_cache"].shape
    buffer = torch.full(
        (2, slots + 2, kv_heads, head_dim), 17, dtype=torch.bfloat16, device=_device()
    ).to(data["k_cache"].dtype)
    buffer[0, 1:-1].copy_(data["k_cache"])
    buffer[1, 1:-1].copy_(data["v_cache"])
    return buffer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens", [1, 4, 64, 512, 2048])
def test_masked_matches_unmasked_on_valid_slots(variant, dtype, tokens):
    geometry, table = GEOMETRIES[variant][0], _table(variant, 128)
    if dtype == torch.bfloat16:
        data = _inputs(variant, tokens, *geometry, slots=2 * tokens, max_position=128, seed=tokens)
        scales = {}
    else:
        data, scales = _fp8_inputs(
            variant, tokens, dtype, geometry=geometry, max_position=128, seed=tokens
        )
    outputs = [
        _call(
            variant,
            _operation(variant, *geometry, cache_dtype=dtype, masked=masked),
            data,
            table,
            *geometry,
            **scales,
        )
        for masked in (False, True)
    ]
    # Compare Q, both complete caches, and the projection rows byte for byte.
    for unmasked, masked in zip(*outputs, strict=True):
        assert torch.equal(unmasked.view(torch.uint8), masked.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.bfloat16, *FP8_DTYPES])
@pytest.mark.parametrize("tokens", [1, 3, 4, 24, 64, 256, 512, 2048])
@pytest.mark.parametrize("padding", ["mixed", "all"])
def test_negative_indices_leave_unwritten_cache_bytes_and_guards_unchanged(
    variant, dtype, tokens, padding
):
    geometry, table = (16, 4), _table(variant, 128)
    if dtype == torch.bfloat16:
        data = _inputs(variant, tokens, *geometry, slots=2 * tokens, max_position=128, seed=tokens)
        scales = {}
    else:
        data, scales = _fp8_inputs(
            variant, tokens, dtype, geometry=geometry, max_position=128, seed=tokens
        )
    if padding == "all":
        data["indices"].fill_(-1)
    else:
        data["indices"][::3] = -1
        data["indices"][-1] = -1
    buffer = _guarded_caches(data)
    before = buffer.clone()
    k_cache, v_cache = buffer[:, 1:-1]
    rows = data["rows"].clone()
    qkv = rows[:, : (geometry[0] + 2 * geometry[1]) * variant[0]]
    op = _operation(variant, *geometry, cache_dtype=dtype, masked=True)
    q = op(
        qkv, data["q_weight"], data["k_weight"], table, data["positions"],
        k_cache, v_cache, data["indices"], **scales,
    )  # fmt: skip
    outputs = q, k_cache, v_cache, rows
    if dtype == torch.bfloat16:
        _check(variant, data, table, *geometry, *outputs)
    else:
        _check_fp8(
            variant, data, table, dtype, scales, outputs,
            compose=op.provider == "compose", geometry=geometry, check_q=True, masked=True,
        )  # fmt: skip
    kept = torch.ones(buffer.shape[1], dtype=torch.bool, device=_device())
    kept[data["indices"][data["indices"] >= 0].long() + 1] = False
    assert torch.equal(buffer.view(torch.uint8)[:, kept], before.view(torch.uint8)[:, kept])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens", [4, 512])
def test_graph_replay_reads_updated_negative_indices(variant, dtype, tokens):
    geometry, table = (16, 4), _table(variant, 128)
    if dtype == torch.bfloat16:
        data = _inputs(variant, tokens, *geometry, slots=2 * tokens, max_position=128, seed=1)
        scales = {}
    else:
        data, scales = _fp8_inputs(
            variant, tokens, dtype, geometry=geometry, max_position=128, seed=1
        )
    indices = data["indices"].clone()
    data["indices"][::3] = -1
    data["indices"][-1] = -1
    buffer = _guarded_caches(data)
    before = buffer.clone()
    data["k_cache"], data["v_cache"] = buffer[:, 1:-1]
    qkv = data["rows"][:, : (geometry[0] + 2 * geometry[1]) * variant[0]]
    args = (
        qkv, data["q_weight"], data["k_weight"], table, data["positions"],
        data["k_cache"], data["v_cache"], data["indices"],
    )  # fmt: skip
    op = _operation(variant, *geometry, cache_dtype=dtype, masked=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*args, **scales)
    torch.cuda.current_stream().wait_stream(stream)
    count = len(op._kernel._compiled) if op.provider == "cute" else 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q = op(*args, **scales)
    for seed, padding in enumerate(("mixed", "all", "none", "mixed"), start=2):
        fresh = _inputs(variant, tokens, *geometry, slots=2 * tokens, max_position=128, seed=seed)
        data["rows"].copy_(fresh["rows"])
        data["indices"].copy_(indices)
        if padding == "mixed":
            data["indices"][1::3] = -1
        elif padding == "all":
            data["indices"].fill_(-1)
        buffer.copy_(before)
        eager = _call(variant, op, data, table, *geometry, **scales)
        graph.replay()
        if op.provider == "cute":
            assert len(op._kernel._compiled) == count
        assert torch.equal(q.view(torch.uint8), eager[0].view(torch.uint8))
        for cache, expected in zip(buffer[:, 1:-1], eager[1:3], strict=True):
            assert torch.equal(cache.view(torch.uint8), expected.view(torch.uint8))
        kept = torch.ones(buffer.shape[1], dtype=torch.bool, device=_device())
        kept[data["indices"][data["indices"] >= 0].long() + 1] = False
        assert torch.equal(buffer.view(torch.uint8)[:, kept], before.view(torch.uint8)[:, kept])


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize(
    "tokens,case",
    [
        pytest.param(
            tokens,
            case,
            marks=()
            if (tokens, case) in {(1, "random"), (3, "tiny"), (37, "large"), (2048, "none")}
            else pytest.mark.exhaustive,
        )
        for tokens in (1, 2, 3, 4, 37, 300, 2048)
        for case in ("random", "large", "tiny", "none")
    ],
)
def test_fp8_caches_against_fp64(variant, dtype, tokens, case):
    data, scales = _fp8_inputs(variant, tokens, dtype, case, seed=tokens)
    geometry, table = GEOMETRIES[variant][0], _table(variant, MAX_POSITION[variant])
    op = _operation(variant, *geometry, cache_dtype=dtype)
    assert op.provider == ("cute" if _cute_runs(variant) else "compose")
    if case == "large":
        _, k = _reference(variant, data, table, *geometry)
        assert bool((k.abs() / scales["k_scale"] > 448).any())
    outputs = _call(variant, op, data, table, *geometry, **scales)
    _check_fp8(variant, data, table, dtype, scales, outputs, compose=op.provider == "compose")


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize(
    "variant,geometry,tokens,band",
    [
        pytest.param(
            variant,
            geometry,
            tokens,
            band,
            marks=()
            if (tokens, band) in {(4, None), (300, "long")}
            or (band == "long" and tokens == 16384 and geometry == GEOMETRIES[variant][0])
            else pytest.mark.exhaustive,
            id=f"hd{variant[0]}-{geometry[0]}x{geometry[1]}-{tokens}-{band or 'random'}",
        )
        for variant in VARIANTS
        for geometry in GEOMETRIES[variant]
        for tokens, band in ((1, None), (4, None), (300, "long"), (4097, None), (16384, "long"))
    ],
)
def test_fp8_geometries_and_long_context(variant, geometry, tokens, band, dtype):
    """Every dense geometry, positions over the model's whole context, and a capped extreme
    band: a 16k-token chunk at the end of the context into a cache of model-max slots."""
    data, scales = _fp8_inputs(
        variant, tokens, dtype, seed=tokens + sum(geometry), geometry=geometry, band=band
    )
    table = _table(variant, MAX_POSITION[variant])
    op = _operation(variant, *geometry, cache_dtype=dtype)
    assert op.provider == ("cute" if _cute_runs(variant) else "compose")
    outputs = _call(variant, op, data, table, *geometry, **scales)
    _check_fp8(
        variant,
        data,
        table,
        dtype,
        scales,
        outputs,
        compose=op.provider == "compose",
        geometry=geometry,
    )


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_fp8_compose(variant, dtype, monkeypatch):
    # Loader resolutions are cached per device/variant, so isolate the forced resolution.
    monkeypatch.setenv(_prologue.VARIABLE, "0")
    monkeypatch.setattr(_prologue, "_launchers", {})
    data, scales = _fp8_inputs(variant, 37, dtype)
    geometry, table = GEOMETRIES[variant][0], _table(variant, MAX_POSITION[variant])
    op = _operation(variant, *geometry, cache_dtype=dtype)
    assert op.provider == "compose"
    outputs = _call(variant, op, data, table, *geometry, **scales)
    _check_fp8(variant, data, table, dtype, scales, outputs, compose=True)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_fp8_graph_reads_updated_scales_positions_and_indices(variant, dtype):
    geometry, tokens = GEOMETRIES[variant][0], 4
    data, scales = _fp8_inputs(variant, tokens, dtype, seed=1)
    table = _table(variant, MAX_POSITION[variant])
    op = _operation(variant, *geometry, cache_dtype=dtype)
    op.warm_up(tokens)
    count = len(op._kernel._compiled) if op.provider == "cute" else 0
    qkv = data["rows"][:, : (geometry[0] + 2 * geometry[1]) * variant[0]]
    args = (
        qkv,
        data["q_weight"],
        data["k_weight"],
        table,
        data["positions"],
        data["k_cache"],
        data["v_cache"],
        data["indices"],
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*args, **scales)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q = op(*args, **scales)
    if op.provider == "cute":
        assert len(op._kernel._compiled) == count
    for seed, values in ((2, (0.0317, 0.0831)), (3, (1.0, 0.0067))):
        fresh, _ = _fp8_inputs(variant, tokens, dtype, seed=seed)
        for name, tensor in fresh.items():
            data[name].copy_(tensor)
        for scale, value in zip(scales.values(), values, strict=True):
            scale.fill_(value)
        graph.replay()
        eager = _call(variant, op, fresh, table, *geometry, **scales)
        assert torch.equal(q, eager[0])
        for name, expected in zip(("k_cache", "v_cache"), eager[1:3], strict=True):
            assert torch.equal(data[name].view(torch.uint8), expected.view(torch.uint8))
        if op.provider == "cute":
            assert len(op._kernel._compiled) == count
    assert op.warm_up(tokens) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, *FP8_DTYPES])
@pytest.mark.parametrize("masked", [False, True])
def test_host_fast_path_engine_layers_and_warmed_capture(variant, dtype, masked, monkeypatch):
    heads, kv_heads = 32, 8
    hd = variant[0]
    op = _operation(variant, heads, kv_heads, dtype, masked=masked)
    full = _operation(variant, heads, kv_heads, dtype, masked=masked)

    def check_full(qkv, *args):
        full._check_full(qkv, *args)
        return qkv.shape[0]

    monkeypatch.setattr(full, "_check", check_full)
    full_checks = []
    original_check = op._check_full

    def counted(*args):
        full_checks.append(1)
        return original_check(*args)

    monkeypatch.setattr(op, "_check_full", counted)
    table = _table(variant, 16)
    layers = [
        _inputs(variant, 8, heads, kv_heads, slots=16, max_position=16, seed=i) for i in range(3)
    ]
    pools = []
    for layer in layers:
        for name in ("q_weight", "k_weight"):
            layer[name] = torch.nn.Parameter(layer[name], requires_grad=False)
        pools.append([layer[name].to(dtype).unsqueeze(1) for name in ("k_cache", "v_cache")])
    reference_pools = [[pool.clone() for pool in pair] for pair in pools]
    scales = (
        {}
        if dtype == torch.bfloat16
        else {
            "k_scale": torch.tensor([0.7], device=_device()),
            "v_scale": torch.tensor([1.3], device=_device()),
        }
    )
    for tokens in (1, 3, 7):
        positions = torch.arange(tokens, device=_device(), dtype=torch.int64)
        indices = torch.arange(tokens, device=_device(), dtype=torch.int32)
        if masked:
            indices[::2] = -1
        for layer, pair, reference_pair in zip(layers, pools, reference_pools, strict=True):
            rows = layer["rows"].clone()
            reference_rows = rows.clone()
            qkv = rows[1 : tokens + 1, : (heads + 2 * kv_heads) * hd]
            reference_qkv = reference_rows[1 : tokens + 1, : (heads + 2 * kv_heads) * hd]
            args = (layer["q_weight"], layer["k_weight"], table, positions)
            q = op(qkv, *args, pair[0].squeeze(1), pair[1].squeeze(1), indices, **scales)
            expected = full(
                reference_qkv,
                *args,
                reference_pair[0].squeeze(1),
                reference_pair[1].squeeze(1),
                indices,
                **scales,
            )
            assert torch.equal(q, expected)
            for cache, reference_cache in zip(pair, reference_pair, strict=True):
                assert torch.equal(cache.view(torch.uint8), reference_cache.view(torch.uint8))
            view = qkv[:, : heads * hd].view(tokens, heads, hd)
            assert q.shape == view.shape and q.stride() == view.stride()
            assert q.storage_offset() == view.storage_offset()
            assert q._base is view._base
            before = len(full_checks)
            # Fresh engine views of the same pools must hit the binding on every call.
            op(qkv, *args, pair[0].squeeze(1), pair[1].squeeze(1), indices, **scales)
            assert len(full_checks) == before
        assert len(full_checks) == (3 if tokens == 1 else 4 if tokens == 3 else 5)
    if op.provider != "cute":
        return
    op.warm_up(7)
    original = qkv.clone()
    cache_originals = [pool.clone() for pool in pair]
    expected = full(
        reference_qkv.copy_(original),
        *args,
        reference_pair[0].squeeze(1),
        reference_pair[1].squeeze(1),
        indices,
        **scales,
    ).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(qkv, *args, pair[0].squeeze(1), pair[1].squeeze(1), indices, **scales)
    qkv.copy_(original)
    for pool, saved in zip(pair, cache_originals, strict=True):
        pool.copy_(saved)
    graph.replay()
    assert torch.equal(captured, expected)
    for cache, reference_cache in zip(pair, reference_pair, strict=True):
        assert torch.equal(cache.view(torch.uint8), reference_cache.view(torch.uint8))


def test_capture_uncompiled_count_does_not_compile(variant):
    op = _operation(variant, 32, 8)
    if op.provider != "cute":
        pytest.skip("CuTe compile guard")
    # Isolate this launcher's compile cache from other tests and warm only the T=1 config.
    op._kernel = _prologue.Launcher(_device(), op._kernel._kernels, variant)
    op.warm_up(1)
    data, table = _inputs(variant, 128, 32, 8, slots=128), _table(variant)
    qkv = data["rows"][:, : 48 * variant[0]]
    before = len(op._kernel._compiled)
    with pytest.raises(RuntimeError, match="warm_up.*eagerly"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(
                qkv,
                data["q_weight"],
                data["k_weight"],
                table,
                data["positions"],
                data["k_cache"],
                data["v_cache"],
                data["indices"],
            )
    assert len(op._kernel._compiled) == before


def test_warmed_layouts_neither_compile_nor_wait_for_queued_work(variant):
    """Two prefix-hit layouts of different configurations, first served after warm_up while
    the device is busy: no compile, and the call returns before the queued work finishes
    (CUDA loads a kernel on its first launch, which waits for the device, so warm_up must
    have launched each one)."""
    q_heads, kv_heads = GEOMETRIES[variant][0]
    op = _operation(variant, q_heads, kv_heads)
    if op.provider != "cute":
        pytest.skip("CuTe warm-up")
    op._kernel = _prologue.Launcher(_device(), op._kernel._kernels, variant)
    configs = {}
    for tokens in range(4096, 0, -1):
        key = _prologue.config(tokens, q_heads, kv_heads, op._kernel._l2_bytes, variant[0])
        configs.setdefault(key, tokens)
    op.warm_up(4096)
    compiled = len(op._kernel._compiled)
    table, busy = _table(variant), torch.randn(4096, 4096, device=_device())
    for tokens in sorted(configs.values())[-3:-1]:
        prefix = 2048
        data = _inputs(
            variant, tokens, q_heads, kv_heads, slots=prefix + tokens,
            positions=torch.arange(prefix, prefix + tokens, device=_device()),
            indices=torch.arange(prefix, prefix + tokens, device=_device()),
        )  # fmt: skip
        qkv = data["rows"][:, : (q_heads + 2 * kv_heads) * variant[0]]
        torch.cuda.synchronize()
        for _ in range(64):
            busy.matmul(busy)
        queued = torch.cuda.Event()
        queued.record()
        op(qkv, data["q_weight"], data["k_weight"], table, data["positions"], data["k_cache"],
           data["v_cache"], data["indices"])  # fmt: skip
        assert not queued.query(), tokens
        torch.cuda.synchronize()
    assert len(op._kernel._compiled) == compiled


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_bound_tensor_in_another_role_takes_the_full_check(variant, dtype, monkeypatch):
    """A tensor bound in one role (q_weight, cos_sin_cache, a cache, a scale, ...) offered in
    another must run that role's full check and raise its original error, if any."""
    head_dim = variant[0]
    data, table = _inputs(variant, 6, 8, 2), _table(variant)
    arguments = {
        "q_weight": data["q_weight"],
        "k_weight": data["k_weight"],
        "cos_sin_cache": table,
        "positions": data["positions"],
        "k_cache": data["k_cache"].to(dtype),
        "v_cache": data["v_cache"].to(dtype),
        "cache_indices": data["indices"],
    }
    if dtype != torch.bfloat16:
        arguments["k_scale"] = torch.tensor([0.7], device=_device())
        arguments["v_scale"] = torch.tensor([1.3], device=_device())
    qkv = data["rows"].clone()[:, : 12 * head_dim]
    op, fresh = _operation(variant, 8, 2, dtype), _operation(variant, 8, 2, dtype)
    op(qkv, **arguments)
    full_checks = []
    for name in ("_check_full", "_check_scale"):
        original = getattr(op, name)

        def counted(*args, _original=original):
            full_checks.append(1)
            return _original(*args)

        monkeypatch.setattr(op, name, counted)
    scale_roles = ("k_scale", "v_scale")
    for source, tensor in arguments.items():
        for role in arguments:
            if role == source:
                continue
            offered = {**arguments, role: tensor}
            tensors = {key: value for key, value in offered.items() if key not in scale_roles}
            expected = None
            try:
                fresh._check_full(qkv, *tensors.values())
                fresh._scales(offered.get("k_scale"), offered.get("v_scale"))
            except (TypeError, ValueError) as error:
                expected = error
            before = len(full_checks)
            if expected is None:
                op(qkv, **offered)
            else:
                with pytest.raises(type(expected)) as raised:
                    op(qkv, **offered)
                assert str(raised.value) == str(expected), (source, role)
            assert len(full_checks) > before, (source, role)
            op(qkv, **arguments)  # rebind the original roles


def _axes(variant, layout):
    """Per-frequency position axes: interleaved (Qwen3.5 M-RoPE) or chunked."""
    frequencies = variant[1] // 2
    if layout == "interleaved":
        return tuple(f % 3 for f in range(frequencies))
    return tuple(min(3 * f // frequencies, 2) for f in range(frequencies))


@pytest.mark.parametrize("layout", ["interleaved", "chunked"])
@pytest.mark.parametrize("provider", ["kernel", "compose"])
@pytest.mark.parametrize("tokens", [1, 3, 64, 300])
def test_multi_axis_positions_match_the_gathered_table(variant, layout, provider, tokens):
    """[3, T] positions equal the 1-D call on the table gathered per token (each column at its
    frequency's axis row), bitwise: both read the same FP32 table entries."""
    q_heads, kv_heads = GEOMETRIES[variant][0]
    head_dim, rotary_dim, zero_centered = variant
    axes = _axes(variant, layout)
    op = Prologue(
        num_q_heads=q_heads, num_kv_heads=kv_heads, eps=EPS, device=_device(),
        head_dim=head_dim, rotary_dim=rotary_dim, zero_centered=zero_centered,
        position_axes=axes,
    )  # fmt: skip
    plain = _operation(variant, q_heads, kv_heads)
    if provider == "compose":
        op._kernel = op._axis_kernel = plain._kernel = None
    data, table = _inputs(variant, tokens, q_heads, kv_heads, seed=5), _table(variant)
    generator = torch.Generator(device=_device()).manual_seed(6)
    positions = torch.randint(8192, (3, tokens), generator=generator, device=_device())
    columns = torch.tensor(axes * 2, device=_device())
    rows = table[positions[columns].T, torch.arange(rotary_dim, device=_device())]
    expected = _call(
        variant, plain, dict(data, positions=torch.arange(tokens, device=_device())), rows,
        q_heads, kv_heads,
    )  # fmt: skip
    got = _call(variant, op, dict(data, positions=positions), table, q_heads, kv_heads)
    for a, b in zip(got, expected, strict=True):
        assert torch.equal(a, b)
    # [T] positions on a multi-axis op run the 1-D kernel: the same bits as a plain op.
    one = positions[0].contiguous()
    got = _call(variant, op, dict(data, positions=one), table, q_heads, kv_heads)
    expected = _call(variant, plain, dict(data, positions=one), table, q_heads, kv_heads)
    for a, b in zip(got, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("tokens", [1, 4])
def test_multi_axis_graph_replay_reads_new_positions(variant, tokens):
    q_heads, kv_heads = GEOMETRIES[variant][0]
    head_dim, rotary_dim, zero_centered = variant
    op = Prologue(
        num_q_heads=q_heads, num_kv_heads=kv_heads, eps=EPS, device=_device(),
        head_dim=head_dim, rotary_dim=rotary_dim, zero_centered=zero_centered,
        position_axes=_axes(variant, "interleaved"),
    )  # fmt: skip
    op.warm_up(tokens)
    table = _table(variant)
    buffers = _inputs(variant, tokens, q_heads, kv_heads, seed=1)
    buffers["positions"] = torch.zeros(3, tokens, dtype=torch.int64, device=_device())
    qkv = buffers["rows"][:, : (q_heads + 2 * kv_heads) * head_dim]
    arguments = (qkv, buffers["q_weight"], buffers["k_weight"], table, buffers["positions"])
    arguments += (buffers["k_cache"], buffers["v_cache"], buffers["indices"])
    op(*arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q = op(*arguments)
    for seed in (2, 3):
        fresh = _inputs(variant, tokens, q_heads, kv_heads, seed=seed)
        fresh["positions"] = torch.randint(8192, (3, tokens), device=_device())
        for name, tensor in fresh.items():
            buffers[name].copy_(tensor)
        graph.replay()
        eager = _call(variant, op, fresh, table, q_heads, kv_heads)
        assert torch.equal(q, eager[0])
        assert torch.equal(buffers["k_cache"], eager[1])
