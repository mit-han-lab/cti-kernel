# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-FP8 gated-MLP input on the CPU: lazy imports, the frozen-weight rule, checks that run
before any launch, the FlashInfer FC1 provider's operand reordering and errors, and dispatch."""

from __future__ import annotations

import importlib.metadata
import math
import sys
from types import SimpleNamespace

import pytest
import torch

from ctikernels.linear import _block_fp8_gated_dispatch as dispatch
from ctikernels.linear import block_fp8_gated
from ctikernels.linear.flashinfer import _block_fp8_gated as _block_fp8_gated_flashinfer

from ..support.child import run_child

E4M3 = torch.float8_e4m3fn


def _device_properties(capability: tuple[int, int], sms: int) -> object:
    """Fake device properties, complete for torch's lazy CUDA init when a real GPU is visible."""
    major, minor = capability
    name = f"sm_{major}{minor}"
    return SimpleNamespace(major=major, minor=minor, multi_processor_count=sms, name=name)


def _weight(n: int = 512, k: int = 384):
    return torch.zeros(n, k, dtype=E4M3), torch.ones(-(-n // 128), -(-k // 128))


def test_importing_the_operation_loads_no_provider() -> None:
    code = (
        "import sys, ctikernels.linear.block_fp8_gated; "
        "print(sorted(m for m in ('triton', 'flashinfer', 'cutlass') if m in sys.modules))"
    )
    loaded = run_child([sys.executable, "-c", code], timeout=600, check=True).stdout
    assert loaded.strip() == "[]"


@pytest.mark.parametrize("frozen", ["weight", "weight_scale_inv"])
def test_a_weight_requiring_grad_raises_before_any_other_check(frozen: str) -> None:
    weight, scale = _weight()
    tensors = {"weight": weight.float().requires_grad_(), "weight_scale_inv": scale}
    if frozen == "weight_scale_inv":
        tensors = {"weight": weight, "weight_scale_inv": scale.requires_grad_()}
    with pytest.raises(ValueError, match=f"{frozen} is frozen"):
        block_fp8_gated.BlockFP8GateUpSiLU(tensors["weight"], tensors["weight_scale_inv"])


def test_cpu_tensors_are_rejected_without_launching() -> None:
    with pytest.raises(ValueError, match="one CUDA device"):
        block_fp8_gated.BlockFP8GateUpSiLU(*_weight())
    with pytest.raises(ValueError, match="CUDA device"):
        block_fp8_gated.silu_and_mul_quantize_fp8(torch.zeros(2, 256, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("gate_up", "error", "match"),
    [
        (torch.zeros(2, 256), TypeError, "BF16 or FP16"),
        ("not a tensor", TypeError, "must be a tensor"),
    ],
)
def test_activation_input_is_checked_before_its_device(gate_up, error, match) -> None:
    with pytest.raises(error, match=match):
        block_fp8_gated.silu_and_mul_quantize_fp8(gate_up)


def test_flashinfer_fc1_reorders_operands_up_first(monkeypatch) -> None:
    monkeypatch.setattr(_block_fp8_gated_flashinfer, "_flashinfer_fc1", lambda *args: object())
    i, k = 256, 384
    weight = torch.arange(2 * i, dtype=torch.float32)[:, None].expand(2 * i, k).to(E4M3)
    scale = torch.arange(4 * 3, dtype=torch.float32).view(4, 3).to(torch.bfloat16)
    provider = _block_fp8_gated_flashinfer.FlashInferFc1ActQ1(weight, scale, capability=(12, 0))
    assert provider._weight.shape == (1, 2 * i, k) and provider._weight.is_contiguous()
    assert torch.equal(provider._weight[0, :i].float(), weight[i:].float())
    assert torch.equal(provider._weight[0, i:].float(), weight[:i].float())
    assert provider._scale.dtype == torch.float32 and provider._scale.shape == (1, 3, 4)
    assert torch.equal(provider._scale[0], torch.cat([scale[2:], scale[:2]]).float().t())
    assert [_block_fp8_gated_flashinfer.padded_rows(m) for m in (1, 4, 5, 8)] == [4, 4, 8, 8]


def test_flashinfer_fc1_reports_a_missing_or_unqualified_install(monkeypatch) -> None:
    fc1 = _block_fp8_gated_flashinfer._flashinfer_fc1
    with pytest.raises(RuntimeError, match="compute capability 12.x"):
        fc1("op", (10, 0), 512, 2048)
    with pytest.raises(ValueError, match="multiples of 128"):
        fc1("op", (12, 0), 200, 2048)

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="flashinfer-python 0.7.1"):
        fc1("op", (12, 0), 512, 2048)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.6.11.post3")
    module = type(sys)("flashinfer.fused_moe.cute_dsl.blackwell_sm12x")
    module.cute_dsl_sm12x_fc1_act_q1_fp8 = object()
    for name in ("flashinfer", "flashinfer.fused_moe", "flashinfer.fused_moe.cute_dsl"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="found 0.6.11.post3"):
        fc1("op", (12, 0), 512, 2048)


def test_dispatch_table_runs_are_well_formed() -> None:
    for shapes in dispatch.TABLE.values():
        for (i, k), runs in shapes.items():
            assert i % 128 == 0 and k % 128 == 0
            lasts = [last for last, _ in runs]
            assert lasts == sorted(set(lasts))
            assert all(name in (dispatch.CUTE, dispatch.FLASHINFER) for _, name in runs)


def test_dispatch_picks_the_run_holding_the_batch() -> None:
    runs = ((4, dispatch.CUTE), (64, dispatch.FLASHINFER), (512, dispatch.CUTE))
    assert dispatch.provider(runs, 1) == dispatch.CUTE
    assert dispatch.provider(runs, 5) == dispatch.FLASHINFER
    assert dispatch.provider(runs, 4096) == dispatch.CUTE
    assert dispatch.buckets((9, 0), 132, 512, 2048) is None
    assert block_fp8_gated.PROVIDERS == (
        dispatch.CUTE,
        dispatch.FLASHINFER,
        dispatch.FUSED,
    )


@pytest.mark.parametrize("sms", [170, 188])
@pytest.mark.parametrize("available", [False, True])
def test_default_dispatch_uses_qualified_flashinfer_in_the_measured_winning_cell(
    monkeypatch, sms, available
) -> None:
    # Exercise real construction/availability/fallback without allocating the 27B weight
    # or querying CUDA. Only the device checks, underlying GEMM and FC1 import are replaced.
    weight = torch.empty(34816, 5120, device="meta", dtype=E4M3)
    scale = torch.empty(272, 40, device="meta")
    monkeypatch.setattr(block_fp8_gated, "_check_weight", lambda *a, **kw: (weight, scale))
    monkeypatch.setattr(block_fp8_gated, "available", lambda capability: True)
    monkeypatch.setattr(block_fp8_gated, "_cuda_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: _device_properties((12, 0), sms)
    )
    gemm = SimpleNamespace(provider_for=lambda _: "triton-fp8")
    monkeypatch.setattr(block_fp8_gated, "BlockFP8Linear", lambda *args, **kwargs: gemm)
    probes = []

    def fc1(*args):
        probes.append(args)
        if not available:
            raise RuntimeError("no qualified FlashInfer FC1")
        return object()

    monkeypatch.setattr(_block_fp8_gated_flashinfer, "_flashinfer_fc1", fc1)
    op = block_fp8_gated.BlockFP8GateUpSiLU(weight, scale)
    assert len(probes) == 1
    assert op.provider == dispatch.FUSED
    assert op.provider_for(64) == (
        dispatch.FLASHINFER if available else "triton-fp8+cute-act-quant"
    )
    assert op.gate_up is gemm
    for m in (0, 1, 4, 32, 63, 65, 128, 4096, 8192):
        assert op.provider_for(m) == (dispatch.FUSED if m == 1 else "triton-fp8+cute-act-quant")
    explicit = block_fp8_gated.BlockFP8GateUpSiLU(weight, scale, provider=dispatch.CUTE)
    assert explicit.provider_for(64) == "triton-fp8+cute-act-quant"
    assert len(probes) == 1  # explicit local activation never probes/imports FC1


def test_dispatch_does_not_extrapolate_a_shape_or_a_subthreshold_win() -> None:
    assert dispatch.buckets((12, 1), 48, 17408, 5120) is None  # GB10: only 2.4%
    assert dispatch.buckets((12, 0), 170, 8704, 5120) is None
    assert dispatch.buckets((12, 0), 188, 512, 2048) is None
    assert dispatch.buckets((12, 0), 169, 17408, 5120) is None


@pytest.mark.parametrize("layout", ["k", "mn"])
@pytest.mark.parametrize("m,i", [(0, 200), (1, 512), (3, 2048), (5, 4352)])
def test_output_scale_shape_and_padding(layout, m, i):
    q, s = block_fp8_gated.act_quant.allocate_outputs(m, i, "cpu", layout)
    assert q.shape == (m, i) and q.dtype == E4M3
    assert s.shape == (m, -(-i // 128)) and s.dtype == torch.float32
    if layout == "mn":
        assert s.stride() == (1, max(1, -(-m // 4) * 4))
    else:
        assert s.is_contiguous()


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan"), "10"])
def test_bad_clamp_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        block_fp8_gated.act_quant.check_options(limit, False, "k")


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("limit", [None, 10.0])
def test_reference_zero_and_clamp(ue8m0, limit):
    from ._block_fp8_gated_reference import activation, quantize, silu_quantize

    x = torch.zeros(3, 1024, dtype=torch.bfloat16)
    q, s = silu_quantize(x, swiglu_limit=limit, scale_ue8m0=ue8m0)
    assert torch.count_nonzero(q.float()) == 0
    expected = 2.0**-42 if ue8m0 else torch.tensor(1e-10) * (1.0 / 448)
    assert torch.all(s == expected)
    x[:, :512] = 20
    x[:, 512:] = -20
    if limit:
        assert torch.all(activation(x, limit) == -100)
    assert torch.equal(
        quantize(activation(x, limit), ue8m0)[0].view(torch.uint8),
        silu_quantize(x, swiglu_limit=limit, scale_ue8m0=ue8m0)[0].view(torch.uint8),
    )


def test_ue8m0_rounds_up_at_power_of_two_boundaries():
    from ._block_fp8_gated_reference import quantize

    values = torch.tensor([0, 448, 449, 896, 897], dtype=torch.float32)[:, None].expand(5, 128)
    q, scale = quantize(values, True)
    assert torch.equal(scale[:, 0], torch.tensor([2.0**-42, 1, 2, 2, 4]))
    assert torch.equal(q[:, 0].float(), torch.tensor([0, 448, 224, 448, 224]))


def test_clamp_fp16_operands_round_to_bf16_before_silu():
    from ._block_fp8_gated_reference import activation

    x = torch.tensor([[1.003, -11, 1.003, 11]], dtype=torch.float16)
    g, u = x.bfloat16().float().chunk(2, dim=1)
    expected = (g.clamp(max=10) / (1 + torch.exp(-g.clamp(max=10))) * u.clamp(-10, 10)).half()
    assert torch.equal(activation(x, 10), expected)


def test_dispatch_names_any_package_gemm_and_preserves_full_fusion():
    for gemm in ("cute-fp8", "triton-fp8", "flashinfer-cutlass", "cuda-sm12x"):
        runs = ((63, dispatch.CUTE), (64, dispatch.FLASHINFER), (65, dispatch.CUTE))
        assert dispatch.resolved_provider(runs, 63, gemm) == gemm + "+cute-act-quant"
        assert dispatch.resolved_provider(runs, 64, gemm) == dispatch.FLASHINFER


def test_dsv4_disables_full_fusion_and_passes_activation_policy(monkeypatch):
    weight, scale = _weight()
    monkeypatch.setattr(block_fp8_gated, "_check_weight", lambda *a, **kw: (weight, scale))
    monkeypatch.setattr(block_fp8_gated, "_cuda_capability", lambda _: (12, 0))
    monkeypatch.setattr(block_fp8_gated, "available", lambda _: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties((12, 0), 170)
    )
    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((64, dispatch.FLASHINFER),))
    calls = []
    gemm = SimpleNamespace(provider_for=lambda _: "cute-fp8")

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return gemm

    monkeypatch.setattr(block_fp8_gated, "BlockFP8Linear", factory)

    def no_fusion(*args):
        pytest.fail("DSV4 must not probe FlashInfer FC1")

    monkeypatch.setattr(block_fp8_gated.BlockFP8GateUpSiLU, "_flashinfer_provider", no_fusion)
    op = block_fp8_gated.BlockFP8GateUpSiLU(weight, scale, swiglu_limit=10, scale_ue8m0=True)
    assert op.provider == dispatch.FUSED
    assert op.provider_for(64) == "cute-fp8+cute-act-quant"
    assert calls == [{"scale_ue8m0": True}]
    with pytest.raises(ValueError, match="DSV4"):
        block_fp8_gated.BlockFP8GateUpSiLU(
            weight, scale, provider=dispatch.FLASHINFER, swiglu_limit=10
        )
    # The source scale must still be checked on every entry, before any launch.
    scale.requires_grad_()
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op(None)
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op.quantized(None, None)


def test_benchmark_imports_on_cpu():
    from benchmark.kernel_tuning import block_fp8_gated as benchmark

    assert "cute-act-quant" in benchmark.PATHS
    assert "cute-fp8-fused" in benchmark.PATHS
    assert "triton-act-quant" not in benchmark.PATHS


def test_only_an_available_measured_cute_gemm_overrides_full_fusion():
    runs = ((64, dispatch.FLASHINFER),)
    measured = ((64, "cute-fp8"),)
    assert dispatch.resolved_provider(runs, 64, "cute-fp8", measured) == "cute-fp8+cute-act-quant"
    assert dispatch.resolved_provider(runs, 64, "triton-fp8", measured) == dispatch.FLASHINFER
    assert dispatch.resolved_provider(runs, 64, "cute-fp8") == dispatch.FLASHINFER


def test_cute_activation_capture_readiness_covers_options_and_layout(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.linear.cute import _block_fp8_act_quant as quant
    from ctikernels.linear.cute import _block_fp8_provider as provider

    state = SimpleNamespace(capturing=True, compiles=[], launches=[])
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(provider, "available", lambda _: True)
    monkeypatch.setattr(quant, "_compiled", {})
    monkeypatch.setattr(quant, "_ready", set())

    def compile_mock(*args, **kwargs):
        state.compiles.append((args, kwargs))
        return lambda *args: state.launches.append(args)

    monkeypatch.setattr(quant, "_compile", compile_mock)
    x = torch.empty(3, 1024, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="warmup"):
        quant.silu_quantize(x)
    assert not state.compiles and not state.launches
    state.capturing = False
    quant.silu_quantize(x)
    state.capturing = True
    quant.silu_quantize(x[:1])  # M is symbolic; same signature
    assert len(state.compiles) == 1 and len(state.launches) == 2
    for options in ({"swiglu_limit": 10}, {"scale_ue8m0": True}, {"scale_layout": "mn"}):
        with pytest.raises(RuntimeError, match="warmup"):
            quant.silu_quantize(x, **options)
    state.capturing = False
    quant.silu_quantize(x, swiglu_limit=10, scale_ue8m0=True, scale_layout="mn")
    args, options = state.compiles[-1]
    assert args[:3] == (512, torch.bfloat16, True)
    assert options == {"gated": True, "swiglu_limit": 10, "scale_ue8m0": True, "scale_layout": "mn"}


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("shape", [(256, 2048), (512, 2048), (2048, 4096), (17408, 5120)])
def test_fused_dispatch_and_explicit_override(monkeypatch, capability, shape):
    i, k = shape
    weight = torch.empty(2 * i, k, device="meta", dtype=E4M3)
    scale = torch.empty(2 * i // 128, k // 128, device="meta")
    monkeypatch.setattr(block_fp8_gated, "_check_weight", lambda *a, **kw: (weight, scale))
    monkeypatch.setattr(block_fp8_gated, "available", lambda _: True)
    monkeypatch.setattr(block_fp8_gated, "_cuda_capability", lambda _: capability)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties(capability, 1)
    )
    gemm = SimpleNamespace(provider_for=lambda _: "cute-fp8")
    monkeypatch.setattr(block_fp8_gated, "BlockFP8Linear", lambda *a, **kw: gemm)
    for selector, ceiling in ((None, 1), (dispatch.FUSED, 16), (dispatch.CUTE, 0)):
        op = block_fp8_gated.BlockFP8GateUpSiLU(weight, scale, provider=selector)
        for m in range(18):
            assert op.provider_for(m) == (
                dispatch.FUSED if 1 <= m <= ceiling else "cute-fp8+cute-act-quant"
            )
        assert op._fused is None  # no allocation/DSL import until eager warmup
    assert not dispatch.fused_supported((9, 0), i, k)
    assert not dispatch.fused_supported(capability, i + 1, k)
    assert not dispatch.fused_supported(capability, i, k + 1)


def test_fused_config_and_workspace_sizing():
    from ctikernels.linear.cute._block_fp8_fused_provider import (
        FusedConfig,
        config_for,
        m_bucket,
        workspace_shapes,
    )

    for capability in ((12, 0), (12, 1)):
        assert config_for(capability, 1024) == FusedConfig(4, 4)
        assert config_for(capability, 1152) == FusedConfig(8, 8)
        for m in range(2, 17):
            assert config_for(capability, 1024, m) == FusedConfig(engine="mma", warps_n=1)
            assert config_for(capability, 1152, m) == FusedConfig(engine="mma", warps_n=2)
    assert [m_bucket(m) for m in range(1, 17)] == [1, 2, 4, 4, 8, 8, 8, 8] + [16] * 8
    for bad in (0, -1, 17, 1.5, True):
        with pytest.raises(ValueError, match="M must"):
            m_bucket(bad)
    for r, w in ((1, 4), (32, 4), (4, 2), (4, 16), (4.0, 4), (4, True)):
        with pytest.raises(ValueError):
            FusedConfig(r, w)
    for bad in (
        {"engine": "auto"},
        {"engine": None},
        {"warps_n": 0},
        {"warps_n": 3},
        {"warps_n": True},
        {"warps_n": 2.0},
        {"warps_k": 0},
        {"warps_k": 3},
        {"warps_k": True},
        {"warps_k": 4.0},
        {"warps_n": 4, "warps_k": 8},
        {"cta_k_split": 0},
        {"cta_k_split": 2},
        {"cta_k_split": True},
    ):
        with pytest.raises(ValueError):
            FusedConfig(**bad)
    for wn in (1, 2, 4):
        for wk in (1, 2, 4, 8):
            if wn * wk <= 16:
                for m in (1, 2, 4, 8, 16):
                    config = FusedConfig(engine="mma", warps_n=wn, warps_k=wk)
                    tokens = max(8, m)
                    assert config.shared_bytes(m) == (
                        tokens * wk * (128 + 4) + 32 * 4 * (tokens // 8) * wn * wk * 4 + 4
                    )
                    assert config.shared_bytes(m) < 48 * 1024
    for bad in (0, 127, 129):
        with pytest.raises(ValueError, match="multiple of 128"):
            workspace_shapes(bad)
    for i in (256, 512, 2048, 17408):
        assert workspace_shapes(i) == ((16, i), (i // 128,))
        # BF16 hbuf plus one int32 counter per 128-column group.
        h, counters = workspace_shapes(i)
        assert math.prod(h) * 2 + math.prod(counters) * 4 == 32 * i + i // 32
    for r in (2, 4, 8, 16):
        for w in (4, 8):
            assert FusedConfig(r, w).shared_bytes(16) <= 65540


def test_fused_readiness_and_instance_workspace(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.linear.cute import _block_fp8_fused_provider as fused
    from ctikernels.linear.cute import _block_fp8_provider as gemm

    state = SimpleNamespace(capturing=False, compiles=[], launches=[])
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(gemm, "available", lambda _: True)
    monkeypatch.setattr(fused, "_compiled", {})
    monkeypatch.setattr(fused, "_ready", set())

    def compile_mock(self, *args):
        state.compiles.append(args)
        return lambda *args: state.launches.append(args)

    monkeypatch.setattr(fused.FusedFC1, "_compile", compile_mock)
    weight, scale = _weight(512, 2048)
    scale = scale.bfloat16()  # exercise a copied source scale's frozen check
    op = fused.FusedFC1(weight, scale, capability=(12, 0))
    x = torch.empty(4, 2048, dtype=torch.bfloat16)
    op.run(x)
    assert state.compiles[0][-1].engine == "mma"
    hbuf, counters = op._scratch
    assert hbuf.shape == (16, 256) and hbuf.dtype == torch.bfloat16
    assert counters.shape == (2,) and counters.dtype == torch.int32
    assert torch.count_nonzero(counters) == 0
    state.capturing = True
    op.run(x[:3])  # same compiled M bucket
    op.run(torch.empty(3, 2049, dtype=torch.bfloat16)[:, :2048])  # symbolic stride
    assert len(state.compiles) == 1 and len(state.launches) == 3
    assert op._scratch[0] is hbuf and op._scratch[1] is counters
    for value, debug in ((x[:2], False), (x.half(), False), (x, True)):
        with pytest.raises(RuntimeError, match="warmup"):
            op.run(value, debug=debug)
    state.capturing = False
    other = fused.FusedFC1(weight, scale, capability=(12, 0))
    state.capturing = True
    # A globally warmed kernel must not hide cold per-instance scratch.
    with pytest.raises(RuntimeError, match="workspace"):
        other.run(x)
    state.capturing = False
    other.run(x)
    assert len(state.compiles) == 1
    assert other._scratch[0] is not hbuf
    op.run(x[:1])
    assert state.compiles[-1][-1].engine == "fma"
    assert op._scratch[0] is hbuf and op._scratch[1] is counters
    # A per-callsite override keeps MMA even at M=1 and has its own cache entry.
    forced = fused.FusedFC1(
        weight, scale, capability=(12, 0), config=fused.FusedConfig(engine="mma")
    )
    forced.run(x[:1])
    assert len(state.compiles) == 3 and state.compiles[-1][-1].engine == "mma"
    state.capturing = True
    for rows in (1, 3, 4):
        op.run(x[:rows])
    assert len(state.compiles) == 3
    state.capturing = False
    # Even a source that was converted/copied is checked before allocation/launch.
    scale.requires_grad_()
    before = len(state.launches)
    for instance in (op, other):
        with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
            instance.run(x)
    assert len(state.launches) == before


def test_fused_engine_table_can_select_inside_an_m_bucket(monkeypatch):
    from ctikernels.linear.cute import _block_fp8_fused_provider as fused

    fma, mma = fused.FusedConfig(), fused.FusedConfig(engine="mma", warps_n=2)
    monkeypatch.setitem(fused.CONFIGS, (12, 0), ((3, None, fma), (16, None, mma)))
    op = object.__new__(fused.FusedFC1)
    op.config, op.capability, op.i = None, (12, 0), 2048
    op._configs = {}
    op.sms = 170
    assert fused.m_bucket(3) == fused.m_bucket(4)
    assert op.config_for(3) == fma and op.config_for(4) == mma
    monkeypatch.setitem(fused.CONFIGS, (12, 0, 170), ((4, None, mma),))
    assert op.config_for(1) == mma  # device-specific override
    assert op.config_for(8) == mma  # capability fallback
    op.sms = 188
    assert op.config_for(1) == fma  # other sm120 device keeps its own policy
    op.config = mma
    assert op.config_for(1) == mma
    for bad in (0, 17, True):
        with pytest.raises(ValueError, match="M must"):
            op.config_for(bad)


@pytest.mark.parametrize("m", [1, 2, 7, 8, 9, 16])
def test_mma_k_permutation_and_lane_fragments_preserve_dot_products(m):
    """Reconstruct hardware A/B tiles from lane registers, then scatter C.

    Reference: public PTX ISA mma.m16n8k32 E4M3 A/B/C fragment formulas.
    This is CPU-only; FP64 here makes the small FP8 dot products exact.
    """
    generator = torch.Generator().manual_seed(71 + m)
    weight = (torch.randn(16, 128, generator=generator) * 3).to(E4M3).double()
    x = (torch.randn(m, 128, generator=generator) * 3).to(E4M3).double()
    padded = torch.zeros(16, 128, dtype=torch.float64)
    padded[:m] = x
    shared = torch.empty(16, 32, 4, dtype=torch.float64)
    for row in range(16):
        shared[row, torch.arange(32) ^ (row % 8)] = padded[row].reshape(32, 4)
    logical = torch.arange(128)
    # L=32*s+16*h+4*t+b -> P=32*t+8*s+4*h+b.
    physical = 32 * ((logical % 16) // 4) + 8 * (logical // 32)
    physical += 4 * ((logical % 32) // 16) + logical % 4
    assert torch.equal(physical.sort().values, logical)
    reconstructed = torch.zeros(16, 16, dtype=torch.float64)
    for tile in range((m + 7) // 8):
        c = torch.zeros(16, 8, dtype=torch.float64)
        for s in range(4):
            a = torch.empty(16, 32, dtype=torch.float64)
            b = torch.empty(32, 8, dtype=torch.float64)
            a_owners = torch.zeros_like(a, dtype=torch.int32)
            b_owners = torch.zeros_like(b, dtype=torch.int32)
            for lane in range(32):
                g, t = divmod(lane, 4)
                # The two 128-bit loads per row provide eight packed words.
                words = torch.stack(
                    (weight[g, t * 32 : t * 32 + 32], weight[g + 8, t * 32 : t * 32 + 32])
                )
                words = words.reshape(2, 8, 4)
                a_regs = torch.stack(
                    (words[0, 2 * s], words[1, 2 * s], words[0, 2 * s + 1], words[1, 2 * s + 1])
                )
                b_regs = shared[tile * 8 + g, [(8 * t + 2 * s) ^ g, (8 * t + 2 * s + 1) ^ g]]
                # Decode the PTX fragment layouts independently of P(L).
                for reg in range(4):
                    row = g + 8 * (reg % 2)
                    k0 = t * 4 + 16 * (reg // 2)
                    a[row, k0 : k0 + 4] = a_regs[reg]
                    a_owners[row, k0 : k0 + 4] += 1
                for reg in range(2):
                    k0 = t * 4 + 16 * reg
                    b[k0 : k0 + 4, g] = b_regs[reg]
                    b_owners[k0 : k0 + 4, g] += 1
            assert torch.all(a_owners == 1) and torch.all(b_owners == 1)
            for half in range(2):
                banks = [((8 * (lane % 4) + 2 * s + half) ^ (lane // 4)) % 32 for lane in range(32)]
                assert len(set(banks)) == 32
            assert torch.equal(a, weight[:, physical[s * 32 : (s + 1) * 32]])
            assert torch.equal(
                b, padded[tile * 8 : tile * 8 + 8, physical[s * 32 : (s + 1) * 32]].T
            )
            c += a @ b
        # Each thread owns gate/up for both tokens: no epilogue shuffle needed.
        owners = torch.zeros(16, 8, dtype=torch.int32)
        for lane in range(32):
            g, t = divmod(lane, 4)
            fragment = (c[g, 2 * t], c[g, 2 * t + 1], c[g + 8, 2 * t], c[g + 8, 2 * t + 1])
            for index, value in enumerate(fragment):
                row, token = g + 8 * (index // 2), 2 * t + index % 2
                reconstructed[row, tile * 8 + token] = value
                owners[row, token] += 1
        assert torch.all(owners == 1)
    assert torch.equal(reconstructed[:, :m], weight @ x.T)


@pytest.fixture
def small_m_op(monkeypatch):
    """Construct the real dispatcher using meta weights and CPU-only provider doubles."""
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(block_fp8_gated, "_check_weight", lambda _, w, s, **kw: (w, s))
    monkeypatch.setattr(block_fp8_gated, "available", lambda _: True)
    monkeypatch.setattr(
        block_fp8_gated,
        "BlockFP8Linear",
        lambda *a, **kw: SimpleNamespace(
            provider_for=lambda _: "triton-fp8",
            _input_layout_key=lambda m, layout, scale=None, **kw: (layout, scale),
        ),
    )
    monkeypatch.setattr(
        block_fp8_gated.BlockFP8GateUpSiLU, "_flashinfer_provider", lambda *a: object()
    )

    def make(hardware, shape=(512, 2048), **kwargs):
        major, minor, sms = hardware
        monkeypatch.setattr(block_fp8_gated, "_cuda_capability", lambda _: (major, minor))
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda _: _device_properties((major, minor), sms)
        )
        i, h = shape
        w = torch.empty(2 * i, h, device="meta", dtype=E4M3)
        s = torch.empty(-(-2 * i // 128), -(-h // 128), device="meta")
        return block_fp8_gated.BlockFP8GateUpSiLU(w, s, **kwargs)

    return make


def test_small_m_placeholder_table():
    shapes = ((512, 2048), (256, 2048), (2048, 4096), (17408, 5120), (8704, 5120))
    assert set(dispatch.SMALL_M) == {
        (device, i, h) for device in dispatch.DEVICES.values() for i, h in shapes
    }
    for runs in dispatch.SMALL_M.values():
        assert dispatch.parse_runs(runs)


@pytest.mark.parametrize("spec", ["fma:r=8,w=8", "mma:wn=1,wk=4", "mma:wn=4,wk=4"])
def test_small_m_config_roundtrip(spec):
    assert dispatch.config_spec(dispatch.parse_config(spec)) == spec


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "fused",
        "other:r=4,w=4",
        "fma:r=4",
        "fma:r=4,w=4,w=8",
        "fma:r=3,w=4",
        "fma:r=4,w=2",
        "mma:wn=4,wk=8",
        "mma:wn=2,w=4",
        "fma:r=4.0,w=4",
        3,
    ],
)
def test_small_m_bad_specs(spec):
    with pytest.raises(ValueError):
        dispatch.parse_config(spec)


@pytest.mark.parametrize(
    "runs",
    [
        (),
        ((0, dispatch.CUTE, None),),
        ((True, dispatch.CUTE, None),),
        ((1, dispatch.CUTE, None), (1, dispatch.FUSED, None)),
        ((4, dispatch.CUTE, None), (2, dispatch.FUSED, None)),
        ((1, "unknown", None),),
        ((1, dispatch.CUTE, "fma:r=4,w=4"),),
        ((17, dispatch.FUSED, None),),
        ((1, dispatch.FUSED, "mma:wn=3,wk=4"),),
    ],
)
def test_small_m_bad_runs(runs):
    with pytest.raises(ValueError):
        dispatch.parse_runs(runs)


@pytest.mark.parametrize("hardware", dispatch.DEVICES)
@pytest.mark.parametrize("shape", [(512, 2048), (2048, 4096), (1536, 2048)])
def test_small_m_resolution_for_each_device(small_m_op, hardware, shape):
    op = small_m_op(hardware, shape)
    runs = dispatch.SMALL_M.get((dispatch.DEVICES[hardware], *shape))
    if runs is None:
        # Unlisted shapes: fused capability default at M=1 only.
        assert op.provider == dispatch.FUSED
        assert op.resolve(1) == (dispatch.FUSED, "fma:r=8,w=8")
        last = 1
    else:
        assert op.provider == runs[0][1]
        previous = 0
        for ceiling, provider, spec in runs:
            for m in range(previous + 1, ceiling + 1):
                resolved = op.resolve(m)
                assert resolved[0] == provider
                if spec is not None:
                    assert resolved[1] == spec
            previous = ceiling
        last = runs[-1][0]
    for m in (0, last + 1, 16, 17, 64, 4096):
        if m == 0 or m > last:
            assert op.resolve(m) == (dispatch.CUTE, None)


@pytest.mark.parametrize("hardware", dispatch.DEVICES)
def test_small_m_mixed_configs_and_overrides(monkeypatch, small_m_op, hardware):
    key = (dispatch.DEVICES[hardware], 512, 2048)
    monkeypatch.setitem(
        dispatch.SMALL_M,
        key,
        (
            (2, dispatch.FUSED, "fma:r=8,w=8"),
            (4, dispatch.FUSED, "mma:wn=1,wk=4"),
            (8, dispatch.CUTE, None),
            (16, dispatch.FUSED, None),
        ),
    )
    op = small_m_op(hardware)
    pinned = block_fp8_gated.FusedConfig(engine="mma", warps_n=2, warps_k=4)
    override = small_m_op(hardware, fused_config=pinned)
    explicit = small_m_op(hardware, provider=dispatch.FUSED, fused_config=pinned)
    local = small_m_op(hardware, provider=dispatch.CUTE, fused_config=pinned)
    flash = small_m_op(hardware, provider=dispatch.FLASHINFER, fused_config=pinned)
    # No parsing or default-config lookup is allowed after construction.
    monkeypatch.setattr(dispatch, "parse_config", lambda _: pytest.fail("parsed on forward"))
    monkeypatch.setattr(block_fp8_gated, "config_for", lambda *a, **k: pytest.fail("late default"))
    for m in range(1, 18):
        if m <= 2:
            expected = (dispatch.FUSED, "fma:r=8,w=8")
        elif m <= 4 or 9 <= m <= 16:
            expected = (dispatch.FUSED, "mma:wn=1,wk=4")
        else:
            expected = (dispatch.CUTE, None)
        assert op.resolve(m) == expected
        assert override.resolve(m) == (
            (dispatch.FUSED, "mma:wn=2,wk=4") if expected[0] == dispatch.FUSED else expected
        )
        assert explicit.resolve(m) == (
            (dispatch.FUSED, "mma:wn=2,wk=4") if m <= 16 else (dispatch.CUTE, None)
        )
        assert local.resolve(m) == (dispatch.CUTE, None)
        assert flash.resolve(m) == (dispatch.FLASHINFER, None)


def test_small_m_calls_use_resolved_configs_and_quantized_bypasses_table(monkeypatch, small_m_op):
    from ctikernels.linear.cute import _block_fp8_fused_provider as fused

    hardware = (12, 0, 170)
    monkeypatch.setitem(
        dispatch.SMALL_M,
        ("rtx5090", 512, 2048),
        (
            (1, dispatch.FUSED, "fma:r=8,w=8"),
            (4, dispatch.FUSED, "mma:wn=2,wk=4"),
        ),
    )
    op = small_m_op(hardware)
    calls = []

    provider_class = fused.FusedFC1

    def factory(*a, config, configs, **kw):
        # Use the actual provider's config resolver without allocating CUDA scratch.
        instance = object.__new__(provider_class)
        instance.config, instance._configs = config, configs
        instance.run = lambda x: calls.append(dispatch.config_spec(instance.config_for(x.shape[0])))
        return instance

    monkeypatch.setattr(fused, "FusedFC1", factory)
    for m in (1, 2, 4, 1):
        op._run(torch.empty(m, 2048), None, torch.bfloat16)
        assert calls[-1] == op.resolve(m)[1]
    assert len(calls) == 4
    op._gate_up = SimpleNamespace(quantized=lambda *a, **kw: "gate_up")
    monkeypatch.setattr(op, "_unfused_provider_for", lambda _: "triton-fp8+cute-act-quant")
    monkeypatch.setattr(block_fp8_gated, "silu_and_mul_quantize_fp8", lambda x, **kw: x)
    assert op._run(torch.empty(1, 2048), object(), torch.bfloat16) == "gate_up"
    assert len(calls) == 4


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_resolve_validates_m(small_m_op, bad):
    with pytest.raises(ValueError, match="M must"):
        small_m_op((12, 1, 48)).resolve(bad)


def test_small_m_overrides_flashinfer_only_through_its_last_ceiling(monkeypatch, small_m_op):
    key = ("rtx5090", 17408, 5120)
    monkeypatch.setitem(dispatch.SMALL_M, key, ((63, dispatch.CUTE, None),))
    op = small_m_op((12, 0, 170), key[1:])
    assert op.resolve(63) == (dispatch.CUTE, None)
    assert op.resolve(64) == (dispatch.FLASHINFER, None)
    assert op.resolve(65) == (dispatch.CUTE, None)
    monkeypatch.setitem(dispatch.SMALL_M, key, ((64, dispatch.CUTE, None),))
    op = small_m_op((12, 0, 170), key[1:])
    assert op.resolve(64) == (dispatch.CUTE, None)


@pytest.mark.parametrize("flash_available,dsv4", [(True, False), (False, False), (True, True)])
def test_small_m_flashinfer_choice_respects_availability_and_options(
    monkeypatch, small_m_op, flash_available, dsv4
):
    monkeypatch.setitem(
        dispatch.SMALL_M,
        ("rtx5090", 512, 2048),
        (
            (1, dispatch.FLASHINFER, None),
            (4, dispatch.CUTE, None),
        ),
    )

    def flash(*args):
        if not flash_available:
            raise RuntimeError("unavailable")
        return object()

    monkeypatch.setattr(block_fp8_gated.BlockFP8GateUpSiLU, "_flashinfer_provider", flash)
    op = small_m_op((12, 0, 170), scale_ue8m0=dsv4)
    assert op.resolve(1) == (
        dispatch.FLASHINFER if flash_available and not dsv4 else dispatch.CUTE,
        None,
    )
    assert op.resolve(2) == (dispatch.CUTE, None)
    assert op.resolve(5) == (dispatch.CUTE, None)


def test_small_m_unfused_run_overrides_legacy_flashinfer_cell(monkeypatch, small_m_op):
    monkeypatch.setitem(dispatch.SMALL_M, ("rtx5090", 17408, 5120), ((64, dispatch.CUTE, None),))
    op = small_m_op((12, 0, 170), (17408, 5120))

    def gemm(x):
        return "gate_up"

    gemm.provider_for = lambda _: "triton-fp8"
    op._gate_up = gemm
    monkeypatch.setattr(op, "_run_flashinfer", lambda *a: pytest.fail("ignored small-M row"))
    monkeypatch.setattr(block_fp8_gated, "silu_and_mul_quantize_fp8", lambda x, **kw: x)
    assert op.resolve(64) == (dispatch.CUTE, None)
    assert op._run(torch.empty(64, 5120, device="meta"), None, torch.bfloat16) == "gate_up"


@pytest.mark.parametrize("hardware", dispatch.DEVICES)
def test_warm_up_max_m_covers_every_resolved_bucket(small_m_op, monkeypatch, hardware):
    op = small_m_op(hardware)
    op._gate_up._dispatch_key = lambda m, **kw: ("triton-fp8", None)
    calls = []
    monkeypatch.setattr(op, "warmup", lambda ms, *, dtype, **kw: calls.append((tuple(ms), dtype)))
    op.warm_up(96, dtype=torch.float16)
    ((sizes, dtype),) = calls
    assert dtype is torch.float16 and sizes[0] == 1 and sizes[-1] == 96
    assert {op.resolve(m) for m in sizes} == {op.resolve(m) for m in range(1, 97)}
    op.warm_up(batch_sizes=[3, 7])
    assert calls[-1] == ((3, 7), torch.bfloat16)
    for kwargs in ({}, {"max_m": 4, "batch_sizes": [1]}, {"max_m": 0}):
        with pytest.raises(ValueError, match="BlockFP8GateUpSiLU"):
            op.warm_up(**kwargs)


def test_warm_up_covers_fused_compile_buckets(small_m_op, monkeypatch):
    from ctikernels.linear.cute._block_fp8_fused_provider import m_bucket

    op = small_m_op((12, 0, 170), provider=dispatch.FUSED)
    op._gate_up._dispatch_key = lambda m, **kw: ("triton-fp8", None)
    calls = []
    monkeypatch.setattr(op, "warmup", lambda ms, **kw: calls.extend(ms))
    op.warm_up(16)
    assert {m_bucket(m) for m in calls} == {1, 2, 4, 8, 16}


def test_warm_up_covers_every_flashinfer_row_offset(small_m_op, monkeypatch):
    op = small_m_op((12, 0, 170), provider=dispatch.FLASHINFER)
    calls = []
    monkeypatch.setattr(op, "warmup", lambda ms, **kw: calls.extend(ms))
    op.warm_up(16)
    assert calls == list(range(1, 17))


@pytest.mark.parametrize("hardware", dispatch.DEVICES)
def test_warm_up_returns_gated_routes(small_m_op, monkeypatch, hardware):
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = []
    monkeypatch.setattr(
        block_fp8_gated.BlockFP8GateUpSiLU, "__call__", lambda self, x: calls.append(x.shape[0])
    )
    op = small_m_op(hardware)
    op._gate_up._dispatch_key = lambda m, **kw: ("triton-fp8", m % 16 == 0)
    op._gate_up.warm_up = lambda *, batch_sizes, dtype, **kw: [
        {
            "provider": "triton-fp8",
            "m": m,
            "bucket": (m, m),
            "config": None,
            "input_layout": "packed",
        }
        for m in batch_sizes
    ]
    routes = op.warm_up(96)
    assert isinstance(routes, list) and all(isinstance(r, dict) for r in routes)
    assert {r["provider"] for r in routes} == {op.provider_for(m) for m in range(1, 97)}
    assert {r["config"] for r in routes} == {op._dispatch_key(m) for m in range(1, 97)}
    assert {r["m"] for r in routes} == set(calls)
    assert op._warmed_inputs == {(r["config"], r["input_layout"]) for r in routes}


def test_warm_up_returns_flashinfer_gated_layout_routes(small_m_op, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Meta tensors have no pointer; the warm-up descriptor still records alignment.
    calls = []
    monkeypatch.setattr(
        block_fp8_gated.BlockFP8GateUpSiLU, "__call__", lambda self, x: calls.append(x.stride(0))
    )
    op = small_m_op((12, 0, 170), provider=dispatch.FLASHINFER)
    routes = op.warm_up(16, input_layouts="all")
    assert len(routes) == len(calls)
    assert {r["m"] for r in routes} == set(range(1, 17))
    assert len({r["input_layout"] for r in routes}) == 6


@pytest.fixture
def workspace_op():
    from ctikernels.linear.block_fp8 import BlockFP8Linear
    from ctikernels.linear.triton._block_fp8_provider import TritonFP8Gemm

    weight, scale = _weight(512, 2048)
    block = BlockFP8Linear.__new__(BlockFP8Linear)
    block._weight, block._scale, block._scale_source = weight, scale, scale
    block._device = weight.device
    block._capability, block._sms = (12, 0), 170
    block._scale_ue8m0 = False
    block._buckets = ((0, "triton-fp8"),)
    block._triton = TritonFP8Gemm(weight, scale, capability=(12, 0), sms=170)
    op = block_fp8_gated.BlockFP8GateUpSiLU.__new__(block_fp8_gated.BlockFP8GateUpSiLU)
    op._weight, op._scale, op._device = weight, scale, weight.device
    op._scale_layout = "mn"
    op._gate_up = block
    op._small_m = ((16, block_fp8_gated.CUTE_FP8_FUSED, None),)
    op._buckets = ((0, block_fp8_gated.CUTE_ACT_QUANT),)
    op._measured_gemm = None
    op._swiglu_limit, op._scale_ue8m0 = None, False
    return op


def test_gated_workspace_covers_dense_and_quantized_routes(workspace_op, monkeypatch):
    op, block = workspace_op, workspace_op._gate_up
    needs = [op._workspace_bytes(m) for m in range(1, 514)]
    for max_m in (1, 16, 17, 256, 320, 512, 513):
        assert op.workspace_bytes(max_m) == max(needs[:max_m])
    for m in (1, 7, 16, 17):
        layout = op._workspace_layout(m)
        assert "out_q" not in layout.slots and "out_scale" not in layout.slots
        if m <= 16:
            assert layout.slots["hidden"][1] == 2 * 16 * 256
            assert layout.slots["counters"][1] == 4 * 2
        else:
            assert layout.slots["gate_up"][1] == 2 * m * 512
            assert layout.slots["gemm"][1] == block._workspace_layout(m).nbytes
    monkeypatch.setattr(op, "_fused_provider", lambda: pytest.fail("launch before validation"))
    x = torch.zeros(7, 2048, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="workspace"):
        op(x, workspace=torch.empty(op.workspace_bytes(7) // 4))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    with pytest.raises(ValueError, match="workspace"):
        op.quantized(x.to(E4M3), torch.ones(7, 16), workspace=torch.empty(1))


@pytest.mark.parametrize("route", ["fused", "unfused", "flashinfer"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("scale_layout", ["k", "mn"])
def test_gated_workspace_outputs_survive_reuse(
    workspace_op, monkeypatch, route, quantized, scale_layout
):
    op = workspace_op
    op._scale_layout = scale_layout
    if route == "unfused":
        op._small_m = ()
    elif route == "flashinfer":
        op._small_m = ()
        op._gate_up = None
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    workspace = torch.empty(op.workspace_bytes(7) // 4)
    x = torch.zeros(7, 2048, dtype=torch.bfloat16)
    q, scale = x.to(E4M3), torch.ones(7, 16)
    calls = []

    def fill(outputs):
        workspace.fill_(len(calls) + 1)
        outputs[0].view(torch.uint8).fill_(len(calls) + 1)
        outputs[1].fill_(len(calls) + 1)
        calls.append(outputs)
        return outputs

    def fused(x, *, buffers, outputs):
        assert buffers.get("hidden").untyped_storage().data_ptr() == workspace.data_ptr()
        return fill(outputs)

    def gemm(x, scale, out, views):
        assert out.untyped_storage().data_ptr() == workspace.data_ptr()
        out.zero_()

    def silu(x, *, outputs, **kwargs):
        return fill(outputs)

    def flash(xq, xs, oq, os, *, indptr):
        for tensor in (xq, xs, indptr):
            assert tensor.untyped_storage().data_ptr() == workspace.data_ptr()
        assert os.shape == (2, 8)
        fill((oq, os))

    monkeypatch.setattr(op, "_fused_provider", lambda: SimpleNamespace(run=fused))
    monkeypatch.setattr(op, "_flashinfer", SimpleNamespace(run=flash), raising=False)
    monkeypatch.setattr(block_fp8_gated.act_quant, "silu_quantize", silu)
    monkeypatch.setattr(block_fp8_gated, "silu_and_mul_quantize_fp8", silu)
    if op._gate_up is not None:
        monkeypatch.setattr(op._gate_up, "_run_workspace", gemm)
    else:
        pytest.importorskip("triton")
        from ctikernels.linear.triton import _block_fp8 as kernel

        monkeypatch.setattr(kernel, "quantize", lambda *a, **kw: None)
        monkeypatch.setattr(block_fp8_gated, "_check_triton_capture_layout", lambda *a: None)

    def call():
        return (
            op.quantized(q, scale, workspace=workspace) if quantized else op(x, workspace=workspace)
        )

    first = call()
    saved = tuple(t.clone() for t in first)
    second = call()
    for a, b, expected in zip(first, second, saved, strict=True):
        assert torch.equal(
            a.view(torch.uint8) if a.dtype == E4M3 else a,
            expected.view(torch.uint8) if a.dtype == E4M3 else expected,
        )
        assert a.untyped_storage().data_ptr() != workspace.data_ptr()
        assert b.untyped_storage().data_ptr() != workspace.data_ptr()
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "dtype",
        "cpu",
        "device",
        "strides",
        "size",
        "alignment",
        "grad",
        "x",
        "scale",
        "weight",
        "weight_scale",
    ],
)
def test_gated_workspace_validation_before_launch(workspace_op, monkeypatch, quantized, bad):
    op = workspace_op
    workspace = torch.empty(max(op.workspace_bytes(7) // 4, op._weight.numel() // 4) + 4)
    x = torch.zeros(7, 2048, dtype=torch.bfloat16)
    q, scale = x.to(E4M3), torch.ones(7, 16)
    if bad != "cpu":
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    if bad == "dtype":
        workspace = workspace.to(torch.float16)
    elif bad == "device":
        op._device = torch.device("meta")
        x, q, scale = x.to("meta"), q.to("meta"), scale.to("meta")
    elif bad == "strides":
        workspace = workspace.expand(2, -1)
    elif bad == "size":
        workspace = workspace[:1]
    elif bad == "alignment":
        workspace = workspace[1:]
    elif bad == "grad":
        workspace.requires_grad_()
    elif bad == "x":
        x = workspace.view(torch.bfloat16)[: x.numel()].view_as(x)
        q = workspace.view(E4M3)[: q.numel()].view_as(q)
    elif bad == "scale":
        if not quantized:
            pytest.skip("dense input has no activation scale")
        scale = workspace[: scale.numel()].view_as(scale)
    elif bad == "weight":
        op._weight = workspace.view(E4M3)[: op._weight.numel()].view_as(op._weight)
    elif bad == "weight_scale":
        op._scale = workspace[: op._scale.numel()].view_as(op._scale)
    monkeypatch.setattr(op, "_fused_provider", lambda: pytest.fail("launch before validation"))
    monkeypatch.setattr(
        op._gate_up, "_run_workspace", lambda *a: pytest.fail("launch before validation")
    )
    with pytest.raises(ValueError, match="workspace"):
        op.quantized(q, scale, workspace=workspace) if quantized else op(x, workspace=workspace)


@pytest.mark.parametrize("quantized", [False, True])
def test_gated_packed_and_explicit_layout_bookkeeping(small_m_op, monkeypatch, quantized):
    from ctikernels.linear import block_fp8

    op = small_m_op((12, 0, 170), provider=dispatch.FLASHINFER)
    # Real CPU tensors allow the same metadata checks as capture without a GPU.
    op._weight = torch.zeros(1024, 2048, dtype=E4M3)
    op._device = torch.device("cpu")
    launches = []
    monkeypatch.setattr(op, "_run_flashinfer", lambda *a: launches.append(a))
    routes = op.warm_up(batch_sizes=[3], quantized=quantized)
    assert len(routes) == 1
    assert op._warmed_inputs == {(r["config"], r["input_layout"]) for r in routes}
    x = block_fp8._layout_tensor(
        3, 2048, (2049, 1, 1), dtype=E4M3 if quantized else torch.bfloat16, device="cpu"
    )
    scale = torch.ones(3, 16)
    call = lambda: op.quantized(x, scale) if quantized else op(x)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    count = len(launches)
    if quantized:
        call()  # FlashInfer stages every q/scale view to the same layout.
    else:
        with pytest.raises(RuntimeError, match="row stride=2049.*warm_up.*input_layouts"):
            call()
        assert len(launches) == count
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    extra = op.warm_up(batch_sizes=[3], quantized=quantized, input_layouts=[(2049, True)])
    assert op._warmed_inputs == {(r["config"], r["input_layout"]) for r in routes + extra}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    call()
