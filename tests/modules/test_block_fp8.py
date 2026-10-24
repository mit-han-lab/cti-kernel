# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-FP8 inference on the CPU: lazy imports, the frozen-weight rule, metadata checks that
run before any launch, launch heuristics and the optional FlashInfer provider's errors."""

from __future__ import annotations

import importlib.metadata
import sys

import pytest
import torch

from ctikernels import moe_impl
from ctikernels.linear import _block_fp8_dispatch, block_fp8
from ctikernels.linear.flashinfer import _block_fp8 as flashinfer_block_fp8

from ..support.child import run_child


def _device_properties(capability: tuple[int, int], sms: int) -> object:
    """Fake device properties, complete for torch's lazy CUDA init when a real GPU is visible."""
    from types import SimpleNamespace

    major, minor = capability
    name = f"sm_{major}{minor}"
    return SimpleNamespace(major=major, minor=minor, multi_processor_count=sms, name=name)


E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _weight(n: int = 256, k: int = 384, *, experts: int | None = None):
    lead = () if experts is None else (experts,)
    weight = torch.zeros(*lead, n, k, dtype=E4M3)
    return weight, torch.ones(*lead, -(-n // 128), -(-k // 128))


def test_importing_the_operations_loads_no_provider() -> None:
    code = (
        "import sys, ctikernels.linear.block_fp8, ctikernels.moe_impl; "
        "import ctikernels.linear.cute._block_fp8_provider; "
        "import ctikernels.linear.cute._block_fp8_act_quant; "
        "print(sorted(m for m in ('triton', 'flashinfer', 'sgl_kernel', 'cutlass') "
        "if m in sys.modules))"
    )
    loaded = run_child([sys.executable, "-c", code], timeout=600, check=True).stdout
    assert loaded.strip() == "[]"


@pytest.mark.parametrize("frozen", ["weight", "weight_scale_inv"])
@pytest.mark.parametrize("scale_dtype", [torch.float32, E8M0])
def test_a_weight_requiring_grad_raises_before_any_other_check(frozen: str, scale_dtype) -> None:
    weight, scale = _weight()
    scale = scale.to(scale_dtype)
    tensors = {"weight": weight.float().requires_grad_(), "weight_scale_inv": scale}
    if frozen == "weight_scale_inv":
        tensors = {"weight": weight, "weight_scale_inv": scale.requires_grad_()}
    # The weight is on the CPU (and, for the first case, not even E4M3): the frozen rule wins.
    with pytest.raises(ValueError, match=f"{frozen} is frozen"):
        block_fp8.BlockFP8Linear(tensors["weight"], tensors["weight_scale_inv"])
    with pytest.raises(ValueError, match=f"{frozen} is frozen"):
        block_fp8.dequantize_block_fp8(tensors["weight"], tensors["weight_scale_inv"])
    with pytest.raises(ValueError, match=f"{frozen} is frozen"):
        block_fp8.slice_block_fp8(tensors["weight"], tensors["weight_scale_inv"], -2, 0, 128)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16])
@pytest.mark.parametrize("operation", ["BlockFP8Linear", "dequantize_block_fp8", "slice_block_fp8"])
def test_public_weight_scale_validation_rejects_unsupported_dtypes(dtype, operation) -> None:
    weight, scale = _weight()
    args = (-2, 0, 128) if operation == "slice_block_fp8" else ()
    with pytest.raises(TypeError, match="weight_scale_inv must be"):
        getattr(block_fp8, operation)(weight, scale.to(dtype), *args)


def test_ue8m0_construction_converts_every_encoding_exactly(monkeypatch) -> None:

    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties((12, 0), 170)
    )
    # All encodings, in a strided grid: byte zero is 2^-127, 147 is 2^20, 255 is NaN.
    scale = torch.arange(256, dtype=torch.uint8).repeat_interleave(2)[::2].view(E8M0)[None]
    weight, _ = _weight(1, 256 * 128)
    op = block_fp8.BlockFP8Linear(weight, scale, provider="triton-fp8")
    expected = torch.tensor([2.0 ** (code - 127) for code in range(255)])
    assert op._scale.dtype == torch.float32
    assert torch.equal(op._scale[0, :255], expected)
    assert torch.equal(op._scale[0, :255], scale.to(torch.float32)[0, :255])
    assert torch.isnan(op._scale[0, 255])
    assert op._triton._scale is op._scale
    # Changing grad metadata after construction must still fail before any launch.
    scale.requires_grad_()
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op(torch.empty(1, weight.shape[1], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op.quantized(torch.empty(1, weight.shape[1], dtype=E4M3), torch.ones(1, 256))


@pytest.mark.parametrize("experts", [None, 2])
def test_ue8m0_dequantization_on_cpu_interpreter(monkeypatch, experts) -> None:
    # The CPU CI env has no Triton; the GPU suites run this with Triton installed.
    InterpretedFunction = pytest.importorskip("triton.runtime.interpreter").InterpretedFunction

    from ctikernels.linear.triton import _block_fp8 as kernel

    # Interpret the real decoding kernel on CPU: no CUDA or JIT compilation.
    monkeypatch.setattr(
        kernel, "_dequantize_kernel", InterpretedFunction(kernel._dequantize_kernel.fn)
    )
    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    weight, _ = _weight(1, 256 * 128, experts=experts)
    weight.fill_(1)
    codes = torch.arange(256, dtype=torch.uint8).repeat_interleave(2)[::2]
    scale = codes.view(E8M0)[None]
    if experts is not None:
        scale = scale[None].expand(experts, -1, -1)
    decoded = block_fp8.dequantize_block_fp8(weight, scale, dtype=torch.float32)
    expected = scale.to(torch.float32).repeat_interleave(128, -1)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0, equal_nan=True)


def test_cpu_tensors_are_rejected_without_launching() -> None:
    with pytest.raises(ValueError, match="one CUDA device"):
        block_fp8.BlockFP8Linear(*_weight())
    with pytest.raises(ValueError, match="one CUDA device"):
        block_fp8.dequantize_block_fp8(*_weight(experts=2))
    with pytest.raises(ValueError, match="CUDA device"):
        block_fp8.quantize_per_token_group_fp8(torch.zeros(2, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="runs on a CUDA device"):
        block_fp8.default_provider("cpu")


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda w, s: (w, s.T), ValueError, "must be \\(2, 3\\)"),
        (lambda w, s: (w, s[:, :1]), ValueError, "must be \\(2, 3\\)"),
        (lambda w, s: (w.view(torch.uint8), s), TypeError, "float8_e4m3fn"),
        (lambda w, s: (w, s.half()), TypeError, "weight_scale_inv must be"),
        (lambda w, s: (w[None], s), ValueError, "2 dimensions"),
        (lambda w, s: (w[:0], s[:0]), ValueError, "must not be empty"),
    ],
)
def test_weight_layout_is_checked_not_guessed(change, error, match) -> None:
    with pytest.raises(error, match=match):
        block_fp8.BlockFP8Linear(*change(*_weight()))


def test_dequantize_accepts_matrices_and_expert_stacks_only() -> None:
    weight, scale = _weight(experts=2)
    with pytest.raises(ValueError, match="\\[N, K\\] or \\[E, N, K\\]"):
        block_fp8.dequantize_block_fp8(weight[None], scale[None])
    with pytest.raises(TypeError, match="dtype must be"):
        block_fp8.dequantize_block_fp8(weight, scale, dtype=torch.float64)


@pytest.mark.parametrize("m", [1, 2, 4, 16, 17, 64, 65, 256, 257, 4096])
@pytest.mark.parametrize(("n", "k"), [(9216, 2048), (2048, 4096), (2048, 512), (200, 328)])
@pytest.mark.parametrize(
    ("sms", "capability"), [(48, (12, 1)), (48, None), (170, None), (188, None)]
)
def test_launch_configurations_cover_every_shape(m, n, k, sms, capability) -> None:
    block_m, block_n, split_k, warps, stages, *group_m = block_fp8._config(m, n, k, sms, capability)
    assert group_m in ([], [32]) and (not group_m or capability == (12, 1))
    assert block_m >= 16 and block_n >= 32 and warps in (4, 8) and stages in (2, 3, 4)
    groups = -(-k // 128)
    assert 1 <= split_k <= max(1, groups // 4) or split_k == 1
    if m <= 512:
        # GB10 reads each weight byte once with one 64-row tile from 17 rows of a wide weight.
        assert block_m <= max(16, (4 if capability == (12, 1) else 2) * m)
        tiles = -(-m // block_m) * -(-n // block_n)
        # A small batch splits K until its grid covers the SMs or the groups run out.
        assert split_k == 8 or tiles * split_k >= sms or 2 * split_k * 4 > groups
    else:
        assert split_k == 1


def test_launch_configurations_of_other_devices_ignore_the_gb10_rule() -> None:
    for m in (1, 16, 17, 64, 256, 512, 4096):
        for n, k in ((9216, 2048), (2048, 4096), (2048, 512)):
            assert block_fp8._config(m, n, k, 48) == block_fp8._config(m, n, k, 48, (12, 0))
            if m > 512:
                assert block_fp8._config(m, n, k, 48) == block_fp8._config(m, n, k, 48, (12, 1))


@pytest.mark.parametrize("capability", [(9, 0), (12, 0), (12, 1)])
def test_moe_configuration_steps_one_scale_group(capability) -> None:
    for tokens in (1, 4, 64, 4096):
        config = moe_impl.block_fp8_moe_config(tokens, 256, 1024, 2048, 8, capability=capability)
        assert config["BLOCK_SIZE_K"] == 128 and config["BLOCK_SIZE_M"] in (16, 32, 64)
        assert config["num_stages"] in ((2, 4) if capability == (12, 1) else (3,))


def _moe_operands(**changes):
    operands = {
        "A": torch.zeros(6, 384, dtype=E4M3),
        "B": torch.zeros(4, 256, 384, dtype=E4M3),
        "C": torch.zeros(3, 2, 256, dtype=torch.bfloat16),
        "a_scale": torch.ones(6, 3),
        "b_scale": torch.ones(4, 2, 3),
        "config": {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 16},
        "topk_weights": torch.ones(3, 2),
        "topk_ids": torch.zeros(3, 2, dtype=torch.int32),
        "sorted_token_ids": torch.zeros(32, dtype=torch.int32),
        "expert_ids": torch.zeros(2, dtype=torch.int32),
        "mul_routed_weight": True,
        "top_k": 1,
    }
    operands.update(changes)
    return operands


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"b_scale": None}, ValueError, "both a_scale and b_scale"),
        ({"A": torch.zeros(6, 384, dtype=torch.bfloat16)}, TypeError, "E4M3 A and B"),
        ({"a_scale": torch.ones(6, 3, dtype=torch.bfloat16)}, TypeError, "FP32 a_scale"),
        ({"C": torch.zeros(3, 2, 256)}, TypeError, "BF16 or FP16 C"),
        ({"B": torch.zeros(4, 384, 256, dtype=E4M3).transpose(1, 2)}, ValueError, "unit K"),
        ({"A": torch.zeros(6, 256, dtype=E4M3)}, ValueError, "A has 256 columns"),
        ({"a_scale": torch.ones(3, 6).T}, ValueError, None),
        ({"b_scale": torch.ones(4, 3, 2)}, ValueError, "b_scale must be"),
        ({"config": {"BLOCK_SIZE_K": 96, "BLOCK_SIZE_M": 16}}, ValueError, "BLOCK_SIZE_K"),
        # Output capacity and layout: the kernel writes N columns of each of the 6 routes.
        ({"C": torch.zeros(3, 2, 1, dtype=torch.bfloat16)}, ValueError, "C must be"),
        ({"C": torch.zeros(2, 2, 256, dtype=torch.bfloat16)}, ValueError, "C must be"),
        ({"C": torch.zeros(6, 256, dtype=torch.bfloat16)}, ValueError, "C must be"),
        (
            {"C": torch.zeros(2, 3, 256, dtype=torch.bfloat16).transpose(0, 1)},
            ValueError,
            "flatten to route rows",
        ),
        ({"A": torch.zeros(5, 384, dtype=E4M3), "a_scale": torch.ones(5, 3)}, ValueError, "rows"),
        ({"top_k": 2}, ValueError, "rows"),
        ({"topk_weights": torch.ones(5)}, ValueError, "topk_weights"),
        ({"topk_weights": torch.ones(2, 3).T}, ValueError, "topk_weights"),
        ({"expert_ids": torch.zeros(1, dtype=torch.int32)}, ValueError, "expert_ids"),
        ({"b_scale": torch.ones(4, 2, 3, requires_grad=True)}, ValueError, "require grad"),
    ],
)
def test_block_fp8_moe_operands_are_checked(changes, error, match) -> None:
    operands = _moe_operands(**changes)
    if changes.keys() == {"a_scale"} and match is None:
        moe_impl._check_block_fp8(**operands)  # any strides of the right shape are read as is
        return
    with pytest.raises(error, match=match):
        moe_impl._check_block_fp8(**operands)


@pytest.mark.parametrize(
    ("name", "tensor"),
    [
        ("expert_ids", torch.tensor([0, 99, 1, 99], dtype=torch.int32)[::2]),
        ("expert_ids", torch.zeros(1, dtype=torch.int32).expand(2)),
        ("sorted_token_ids", torch.arange(64, dtype=torch.int32)[::2]),
        ("sorted_token_ids", torch.zeros(1, dtype=torch.int32).expand(32)),
        ("topk_weights", torch.ones(6, 2)[::2]),  # unit inner stride, gaps between rows
        ("topk_weights", torch.ones(3, 4)[:, ::2]),
    ],
)
@pytest.mark.parametrize("launcher", [False, True])
def test_block_fp8_moe_rejects_unpacked_routes_on_cpu(name, tensor, launcher) -> None:
    operands = _moe_operands(**{name: tensor})
    check = moe_impl._check_block_fp8
    if launcher:
        check = moe_impl.fused_moe_kernel_triton
        operands.update(
            num_tokens_post_padded=torch.tensor([32], dtype=torch.int32),
            compute_type=torch.bfloat16,
        )
    with pytest.raises(ValueError, match=f"{name} must be contiguous"):
        check(**operands)


def test_block_fp8_moe_accepts_packed_routes_with_storage_offsets() -> None:
    moe_impl._check_block_fp8(
        **_moe_operands(
            expert_ids=torch.tensor([99, 0, 1], dtype=torch.int32)[1:],
            sorted_token_ids=torch.arange(33, dtype=torch.int32)[1:],
            topk_weights=torch.ones(4, 2)[1:],
            # Only numel() is used; the kernel never reads these IDs.
            topk_ids=torch.zeros(3, 4, dtype=torch.int32)[:, ::2],
        )
    )


def test_flashinfer_provider_refuses_strided_shards_before_importing() -> None:
    from ctikernels.linear.flashinfer._block_fp8 import FlashInferCutlassGemm

    weight = torch.zeros(256, 512, dtype=E4M3)
    shard, scale = block_fp8.slice_block_fp8(weight, torch.ones(2, 4), -1, 128, 384)
    with pytest.raises(ValueError, match="packed"):
        FlashInferCutlassGemm(shard, scale, capability=(12, 0))


def test_flashinfer_provider_reports_a_missing_or_unqualified_install(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="compute capability 10.x or 12.x"):
        block_fp8._flashinfer_gemm("op", (9, 0), (256, 256))
    with pytest.raises(ValueError, match="multiples of 128"):
        block_fp8._flashinfer_gemm("op", (12, 0), (200, 256))

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="flashinfer-python==0.7.0.post1"):
        block_fp8._flashinfer_gemm("op", (12, 0), (256, 256))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.1")
    gemm = type(sys)("flashinfer.gemm")
    gemm.gemm_fp8_nt_groupwise = object()
    monkeypatch.setitem(sys.modules, "flashinfer", type(sys)("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", gemm)
    with pytest.raises(RuntimeError, match="found 0.0.1"):
        block_fp8._flashinfer_gemm("op", (12, 0), (256, 256))
    assert flashinfer_block_fp8.QUALIFIED_VERSIONS == ("0.6.11.post3", "0.7.0.post1", "0.7.1")
    for qualified in flashinfer_block_fp8.QUALIFIED_VERSIONS:
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v=qualified: v)
        assert block_fp8._flashinfer_gemm("op", (12, 0), (256, 256)) is gemm.gemm_fp8_nt_groupwise


@pytest.mark.parametrize("experts", [None, 4])
@pytest.mark.parametrize("dim", [-2, -1])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16, E8M0])
def test_tp_slices_are_matching_views(experts, dim, scale_dtype) -> None:
    weight, scale = _weight(1000, 1000, experts=experts)
    weight.view(torch.uint8).copy_(torch.arange(weight.numel()).reshape(weight.shape) % 127)
    scale = torch.arange(scale.numel(), dtype=torch.float32).to(scale_dtype).reshape(scale.shape)
    for start, stop in ((128, 512), (512, 1000)):
        w, s = block_fp8.slice_block_fp8(weight, scale, dim, start, stop)
        expected_w = weight.narrow(dim, start, stop - start)
        expected_s = scale.narrow(dim, start // 128, -(-stop // 128) - start // 128)
        assert torch.equal(w.view(torch.uint8), expected_w.view(torch.uint8))
        assert torch.equal(s.view(torch.uint8), expected_s.view(torch.uint8))
        for view, original, expected in ((w, weight, expected_w), (s, scale, expected_s)):
            assert view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
            assert view.storage_offset() == expected.storage_offset()
            assert view.stride() == expected.stride()


def test_expert_slices_keep_weight_and_scale_experts_together() -> None:
    weight, scale = _weight(experts=5)
    scale.copy_(torch.arange(scale.numel()).reshape(scale.shape))
    w, s = block_fp8.slice_block_fp8(weight, scale, 0, 1, 4)
    assert torch.equal(w.view(torch.uint8), weight.narrow(0, 1, 3).view(torch.uint8))
    assert torch.equal(s, scale.narrow(0, 1, 3))
    assert w.data_ptr() == weight[1].data_ptr()
    assert s.data_ptr() == scale[1].data_ptr()
    s.fill_(17)
    assert torch.all(scale[1:4] == 17)


@pytest.mark.parametrize("dim", [-2, -1])
@pytest.mark.parametrize(("start", "stop"), [(1, 256), (128, 255), (129, 1000)])
def test_tp_slices_reject_partial_scale_blocks(dim, start, stop) -> None:
    with pytest.raises(ValueError, match="128-block rule"):
        block_fp8.slice_block_fp8(*_weight(1000, 1000), dim, start, stop)


@pytest.mark.parametrize("experts", [None, 3])
def test_column_shards_reassemble_the_scale_grid(experts) -> None:
    weight, scale = _weight(1000, 384, experts=experts)
    scale.copy_(torch.arange(scale.numel()).reshape(scale.shape))
    _, first = block_fp8.slice_block_fp8(weight, scale, -2, 0, 512)
    _, last = block_fp8.slice_block_fp8(weight, scale, -2, 512, 1000)
    assert torch.equal(torch.cat((first, last), dim=-2), scale)


@pytest.mark.parametrize("frozen", ["weight", "weight_scale_inv"])
def test_slice_checks_frozen_before_shape_dtype_and_dimension(frozen) -> None:
    weight, scale = _weight()
    if frozen == "weight":
        weight = weight.float()[None, None].requires_grad_()
    else:
        scale.requires_grad_()
    with pytest.raises(ValueError, match=f"{frozen} is frozen"):
        block_fp8.slice_block_fp8(weight, scale, 99, 1, 2)


@pytest.mark.parametrize(
    ("dim", "start", "stop"),
    [(0, 0, 1), (1, 0, 128), (-3, 0, 128), (-2, -128, 128), (-1, 256, 128), (-1, 0, 512)],
)
def test_slice_rejects_invalid_dimensions_and_ranges(dim, start, stop) -> None:
    with pytest.raises(ValueError):
        block_fp8.slice_block_fp8(*_weight(), dim, start, stop)


def test_slice_validates_scale_shape_and_weight_stride_on_cpu() -> None:
    weight, scale = _weight()
    with pytest.raises(ValueError, match="128 x 128 blocks"):
        block_fp8.slice_block_fp8(weight, scale[:, :1], -2, 0, 128)
    with pytest.raises(ValueError, match="unit stride along K"):
        block_fp8.slice_block_fp8(weight.T.contiguous().T, scale, -2, 0, 128)


@pytest.mark.parametrize("provider", block_fp8.PROVIDERS)
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, E8M0])
def test_providers_own_launch_policy_and_flashinfer_scale_copy(
    monkeypatch, provider, scale_dtype
) -> None:
    from types import SimpleNamespace

    from ctikernels.linear import triton

    calls = []
    kernel = SimpleNamespace(
        quantize=lambda *args, **kw: calls.append(("quantize", args, kw)),
        gemm=lambda *args, **kw: calls.append(("triton", args, kw)),
    )
    monkeypatch.setattr(triton, "_block_fp8", kernel, raising=False)
    from ctikernels.linear.cuda import _block_fp8_sm12x

    sm12x = SimpleNamespace(
        run=lambda *args: calls.append(("sm12x", args, {})),
        run_fused=lambda *args: calls.append(("sm12x-fused", args, {})),
    )
    monkeypatch.setattr(_block_fp8_sm12x, "_module", lambda *args: sm12x)
    monkeypatch.setitem(sys.modules, "tvm_ffi", type(sys)("tvm_ffi"))
    from contextlib import nullcontext

    from ctikernels.linear.cute import _block_fp8_provider

    monkeypatch.setattr(_block_fp8_provider, "available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=1))
    monkeypatch.setattr(
        _block_fp8_provider.CuteFP8Gemm,
        "_function",
        lambda *args: lambda *operands: calls.append(("cute", operands, {})),
    )
    gemm = type(sys)("flashinfer.gemm")
    gemm.gemm_fp8_nt_groupwise = lambda *args, **kw: calls.append(("flashinfer", args, kw))
    monkeypatch.setitem(sys.modules, "flashinfer", type(sys)("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", gemm)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: block_fp8.FLASHINFER_VERSION)
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties((12, 0), 170)
    )
    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    # The operation still validates its public operands; only CUDA metadata and launches are mocked.
    weight, scale = _weight(256, 1024)
    scale = torch.arange(scale.numel(), dtype=torch.bfloat16).reshape(scale.shape).T.contiguous().T
    if scale_dtype == E8M0:
        scale = scale.to(E8M0)
    op = block_fp8.BlockFP8Linear(weight, scale, provider=provider)
    assert op.provider == provider and op.provider_for(17) == provider
    if scale_dtype == E8M0:
        assert op._scale.dtype == torch.float32 and torch.equal(op._scale, scale.float())
    monkeypatch.setattr(
        block_fp8,
        "quantize_per_token_group_fp8",
        lambda x: (torch.empty(x.shape, dtype=E4M3), torch.empty(x.shape[0], x.shape[1] // 128)),
    )
    for m in (0, 16, 17):
        calls.clear()
        out = op(torch.empty(m, 1024, dtype=torch.bfloat16))
        assert out.shape == (m, 256)
        if not m:
            assert not calls
            continue
        source, args, kw = calls[-1]
        if provider == "cute-fp8":
            assert len(calls) == 1 and source == "cute" and args[0].dtype == E4M3
            assert args[1] is weight and torch.equal(args[3], scale.float())
            assert args[4].data_ptr() == out.data_ptr() and args[5] is None
        elif provider == "cuda-sm12x":
            # Quantization happens inside the GEMM: one launch on the BF16 activation.
            assert len(calls) == 1 and source == "sm12x-fused" and args[0].dtype == torch.bfloat16
            assert args[1].dtype == torch.uint8 and args[2].dtype == torch.float32
            assert torch.equal(args[2], scale.float()) and args[3].data_ptr() == out.data_ptr()
            assert args[4] == 170
        elif provider == "flashinfer-cutlass":
            assert source == "flashinfer" and args[0].dtype == E4M3
            assert args[3].dtype == torch.float32 and args[3].is_contiguous()
            assert torch.equal(args[3], scale.float())
            assert args[3].data_ptr() != scale.data_ptr()
            assert kw.pop("out").data_ptr() == out.data_ptr()
            assert kw == {
                "scale_major_mode": "K",
                "out_dtype": out.dtype,
                "backend": "cutlass",
            }
        else:
            assert source == "triton"
            assert (args[1] is None) == (m <= 16)
            assert len(calls) == (1 if m <= 16 else 2)
    calls.clear()
    q = torch.empty(17, 1024, dtype=E4M3)
    s = torch.empty(8, 17).T
    op.quantized(q, s)
    assert len(calls) == 1
    assert calls[0][0] != "sm12x-fused"
    scale_index = 2 if provider == "cute-fp8" else (2 if provider == "flashinfer-cutlass" else 1)
    assert calls[0][1][scale_index].shape == s.shape


@pytest.mark.parametrize("capability", [(8, 9), (10, 0), (10, 3), (12, 0), (12, 1)])
@pytest.mark.parametrize("qualified", [True, False])
def test_default_dispatch_keeps_device_thresholds(monkeypatch, capability, qualified) -> None:

    from ctikernels.linear.flashinfer import _block_fp8

    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    monkeypatch.setattr(block_fp8, "_cuda_device", lambda name, device: torch.device(device))
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: capability)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties(capability, 170)
    )

    def load(*args):
        if not qualified:
            raise RuntimeError("FlashInfer unavailable")
        return lambda *args, **kw: None

    monkeypatch.setattr(_block_fp8, "_flashinfer_gemm", load)
    weight, scale = _weight(256, 1024)
    op = block_fp8.BlockFP8Linear(weight, scale)
    small = "triton-fp8"
    hybrid = qualified and capability in ((12, 0), (12, 1))
    assert op.provider == (block_fp8.HYBRID if hybrid else small)
    assert op.provider_for(0) == small
    if hybrid:  # a shape the measured table lacks: the size rule
        flops = 16_000_000_000 if capability == (12, 0) else 2_000_000_000
        first = -(-flops // (2 * 256 * 1024))
        assert op.provider_for(first - 1) == small
        assert op.provider_for(first) == "flashinfer-cutlass"
    else:
        assert op.provider_for(1 << 20) == small
    forced = block_fp8.BlockFP8Linear(weight, scale, provider=small)
    assert forced.provider_for(1 << 20) == small


def test_dispatch_table_runs_are_well_formed() -> None:
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    providers = {dispatch.TRITON, dispatch.FLASHINFER, dispatch.SM12X, dispatch.CUTE}
    for device, table in dispatch.TABLE.items():
        # Every table device has a size rule for unmeasured shapes, except GB300 (Triton).
        assert device[:2] in dispatch.FLASHINFER_FLOPS or device[:2] == (10, 3)
        for (n, k), runs in table.items():
            assert n % 128 == 0 and k % 128 == 0
            lasts = [last for last, _ in runs]
            assert lasts == sorted(set(lasts)) and lasts[0] >= 1
            names = [name for _, name in runs]
            assert set(names) <= providers
            # FlashInfer's sm100 GEMM only from M = 33 (wrong at 18..32 unpadded).
            assert device[:2] != (10, 3) or all(
                last >= 32 for last, name in runs[:-1] if name == dispatch.TRITON
            )
            # The local CUDA GEMM is measured on GB10 (12.1) only.
            assert dispatch.SM12X not in names or device[:2] == (12, 1)
            assert all(a != b for a, b in zip(names, names[1:], strict=False)), (
                "runs must alternate"
            )


def test_dispatch_picks_the_run_holding_the_batch() -> None:
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    runs = ((192, "triton-fp8"), (256, "flashinfer-cutlass"), (448, "triton-fp8"))
    picks = [dispatch.provider(runs, m) for m in (1, 192, 193, 256, 257, 448, 10_000)]
    assert picks == ["triton-fp8"] * 2 + ["flashinfer-cutlass"] * 2 + ["triton-fp8"] * 3
    # Shapes the table lacks: FlashInfer from the device's GEMM size on.
    rule = dispatch.buckets((12, 0), 170, 256, 1024)
    first = -(-16_000_000_000 // (2 * 256 * 1024))
    assert dispatch.provider(rule, first - 1) == "triton-fp8"
    assert dispatch.provider(rule, first) == "flashinfer-cutlass"
    assert dispatch.buckets((9, 0), 132, 1024, 2048) is None
    assert dispatch.buckets((10, 3), 148, 1024, 2048) is None
    measured = dispatch.TABLE[(12, 0, 170)][(9216, 2048)]
    assert dispatch.buckets((12, 0), 170, 9216, 2048) is measured
    # Providers an operation could not construct hand their runs to the Triton GEMM.
    three = ((32, "triton-fp8"), (2048, "flashinfer-cutlass"), (4096, "cuda-sm12x"))
    assert dispatch.restrict(three, {"triton-fp8", "flashinfer-cutlass"}) == (
        (32, "triton-fp8"),
        (2048, "flashinfer-cutlass"),
        (4096, "triton-fp8"),
    )
    assert dispatch.restrict(three, {"triton-fp8", "cuda-sm12x"}) == (
        (2048, "triton-fp8"),
        (4096, "cuda-sm12x"),
    )
    assert dispatch.restrict(three, {"triton-fp8"}) == ((4096, "triton-fp8"),)


def test_quantizer_tiles_are_well_formed() -> None:
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    for device, runs in dispatch.QUANTIZER.items():
        assert device in dispatch.TABLE
        lasts = [last for last, _ in runs]
        assert lasts == sorted(set(lasts)) and lasts[0] >= 1
        tiles = [tile for _, tile in runs]
        assert all(a != b for a, b in zip(tiles, tiles[1:], strict=False)), "runs must alternate"
        for block_m, warps in tiles:
            assert block_m in (1, 2, 4, 8, 16, 32, 64) and warps in (1, 2, 4, 8, 16)
            assert block_m * 128 >= 32 * warps, "a warp must hold at least one element"


def test_quantizer_tile_follows_the_device_and_program_count() -> None:
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    # A device without a sweep: 4, 8 or 16 rows by batch size, 4 warps, whatever K.
    for rows, block_m in ((1, 4), (64, 4), (65, 8), (1024, 8), (1025, 16), (1 << 20, 16)):
        for groups in (1, 16, 32):
            assert dispatch.quantizer_tile((9, 0), 132, rows, groups) == (block_m, 4)
    for device, runs in dispatch.QUANTIZER.items():
        capability, sms = device[:2], device[2]
        for (last, tile), (previous, _) in zip(runs, ((0, None), *runs), strict=False):
            # Units are rows x groups: the run is chosen by their product.
            assert dispatch.quantizer_tile(capability, sms, last, 1) == tile
            assert dispatch.quantizer_tile(capability, sms, 1, last) == tile
            assert dispatch.quantizer_tile(capability, sms, previous + 1, 1) == tile
        assert dispatch.quantizer_tile(capability, sms, runs[-1][0] + 1, 64) == runs[-1][1]


@pytest.fixture
def cute_cpu(monkeypatch):
    """Keep real host policy/tensors, substituting only CUDA metadata and compilation."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ctikernels.linear.cute import _block_fp8_provider as impl

    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    monkeypatch.setattr(block_fp8, "_cuda_device", lambda name, device: torch.device(device))
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: (12, 1))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: _device_properties((12, 1), 48)
    )
    state = SimpleNamespace(capturing=False, stream=1, compiles=[], launches=[])
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=state.stream)
    )
    monkeypatch.setattr(impl, "available", lambda _: True)
    monkeypatch.setattr(impl, "_compiled", {})

    def compile_fn(self, config, dtype, strided=False):
        state.compiles.append(self._key(config, dtype, strided))
        return lambda *args: state.launches.append(args)

    monkeypatch.setattr(impl.CuteFP8Gemm, "_compile", compile_fn)
    return impl, state


@pytest.mark.parametrize("quantized", [False, True])
def test_linear_out_validation_precedes_launch(cute_cpu, monkeypatch, quantized):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    launches = []
    monkeypatch.setattr(op._triton, "run", lambda *args: launches.append(args))
    x = torch.zeros(2, 384, dtype=torch.bfloat16)
    q, scale = torch.zeros(2, 384, dtype=E4M3), torch.ones(2, 3)

    def call(out, **kwargs):
        return op.quantized(q, scale, out=out, **kwargs) if quantized else op(x, out=out, **kwargs)

    bad = [
        (object(), TypeError, "tensor"),
        (torch.empty(2, 256), TypeError, "out must be"),
        (torch.empty(2, 1, 256, dtype=torch.bfloat16), ValueError, "dimensions"),
        (torch.empty(3, 256, dtype=torch.bfloat16), ValueError, "out must be"),
        (torch.empty(2, 512, dtype=torch.bfloat16)[:, ::2], ValueError, "stride"),
        (torch.empty(256, dtype=torch.bfloat16).expand(2, -1), ValueError, "stride"),
        (torch.empty_strided((2, 256), (255, 1), dtype=torch.bfloat16), ValueError, "stride"),
        (torch.empty(2, 256, device="meta", dtype=torch.bfloat16), ValueError, "out must be on"),
    ]
    for out, error, match in bad:
        with pytest.raises(error, match=match):
            call(out)
    out = torch.empty(2, 256, dtype=torch.float16)
    with pytest.raises(ValueError, match="out_dtype must equal"):
        call(out, out_dtype=torch.bfloat16)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        call(out.requires_grad_())
    assert not launches
    with torch.no_grad():
        assert call(out) is out
    out.requires_grad_(False)
    assert call(out) is out and launches[-1][-1] is out
    assert call(out, out_dtype=torch.float16) is out
    # The frozen rule precedes output checks, too.
    op._weight.requires_grad_()
    with pytest.raises(ValueError, match="weight is frozen"):
        call(object())


@pytest.mark.parametrize("quantized", [False, True])
def test_linear_empty_out_and_default_dtype(cute_cpu, monkeypatch, quantized):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")

    def unexpected(*args):
        pytest.fail("empty input launched a provider")

    monkeypatch.setattr(op._triton, "run", unexpected)
    out = torch.empty(0, 512, dtype=torch.float16)[:, 256:]
    if quantized:
        args = (torch.empty(0, 384, dtype=E4M3), torch.empty(0, 3))
        assert op.quantized(*args, out=out) is out
        assert op.quantized(*args).dtype == torch.bfloat16
    else:
        x = torch.empty(2, 0, 384, dtype=torch.bfloat16)
        assert op(x, out=out) is out
        result = op(x)
        assert result.shape == (2, 0, 256) and result.dtype == x.dtype


@pytest.mark.parametrize("provider", ["cute-fp8", "cuda-sm12x", "flashinfer-cutlass"])
@pytest.mark.parametrize("stride,offset", [(512, 0), (512, 8), (512, 2), (512, 1), (513, 0)])
@pytest.mark.parametrize("m", [1, 3])
def test_provider_strided_out_direct_or_copy(cute_cpu, monkeypatch, provider, stride, offset, m):
    from types import SimpleNamespace

    from ctikernels.linear.cuda import _block_fp8_sm12x

    impl, _ = cute_cpu
    targets = []

    def write(out):
        targets.append(out)
        out.fill_(7)

    monkeypatch.setattr(impl.CuteFP8Gemm, "_compile", lambda *a: lambda *args: write(args[4]))
    monkeypatch.setitem(sys.modules, "tvm_ffi", type(sys)("tvm_ffi"))
    monkeypatch.setattr(
        _block_fp8_sm12x,
        "_module",
        lambda *a: SimpleNamespace(run=lambda *args: write(args[4])),
    )
    monkeypatch.setattr(
        flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: lambda *args, **kw: write(kw["out"])
    )
    op = block_fp8.BlockFP8Linear(*_weight(), provider=provider)
    parent = torch.full((m, stride), -19, dtype=torch.bfloat16)
    out = parent[:, offset : offset + 256]
    assert op.quantized(torch.zeros(m, 384, dtype=E4M3), torch.ones(m, 3), out=out) is out
    assert torch.all(out == 7)
    assert torch.all(parent[:, :offset] == -19) and torch.all(parent[:, offset + 256 :] == -19)
    alignment = 8 if provider == "cute-fp8" else 2
    direct = provider != "flashinfer-cutlass" and not (stride % alignment or offset % alignment)
    if provider == "flashinfer-cutlass":
        direct = m == 1 and offset % 8 == 0
    assert (targets[-1] is out) == direct


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("stride", [None, 512, 513])
def test_linear_warm_up_uses_public_dispatch(cute_cpu, monkeypatch, quantized, stride):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    launches = []
    method = "run" if quantized else "run_x"
    monkeypatch.setattr(op._cute, method, lambda *args: launches.append(args))
    routes = op.warm_up(
        batch_sizes=[0, 1, 4, 64, 512],
        dtype=torch.float16,
        quantized=quantized,
        out_row_stride=stride,
    )
    variants = 1 if stride is None else 3
    assert len(launches) == len(routes)
    assert {route["m"] for route in routes} == {1, 4, 64, 512}
    # warm_up launches largest first; routes come back in ascending M.
    in_launch_order = sorted(routes, key=lambda route: -route["m"])
    for index, (args, route) in enumerate(zip(launches, in_launch_order, strict=True)):
        m = route["m"]
        x, *_, out, config = args
        assert x.shape == (m, 384) and torch.count_nonzero(x.float()) == 0
        assert x.dtype == (E4M3 if quantized else torch.float16)
        assert out.dtype == (torch.bfloat16 if quantized else torch.float16)
        assert out.stride(0) == (256 if index % variants == 0 else stride)
        assert out.data_ptr() % 16 == (2 if index % variants == 2 else 0)
        assert config == op._cute_config(m) and op.provider_for(m) == "cute-fp8"
    launches.clear()
    for kwargs in (
        {"batch_sizes": [1, -1]},
        {"batch_sizes": [1, 1.5]},
        {"batch_sizes": [1], "out_row_stride": 255},
        {},
        {"max_m": 4, "batch_sizes": [1]},
        {"max_m": 0},
        {"max_m": True},
    ):
        with pytest.raises(ValueError):
            op.warm_up(**kwargs)
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        op.warm_up(1)
    assert not launches


@pytest.mark.parametrize("n", [136, 256])
def test_split_sum_writes_row_strided_out_on_cpu_interpreter(n):
    InterpretedFunction = pytest.importorskip("triton.runtime.interpreter").InterpretedFunction

    from ctikernels.linear.triton import _block_fp8 as kernel

    m, splits = 3, 4
    # Exact BF16 sums isolate addressing from the interpreter's truncating BF16 cast.
    partial = torch.arange(splits * m * n, dtype=torch.float32).reshape(splits, m, n) % 17 / 16
    parent = torch.full((m, n + 17), -19, dtype=torch.bfloat16)
    out = parent[:, 3 : n + 3]
    InterpretedFunction(kernel._split_sum_kernel.fn)[(-(-(m * n) // 1024),)](
        partial, out, m * n, n, out.stride(0), SPLIT_K=splits, BLOCK=1024
    )
    assert torch.equal(out, partial.sum(0).to(out.dtype))
    assert torch.all(parent[:, :3] == -19) and torch.all(parent[:, n + 3 :] == -19)


def test_cute_dependency_probe_is_exact_and_lazy(monkeypatch):
    from ctikernels.linear.cute import _block_fp8_provider as impl

    monkeypatch.delenv("CTI_BLOCK_FP8_CUTE", raising=False)
    for version in ("4.7.0", "4.8.0", "4.8.1"):
        monkeypatch.setattr(importlib.metadata, "version", lambda _, version=version: version)
        assert impl.available((12, 0)) == (version == "4.8.0")
        assert impl.available((12, 1)) == (version == "4.8.0")
        assert not impl.available((12, 2))
        assert not impl.available((10, 0))

    def missing(_):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert not impl.available((12, 1))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "4.8.0")
    monkeypatch.setenv("CTI_BLOCK_FP8_CUTE", "0")
    assert not impl.available((12, 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bm": 16},
        {"bn": 256},
        {"warps": 2},
        {"stages": 1},
        {"raster": "unknown"},
        {"group": 0},
        {"scheduler": "unknown"},
        {"epi": "unknown"},
        {"epi": "private", "stages": 3},
        {"epi": "private", "bm": 64, "bn": 128, "ctas_per_sm": 2},
        {"ctas_per_sm": 3},
        {"min_sk_iters": 0},
        {"min_sk_iters": 1.5},
        {"sk_order": "unknown"},
        {"sk_waves": 0},
        {"sk_waves": 3},
        {"tail_split_threshold": 0},
        {"tail_split_threshold": 1.1},
        {"stages": 4},
        {"ctas_per_sm": 2},
        {"bm": 32, "bn": 32, "warps": 8},
    ],
)
def test_cute_config_rejects_invalid_resources(kwargs):
    from ctikernels.linear.cute._block_fp8_provider import Config

    with pytest.raises(ValueError):
        Config(**kwargs)


def test_cute_config_names_and_resource_limits():
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS

    bases = {
        "swap128x64s2",
        "swap128x32s3",
        "normal64x128s2w4",
        "swap128x32s4",
        "swap128x32s4w8",
        "swap128x64s3",
        "swap128x64s3w8",
        "normal32x128s4",
        "normal64x128s2",
        "normal128x64s2",
        "normal64x128s3",
        "normal64x128s3w8",
        "normal128x128s2",
        "normal128x128s3",
        "normal128x128s2w8",
        "normal128x64s3",
        "normal64x128s4",
        "normal128x128s2w8g4",
        "normal128x128s2w8g8",
        "normal128x128s2w8g16",
        "swap128x32s2c2",
        "swap64x64s2c2",
        "swap64x32s3c2",
        "swap128x64s4",
        "swap128x32s4g4",
    }
    alias_names = {
        f"{base}{scheduler}-{atom}{pdl}"
        for base in bases
        for scheduler in ("", "-persistent", "-tailsplit")
        for atom in ("mx", "plain")
        for pdl in ("", "-pdl")
    }
    from dataclasses import replace

    excluded = {
        "normal64x128s2",
        "normal128x64s2",
        "normal128x128s3",
        "normal64x128s4",
        "swap128x64s4",
    }
    private_names = {
        f"{base}-{scheduler}-ovl{pdl}-{atom}"
        for base in bases - excluded
        for scheduler in ("persistent", "tailsplit")
        for atom in ("mx", "plain")
        for pdl in ("", "-pdl")
    }
    normalized_pdl_names = {
        f"{base}{scheduler}-pdl-{atom}"
        for base in bases
        for scheduler in ("", "-persistent", "-tailsplit")
        for atom in ("mx", "plain")
    }
    named = alias_names | private_names | normalized_pdl_names
    threshold_names = {
        f"{name}-t{percent}" for name in named if "-tailsplit" in name for percent in (75, 100)
    }
    policy_names = set()
    for name in named:
        if "-tailsplit" not in name:
            continue
        for threshold in ("", "-t100"):
            for order in ("", "-dpf"):
                for waves in ("", "-w1"):
                    variant = f"{name}{threshold}{order}{waves}"
                    policy_names.add(variant)
                    assert CONFIGS[variant] == replace(
                        CONFIGS[name],
                        tail_split_threshold=1.0 if threshold else 0.5,
                        sk_order="dp_first" if order else "helper_first",
                        sk_waves=1 if waves else 2,
                    )
    assert set(CONFIGS) == named | threshold_names | policy_names
    for name in threshold_names:
        base, percent = name.rsplit("-t", 1)
        assert CONFIGS[name] == replace(CONFIGS[base], tail_split_threshold=int(percent) / 100)
    for name in normalized_pdl_names:
        base, atom = name.rsplit("-pdl-", 1)
        assert CONFIGS[name] == CONFIGS[f"{base}-{atom}-pdl"]
    for name in alias_names:
        config = CONFIGS[name]
        assert config.epi == "alias"
        assert config.pdl == name.endswith("-pdl")
        if config.pdl:
            assert replace(config, pdl=False) == CONFIGS[name.removesuffix("-pdl")]
    for name in private_names:
        config = CONFIGS[name]
        base_policy, suffix = name.split("-ovl")
        base, scheduler = base_policy.rsplit("-", 1)
        atom = suffix.split("-")[-1]
        assert config == replace(
            CONFIGS[f"{base}-{atom}"],
            scheduler=scheduler,
            epi="private",
            pdl="-pdl-" in name,
        )
    for config in CONFIGS.values():
        assert config.smem_bytes <= 99 * 1024
        if config.ctas_per_sm == 2:
            assert 2 * (config.smem_bytes + 1024) <= 100 * 1024
        assert config.register_budget <= 255


@pytest.mark.parametrize(
    "device,shape,ceiling,config_name",
    [
        (device, shape, ceiling, config_name)
        for device, shapes in _block_fp8_dispatch.CUTE_CONFIG.items()
        for shape, runs in shapes.items()
        for ceiling, config_name in runs
    ],
)
def test_cute_dispatch_configs_exist(device, shape, ceiling, config_name):
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS

    assert config_name in CONFIGS, (device, shape, ceiling, config_name)


@pytest.mark.parametrize(
    "base,expected_bytes",
    [
        ("swap128x64s2", 66176),
        ("normal64x128s2w4", 66176),
        ("swap128x32s3", 70144),
        ("swap128x32s4", 90752),
        ("normal128x128s2", 99456),
        ("swap128x32s2c2", 49536),
        ("swap64x64s2c2", 41600),
        ("swap64x32s3c2", 41472),
    ],
)
def test_cute_private_epilogue_storage(base, expected_bytes):
    from dataclasses import replace

    from ctikernels.linear.cute._block_fp8_provider import CONFIGS

    alias = CONFIGS[base + "-mx"]
    private = CONFIGS[base + "-persistent-ovl-mx"]
    assert private.smem_bytes == expected_bytes
    assert private.smem_bytes - alias.smem_bytes == 2 * alias.token_tile * alias.weight_tile
    assert replace(private, scheduler="grid").smem_bytes == expected_bytes
    # Swizzle must be a permutation within the allocation and preserve each
    # four-element (eight-byte) vector used by finish_tile.
    bits, low, shift = private.epilogue_swizzle
    mask = ((1 << bits) - 1) << (low + shift)
    offsets = [i ^ ((i & mask) >> shift) for i in range(private.bm * private.bn)]
    assert sorted(offsets) == list(range(private.bm * private.bn))
    for i in range(0, len(offsets), 4):
        assert offsets[i : i + 4] == list(range(offsets[i], offsets[i] + 4))


def test_cute_config_dispatch_and_restrict(monkeypatch):
    from ctikernels.linear import _block_fp8_dispatch as dispatch
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS

    key = (12, 1, 48)
    runs = ((7, "swap128x32s2c2-mx"), (256, "normal64x128s3-persistent-plain"))
    monkeypatch.setitem(dispatch.CUTE_CONFIG, key, {(1536, 4096): runs})
    for m, expected in ((1, runs[0][1]), (7, runs[0][1]), (8, runs[1][1]), (9999, runs[1][1])):
        assert dispatch.cute_config((12, 1), 48, 1536, 4096, m) == expected
    for n in (512, 1024, 1536, 8192):
        for m in (1, 128, 129, 512, 513, 4096):
            selected = dispatch.cute_config((12, 0), 170, n, 4096, m)
            assert selected in CONFIGS and "tailsplit" not in selected
    assert dispatch.cute_config((12, 0), 170, 512, 4096, 128) == "swap128x32s4-mx"
    assert dispatch.cute_config((12, 0), 170, 1536, 4096, 128) == "swap128x64s3-mx"
    assert dispatch.cute_config((12, 0), 170, 8192, 4096, 129) == "normal128x128s2-mx"
    table = ((7, dispatch.CUTE), (128, dispatch.TRITON), (256, dispatch.CUTE))
    assert dispatch.restrict(table, {dispatch.TRITON}) == ((256, dispatch.TRITON),)
    assert dispatch.restrict(table, {dispatch.TRITON, dispatch.CUTE}) == table


def test_cute_cache_workspace_and_capture(cute_cpu):
    impl, state = cute_cpu
    op = impl.CuteFP8Gemm(*_weight(), capability=(12, 1), sms=48)
    config = "swap128x32s2c2-tailsplit-mx"
    state.capturing = True
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        op.run(
            torch.empty(7, 384, dtype=E4M3),
            torch.ones(7, 3),
            torch.empty(7, 256, dtype=torch.bfloat16),
            config,
        )
    state.capturing = False
    op.prepare(config)
    assert len(state.compiles) == 1 and not state.launches
    state.capturing = True
    with pytest.raises(RuntimeError, match="warmup"):
        op.run(
            torch.empty(7, 384, dtype=E4M3),
            torch.ones(7, 3),
            torch.empty(7, 256, dtype=torch.bfloat16),
            config,
        )
    state.capturing = False
    op.warmup((1, 7), config)
    assert len(state.compiles) == 1
    scratch = state.launches[-1][-1]
    assert scratch.shape == (96 * 32 * 128 + 2 * 96,)
    state.capturing = True
    op.run(
        torch.empty(200, 384, dtype=E4M3),
        torch.ones(200, 3),
        torch.empty(200, 256, dtype=torch.bfloat16),
        config,
    )
    # Unsupplied tail scratch is per call (the graph's pool under capture), never cached.
    assert state.launches[-1][-1] is not scratch
    assert state.launches[-1][-1].shape == scratch.shape and not hasattr(op, "_workspaces")
    state.capturing = False
    state.stream = 2
    op.warmup(7, config)
    assert len(state.compiles) == 1
    # Separate operations reuse code but own independent scratch.
    another = impl.CuteFP8Gemm(*_weight(), capability=(12, 1), sms=48)
    another.warmup(7, config)
    assert len(state.compiles) == 1
    assert state.launches[-1][-1] is not scratch
    for capability, sms in (((12, 0), 48), ((12, 1), 49)):
        other = impl.CuteFP8Gemm(*_weight(), capability=capability, sms=sms)
        other.prepare(config)
    assert len(state.compiles) == 3
    op.prepare(config, out_dtype=torch.float16)
    assert len(state.compiles) == 4


@pytest.mark.parametrize("field", ["weight", "scale"])
def test_cute_frozen_checks_precede_compilation_and_launch(cute_cpu, field):
    impl, state = cute_cpu
    weight, scale = _weight()
    op = impl.CuteFP8Gemm(weight, scale, capability=(12, 1), sms=48)
    (weight if field == "weight" else scale).requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        op.warmup(1, "swap128x32s4-mx")
    with pytest.raises(ValueError, match="frozen"):
        op.run(
            torch.empty(1, 384, dtype=E4M3),
            torch.ones(1, 3),
            torch.empty(1, 256),
            "swap128x32s4-mx",
        )
    with pytest.raises(ValueError, match="frozen"):
        impl.CuteFP8Gemm(weight, scale, capability=(8, 0), sms=1)
    assert not state.compiles and not state.launches


@pytest.mark.parametrize("all_cute", [False, True])
def test_cute_table_instantiation_quantizer_and_per_batch_config(cute_cpu, monkeypatch, all_cute):
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    impl, state = cute_cpu
    first, last = "swap128x32s2c2-mx", "normal64x128s3-persistent-plain"
    monkeypatch.setitem(
        dispatch.TABLE,
        (12, 1, 48),
        {
            (256, 384): ((256, dispatch.CUTE),)
            if all_cute
            else ((7, dispatch.TRITON), (256, dispatch.CUTE)),
        },
    )
    monkeypatch.setitem(dispatch.CUTE_CONFIG, (12, 1, 48), {(256, 384): ((16, first), (256, last))})
    quantized = []

    def quantize(x):
        q, scale = torch.empty_like(x, dtype=E4M3), torch.ones(x.shape[0], 3)
        quantized.append((q, scale))
        return q, scale

    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quantize)
    op = block_fp8.BlockFP8Linear(*_weight())
    assert op._cute is not None
    assert op.provider_for(8) == dispatch.CUTE
    for m, expected in ((8, first), (200, last)):
        out = op(torch.empty(m, 384, dtype=torch.bfloat16))
        args = state.launches[-1]
        assert args[0] is quantized[-1][0] and args[2] is quantized[-1][1]
        assert args[4].data_ptr() == out.data_ptr()
        assert state.compiles[-1][3] == impl.CONFIGS[expected]
        op.quantized(*quantized[-1])
    assert len(quantized) == 2 and len(state.launches) == 4
    monkeypatch.setattr(impl, "available", lambda _: False)
    unavailable = block_fp8.BlockFP8Linear(*_weight())
    assert unavailable._cute is None and unavailable.provider_for(200) == dispatch.TRITON
    with pytest.raises(RuntimeError, match="4.8.0"):
        block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    monkeypatch.setattr(impl, "available", lambda _: pytest.fail("unrequested CuTe constructed"))
    monkeypatch.setitem(dispatch.TABLE, (12, 1, 48), {(256, 384): ((256, dispatch.TRITON),)})
    assert block_fp8.BlockFP8Linear(*_weight())._cute is None


def test_cute_compile_cache_serializes_concurrent_first_use(cute_cpu):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    impl, state = cute_cpu
    op = impl.CuteFP8Gemm(*_weight(), capability=(12, 1), sms=48)
    barrier = Barrier(8)

    def prepare(_):
        barrier.wait()
        return op._function(impl.CONFIGS["swap128x32s4-mx"], torch.bfloat16)

    with ThreadPoolExecutor(max_workers=8) as pool:
        functions = list(pool.map(prepare, range(8)))
    assert len(state.compiles) == 1 and not state.launches
    assert all(fn is functions[0] for fn in functions)


@pytest.mark.parametrize("pdl", [False, True])
def test_cute_run_x_selects_quantizer(cute_cpu, monkeypatch, pdl):
    from ctikernels.linear.cute import _block_fp8_act_quant as quant

    impl, state = cute_cpu
    op = impl.CuteFP8Gemm(*_weight(256, 256), capability=(12, 1), sms=48)
    x = torch.zeros(3, 256, dtype=torch.bfloat16)
    q, scale = torch.zeros(3, 256, dtype=E4M3), torch.ones(3, 2)
    calls = []

    def quantizer(name):
        def run(value):
            assert value is x
            calls.append(name)
            return q, scale

        return run

    monkeypatch.setattr(quant, "quantize", quantizer("cute"))
    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quantizer("triton"))
    config = "swap128x32s4-mx" + ("-pdl" if pdl else "")
    out = torch.empty(3, 256, dtype=torch.bfloat16)
    op.run_x(x, out, config)
    assert calls == ["cute" if pdl else "triton"]
    assert state.compiles[0][3].pdl == pdl
    assert state.launches[0][0] is q and state.launches[0][2] is scale
    assert state.launches[0][4] is out


@pytest.mark.parametrize("layout", ["k", "mn"])
def test_ue8m0_activation_policy_and_direct_down_input(cute_cpu, monkeypatch, layout):
    from ctikernels.linear.cute import _block_fp8_act_quant as quant

    _, state = cute_cpu
    weight, scale = _weight()
    scale = scale.to(torch.float8_e8m0fnu)
    op = block_fp8.BlockFP8Linear(weight, scale, provider="cute-fp8", scale_ue8m0=True)
    q, s = quant.allocate_outputs(3, 384, "cpu", layout)
    calls = []

    def quantize(x, *, scale_ue8m0):
        assert x.shape == (3, 384) and scale_ue8m0
        calls.append(x)
        return q, s

    monkeypatch.setattr(quant, "quantize", quantize)
    x = torch.empty(3, 1, 384, dtype=torch.bfloat16)
    assert op(x).shape == (3, 1, 256)
    assert len(calls) == 1
    assert state.launches[-1][0] is q and state.launches[-1][2] is s
    op.quantized(q, s)
    assert len(calls) == 1  # down GEMM reads the supplied FP8/scales, no quantization
    assert state.launches[-1][0] is q and state.launches[-1][2] is s
    assert op._scale.dtype == torch.float32 and op._scale_source is scale
    weight.requires_grad_()
    with pytest.raises(ValueError, match="weight is frozen"):
        op(x)
    with pytest.raises(ValueError, match="weight is frozen"):
        op.quantized(q, s)
    assert len(state.launches) == 2


def test_warm_up_max_m_covers_every_dispatch_bucket():
    keys = {m: ("a" if m <= 5 else "b" if m <= 40 else "c") for m in range(1, 100)}
    sizes = block_fp8._warm_up_sizes("op", 64, None, keys.__getitem__)
    assert sizes == (1, 5, 40, 64)  # the largest of each key
    assert {keys[m] for m in sizes} == {keys[m] for m in range(1, 65)}
    assert block_fp8._warm_up_sizes("op", 1, None, keys.__getitem__) == (1,)
    assert block_fp8._warm_up_sizes("op", None, 3, keys.__getitem__) == (3,)
    assert block_fp8._warm_up_sizes("op", None, [0, 2], keys.__getitem__) == (0, 2)


def test_linear_warm_up_max_m_runs_bucket_representatives(cute_cpu, monkeypatch):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    launches = []
    monkeypatch.setattr(op._cute, "run_x", lambda *args: launches.append(args[0].shape[0]))
    op.warm_up(200)
    expected = block_fp8._warm_up_sizes("op", 200, None, op._dispatch_key)
    assert tuple(dict.fromkeys(launches)) == expected[::-1]  # largest first
    assert {op._dispatch_key(m) for m in launches} == {op._dispatch_key(m) for m in range(1, 201)}


@pytest.mark.parametrize("provider", ["triton-fp8"])
@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
@pytest.mark.parametrize("shape", [(256, 2048), (129, 2048), (136, 256), (136, 328)])
@pytest.mark.parametrize("quantized", [False, True])
def test_warm_up_covers_triton_configs_and_scalar_classes(
    cute_cpu, provider, hardware, shape, quantized
):
    from ctikernels.linear.triton._block_fp8_provider import _config

    n, k = shape
    capability, sms = hardware[:2], hardware[2]
    op = block_fp8.BlockFP8Linear(*_weight(n, k), provider=provider)
    op._capability, op._sms = capability, sms
    op._triton._capability, op._triton._sms = capability, sms
    key = lambda m: op._dispatch_key(m, quantized=quantized)
    sizes = block_fp8._warm_up_sizes("op", 512, None, key)

    def scalar(value):
        return value == 1, value % 16 == 0

    def signature(m):
        config = _config(m, n, k, sms, capability)
        quantizer = None
        if not quantized and m > 16:
            groups = (k + 127) // 128
            tile = block_fp8._block_fp8_dispatch.quantizer_tile(capability, sms, m, groups)
            quantizer = tile, scalar(m * groups if k % 128 == 0 else m)
        return (
            config,
            not quantized and m <= 16,
            scalar(m),
            scalar(m * n) if config[2] > 1 else None,
            quantizer,
        )

    assert {signature(m) for m in sizes} == {signature(m) for m in range(1, 513)}
    assert key(16) != key(32)
    if hardware == (12, 0, 170) and shape == (256, 2048) and not quantized:
        assert 32 in sizes


def test_flashinfer_warm_up_runs_the_largest_size_of_each_class(cute_cpu, monkeypatch):
    monkeypatch.setattr(flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: object())
    op = block_fp8.BlockFP8Linear(*_weight(), provider="flashinfer-cutlass")
    sizes = block_fp8._warm_up_sizes("op", 4096, None, op._dispatch_key)
    keys = {m: op._dispatch_key(m) for m in range(1, 4097)}
    assert {keys[m] for m in sizes} == set(keys.values())
    assert all(keys[later] != keys[m] for m in sizes[1:] for later in range(m + 1, 4097))
    # sm12x: the library adds no M class (the quantizer's remain); sm100 adds its row classes.
    assert {key[3][1] for key in keys.values()} == {None}
    op._flashinfer._capability = (10, 0)
    classes = {op._flashinfer.row_class(m) for m in range(1, 4097)}
    assert classes == {(True, False), (True, True), (False, True), (False, False)}
    assert op._flashinfer.row_class(17) == op._flashinfer.row_class(32) == (False, True)


def test_flashinfer_keeps_no_instance_scratch_and_stages_inputs(cute_cpu, monkeypatch):
    _, state = cute_cpu
    calls = []
    monkeypatch.setattr(
        flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: lambda *a, **kw: calls.append(a)
    )
    op = block_fp8.BlockFP8Linear(*_weight(), provider="flashinfer-cutlass")
    op.warm_up(16, quantized=True)
    assert not hasattr(op._flashinfer, "_temporaries")  # staging is per call
    q = torch.full((1, 385), 2.0).to(E4M3)[:, 1:]
    scale = torch.full((1, 4), 3.0)[:, 1:]
    out = torch.empty(1, 257, dtype=torch.bfloat16)[:, 1:]
    op.warm_up(batch_sizes=[1], quantized=True, input_layouts=[((385, True), (4, 1, True))])
    state.capturing = True
    op.quantized(q, scale, out=out)
    assert calls[-1][0].data_ptr() % 16 == calls[-1][2].data_ptr() % 16 == 0
    assert torch.equal(calls[-1][0].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(calls[-1][2], scale)


@pytest.mark.parametrize("k", [256, 384, 2048])
@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
def test_cute_warm_up_covers_quantizer_tiles_and_flattened_scalar_classes(cute_cpu, k, hardware):
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS

    op = block_fp8.BlockFP8Linear(*_weight(256, k), provider="cute-fp8")
    op._capability, op._sms = hardware[:2], hardware[2]
    sizes = block_fp8._warm_up_sizes("op", 512, None, op._dispatch_key)

    def signature(m):
        config = CONFIGS[op._cute_config(m)]
        if config.pdl:
            return config, "cute-act-quant"
        units = m * (k // 128)
        return (
            config,
            block_fp8._block_fp8_dispatch.quantizer_tile(hardware[:2], hardware[2], m, k // 128),
            units == 1,
            units % 16 == 0,
        )

    assert {signature(m) for m in sizes} == {signature(m) for m in range(1, 513)}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flashinfer_strided_output_goes_through_per_call_staging(cute_cpu, monkeypatch, dtype):
    _, state = cute_cpu
    targets = []

    def gemm(*args, out, **kwargs):
        targets.append(out)
        out.fill_(7)

    monkeypatch.setattr(flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: gemm)
    op = block_fp8.BlockFP8Linear(*_weight(), provider="flashinfer-cutlass")
    op.warm_up(5, quantized=True, out_dtype=dtype, out_row_stride=264)
    state.capturing = True
    for m in range(1, 6):
        q, scale = torch.zeros(m, 384, dtype=E4M3), torch.ones(m, 3)
        parent = torch.full((m, 264), -19, dtype=dtype)
        out = parent[:, 1:257]
        op.quantized(q, scale, out=out)
        assert targets[-1].data_ptr() != out.data_ptr() and targets[-1].shape == out.shape
        assert torch.all(out == 7)
        assert torch.all(parent[:, :1] == -19) and torch.all(parent[:, 257:] == -19)


def test_cute_strided_warm_up_prepares_packed_unaligned_fallback(cute_cpu):
    impl, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    # Quantized entry isolates the output dispatch without a mocked GPU quantizer.
    op.warm_up(batch_sizes=[4], quantized=True, out_row_stride=264)
    config = impl.CONFIGS[op._cute_config(4)]
    assert op._cute._key(config, torch.bfloat16, False) in impl._launched
    assert op._cute._key(config, torch.bfloat16, True) in impl._launched
    state.capturing = True
    out = torch.empty(4, 264, dtype=torch.bfloat16)[:, 1:257]
    op.quantized(torch.zeros(4, 384, dtype=E4M3), torch.ones(4, 3), out=out)


@pytest.mark.parametrize("quantized", [False, True])
def test_warm_up_covers_actual_triton_input_signatures(cute_cpu, monkeypatch, quantized):
    pytest.importorskip("triton")
    from ctikernels.linear.triton import _block_fp8 as kernels

    seen = set()
    capturing = False

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                signature = (
                    self.name,
                    tuple(
                        (a.dtype, a.data_ptr() % 16 == 0)
                        if isinstance(a, torch.Tensor)
                        else (a == 1, a % 16 == 0)
                        for a in args
                    ),
                    tuple(kwargs.items()),
                )
                if capturing:
                    assert signature in seen, signature
                else:
                    seen.add(signature)

            return launch

    for name in ("_quantize_kernel", "_gemm_kernel", "_split_sum_kernel"):
        monkeypatch.setattr(kernels, name, Kernel(name))
    monkeypatch.setattr(kernels, "_device_key", lambda _: ((12, 1), 48))
    op = block_fp8.BlockFP8Linear(*_weight(256, 2048), provider="triton-fp8")
    routes = op.warm_up(64, quantized=quantized, input_layouts="all")
    capturing = True
    cute_cpu[1].capturing = True
    # Independent strides include broadcasts, overlaps, padding and MN-major scales.
    for m in (1, 2, 15, 16, 17, 31, 32, 33, 48, 64):
        for row, offset in ((2048, 0), (2048, 1), (2051, 0), (2064, 1), (0, 0), (1, 1)):
            x = block_fp8._layout_tensor(
                m,
                2048,
                (row, 1, offset),
                dtype=E4M3 if quantized else torch.bfloat16,
                device="cpu",
            )
            if quantized:
                for sr, sc, so in (
                    (16, 1, 0),
                    (19, 1, 1),
                    (1, 48, 1),
                    (0, 0, 0),
                    (1, 1, 0),
                    (2, 2, 1),
                ):
                    scale = block_fp8._layout_tensor(
                        m, 16, (sr, sc, so), dtype=torch.float32, device="cpu", scale=True
                    )
                    op.quantized(x, scale)
            else:
                op(x)
    assert {r["provider"] for r in routes} == {op.provider_for(m) for m in range(1, 65)}


@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
@pytest.mark.parametrize(
    "shape,provider",
    [((256, 384), p) for p in block_fp8.PROVIDERS] + [((9216, 2048), None), ((2304, 2048), None)],
)
def test_warm_up_returns_every_provider_route(cute_cpu, monkeypatch, hardware, shape, provider):
    from ctikernels.linear.cuda import _block_fp8_sm12x
    from ctikernels.linear.cute import _block_fp8_provider

    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: hardware[:2])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: _device_properties(hardware[:2], hardware[2]),
    )
    monkeypatch.setattr(flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "tvm_ffi", type(sys)("tvm_ffi"))
    monkeypatch.setattr(_block_fp8_sm12x.CudaSm12xGemm, "run", lambda *a: None)
    monkeypatch.setattr(_block_fp8_provider.CuteFP8Gemm, "run", lambda *a: None)
    monkeypatch.setattr(block_fp8.TritonFP8Gemm, "run", lambda *a: None)
    op = block_fp8.BlockFP8Linear(*_weight(*shape), provider=provider)
    routes = op.warm_up(96, quantized=True)
    expected = {op._dispatch_key(m, quantized=True) for m in range(1, 97)}
    assert isinstance(routes, list) and all(isinstance(r, dict) for r in routes)
    assert {r["provider"] for r in routes} == {op.provider_for(m) for m in range(1, 97)}
    assert {r["config"] for r in routes} == expected
    assert all(r["bucket"] == (r["m"], r["m"]) for r in routes)


@pytest.mark.parametrize("quantized", [False, True])
def test_warm_up_rejects_unbounded_stride_classes_in_capture(cute_cpu, monkeypatch, quantized):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    calls = []
    monkeypatch.setattr(op._triton, "run", lambda *args: calls.append(args))
    x = torch.zeros(384, dtype=E4M3 if quantized else torch.bfloat16).as_strided(
        (1, 384), (2**32, 1)
    )
    scale = torch.ones(1, 3)
    call = lambda: op.quantized(x, scale) if quantized else op(x)
    call()
    assert len(calls) == 1

    op.warm_up(batch_sizes=[1], quantized=quantized)
    count = len(calls)
    state.capturing = True
    if not quantized:
        # Flattening normalizes singleton row strides before the kernel sees them.
        call()
        assert calls[-1][0].stride(0) == 384
        return
    with pytest.raises(RuntimeError, match="warm_up.*int32 input strides"):
        call()
    assert len(calls) == count


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_warm_up_ue8m0_covers_scalar_and_vector_quantizer(cute_cpu, monkeypatch, dtype):
    from ctikernels.linear.cute import _block_fp8_act_quant as quantizer

    _, state = cute_cpu
    warmed = set()

    def quantize(x, *, scale_ue8m0):
        key = (x.dtype, x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0, scale_ue8m0)
        if state.capturing:
            assert key in warmed
        else:
            warmed.add(key)
        return torch.zeros(x.shape, dtype=E4M3), torch.ones(x.shape[0], x.shape[1] // 128)

    monkeypatch.setattr(quantizer, "quantize", quantize)
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8", scale_ue8m0=True)
    monkeypatch.setattr(op._triton, "run", lambda *a: None)
    op.warm_up(32, dtype=dtype)
    assert warmed == {(dtype, True, True)}
    x = torch.zeros(32, 385, dtype=dtype)[:, 1:]
    state.capturing = True
    with pytest.raises(RuntimeError, match="row stride=385.*warm_up.*input_layouts"):
        op(x)
    state.capturing = False
    op.warm_up(batch_sizes=[32], dtype=dtype, input_layouts=[(385, True)])
    assert warmed == {(dtype, True, True), (dtype, False, True)}
    state.capturing = True
    op(torch.zeros(32, 385, dtype=dtype)[:, 1:])


@pytest.mark.parametrize("quantized", [False, True])
def test_warm_up_layout_selection_and_bookkeeping(cute_cpu, monkeypatch, quantized):
    op = block_fp8.BlockFP8Linear(*_weight(256, 2048), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a: None)
    routes = op.warm_up(batch_sizes=[17], quantized=quantized)
    assert len(routes) == (2 if quantized else 1)
    assert op._warmed_inputs == {(r["config"], r["input_layout"]) for r in routes}
    assert all(
        layout == (2048, 1, 0) for layout, _, _ in op._input_layouts(17, quantized=quantized)
    )
    examples = [((2051, True), (19, 2, True))] if quantized else [(2051, True)]
    extra = op.warm_up(batch_sizes=[17], quantized=quantized, input_layouts=examples * 2)
    assert len(extra) == 1
    assert op._warmed_inputs == {(r["config"], r["input_layout"]) for r in routes + extra}
    # Independent reconstruction of the exhaustive set from the previous candidates.
    from itertools import product

    scales = block_fp8._layout_candidates(16, scale=True) if quantized else (None,)
    expected = {
        op._input_layout_key(
            17,
            (row, offset == 0),
            (*scale[:2], scale[2] == 0) if scale else None,
            quantized=quantized,
        )
        for (row, _, offset), scale in product(block_fp8._layout_candidates(2048), scales)
    }
    exhaustive = op.warm_up(batch_sizes=[17], quantized=quantized, input_layouts="all")
    assert {r["input_layout"] for r in exhaustive} == expected
    assert len(exhaustive) == len(expected) == (108 if quantized else 8)


# Strided scales exist only for quantized inputs.
@pytest.mark.parametrize(
    ("quantized", "layout"),
    [
        (False, "padded"),
        (False, "misaligned"),
        (True, "padded"),
        (True, "misaligned"),
        (True, "scale"),
    ],
)
def test_capture_rejects_unwarmed_layout_before_launch(cute_cpu, monkeypatch, quantized, layout):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    launches = []
    monkeypatch.setattr(op._triton, "run", lambda *a: launches.append(a))
    op.warm_up(32, quantized=quantized)
    x = block_fp8._layout_tensor(
        17,
        384,
        (385 if layout == "padded" else 384, 1, int(layout == "misaligned")),
        dtype=E4M3 if quantized else torch.bfloat16,
        device="cpu",
    )
    scale = block_fp8._layout_tensor(
        17,
        3,
        (7, 2, 1) if layout == "scale" else (3, 1, 0),
        dtype=torch.float32,
        device="cpu",
        scale=True,
    )
    call = lambda: op.quantized(x, scale) if quantized else op(x)
    call()  # Eager calls remain unrestricted, but do not register a warm_up route.
    count = len(launches)
    state.capturing = True
    with pytest.raises(
        RuntimeError, match="input layout class.*row stride=.*warm_up.*input_layouts"
    ):
        call()
    assert len(launches) == count
    state.capturing = False
    example = (x.stride(0), x.data_ptr() % 16 != 0)
    if quantized:
        example = (example, (*scale.stride(), scale.data_ptr() % 16 != 0))
    op.warm_up(batch_sizes=[17], quantized=quantized, input_layouts=[example])
    state.capturing = True
    call()
    count = len(launches)
    with pytest.raises(RuntimeError, match="matching batch and dtypes"):
        if quantized:
            op.quantized(x, scale, out_dtype=torch.float16)
        else:
            op(x.to(torch.float16))
    assert len(launches) == count


@pytest.mark.parametrize("layouts", ["invalid", None, [(384, 0)], [(-1, False)], [(2**32, False)]])
def test_warm_up_rejects_invalid_input_layouts(cute_cpu, monkeypatch, layouts):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a: pytest.fail("launched invalid layout"))
    with pytest.raises(ValueError, match="input_layouts"):
        op.warm_up(batch_sizes=[2], input_layouts=layouts)


def test_default_quantized_warm_up_covers_native_scale_padding(cute_cpu, monkeypatch):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a: None)
    op.warm_up(64, quantized=True)
    state.capturing = True
    for m in range(1, 65):
        q = torch.zeros(m, 384, dtype=E4M3)
        op.quantized(q, torch.ones(m, 3))
        op.quantized(q, torch.ones(3, (m + 3) // 4 * 4).t()[:m])


def test_explicit_singleton_layout_uses_flattened_stride(cute_cpu, monkeypatch):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a: None)
    routes = op.warm_up(batch_sizes=[1], input_layouts=[(385, True)])
    x = torch.zeros(1, 385, dtype=torch.bfloat16)[:, 1:]
    state.capturing = True
    op(x)
    assert routes[0]["input_layout"] == op._input_layout_key(1, (384, False))


def test_capture_checks_before_flattening_copy(cute_cpu, monkeypatch):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a: None)
    x = torch.zeros(2, 3, 384, dtype=torch.bfloat16).transpose(0, 1)
    state.capturing = True
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "reshape", lambda *a: pytest.fail("copy before guard"))
        with pytest.raises(RuntimeError, match="warm_up.*input_layouts"):
            op(x)
    state.capturing = False
    op.warm_up(batch_sizes=[6])
    state.capturing = True
    assert op(x).shape == (3, 2, 256)


@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
@pytest.mark.parametrize("shape", [(2048, 2048), (2304, 2048), (9216, 2048)])
def test_workspace_bytes_bounds_every_dispatched_m(cute_cpu, monkeypatch, hardware, shape):
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS, tail_workspace

    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: hardware[:2])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: _device_properties(hardware[:2], hardware[2]),
    )
    monkeypatch.setattr(flashinfer_block_fp8, "_flashinfer_gemm", lambda *a: object())
    monkeypatch.setattr(
        flashinfer_block_fp8.FlashInferCutlassGemm,
        "workspace_bytes",
        property(lambda _: 40 * 1024 * 1024),
    )
    monkeypatch.setitem(sys.modules, "tvm_ffi", type(sys)("tvm_ffi"))
    op = block_fp8.BlockFP8Linear(*_weight(*shape))
    n, k = shape
    aligned = lambda size: (size + 15) // 16 * 16
    needs = []
    for m in range(1, 514):
        chosen = op.provider_for(m)
        sizes = [2 * m * k]
        if chosen in ("cute-fp8", "flashinfer-cutlass", "cuda-sm12x"):
            sizes += [2 * m * n]
        if chosen in ("cute-fp8", "flashinfer-cutlass") or (
            chosen.startswith("triton-") and m > 16
        ):
            sizes += [m * k, 4 * m * ((k + 127) // 128)]
        if chosen.startswith("triton-"):
            splits = op._triton.config_for(m)[2]
            if splits > 1:
                sizes += [4 * splits * m * n]
        elif chosen == "cute-fp8":
            cfg = CONFIGS[op._cute_config(m)]
            if cfg.scheduler == "tailsplit":
                sizes += [4 * sum(tail_workspace(cfg, hardware[2]))]
        elif chosen == "flashinfer-cutlass":
            sizes += [op._flashinfer.workspace_bytes]
        assert op._workspace_layout(m).nbytes == sum(map(aligned, sizes))
        needs.append(op._workspace_bytes(m))
    for max_m in (16, 256, 320, 512, 513):
        assert op.workspace_bytes(max_m) == max(needs[:max_m])
    if hardware == (12, 0, 170) and shape == (2048, 2048):
        assert op.provider_for(320) == "triton-fp8"
        assert needs[319] > needs[255] and needs[319] > needs[511]
        assert op.workspace_bytes(512) > op._workspace_bytes(512)
    assert not cute_cpu[1].launches and not cute_cpu[1].compiles


@pytest.mark.parametrize("max_m", [0, -1, True, 1.5, None])
def test_workspace_bytes_rejects_invalid_max_m(cute_cpu, max_m):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    with pytest.raises(ValueError, match="max_m must be positive"):
        op.workspace_bytes(max_m)


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
        "overlap",
        "weight",
        "weight_scale",
        "out",
    ],
)
def test_workspace_validation_before_launch(cute_cpu, monkeypatch, quantized, bad):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    count = max(op.workspace_bytes(2) // 4, op._weight.numel() // 4)
    workspace = torch.empty(count + 4)
    x = torch.zeros(2, 384, dtype=torch.bfloat16)
    q, scale = torch.zeros(2, 384, dtype=E4M3), torch.ones(2, 3)
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
    elif bad == "overlap":
        if quantized:
            scale = workspace[:6].view(2, 3)
        else:
            x = workspace.view(torch.bfloat16)[:768].view(2, 384)
    out = None
    if bad == "weight":
        op._weight = workspace.view(E4M3)[: op._weight.numel()].view_as(op._weight)
    elif bad == "weight_scale":
        op._scale = workspace[: op._scale.numel()].view_as(op._scale)
    elif bad == "out":
        out = workspace.view(torch.bfloat16)[: 2 * op.out_features].view(2, op.out_features)
    monkeypatch.setattr(op, "_run_workspace", lambda *a: pytest.fail("launch before validation"))
    with pytest.raises(ValueError, match="workspace"):
        op.quantized(q, scale, out=out, workspace=workspace) if quantized else op(
            x, out=out, workspace=workspace
        )


def test_workspace_cute_views_and_counters(cute_cpu, monkeypatch):
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS, tail_workspace

    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    config = "swap128x32s2c2-tailsplit-mx"
    monkeypatch.setattr(op, "_cute_config", lambda _: config)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    workspace = torch.full((op.workspace_bytes(7) // 4,), float("nan"))
    q = torch.zeros(7, 385, dtype=E4M3)[:, 1:]
    scale = torch.ones(7, 3)
    out = torch.empty(7, 257, dtype=torch.bfloat16)[:, 1:]
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace allocation"))
    assert op.quantized(q, scale, out=out, workspace=workspace) is out
    launch = cute_cpu[1].launches[-1]
    for t in (launch[0], launch[4], launch[5]):
        assert t.untyped_storage().data_ptr() == workspace.data_ptr()
        assert t.data_ptr() % 16 == 0
    words, _ = tail_workspace(CONFIGS[config], 48)
    assert torch.count_nonzero(launch[5][words:]) == 0
    assert torch.isnan(launch[5][:words]).all()


@pytest.mark.parametrize("quantized", [False, True])
def test_workspace_outputs_survive_reuse(cute_cpu, monkeypatch, quantized):
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    workspace = torch.empty(op.workspace_bytes(2) // 4)
    x = torch.zeros(2, 384, dtype=torch.bfloat16)
    q, scale = x.to(E4M3), torch.ones(2, 3)
    calls = []

    def run(x, scale, out, views):
        views.storage.fill_(len(calls) + 1)
        out.fill_(len(calls) + 1)
        calls.append(out)

    monkeypatch.setattr(op, "_run_workspace", run)

    def call():
        return (
            op.quantized(q, scale, workspace=workspace) if quantized else op(x, workspace=workspace)
        )

    first = call()
    saved = first.clone()
    second = call()
    assert torch.equal(first, saved)
    for output in (first, second):
        assert output.untyped_storage().data_ptr() != workspace.untyped_storage().data_ptr()
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()


@pytest.mark.parametrize("warm_packed", [False, True])
def test_workspace_capture_checks_caller_layout(cute_cpu, monkeypatch, warm_packed):
    # Same layout key as warm_up and the workspace-free path: the caller's
    # layout, whose staging repair into the workspace is part of the class.
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="cute-fp8")
    q = torch.zeros(3, 385, dtype=E4M3)[:, 1:]
    scale = torch.ones(3, 3)
    workspace = torch.empty(op.workspace_bytes(3) // 4)
    out = torch.empty(3, 256, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    op.warm_up(
        batch_sizes=[3], quantized=True, input_layouts="packed" if warm_packed else [(385, True)]
    )
    state.launches.clear()
    state.capturing = True
    if warm_packed:
        monkeypatch.setattr(torch.Tensor, "copy_", lambda *a: pytest.fail("copy before guard"))
        with pytest.raises(RuntimeError, match="input layout class.*was not warmed"):
            op.quantized(q, scale, out=out, workspace=workspace)
        assert not state.launches
    else:
        monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace allocation"))
        assert op.quantized(q, scale, out=out, workspace=workspace) is out
        launched = state.launches[-1][0]
        assert launched.stride() == (384, 1) and launched.data_ptr() % 16 == 0
        assert launched.untyped_storage().data_ptr() == workspace.data_ptr()


@pytest.mark.parametrize("quantized", [False, True])
def test_workspace_capture_checks_unrepaired_layout(cute_cpu, monkeypatch, quantized):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    calls = []
    monkeypatch.setattr(op._triton, "run", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    op.warm_up(batch_sizes=[3], quantized=quantized)
    calls.clear()
    x = torch.zeros(3, 385, dtype=E4M3 if quantized else torch.bfloat16)[:, 1:]
    scale = torch.ones(3, 3)
    workspace = torch.empty(op.workspace_bytes(3) // 4)
    out = torch.empty(3, 256, dtype=torch.bfloat16)

    def call():
        return (
            op.quantized(x, scale, out=out, workspace=workspace)
            if quantized
            else op(x, out=out, workspace=workspace)
        )

    state.capturing = True
    with pytest.raises(RuntimeError, match="input layout class.*was not warmed"):
        call()
    assert not calls
    state.capturing = False
    op.warm_up(batch_sizes=[3], quantized=quantized, input_layouts=[(385, True)])
    state.capturing = True
    assert call() is out


def test_workspace_capture_checks_before_flattening_copy(cute_cpu, monkeypatch):
    _, state = cute_cpu
    op = block_fp8.BlockFP8Linear(*_weight(), provider="triton-fp8")
    monkeypatch.setattr(op._triton, "run", lambda *a, **kw: None)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    x = torch.zeros(2, 3, 384, dtype=torch.bfloat16).transpose(0, 1)
    workspace = torch.empty(op.workspace_bytes(6) // 4)
    state.capturing = True
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", lambda *a: pytest.fail("copy before guard"))
        with pytest.raises(RuntimeError, match="warm_up.*input_layouts"):
            op(x, workspace=workspace)
    state.capturing = False
    op.warm_up(batch_sizes=[6])
    state.capturing = True
    assert op(x, workspace=workspace).shape == (3, 2, 256)


def test_sm12x_gemm_asks_the_driver_for_a_versioned_entry_point():
    # An unversioned cudaGetDriverEntryPoint asks for the compiling runtime's version, which an
    # older driver rejects (CUDA 13.0 driver, 13.1 toolkit: cudaErrorInvalidValue on GB10).
    from pathlib import Path

    import ctikernels

    source = (Path(ctikernels.__file__).parent / "csrc/jit/block_fp8_gemm_sm12x.cu").read_text()
    versioned = source.index("cudaGetDriverEntryPointByVersion(")
    assert "12000" in source[versioned : versioned + 200]  # the 12.0 ABI, where it appeared
    unversioned = source.find("::cudaGetDriverEntryPoint(")
    assert unversioned == -1 or "#else" in source[versioned:unversioned]
