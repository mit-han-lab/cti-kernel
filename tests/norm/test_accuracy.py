# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU accuracy of the norms: FP64 oracle, PyTorch at the same precision as the yardstick.

See docs/testing.md. Bitwise contracts (stored sums, buffers left alone) stay exact.
"""

from __future__ import annotations

from itertools import product

import pytest
import torch

from ctikernels import _flashinfer_version
from ctikernels.norm import cute, training
from ctikernels.norm import inference as norm
from ctikernels.norm._dispatch import TABLE, Config, entry_for, tune_configs

from ..gpu._exhaustive import ci_cases
from ..support.accuracy import assert_accurate, errors
from . import oracle

pytestmark = pytest.mark.gpu


def _check_cute_norm(m, hidden, dtype, zero, config=None, fp32_weight=False):
    _check_norm_provider("cute", m, hidden, dtype, zero, config or Config(), fp32_weight)


def _check_norm_provider(provider, m, hidden, dtype, zero, config, fp32_weight=False):
    """Check a selected provider against FP64 with the PyTorch precision baseline."""
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("FlashInfer provider requires the qualified pin")
    torch.manual_seed(20261015 + m)
    x = torch.randn((m, hidden), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    if fp32_weight:
        weight = weight.float()
    ours_x, ours_residual = x.clone(), residual.clone()
    op = norm.AddRMSNorm(provider=provider, config=config, static_weight=config.static_weight)
    returned = op(ours_x, ours_residual, weight, 1e-6, zero_centered=zero)
    assert (
        returned[0] is ours_x
        and returned[1] is ours_residual
        and op.provider == (provider if m else "empty")
    )
    if not m:
        return
    oracle.check_add_rms_norm(ours_x, x, residual, weight, zero_centered=zero)
    _, stored = oracle.add_rms_norm(x, residual, weight, zero_centered=zero, upcast=False)
    assert_accurate(
        ours_residual, x.double() + residual.double(), stored, family="norm", name="stored_sum"
    )


def _shipped_norm_cases():
    seen = set()
    cases = []
    for kind, table in TABLE.items():
        for (hidden, dtype, _residual_dtype, zero, static), bands in table.items():
            for (lo, _hi), entry in bands.items():
                if _hi == 0:
                    continue
                # One BF16 row per shipped width/provider/config/weight convention
                # in core; every device/dtype/band remains in qualification.
                key = (hidden, zero, entry)
                core = dtype == "bfloat16" and key not in seen
                if core:
                    seen.add(key)
                cases.append(
                    pytest.param(
                        kind,
                        hidden,
                        dtype,
                        zero,
                        static,
                        max(1, lo),
                        entry,
                        marks=() if core else pytest.mark.exhaustive,
                    )
                )
    return cases


def _check_fp32_residual(
    provider, m, hidden, dtype, zero, config=None, fp32_weight=False, *, mixed=True
):
    torch.manual_seed(20261016 + m)
    x = torch.randn(m, hidden, device="cuda", dtype=dtype)
    residual = torch.randn(m, hidden, device="cuda", dtype=torch.float32 if mixed else dtype)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32 if fp32_weight else dtype)
    _, summed = oracle.add_rms_norm(x, residual, weight, zero_centered=zero, upcast=False)
    for store in (True, False):
        actual, actual_r = x.clone(), residual.clone()
        aux = torch.full((m, 3 * hidden), 17.0, device="cuda", dtype=residual.dtype)
        sum_out = aux[:, hidden : 2 * hidden]
        op = norm.AddRMSNorm(
            provider=provider,
            config=config,
            static_weight=config.static_weight if config else False,
        )
        op(
            actual,
            actual_r,
            weight,
            1e-6,
            zero_centered=zero,
            store_residual=store,
            sum_out=sum_out,
        )
        assert op.provider in ("cute", "triton")
        if provider != "auto":
            assert op.provider == provider
        _assert_norm_bits(actual_r, summed if store else residual)
        _assert_norm_bits(sum_out, summed)
        assert (aux[:, :hidden] == 17).all() and (aux[:, 2 * hidden :] == 17).all()
        oracle.check_add_rms_norm(actual, x, residual, weight, zero_centered=zero)
        assert_accurate(
            sum_out, x.double() + residual.double(), summed, family="norm", name="stored_sum"
        )

    return op


def _assert_norm_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("hidden_size", (128, 256, 4096, 4097, 5120))
def test_zero_centered_rms_norm_matches_reference_on_strided_rows(
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261001)
    eps = 1e-6
    storage = torch.randn((7, 9, hidden_size), device="cuda", dtype=dtype)
    x = storage[:, :3, :]
    weight = torch.randn(hidden_size, device="cuda", dtype=weight_dtype)
    original_storage = storage.clone()

    with torch.inference_mode():
        output = norm.rms_norm(x, weight, eps, zero_centered=True)
        from_copy = norm.rms_norm(x.contiguous(), weight, eps, zero_centered=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)

    with torch.inference_mode():
        returned = norm.rms_norm_(x, weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert returned is x
    assert output.is_contiguous()
    oracle.check_rms_norm(output, original_storage[:, :3, :], weight, eps, zero_centered=True)
    torch.testing.assert_close(output, from_copy, rtol=0, atol=0)
    torch.testing.assert_close(x, output, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, 3:], original_storage[:, 3:], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("hidden_size", (128, 4096, 4097, 5120))
def test_zero_centered_add_rms_norm_equals_unfused_add_then_norm(
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261003)
    eps = 1e-6
    x = torch.randn((5, hidden_size), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    summed = x + residual

    with torch.inference_mode():
        unfused = norm.rms_norm(summed, weight, eps, zero_centered=True)
        output, residual_sum = norm.add_rms_norm_(
            x, residual, weight, eps, zero_centered=True, normalize="stored_sum"
        )
    torch.cuda.synchronize()

    assert output is x and residual_sum is residual
    torch.testing.assert_close(residual, summed, rtol=0, atol=0)
    torch.testing.assert_close(x, unfused, rtol=0, atol=0)
    oracle.check_rms_norm(x, summed, weight, eps, zero_centered=True)


@pytest.mark.parametrize("fp32_residual", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("hidden_size", (128, 2048, 4096, 4097, 5120))
def test_zero_centered_add_rms_norm_normalizes_the_fp32_sum(
    fp32_residual: bool,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261012)
    eps = 1e-6
    residual_dtype = torch.float32 if fp32_residual else dtype
    x = torch.randn((37, hidden_size), device="cuda", dtype=dtype)
    residual = torch.randn((37, hidden_size), device="cuda", dtype=residual_dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=weight_dtype)
    # The norm reads the FP32 sum, which the residual stores rounded once to its dtype.
    original_x, original_residual = x.clone(), residual.clone()
    fp32_sum = residual.float() + x.float()

    with torch.inference_mode():
        unfused = norm.rms_norm(fp32_sum, weight, eps, zero_centered=True, out_dtype=dtype)
        output, residual_sum = norm.add_rms_norm_(x, residual, weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert output is x and residual_sum is residual
    assert x.dtype == dtype and residual.dtype == residual_dtype
    torch.testing.assert_close(residual, fp32_sum.to(residual_dtype), rtol=0, atol=0)
    if norm._DEFAULT_ADD.provider == "triton":
        # Same Triton reduction as the unfused norm; CuTe rows reduce in another order.
        torch.testing.assert_close(x, unfused, rtol=0, atol=0)
    oracle.check_add_rms_norm(x, original_x, original_residual, weight, eps, zero_centered=True)


def test_fp32_residual_accumulates_less_error_than_a_rounded_one() -> None:
    torch.manual_seed(20261014)
    eps = 1e-6
    num_layers, num_tokens, hidden_size = 64, 32, 4096
    embedding = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    updates = torch.randn(
        (num_layers, num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    # FP64 holds these sums of BF16 values exactly.
    exact = embedding.double() + updates.double().sum(dim=0)
    expected = oracle.rms_norm(exact, weight, eps, zero_centered=True)

    residual_errors, output_errors = {}, {}
    for residual_dtype in (torch.bfloat16, torch.float32):
        residual = embedding.to(residual_dtype, copy=True)
        with torch.inference_mode():
            for update in updates:
                output, _ = norm.add_rms_norm_(
                    update.clone(), residual, weight, eps, zero_centered=True
                )
        torch.cuda.synchronize()
        residual_errors[residual_dtype] = errors(residual, exact).mean
        output_errors[residual_dtype] = errors(output, expected).mean

    assert residual_errors[torch.float32] < residual_errors[torch.bfloat16]
    assert output_errors[torch.float32] < output_errors[torch.bfloat16]


@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (256, 4096, 4097, 5120))
def test_zero_centered_rms_norm_rounds_an_fp32_input_once_to_out_dtype(
    out_dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261017)
    eps = 1e-6
    x_data = torch.randn((3, 7, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn((3, 7, hidden_size), device="cuda", dtype=out_dtype)

    baseline_x = x_data.clone().requires_grad_()
    oracle.rms_norm(
        baseline_x, weight, eps, zero_centered=True, out_dtype=out_dtype, upcast=False
    ).backward(upstream)

    reference_x = x_data.double().requires_grad_()
    oracle.rms_norm(reference_x, weight, eps, zero_centered=True).backward(upstream.double())

    x = x_data.clone().requires_grad_()
    output = training.rms_norm(x, weight, eps, zero_centered=True, out_dtype=out_dtype)
    output.backward(upstream)
    with torch.inference_mode():
        inference_output = norm.rms_norm(
            x_data, weight, eps, zero_centered=True, out_dtype=out_dtype
        )
        fp32_output = norm.rms_norm(x_data, weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert output.dtype == out_dtype and x.grad.dtype == torch.float32
    torch.testing.assert_close(output, inference_output, rtol=0, atol=0)
    torch.testing.assert_close(output, fp32_output.to(out_dtype), rtol=0, atol=0)
    oracle.check_rms_norm(output, x_data, weight, eps, zero_centered=True)
    assert_accurate(
        x.grad, reference_x.grad, baseline_x.grad.to(x.grad.dtype), family="norm", name="dx"
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (256, 4096, 4097, 5120))
def test_zero_centered_rms_norm_widens_to_an_fp32_out_dtype_unrounded(
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261020)
    eps = 1e-6
    x_data = torch.randn((3, 7, hidden_size), device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn((3, 7, hidden_size), device="cuda", dtype=torch.float32)

    baseline_x = x_data.clone().requires_grad_()
    baseline = oracle.rms_norm(
        baseline_x, weight, eps, zero_centered=True, out_dtype=torch.float32, upcast=False
    )
    baseline.backward(upstream)

    reference_x = x_data.double().requires_grad_()
    reference = oracle.rms_norm(reference_x, weight, eps, zero_centered=True)
    reference.backward(upstream.double())

    x = x_data.clone().requires_grad_()
    output = training.rms_norm(x, weight, eps, zero_centered=True, out_dtype=torch.float32)
    output.backward(upstream)
    with torch.inference_mode():
        inference_output = norm.rms_norm(
            x_data, weight, eps, zero_centered=True, out_dtype=torch.float32
        )
        # x widens exactly, so its FP32 copy normalizes to the same FP32 result.
        widened_input = norm.rms_norm(x_data.float(), weight, eps, zero_centered=True)
        default_output = norm.rms_norm(x_data, weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert output.dtype == torch.float32 and x.grad.dtype == dtype
    torch.testing.assert_close(output, inference_output, rtol=0, atol=0)
    torch.testing.assert_close(output, widened_input, rtol=0, atol=0)
    # The one rounding is to FP32: rounding the result on to x's dtype gives the default
    # output, and the FP32 result keeps bits that output lacks.
    torch.testing.assert_close(output.to(dtype), default_output, rtol=0, atol=0)
    assert (output != default_output.float()).any().item()
    oracle.check_rms_norm(output, x_data, weight, eps, zero_centered=True)
    assert_accurate(
        x.grad, reference_x.grad, baseline_x.grad.to(x.grad.dtype), family="norm", name="dx"
    )


@pytest.mark.parametrize(
    "m,hidden,dtype,zero",
    ci_cases(
        (0, 1, 2, 3, 4, 16, 17, 64, 256, 300, 512, 1024, 2048, 4096, 8192),
        (8, 136, 4096),
        (torch.bfloat16, torch.float16),
        (False, True),
        core={
            (m, 4096, torch.bfloat16, z)
            for m in (0, 1, 3, 16, 64, 256, 1024)
            for z in (False, True)
        }
        | {(17, 136, torch.float16, False), (300, 4096, torch.float16, True)},
    ),
)
def test_cute_add_norm_accuracy(m, hidden, dtype, zero):
    _check_cute_norm(m, hidden, dtype, zero)


@pytest.mark.parametrize("config", tune_configs())
@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.exhaustive
def test_cute_add_norm_configuration_numerics(config, zero):
    _check_cute_norm(17, 4096, torch.bfloat16, zero, config)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_gemma_add_norm_fp32_weight(dtype):
    _check_cute_norm(17, 4096, dtype, True, Config(256, 2), fp32_weight=True)


@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_add_norm_cancellation_accuracy(zero, dtype):
    x = torch.randn((3, 4096), device="cuda", dtype=dtype)
    residual = -x
    weight = torch.randn(4096, device="cuda", dtype=dtype)
    cx, cr = x.clone(), residual.clone()
    norm.AddRMSNorm(provider="cute")(cx, cr, weight, 1e-6, zero_centered=zero)
    oracle.check_add_rms_norm(cx, x, residual, weight, zero_centered=zero)
    _, stored = oracle.add_rms_norm(x, residual, weight, zero_centered=zero, upcast=False)
    assert_accurate(cr, x.double() + residual.double(), stored, family="norm", name="stored_sum")


@pytest.mark.parametrize("kind,hidden,dtype,zero,static,m,entry", _shipped_norm_cases())
def test_shipped_norm_configurations(kind, hidden, dtype, zero, static, m, entry):
    # Every shipped provider (CuTe, FlashInfer, Triton) takes the FP64 gate.
    _check_norm_provider(entry.provider, m, hidden, getattr(torch, dtype), zero, entry.config)
    x = torch.randn((m, hidden), device="cuda", dtype=getattr(torch, dtype))
    residual, weight = torch.randn_like(x), torch.randn(hidden, device=x.device, dtype=x.dtype)
    expected_x, expected_r = x.clone(), residual.clone()
    norm.AddRMSNorm(provider=entry.provider, config=entry.config, static_weight=static)(
        expected_x, expected_r, weight, 1e-6, zero_centered=zero
    )
    if norm._hardware(x.device)[0] == kind:
        op = norm.AddRMSNorm(provider="auto", static_weight=static)
        op(x, residual, weight, 1e-6, zero_centered=zero)
        assert op.provider == entry.provider
        assert torch.equal(x.view(torch.int16), expected_x.view(torch.int16))
        assert torch.equal(residual.view(torch.int16), expected_r.view(torch.int16))


@pytest.mark.parametrize(
    "provider,m,hidden,dtype,zero",
    ci_cases(
        ("auto", "cute", "triton"),
        (1, 3, 16, 64, 256, 1024),
        (2048, 4096, 4104, 2560),
        (torch.bfloat16, torch.float16),
        (False, True),
        core={
            (p, m, h, d, False)
            for p in ("auto", "cute", "triton")
            for m in (1, 3, 16, 64, 256, 1024)
            for h in (2048, 4096)
            for d in (torch.bfloat16, torch.float16)
        }
        | {
            ("cute", m, h, d, True)
            for m in (3, 64)
            for h in (2048, 4096)
            for d in (torch.bfloat16, torch.float16)
        }
        | {
            (p, 3, h, d, False)
            for p, h in (("triton", 4104), ("cute", 2560))
            for d in (torch.bfloat16, torch.float16)
        },
    ),
)
def test_fp32_residual_weight_modes_accuracy_and_stores(provider, m, hidden, dtype, zero):
    _check_fp32_residual(provider, m, hidden, dtype, zero)


@pytest.mark.exhaustive
@pytest.mark.parametrize("config", tune_configs())
@pytest.mark.parametrize("zero", (False, True))
def test_fp32_residual_cute_configurations(config, zero):
    _check_fp32_residual("cute", 3, 2048, torch.bfloat16, zero, config)


_H = (64, 128, 256, 1000, 1024, 2048, 2560, 2880, 4096, 5120, 8192, 16384)
_ADD_H = (64, 1000, 1024, 2048, 2560, 2880, 4096, 5120, 8192, 16384)
_M = (0, 1, 2, 3, 4, 5, 16, 64, 65, 255, 256, 4096, 16384)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_EPS = 1e-6
_STANDALONE = [
    (dtype, mixed, out, zero, layout, inplace)
    for dtype, mixed, out, zero, layout, inplace in product(
        _DTYPES, (False, True), _DTYPES, (False, True), ("dense", "rows", "heads"), (False, True)
    )
    if (not inplace or out == dtype) and (dtype != torch.float32 or not mixed)
]
_ADD = list(
    product(
        _DTYPES,
        (False, True),
        (False, True),
        ("fp32_sum", "stored_sum"),
        ("store", "no-store", "sum_out"),
        ("dense", "rows"),
    )
)


@pytest.fixture
def range_auto_policy(monkeypatch):
    monkeypatch.setenv(norm.VARIABLE, "auto")
    monkeypatch.delenv("CTI_NORM_CONFIG", raising=False)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)


def _cases(widths):
    offset = 0
    core = {(64, 0), (1000, 3), (4096, 1), (5120, 5)}
    for hidden, m in product(widths, _M):
        count = 12 if m <= 256 else 3
        yield pytest.param(
            hidden,
            m,
            offset,
            count,
            id=f"H{hidden}-M{m}",
            marks=() if (hidden, m) in core else pytest.mark.exhaustive,
        )
        offset += count


def _input(m, hidden, dtype, layout):
    if layout == "heads":
        # Interleaved components of a merged QKV: head stride 3H+1, row stride
        # 3*(3H+1)+5. Three heads per token also exercises M=1 with multiple groups.
        return torch.randn(m, 3 * (3 * hidden + 1) + 5, device="cuda", dtype=dtype).as_strided(
            (m, 3, hidden), (3 * (3 * hidden + 1) + 5, 3 * hidden + 1, 1)
        )
    return torch.randn(m, hidden + (1 if layout == "rows" else 0), device="cuda", dtype=dtype)[
        :, :hidden
    ]


def _bits(a, b):
    integer = torch.int32 if a.dtype == torch.float32 else torch.int16
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer))


def _assert_standalone_provider(name, x, weight, zero, out):
    actual = norm.last_standalone_provider[name]
    if not x.numel():
        assert actual == "empty"
        return
    compatible = (
        not zero and x.dtype != torch.float32 and weight.dtype == x.dtype and out == x.dtype
    )
    direct = norm._ordinary_rms_direct(x) and weight.data_ptr() % 16 == 0
    expected = "flashinfer" if compatible and direct and _flashinfer_version.allowed() else "triton"
    assert actual == expected
    assert bool(getattr(actual, "reason", None)) == (compatible and expected == "triton")


def _assert_add_provider(op, x, residual, weight, zero, normalize, store, summed):
    if not x.numel():
        assert op.provider == "empty"
        return
    kind, capability = norm._hardware(x.device)
    admitted = norm._admitted(x, residual, weight, zero, store, normalize)
    # The runtime admits sum_out through its [M, H] view: one row has no row pitch.
    rows = None if summed is None else summed.view(-1, summed.shape[-1])
    admitted = admitted and (
        rows is None or (rows.data_ptr() % 16 == 0 and rows.stride(0) % 8 == 0)
    )
    if capability not in ((12, 0), (12, 1)) or not admitted:
        assert op.provider == "triton" and not getattr(op.provider, "reason", None)
        return
    if not cute.available():
        assert op.provider == "triton" and "nvidia-cutlass-dsl" in op.provider.reason
        return
    entry = entry_for(
        kind,
        x.shape[-1],
        str(x.dtype).removeprefix("torch."),
        x.numel() // x.shape[-1],
        zero,
        residual_dtype=str(residual.dtype).removeprefix("torch."),
    )
    expected = entry.provider if entry else "cute"
    if expected == "flashinfer" and (summed is not None or not store or weight.dtype != x.dtype):
        expected = "cute"
    if expected == "flashinfer" and not _flashinfer_version.allowed():
        expected = "cute"
        assert "FlashInfer" in op.provider.reason
    # A compile failure is observable, but fails device qualification here.
    assert op.provider == expected
    if expected == "cute":
        assert op.dispatched_config == (entry.config if entry else Config())


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("hidden,m,offset,count", _cases(_H))
def test_standalone_range(hidden, m, offset, count):
    for i in range(offset, offset + count):
        _standalone_case(hidden, m, _STANDALONE[i % len(_STANDALONE)])


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("m", [1, 65])
@pytest.mark.parametrize("hidden", [128, 1000, 5120])
def test_standalone_routes(hidden, m):
    for variant in _STANDALONE:
        _standalone_case(hidden, m, variant)


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("m", [1, 65])
@pytest.mark.parametrize("hidden", [1000, 4096, 5120])
def test_add_routes(hidden, m):
    op = norm.AddRMSNorm()
    for i, variant in enumerate(_ADD):
        _add_case(op, hidden, m, i, variant)


def _standalone_case(hidden, m, variant):
    dtype, mixed, out, zero, layout, inplace = variant
    x = _input(m, hidden, dtype, layout)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32 if mixed else dtype)
    saved = x.clone()
    name = "rms_norm_" if inplace else "rms_norm"
    result = (
        norm.rms_norm_(x, weight, _EPS, zero_centered=zero)
        if inplace
        else norm.rms_norm(x, weight, _EPS, zero_centered=zero, out_dtype=out)
    )
    assert result.shape == x.shape and result.dtype == out
    _assert_standalone_provider(name, x, weight, zero, out)
    oracle.check_rms_norm(result, saved, weight, _EPS, zero_centered=zero)
    if inplace:
        assert result is x
    else:
        assert torch.equal(saved, x)
    del result, saved, x


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("hidden,m,offset,count", _cases(_ADD_H))
def test_add_range(hidden, m, offset, count):
    op = norm.AddRMSNorm()
    for i in range(offset, offset + count):
        _add_case(op, hidden, m, i, _ADD[i % len(_ADD)])


def _add_case(op, hidden, m, i, variant):
    dtype, zero, mixed, normalize, mode, layout = variant
    x = _input(m, hidden, dtype, layout)
    residual = _input(m, hidden, torch.float32 if mixed else dtype, layout)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    saved_x, saved_r = x.clone(), residual.clone()
    summed = torch.empty_like(residual) if mode == "sum_out" else None
    store = mode == "store"
    # Exercise both the shared public call-site and dedicated instances.
    call = norm.add_rms_norm_ if i % 2 else op
    call(
        x,
        residual,
        weight,
        _EPS,
        zero_centered=zero,
        normalize=normalize,
        store_residual=store,
        sum_out=summed,
    )
    used = norm._DEFAULT_ADD if i % 2 else op
    _assert_add_provider(used, x, residual, weight, zero, normalize, store, summed)
    for start in range(0, m, max(1, (1 << 22) // hidden)):
        sl = slice(start, start + max(1, (1 << 22) // hidden))
        stored = (saved_x[sl].float() + saved_r[sl].float()).to(residual.dtype)
        oracle.check_add_rms_norm(
            x[sl], saved_x[sl], saved_r[sl], weight, _EPS, zero_centered=zero, normalize=normalize
        )
        _bits(residual[sl], stored if store else saved_r[sl])
        if summed is not None:
            _bits(summed[sl], stored)
    # Same contiguous layout only: Triton keys its reduction vector on alignment.
    if normalize == "stored_sum" and m and layout == "dense":
        first = (saved_x[:3].float() + saved_r[:3].float()).to(residual.dtype)
        residual_only = norm.residual_rms_norm(
            first, weight, _EPS, zero_centered=zero, out_dtype=dtype
        )
        _bits(residual_only, x[:3])
    del x, residual, saved_x, saved_r, summed


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("layout", ["hidden", "prefix"])
def test_pack_and_residual_range(zero, layout, monkeypatch):
    monkeypatch.setenv(norm.VARIABLE, "triton")
    base = torch.randn(2, 3, 4, 2000, device="cuda", dtype=torch.bfloat16)
    x = base[..., ::2] if layout == "hidden" else base[..., :1000].transpose(0, 1)
    residual = torch.randn_like(base, dtype=torch.float32)
    residual = residual[..., ::2] if layout == "hidden" else residual[..., :1000].transpose(0, 1)
    weight = torch.randn(1000, device="cuda", dtype=torch.float32)
    for name in ("rms_norm", "rms_norm_"):
        saved = x.clone()
        actual = getattr(norm, name)(x, weight, _EPS, zero_centered=zero)
        assert norm.last_standalone_provider[name].reason == "pack"
        oracle.check_rms_norm(actual, saved, weight, _EPS, zero_centered=zero)
    saved = residual.clone()
    actual = norm.residual_rms_norm(residual, weight, _EPS, zero_centered=zero, out_dtype=x.dtype)
    assert norm.last_standalone_provider["residual_rms_norm"].reason == "pack"
    oracle.check_rms_norm(actual, saved, weight, _EPS, zero_centered=zero)
    op = norm.AddRMSNorm(provider="triton")
    stored = (x.float() + residual.float()).to(residual.dtype)
    op(x, residual, weight, _EPS, zero_centered=zero, normalize="stored_sum")
    assert op.provider.reason == "pack"
    _bits(residual, stored)
    oracle.check_rms_norm(x, stored, weight, _EPS, zero_centered=zero)


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("hidden", [1, 7, 1000, 5120, 16384, 65544, 131080], ids=lambda h: f"H{h}")
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_stored_sum_training_and_residual_bits(hidden, zero, dtype, monkeypatch):
    monkeypatch.setenv(norm.VARIABLE, "triton")
    x = torch.randn(3, hidden, device="cuda", dtype=dtype)
    residual = torch.randn(3, hidden, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
    expected, summed = training.add_rms_norm(
        x, residual, weight, _EPS, zero_centered=zero, save="inputs"
    )
    op = norm.AddRMSNorm(provider="triton")
    op(x, residual, weight, _EPS, zero_centered=zero, normalize="stored_sum")
    assert op.provider == "triton"
    _bits(x, expected)
    _bits(residual, summed)
    actual = norm.residual_rms_norm(residual, weight, _EPS, zero_centered=zero, out_dtype=x.dtype)
    _bits(actual, x)
    oracle.check_rms_norm(x, residual, weight, _EPS, zero_centered=zero)
    # Odd-pitch residual rows and unaligned weights are native on Triton. Triton
    # picks its per-thread vector (and so the FP32 reduction order) from pointer
    # alignment, so these layouts meet the FP64 gate rather than the bits above.
    strided = torch.empty(3, hidden + 1, device="cuda", dtype=residual.dtype)[:, :hidden]
    strided.copy_(residual)
    unaligned_weight = torch.empty(hidden + 1, device="cuda", dtype=weight.dtype)[1:]
    unaligned_weight.copy_(weight)
    actual = norm.residual_rms_norm(
        strided, unaligned_weight, _EPS, zero_centered=zero, out_dtype=x.dtype
    )
    assert not getattr(norm.last_standalone_provider["residual_rms_norm"], "reason", None)
    oracle.check_rms_norm(actual, residual, weight, _EPS, zero_centered=zero)
    standalone = norm.rms_norm(residual, weight, _EPS, zero_centered=zero, out_dtype=x.dtype)
    _bits(standalone, x)


@pytest.mark.usefixtures("range_auto_policy")
@pytest.mark.parametrize("family", ["rms_norm", "rms_norm_", "add_rms_norm_"])
def test_norm_over_2_31(family, monkeypatch):
    monkeypatch.setenv(norm.VARIABLE, "triton")
    m, hidden = 131073, 16384
    bytes_per_tensor = m * hidden * 2
    free, _ = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if free < 3 * bytes_per_tensor + (1 << 30):
        pytest.skip("requires room for two >2^31-element BF16 tensors and allocator headroom")
    x = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    weight = torch.randn(hidden, device="cuda", dtype=x.dtype)
    indices = torch.tensor([0, m // 2, m - 1], device="cuda")
    saved = x[indices].clone()
    assert x.numel() > 2**31
    if family == "add_rms_norm_":
        residual = torch.empty_like(x).fill_(0.25)
        stored = (x[indices].float() + residual[indices].float()).to(x.dtype)
        norm.add_rms_norm_(x, residual, weight, _EPS)
        assert norm._DEFAULT_ADD.provider == "triton"
        _bits(residual[indices], stored)
        oracle.check_add_rms_norm(x[indices], saved, torch.full_like(saved, 0.25), weight, _EPS)
    else:
        result = getattr(norm, family)(x, weight, _EPS)
        assert norm.last_standalone_provider[family] == "triton"
        oracle.check_rms_norm(result[indices], saved, weight, _EPS)


@pytest.mark.usefixtures("range_auto_policy")
def test_cached_flashinfer_plan_rechecks_alignment():
    # A plan warmed on aligned row-strided rows must not launch FlashInfer on an
    # identically shaped and strided view that is offset by one element.
    if not _flashinfer_version.allowed():
        pytest.skip("FlashInfer unavailable")
    op = norm.AddRMSNorm(provider="flashinfer")
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    def views(offset):
        storage = torch.randn(2 * 8192 + 8, device="cuda", dtype=torch.bfloat16)
        return storage[offset:].as_strided((2, 4096), (8192, 1))

    x, residual = views(0), views(0)
    op(x, residual, weight, _EPS)
    assert op.provider == "flashinfer"
    x, residual = views(1), views(1)
    with pytest.raises(RuntimeError, match="alignment"):
        op(x, residual, weight, _EPS)
    auto = norm.AddRMSNorm()
    x, residual = views(0), views(0)
    auto(x, residual, weight, _EPS)
    x, residual = views(1), views(1)
    saved_x, saved_r = x.clone(), residual.clone()
    auto(x, residual, weight, _EPS)
    assert auto.provider != "flashinfer"
    oracle.check_add_rms_norm(x, saved_x, saved_r, weight, _EPS)
