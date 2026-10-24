# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU norm contracts for mutation, aliasing, routing, streams, graphs, and exact behavior."""

from __future__ import annotations

import pytest
import torch

from ctikernels import _flashinfer_version
from ctikernels._norm_act_reroute import Provider, RerouteError
from ctikernels.norm import _zero_centered_rms_norm as zero_centered_kernels
from ctikernels.norm import inference as inference_norm
from ctikernels.norm import training as training_norm
from ctikernels.norm._dispatch import TABLE, Config, entry_for

from ..gpu._exhaustive import ci_cases
from ..support.accuracy import assert_accurate
from . import oracle

pytestmark = pytest.mark.gpu

_SUM_PROVIDERS = (
    ("cute", False, False),
    ("cute", True, False),
    ("cute", False, True),
    ("cute", True, True),
    ("triton", False, True),
    ("auto", False, True),
    ("triton", True, False),
    ("triton", True, True),
    ("auto", False, False),
    ("auto", True, False),
    ("auto", True, True),
)


def _assert_norm_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


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
        op = inference_norm.AddRMSNorm(
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


def test_inference_rms_norm_mutation_contract() -> None:
    torch.manual_seed(20260724)
    eps = 1e-6
    x = torch.randn((19, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    original = x.clone()

    with torch.inference_mode():
        output = inference_norm.rms_norm(x, weight, eps)
        in_place_output = inference_norm.rms_norm_(x, weight, eps)
    torch.cuda.synchronize()

    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert in_place_output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    oracle.check_rms_norm(output, original, weight, eps)
    oracle.check_rms_norm(in_place_output, original, weight, eps)


def test_inference_add_rms_norm_mutation_contract() -> None:
    torch.manual_seed(20260725)
    eps = 1e-6
    x = torch.randn((19, 128), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    original_x = x.clone()
    original_residual = residual.clone()
    expected_residual = original_x + original_residual

    with torch.inference_mode():
        output, residual_sum = inference_norm.add_rms_norm_(x, residual, weight, eps)
    torch.cuda.synchronize()

    assert output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert residual_sum.untyped_storage().data_ptr() == residual.untyped_storage().data_ptr()
    oracle.check_add_rms_norm(output, original_x, original_residual, weight, eps)
    torch.testing.assert_close(residual_sum, expected_residual, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ("rms_norm", "add_rms_norm"))
def test_inference_mutation_rejects_requires_grad_storage_before_writing(
    operation: str,
) -> None:
    torch.manual_seed(20260726)
    eps = 1e-6
    x = torch.randn((11, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    original_x = x.detach().clone()
    original_residual = residual.detach().clone()

    with torch.no_grad(), pytest.raises(RuntimeError, match="cannot modify"):
        if operation == "rms_norm":
            inference_norm.rms_norm_(x, weight, eps)
        else:
            inference_norm.add_rms_norm_(x, residual, weight, eps)

    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


def test_inference_rms_norm_writes_through_qkv_view() -> None:
    torch.manual_seed(20260727)
    eps = 1e-6
    num_tokens = 13
    num_heads = 4
    head_dim = 128
    qkv = torch.randn(
        (num_tokens, 3 * num_heads * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = qkv.split(num_heads * head_dim, dim=-1)
    q = q.view(num_tokens, num_heads, head_dim)
    original_q = q.clone()
    original_k = k.clone()
    original_v = v.clone()
    weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        inference_norm.rms_norm_(q, weight, eps)
    torch.cuda.synchronize()

    oracle.check_rms_norm(q, original_q, weight, eps)
    torch.testing.assert_close(k, original_k, rtol=0, atol=0)
    torch.testing.assert_close(v, original_v, rtol=0, atol=0)


@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
def test_zero_centered_rms_norm_writes_through_qk_heads_of_merged_projection(
    weight_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20261002)
    eps = 1e-6
    num_tokens, num_q_heads, num_kv_heads, head_dim = 13, 8, 2, 256
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    qkv = torch.randn(
        (num_tokens, q_size + 2 * kv_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(num_tokens, num_q_heads, head_dim)
    k = k.view(num_tokens, num_kv_heads, head_dim)
    q_weight = torch.randn(head_dim, device="cuda", dtype=weight_dtype)
    k_weight = torch.randn(head_dim, device="cuda", dtype=weight_dtype)
    original_q, original_k, original_v = q.clone(), k.clone(), v.clone()

    with torch.inference_mode():
        expected_q = inference_norm.rms_norm(original_q, q_weight, eps, zero_centered=True)
        expected_k = inference_norm.rms_norm(original_k, k_weight, eps, zero_centered=True)
        inference_norm.rms_norm_(q, q_weight, eps, zero_centered=True)
        inference_norm.rms_norm_(k, k_weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert not q.is_contiguous() and not k.is_contiguous()
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v, original_v, rtol=0, atol=0)
    oracle.check_rms_norm(q, original_q, q_weight, eps, zero_centered=True)
    oracle.check_rms_norm(k, original_k, k_weight, eps, zero_centered=True)


@pytest.mark.parametrize("residual_dtype", (torch.bfloat16, torch.float32))
def test_zero_centered_add_rms_norm_row_strided_residual(residual_dtype: torch.dtype) -> None:
    """Inference takes residual rows at their own stride; training takes contiguous rows only."""

    torch.manual_seed(20261005)
    x = torch.randn((5, 3, 128), device="cuda", dtype=torch.bfloat16)
    residual_storage = torch.randn((5, 3, 256), device="cuda", dtype=residual_dtype)
    residual = residual_storage[..., 128:]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    original_x, original_storage = x.clone(), residual_storage.clone()

    with pytest.raises(ValueError, match="contiguous"):
        training_norm.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual_storage, original_storage, rtol=0, atol=0)

    expected_x, expected_residual = x.clone(), residual.contiguous()
    inference_norm.add_rms_norm_(expected_x, expected_residual, weight, 1e-6, zero_centered=True)
    with torch.inference_mode():
        inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True)
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(
        residual_storage[..., :128], original_storage[..., :128], rtol=0, atol=0
    )


@pytest.mark.parametrize("residual_dtype", (torch.bfloat16, torch.float32))
def test_zero_centered_add_rms_norm_stores_the_sum_in_the_residual_dtype(
    residual_dtype: torch.dtype,
) -> None:
    x = torch.full((3, 256), 0.5, device="cuda", dtype=torch.bfloat16)
    residual = torch.full((3, 256), 256.0, device="cuda", dtype=residual_dtype)
    weight = torch.zeros(256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True)
    torch.cuda.synchronize()

    # BF16 values near 256 are 2 apart, so 256.5 rounds back to 256; FP32 holds it.
    expected = 256.5 if residual_dtype == torch.float32 else 256.0
    assert (residual == expected).all().item()


def test_zero_centered_norms_replay_in_cuda_graph() -> None:
    torch.manual_seed(20261004)
    eps = 1e-6
    num_tokens, hidden_size, num_heads, head_dim = 6, 5120, 4, 256
    hidden = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros_like(hidden)
    qkv = torch.zeros((num_tokens, 3 * num_heads * head_dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    head_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)

    def decoder_norms(hidden, residual, qkv):
        inference_norm.add_rms_norm_(hidden, residual, weight, eps, zero_centered=True)
        q, k, _ = qkv.view(num_tokens, 3 * num_heads, head_dim).split(num_heads, dim=1)
        inference_norm.rms_norm_(q, head_weight, eps, zero_centered=True)
        inference_norm.rms_norm_(k, head_weight, eps, zero_centered=True)

    with torch.inference_mode():
        decoder_norms(hidden, residual, qkv)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        decoder_norms(hidden, residual, qkv)

    for _ in range(3):
        fresh = [torch.randn_like(tensor) for tensor in (hidden, residual, qkv)]
        expected = [tensor.clone() for tensor in fresh]
        with torch.inference_mode():
            decoder_norms(*expected)
        for static, values in zip((hidden, residual, qkv), fresh, strict=True):
            static.copy_(values)
        graph.replay()
        torch.cuda.synchronize()

        for static, values in zip((hidden, residual, qkv), expected, strict=True):
            torch.testing.assert_close(static, values, rtol=0, atol=0)


def test_zero_centered_fp32_residual_replays_in_cuda_graph() -> None:
    torch.manual_seed(20261015)
    eps = 1e-6
    num_tokens, hidden_size = 6, 5120
    hidden = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    def norms(hidden, residual):
        inference_norm.add_rms_norm_(hidden, residual, weight, eps, zero_centered=True)
        # The next layer's input norm reads the FP32 residual into the activation dtype.
        return inference_norm.rms_norm(
            residual, weight, eps, zero_centered=True, out_dtype=hidden.dtype
        )

    with torch.inference_mode():
        norms(hidden, residual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        captured = norms(hidden, residual)

    for _ in range(3):
        fresh = [torch.randn_like(tensor) for tensor in (hidden, residual)]
        expected = [tensor.clone() for tensor in fresh]
        with torch.inference_mode():
            expected.append(norms(*expected))
        for static, values in zip((hidden, residual), fresh, strict=True):
            static.copy_(values)
        graph.replay()
        torch.cuda.synchronize()

        for static, values in zip((hidden, residual, captured), expected, strict=True):
            torch.testing.assert_close(static, values, rtol=0, atol=0)


def test_zero_centered_rms_norm_accepts_empty_rows() -> None:
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    weight = torch.zeros(128, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        output = inference_norm.rms_norm(x, weight, 1e-6, zero_centered=True)
        in_place = inference_norm.rms_norm_(x, weight, 1e-6, zero_centered=True)
        fused = inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True)

    assert output.shape == x.shape
    assert in_place is x
    assert fused[0] is x and fused[1] is residual


def test_zero_centered_rms_norm_rejects_invalid_inputs() -> None:
    x = torch.randn((4, 2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="CUDA"):
        inference_norm.rms_norm(x.cpu(), weight.cpu(), 1e-6, zero_centered=True)
    with pytest.raises(TypeError, match="FP16, BF16 or FP32"):
        inference_norm.rms_norm(x.to(torch.int32), weight, 1e-6, zero_centered=True)
    with pytest.raises(ValueError, match="weight"):
        inference_norm.rms_norm(x, weight[:127], 1e-6, zero_centered=True)
    packed = torch.randn((2, 3, 4, 128), device="cuda").transpose(0, 1)
    inference_norm.rms_norm(packed, weight.float(), 1e-6, zero_centered=True)
    assert inference_norm.last_standalone_provider["rms_norm"].reason == "pack"
    hidden_strided = torch.randn((4, 2, 256), device="cuda", dtype=torch.bfloat16)[..., ::2]
    inference_norm.rms_norm_(hidden_strided, weight, 1e-6, zero_centered=True)
    assert inference_norm.last_standalone_provider["rms_norm_"].reason == "pack"
    with pytest.raises(ValueError, match="distinct"):
        inference_norm.add_rms_norm_(x, x, weight, 1e-6, zero_centered=True)


def test_zero_centered_mixed_dtypes_accept_empty_rows() -> None:
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty((0, 128), device="cuda", dtype=torch.float32)
    weight = torch.zeros(128, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        output = inference_norm.rms_norm(
            residual, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16
        )
        fused = inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True)
        residual_output = inference_norm.residual_rms_norm(
            residual, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16
        )
    normalized, summed = training_norm.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)
    residual_leaf = residual.clone().requires_grad_()
    residual_normalized, passed = training_norm.residual_rms_norm(
        residual_leaf, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16
    )
    (residual_normalized.float().sum() + passed.sum()).backward()

    assert output.shape == x.shape and output.dtype == torch.bfloat16
    assert fused[0] is x and fused[1] is residual
    assert normalized.dtype == torch.bfloat16 and summed.dtype == torch.float32
    assert residual_output.shape == x.shape and residual_output.dtype == torch.bfloat16
    assert residual_leaf.grad.shape == residual_leaf.shape
    assert residual_leaf.grad.dtype == torch.float32


def test_norms_reject_unsupported_dtype_mixes_before_writing() -> None:
    torch.manual_seed(20261016)
    x = torch.randn((4, 2, 128), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((4, 2, 128), device="cuda", dtype=torch.float32)
    weight = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    original_x, original_residual = x.clone(), residual.clone()

    zero_centered_cases = (
        (x, x.half(), TypeError, "input dtype or FP32"),
        (x.float(), x.clone(), TypeError, "input dtype or FP32"),
        (residual.view(torch.bfloat16)[..., :128], residual, ValueError, "overlap"),
        (x, residual[:, :1], ValueError, "shape and device"),
    )
    for case_x, case_residual, error, message in zero_centered_cases:
        with torch.inference_mode(), pytest.raises(error, match=message):
            inference_norm.add_rms_norm_(case_x, case_residual, weight, 1e-6, zero_centered=True)
        with pytest.raises(error, match=message):
            training_norm.add_rms_norm(case_x, case_residual, weight, 1e-6, zero_centered=True)
    with torch.inference_mode(), pytest.raises(TypeError, match="out_dtype"):
        inference_norm.rms_norm(x, weight, 1e-6, zero_centered=True, out_dtype=torch.int32)
    with pytest.raises(TypeError, match="out_dtype"):
        training_norm.rms_norm(x, weight, 1e-6, zero_centered=True, out_dtype=torch.int32)
    # Forced FlashInfer still rejects mixed residual dtypes.
    with torch.inference_mode(), pytest.raises(RuntimeError, match="forced flashinfer requires"):
        inference_norm.AddRMSNorm(provider="flashinfer")(x, residual, weight, 1e-6)
    with torch.inference_mode():
        assert (
            inference_norm.rms_norm(x, weight, 1e-6, out_dtype=torch.float32).dtype == torch.float32
        )
    for train_weight in (False, True):
        direct_weight = weight.clone().requires_grad_(train_weight)
        with pytest.raises(TypeError, match="zero_centered=True"):
            training_norm.rms_norm(x, direct_weight, 1e-6, out_dtype=torch.float32)
        if train_weight:
            with pytest.raises(TypeError, match="zero_centered=True"):
                training_norm.add_rms_norm(x, residual, direct_weight, 1e-6)
    torch.cuda.synchronize()

    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


@pytest.mark.parametrize("fp32_residual", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (2048, 4096, 5120))
@pytest.mark.parametrize("rows", (1, 7, 512))
def test_residual_norms_keep_the_forward_bits_of_the_grouped_kernel(
    fp32_residual: bool,
    dtype: torch.dtype,
    hidden_size: int,
    rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracles add in torch, which rounds the FP32 sum once as the kernel does, and
    normalize with the grouped kernel (the one q_norm and k_norm run) into x's dtype. Both
    kernels normalize a row with one FP32 formula and one row reduction, so inference, with
    or without storing the sum, gives the bits of the FP32 sum's norm, and training and every
    norm of the stored sum those of the stored sum's. An FP32 residual holds the FP32 sum, so
    only a residual in x's dtype tells the two apart."""
    # Bitwise parity with training is the Triton residual-stream kernel's contract; the
    # auto table may pick CuTe or FlashInfer for prefill rows (no worse vs FP64 instead).
    monkeypatch.setenv("CTI_NORM_PROVIDER", "triton")

    torch.manual_seed(20261022)
    eps = 1e-6
    residual_dtype = torch.float32 if fp32_residual else dtype
    x = torch.randn((rows, hidden_size), device="cuda", dtype=dtype)
    residual = torch.randn((rows, hidden_size), device="cuda", dtype=residual_dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    fp32_sum = x.float() + residual.float()
    summed = fp32_sum.to(residual_dtype)

    fused_x, fused_residual = x.clone(), residual.clone()
    unstored_x, unstored_residual = x.clone(), residual.clone()
    stored_sum_x, stored_sum_residual = x.clone(), residual.clone()
    with torch.inference_mode():
        expected = inference_norm.rms_norm(
            fp32_sum, weight, eps, zero_centered=True, out_dtype=dtype
        )
        expected_stored_sum = inference_norm.rms_norm(
            summed, weight, eps, zero_centered=True, out_dtype=dtype
        )
        inference_norm.AddRMSNorm(provider="triton")(
            fused_x, fused_residual, weight, eps, zero_centered=True
        )
        inference_norm.add_rms_norm_(
            unstored_x, unstored_residual, weight, eps, zero_centered=True, store_residual=False
        )
        inference_norm.add_rms_norm_(
            stored_sum_x,
            stored_sum_residual,
            weight,
            eps,
            zero_centered=True,
            normalize="stored_sum",
        )
        residual_output = inference_norm.residual_rms_norm(
            summed, weight, eps, zero_centered=True, out_dtype=dtype
        )
    trained, trained_sum = training_norm.add_rms_norm(x, residual, weight, eps, zero_centered=True)
    trained_residual_output, _ = training_norm.residual_rms_norm(
        summed, weight, eps, zero_centered=True, out_dtype=dtype
    )
    torch.cuda.synchronize()

    for actual in (fused_x, unstored_x):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual in (stored_sum_x, trained, residual_output, trained_residual_output):
        torch.testing.assert_close(actual, expected_stored_sum, rtol=0, atol=0)
    for actual in (fused_residual, stored_sum_residual, trained_sum):
        torch.testing.assert_close(actual, summed, rtol=0, atol=0)
    torch.testing.assert_close(unstored_residual, residual, rtol=0, atol=0)
    oracle.check_add_rms_norm(fused_x, x, residual, weight, eps, zero_centered=True)
    assert torch.equal(expected, expected_stored_sum) == fp32_residual


def test_residual_norms_replay_in_cuda_graph() -> None:
    torch.manual_seed(20261026)
    eps = 1e-6
    num_tokens, hidden_size = 6, 5120
    hidden = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    branch = torch.zeros_like(hidden)
    residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    def norms(hidden, branch, residual):
        # The first layer's input norm, an add+norm, and a last norm that leaves the residual.
        first = inference_norm.residual_rms_norm(
            residual, weight, eps, zero_centered=True, out_dtype=hidden.dtype
        )
        inference_norm.add_rms_norm_(hidden, residual, weight, eps, zero_centered=True)
        inference_norm.add_rms_norm_(
            branch, residual, weight, eps, zero_centered=True, store_residual=False
        )
        return first

    with torch.inference_mode():
        norms(hidden, branch, residual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        captured = norms(hidden, branch, residual)

    for _ in range(3):
        fresh = [torch.randn_like(tensor) for tensor in (hidden, branch, residual)]
        expected = [tensor.clone() for tensor in fresh]
        with torch.inference_mode():
            expected.append(norms(*expected))
        for static, values in zip((hidden, branch, residual), fresh, strict=True):
            static.copy_(values)
        graph.replay()
        torch.cuda.synchronize()

        for static, values in zip((hidden, branch, residual, captured), expected, strict=True):
            torch.testing.assert_close(static, values, rtol=0, atol=0)


def test_residual_norms_reject_unaligned_and_mixed_operands_before_writing() -> None:
    torch.manual_seed(20261027)
    x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((4, 256), device="cuda", dtype=torch.float32)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    unaligned = torch.randn(4 * 256 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(4, 256)
    strided = torch.randn((4, 512), device="cuda", dtype=torch.float32)[:, :256]
    tensors = (x, residual, unaligned, strided)
    originals = [tensor.clone() for tensor in tensors]

    with pytest.raises(ValueError, match="aligned"):
        training_norm.add_rms_norm(unaligned, residual, weight, 1e-6, zero_centered=True)
    for case_residual, error, message, out_dtype in (
        (unaligned, ValueError, "aligned", torch.bfloat16),
        (strided, ValueError, "contiguous", torch.bfloat16),
        (x, TypeError, "out_dtype or FP32", torch.float32),
        (x.half(), TypeError, "out_dtype or FP32", torch.bfloat16),
    ):
        if error is TypeError:
            with torch.inference_mode(), pytest.raises(error, match=message):
                inference_norm.residual_rms_norm(
                    case_residual, weight, 1e-6, zero_centered=True, out_dtype=out_dtype
                )
        else:
            actual = inference_norm.residual_rms_norm(
                case_residual, weight, 1e-6, zero_centered=True, out_dtype=out_dtype
            )
            expected = inference_norm.residual_rms_norm(
                case_residual.contiguous(), weight, 1e-6, zero_centered=True, out_dtype=out_dtype
            )
            assert torch.equal(actual, expected)
        with pytest.raises(error, match=message):
            training_norm.residual_rms_norm(
                case_residual, weight, 1e-6, zero_centered=True, out_dtype=out_dtype
            )
    # Forced FlashInfer still rejects stored-sum normalization; residual-only
    # training requires a frozen weight under either convention.
    with torch.inference_mode(), pytest.raises(RuntimeError, match="forced flashinfer"):
        inference_norm.AddRMSNorm(provider="flashinfer")(
            x, x.clone(), weight, 1e-6, normalize="stored_sum"
        )
    trainable = weight.clone().requires_grad_()
    with pytest.raises(ValueError, match="zero_centered=True"):
        training_norm.add_rms_norm(x, x.clone(), trainable, 1e-6, save="inputs")
    with pytest.raises(ValueError, match="frozen weight"):
        training_norm.residual_rms_norm(residual, trainable, 1e-6, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    for tensor, original in zip(tensors, originals, strict=True):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("zero", (False, True))
def test_cute_add_norm_graph_dynamic_rows_and_eps(pdl, dtype, zero):
    from ctikernels.norm.cute import _provider

    op = inference_norm.AddRMSNorm(provider="cute", config=Config(256, 2, pdl))
    weight = torch.randn(4096, device="cuda", dtype=dtype)
    x = torch.randn((3, 4096), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    op(x, residual, weight, 1e-6, zero_centered=zero)
    keys = set(_provider._COMPILED)
    x = torch.randn((1, 17, 4096), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    op(x, residual, weight, 1e-4, zero_centered=zero)
    assert set(_provider._COMPILED) == keys
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = op(x, residual, weight, 1e-4, zero_centered=zero)
    for _ in range(3):
        fresh_x, fresh_r = torch.randn_like(x), torch.randn_like(residual)
        expected_x, expected_r = fresh_x.clone(), fresh_r.clone()
        op(expected_x, expected_r, weight, 1e-4, zero_centered=zero)
        x.copy_(fresh_x)
        residual.copy_(fresh_r)
        graph.replay()
        assert returned[0] is x and returned[1] is residual
        assert torch.equal(x, expected_x) and torch.equal(residual, expected_r)
    assert set(_provider._COMPILED) == keys


@pytest.mark.parametrize("m,zero", [(1, False), (64, False), (256, True), (1024, True)])
def test_add_norm_dispatch_matches_table(m, zero):
    x = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16)
    residual, weight = torch.randn_like(x), torch.randn(4096, device="cuda", dtype=x.dtype)
    kind, _ = inference_norm._hardware(x.device)
    entry = entry_for(kind, 4096, "bfloat16", m, zero)
    assert entry is not None
    op = inference_norm.AddRMSNorm(provider="auto")
    op(x, residual, weight, 1e-6, zero_centered=zero)
    if entry.provider == "flashinfer" and not _flashinfer_version.allowed():
        assert isinstance(op.provider, Provider) and op.provider.reason
        assert op.provider in ("cute", "triton")
    else:
        assert op.provider == entry.provider


@pytest.mark.parametrize("mode", ("fp32_input", "stored_sum", "hidden"))
def test_add_norm_unsupported_cute_modes_keep_triton(mode):
    dtype = torch.float32 if mode == "fp32_input" else torch.bfloat16
    hidden = 4097 if mode == "hidden" else 4096
    x = torch.randn((3, hidden), device="cuda", dtype=dtype)
    residual = torch.randn_like(x).float() if mode == "fp32_residual" else torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    kwargs = {
        "zero_centered": True,
        "store_residual": mode != "no_store",
        "normalize": "stored_sum" if mode == "stored_sum" else "fp32_sum",
    }
    expected_x, expected_r = x.clone(), residual.clone()
    inference_norm.AddRMSNorm(provider="triton")(expected_x, expected_r, weight, 1e-6, **kwargs)
    op = inference_norm.AddRMSNorm(provider="auto")
    op(x, residual, weight, 1e-6, **kwargs)
    assert op.provider == "triton"
    assert torch.equal(x, expected_x) and torch.equal(residual, expected_r)
    with pytest.raises(RuntimeError, match="forced cute"):
        inference_norm.AddRMSNorm(provider="cute")(x, residual, weight, 1e-6, **kwargs)


def test_cute_add_norm_rejects_aliases_and_accepts_disjoint_views():
    storage = torch.randn(3 * 16 + 3 * 16 + 16, device="cuda", dtype=torch.bfloat16)
    x, residual, weight = storage[:48].view(3, 16), storage[48:96].view(3, 16), storage[96:]
    op = inference_norm.AddRMSNorm(provider="cute")
    original_x, original_residual = x.clone(), residual.clone()
    _, summed = oracle.add_rms_norm(x, residual, weight, upcast=False)
    op(x, residual, weight, 1e-6)
    assert torch.equal(residual, summed)
    oracle.check_add_rms_norm(x, original_x, original_residual, weight)
    before = storage.clone()
    for operands in ((x, x, weight), (x, residual, x[0]), (x, residual, residual[0])):
        with pytest.raises(RuntimeError, match="forced cute"):
            op(*operands, 1e-6)
    assert torch.equal(before, storage)


@pytest.mark.parametrize("shape", [(2, 1, 4096), (2, 256, 4096), (2, 600, 4096)])
@pytest.mark.parametrize(
    "provider,zero",
    [(p, z) for p in ("cute", "flashinfer", "auto") for z in (False, True)] + [("triton", True)],
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, pytest.param(torch.float16, marks=pytest.mark.exhaustive)]
)
def test_add_norm_higher_rank_is_bitwise_flattened(shape, provider, zero, dtype):
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("forced FlashInfer requires the qualified pin")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    residual, weight = torch.randn_like(x), torch.randn(shape[-1], device=x.device, dtype=dtype)
    flat_x, flat_r = x.clone().view(-1, shape[-1]), residual.clone().view(-1, shape[-1])
    op = inference_norm.AddRMSNorm(provider=provider)
    op(flat_x, flat_r, weight, 1e-6, zero_centered=zero)
    flat_provider = op.provider
    returned = op(x, residual, weight, 1e-6, zero_centered=zero)
    assert returned[0] is x and returned[1] is residual
    assert op.provider == flat_provider
    assert torch.equal(x.view(torch.int16), flat_x.view_as(x).view(torch.int16))
    assert torch.equal(residual.view(torch.int16), flat_r.view_as(residual).view(torch.int16))


@pytest.mark.parametrize(
    "provider,zero,fp32_residual,m,hidden,dtype,store",
    ci_cases(
        _SUM_PROVIDERS,
        (1, 4, 64, 300),
        (4096, 2560),
        (torch.bfloat16, torch.float16),
        (True, False),
        core={
            (*p, m, h, d, store)
            for p in _SUM_PROVIDERS
            for m, h, d, store in (
                (4, 4096, torch.bfloat16, True),
                (300, 2560, torch.float16, False),
            )
        }
        | {("cute", False, False, m, 4096, torch.bfloat16, True) for m in (1, 64)},
    ),
)
def test_add_norm_sum_out_bits_and_column_slice(
    provider, zero, fp32_residual, m, hidden, dtype, store, monkeypatch
):
    monkeypatch.delenv(inference_norm.VARIABLE, raising=False)
    torch.manual_seed(20261015 + m)
    x = torch.randn(m, hidden, device="cuda", dtype=dtype)
    residual = torch.randn(
        m, hidden, device="cuda", dtype=torch.float32 if fp32_residual else dtype
    )
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    before_x, before_r = x.clone(), residual.clone()
    expected_sum = torch.add(x, residual)
    aux = torch.full((m, 3 * hidden), 17.0, device="cuda", dtype=residual.dtype)
    summed = aux[:, hidden : 2 * hidden]
    op = inference_norm.AddRMSNorm(provider=provider)
    result = op(x, residual, weight, 1e-6, zero_centered=zero, store_residual=store, sum_out=summed)
    assert result[0] is x and result[1] is residual
    assert op.provider in ("cute", "triton")
    _assert_norm_bits(summed, expected_sum)
    _assert_norm_bits(residual, expected_sum if store else before_r)
    assert (aux[:, :hidden] == 17).all() and (aux[:, 2 * hidden :] == 17).all()
    # Auto may replace a FlashInfer band with CuTe for this output. Compare the
    # same provider with the auxiliary store removed, isolating its arithmetic.
    inference_norm.AddRMSNorm(provider=op.provider)(
        before_x, before_r, weight, 1e-6, zero_centered=zero, store_residual=store
    )
    _assert_norm_bits(x, before_x)
    _assert_norm_bits(residual, before_r)


@pytest.mark.parametrize("provider,zero,fp32_residual", _SUM_PROVIDERS)
@pytest.mark.parametrize("store", (False, True))
def test_add_norm_sum_out_graph_and_collapsible_rows(provider, zero, fp32_residual, store):
    dtype, hidden = torch.bfloat16, 4096
    x = torch.randn(2, 2, hidden, device="cuda", dtype=dtype)
    residual = torch.randn(x.shape, device="cuda", dtype=torch.float32 if fp32_residual else dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    aux = torch.full((2, 2, 3 * hidden), 17.0, device="cuda", dtype=residual.dtype)
    summed = aux[..., hidden : 2 * hidden]
    op = inference_norm.AddRMSNorm(provider=provider)

    def call():
        return op(
            x, residual, weight, 1e-6, zero_centered=zero, store_residual=store, sum_out=summed
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        fresh_x, fresh_r = torch.randn_like(x), torch.randn_like(residual)
        expected_x, expected_r = fresh_x.clone(), fresh_r.clone()
        op(expected_x, expected_r, weight, 1e-6, zero_centered=zero, store_residual=store)
        x.copy_(fresh_x)
        residual.copy_(fresh_r)
        graph.replay()
        _assert_norm_bits(x, expected_x)
        _assert_norm_bits(residual, expected_r)
        _assert_norm_bits(summed, torch.add(fresh_x, fresh_r))
        assert (aux[..., :hidden] == 17).all() and (aux[..., 2 * hidden :] == 17).all()


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize(
    "static_weight,captured",
    [(False, False), (False, True), (True, True)],
    ids=["safe-eager", "safe-graph", "stale-mutant-graph"],
)
@pytest.mark.parametrize("split", [1, 3], ids=["single", "cluster"])
def test_cute_norm_delayed_pdl_weight_producer(zero, m, captured, static_weight, split):
    # The mutant runs only as a captured graph: CUDA may serialize eager PDL launches
    # (seen on the RTX 5090 in about 1 of 5 eager runs even with the stream held),
    # while the graph's programmatic edge exposed the early prefetch in every run.
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL requires compute capability >= 9")
    from ..gpu._pdl_producer import compile_producer

    delay_ns = 200_000
    producer = compile_producer(4096, "bfloat16", torch.cuda.get_device_capability())
    consumer = inference_norm.AddRMSNorm(
        provider="cute",
        config=Config(pdl=True, static_weight=static_weight, split=split),
        static_weight=static_weight,
    )
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    initial_x, initial_r = x.clone(), residual.clone()
    weight = torch.full((4096,), 0.5, device="cuda", dtype=x.dtype)
    new_weight = torch.full_like(weight, 3.0)
    stamps = torch.empty(2, device="cuda", dtype=torch.int64)

    # Warm both launches and generate old/new references using the SAME consumer.
    consumer(x, residual, weight, 1e-6, zero_centered=zero)
    old_x = x.clone()
    producer(weight, new_weight, stamps, delay_ns)
    torch.cuda.synchronize()
    x.copy_(initial_x)
    residual.copy_(initial_r)
    consumer(x, residual, weight, 1e-6, zero_centered=zero)
    expected_x, expected_r = x.clone(), residual.clone()
    torch.cuda.synchronize()
    assert not torch.equal(old_x, expected_x)

    def pair():
        producer(weight, new_weight, stamps, delay_ns)
        consumer(x, residual, weight, 1e-6, zero_centered=zero)

    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pair()
    for _ in range(10):
        x.copy_(initial_x)
        residual.copy_(initial_r)
        weight.fill_(0.5)
        stamps.zero_()
        if graph is None:
            # Hold the stream so both eager launches are queued before the producer
            # runs; otherwise host launch latency can exceed the delay and hide the race.
            torch.cuda._sleep(5_000_000)
        # No kernel or event between producer and consumer. Resets happen before
        # the pair so they cannot introduce a serialization edge inside it.
        if graph is None:
            pair()
        else:
            graph.replay()
        torch.cuda.current_stream().synchronize()
        assert stamps[1].item() - stamps[0].item() >= delay_ns
        assert torch.equal(weight, new_weight)
        assert torch.equal(residual, expected_r)
        if static_weight:
            # Strict mutant: serialization is a failure, never an XPASS/skip.
            assert not torch.equal(x, expected_x), "PDL did not expose early weight prefetch"
            assert torch.equal(x, old_x), "mutant must read the complete old weight"
        else:
            assert torch.equal(x, expected_x)


@pytest.mark.parametrize("threads,rows,hidden", ((128, 2, 8192), (128, 4, 4096), (256, 4, 4096)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("static", (False, True))
def test_cute_norm_early_trigger_chain_core(threads, rows, hidden, dtype, zero, static):
    """Odd final CTA, 1/2 warp rows, unchanged arithmetic and two-call PDL graphs."""
    from dataclasses import replace

    torch.manual_seed(34)
    config = Config(threads, rows, True, static)
    end = inference_norm.AddRMSNorm(provider="cute", config=config, static_weight=static)
    early = inference_norm.AddRMSNorm(
        provider="cute", config=replace(config, trigger="early"), static_weight=static
    )
    x = torch.randn(3, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=dtype) * 0.125
    _, stored = oracle.add_rms_norm(x, residual, weight, zero_centered=zero, upcast=False)
    end_x, end_r, early_x, early_r = x.clone(), residual.clone(), x.clone(), residual.clone()
    for op, xx, rr in ((end, end_x, end_r), (early, early_x, early_r)):
        op(xx, rr, weight, 1e-6, zero_centered=zero)
        oracle.check_add_rms_norm(xx, x, residual, weight, zero_centered=zero)
        assert torch.equal(rr, stored)
    assert torch.equal(end_x.view(torch.uint8), early_x.view(torch.uint8))
    assert torch.equal(end_r.view(torch.uint8), early_r.view(torch.uint8))
    # The second call consumes both outputs of the first. The end-trigger chain
    # supplies the bitwise oracle; early release must not expose partial stores.
    end(end_x, end_r, weight, 1e-6, zero_centered=zero)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        early(early_x, early_r, weight, 1e-6, zero_centered=zero)
        early(early_x, early_r, weight, 1e-6, zero_centered=zero)
    for _ in range(2):
        early_x.copy_(x)
        early_r.copy_(residual)
        graph.replay()
        assert torch.equal(end_x.view(torch.uint8), early_x.view(torch.uint8))
        assert torch.equal(end_r.view(torch.uint8), early_r.view(torch.uint8))


def test_cute_norm_sweep_capture_hot_then_engine_core():
    from types import SimpleNamespace

    from benchmark.kernel_tuning.norm_act_norm import NormCall
    from benchmark.kernel_tuning.norm_act_paired import capture_bank
    from benchmark.kernel_tuning.norm_act_pools import build_norm_pools, pool_spec

    args = SimpleNamespace(op="add_norm", width=4096, layers=2, dtype="bf16", static_weight=0)
    candidates = [
        NormCall(False, "cute", Config(128, rows, True, False, trigger), static_weight=False)
        for rows in (2, 4)
        for trigger in ("end", "early")
    ]
    for regime in ("hot", "engine"):
        pools = build_norm_pools(pool_spec(args, 1, regime, 1024))
        for call in candidates:
            pools.restore()
            bank = capture_bank(call, pools.inputs, [None] * pools.count, args.layers)
            # Private warmup and capture must not modify the benchmark's inputs.
            assert torch.equal(pools.x, pools.saved_x)
            assert torch.equal(pools.residual, pools.saved_r)
            for graph, _, _ in bank[0]:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(pools.x).all()
            assert call.provider == "cute"


@pytest.mark.parametrize("provider", ("cute", "triton"))
@pytest.mark.parametrize("zero", (False, True))
def test_fp32_residual_and_weight(provider, zero):
    _check_fp32_residual(provider, 3, 2048, torch.bfloat16, zero, fp32_weight=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(136,), (2, 3, 4, 136), (2, 1, 3, 4, 136)])
@pytest.mark.parametrize("inplace", [False, True])
def test_ordinary_rmsnorm_rank_adaptation_bitwise(dtype, shape, inplace, monkeypatch):
    monkeypatch.setenv(inference_norm.VARIABLE, "triton")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    saved = x.clone()
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected = inference_norm.rms_norm(x.view(-1, shape[-1]), weight, 1e-6).view(shape)
    operation = inference_norm.rms_norm_ if inplace else inference_norm.rms_norm
    actual = operation(x, weight, 1e-6)
    assert actual.shape == x.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if inplace:
        assert actual is x
    else:
        assert torch.equal(x, saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["transpose_prefix", "row_gaps", "odd_row_pitch", "hidden_stride"]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_ordinary_rmsnorm_strided_prefix_bitwise(dtype, layout, inplace, monkeypatch):
    monkeypatch.setenv(inference_norm.VARIABLE, "triton")
    base = torch.randn(2, 3, 4, 272, device="cuda", dtype=dtype)
    x = (
        base.transpose(0, 1)
        if layout == "transpose_prefix"
        else (base[..., :136] if layout == "row_gaps" else base[..., ::2])
    )
    if layout == "odd_row_pitch":
        x = torch.randn(2, 3, 4, 273, device="cuda", dtype=dtype)[..., :136]
    saved = x.clone()
    weight = torch.randn(x.shape[-1], device="cuda", dtype=dtype)
    rows = x.contiguous().view(-1, x.shape[-1])
    expected = inference_norm.rms_norm(rows, weight, 1e-6).view(x.shape)
    operation = inference_norm.rms_norm_ if inplace else inference_norm.rms_norm
    actual = operation(x, weight, 1e-6)
    assert torch.equal(actual.contiguous().view(torch.int16), expected.view(torch.int16))
    if inplace:
        assert actual is x
    else:
        assert torch.equal(x, saved)


@pytest.mark.parametrize(
    "split,threads,m,hidden,dtype,mixed,zero",
    ci_cases(
        (2, 3, 4, 8),
        (64, 128),
        (1, 3, 64, 65, 257),
        (2048, 4096),
        (torch.bfloat16, torch.float16),
        (False, True),
        (False, True),
        core={
            (s, t, 65, h, torch.bfloat16, True, False)
            for s in (2, 3, 4, 8)
            for t, h in ((64, 2048), (128, 4096))
        }
        | {
            (3, 64, 3, 2048, torch.float16, False, True),
            (8, 128, 257, 4096, torch.float16, True, True),
        },
    ),
)
def test_cute_cluster_norm_accuracy_and_stores(split, threads, m, hidden, dtype, mixed, zero):
    config = Config(threads, pdl=True, trigger="early", split=split)
    op = _check_fp32_residual("cute", m, hidden, dtype, zero, config, mixed=mixed)
    assert op.dispatched_config == config


@pytest.mark.parametrize("split", [2, 3, 4, 8])
@pytest.mark.parametrize("pdl", [False, True], ids=["no-pdl", "pdl"])
def test_cute_cluster_norm_graph_bits(split, pdl):
    config = Config(64, pdl=pdl, trigger="early" if pdl else "end", split=split)
    op = inference_norm.AddRMSNorm(provider="cute", config=config)
    x = torch.randn(65, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x, dtype=torch.float32)
    weight = torch.randn(2048, device="cuda", dtype=x.dtype)
    summed = torch.empty_like(residual)
    saved_x, saved_r = x.clone(), residual.clone()

    def call():
        op(x, residual, weight, 1e-6, sum_out=summed, store_residual=False)

    call()
    expected_x, expected_sum = x.clone(), summed.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        x.copy_(saved_x)
        residual.copy_(saved_r)
        graph.replay()
        _assert_norm_bits(x, expected_x)
        _assert_norm_bits(summed, expected_sum)
        _assert_norm_bits(residual, saved_r)
    assert op.dispatched_config == config


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(0, 4096), (0, 3, 4096)])
def test_ordinary_rmsnorm_empty_no_launch(monkeypatch, dtype, shape):
    monkeypatch.setattr(inference_norm, "_standalone", lambda *a: pytest.fail("empty launch"))
    x = torch.empty(shape, device="cuda", dtype=dtype)
    weight = torch.ones(4096, device="cuda", dtype=dtype)
    output = inference_norm.rms_norm(x, weight, 1e-6)
    assert output.shape == x.shape and output.dtype == x.dtype
    assert inference_norm.rms_norm_(x, weight, 1e-6) is x
    output = inference_norm.residual_rms_norm(x, weight, 1e-6, out_dtype=torch.float32)
    assert output.shape == x.shape and output.dtype == torch.float32


@pytest.mark.parametrize("provider", ["auto", "cute", "flashinfer", "triton"])
@pytest.mark.parametrize("mixed", [False, True])
def test_add_rmsnorm_empty_no_launch(monkeypatch, provider, mixed):
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("forced FlashInfer requires the qualified pin")
    monkeypatch.setattr(
        inference_norm.AddRMSNorm, "_resolve", lambda *a: pytest.fail("empty resolution")
    )
    monkeypatch.setenv(inference_norm.VARIABLE, provider)
    x = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.empty_like(x, dtype=torch.float32 if mixed else x.dtype)
    weight = torch.ones(4096, device="cuda", dtype=x.dtype)
    for call in (inference_norm.add_rms_norm_, inference_norm.AddRMSNorm(provider=provider)):
        output, summed = call(x, residual, weight, 1e-6, sum_out=torch.empty_like(residual))
        assert output is x and summed is residual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("inplace", [False, True])
def test_ordinary_rmsnorm_hidden_stride_rank23_bits(dtype, rank, inplace, monkeypatch):
    monkeypatch.setenv(inference_norm.VARIABLE, "triton")
    shape = (3, 4096) if rank == 2 else (2, 3, 4096)
    x = torch.randn(*shape[:-1], 2 * shape[-1], device="cuda", dtype=dtype)[..., ::2]
    saved = x.clone()
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    # Native and packed layouts use the same Triton reduction.
    expected = inference_norm.rms_norm(x.contiguous().view(-1, shape[-1]), weight, 1e-6).view(shape)
    call = inference_norm.rms_norm_ if inplace else inference_norm.rms_norm
    actual = call(x, weight, 1e-6)
    _assert_norm_bits(actual, expected)
    if inplace:
        assert actual is x
    else:
        _assert_norm_bits(x, saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("zero", [False, True])
def test_add_norm_warm_up_graph_routes_core(zero):
    op = inference_norm.AddRMSNorm()
    device = torch.device("cuda", torch.cuda.current_device())
    routes = op.warm_up(128, hidden=4096, dtype=torch.bfloat16, device=device, zero_centered=zero)
    op._require_cached_plan = True
    weight = torch.randn(4096, device=device, dtype=torch.bfloat16)
    for m in (1, 4, 16, 64, 65, 128):
        route = next(r for r in routes if r["m"][0] <= m <= r["m"][1])
        x = torch.randn((m, 4096), device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        original_x, original_r = x.clone(), residual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, residual, weight, 1e-6, zero_centered=zero)
        assert op.provider == route["provider"]
        assert (None if op.dispatched_config is None else op.dispatched_config.name) == route[
            "config"
        ]
        x.copy_(original_x)
        residual.copy_(original_r)
        graph.replay()
        expected_x, expected_r = original_x.clone(), original_r.clone()
        op(expected_x, expected_r, weight, 1e-6, zero_centered=zero)
        assert torch.equal(x.view(torch.int16), expected_x.view(torch.int16))
        assert torch.equal(residual.view(torch.int16), expected_r.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "provider,store,summed",
    [
        ("flashinfer", True, False),
        ("triton", True, False),
        ("triton", False, False),
        ("triton", True, True),
        ("triton", False, True),
    ],
)
def test_zero_centered_norm_warm_up_lazy_provider_graph(provider, store, summed):
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("forced FlashInfer requires the qualified pin")
    device = torch.device("cuda", torch.cuda.current_device())
    kind, _ = inference_norm._hardware(device)
    dtype = torch.bfloat16
    residual_dtype = dtype
    override = provider
    shapes = (1, 16, 17, 128)
    # RTX 5090 has a zero-centered FP32-residual Triton row at M=9..40.
    # Prefer a real table row; other devices/providers need a forced instance.
    for candidate_dtype in (dtype, torch.float32):
        matches = [
            m
            for m in range(1, 129)
            if (
                entry := entry_for(
                    kind,
                    4096,
                    "bfloat16",
                    m,
                    True,
                    residual_dtype=str(candidate_dtype).removeprefix("torch."),
                )
            )
            is not None
            and entry.provider == provider
        ]
        if matches:
            residual_dtype = candidate_dtype
            override = "auto"
            shapes = tuple(m for m in shapes if m in matches) or (matches[0],)
            break
    op = inference_norm.AddRMSNorm(provider=override)
    kwargs = {"zero_centered": True, "store_residual": store}
    routes = op.warm_up(
        128,
        hidden=4096,
        dtype=dtype,
        device=device,
        residual_dtype=residual_dtype,
        sum_out=summed,
        **kwargs,
    )
    op._require_cached_plan = True
    weight = torch.randn(4096, device=device, dtype=dtype)
    for m in shapes:
        route = next(r for r in routes if r["m"][0] <= m <= r["m"][1])
        assert route["provider"] == provider
        x = torch.randn(m, 4096, device=device, dtype=dtype)
        residual = torch.randn_like(x, dtype=residual_dtype)
        original_x, original_r = x.clone(), residual.clone()
        sum_out = torch.empty_like(residual) if summed else None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, residual, weight, 1e-6, sum_out=sum_out, **kwargs)
        assert op.provider == provider
        x.copy_(original_x)
        residual.copy_(original_r)
        graph.replay()
        expected_x, expected_r = original_x.clone(), original_r.clone()
        expected_sum = torch.empty_like(residual) if summed else None
        op(expected_x, expected_r, weight, 1e-6, sum_out=expected_sum, **kwargs)
        _assert_norm_bits(x, expected_x)
        _assert_norm_bits(residual, expected_r)
        if summed:
            _assert_norm_bits(sum_out, expected_sum)


@pytest.mark.parametrize(
    "dtype,fp32_weight,out_dtype,hidden",
    ci_cases(
        (torch.float16, torch.bfloat16, torch.float32),
        (False, True),
        (torch.float16, torch.bfloat16, torch.float32),
        (1, 7, 64, 128, 1000, 4096),
        core={
            (torch.float16, False, torch.float16, 1),
            (torch.bfloat16, True, torch.float32, 7),
            (torch.float32, False, torch.bfloat16, 64),
            (torch.float16, True, torch.bfloat16, 128),
            (torch.bfloat16, False, torch.float16, 1000),
            (torch.float32, True, torch.float32, 4096),
        },
    ),
)
def test_standard_rmsnorm_triton_generic(monkeypatch, dtype, fp32_weight, out_dtype, hidden):
    monkeypatch.setenv(inference_norm.VARIABLE, "triton")
    torch.manual_seed(20261017)
    storage = torch.randn(2, 3, hidden * 3, device="cuda", dtype=dtype)
    x = storage[..., hidden : 2 * hidden]
    before = storage.clone()
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32 if fp32_weight else dtype)
    saved = x.clone()
    out = inference_norm.rms_norm(x, weight, 1e-6, out_dtype=out_dtype)
    assert out.dtype == out_dtype and out.data_ptr() != x.data_ptr()
    oracle.check_rms_norm(out, saved, weight)
    assert inference_norm.last_standalone_provider["rms_norm"] == "triton"
    returned = inference_norm.rms_norm_(x, weight, 1e-6)
    assert returned is x
    oracle.check_rms_norm(x, saved, weight)
    assert inference_norm.last_standalone_provider["rms_norm_"] == "triton"
    assert torch.equal(storage[..., :hidden], before[..., :hidden])
    assert torch.equal(storage[..., 2 * hidden :], before[..., 2 * hidden :])


@pytest.mark.parametrize("inplace", [False, True])
def test_standard_rmsnorm_unusable_fi_reroute(monkeypatch, inplace):
    monkeypatch.setenv(inference_norm.VARIABLE, "auto")
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(_flashinfer_version, "allowed", lambda: False)
    inference_norm._resolve_standalone.cache_clear()
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    saved = x.clone()
    name = "rms_norm_" if inplace else "rms_norm"
    call = getattr(inference_norm, name)
    out = call(x, weight, 1e-6)
    oracle.check_rms_norm(out, saved, weight)
    provider = inference_norm.last_standalone_provider[name]
    assert isinstance(provider, Provider) and provider == "triton" and provider.reason
    launches = []
    monkeypatch.setattr(zero_centered_kernels, "_launch", lambda *a, **k: launches.append(1))
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError):
        call(x, weight, 1e-6)
    assert launches == []


@pytest.mark.parametrize("zero", [False, True])
def test_add_norm_row_strided_inputs_admitted(monkeypatch, zero):
    monkeypatch.setenv(inference_norm.VARIABLE, "auto")
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    op = inference_norm._DEFAULT_ADD
    op._plans.clear()
    op._pinned.clear()
    x_storage = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16)
    residual_storage = torch.randn_like(x_storage)
    x, residual = x_storage[:, :4096], residual_storage[:, :4096]
    x_tail, residual_tail = x_storage[:, 4096:].clone(), residual_storage[:, 4096:].clone()
    weight = torch.randn(4096, device="cuda", dtype=x.dtype)
    summed = x.float() + residual.float()
    saved_x, saved_r = x.clone(), residual.clone()
    inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero)
    oracle.check_add_rms_norm(x, saved_x, saved_r, weight, zero_centered=zero)
    torch.testing.assert_close(residual, summed.to(residual.dtype), rtol=0, atol=0)
    assert torch.equal(x_storage[:, 4096:], x_tail)
    assert torch.equal(residual_storage[:, 4096:], residual_tail)


@pytest.mark.parametrize("zero", [False, True])
def test_add_norm_unusable_fi_table_reroute(monkeypatch, zero):
    device = torch.device("cuda", torch.cuda.current_device())
    kind, _ = inference_norm._hardware(device)
    bands = TABLE.get(kind, {}).get((4096, "bfloat16", "bfloat16", zero, False), {})
    rows = [max(1, lo) for (lo, _), entry in bands.items() if entry.provider == "flashinfer"]
    if not rows:
        pytest.skip("device has no FlashInfer table row for this weight convention")
    monkeypatch.setenv(inference_norm.VARIABLE, "auto")
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(_flashinfer_version, "allowed", lambda: False)
    op = inference_norm._DEFAULT_ADD
    op._plans.clear()
    op._pinned.clear()
    x = torch.randn(min(rows), 4096, device=device, dtype=torch.bfloat16)
    residual, weight = torch.randn_like(x), torch.randn(4096, device=device, dtype=x.dtype)
    summed = x.float() + residual.float()
    saved_x, saved_r = x.clone(), residual.clone()
    inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero)
    oracle.check_add_rms_norm(x, saved_x, saved_r, weight, zero_centered=zero)
    torch.testing.assert_close(residual, summed.to(residual.dtype), rtol=0, atol=0)
    assert isinstance(op.provider, Provider) and op.provider.reason
    assert op.provider in ("cute", "triton")
    from ctikernels.norm import cute

    launches = []
    monkeypatch.setattr(zero_centered_kernels, "add_rms_norm_", lambda *a, **k: launches.append(1))
    monkeypatch.setattr(
        inference_norm.flashinfer, "add_rms_norm_", lambda *a, **k: launches.append(1)
    )
    monkeypatch.setattr(cute, "get_compiled", lambda *a, **k: launches.append(1))
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError):
        inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero)
    assert launches == []


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_inference_frozen_weight_rejects_without_launch(monkeypatch, grad_enabled, zero):
    from ctikernels.norm import cute

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(128, device="cuda", dtype=x.dtype, requires_grad=True)
    launches = []

    def launch(*a, **k):
        launches.append(1)

    for name in ("_launch", "_launch_residual_norm", "add_rms_norm_"):
        monkeypatch.setattr(zero_centered_kernels, name, launch)
    monkeypatch.setattr(inference_norm, "_resolve_standalone", launch)
    monkeypatch.setattr(inference_norm.flashinfer, "add_rms_norm_", launch)
    monkeypatch.setattr(cute, "get_compiled", launch)
    for call in (
        lambda: inference_norm.rms_norm(x, weight, 1e-6, zero_centered=zero),
        lambda: inference_norm.rms_norm_(x, weight, 1e-6, zero_centered=zero),
        lambda: inference_norm.residual_rms_norm(
            residual, weight, 1e-6, zero_centered=zero, out_dtype=x.dtype
        ),
        lambda: inference_norm.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero),
        lambda: inference_norm.AddRMSNorm()(x, residual, weight, 1e-6, zero_centered=zero),
    ):
        with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="frozen weight"):
            call()
    assert launches == []
