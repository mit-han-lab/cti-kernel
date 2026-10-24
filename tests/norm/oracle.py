# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Reference RMS norms for tests/norm, after flash-attention's ``attention_ref(upcast=...)``.

Each formula has two modes. ``upcast=True`` computes in FP64 and returns FP64: the reference.
``upcast=False`` is the plain PyTorch implementation at the op's precision, written as the
frameworks write it: the yardstick of tests/support/accuracy.py.

- Standard weight (vLLM ``RMSNorm.forward_native``, Hugging Face Llama): FP32 statistics, the
  normalized row rounded to the input dtype, then multiplied by the weight.
- Zero-centered weight (Qwen3.5, Gemma): FP32 statistics times ``1 + weight`` in FP32, rounded
  once to the output dtype.
- Fused add (vLLM ``fused_add_rms_norm``): the sum in FP32, stored rounded to the residual dtype;
  ``normalize="fp32_sum"`` normalizes the FP32 sum, ``"stored_sum"`` the rounded one.

All functions are differentiable, so FP64 autograd through ``upcast=True`` is the gradient
reference and autograd through ``upcast=False`` the gradient yardstick.
"""

from __future__ import annotations

import torch

from ..support.accuracy import assert_accurate

EPS = 1e-6
# Rows per check chunk: bounds FP64 scratch to about 2**22 elements.
CHUNK_ELEMENTS = 1 << 22


def _framework(values, weight, eps, zero_centered, dtype, out_dtype):
    """FP32 ``values`` normalized the frameworks' way for an op whose input dtype is ``dtype``."""
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    if zero_centered:
        return (normalized * (1.0 + weight.float())).to(out_dtype)
    if weight.dtype == dtype:
        return (normalized.to(dtype) * weight).to(out_dtype)
    return (normalized * weight.float()).to(out_dtype)


def rms_norm(x, weight, eps=EPS, *, zero_centered=False, out_dtype=None, upcast=True):
    """RMS norm of the last dimension; FP64 with ``upcast``, else torch at x's precision."""
    if upcast:
        values = x.double()
        scale = weight.double() + 1 if zero_centered else weight.double()
        return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * scale
    return _framework(x.float(), weight, eps, zero_centered, x.dtype, out_dtype or x.dtype)


def add_rms_norm(
    x, residual, weight, eps=EPS, *, zero_centered=False, normalize="fp32_sum", upcast=True
):
    """Returns ``(normalized, stored_sum)``; the stored sum is always in residual's dtype."""
    summed = x.float() + residual.float()
    stored = summed.to(residual.dtype)
    stored_sum = normalize == "stored_sum"
    if upcast:
        values = stored.double() if stored_sum else x.double() + residual.double()
        return rms_norm(values, weight, eps, zero_centered=zero_centered), stored
    values = stored.float() if stored_sum else summed
    return _framework(values, weight, eps, zero_centered, x.dtype, x.dtype), stored


def _rows(tensor):
    return tensor.reshape(-1, tensor.shape[-1]) if tensor.ndim else tensor.reshape(1, 1)


def check(actual, reference_fn, baseline_fn, *inputs, name="output"):
    """Gate ``actual`` row chunk by row chunk; the fns take row chunks of ``inputs``."""
    if not actual.numel():
        return
    rows = [_rows(t) for t in inputs]
    actual_rows = _rows(actual)
    step = max(1, CHUNK_ELEMENTS // actual_rows.shape[-1])
    for start in range(0, actual_rows.shape[0], step):
        part = [t[start : start + step] for t in rows]
        assert_accurate(
            actual_rows[start : start + step],
            reference_fn(*part),
            baseline_fn(*part),
            family="norm",
            name=f"{name}[rows {start}:{start + len(part[0])}]",
        )


def check_rms_norm(actual, x, weight, eps=EPS, *, zero_centered=False, name="rms_norm"):
    """``actual`` is the norm of ``x`` (any layout) rounded to ``actual.dtype``."""
    check(
        actual,
        lambda rows: rms_norm(rows, weight, eps, zero_centered=zero_centered),
        lambda rows: rms_norm(
            rows, weight, eps, zero_centered=zero_centered, out_dtype=actual.dtype, upcast=False
        ),
        x,
        name=name,
    )


def check_add_rms_norm(
    actual, x, residual, weight, eps=EPS, *, zero_centered=False, normalize="fp32_sum"
):
    """``actual`` is the normalized output of an add norm of the original ``x``/``residual``."""

    def fn(upcast):
        return lambda xs, rs: add_rms_norm(
            xs, rs, weight, eps, zero_centered=zero_centered, normalize=normalize, upcast=upcast
        )[0]

    check(actual, fn(True), fn(False), x, residual, name=f"add_rms_norm[{normalize}]")
