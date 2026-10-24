# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Qualification gates: no floors, additive slack or expected failures."""

import pytest
import torch
import torch.nn.functional as F


def _max_relative(got, reference):
    difference = (got.double() - reference.double()).abs()
    relative = torch.where(
        reference != 0,
        difference / reference.double().abs(),
        torch.where(difference == 0, 0.0, torch.inf),
    )
    return float(relative.max())


def _assert_gate(got, tp1, *, checks, label, dtype, oracle=None, output=False, factor=1.1):
    valid = got.shape == tp1.shape and bool(torch.isfinite(got).all() & torch.isfinite(tp1).all())
    details = ""
    if oracle is None:
        valid = valid and got.dtype == tp1.dtype == torch.float32
        metric, limit = _max_relative(got, tp1), 1e-5
        name = "max elementwise rel diff"
    else:
        ours = float((got.double() - oracle.double()).abs().max())
        previous = float((tp1.double() - oracle.double()).abs().max())
        ratio = ours / previous if previous else (0.0 if ours == 0 else float("inf"))
        details = f"; TP2 err={ours:.8g}, TP1 err={previous:.8g}, ratio={ratio:.8g}"
        valid = valid and bool(torch.isfinite(oracle).all())
        if output:
            valid = valid and got.dtype == tp1.dtype == torch.float32
        if dtype == torch.float32 and not output:
            metric = float((got.double() - tp1.double()).abs().max())
            limit = 1e-5 * float(tp1.abs().max())
            name = "max abs diff"
        else:
            metric, limit = ours, factor * previous
            name = "TP2 oracle err"
    check = {
        "label": label,
        "metric": metric,
        "limit": limit,
        "passed": valid and metric <= limit,
        "valid": valid,
        "name": name,
        "details": details,
        "dtype": str(dtype),
    }
    checks.append(check)
    print(_line(check), flush=True)
    return check


def _line(check):
    status = "PASS" if check["passed"] else "FAIL"
    invalid = "" if check["valid"] else "; invalid shape/dtype or nonfinite values"
    return (
        f"{status} {check['dtype']} {check['label']}: {check['name']}={check['metric']:.8g} "
        f"limit={check['limit']:.8g}{check['details']}{invalid}"
    )


def _finish(checks):
    if any(not check["passed"] for check in checks):
        table = "\n".join(_line(check) for check in checks)
        pytest.fail(f"Strict vocab-parallel gates:\n{table}")


def _head_dtype_mode(dtype, fp32_logits):
    """BF16/FP16 logits: FP32 outputs use the FP64-oracle rule (TP1 itself
    changes by one logit ulp with GEMM M/N). FP32 logits (fp32_logits=True or an
    FP32 head) keep the elementwise 1e-5 rule against TP1."""
    return not fp32_logits and dtype in (torch.bfloat16, torch.float16)


@torch.no_grad()
def _stats(logits, y, start, temperature):
    logits = logits / temperature
    maximum = logits.max(1).values
    local = y - start
    owned = (y != -100) & (local >= 0) & (local < logits.shape[1])
    target = torch.where(
        owned, logits.gather(1, local.clamp(0, logits.shape[1] - 1)[:, None])[:, 0], 0.0
    )
    reference = torch.stack((maximum, (logits - maximum[:, None]).exp().sum(1), target), 1)
    reference[y == -100] = reference.new_tensor([0.0, 1.0, 0.0])
    return reference


@torch.no_grad()
def _tp1_logits(x, w, b, chunk_size, kind):
    """TP1's own projection: the full vocabulary head over TP1's row chunks."""
    chunks = []
    for rows in x.split(chunk_size):
        if kind == "ce":
            chunks.append(rows @ w.t() if b is None else torch.addmm(b, rows, w.t()))
        else:
            chunks.append(F.linear(rows, w, b))
    return torch.cat(chunks)


@torch.no_grad()
def _check_stats(part, x, w, b, y, start, temperature, fp32_logits, *, checks, tp1=None):
    """``tp1=(full_weight, full_bias, tp1_chunk_size, kind)`` is required for
    head-dtype logits: the baseline is TP1's full-head projection sliced to this
    shard's columns, so its GEMM-shape rounding is TP1's real rounding."""
    dtype = x.dtype

    def project(rows):
        if not fp32_logits:
            return F.linear(rows, w, b)
        if rows.is_cuda and rows.dtype in (torch.bfloat16, torch.float16):
            if b is None:
                return torch.mm(rows, w.t(), out_dtype=torch.float32)
            return torch.addmm(b.float(), rows, w.t(), out_dtype=torch.float32)
        return F.linear(rows.float(), w.float(), b.float() if b is not None else None)

    head_dtype_mode = _head_dtype_mode(dtype, fp32_logits)
    # FP32-logit modes gate the shard's reduction at 1e-5, so the reference
    # projects the same row chunks (same GEMM M); TP1 equivalence is gated on
    # the outputs. Head-dtype mode slices TP1's full-vocabulary projection.
    if head_dtype_mode:
        if tp1 is None:
            raise ValueError("head-dtype shard stats need the TP1 full head (tp1=...)")
        full_w, full_b, tp1_chunk, kind = tp1
        logits = _tp1_logits(x, full_w, full_b, tp1_chunk, kind)
        logits = logits[:, start : start + w.shape[0]]
    else:
        logits = torch.cat([project(r) for r in x.split(part._chunk_size)])
    reference = _stats(logits.float(), y, start, temperature)
    oracle = _stats(
        F.linear(x.double(), w.double(), b.double() if b is not None else None),
        y,
        start,
        temperature,
    )
    for column, name in enumerate(("max", "sumexp", "target")):
        _assert_gate(
            part.stats[:, column],
            reference[:, column],
            checks=checks,
            label=f"shard {start} {name}",
            dtype=dtype,
            oracle=oracle[:, column] if head_dtype_mode else None,
            output=True,
        )
    torch.testing.assert_close(part.stats[y == -100], reference[y == -100], rtol=0, atol=0)


def _check_results(actual, baseline, oracle, dtype, *, checks, fp32_logits=False):
    head_dtype_mode = _head_dtype_mode(dtype, fp32_logits)
    for index, (got, old, ref) in enumerate(zip(actual, baseline, oracle, strict=True)):
        _assert_gate(
            got,
            old,
            label=("output", "dX", "dW", "db")[index],
            checks=checks,
            dtype=dtype,
            oracle=ref if index != 0 or head_dtype_mode else None,
            output=index == 0,
        )
