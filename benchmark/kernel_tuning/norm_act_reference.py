# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Torch-only numerical oracle and market formulas; usable on CPU and CUDA.

The numerical gate compares EACH metric to the smallest comparator error on
that same input: ours <= min(markets) * (1 + TAU[metric]). TAU is the measured
multi-seed FP32 evaluation-order noise floor, rounded up per metric. An absolute
1e-12 floor applies only when the best comparator error is exactly zero.
"""

import math

import torch
import torch.nn.functional as F

FP32_ACTIVATION_KINDS = ("silu", "gelu", "gelu_tanh", "relu2")
FP32_ACTIVATION_ROWS = (0, 1, 3, 64, 257)
FP32_ACTIVATION_WIDTHS = (4, 8, 12, 13, 6144, 12288)
RELU2_WIDTHS = (1, 13, 6144, 12288)


def support_range_formula(kind, x, *, exact):
    """Unrounded FP64 oracle or torch eager FP32 expression of SGLang's math.

    Preserve comparison-based relu2 semantics, including NaN -> zero. The
    FP16/BF16 support-range tests also use this expression, rounding only once.
    """
    values = x.double() if exact else x.float()
    if kind == "relu2":
        positive = torch.where(values > 0, values, 0.0)
        result = positive.square()
    else:
        gate, up = values.chunk(2, dim=-1)
        if kind == "silu":
            activated = gate * torch.sigmoid(gate) if exact else gate / (1 + torch.exp(-gate))
        elif kind == "gelu":
            activated = gate * (0.5 * (1 + torch.erf(gate * (2**-0.5))))
        elif kind == "gelu_tanh":
            activated = gate * (
                0.5
                * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (gate + 0.044715 * gate * gate * gate)))
            )
        else:
            raise ValueError(f"unknown activation: {kind}")
        result = activated * up
    return result if exact else result.to(x.dtype)


def fp32_activation_formulas(kind, x):
    """Independent faithful FP32 comparators for the standard-normal test sweep.

    No CuTe output participates in calibration. Native torch GELU is explicitly
    exact-erf or tanh as appropriate; reordered expressions change FP32 rounding
    but not the real-valued formula. relu2 has only one multiplication, so its
    square/multiply/pow expressions can agree bitwise (zero measured noise).
    """
    if x.dtype != torch.float32:
        raise ValueError("FP32 activation comparators require float32 input")
    eager = support_range_formula(kind, x, exact=False)
    if kind == "relu2":
        positive = torch.where(x > 0, x, 0.0)
        native = torch.mul(positive, positive)
        reordered = torch.pow(positive, 2)
    else:
        gate, up = x.chunk(2, dim=-1)
        if kind == "silu":
            native = F.silu(gate) * up
            reordered = (gate * up) / (1 + torch.exp(-gate))
        elif kind == "gelu":
            native = F.gelu(gate, approximate="none") * up
            reordered = (gate * up) * (0.5 * (1 + torch.erf(gate * (2**-0.5))))
        else:
            native = F.gelu(gate, approximate="tanh") * up
            argument = (2 / torch.pi) ** 0.5 * (gate * (1 + 0.044715 * gate.square()))
            reordered = (gate * up) * (0.5 * (1 + torch.tanh(argument)))
    return {"sglang_eager": eager, "torch_native": native, "reordered": reordered}


def norm_reference(x, residual, weight, eps=1e-6, *, zero_centered=False):
    """FP64 normalization of the FP32 sum; residual storage rounds separately.

    Weight offset has the public FP32 semantics, including FP32 weights. The
    normalization oracle never reads the lower-precision stored residual.
    """
    values = (x.float() + residual.float()).double()
    scale = (weight.float() + 1.0 if zero_centered else weight.float()).double()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return normalized * scale, values.to(residual.dtype)


def vllm_norm_native(x, residual, weight, eps=1e-6, *, zero_centered=False):
    # Independently expressed from vLLM 0.31.0 ir/ops/layernorm.py and
    # GemmaRMSNorm.forward_native. Standard rounds normalized values to weight's
    # dtype BEFORE multiplying; Gemma creates weight.float()+1 and stays FP32.
    scale = weight.float() + 1.0 if zero_centered else weight
    summed = residual.float() + x.float()
    inverse = torch.rsqrt(torch.mean(torch.square(summed), dim=-1, keepdim=True) + eps)
    output = (summed * inverse).to(scale.dtype) * scale
    return output.to(x.dtype), summed.to(x.dtype)


def norm_market_formulas(x, residual, weight, eps=1e-6, *, zero_centered=False):
    """Supplement real GPU markets with both public FP32 rounding conventions."""
    summed = x.float() + residual.float()
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    inverse = torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
    return {
        "vllm_formula": vllm_norm_native(x, residual, weight, eps, zero_centered=zero_centered),
        "flashinfer_formula": ((summed * inverse * scale).to(x.dtype), summed.to(residual.dtype)),
    }


def norm_pair_metrics(pair, exact, residual_exact):
    return {
        "normalized": error_metrics(pair[0], exact),
        "residual": error_metrics(pair[1], residual_exact),
    }


def reference(x):
    gate, up = x.double().chunk(2, dim=-1)
    # Stable for large finite negative inputs; double covers the BF16 exponent range.
    return (gate * torch.sigmoid(gate)) * up


def market_formulas(x):
    gate, up = x.float().chunk(2, dim=-1)
    exp = torch.exp(-gate)
    return {
        "sglang_formula": ((gate / (1.0 + exp)) * up).to(x.dtype),
        "inductor_formula": ((gate * (1.0 / (1.0 + exp))) * up).to(x.dtype),
    }


def error_metrics(actual, exact):
    """Error against unrounded FP64, in output ULP and absolute RMS units.

    A binade's spacing is used at powers of two, including gradual underflow.
    Nonfinite mismatches fail instead of disappearing in a finite-only reduction.
    """
    if actual.numel() == 0:
        return {"max_ulp": 0.0, "rms": 0.0, "rms_ulp": 0.0}
    info = torch.finfo(actual.dtype)
    expected = exact.to(actual.dtype)
    same_nonfinite = (~torch.isfinite(expected)) & (
        (actual == expected) | (torch.isnan(actual) & torch.isnan(expected))
    )
    valid = torch.isfinite(actual) & torch.isfinite(expected)
    if not bool((valid | same_nonfinite).all()):
        return {"max_ulp": float("inf"), "rms": float("inf"), "rms_ulp": float("inf")}
    error = torch.where(valid, (actual.double() - exact).abs(), 0.0)
    exponent = torch.floor(torch.log2(exact.abs().clamp_min(info.tiny)))
    spacing = torch.exp2(exponent) * info.eps
    ulps = error / spacing
    return {
        "max_ulp": ulps.max().item(),
        "rms": error.square().mean().sqrt().item(),
        "rms_ulp": ulps.square().mean().sqrt().item(),
    }


# Calibrated by `python -m benchmark.kernel_tuning.norm_act_tau` on torch 2.13.0+cpu
# (one thread, 208.1s): BF16/FP16; seeds 0..31 (0..7 above 262144 output
# elements), plus 20261015+M for every random shape. SiLU widths 8/128/12288 x
# M=1/3/4/16/17/64/256/1024; standard/zero-centered norm hidden 8/136/4096 x
# M=1/2/3/4/16/17/64/256/300/1024/2048. Also every finite gate encoding at width
# 12288 with up=0/0.25/1/-3/128 and shuffled gates (seed 20261016) paired with
# every finite up encoding, both dtypes. Norm includes chunk8/32/128 reductions.
# Max symmetric relative excess across equal-quality FP32 variants:
# max_ulp=0.0010644159089683303: zero norm, FP16, M=4, hidden=136, seed=18;
# rms=3.2340738394420754e-5: zero norm, FP16, M=2, hidden=136, seed=24;
# rms_ulp=9.633508800677149e-6: SiLU, FP16, M=1, width=128, seed=21.
# Round each maximum UP to a power of two; this is an empirical calibration,
# not a universal bound on other input distributions or device math libraries.
TAU = {"max_ulp": 2.0**-9, "rms": 2.0**-14, "rms_ulp": 2.0**-16}


def no_worse(ours, markets, *, tau=TAU):
    """Every metric must beat the best comparator up to measured relative noise.

    Callers supply metric dictionaries for the same input and FP64 reference.
    The best comparator can differ by metric. Only exact-zero comparator errors
    receive the tiny absolute floor; nonzero errors get no additive slack.
    FP32 activation callers supply their per-kind calibration explicitly; the
    default preserves the existing BF16/FP16 gate.
    """
    for key in TAU:
        best = min(m[key] for m in markets)
        limit = best * (1 + tau[key]) if best else 1e-12
        if not math.isfinite(ours[key]) or not ours[key] <= limit:
            return False
    return True


def sglang_fp32_residual_native(x, residual, weight, eps=1e-6, *, zero_centered=False):
    """Torch eager FP32-residual comparator, with independent output allocations.

    Design reference: SGLang 72730d6, python/sglang/srt/layers/layernorm.py:821-870
    (supplied source export), RMSNorm.forward_native: FP32 add/clone, variance,
    and final output-multiply
    branches (cast_x_before_out_mul=False). Expressed independently, not copied.
    The zero-centered variant uses the same arithmetic with FP32 weight + 1.
    """
    summed = torch.add(x.float(), residual.float())
    saved_residual = summed.clone()
    second_moment = torch.mean(torch.pow(summed, 2), dim=-1, keepdim=True)
    normalized = torch.mul(summed, torch.rsqrt(second_moment + eps))
    scale = weight.float() + 1.0 if zero_centered else weight
    return torch.mul(normalized, scale).to(x.dtype), saved_residual


def fp32_residual_formulas(x, residual, weight, eps=1e-6, *, zero_centered=False):
    """Same-contract comparators; BF16-residual market kernels are timing only."""
    summed = x.float() + residual
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    eager = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * scale).to(x.dtype)
    return {
        "torch_fp32": (eager, summed),
        "sgl_fp32": sglang_fp32_residual_native(
            x, residual, weight, eps, zero_centered=zero_centered
        ),
    }
