# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Measure FP32 evaluation-order noise against the unrounded FP64 oracles.

Run from the repository root with:
    python -m benchmark.kernel_tuning.norm_act_tau [--device cuda]
    python -m benchmark.kernel_tuning.norm_act_tau --fp32-activation

--fp32-activation selects a separate FP32-output calibration on CUDA by default
(--device cpu is a diagnostic only). It covers every support-range shape for
SiLU, GELU, tanh-GELU and relu2, including empty inputs, using shared grids and
comparators from norm_act_reference. Draws are native to the requested device,
matching the GPU tests: seeds 0..31 (0..7 above 262144 output elements), plus
20261016 + M + width. Explicit --seeds must be >=8. Each kind gets its own
per-metric maximum and rounded-up TAU; no CuTe output enters the measurement.

The default sweep covers BF16/FP16 SiLU widths 8/128/12288 at
M=1/3/4/16/17/64/256/1024, and standard/zero-centered add-RMSNorm hidden
8/136/4096 at M=1/2/3/4/16/17/64/256/300/1024/2048. Random shapes use seeds
0..31 (0..7 above 262144 output elements), plus 20261015 + M in every case.
Inputs are independent standard-normal FP32 CPU draws rounded to the input
dtype, then copied to the requested device; seeds do not imply identical
CUDA-native RNG draws. Every finite gate encoding is also tested at width
12288 with up=0/0.25/1/-3/128 and a shuffled-gate/all-encoding-up pass.
All variants compute in FP32 and round only the final output to 16 bits;
only evaluation order, reduction order, and rsqrt versus reciprocal sqrt vary.
SiLU variants share the split-exponent negative-tail treatment to avoid
calibrating overflow/underflow bugs as rounding noise. Residual storage is
identical across norm variants and contributes no noise.

For each metric and identical sample, measure every unordered pair both ways:
max(a/b - 1, b/a - 1, 0). A zero denominator alone uses 1e-12. Maximize over
pairs, seeds, operations, dtypes and shapes, then round UP to a power of two.
This is an empirical noise floor for this sweep, not a universal error bound.
Only torch and the Python standard library are required. The default 16-bit
sweep needs no GPU; FP32 activation qualification needs the target GPU.
"""

import argparse
import itertools
import math
import time

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.norm_act_reference import (
    FP32_ACTIVATION_KINDS,
    FP32_ACTIVATION_ROWS,
    FP32_ACTIVATION_WIDTHS,
    RELU2_WIDTHS,
    error_metrics,
    fp32_activation_formulas,
    norm_reference,
    reference,
    support_range_formula,
)

METRICS = ("max_ulp", "rms", "rms_ulp")
SILU_ROWS = (1, 3, 4, 16, 17, 64, 256, 1024)
NORM_ROWS = (1, 2, 3, 4, 16, 17, 64, 256, 300, 1024, 2048)
SILU_WIDTHS = (8, 128, 12288)
NORM_HIDDEN = (8, 136, 4096)
LARGE_ELEMENTS = 262144
LARGE_SEEDS = 8
DEFAULT_SEEDS = 32


def silu_variants(x):
    gate, up = x.float().chunk(2, dim=-1)
    denominator = 1.0 + torch.exp(-gate)
    divided = (gate / denominator) * up
    product = gate * up
    # Multiplication before division can overflow even when the result is finite.
    multiplied = torch.where(torch.isfinite(product), product / denominator, divided)
    half_exp = torch.exp(gate.clamp_max(0) * 0.5)
    tail = ((gate * half_exp) * up) * half_exp
    return {
        name: torch.where(gate < -80, tail, value).to(x.dtype)
        for name, value in {
            "divide_then_mul": divided,
            "sigmoid_then_mul": (gate * torch.sigmoid(gate)) * up,
            "mul_then_divide": multiplied,
        }.items()
    }


def norm_variants(x, residual, weight, *, zero_centered):
    values = x.float() + residual.float()
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    squares = values.square()
    tree = F.pad(squares, (0, (1 << (squares.shape[-1] - 1).bit_length()) - squares.shape[-1]))
    while tree.shape[-1] > 1:
        tree = tree[..., 0::2] + tree[..., 1::2]
    sums = {
        "native": squares.sum(-1, keepdim=True),
        "pairwise": tree,
    }
    for chunk in (8, 32, 128):
        padded = F.pad(squares, (0, (-squares.shape[-1]) % chunk))
        sums[f"chunk{chunk}"] = (
            padded.reshape(*squares.shape[:-1], -1, chunk).sum(-1).sum(-1, keepdim=True)
        )
    result = {}
    for name, total in sums.items():
        variance = total / values.shape[-1] + 1e-6
        for inverse_name, inverse in (
            ("rsqrt", torch.rsqrt(variance)),
            ("sqrt_reciprocal", 1.0 / torch.sqrt(variance)),
        ):
            result[f"{name}/{inverse_name}"] = (values * inverse * scale).to(x.dtype)
    return result


def pair_excess(a, b):
    """Symmetric relative excess; exact-zero metrics use only the absolute floor."""
    return max(a / (b if b else 1e-12) - 1, b / (a if a else 1e-12) - 1, 0.0)


def finite_encoding_inputs(dtype, *, width=12288, device="cpu"):
    """Mirror the shipped-config GPU test, including its shuffled paired pass."""
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    encodings = encodings[torch.isfinite(encodings)]
    padding = (-encodings.numel()) % width
    gates = F.pad(encodings, (0, padding)).view(-1, width).to(device)
    generator = torch.Generator().manual_seed(20261016)
    shuffled = encodings[torch.randperm(encodings.numel(), generator=generator)]
    shuffled = F.pad(shuffled, (0, padding)).view_as(gates).to(device)
    for multiplier in (0.0, 0.25, 1.0, -3.0, 128.0, None):
        gate = gates if multiplier is not None else shuffled
        up = torch.full_like(gates, multiplier) if multiplier is not None else gates
        yield str(multiplier) if multiplier is not None else "shuffled", torch.cat((gate, up), -1)


def _update(row, exact, variants, seed):
    metrics = {name: error_metrics(value, exact) for name, value in variants.items()}
    if any(not math.isfinite(v) for metric in metrics.values() for v in metric.values()):
        raise ValueError(f"nonfinite calibration metrics: {row} seed={seed}")
    for a, b in itertools.combinations(metrics, 2):
        for key in METRICS:
            excess = pair_excess(metrics[a][key], metrics[b][key])
            if excess > row["maxima"][key]:
                row["maxima"][key] = excess
                row["witnesses"][key] = (seed, a, b)


def _measurement(op, dtype, m, width, layout, seed_count):
    return {
        "op": op,
        "dtype": str(dtype),
        "m": m,
        "width": width,
        "layout": layout,
        "seed_count": seed_count,
        "maxima": dict.fromkeys(METRICS, 0.0),
        "witnesses": dict.fromkeys(METRICS, None),
    }


@torch.no_grad()
def measure(
    *,
    seeds=None,
    rows=None,
    device="cpu",
    silu_widths=SILU_WIDTHS,
    norm_hidden=NORM_HIDDEN,
    include_encodings=True,
    progress=None,
):
    """Return per-shape maxima and seed/pair witnesses; explicit seeds bypass the cap.

    Always add the GPU test's seed 20261015 + M. ``rows`` overrides both operation
    grids for small CPU tests. The default seed cap applies only to large shapes.
    """
    seeds = None if seeds is None else tuple(seeds)
    rows = None if rows is None else tuple(rows)
    if seeds == () or rows == () or (rows is not None and any(m < 1 for m in rows)):
        raise ValueError("measurement requires seeds and positive row counts")
    measurements = []

    def record(row):
        measurements.append(row)
        if progress is not None:
            progress(row)

    for op in ("silu_mul", "add_rms_norm", "zero_add_rms_norm"):
        widths = silu_widths if op == "silu_mul" else norm_hidden
        op_rows = rows if rows is not None else (SILU_ROWS if op == "silu_mul" else NORM_ROWS)
        for dtype, width, m in itertools.product((torch.bfloat16, torch.float16), widths, op_rows):
            count = LARGE_SEEDS if m * width > LARGE_ELEMENTS else DEFAULT_SEEDS
            base_seeds = range(count) if seeds is None else seeds
            sample_seeds = tuple(dict.fromkeys((*base_seeds, 20261015 + m)))
            row = _measurement(op, dtype, m, width, "random", len(sample_seeds))
            for seed in sample_seeds:
                generator = torch.Generator(device="cpu").manual_seed(seed)

                def sample(*shape, generator=generator, dtype=dtype):
                    return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
                        dtype=dtype, device=device
                    )

                if op == "silu_mul":
                    x = sample(m, 2 * width)
                    exact, variants = reference(x), silu_variants(x)
                else:
                    x, residual, weight = sample(m, width), sample(m, width), sample(width)
                    zero = op == "zero_add_rms_norm"
                    exact, _ = norm_reference(x, residual, weight, zero_centered=zero)
                    variants = norm_variants(x, residual, weight, zero_centered=zero)
                _update(row, exact, variants, seed)
            record(row)
    if include_encodings:
        for dtype in (torch.bfloat16, torch.float16):
            for layout, x in finite_encoding_inputs(dtype, device=device):
                row = _measurement("silu_mul", dtype, x.shape[0], 12288, layout, 1)
                _update(
                    row, reference(x), silu_variants(x), 20261016 if layout == "shuffled" else None
                )
                record(row)
    return measurements


def measured_maxima(measurements):
    return {key: max(row["maxima"][key] for row in measurements) for key in METRICS}


@torch.no_grad()
def measure_fp32_activation(*, seeds=None, device="cuda", progress=None):
    """Calibrate all gated FP32 shapes with >=8 independent seeds per shape.

    Device-native draws also reproduce each GPU support-range test's seed.
    CPU results are diagnostic, not a substitute for CUDA math-library noise.
    """
    seeds = None if seeds is None else tuple(dict.fromkeys(seeds))
    if seeds is not None and len(seeds) < LARGE_SEEDS:
        raise ValueError("FP32 activation calibration requires at least 8 distinct seeds")
    measurements = []
    for kind in FP32_ACTIVATION_KINDS:
        widths = RELU2_WIDTHS if kind == "relu2" else FP32_ACTIVATION_WIDTHS
        for m, width in itertools.product(FP32_ACTIVATION_ROWS, widths):
            count = LARGE_SEEDS if m * width > LARGE_ELEMENTS else DEFAULT_SEEDS
            base_seeds = range(count) if seeds is None else seeds
            sample_seeds = tuple(dict.fromkeys((*base_seeds, 20261016 + m + width)))
            row = _measurement(kind, torch.float32, m, width, "random", len(sample_seeds))
            for seed in sample_seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                x = torch.randn(
                    m,
                    width if kind == "relu2" else 2 * width,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                _update(
                    row,
                    support_range_formula(kind, x, exact=True),
                    fp32_activation_formulas(kind, x),
                    seed,
                )
            measurements.append(row)
            if progress is not None:
                progress(row)
    return measurements


def derive_taus(maxima):
    return {
        key: 2.0 ** math.ceil(math.log2(value)) if value else 0.0 for key, value in maxima.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="default: cpu; cuda with --fp32-activation")
    parser.add_argument("--fp32-activation", action="store_true")
    parser.add_argument("--seeds", type=int, help="override seed count, including large shapes")
    parser.add_argument("--rows", type=int, nargs="+", help="override both random M grids")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.fp32_activation:
        if args.rows is not None:
            parser.error("--fp32-activation always covers all gated shapes; omit --rows")
        if args.seeds is not None and args.seeds < LARGE_SEEDS:
            parser.error("--fp32-activation requires --seeds >= 8")
    if args.device is None:
        args.device = "cuda" if args.fp32_activation else "cpu"
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; --device cpu is diagnostic only for FP32 activations")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    print(f"torch={torch.__version__} device={args.device} threads={args.threads}", flush=True)
    if torch.device(args.device).type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(args.device)} CUDA={torch.version.cuda}")
    test_seed = "20261016+M+width" if args.fp32_activation else "20261015+M"
    print(f"Default seeds: 0..31; 0..7 above {LARGE_ELEMENTS} elements; plus {test_seed}")

    def report(row):
        values = " ".join(f"{key}={row['maxima'][key]:.10g}" for key in METRICS)
        print(
            f"{row['op']} {row['dtype']} M={row['m']} width={row['width']} "
            f"layout={row['layout']} seeds={row['seed_count']} {values}",
            flush=True,
        )

    if args.fp32_activation:
        measurements = measure_fp32_activation(
            seeds=None if args.seeds is None else range(args.seeds),
            device=args.device,
            progress=report,
        )
        taus = {}
        for kind in FP32_ACTIVATION_KINDS:
            rows = [row for row in measurements if row["op"] == kind]
            maxima = measured_maxima(rows)
            taus[kind] = derive_taus(maxima)
            print(f"{kind} measured maxima: {maxima}")
            for key in METRICS:
                row = max(rows, key=lambda row: row["maxima"][key])
                print(
                    f"{kind}/{key} witness: M={row['m']} width={row['width']} "
                    f"(seed, comparator_a, comparator_b)={row['witnesses'][key]}"
                )
        if torch.device(args.device).type != "cuda":
            print("CPU DIAGNOSTIC ONLY: remeasure on target GPUs before qualification.")
        print("Derived TAU_FP32 =", taus)
        print(f"Elapsed: {time.monotonic() - started:.1f}s")
        return

    measurements = measure(
        seeds=None if args.seeds is None else range(args.seeds),
        rows=args.rows,
        device=args.device,
        progress=report,
    )
    maxima = measured_maxima(measurements)
    print("Measured maxima:", maxima)
    print("Derived TAU:", derive_taus(maxima))
    for key in METRICS:
        row = max(measurements, key=lambda row: row["maxima"][key])
        print(
            f"{key} witness: {row['op']} {row['dtype']} M={row['m']} width={row['width']} "
            f"layout={row['layout']} {row['witnesses'][key]}"
        )
    print(f"Elapsed: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
