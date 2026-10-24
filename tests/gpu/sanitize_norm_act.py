# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Bounded compute-sanitizer driver for norm/activation CuTe configurations.

Run ``python -m tests.gpu.sanitize_norm_act [--full] [--only activation|norm]``
under memcheck, racecheck or synccheck. Pair shapes with row counts instead of
crossing the full matrix so even --full stays below 1500 operator calls.
No GPU work happens at import, and pytest collects no tests from this module.
"""

import argparse
from dataclasses import replace

import torch

from ctikernels.activation import inference as act
from ctikernels.activation._dispatch import _BANDS
from ctikernels.activation._dispatch import Config as ActConfig
from ctikernels.norm import inference as norm
from ctikernels.norm._dispatch import TABLE
from ctikernels.norm._dispatch import Config as NormConfig


def run_act_configs():
    configs = [config for bands in _BANDS.values() for config in bands.values()]
    configs += [
        ActConfig(128, 4, False, 8),
        ActConfig(64, 2, True, 4),
        ActConfig(256, 1, True, 2),
        ActConfig(512, 1, False, 8),
        ActConfig(128, 2, True, 4, "early"),
    ]
    return list(dict.fromkeys(configs))


def run_norm_configs():
    configs = [
        entry.config
        for table in TABLE.values()
        for bands in table.values()
        for entry in bands.values()
        if entry.provider == "cute"
    ]
    configs += [
        NormConfig(128, 1, False),
        NormConfig(128, 2, True, trigger="early"),
        NormConfig(128, 4, True, trigger="early"),
        NormConfig(256, 2, True),
        NormConfig(512, 4, True),
        NormConfig(1024, 4, False),
        NormConfig(1024, 1, True),
    ]
    return list(
        dict.fromkeys(
            replace(config, static_weight=static) for config in configs for static in (False, True)
        )
    )


def run_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def run_activation(full):
    cases = [(m, 12288) for m in (1, 3, 4, 16, 64, 257)]
    cases += [(3, 8), (4, 24), (257, 4104)]
    if full:
        cases += [(16, 17408), (64, 9728), (1024, 12288)]
    configs = run_act_configs()
    n = 0
    for config in configs:
        op = act.SiluAndMul(provider="cute", config=config)
        for m, width in cases:
            for dtype in (torch.bfloat16, torch.float16):
                x = torch.randn(m, 2 * width, device="cuda", dtype=dtype) * 4
                # Mix ordinary and negative-tail lanes in the same vector.
                gates = torch.tensor(
                    [-81, 1, -90, -1, -100, -80, -120, 3], device="cuda", dtype=dtype
                )
                x[:, :8] = gates
                x[:, width : width + 8] = 128
                out = torch.empty(m, width, device="cuda", dtype=dtype)
                op(x, out)
                torch.cuda.synchronize()
                g, u = x.float().chunk(2, -1)
                expected = (torch.nn.functional.silu(g) * u).to(dtype)
                # SGLang's g / (1 + expf(-g)) * u is F.silu(g) * u, bit for bit.
                run_bits(out, expected)
                n += 1
    print(f"SANITIZE activation calls={n} configs={len(configs)}", flush=True)
    return n


def run_norm_case(op, m, hidden, zero, dtype, weight_dtype):
    x = torch.randn(m, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=weight_dtype)
    summed = x.float() + residual.float()
    rounded = summed.to(dtype)
    scale = weight.float() + 1 if zero else weight.float()
    expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6) * scale
    # Exercise all four compiled has_sum/store_residual combinations across the
    # sweep, with both stores for sum_out in every shape/weight/config case.
    variants = [(False, True), (True, True), (True, False)]
    n = 0
    for has_sum, store in variants:
        xx, rr = x.clone(), residual.clone()
        aux = torch.full((m, 3 * hidden), 17.0, device="cuda", dtype=dtype)
        sum_out = aux[:, hidden : 2 * hidden] if has_sum else None
        op(xx, rr, weight, 1e-6, zero_centered=zero, store_residual=store, sum_out=sum_out)
        torch.cuda.synchronize()
        err = (xx.float() - expected).abs().max().item()
        assert err < 0.1 * max(1.0, expected.abs().max().item()), (op.config, hidden, m, err)
        run_bits(rr, rounded if store else residual)
        if has_sum:
            run_bits(sum_out, rounded)
        else:
            assert (aux[:, hidden : 2 * hidden] == 17).all()
        assert (aux[:, :hidden] == 17).all() and (aux[:, 2 * hidden :] == 17).all()
        n += 1
    return n


def run_norm(full):
    cases = [(1, 8), (3, 128), (16, 2560), (3, 4096), (64, 4096), (257, 4104), (4, 5120)]
    if full:
        cases += [(16, 8192), (3, 12288), (1024, 4096)]
    configs = run_norm_configs()
    n = 0
    for config in configs:
        op = norm.AddRMSNorm(provider="cute", config=config, static_weight=config.static_weight)
        for m, hidden in cases:
            if not config.admits(hidden):
                continue
            dtype = (
                torch.float16 if hidden == 128 or (hidden == 4096 and m == 3) else torch.bfloat16
            )
            for zero in (False, True):
                n += run_norm_case(op, m, hidden, zero, dtype, dtype)
        n += run_norm_case(op, 3, 128, True, torch.bfloat16, torch.float32)
        # No auxiliary output and no residual store uses the fourth flag pair.
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        residual, weight = torch.randn_like(x), torch.ones(128, device="cuda", dtype=x.dtype)
        before = residual.clone()
        op(x, residual, weight, 1e-6, store_residual=False)
        torch.cuda.synchronize()
        run_bits(residual, before)
        n += 1
    print(f"SANITIZE norm calls={n} configs={len(configs)}", flush=True)
    return n


def run_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--only", choices=("activation", "norm"))
    args = parser.parse_args()
    torch.manual_seed(0)
    calls = 0
    if args.only != "norm":
        calls += run_activation(args.full)
    if args.only != "activation":
        calls += run_norm(args.full)
    assert calls < 2000, calls
    print(f"SANITIZE OK calls={calls}", flush=True)


if __name__ == "__main__":
    run_main()
