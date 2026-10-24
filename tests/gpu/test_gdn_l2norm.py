# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Fixed FLA L2 norm: precision, compiled spills, and cold-cache training stack usage."""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
from contextlib import nullcontext
from itertools import accumulate
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetTraining, _cute, _fla, _fla_fixed, _l2norm

from ..support.child import run_child

SEEDS = 8


@pytest.fixture(autouse=True)
def _cuda_required():
    # Defer the CUDA check until execution; collecting this module never touches CUDA.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _configure_fla():
    _fla.load("GDN l2norm", torch.device("cuda", torch.cuda.current_device()))


def _bf16_ulp_errors(actual, expected):
    """RMS and maximum error in elementwise BF16 ULPs at the FP64 reference value."""
    _, exponent = torch.frexp(expected.abs())
    # BF16 has seven fraction bits; include its subnormal spacing at zero.
    ulp = torch.exp2((exponent - 8).double()).clamp_min(2.0**-133)
    ulp = torch.where(expected == 0, 2.0**-133, ulp)
    error = (actual.double() - expected) / ulp
    return error.square().mean().sqrt().item(), error.abs().max().item()


def _backward_reference(y, rstd, dy):
    # Use exactly the rounded operands the kernel reads, including FP32 rstd.
    y, r, dy = y.double(), rstd.double().unsqueeze(-1), dy.double()
    return dy * r - (dy * y).sum(-1, keepdim=True) * y * r


def _dx_bound_use(dx, y, rstd, dy):
    """Largest |dx - reference| over its allowance: one BF16 ULP of the reference (the single
    output rounding) plus FP32 evaluation error of dy r - (sum dy y) y r. Where the two terms
    cancel, the FP32 error scales with the terms, not with dx, so a bare ULP bound at the
    reference value is unattainable (measured 1.5-1.8 ULP there for this kernel); 2^-20 times
    the term magnitudes allows 16x the FP32 unit roundoff over a 128-term sum."""
    expected = _backward_reference(y, rstd, dy)
    _, exponent = torch.frexp(expected.abs())
    ulp = torch.where(expected == 0, 2.0**-133, torch.exp2((exponent - 8).double()))
    y, r, dy64 = y.double(), rstd.double().unsqueeze(-1), dy.double()
    terms = dy64.abs() * r + (dy64 * y).abs().sum(-1, keepdim=True) * y.abs() * r
    allowance = ulp + 2.0**-20 * terms
    return ((dx.double() - expected).abs() / allowance).max().item()


def _l2norm_errors(y, rstd, dx, dy, y_ref, r_ref):
    y_rms, y_peak = _bf16_ulp_errors(y, y_ref)
    dx_rms, dx_peak = _bf16_ulp_errors(dx, _backward_reference(y, rstd, dy))
    return {
        "y_rms_ulp": y_rms,
        "y_ulp": y_peak,
        "dx_rms_ulp": dx_rms,
        "dx_ulp": dx_peak,
        "dx_bound_use": _dx_bound_use(dx, y, rstd, dy),
        "rstd_relative": ((rstd.double() - r_ref).abs() / r_ref).max().item(),
    }


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 1000, 8193])
def test_l2norm_matches_fla_and_fp64(tokens):
    """Absolute bounds hold per seed; comparisons use the eight-seed spread rule.

    Fixed versus autotuned reduction orders cause symmetric rounding noise, including
    in FLA's own outputs as its timing-dependent config choice changes. Per metric:
    mean(ours) <= mean(FLA) * (1 + 2 * stdev(ours/FLA) / sqrt(SEEDS)), with sample
    stdev (n-1), zero when every ratio is equal. See the fla-core capture precision rule.
    """
    _configure_fla()
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd

    errors = {name: [] for name in ("y_rms_ulp", "y_ulp", "dx_rms_ulp", "dx_bound_use")}
    for seed in range(SEEDS):
        generator = torch.Generator(device="cuda").manual_seed(42 + seed)
        x = torch.randn(tokens, 16, 128, device="cuda", generator=generator)
        x[:, 0] *= 1e-3
        x[:, 1] *= 30
        x = x.bfloat16()
        dy = torch.randn(x.shape, device="cuda", generator=generator).bfloat16()
        y, rstd = _l2norm.l2norm_forward(x)
        dx = _l2norm.l2norm_backward(y, rstd, dy)
        r_ref = (x.double().square().sum(-1) + 1e-6).rsqrt()
        y_ref = x.double() * r_ref.unsqueeze(-1)

        ours = _l2norm_errors(y, rstd, dx, dy, y_ref, r_ref)
        # The unmodified autotuned baseline can raise this process's stack limit;
        # the cold-cache regression below deliberately runs in a fresh child process.
        fla_y, fla_r = l2norm_fwd(x)
        fla_dx = l2norm_bwd(fla_y, fla_r, dy)
        flas = _l2norm_errors(fla_y, fla_r, fla_dx, dy, y_ref, r_ref)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert dx.shape == y.shape and dx.dtype == y.dtype
        assert rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32
        assert ours["y_ulp"] <= 1, (seed, ours)
        assert ours["dx_bound_use"] <= 1, (seed, ours)
        assert ours["rstd_relative"] <= 1e-6, (seed, ours)
        for name, samples in errors.items():
            samples.append((ours[name], flas[name]))

    report = {}
    for name, samples in errors.items():
        ours, flas = zip(*samples, strict=True)
        # Guard only the ratio's zero denominator, without adding gate slack.
        ratios = [o / max(f, 2.0**-126) for o, f in samples]
        report[name] = {
            "fixed": statistics.mean(ours),
            "autotuned": statistics.mean(flas),
            "margin": 2 * statistics.stdev(ratios) / SEEDS**0.5,
        }
    print(json.dumps({"tokens": tokens, "seeds": SEEDS, "metrics": report}), flush=True)
    for name, measured in report.items():
        assert measured["fixed"] <= measured["autotuned"] * (1 + measured["margin"]), (
            name,
            measured,
            errors[name],
        )


@pytest.mark.parametrize(("k_heads", "v_heads"), [(16, 32), (8, 24), (4, 12), (16, 16)])
@pytest.mark.parametrize("lengths", [[1, 63, 65, 130], [1, 63, 65, 2047, 2111, 1025]])
def test_wy_backward_fixed_config_is_stable(monkeypatch, k_heads, v_heads, lengths):
    """Qualify 4 warps / 1 stage against Blackwell's 2/4 workaround for FLA #913.

    Repeat twenty synchronized launches with odd sequence tails and head ratios 1-3.
    The large case launches at least 1024 blocks even with only twelve value heads.
    Both configurations accumulate in FP32; require <= 1e-2 relative RMS per output.
    """
    _configure_fla()
    generator = torch.Generator(device="cuda").manual_seed(42)
    tokens, bt = sum(lengths), 64
    chunks = [(seq, c) for seq, n in enumerate(lengths) for c in range((n + bt - 1) // bt)]
    # Pack independently generated full chunks into FLA's [B, T, HV, ...] layout.
    rows = torch.tensor(
        [
            i * bt + t
            for i, (s, c) in enumerate(chunks)
            for t in range(min(bt, lengths[s] - c * bt))
        ],
        device="cuda",
    )
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda")
    indices = torch.tensor(chunks, dtype=torch.int32, device="cuda")
    blocks = len(chunks) * v_heads  # B=1, grid=(NT, B*HV)
    if tokens >= 4096:
        assert blocks >= 1024
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, k_heads, 128, generator=generator, device="cuda"), dim=-1
    ).bfloat16()
    v = torch.randn(1, tokens, v_heads, 128, generator=generator, device="cuda").bfloat16()
    beta = 0.05 + 0.9 * torch.rand(1, tokens, v_heads, generator=generator, device="cuda")
    decays = -(
        0.01 + 0.09 * torch.rand(len(chunks), bt, v_heads, generator=generator, device="cuda")
    )
    g = decays.cumsum(1).flatten(0, 1)[rows].unsqueeze(0)
    # Unit diagonal plus small random strict-lower entries: well-conditioned 64x64 A.
    matrices = 0.01 * torch.randn(len(chunks), v_heads, bt, bt, generator=generator, device="cuda")
    matrices = (matrices.tril(-1) + torch.eye(bt, device="cuda")).bfloat16()
    A = matrices.permute(0, 2, 1, 3).reshape(-1, v_heads, bt)[rows].unsqueeze(0)
    dw = torch.randn(v.shape, generator=generator, device="cuda").bfloat16()
    du = torch.randn(v.shape, generator=generator, device="cuda").bfloat16()
    arguments = {
        "k": k,
        "v": v,
        "beta": beta,
        "A": A,
        "dw": dw,
        "du": du,
        "g": g,
        "cu_seqlens": cu,
        "chunk_indices": indices,
    }
    first = None
    for repeat in range(20):
        outputs = _fla_fixed.prepare_wy_repr_bwd(**arguments)
        torch.cuda.synchronize()  # Surface asynchronous faults on every launch.
        if first is None:
            first = outputs
        for name, actual, original in zip(("dk", "dv", "db", "dg"), outputs, first, strict=True):
            assert torch.isfinite(actual).all(), (name, repeat)
            bits = torch.int16 if actual.element_size() == 2 else torch.int32
            assert torch.equal(actual.view(bits), original.view(bits)), (name, repeat)

    # The same wrapper calls _fla_fixed._launch on the same FLA JIT kernel and tiles;
    # only warps/stages change, bypassing autotuning for the Blackwell baseline too.
    with monkeypatch.context() as baseline:
        baseline.setitem(
            _fla_fixed.CONFIGS, "prepare_wy_repr_bwd_kernel", {"num_warps": 2, "num_stages": 4}
        )
        theirs = _fla_fixed.prepare_wy_repr_bwd(**arguments)
        torch.cuda.synchronize()
    relative_rms = {}
    for name, actual, expected in zip(("dk", "dv", "db", "dg"), first, theirs, strict=True):
        assert torch.isfinite(expected).all(), name
        actual_fp32, expected_fp32 = actual.float(), expected.float()
        relative_rms[name] = (
            (actual_fp32 - expected_fp32).square().mean().sqrt()
            / expected_fp32.square().mean().sqrt().clamp_min(torch.finfo(torch.float32).tiny)
        ).item()
    print(
        json.dumps(
            {
                "geometry": [k_heads, v_heads],
                "lengths": lengths,
                "blocks": blocks,
                "relative_rms": relative_rms,
            }
        ),
        flush=True,
    )
    for name, measured in relative_rms.items():
        assert measured <= 1e-2, (name, measured)


def test_compiled_configs_do_not_spill():
    _configure_fla()
    x = torch.randn(17, 16, 128, device="cuda", dtype=torch.bfloat16)
    y, dx, dy = torch.empty_like(x), torch.empty_like(x), torch.ones_like(x)
    rstd = torch.empty(x.shape[:-1], device="cuda", dtype=torch.float32)
    forward = _l2norm._launch_forward(x, y, rstd)
    backward = _l2norm._launch_backward(y, rstd, dy, dx)
    torch.cuda.synchronize()
    for name, kernel in (("forward", forward), ("backward", backward)):
        assert kernel.n_spills == 0, (name, kernel.n_spills)


def _stack_limit():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        # Torch has already initialized CUDA. Use its loaded libcudart (including
        # wheels whose runtime lives in nvidia/cu13/lib rather than torch/lib).
        libraries = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line and line.split()[-1].startswith("/")
        }
        assert len(libraries) == 1, libraries
        runtime = ctypes.CDLL(libraries.pop())
        get_limit = runtime.cudaDeviceGetLimit
        get_limit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        get_limit.restype = ctypes.c_int
        value = ctypes.c_size_t()
        status = get_limit(ctypes.byref(value), 0)  # cudaLimitStackSize = 0
        assert status == 0, status
        return value.value
    status, value = cudart.cudaDeviceGetLimit(cudart.cudaLimit.cudaLimitStackSize)
    assert status == cudart.cudaError_t.cudaSuccess, status
    return int(value)


def _cold_cache_training(provider, k_heads, v_heads):
    """Fresh-process check: every loaded kernel, all autotuners, both varlen cases."""
    import triton.compiler.compiler as compiler
    from triton.runtime.autotuner import Autotuner

    torch.cuda.init()
    torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    before = _stack_limit()  # before any Triton launch, including provider construction
    capability = torch.cuda.get_device_capability()
    reason = _fla.WRONG_ON.get((GatedDeltaNetTraining.__name__, capability))
    if reason is not None or (provider == "cute" and capability not in _cute.CAPABILITIES):
        print(json.dumps({"skip": reason or "CuTe requires sm12x"}))
        return
    loaded, autotunes = {}, []
    original_init = compiler.CompiledKernel._init_handles
    original_run = Autotuner.run

    def init(kernel):
        original_init(kernel)
        loaded[(kernel.name, kernel.hash)] = {
            "name": kernel.name,
            "warps": kernel.metadata.num_warps,
            "regs": kernel.n_regs,
            "spills": kernel.n_spills,
        }

    def run(tuner, *args, **kwargs):
        autotunes.append(getattr(tuner.fn, "__name__", type(tuner.fn).__name__))
        return original_run(tuner, *args, **kwargs)

    force = (
        patch.object(_cute, "training", return_value=None)
        if provider == "fla-core"
        else nullcontext()
    )
    with (
        patch.object(compiler.CompiledKernel, "_init_handles", init),
        patch.object(Autotuner, "run", run),
        force,
    ):
        op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
        assert op.provider == provider
        conv_dim = (2 * k_heads + v_heads) * 128
        width = conv_dim + v_heads * 128 + 2 * v_heads
        generator = torch.Generator().manual_seed(42)
        weights = (
            (0.5 * torch.randn(conv_dim, 4, generator=generator)).bfloat16().cuda(),
            (torch.rand(v_heads, generator=generator) * 2.8 - 0.7).cuda(),
            (-2.0 + 4.0 * torch.rand(v_heads, generator=generator)).cuda(),
            (1 + 0.1 * torch.randn(128, generator=generator)).cuda(),
        )
        limits = []
        for lengths in ([64, 130, 7], [1000, 70]):
            tokens = sum(lengths)
            cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda")
            leaf = (
                torch.randn(tokens, width, generator=generator)
                .bfloat16()
                .cuda()
                .requires_grad_(True)
            )
            dy = torch.randn(tokens, v_heads * 128, generator=generator).bfloat16().cuda()
            y = op(leaf, *weights, cu, max_sequence_length=max(lengths))
            torch.cuda.synchronize()
            limits.append(_stack_limit())
            torch.autograd.grad(y, leaf, dy)
            torch.cuda.synchronize()
            limits.append(_stack_limit())
    report = {
        "provider": provider,
        "geometry": [k_heads, v_heads],
        "before": before,
        "limits": limits,
        "autotunes": autotunes,
        "loaded": list(loaded.values()),
    }
    print(json.dumps(report), flush=True)
    assert loaded, "CompiledKernel hook did not observe any launches"
    assert not autotunes, autotunes
    spilled = [record for record in loaded.values() if record["spills"]]
    assert not spilled, spilled
    assert all(limit == before for limit in limits), (before, limits)


@pytest.mark.parametrize(
    ("provider", "k_heads", "v_heads"),
    [
        ("cute", 16, 32),
        ("cute", 16, 48),
        ("cute", 8, 16),
        ("fla-core", 8, 24),
        ("fla-core", 4, 8),
        ("fla-core", 4, 12),
        ("fla-core", 16, 32),
    ],
)
def test_cold_cache_training_keeps_the_stack_limit(tmp_path, provider, k_heads, v_heads):
    cache = tmp_path / "triton"
    cache.mkdir()
    environment = os.environ.copy()
    environment.pop("FLA_CACHE_MODE", None)
    environment["TRITON_CACHE_DIR"] = str(cache)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), *sys.path]
    )
    environment.pop("CTI_GDN_CUTE", None)
    result = run_child(
        [
            sys.executable,
            "-c",
            "from tests.gpu.test_gdn_l2norm import _cold_cache_training; "
            f"_cold_cache_training({provider!r}, {k_heads}, {v_heads})",
        ],
        env=environment,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if "skip" in report:
        pytest.skip(report["skip"])
    print(json.dumps(report))
    assert report["provider"] == provider
    assert report["autotunes"] == []
    assert all(item["spills"] == 0 for item in report["loaded"]), report
    assert all(limit == report["before"] for limit in report["limits"]), report
