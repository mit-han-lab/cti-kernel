# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Small FP32 decomposition checks of the shared-A backward steps and the CPU reference path."""

import pytest
import torch
import torch.nn.functional as F

from ctikernels.moe import _lora_backward as _backward
from ctikernels.moe import lora as _ops
from ctikernels.moe._lora_weights import routed_slices
from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora, shared_expert_lora
from ctikernels.moe.torch import _lora_reference as _reference


def data(*, dtype=torch.float32, empty=False, remote=False, zero=None, shared=False):
    """x, w13, w2, rw, ids and the shared-A routed leaves (A [E,r,H], B [2,E,I,r], A2, B2)."""
    torch.manual_seed(123)
    t, e, h, i, r, k = 7, 3, 24, 16, 8, 2
    if shared:
        e, k = 1, 1
    if empty:
        t = 0
    rand = lambda *shape: (torch.randn(shape) * 0.12).to(dtype)
    x, w13, w2 = rand(t, h), rand(e, 2 * i, h), rand(e, h, i)
    fs = [rand(e, r, h), rand(2, e, i, r), rand(e, r, i), rand(e, h, r)]
    ids = torch.randint(e, (t, k), dtype=torch.int32)
    if not shared and t:
        ids[0] = torch.tensor([-1, e])
    if remote:
        ids.fill_(-1)
    rw = torch.rand(t, k)
    if shared:
        rw.fill_(1)
    if zero:
        for j in (0, 2) if zero == "a" else (1, 3):
            fs[j].zero_()
    return x, w13, w2, rw, ids, fs


def fp32_state(x, w13, w2, ids, fs, scale):
    a, b, a2 = fs[0], fs[1], fs[2]
    p, k, i, r = ids.numel(), ids.shape[1], w2.shape[-1], a2.shape[-2]
    c1 = torch.zeros(p, 2 * i)
    z13, z2 = torch.zeros(p, 1, r), torch.zeros(p, 1, r)
    for e in range(w13.shape[0]):
        pairs = torch.where(ids.flatten() == e)[0]
        inp = x[pairs // k]
        c1[pairs] = inp @ w13[e].T
        z13[pairs, 0] = inp @ a[e].T
        for s in range(2):
            c1[pairs, s * i : (s + 1) * i] += scale * (z13[pairs, 0] @ b[s, e].T)
        g, u = c1[pairs].chunk(2, -1)
        z2[pairs, 0] = (F.silu(g) * u) @ a2[e].T
    return {"c1": c1, "z13": z13, "z2": z2}


@pytest.mark.parametrize("zero", [None, "a", "b"])
@pytest.mark.parametrize("routes", ["mixed", "remote", "empty", "shared"])
def test_torch_decomposition_matches_fp32_autograd(zero, routes):
    x, w13, w2, rw, ids, fs = data(
        empty=routes == "empty", remote=routes == "remote", zero=zero, shared=routes == "shared"
    )
    scale = torch.tensor([0.7])
    state = fp32_state(x, w13, w2, ids, fs, scale)
    leaves = [v.requires_grad_() for v in (x, rw, *fs)]
    y = _reference.fused_experts_autograd(x, w13, w2, rw, ids, routed_slices(*fs), scale)
    dy = torch.randn_like(y)
    expected = torch.autograd.grad(y, leaves, dy)
    actual = _backward.backward_steps(dy, x, w13, w2, rw, ids, fs, scale, state, (True,) * 6)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, atol=1e-7, rtol=2e-5)
    if zero == "b" and routes in ("mixed", "shared"):
        assert all(actual[j].abs().sum() > 0 for j in (3, 5))


@pytest.mark.parametrize("shared", [False, True])
def test_validation_before_function(monkeypatch, shared):
    x, w13, w2, rw, ids, fs = data(dtype=torch.bfloat16, shared=shared)
    x.requires_grad_()
    w13.requires_grad_()

    def forbidden(*args, **kwargs):
        pytest.fail("launched before frozen-base validation")

    monkeypatch.setattr(_ops.TritonBlockLoRA, "apply", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        if shared:
            a, b, a2, b2 = fs
            factors = (torch.cat((a, a)), b[:, 0], a2[0], b2[0])
            shared_expert_lora(x, w13[0], w2[0], factors, 1.0)
        else:
            fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights(*fs), scaling=1.0)


@pytest.mark.parametrize("provider", ["torch", "auto", "cute"])
def test_cpu_keeps_reference_path(provider):
    x, w13, w2, rw, ids, fs = data(dtype=torch.bfloat16)
    x.requires_grad_()
    y = fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights(*fs), scaling=0.7, provider=provider)
    assert "TritonBlockLoRA" not in type(y.grad_fn).__name__
    y.sum().backward()
    assert x.grad is not None


def test_separate_layout_is_inference_only():
    x, w13, w2, rw, ids, fs = data(dtype=torch.bfloat16)
    a, b, a2, b2 = fs
    separate = MoELoRAWeights.separate(a, a.clone(), b[0], b[1], a2, b2)
    assert separate.layout == "separate" and MoELoRAWeights(*fs).layout == "shared_a"
    with torch.no_grad():
        fused_experts_lora(x, w13, w2, rw, ids, separate, scaling=0.7)
    x.requires_grad_()
    with pytest.raises(ValueError, match="inference-only"):
        fused_experts_lora(x, w13, w2, rw, ids, separate, scaling=0.7)


@pytest.mark.parametrize("divisor", [1, 2])
def test_strided_shrink_expand_and_outer(divisor):
    x, _, _, rw, ids, fs = data()
    inp = torch.randn(ids.numel() // divisor, 48)[:, 24:]
    factor = fs[0]
    z = _reference.backward_shrink(inp, factor, ids, row_divisor=divisor)
    base = torch.zeros(ids.numel(), 24)
    out = _reference.backward_expand(base, z, factor.transpose(-1, -2), ids, 0.7)
    grad = _reference.backward_grouped_outer(
        inp, z, ids, 3, 0.7, left_divisor=divisor, pair_scale=rw
    )
    expected = torch.zeros_like(grad)
    for p, e in enumerate(ids.flatten()):
        if 0 <= e < 3:
            zr = factor[e] @ inp[p // divisor]
            torch.testing.assert_close(z[p], zr)
            torch.testing.assert_close(out[p], 0.7 * factor[e].T @ zr)
            expected[e] += 0.7 * rw.flatten()[p] * torch.outer(inp[p // divisor], zr)
        else:
            assert not out[p].count_nonzero() and not z[p].count_nonzero()
    torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-5)


def test_base_only_reference():
    x, w13, w2, rw, ids, _ = data(dtype=torch.bfloat16)
    x.requires_grad_()
    rw.requires_grad_()
    y = fused_experts_lora(x, w13, w2, rw, ids, None, scaling=1.0)
    y.sum().backward()
    assert x.grad is not None and rw.grad is not None


@pytest.mark.parametrize("trainable", (0, 2, 3, 4, 5))
def test_reference_gradient_subsets(trainable):
    x, w13, w2, rw, ids, fs = data(dtype=torch.bfloat16)
    leaves = (x, rw, *fs)
    leaves[trainable].requires_grad_()
    y = fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights(*fs), scaling=0.7)
    y.sum().backward()
    assert [v.grad is not None for v in leaves] == [j == trainable for j in range(len(leaves))]


@pytest.mark.parametrize("bad", ["scale", "provider", "adapter", "shape"])
def test_training_validation(bad):
    x, w13, w2, rw, ids, fs = data(dtype=torch.bfloat16)
    x.requires_grad_()
    options = {"scaling": 0.7}
    if bad == "scale":
        options["scaling"] = torch.ones(2)
    elif bad == "provider":
        options["provider"] = "invalid"
    elif bad == "adapter":
        options["adapter_id"] = 1
    else:
        fs = [f[..., :16] if j == 0 else f for j, f in enumerate(fs)]
    with pytest.raises(ValueError):
        fused_experts_lora(x, w13, w2, rw, ids, tuple(fs), **options)


def test_backward_benchmark_protocol(capsys):
    import json

    from benchmark.kernel_tuning import moe_lora_backward_bench as bench

    args = bench.parse_args(["--quick", "--ep", "2"])
    assert args.tokens == (1, 8) and args.ranks == (16,) and args.samples == 21
    for argv in (["--samples", "0"], ["--tokens", "0"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)
    capsys.readouterr()
    bench.main(["--quick", "--dry-run"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2 and {r["kind"] for r in rows} == {"routed"}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("remote", [False, True])
def test_benchmark_torch_comparator(dtype, remote):
    from benchmark.kernel_tuning import moe_lora_backward_bench as bench

    x, w13, w2, rw, ids, fs = data(dtype=dtype, remote=remote)
    leaves = [v.requires_grad_() for v in (x, rw, *fs)]
    pairs = [(e, torch.where(ids.flatten() == e)[0]) for e in range(3)]
    out = bench.bf16_experts(x, w13, w2, rw, fs, pairs, 2, torch.tensor([0.7]))
    grads = torch.autograd.grad(out, leaves, torch.ones_like(out))
    xf, rf, *ff = [v.detach().float().requires_grad_() for v in leaves]
    ref = _reference.fused_experts_autograd(
        xf, w13.float(), w2.float(), rf, ids, routed_slices(*ff), 0.7
    )
    reference = torch.autograd.grad(ref, (xf, rf, *ff), torch.ones_like(ref))
    errors = bench.relative_errors(grads, reference, bench.GRAD_NAMES)
    assert set(errors) == set(bench.GRAD_NAMES)
    assert all(error <= 0.025 for error in errors.values())


def test_outer_oracle_writes_only_destination_view():
    x, _, _, rw, ids, fs = data()
    left = _reference.backward_shrink(x, fs[0], ids, row_divisor=2)
    expected = _reference.backward_grouped_outer(
        left, x, ids, 3, 0.7, right_divisor=2, pair_scale=rw
    )
    storage = torch.full((3, 2, *expected.shape[1:]), float("nan"))
    out = storage[:, 1]
    actual = _reference.backward_grouped_outer(
        left, x, ids, 3, 0.7, right_divisor=2, pair_scale=rw, out=out
    )
    assert actual is out
    torch.testing.assert_close(actual, expected)
    assert storage[:, 0].isnan().all()
