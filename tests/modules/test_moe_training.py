# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU metadata checks and isolated Triton-interpreter numerical qualification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from ctikernels.moe import moe_block, training

from ..support.child import run_child
from . import _moe_reference as ref
from ._moe_layout_reference import build_slot_layout


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("saved", (False, True))
@pytest.mark.parametrize("mapped", (False, True))
def test_empty_tokens_before_dispatch(monkeypatch, shared, masked, saved, mapped):
    pytest.importorskip("triton")  # the training backend imports Triton; the CPU CI env has none
    from ctikernels.moe import _training_backend as block
    from ctikernels.moe.triton import _provider

    def unexpected(*args, **kwargs):
        pytest.fail("empty tokens reached kernel dispatch")

    monkeypatch.setattr(training, "_provider_for", lambda *_: object())
    monkeypatch.setattr(_provider, "_blocks", unexpected)
    monkeypatch.setattr(block, "_route", unexpected)
    wt = ref.make_weights(h=64, e=8, ni=64, nis=128, has_shared=shared)
    mapping = None
    if mapped:
        wt, mapping = ref.shard_weights(wt, 2, 0)
    x = ref.make_x(0).requires_grad_()
    valid = torch.tensor(0, dtype=torch.int32) if masked else None
    routing = torch.empty(0, 2, dtype=torch.uint8)
    for norm in (False, True):
        for replay in (None, routing):
            out, record = moe_block(
                x,
                *wt.values(),
                top_k=2,
                expert_map=mapping,
                routing=replay,
                valid_tokens=valid,
                save_activations=saved,
                norm_topk_prob=norm,
            )
            dx = torch.autograd.grad(out, x, torch.empty_like(out))[0]
            for result in (out, dx):
                assert result.shape == (0, 64) and result.dtype == x.dtype
                assert result.device == x.device and result.is_contiguous()
            assert record.shape == (0, 2) and record.dtype == torch.uint8
            assert record.device == x.device and record.is_contiguous()
            assert not record.requires_grad


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("grad_enabled", (True, False))
def test_frozen_check_before_dispatch(monkeypatch, index, grad_enabled):
    wt = ref.make_weights()
    list(wt.values())[index].requires_grad_()
    monkeypatch.setattr(training, "_provider_for", lambda *_: pytest.fail("provider loaded"))
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="frozen"):
        moe_block(torch.empty(1), *wt.values(), top_k=4)


@pytest.mark.parametrize(
    "case",
    (
        "x_rank",
        "x_dtype",
        "x_stride",
        "weight_rank",
        "weight_dtype",
        "weight_device",
        "weight_shape",
        "top_k",
        "top_k_float",
        "too_many_experts",
        "chunk",
        "chunk_float",
        "map_dtype",
        "map_shape",
        "map_device",
        "map_stride",
        "local_count",
        "routing_dtype",
        "routing_shape",
        "routing_device",
        "valid_dtype",
        "valid_shape",
        "valid_rank",
        "valid_device",
        "bad_h",
        "bad_i",
        "bad_is",
        "norm_int",
    ),
)
def test_validation_before_dispatch(monkeypatch, case):
    wt = ref.make_weights()
    x = ref.make_x(3)
    kw = {"top_k": 4}
    if case == "x_rank":
        x = x[0]
    elif case == "x_dtype":
        x = x.float()
    elif case == "x_stride":
        x = torch.empty(64, 3, dtype=x.dtype).T
    elif case == "weight_rank":
        wt["w2"] = wt["w2"][0]
    elif case == "weight_dtype":
        wt["sg"] = wt["sg"].float()
    elif case == "weight_device":
        wt["sg"] = wt["sg"].to("meta")
    elif case == "weight_shape":
        wt["sg"] = wt["sg"][:, :-1]
    elif case == "top_k":
        kw["top_k"] = 17
    elif case == "top_k_float":
        kw["top_k"] = 2.0
    elif case == "too_many_experts":
        wt = ref.make_weights(e=257)
    elif case == "chunk":
        kw["chunk_tokens"] = 0
    elif case == "chunk_float":
        kw["chunk_tokens"] = 1.5
    elif case == "map_dtype":
        kw["expert_map"] = torch.arange(16)
    elif case == "map_shape":
        kw["expert_map"] = torch.zeros(15, dtype=torch.int32)
    elif case == "map_device":
        kw["expert_map"] = torch.empty(16, dtype=torch.int32, device="meta")
    elif case == "map_stride":
        kw["expert_map"] = torch.zeros(32, dtype=torch.int32)[::2]
    elif case == "local_count":
        wt["w13"], wt["w2"] = wt["w13"][:8], wt["w2"][:8]
    elif case == "routing_dtype":
        kw["routing"] = torch.zeros(3, 4, dtype=torch.int32)
    elif case == "routing_shape":
        kw["routing"] = torch.zeros(3, 3, dtype=torch.uint8)
    elif case == "routing_device":
        kw["routing"] = torch.empty(3, 4, dtype=torch.uint8, device="meta")
    elif case == "valid_dtype":
        kw["valid_tokens"] = torch.tensor(2)
    elif case == "valid_shape":
        kw["valid_tokens"] = torch.zeros(2, dtype=torch.int32)
    elif case == "valid_rank":
        kw["valid_tokens"] = torch.zeros(1, 1, dtype=torch.int32)
    elif case == "valid_device":
        kw["valid_tokens"] = torch.empty((), dtype=torch.int32, device="meta")
    elif case == "bad_h":
        wt, x = ref.make_weights(h=48), ref.make_x(3, h=48)
    elif case == "bad_i":
        wt = ref.make_weights(ni=48)
    elif case == "bad_is":
        wt = ref.make_weights(nis=48)
    elif case == "norm_int":
        kw["norm_topk_prob"] = 0
    monkeypatch.setattr(training, "_provider_for", lambda *_: pytest.fail("provider loaded"))
    with pytest.raises(ValueError, match="moe_block"):
        moe_block(x, *wt.values(), **kw)


@pytest.mark.parametrize("missing", range(1, 7))
def test_shared_none_mixing(monkeypatch, missing):
    wt = ref.make_weights()
    for bit, name in enumerate(("sg", "s13", "s2")):
        if missing & (1 << bit):
            wt[name] = None
    monkeypatch.setattr(training, "_provider_for", lambda *_: pytest.fail("provider loaded"))
    with pytest.raises(ValueError, match="must all be None or all be tensors"):
        moe_block(ref.make_x(1), *wt.values(), top_k=4)
    wt["w13"].requires_grad_()
    with pytest.raises(ValueError, match="must be frozen"):
        moe_block(ref.make_x(1), *wt.values(), top_k=4)


def test_provider_dispatch(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("CTI_MOE_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="CUDA"):
        moe_block(ref.make_x(1), *ref.make_weights().values(), top_k=4)
    device = torch.device("cuda")
    shape = (device, 64, 32, 128)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (99, 0))
    with pytest.raises(ValueError, match="unsupported"):
        training._provider_for(*shape)
    modules = []
    triton_backend = object()
    cute_backend = object()
    cute_supports = {"value": True}

    def load(name, package):
        modules.append(name)
        assert package == "ctikernels.moe"

        def backend(*args):
            assert args == shape
            return cute_backend if name == ".cute._provider" else triton_backend

        def supports(h, ni, nis):
            return cute_supports["value"] and (h, ni, nis) == (128, 64, 32)

        return SimpleNamespace(backend=backend, supports=supports)

    monkeypatch.setattr(training, "import_module", load)
    for capability, expected in (
        ((9, 0), triton_backend),
        ((10, 3), triton_backend),
        ((12, 0), cute_backend),
        ((12, 1), cute_backend),
    ):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _, cap=capability: cap)
        assert training._provider_for(*shape) is expected
    assert modules == [".triton._provider"] * 2 + [".cute._provider"] * 2
    monkeypatch.setitem(training._AUTO, (12, 1), "triton")
    for name, expected in (
        ("cute", cute_backend),
        ("triton", triton_backend),
        ("auto", triton_backend),
    ):
        monkeypatch.setenv("CTI_MOE_PROVIDER", name)
        assert training._provider_for(*shape) is expected
    monkeypatch.setitem(training._AUTO, (12, 1), "cute")
    assert training._provider_for(*shape) is cute_backend
    # Unsupported shapes, and devices other than sm_120/sm_121, fall back to Triton.
    cute_supports["value"] = False
    assert training._provider_for(*shape) is triton_backend
    assert modules[-2:] == [".cute._provider", ".triton._provider"]
    cute_supports["value"] = True
    monkeypatch.setenv("CTI_MOE_PROVIDER", "cute")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    assert training._provider_for(*shape) is triton_backend
    monkeypatch.setenv("CTI_MOE_PROVIDER", "invalid")
    with pytest.raises(ValueError, match="CTI_MOE_PROVIDER"):
        training._provider_for(*shape)

    def unavailable(*_):
        raise ModuleNotFoundError("backend unavailable")

    monkeypatch.setattr(training, "import_module", unavailable)
    monkeypatch.setenv("CTI_MOE_PROVIDER", "cute")
    with pytest.raises(ImportError, match="cute.*could not be imported"):
        training._provider_for(*shape)


@pytest.mark.parametrize("num_experts", (0, 5))
def test_layout_invariants(num_experts):
    e, bm = num_experts, 4
    se = torch.randint(e + 2, (37,), generator=torch.Generator().manual_seed(3))
    lay = build_slot_layout(se, e, bm)
    live = se <= e
    assert (lay.sorted_slot >= 0).sum() == live.sum()
    assert torch.all(lay.inv[~live] == -1)
    for slot in live.nonzero().flatten():
        pos = lay.inv[slot].long()
        assert lay.sorted_slot[pos] == slot
        assert lay.block_expert[pos // bm] == se[slot]
    for b in range(lay.num_blocks):
        slots = lay.sorted_slot[b * bm : (b + 1) * bm]
        slots = slots[slots >= 0].long()
        assert torch.all(se[slots] == lay.block_expert[b])
        assert torch.all(slots[1:] > slots[:-1])


def test_gate_rejects_nonfinite_reference():
    o = torch.ones(4, 8, dtype=torch.float64)
    m = lambda v: ref.metrics(v, o, o)
    good, bad, nan = m(o.bfloat16()), m(2 * o), m(torch.full_like(o, float("nan")))
    assert ref.paired_gate(bad, {"U": nan})
    assert ref.paired_gate(bad, {"U": good, "other": nan})
    assert not ref.paired_gate(good, {"U": good, "other": nan})


def test_interpreter_numerics():
    pytest.importorskip("triton")
    if os.environ.get("CTI_MOE_INTERPRETER_TEST") == "1":
        assert os.environ.get("TRITON_INTERPRET") == "1"
        _interpreter_cases()
        return
    env = os.environ.copy()
    env.update(
        TRITON_INTERPRET="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CTI_MOE_INTERPRETER_TEST="1",
    )
    result = run_child(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(Path(__file__).resolve()) + "::test_interpreter_numerics",
        ),
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _interpreter_cases():
    from ctikernels.moe.triton import _provider as provider

    torch.set_num_threads(1)
    # CPU execution is a test-only provider substitution; the public contract stays CUDA-only.
    training._provider_for = provider.backend
    assert provider._blocks(torch.device("cpu"), 32, 16)["warps"] == 4
    old_cap = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda _: (9, 0)
    assert provider._blocks(torch.device("cuda"), 64, 16)["warps"] == 8
    assert provider._blocks(torch.device("cuda"), 64, 64)["warps"] == 4
    torch.cuda.get_device_capability = old_cap
    wt = ref.make_weights()
    x, dy = ref.make_x(7), ref.make_x(7, seed=2)
    baseline = ref.op_grad(x, wt, dy)
    ref.check_gate(x, wt, dy, *baseline)
    ref.check_gate(x, wt, dy, *baseline, dtype=torch.float32)
    for mapping in (
        torch.arange(16, dtype=torch.int32),
        torch.tensor(list(range(16)), dtype=torch.int32),
    ):
        mapped = ref.op_grad(x, wt, dy, expert_map=mapping)
        assert all(torch.equal(a, b) for a, b in zip(baseline, mapped, strict=True))
    for kw in ({"save_activations": True}, {"chunk_tokens": 3}, {"routing": baseline[1]}):
        got = ref.op_grad(x, wt, dy, **kw)
        assert all(torch.equal(a, b) for a, b in zip(baseline, got, strict=True))
    # Nonidentity local indexing must use local weights but retain global records.
    perm = torch.arange(15, -1, -1)
    permuted = dict(wt, w13=wt["w13"][perm], w2=wt["w2"][perm])
    got = ref.op_grad(x, permuted, dy, expert_map=perm.int())
    assert all(torch.equal(a, b) for a, b in zip(baseline, got, strict=True))
    for h, ni, nis in ((16, 16, 16), (32, 32, 16), (64, 64, 32), (64, 16, 64), (128, 32, 32)):
        w = ref.make_weights(h=h, e=4, ni=ni, nis=nis)
        xx, dd = ref.make_x(3, h=h), ref.make_x(3, h=h, seed=2)
        a, b = (
            ref.op_grad(xx, w, dd, top_k=2),
            ref.op_grad(xx, w, dd, top_k=2, save_activations=True),
        )
        assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True)), (h, ni, nis)
        oo, dx, _, _ = ref.oracle(
            xx, *w.values(), a[1].long(), dd, dtype=torch.float32, kappa=False
        )
        for got, want in ((a[0], oo), (a[2], dx)):
            assert (got.float() - want).norm() / want.norm() < 0.02, (h, ni, nis)
    # The GB10 entry's 128-wide N tiles, also where 128 does not divide H, I or Is (64 fallback).
    portable = provider._blocks
    provider._blocks = lambda *_: {"bm": 64, "bn": 128, "bk": 64, "warps": 4, "stages": 2}
    try:
        for h, ni, nis in ((128, 128, 128), (192, 64, 32), (128, 192, 128)):
            w = ref.make_weights(h=h, e=4, ni=ni, nis=nis, seed=h + ni + nis)
            xx, dd = ref.make_x(5, h=h), ref.make_x(5, h=h, seed=2)
            a = ref.op_grad(xx, w, dd, top_k=2)
            b = ref.op_grad(xx, w, dd, top_k=2, save_activations=True)
            assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True)), (h, ni, nis)
            oo, dx, _, _ = ref.oracle(
                xx, *w.values(), a[1].long(), dd, dtype=torch.float32, kappa=False
            )
            for got, want in ((a[0], oo), (a[2], dx)):
                assert (got.float() - want).norm() / want.norm() < 0.02, (h, ni, nis)
    finally:
        provider._blocks = portable
    for t in (1, 2, 3, 8, 16):
        xx, dd = ref.make_x(t), ref.make_x(t, seed=2)
        got = ref.op_grad(xx, wt, dd)
        assert ref.admissible_sets(xx, wt["r"], got[1].long()).all()
    # Split shared I=64 into independently sized I=32/16 shards; routed I stays 32.
    w = ref.make_weights(nis=64, seed=10)
    xx, dd = ref.make_x(64, seed=110), ref.make_x(64, seed=210)
    full = ref.op_grad(xx, w, dd)
    for ep in (2, 4):
        parts = []
        for rank in range(ep):
            shard, mapping = ref.shard_weights(w, ep, rank)
            part = ref.op_grad(xx, shard, dd, expert_map=mapping, routing=full[1])
            assert torch.equal(part[1], full[1])
            parts.append(part)
        out = sum(p[0].float() for p in parts).bfloat16()
        dx = sum(p[2].float() for p in parts).bfloat16()
        ref.check_gate(xx, w, dd, out, full[1], dx, row_se=True, ep=ep)
        ref.check_gate(xx, w, dd, out, full[1], dx, dtype=torch.float32, ep=ep)
        for total, original in ((out, full[0]), (dx, full[2])):
            assert (total.float() - original.float()).norm() / original.float().norm() < 0.02
    # All-remote live rows: poison local weights so any accidental compute/read is visible.
    shard, mapping = ref.shard_weights(w, 4, 3)
    shard["w13"] = torch.full_like(shard["w13"], float("nan"))
    shard["w2"] = torch.full_like(shard["w2"], float("nan"))
    routing = torch.arange(4, dtype=torch.uint8).expand(7, -1).contiguous()
    got = ref.op_grad(x, shard, dy, expert_map=mapping, routing=routing)
    zero = dict(
        shard, w13=torch.zeros(16, 64, 64, dtype=x.dtype), w2=torch.zeros(16, 64, 32, dtype=x.dtype)
    )
    want = ref.op_grad(x, zero, dy, routing=routing)
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    empty = dict(shard, w13=shard["w13"][:0], w2=shard["w2"][:0])
    all_remote = torch.full((16,), -1, dtype=torch.int32)
    got = ref.op_grad(x, empty, dy, expert_map=all_remote, routing=routing)
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    for valid in (-(2**31), -1, 0, 1, 4, 7, 99):
        n = max(0, min(valid, 7))
        xn, dn = x.clone(), dy.clone()
        xn[n:], dn[n:] = float("nan"), float("nan")
        v = torch.tensor(valid, dtype=torch.int32)
        out, rec, dx = ref.op_grad(xn, wt, dn, valid_tokens=v, chunk_tokens=3)
        assert torch.isfinite(out).all() and torch.isfinite(dx).all()
        assert (
            not out[n:].count_nonzero()
            and not rec[n:].count_nonzero()
            and not dx[n:].count_nonzero()
        )
    # Backward retains the forward's valid count, independent of subsequent device-scalar writes.
    v = torch.tensor(3, dtype=torch.int32)
    xr = x.clone().requires_grad_()
    out, _ = moe_block(xr, *wt.values(), top_k=4, valid_tokens=v)
    v.fill_(7)
    dx = torch.autograd.grad(out, xr, dy)[0]
    assert not dx[3:].count_nonzero()
    # Tied logits select as Unsloth's router does (torch.topk on the FP32 probabilities).
    tied = dict(wt, r=torch.zeros_like(wt["r"]))
    want = ref.unsloth_routing(x, tied["r"], tied["sg"], 4)[0].to(torch.uint8)
    assert torch.equal(ref.op_grad(x, tied, dy)[1], want)
    _no_shared_cases()
    _norm_topk_prob_cases()
    _backend_cases()
    _check_ablations()
    ref.check_planted_bias(
        ref.make_x(64, seed=160), ref.make_weights(seed=60), ref.make_x(64, seed=260)
    )


def _no_shared_cases():
    wt = ref.make_weights(e=8, ni=64, has_shared=False, seed=80)
    x, dy = ref.make_x(7, seed=81), ref.make_x(7, seed=82)
    full = ref.op_grad(x, wt, dy, top_k=2)
    assert ref.admissible_sets(x, wt["r"], full[1].long()).all()
    ref.check_gate(x, wt, dy, *full)
    oo, dd, kp, kd = ref.oracle(x, *wt.values(), full[1].long(), dy)
    assert torch.isfinite(kp).all() and torch.isfinite(kd).all()
    for got, want in ((full[0], oo), (full[2], dd)):
        assert (got.double() - want).norm() / want.norm() < 0.02
    lo, ld = ref.unsloth(x, *wt.values(), full[1].long(), dy)
    assert torch.isfinite(lo).all() and torch.isfinite(ld).all()
    for kw in ({"save_activations": True}, {"chunk_tokens": 3}, {"routing": full[1]}):
        got = ref.op_grad(x, wt, dy, top_k=2, **kw)
        assert all(torch.equal(a, b) for a, b in zip(full, got, strict=True))
    for ep in (2, 4):
        parts = []
        for rank in range(ep):
            shard, mapping = ref.shard_weights(wt, ep, rank)
            part = ref.op_grad(x, shard, dy, top_k=2, expert_map=mapping, routing=full[1])
            assert torch.equal(part[1], full[1])
            parts.append(part)
        for index, want in ((0, oo), (2, dd)):
            total = sum(p[index].float() for p in parts)
            assert (total.double() - want).norm() / want.norm() < 0.02
    for valid in (0, 1, 4, 7):
        xn, dn = x.clone(), dy.clone()
        xn[valid:], dn[valid:] = float("nan"), float("nan")
        got = ref.op_grad(
            xn,
            wt,
            dn,
            top_k=2,
            valid_tokens=torch.tensor(valid, dtype=torch.int32),
            chunk_tokens=3,
            save_activations=True,
        )
        assert all(torch.isfinite(a).all() and not a[valid:].count_nonzero() for a in got)
        assert all(torch.equal(a[:valid], b[:valid]) for a, b in zip(full, got, strict=True))
    empty = dict(wt, w13=wt["w13"][:0], w2=wt["w2"][:0])
    got = ref.op_grad(x, empty, dy, top_k=2, expert_map=torch.full((8,), -1, dtype=torch.int32))
    assert not got[0].count_nonzero() and not got[2].count_nonzero()
    wt = ref.make_weights(e=4, ni=768, has_shared=False)
    got = ref.op_grad(x[:1], wt, dy[:1], top_k=2)
    oo, dd, _, _ = ref.oracle(x[:1], *wt.values(), got[1].long(), dy[:1])
    for actual, want in ((got[0], oo), (got[2], dd)):
        assert (actual.double() - want).norm() / want.norm() < 0.02


def _norm_topk_prob_cases():
    # Both router modes against FP64 autograd; with norm_topk_prob off the derivative
    # reaches every expert's router row, which the selected-only ablation misses.
    for wt, k in (
        (ref.make_weights(seed=90, logit_std=0.5), 4),
        (ref.make_weights(e=8, ni=64, has_shared=False, seed=91, logit_std=0.5), 2),
    ):
        e = wt["r"].shape[0]
        x, dy = ref.make_x(7, seed=92), ref.make_x(7, seed=93)
        for norm in (True, False):
            full = ref.op_grad(x, wt, dy, top_k=k, norm_topk_prob=norm)
            assert ref.admissible_sets(x, wt["r"], full[1].long()).all()
            args = (x, *wt.values(), full[1].long(), dy)
            oo, dd, _, _ = ref.oracle(*args, kappa=False, renormalize=norm)
            errors = [
                (got.double() - want).norm() / want.norm()
                for got, want in ((full[0], oo), (full[2], dd))
            ]
            assert max(errors) < 0.01, (norm, errors)
            wrong, _, _, _ = ref.oracle(*args, kappa=False, renormalize=not norm)
            assert (full[0].double() - wrong).norm() / oo.norm() > 0.05
            if not norm:
                _, ablated, _, _ = ref.oracle(
                    *args, kappa=False, renormalize=False, unselected_grad=False
                )
                assert (full[2].double() - ablated).norm() / dd.norm() > 5 * errors[1]
            for kw in ({"save_activations": True}, {"chunk_tokens": 3}, {"routing": full[1]}):
                got = ref.op_grad(x, wt, dy, top_k=k, norm_topk_prob=norm, **kw)
                assert all(torch.equal(a, b) for a, b in zip(full, got, strict=True))
            # The shared I-shard needs Is/ep >= 16.
            for ep in (2,) if wt["sg"] is not None else (2, 4):
                parts = []
                for rank in range(ep):
                    shard, mapping = ref.shard_weights(wt, ep, rank)
                    parts.append(
                        ref.op_grad(x, shard, dy, top_k=k, expert_map=mapping, norm_topk_prob=norm)
                    )
                    assert torch.equal(parts[-1][1], full[1])
                for index, want in ((0, oo), (2, dd)):
                    total = sum(p[index].float() for p in parts)
                    assert (total.double() - want).norm() / want.norm() < 0.01
            remote = ref.op_grad(
                x,
                dict(wt, w13=wt["w13"][:0], w2=wt["w2"][:0]),
                dy,
                top_k=k,
                expert_map=torch.full((e,), -1, dtype=torch.int32),
                norm_topk_prob=norm,
            )
            if wt["sg"] is None:
                assert not remote[0].count_nonzero() and not remote[2].count_nonzero()
            for valid in (0, 1, 4, 7):
                xn, dn = x.clone(), dy.clone()
                xn[valid:], dn[valid:] = float("nan"), float("nan")
                got = ref.op_grad(
                    xn,
                    wt,
                    dn,
                    top_k=k,
                    valid_tokens=torch.tensor(valid, dtype=torch.int32),
                    chunk_tokens=3,
                    norm_topk_prob=norm,
                )
                for a, b in zip(full, got, strict=True):
                    assert torch.isfinite(a).all() and not b[valid:].count_nonzero()
                    assert torch.equal(a[:valid], b[:valid])
                    if b.is_floating_point():
                        assert not torch.signbit(b[valid:]).any()


def _backend_cases():
    from ctikernels.moe._training_backend import _MoEBlock
    from ctikernels.moe.triton import _provider as provider

    assert provider._MoEBlock is _MoEBlock
    wt = ref.make_weights(e=4, ni=64, nis=32)
    x, dy = ref.make_x(7), ref.make_x(7, seed=2)
    full = ref.op_grad(x, wt, dy, top_k=2, chunk_tokens=3)
    xr = x.clone().requires_grad_()
    out, rec = provider._MoEBlock.apply(xr, *wt.values(), 2, None, None, None, False, 3)
    dx = torch.autograd.grad(out, xr, dy)[0]
    assert all(torch.equal(a, b) for a, b in zip(full, (out, rec, dx), strict=True))

    class ChunkBackend(provider._TritonBackend):
        def block_m(self, device, tokens, ni, nis):
            rows.append(tokens)
            self.blk = {"bm": 16 if tokens == 1 else 32, "bn": 16, "bk": 64, "warps": 4}
            return super().block_m(device, tokens, ni, nis)

        def dw_tiles(self, device, tokens, ni, nis, *, saved=False):
            tiles.append(tokens)
            self.blk = {"bm": 16 if tokens == 1 else 32, "bn": 16, "bk": 64, "warps": 4}
            return super().dw_tiles(device, tokens, ni, nis, saved=saved)

    rows, tiles = [], []
    custom = ChunkBackend({})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "_provider_for", lambda *_: custom)
        xr = x.clone().requires_grad_()
        out, rec = moe_block(xr, *wt.values(), top_k=2, chunk_tokens=3, save_activations=True)
        patch.setattr(training, "_provider_for", lambda *_: pytest.fail("backward redispatched"))
        dx = torch.autograd.grad(out, xr, dy)[0]
    assert rows == tiles == [3, 3, 1]
    oo, dd, _, _ = ref.oracle(x, *wt.values(), rec.long(), dy)
    for actual, want in ((out, oo), (dx, dd)):
        assert (actual.double() - want).norm() / want.norm() < 0.02


# Port of cti's own frozen MoE-block dX p2c ablation regressions (2026-10-11).
def _ablation_candidate(x, wt, dy, sel, *, no_router=False, no_sgate=False, no_renorm=False):
    # Our recipe in torch (FP32 everywhere but the inputs), with one term removed.
    ni = wt["w2"].shape[2]
    xr = x.float().detach().requires_grad_()
    z = (xr @ wt["r"].float().T).to(torch.bfloat16).float()
    z = xr @ wt["r"].float().T + (z - xr @ wt["r"].float().T).detach()
    if no_renorm:
        w = torch.softmax(z, dim=1).gather(1, sel)
    else:
        zs = z.gather(1, sel)
        w = torch.softmax(zs - zs.max(dim=1, keepdim=True).values.detach(), dim=1)
    if no_router:
        w = w.detach()
    d = ref.routed(xr, sel, wt["w13"], wt["w2"], torch.float32)
    out = (w[:, :, None] * d).sum(dim=1)
    gu = xr @ wt["s13"].float().T
    sh = (torch.nn.functional.silu(gu[:, :ni]) * gu[:, ni:]) @ wt["s2"].float().T
    game = torch.sigmoid(xr @ wt["sg"].float().T)
    if no_sgate:
        game = game.detach()
    out = out + game * sh
    (dx,) = torch.autograd.grad(out, xr, dy.float())
    return out.detach(), dx


def _check_ablations():
    """Each missing term fails the gate on a targeted fixture; the op passes the same fixture."""

    torch.use_deterministic_algorithms(True)
    c = {"H": 64, "I": 32, "E": 16, "K": 4, "sweep": [64]}
    t = min(64, c["sweep"][-1])
    res = {}
    args_of = lambda wt: (wt["r"], wt["sg"], wt["w13"], wt["w2"], wt["s13"], wt["s2"])

    def gate_view(x, wt, dy, sel, cand, view):
        st = ref.oracle_state(x, *args_of(wt), sel, dy)
        lv = ref.unsloth(x, *args_of(wt), sel, dy)
        i = 0 if view == "out" else 1
        pick = lambda v: (v, None) if view == "out" else (None, v)
        rc, rl = ref.resolve(st, *pick(cand)), ref.resolve(st, *pick(lv[i]))
        # bias SE clustered by row: elements of a row share error terms (row-clustered standard error)
        mc = ref.metrics(cand, rc[i], rc[2 + i], row_se=True)
        ml = ref.metrics(lv[i], rl[i], rl[2 + i], row_se=True)
        return ref.paired_gate(mc, {"U": ml}), mc["rms_error"] / (1.05 * ml["rms_error"])

    def gate_dx(x, wt, dy, sel, cand_dx):
        return gate_view(x, wt, dy, sel, cand_dx, "dx")[0]

    def strength(x, wt, dy, sel, cand, view):
        # how far past the gate's rms allowance the ablated candidate lands (>= 10 required)
        return gate_view(x, wt, dy, sel, cand, view)[1]

    # router term: moderate spread, dy along the difference of two selected experts' outputs
    wt = ref.make_weights(seed=50, logit_std=0.5)
    x = ref.make_x(t, seed=51)
    _, rec, _ = ref.op_grad(x, wt, ref.make_x(t, seed=52))
    sel = rec.long()
    d = ref.routed(x.double(), sel, wt["w13"], wt["w2"], torch.float64)
    dy = (d[:, 0] - d[:, 1]).to(torch.bfloat16)
    _, _, dx = ref.op_grad(x, wt, dy)
    fails = gate_dx(x, wt, dy, sel, dx)
    assert not fails, f"op fails the router fixture: {fails}"
    _, adx = _ablation_candidate(x, wt, dy, sel, no_router=True)
    res["router"] = gate_dx(x, wt, dy, sel, adx)
    res["router_strength"] = strength(x, wt, dy, sel, adx, "dx")
    assert res["router"] and res["router_strength"] >= 10, f"router ablation too weak: {res}"
    # shared-gate term: s = 0 exactly (sigma' maximal) through a coordinate where x is zero and w_sg
    # is large, with a large shared output aligned with dy
    wt = ref.make_weights(seed=53, sg_scale=0.0, shared_scale=4.0)
    xs = x.clone()
    xs[:, -1] = 0
    wt["sg"] = torch.zeros_like(wt["sg"])
    wt["sg"][0, -1] = 4.0
    _, rec, _ = ref.op_grad(xs, wt, ref.make_x(t, seed=55))
    sel = rec.long()
    ni = c["I"]
    gu = xs.double() @ wt["s13"].double().T
    sh = (torch.nn.functional.silu(gu[:, :ni]) * gu[:, ni:]) @ wt["s2"].double().T
    dy = (sh / sh.norm(dim=1, keepdim=True)).to(torch.bfloat16)
    x_keep, x = x, xs
    _, _, dx = ref.op_grad(x, wt, dy)
    fails = gate_dx(x, wt, dy, sel, dx)
    assert not fails, f"op fails the shared-gate fixture: {fails}"
    _, adx = _ablation_candidate(x, wt, dy, sel, no_sgate=True)
    res["shared_gate"] = gate_dx(x, wt, dy, sel, adx)
    res["shared_gate_strength"] = strength(x, wt, dy, sel, adx, "dx")
    assert res["shared_gate"] and res["shared_gate_strength"] >= 10, (
        f"shared-gate ablation too weak: {res}"
    )
    x = x_keep
    # renormalisation
    wt = ref.make_weights(seed=56, logit_std=0.5)
    dy = ref.make_x(t, seed=57)
    out, rec, dx = ref.op_grad(x, wt, dy)
    sel = rec.long()
    about, _ = _ablation_candidate(x, wt, dy, sel, no_renorm=True)
    res["renorm"] = gate_view(x, wt, dy, sel, about, "out")[0]
    res["renorm_strength"] = strength(x, wt, dy, sel, about, "out")
    assert res["renorm"] and res["renorm_strength"] >= 10, f"renorm ablation too weak: {res}"
    fails = gate_view(x, wt, dy, sel, out, "out")[0]
    assert not fails, f"op fails the renorm fixture: {fails}"
    # the BF16 rounding of the logits: at the k / k+1 boundary, expert 1's FP32 logit 1 + 2^-10 beats
    # expert 0's 1.0, but both round to 1.0 in BF16, so the contract keeps the lower index 0
    wt = ref.make_weights(seed=58)
    wt["r"] = torch.zeros_like(wt["r"])
    hi_e, lo_e = 1, 0
    xr = torch.zeros(t, c["H"], dtype=torch.bfloat16, device=x.device)
    xr[:, 0] = 1.0
    xr[:, 1] = 2.0**-10
    wt["r"][:, 0] = -8.0
    for j in range(2, c["K"] + 1):  # k - 1 experts clearly above the boundary
        wt["r"][j, 0] = 4.0 - 0.25 * j
    wt["r"][lo_e, 0] = 1.0
    wt["r"][hi_e, 0] = 1.0
    wt["r"][hi_e, 1] = 1.0
    _, rec, _ = ref.op_grad(xr, wt, ref.make_x(t, seed=59))
    sel = rec.long()
    assert torch.all((sel == lo_e).any(dim=1)) and not torch.any((sel == hi_e).any(dim=1)), (
        "rounded tie rule"
    )
    unrounded = torch.sort(
        xr.float() @ wt["r"].float().T, dim=1, descending=True, stable=True
    ).indices[:, : c["K"]]
    assert torch.all((unrounded == hi_e).any(dim=1)), (
        "fixture: FP32 logits should pick the higher index"
    )
    adm = ref.admissible_sets(xr, wt["r"], unrounded)
    res["logit_rounding"] = int((~adm).sum())
    assert res["logit_rounding"] == t, "a selection from unrounded logits was admissible"
    return res
