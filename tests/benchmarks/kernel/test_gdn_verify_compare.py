# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks of dispatch layouts, copy guards, CLI plumbing and row schema."""

import json
import os
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import gdn_verify_compare as compare
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_verify_compare import (
    External,
    audit_contiguous_copies,
    summarize,
)
from benchmark.kernel_tuning.gdn_verify_external import load_vllm_rearrange


def test_paired_statistics_use_ratios_before_median():
    comparisons = {
        "vllm": [(1.0, 2.0), (10.0, 10.0), (100.0, 50.0)],
        "sglang": [(2.0, 3.0), (4.0, 6.0), (6.0, 9.0)],
    }
    result = summarize(comparisons, [(10.0, 11.0), (10.0, 9.0), (10.0, 12.0)])
    assert result["ratios_comparator_over_ours"] == {"vllm": 1.0, "sglang": 1.5}
    assert result["eps"] == pytest.approx(0.19)
    assert result["faster_comparator"] == "vllm"
    assert result["speedup_vs_faster_comparator"] == 1.0
    assert result["samples_us"]["vllm"] == comparisons["vllm"]


@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48), (8, 16), (8, 24)])
@pytest.mark.parametrize("b,q", [(1, 4), (2, 5), (3, 9), (4, 16)])
def test_external_acceptance_layouts_and_restore(b, q, hk, hv):
    dc = (2 * hk + hv) * 128
    weights, x, initial, history = case_inputs(hk, hv, b, b, q + 1, "ordinary", 23, "cpu")
    drafts = x[-q:].transpose(0, 1).contiguous()
    vllm = External("vllm", None, weights, drafts, initial, history, hk=hk, hv=hv)
    sglang = External("sglang", None, weights, drafts, initial, history, hk=hk, hv=hv)
    assert vllm.qkvz.is_contiguous() and vllm.ba.is_contiguous()
    assert sglang.qkvz.stride(0) == sglang.ba.stride(0) == drafts.shape[-1]
    assert not sglang.qkvz.is_contiguous() and not sglang.ba.is_contiguous()
    assert sglang.qkvz.untyped_storage().data_ptr() == sglang.ba.untyped_storage().data_ptr()
    accepted = q // 2
    assert vllm.indices.shape == (b, q)
    assert vllm.indices.min() > 0  # vLLM skips slot zero.
    assert vllm.state.shape[0] == b * q + 1
    torch.testing.assert_close(vllm.state[vllm.indices[:, accepted - 1].long()], initial)
    torch.testing.assert_close(
        vllm.conv[vllm.slots.long(), :, accepted - 1 : accepted + 2],
        history.transpose(1, 2),
    )
    torch.testing.assert_close(sglang.state[sglang.slots.long()], initial)
    torch.testing.assert_close(sglang.conv[sglang.slots.long()], history.transpose(1, 2))
    assert sglang.intermediate.shape == (b + 1, q, hv, 128, 128)
    # Neighboring speculative conv windows alias the same physical history.
    sglang.conv_windows[0, 0, 0, 1] = 7
    assert sglang.conv_windows[0, 1, 0, 0].item() == 7
    assert sglang.conv_storage.numel() == (b + 1) * dc * (q + 2)
    for provider in (vllm, sglang):
        torch.testing.assert_close(provider.qkvz[:, :dc], drafts[..., :dc].reshape(b * q, dc))
        torch.testing.assert_close(provider.beta, drafts[..., -2 * hv : -hv].reshape(b * q, hv))
        torch.testing.assert_close(provider.alpha, drafts[..., -hv:].reshape(b * q, hv))
        provider.state.zero_()
        provider.conv.zero_()
        provider.prepare()
    torch.testing.assert_close(vllm.state[vllm.indices[:, accepted - 1].long()], initial)
    torch.testing.assert_close(sglang.state[sglang.slots.long()], initial)


@pytest.mark.parametrize(
    "extra,expected_checks,expected_fused,mode,window",
    [
        ([], "trusted", "on", "eager", 8),
        (["--ours-checks", "strict", "--ours-fused", "off"], "strict", "off", "eager", 8),
        (["--ours-commit-mode", "delayed", "--ours-window", "16"], "trusted", "on", "delayed", 16),
    ],
)
@pytest.mark.parametrize("shape", ["16,32", "16,48", "8,16", "8,24"])
@pytest.mark.parametrize("audit", [True, False])
def test_cli_modes_reach_measure_and_every_row(
    monkeypatch, capsys, extra, audit, expected_checks, expected_fused, mode, window, shape
):
    """Exercise main -> measure -> provider/graph plumbing without a GPU."""
    from benchmark.kernel_tuning import gdn_replay_inputs

    modes, audit_flags = [], []
    expected_heads = tuple(map(int, shape.split(",")))

    def ours(*args, checks, fused, commit_mode, window, hk, hv, provider):
        assert (hk, hv) == expected_heads
        modes.append((checks, fused))
        return SimpleNamespace(
            effective_tile=64,
            commit_mode=commit_mode,
            rounds=compare.fold_period(commit_mode, args[1].shape[1], window),
        )

    def graph(provider, *, assert_no_extra_copies):
        audit_flags.append(assert_no_extra_copies)
        return SimpleNamespace(
            sample=lambda: 1.0,
            copy_audit={"extra_copies": 0} if assert_no_extra_copies else None,
        )

    monkeypatch.setattr(compare, "Ours", ours)

    def external(*args, hk, hv):
        assert (hk, hv) == expected_heads
        return object()

    monkeypatch.setattr(compare, "External", external)
    monkeypatch.setattr(compare, "GraphRound", graph)

    def inputs(hk, hv, b, slots, steps, *args):
        assert (hk, hv) == expected_heads
        return None, torch.zeros(steps, b, 2), None, None

    monkeypatch.setattr(gdn_replay_inputs, "case_inputs", inputs)
    monkeypatch.setattr(
        compare,
        "correctness",
        lambda *args: {"providers": {name: {"allclose": True} for name in ("vllm", "sglang")}},
    )
    monkeypatch.setattr(compare, "source_hashes", lambda roots: {"fixture": "digest"})
    monkeypatch.setattr(compare, "load_vllm", lambda *args: None)
    monkeypatch.setattr(compare, "load_sglang", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    compare.main(
        [
            "--vllm-root",
            "unused-vllm",
            "--sglang-root",
            "unused-sglang",
            "--q",
            "4",
            "8",
            "--batches",
            "1",
            "2",
            "--pairs",
            "2",
        ]
        + ["--shape", shape]
        + extra
        + ([] if audit else ["--no-assert-no-extra-copies"])
    )
    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert modes == [(expected_checks, expected_fused == "on")] * 4
    assert audit_flags == [audit] * 12
    assert [(row["q"], row["b"]) for row in rows] == [(4, 1), (4, 2), (8, 1), (8, 2)]
    for row in rows:
        config = row["config"]
        assert config["shape"] == list(expected_heads)
        assert (row["shape"]["hk"], row["shape"]["hv"]) == expected_heads
        assert config["ours_checks"] == expected_checks
        assert config["ours_fused"] == expected_fused
        assert config["ours_commit_mode"] == mode
        assert config["ours_window"] == window
        assert config["ours_rounds"] == compare.fold_period(mode, row["q"], window)
        assert config["ours_tile"] == "auto"
        assert config["ours_effective_tile"] == 64
        assert config["ours_pdl"] is True
        assert config["pdl"] is True
        assert config["assert_no_extra_copies"] is audit
        assert config["vllm_projection_layout"] == "separate_qkvz_ba"
        assert config["sglang_projection_layout"] == "fused_qkvzba_views"
        assert set(row["copy_audit"]) == {"ours", "vllm", "sglang"}
        assert row["copy_audit"]["vllm"] == ({"extra_copies": 0} if audit else None)
        assert row["source_sha256"] == {"fixture": "digest"}
        assert row["correctness_ok"] is True
    summaries = captured.err.splitlines()
    assert len(summaries) == 4
    for summary in summaries:
        assert f"ours_checks={expected_checks} ours_fused={expected_fused}" in summary
        assert "ours_tile=auto (64) ours_pdl=True" in summary


@pytest.mark.parametrize("flag,value", [("--ours-checks", "invalid"), ("--ours-fused", "invalid")])
def test_invalid_modes_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        compare.parse_args(["--vllm-root", "unused", "--sglang-root", "unused", flag, value])


@pytest.mark.parametrize("provider_name", ["vllm", "sglang", "ours"])
@pytest.mark.parametrize("extra_copy", [False, True])
def test_copy_audit_distinguishes_required_copies_and_restores_patch(provider_name, extra_copy):
    original = torch.Tensor.contiguous
    dense = torch.ones(2, 4)
    strided = dense[:, :2]
    leaf = SimpleNamespace(recurrent=lambda x: x.contiguous())
    provider = SimpleNamespace(provider=provider_name, leaf=leaf)

    def call():
        if provider_name == "vllm":
            strided.contiguous()
            strided.contiguous()
        provider.leaf.recurrent(strided if extra_copy else dense)

    provider.call = call
    if extra_copy:
        with pytest.raises(AssertionError, match="contiguous-copy audit failed"):
            audit_contiguous_copies(provider)
    else:
        result = audit_contiguous_copies(provider)
        assert result["copies"] == (2 if provider_name == "vllm" else 0)
        assert result["extra_copies"] == 0
        assert result["stages"]["recurrent"] == {"calls": 1, "copies": 0}
    assert torch.Tensor.contiguous is original
    assert provider.leaf is leaf


def test_copy_audit_restores_patch_after_wrapper_exception():
    original = torch.Tensor.contiguous

    def fail():
        raise ValueError("wrapper failure")

    leaf = SimpleNamespace(recurrent=fail)
    provider = SimpleNamespace(leaf=leaf)
    provider.call = lambda: provider.leaf.recurrent()
    with pytest.raises(ValueError, match="wrapper failure"):
        audit_contiguous_copies(provider)
    assert torch.Tensor.contiguous is original
    assert provider.leaf is leaf


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("rounds", [1, 4])
def test_graph_audits_once_in_warmup_only(monkeypatch, enabled, rounds):
    phase, calls, audits = "warmup", [], []
    original = torch.Tensor.contiguous
    real_audit = compare.audit_contiguous_copies
    stream = SimpleNamespace(wait_stream=lambda other: None)

    def call():
        calls.append(phase)
        if phase == "capture":
            assert torch.Tensor.contiguous is original

    provider = SimpleNamespace(prepare=lambda: None, call=call, finish=lambda: None, rounds=rounds)

    def audit(provider):
        assert phase == "warmup" and len(calls) == 2
        audits.append(real_audit(provider))
        return audits[-1]

    @contextmanager
    def capture(*args, **kwargs):
        nonlocal phase
        phase = "capture"
        yield
        phase = "replay"

    def replay():
        assert phase == "replay"
        assert torch.Tensor.contiguous is original

    monkeypatch.setattr(compare, "audit_contiguous_copies", audit)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=replay))
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda stop: 1.0
        ),
    )
    graph = compare.GraphRound(provider, assert_no_extra_copies=enabled)
    assert graph.sample() == 1000 / rounds
    assert calls == ["warmup"] * 3 + ["capture"]
    assert len(audits) == int(enabled)
    assert graph.copy_audit == (audits[0] if enabled else None)


@pytest.mark.parametrize("checks", ["trusted", "strict"])
@pytest.mark.parametrize("rounds", [1, 4])
def test_ours_passes_projection_and_precomputed_positions_directly(checks, rounds):
    calls = []

    def record(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    provider = compare.Ours.__new__(compare.Ours)
    provider.checks, provider.provider = checks, "cute"
    provider.drafts, provider.positions, provider.accepted = (
        object(),
        [object() for _ in range(rounds)],
        object(),
    )
    runner = SimpleNamespace(
        op=SimpleNamespace(**{name: record(name) for name in ("begin", "verify", "commit", "end")}),
        weights=(1, 2, 3, 4),
        slots=object(),
        state=object(),
        pending=object(),
        verify_out=object(),
        status=object(),
        phase_workspace=object(),
    )
    provider.runner = runner
    provider.outputs = [runner.verify_out] + [object() for _ in range(rounds - 1)]
    provider.output = runner.verify_out
    provider.call()
    assert [name for name, _, _ in calls] == (
        ["verify", "commit"] * rounds
        if checks == "trusted"
        else ["begin"] + ["verify", "commit"] * rounds + ["end"]
    )
    verifies = [(args, kwargs) for name, args, kwargs in calls if name == "verify"]
    for r, (args, kwargs) in enumerate(verifies):
        assert args[0] is provider.drafts
        assert kwargs["positions"] is provider.positions[r]
        assert kwargs["out"] is provider.outputs[r]
    assert provider.output is runner.verify_out


# The vLLM case runs only when GDN_VERIFY_VLLM_ROOT supplies a vLLM source tree (never a skip).
@pytest.mark.parametrize(
    "provider_name", ["sglang", *(["vllm"] if os.environ.get("GDN_VERIFY_VLLM_ROOT") else [])]
)
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48), (8, 16), (8, 24)])
@pytest.mark.parametrize("b,q", [(1, 4), (2, 5)])
def test_dispatch_tensor_layouts_at_leaf_boundaries(monkeypatch, provider_name, b, q, hk, hv):
    """Use the real vLLM packing method when its read-only checkout is supplied.

    Kernel stand-ins only allocate their documented output layouts; they do
    not implement any third-party numerical algorithm.
    """
    root = os.environ.get("GDN_VERIFY_VLLM_ROOT")
    k = 128
    dc = (2 * hk + hv) * k
    drafts = torch.randn(b, q, dc + hv * k + 2 * hv)
    weights = SimpleNamespace(conv_weight=None, A_log=None, dt_bias=None, norm_weight=None)
    events, buffers = [], {}

    def unpack(qkvz, ba, *geometry):
        events.append("unpack")
        assert geometry == (hk, hv, k, k)
        assert qkvz.stride(0) == ba.stride(0) == drafts.shape[-1]
        mixed = torch.empty(b * q, dc)
        z = torch.empty(b * q, hv, k)
        buffers["z"] = z
        return mixed, z, torch.empty(b * q, hv), torch.empty(b * q, hv)

    def conv(mixed, *args, **kwargs):
        events.append("conv")
        if provider_name == "vllm":
            assert mixed.stride() == (dc + hv * k, 1)
            assert "num_accepted_tokens" in kwargs
            return mixed  # vLLM overwrites its strided input.
        assert mixed.stride() == (q * dc, 1, dc)
        assert "intermediate_conv_window" in kwargs
        result = torch.empty_like(mixed)
        assert result.stride() == mixed.stride()
        return result

    def recurrent(**kwargs):
        events.append("recurrent")
        query, key, value = (kwargs[name] for name in ("q", "k", "v"))
        for tensor, heads in ((query, hk), (key, hk), (value, hv)):
            assert tensor.shape == (1, b * q, heads, k)
            assert tensor.stride(-1) == 1
            assert tensor.stride(-2) == k
            if provider_name == "vllm":
                assert tensor.is_contiguous()
                assert tensor.contiguous() is tensor
            else:
                assert tensor.stride(1) == dc
                assert not tensor.is_contiguous()
        assert kwargs["a"].is_contiguous() and kwargs["b"].is_contiguous()
        result = torch.empty_like(value, memory_format=torch.contiguous_format)
        if provider_name == "vllm":
            assert query.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            expected = drafts[..., :dc].reshape(b * q, dc)
            torch.testing.assert_close(query.reshape(b * q, -1), expected[:, : hk * k])
            torch.testing.assert_close(key.reshape(b * q, -1), expected[:, hk * k : 2 * hk * k])
            torch.testing.assert_close(value.reshape(b * q, -1), expected[:, 2 * hk * k :])
            return result, None
        assert kwargs["disable_state_update"] is True
        return result

    def norm(output, *args, z, **kwargs):
        events.append("norm")
        assert output.is_contiguous() and z.is_contiguous()
        if provider_name == "sglang":
            assert z.data_ptr() == buffers["z"].data_ptr()
        return output, None, None

    def scatter(caches, slots, steps, *args):
        events.append("scatter")
        assert slots.is_contiguous() and steps.is_contiguous()
        assert slots.dtype == steps.dtype == torch.int32

    leaf = SimpleNamespace(conv=conv, recurrent=recurrent, norm=norm)
    if provider_name == "vllm":
        leaf.rearrange = load_vllm_rearrange(root)
    else:
        leaf.unpack, leaf.scatter = unpack, scatter
    provider = External(
        provider_name,
        leaf,
        weights,
        drafts,
        torch.zeros(b, hv, k, k),
        torch.zeros(b, 3, dc),
        hk=hk,
        hv=hv,
    )
    result = audit_contiguous_copies(provider)
    assert result["extra_copies"] == 0
    assert provider.output.shape == (b, q, hv * k)
    assert events == (
        ["conv", "recurrent", "norm"]
        if provider_name == "vllm"
        else ["unpack", "conv", "recurrent", "norm", "scatter"]
    )


@pytest.mark.parametrize("q", [4, 5, 6, 8, 9, 12, 16])
@pytest.mark.parametrize("window", [4, 8, 16])
def test_delayed_fold_period(q, window):
    period = compare.fold_period("delayed", q, window)
    assert (period - 1) * (q // 2) < window <= period * (q // 2)
    assert compare.fold_period("eager", q, window) == 1
