# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU geometry contracts for TP2 benchmark entry points and engine adapters."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import (
    gdn_decode_compare as decode,
)
from benchmark.kernel_tuning import (
    gdn_decode_external as external,
)
from benchmark.kernel_tuning import (
    gdn_k2_commit_ab as commit_ab,
)
from benchmark.kernel_tuning import (
    gdn_k2_round as k2_round,
)
from benchmark.kernel_tuning import (
    gdn_verify_compare as verify,
)
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs


@pytest.mark.parametrize("shape", ["16,32", "16,48", "8,16", "8,24"])
@pytest.mark.parametrize("module", [verify, commit_ab, k2_round, decode])
def test_shape_arguments(module, shape):
    roots = ["--vllm-root", "unused", "--sglang-root", "unused"] if module == verify else []
    args = module.parse_args(
        roots + (["--providers", "ours", "--shapes"] if module == decode else ["--shape"]) + [shape]
    )
    expected = tuple(map(int, shape.split(",")))
    assert (args.shapes if module == decode else [args.shape]) == [expected]


@pytest.mark.parametrize("shape", ["0,16", "8,0", "8,23", "8", "a,b", "8,16,24"])
@pytest.mark.parametrize("module", [verify, commit_ab, k2_round])
def test_invalid_geometry_rejected_before_cuda(module, shape):
    roots = ["--vllm-root", "unused", "--sglang-root", "unused"] if module == verify else []
    with pytest.raises(SystemExit):
        module.parse_args(roots + ["--shape", shape])


@pytest.mark.parametrize("module", [commit_ab, k2_round])
@pytest.mark.parametrize("hk,hv", [(8, 16), (8, 24)])
def test_k2_cli_threads_geometry(monkeypatch, capsys, module, hk, hv):
    calls = []

    def measure(*args):
        calls.append(args)
        return {"shape": [args[0], args[1]]}

    monkeypatch.setattr(module, "measure", measure)
    flags = ["--shape", f"{hk},{hv}", "--q", "4", "--batches", "1"]
    if module == k2_round:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=16)
        )
        monkeypatch.setattr(module, "select", lambda _: SimpleNamespace(warps=4, min_blocks=1))
        for name in (
            "VECTOR_PUBLISH",
            "FUSED_PUBLISH",
            "PDL",
            "PHASE_TILE",
            "PHASE_WARPS",
            "PHASE_MIN_BLOCKS",
        ):
            monkeypatch.setattr(module.launch, name, getattr(module.launch, name))
        flags += ["--tiles", "64", "--checks", "trusted", "--pdl", "off", "--fused", "on"]
    module.main(flags)
    assert len(calls) == 1 and calls[0][:4] == (hk, hv, 4, 1)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["shape"] == ([hk, hv] if module == commit_ab else f"{hk},{hv}")


@pytest.mark.parametrize("hk,hv", [(8, 16), (8, 24)])
@pytest.mark.parametrize("provider", ["vllm", "sglang", "sglang_replay"])
@pytest.mark.parametrize("batch", [1, 3])
def test_decode_leaf_geometry(monkeypatch, tmp_path, hk, hv, provider, batch):
    from benchmark.kernel_tuning import gdn_replay_external

    weights, xs, state, history = case_inputs(hk, hv, batch, batch + 2, 1, "ordinary", 11, "cpu")
    dc = (2 * hk + hv) * 128
    assert xs.shape == (1, batch, dc + hv * 128 + 2 * hv)
    assert weights.conv_weight.shape == (dc, 4)
    assert state.shape == (batch + 2, hv, 128, 128)
    assert history.shape == (batch + 2, 3, dc)
    slots = torch.arange(batch, 0, -1, dtype=torch.int32)
    calls = []

    def conv(qkvz, *args, **kwargs):
        if provider == "vllm":
            assert qkvz.shape == (batch, dc)
            assert args[0].shape == (batch + 2, dc, 3)
            return qkvz
        assert kwargs["qkv_dim"] == dc and kwargs["num_v_heads"] == hv
        assert args[1].shape == (batch + 2, dc, 3)
        return (
            qkvz[:, :dc],
            torch.empty(batch, hv, 128),
            torch.empty(batch, hv),
            torch.empty(batch, hv),
        )

    def recurrent(**kwargs):
        calls.append(kwargs)
        assert kwargs["mixed_qkv"].shape == (batch, dc)
        assert kwargs["initial_state"].shape == (batch + 2, hv, 128, 128)
        assert kwargs["out"].shape == (batch, 1, hv, 128)
        assert kwargs["a"].shape == kwargs["b"].shape == (batch, hv)
        assert torch.equal(kwargs["ssm_state_indices"], slots)
        if provider == "sglang_replay":
            assert kwargs["d_cache"].shape == (batch + 2, hv, 8, 128)
            assert kwargs["k_cache"].shape == (batch + 2, hk, 8, 128)
            assert kwargs["g_cache"].shape == (batch + 2, hv, 8)
            assert kwargs["write_pos"].tolist() == [2] * batch
            assert kwargs["force_flush"].tolist() == [1] * batch
        kwargs["out"].zero_()

    leaf = SimpleNamespace(
        conv=conv, recurrent=recurrent, norm=lambda **kw: kw["x"], metadata={"source_sha256": {}}
    )
    monkeypatch.setattr(external, "load_vllm", lambda *args: leaf)
    monkeypatch.setattr(external, "load_sglang", lambda *args: leaf)
    if provider == "sglang_replay":
        path = tmp_path / gdn_replay_external.SGLANG_FILE
        path.parent.mkdir(parents=True)
        path.write_text("# stand-in leaf\n")
        monkeypatch.setattr(gdn_replay_external, "load_sglang_replay", lambda _: recurrent)
        runner = external.SGLangReplayDecode(tmp_path, 90, hk, hv, weights.args(), state, history)
        prepared = runner.prepare(xs[0], 2, force_flush=True)
    else:
        runner = external.ExternalDecode(
            provider, tmp_path, 90, hk, hv, weights.args(), state, history
        )
        prepared = runner.prepare(xs[0])
    assert runner.step(prepared, slots).shape == (batch, hv * 128)
    assert len(calls) == 1
    torch.testing.assert_close(runner.canonical_history(), history)


def test_replay_option_uses_sglang_root():
    args = decode.parse_args(
        ["--providers", "sglang_replay", "--sglang-root", "unused", "--shapes", "8,16", "8,24"]
    )
    assert args.providers == ["ours", "sglang_replay"]
    assert args.shapes == [(8, 16), (8, 24)]
    with pytest.raises(SystemExit):
        decode.parse_args(["--providers", "sglang_replay"])


def test_missing_replay_leaf_has_explicit_reason(tmp_path):
    with pytest.raises(external.UnavailableComparator, match="no standalone ReplaySSM leaf"):
        external.SGLangReplayDecode(tmp_path, 90, 8, 24, (), None, None)


@pytest.mark.parametrize("fault", [None, "late_output", "checkpoint", "inactive", "history"])
def test_replay_correctness_wraps_flushes_and_resets(fault):
    initial = torch.ones(5, 1, 2, 2)
    history = torch.zeros(5, 3, 4)
    xs = torch.arange(8 * 3 * 4).reshape(8, 3, 4).float()
    ref_state, ref_history = initial.clone(), history.clone()
    state, cache = initial.clone(), history.clone()
    calls, resets = [], []

    def reference_step(x, slots, state, history):
        indices = slots.long()
        state[indices] += 1
        history[indices] = torch.cat((history[indices, 1:], x[:, None]), dim=1)
        return x

    def step(projected, slots):
        x, phase, flush = projected
        assert phase == len(calls) % 8
        assert flush == (len(calls) == 18)
        assert slots.tolist() == [3, 2, 1]
        calls.append(phase)
        cache.copy_(ref_history)
        if phase == 7 or flush:
            state.copy_(ref_state)
            if fault == "checkpoint":
                state[slots.long()] += 10
        if fault == "inactive":
            state[0] += 1
        if fault == "history":
            cache[0] += 1
        return x + 100 if fault == "late_output" and len(calls) == 17 else x

    runner = SimpleNamespace(
        state=state,
        prepare=lambda x, phase, force_flush: (x, phase, force_flush),
        step=step,
        canonical_history=lambda: cache,
        reset=lambda *args: resets.append(args),
    )
    reference = SimpleNamespace(op=reference_step, weights=(), state=ref_state, history=ref_history)
    result = decode.check_replay_correctness(
        runner, reference, xs, initial, history, 3, decode.Budget(30)
    )
    assert result["correct"] == (fault is None)
    assert len(result["outputs"]) == result["steps"] == 19
    assert len(result["checkpoints"]) == 3
    assert len(resets) == 1 and resets[0][0] is initial and resets[0][1] is history


@pytest.mark.parametrize("mode", ["auto", "immediate", "delayed"])
def test_decode_mode_cli(mode):
    args = decode.parse_args(["--providers", "ours", "--ours", "unified", "--ours-mode", mode])
    assert args.ours_mode == mode
    if mode != "auto":
        with pytest.raises(SystemExit):
            decode.parse_args(["--providers", "ours", "--ours-mode", mode])
