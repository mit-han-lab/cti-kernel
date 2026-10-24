# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU coverage of layer geometry and isolated comparator workers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import dflash_draft_layers as layers


def test_shapes_and_provider_arguments():
    assert layers.SHAPES == {
        "q3_target": {"hq": 32, "hk": 8, "d": 128, "causal": True, "layers": 36},
        "draft_noncausal": {"hq": 32, "hk": 8, "d": 128, "causal": False, "layers": 5},
        "q35_target": {"hq": 16, "hk": 4, "d": 256, "causal": True, "layers": 8},
        "q35moe_target": {"hq": 16, "hk": 2, "d": 256, "causal": True, "layers": 10},
        "q35moe_tp2": {"hq": 8, "hk": 1, "d": 256, "causal": True, "layers": 10},
        # K-Q7: Qwen3-30B-A3B DFlash draft geometry (GQA 8, D=128), causal layers.
        "q3moe_draft": {"hq": 32, "hk": 4, "d": 128, "causal": True, "layers": 5},
    }
    args, combos = layers.parse_args([])
    assert args.providers == list(layers.PROVIDERS)
    assert layers.COMPARATOR_LABELS["draft_fi"] == "ctikernels DraftAttention FlashInfer"
    assert args.sglang_flashinfer is None
    assert combos == [{}]
    args, _ = layers.parse_args(
        [
            "--providers",
            "fi_fixed_s16,fi_fixed_s32,fi_fixed_s64,sglang",
            "--sglang-flashinfer",
            "/tmp/flashinfer071",
            "--shapes",
            "q35_target",
        ]
    )
    assert args.providers == ["fi_fixed_s16", "fi_fixed_s32", "fi_fixed_s64", "sglang"]
    assert args.sglang_flashinfer == Path("/tmp/flashinfer071")
    assert args.shapes == ["q35_target"]


@pytest.mark.parametrize("providers", ["", "unknown", "fi_fixed,", "sglang,cute_typo"])
def test_unknown_providers_rejected(providers):
    with pytest.raises(SystemExit):
        layers.parse_args(["--providers", providers])


def test_mixed_versions_rejected_in_direct_worker():
    with pytest.raises(SystemExit):
        layers.parse_args(
            [
                "--worker",
                "--providers",
                "fi_fixed,sglang",
                "--sglang-flashinfer",
                "/tmp/flashinfer071",
            ]
        )


@pytest.mark.parametrize("isolated", [False, True])
def test_combo_commands_preserve_options_and_split_versions(monkeypatch, capsys, isolated):
    commands = []

    def run(command, *, env, check):
        assert check is False
        args, _ = layers.parse_args(command[3:])
        assert args.worker and args.combos is None
        assert env["CTI_DRAFT_ROWS"] == str(32 if args.combo_index == 0 else 64)
        assert env["CTI_DRAFT_MERGE"] == ("legacy" if args.combo_index == 0 else "distributed")
        assert env["CTI_DRAFT_COOP_PDL"] == str(args.combo_index)
        assert env["CTI_DRAFT_INLINE"] == str(args.combo_index)
        assert env["CTI_DRAFT_SHORT"] == str(0 if args.combo_index == 0 else 192)
        commands.append((command, args))
        return SimpleNamespace(returncode=int(args.providers == ["sglang"]))

    monkeypatch.setattr(layers.subprocess, "run", run)
    argv = [
        "--providers",
        "cute,fi_fixed_s32,sglang",
        "--shapes",
        "q35_target,q3_target",
        "--batches",
        "1,4",
        "--contexts",
        "100,4000",
        "--query",
        "16",
        "--capacity",
        "32768",
        "--vllm-fa2",
        "/tmp/vllm.so",
        "--combos",
        '[{"Q":4,"ROWS":32,"INLINE":0,"SHORT":0,"MERGE":"legacy","COOP_PDL":0},{"CAP":65536,"ROWS":64,"INLINE":1,"SHORT":192,"MERGE":"distributed","COOP_PDL":1}]',
    ]
    if isolated:
        argv += ["--sglang-flashinfer", "/tmp/flashinfer071"]
    assert layers.main(argv) == int(isolated)
    expected = (
        [["cute", "fi_fixed_s32"], ["sglang"]] if isolated else [["cute", "fi_fixed_s32", "sglang"]]
    )
    assert [args.providers for _, args in commands] == expected * 2
    for command, args in commands:
        assert command[:3] == [sys.executable, "-m", "benchmark.kernel_tuning.dflash_draft_layers"]
        assert args.shapes == ["q35_target", "q3_target"]
        assert args.batches == [1, 4] and args.contexts == [100, 4000]
        assert args.vllm_fa2 == Path("/tmp/vllm.so")
        assert (args.query, args.capacity) == [(4, 32768), (16, 65536)][args.combo_index]
        expected_path = Path("/tmp/flashinfer071") if args.providers == ["sglang"] else None
        assert args.sglang_flashinfer == expected_path
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(row["type"] == "combo_result" for row in receipts) == len(commands)


@pytest.mark.parametrize("providers", ["sglang", "fi_fixed_s64"])
def test_filtered_combos_do_not_launch_empty_workers(monkeypatch, providers):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(layers.subprocess, "run", run)
    assert layers.main(["--providers", providers, "--sglang-flashinfer", "/tmp/flashinfer071"]) == 0
    assert len(commands) == 1
    args, _ = layers.parse_args(commands[0][3:])
    assert args.providers == [providers]


def test_unavailable_providers_skip_before_allocating(monkeypatch, capsys):
    # No tensor or CUDA operations are available in this stub.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    args, _ = layers.parse_args(["--providers", "sglang"])
    assert layers.run_case("q35_target", 4, 100, 1, args)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 1
    for row in rows:
        assert row["status"] == row["correct"] == "SKIP"
        assert (row["hq"], row["hk"], row["d"]) == (16, 4, 256)
        assert row["n"] == 8
        assert row["rotating_kv_bytes"] == 8 * 2 * 4 * 256 * 4 * 256 * 2
        assert row["comparator_label"] == layers.COMPARATOR_LABELS[row["provider"]]
    assert rows[0]["reason"] == "--sglang-flashinfer directory absent"


def test_sglang_wrong_version_emits_error_before_allocating(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__="0.6.11.post3"))
    args, _ = layers.parse_args(
        ["--providers", "sglang", "--sglang-flashinfer", "/tmp/flashinfer071"]
    )
    assert not layers.run_case("q35_target", 1, 100, 1, args)
    row = json.loads(capsys.readouterr().out)
    assert row["status"] == "ERROR"
    assert row["reason"] == "SGLang requires FlashInfer 0.7.1"
    assert row["flashinfer_version"] == "0.6.11.post3"


def test_sglang_path_precedes_worker_imports(monkeypatch):
    import builtins

    args, _ = layers.parse_args(
        ["--worker", "--providers", "sglang", "--sglang-flashinfer", "/tmp/flashinfer071"]
    )
    original_import = builtins.__import__
    imports = []

    def import_module(name, *positional, **kwargs):
        if name == "torch":
            assert sys.path[0] == "/tmp/flashinfer071"
            imports.append(name)
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(layers, "pin_cores", lambda: {})
    assert layers.run_worker(args) == 1
    assert imports == ["torch"]


@pytest.mark.parametrize(
    "provider,name",
    [
        ("auto", "q3_target"),
        ("auto", "q35_target"),
        ("cute", "q3_target"),
        ("cute_fused", "q3_target"),
        ("cute", "q35_target"),
        ("cute_fused", "q35_target"),
        ("draft_fi", "q3_target"),
        ("draft_fi", "q35_target"),
    ],
)
@pytest.mark.parametrize("q", [4, 16])
def test_target_constructs_public_causal_full(monkeypatch, provider, name, q):
    import torch

    from ctikernels.attention import draft

    expected_provider = {"auto": None, "draft_fi": "flashinfer"}.get(provider, "cute")
    original_geometry = draft._geometry
    calls = []

    def workspace_size(**geometry):
        assert draft._geometry is original_geometry
        assert geometry.pop("provider") == expected_provider
        assert geometry["causal"] is True and geometry["window"] is None
        original_geometry(**(geometry | {"device": "cuda"}))
        calls.append("size")
        return 256, 256

    def construct(self, *workspaces, **kwargs):
        assert draft._geometry is original_geometry
        assert kwargs["causal"] is True and kwargs["window"] is None
        assert kwargs["provider"] == expected_provider
        assert kwargs["positions"].dtype == torch.int64
        assert kwargs["positions"].tolist() == list(range(8, 8 + q))
        calls.append("construct")
        raise RuntimeError("stop before device execution")

    monkeypatch.setattr(draft.DraftAttention, "workspace_size", staticmethod(workspace_size))
    monkeypatch.setattr(draft.DraftAttention, "__init__", construct)
    args, _ = layers.parse_args(["--query", str(q), "--providers", provider])
    shape = layers.SHAPES[name]
    query = torch.zeros(q, shape["hq"], shape["d"], dtype=torch.bfloat16)
    kv = torch.zeros(32, shape["hk"], shape["d"], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="stop before device execution"):
        layers.benchmark_provider(
            provider,
            query,
            [kv],
            [kv],
            1,
            8,
            32,
            shape["causal"],
            shape["hq"],
            shape["hk"],
            shape["d"],
            {},
            args,
        )
    assert calls == ["size", "construct"]


@pytest.mark.parametrize("key", ["PIECES", "INLINE", "SHORT"])
def test_zero_tuning_values_accepted(key):
    _, combos = layers.parse_args(["--combos", json.dumps([{key: 0}])])
    assert combos == [{key: 0}]
    with pytest.raises(SystemExit):
        layers.parse_args(["--combos", json.dumps([{key: -1}])])


@pytest.mark.parametrize("mode", ["distributed", "tree", "legacy"])
def test_merge_tuning_values(mode):
    _, combos = layers.parse_args(["--combos", json.dumps([{"MERGE": mode, "COOP_PDL": 0}])])
    assert combos == [{"MERGE": mode, "COOP_PDL": 0}]


@pytest.mark.parametrize("value", [0, "invalid", None, []])
def test_invalid_merge_values(value):
    with pytest.raises(SystemExit):
        layers.parse_args(["--combos", json.dumps([{"MERGE": value}])])
