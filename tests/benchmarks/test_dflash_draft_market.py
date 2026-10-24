# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only coverage of market rows, dispatch metadata, and measurement protocol."""

import hashlib
import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import dflash_draft_market as market

from ..support.child import run_child


def default_rows(**overrides):
    args = market.parse_args(["--comparators", "flashinfer"])
    options = {
        name: getattr(args, name)
        for name in ("queries", "batches", "contexts", "masks", "heads", "shard")
    }
    return market.generate_rows(**(options | overrides))


def test_default_rows_include_boundaries_without_duplicates():
    rows = default_rows()
    assert len(rows) == 896
    assert [row["row_id"] for row in rows] == list(range(len(rows)))
    assert {row["query_length"] for row in rows} == {4, 5, 6, 8, 9, 12, 16}
    assert {row["batch_size"] for row in rows} == {1, 2, 3, 4}
    assert {(row["num_qo_heads"], row["num_kv_heads"]) for row in rows} == {(32, 8), (16, 4)}
    for mask, boundary in (("full", 4096), ("causal_sliding", 4096), ("noncausal_sliding", 2048)):
        selected = [row for row in rows if row["mask"] == mask]
        assert {row["context"] for row in selected} == {
            128,
            512,
            4096,
            boundary - 1,
            boundary,
            boundary + 1,
        }
        assert all((row["causal"], row["window"]) == market.MASKS[mask] for row in selected)
    identities = [tuple(value for key, value in row.items() if key != "row_id") for row in rows]
    assert len(set(identities)) == len(rows)


@pytest.mark.parametrize("count", [1, 2, 7, 19])
def test_shards_partition_rows_exactly_once_in_stable_order(count):
    rows = default_rows()
    shards = [default_rows(shard=(index, count)) for index in range(count)]
    assert all(shard == rows[index::count] for index, shard in enumerate(shards))
    assert Counter(row["row_id"] for shard in shards for row in shard) == Counter(range(len(rows)))


def test_custom_grid_adds_boundaries_and_deduplicates():
    rows = default_rows(
        queries=[5, 5],
        batches=[3],
        contexts=[0, 2048, 2048],
        masks=["noncausal_sliding"],
        heads=[(16, 4)],
    )
    assert [row["context"] for row in rows] == [0, 2047, 2048, 2049]
    assert default_rows(shard=(999, 1000)) == []


CPUINFO = """processor : 0
CPU implementer : 0x41
CPU part : 0xd85

processor : 1
CPU part : 0xd81

processor\t: 3
CPU part\t: 0xD85

processor : malformed
CPU part : 0xd85

CPU part : 0xd85

processor : 8
model name : unrelated
"""


def test_x925_cpuinfo_parsing():
    assert market.x925_cores(CPUINFO) == [0, 3]
    assert market.x925_cores("") == []
    assert market.x925_cores("processor : 0\nCPU part : invalid\n") == []


def test_pinning_intersects_allowed_cores_and_records_result(monkeypatch):
    affinity = {1, 3}
    monkeypatch.setattr(market.Path, "read_text", lambda _: CPUINFO)
    monkeypatch.setattr(market.os, "sched_getaffinity", lambda _: affinity.copy())

    def set_affinity(pid, cores):
        assert pid == 0
        affinity.intersection_update(cores)

    monkeypatch.setattr(market.os, "sched_setaffinity", set_affinity)
    assert market.pin_x925(True) == {"requested": True, "cpu_part": "0xd85", "cores": [3]}
    affinity.clear()
    with pytest.raises(RuntimeError, match="no allowed Cortex-X925"):
        market.pin_x925(True)


@pytest.mark.parametrize(
    "causal,window,expected",
    [
        (False, None, (-1, -1)),
        (True, 4096, (4095, 0)),
        (False, 2048, (2047, 2047)),
    ],
)
def test_vllm_window_mapping(causal, window, expected):
    assert market.vllm_window(causal, window) == expected


def test_row_keys_select_generated_rows():
    key = market.parse_row_key("full:2:9:4096:16/4")
    assert key == ("full", 2, 9, 4096, 16, 4)
    assert sum(market.row_key(row) == key for row in default_rows()) == 1
    for value in ("full:2:9:4096", "diag:2:9:4096:16/4", "full:2:9:4096:16/5"):
        with pytest.raises(market.argparse.ArgumentTypeError):
            market.parse_row_key(value)


@pytest.mark.parametrize("value", ["1/0", "-1/3", "3/3", "1", "1/2/3", "x/3"])
def test_invalid_shards_are_rejected(value):
    with pytest.raises(market.argparse.ArgumentTypeError):
        market.parse_shard(value)


def test_import_does_not_load_gpu_dependencies():
    result = run_child(
        [
            sys.executable,
            "-c",
            "import sys, json; from benchmark.kernel_tuning import dflash_draft_market; "
            "print(json.dumps([m for m in ('torch', 'flashinfer', 'ctikernels') if m in sys.modules]))",
        ],
        timeout=600,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_parent_starts_one_sequential_worker_per_comparator(monkeypatch, tmp_path):
    (tmp_path / "flashinfer").mkdir()
    library = tmp_path / "_vllm_fa2_C.abi3.so"
    library.write_bytes(b"test artifact")
    commands = []

    def run(command, *, check):
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=int(command[-1] == "sglang"))

    monkeypatch.setattr(market.subprocess, "run", run)
    assert (
        market.main(
            ["--sglang-flashinfer", str(tmp_path), "--vllm-fa2", str(library), "--shard", "1/3"]
        )
        == 1
    )
    assert [command[-2:] for command in commands] == [
        ["--worker", kind] for kind in market.COMPARATORS
    ]
    assert all(
        command[:3] == [sys.executable, "-m", "benchmark.kernel_tuning.dflash_draft_market"]
        for command in commands
    )
    assert all(command[-4:-2] == ["--shard", "1/3"] for command in commands)


def test_nondefault_pairs_keep_twenty_self_pairs_and_alternate(monkeypatch):
    measurements = []
    samples = iter([value for i in range(20) for value in (1 + i / 100, 1)] + [1] * 6)

    def measure(call, repetitions):
        measurements.append((call, repetitions))
        return next(samples)

    from benchmark import kernel_tuning

    protocol = SimpleNamespace(measure=measure)
    monkeypatch.setattr(kernel_tuning, "dflash_draft_attention", protocol, raising=False)
    monkeypatch.setitem(sys.modules, "benchmark.kernel_tuning.dflash_draft_attention", protocol)
    result = market.paired_timings("candidate", "reference", pairs=3, calls=100)
    assert result["epsilon"] == pytest.approx(0.18)
    assert result["speed"] == "PASS"
    assert len(result["reference_self_pairs_us"]) == 20
    assert len(result["candidate_reference_pairs_us"]) == 3
    assert measurements[:40] == [("reference", 100)] * 40
    assert measurements[40:] == [
        (name, 100)
        for name in ("candidate", "reference", "reference", "candidate", "candidate", "reference")
    ]


def test_vllm_artifact_hash_is_verified_and_only_wheel_basename_is_emitted(tmp_path):
    library = tmp_path / "_vllm_fa2_C.abi3.so"
    library.write_bytes(b"test artifact")
    wheel = "vllm-0.31.0-cp38-abi3-manylinux_2_35_aarch64.whl"
    (tmp_path / "wheel.txt").write_text(wheel + "\n")
    sha = hashlib.sha256(library.read_bytes()).hexdigest()
    args = SimpleNamespace(vllm_fa2=library, vllm_fa2_sha256=sha, vllm_wheel=None)
    assert market.vllm_artifact(args) == {"wheel": wheel, "so_sha256": sha, "version": "0.31.0"}
    args.vllm_wheel = str(tmp_path / wheel)
    assert market.vllm_artifact(args)["wheel"] == wheel
    args.vllm_fa2_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        market.vllm_artifact(args)


def test_default_pairs_reuse_existing_protocol(monkeypatch):
    from benchmark import kernel_tuning

    def paired(candidate, reference, *, calls):
        assert (candidate, reference, calls) == ("candidate", "reference", 100)
        return {"protocol": "existing"}

    protocol = SimpleNamespace(paired_timings=paired)
    monkeypatch.setattr(kernel_tuning, "dflash_draft_attention", protocol, raising=False)
    monkeypatch.setitem(sys.modules, "benchmark.kernel_tuning.dflash_draft_attention", protocol)
    assert market.paired_timings("candidate", "reference", pairs=20, calls=100) == {
        "protocol": "existing"
    }


def test_vllm_worker_loads_library_without_flashinfer_and_summarizes(monkeypatch, capsys):
    args = market.parse_args(["--comparators", "flashinfer"])
    args.vllm_fa2 = "_vllm_fa2_C.abi3.so"
    libraries = []
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        ops=SimpleNamespace(load_library=libraries.append),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    # An attempted FlashInfer import raises ModuleNotFoundError even if installed.
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(market, "pin_x925", lambda _: {"requested": False, "cores": [0]})
    monkeypatch.setattr(market, "vllm_artifact", lambda _: {"version": "test"})
    rows = default_rows()[:2]

    def run_case(kind, row, options):
        assert kind == "vllm" and options is args
        return {"status": "PASS" if row["row_id"] == 0 else "FAIL"}

    monkeypatch.setattr(market, "run_case", run_case)
    assert market.worker("vllm", rows, args) == 1
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert libraries == [args.vllm_fa2]
    assert output[0]["type"] == "header"
    assert output[0]["versions"]["flashinfer"] is None
    assert output[0]["setup_error"] is None
    assert [row["page_size"] for row in output[1:-1]] == [16, 16]
    assert output[-1] == {
        "type": "summary",
        "comparator": "vllm",
        "pass": 1,
        "fail": 1,
        "status": "FAIL",
    }


def test_failed_worker_setup_still_emits_header_rows_and_summary(monkeypatch, capsys):
    args = market.parse_args(["--comparators", "flashinfer"])
    monkeypatch.setitem(sys.modules, "torch", None)
    rows = default_rows()[:2]
    assert market.worker("flashinfer", rows, args) == 1
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["type"] == "header"
    assert output[0]["setup_error"] == "ModuleNotFoundError"
    for row in output[1:-1]:
        assert row["status"] == "FAIL" and row["error_stage"] == "setup"
        assert row["median_ratio"] is None
        assert row["candidate_reference_pairs_us"] == []
    assert output[-1]["pass"] == 0 and output[-1]["fail"] == len(rows)
