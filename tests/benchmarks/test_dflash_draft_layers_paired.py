# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU coverage of the persistent paired layer protocol and market gate."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import dflash_draft_layers_paired as paired

from ..support.child import run_child


def test_import_does_not_load_gpu_dependencies():
    result = run_child(
        [
            sys.executable,
            "-c",
            f"import sys, json; import {paired.MODULE}; "
            "print(json.dumps([m for m in ('torch', 'flashinfer', 'ctikernels') if m in sys.modules]))",
        ],
        timeout=600,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_defaults_and_single_query_grid():
    args = paired.parse_args([])
    assert args.candidate == "auto"
    assert (args.pairs, args.calls, args.warmup) == (20, 3, 5)
    assert args.shapes == ["draft_noncausal", "q3_target"]
    assert args.batches == [1, 2, 3, 4]
    assert args.contexts == [100, 400, 1000]
    assert args.queries == [4, 5, 6, 8, 9, 12, 16]
    assert args.capacities == [0, 32768]
    assert len(list(paired.generate_rows(args))) == 336
    rows = list(paired.generate_rows(paired.parse_args(["--queries", "9"])))
    assert len(rows) == 48
    assert all(row["Q"] == 9 and row["capacity"] in (row["ctx"] + 9, 32768) for row in rows)
    assert len({tuple(row.items()) for row in rows}) == 48


@pytest.mark.parametrize(
    "argv",
    [
        ["--pairs", "19"],
        ["--calls", "0"],
        ["--warmup", "-1"],
        ["--queries", "1"],
        ["--queries", "33"],
        ["--queries", ""],
        ["--contexts", "-1"],
        ["--batches", "0"],
        ["--shapes", "unknown"],
        ["--capacities", "tiny"],
        ["--capacities", "-1"],
        ["--capacities", "1000"],
        ["--candidate", "sglang"],
        ["--worker", "sglang"],
    ],
)
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        paired.parse_args(argv)


def test_epsilon_uses_nineteenth_sorted_deviation():
    samples = [[1 + index / 100, 1] for index in reversed(range(20))]
    assert paired.epsilon(samples) == pytest.approx(0.18)
    with pytest.raises(ValueError, match="exactly 20"):
        paired.epsilon(samples[:-1])


def test_ratio_median_is_not_ratio_of_medians_and_gate_is_inclusive():
    result = paired.summarize_reference([[1, 1]] * 20, [[1, 2], [10, 10], [100, 20]], True)
    assert result["ratios"] == [0.5, 1, 5]
    assert result["median_ratio"] == 1
    assert result["status"] == "PASS"
    samples = [[1, 10], [10, 20], [100, 1]]
    result = paired.summarize_reference([[1, 1]] * 20, samples, True)
    assert result["median_ratio"] == 0.5
    assert result["candidate_median_us"] / result["reference_median_us"] == 1
    assert paired.summarize_reference([[1, 1]] * 20, [[1.01, 1]], True)["status"] == "FAIL"
    assert paired.summarize_reference([[1, 1]] * 20, [[1, 1]], False)["status"] == "FAIL"


def test_both_references_interleave_and_calibration_is_always_twenty_pairs():
    calls = []
    calibration = [1 + index / 100 for index in range(20)]
    values = iter([value for a in calibration for value in (a, 1)] * 2)

    def measure(name, count):
        calls.append((name, count))
        if len(calls) <= 80:
            return next(values)
        return {"candidate": 1, "vllm_fa2": 2, "sglang": 3}[name]

    correct = dict.fromkeys(("candidate", *paired.REFERENCES), "PASS")
    result = paired.paired_timings(
        measure, paired.REFERENCES, pairs=23, calls=3, correctness=correct
    )
    assert calls[:80] == [("vllm_fa2", 3)] * 40 + [("sglang", 3)] * 40
    assert [name for name, _ in calls[80:88]] == [
        "candidate",
        "vllm_fa2",
        "candidate",
        "sglang",
        "vllm_fa2",
        "candidate",
        "sglang",
        "candidate",
    ]
    assert len(calls) == 80 + 23 * 4
    for name, ref in result["references"].items():
        assert ref["epsilon"] == pytest.approx(0.18)
        assert len(ref["reference_self_pairs_us"]) == 20
        assert len(ref["candidate_reference_pairs_us"]) == 23
        assert ref["candidate_reference_pairs_us"] == [[1, 2 if name == "vllm_fa2" else 3]] * 23
    assert result["faster_market"] == "vllm_fa2"
    assert result["market_gate"] == "PASS"


@pytest.mark.parametrize("failed", ["candidate", "vllm_fa2", "sglang", None])
def test_market_requires_all_correctness_checks(failed):
    refs = {
        name: paired.summarize_reference([[1, 1]] * 20, [[1, 2]], True)
        for name in paired.REFERENCES
    }
    correctness = dict.fromkeys(("candidate", *paired.REFERENCES), "PASS")
    if failed:
        correctness[failed] = "FAIL"
    assert paired.market_decision(refs, correctness)["market_gate"] == (
        "FAIL" if failed else "PASS"
    )
    refs["sglang"]["status"] = "FAIL"
    assert paired.market_decision(refs, correctness)["market_gate"] == "FAIL"


@pytest.mark.parametrize("available", [[], ["vllm_fa2"], ["sglang"]])
def test_missing_comparator_never_passes(available):
    correctness = {
        name: "PASS" if name in ("candidate", *available) else "N/A"
        for name in ("candidate", *paired.REFERENCES)
    }
    result = paired.paired_timings(
        lambda *_: 1, available, pairs=20, calls=3, correctness=correctness
    )
    assert result["market_gate"] == "N/A"
    assert result["faster_market"] is None
    for name in paired.REFERENCES:
        assert result["references"][name]["status"] == ("PASS" if name in available else "N/A")


def test_artifact_records_only_basename_and_digest(tmp_path):
    library = tmp_path / "vllm.so"
    library.write_bytes(b"test artifact")
    args = SimpleNamespace(vllm_fa2=library, vllm_fa2_wheel=str(tmp_path / "vllm.whl"))
    assert paired.artifact(args) == {
        "wheel": "vllm.whl",
        "so_sha256": hashlib.sha256(b"test artifact").hexdigest(),
    }


@pytest.mark.parametrize("crash_command", [None, "setup", "measure", "teardown"])
def test_parent_persists_workers_and_restarts_after_row_error(
    monkeypatch, capsys, tmp_path, crash_command
):
    library = tmp_path / "vllm.so"
    library.write_bytes(b"artifact")
    instances = []
    requests = []
    crashed = False

    class Worker:
        def __init__(self, kind, args):
            self.kind = kind
            self.closed = False
            instances.append(self)

        def receive(self):
            return {"device": "test", "device_uuid": "test-id", "sms": 1, "l2_bytes": 1024}

        def request(self, command, **fields):
            nonlocal crashed
            requests.append((self.kind, command, fields))
            if command == crash_command and not crashed:
                crashed = True
                raise paired.WorkerError("test failure")
            if command == "setup":
                providers = ["candidate", "vllm_fa2"] if self.kind == "main" else ["sglang"]
                return {
                    "providers": {
                        name: {"correct": "PASS", "op.provider": "cute", "cute_tuning": {}}
                        for name in providers
                    },
                    "geometry": {"n": 5},
                }
            return {"us": 1}

        def close(self):
            self.closed = True

    monkeypatch.setattr(paired, "Worker", Worker)
    args = paired.parse_args(
        [
            "--queries",
            "4",
            "--shapes",
            "draft_noncausal",
            "--batches",
            "1",
            "--contexts",
            "100",
            "--vllm-fa2",
            str(library),
            "--sglang-flashinfer",
            str(tmp_path),
        ]
    )
    assert paired.run_parent(args) == int(crash_command is not None)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["type"] for r in receipts] == ["header", "row", "row"]
    assert receipts[-1]["status"] == "PASS"
    if crash_command:
        assert receipts[1]["status"] == "ERROR"
        assert len(instances) == 4
        assert receipts[-1]["worker_generation"] == 2
    else:
        assert len(instances) == 2
        assert receipts[-1]["worker_generation"] == 1
        assert sum(command == "setup" for _, command, _ in requests) == 4
        assert sum(command == "measure" for _, command, _ in requests) == 320
        assert [fields["calls"] for _, command, fields in requests if command == "warmup"] == [
            5
        ] * 6
        assert sum(command == "teardown" for _, command, _ in requests) == 4
    assert all(worker.closed for worker in instances)


def test_worker_commands_isolate_flashinfer_and_close_with_eof(monkeypatch):
    import io

    processes = []

    def popen(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", paired.MODULE]
        args = paired.parse_args(command[3:])
        assert (args.sglang_flashinfer is not None) == (args.worker == "sglang")
        assert (args.vllm_fa2 is not None) == (args.worker == "main")
        process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), wait=lambda: 0)
        processes.append(process)
        return process

    monkeypatch.setattr(paired.subprocess, "Popen", popen)
    args = paired.parse_args(["--vllm-fa2", "vllm.so", "--sglang-flashinfer", "flashinfer071"])
    for kind in ("main", "sglang"):
        worker = paired.Worker(kind, args)
        with pytest.raises(paired.WorkerError):
            worker.receive()
        worker.close()
    assert all(process.stdin.closed and process.stdout.closed for process in processes)


def test_worker_row_captures_once_checks_last_layer_and_times_replays(monkeypatch):
    events = []
    captures = []
    expected = object()
    query = object()
    keys, values = [object(), object()], [object(), object()]

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            events.append("record")

        def synchronize(self):
            events.append("event_sync")

        def elapsed_time(self, end):
            return 0.06

    def reference(q, k, v, *args):
        assert (q, k, v) == (query, keys[-1], values[-1])
        assert events[-1] == "sync"
        return expected

    def build(provider, *args):
        metadata = args[-2]
        metadata["op.provider"] = "cute-splitkv" if provider == "auto" else None
        return (
            lambda i: events.append((provider, i)),
            SimpleNamespace(float=lambda: expected),
            [object()],
        )

    def capture(fn, n):
        assert n == 2
        captures.append(fn)
        return SimpleNamespace(replay=lambda: events.append("replay"))

    def close(actual, oracle, **tolerance):
        assert actual is oracle is expected
        assert tolerance == {"atol": 2e-2, "rtol": 2e-2}
        return True

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=Event,
            synchronize=lambda: events.append("sync"),
            empty_cache=lambda: events.append("empty_cache"),
        ),
        allclose=close,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        paired.layers,
        "layer_geometry",
        lambda *_: {"n": 2, "lpad": 256, "causal": False, "hq": 32, "hk": 8, "d": 128},
    )
    monkeypatch.setattr(paired.layers, "layer_buffers", lambda *_: (query, keys, values))
    monkeypatch.setattr(paired.layers, "build_provider", build)
    monkeypatch.setattr(paired.layers, "capture_graph", capture)
    monkeypatch.setattr(paired.layers, "reference", reference)
    args = paired.parse_args(["--worker", "main", "--vllm-fa2", "vllm.so"])
    row = paired.WorkerRow(
        {"shape": "draft_noncausal", "B": 1, "ctx": 100, "Q": 4, "capacity": 104}, args, 1024
    )
    assert len(captures) == 2
    assert all(provider["correct"] == "PASS" for provider in row.report.values())
    assert row.report["candidate"]["op.provider"] == "cute-splitkv"
    events.clear()
    assert row.measure("candidate", 3) == pytest.approx(10)
    assert events == ["sync", "record", "replay", "replay", "replay", "record", "event_sync"]
    events.clear()
    row.warmup("vllm_fa2", 5)
    assert events == ["replay"] * 5 + ["sync"]
    assert len(captures) == 2
    row.close()
    assert not row.graphs and not row.retained
    assert events[-1] == "empty_cache"


@pytest.mark.parametrize(
    "version,valid_path", [("0.7.1", True), ("0.6.11.post3", True), ("0.7.1", False)]
)
def test_sglang_worker_checks_version_path_and_reports_relative_import(
    monkeypatch, tmp_path, version, valid_path
):
    from pathlib import Path

    args = paired.parse_args(["--worker", "sglang", "--sglang-flashinfer", str(tmp_path)])
    package = tmp_path / ("flashinfer" if valid_path else "elsewhere") / "__init__.py"
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        cuda=SimpleNamespace(
            set_device=lambda index: None,
            get_device_properties=lambda index: SimpleNamespace(
                name="test", multi_processor_count=1, L2_cache_size=1024
            ),
        ),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(__version__=version, __file__=str(package))
    )
    monkeypatch.setattr(paired.layers, "pin_cores", lambda: {"cores": [3]})
    monkeypatch.setattr(paired.importlib.metadata, "version", lambda name: "test-cti")
    monkeypatch.setattr(
        paired.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(tmp_path / "ctikernels" / "__init__.py")),
    )
    if version != "0.7.1" or not valid_path:
        with pytest.raises(ValueError, match="supplied FlashInfer 0.7.1"):
            paired.worker_info(args)
    else:
        info = paired.worker_info(args)
        assert info["cores"] == [3]
        assert info["versions"]["cti-kernels"] == "test-cti"
        assert info["versions"]["flashinfer"] == "0.7.1"
        assert info["versions"]["ctikernels_import_path"] == "ctikernels/__init__.py"
        assert not fake_torch.backends.cuda.matmul.allow_tf32
    assert Path(sys.path[0]) == tmp_path


def test_worker_protocol_redirects_python_and_native_chatter_without_cuda():
    script = """
import contextlib
import os
import sys
from types import SimpleNamespace
from benchmark.kernel_tuning import dflash_draft_layers_paired as paired
sys.modules["torch"] = SimpleNamespace(no_grad=contextlib.nullcontext)
def info(args):
    print("python chatter")
    os.write(1, b"native chatter\\n")
    return {"l2_bytes": 1024}
class Row:
    report = {"candidate": {"correct": "PASS"}}
    geometry = {"n": 5}
    def __init__(self, *args):
        pass
    def measure(self, *args):
        return 2.0
    def close(self):
        print("teardown chatter")
paired.worker_info = info
paired.WorkerRow = Row
raise SystemExit(paired.run_worker(SimpleNamespace()))
"""
    requests = [
        {"command": "setup", "row": {}},
        {"command": "measure", "provider": "candidate", "calls": 3},
        {"command": "teardown"},
    ]
    result = run_child(
        [sys.executable, "-c", script],
        input="".join(json.dumps(request) + "\n" for request in requests),
        timeout=600,
        check=True,
    )
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [reply["status"] for reply in replies] == ["READY", "OK", "OK", "OK"]
    assert replies[2]["us"] == 2.0
    assert all(
        text in result.stderr for text in ("python chatter", "native chatter", "teardown chatter")
    )
