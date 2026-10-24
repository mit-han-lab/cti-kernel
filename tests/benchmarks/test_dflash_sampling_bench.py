# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Benchmark diagnostics and measured launch accounting, without CUDA execution."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[2] / "benchmark/dflash_sampling/bench.py"
    spec = importlib.util.spec_from_file_location("dflash_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_count_uses_cuda_kernel_events(bench, monkeypatch):
    events = [
        SimpleNamespace(device_type=device, name=name)
        for device, name in (
            (torch.autograd.DeviceType.CPU, "cudaLaunchKernel"),
            (torch.autograd.DeviceType.CUDA, "chunk_partials"),
            (torch.autograd.DeviceType.CUDA, "chunk_finalize"),
            (torch.autograd.DeviceType.CUDA, "Memcpy DtoD (Device -> Device)"),
            (torch.autograd.DeviceType.CUDA, "Memset (Device)"),
        )
    ]
    monkeypatch.setattr(
        torch.profiler, "profile", lambda **_: nullcontext(SimpleNamespace(events=lambda: events))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []
    result = bench.count_launches(lambda: calls.append(True))
    assert calls == [True]
    assert result == {"kernel_launches": 2, "launch_count_source": "torch.profiler eager call"}


def test_missing_cupti_is_not_a_static_launch_count(bench, monkeypatch, capsys):
    monkeypatch.setattr(
        torch.profiler, "profile", lambda **_: nullcontext(SimpleNamespace(events=lambda: []))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    result = bench.count_launches(lambda: None)
    assert result["kernel_launches"] is None
    assert "CUPTI" in result["launch_count_detail"]
    assert "Traceback" in capsys.readouterr().err


def test_comparator_failure_reports_detail_and_traceback(bench, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    detail = "comparator import failed: " + "x" * 600

    def fail(*_, **kwargs):
        raise RuntimeError(detail)

    monkeypatch.setattr(bench, "make_case", fail)
    bench.paired_worker(1)
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert [r["case"] for r in records] == ["rejection", "bonus"]
    assert all(r["detail"] == detail[:500] and r["reason"] == "RuntimeError" for r in records)
    assert captured.err.count("Traceback") == 2
    assert detail in captured.err


def test_paired_cli_preserves_interpreter_and_appends_paths(bench, monkeypatch, tmp_path):
    first, second = tmp_path / "sglang", tmp_path / "dependencies"
    first.mkdir()
    second.mkdir()
    original = list(bench.sys.path)
    monkeypatch.setattr(bench.sys, "path", list(original))
    monkeypatch.setattr(
        bench.sys, "argv", ["bench", "--sglang-path", str(first), "--sglang-path", str(second)]
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(bench, "measurements", lambda: calls.append("measurements"))
    monkeypatch.setattr(bench, "paired_worker", lambda b, paths, **kw: calls.append((b, paths, kw)))
    bench.main()
    paths = (str(first), str(second))
    assert bench.sys.path == original + list(paths)
    assert calls == [
        "measurements",
        *((b, paths, kw) for b in (1, 2, 3, 4, 8) for kw in ({}, {"point": True})),
    ]


def test_deprecated_interpreter_errors_before_measurements(bench, monkeypatch, capsys):
    monkeypatch.setattr(bench.sys, "argv", ["bench", "--sglang-python", "python"])
    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "use --sglang-path DIR" in capsys.readouterr().err


def test_import_provenance_is_relative_and_emitted_once(bench, monkeypatch, tmp_path, capsys):
    name = "sglang.kernels.ops.speculative.reject_sampling"
    relative = "sglang/kernels/ops/speculative/reject_sampling.py"
    chain = object()
    module = SimpleNamespace(
        __name__=name,
        __file__=str(tmp_path / relative),
        chain_speculative_sampling_triton=chain,
    )
    imports = []

    def import_module(requested):
        imports.append(requested)
        return module

    monkeypatch.setattr(bench.importlib, "import_module", import_module)
    paths = (str(tmp_path),)
    assert bench.load_comparator(paths) == ("sglang-chain-triton", chain)
    assert bench.load_comparator(paths) == ("sglang-chain-triton", chain)
    assert imports == [name]
    captured = capsys.readouterr()
    assert captured.out == ""
    record = json.loads(captured.err)
    assert record["module_file"] == relative
    assert record["relative_to"] == "sglang-path[0]"
    assert str(tmp_path) not in captured.err


@pytest.mark.parametrize("mismatch", [None, "tokens", "counts", "indices"])
def test_output_match_gathers_emitted_prefix_and_ignores_unused_tail(bench, mismatch):
    # Row 0 rejects immediately; row 1 accepts both drafts and emits a bonus.
    ours = torch.tensor([[7, -1, -1], [4, 5, 6]], dtype=torch.int32)
    ours_counts = torch.tensor([0, 2], dtype=torch.int32)
    predicts = torch.tensor([5, 7, 6, 4, 99, 99], dtype=torch.int32)
    indices = torch.tensor([[1, -99, 999], [3, 0, 2]])
    counts = ours_counts.clone()
    if mismatch == "tokens":
        predicts[2] = 8
    elif mismatch == "counts":
        counts[0] = 1
    elif mismatch == "indices":
        indices[0, 0] = 999
    result = bench.compare_outputs(ours, ours_counts, predicts, indices, counts)
    assert result["output_match"] is (mismatch is None)
    assert result["counts_match"] is (mismatch != "counts")
    assert result["tokens_match"] is (mismatch not in ("tokens", "indices"))


def test_probability_benchmark_hints_and_launch_records(bench, monkeypatch, capsys):
    from ctikernels.sampling._dflash import RequestLocalChainSampling

    class Sampler:
        probability_launch_counts = RequestLocalChainSampling.probability_launch_counts

        def __init__(self, *_):
            self.hint = None

        def workspace_size(self, *_):
            return 256

        def probabilities(self, logits, temperatures, top_k, top_p, *, filters, **_):
            self.hint = filters
            if filters in ("none", "top_p"):
                assert (top_k == logits.shape[-1]).all()
            if filters in ("none", "top_k"):
                assert (top_p == 1).all()

    def on_cpu(factory):
        return lambda *args, **kwargs: factory(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        )

    fake_torch = SimpleNamespace(
        **{
            name: on_cpu(getattr(torch, name))
            for name in ("Generator", "randn", "empty", "arange", "full")
        },
        **{name: getattr(torch, name) for name in ("float32", "bfloat16", "int32", "uint8")},
        cuda=SimpleNamespace(get_device_name=lambda: "test device"),
    )
    monkeypatch.setattr(bench, "torch", fake_torch)
    monkeypatch.setattr(bench, "RequestLocalChainSampling", Sampler)
    monkeypatch.setattr(bench, "V", 64)
    monkeypatch.setattr(bench, "PAIRS", 1)
    monkeypatch.setattr(bench, "capture", lambda call: call())
    monkeypatch.setattr(bench, "count_launches", lambda call: {"kernel_launches": None})
    monkeypatch.setattr(bench, "time_graph", lambda _: 1.0)
    bench.probability_measurements()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 35
    expected_modes = [
        ("identity", "none"),
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("top_k+top_p", "both"),
        ("top_k+top_p", "auto"),
        ("production", "both"),
        ("greedy", "none"),
    ]
    for batch in (1, 2, 3, 4, 8):
        subset = [record for record in records if record["batch"] == batch]
        assert [(record["case"], record["filters"]) for record in subset] == expected_modes
        for record in subset:
            assert (
                record["expected_kernel_launches"]
                == Sampler.probability_launch_counts[record["filters"]]
            )
            assert record["kernel_launches"] is None  # measured count never fabricated
        production = subset[5]
        assert (production["temperature"], production["top_k"], production["top_p"]) == (
            0.6,
            20,
            0.95,
        )


@pytest.mark.parametrize("case", ["rejection", "bonus"])
def test_point_mass_fixture_and_sglang_fixed_uniforms(bench, monkeypatch, case):
    from ctikernels.sampling._dflash import RequestLocalChainSampling
    from ctikernels.sampling._dflash.reference import ACCEPT, BONUS, CORRECTION, uniform

    def on_cpu(factory):
        return lambda *args, **kwargs: factory(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        )

    fake_torch = SimpleNamespace(
        **{name: getattr(torch, name) for name in dir(torch)},
    )
    for name in ("full", "zeros", "ones", "empty", "tensor", "arange"):
        setattr(fake_torch, name, on_cpu(getattr(torch, name)))
    monkeypatch.setattr(bench, "torch", fake_torch)
    monkeypatch.setattr(bench, "V", 17)
    monkeypatch.setattr(
        bench,
        "RequestLocalChainSampling",
        lambda _, q, v: RequestLocalChainSampling("cpu", q, v),
    )
    op, call, data = bench.make_case(3, False, case, point=True)
    p, q, drafts, temp, common, out, counts = data
    assert torch.equal(q.argmax(-1).int(), drafts)
    assert (q[:, :, 0] == 1).all() and (q[:, :, 1:] == 0).all()
    call()
    assert counts.tolist() == [0 if case == "rejection" else 15] * 3
    expected = out.clone()
    expected_counts = counts.clone()
    op.verify_dense(p, drafts, q, temp, **common, out_tokens=out, accepted_counts=counts)
    assert torch.equal(out, expected) and torch.equal(counts, expected_counts)
    received = []
    monkeypatch.setattr(
        bench,
        "load_comparator",
        lambda _: ("sglang-chain-triton", lambda *args, **kw: received.append((args, kw))),
    )
    _, comparator, _ = bench.comparator_call(3, case, data)
    comparator()
    args, kwargs = received[0]
    assert torch.equal(args[10][:, :15], q)
    assert (args[10][:, 15] == 0).all()
    assert args[9] is p
    assert kwargs == {"block_verification": False}
    assert torch.equal(
        args[7],
        torch.tensor([[uniform(42, 2**32 + b, ACCEPT, i) for i in range(16)] for b in range(3)]),
    )
    domain = CORRECTION if case == "rejection" else BONUS
    assert torch.equal(args[8], torch.tensor([uniform(42, 2**32 + b, domain, 0) for b in range(3)]))


@pytest.mark.parametrize("match", [False, True])
def test_point_mass_comparator_gates_and_pairs(bench, monkeypatch, capsys, match):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench, "make_case", lambda *args, **kw: (None, lambda: None, None))
    matches = {"output_match": match, "tokens_match": match, "counts_match": match}
    monkeypatch.setattr(
        bench,
        "comparator_call",
        lambda *args: ("sglang-chain-triton", lambda: None, lambda: matches),
    )
    monkeypatch.setattr(bench, "capture", lambda _: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(bench, "count_launches", lambda _: {"kernel_launches": 2})
    timings = []
    monkeypatch.setattr(bench, "time_graph", lambda graph: timings.append(graph) or 1.0)
    bench.paired_worker(3, point=True)
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert all(r["op"] == "verify_point_mass" for r in records)
    # A mismatch is reported (not dropped) and timing continues with the flag carried.
    mismatches = [r for r in records if r.get("record_type") == "output_mismatch"]
    assert len(mismatches) == (0 if match else 2)
    assert len(records) == 42 + len(mismatches)  # 20 pairs + summary, rejection and bonus
    summaries = [r for r in records if r.get("record_type") == "paired_summary"]
    assert len(summaries) == 2
    assert all(r["pairs"] == 20 and r["output_match"] == match for r in summaries)
    assert all(r["epsilon"] == 0 and r["ci_lower"] == r["ci_upper"] == 1 for r in summaries)
    assert len(timings) == 240  # 20 comparator self-pairs + 20 measured/self-pairs per case
