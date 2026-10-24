# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import norm_act_probe as probe
from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard, profiler_clean
from benchmark.kernel_tuning.norm_act_vllm import MarketError
from ctikernels.norm._dispatch import Config

from ..support.child import run_child


def spec_for(kind="worker", op="gemma_add_norm_sum"):
    return {
        "kind": kind,
        "op": op,
        "m": 16,
        "width": 4096,
        "dtype": "bf16",
        "market_root": Path("/tmp/market with spaces"),
        "static_weight": False,
        "candidate": "cute_remote",
        "config": Config(),
    }


KERNELS = [{"name": "test_kernel", "grid": [16, 1, 1], "block": [256, 1, 1]}]


@pytest.mark.parametrize("kind", ["vllm", "provider", "worker", "sglang_silu"])
def test_probe_spec_json_roundtrip(kind):
    spec = spec_for(kind)
    config = {"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 8, "num_stages": 1}
    spec.update(site="d2", mode="shipped", configs=[config], aux_configs=[config], aux=True)
    encoded = probe.encode_spec(spec)
    result = json.loads(encoded)
    assert result == spec | {
        "config": asdict(spec["config"]),
        "market_root": str(spec["market_root"]),
    }
    assert probe.encode_spec(result) == encoded


@pytest.mark.parametrize(
    "change", [{"unexpected": "field"}, {"dtype": "fp32"}, {"m": 0}, {"kind": "unknown"}]
)
def test_probe_spec_rejects_bad_input(change):
    with pytest.raises(ValueError):
        probe.encode_spec(spec_for() | change)


def test_probe_handoff_synchronizes_and_preserves_device_and_overlay(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5")
    monkeypatch.setattr(sys, "path", ["/tmp/fi071-overlay", *sys.path])

    def run(command, **kwargs):
        assert events == ["sync"]
        events.append("probe")
        assert command[:4] == [
            sys.executable,
            "-m",
            "benchmark.kernel_tuning.norm_act_probe",
            "--spec",
        ]
        assert json.loads(command[4])["device"] == 2
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,4,5"
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == "/tmp/fi071-overlay"
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert "timeout" not in kwargs  # no timeout-induced process termination
        return SimpleNamespace(returncode=0, stdout=json.dumps({"kernels": KERNELS}))

    monkeypatch.setattr(subprocess, "run", run)
    assert probe.run_probe(spec_for()) == {"kernels": KERNELS}
    assert events == ["sync", "probe"]


@pytest.mark.parametrize(
    "code,stdout",
    [
        (1, "native diagnostic"),
        (0, "bad JSON"),
        (0, '{"kernels":[]}'),
        (0, '{"kernels":[{"name":"x","grid":null,"block":null}]}'),
    ],
)
def test_probe_failure_is_market_error_without_native_diagnostics(monkeypatch, code, stdout):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout)
    )
    with pytest.raises(MarketError) as error:
        probe.run_probe(spec_for())
    assert str(error.value) in {"kernel probe failed", "invalid kernel probe receipt"}


@pytest.mark.parametrize("api", ["context", "start", "legacy", "prepare"])
def test_timing_guard_rejects_activation_before_kineto(api):
    install_profiler_guard()
    install_profiler_guard()
    assert profiler_clean()
    with pytest.raises(RuntimeError, match="profiling is forbidden"):
        if api == "context":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
                pytest.fail("entered profiler")
        elif api == "start":
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]).start()
        elif api == "prepare":
            torch.profiler._KinetoProfile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ).prepare_trace()
        else:
            with torch.autograd.profiler.profile():
                pytest.fail("entered legacy profiler")
    assert profiler_clean()


def test_guard_rejects_active_profiler_and_tampering(monkeypatch):
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", True)
    with pytest.raises(RuntimeError, match="already active"):
        install_profiler_guard()
    with pytest.raises(RuntimeError, match="not intact"):
        profiler_clean()
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", False)
    monkeypatch.setattr(torch.profiler.profile, "start", lambda self: None)
    with pytest.raises(RuntimeError, match="not intact"):
        profiler_clean()


def test_probe_interpreter_can_profile_without_touching_parent_guard():
    # Exercise the actual module import graph in a disposable CPU probe interpreter.
    result = run_child(
        [
            sys.executable,
            "-c",
            "import benchmark.kernel_tuning.norm_act_probe; "
            "import benchmark.kernel_tuning.norm_act_paired; import torch; "
            "p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]); "
            "p.start(); torch.ones(1).add_(1); p.stop(); print('probe-ok')",
        ],
        timeout=600,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "probe-ok"
    assert profiler_clean()


def test_worker_profile_reconstructs_spec_without_using_shared_tensors(monkeypatch):
    from benchmark.kernel_tuning import norm_act_market as market

    worker = market.Worker.__new__(market.Worker)
    worker.root = Path("/tmp/market")
    worker.spec = spec_for()
    worker.cute_config = asdict(Config())
    worker.pools = object()  # profiling must never read or restore IPC tensors
    seen = []
    monkeypatch.setattr(market, "run_probe", lambda spec: seen.append(spec) or {"kernels": KERNELS})
    assert worker.dispatch("profile", {"candidate": "cute_remote"}) == {"kernels": KERNELS}
    assert seen[0]["config"] == worker.cute_config
    assert seen[0]["static_weight"] is False
    assert seen[0]["op"] == "gemma_add_norm_sum"
    assert seen[0]["market_root"] == worker.root


@pytest.mark.parametrize("op", ["silu_mul", "add_norm", "gemma_add_norm_sum"])
def test_probe_fresh_tensors_match_shape_and_aux_stride(monkeypatch, op):
    for name in ("randn", "empty"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *a, _fn=original, **kw: _fn(*a, **(kw | {"device": "cpu"}))
        )
    inputs, out = probe.fresh_tensors(spec_for(op=op))
    if op == "silu_mul":
        assert inputs.shape == (16, 8192) and out.shape == (16, 4096)
    else:
        assert inputs[0].shape == inputs[1].shape == (16, 4096)
        assert inputs[2].shape == (4096,) and out is None
        if op.endswith("_sum"):
            assert inputs[3].shape == inputs[0].shape
            assert inputs[3].stride() == (12288, 1)
            assert inputs[3].storage_offset() == 4096


def test_sum_probe_warms_both_parities_before_profiling(monkeypatch):
    events = []
    inputs = (object(), object(), object(), object())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(probe, "fresh_tensors", lambda spec: (inputs, None))
    monkeypatch.setattr(
        probe, "build_call", lambda spec, inputs: lambda x, out: events.append(("warm", len(x)))
    )
    monkeypatch.setattr(
        probe, "profile_kernels", lambda fn, x, out: events.append(("profile", len(x))) or KERNELS
    )
    assert probe.probe(spec_for()) == {"kernels": KERNELS, "kernels_even": KERNELS}
    assert events == [("warm", 4), ("warm", 3), "sync", ("profile", 4), ("profile", 3)]


def test_only_probe_module_activates_torch_profiler():
    import ast

    for path in Path("benchmark/kernel_tuning").glob("norm_act_*.py"):
        if path.name == "norm_act_probe.py":
            continue
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
        ]
        assert "profile_kernels" not in calls
        assert "torch.profiler.profile" not in calls
        assert "torch.autograd.profiler.profile" not in calls


def test_compiled_probe_pins_parent_launcher_and_rejects_missing_config(monkeypatch):
    from benchmark.kernel_tuning.norm_act_vllm import compiled_configs

    def launcher(warps):
        return SimpleNamespace(
            config=SimpleNamespace(kwargs={"XBLOCK": 1}, num_warps=warps, num_stages=1)
        )

    class Autotuner:
        fn = SimpleNamespace(__name__="triton_test")

        def __init__(self):
            self.launchers = [launcher(4), launcher(8)]
            self.inductor_meta = {}
            self._cached_launcher = object()

        def run(self):
            assert self._cached_launcher is None
            assert len(self.launchers) == 1
            assert self.launchers[0].config.num_warps == 8
            return "launched"

    monkeypatch.setitem(
        sys.modules,
        "torch._inductor.runtime.triton_heuristics",
        SimpleNamespace(CachingAutotuner=Autotuner),
    )
    expected = {"triton_test": [{"XBLOCK": 1, "num_warps": 8, "num_stages": 1}]}
    original = Autotuner.run
    with compiled_configs(expected) as observed:
        assert Autotuner().run() == "launched"
    assert observed == expected and Autotuner.run is original
    with pytest.raises(MarketError, match="cannot reproduce"):
        with compiled_configs({"triton_test": [{"XBLOCK": 1, "num_warps": 16, "num_stages": 1}]}):
            Autotuner().run()
    assert Autotuner.run is original


def test_provider_spec_preserves_sum_route_static_weight_and_config():
    from benchmark.kernel_tuning.norm_act_norm import NormCall
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall

    fn = SumNormCall(NormCall(True, "cute", Config(), static_weight=False), fused=True)
    spec = json.loads(
        probe.encode_spec(
            probe.provider_spec(fn, op="gemma_add_norm_sum", m=4, width=4096, dtype="bf16")
        )
    )
    rebuilt = probe.build_call(spec, None)
    assert rebuilt.fused and rebuilt.norm.zero_centered
    assert rebuilt.norm.op.config == Config()
    assert rebuilt.norm.op._override == "cute"
    assert rebuilt.norm.op.static_weight is False


def test_shipped_probe_rebuilds_parent_config_and_aux_layout(monkeypatch):
    from benchmark.kernel_tuning import norm_act_vllm as vllm

    configs = [{"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 8, "num_stages": 1}]
    aux_configs = [{"XBLOCK": 1, "R0_BLOCK": 2048, "num_warps": 16, "num_stages": 1}]
    loads, launches = [], []

    def load(path, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(run=lambda *a, **kw: launches.append((kwargs, a))), None

    monkeypatch.setattr(vllm, "load_shipped", load)
    monkeypatch.setattr(vllm, "site_file", lambda *a: Path("synthetic.py"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1))
    inputs = (
        torch.ones(2, 4096),
        torch.ones(2, 4096),
        torch.ones(4096),
        torch.empty(2, 12288)[:, 4096:8192],
    )
    spec = spec_for("vllm") | {
        "site": "d2",
        "mode": "shipped",
        "aux": True,
        "configs": configs,
        "aux_configs": aux_configs,
    }
    fn = probe.build_call(spec, inputs)
    fn(inputs, None)
    assert loads == [{"configs": configs}, {"aux_stride": 12288, "configs": aux_configs}]
    assert launches[0][0] == loads[1]
    assert launches[0][1][6] is inputs[3]
