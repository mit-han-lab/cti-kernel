# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks for fixture qualification, FP32 market timing, and the DRAM bar."""

import sys
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.norm_act_paired import paired_summary, sol_summary
from benchmark.kernel_tuning.norm_act_vllm import (
    MarketError,
    chosen_configs,
    geometry_receipt,
    load_shipped,
    optional_sites,
    site_file,
)


@pytest.mark.parametrize(
    "width,zero,site,directory,filename",
    [
        (4096, False, "a", "qwen3", "triton_red_fused_fused_add_rms_norm_0.hash.py"),
        (4096, True, "a", "qwen35", "triton_red_fused_to_copy_add_fused_add_rms_norm_0.hash.py"),
        (4096, False, "b", "qwen3", "kernel.ctzv25hash.py"),
        (4096, True, "b", "qwen35", "kernel.c7zxzihash.py"),
        (4096, False, "d2", "qwen3", "kernel.ciqqjvb7hash.py"),
        (4096, True, "d2", "qwen35", "kernel.cxy7p5q7hash.py"),
        (12288, False, "silu", "qwen3", "triton_poi_fused_mul_silu_slice_1.hash.py"),
        (
            2048,
            False,
            "a",
            "qwen3moe-h2048",
            "triton_red_fused_fused_add_rms_norm_moe_forward_0.hash.py",
        ),
        (2048, False, "b", "qwen3moe-h2048", "kernel.cw4iw2rhhash.py"),
    ],
)
def test_fixture_selection(tmp_path, width, zero, site, directory, filename):
    path = tmp_path / "vllm-kernels" / directory / filename
    path.parent.mkdir(parents=True)
    path.touch()
    assert site_file(tmp_path, site, zero, width) == path


@pytest.mark.parametrize(
    "width,zero,site",
    [
        (1024, False, "a"),
        (2048, True, "a"),
        (2048, True, "b"),
        (2048, False, "d2"),
        (2048, False, "silu"),
        (4096, False, "unknown"),
        (4096, False, "silu"),
        (12288, False, "b"),
    ],
)
def test_unsupported_capture_is_explicit_omission(tmp_path, width, zero, site):
    def build():
        return site_file(tmp_path, site, zero, width)

    with pytest.raises(MarketError, match="no vLLM 0.31.0 capture for"):
        build()
    sites, receipt = optional_sites(build)
    assert sites is None and receipt["omitted"]
    assert f"width={width}" in receipt["reason"] and f"site={site}" in receipt["reason"]


def test_proof_failure_is_never_an_omission():
    def build():
        raise MarketError("compiled vLLM kernel was not proven equal")

    with pytest.raises(MarketError, match="not proven equal"):
        optional_sites(build)


@pytest.mark.parametrize("best", [False, True])
def test_shipped_loader_only_pins_existing_best_config(monkeypatch, tmp_path, best):
    from torch._inductor.runtime import hints

    path = tmp_path / "kernel.py"
    path.write_text("""import triton
@triton.jit(inductor_meta={'dynamic_scale_rblock': True})
def triton_synthetic():
    pass
""")
    if best:
        path.with_suffix(".best_config").write_text('{"XBLOCK":1,"num_warps":16,"num_stages":1}')
    kernel = SimpleNamespace(configs=["decorator config"])
    observed = {}

    def jit(**kwargs):
        observed.update(kwargs["inductor_meta"])
        return lambda fn: kernel

    monkeypatch.setitem(
        sys.modules, "triton", SimpleNamespace(jit=jit, Config=lambda *a, **k: (a, k))
    )
    monkeypatch.setattr(hints, "DeviceProperties", SimpleNamespace(create=lambda device: None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "RTX 5090")
    loaded, pinned = load_shipped(path)
    assert loaded is kernel
    assert observed["force_disable_caches"]
    assert not observed["autotune_local_cache"] and not observed["autotune_remote_cache"]
    assert observed["dynamic_scale_rblock"] is (not best)
    if best:
        assert pinned["XBLOCK"] == 1 and kernel.configs != ["decorator config"]
    else:
        assert pinned is None and kernel.configs == ["decorator config"]


@pytest.mark.parametrize("device", ["RTX 5090", "RTX PRO 6000", "GB10"])
def test_autotuned_geometry_checks_selected_launcher_on_every_device(device):
    launcher = SimpleNamespace(
        config=SimpleNamespace(kwargs={"XBLOCK": 4}, num_warps=8, num_stages=1)
    )
    configs = chosen_configs(SimpleNamespace(launchers=[launcher]))
    kernels = [{"grid": [5, 1, 1], "block": [256, 1, 1]}]
    receipt = geometry_receipt("a", 17, kernels, device, width=2048, configs=configs)
    assert receipt["basis"] == "autotuned-launcher"
    assert receipt["checked"] and receipt["valid"] and receipt["matches"]
    for wrong in (
        [{"grid": [17, 1, 1], "block": [256, 1, 1]}],
        [{"grid": [5, 1, 1], "block": [512, 1, 1]}],
        [],
        kernels * 2,
    ):
        assert not geometry_receipt("a", 17, wrong, device, width=2048, configs=configs)["valid"]
    assert not geometry_receipt("a", 17, kernels, device, width=2048)["valid"]


@pytest.mark.parametrize(
    "provider,bar", [("cute", 0.95), ("cute_t64_r1_p1_s0_te_c3", 0.95), ("triton", 0.90)]
)
def test_sol_fp32_bytes_and_traffic_threshold(provider, bar):
    moved = 12 * 256 * 2048 + 2 * 2048
    bandwidth = moved / 4000
    row = sol_summary(
        256,
        2048,
        {"ours": 4.2, "static": 5, "triton": 6, "cute_tune": 8, "sgl_fp32": 4, "cute_remote": 3},
        bandwidth,
        provider,
    )
    assert row["dram_bytes"] == moved and row["t_dram_us"] == 4
    assert row["traffic_bound"] and row["sol_bar"] == bar and row["sol_pass"]
    assert row["sol"] == {"ours": 4 / 4.2, "static": 0.8, "triton": 4 / 6, "cute_tune": 0.5}
    slow = sol_summary(256, 2048, {"ours": 5}, bandwidth, provider)
    assert not slow["sol_pass"]
    latency = sol_summary(256, 2048, {"ours": 5}, bandwidth * 1.001, provider)
    assert not latency["traffic_bound"] and "sol_bar" not in latency and "sol_pass" not in latency


def test_fp32_market_uses_faster_comparator_within_each_round():
    result = paired_summary(
        {
            "ours": [2.0] * 30,
            "aa": [2.0] * 30,
            "sglang": [10.0] * 30,
            "sgl_fp32": [1.0, 8.0] * 15,
            "sgl_fp32c": [8.0, 1.0] * 15,
        }
    )
    assert result["sgl_fp32_ratio"] == result["sgl_fp32c_ratio"] == 2.25
    assert result["fp32_market_ratio"] == 0.5
    assert result["fp32_market_ci95"] == [0.5, 0.5]
    assert result["fp32_market_ci95_lower"] == 0.5
    assert result["markets"] == ["sglang"]


def test_compiled_fp32_call_uses_default_options_and_probe(monkeypatch):
    from benchmark.kernel_tuning.norm_act_norm import SglangFP32Call
    from benchmark.kernel_tuning.norm_act_probe import build_call, encode_spec, provider_spec
    from benchmark.kernel_tuning.norm_act_reference import fp32_residual_formulas

    calls = []

    def compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", compile)
    fn = SglangFP32Call(False, compiled=True)
    tensors = (
        torch.ones(2, 16, dtype=torch.bfloat16),
        torch.ones(2, 16),
        torch.ones(16, dtype=torch.bfloat16),
    )
    spec = provider_spec(fn, op="add_norm", m=2, width=16, dtype="bf16", residual_dtype="fp32")
    assert spec["kind"] == "sgl_fp32c" and encode_spec(spec)
    probed = build_call(spec, tensors)
    assert all(torch.equal(a, b) for a, b in zip(fn(tensors), probed(tensors), strict=True))
    assert calls == [{"fullgraph": True, "dynamic": False}] * 2
    assert set(fp32_residual_formulas(*tensors)) == {"torch_fp32", "sgl_fp32"}


@pytest.mark.parametrize("valid", [False, True])
def test_2048_parent_and_probe_share_two_output_abi(monkeypatch, tmp_path, valid):
    from benchmark.kernel_tuning import norm_act_probe as probe
    from benchmark.kernel_tuning import norm_act_vllm as vllm

    directory = tmp_path / "vllm-kernels/qwen3moe-h2048"
    directory.mkdir(parents=True)
    path = directory / "triton_red_fused_fused_add_rms_norm_moe_forward_0.hash.py"
    path.write_text("# fixture")
    (directory / "PROVENANCE.txt").write_text("Qwen3-30B-A3B capture")
    (directory / "layer_graph.wrapper.txt").write_text("wrapper")
    (tmp_path / "vllm-ir").mkdir()
    (tmp_path / "vllm-ir/layernorm.py").write_text("# IR")
    configs = [{"XBLOCK": 4, "num_warps": 8, "num_stages": 1}]
    launches = []
    kernel = SimpleNamespace(run=lambda *a, **kw: launches.append(a))
    monkeypatch.setattr(vllm, "load_shipped", lambda *a, **kw: (kernel, None))
    monkeypatch.setattr(vllm, "chosen_configs", lambda kernel: configs)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "GB10")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1))
    tensors = (
        torch.ones(17, 2048, dtype=torch.bfloat16),
        torch.ones(17, 2048, dtype=torch.bfloat16),
        torch.ones(2048, dtype=torch.bfloat16),
    )

    def run_probe(spec):
        assert spec["configs"] == configs and spec["width"] == 2048
        probe.build_call(spec, tensors)(tensors, None)
        return {"kernels": [{"grid": [5 if valid else 17, 1, 1], "block": [256, 1, 1]}]}

    monkeypatch.setattr(probe, "run_probe", run_probe)
    args = SimpleNamespace(market_root=tmp_path, vllm_mode="shipped", allow_vllm_mismatch=True)
    if not valid:
        with pytest.raises(MarketError, match="geometry proof failed"):
            vllm.VllmSite(args, "a", tensors)
    else:
        site = vllm.VllmSite(args, "a", tensors)
        meta = site.metadata
        assert meta["best_config"] is None and meta["autotuned"]
        assert meta["model"] == "Qwen3-30B-A3B" and meta["fixture"] == directory.name
        import hashlib

        assert (
            meta["provenance_sha256"]
            == hashlib.sha256((directory / "PROVENANCE.txt").read_bytes()).hexdigest()
        )
        assert meta["geometry"]["checked"] and meta["geometry"]["valid"]
        result = site(*tensors)
        assert result is launches[-1][3]
    assert len(launches) >= 2
    for launch in launches:
        assert len(launch) == 7 and launch[-2:] == (17, 2048)
        assert launch[3].shape == launch[4].shape == tensors[0].shape
        assert launch[3].data_ptr() != launch[4].data_ptr()


def test_bandwidth_counts_traffic_uses_events_and_runs_once(monkeypatch):
    from benchmark.kernel_tuning import norm_act_paired as bench

    counts = dict.fromkeys(("copy", "add", "sum"), 0)
    allocated = []

    class Buffer:
        def __init__(self, elements):
            self.elements = elements
            allocated.append(elements * 4)

        def numel(self):
            return self.elements

        def element_size(self):
            return 4

        def copy_(self, other):
            counts["copy"] += 1

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, end):
            return 0.25 if counts["sum"] else 2.0

    def add(a, b, *, out):
        counts["add"] += 1

    def sum_(a):
        counts["sum"] += 1

    monkeypatch.setattr(torch, "ones", lambda n, **kw: Buffer(n))
    monkeypatch.setattr(torch, "ones_like", lambda t: Buffer(t.numel()))
    monkeypatch.setattr(torch, "empty_like", lambda t: Buffer(t.numel()))
    monkeypatch.setattr(torch, "add", add)
    monkeypatch.setattr(torch, "sum", sum_)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(L2_cache_size=256 * 1024**2)
    )
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    bench.measure_bandwidth.cache_clear()
    try:
        receipt = bench.measure_bandwidth()
        assert bench.measure_bandwidth() is receipt
        assert counts == {"copy": 25, "add": 25, "sum": 25}
        assert allocated == [1024**3] * 3
        assert receipt["bw_gbs"] == 3 * 1024**3 / (2000 * 1000)
        assert receipt["bw_read_gbs"] == 1024**3 / (250 * 1000)
        assert receipt["bw_read_gbs"] > receipt["bw_gbs"]
        assert {name: value["bytes"] for name, value in receipt["measurements"].items()} == {
            "copy": 2 * 1024**3,
            "add": 3 * 1024**3,
            "sum": 1024**3,
        }
    finally:
        bench.measure_bandwidth.cache_clear()


def test_fp32_main_reports_bandwidth_and_sol_on_every_row(monkeypatch, tmp_path, capsys):
    import json

    from benchmark.kernel_tuning import norm_act_paired as bench

    calls = []
    bandwidth = {
        "bw_gbs": 1000,
        "bw_read_gbs": 1200,
        "measurements": {"copy": {}, "add": {}, "sum": {}},
    }
    monkeypatch.setattr(bench, "pin_x925", lambda: None)
    monkeypatch.setattr(bench, "measure_bandwidth", lambda: calls.append("bandwidth") or bandwidth)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def row(args, m, regime):
        assert calls[0] == "bandwidth"
        calls.append("row")
        return {
            "M": m,
            "regime": regime,
            "dispatch": "cute",
            "verdict": "PASS",
            "us": {"ours": 4, "rw_copy": 3},
        }

    monkeypatch.setattr(bench, "run_row", row)
    assert (
        bench.main(
            [
                "--op",
                "add_norm",
                "--market-root",
                str(tmp_path),
                "--residual-dtype",
                "fp32",
                "--rows",
                "256",
                "512",
                "--regimes",
                "hot",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert calls == ["bandwidth", "row", "row"]
    assert json.loads(lines[0].removeprefix("HEADER "))["bw_gbs"] == 1000
    assert json.loads(lines[0].removeprefix("HEADER "))["bw_read_gbs"] == 1200
    rows = [json.loads(line.removeprefix("ROW ")) for line in lines if line.startswith("ROW ")]
    assert len(rows) == 2
    for receipt in rows:
        assert receipt["bandwidth"] == bandwidth and receipt["bw_gbs"] == 1000
        assert receipt["bw_read_gbs"] == 1200
        assert receipt["traffic_bound"] and receipt["sol_bar"] == 0.95
        assert receipt["sol"]["ours"] == receipt["t_dram_us"] / 4
        assert receipt["us"]["rw_copy"] == 3
        assert receipt["sol_witness"] == {"ours": 0.75}
    assert any("rw_copy_us" in line for line in lines)
    assert any("sgl_fp32c_ratio CI95 fp32_market_ratio CI95" in line for line in lines)


def test_compiled_probe_rejects_unequal_source(monkeypatch, tmp_path):
    import contextlib

    import torch._inductor.utils

    from benchmark.kernel_tuning import norm_act_probe as probe
    from benchmark.kernel_tuning import norm_act_vllm as vllm

    monkeypatch.setattr(vllm, "site_file", lambda *a: tmp_path / "fixture.py")
    monkeypatch.setattr(vllm, "fragments", lambda *a: {"a": lambda x, r, w: x})
    monkeypatch.setattr(vllm, "compiled_configs", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(vllm, "equality_receipt", lambda *a: {"equal": False})
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *a: None)
    monkeypatch.setattr(
        torch._inductor.utils, "run_and_get_code", lambda fn, *a: (fn(*a), ["different"])
    )
    tensors = (torch.ones(2, 2048), torch.ones(2, 2048), torch.ones(2048))
    call = probe.build_call(
        {
            "kind": "vllm",
            "op": "add_norm",
            "site": "a",
            "width": 2048,
            "market_root": tmp_path,
            "mode": "compiled",
            "compiled_configs": {},
        },
        tensors,
    )
    with pytest.raises(MarketError, match="not proven equal"):
        call(tensors, None)


@pytest.mark.parametrize("sweep", [False, True])
def test_explicit_cluster_candidates_preserve_static_weight(monkeypatch, sweep, tmp_path):
    from benchmark.kernel_tuning import norm_act_norm as bench
    from benchmark.kernel_tuning.norm_act_paired import parse_args
    from ctikernels.norm._dispatch import Config

    args = parse_args(
        [
            "--op",
            "add_norm",
            "--residual-dtype",
            "fp32",
            "--candidates",
            "ours",
            "--market-root",
            str(tmp_path),
            "--configs",
            "64,1,1,0,1,3;128,1,1,1,0,8",
            "--sweep-filter",
            "_c",
            *(["--sweep"] if sweep else []),
        ]
    )
    monkeypatch.setattr("benchmark.kernel_tuning.norm_act_pools.warm_norm_call", lambda *a: None)
    monkeypatch.setattr(bench, "SglangFP32Call", lambda *a, **k: SimpleNamespace(provider="native"))
    candidates, _ = bench.norm_candidates(args, torch.bfloat16, (None, None, None))
    configs = [Config(64, 1, True, False, "early", 3), Config(128, 1, True, True, "end", 8)]
    assert bench.candidate_configs(args) == configs
    for config in configs:
        op = candidates[config.name].op
        assert op.config == config and op.static_weight == config.static_weight
    ours = candidates["ours"]
    ours.op.provider, ours.op.dispatched_config = "cute", configs[0]
    assert ours.provider == "cute" and ours.dispatch == configs[0].name
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall

    assert SumNormCall(ours).dispatch == configs[0].name
