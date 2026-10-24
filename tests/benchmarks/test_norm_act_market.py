# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import ast
import io
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.norm_act_market import (
    PROTOCOL_VERSION,
    MarketWorker,
    RemoteCandidate,
    decode_message,
    encode_message,
    sample_us,
)
from benchmark.kernel_tuning.norm_act_norm import VllmNormCall, merge_market_metrics
from benchmark.kernel_tuning.norm_act_paired import measure, paired_summary
from benchmark.kernel_tuning.norm_act_pools import build_norm_pools, pool_count, pool_spec
from benchmark.kernel_tuning.norm_act_vllm import (
    MarketError,
    RuntimeMetadata,
    equality_receipt,
    extract_kernels,
    fragments,
    geometry_receipt,
    normalize_kernel,
    site_file,
)
from ctikernels.norm._dispatch import Config, entry_for


# Small invented programs test syntax handling; no vendor source is embedded.
def kernel_source(name="test", pointer=7, temporary=11, delta="1.0"):
    return f"""@triton_heuristics.pointwise(
    size_hints={{'x': 128}}, filename=__file__,
    triton_meta={{'device': DeviceProperties(index=99)}},
    inductor_meta={{'kernel_name': 'host_metadata'}})
@triton.jit
def triton_{name}(in_ptr{pointer}, out_ptr8, XBLOCK: tl.constexpr):
    tmp{temporary} = tl.load(in_ptr{pointer})
    tmp12 = tmp{temporary} + {delta}
    tl.store(out_ptr8, tmp12)
"""


def test_normalization_ignores_metadata_names_but_preserves_math_and_stores(tmp_path):
    source = kernel_source()
    changed = kernel_source("other", 19, 23).replace("index=99", "index=0")
    assert normalize_kernel(source) == normalize_kernel(changed)
    assert normalize_kernel(source) != normalize_kernel(kernel_source(delta="2.0"))
    assert normalize_kernel(source) != normalize_kernel(source.replace("tl.store", "tl.atomic_add"))
    assert normalize_kernel(source) != normalize_kernel(
        source.replace("tmp12)", "tmp12.to(tl.bfloat16))")
    )
    path = tmp_path / "generated.py"
    path.write_text(source)
    assert equality_receipt("a", [changed], path)["equal"]
    proof = equality_receipt("a", [kernel_source(delta="2.0")], path)
    assert not proof["equal"] and 0 < len(proof["diff"]) <= 40
    assert not equality_receipt("a", [changed, changed], path)["equal"]


def test_extracts_quoted_inductor_sources_and_keeps_tiling():
    source = kernel_source()
    wrapper = "kernel = compile_source(" + repr(source) + ")"
    assert extract_kernels(wrapper) == extract_kernels(source)
    assert normalize_kernel(wrapper) == normalize_kernel(source)
    assert normalize_kernel(source) != normalize_kernel(source.replace("'x': 128", "'x': 256"))
    with pytest.raises(ValueError):
        normalize_kernel("pass")


def test_device_patch_preserves_jit_body():
    source = kernel_source()
    transformed = ast.unparse(ast.fix_missing_locations(RuntimeMetadata().visit(ast.parse(source))))
    assert "_current_device" in transformed
    assert "DeviceProperties(" not in transformed
    # Only the host decorator changes, with no rewritten math or source file.
    original_body = ast.parse(source).body[0].body
    patched_body = ast.parse(transformed).body[0].body
    assert [ast.dump(node) for node in original_body] == [ast.dump(node) for node in patched_body]


def test_protocol_roundtrip_and_nonfinite_error_metrics():
    message = {
        "version": PROTOCOL_VERSION,
        "id": 3,
        "command": "setup",
        "label": "line\nbreak",
        "metrics": [float("inf"), -float("inf"), float("nan")],
    }
    encoded = encode_message(message)
    assert encoded.count("\n") == 1
    parsed = json.loads(encoded)
    assert parsed["label"] == "line\nbreak"
    decoded = decode_message(encoded)
    assert decoded["metrics"][:2] == [float("inf"), -float("inf")]
    assert math.isnan(decoded["metrics"][2])
    with pytest.raises(ValueError):
        decode_message("[]")
    with pytest.raises(ValueError):
        decode_message('{"metric":{"$float":"unexpected"}}')


def test_event_receipt_math_uses_all_calls():
    assert sample_us([0.12, 0.24], 6) == pytest.approx(30)
    for times, layers in (([], 1), ([1], 0), ([-1], 2), ([float("nan")], 2)):
        with pytest.raises(ValueError):
            sample_us(times, layers)
    samples = {
        "ours": [2] * 30,
        "aa": [2] * 30,
        "vllm": [3] * 30,
        "vllm_a": [1] * 30,
        "vllm_b": [5] * 30,
        "sglang": [4] * 30,
    }
    receipt = paired_summary(samples)
    assert receipt["faster_ratio"] == 1.5  # A alone must never decide the verdict.


def test_cross_process_receipt_uses_paired_remote_over_local_ratios():
    # Median(remote) / median(local) is 2; median of round ratios is 3.
    samples = {name: [7.0] * 30 for name in ("ours", "aa", "vllm", "sglang")}
    samples.update(cute=[1.0, 10.0, 100.0] * 10, cute_remote=[3.0, 20.0, 400.0] * 10)
    control = paired_summary(samples)["cross_process_control"]
    assert control == {
        "local": "cute",
        "remote": "cute_remote",
        "ratio_median": 3.0,
        "ci95": [2.0, 4.0],
    }
    samples["cute_remote"].pop()
    with pytest.raises(ValueError):
        paired_summary(samples)


@pytest.mark.parametrize("static_weight", [False, True])
@pytest.mark.parametrize("op", ["add_norm", "gemma_add_norm", "add_norm_sum", "gemma_add_norm_sum"])
@pytest.mark.parametrize("m", [16, 256, 512, 1024])
def test_control_config_matches_auto_table_or_default(op, m, static_weight):
    from benchmark.kernel_tuning.norm_act_norm import cross_process_config

    args = SimpleNamespace(op=op, width=4096, static_weight=static_weight)
    props = SimpleNamespace(major=12, minor=0, multi_processor_count=170)
    entry = entry_for("rtx5090", 4096, "bfloat16", m, op.startswith("gemma"), static_weight)
    config = cross_process_config(args, m, torch.bfloat16, props)
    assert config == (entry.config if entry.provider == "cute" else Config())
    args.width = 16  # No table row: pin the default explicitly.
    assert cross_process_config(args, m, torch.bfloat16, props) == Config()


@pytest.mark.parametrize("static_weight", [False, True])
def test_setup_sends_all_explicit_config_fields_as_json(monkeypatch, static_weight):
    from benchmark.kernel_tuning import norm_act_market as market

    config = Config(256, 2, True, static_weight, "early")
    args = SimpleNamespace(fi071=Path("unused-overlay"), market_root=Path("unused-market"))
    spec = {"op": "gemma_add_norm_sum", "static_weight": static_weight}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    worker = MarketWorker.__new__(MarketWorker)
    worker.sequence = 0
    worker.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            encode_message(
                {
                    "version": PROTOCOL_VERSION,
                    "id": 1,
                    "ok": True,
                    "result": {"shared_buffers": True},
                }
            )
        ),
    )
    monkeypatch.setattr(market, "MarketWorker", lambda root, target: worker)
    pools = SimpleNamespace(spec=spec)
    monkeypatch.setattr(market, "encode_pool_handles", lambda actual: {"mock": actual is pools})
    assert market.start_market(args, pools, cute_config=config) == (
        worker,
        {"shared_buffers": True},
    )
    assert worker.shared_pools is pools
    message = json.loads(worker.process.stdin.getvalue())
    assert message["command"] == "setup" and message["spec"] == spec
    assert message["pool_handles"] == {"mock": True}
    assert message["cute_config"] == {
        "threads": 256,
        "rows": 2,
        "pdl": True,
        "static_weight": static_weight,
        "trigger": "early",
        "split": 1,
    }
    assert Config(**message["cute_config"]) == config


@pytest.mark.parametrize("regime", ["hot", "engine", "cold"])
@pytest.mark.parametrize("summed", [False, True])
def test_identical_worker_pools_restore_including_hot_h1(regime, summed):
    args = SimpleNamespace(
        op="gemma_add_norm" + ("_sum" if summed else ""),
        width=16,
        dtype="bf16",
        layers=4,
        static_weight=0,
    )
    spec = pool_spec(args, 3, regime, 1024)
    parent = build_norm_pools(spec, device="cpu")
    torch.manual_seed(817)
    child = build_norm_pools(json.loads(json.dumps(spec)), device="cpu")
    assert parent.count == pool_count(spec) == len(parent.inputs)
    assert parent.count % args.layers == 0
    for name in ("x", "residual", "weight", "h1"):
        assert torch.equal(getattr(parent, name), getattr(child, name))
    if regime != "hot":
        rotating = [parent.residual, parent.weight] + ([parent.x] if regime == "cold" else [])
        assert sum(t.numel() * t.element_size() for t in rotating) > 2 * spec["l2_bytes"]
    parent.x.zero_()
    parent.residual.zero_()
    parent.h1.zero_()
    parent.restore()
    for name in ("x", "residual", "h1"):
        assert torch.equal(getattr(parent, name), getattr(child, name))
    assert parent.previous_hidden(1).data_ptr() == parent.h1[0 if regime == "hot" else 1].data_ptr()
    if summed:
        assert parent.tensors[0][3].stride() == (48, 1)


def test_alternation_is_indexed_not_affected_by_warmup():
    calls = []
    sites = {site: lambda *args, site=site: calls.append((site, args)) for site in ("a", "b")}
    pool = SimpleNamespace(previous_hidden=lambda index: f"h1:{index}")
    candidate = VllmNormCall(sites, pool)
    for index in (0, 1, 0, 1, 2, 3):
        candidate.call_index(("h2", "r", "w"), None, index)
    assert [site for site, _ in calls] == ["a", "b", "a", "b", "a", "b"]
    assert calls[1][1] == ("h2", "h1:1", "r", "w")


def test_lockstep_remote_samples_follow_same_seeded_round_order(monkeypatch):
    log = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: log.append("sync"))

    class Graph:
        def replay(self):
            log.append("local")

    class Event:
        def synchronize(self):
            log.append("event_sync")

        def elapsed_time(self, end):
            return 0.008

    class Worker:
        def request(self, command, **payload):
            assert log[-1] == "sync"
            log.append(payload["candidate"])
            return {"us": 4.0}

    banks = {
        "ours": ([(Graph(), Event(), Event())], []),
        "sglang": RemoteCandidate(Worker(), "sglang"),
        "fi071_direct": RemoteCandidate(Worker(), "fi071_direct"),
        "cute_remote": RemoteCandidate(Worker(), "cute_remote"),
    }
    result = measure(banks, 6, 2, before=lambda name: log.append("restore:" + name))
    assert result == {name: [4.0] * 6 for name in banks}
    measured_order = [
        item for item in log if item in ("local", "sglang", "fi071_direct", "cute_remote")
    ][len(banks) :]
    expected = []
    for index in range(6):
        names = list(banks)
        random.Random(20261015 + index).shuffle(names)
        expected.extend("local" if name == "ours" else name for name in names)
    assert measured_order == expected
    assert log.count("restore:ours") == 7


def test_parent_rejects_worker_sequence_mismatch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    worker = MarketWorker.__new__(MarketWorker)
    worker.sequence = 0
    worker.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            encode_message({"version": PROTOCOL_VERSION, "id": 9, "ok": True, "result": {}})
        ),
    )
    with pytest.raises(MarketError, match="sequence/version"):
        worker.request("free")


def test_remote_invalid_samples_fail():
    worker = SimpleNamespace(request=lambda *a, **kw: {"us": float("nan")})
    with pytest.raises((ValueError, MarketError)):
        RemoteCandidate(worker, "sglang").sample()


def test_geometry_trace_is_device_specific():
    kernels = [{"name": "kernel", "grid": [16, 1, 1], "block": [512, 1, 1]}]
    assert geometry_receipt("a", 16, kernels, "NVIDIA GeForce RTX 5090")["valid"]
    assert not geometry_receipt("b", 64, kernels, "NVIDIA GeForce RTX 5090")["valid"]
    assert geometry_receipt("b", 64, kernels, "GB10")["warning"]
    silu = [{"name": "kernel", "grid": [192, 1, 1], "block": [128, 1, 1]}]
    assert geometry_receipt("silu", 16, silu, "RTX 5090")["matches"]


def test_market_best_is_per_metric_and_per_input():
    a = {name: {"max_ulp": 1, "rms": 2, "rms_ulp": 3} for name in ("normalized", "residual")}
    b = {name: {"max_ulp": 3, "rms": 1, "rms_ulp": 2} for name in a}
    assert merge_market_metrics(a, b)["normalized"] == {"max_ulp": 1, "rms": 1, "rms_ulp": 2}


def test_shipped_ir_loader_and_snapshot_selection_when_available():
    root = Path("/tmp/na1/market")
    if not root.exists():
        pytest.skip("external orchestrator market snapshot not installed")
    h, h1, r, w = (
        torch.randn(2, 16).bfloat16(),
        torch.randn(2, 16).bfloat16(),
        torch.randn(2, 16).bfloat16(),
        torch.randn(16).bfloat16(),
    )
    from benchmark.kernel_tuning.norm_act_reference import vllm_norm_native

    for zero in (False, True):
        funcs = fragments(root, zero)
        expected = vllm_norm_native(h, r, w, zero_centered=zero)
        assert torch.equal(funcs["a"](h, r, w), expected[0])
        s1 = (h1.float() + r.float()).to(h.dtype)
        expected_b = vllm_norm_native(h, s1, w, zero_centered=zero)
        assert all(
            torch.equal(a, b) for a, b in zip(funcs["b"](h, h1, r, w), expected_b, strict=True)
        )
        expected_aux = (*expected_b, h + s1)
        assert all(
            torch.equal(a, b) for a, b in zip(funcs["d2"](h, h1, r, w), expected_aux, strict=True)
        )
        for site in ("a", "b", "d2", "silu"):
            path = site_file(root, site, zero, 12288 if site == "silu" else 4096)
            assert equality_receipt(site, [path.read_text()], path)["equal"]


@pytest.mark.parametrize("op", ["add_norm", "gemma_add_norm", "add_norm_sum", "gemma_add_norm_sum"])
@pytest.mark.parametrize("static_weight", [0, 1])
@pytest.mark.parametrize("residual_dtype", ["same", "fp32"])
@pytest.mark.parametrize("omit_vllm", [False, True])
@pytest.mark.parametrize("regime", ["hot", "engine", "cold"])
def test_full_norm_row_protocol_capture_and_receipt_on_cpu(
    monkeypatch, op, static_weight, residual_dtype, omit_vllm, regime
):
    from benchmark.kernel_tuning import norm_act_market as market
    from benchmark.kernel_tuning import norm_act_norm as norm
    from benchmark.kernel_tuning import norm_act_paired as paired
    from benchmark.kernel_tuning import norm_act_pools as pool_module
    from benchmark.kernel_tuning import norm_act_vllm as vllm
    from benchmark.kernel_tuning.norm_act_reference import norm_reference

    zero = op.startswith("gemma")
    args = SimpleNamespace(
        op=op,
        width=16,
        dtype="bf16",
        residual_dtype=residual_dtype,
        layers=2,
        rounds=6,
        check=True,
        static_weight=static_weight,
        fi07=None,
        sweep=False,
        market_root=Path("unused"),
        candidates="ours,flashinfer",
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", L2_cache_size=512, major=12, minor=0, multi_processor_count=170
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    builder = pool_module.build_norm_pools
    monkeypatch.setattr(
        pool_module, "build_norm_pools", lambda spec, **kw: builder(spec, device="cpu", **kw)
    )
    monkeypatch.setattr(
        market,
        "run_probe",
        lambda spec: {"kernels": [{"name": "test", "grid": [1, 1, 1], "block": [32, 1, 1]}]},
    )

    constructed = []

    class Local:
        provider = "test"
        zero_centered = zero

        def __init__(self, zero=zero, provider="auto", config=None, *, static_weight=True):
            self.zero_centered = zero
            constructed.append((provider, config, static_weight))

        def __call__(self, tensors, unused=None):
            out, residual = norm_reference(*tensors, zero_centered=zero)
            tensors[0].copy_(out)
            tensors[1].copy_(residual)
            return tensors[:2]

        def op(self, x, r, w, eps, *, zero_centered, sum_out):
            torch.add(x, r, out=sum_out)
            return self((x, r, w))

    monkeypatch.setattr(norm, "NormCall", Local)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(norm, "flashinfer_metadata", lambda: {})
    # Exercise every field across the parent/worker boundary, independently of
    # the table-selection tests above (which also cover default-config routes).
    control_config = Config(256, 2, True, bool(static_weight), "early")
    monkeypatch.setattr(norm, "cross_process_config", lambda *args: control_config)

    class Site:
        compiled = None

        def __init__(self, args, site, example, zero):
            if omit_vllm:
                raise vllm.MarketError("no vLLM 0.31.0 capture for this shape")
            self.site = site
            self.metadata = {"mode": "shipped", "site": site}

        def __call__(self, *tensors, output=None, residual_output=None, aux_output=None):
            if self.site in ("b", "d2"):
                h, h1, residual, w = tensors
                tensors = (h, (h1.float() + residual.float()).to(h.dtype), w)
            out, residual = norm_reference(*tensors, zero_centered=zero)
            out = out.to(tensors[0].dtype)
            if output is not None:
                output.copy_(out)
                out = output
            if residual_output is not None:
                residual_output.copy_(residual)
                residual = residual_output
            if self.site == "d2":
                if aux_output is None:
                    aux_output = torch.empty_like(residual)
                aux_output.copy_(residual)
                return out, residual, aux_output
            return out if self.site == "a" else (out, residual)

    monkeypatch.setattr(vllm, "VllmSite", Site)

    class Graph:
        def __init__(self, fn, inputs):
            self.fn, self.inputs = fn, inputs

        def replay(self):
            for i, tensors in enumerate(self.inputs):
                if hasattr(self.fn, "call_index"):
                    self.fn.call_index(tensors, None, i)
                else:
                    self.fn(tensors)

    class Event:
        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 0.002

    def capture(fn, inputs, outputs, layers, **kwargs):
        assert len(inputs) % layers == 0
        if fn is norm.rw_copy:
            assert len(inputs) == pool_module.pool_count(
                pool_module.pool_spec(args, 4, regime, 512)
            )
            for source, destination in inputs:
                assert source.dtype == destination.dtype == torch.float32
                assert source.numel() * 4 == destination.numel() * 4 == 6 * 4 * args.width
        base = getattr(fn, "norm", fn)
        if isinstance(base, norm.SglangFP32Call) and residual_dtype == "fp32":
            assert inputs[0][1].dtype == torch.float32
        return [
            (Graph(fn, inputs[start : start + layers]), Event(), Event())
            for start in range(0, len(inputs), layers)
        ], []

    monkeypatch.setattr(paired, "capture_bank", capture)
    calls = []

    def start(args, pools, *, cute_config):
        spec = pools.spec
        assert pools.residual.dtype == torch.bfloat16
        monkeypatch.setattr(market, "decode_pool_handles", lambda spec, handles: pools)
        worker = market.Worker.__new__(market.Worker)
        worker.pools = None
        worker.root = args.market_root
        worker.banks = {}
        worker.metadata = {"version": "0.7.1"}

        def operation(x, r, w, eps, enable_pdl):
            assert enable_pdl is None
            return Local()((x, r, w))

        module = SimpleNamespace(fused_add_rmsnorm=operation, gemma_fused_add_rmsnorm=operation)
        worker.wrapper = worker.direct = module

        class Proxy:
            def request(self, command, **payload):
                calls.append(command)
                return worker.dispatch(command, payload)

            def close(self):
                calls.append("exit")
                worker.dispatch("exit", {})

        proxy = Proxy()
        payload = decode_message(encode_message({"spec": spec, "cute_config": asdict(cute_config)}))
        return proxy, proxy.request("setup", **payload)

    monkeypatch.setattr(market, "start_market", start)
    row = norm.run_norm_row(args, 4, regime)
    expected = {"ours", "aa", "cute", "cute_remote", "flashinfer", "vllm", "sglang", "fi071_direct"}
    expected |= {"unfused"} if op.endswith("_sum") else {"vllm_a", "vllm_b"}
    if residual_dtype == "fp32":
        expected.remove("flashinfer")
        expected.update(("sgl_fp32", "sgl_fp32c"))
        assert row["sgl_fp32_ratio"] == 1.0
        assert row["check"]["markets"] == ["torch_fp32", "sgl_fp32"]
        assert "forward_native" in row["providers"]["sgl_fp32"]["kernel"]
        if op == "add_norm":
            expected.add("rw_copy")
            assert row["providers"]["rw_copy"]["read_bytes"] == 6 * 4 * args.width
            assert row["providers"]["rw_copy"]["write_bytes"] == 6 * 4 * args.width
            assert row["providers"]["rw_copy"]["graphs"] == row["graphs_per_candidate"]
    assert "rw_copy" not in row["ratios"]
    assert "rw_copy" not in row["markets"]
    assert "rw_copy" not in row["candidate_verdicts"]
    assert "rw_copy" not in row["check"]["candidate_passed"]
    if omit_vllm:
        expected -= {"vllm", "vllm_a", "vllm_b"}
        assert row["omitted"]["vllm"] == "no vLLM 0.31.0 capture for this shape"
        assert row["providers"]["vllm"]["omitted"]
        assert not row["check"]["vllm_sites"]
    assert set(row["us"]) == expected
    assert row["verdict"] == "PASS" and row["check"]["passed"]
    assert row["providers"]["sglang"]["version"] == "0.7.1"
    assert row["providers"]["sglang"]["kernels"]
    assert row["check"]["remote_metrics"].keys() == {"sglang", "fi071_direct", "cute_remote"}
    assert "cute_remote" not in row["check"]["markets"]
    assert [item for item in constructed if item[0] == "cute"] == [
        ("cute", control_config, bool(static_weight)),
        ("cute", control_config, bool(static_weight)),
    ]
    if residual_dtype == "same":
        assert row["cross_process_control"] == {
            "local": "cute",
            "remote": "cute_remote",
            "ratio_median": 1.0,
            "ci95": [1.0, 1.0],
        }
    else:
        assert "cross_process_control" not in row
    assert "per_input" not in row["check"]
    assert row["shared_buffers"] is True
    assert calls[0] == "setup" and calls[-2:] == ["free", "exit"]
    assert calls.count("sample") == 3 * (args.rounds + 1)  # parent warm plus measured samples
    if (op, static_weight, residual_dtype, omit_vllm, regime) == (
        "add_norm",
        0,
        "fp32",
        False,
        "engine",
    ):
        args.no_rw_copy = True
        without = norm.run_norm_row(args, 4, regime)
        assert "rw_copy" not in without["us"] and "rw_copy" not in without["providers"]
        for key in ("verdict", "candidate_verdicts", "fp32_market_ratio", "fp32_market_ci95"):
            assert without[key] == row[key]


def test_equality_proof_rejects_pointer_dtype_mismatch(tmp_path):
    source = kernel_source().replace(
        "'device': DeviceProperties(index=99)",
        "'signature': {'in_ptr7': '*bf16', 'out_ptr8': '*bf16', 'XBLOCK': 'constexpr'}",
    )
    path = tmp_path / "example.py"
    path.write_text(source)
    # Equal bodies alone are insufficient: implicit load/store dtypes differ.
    other = source.replace("*bf16", "*fp16")
    assert normalize_kernel(source) == normalize_kernel(other)
    receipt = equality_receipt("a", [other], path)
    assert not receipt["equal"] and not receipt["abi_equal"]
    assert receipt["diff"][0].startswith("shipped ABI:")


@pytest.mark.parametrize("static_weight", [0, 1])
def test_candidate_filter_and_sweep_respect_static_weight(monkeypatch, static_weight):
    import inspect

    from benchmark.kernel_tuning import norm_act_norm as norm

    class Local:
        def __init__(self, zero, provider="auto", config=None, *, static_weight=True):
            self.provider, self.config, self.static_weight = provider, config, static_weight

        def __call__(self, tensors):
            return tensors[:2]

    monkeypatch.setattr(norm, "NormCall", Local)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        inspect, "getsource", lambda _: "triton provider requires zero_centered=True"
    )
    monkeypatch.setattr(norm, "flashinfer_metadata", lambda: {"version": "test"})
    monkeypatch.setattr(
        norm,
        "tune_configs",
        lambda hidden: [
            SimpleNamespace(name="static_config", static_weight=True),
            SimpleNamespace(name="dynamic_config", static_weight=False),
        ],
    )
    args = SimpleNamespace(
        op="add_norm",
        width=16,
        static_weight=static_weight,
        sweep=True,
        candidates="ours,fi_default,triton",
    )
    candidates, metadata = norm.norm_candidates(args, torch.bfloat16, (torch.zeros(2, 16),) * 3)
    assert {"ours", "aa", "fi_default", "static_config"} <= candidates.keys()
    assert ("dynamic_config" in candidates) == (static_weight == 0)
    assert "cute" not in candidates and "triton" in metadata["skipped"]
    assert candidates["fi_default"].provider == "flashinfer"


@pytest.mark.parametrize("allow_mismatch", [False, True])
def test_2048_market_rejects_failed_compiled_proof(monkeypatch, tmp_path, allow_mismatch):
    import contextlib

    import torch._inductor.utils

    from benchmark.kernel_tuning import norm_act_probe as probe
    from benchmark.kernel_tuning import norm_act_vllm as vllm

    fixture = tmp_path / "kernel.py"
    fixture.write_text("# SPDX-License-Identifier: Apache-2.0\n# H=4096 fixture\n")
    (tmp_path / "layer_graph.wrapper.txt").write_text("wrapper")
    (tmp_path / "PROVENANCE.txt").write_text("Qwen3-30B-A3B")
    (tmp_path / "vllm-ir").mkdir()
    (tmp_path / "vllm-ir/layernorm.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    monkeypatch.setattr(vllm, "site_file", lambda *a: fixture)

    def native(x, r, w):
        return x + r

    monkeypatch.setattr(vllm, "fragments", lambda *a: {"a": native})
    monkeypatch.setattr(torch, "compile", lambda fn, **k: fn)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *a: None)
    monkeypatch.setattr(
        torch._inductor.utils, "run_and_get_code", lambda fn, *a: (fn(*a), ["H=2048"])
    )
    monkeypatch.setattr(
        vllm, "equality_receipt", lambda *a: {"equal": False, "diff": ["different width"]}
    )
    monkeypatch.setattr(
        vllm, "compiled_configs", lambda: contextlib.nullcontext({"compiled": [{}]})
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA RTX 5090")
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda spec: {
            "kernels": [{"name": "compiled_2048", "grid": [3, 1, 1], "block": [128, 1, 1]}]
        },
    )
    args = SimpleNamespace(
        market_root=tmp_path, vllm_mode="compiled", allow_vllm_mismatch=allow_mismatch
    )
    x = torch.ones(3, 2048, dtype=torch.bfloat16)
    example = (x, x.clone(), torch.ones(2048, dtype=x.dtype))
    with pytest.raises(vllm.MarketError, match="not proven equal"):
        vllm.VllmSite(args, "a", example)
