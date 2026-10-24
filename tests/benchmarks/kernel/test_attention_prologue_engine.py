# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU span slicing and paired protocol tests; no CUDA workloads."""

import json
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning.attention_prologue_engine import (
    InvalidSample,
    KernelEvent,
    SpanLayout,
    collect,
    summarize,
)

from ...support.child import run_child


def test_graph_releases_outputs_before_filler_and_next_projection(monkeypatch):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    outputs, calls = [], []

    class Output:
        pass

    def check_released(*args, **kwargs):
        assert all(ref() is None for ref in outputs)

    def operation():
        check_released()
        output = Output()
        outputs.append(weakref.ref(output))
        calls.append("op")
        return output

    stream = SimpleNamespace(wait_stream=lambda other: None)
    cuda = SimpleNamespace(
        Stream=lambda: stream,
        current_stream=lambda: stream,
        stream=lambda stream: nullcontext(),
        synchronize=lambda: None,
        CUDAGraph=lambda: object(),
        graph=lambda graph: nullcontext(),
        Event=lambda **kwargs: object(),
    )
    monkeypatch.setattr(engine, "torch", SimpleNamespace(cuda=cuda), raising=False)
    monkeypatch.setattr(engine.Graph, "sample", lambda self: 0)
    b = SimpleNamespace(layers=[0, 1, 2], project=check_released, filler=check_released)
    graph = engine.Graph(b, [operation] * 3)
    check_released()
    assert len(calls) == 12  # Three warmups and capture, each with three layers.
    assert not hasattr(graph, "outputs")


def observations(ours=2.0, market=3.0):
    # Baseline varies far more than the measured prologue.
    return [
        {
            "baseline": 100 + i * 11,
            "ours": 100 + i * 11 + ours,
            "self": 100 + i * 11 + ours * 1.01,
            "sglang": 100 + i * 11 + market,
            "vllm": 100 + i * 11 + market * 2,
        }
        for i in range(20)
    ]


@pytest.mark.parametrize(
    "ours,market,verdict", [(2, 3, "faster"), (3, 2, "slower"), (3, 3, "noise")]
)
def test_each_pair_uses_its_own_baseline(ours, market, verdict):
    result = summarize(observations(ours, market), ["ours", "sglang", "vllm"])
    assert result["us_per_layer"]["ours"] == ours
    assert result["speedup"] == pytest.approx(market / ours)
    assert result["epsilon"] == pytest.approx(0.01)
    assert result["market"] == "sglang"
    assert result["verdict"] == verdict
    assert result["ci95"] == pytest.approx([market / ours] * 2)


def test_one_nonpositive_difference_invalidates_ratio_without_clamping():
    samples = observations()
    samples[7]["ours"] = samples[7]["baseline"] - 0.01
    result = summarize(samples, ["ours", "sglang", "vllm"])
    assert result["speedup"] is None
    assert result["verdict"] == "noise"
    assert result["samples_prologue_us_per_layer"]["ours"][7] < 0


def test_surrogate_cannot_claim_a_real_market_win():
    samples = observations()
    for sample in samples:
        sample["vllm_surrogate"] = sample.pop("vllm")
    result = summarize(samples, ["ours", "sglang", "vllm_surrogate"])
    assert result["speedup"] is None
    assert "vllm_surrogate" in result["us_per_layer"]


def test_collect_interleaves_and_samples_one_baseline_and_self_per_pair():
    order = []

    class Clock:
        def __init__(self, name):
            self.name = name

        def sample(self):
            order.append(self.name)
            return 10.0

    graphs = {name: Clock(name) for name in ("ours", "sglang", "vllm")}
    samples = collect(graphs, Clock("baseline"), 20, self_graph=Clock("self"))
    assert len(samples) == 20
    assert order.count("baseline") == 20
    assert order.count("ours") == 20
    assert order.count("self") == 20
    assert order[0] == "baseline"
    assert order[9] == "baseline"
    assert all(set(sample) == {"baseline", "ours", "sglang", "vllm", "self"} for sample in samples)


def kernel_events(names, *, offset=0):
    return [KernelEvent(name, offset + 10 * i, offset + 10 * i + 2) for i, name in enumerate(names)]


def calibrated_layout():
    # Both projection and filler have multiple kernels; their names overlap.
    return SpanLayout.calibrate(
        kernel_events(["gemm", "reduce"] * 2),
        kernel_events(["gemm", "reduce", "gemm", "add"] * 2),
        2,
    )


def test_span_slices_by_count_and_includes_all_prologue_launch_gaps():
    layout = calibrated_layout()
    # Prologue names deliberately collide with projection/filler names. They
    # must not be classified by name or subtracted from a global multiset.
    events = kernel_events(["gemm", "reduce", "gemm", "add", "gemm", "add"])
    events += [
        KernelEvent("gemm", 100, 104),
        KernelEvent("reduce", 106, 110),
        KernelEvent("gemm", 113, 118),
        KernelEvent("add", 122, 129),
        KernelEvent("gemm", 150, 300),
        KernelEvent("add", 310, 500),
    ]
    # Input order is deliberately scrambled; timestamps define execution order.
    events.reverse()
    assert layout.prologue_count(events, 2) == 2
    assert layout.spans(events, 2, 2) == [20, 19]
    from benchmark.kernel_tuning.attention_prologue_engine import Graph

    graph = Graph.__new__(Graph)
    graph.layers, graph.span_layout, graph.prologue_kernels = 2, layout, 2
    graph.profile = lambda: (events, 800)
    assert graph.span_sample() == (19.5, 800)


@pytest.mark.parametrize("delta", [-1, 1, 2, 12])
def test_span_rejects_missing_extra_and_whole_replays(delta):
    events = kernel_events(["gemm", "reduce", "norm", "rope", "gemm", "add"] * 2)
    events = events[:delta] if delta < 0 else events + kernel_events(["extra"] * delta, offset=1000)
    with pytest.raises(InvalidSample, match="kernel count mismatch"):
        calibrated_layout().spans(events, 2, 2)


@pytest.mark.parametrize(
    "index,region", [(0, "projection"), (4, "filler"), (6, "projection"), (11, "filler")]
)
def test_span_rejects_wrong_boundary_names_even_with_correct_total(index, region):
    events = kernel_events(["gemm", "reduce", "norm", "rope", "gemm", "add"] * 2)
    old = events[index]
    events[index] = KernelEvent("unexpected", old.start, old.end)
    with pytest.raises(InvalidSample, match=f"{region} name multiset mismatch"):
        calibrated_layout().spans(events, 2, 2)


@pytest.mark.parametrize(
    "projection,baseline,reason",
    [
        ([], ["p", "f"] * 2, "no GPU kernels"),
        (["p"] * 3, ["p", "f"] * 2, "multiple"),
        (["p"] * 2, ["p", "f", "p"], "multiple"),
        (["p", "other"], ["p", "f"] * 2, "projection name"),
        (["p"] * 2, ["p"] * 2, "filler kernels"),
        (["p"] * 2, ["p", "f", "p", "other"], "filler name"),
        (["p"] * 2, ["p", "f", "other", "f"], "projection name"),
    ],
)
def test_span_calibration_rejects_incomplete_or_inconsistent_layers(projection, baseline, reason):
    with pytest.raises(InvalidSample, match=reason):
        SpanLayout.calibrate(kernel_events(projection), kernel_events(baseline), 2)


@pytest.mark.parametrize("start,end", [(float("nan"), 24), (20, float("inf")), (20, 20)])
def test_span_rejects_invalid_timestamps(start, end):
    events = kernel_events(["gemm", "reduce", "norm", "gemm", "add"] * 2)
    events[2] = KernelEvent("norm", start, end)
    with pytest.raises(InvalidSample, match="timestamps"):
        calibrated_layout().spans(events, 2, 1)


def test_span_accepts_pdl_overlap():
    # A PDL dependent starts before its primary ends; the span runs from the
    # projection's last end to the prologue's last end.
    events = kernel_events(["gemm", "reduce", "norm", "gemm", "add"] * 2)
    reduce_end = events[1].end
    events[2] = KernelEvent("norm", reduce_end - 1, reduce_end + 2)
    assert calibrated_layout().spans(events, 2, 1)[0] == pytest.approx(2)


def span_observations(ours=2.0, market=3.0):
    return [
        {
            "baseline": 0,
            "ours": ours,
            "self": ours * 1.01,
            "sglang": market,
            "vllm": market * 2,
            # Deliberately pathological subtraction, irrelevant to span ratios.
            "graph_us_per_layer": {
                "baseline": 706,
                "ours": 355,
                "self": 708,
                "sglang": 360,
                "vllm": 380,
            },
            "invalid": {},
        }
        for _ in range(20)
    ]


@pytest.mark.parametrize(
    "ours,market,verdict", [(2, 6, "faster"), (6, 2, "slower"), (3, 3, "noise")]
)
def test_span_summary_ignores_graph_jitter_and_preserves_paired_statistics(ours, market, verdict):
    samples = span_observations(ours, market)
    result = summarize(samples, ["ours", "sglang", "vllm"], timing="span")
    assert result["us_per_layer"]["ours"] == ours
    assert result["speedup"] == pytest.approx(market / ours)
    assert result["ci95"] == pytest.approx([market / ours] * 2)
    assert result["epsilon"] == pytest.approx(0.01)
    assert result["verdict"] == verdict
    assert result["samples_subtraction_us_per_layer"][0]["ours"] == -351


def test_span_uses_faster_real_market_and_median_of_paired_ratios():
    samples = span_observations()
    for index, sample in enumerate(samples):
        sample.update(
            ours=index + 1, self=(index + 1) * 1.01, sglang=100, vllm=30, vllm_surrogate=0.1
        )
    result = summarize(samples, ["ours", "sglang", "vllm", "vllm_surrogate"], timing="span")
    assert result["market"] == "vllm"
    assert result["speedup"] == pytest.approx((30 / 10 + 30 / 11) / 2)


def test_span_discards_whole_invalid_pairs_and_reports_them():
    samples = span_observations()
    samples[2]["ours"] = None
    samples[2]["sglang"] = 1000
    samples[2]["invalid"]["ours"] = {"reason": "kernel count mismatch: expected 12, got 11"}
    result = summarize(samples, ["ours", "sglang", "vllm"], timing="span")
    assert result["pairs"] == 19
    assert result["invalid_pairs"] == 1
    assert result["attempted_pairs"] == 20
    assert len(result["samples_prologue_us_per_layer"]["sglang"]) == 19
    assert result["speedup"] == 1.5
    assert result["samples"][2]["invalid"] == samples[2]["invalid"]


def test_span_requires_six_valid_pairs_for_exact_ci():
    samples = span_observations()
    for sample in samples[5:]:
        sample["invalid"]["self"] = {"reason": "missing kernels"}
        sample["self"] = None
    result = summarize(samples, ["ours", "sglang", "vllm"], timing="span")
    assert result["pairs"] == 5
    assert result["verdict"] == "noise"
    assert result["speedup"] is None
    assert "fewer than six" in result["reason"]


def test_span_collect_rotates_with_distinct_self_and_reports_invalid_replay():
    order = []

    class Clock:
        def __init__(self, name):
            self.name = name

        def span_sample(self):
            order.append(self.name)
            if self.name == "vllm":
                raise InvalidSample("kernel count mismatch: expected 12, got 11")
            return 2, 355

    samples = collect(
        {name: Clock(name) for name in ("ours", "sglang", "vllm")},
        Clock("baseline"),
        2,
        self_graph=Clock("self"),
        timing="span",
    )
    assert order == [
        "baseline",
        "ours",
        "sglang",
        "vllm",
        "self",
        "ours",
        "vllm",
        "sglang",
        "self",
        "baseline",
    ]
    assert all(sample["vllm"] is None and "vllm" in sample["invalid"] for sample in samples)
    assert all(sample["graph_us_per_layer"]["ours"] == 355 for sample in samples)


@pytest.mark.parametrize("timing", ["span", "subtract"])
@pytest.mark.parametrize("filler", ["model", "legacy"])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3"])
def test_row_builds_independent_self_graph_and_retries_noise(monkeypatch, timing, filler, kv_dtype):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    args = engine.parser().parse_args(
        ["--layers", "2", "--timing", timing, "--filler", filler, "--kv-dtype", kv_dtype]
    )
    filler_names = ["o_proj", "gate_up", "down"] if filler == "model" else ["filler", "eviction"]
    buffers = SimpleNamespace(
        table=SimpleNamespace(to=lambda dtype: None),
        vllm_table=None,
        width=6144,
        eviction_bytes=0 if filler == "model" else 200_000_000,
        layers=[None, None],
        cache=lambda: object(),
        qsize=4096,
        ksize=1024,
        hd=128,
        rd=128,
    )
    monkeypatch.setattr(engine, "Inputs", lambda tokens, args: buffers)
    monkeypatch.setattr(
        engine,
        "torch",
        SimpleNamespace(bfloat16=None, cuda=SimpleNamespace(get_device_name=lambda device: "5090")),
        raising=False,
    )
    errors = [{"k": {"max_fp8_ulp": 0.5}, "v": {"max_fp8_ulp": 0.25}}]
    monkeypatch.setattr(engine, "correctness", lambda *args: (True, errors))
    monkeypatch.setattr(
        engine, "operations", lambda name, b, caches, providers: ([object(), object()], name)
    )
    built, collections = [], []

    class FakeGraph:
        def __init__(self, b, ops, *, filler=True):
            self.ops, self.filler = ops, filler
            built.append(self)

        def profile(self):
            names = ["projection"] + (["prologue"] if self.ops else [])
            names += filler_names if self.filler else []
            return kernel_events(names * 2), 353

    def fake_collect(graphs, baseline, count, *, self_graph, timing):
        assert self_graph is not graphs["ours"]
        assert self_graph.owners[0] is not graphs["ours"].owners[0]
        assert self_graph.owners[1][0] is not graphs["ours"].owners[1][0]
        if timing == "span":
            assert self_graph.prologue_kernels == graphs["ours"].prologue_kernels == 1
        collections.append(count)
        factory = span_observations if timing == "span" else observations
        samples = (factory(3, 3) * (count // 20))[:count]
        for sample in samples:
            sample["compose"] = sample["ours"]
            if kv_dtype != "bf16":
                sample["sglang_noscale"] = sample["sglang"]
        return samples

    monkeypatch.setattr(engine, "Graph", FakeGraph)
    monkeypatch.setattr(engine, "collect", fake_collect)
    providers = {"sglang": True, "vllm": True}
    if kv_dtype != "bf16":
        providers["sglang_noscale"] = True
    row = engine.run_row(16, args, providers)
    if kv_dtype != "bf16":
        assert row["stacks"]["sglang_noscale"]["status"] == "ok"
        assert row["stacks"]["ours"]["max_fp8_ulp"] == {"k": 0.5, "v": 0.25}
    assert collections == [20, 40]
    assert row["comparison"]["pairs"] == 40
    assert row["comparison"]["initial"]["pairs"] == 20
    expected_verdict = "diagnostic" if timing == "subtract" or filler == "legacy" else "noise"
    assert row["comparison"]["verdict"] == expected_verdict
    assert row["comparison"]["initial"]["verdict"] == expected_verdict
    assert row["protocol"]["lifetime_model"] == engine.LIFETIME_MODEL
    assert row["protocol"]["filler_kind"] == filler
    if filler == "model":
        assert row["protocol"]["filler_bytes_per_layer"] == 320 * 1024**2 + 16 * 106496
    assert row["stacks"]["self"]["status"] == "ok"
    if timing == "span":
        assert row["span_layout"] == {
            "projection_per_layer": 1,
            "filler_per_layer": len(filler_names),
        }
        assert sum(not graph.filler for graph in built) == 1


def test_print_row_reports_invalid_samples_without_json(capsys):
    from benchmark.kernel_tuning.attention_prologue_engine import print_row

    samples = span_observations()
    samples[0]["invalid"]["ours"] = {"reason": "kernel count mismatch: expected 12, got 11"}
    print_row(
        {
            "tokens": 16,
            "timing": "span",
            "comparison": summarize(samples, ["ours", "sglang", "vllm"], timing="span"),
            "sol": {"us_per_layer": 1},
        }
    )
    output = capsys.readouterr().out
    assert "T=16 timing=span" in output
    assert "invalid_pairs=1/20" in output
    assert "final pair=0" in output
    assert "expected 12, got 11" in output
    assert "SOL=1.000us/layer" in output


def test_engine_import_and_help_do_not_import_torch_or_gpu_providers():
    run_child(
        [
            sys.executable,
            "-c",
            """
import sys
from benchmark.kernel_tuning import attention_prologue_engine as engine
assert engine.parser().parse_args([]).layers == 8
assert engine.parser().parse_args([]).pairs == 20
assert engine.parser().parse_args([]).timing == 'span'
assert engine.parser().parse_args(['--timing', 'subtract']).timing == 'subtract'
for prefix in ('torch', 'triton', 'cutlass', 'ctikernels', 'sglang', 'vllm'):
    assert not any(n == prefix or n.startswith(prefix + '.') for n in sys.modules), prefix
""",
        ],
        timeout=600,
        check=True,
    )


@pytest.mark.parametrize("variant,qh,kh,hd", [("hd128", 32, 8, 128), ("hd256", 16, 4, 256)])
@pytest.mark.parametrize("tokens", [1, 16, 1024])
def test_model_protocol_and_filler_bytes(variant, qh, kh, hd, tokens, capsys):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    args = engine.parser().parse_args(["--variant", variant, "--rows", str(tokens)])
    protocol = engine.protocol_metadata(args, tokens=tokens)
    assert protocol["lifetime_model"] == engine.LIFETIME_MODEL
    assert protocol["layers"] == 8
    assert protocol["filler_kind"] == "model"
    assert protocol["filler_dtype"] == "bfloat16"
    assert protocol["num_q_heads"] == qh
    assert protocol["num_kv_heads"] == kh
    assert protocol["head_dim"] == hd
    assert protocol["hidden"] == 4096
    assert protocol["intermediate"] == 12288
    assert protocol["filler_weight_shapes"] == {
        "o_proj": (qh * hd, 4096),
        "gate_up": (4096, 24576),
        "down": (12288, 4096),
    }
    assert protocol["filler_weight_bytes_per_layer"] == (32 + 192 + 96) * 1024**2
    assert protocol["filler_bytes_per_layer"] == 320 * 1024**2 + tokens * 106496
    assert engine.protocol_metadata(args)["filler_bytes_per_layer"] == {
        str(tokens): protocol["filler_bytes_per_layer"]
    }
    assert protocol["eviction_bytes"] == 0
    assert not protocol["diagnostic"]
    assert not any(protocol["diagnostic_flags"].values())
    engine.print_protocol(protocol)
    output = capsys.readouterr().out
    assert "filler=model" in output and "diagnostic=False" in output


def test_qwen3_moe_protocol_uses_its_own_model_sizes():
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    args = engine.parser().parse_args(["--variant", "hd128-moe", "--rows", "4"])
    protocol = engine.protocol_metadata(args, tokens=4)
    assert (protocol["num_q_heads"], protocol["num_kv_heads"], protocol["head_dim"]) == (32, 4, 128)
    assert (protocol["hidden"], protocol["intermediate"]) == (2048, 6144)
    assert protocol["filler_weight_shapes"] == {
        "o_proj": (4096, 2048),
        "gate_up": (2048, 12288),
        "down": (6144, 2048),
    }
    assert protocol["filler_bytes_per_layer"] == 88 * 1024**2 + 4 * 57344


@pytest.mark.parametrize("variant", ["hd128", "hd256"])
def test_legacy_filler_byte_accounting_and_diagnostics(variant):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    args = engine.parser().parse_args(
        [
            "--variant",
            variant,
            "--filler",
            "legacy",
            "--filler-width",
            "256",
            "--evict-mib",
            "16",
        ]
    )
    protocol = engine.protocol_metadata(args, tokens=16, eviction_bytes=32 * 1024**2)
    assert protocol["filler_kind"] == "legacy"
    assert protocol["filler_weight_bytes_per_layer"] == 4096 * 256 * 2
    assert (
        protocol["filler_bytes_per_layer"] == 4096 * 256 * 2 + 16 * (4096 + 256) * 2 + 64 * 1024**2
    )
    assert protocol["diagnostic"]
    assert protocol["diagnostic_flags"] == {
        "legacy_filler": True,
        "filler_width_override": True,
        "evict_mib_override": True,
        "subtract_timing": False,
    }


@pytest.mark.parametrize(
    "option", [["--filler-width", "256"], ["--evict-mib", "16"], ["--timing", "subtract"]]
)
def test_diagnostic_options_are_recorded_even_with_model_filler(option):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    protocol = engine.protocol_metadata(engine.parser().parse_args(option), tokens=1)
    assert protocol["diagnostic"]
    assert protocol["filler_kind"] == "model"
    assert protocol["eviction_bytes"] == 0


@pytest.mark.parametrize("diagnostic", [False, True])
def test_paired_protocol_saves_flag_and_suppresses_all_diagnostic_verdicts(tmp_path, diagnostic):
    from benchmark.kernel_tuning import attention_prologue_paired as paired

    args = paired.parser().parse_args(["--vllm-distinct-outputs"] if diagnostic else [])
    report = {
        "protocol": paired.protocol_metadata(args),
        "rows": [
            {
                "bar": {"verdict": "faster"},
                "bar_sglang": {"verdict": "below"},
                "comparisons": {"vllm": {"verdict": "faster", "initial": {"verdict": "noise"}}},
                "sweep": {"results": [{"paired": {"verdict": "below"}}]},
            }
        ],
    }
    path = tmp_path / "paired.json"
    paired.write_report(report, path)
    saved = json.loads(path.read_text())
    assert saved["protocol"]["vllm_distinct_outputs"] is diagnostic
    assert saved["protocol"]["diagnostic"] is diagnostic
    row = saved["rows"][0]
    assert row["bar"]["verdict"] == ("diagnostic" if diagnostic else "faster")
    assert row["bar_sglang"]["verdict"] == ("diagnostic" if diagnostic else "below")
    assert row["comparisons"]["vllm"]["initial"]["verdict"] == (
        "diagnostic" if diagnostic else "noise"
    )
    assert row["sweep"]["results"][0]["paired"]["verdict"] == (
        "diagnostic" if diagnostic else "below"
    )
    assert "cache-pressure diagnostic" in paired.parser().format_help()
    assert "never produces a verdict" in paired.parser().format_help()


def test_span_calibrates_three_model_gemms_with_overlapping_names():
    projection = kernel_events(["gemm", "reduce"] * 2)
    baseline = kernel_events(["gemm", "reduce", "gemm", "gate_up", "gemm", "reduce"] * 2)
    layout = SpanLayout.calibrate(projection, baseline, 2)
    assert layout.filler == ("gemm", "gate_up", "gemm", "reduce")
    events = kernel_events(
        ["gemm", "reduce", "norm", "rope", "gemm", "gate_up", "gemm", "reduce"] * 2
    )
    assert layout.prologue_count(events, 2) == 2
    assert layout.spans(events, 2, 2) == [20, 20]
    events[-1] = KernelEvent("unexpected", events[-1].start, events[-1].end)
    with pytest.raises(InvalidSample, match="filler name multiset mismatch"):
        layout.spans(events, 2, 2)


@pytest.mark.parametrize("variant,no_gate", [("hd128", False), ("hd256", False), ("hd256", True)])
@pytest.mark.parametrize("filler", ["model", "legacy"])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3", "fp8_e5m2"])
def test_inputs_share_activations_and_use_distinct_model_weights(
    monkeypatch, variant, no_gate, filler, kv_dtype
):
    import torch

    from benchmark.kernel_tuning import attention_prologue_engine as engine

    # Meta tensors exercise the actual builder and layout code without memory
    # allocation or CUDA. Intercept every explicit CUDA factory/transfer.
    real_to = torch.Tensor.to

    def meta_to(tensor, *args, **kwargs):
        args = tuple(
            "meta" if isinstance(arg, str) and arg.startswith("cuda:") else arg for arg in args
        )
        return real_to(tensor, *args, **kwargs)

    def factory(name):
        def create(*args, **kwargs):
            kwargs.pop("generator", None)
            kwargs["device"] = "meta"
            return getattr(torch, name)(*args, **kwargs)

        return create

    calls = []

    def matmul(left, right, *, out):
        assert left.shape[1] == right.shape[0]
        assert out.shape == (left.shape[0], right.shape[1])
        calls.append((left, right, out))

    def properties(device):
        assert filler == "legacy"
        return SimpleNamespace(L2_cache_size=1024)

    proxy = SimpleNamespace(
        **{
            name: factory(name)
            for name in ("randn", "empty", "arange", "randperm", "ones", "zeros", "full")
        },
        bfloat16=torch.bfloat16,
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e5m2=torch.float8_e5m2,
        float32=torch.float32,
        float64=torch.float64,
        int64=torch.int64,
        int32=torch.int32,
        cat=torch.cat,
        matmul=matmul,
        Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: None),
        cuda=SimpleNamespace(get_device_properties=properties),
    )
    monkeypatch.setattr(torch.Tensor, "to", meta_to)
    monkeypatch.setattr(engine, "torch", proxy, raising=False)
    args = engine.parser().parse_args(
        [
            "--variant",
            variant,
            "--layers",
            "3",
            "--kv-dtype",
            kv_dtype,
            "--filler",
            filler,
            *(["--no-output-gate"] if no_gate else []),
        ]
    )
    b = engine.Inputs(16, args)
    assert all(layer.raw is b.raw for layer in b.layers)
    assert all(layer.x is b.layers[0].x for layer in b.layers)
    assert len({id(layer.w) for layer in b.layers}) == 3
    if b.gated:
        assert all(layer.raw_engine is b.raw for layer in b.layers)
    caches = [b.cache() for _ in b.layers]
    assert len({id(cache) for pair in caches for cache in pair}) == 6
    assert all(
        cache.dtype == getattr(torch, engine.KV_DTYPES[kv_dtype])
        for pair in caches
        for cache in pair
    )
    assert b.k_scale.dtype == b.v_scale.dtype == torch.float32
    assert b.k_scale.numel() == b.v_scale.numel() == 1
    for layer in b.layers:
        b.project(layer, engine=b.gated)
    assert all(out is b.raw for _, _, out in calls)
    calls.clear()
    for layer in b.layers:
        b.filler(layer)
    if filler == "model":
        assert len(calls) == 9
        assert b.eviction_bytes == 0
        assert not hasattr(b, "eviction")
        for name, shape in engine.filler_shapes(variant).items():
            weights = [getattr(layer, name) for layer in b.layers]
            assert all(
                weight.shape == shape and weight.dtype == torch.bfloat16 for weight in weights
            )
            assert len({id(weight) for weight in weights}) == 3
        assert (
            sum(getattr(b.layers[0], name).numel() * 2 for name in engine.filler_shapes(variant))
            == 320 * 1024**2
        )
        for index, layer in enumerate(b.layers):
            o_proj, gate_up, down = calls[3 * index : 3 * index + 3]
            assert o_proj[0] is b.filler_input and o_proj[1] is layer.o_proj
            assert gate_up[0] is o_proj[2] and gate_up[1] is layer.gate_up
            assert down[0]._base is gate_up[2] and down[1] is layer.down
            assert down[2] is b.filler_hidden
    else:
        assert len(calls) == 3
        assert b.eviction_bytes == 2048
        assert all(weight is b.filler_weight and out is b.filler_output for _, weight, out in calls)


@pytest.mark.parametrize("filler", ["model", "legacy"])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3", "fp8_e5m2"])
def test_engine_saves_full_protocol_and_prints_it_before_rows(
    monkeypatch, tmp_path, capsys, filler, kv_dtype
):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    cuda = SimpleNamespace(
        set_device=lambda device: None,
        get_device_name=lambda device: "CPU test fake",
        empty_cache=lambda: None,
    )
    fake_torch = SimpleNamespace(cuda=cuda, no_grad=nullcontext)
    real_import = engine.importlib.import_module
    monkeypatch.setattr(engine, "torch", fake_torch, raising=False)
    monkeypatch.setattr(
        engine.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import(name),
    )
    monkeypatch.setattr(engine, "load_providers", lambda args: ({}, {}))

    def row(tokens, args, providers):
        protocol = engine.protocol_metadata(args, tokens=tokens, eviction_bytes=2048)
        return {
            "tokens": tokens,
            "protocol": protocol,
            "comparison": {"verdict": "diagnostic" if protocol["diagnostic"] else "noise"},
        }

    monkeypatch.setattr(engine, "run_row", row)
    path = tmp_path / "engine.json"
    engine.main(
        [
            "--rows",
            "1,16",
            "--layers",
            "3",
            "--filler",
            filler,
            "--kv-dtype",
            kv_dtype,
            "--json",
            str(path),
        ]
    )
    saved = json.loads(path.read_text())
    protocol = saved["protocol"]
    assert protocol["kv_dtype"] == kv_dtype
    assert protocol["k_scale"] == (None if kv_dtype == "bf16" else 0.0478)
    assert protocol["v_scale"] == (None if kv_dtype == "bf16" else 0.0213)
    assert protocol["lifetime_model"] == engine.LIFETIME_MODEL
    assert protocol["layers"] == 3
    assert protocol["filler_kind"] == filler
    assert protocol["diagnostic"] is (filler == "legacy")
    assert protocol["eviction_bytes"] == (2048 if filler == "legacy" else 0)
    assert protocol["filler_bytes_per_layer"] == {
        str(row["tokens"]): row["protocol"]["filler_bytes_per_layer"] for row in saved["rows"]
    }
    assert all(value > 0 for value in protocol["filler_bytes_per_layer"].values())
    output = capsys.readouterr().out
    assert output.startswith("Protocol:")
    assert output.index("Protocol:") < output.index("T=1")


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3", "fp8_e5m2"])
def test_kv_dtype_flag_protocol_and_bytes(kv_dtype, capsys):
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    args = engine.parser().parse_args(["--kv-dtype", kv_dtype])
    assert engine.parser().parse_args([]).kv_dtype == "bf16"
    protocol = engine.protocol_metadata(args)
    assert protocol["kv_dtype"] == kv_dtype
    fp8 = kv_dtype != "bf16"
    assert protocol["k_scale"] == (0.0478 if fp8 else None)
    assert protocol["v_scale"] == (0.0213 if fp8 else None)
    assert protocol["sglang_noscale"] is fp8
    assert protocol["sglang_noscale_scales"] == {"k_scale": None, "v_scale": None}
    saved = json.loads(json.dumps(protocol))
    for name in ("kv_dtype", "k_scale", "v_scale", "sglang_noscale_scales"):
        assert saved[name] == protocol[name]
    engine.print_protocol(protocol)
    output = capsys.readouterr().out
    assert f"kv_dtype={kv_dtype}" in output
    assert f"k_scale={protocol['k_scale']}" in output
    assert f"v_scale={protocol['v_scale']}" in output
    bf16 = engine.prologue_bytes(37, 4096, 1024, 128, 128, "bf16")
    moved = engine.prologue_bytes(37, 4096, 1024, 128, 128, kv_dtype)
    assert moved == bf16 - (2 * 37 * 1024 - 8 if fp8 else 0)


def test_kv_dtype_flag_rejects_unknown_formats():
    from benchmark.kernel_tuning import attention_prologue_engine as engine

    with pytest.raises(SystemExit):
        engine.parser().parse_args(["--kv-dtype", "float16"])


@pytest.mark.parametrize("winner", ["sglang", "sglang_noscale", "vllm"])
def test_fp8_market_uses_fastest_real_stack(winner):
    samples = span_observations()
    for sample in samples:
        sample.update(sglang=5, sglang_noscale=4, vllm=6, vllm_surrogate=0.01)
        sample[winner] = 3
    names = ["ours", "sglang", "sglang_noscale", "vllm", "vllm_surrogate"]
    result = summarize(samples, names, timing="span", kv_dtype="fp8_e4m3")
    assert result["market"] == winner
    assert result["speedup"] == 1.5
    assert result["verdict"] == "faster"
    names.remove("sglang_noscale")
    assert summarize(samples, names, timing="span", kv_dtype="fp8_e4m3")["speedup"] is None


@pytest.mark.parametrize(
    "prologue",
    [
        ["norm", "rope", "div", "div", "cast", "cast", "store"],
        ["norm", "rope", "cast", "cast", "store"],
        ["norm", "rope", "reshape_and_cache_flash"],
    ],
)
def test_span_calibrates_fp8_conversion_kernels(prologue):
    layout = calibrated_layout()
    events = kernel_events((["gemm", "reduce"] + prologue + ["gemm", "add"]) * 2)
    count = layout.prologue_count(events, 2)
    assert count == len(prologue)
    assert layout.spans(events, 2, count) == [10 * count] * 2
    with pytest.raises(InvalidSample, match="count mismatch"):
        layout.spans(events[:-1], 2, count)


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_ulp_gate_handles_subnormals_saturation_and_bitwise_v(monkeypatch, dtype_name):
    import torch

    from benchmark.kernel_tuning import attention_prologue_engine as engine

    monkeypatch.setattr(engine, "torch", torch, raising=False)
    dtype = getattr(torch, dtype_name)
    info = torch.finfo(dtype)
    reference = torch.tensor([0, info.tiny / 2, 1, info.max * 2, -info.max * 2])
    expected = reference.clamp(-info.max, info.max).to(dtype)
    scale = torch.ones(1)
    record = engine.fp8_error(expected, reference, scale, dtype, limit=1, bitwise=True)
    assert record["passed"] and record["max_fp8_ulp"] == 0
    assert engine.fp8_ulp(reference, dtype)[0].item() == info.tiny * info.eps
    got = torch.tensor([1 + 2 * info.eps], dtype=dtype)
    ref = torch.tensor([1 + info.eps / 4])
    assert not engine.fp8_error(got, ref, scale, dtype, limit=1)["passed"]
    assert engine.fp8_error(got, ref, scale, dtype, limit=2)["passed"]
    got = torch.tensor([1 + info.eps], dtype=dtype)
    assert engine.fp8_error(got, ref, scale, dtype, limit=1)["passed"]
    assert not engine.fp8_error(got, ref, scale, dtype, limit=1, bitwise=True)["passed"]
    negative_zero = torch.tensor([-0.0], dtype=dtype)
    assert not engine.fp8_error(negative_zero, torch.zeros(1), scale, dtype, limit=1, bitwise=True)[
        "passed"
    ]
    record = engine.fp8_error(torch.tensor([float("nan")], dtype=dtype), ref, scale, dtype, limit=2)
    assert not record["passed"]
    assert record["max_fp8_ulp"] is None
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("stack", ["ours", "self", "compose", "sglang", "sglang_noscale", "vllm"])
@pytest.mark.parametrize("wrong", ["k", "v", "untouched"])
def test_fp8_correctness_applies_stack_gates(monkeypatch, stack, wrong):
    import torch

    from benchmark.kernel_tuning import attention_prologue_engine as engine

    monkeypatch.setattr(engine, "torch", torch, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    dtype = torch.float8_e4m3fn
    scale = torch.tensor([0.5])
    b = SimpleNamespace(
        layers=[None],
        project=lambda layer: None,
        fp8=True,
        cache_dtype=dtype,
        k_scale=scale,
        v_scale=scale,
        slots=2,
        indices64=torch.tensor([1]),
    )
    effective_scale = 1 if stack == "sglang_noscale" else 0.5
    q = torch.ones(1, 1, 1)
    k = torch.full_like(q, 1.03125 * effective_scale)
    v = torch.full_like(q, effective_scale)
    monkeypatch.setattr(engine, "reference", lambda *args: ((q, k, v), (q, k, v)))
    kc, vc = (torch.zeros(2, 1, 1, dtype=dtype) for _ in range(2))

    def run():
        kc[1].copy_(torch.tensor(1.25 if wrong == "k" else 1, dtype=dtype))
        vc[1].copy_(torch.tensor(1.125 if wrong == "v" else 1, dtype=dtype))
        if wrong == "untouched":
            kc[0].copy_(torch.tensor(1, dtype=dtype))
        return q

    passed, records = engine.correctness(b, [run], [(kc, vc)], stack)
    expected = (
        stack not in ("ours", "self")
        if wrong == "k"
        else stack == "sglang"
        if wrong == "v"
        else False
    )
    assert passed is expected
    assert records[0]["k"]["ulp_limit"] == (1 if stack in ("ours", "self") else 2)
    assert records[0]["v"]["requires_bitwise"] is (stack != "sglang")


@pytest.mark.parametrize("kv_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("stack", ["sglang", "sglang_noscale", "vllm", "compose", "ours"])
def test_fp8_stack_storage_and_scale_dispatch(monkeypatch, kv_dtype, stack):
    import torch

    from benchmark.kernel_tuning import attention_prologue_engine as engine

    monkeypatch.setattr(engine, "torch", torch, raising=False)
    noop = lambda *args, **kwargs: None
    for module, functions in (
        ("ctikernels.norm.inference", {"rms_norm_": noop}),
        ("ctikernels.rotary.inference", {"apply_rotary_": noop}),
    ):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**functions))
    b = engine.Inputs.__new__(engine.Inputs)
    b.qh, b.kh, b.hd, b.rd = 1, 1, 16, 16
    b.qsize, b.ksize, b.tokens, b.slots = 16, 16, 2, 16
    b.zero, b.gated, b.fp8 = False, False, True
    b.cache_dtype = getattr(torch, engine.KV_DTYPES[kv_dtype])
    b.cache_kind = "fp8" if kv_dtype == "fp8_e4m3" else "fp8_e5m2"
    b.table, b.positions = torch.zeros(2, 16), torch.tensor([0, 1])
    b.indices64, b.indices = torch.tensor([3, 7]), torch.tensor([3, 7], dtype=torch.int32)
    b.k_scale, b.v_scale = [torch.tensor([v]) for v in engine.KV_SCALES]
    raw = torch.linspace(-2, 2, 96).reshape(2, 48).bfloat16()
    original = raw.clone()
    b.layers = [SimpleNamespace(raw=raw, qw=torch.ones(16), kw=torch.ones(16))]
    kc, vc = b.cache()
    calls = []

    def store(k, v, out_k, out_v, indices, **kwargs):
        calls.append(kwargs)
        assert k.dtype == v.dtype == out_k.dtype == out_v.dtype == torch.uint8
        assert out_k.data_ptr() == kc.data_ptr() and out_v.data_ptr() == vc.data_ptr()
        out_k.view(16, 16)[indices.long()] = k.view(2, 16)
        out_v.view(16, 16)[indices.long()] = v.view(2, 16)

    monkeypatch.setitem(
        sys.modules,
        "ctikernels.store",
        SimpleNamespace(store_cache=lambda ck, cv, indices, k, v: store(k, v, ck, cv, indices)),
    )

    def cache_op(k, v, ck, cv, indices, kind, ks, vs):
        assert kind == b.cache_kind
        assert ks is b.k_scale and vs is b.v_scale
        assert ck.shape == cv.shape == (1, 16, 1, 16)
        limit = torch.finfo(b.cache_dtype).max
        quantized = [
            (x.float() / scale).clamp(-limit, limit).to(b.cache_dtype).view(torch.uint8)
            for x, scale in ((k, ks), (v, vs))
        ]
        store(*quantized, ck, cv, indices)

    class FakePrologue:
        provider = "cute"

        def __init__(self, **kwargs):
            assert kwargs["cache_dtype"] == b.cache_dtype

        def __call__(self, packed, qw, kw, table, positions, ck, cv, indices, **scales):
            q, k, v = b.split(packed)
            assert ck.dtype == cv.dtype == b.cache_dtype
            cache_op(
                k,
                v,
                ck.view(torch.uint8).view(1, 16, 1, 16),
                cv.view(torch.uint8).view(1, 16, 1, 16),
                indices,
                b.cache_kind,
                scales["k_scale"],
                scales["v_scale"],
            )
            return q

    monkeypatch.setitem(
        sys.modules, "ctikernels.attention.inference", SimpleNamespace(Prologue=FakePrologue)
    )
    monkeypatch.setattr(engine, "native_region", lambda *args: lambda raw: b.split(raw)[:2])
    providers = {
        "sglang": (
            SimpleNamespace(fused_inplace_qknorm=noop),
            SimpleNamespace(apply_rope_with_cos_sin_cache_inplace=noop),
            SimpleNamespace(store_cache=store),
        ),
        "vllm": (),
        "cache_op": cache_op,
    }
    operations, _ = engine.operations(stack, b, [(kc, vc)], providers)
    q = operations[0]()
    assert torch.equal(q, b.split(original)[0])
    for cache, ref, value in zip((kc, vc), b.split(original)[1:], engine.KV_SCALES, strict=True):
        scale = 1 if stack == "sglang_noscale" else value
        expected = ref / scale if stack.startswith("sglang") else ref.float() / scale
        expected = expected.to(b.cache_dtype).view(torch.uint8)
        assert torch.equal(cache.view(torch.uint8)[b.indices64], expected)
    if stack.startswith("sglang"):
        assert calls == [{"row_bytes": 16, "v_row_bytes": 16, "size_limit": 16}]
        expected_raw = original.clone()
        if stack == "sglang":
            k, v = b.split(expected_raw)[1:]
            k.div_(engine.KV_SCALES[0])
            v.div_(engine.KV_SCALES[1])
        assert torch.equal(raw, expected_raw)
