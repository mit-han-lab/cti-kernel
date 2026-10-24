# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import math

import pytest
import torch

from benchmark.kernel_tuning.norm_act_norm import check_norm_candidates
from benchmark.kernel_tuning.norm_act_paired import median_ci, paired_summary
from benchmark.kernel_tuning.norm_act_reference import norm_reference


def test_exact_median_order_statistics():
    median, ci = median_ci(list(range(1, 31)))
    assert median == 15.5
    assert ci == [10, 21]
    tail = sum(math.comb(30, i) for i in range(10)) / 2**30
    assert 1 - 2 * tail >= 0.95
    assert 1 - 2 * (tail + math.comb(30, 10) / 2**30) < 0.95
    assert median_ci([1] * 5)[1] == [0, float("inf")]
    with pytest.raises(ValueError):
        median_ci([])


def test_paired_ratios_use_within_round_fastest_and_aa_epsilon():
    samples = {
        "ours": [2.0] * 30,
        "aa": [2.0] * 30,
        "vllm": [1.0, 4.0] * 15,
        "sglang": [4.0, 1.0] * 15,
    }
    result = paired_summary(samples)
    assert result["epsilon"] == 0.005
    assert result["faster_ratio"] == 0.5
    assert result["verdict"] == "FAIL"
    samples["vllm"] = [2.0] * 30
    samples["sglang"] = [3.0] * 30
    assert paired_summary(samples)["verdict"] == "PASS"
    samples["aa"] = [2.04] * 30
    assert paired_summary(samples)["epsilon"] == pytest.approx(0.02)


def test_rw_copy_receipt_is_diagnostic_only():
    from benchmark.kernel_tuning.norm_act_paired import parse_args, sol_summary

    args = ["--op", "add_norm", "--market-root", ".", "--residual-dtype", "fp32"]
    assert not parse_args(args).no_rw_copy
    assert parse_args([*args, "--no-rw-copy"]).no_rw_copy
    samples = {
        "ours": [2.0] * 30,
        "aa": [2.0] * 30,
        "static": [1.5] * 30,
        "triton": [3.0] * 30,
        "cute": [2.5] * 30,
        "cute_t128_r1_p0_s0_tn_c2": [4.0] * 30,
        "cute_remote": [0.1] * 30,
        "sglang": [3.0] * 30,
        "sgl_fp32": [5.0] * 30,
        "sgl_fp32c": [4.0] * 30,
    }
    baseline = paired_summary(samples)
    result = paired_summary(samples | {"rw_copy": [0.25, 0.75] * 15})
    assert result["us"].pop("rw_copy") == 0.5
    assert result == baseline  # No market ratio, (a) bar or verdict changes.
    sol = sol_summary(64, 2048, baseline["us"] | {"rw_copy": 0.5}, 200, "cute")
    assert sol["sol_witness"] == {
        "ours": 0.25,
        "static": 0.5 / 1.5,
        "triton": 0.5 / 3,
        "cute": 0.2,
        "cute_t128_r1_p0_s0_tn_c2": 0.125,
    }
    sol.pop("sol_witness")
    assert sol == sol_summary(64, 2048, baseline["us"], 200, "cute")


@pytest.mark.parametrize("regime", ["hot", "engine", "cold"])
@pytest.mark.parametrize("width", [3, 16])
def test_rw_copy_volume_rotation_and_restore(regime, width):
    from types import SimpleNamespace

    from benchmark.kernel_tuning.norm_act_norm import rw_copy
    from benchmark.kernel_tuning.norm_act_pools import (
        build_norm_pools,
        build_rw_copy_pools,
        fp32_residual_pools,
        pool_spec,
    )

    args = SimpleNamespace(op="add_norm", width=width, layers=4, dtype="bf16", static_weight=0)
    norm = fp32_residual_pools(build_norm_pools(pool_spec(args, 3, regime, 512), device="cpu"))
    copy = build_rw_copy_pools(norm)
    assert len(copy.inputs) == len(norm.inputs) == norm.count
    assert copy.source.shape[0] == len(norm.tensors)
    assert len({src.data_ptr() for src, _ in copy.inputs}) == len(norm.tensors)
    assert len({dst.data_ptr() for _, dst in copy.inputs}) == len(norm.tensors)
    for source, destination in copy.inputs:
        assert source.dtype == destination.dtype == torch.float32
        assert 6 * 3 * width <= source.numel() * 4 < 6 * 3 * width + 4
        assert source.shape == destination.shape and source.data_ptr() != destination.data_ptr()
        source.fill_(7)
        destination.zero_()
        assert torch.equal(rw_copy((source, destination)), source)
    copy.restore()
    assert torch.equal(copy.source, copy.saved_source)
    assert torch.equal(copy.destination, copy.saved_source)
    if regime != "hot":
        assert copy.source.numel() * 4 > 2 * norm.spec["l2_bytes"]


def test_norm_checks_clone_each_inplace_candidate_and_check_residual():
    x, residual, weight = torch.randn(3, 16), torch.randn(3, 16), torch.randn(16)
    inputs = (x, residual, weight)
    originals = tuple(t.clone() for t in inputs)

    def correct(tensors):
        out, summed = norm_reference(*tensors)
        tensors[0].copy_(out)
        tensors[1].copy_(summed)
        return tensors[:2]

    def bad_residual(tensors):
        out, summed = correct(tensors)
        summed.add_(1)
        return out, summed

    candidates = {
        "ours": correct,
        "aa": correct,
        "vllm": correct,
        "sglang": correct,
        "flashinfer": correct,
        "cute_bad": bad_residual,
    }
    result = check_norm_candidates(candidates, [inputs], False)
    assert not result["passed"] and not result["candidate_passed"]["cute_bad"]
    assert result["candidate_passed"]["ours"]
    assert result["bitwise_flashinfer"]["ours"]
    assert all(torch.equal(a, b) for a, b in zip(inputs, originals, strict=True))


@pytest.mark.parametrize(
    "op,rows,width,layers",
    [
        ("silu_mul", [1, 4, 16, 64, 256, 1024], 12288, 32),
        ("add_norm", [1, 2, 3, 4, 16, 64], 4096, 36),
        ("add_norm_sum", [1, 4, 16, 64, 256, 1024], 4096, 36),
        ("gemma_add_norm_sum", [1, 4, 16, 64, 256, 1024], 4096, 36),
        ("gemma_add_norm", [256, 512, 1024, 2048, 4096, 8192], 4096, 36),
    ],
)
def test_operation_defaults_preserve_activation_and_cover_norm_bands(
    monkeypatch, op, rows, width, layers
):
    from benchmark.kernel_tuning import norm_act_paired as bench

    calls = []

    def run(args, m, regime):
        calls.append((m, args.width, args.layers, regime))
        return {"M": m, "regime": regime, "dispatch": "test", "verdict": "PASS"}

    monkeypatch.setattr(bench, "pin_x925", lambda: None)
    monkeypatch.setattr(bench, "run_row", run)
    assert bench.main(["--op", op, "--sglang-root", ".", "--market-root", "."]) == 0
    regimes = ["hot", "cold"] if op == "silu_mul" else ["hot", "engine", "cold"]
    if op.endswith("_sum"):
        regimes = ["engine"]
    assert calls == [(m, width, layers, regime) for m in rows for regime in regimes]


def test_activation_rejects_norm_engine_regime():
    from benchmark.kernel_tuning import norm_act_paired as bench

    with pytest.raises(SystemExit, match="2"):
        bench.main(["--sglang-root", ".", "--regimes", "engine"])


@pytest.mark.parametrize("buffer", (0, 1))
@pytest.mark.parametrize("name", ("cute", "triton"))
def test_worse_real_market_cannot_relax_norm_gate_for_either_output(buffer, name):
    inputs = (
        torch.ones(1, 16, dtype=torch.bfloat16),
        torch.zeros(1, 16, dtype=torch.bfloat16),
        torch.ones(16, dtype=torch.bfloat16),
    )

    def rounded(tensors):
        exact, residual = norm_reference(*tensors)
        pair = [exact.to(tensors[0].dtype), residual]
        pair[buffer] = torch.nextafter(pair[buffer], torch.full_like(pair[buffer], float("inf")))
        return pair

    # A worse real market cannot override the better torch formulas, for either
    # normalized output or residual; qualification uses each metric's minimum.
    formulas_only = check_norm_candidates({name: rounded}, [inputs], False)
    assert not formulas_only["candidate_passed"][name]
    with_market = check_norm_candidates({name: rounded, "flashinfer": rounded}, [inputs], False)
    assert not with_market["candidate_passed"][name]
    assert "residual_metrics" in with_market


def test_fi07_numerics_are_compared_per_input_and_triton_is_gated():
    inputs = [
        (
            torch.ones(1, 16, dtype=torch.bfloat16),
            torch.zeros(1, 16, dtype=torch.bfloat16),
            torch.ones(16, dtype=torch.bfloat16),
        )
    ] * 2

    def correct(tensors):
        exact, residual = norm_reference(*tensors, zero_centered=True)
        return exact.to(tensors[0].dtype), residual

    child = check_norm_candidates({"flashinfer07": correct}, inputs, True, record_inputs=True)
    extra = [row["flashinfer07"] for row in child["per_input"]]
    result = check_norm_candidates({"triton": correct}, inputs, True, extra)
    assert result["candidate_passed"]["triton"]
    assert "flashinfer07" in result["markets"]
    # A good market on input 0 must not inherit the worse error on input 1.
    extra[0]["normalized"] = dict.fromkeys(extra[0]["normalized"], 0.0)
    assert not check_norm_candidates({"triton": correct}, inputs, True, extra)["passed"]
    with pytest.raises(ValueError, match="input count"):
        check_norm_candidates({"triton": correct}, inputs, True, extra[:1])


@pytest.mark.parametrize(
    "returncode,stdout,stderr,reason",
    [
        (1, "", "ModuleNotFoundError: No module named 'einops'", "native stderr omitted"),
        (0, "no receipt\n", "", "receipts=0"),
        (0, "ROW not-json\n", "", "JSONDecodeError"),
        (0, 'ROW {"us": {"flashinfer07": -1}}\n', "", "invalid"),
    ],
)
def test_fi07_child_failure_is_loud(
    monkeypatch, tmp_path, capsys, returncode, stdout, stderr, reason
):
    from types import SimpleNamespace

    from benchmark.kernel_tuning import norm_act_norm as norm_bench

    args = SimpleNamespace(
        fi07=tmp_path,
        op="gemma_add_norm",
        width=4096,
        dtype="bf16",
        layers=36,
        rounds=30,
        check=True,
    )
    monkeypatch.setattr(
        norm_bench.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr),
    )
    child = norm_bench.fi07_child(args, 256, "cold")
    assert child["status"] == "ERROR" and reason in child["error"]
    assert reason in capsys.readouterr().err
    receipt = {"verdict": "PASS"}
    norm_bench.apply_fi07_market(receipt, child, {}, True)
    assert receipt["verdict"] == "MARKET_ERROR"
    assert reason in receipt["flashinfer07_error"]


@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("fastest", ("vllm", "sglang", "flashinfer07"))
@pytest.mark.parametrize("parent_epsilon,child_epsilon", ((0.005, 0.02), (0.02, 0.005)))
def test_fi07_row_selects_market_median_and_uses_paired_child_ratio(
    zero, fastest, parent_epsilon, child_epsilon
):
    from benchmark.kernel_tuning.norm_act_norm import apply_fi07_market

    samples = {
        "ours": [2.0] * 30,
        "aa": [2.0] * 30,
        "vllm": [1.0, 4.0] * 15,
        "sglang": [4.0, 1.0] * 15,
    }
    receipt = paired_summary(samples)
    assert receipt["verdict"] == "FAIL"  # Legacy per-round minimum, before --fi07.
    receipt["epsilon"] = parent_epsilon
    child = {
        "us": {"flashinfer07": 1.0, "ours": 1.0},
        "ratios": {"flashinfer07": {"median": 1.2, "ci95": [1.1, 1.3]}},
        "ci95_us": [1.0, 1.0],
        "epsilon": child_epsilon,
    }
    selected = {"median": 1.0, "ci95": [0.985, 1.05]}
    if fastest == "flashinfer07":
        child["ratios"][fastest] = selected
    else:
        receipt["ratios"][fastest] = selected
    apply_fi07_market(receipt, child, samples, zero)
    assert receipt["flashinfer07"] is child
    assert receipt["flashinfer07_role"] == "market"
    assert receipt["us"]["flashinfer07"] == 1.0
    assert receipt["us"]["ours_fi07"] == 1.0
    assert receipt["ratios"]["flashinfer07"] == child["ratios"]["flashinfer07"]
    assert receipt["epsilon_fi07"] == child_epsilon
    assert receipt["epsilon"] == 0.02
    assert receipt["faster_ratio"] == selected["median"]
    assert receipt["faster_ci95"] == selected["ci95"]
    assert receipt["verdict"] == "PASS"
    assert "ours" not in receipt["flashinfer07_comparison"]["candidates"]


@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("numerics_fail", (False, True))
def test_fi07_paired_market_can_fail_row_without_overriding_numerics(zero, numerics_fail):
    from benchmark.kernel_tuning.norm_act_norm import apply_fi07_market

    samples = {name: [1.0] * 30 for name in ("ours", "aa", "vllm", "sglang")}
    receipt = paired_summary(samples)
    if numerics_fail:
        receipt["verdict"] = "NUMERICS_FAIL"
    child = {
        "us": {"flashinfer07": 4.0, "ours": 8.0},
        "ratios": {"flashinfer07": {"median": 0.5, "ci95": [0.49, 0.51]}},
        "epsilon": 0.005,
    }
    apply_fi07_market(receipt, child, samples, zero)
    assert receipt["faster_ratio"] == 0.5
    assert receipt["faster_ci95"] == [0.49, 0.51]
    assert receipt["verdict"] == ("NUMERICS_FAIL" if numerics_fail else "FAIL")


@pytest.mark.parametrize("op", ("add_norm", "gemma_add_norm"))
@pytest.mark.parametrize("regime", ("hot", "engine", "cold"))
@pytest.mark.parametrize("l2_bytes", (128, 144))
@pytest.mark.parametrize("static_weight", (0, 1))
def test_fi07_child_measures_auto_and_aa_with_shared_restore(
    monkeypatch, op, regime, l2_bytes, static_weight
):
    from types import SimpleNamespace

    from benchmark.kernel_tuning import norm_act_norm as norm_bench
    from benchmark.kernel_tuning import norm_act_paired as bench

    args = SimpleNamespace(
        op=op,
        dtype="bf16",
        width=8,
        layers=2,
        rounds=30,
        fi07=None,
        fi07_child=True,
        check=True,
        static_weight=static_weight,
    )
    props = SimpleNamespace(
        name="test",
        major=12,
        minor=0,
        multi_processor_count=1,
        L2_cache_size=l2_bytes,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **k: randn(*a, **(k | {"device": "cpu"})))
    monkeypatch.setattr(norm_bench, "flashinfer_metadata", lambda: {"version": "0.7.0.post1"})
    handles = []

    class Call:
        def __init__(self, zero, provider="auto", *, static_weight=True):
            assert zero == (op == "gemma_add_norm")
            self.zero = zero
            self.provider = provider
            self.static_weight = static_weight
            handles.append(self)

        def __call__(self, tensors):
            out, residual = norm_reference(*tensors, zero_centered=self.zero)
            tensors[0].copy_(out)
            tensors[1].copy_(residual)
            assert all(torch.isfinite(t).all() and t.abs().max() < 32 for t in tensors[:2])
            return tensors[:2]

    monkeypatch.setattr(norm_bench, "NormCall", Call)
    original = []

    def check(candidates, tensors, zero, extra, *, record_inputs):
        assert list(candidates) == ["flashinfer07"]  # Existing numerical checks.
        assert extra is None and record_inputs
        original.extend(tuple(t.clone() for t in triple) for triple in tensors)
        return {"passed": True}

    monkeypatch.setattr(norm_bench, "check_norm_candidates", check)
    captured = []

    def assert_restored(inputs):
        expected = original * args.layers if regime == "hot" else original
        for actual, saved in zip(inputs, expected, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(actual, saved, strict=True))

    def capture(fn, inputs, outputs, layers):
        assert_restored(inputs)
        assert outputs == [None] * len(inputs) and layers == args.layers
        assert len(inputs) % layers == 0
        unique_buffers = [{t[i].data_ptr(): t[i] for t in inputs} for i in range(3)]
        assert len(unique_buffers[0]) == (len(inputs) if regime == "cold" else 1)
        for buffers in unique_buffers[1:]:
            assert len(buffers) == (1 if regime == "hot" else len(inputs))
        if regime != "hot":
            rotating = unique_buffers[1:] if regime == "engine" else unique_buffers
            assert (
                sum(t.numel() * t.element_size() for buffers in rotating for t in buffers.values())
                > 2 * props.L2_cache_size
            )
        captured.append(fn)
        for tensors in inputs:
            fn(tensors)
        return fn, inputs

    def measure(banks, rounds, layers, before):
        assert list(banks) == ["ours", "aa", "flashinfer07"]
        assert rounds == 30 and layers == args.layers
        assert captured == handles
        assert [fn.provider for fn in handles] == ["auto", "auto", "flashinfer"]
        assert all(fn.static_weight == bool(static_weight) for fn in handles[:2])
        for name, (fn, inputs) in banks.items():
            before(name)
            assert_restored(inputs)
            for tensors in inputs:
                fn(tensors)
        return {
            "ours": [10.0, 40.0] * 15,
            "aa": [10.0, 40.0] * 15,
            "flashinfer07": [20.0] * 30,
        }

    monkeypatch.setattr(bench, "capture_bank", capture)
    monkeypatch.setattr(bench, "measure", measure)
    row = norm_bench.run_norm_row(args, 2, regime)
    assert row["regime"] == regime
    assert row["distinct_weights"] == len(original)
    unique_x = len(original) if regime == "cold" else 1
    assert row["pool_input_bytes"] == (unique_x * 2 + len(original) * 3) * args.width * 2
    assert row["graphs_per_candidate"] == (1 if regime == "hot" else len(original) // args.layers)
    assert row["us"] == {"ours": 25.0, "aa": 25.0, "flashinfer07": 20.0}
    # Median of within-round ratios differs from ratio of medians (20/25).
    assert row["ratios"]["flashinfer07"] == {"median": 1.25, "ci95": [0.5, 2.0]}
    assert row["ratios"]["aa"] == {"median": 1.0, "ci95": [1.0, 1.0]}
    assert row["epsilon"] == 0.005
    assert row["ci95_us"] == [20.0, 20.0]


def test_fi07_installer_only_adds_missing_python_dependencies(monkeypatch):
    from benchmark.kernel_tuning import norm_act_fi07 as install

    def version(name):
        if name == "einops":
            raise install.importlib.metadata.PackageNotFoundError(name)
        return "24.0"

    monkeypatch.setattr(install.importlib.metadata, "version", version)
    assert install.missing_pure_requirements(
        [
            "einops",
            "packaging>=24.2",
            "requests",
            "torch",
            "nvidia-cutlass-dsl>=4.6.2",
            'click; extra == "unused"',
            "nccl4py>=0.4.1",
        ]
    ) == ["einops", "packaging>=24.2"]


@pytest.mark.parametrize("regime", ("hot", "engine", "cold"))
def test_fi07_child_keeps_successful_receipt(monkeypatch, tmp_path, regime):
    import json
    from types import SimpleNamespace

    from benchmark.kernel_tuning import norm_act_norm as norm_bench

    args = SimpleNamespace(
        fi07=tmp_path,
        op="gemma_add_norm",
        width=4096,
        dtype="bf16",
        layers=36,
        rounds=30,
        check=True,
    )
    receipt = {
        "us": {"flashinfer07": 2.0, "ours": 1.0, "aa": 1.0},
        "ratios": {"flashinfer07": {"median": 2.0, "ci95": [1.9, 2.1]}},
        "epsilon": 0.005,
        "ci95_us": [1.9, 2.1],
        "providers": {"flashinfer07": {"version": "0.7.0.post1"}},
        "check": {"per_input": [{"flashinfer07": {"normalized": {}, "residual": {}}}]},
    }

    def run(command, **kwargs):
        assert "--fi07-child" in command and "--check" in command
        assert command[command.index("--regimes") + 1] == regime
        assert str(tmp_path) in command
        assert kwargs["capture_output"] and not kwargs["check"]
        return SimpleNamespace(returncode=0, stdout="info\nROW " + json.dumps(receipt), stderr="")

    monkeypatch.setattr(norm_bench.subprocess, "run", run)
    assert norm_bench.fi07_child(args, 256, regime) == receipt


@pytest.mark.parametrize("zero", (False, True))
def test_fi07_missing_market_cannot_qualify_row_or_exit_success(
    monkeypatch, tmp_path, capsys, zero
):
    from benchmark.kernel_tuning import norm_act_paired as bench
    from benchmark.kernel_tuning.norm_act_norm import apply_fi07_market

    (tmp_path / "flashinfer").mkdir()
    receipt = {
        "M": 256,
        "regime": "hot",
        "dispatch": "cute",
        "verdict": "PASS",
        "candidate_verdicts": {"cute_good": "PASS", "cute_bad": "NUMERICS_FAIL"},
    }
    apply_fi07_market(receipt, {"status": "ERROR", "error": "missing dependency"}, {}, zero)
    assert receipt["candidate_verdicts"] == {
        "cute_good": "MARKET_ERROR" if zero else "PASS",
        "cute_bad": "NUMERICS_FAIL",
    }
    monkeypatch.setattr(bench, "pin_x925", lambda: None)
    monkeypatch.setattr(bench, "run_row", lambda *a: receipt)
    assert (
        bench.main(
            [
                "--op",
                "gemma_add_norm" if zero else "add_norm",
                "--rows",
                "256",
                "--regimes",
                "hot",
                "--fi07",
                str(tmp_path),
                "--market-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    stdout = capsys.readouterr().out
    assert '"flashinfer07_error":"missing dependency"' in stdout
    assert "flashinfer07_us" in stdout
    assert "ours_fi07_us" in stdout


@pytest.mark.parametrize("fused", (False, True))
def test_sum_pair_observes_pre_norm_values_and_only_writes_aux_slice(fused):
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall

    class Norm:
        zero_centered = True
        provider = "test"

        def op(self, x, residual, weight, eps, *, zero_centered, sum_out):
            assert zero_centered and eps == 1e-6
            torch.add(x, residual, out=sum_out)
            return self((x, residual, weight))

        def __call__(self, tensors):
            x, residual, _ = tensors
            residual.add_(x)
            x.fill_(-5)
            return x, residual

    x, residual, weight = torch.randn(4, 16), torch.randn(4, 16), torch.randn(16)
    expected = x + residual
    aux = torch.full((4, 48), 17.0)
    summed = aux[:, 16:32]
    call = SumNormCall(Norm(), fused=fused)
    result = call((x, residual, weight, summed))
    assert result[0] is x and result[1] is residual
    assert torch.equal(summed, expected) and torch.equal(residual, expected)
    assert (aux[:, :16] == 17).all() and (aux[:, 32:] == 17).all()


def test_sum_pair_reports_unfused_within_round_ratio():
    samples = {
        "ours": [1.0, 4.0] * 15,
        "aa": [1.0, 4.0] * 15,
        "unfused": [2.0] * 30,
        "vllm": [3.0] * 30,
        "flashinfer": [4.0] * 30,
    }
    result = paired_summary(samples, markets=("vllm", "flashinfer"))
    assert result["ratios"]["unfused"] == {"median": 1.25, "ci95": [0.5, 2.0]}


@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("static_weight", (0, 1))
def test_sum_engine_pair_restores_shared_inputs_outside_graphs(zero, static_weight):
    from types import SimpleNamespace

    from benchmark.kernel_tuning.norm_act_paired import SumNormCall
    from benchmark.kernel_tuning.norm_act_pools import build_norm_pools, pool_spec

    args = SimpleNamespace(
        op="gemma_add_norm_sum" if zero else "add_norm_sum",
        width=16,
        dtype="bf16",
        layers=2,
        static_weight=static_weight,
    )
    pools = build_norm_pools(pool_spec(args, 4, "engine", 512), device="cpu")

    class Call:
        zero_centered = zero

        def op(self, x, residual, weight, eps, *, zero_centered, sum_out):
            torch.add(x, residual, out=sum_out)
            return self((x, residual, weight))

        def __call__(self, tensors):
            out, residual = norm_reference(*tensors, zero_centered=zero)
            tensors[0].copy_(out)
            tensors[1].copy_(residual)
            return tensors[:2]

    for fused in (True, False, True):
        pools.restore()
        assert torch.equal(pools.x, pools.saved_x)
        assert torch.equal(pools.residual, pools.saved_r)
        fn = SumNormCall(Call(), fused=fused)
        for triple in pools.inputs:
            expected = triple[0] + triple[1]
            fn(triple)
            assert torch.equal(triple[3], expected)
        assert (pools.aux[:, :, :16] == 17).all()
        assert (pools.aux[:, :, 32:] == 17).all()
    assert pools.input_bytes > 1024


@pytest.mark.parametrize("rows", (1, 3))
def test_capture_warmup_preserves_pool_views_and_input_values(rows):
    from benchmark.kernel_tuning.norm_act_paired import warm_capture_call
    from benchmark.kernel_tuning.norm_act_pools import clone_norm_tensors

    # Nonzero pool offsets and singleton strides are lost by ordinary clone().
    x = torch.arange(8 * rows * 16).reshape(8, rows, 16)[3]
    residual = torch.ones_like(x)
    weight = torch.ones(8, 16)[5]
    aux = torch.full((8, rows, 48), 17)[3, :, 16:32]
    tensors = (x, residual, weight, aux)
    saved = tuple(t.clone() for t in tensors)
    clones = clone_norm_tensors(tensors)
    for actual, private in zip(tensors, clones, strict=True):
        assert private.shape == actual.shape and private.stride() == actual.stride()
        assert private.storage_offset() == actual.storage_offset()
        assert private.data_ptr() % 16 == actual.data_ptr() % 16
        assert private.untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
        assert torch.equal(private, actual)
    seen = []

    class Call:
        def call_index(self, private, out, index):
            seen.append(index)
            for actual, warm in zip(tensors, private, strict=True):
                assert actual.stride() == warm.stride()
                assert actual.storage_offset() == warm.storage_offset()
            private[0].zero_()
            private[3].zero_()

    warm_capture_call(Call(), [tensors, tensors], [None, None])
    assert seen == [0, 1]
    assert all(torch.equal(t, before) for t, before in zip(tensors, saved, strict=True))


def test_capture_guard_rejects_resolution_before_provider_setup(monkeypatch):
    from benchmark.kernel_tuning.norm_act_norm import NormCall
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall, cached_capture_plans

    call = NormCall(False)
    calls = []
    monkeypatch.setattr(
        call.op, "_resolve", lambda *args: (calls.append(1) or "cute", lambda *a: None)
    )
    tensors = (torch.ones(1, 16), torch.ones(1, 16), torch.ones(16))
    call(tensors)
    with cached_capture_plans(SumNormCall(call, fused=True)):
        call(tensors)
        with pytest.raises(RuntimeError, match="resolution cache miss"):
            call(tuple(t.double() for t in tensors))
    assert calls == [1]
    assert not call.op._require_cached_plan


def test_capture_bank_warms_all_views_on_capture_stream_before_graph(monkeypatch):
    import contextlib

    from benchmark.kernel_tuning.norm_act_norm import NormCall
    from benchmark.kernel_tuning.norm_act_paired import capture_bank

    state = {"stream": None, "capturing": False}
    warmed = set()
    launches = []

    class Stream:
        def wait_stream(self, other):
            assert not state["capturing"]

        def synchronize(self):
            assert not state["capturing"]

    @contextlib.contextmanager
    def stream_context(stream):
        state["stream"] = stream
        yield
        state["stream"] = None

    @contextlib.contextmanager
    def graph_context(graph, *, stream, pool):
        assert state["stream"] is stream
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            assert state["capturing"]

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    op = NormCall(False)

    def resolve(*args):
        assert not state["capturing"], "resolution performed CUDA setup in capture"

        def launch(x, r, w, eps, summed):
            key = (state["stream"], x.stride(), x.storage_offset(), r.storage_offset())
            if state["capturing"]:
                assert key in warmed, "lazy launch initialization inside capture"
                assert op.op._require_cached_plan
                launches.append(key)
            else:
                warmed.add(key)
            x.zero_()

        return "cute", launch

    monkeypatch.setattr(op.op, "_resolve", resolve)
    # Reuse the same candidate for hot and then engine, with fresh capture streams.
    for count in (1, 4):
        x, r, w = torch.ones(1, 1, 16), torch.ones(count, 1, 16), torch.ones(count, 16)
        inputs = [(x[0], r[i], w[i]) for i in range(count)]
        bank = capture_bank(op, inputs * 2, [None] * (2 * count), 2)
        assert len(bank[0]) == count
    assert len(launches) == 10
    assert not op.op._require_cached_plan


@pytest.mark.parametrize("op", ["add_norm", "gemma_add_norm"])
def test_fp32_residual_cli_and_market_ratio(op):
    from benchmark.kernel_tuning.norm_act_paired import parse_args

    args = parse_args(
        ["--op", op, "--market-root", ".", "--residual-dtype", "fp32", "--width", "2048"]
    )
    assert args.width == 2048 and args.residual_dtype == "fp32" and args.static_weight == 0
    assert parse_args(["--op", op, "--market-root", "."]).residual_dtype == "same"
    result = paired_summary(
        {
            "ours": [2.0] * 30,
            "aa": [2.0] * 30,
            "vllm": [3.0] * 30,
            "sglang": [4.0] * 30,
            "sgl_fp32": [1.0] * 30,
        }
    )
    assert result["faster_ratio"] == 1.5  # Native FP32 is diagnostic, not the acceptance market.
    assert result["ratios"]["sgl_fp32"]["median"] == 0.5


@pytest.mark.parametrize("zero", [False, True])
def test_fp32_residual_native_reference_and_gate(zero):
    from benchmark.kernel_tuning.norm_act_reference import fp32_residual_formulas

    x = torch.randn(3, 64, dtype=torch.bfloat16)
    residual = torch.randn(3, 64)
    weight = torch.randn(64, dtype=x.dtype)
    originals = [t.clone() for t in (x, residual, weight)]
    formulas = fp32_residual_formulas(x, residual, weight, zero_centered=zero)
    eager, native = formulas["torch_fp32"], formulas["sgl_fp32"]
    assert torch.equal(eager[0], native[0])
    assert torch.equal(native[1], x.float() + residual)
    assert native[1].data_ptr() not in (residual.data_ptr(), eager[1].data_ptr())
    for before, after in zip(originals, (x, residual, weight), strict=True):
        assert torch.equal(before, after)
    result = check_norm_candidates(
        {"ours": lambda ts: fp32_residual_formulas(*ts, zero_centered=zero)["sgl_fp32"]},
        [(x, residual, weight)],
        zero,
    )
    assert result["passed"] and result["markets"] == ["torch_fp32", "sgl_fp32"]
