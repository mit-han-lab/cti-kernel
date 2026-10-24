# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only parity statistics, SOL records, weights and startup protocol checks."""

import json
import random
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import nvfp4_parity as parity

CPUINFO = """processor : 7
CPU part : 0xd87

processor : 4
CPU part : 0xd85

CPU part : 0xD85
processor : 2

processor : 9
model name : other
"""


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], "dispatch"),
        (["--provider", "dispatch"], "dispatch"),
        (["--provider", "flashinfer"], "flashinfer"),
        (["--provider", "dispatch", "--candidates"], "dispatch"),
    ],
)
def test_provider_option(flags, expected):
    assert parity._parser().parse_args(flags).provider == expected


def test_provider_option_rejects_unknown():
    with pytest.raises(SystemExit) as error:
        parity._parser().parse_args(["--provider", "unknown"])
    assert error.value.code == 2


def test_flashinfer_rejects_candidates_before_setup(monkeypatch, capsys):
    monkeypatch.setattr(parity.sys, "argv", ["parity", "--provider", "flashinfer", "--candidates"])

    def unexpected(*args):
        pytest.fail("invalid provider/candidates combination reached setup")

    monkeypatch.setattr(parity.importlib.metadata, "version", unexpected)
    monkeypatch.setattr(parity, "_device_name", unexpected)
    with pytest.raises(SystemExit) as error:
        parity.main()
    assert error.value.code == 2
    assert "--candidates cannot be combined with --provider flashinfer" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["dispatch", "flashinfer"])
@pytest.mark.parametrize("kind", parity.OPS)
@pytest.mark.parametrize("name", parity.SHAPES)
@pytest.mark.parametrize("comparator", ["sglang", "vllm", "vllm-eager"])
def test_operations_construct_and_force_every_copy(monkeypatch, provider, kind, name, comparator):
    from ctikernels.linear import nvfp4
    from ctikernels.linear._nvfp4_dispatch import UNFUSED, FlashInfer

    scale, original_plan = object(), object()
    constructed, references = [], []

    def operation(*args, **kwargs):
        constructed.append((args, kwargs))
        return SimpleNamespace(
            provider=kwargs.get("provider", "quack"), plan_for=lambda m: original_plan
        )

    def reference(*args, **kwargs):
        references.append((args, kwargs))
        return object()

    def comparators(selected):
        assert selected == comparator
        return reference, reference

    def unexpected(*args, **kwargs):
        pytest.fail("constructed the wrong operation kind")

    monkeypatch.setattr(nvfp4, "NVFP4Linear", operation if kind == "linear" else unexpected)
    monkeypatch.setattr(nvfp4, "NVFP4GateUpSiLU", operation if kind == "gated" else unexpected)
    monkeypatch.setattr(parity, "_comparator_classes", comparators)
    monkeypatch.setattr(parity.torch, "full", lambda *args, **kwargs: scale)
    monkeypatch.setattr(
        parity,
        "_weight",
        lambda *args: nvfp4.NVFP4Weight(object(), object(), parity.torch.tensor([0.5])),
    )
    h, i = parity.SHAPES[name]
    packed_bytes = h * i // 2 if kind == "linear" else h * i
    ours, refs, cold_bytes = parity._operations(
        kind, name, 2 * packed_bytes + 1, comparator, provider
    )
    assert len(ours) == len(refs) == len(constructed) == len(references) == 3
    assert cold_bytes == 3 * packed_bytes
    for op, (args, kwargs), (ref_args, ref_kwargs) in zip(
        ours, constructed, references, strict=True
    ):
        options = {"provider": "flashinfer"} if provider == "flashinfer" else {}
        if kind == "linear":
            assert kwargs == options
            assert args[1] is scale
            assert ref_args == ((args[0],), scale)
            assert ref_kwargs == {}
        else:
            assert kwargs == {"input_scale": scale, "output_scale": scale, **options}
            gate, up_ours = args
            market_gate, up_market, *ref_scales = ref_args
            assert market_gate is gate
            assert up_ours.global_scale.data_ptr() != gate.global_scale.data_ptr()
            assert up_ours.global_scale.item() == 1.25 * gate.global_scale.item()
            assert up_market.global_scale is gate.global_scale
            assert up_ours.data is up_market.data
            assert up_ours.scale is up_market.scale
            assert ref_scales == [scale, scale]
            assert ref_kwargs == {"shared": name == "35b-shared"}
        for m in parity.BATCHES:
            if provider == "dispatch":
                assert op.plan_for(m) is original_plan
            else:
                assert op.plan_for(m) == (
                    FlashInfer() if kind == "linear" else (UNFUSED, FlashInfer())
                )
                metadata = parity._dispatch_metadata(op, m, gated=kind == "gated")
                assert metadata["provider"] == metadata["dispatched_provider"] == "flashinfer"


def test_bootstrap_is_deterministic_and_leaves_global_rng_alone():
    ratios = [0.91, 1.01, 1.02, 1.05, 1.08, 1.11]
    state = random.getstate()
    first = parity.bootstrap_ci(ratios)
    assert parity.bootstrap_ci(ratios) == first
    assert random.getstate() == state


@pytest.mark.parametrize(
    ("ratios", "expected"),
    [([1.25] * 60, (1.25, 1.25)), ([1, 2, 3], (1, 3)), ([1, 2], (1, 2))],
)
def test_bootstrap_known_answers(ratios, expected):
    assert parity.bootstrap_ci(ratios) == expected


def test_bootstrap_resamples_ratios_and_interpolates_percentiles(monkeypatch):
    class Resampler:
        def __init__(self, seed):
            assert seed == 0
            self.calls = 0

        def choices(self, ratios, *, k):
            assert ratios == [1, 2, 3] and k == 3
            self.calls += 1
            return [self.calls] * k

    rng = Resampler(0)
    monkeypatch.setattr(parity.random, "Random", lambda seed: rng)
    assert parity.bootstrap_ci([1, 2, 3]) == pytest.approx((50.975, 1950.025))
    assert rng.calls == 2000


@pytest.mark.parametrize("ratios", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_bootstrap_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        parity.bootstrap_ci(ratios)


def test_cpuinfo_selects_x925_logical_ids():
    assert parity.x925_cpus(CPUINFO) == [2, 4]
    assert parity.x925_cpus("processor: 3\nCPU part: 0xd85") == [3]
    assert parity.x925_cpus("processor: 0\nCPU part: 0xd87") == []
    assert parity.x925_cpus("") == []


@pytest.mark.parametrize(
    ("device", "flags", "pairs", "pinned"),
    [
        ("NVIDIA GB10", [], 60, True),
        ("NVIDIA GB10", ["--pairs", "30"], 60, True),
        ("NVIDIA GB10", ["--pairs", "80"], 80, True),
        ("NVIDIA RTX 5090", [], 20, False),
        ("NVIDIA RTX 5090", ["--pairs", "80"], 80, False),
        ("Other GPU", ["--pin-x925"], 20, True),
    ],
)
def test_device_selects_pair_count_and_affinity(monkeypatch, device, flags, pairs, pinned):
    monkeypatch.setattr(parity, "_device_name", lambda: device)
    monkeypatch.setattr(parity.Path, "read_text", lambda path: CPUINFO)
    calls = []
    monkeypatch.setattr(parity.os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, cpus)))
    args = parity._parser().parse_args(flags)
    name, cpus = parity._timing_setup(args)
    assert name == device
    assert args.pairs == pairs
    assert args.pin_x925 is pinned
    assert cpus == ([2, 4] if pinned else [])
    assert calls == ([(0, [2, 4])] if pinned else [])


def test_gb10_without_x925_fails_before_cuda_or_comparator_load(monkeypatch):
    monkeypatch.setattr(parity, "_device_name", lambda: "NVIDIA GB10")
    monkeypatch.setattr(parity.Path, "read_text", lambda path: "CPU part: 0xd87\n")
    monkeypatch.setattr(parity.importlib.metadata, "version", lambda name: "0.7.1")
    monkeypatch.setattr(parity.sys, "argv", ["parity", "--rows", "linear/9b/1"])

    def unexpected(*args):
        pytest.fail("startup proceeded without X925 pinning")

    from benchmark.kernel_tuning import nvfp4_sglang

    monkeypatch.setattr(nvfp4_sglang, "load", unexpected)
    monkeypatch.setattr(parity.torch.cuda, "get_device_properties", unexpected)
    monkeypatch.setattr(parity.os, "sched_setaffinity", unexpected)
    with pytest.raises(RuntimeError, match="no Cortex-X925 CPUs"):
        parity.main()


def test_affinity_failure_is_not_ignored(monkeypatch):
    monkeypatch.setattr(parity.Path, "read_text", lambda path: CPUINFO)

    def denied(pid, cpus):
        raise OSError("affinity denied")

    monkeypatch.setattr(parity.os, "sched_setaffinity", denied)
    with pytest.raises(OSError, match="affinity denied"):
        parity._pin_x925()


def test_device_discovery_is_isolated(monkeypatch):
    def discover(command, **kwargs):
        assert command[:2] == [parity.sys.executable, "-c"]
        assert "torch.cuda.get_device_name(0)" in command[2]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=json.dumps("NVIDIA GB10"))

    monkeypatch.setattr(parity.subprocess, "run", discover)
    assert parity._device_name() == "NVIDIA GB10"


@pytest.mark.parametrize("comparator", ["sglang", "vllm", "vllm-eager"])
def test_statistics_use_median_of_paired_ratios(comparator):
    result = parity.parity_statistics(
        [1, 10, 100], [2, 40, 100], [1] * 3, [1] * 3, comparator=comparator
    )
    assert result["speedup"] == 2  # Ratio of medians would be 4.
    assert result["speedup_ci_low"] == 1
    assert result["speedup_ci_high"] == 4
    assert result[f"{comparator}_us"] == 40
    assert result["pairs"] == 3


@pytest.mark.parametrize(
    ("device", "ratio", "noise", "verdict"),
    [
        ("NVIDIA GB10", 1.01, 0, "pass"),
        ("NVIDIA GB10", 1.01, 0.1, "pass"),
        ("NVIDIA GB10", 1, 0, "fail"),
        ("NVIDIA GB10", 0.99, 0.1, "fail"),
        ("NVIDIA RTX 5090", 1, 0, "pass"),
        ("NVIDIA RTX 5090", 0.99, 0.015, "pass"),
        ("NVIDIA RTX 5090", 0.98, 0.015, "fail"),
        ("NVIDIA RTX 5090", 1.2, 0.03, "inconclusive"),
    ],
)
def test_device_verdict_rules(device, ratio, noise, verdict):
    result = parity.parity_statistics(
        [1] * 60, [ratio] * 60, [1 + noise] * 60, [1] * 60, device_name=device
    )
    assert result["verdict"] == verdict
    assert result["eps"] == pytest.approx(noise)
    assert result["speedup_ci_low"] == ratio == result["speedup_ci_high"]


@pytest.mark.parametrize(
    ("pairs", "ours", "noise", "tie"),
    [
        (60, [1] * 60, 0, "pass"),
        (60, [1] * 31 + [1.004] * 29, 0.01, "pass"),  # median 1.00, CI_low >= 1 - eps
        (60, [1.004] * 60, 0.01, "fail"),  # median below 1.00
        (60, [1] * 31 + [1.004] * 29, 0, "fail"),  # CI_low below 1 - eps
        (20, [1] * 20, 0, "insufficient-pairs"),
    ],
)
def test_identical_kernel_tie_rule(pairs, ours, noise, tie):
    result = parity.parity_statistics(
        ours, [1] * pairs, [1 + noise] * pairs, [1] * pairs, device_name="NVIDIA GB10",
        identical_kernel=True,
    )  # fmt: skip
    assert result["tie_verdict"] == tie
    assert result["acceptance"] == ("pass" if tie == "pass" else "fail")


@pytest.mark.parametrize("identical", [None, False])
def test_tie_rule_needs_identical_kernels(identical):
    tie = [1] * 31 + [1.004] * 29  # passes the tie rule when the kernels are identical
    result = parity.parity_statistics(
        tie, [1] * 60, [1.01] * 60, [1] * 60, device_name="NVIDIA GB10", identical_kernel=identical
    )
    assert result["tie_verdict"] == (
        "kernel-identity-unknown" if identical is None else "not-identical-kernel"
    )
    assert result["acceptance"] == result["verdict"] == "fail"


@pytest.mark.parametrize("identical", [None, False, True])
def test_different_kernel_win_needs_ci_low_above_one(identical):
    # eps 0.1 makes the RTX 5090 verdict inconclusive; CI_low > 1 at 60 pairs still accepts.
    result = parity.parity_statistics(
        [1] * 60, [1.5] * 60, [1.1] * 60, [1] * 60, device_name="NVIDIA RTX 5090",
        identical_kernel=identical,
    )  # fmt: skip
    assert result["verdict"] == "inconclusive" and result["win_ci_low_above_1"]
    assert result["acceptance"] == "pass"
    short = parity.parity_statistics(
        [1] * 20, [1.5] * 20, [1.1] * 20, [1] * 20, device_name="NVIDIA RTX 5090",
        identical_kernel=identical,
    )  # fmt: skip
    assert not short["win_ci_low_above_1"] and short["acceptance"] != "pass"


def test_identical_gemm_compares_gemm_kernel_names():
    fi = "void cutlass::device_kernel<flashinfer::gemm::Fp4Gemm<128>>"
    assert parity.identical_gemm(["quantize", fi], ["other_quantize", fi])
    assert not parity.identical_gemm(["quantize", fi], ["quantize", "kernel_cutlass_GemmSm120"])
    assert not parity.identical_gemm(["kernel_cutlass_small"], ["kernel_cutlass_small"])


def test_gb10_nominal_win_with_ci_crossing_one_fails():
    result = parity.parity_statistics(
        [1] * 60, [0.9] * 29 + [1.1] * 31, [1] * 60, [1] * 60, device_name="GB10"
    )
    assert result["speedup"] > 1
    assert result["speedup_ci_low"] < 1 < result["speedup_ci_high"]
    assert result["verdict"] == "fail"


def test_candidate_remeasurement_preserves_larger_pair_count(monkeypatch):
    counts = []

    def sample(graphs, pairs, replays, deadline):
        counts.append(pairs)
        return [[1] * pairs, [1.1] * pairs, [1] * pairs]

    monkeypatch.setattr(parity, "_round_robin", sample)
    times, noise = parity._measure_candidates([None] * 3, 80, 5)
    assert counts == [80, 80]
    assert all(len(samples) == 80 for samples in times)
    assert noise["remeasured"]


@pytest.mark.parametrize("provider", ["dispatch", "flashinfer"])
def test_main_pins_before_comparator_loading_and_cuda(monkeypatch, capsys, provider):
    from benchmark.kernel_tuning import nvfp4_sglang

    events = []
    monkeypatch.setattr(
        parity.sys, "argv", ["parity", "--rows", "linear/9b/1", "--provider", provider]
    )
    monkeypatch.setattr(parity.importlib.metadata, "version", lambda name: "0.7.1")
    monkeypatch.setattr(parity, "_device_name", lambda: "NVIDIA GB10")
    monkeypatch.setattr(parity.Path, "read_text", lambda path: CPUINFO)
    monkeypatch.setattr(parity.os, "sched_setaffinity", lambda *args: events.append("pin"))
    monkeypatch.setattr(nvfp4_sglang, "load", lambda *args: events.append("load"))

    def properties(index):
        assert events == ["pin", "load"]
        events.append("cuda")
        return SimpleNamespace(L2_cache_size=1024)

    monkeypatch.setattr(parity.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(parity.torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(parity.torch.cuda, "empty_cache", lambda: None)

    def operations(kind, name, rotation, comparator, selected_provider):
        assert comparator == "sglang"
        assert selected_provider == provider
        return [], [], 2049

    monkeypatch.setattr(parity, "_operations", operations)

    def row(kind, name, m, ours, reference, cold, device, pairs, *args, **kwargs):
        assert pairs == 60
        assert device["x925_cpus"] == [2, 4]
        assert events == ["pin", "load", "cuda"]
        return parity.parity_statistics(
            [1] * pairs,
            [1.1] * pairs,
            [1] * pairs,
            [1] * pairs,
            device_name=device["device"],
        )

    monkeypatch.setattr(parity, "_row", row)
    parity.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("PARITY ")
    records = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert all(record["x925_cpus"] == [2, 4] for record in records)
    assert all(record["provider_mode"] == provider for record in records)
    assert records[0]["L2_bytes"] == 1024
    assert records[0]["rotation_target_bytes"] == 2049
    assert records[0]["speedup_ci_low"] == records[0]["speedup_ci_high"] == 1.1
    assert records[0]["verdict"] == "pass"


def test_candidate_records_include_device_verdict_and_ci(monkeypatch, capsys):
    monkeypatch.setattr(parity.torch, "randn", lambda *args, **kwargs: SimpleNamespace(device=0))
    monkeypatch.setattr(parity, "candidate_plans", lambda *args: [("quack", None, (), True)])
    monkeypatch.setattr(parity, "_cute_plans", lambda *args: (None, ()))
    monkeypatch.setattr(parity, "_forced", lambda *args: None)
    monkeypatch.setattr(parity, "kernels_per_call", lambda *args: 1)
    monkeypatch.setattr(parity, "_timed_graph", lambda calls: (None, len(calls)))
    # A nominal win with a CI crossing 1 must fail even though eps is zero.
    monkeypatch.setattr(
        parity,
        "_round_robin",
        lambda *args: [[1] * 60, [0.9] * 29 + [1.1] * 31, [0.9] * 29 + [1.1] * 31],
    )
    device = {
        "device": "NVIDIA GB10",
        "capability": [12, 1],
        "flashinfer_version": "0.7.1",
        "x925_cpus": [2, 4],
        "provider_mode": "dispatch",
    }
    ours = SimpleNamespace(plan_for=lambda m: ())
    reference = SimpleNamespace(probe=lambda x: None, metadata=lambda m: {})
    parity._candidate_row("linear", "9b", 1, [ours], [reference], 2049, device, 60, 5, None)
    record = json.loads(capsys.readouterr().out.removeprefix("CAND "))
    assert record["speedup"] == 1.1
    assert record["speedup_ci_low"] < 1 < record["speedup_ci_high"]
    assert record["verdict"] == "fail"
    assert record["pairs"] == record["samples"] == 60
    assert record["x925_cpus"] == [2, 4]
    assert record["provider_mode"] == "dispatch"
    assert record["sol"] is None
    assert record["sol_note"] == "vendor-dispatched row: no SOL gate"


@pytest.mark.parametrize("comparator", ["sglang", "vllm", "vllm-eager"])
def test_gated_weights_preserve_bytes_and_use_independent_globals(comparator):
    from ctikernels.linear.nvfp4 import NVFP4Weight

    def weight(value):
        return NVFP4Weight(
            parity.torch.zeros((32, 32), dtype=parity.torch.uint8),
            parity.torch.zeros((32, 4), dtype=parity.torch.float8_e4m3fn),
            parity.torch.tensor([value], dtype=parity.torch.float32),
        )

    gate, up = weight(0.5), weight(0.75)
    ours_up, market_up = parity._gated_weights(gate, up)
    assert ours_up.global_scale.shape == (1,)
    assert ours_up.global_scale.dtype == parity.torch.float32
    assert ours_up.global_scale.item() == 0.625
    assert ours_up.global_scale.data_ptr() not in (
        gate.global_scale.data_ptr(),
        up.global_scale.data_ptr(),
    )
    assert market_up.global_scale is gate.global_scale
    assert gate.global_scale.item() == 0.5
    assert up.global_scale.item() == 0.75
    assert ours_up.data is market_up.data is up.data
    assert ours_up.scale is market_up.scale is up.scale
    assert parity._gated_metadata("gated", comparator) == {
        "ours_gate_up_globals": "independent-unequal",
        f"{comparator}_gate_up_globals": "collapsed-to-max (same kernels as unequal)",
    }
    assert parity._gated_metadata("linear", comparator) == {}


@pytest.mark.parametrize("kind", parity.OPS)
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    ("device", "ceilings"),
    [
        ("NVIDIA GeForce RTX 5090", "rtx5090"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "rtx-pro-6000"),
        ("NVIDIA GB10", "gb10"),
    ],
)
def test_cute_sol_uses_shared_ceilings(monkeypatch, kind, pdl, device, ceilings):
    from ctikernels.linear._nvfp4_dispatch import Cute

    estimate = parity.sol.estimate
    calls = []

    def capture(call, source, **kwargs):
        calls.append(call)
        return estimate(call, source, **kwargs)

    monkeypatch.setattr(parity.sol, "estimate", capture)
    record = parity._sol_record(kind, 4, 5120, 17408, Cute(pdl=pdl), device, 80)
    assert record["sol_ceilings"] == f"{ceilings}.json"
    assert record["sol_us"] > 0
    assert record["sol_fraction"] == record["sol_us"] / 80
    assert record["sol_binding_resource"] == record["sol"]["limiting_resource"]
    assert record["sol"]["estimate_type"] == "useful"
    assert len(calls) == len(calls[0].kernels) == len(record["sol"]["per_kernel"]) == 1
    kernel = calls[0].kernels[0]
    assert kernel.cache_state == "cold" and not kernel.l2_resident_bytes
    assert kernel.dram_mode == "read" and kernel.pdl is pdl
    n, k = (34816, 5120) if kind == "gated" else (5120, 17408)
    assert kernel.dram_bytes == {
        "weights": n * k // 2,
        "scales": n * k // 16 + (16 + 272 * 512 if kind == "gated" else 8),
        "activations": 8 * k,
        "outputs": 34816 if kind == "gated" else 40960,
    }
    assert kernel.flops == {"mxf4nvf4_m16n8k64": 8 * n * k}
    table = parity.sol.load_ceilings(ceilings)
    floor = table["fixed_costs"]["graph_node_pdl_us" if pdl else "graph_node_us"]
    assert record["sol"]["per_kernel"][0]["terms_us"]["floor"] == floor
    expected = estimate(calls[0], table)
    assert record["sol_us"] == expected.t_ref_us
    assert record["sol_us"] >= floor


@pytest.mark.parametrize("provider", ["flashinfer", "quack", "triton", "dequant"])
def test_vendor_sol_has_no_gate_or_ceiling_lookup(monkeypatch, provider):
    from ctikernels.linear._nvfp4_dispatch import FlashInfer, Small
    from ctikernels.linear._nvfp4_quack import DEFAULT

    def unexpected(*args, **kwargs):
        pytest.fail("vendor row tried to estimate SOL")

    monkeypatch.setattr(parity.sol, "estimate", unexpected)
    config = {
        "flashinfer": FlashInfer(),
        "quack": DEFAULT,
        "triton": Small(1, 128, 256, 4, 2, True),
        "dequant": None,
    }[provider]
    assert parity._sol_record("gated", 4, 5120, 17408, config, "RTX 5090", 80) == {
        "sol": None,
        "sol_note": "vendor-dispatched row: no SOL gate",
    }


def test_cute_sol_missing_ceiling_reports_reason(monkeypatch):
    from ctikernels.linear._nvfp4_dispatch import Cute

    record = parity._sol_record("linear", 1, 5120, 17408, Cute(), "unknown", 80)
    assert record == {"sol": None, "sol_note": "no shared SOL ceilings for device"}
    table = parity.sol.load_ceilings("rtx5090")
    del table["tensor_core"]["mxf4nvf4_m16n8k64"]
    table["missing_measurements"] = ["tensor_core.mxf4nvf4_m16n8k64"]
    load = parity.sol.load_ceilings
    monkeypatch.setattr(parity.sol, "load_ceilings", lambda source: load(table))
    record = parity._sol_record("linear", 1, 5120, 17408, Cute(), "RTX 5090", 80)
    assert record["sol"] is None
    assert record["sol_ceilings"] == "rtx5090.json"
    assert "Missing ceiling key: tensor_core.mxf4nvf4_m16n8k64" in record["sol_note"]
    assert "sol_us" not in record and "sol_fraction" not in record


@pytest.mark.parametrize("kind", parity.OPS)
@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
def test_row_reports_actual_plan_sol_and_profiled_counts(monkeypatch, kind, provider):
    from ctikernels.linear._nvfp4_dispatch import CUTE, UNFUSED, Cute, FlashInfer

    config = Cute() if provider == "cute" else FlashInfer()
    path = CUTE if provider == "cute" else UNFUSED
    plan = (path, config) if kind == "gated" else config
    ours = SimpleNamespace(provider="quack", plan_for=lambda m: plan)
    reference = SimpleNamespace(probe=lambda x: None, metadata=lambda m: {})
    profiled = []

    def count(op, x):
        profiled.append(op)
        return 1 if provider == "cute" and op is ours else 4

    monkeypatch.setattr(parity.torch, "randn", lambda *args, **kwargs: object())
    monkeypatch.setattr(parity, "kernel_names", lambda op, x: ["k"] * count(op, x))
    monkeypatch.setattr(parity, "_timed_graph", lambda calls: (None, len(calls)))
    monkeypatch.setattr(
        parity, "_measure", lambda *args, **kwargs: {"ours_us": 80, "vllm_us": 100, "eps": 0}
    )
    record = parity._row(
        kind,
        "27b",
        4,
        [ours],
        [reference],
        4096,
        {"device": "RTX 5090"},
        20,
        5,
        comparator="vllm",
    )
    assert record["provider"] == "quack"
    assert record["dispatched_provider"] == provider
    assert profiled == [ours, reference]
    assert record["kernels_per_call"] == {"ours": 1 if provider == "cute" else 4, "vllm": 4}
    if provider == "cute":
        assert record["sol_fraction"] == record["sol_us"] / 80
        assert record["sol_ceilings"] == "rtx5090.json"
    else:
        assert record["sol"] is None
        assert record["sol_note"] == "vendor-dispatched row: no SOL gate"
    if kind == "gated":
        assert record["ours_gate_up_globals"] == "independent-unequal"
        assert record["vllm_gate_up_globals"] == "collapsed-to-max (same kernels as unequal)"
    else:
        assert "ours_gate_up_globals" not in record
    assert (
        not {
            "roofline_us",
            "utilization",
            "vllm_utilization",
            "bound",
            "F",
            "D_dram",
            "D_total",
            "activation_output_bytes",
            "activation_output_in_l2",
            "known_partial_bytes_written",
        }
        & record.keys()
    )
