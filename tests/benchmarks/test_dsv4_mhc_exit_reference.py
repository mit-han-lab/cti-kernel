# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only numerical and benchmark protocol coverage; never initialize CUDA."""

from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.dsv4_mhc_exit import bench, market, pair
from benchmark.kernel_tuning.dsv4_mhc_exit import reference as ref


def straightforward(cfg, weights, state):
    """Independent loops over streams and tokens; official model.py:674-690,
    kernel.py:398-424. No calls to the reference's coefficient/Sinkhorn helpers.
    """
    x, R, post, comb = [state[key].double() for key in ("x", "R", "post", "comb")]
    w = {key: value.double() for key, value in weights.items()}
    outputs = {key: [] for key in ("residual", "pre", "post", "comb", "layer_input", "out")}
    for t in range(x.shape[0]):
        streams = []
        for j in range(cfg.mult):
            streams.append(
                post[t, j] * x[t] + sum(comb[t, i, j] * R[t, i] for i in range(cfg.mult))
            )
        residual = torch.stack(streams)
        flat = residual.reshape(-1)
        rms = (sum(v * v for v in flat) / flat.numel() + cfg.rms_eps).sqrt()
        mixes = torch.stack(
            [sum(a * b for a, b in zip(flat, row, strict=True)) / rms for row in w["fn"]]
        )
        pre = torch.stack(
            [
                1 / (1 + torch.exp(-(mixes[j] * w["scale"][0] + w["base"][j]))) + cfg.hc_eps
                for j in range(cfg.mult)
            ]
        )
        p = cfg.post_mult * torch.sigmoid(
            mixes[cfg.mult : 2 * cfg.mult] * w["scale"][1] + w["base"][cfg.mult : 2 * cfg.mult]
        )
        logits = (mixes[2 * cfg.mult :] * w["scale"][2] + w["base"][2 * cfg.mult :]).reshape(
            cfg.mult, cfg.mult
        )
        c = torch.stack(
            [
                torch.exp(row - row.max()) / torch.exp(row - row.max()).sum() + cfg.hc_eps
                for row in logits
            ]
        )
        for j in range(cfg.mult):
            c[:, j] /= c[:, j].sum() + cfg.hc_eps
        for _ in range(1, cfg.sinkhorn_iters):
            for i in range(cfg.mult):
                c[i] /= c[i].sum() + cfg.hc_eps
            for j in range(cfg.mult):
                c[:, j] /= c[:, j].sum() + cfg.hc_eps
        y = sum(pre[j] * residual[j] for j in range(cfg.mult))
        out = y / (sum(v * v for v in y) / cfg.hidden + cfg.norm_eps).sqrt() * w["norm_weight"]
        for key, value in zip(outputs, (residual, pre, p, c, y, out), strict=True):
            outputs[key].append(value)
    return {key: torch.stack(value) for key, value in outputs.items()}


@pytest.mark.parametrize("iters", [1, 20])
def test_fp64_matches_independent_formulas(iters):
    cfg = ref.MhcConfig(hidden=64, sinkhorn_iters=iters, post_mult=1.7)
    weights, state = ref.make_weights(cfg, 42), ref.make_state(cfg, 3, 43)
    actual = ref.exit_fp64(cfg, weights, *bench.inputs(state))
    expected = straightforward(cfg, weights, state)
    for key in actual:
        assert actual[key].dtype == torch.float64
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-13, atol=2e-13)


def test_state_weights_and_sinkhorn():
    cfg = ref.MhcConfig(hidden=64)
    weights, state = ref.make_weights(cfg, 11), ref.make_state(cfg, 4, 12)
    assert weights["fn"].shape == (24, 256)
    assert weights["scale"].shape == (3,)
    assert weights["base"].shape == (24,)
    assert weights["norm_weight"].dtype == torch.bfloat16
    assert torch.equal(state["x"], (state["x0"].float() + state["x1"].float()).bfloat16())
    result = ref.exit_fp64(cfg, weights, *bench.inputs(state))
    for comb in (state["comb"], result["comb"]):
        assert (comb > 0).all()
        torch.testing.assert_close(comb.sum(-1), torch.ones_like(comb.sum(-1)), rtol=0, atol=3e-6)
        torch.testing.assert_close(comb.sum(-2), torch.ones_like(comb.sum(-2)), rtol=0, atol=3e-6)
    assert torch.equal(state["R"], ref.make_state(cfg, 4, 12)["R"])
    assert not torch.equal(state["R"], ref.make_state(cfg, 4, 13)["R"])


def test_bf16_rounding_errors_and_metric_schema():
    cfg = ref.MhcConfig(hidden=64)
    weights, state = ref.make_weights(cfg, 10), ref.make_state(cfg, 4, 20)
    oracle = ref.exit_fp64(cfg, weights, *bench.inputs(state))
    actual = ref.exit_torch_bf16(cfg, weights, *bench.inputs(state))
    metrics = ref.error_stats(actual, oracle)
    assert set(metrics) == set(oracle)
    for key, stats in metrics.items():
        assert set(stats) == {"max_abs", "rms"}
        assert 0 <= stats["rms"] <= stats["max_abs"] < 0.06
        assert actual[key].dtype == (
            torch.bfloat16 if key in ("residual", "layer_input", "out") else torch.float32
        )
    y = actual["layer_input"].float()
    expected = (
        y
        * torch.rsqrt(y.square().mean(-1, keepdim=True) + cfg.norm_eps)
        * weights["norm_weight"].float()
    ).bfloat16()
    assert torch.equal(actual["out"], expected)
    assert metrics["residual"]["rms"] > 0
    assert bench.check(cfg, weights, state, actual)["provider_le_torch_bf16"]
    with pytest.raises(ValueError, match="shape"):
        ref.error_stats({"out": actual["out"][:1]}, oracle)


def test_nonfinite_error_never_passes_floor():
    cfg = ref.MhcConfig(hidden=64)
    w, state = ref.make_weights(cfg), ref.make_state(cfg, 1)
    result = ref.exit_torch_bf16(cfg, w, *bench.inputs(state))
    result["out"][0, 0] = float("nan")
    check = bench.check(cfg, w, state, result)
    assert not check["finite"] and not check["provider_le_torch_bf16"]
    assert check["errors"]["out"]["rms"] is None


@pytest.mark.parametrize("name", market.NAMES)
def test_provider_cpu_status_without_cuda_probe(name, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preparation must not probe CUDA or load optional libraries")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(market.importlib, "import_module", forbidden)
    cfg = ref.MhcConfig(hidden=64)
    w, state = ref.make_weights(cfg), ref.make_state(cfg, 2)
    provider = market.Provider(name)
    fn = provider.prepare(cfg, w, 2, "cpu")
    if name == "torch":
        assert provider.status == "available"
        result = fn(*bench.inputs(state))
        assert [value.shape for value in result] == [(2, 4, 64), (2, 4), (2, 4, 4), (2, 64)]
    else:
        assert fn is None
        assert provider.status.startswith("unavailable: requires CUDA")


def test_cold_working_set_strictly_exceeds_twice_l2():
    cfg = ref.MhcConfig()
    count, size = bench.rotation_count(cfg, 1, 16, 32 * 1024**2)
    assert count > 16 and size > 64 * 1024**2
    assert bench.rotation_count(cfg, 1, 16, 0, hot=True)[0] == 1
    with pytest.raises(ValueError, match="L2"):
        bench.rotation_count(cfg, 1, 16, 0)


def test_paired_samples_preserve_pairing_and_shuffle_order():
    seen = []

    def timer(name, repeats, calls):
        seen.append(name)
        return {"a": 2.0, "b": 4.0}[name]

    result = bench.measure(
        {"a": "a", "b": "b"}, SimpleNamespace(calls=86, pairs=10, seed=11), timer
    )
    assert result["b"]["fastest_provider"] == "a"
    assert result["b"]["ratio_ci95"] == [2.0, 2.0]
    assert all(set(seen[i : i + 2]) == {"a", "b"} for i in range(2, len(seen), 2))
    assert len({tuple(seen[i : i + 2]) for i in range(2, len(seen), 2)}) == 2


def test_block_rotates_states_and_weights_and_hash_checks_all_outputs():
    cfg = ref.MhcConfig(hidden=64)
    states = [ref.make_state(cfg, 1, i) for i in range(3)]
    visited = []
    functions = [lambda x, R, p, c, i=i: visited.append((i, x.data_ptr())) for i in range(3)]
    bench.block_calls(functions, states, 7)()
    assert [i for i, _ in visited] == [0, 1, 2, 0, 1, 2, 0]
    assert all(ptr == states[i]["x"].data_ptr() for i, ptr in visited)
    result = ref.exit_torch_bf16(cfg, ref.make_weights(cfg), *bench.inputs(states[0]))
    before = pair.digest(result)
    result["comb"][0, 0, 0] += 1e-5
    assert pair.digest(result) != before


def test_error_messages_do_not_echo_exception_payloads():
    assert "untrusted detail" not in market.failure_reason(
        RuntimeError("untrusted detail"), "warmup"
    )


def test_spacer_sequence_and_standalone_count():
    state = {key: key for key in ("x", "R", "post", "comb")}
    visited = []

    def exit_call(*args):
        visited.append("exit")
        return "last-output"

    def spacer():
        visited.append("spacer")

    assert bench.block_calls([exit_call], [state], 3, spacer)() == "last-output"
    assert visited == ["exit", "spacer", "exit", "spacer", "exit"]
    visited.clear()
    bench.spacer_block(spacer, 3)()
    assert visited == ["spacer", "spacer"]
    assert bench.make_spacer(0, "cpu") is None
    # Exercise the allocation-free operation itself; buffer storage stays private.
    actual = bench.make_spacer(0.001, "cpu")
    assert actual.read_bytes == 1052
    actual()
    actual()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_spacer_argument_rejects_invalid_size(value):
    parser = bench.arguments()
    args = parser.parse_args(["--spacer-mib", value])
    with pytest.raises(SystemExit):
        bench.validate(parser, args)


def test_iket_summary_uses_per_cta_max_and_skips_warmup():
    from benchmark.kernel_tuning.dsv4_mhc_exit.iket_summary import summarize

    document = {
        "stringTable": ["mhc.coefficients"],
        "locationTable": [
            {"ctaId": [0, 0, 0], "warpId": 0},
            {"ctaId": [0, 0, 0], "warpId": 1},
            {"ctaId": [1, 0, 0], "warpId": 0},
        ],
        "launches": [],
    }
    for grid, durations in enumerate(((900, 800, 700), (10, 20, 30))):
        document["launches"].append(
            {
                "gridId": grid,
                "contextId": 1,
                "trace": {
                    "ranges": [
                        {
                            "rangeNameIdx": 0,
                            "warpLocIdxs": [warp, warp],
                            "startTs": 100,
                            "endTs": 100 + dt,
                        }
                        for warp, dt in enumerate(durations)
                    ]
                },
            }
        )
    # Repeated chunks add per warp. Parallel warp durations must not be summed.
    document["launches"][-1]["trace"]["ranges"].append(
        {
            "rangeNameIdx": 0,
            "warpLocIdxs": [0, 0],
            "startTs": 200,
            "endTs": 215,
        }
    )
    assert summarize(document, last_launches=1) == {"mhc.coefficients": [25, 30]}
    assert summarize(document, last_launches=1, by_cta=True) == {
        ("mhc.coefficients", (0, 0, 0)): [25],
        ("mhc.coefficients", (1, 0, 0)): [30],
    }
    with pytest.raises(ValueError, match="no mHC ranges"):
        summarize({})


def test_iket_timeline_envelopes_launch_averages_and_fallback(tmp_path, monkeypatch, capsys):
    import json

    from benchmark.kernel_tuning.dsv4_mhc_exit import iket_summary

    def span(name, cta, warp, start, end):
        return {
            "range_name": name,
            "warp_locs": [{"cta_id": {"x": cta, "y": 0, "z": 0}, "warp_id": warp}],
            "start_ts": start,
            "end_ts": end,
        }

    document = {
        "launches": [
            {"gridId": 0, "ranges": [span("mhc.output", 0, 0, 0, 9999)]},
            {
                "gridId": 1,
                "ranges": [
                    span("mhc.grid_wait", 0, 0, 80, 90),
                    span("mhc.grid_wait", 0, 1, 85, 100),
                    span("mhc.grid_wait", 1, 0, 84, 105),
                    span("mhc.arrival", 0, 0, 106, 107),
                    span("mhc.arrival", 1, 0, 109, 110),
                    span("mhc.last_gains", 1, 0, 110, 120),
                    span("mhc.partial_read", 1, 0, 110, 115),
                    span("mhc.flag_wait", 0, 0, 107, 121),
                    span("mhc.output", 0, 0, 112, 114),
                    span("mhc.output", 0, 0, 116, 117),
                    span("mhc.output", 0, 1, 113, 118),
                    span("mhc.output", 1, 0, 122, 130),
                    span("mhc.last_sinkhorn", 1, 0, 130, 135),
                    span("unrelated", 0, 0, 0, 9000),
                ],
            },
            {
                "gridId": 2,
                "ranges": [
                    span("mhc.cluster_exchange", 0, 0, 200, 210),
                    *[
                        span("mhc.exchange", cta, 0, 215 + 10 * cta, 218 + 10 * cta)
                        for cta in range(3)
                    ],
                    *[
                        span("mhc.output", cta, 0, 220 + 20 * cta, 230 + 20 * cta)
                        for cta in range(3)
                    ],
                    span("mhc.projection", 0, 0, 210, 212),
                    span("mhc.gram", 0, 0, 212, 214),
                ],
            },
        ]
    }
    # Round-trip a small expanded JSON, with different CTA counts per launch.
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(document))
    document = json.loads(path.read_text())
    phases, boundaries = iket_summary.timeline(document, 2, ns_per_tick=1000)
    assert phases["mhc.output"] == {"launches": 2, "start": (16, 28.5, 41), "end": (24, 37, 50)}
    assert phases["mhc.last_gains"] == {"launches": 1, "start": (10, 10, 10), "end": (20, 20, 20)}
    assert phases["mhc.last_sinkhorn"] == {
        "launches": 1,
        "start": (30, 30, 30),
        "end": (35, 35, 35),
    }
    assert boundaries == [
        {"launch": (0, 1), "t0": 100, "last_cta_arrival_us": 9, "kernel_end_us": 35},
        {"launch": (0, 2), "t0": 200, "last_cta_arrival_us": 35, "kernel_end_us": 70},
    ]
    assert {"mhc.cluster_exchange", "mhc.projection", "mhc.gram", "mhc.flag_wait"} <= phases.keys()
    assert set(iket_summary.summarize(document, 2)) == set(phases)
    assert iket_summary.timeline(document, 1)[1][0]["kernel_end_us"] == pytest.approx(0.070)
    monkeypatch.setattr(
        "sys.argv", ["iket_summary", str(path), "--timeline", "--last-launches", "2"]
    )
    iket_summary.main()
    output = capsys.readouterr().out
    assert "start_us[min median max]" in output
    assert "last CTA arrival_us=0.009 kernel_end_us=0.035" in output
    assert "last CTA arrival_us=0.035 kernel_end_us=0.070" in output
    assert "mhc.cluster_exchange" in output
    with pytest.raises(ValueError, match="no mHC ranges"):
        iket_summary.timeline({})


def test_iket_timeline_missing_boundaries():
    from benchmark.kernel_tuning.dsv4_mhc_exit.iket_summary import timeline

    phases, boundaries = timeline(
        {
            "gridId": 1,
            "rangeName": "mhc.new_phase",
            "warp_locs": [{"ctaId": [0, 0, 0], "warpId": 0}],
            "startTs": 100,
            "endTs": 200,
        }
    )
    assert phases["mhc.new_phase"]["end"] == (0.1, 0.1, 0.1)
    assert boundaries[0]["last_cta_arrival_us"] is None
    assert boundaries[0]["kernel_end_us"] is None


def test_pair_fused_functions_keep_partials_and_outputs(monkeypatch):
    import ctikernels.mhc

    prepared, calls = [], []

    class FakeExit:
        def __init__(self, *args, **kwargs):
            pass

        def _fused(self, tokens, dtype, ar):
            return ar.active

        def prepare(self, counts, all_reduce):
            prepared.append((counts, all_reduce))

        def __call__(self, x, R, post, comb, **kwargs):
            calls.append((x, kwargs))
            return tuple(kwargs[name] for name in ("residual_out", "post_out", "comb_out", "out"))

    monkeypatch.setattr(ctikernels.mhc, "MhcExit", FakeExit)
    cfg = ref.MhcConfig(hidden=16)
    weights = [ref.make_weights(cfg, seed) for seed in (21, 22)]
    states = [ref.make_state(cfg, 3, seed) for seed in (31, 32)]
    ar = SimpleNamespace(active=True)
    functions = pair.fused_functions(cfg, weights, states, ar, rank=1)
    assert prepared == [([3], ar)] and not calls
    for fn, w, state in zip(functions, weights, states, strict=True):
        first = fn(*bench.inputs(state))
        second = fn(*bench.inputs(state))
        assert all(a is b for a, b in zip(first, second, strict=True))
        assert calls[-1][0] is state["x1"]
        assert calls[-1][1]["fn"] is w["fn"]
        assert calls[-1][1]["all_reduce"] is ar
    ar.active = False
    with pytest.raises(ValueError, match="active B1"):
        pair.fused_functions(cfg, weights, states, ar, rank=0)
