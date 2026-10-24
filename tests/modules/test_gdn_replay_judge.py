# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks of the frozen numerical judge and independent batched FP64 oracle."""

import pytest
import torch

from benchmark.kernel_tuning.gdn_replay_bench import pair_stats
from benchmark.kernel_tuning.gdn_replay_common import Oracle
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_replay_precision import calibrated, errors
from ctikernels.gdn import GatedDeltaNetReplay

from ..gpu._gdn_reference import decode


def test_zero_denominator_and_noise_are_not_slack():
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    pairs = base.expand(20, 2, 2, 3).clone()
    candidate = base.clone()
    candidate[0, 1] = 1e-20
    candidate[1, 0] = 1.0001
    rr, eta, passed, invalid = calibrated(candidate, base, pairs)
    assert rr[0, 0] == 1 and torch.isinf(rr[0, 1])
    assert not passed[0, 1] and not passed[1, 0]
    assert torch.equal(eta, torch.zeros_like(eta))
    assert not invalid.any()
    pairs[0, 0, 0, 0] = 1
    assert calibrated(base, base, pairs)[3][0, 0]


def test_noise_nearest_rank_and_strict_comparison():
    base = torch.ones(1, 3, dtype=torch.float64)
    pairs = base.expand(20, 2, 1, 3).clone()
    pairs[:, 0, :, 0] += torch.arange(20, dtype=torch.float64)[:, None] / 100
    candidate = base.clone()
    candidate[0, 0] = 1.18
    _, eta, passed, _ = calibrated(candidate, base, pairs)
    assert eta[0, 0].item() == pytest.approx(0.18)
    assert passed.all()
    candidate[0, 0] = 1.18000001
    assert not calibrated(candidate, base, pairs)[2][0, 0]


def test_ulp_preserves_value_head_and_row_axes():
    truth = torch.zeros(2, 3, 128, dtype=torch.float64)
    truth[:, 0] = 1e5
    truth[:, 1] = 1e-4
    got = truth.clone()
    got[:, 1, 0] += 0.01
    assert (errors(got, truth)[:, 2] > 1000).all()
    with pytest.raises(ValueError, match="nonfinite"):
        errors(got, truth * float("nan"))


def test_batched_oracle_matches_pinned_token_reference():
    torch.set_num_threads(1)
    weights, x, state, history = case_inputs(1, 2, 2, 2, 5, "heavy", 23, "cpu")
    oracle = Oracle(1, weights.args(), state, history)
    for row in x:
        got = oracle.step(row)
        expected, state, history = decode(row, weights, 1, torch.arange(2), state, history)
        torch.testing.assert_close(got.reshape_as(expected), expected, rtol=2e-13, atol=1e-14)
        torch.testing.assert_close(oracle.canonical(), state, rtol=2e-13, atol=1e-14)
        assert torch.equal(oracle.history, history)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("kind", ["ordinary", "heavy"])
def test_oracle_capture_buffers_match_eager_on_cpu(batch, kind):
    torch.set_num_threads(1)
    weights, xs, state, history = case_inputs(1, 2, batch, batch, 10, kind, 11, "cpu")
    eager = Oracle(1, weights.args(), state, history)
    buffered = Oracle(1, weights.args(), state, history, graph=True)
    saved_outputs = []

    def compare(x):
        expected = eager.step(x)
        actual = buffered.step(x)
        assert torch.equal(actual, expected)
        assert torch.equal(buffered.canonical(), eager.canonical())
        assert torch.equal(buffered.history, eager.history)
        saved_outputs.append((actual, expected.clone()))

    for x in xs[:3]:
        compare(x)
    assert buffered._captured is not None  # CPU must exercise the reusable-buffer path.
    addresses = [v.data_ptr() for v in buffered._captured[1:]]
    for oracle in (eager, buffered):
        # Handoff can replace both public tensors while graph addresses stay fixed.
        oracle.state = oracle.state.float().double() * 0.5
        oracle.history = history.clone()
    compare(xs[3])
    drafts = xs[4:9].transpose(0, 1)  # Noncontiguous projected rows at B>1.
    for lengths in ([0] * batch, [5] * batch, [0, 1, 3, 5][:batch]):
        accepted = torch.tensor(lengths)
        assert torch.equal(buffered.verify(drafts, accepted), eager.verify(drafts, accepted))
        assert torch.equal(buffered.canonical(), eager.canonical())
        assert torch.equal(buffered.history, eager.history)
        compare(xs[9])
    assert [v.data_ptr() for v in buffered._captured[1:]] == addresses
    assert all(torch.equal(actual, expected) for actual, expected in saved_outputs)


def test_timing_pairs_require_all_samples_and_exact_bar():
    same = [[1.0, 1.0]] * 20
    assert pair_stats(same, same)["passed"]
    assert not pair_stats([[1.0, 1.000001]] * 20, same)["passed"]
    with pytest.raises(ValueError):
        pair_stats(same[:-1], same)


def test_explicit_provider_refusal():
    with pytest.raises(ValueError, match="requires"):
        GatedDeltaNetReplay(num_k_heads=1, num_v_heads=2, device="cpu", provider="cute")


def test_sol_curve_matches_pinned_interpolation_and_clamping():
    from benchmark.kernel_tuning.gdn_replay_sol import transfer

    memory = {
        "small_transfer": [{"bytes": 100, "time_us": 2}, {"bytes": 10000, "time_us": 8}],
        "dram_read_gbs": 100,
        "dram_copy_gbs": 50,
    }
    assert transfer(1, memory, "read") == 2
    assert transfer(1000, memory, "read") == pytest.approx(4)
    assert transfer(20000, memory, "copy") == pytest.approx(0.4)


def test_sol_one_launch_replay_keeps_state_traffic(tmp_path):
    import json

    from benchmark.kernel_tuning.gdn_replay_sol import report

    memory = {"small_transfer": [{"bytes": 1, "time_us": 0.0}], "dram_copy_gbs": 1.0}
    path = tmp_path / "ceil.json"
    path.write_text(json.dumps({"memory": memory, "fixed_costs": {"graph_node_us": 0.0}}))
    state = 4 * 32 * 128 * 128
    one = report("cute", 1, 16, 32, 8, 1.0, 1, path, "cold")["reference_us"]["useful"]
    two = report("cute", 1, 16, 32, 8, 1.0, 2, path, "cold")["reference_us"]["useful"]
    # Delayed reads the state every step and writes it once per window.
    assert one * 1000 > state * 9 / 8
    assert one == pytest.approx(two)


@pytest.mark.parametrize("batch", [1, 2])
def test_precision_adapter_multiphase_capture_on_cpu(batch):
    from benchmark.kernel_tuning.gdn_replay_common import Replay

    weights, x, state, history = case_inputs(1, 4, batch, batch, 17, "ordinary", 11, "cpu")
    runner = Replay(
        "torch_reference",
        1,
        4,
        batch,
        4,
        2,
        weights.args(),
        state,
        history,
        torch.tensor([2**33 + 7] * batch, dtype=torch.int64),
    )
    for row in x[:9]:
        runner.step(row)
    assert runner.expected_cursor.tolist() == [2] * batch
    assert runner.expected_n.tolist() == [1] * batch
    before = runner.state.checkpoint[:batch].clone()
    runner.materialize()
    assert torch.equal(before, runner.state.checkpoint[:batch])
    drafts = x[9:11].transpose(0, 1)  # Includes singleton-B misleading contiguity.
    if batch == 1:
        assert drafts.is_contiguous() and drafts.stride(0) < drafts.shape[1] * drafts.shape[2]
    runner.verify(drafts)
    runner.resolve(
        torch.tensor([0, 2][:batch], dtype=torch.int32), offsets=torch.tensor([0, 1][:batch])
    )
    assert runner.expected_cursor.tolist() == [0] * batch
    assert runner.expected_n.tolist() == [0] * batch
    runner.verify(drafts)
    before = runner.materialize().clone()
    runner.resolve(torch.zeros(batch, dtype=torch.int32), reject=True)
    assert torch.equal(runner.materialize(), before)
    runner.check()


def test_cute_workspace_exact_payload_and_initialization():
    from ctikernels.gdn.replay.cute import Config

    # Explicit device and config avoid CUDA discovery; descriptor queries are host-only.
    for snapshot in (False, True):
        for sections in (False, True):
            op = GatedDeltaNetReplay(
                num_k_heads=16,
                num_v_heads=48,
                device="cuda:0",
                provider="cute",
                cute_config=Config(sections=sections, snapshot_controls=snapshot, trusted=False),
                checks="strict",
            )
            for batch in (0, 1, 4):
                spec = op.workspace_spec("step", batch_size=batch)
                scratch = {k: v for k, v in spec.items() if k != "out"}
                assert "pre_norm" not in spec
                assert ("arrivals" in spec) == (not snapshot)
                if snapshot:
                    assert spec["controls"].shape == (2 * batch + 1,)
                    assert spec["controls"].initial_value is None
                else:
                    assert spec["arrivals"].shape == (1,)
                    assert spec["arrivals"].initial_value == 0
                assert ("sections" in spec) == sections
                if sections:
                    assert spec["sections"].shape == (batch * 48 + 1, 7)
                assert all(r.lifetime == "invocation" for r in scratch.values())
                total = sum(r.payload_bytes for r in scratch.values())
                assert total == (
                    batch * op.capabilities["step_scratch_bytes_per_row"]
                    + op.capabilities["step_scratch_fixed_bytes"]
                )
                if batch:
                    graph = op.graph_data_spec(batch_size=batch)
                    assert all(graph[k].shape == r.shape for k, r in scratch.items())
                assert all(s.offset % 256 == 0 for s in spec.values())
            assert op.provider["step"] == "cute"
            assert all(
                p
                == (
                    "triton"
                    if mode
                    in (
                        "reset",
                        "initialize",
                        "handoff",
                        "import_snapshot",
                        "release",
                        "begin",
                        "end",
                    )
                    else "cute"
                )
                for mode, p in op.provider.items()
                if mode != "handoff_layers"  # Copy provider qualification is device-independent.
            )
            assert op.capabilities["precision_tiers"] == ("default", "strict")
            assert not op.capabilities["strict_qualified"]


def test_section_timing_uses_per_cta_differences():
    from benchmark.kernel_tuning.gdn_replay_profile import section_medians

    # Different absolute launch times must not enter a duration comparison.
    rows = [
        [100, 1100, 3100, 6100, 10100, 15100, 21100],
        [1000000, 1003000, 1007000, 1012000, 1018000, 1025000, 1033000],
    ]
    assert list(section_medians(rows).values()) == [2, 3, 4, 5, 6, 7]
    assert section_medians(rows)["epilogue"] == 6
    assert section_medians(rows)["output_norm"] == 7
    assert section_medians([]) is None
    with pytest.raises(ValueError, match="nonmonotonic"):
        section_medians([[7, 6, 5, 4, 3, 2, 1]])


def test_batched_calibration_metrics_preserve_each_replica_and_row():
    from benchmark.kernel_tuning.gdn_replay_precision import calibration_errors

    generator = torch.Generator().manual_seed(111)
    for shape in ((2, 3, 128), (2, 3, 8, 128)):
        truth = torch.randn(shape, dtype=torch.float64, generator=generator)
        replicas = [truth.float() + i * 1e-7 for i in range(40)]
        got = calibration_errors(replicas, truth)
        expected = torch.stack([errors(v, truth) for v in replicas]).reshape(20, 2, 2, 3)
        assert torch.equal(got[..., :2], expected[..., :2])


def test_buffered_score_preserves_worst_position_and_failure():
    from benchmark.kernel_tuning.gdn_replay_precision import Score

    score = Score("ordinary")
    baseline = torch.ones(2, 3, dtype=torch.float64)
    pairs = baseline.expand(20, 2, 2, 3).clone()
    score.add("state/decode", 1, baseline, baseline, pairs)
    candidate = baseline.clone()
    candidate[1, 0] = 1.00001
    score.add("state/flush", 2, candidate, baseline, pairs)
    # Storage may be reused by the caller after add; queued metrics own values.
    candidate.zero_()
    score.drain()
    assert score.observations == 4 and not score.passed
    assert score.worst["state_rms"]["tuple"] == ["state/flush", 2, 1]
    assert score.worst["state_rms"]["eta"] == 0
    assert score.worst["state_rms"]["ratio"] == 1.00001
    assert not score.pending


@pytest.mark.parametrize(
    "mode,backend",
    [
        (mode, "cute._phase_launch")
        for mode in ("verify", "commit", "reject", "flush", "export_snapshot", "bridge_step")
    ]
    + [
        (mode, "triton")
        for mode in ("reset", "initialize", "handoff", "import_snapshot", "release")
    ],
)
def test_cute_phase_dispatch_preserves_borrowed_arguments(monkeypatch, mode, backend):
    import sys
    from types import SimpleNamespace

    from ctikernels.gdn.replay import cute

    calls = []
    monkeypatch.setitem(
        sys.modules,
        f"ctikernels.gdn.replay.{backend}",
        SimpleNamespace(run=lambda *args, **kw: calls.append((args, kw))),
    )
    op, state, status, weights, controls = (object() for _ in range(5))
    cute.run(op, mode, state=state, status=status, weights=weights, slot_indices=controls)
    assert calls == [
        (
            (op, mode),
            {"state": state, "status": status, "weights": weights, "slot_indices": controls},
        )
    ]


@pytest.mark.parametrize("provider", ["triton"])
def test_batch_invariant_tier_refuses_unqualified_cuda_providers(provider):
    with pytest.raises(ValueError, match="strict"):
        GatedDeltaNetReplay(
            num_k_heads=16,
            num_v_heads=32,
            device="cuda:0",
            provider=provider,
            precision="strict",
        )


def test_product_dispatch_fma_orders_are_distinct_fp32_functions():
    # These normal FP32 operands have exactly representable FP64 products.
    # Each expression models the PTX's rounded product plus its RN FMA;
    # the result is far from a FP32 midpoint, so FP64 summation is sufficient.
    alpha, sq, delta, kq = torch.tensor(
        [0.29611194133758545, 0.516562283039093, 0.2516707181930542, 0.05508454144001007],
        dtype=torch.float32,
    )
    scalar_tile = (alpha.double() * sq.double() + (delta * kq).double()).float()
    packed_tile = ((alpha * sq).double() + delta.double() * kq.double()).float()
    assert scalar_tile.item() == 0.16682343184947968
    assert packed_tile.item() == 0.16682341694831848
    assert abs(scalar_tile.view(torch.int32).item() - packed_tile.view(torch.int32).item()) == 1
