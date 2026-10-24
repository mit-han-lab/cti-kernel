# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only warm-up policy, persistence and mocked measurement contracts."""

import json
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from ctikernels.linear import Linear, autotune, autotune_entries, cute, inference
from ctikernels.linear import _autotune as tuning
from ctikernels.linear import _dispatch as dispatch

MEASURE = tuning._measure  # The real one; the host fixture replaces it.

DEVICE = torch.device("cuda:0")
REAL_MEASURE = tuning._measure  # The autouse fixture mocks it.
HARDWARE = (12, 0, 170, "NVIDIA GeForce RTX 5090")
N, K = 3072, 4096


def forbidden(*args, **kwargs):
    pytest.fail("unexpected tuning, compilation or cache access")


@pytest.fixture(autouse=True)
def host(monkeypatch, tmp_path):
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE", "cache")
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    monkeypatch.setattr(tuning, "_CACHES", {})
    monkeypatch.setattr(tuning, "_WARNED", set())
    monkeypatch.setattr(tuning, "_driver_version", lambda: "driver")
    monkeypatch.setattr(tuning, "_version", lambda name: name + "-version")
    monkeypatch.setattr(tuning, "_cublas_version", lambda: "cublas")
    monkeypatch.setattr(tuning, "_fingerprint", lambda: "kernels")
    monkeypatch.setattr(tuning, "_device_name", lambda _: HARDWARE[3])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", HARDWARE[:2], HARDWARE[2]))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(cute, "get_compiled", lambda *args: object())
    monkeypatch.setattr(tuning, "_measure", measurement)
    monkeypatch.setattr(tuning, "_release", lambda: None)  # No GPU memory to free.


def qualified(bucket=(3, 4), speedup=1.1, epsilon=0.01, pairs=20, ms=()):
    measured = sorted({*bucket, *ms})
    record = tuning._record(bucket)
    record.update(
        config=asdict(dispatch.Config(splits=2)),
        correct=True,
        deterministic=True,
        measured_m=measured,
        regimes={
            regime: {
                str(m): {
                    "speedup": speedup,
                    "epsilon": epsilon,
                    "ours_us": 10.0,
                    "cublas_us": 10 * speedup,
                    "pairs": pairs,
                }
                for m in measured
            }
            for regime in ("cold", "engine", "hot")
        },
    )
    tuning._qualify(record, "rtx5090")
    return record


def measurement(weight, hardware, stride, bucket, ms, configs, pairs, record):
    record.update(qualified(bucket, ms=ms))


def resolve(op=None, n=N, edge=4, stride=0):
    return (op or Linear())._resolve(DEVICE, n, K, edge, stride, weight=object())


def seed(record=None, n=N, stride=0, bucket=(3, 4)):
    key = tuning.cache_key(HARDWARE, n, K, stride, bucket)
    with autotune():
        tuning._write(key, record or qualified(bucket))
    return key


def test_key_contains_all_identity_fields(monkeypatch):
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    assert json.loads(key) == [
        list(HARDWARE),
        "driver",
        torch.__version__,
        torch.version.cuda,
        "cublas",
        "nvidia-cutlass-dsl-version",
        "cti-kernels-version",
        "kernels",
        N,
        K,
        0,
        [3, 4],
    ]
    seed()
    assert Linear().decision(N, K, 3)["source"] == "autotune"
    for package in ("nvidia-cutlass-dsl", "cti-kernels"):
        with monkeypatch.context() as patch:
            patch.setattr(
                tuning,
                "_version",
                lambda name, package=package: "changed" if name == package else name + "-version",
            )
            assert Linear().decision(N, K, 3)["source"] == "default"
    for target, field, value in (
        (tuning, "_driver_version", lambda: "new"),
        (torch.version, "cuda", "new"),
        (torch, "__version__", "new"),
        (tuning, "_cublas_version", lambda: "new"),
        (tuning, "_fingerprint", lambda: "new"),
        (tuning, "_device_name", lambda _: "NVIDIA RTX PRO 6000 Blackwell"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(target, field, value)
            assert Linear().decision(N, K, 3)["source"] == "default"
    for hardware, n, k, stride, bucket in (
        ((12, 1, 48, HARDWARE[3]), N, K, 0, (3, 4)),
        (HARDWARE, N + 8, K, 0, (3, 4)),
        (HARDWARE, N, K + 8, 0, (3, 4)),
        (HARDWARE, N, K, K, (3, 4)),
        (HARDWARE, N, K, 0, (5, 8)),
    ):
        assert tuning.lookup(hardware, n, k, stride, bucket) is None


@pytest.mark.parametrize("stride", [0, 10240])
@pytest.mark.parametrize("mode", ["cache", "0"])
def test_table_always_wins(monkeypatch, stride, mode):
    monkeypatch.setattr(tuning, "lookup", forbidden)
    monkeypatch.setattr(tuning, "tune", forbidden)
    with autotune():
        op = Linear(autotune=mode)
        result = op.decision(4096, 4096, 3, gate_stride=stride)
        assert result["source"] == "table" and result["record"] is None
        plan = op._resolve(DEVICE, 4096, 4096, 4, stride, object())
        assert plan.launch["source"] == "table"


@pytest.mark.parametrize(
    "provider,mode,source",
    [("auto", "0", "off"), ("cute", "cache", "forced"), ("cublas", "cache", "forced")],
)
def test_off_and_forced_ignore_cache(monkeypatch, provider, mode, source):
    monkeypatch.setattr(tuning, "lookup", forbidden)
    monkeypatch.setattr(tuning, "tune", forbidden)
    monkeypatch.setattr(tuning, "_write", forbidden)
    with autotune():
        op = Linear(provider=provider, autotune=mode)
        assert op.decision(N, K, 3)["source"] == source
        assert (resolve(op) is not None) == (provider == "cute")
    assert tuning._CACHES == {}


def test_default_reads_cache_only_inside_context(monkeypatch):
    seed()  # A qualified CuTe decision left on this node.
    monkeypatch.delenv("CTI_LINEAR_AUTOTUNE")
    lookup = tuning.lookup
    monkeypatch.setattr(tuning, "lookup", forbidden)
    op = Linear()
    assert op.decision(N, K, 3)["source"] == "default" and resolve(op) is None
    monkeypatch.setattr(tuning, "lookup", lookup)
    with autotune():
        assert Linear().decision(N, K, 3)["source"] == "autotune"
    assert Linear(autotune="cache").decision(N, K, 3)["source"] == "autotune"


def test_context_exit_during_selection_never_reads_node_cache(monkeypatch, tmp_path):
    seed()  # Node-default cache (CTI_LINEAR_AUTOTUNE_CACHE) holds a CuTe decision.
    monkeypatch.delenv("CTI_LINEAR_AUTOTUNE")
    identity = tuning.identity

    def exiting(device):  # Another thread's context exits between check and lookup.
        tuning._CONTEXT, tuning.ACTIVE = None, False
        return identity(device)

    monkeypatch.setattr(tuning, "identity", exiting)
    with autotune(cache=str(tmp_path / "explicit.json")):
        assert Linear().decision(N, K, 3)["source"] == "default"


def test_outermost_exit_releases_once_after_restoring(monkeypatch):
    released = []
    monkeypatch.setattr(tuning, "_release", lambda: released.append(tuning._CONTEXT))
    with autotune():
        pass  # Nothing tuned: nothing to release.
    with pytest.raises(KeyError), autotune():
        with autotune():
            tuning._CONTEXT.devices.add("cuda:1")  # As _measure records it.
        assert released == []
        raise KeyError  # An exception exit still releases.
    assert released == [None] and not tuning.ACTIVE


def test_expiry_before_gpu_work_marks_no_device(monkeypatch):
    released = []
    monkeypatch.setattr(tuning, "_release", lambda: released.append(True))
    weight = SimpleNamespace(device="cuda:0")
    with autotune(budget_seconds=0):
        with pytest.raises(tuning._BudgetExpired):
            MEASURE(weight, HARDWARE, 0, (3, 4), [3, 4], [], 20, tuning._record((3, 4)))
        assert tuning._CONTEXT.devices == set()
    assert released == []


def test_cleanup_warning_as_error_never_replaces_the_body_error(monkeypatch):
    def failing():
        raise RuntimeError("cleanup")

    monkeypatch.setattr(tuning, "_release", failing)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError), autotune():
            tuning._CONTEXT.devices.add("cuda:0")
            raise KeyError
        with pytest.raises(UserWarning, match="cleanup failed"), autotune():
            tuning._CONTEXT.devices.add("cuda:0")  # A successful body may see the warning.


@pytest.mark.parametrize("body_error", [False, True])
def test_cleanup_failure_warns_and_keeps_the_body_outcome(monkeypatch, body_error):
    def failing():
        raise RuntimeError("cleanup")

    monkeypatch.setattr(tuning, "_release", failing)
    with pytest.warns(UserWarning, match="cleanup failed"):
        with pytest.raises(KeyError) if body_error else nullcontext():
            with autotune():
                tuning._CONTEXT.devices.add("cuda:0")
                if body_error:
                    raise KeyError


def test_mode_read_at_construction(monkeypatch):
    off = Linear(autotune="0")
    seed()
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE", "0")
    assert Linear().decision(N, K, 3)["source"] == "off"
    cached = Linear(autotune="cache")
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE", "bad")
    assert cached.decision(N, K, 3)["source"] == "autotune"
    assert off.decision(N, K, 3)["source"] == "off"
    with pytest.raises(ValueError, match="autotune"):
        Linear()


def test_capture_never_tunes_or_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with autotune():
        assert tuning.tune(object(), HARDWARE, "rtx5090", N, K, 0, (3, 4)) is None
        monkeypatch.setattr(tuning, "tune", forbidden)
        assert resolve() is None
        tuning._write("key", qualified())
    assert not (tmp_path / "cache.json").exists()


def test_capture_uses_cached_decision(monkeypatch):
    seed()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(tuning, "tune", forbidden)
    with autotune():
        plan = resolve()
        assert plan.launch["source"] == "autotune"


def test_reload_and_query_are_read_only(monkeypatch):
    with autotune(ms=[3, 4]):
        plan = resolve()
    assert plan.launch["source"] == "autotune"
    before = Linear().decision(N, K, 3)
    tuning._CACHES.clear()  # Fresh-process cache simulation, retaining only the file.
    monkeypatch.setattr(tuning, "tune", forbidden)
    monkeypatch.setattr(cute, "get_compiled", forbidden)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
    with autotune():
        other = Linear()
        assert other.decision(N, K, 4) == before
        assert other.workspace_bytes(N, K, max_m=4) == 2 * 4 * N * 4
        entries = autotune_entries()
        assert len(entries) == 1
        next(iter(entries.values()))["provider"] = "mutated"
        assert other.decision(N, K, 4) == before
    assert autotune_entries(device="cpu") == {}


def test_bucket_edges_and_ms_restrictions():
    with autotune(ms=[6, 7]):
        assert resolve(edge=4) is None
        assert not tuning._entries()
        assert resolve(edge=8) is not None
        record = Linear().decision(N, K, 6)["record"]
        assert record["measured_m"] == [5, 6, 7, 8]
        assert resolve(edge=16) is None
    bad = qualified()
    bad["measured_m"] = [4]
    seed(bad)
    decision = Linear().decision(N, K, 3)
    assert decision["source"] == "default" and decision["provider"] == "cublas"
    assert resolve() is None
    assert len(tuning._entries()) == 2


def test_budget_includes_measurement_and_skips_without_caching(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(tuning.time, "monotonic", lambda: clock[0])

    def expensive(*args):
        measurement(*args)
        clock[0] += 2

    monkeypatch.setattr(tuning, "_measure", expensive)
    with autotune(budget_seconds=1):
        assert resolve() is None
        assert resolve(n=N + 8) is None
    assert not tuning._entries()
    assert Linear().decision(N, K, 3)["record"] is None
    assert Linear().decision(N + 8, K, 3)["record"] is None
    with autotune(budget_seconds=0):
        assert resolve(n=N + 8) is None
    monkeypatch.setattr(tuning, "_measure", measurement)
    with autotune(budget_seconds=1):
        assert resolve(n=N + 8) is not None


def test_nested_context_restores_budget_path_and_ms(tmp_path):
    with autotune(ms=[3], cache=tmp_path / "outer.json"):
        outer = tuning._CONTEXT
        with autotune(ms=[1], budget_seconds=0, cache=tmp_path / "inner.json"):
            assert resolve(edge=1) is None
        assert tuning._CONTEXT is outer and tuning.ACTIVE
        assert resolve() is not None
    assert not tuning.ACTIVE and tuning._CONTEXT is None
    assert (tmp_path / "outer.json").exists()
    assert not (tmp_path / "inner.json").exists()


@pytest.mark.parametrize("exception", [RuntimeError, MemoryError])
def test_exception_records_cublas(monkeypatch, exception):
    def fail(*args):
        raise exception("do not persist exception details")

    monkeypatch.setattr(tuning, "_measure", fail)
    with autotune():
        assert resolve() is None
    record = Linear().decision(N, K, 3)["record"]
    assert record["provider"] == "cublas"
    assert record["reason"] == f"error: {exception.__name__}"
    assert "details" not in json.dumps(record)


@pytest.mark.parametrize(
    "ratio,epsilon,provider",
    [
        (1.029, 0.001, "cublas"),
        (1.04, 0.05, "cublas"),
        (1.03, 0.03, "cublas"),
        (1.0625, 0.0625, "cublas"),
        (1.0626, 0.0625, "cute"),
        (1.031, 0.001, "cute"),
    ],
)
def test_acceptance_margin_and_noise(monkeypatch, ratio, epsilon, provider):
    def measure(*args):
        args[-1].update(qualified(speedup=ratio, epsilon=epsilon))

    monkeypatch.setattr(tuning, "_measure", measure)
    with autotune():
        resolve()
    assert Linear().decision(N, K, 3)["provider"] == provider
    assert (provider == "cute") == dispatch.qualifies(ratio, epsilon)


def test_stale_writers_merge_without_overwriting():
    first, second = qualified(), qualified(speedup=1.3)
    one, two = (tuning.cache_key(HARDWARE, n, K, 0, (3, 4)) for n in (N, N + 8))
    with autotune():
        tuning._write(one, first)
        tuning._CACHES[tuning._path()] = {}  # Second writer's stale in-memory read.
        tuning._write(two, second)
        tuning._CACHES[tuning._path()] = {}
        assert tuning._write(one, second) == first
    tuning._CACHES.clear()
    assert tuning._entries() == {one: first, two: second}


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "empty cache"),
        ('{"format":6,"entries":{}}', "empty cache"),
        ("[]", "empty cache"),
        ('{"format":5,"entries":{"bad":null}}', "invalid records"),
    ],
)
def test_corrupt_cache_warns_once(tmp_path, contents, message):
    (tmp_path / "cache.json").write_text(contents)
    with pytest.warns(UserWarning, match=message) as captured:
        assert tuning._entries() == {}
        assert tuning._entries() == {}
        tuning._CACHES.clear()
        assert tuning._entries() == {}
    assert len(captured) == 1


def test_rank_deterministic_filtering_and_source_ties(monkeypatch):
    # Equal score, worse lexical field order: same-shape table provenance wins.
    same = dispatch.Config(64, 64, 3)
    other = dispatch.Config(64, 64, 2)
    grid = dispatch.Config(64, 128, 2)
    overflow = dispatch.Config(64, 256, 2)
    empty = dispatch.Config(16, 512, 2, 64)
    monkeypatch.setitem(
        dispatch.TUNED, "rtx5090", {(N, K): {(1, 1): same}, (N + 8, K): {(3, 4): other}}
    )
    monkeypatch.setattr(tuning, "bxq_configs", lambda *a: [grid, overflow, empty])
    args = ("rtx5090", 1, N, K, 64, 0, 3)
    ranked = tuning.rank_candidates(*args)
    assert ranked == tuning.rank_candidates(*args) == [same, other, grid]
    assert tuning.rank_candidates(*args[:-1], 1) == [same]
    for kind, sms in (("rtx5090", 170), ("pro6000", 188), ("gb10", 48), (None, 100)):
        result = tuning.rank_candidates(kind, sms, N, K, 64, K, 3)
        assert len(result) <= 3
        assert all(c.shared_bytes_for(8, True) <= 99 * 1024 for c in result)


def test_none_plan_retries_in_context_and_cute_hot_path_is_unchanged(monkeypatch):
    x = SimpleNamespace(get_device=lambda: 0, device=DEVICE)
    op = Linear()
    monkeypatch.setattr(op, "_launch_plan", lambda plan, *args: plan)
    assert op._cute(x, object(), None, None, 3, N, K) is None
    with autotune():
        plan = op._cute(x, object(), None, None, 3, N, K)
        assert plan.launch["source"] == "autotune"
        monkeypatch.setattr(op, "_resolve", forbidden)
        monkeypatch.setattr(tuning, "lookup", forbidden)
        assert op._cute(x, object(), None, None, 4, N, K) is plan


def test_gated_cache_queries_and_plan(monkeypatch):
    seed(stride=K + 8)
    monkeypatch.setattr(tuning, "tune", forbidden)
    op = Linear()
    assert op.gate_fused(N, K, K + 8, 3)
    assert op.workspace_bytes(N, K, 4, gate_stride=K + 8) == 2 * 4 * N * 4
    assert not op.gate_fused(N, K, K + 16, 3)
    plan = resolve(op, stride=K + 8)
    assert plan.launch["source"] == "autotune" and plan.launch["gate_stride"] == K + 8


# Preserve the real CUDA orchestration function for mocked Trial tests below.
_REAL_MEASURE = tuning._measure


@pytest.mark.parametrize(
    "failure", ["compile", "launch", "incorrect", "screen", "noise", "bits", None]
)
def test_measurement_orchestration(monkeypatch, failure):
    calls = []
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)

    class Trial:
        n, k = N, K

        def __init__(self, *args):
            pass

        def call(self, compiled, config, m):
            def run():
                if failure == "launch":
                    raise RuntimeError("launch failed")
                value = 2 if failure == "bits" and m == 3 else 1
                return torch.full((m, N), value, dtype=torch.bfloat16)

            return run

        def correct(self, output, m):
            return failure != "incorrect"

        def benchmark(self, call, m, regime, pairs, calibrate=True):
            calls.append((m, regime, pairs, calibrate))
            return {
                "speedup": 0.9 if failure == "screen" else 1.1,
                "epsilon": 0.2 if failure == "noise" else 0.01,
                "ours_us": 10,
                "cublas_us": 11,
            }

    def compile_candidate(*args):
        if failure == "compile":
            raise RuntimeError("compiler failure")
        return object()

    monkeypatch.setattr(tuning, "_Trial", Trial)
    monkeypatch.setattr(cute, "get_compiled", compile_candidate)
    monkeypatch.setattr(tuning, "_measure", _REAL_MEASURE)
    monkeypatch.setattr(tuning, "rank_candidates", lambda *args: [dispatch.Config()])
    with autotune():
        record = tuning.tune(SimpleNamespace(device=DEVICE), HARDWARE, "rtx5090", N, K, 0, (3, 4))
    assert Linear().decision(N, K, 3)["record"] == record
    assert record["provider"] == ("cute" if failure is None else "cublas")
    assert record["cublas_bf16_reduced_precision"] is True
    if failure in ("compile", "launch"):
        assert record["candidates"][0]["error"] == "RuntimeError"
        assert record["reason"] == "error: RuntimeError"
    if failure not in ("compile", "launch", "incorrect"):
        assert calls[0] == (4, "cold", 5, False)
    if failure in (None, "noise", "bits"):
        assert set(calls[1:]) == {
            (m, regime, 20, True) for m in (3, 4) for regime in ("cold", "engine", "hot")
        }
        assert record["measured_m"] == [3, 4]
    else:
        assert record["measured_m"] == []
    if failure == "screen":
        # The screen loss is confirmed with 20 pairs before cuBLAS is persisted.
        assert calls[1:] == [(4, "cold", 20, True)]
        assert record["regimes"]["cold"]["4"]["pairs"] == 20


@pytest.mark.parametrize("stride", [0, 24])
def test_real_trial_inputs_reference_and_comparator_flags(monkeypatch, stride):
    weight = torch.randn(7, 16, dtype=torch.bfloat16)
    before = weight.clone()
    rng = torch.random.get_rng_state()
    trial = tuning._Trial(weight, stride, 4)
    assert torch.equal(torch.random.get_rng_state(), rng)
    if stride:
        assert trial.gate.stride() == (stride, 1)
        expected = (trial.x.float() * trial.gate.float().sigmoid()).bfloat16()
    else:
        expected = trial.x
    assert torch.equal(trial.a, expected)
    torch.testing.assert_close(trial.ref, expected.double() @ weight.double().T, rtol=0, atol=0)
    torch.testing.assert_close(
        trial.condition, expected.double().abs() @ weight.double().abs().T, rtol=0, atol=0
    )
    output = torch.nn.functional.linear(expected, weight)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    assert trial.correct(output, 4)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    original_linear = tuning.F.linear
    seen = []

    def comparator(x, w):
        seen.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        assert x.data_ptr() == trial.x.data_ptr() and w is weight
        return original_linear(x, w)

    def benchmark(ours, baseline, *args, **kwargs):
        baseline(weight)

    monkeypatch.setattr(tuning.F, "linear", comparator)
    monkeypatch.setattr(tuning, "_benchmark", benchmark)
    trial.benchmark(lambda: output, 4, "cold", 20)
    assert seen == [True]
    assert torch.equal(weight, before)

    def failing_baseline(*args):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(tuning.F, "linear", failing_baseline)
    with pytest.raises(RuntimeError):
        trial.correct(output, 4)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True


class RotationWeight:
    """Track clones without allocating GPU-sized CPU tensors."""

    device = DEVICE

    def __init__(self, nbytes):
        self.shape = (nbytes // 2, 1)

    def clone(self):
        return RotationWeight(self.shape[0] * 2)


@pytest.mark.parametrize(
    "weight_mib,l2_mib,copies,cold_calls,engine_calls,hot_calls",
    [
        (24, 96, 12, 24, 24, 22),
        (80, 96, 4, 8, 8, 7),
        (160, 96, 2, 4, 4, 4),
        (192, 96, 1, 3, 4, 3),  # Exactly twice L2: self-evicting.
        (256, 96, 1, 2, 2, 2),
        (1024, 96, 1, 1, 2, 1),  # One full pattern even above the target.
        (1, 4, 12, 252, 240, 256),  # Cap while retaining whole patterns.
    ],
)
@pytest.mark.parametrize("regime", ["cold", "engine", "hot"])
def test_rotation_formulas(
    monkeypatch, weight_mib, l2_mib, copies, cold_calls, engine_calls, hot_calls, regime
):
    weight = RotationWeight(weight_mib << 20)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(L2_cache_size=l2_mib << 20),
    )
    weights, actual_copies, calls = tuning._rotation(weight, regime)
    assert calls == {"cold": cold_calls, "engine": engine_calls, "hot": hot_calls}[regime]
    assert calls % len(weights) == 0
    assert len(weights) <= calls <= 256
    if regime == "hot":
        assert actual_copies == 0 and weights == [weight]
    else:
        assert actual_copies == copies
        cold = weights if regime == "cold" else weights[::2]
        assert len(cold) == len({id(w) for w in cold}) == copies
        assert all(w is not weight and w.shape == weight.shape for w in cold)
        if weight_mib < 2 * l2_mib:
            assert copies * weight_mib <= 3 * l2_mib + weight_mib
        if regime == "engine":
            assert len(weights) == 2 * copies
            assert all(w is weight for w in weights[1::2])


@pytest.mark.parametrize("regime,copies,calls", [("cold", 5, 10), ("engine", 5, 10), ("hot", 0, 8)])
@pytest.mark.parametrize("calibrate", [False, True])
def test_graph_order_regimes_and_self_pairs(monkeypatch, regime, copies, calls, calibrate):
    ours, baseline = object(), object()
    weight = RotationWeight(64 << 20)
    captured, paired = [], []

    def graph(fn, weights, count):
        captured.append((fn, weights, count))
        return fn, object()

    def pairs(left, right, count, calls):
        paired.append((left, right, count, calls))
        return [10.0, 20.0], [11.0, 22.0]

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=96 << 20)
    )
    monkeypatch.setattr(tuning, "_graph", graph)
    monkeypatch.setattr(tuning, "_paired", pairs)
    result = tuning._benchmark(ours, baseline, regime, 7, weight, calibrate=calibrate)
    assert [g[0] for g in captured] == [ours, baseline]
    assert captured[0][1] is captured[1][1]
    assert [g[2] for g in captured] == [calls, calls]
    assert paired == [(ours, baseline, 7, calls)] + (
        [(baseline, baseline, 20, calls)] if calibrate else []
    )
    assert result["copies"] == copies
    assert result["calls_per_graph"] == calls
    assert result["speedup"] == pytest.approx(1.1)
    assert result["epsilon"] == pytest.approx(abs(10 / 11 - 1) if calibrate else 0)


def test_graph_repeats_rotation_and_reuses_cute_buffers(monkeypatch):
    stream = SimpleNamespace(wait_stream=lambda stream: None)
    graph = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    weight = torch.ones(7, 16, dtype=torch.bfloat16)
    trial = tuning._Trial(weight, 24, 4)
    launches = []
    call = trial.call(lambda *args: launches.append(args), dispatch.Config(splits=2), 3)
    rotation = [weight.clone(), weight, weight.clone(), weight]
    actual_graph, output = tuning._graph(call, rotation, 12)
    assert actual_graph is graph
    assert [id(args[1]) for args in launches[2:]] == [id(w) for w in rotation] * 3
    assert all(args[2] is output for args in launches)
    assert all(args[3] is launches[0][3] for args in launches)
    assert launches[0][3].shape == (2, 3, 7)
    assert all(args[4].data_ptr() == trial.gate.data_ptr() for args in launches)
    call()  # Correctness still defaults to the caller's weight.
    assert launches[-1][1] is weight


def test_event_time_is_per_call():
    seen = []
    graph = SimpleNamespace(replay=lambda: seen.append("replay"))
    start = SimpleNamespace(record=lambda: seen.append("start"), elapsed_time=lambda end: 0.4)
    end = SimpleNamespace(
        record=lambda: seen.append("end"), synchronize=lambda: seen.append("synchronize")
    )
    assert tuning._timed(graph, (start, end), 20) == 20
    assert seen == ["start", "replay", "end", "synchronize"]


def test_unreadable_cache_and_xdg_default(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.delenv("CTI_LINEAR_AUTOTUNE_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert tuning._path() == tmp_path / "ctikernels" / "linear-autotune.json"

    def unreadable(*args):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.warns(UserWarning, match="empty cache"):
        assert tuning._entries() == {}


def test_shipped_dispatch_disables_cache_and_restores_policy(monkeypatch):
    from benchmark.kernel_tuning.linear_smallm import shipped_dispatch

    seed()
    monkeypatch.setattr(tuning, "lookup", forbidden)
    with pytest.raises(RuntimeError), shipped_dispatch():
        assert Linear().decision(N, K, 3)["source"] == "off"
        raise RuntimeError("restore on error")
    assert Linear()._autotune_mode == "cache"


_REAL_DRIVER_VERSION = tuning._driver_version


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("failure", [None, "ours", "baseline", "pairs"])
def test_timing_uses_default_flag_and_restores(monkeypatch, ambient, failure):
    flags = torch.backends.cuda.matmul
    monkeypatch.setattr(flags, "allow_bf16_reduced_precision_reduction", ambient)
    seen = []

    def graph(fn, weights, calls):
        seen.append(fn)
        assert flags.allow_bf16_reduced_precision_reduction is True
        if failure == fn:
            raise RuntimeError("capture failed")
        return fn, object()

    def paired(*args):
        assert flags.allow_bf16_reduced_precision_reduction is True
        if failure == "pairs":
            raise RuntimeError("timing failed")
        return [10], [11]

    monkeypatch.setattr(tuning, "_graph", graph)
    monkeypatch.setattr(tuning, "_paired", paired)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=96 << 20)
    )
    weight = RotationWeight(64 << 20)
    if failure:
        with pytest.raises(RuntimeError):
            tuning._benchmark("ours", "baseline", "cold", 1, weight)
    else:
        result = tuning._benchmark("ours", "baseline", "cold", 1, weight)
        assert result["cublas_bf16_reduced_precision"] is True
    assert seen == (["ours"] if failure == "ours" else ["ours", "baseline"])
    assert flags.allow_bf16_reduced_precision_reduction is ambient


@pytest.mark.parametrize("failure", [None, "load", "init", "get", "raise", "cuda"])
def test_driver_release_nvml_and_fallback_cached(monkeypatch, failure):
    import ctypes

    calls = []

    def init():
        calls.append("init")
        return int(failure == "init")

    def get(buffer, size):
        calls.append("get")
        if failure == "raise":
            raise OSError("driver unavailable")
        buffer.value = b"580.126.09"
        return int(failure == "get")

    def shutdown():
        calls.append("shutdown")
        return 0

    def cuda(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 13000
        return int(failure == "cuda")

    def load(name):
        calls.append(name)
        if name == "libnvidia-ml.so.1":
            if failure in ("load", "cuda"):
                raise OSError("no NVML")
            return SimpleNamespace(
                nvmlInit_v2=init, nvmlSystemGetDriverVersion=get, nvmlShutdown=shutdown
            )
        assert name == "libcuda.so.1"
        return SimpleNamespace(cuDriverGetVersion=cuda)

    monkeypatch.setattr(tuning.ctypes, "CDLL", load)
    _REAL_DRIVER_VERSION.cache_clear()
    try:
        expected = (
            "580.126.09" if failure is None else "unknown" if failure == "cuda" else "cuda13000"
        )
        assert _REAL_DRIVER_VERSION() == expected
        count = len(calls)
        assert _REAL_DRIVER_VERSION() == expected and len(calls) == count
        assert calls.count("shutdown") == int(failure in (None, "get", "raise"))
    finally:
        _REAL_DRIVER_VERSION.cache_clear()


@pytest.mark.parametrize("expire_at", ["trial", "compile", "screen", "qualification"])
def test_mid_key_deadline_aborts_without_cache_and_can_retry(monkeypatch, expire_at):
    now, calls = [0.0], []

    def expire(stage):
        calls.append(stage)
        if stage == expire_at:
            now[0] = 2.0

    class Trial:
        n, k = N, K

        def __init__(self, *args):
            expire("trial")

        def call(self, *args):
            return lambda: torch.ones((4, 1), dtype=torch.bfloat16)

        def correct(self, *args):
            return True

        def benchmark(self, call, m, regime, pairs, calibrate=True):
            expire("qualification" if calibrate else "screen")
            return {"speedup": 1.1, "epsilon": 0.01}

    def compile_candidate(*args):
        expire("compile")
        return object()

    monkeypatch.setattr(tuning.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tuning, "_Trial", Trial)
    monkeypatch.setattr(cute, "get_compiled", compile_candidate)
    monkeypatch.setattr(tuning, "_measure", _REAL_MEASURE)
    monkeypatch.setattr(tuning, "rank_candidates", lambda *args: [dispatch.Config()] * 2)
    with autotune(budget_seconds=1):
        assert (
            tuning.tune(SimpleNamespace(device=DEVICE), HARDWARE, "rtx5090", N, K, 0, (3, 4))
            is None
        )
    assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None
    assert not tuning._path().exists()
    if expire_at == "trial":
        assert calls == ["trial"]
    elif expire_at == "compile":
        assert calls == ["trial", "compile"]
    elif expire_at == "screen":
        assert calls == ["trial", "compile", "screen"]
    monkeypatch.setattr(tuning, "_measure", measurement)
    with autotune(budget_seconds=1):
        assert resolve() is not None


@pytest.mark.parametrize("stride", [0, K])
def test_cached_compile_failure_warns_once_and_falls_back(monkeypatch, stride):
    seed(stride=stride)
    calls = []

    def compile_candidate(*args):
        calls.append(args)
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(cute, "get_compiled", compile_candidate)
    monkeypatch.setattr(inference, "_AUTOTUNE_COMPILE_FAILURES", set(), raising=False)
    with pytest.warns(UserWarning, match="autotune.*compil") as captured:
        assert resolve(stride=stride) is None
        with autotune():
            assert resolve(stride=stride) is None
    assert len(captured) == 1 and len(calls) == 1
    # Queries then report the cuBLAS this process actually runs.
    config, source, record = Linear()._selection("rtx5090", N, K, 4, stride, DEVICE)
    assert config is None and source == "autotune" and record["provider"] == "cublas"


@pytest.mark.parametrize("source", ["table", "forced"])
def test_other_compile_failures_still_raise(monkeypatch, source):
    def fail(*args):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(cute, "get_compiled", fail)
    op = Linear(provider="cute") if source == "forced" else Linear()
    with pytest.raises(RuntimeError, match="compiler failed"):
        resolve(op, n=4096)


@pytest.mark.parametrize("regime", ["cold", "engine"])
def test_rotation_caps_clones_for_tiny_weights(monkeypatch, regime):
    weight = RotationWeight(4096)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(L2_cache_size=96 << 20),
    )
    weights, copies, calls = tuning._rotation(weight, regime)
    assert copies == 64 and calls <= 256 and calls % len(weights) == 0


def test_queries_follow_plan_after_custom_cache_context(tmp_path):
    custom = tmp_path / "custom.json"
    op = Linear()
    with autotune(cache=str(custom)):
        plan = resolve(op)
        op._plans[0, N, K, 4] = plan
    assert plan.launch["source"] == "autotune" and plan.splits == 2
    # The default cache has no entry, but this op still launches the custom-cache plan.
    assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None
    decision = op.decision(N, K, 3)
    assert decision["provider"] == "cute" and decision["source"] == "autotune"
    assert op.workspace_bytes(N, K, 4) == 2 * 4 * N * 4
    assert Linear().decision(N, K, 3)["provider"] == "cublas"
    assert Linear().workspace_bytes(N, K, 4) == 0


def test_gated_queries_follow_plan_after_custom_cache_context(tmp_path):
    op = Linear()
    with autotune(cache=str(tmp_path / "custom.json")):
        assert resolve(op, stride=K + 8).launch["gate_stride"] == K + 8
    assert op.gate_fused(N, K, K + 8, 3)
    assert op.workspace_bytes(N, K, 4, gate_stride=K + 8) == 2 * 4 * N * 4
    assert not Linear().gate_fused(N, K, K + 8, 3)


@pytest.mark.parametrize("pairs", [1, 19])
def test_records_below_twenty_pairs_never_dispatch(pairs):
    record = qualified(pairs=pairs)
    assert record["provider"] == "cublas" and record["reason"] == "fewer than 20 pairs"
    forged = dict(record, provider="cute")
    assert tuning.record_config(forged, (3, 4), "rtx5090") is None
    assert tuning.record_config(qualified(), (3, 4), "rtx5090") == dispatch.Config(splits=2)
    seed(forged)
    assert Linear().decision(N, K, 3)["provider"] == "cublas"


@pytest.mark.parametrize("pairs", [1, 3, 40])
def test_low_pairs_only_shorten_screen(monkeypatch, pairs):
    calls = []

    class Trial:
        n, k = N, K

        def __init__(self, *args):
            pass

        def call(self, compiled, config, m):
            return lambda: torch.ones((m, N), dtype=torch.bfloat16)

        def correct(self, output, m):
            return True

        def benchmark(self, call, m, regime, count, calibrate=True):
            calls.append((regime, count, calibrate))
            return {"speedup": 1.1, "epsilon": 0.01, "ours_us": 10, "cublas_us": 11}

    monkeypatch.setattr(tuning, "_Trial", Trial)
    monkeypatch.setattr(cute, "get_compiled", lambda *args: object())
    monkeypatch.setattr(tuning, "_measure", _REAL_MEASURE)
    monkeypatch.setattr(tuning, "rank_candidates", lambda *args: [dispatch.Config()])
    with autotune(pairs=pairs):
        record = tuning.tune(SimpleNamespace(device=DEVICE), HARDWARE, "rtx5090", N, K, 0, (3, 4))
    assert calls[0] == ("cold", min(5, pairs), False)
    assert {count for _, count, _ in calls[1:]} == {max(20, pairs)}
    assert record["provider"] == "cute"
    assert all(p["pairs"] == max(20, pairs) for r in record["regimes"].values() for p in r.values())


def test_queries_follow_retained_cublas_fallback(tmp_path):
    seed()  # The default cache holds a CuTe winner for this key.
    op = Linear()
    with autotune(cache=str(tmp_path / "empty.json"), budget_seconds=0):
        assert resolve(op) is None  # Budget expired: uncached cuBLAS for this op.
    decision = op.decision(N, K, 3)
    assert decision["provider"] == "cublas" and decision["source"] == "default"
    assert op.workspace_bytes(N, K, 4) == 0
    assert Linear().decision(N, K, 3)["provider"] == "cute"


@pytest.mark.parametrize("pairs", [1, 5])
def test_screen_loss_persists_only_with_twenty_pairs(monkeypatch, pairs):
    counts = []

    class Trial:
        n, k = N, K

        def __init__(self, *args):
            pass

        def call(self, compiled, config, m):
            return lambda: torch.ones((m, N), dtype=torch.bfloat16)

        def correct(self, output, m):
            return True

        def benchmark(self, call, m, regime, count, calibrate=True):
            counts.append(count)
            return {"speedup": 0.5 if count < 20 else 0.9, "epsilon": 0.01}

    monkeypatch.setattr(tuning, "_Trial", Trial)
    monkeypatch.setattr(cute, "get_compiled", lambda *args: object())
    monkeypatch.setattr(tuning, "_measure", _REAL_MEASURE)
    monkeypatch.setattr(tuning, "rank_candidates", lambda *args: [dispatch.Config()])
    with autotune(pairs=pairs):
        record = tuning.tune(SimpleNamespace(device=DEVICE), HARDWARE, "rtx5090", N, K, 0, (3, 4))
    assert counts == [min(5, pairs), 20]
    assert record["provider"] == "cublas" and record["regimes"]["cold"]["4"]["pairs"] == 20
    assert Linear().decision(N, K, 3)["record"] == record


def test_short_evidence_is_never_persisted(monkeypatch):
    def measure(*args):
        args[-1].update(qualified(pairs=19))

    monkeypatch.setattr(tuning, "_measure", measure)
    with autotune():
        assert resolve() is None
    tuning._CACHES.clear()  # Nothing reached disk; a later process retunes.
    assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None


@pytest.mark.parametrize("reason", ["incorrect", "no candidates"])
def test_untimed_cublas_outcomes_stay_in_process(monkeypatch, tmp_path, reason):
    monkeypatch.setattr(tuning, "_measure", lambda *args: args[-1].update(reason=reason))
    with autotune():
        assert resolve() is None
    assert Linear().decision(N, K, 3)["record"]["reason"] == reason
    assert not (tmp_path / "cache.json").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_old_format_cache_is_ignored_and_rewritten(tmp_path, monkeypatch, version):
    path = tmp_path / "cache.json"
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    old = qualified() if version == 2 else dict(qualified(), regimes={})
    path.write_text(json.dumps({"format": version, "entries": {key: old}}))
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE_CACHE", str(path))
    assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None
    with autotune():
        assert resolve().launch["source"] == "autotune"
    data = json.loads(path.read_text())
    assert data["format"] == tuning.FORMAT and data["entries"][key]["regimes"]


@pytest.mark.parametrize(
    ("speedup", "epsilon", "pairs", "edit", "reused"),
    [
        (1.0625, 0.0625, 20, {}, False),  # Equality with epsilon: fails the predicate.
        (1.03, 0.01, 20, {}, False),  # Equality with the margin.
        (1.0626, 0.0625, 20, {}, True),
        (1.1, 0.01, 19, {}, False),
        (1.1, 0.01, 20, {"correct": False}, False),
        (1.1, 0.01, 20, {"deterministic": False}, False),
    ],
)
def test_cached_cute_records_revalidated_on_load(
    tmp_path, monkeypatch, speedup, epsilon, pairs, edit, reused
):
    """A current-format CuTe record is reused only if the shared predicate accepts it now."""
    path = tmp_path / "cache.json"
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    record = qualified()  # Written as qualified, then its evidence is replaced.
    for regime in record["regimes"].values():
        for point in regime.values():
            point.update(speedup=speedup, epsilon=epsilon, pairs=pairs)
    record.update(edit)
    assert record["provider"] == "cute"
    path.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: record}}))
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE_CACHE", str(path))
    assert reused == (
        all(
            p["pairs"] >= 20 and dispatch.qualifies(p["speedup"], p["epsilon"])
            for r in record["regimes"].values()
            for p in r.values()
        )
        and not edit
    )
    found = tuning.lookup(HARDWARE, N, K, 0, (3, 4))
    assert (found is not None) == reused
    assert (tuning.record_config(record, (3, 4), "rtx5090") is not None) == reused
    decision = Linear(autotune="cache").decision(N, K, 3)
    assert decision["provider"] == ("cute" if reused else "cublas")


GB10_HARDWARE = (12, 1, 48, "NVIDIA GB10")


def gb10_record(pairs, ci_low):
    record = qualified()
    for regime in record["regimes"].values():
        for point in regime.values():
            point["pairs"] = pairs
            if ci_low is not None:
                point.update(ci_low=ci_low, ci_high=1.2)
    return record


@pytest.mark.parametrize(
    ("pairs", "ci_low", "accepted"),
    [(20, None, False), (60, None, False), (60, 0.99, False), (59, 1.05, False), (60, 1.05, True)],
)
def test_gb10_qualification_and_cache_load_need_60_pairs_and_ci(
    tmp_path, monkeypatch, pairs, ci_low, accepted
):
    """GB10 decisions follow the offline GB10 rule: >= 60 pairs and bootstrap ci_low > 1,
    both when qualifying a measurement and when reusing a cached record."""
    record = gb10_record(pairs, ci_low)
    record["provider"] = "cublas"
    tuning._qualify(record, "gb10")
    assert (record["provider"] == "cute") == accepted
    # Other devices keep the 20-pair gate without CI.
    assert tuning.record_config(gb10_record(20, None), (3, 4), "rtx5090") is not None
    cached = gb10_record(pairs, ci_low)  # Persisted as CuTe regardless of its evidence.
    assert cached["provider"] == "cute"
    assert (tuning.record_config(cached, (3, 4), "gb10") is not None) == accepted
    path = tmp_path / "gb10.json"
    key = tuning.cache_key(GB10_HARDWARE, N, K, 0, (3, 4))
    path.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: cached}}))
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE_CACHE", str(path))
    assert (tuning.lookup(GB10_HARDWARE, N, K, 0, (3, 4)) is not None) == accepted


def test_gb10_measurement_uses_60_pairs_and_records_ci(monkeypatch):
    seen = []

    def bench(self, call, m, regime, pairs, *, calibrate=True):
        seen.append(pairs)
        return {
            "speedup": 1.2,
            "epsilon": 0.01,
            "ours_us": 1.0,
            "cublas_us": 1.2,
            "ci_low": 1.1,
            "ci_high": 1.3,
        }

    monkeypatch.setattr(
        tuning._Trial, "__init__", lambda self, *a: setattr(self, "n", N) or setattr(self, "k", K)
    )
    monkeypatch.setattr(tuning._Trial, "call", lambda self, *a: lambda: torch.zeros(1))
    monkeypatch.setattr(tuning._Trial, "correct", lambda self, *a: True)
    monkeypatch.setattr(tuning._Trial, "benchmark", bench)
    monkeypatch.setitem(
        __import__("sys").modules,
        "ctikernels.linear.cute",
        SimpleNamespace(get_compiled=lambda *a: object(), mtiles_for=cute.mtiles_for),
    )
    monkeypatch.setattr(tuning, "_release", lambda: None)
    record = tuning._record((3, 4))
    with autotune():  # Measurement records its device in the active context.
        REAL_MEASURE(
            torch.empty(N, K),
            GB10_HARDWARE,
            0,
            (3, 4),
            [3, 4],
            [dispatch.Config(splits=2)],
            20,
            record,
        )
    assert seen[0] == tuning.SCREEN_PAIRS and set(seen[1:]) == {60}
    assert record["provider"] == "cute"
    assert all(p["pairs"] == 60 for r in record["regimes"].values() for p in r.values())


def test_bootstrap_ci_is_shared_and_deterministic():
    ratios = [1.0 + 0.01 * (i % 7) for i in range(60)]
    low, high = dispatch.bootstrap_median_ci(ratios, seed="x")
    assert (low, high) == dispatch.bootstrap_median_ci(ratios, seed="x")
    assert low <= high
    with pytest.raises(ValueError):
        dispatch.bootstrap_median_ci([1.0, float("nan")], seed="x")


def test_errors_stay_in_process(monkeypatch, tmp_path):
    def fail(*args):
        raise RuntimeError("transient")

    monkeypatch.setattr(tuning, "_measure", fail)
    with autotune():
        assert resolve() is None
    assert Linear().decision(N, K, 3)["record"]["reason"] == "error: RuntimeError"
    assert not (tmp_path / "cache.json").exists()
    tuning._CACHES.clear()  # A later process retunes.
    assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None


@pytest.mark.parametrize("warm", ["default", "autotune"])
def test_warmed_route_is_frozen_under_capture_and_other_cache(monkeypatch, tmp_path, warm):
    """A warmed key keeps its route and workspace bound when capture (or a later eager
    call) runs under an autotune context whose cache holds a different CuTe plan."""
    x = SimpleNamespace(get_device=lambda: 0, device=DEVICE)
    other = tmp_path / "other.json"
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    other.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: qualified()}}))
    op = Linear()
    monkeypatch.setattr(op, "_launch_plan", lambda plan, *args: plan)
    if warm == "autotune":  # Warm-up decided cuBLAS through autotune.
        loss = dict(qualified(speedup=0.9), provider="cublas")
        monkeypatch.setattr(tuning, "_measure", lambda *args: args[-1].update(loss))
        with autotune():
            assert op._cute(x, object(), None, None, 3, N, K) is None
    else:  # Eager warm-up outside any context: no decision, cuBLAS.
        assert op._cute(x, object(), None, None, 3, N, K) is None
    route = op.decision(N, K, 3)
    assert route["provider"] == "cublas" and route["source"] == warm
    assert op.workspace_bytes(N, K, max_m=4) == 0
    monkeypatch.setattr(tuning, "tune", forbidden)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with autotune(cache=str(other)):
        assert tuning.record_config(tuning.lookup(HARDWARE, N, K, 0, (3, 4)), (3, 4), "rtx5090")
        assert op._cute(x, object(), None, None, 3, N, K) is None
        assert op.decision(N, K, 3) == route
        assert op.workspace_bytes(N, K, max_m=4) == 0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if warm == "autotune":  # A decided route also stays frozen eagerly.
        with autotune(cache=str(other)):
            assert op._cute(x, object(), None, None, 4, N, K) is None
            assert op.workspace_bytes(N, K, max_m=4) == 0


def test_timing_loop_checks_deadline_between_launches(monkeypatch):
    now, launches = [0.0], []

    def timed(graph, events, calls):
        launches.append(graph)
        now[0] += 0.4
        return 1.0

    monkeypatch.setattr(tuning, "_timed", timed)
    monkeypatch.setattr(tuning.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    with autotune(budget_seconds=1), pytest.raises(tuning._BudgetExpired):
        tuning._paired("ours", "cublas", 20, 1)
    assert len(launches) == 3  # Launched at 0.0, 0.4 and 0.8; stopped at 1.2.
    assert tuning._paired("ours", "cublas", 2, 1) == ([1.0, 1.0], [1.0, 1.0])  # No context.


def _edit(record, path, value):
    *parents, field = path
    target = record
    for name in parents:
        target = target[name]
    target[field] = value


@pytest.mark.parametrize("provider", ["cublas", "cute"])
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("regimes", "cold", "4", "speedup"), float("nan")),
        (("regimes", "hot", "3", "ours_us"), float("inf")),
        (("regimes", "engine", "4", "cublas_us"), -1.0),
        (("regimes", "cold", "3", "speedup"), 0.0),
        (("regimes", "cold", "3", "epsilon"), -0.1),
        (("min_speedup",), float("nan")),
        (("seconds",), float("-inf")),
        (("regimes",), {}),
    ],
)
def test_corrupt_records_are_dropped_on_load_and_warmup_rewrites(
    tmp_path, monkeypatch, provider, path, value
):
    cache = tmp_path / "cache.json"
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    record = dict(qualified(), provider=provider)
    _edit(record, path, value)
    cache.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: record}}))
    with pytest.warns(UserWarning, match="invalid records"):
        assert tuning.lookup(HARDWARE, N, K, 0, (3, 4)) is None
    with autotune():  # Warm-up retunes and the write succeeds.
        assert resolve().launch["source"] == "autotune"
    assert json.loads(cache.read_text())["entries"][key]["provider"] == "cute"


@pytest.mark.parametrize(
    "drop",
    [
        [("engine", None), ("hot", None), ("cold", "5")],  # Upper-edge cold timing only.
        [("engine", None)],
        [("hot", None)],
        [("cold", "5")],
        [("hot", "8")],
        [("engine", "6")],  # A measured interior M without engine evidence.
    ],
)
def test_partial_cute_records_never_dispatch(tmp_path, monkeypatch, drop):
    record = qualified((5, 8), ms=(6,))
    for regime, m in drop:
        if m is None:
            del record["regimes"][regime]
        else:
            del record["regimes"][regime][m]
    assert record["measured_m"] == [5, 6, 8] and record["provider"] == "cute"
    assert tuning.record_config(record, (5, 8), "rtx5090") is None
    assert tuning.record_config(qualified((5, 8), ms=(6,)), (5, 8), "rtx5090") is not None
    key = tuning.cache_key(HARDWARE, N, K, 0, (5, 8))
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: record}}))
    with pytest.warns(UserWarning, match="invalid records"):
        assert Linear().decision(N, K, 6)["provider"] == "cublas"
    no_edge = qualified((5, 8), ms=(6,))
    no_edge["measured_m"] = [6, 8]
    assert tuning.record_config(no_edge, (5, 8), "rtx5090") is None


def test_capture_on_the_call_device_freezes_route(monkeypatch, tmp_path):
    """Capture is read on the call's device, not the thread's current device."""
    current = [0]

    @contextmanager
    def device(index):
        previous, current[0] = current[0], index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "device", device)
    x = SimpleNamespace(get_device=lambda: 1, device=torch.device("cuda:1"))
    other = tmp_path / "other.json"
    key = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    other.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: qualified()}}))
    op = Linear()
    monkeypatch.setattr(op, "_launch_plan", lambda plan, *args: plan)
    assert op._cute(x, object(), None, None, 3, N, K) is None  # Eager warm-up: cuBLAS.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: current[0] == 1)
    monkeypatch.setattr(tuning, "tune", forbidden)
    with autotune(cache=str(other)):
        assert op._cute(x, object(), None, None, 3, N, K) is None
        assert op._plans == {(1, N, K, 4): None}


_REAL_CUBLAS_VERSION = tuning._cublas_version


@pytest.mark.parametrize(
    ("mapped", "expected"),
    [
        (["/lib/cu12/libcublasLt.so.12"], "120900"),
        ([], "120900"),  # Not mapped: torch's CUDA major soname, never a newer one.
        (["/lib/cu12/libcublasLt.so.12", "/lib/cu13/libcublasLt.so.13"], "120900,130100"),
    ],
)
def test_cublas_version_follows_the_library_torch_uses(monkeypatch, mapped, expected):
    versions = {"libcublasLt.so.12": 120900, "libcublasLt.so.13": 130100}

    def load(name):
        version = versions[name.rsplit("/", 1)[-1]]
        return SimpleNamespace(cublasLtGetVersion=lambda: version)

    monkeypatch.setattr(tuning.ctypes, "CDLL", load)
    monkeypatch.setattr(tuning, "_loaded_cublaslt", lambda: mapped)
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    _REAL_CUBLAS_VERSION.cache_clear()
    try:
        assert _REAL_CUBLAS_VERSION() == expected
    finally:
        _REAL_CUBLAS_VERSION.cache_clear()


def test_unparsable_keys_and_huge_numbers_are_dropped(tmp_path):
    good = tuning.cache_key(HARDWARE, N, K, 0, (3, 4))
    huge = tuning.cache_key(HARDWARE, N + 8, K, 0, (3, 4))
    entries = {"[]": qualified(), "{}": qualified(), "x": qualified(), good: qualified()}
    text = json.dumps({"format": tuning.FORMAT, "entries": entries})
    record = json.dumps(dict(qualified(), seconds=0)).replace(
        '"seconds": 0', '"seconds": 1' + "0" * 400
    )
    text = text[:-2] + f', "{huge.replace(chr(34), chr(92) + chr(34))}": {record}}}}}'
    (tmp_path / "cache.json").write_text(text)
    with pytest.warns(UserWarning, match="invalid records"):
        assert list(tuning._entries()) == [good]
