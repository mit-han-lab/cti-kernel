# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Pure CPU checks for the comparator's statistical contract and case selection."""

import math

import pytest

from benchmark.dflash_sampling.bench import paired_summary
from benchmark.dflash_sampling.vllm_bench import grid, parse_args


def test_ci_uses_log_ratios_sample_sd_and_nearest_rank_p95():
    ratios = [math.exp(-0.1)] * 10 + [math.exp(0.1)] * 10
    # Exactly one extreme self-pair must not affect nearest-rank p95 (rank 19).
    self_pairs = [(1, 1)] * 18 + [(1, 1.02), (1, 100)]
    result = paired_summary(ratios, self_pairs)
    margin = 2.093024 * math.sqrt(0.2 / 19) / math.sqrt(20)
    assert result["gm"] == pytest.approx(1)
    assert result["lo"] == pytest.approx(math.exp(-margin))
    assert result["hi"] == pytest.approx(math.exp(margin))
    assert result["eps"] == pytest.approx(0.02)
    assert not result["claim"]
    assert paired_summary([2] * 20, [(1, 1)] * 20)["claim"]
    with pytest.raises(ValueError, match="twenty"):
        paired_summary([1] * 19, self_pairs)


def test_default_grid_preserves_all_batches_blocks_and_modes():
    rows = list(grid(parse_args(["--comparator-root", "bundle"])))
    assert len(rows) == 432
    assert len(set(rows)) == 432
    assert {r[0] for r in rows} == {1, 2, 3, 4}
    assert {r[1] for r in rows} == {4, 5, 6, 8, 9, 16}
    assert {r[3] for r in rows} == {"G", "T", "KP"}
    assert all(r[2] == 248320 for r in rows if r[-1] == "verify")
    quick = list(grid(parse_args(["--comparator-root", "bundle", "--quick"])))
    assert len(quick) == 12


def test_primary_selection_excludes_v1_and_partial_sglang():
    from benchmark.dflash_sampling.vllm_bench import fixture, summarize_row

    f = fixture(1, 4, 32, "T", "bonus", device="cpu")
    times = {"ours": [10], "vllm": [8], "vllm_v1": [1], "sglang": [6], "sglang_triton_proxy": [0.1]}
    stats = {name: paired_summary([2] * 20, [(1, 1)] * 20) for name in times}
    row = summarize_row(f, "pipeline", times, stats, {}, {}, "pipeline")
    assert row["faster"] == "sglang"
    assert row["faster_incl_v1"] == "vllm_v1"
    assert row["vllm"]["runner"] == "v2"
    assert row["vllm_v1"]["runner"] == "v1"
    row = summarize_row(f, "pipeline", times, stats, {}, {}, "verify_only")
    assert row["faster"] == "vllm"
    assert not row["sglang"]["stats"][-1]
    assert not row["sglang_triton_proxy"]["stats"][-1]
    del times["vllm"]
    row = summarize_row(f, "verify", times, stats, {}, {}, "verify")
    assert row["vllm"]["na"] == "no probability-stage analogue"
    assert row["faster"] == "vllm_v1"


def test_full_grid_stdout_budget():
    import json
    from types import SimpleNamespace

    from benchmark.dflash_sampling.vllm_bench import availability, summarize_row

    total = 0
    for b, q, v, mode, case, stage in grid(parse_args(["--comparator-root", "bundle"])):
        f = SimpleNamespace(b=b, q=q, v=v, mode=mode, case=case)
        names = [name for name, reason in availability(f, stage).items() if reason is None]
        times = {name: [1234.5678] for name in ["ours", *names]}
        stats = {
            name: {"gm": 1.234567, "lo": 0.123456, "hi": 12.34567, "eps": 0.012345, "claim": False}
            for name in names
        }
        greedy = stage == "pipeline" and mode == "G"
        if greedy:
            times["ours_greedy"] = [12.345678]
            times["ours_greedy_torch"] = [23.456789]
            stats["ours_greedy_torch"] = {
                "gm": 2.0,
                "lo": 1.8,
                "hi": 2.2,
                "eps": 0.01,
                "claim": True,
            }
            stats["ours_greedy"] = {
                name: {"gm": 2.0, "lo": 1.8, "hi": 2.2, "eps": 0.01, "claim": True}
                for name in names
            }
        row = summarize_row(
            f,
            stage,
            times,
            stats,
            dict.fromkeys(times, 123),
            dict.fromkeys(names, "tc"),
            "pipeline" if stage == "pipeline" else "verify",
        )
        assert ("ours_greedy" in row) == greedy
        assert ("ours_greedy_torch" in row) == greedy
        assert row["ours_us"] == 1235
        if greedy:
            assert row["ours_greedy"] == {
                "us": 12.35,
                "vs": {name: [2.0, 1.8, 2.2, 0.01, True] for name in names},
                "launches": 123,
            }
            assert row["vllm"]["stats"] == [1.235, 0.1235, 12.35, 0.01235, False]
            assert row["ours_greedy_torch"] == {
                "us": 23.46,
                "stats": [2.0, 1.8, 2.2, 0.01, True],
                "launches": 123,
            }
        total += len(json.dumps(row, separators=(",", ":"))) + 1
    assert total < 250_000


@pytest.mark.parametrize("b,q", [(1, 4), (4, 16)])
@pytest.mark.parametrize("case", ["reject", "bonus"])
@pytest.mark.parametrize("use_torch", [False, True])
def test_greedy_dispatch_outputs_and_repeatability_cpu(monkeypatch, b, q, case, use_torch):
    import torch

    from benchmark.dflash_sampling import vllm_bench as module

    f = module.fixture(b, q, 64, "G", case, device="cpu")
    # Equal maxima must retain the smaller token id, as in the engine's argmax.
    winners = f.logits.argmax(-1)
    f.logits.scatter_(2, (winners + q + 1).unsqueeze(-1), 1024)
    states = []
    commit = module.round_bench.commit

    def record(state):
        if not states:
            assert not state["done_table"].any()
            states.append(state)
        commit(state)

    monkeypatch.setattr(module.round_bench, "commit", record)
    call, outputs = module.ours_greedy_call(f, use_torch=use_torch)
    expected = 0 if case == "reject" else q - 1
    reference = lambda: (winners, torch.full((b,), expected, dtype=torch.int32))
    for _ in range(2 * module.R):
        call()
        assert module.check(f, reference, outputs) == "tokens+counts:ok"
    state = states[0]
    assert state["target_ids"].dtype == torch.int64
    assert state["target_ids"].shape == (b * q,)
    assert (state["start_frontiers"] == 4096).all()
    assert not state["round_done"].any()
    assert not state["done_table"].any()
    assert not state["record_done"].any()
    assert state["stop_token_ids"].numel() == 0
    assert (state["output_ends"] > 4096 + q + 1).all()
    assert (state["frontier_table"][state["request_indices"].long()] == 4096 + expected + 1).all()


@pytest.mark.parametrize("mode,stage", [("G", "pipeline"), ("G", "verify"), ("T", "pipeline")])
def test_run_row_pairs_each_variant_cpu(monkeypatch, mode, stage):
    from types import SimpleNamespace

    import torch

    from benchmark.dflash_sampling import vllm_bench as module

    f = module.fixture(1, 4, 32, mode, "bonus", device="cpu")
    outputs = lambda: (f.logits.argmax(-1), torch.tensor([3], dtype=torch.int32))
    calls = []

    def factory(name):
        def setup(*args):
            def call(**kwargs):
                calls.append(name)

            return call, outputs

        return setup

    monkeypatch.setattr(module, "ours_call", factory("ours"))
    monkeypatch.setattr(
        module,
        "ours_greedy_call",
        lambda f, use_torch=False: factory("ours_greedy_torch" if use_torch else "ours_greedy")(f),
    )
    monkeypatch.setattr(module, "vllm_call", factory("vllm"))
    monkeypatch.setattr(module, "vllm_v1_call", factory("vllm_v1"))
    monkeypatch.setattr(module, "sglang_call", lambda *args: (*factory("sglang")(), stage))
    monkeypatch.setattr(
        module,
        "sglang_triton_proxy_call",
        lambda *args: (*factory("sglang_triton_proxy")(), stage),
    )

    def capture(repeated):
        start = len(calls)
        repeated()
        assert calls[start:] == [calls[start]] * module.R
        return SimpleNamespace(name=calls[start], replay=repeated)

    pairs = []

    def measure(left, right):
        pairs.append((left.name, right.name))
        return [1] * 20, [2] * 20, paired_summary([2] * 20, [(1, 1)] * 20)

    monkeypatch.setattr(module.bench, "capture", capture)
    monkeypatch.setattr(module.bench, "count_launches", lambda call: {"kernel_launches": 2})
    monkeypatch.setattr(module, "measure", measure)
    row = module.run_row(f, stage, SimpleNamespace())
    names = [name for name, reason in module.availability(f, stage).items() if reason is None]
    variants = ["ours", "ours_greedy"] if mode == "G" and stage == "pipeline" else ["ours"]
    expected = [(variant, name) for variant in variants for name in names]
    if len(variants) == 2:
        expected.append(("ours_greedy", "ours_greedy_torch"))
    assert pairs == expected
    assert ("ours_greedy" in row) == (len(variants) == 2)
    assert ("ours_greedy_torch" in row) == (len(variants) == 2)


@pytest.mark.parametrize("mode", ["G", "T", "KP"])
def test_v2_upstream_entry_dispatch_cpu(monkeypatch, mode):
    """Execute intact wrappers on CPU; replace only GPU launches with recorders.

    The external pinned bundle is optional for ordinary CI. No upstream source
    is copied into tests; the configured CPU validation run exercises this test.
    """
    import os
    from types import SimpleNamespace

    import torch

    from benchmark.dflash_sampling import upstream
    from benchmark.dflash_sampling.vllm_bench import fixture, vllm_call

    root = os.environ.get("DFLASH_COMPARATOR_ROOT")
    if not root:
        pytest.skip("set DFLASH_COMPARATOR_ROOT to exercise the pinned V2 wrappers")
    upstream.validate_bundle(root)
    launches = []

    def filter_logits(logits, k, p):
        assert logits.dtype == torch.float32
        assert k.shape == p.shape == (8,)
        assert (k == 20).all() and (p == 0.95).all()
        launches.append("filter")
        return logits

    src = upstream.load_v2(root, filter_logits, device="cpu")
    f = fixture(2, 4, 32, mode, "bonus", device="cpu")

    class RecordingKernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def record(*args, **kwargs):
                launches.append(self.name)
                if self.name == "_gather_draft_sampled_kernel":
                    assert args[2].reshape(2, 4)[:, 1:].equal(f.drafts)
                    assert args[3].tolist() == [100, 101, 102, 103] * 2
                    assert args[5].tolist() == [0] * 4 + [1] * 4
                if self.name == "_compute_local_logits_stats_kernel":
                    assert args[10].shape == (8, 32)
                    assert args[10].dtype == (torch.bfloat16 if mode == "G" else torch.float32)
                    assert args[12] is None  # greedy drafts: no draft logits

            return record

    kernels = (
        (src.rejection, "_gather_draft_sampled_kernel"),
        (src.gumbel, "_temperature_kernel"),
        (src.utils, "_compute_local_logits_stats_kernel"),
        (src.utils, "_rejection_kernel"),
        (src.utils, "_resample_kernel"),
        (src.utils, "_insert_resampled_kernel"),
        (src.batch, "_get_num_sampled_and_rejected_kernel"),
    )
    # Dependency hashing walks the actual JIT call closure without a CUDA driver.
    for module, name in kernels:
        assert getattr(module, name).cache_key
    for module, name in kernels:
        monkeypatch.setattr(module, name, RecordingKernel(name))
    call, _ = vllm_call(f, "pipeline", SimpleNamespace(v2=src))
    call()
    expected = ["_gather_draft_sampled_kernel"]
    if mode != "G":
        expected += ["_temperature_kernel"]
    if mode == "KP":
        expected += ["filter"]
    expected += [name for _, name in kernels[2:]]
    assert launches == expected
    with pytest.raises(ValueError, match="no probability-stage analogue"):
        vllm_call(f, "verify", SimpleNamespace(v2=src))


@pytest.mark.parametrize(
    "mode,case",
    [("G", "reject"), ("T", "reject"), ("KP", "reject"), ("T", "bonus"), ("KP", "bonus")],
)
def test_output_contract(mode, case):
    import torch

    from benchmark.dflash_sampling.vllm_bench import check, fixture

    f = fixture(2, 4, 32, mode, case, device="cpu")
    tokens = f.logits.argmax(-1)
    counts = torch.full((2,), 0 if case == "reject" else 3, dtype=torch.int32)
    ours = lambda: (tokens, counts)
    assert check(f, ours, ours) == "tokens+counts:ok"
    assert check(f, ours, lambda: (tokens, counts + 1)) == "FAIL"
    # A wrong correction/bonus token fails even when the counts match.
    assert check(f, ours, lambda: (tokens + 1, counts)) == "FAIL"
    wrong = tokens.clone()
    wrong[:, 0 if case == "reject" else 3] += 1
    assert check(f, ours, lambda: (wrong, counts)) == "FAIL"


@pytest.mark.parametrize(
    "mode,stage,kp_available",
    [
        ("T", "pipeline", True),
        ("KP", "pipeline", True),
        ("T", "verify", True),
        ("KP", "pipeline", False),
    ],
)
def test_sglang_production_dispatch_and_fixed_checks_cpu(mode, stage, kp_available):
    from types import SimpleNamespace

    import torch

    from benchmark.dflash_sampling import vllm_bench as module
    from ctikernels.sampling._dflash.reference import ACCEPT, BONUS, CORRECTION, uniform

    f = module.fixture(2, 4, 32, mode, "bonus", device="cpu")
    counts = torch.tensor([0, 3], dtype=torch.int32)
    calls = []
    builders = []

    def verify(**kwargs):
        calls.append(kwargs)
        assert kwargs["candidates"].dtype == torch.int64
        assert kwargs["candidates"].tolist() == [[0, 1, 2, 3]] * 2
        assert kwargs["threshold_single"] == kwargs["threshold_acc"] == 1.0
        assert kwargs["next_token_logits"].shape == (8, 32)
        assert kwargs["sampling_info"].need_top_k_sampling == (mode == "KP")
        return counts, torch.tensor([7, 8])

    def commit(candidates, accepted, bonus):
        assert accepted is counts
        assert bonus.tolist() == [7, 8]
        return candidates.clone(), accepted + 1

    def factory(builder):
        builders.append(builder)
        return SimpleNamespace(verify=verify, commit=commit)

    src = SimpleNamespace(sgl_kp_available=kp_available, sgl_verify_factory=factory)
    call, outputs, scope = module.sglang_call(f, stage, src)
    assert scope == ("verify_only" if not kp_available else stage)
    if scope == "pipeline":
        assert builders == [None]
    else:
        assert builders[0]() is f.p
    call()
    assert calls[-1]["uniform_samples"] is None
    assert calls[-1]["uniform_samples_for_final_sampling"] is None
    fixed = module.sglang_check_uniforms(f, counts)
    assert fixed["uniform_samples"].tolist() == [
        [uniform(42, 2**32 + b, ACCEPT, i) for i in range(4)] for b in range(2)
    ]
    assert fixed["uniform_samples_for_final_sampling"].tolist() == [
        uniform(42, 2**32, CORRECTION, 0),
        uniform(42, 2**32 + 1, BONUS, 0),
    ]
    call(**fixed)
    assert calls[-1]["uniform_samples"] is fixed["uniform_samples"]
    assert outputs()[1] is counts


def test_sglang_upstream_chain_rng_gather_commit_cpu(monkeypatch):
    """Execute upstream wrappers on CPU, replacing only the CUDA verifier."""
    import os
    from types import SimpleNamespace

    import torch

    from benchmark.dflash_sampling import upstream
    from benchmark.dflash_sampling.vllm_bench import fixture, sglang_call

    root = os.environ.get("DFLASH_COMPARATOR_ROOT")
    if not root:
        pytest.skip("set DFLASH_COMPARATOR_ROOT to exercise pinned DFlash wrappers")
    upstream.validate_bundle(root)
    f = fixture(2, 4, 32, "T", "bonus", device="cpu")
    events = []
    random = torch.rand

    def rand(shape, **kwargs):
        events.append(("rand", shape))
        return random(shape, **kwargs)

    monkeypatch.setattr(torch, "rand", rand)

    def probabilities(**kwargs):
        events.append("probabilities")
        return f.p

    def tree(**kw):
        events.append("tree")
        assert kw["candidates"].tolist() == [[0, 1, 2, 3]] * 2
        assert kw["retrive_index"].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert kw["retrive_next_token"].tolist() == [[1, 2, 3, -1]] * 2
        assert kw["retrive_next_sibling"].tolist() == [[-1] * 4] * 2
        assert kw["retrive_index"].dtype == torch.int64
        assert kw["accept_index"].dtype == torch.int32
        assert kw["predicts"].dtype == kw["accept_token_num"].dtype == torch.int32
        assert kw["uniform_samples"].shape == (2, 4)
        assert kw["uniform_samples_for_final_sampling"].shape == (2,)
        assert kw["threshold_single"] == kw["threshold_acc"] == 1
        assert kw["deterministic"] is True
        assert kw["target_probs"] is f.p
        assert not kw["draft_probs"].any()
        kw["draft_probs"].fill_(1)  # The next call must get fresh zero scratch.
        # Mixed lengths exercise global retrieval offsets and commit boundary.
        kw["accept_token_num"].copy_(torch.tensor([0, 2], dtype=torch.int32))
        kw["accept_index"].fill_(-100)  # Unused slots must never be gathered.
        kw["accept_index"][0, 0] = 0
        kw["accept_index"][1, :3] = torch.tensor([4, 5, 6], dtype=torch.int32)
        kw["predicts"].fill_(-99)
        kw["predicts"][0] = 9
        kw["predicts"][6] = 10

    src = SimpleNamespace(
        sgl_kp_available=True,
        sgl_verify_factory=lambda builder: upstream.load_sglang_verify(root, probabilities, tree),
    )
    call, outputs, _ = sglang_call(f, "pipeline", src)
    for _ in range(2):
        events.clear()
        call()
        assert events == [("rand", (2, 4)), ("rand", (2,)), "probabilities", "tree"]
        out, counts = outputs()
        assert counts.tolist() == [0, 2]
        assert out[0, 0] == 9
        assert out[1, :3].tolist() == [1, 2, 10]
    events.clear()
    call(
        uniform_samples=torch.full((2, 4), 0.5),
        uniform_samples_for_final_sampling=torch.full((2,), 0.5),
    )
    assert events == ["probabilities", "tree"]


def test_sglang_loader_skips_server_initializers_cpu(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from benchmark.dflash_sampling import upstream

    upstream.load_sglang_sampling.cache_clear()
    package = tmp_path / "sglang"
    for suffix in ("", "kernels", "kernels/ops", "kernels/ops/speculative", "srt", "srt/utils"):
        directory = package / suffix
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text(
            "raise AssertionError('server initializer executed')\n"
        )
    (package / "kernels/ops/speculative/sampling.py").write_text("production_entry = True\n")
    previous = {
        key: value
        for key, value in sys.modules.items()
        if key == "sglang" or key.startswith("sglang.")
    }
    for key in previous:
        del sys.modules[key]
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(upstream, "load_definitions", lambda *a, **kw: SimpleNamespace())
    try:
        module = upstream.load_sglang_sampling(tmp_path)
        assert module.production_entry
        assert (tmp_path / "cache").is_dir()
        with pytest.raises(RuntimeError, match="outside this comparator bundle"):
            upstream.load_sglang_sampling(tmp_path / "another")
    finally:
        for key in list(sys.modules):
            if key == "sglang" or key.startswith("sglang."):
                del sys.modules[key]
        sys.modules.update(previous)
        upstream.load_sglang_sampling.cache_clear()
