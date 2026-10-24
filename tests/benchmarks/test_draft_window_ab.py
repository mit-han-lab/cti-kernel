# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only checks for paired draft-window benchmarking helpers."""

import builtins
import importlib
import sys
from types import ModuleType

import pytest

from benchmark.kernel_tuning import draft_window_ab as bench


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("candidate=trees/candidate", ("candidate", "trees/candidate", {})),
        (
            "baseline=trees/baseline@CTI_DRAFT_SPLITS=4+CTI_DRAFT_MERGE=tree",
            ("baseline", "trees/baseline", {"CTI_DRAFT_SPLITS": "4", "CTI_DRAFT_MERGE": "tree"}),
        ),
        ("trial=tree@OPTION=a=b+EMPTY=+", ("trial", "tree", {"OPTION": "a=b", "EMPTY": ""})),
    ],
)
def test_parse_arm(spec, expected):
    assert bench.parse_arm(spec) == expected


def test_parse_args_preserves_arm_order_and_defaults():
    specs = ["candidate=trees/new@CTI_DRAFT_SPLITS=4", "baseline=trees/old"]
    args = bench.parse_args(["--arm", specs[0], "--arm", specs[1]])
    assert args.arm == specs
    assert args.masks == "full,c4096,n2048"
    assert args.batches == "1,2,3,4"
    assert args.ctxs == "128,512,4096"
    assert args.lens is None
    assert (args.pairs, args.rounds, args.cap) == (60, 3, 32768)
    assert args.heads == "32,8,128"
    assert args.label == ""


def test_parse_args_overrides():
    args = bench.parse_args(
        [
            "--arm",
            "candidate=tree",
            "--masks",
            "dsv4",
            "--batches",
            "2,4",
            "--ctxs",
            "100,200",
            "--lens",
            "128,256",
            "--pairs",
            "23",
            "--rounds",
            "2",
            "--cap",
            "4096",
            "--heads",
            "16,4,128",
            "--label",
            "trial",
        ]
    )
    assert (args.masks, args.batches, args.ctxs, args.lens) == ("dsv4", "2,4", "100,200", "128,256")
    assert (args.pairs, args.rounds, args.cap) == (23, 2, 4096)
    assert (args.heads, args.label) == ("16,4,128", "trial")


def test_load_arm_rewrites_imports_without_touching_original(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "python" / "ctikernels"
    (package / "attention").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "attention" / "__init__.py").write_text("")
    (package / "x.py").write_text("y = object()\n")
    (package / "attention" / "draft.py").write_text(
        "from ctikernels.x import y\n"
        "import ctikernels.x\n"
        "\n"
        "class DraftAttention:\n"
        "    imported_y = y\n"
        "    imported_package = ctikernels\n"
        "    imported_module = ctikernels.x\n"
    )
    original_files = {path.relative_to(source): path.read_bytes() for path in source.rglob("*.py")}
    original = ModuleType("ctikernels")
    original_x = ModuleType("ctikernels.x")
    original_x.y = object()
    original.x = original_x
    monkeypatch.setitem(sys.modules, "ctikernels", original)
    monkeypatch.setitem(sys.modules, "ctikernels.x", original_x)
    original_attributes = vars(original).copy()
    original_x_attributes = vars(original_x).copy()
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "path", sys.path[:])
    # Keep imports isolated from the rest of the test suite, including on failure.
    names = ("ab_test_candidate", "ab_test_baseline")
    prefixes = tuple("ctikernels_" + name for name in names)
    try:
        loaded = []
        for name, prefix in zip(names, prefixes, strict=True):
            draft_attention = bench.load_arm(name, source)
            loaded.append(draft_attention)
            assert draft_attention.__module__ == prefix + ".attention.draft"
            assert draft_attention.imported_package is sys.modules[prefix]
            assert draft_attention.imported_module is sys.modules[prefix + ".x"]
            assert draft_attention.imported_y is sys.modules[prefix + ".x"].y
            assert draft_attention.imported_y is not original_x.y
            copied = cwd / "_arms" / name / prefix / "attention" / "draft.py"
            assert copied.is_file()
            assert f"from {prefix}.x import y" in copied.read_text()
            assert f"import {prefix}.x" in copied.read_text()
        assert loaded[0].imported_y is not loaded[1].imported_y
        assert sys.modules["ctikernels"] is original
        assert sys.modules["ctikernels.x"] is original_x
        assert vars(original) == original_attributes
        assert vars(original_x) == original_x_attributes
        assert {
            path.relative_to(source): path.read_bytes() for path in source.rglob("*.py")
        } == original_files
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                del sys.modules[name]


@pytest.mark.parametrize("ratio", [0.75, 1.0, 1.25])
def test_bootstrap_constant_ratios(ratio):
    assert bench.bootstrap([ratio] * 8) == (ratio, ratio, ratio)


def test_bootstrap_known_ratios():
    ratios = [0.7, 1.4, 1.0, 0.9, 1.2, 1.1]
    middle, low, high = bench.bootstrap(ratios)
    assert middle == 1.1  # Upper middle sample, not the average of the middle two.
    assert low <= middle <= high
    assert bench.bootstrap(ratios) == (middle, low, high)


def test_module_import_does_not_import_torch(monkeypatch):
    real_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("benchmark module imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    importlib.reload(bench)


def test_pair_records_reproduce_bootstrap():
    import json

    timings = [(2.0, 3.0, 2.5), (4.0, 3.0, 3.5), (3.0, 4.0, 4.5), (2.5, 2.0, 3.0)]
    records = [
        json.loads(
            json.dumps(
                bench.pair_record(
                    "full",
                    2,
                    512,
                    index // 2,
                    index,
                    dict(zip(("candidate", "base", "base2"), values, strict=True)),
                )
            )
        )
        for index, values in enumerate(timings)
    ]
    for index, record in enumerate(records):
        assert (record["type"], record["mask"], record["B"], record["L"]) == (
            "pair",
            "full",
            2,
            512,
        )
        assert (record["round"], record["pair"]) == (index // 2, index)
        assert record["candidate"] == "candidate"
        assert record["ratio_vs_candidate"]["candidate"] == 1
    ratios = [r["us"]["base"] / r["us"]["candidate"] for r in records]
    assert ratios == [r["ratio_vs_candidate"]["base"] for r in records]
    assert bench.bootstrap(ratios) == bench.bootstrap(
        ratios,
        it=records[0]["bootstrap_resamples"],
        seed=records[0]["bootstrap_seed"],
    )


def test_qwen3_moe_masks():
    for mask, causal in (("q3moe", False), ("q3moe_causal", True)):
        assert bench.MASKS[mask][:3] == (16, causal, None)
        assert bench.mask_heads(mask, (64, 1, 256)) == (32, 4, 128)
    assert bench.mask_heads("dsv4", (64, 1, 256)) == (64, 1, 256)


def test_benchmark_emits_each_pair_and_reconstructable_ci(monkeypatch):
    from types import SimpleNamespace

    records = []
    monkeypatch.setattr(bench, "emit", records.append)
    monkeypatch.setattr(bench, "setup_buffers", lambda *args: SimpleNamespace(num_layers=5))
    monkeypatch.setattr(bench, "reference", lambda *args: None)
    monkeypatch.setattr(bench, "build_graph", lambda *args: (object(), {}, object()))

    def time_pairs(graphs, names, times, pairs, num_layers, rng):
        for _ in range(pairs):
            i = len(times["candidate"])
            times["candidate"].append(2.0 + i)
            times["base"].append(3.0 + i)
            times["base2"].append(3.2 + i)

    monkeypatch.setattr(bench, "time_pairs", time_pairs)
    result = bench.benchmark_case(
        {name: (None, {}) for name in ("candidate", "base", "base2")},
        2,
        496,
        "full",
        SimpleNamespace(cap=1024, rounds=2, pairs=6),
        (32, 8, 128),
        "cpu",
        __import__("random").Random(7),
    )
    assert len(records) == result["pairs"] == 6
    assert [r["round"] for r in records] == [0, 0, 0, 1, 1, 1]
    assert [r["pair"] for r in records] == list(range(6))
    assert all(r["L"] == 512 and r["B"] == 2 for r in records)
    for name in ("base", "base2"):
        median, low, high = bench.bootstrap(
            [r["us"][name] / r["us"]["candidate"] for r in records],
            it=result["bootstrap_resamples"],
            seed=result["bootstrap_seed"],
        )
        assert result["vs_" + name] == {
            "speed_median": median,
            "ci_low": low,
            "ci_high": high,
            "not_slower": high >= 1,
        }
