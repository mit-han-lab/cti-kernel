# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.kernel_tuning import nvfp4_dispatch
from ctikernels.linear._nvfp4_dispatch import (
    FUSED,
    TRITON,
    UNFUSED,
    Cute,
    FlashInfer,
    Small,
)
from ctikernels.linear._nvfp4_quack import Config

DEVICE = {"device": "Test GPU", "capability": [12, 0]}


def _line(op: str, m: int, path: str, config: str | None, us: float) -> str:
    shape = {"H": 64, "I": 32} if op == "gated" else {"N": 64, "K": 32}
    record = {**DEVICE, "op": op, **shape, "M": m, "path": path, "config": config, "us": us}
    return "BENCH " + json.dumps(record) + "\n"


def _emitted(tmp_path: Path, lines: list[str], capsys) -> dict:
    log = tmp_path / "bench.log"
    log.write_text("noise\n" + "".join(lines))
    nvfp4_dispatch.emit([str(log)])
    namespace = {"_Q": Config, "_S": Small, "FUSED": FUSED, "UNFUSED": UNFUSED}
    namespace["TRITON"] = TRITON
    namespace["_F"] = FlashInfer
    namespace["Cute"] = Cute
    exec(capsys.readouterr().out, namespace)  # the emitter prints Python literals
    return namespace


def test_runs_of_winners_become_plan_entries(tmp_path: Path, capsys) -> None:
    lines = [
        _line("gated", 1, "triton", "t2n32k128q", 5.0),
        _line("gated", 1, "fused", "p128x128", 9.0),
        _line("gated", 2, "triton", "t2n32k128q", 5.0),
        _line("gated", 2, "fused", "p128x128", 9.0),
        _line("gated", 4, "fused", "p128x128", 9.0),
        _line("gated", 4, "unfused", "c128x128", 7.0),
        _line("gated", 4, "bf16", None, 1.0),  # baselines never win
        _line("gated", 4, "ours", None, 0.5),
        _line("gated", 4, "sglang", None, 0.1),
        _line("linear", 8, "sglang", None, 0.1),
        _line("linear", 8, "quack", "p128x128", 4.0),
        _line("linear", 8, "quack", "c128x128", 3.95),  # within the margin: the default stays
    ]
    tables = _emitted(tmp_path, lines, capsys)
    device = ("Test GPU", (12, 0))
    small = Small(*nvfp4_dispatch.SMALL["t2n32k128q"])
    assert tables["GATED"][device][(64, 32)] == (
        (2, TRITON, small),
        (4, UNFUSED, Config(*nvfp4_dispatch.CONFIGS["c128x128"])),
    )
    assert "LINEAR" not in tables


def test_the_previous_winner_keeps_a_batch_within_the_margin(tmp_path: Path, capsys) -> None:
    lines = [
        _line("gated", 1, "triton", "t2n32k128q", 5.0),
        _line("gated", 1, "triton", "t4n32k128q", 5.1),
        _line("gated", 1, "fused", "p128x128", 9.0),
        _line("gated", 2, "triton", "t2n32k128q", 5.1),
        _line("gated", 2, "triton", "t4n32k128q", 5.0),
        _line("gated", 2, "fused", "p128x128", 9.0),
    ]
    tables = _emitted(tmp_path, lines, capsys)
    small = Small(*nvfp4_dispatch.SMALL["t2n32k128q"])
    assert tables["GATED"][("Test GPU", (12, 0))][(64, 32)] == ((2, TRITON, small),)


def test_flashinfer_winners_emit_unfused_entries(tmp_path, capsys):
    lines = [
        _line("linear", 1, "quack", "p128x128", 10),
        _line("linear", 1, "flashinfer", "flashinfer", 5),
        _line("gated", 1, "fused", "p128x128", 10),
        _line("gated", 1, "unfused", "flashinfer", 5),
        _line("gated", 1, "sglang", None, 1),
    ]
    tables = _emitted(tmp_path, lines, capsys)
    device = ("Test GPU", (12, 0))
    assert "LINEAR" not in tables
    assert tables["GATED"][device][(64, 32)] == ((1, UNFUSED, FlashInfer()),)


def test_flashinfer_benchmark_candidate_is_selectable():
    from benchmark.kernel_tuning.nvfp4 import CONFIGS, _candidates

    assert list(_candidates({"flashinfer": CONFIGS["flashinfer"]}, {}, 128, 512)) == [
        ("flashinfer", "flashinfer", FlashInfer())
    ]


@pytest.mark.parametrize("candidates", [False, True])
def test_linear_records_emit_nothing(tmp_path, capsys, candidates):
    records = []
    for device, capability in (("NVIDIA GB10", (12, 1)), ("NVIDIA GeForce RTX 5090", (12, 0))):
        for n, k in ((2048, 512), (4096, 12288), (5120, 17408), (256, 256)):
            for m in (1, 8, 16, 32, 4096, 8192):
                for provider, config in (
                    ("quack", [128, 128, True, True]),
                    ("cute", {"n_tile": 64, "split_k": 8}),
                ):
                    record = {
                        "device": device,
                        "capability": capability,
                        "op": "linear",
                        "N": n,
                        "K": k,
                        "M": m,
                        "provider": provider,
                        "path": provider,
                        "config": config if candidates else "p128x128",
                        "us": 1,
                        "median_us": 1,
                        "current": provider == "quack",
                    }
                    records.append(("CAND " if candidates else "BENCH ") + json.dumps(record))
    log = tmp_path / "linear.log"
    log.write_text("\n".join(records))
    emit = nvfp4_dispatch.emit_candidates if candidates else nvfp4_dispatch.emit
    emit([str(log)])
    namespace = {"_F": FlashInfer, "Cute": Cute}
    exec(capsys.readouterr().out, namespace)
    assert "LINEAR" not in namespace
    assert namespace["GATED"] == {}
