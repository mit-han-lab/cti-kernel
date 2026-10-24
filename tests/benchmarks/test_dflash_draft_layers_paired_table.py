# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import csv
import json
import sys

import pytest

from benchmark.kernel_tuning import dflash_draft_layers_paired_table as table

from ..support.child import run_child


def receipt(*, q=4, shape="draft_noncausal", ratio=0.8, faster="vllm_fa2", status="PASS"):
    return {
        "type": "row",
        "row_id": 0,
        "shape": shape,
        "d": 128 if shape != "q35_target" else 256,
        "Q": q,
        "B": 1,
        "ctx": 100,
        "capacity": 100 + q,
        "op.provider": "cute-splitkv",
        "status": status,
        "market_gate": status,
        "faster_market": faster,
        "correctness": dict.fromkeys(("candidate", "vllm_fa2", "sglang"), "PASS"),
        "references": {
            name: {
                "candidate_median_us": 8 if name == "vllm_fa2" else 9,
                "reference_median_us": 10 if name == faster else 20,
                "median_ratio": ratio if name == faster else 0.45,
                "epsilon": 0.02 if name == faster else 0.03,
                "status": status if name == faster else "PASS",
            }
            for name in ("vllm_fa2", "sglang")
        },
    }


def test_cli_mixed_logs_paired_values_duplicates_and_provenance(tmp_path):
    first, retry = tmp_path / "first.log", tmp_path / "retry.log"
    output = tmp_path / "rows.csv"
    workers = {
        name: {
            "device": "Example GPU",
            "cores": [2, 3],
            "x925_cores": [2, 3],
            "versions": {
                "torch": "test-torch",
                "cti-kernels": "test-cti",
                "flashinfer": "0.7.1" if name == "sglang" else "test-fi",
                "ctikernels_import_path": "ctikernels/__init__.py",
            },
        }
        for name in ("main", "sglang")
    }
    installed = {
        "version": "test-cti",
        "import": "site-packages/ctikernels/__init__.py",
        "init_matches_RECORD": True,
        "source_tree_present": False,
    }
    failed = receipt(q=8, shape="q3_target", ratio=1.15, faster="sglang", status="FAIL")
    failed["capacity"] = 32768
    spliced = json.dumps(failed).replace('"references":', '"refUSDT:probe chatter\nerences":')
    error = {
        "type": "row",
        "shape": "q3_target",
        "Q": 8,
        "B": 2,
        "ctx": 400,
        "capacity": 408,
        "status": "ERROR",
        "error_type": "WorkerError",
    }
    first.write_text(
        "\n".join(
            [
                "runner bootstrap",
                'HOST {"ignored": true}',
                'WHEEL {"file": "artifacts/test.whl", "sha256": "abc123"}',
                "INSTALLED " + json.dumps(installed),
                json.dumps({"type": "header", "workers": workers}),
                json.dumps(receipt(status="FAIL", ratio=1.4)),
                spliced,
                json.dumps(error),
                "PAIRED_DONE rc=1",
            ]
        )
    )
    # The paired median ratio deliberately differs from candidate_us/reference_us.
    replacement = receipt(ratio=0.7)
    retry.write_text(
        "\n".join(
            [
                json.dumps(replacement),
                json.dumps(receipt(shape="q35_target")),
                json.dumps({**replacement, "references": {}}),
                '{"type": "row", "shape": "draft_noncausal",',
            ]
        )
    )
    result = run_child(
        [
            sys.executable,
            "-m",
            table.__name__,
            "--log",
            f"GPU={first}",
            "--log",
            f"GPU={retry}",
            "--log",
            f"GPU2={retry}",
            "--csv",
            str(output),
        ],
        timeout=600,
        check=True,
    )
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert "1 duplicate row(s)" in result.stderr
    assert "2 incomplete row receipt(s)" in result.stderr
    passed, failed, error, hd256 = [row for row in rows if row["device"] == "GPU"]
    assert passed["market_gate"] == "PASS"
    assert passed["median_ratio"] == "0.7"
    assert passed["candidate_us"] == "8"
    assert passed["vllm_flash_attn_us"] == "10"
    assert passed["sglang_fi071_us"] == "20"
    assert passed["faster_market"] == "vllm_flash_attn"
    assert passed["capacity"] == "exact"
    assert failed["candidate_us"] == "9"
    assert failed["median_ratio"] == "1.15"
    assert failed["epsilon"] == "0.02"
    assert failed["other_median_ratio"] == "0.45"
    assert failed["other_epsilon"] == "0.03"
    assert failed["other_reference"] == "vllm_flash_attn"
    assert failed["faster_market"] == "SGLang FI 0.7.1"
    assert failed["sglang_fi071_status"] == "FAIL"
    assert failed["vllm_flash_attn_status"] == "PASS"
    assert failed["op.provider"] == "cute-splitkv"
    assert json.loads(failed["correctness"])["candidate"] == "PASS"
    assert error["status"] == error["market_gate"] == "ERROR"
    assert error["head_dim"] == "128"
    assert error["candidate_us"] == error["median_ratio"] == ""
    assert hd256["head_dim"] == "256"
    markdown = result.stdout
    assert "| GPU | Example GPU | test.whl | abc123 | test-cti |" in markdown
    assert "site-packages/ctikernels/__init__.py | True | True | main: [2, 3]" in markdown
    assert "sglang: flashinfer=0.7.1" in markdown
    assert "main: torch=test-torch" in markdown
    assert "| GPU | q3_target | 100 | 0 / 1 | 0 / 1 | 0.020000 | 1.150000 | 0.020000 |" in markdown
    assert (
        "| GPU | 4, 8 | 1 / 3 | 1 / 3 | 0.020000 | 1.150000 | 0.020000 | 1 | 0 | 96 |" in markdown
    )
    assert (
        "| GPU | q3_target | 8 | 1 | 100 | C32768 | 1.150000 | 0.020000 | SGLang FI 0.7.1 | FAIL |"
        in markdown
    )
    assert "| GPU | q3_target | 8 | 2 | 400 | exact | N/A | N/A | N/A | ERROR |" in markdown


def test_unavailable_reference_and_correctness_failure_are_not_passes(tmp_path, capsys):
    unavailable = receipt()
    unavailable.update(status="N/A", market_gate="N/A", faster_market=None)
    unavailable["references"]["sglang"] = {"status": "N/A"}
    unavailable["correctness"]["sglang"] = "N/A"
    wrong = receipt(q=8)
    wrong.update(status="FAIL", market_gate="FAIL")
    wrong["correctness"]["candidate"] = "FAIL"
    wrong["references"]["sglang"]["status"] = "FAIL"
    path = tmp_path / "log"
    path.write_text("\n".join(map(json.dumps, [unavailable, wrong])))
    rows, provenance = table.read_logs([("GPU", path)])
    assert not capsys.readouterr().err
    assert rows[0]["faster_market"] is None
    assert rows[0]["candidate_us"] is None
    assert rows[0]["sglang_fi071_status"] == "N/A"
    assert table.summary(rows) == ["0 / 2", "0 / 2", 0.02, 0.8, 0.02, 0, 1]
    assert "| GPU | draft_noncausal | 8 | 1 | 100 | exact | 0.800000 |" in table.markdown(
        rows, provenance
    )
    error = {**wrong, "status": "ERROR", "market_gate": "ERROR"}
    path.write_text("\n".join(map(json.dumps, [wrong, error])))
    rows, _ = table.read_logs([("GPU", path)])
    assert len(rows) == 1 and rows[0]["status"] == "ERROR"
    assert "1 duplicate row(s)" in capsys.readouterr().err


def test_import_is_cpu_only():
    run_child(
        [
            sys.executable,
            "-c",
            f"import sys; import {table.__name__}; "
            "assert not any(name in sys.modules for name in ('torch', 'flashinfer', 'ctikernels'))",
        ],
        timeout=600,
        check=True,
    )


@pytest.mark.parametrize("value", ["missing-label", "=path", "GPU="])
def test_log_requires_label_and_path(value):
    with pytest.raises(SystemExit):
        table.main(["--log", value, "--csv", "unused.csv"])
