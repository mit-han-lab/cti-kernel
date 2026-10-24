# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Shape derivation and standard-library-only offline pipeline orchestration."""

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import linear_table_pipeline as pipeline

from ...support.child import run_child


@pytest.fixture
def config():
    return {
        "text_config": {
            "hidden_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "attn_output_gate": True,
            "intermediate_size": 768,
            "vocab_size": 1024,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 64,
            "full_attention_interval": 4,
        }
    }


def test_qwen_shapes_and_tp(config):
    rows = {row["name"]: row for row in pipeline.derive_shapes(config, tp=2)["models"]["model"]}
    expected = {
        "attention.q_gate": (512, 256, 2),
        "attention.k": (64, 256, 2),
        "attention.v": (64, 256, 2),
        "attention.qkv": (640, 256, 2),
        "attention.o": (256, 256, 2),
        "mlp.gate_up": (768, 256, 8),
        "mlp.down": (256, 384, 8),
        "lm_head": (512, 256, 1),
        "gdn.in_proj_qkvz": (640, 256, 6),
        "gdn.in_proj_ba": (8, 256, 6),
        "gdn.in_proj_qkvzba": (648, 256, 6),
        "gdn.out_proj": (256, 256, 6),
    }
    assert {name: (r["N"], r["K"], r["count_per_step"]) for name, r in rows.items()} == expected
    assert rows["gdn.in_proj_qkvzba"]["layout"] == "fused"
    assert rows["attention.q_gate"]["layout"] == "split"
    config["text_config"]["attn_output_gate"] = False
    rows = {r["name"]: r for r in pipeline.derive_shapes(config)["models"]["model"]}
    assert rows["attention.q"]["N"] == 512
    assert rows["attention.qkv"]["N"] == 768
    with pytest.raises(ValueError, match="divisible"):
        pipeline.derive_shapes(config, tp=3)
    # Two KV heads on four ranks are replicated: one head (64 wide) per rank.
    rows = {r["name"]: r for r in pipeline.derive_shapes(config, tp=4)["models"]["model"]}
    assert rows["attention.k"]["N"] == rows["attention.v"]["N"] == 64
    assert rows["attention.qkv"]["N"] == 128 + 2 * 64


def test_layer_types(config):
    config["text_config"]["layer_types"] = ["full_attention"] * 3 + ["linear_attention"] * 5
    rows = {r["name"]: r for r in pipeline.derive_shapes(config)["models"]["model"]}
    assert rows["attention.o"]["count_per_step"] == 3
    assert rows["gdn.out_proj"]["count_per_step"] == 5


# Published config.json dimensions (text_config keys the derivation reads).
QWEN35 = {"num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True, "vocab_size": 248320}
QWEN35.update(linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128)
QWEN35.update(full_attention_interval=4)
GDN4 = ["linear_attention"] * 3 + ["full_attention"]
DRAFT = {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
SLIDING = ["sliding_attention"] * 5 + ["full_attention"]
CONFIGS = {
    "qwen3-8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151936,
    },
    "qwen3.5-9b": {
        "text_config": dict(
            QWEN35,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=16,
            intermediate_size=12288,
            linear_num_value_heads=32,
            layer_types=GDN4 * 8,
        )
    },
    "qwen3.8-27b": {
        "text_config": dict(
            QWEN35,
            hidden_size=5120,
            num_hidden_layers=64,
            num_attention_heads=24,
            intermediate_size=17408,
            linear_num_value_heads=48,
            layer_types=GDN4 * 16,
        )
    },
    # Qwen/Qwen3-30B-A3B config.json (every layer sparse).
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "vocab_size": 151936,
    },
    "qwen3-8b-dflash": dict(
        DRAFT,
        hidden_size=4096,
        num_hidden_layers=5,
        intermediate_size=12288,
        vocab_size=151936,
        layer_types=["full_attention"] * 5,
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
    ),
    "qwen3.5-9b-dflash": dict(
        DRAFT,
        hidden_size=4096,
        num_hidden_layers=6,
        intermediate_size=12288,
        vocab_size=248320,
        layer_types=SLIDING,
        dflash_config={"target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29]},
    ),
    "qwen3.5-27b-dflash": dict(
        DRAFT,
        hidden_size=5120,
        num_hidden_layers=6,
        intermediate_size=17408,
        vocab_size=248320,
        layer_types=SLIDING,
        dflash_config={"target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61]},
    ),
}
EXPECTED = {
    ("qwen3-8b", 1): {
        "attention.q": (4096, 4096, 36),
        "attention.k": (1024, 4096, 36),
        "attention.v": (1024, 4096, 36),
        "attention.qkv": (6144, 4096, 36),
        "attention.o": (4096, 4096, 36),
        "mlp.gate_up": (24576, 4096, 36),
        "mlp.down": (4096, 12288, 36),
        "lm_head": (151936, 4096, 1),
    },
    ("qwen3.5-9b", 1): {
        "attention.q_gate": (8192, 4096, 8),
        "attention.k": (1024, 4096, 8),
        "attention.v": (1024, 4096, 8),
        "attention.qkv": (10240, 4096, 8),
        "attention.o": (4096, 4096, 8),
        "mlp.gate_up": (24576, 4096, 32),
        "mlp.down": (4096, 12288, 32),
        "lm_head": (248320, 4096, 1),
        "gdn.in_proj_qkvz": (12288, 4096, 24),
        "gdn.in_proj_ba": (64, 4096, 24),
        "gdn.in_proj_qkvzba": (12352, 4096, 24),
        "gdn.out_proj": (4096, 4096, 24),
    },
    ("qwen3.8-27b", 1): {
        "attention.q_gate": (12288, 5120, 16),
        "attention.k": (1024, 5120, 16),
        "attention.v": (1024, 5120, 16),
        "attention.qkv": (14336, 5120, 16),
        "attention.o": (5120, 6144, 16),
        "mlp.gate_up": (34816, 5120, 64),
        "mlp.down": (5120, 17408, 64),
        "lm_head": (248320, 5120, 1),
        "gdn.in_proj_qkvz": (16384, 5120, 48),
        "gdn.in_proj_ba": (96, 5120, 48),
        "gdn.in_proj_qkvzba": (16480, 5120, 48),
        "gdn.out_proj": (5120, 6144, 48),
    },
    ("qwen3.8-27b", 2): {
        "attention.q_gate": (6144, 5120, 16),
        "attention.k": (512, 5120, 16),
        "attention.v": (512, 5120, 16),
        "attention.qkv": (7168, 5120, 16),
        "attention.o": (5120, 3072, 16),
        "mlp.gate_up": (17408, 5120, 64),
        "mlp.down": (5120, 8704, 64),
        "lm_head": (124160, 5120, 1),
        "gdn.in_proj_qkvz": (8192, 5120, 48),
        "gdn.in_proj_ba": (48, 5120, 48),
        "gdn.in_proj_qkvzba": (8240, 5120, 48),
        "gdn.out_proj": (5120, 3072, 48),
    },
    ("qwen3-30b-a3b", 1): {
        "attention.q": (4096, 2048, 48),
        "attention.k": (512, 2048, 48),
        "attention.v": (512, 2048, 48),
        "attention.qkv": (5120, 2048, 48),
        "attention.o": (2048, 4096, 48),
        "router.gate": (128, 2048, 48),
        "lm_head": (151936, 2048, 1),
    },
    ("qwen3-30b-a3b", 4): {
        "attention.q": (1024, 2048, 48),
        "attention.k": (128, 2048, 48),
        "attention.v": (128, 2048, 48),
        "attention.qkv": (1280, 2048, 48),
        "attention.o": (2048, 1024, 48),
        "router.gate": (128, 2048, 48),
        "lm_head": (37984, 2048, 1),
    },
    ("qwen3-8b-dflash", 1): {
        "fc": (4096, 20480, 1),
        "context_kv": (2048, 4096, 5),
        "attention.qkv": (6144, 4096, 5),
        "attention.o": (4096, 4096, 5),
        "mlp.gate_up": (24576, 4096, 5),
        "mlp.down": (4096, 12288, 5),
    },
    ("qwen3.5-9b-dflash", 1): {
        "fc": (4096, 32768, 1),
        "context_kv": (2048, 4096, 6),
        "attention.qkv": (6144, 4096, 6),
        "attention.o": (4096, 4096, 6),
        "mlp.gate_up": (24576, 4096, 6),
        "mlp.down": (4096, 12288, 6),
    },
    ("qwen3.5-27b-dflash", 1): {
        "fc": (5120, 40960, 1),
        "context_kv": (2048, 5120, 6),
        "attention.qkv": (6144, 5120, 6),
        "attention.o": (5120, 4096, 6),
        "mlp.gate_up": (34816, 5120, 6),
        "mlp.down": (5120, 17408, 6),
    },
}


@pytest.mark.parametrize(("model", "tp"), EXPECTED, ids=lambda v: str(v))
def test_published_configs_give_every_linear(model, tp):
    rows = pipeline.derive_shapes(CONFIGS[model], tp=tp)["models"]["model"]
    got = {r["name"]: (r["N"], r["K"], r["count_per_step"]) for r in rows}
    assert got == EXPECTED[(model, tp)]
    if "dflash" in model:
        domains = {r["name"]: r["row_domain"] for r in rows}
        assert domains.pop("fc") == domains.pop("context_kv") == "context"
        assert set(domains.values()) == {"block"}


def test_dflash_draft_rows_match_the_builtin_tables():
    # The one-command pipeline and the hand-written acceptance shapes agree.
    builtin = pytest.importorskip("benchmark.kernel_tuning.linear_smallm").builtin_shapes()
    for model, key in (
        ("qwen3-8b-dflash", "dflash-qwen3-8b"),
        ("qwen3.5-9b-dflash", "dflash-qwen3.5-9b"),
        ("qwen3.5-27b-dflash", "dflash-qwen3.5-27b"),
    ):
        rows = pipeline.derive_shapes(CONFIGS[model])["models"]["model"]
        assert rows == builtin[key]


def test_moe_dense_layers_and_shared_expert():
    config = dict(CONFIGS["qwen3-30b-a3b"], num_hidden_layers=4, mlp_only_layers=[0])
    config["decoder_sparse_step"] = 2  # Layers 1 and 3 are sparse; 0 is dense-only.
    rows = {r["name"]: r for r in pipeline.derive_shapes(config)["models"]["model"]}
    assert rows["router.gate"]["count_per_step"] == 2
    assert (rows["mlp.gate_up"]["N"], rows["mlp.gate_up"]["count_per_step"]) == (12288, 2)
    config = dict(config, shared_expert_intermediate_size=512, decoder_sparse_step=1)
    config["mlp_only_layers"] = []
    rows = {r["name"]: r for r in pipeline.derive_shapes(config)["models"]["model"]}
    assert "mlp.gate_up" not in rows
    assert {k: (r["N"], r["K"]) for k, r in rows.items() if k.startswith("shared")} == {
        "shared_expert.gate_up": (1024, 2048),
        "shared_expert.down": (2048, 512),
        "shared_expert.gate": (1, 2048),
    }


def test_sliding_attention_layers_count_as_attention():
    config = dict(CONFIGS["qwen3-8b"], num_hidden_layers=4)
    config["layer_types"] = ["sliding_attention", "full_attention"] * 2
    rows = {r["name"]: r for r in pipeline.derive_shapes(config)["models"]["model"]}
    assert rows["attention.o"]["count_per_step"] == 4
    config["layer_types"][0] = "chunked_attention"
    with pytest.raises(ValueError, match="layer_types"):
        pipeline.derive_shapes(config)
    with pytest.raises(ValueError, match="tp=1"):
        pipeline.derive_shapes(CONFIGS["qwen3.5-9b-dflash"], tp=2)


def test_dry_run_commands_without_torch(tmp_path, config):
    source, logs = tmp_path / "config.json", tmp_path / "logs"
    source.write_text(json.dumps(config))
    # A fresh interpreter actively refuses torch, so an accidental import fails.
    code = """
import runpy, sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('dry-run imported torch')
sys.meta_path.insert(0, NoTorch())
sys.argv = ['linear_table_pipeline.py', *sys.argv[1:]]
runpy.run_path('benchmark/kernel_tuning/linear_table_pipeline.py', run_name='__main__')
"""
    result = run_child(
        [
            sys.executable,
            "-c",
            code,
            "--hf-config",
            str(source),
            "--tp",
            "2",
            "--logs",
            str(logs),
            "--dry-run",
        ],
        timeout=600,
        check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 8
    assert commands[0] == [
        sys.executable,
        str(pipeline.TOOLS / "linear_table_pipeline.py"),
        "--hf-config",
        str(source),
        "--tp",
        "2",
        "--logs",
        str(logs),
        "--stage",
        "shapes",
    ]
    assert commands[1][1].endswith("linear_smallm_bxq_sweep.py")
    assert commands[1][2:6] == [
        "--shapes",
        "8:256,64:256,256:256,256:384,512:256,640:256,648:256,768:256",
        "--m",
        "accept",
    ]
    assert commands[2][2:-2] == [
        "--candidate",
        "--probe",
        str(logs / "probe.log"),
        "--bucketize",
        "--only-measured",
        "--rewrite",
    ]
    for command, regime in zip(commands[3:6], ("cold", "engine", "hot"), strict=True):
        assert command == [
            sys.executable,
            str(pipeline.TOOLS / "linear_smallm.py"),
            "--shapes",
            str(logs / "shapes.json"),
            "--model",
            "all",
            "--m",
            "accept",
            "--accept",
            "--l2",
            regime,
            "--pairs",
            "60",
            ">",
            str(logs / f"accept-{regime}.log"),
        ]
    accept = [str(logs / f"accept-{regime}.log") for regime in ("cold", "engine", "hot")]
    assert commands[6][2:-2] == ["--accept", *accept, "--only-measured", "--rewrite"]
    assert commands[7][2:-2] == ["--accept", *accept, "--markdown"]
    assert not logs.exists()


def test_engine_ms_measure_the_edges_of_their_buckets_only():
    from benchmark.kernel_tuning.linear_smallm import accept_ms, parse_ms

    assert set(parse_ms(pipeline.measured_ms("7,42,63"))) == {5, 7, 8, 41, 42, 48, 57, 63, 64}
    assert set(parse_ms(pipeline.measured_ms("1,2,3"))) == {1, 2, 3, 4}
    assert set(parse_ms(pipeline.measured_ms("bxq"))) == set(accept_ms())


def test_shapes_stage_and_existing_models(tmp_path, config):
    source, logs = tmp_path / "config.json", tmp_path / "logs"
    source.write_text(json.dumps(config))
    assert (
        pipeline.main(["--hf-config", str(source), "--logs", str(logs), "--stage", "shapes"]) == 0
    )
    shapes = json.loads((logs / "shapes.json").read_text())
    assert shapes == pipeline.derive_shapes(config)
    other = tmp_path / "copy"
    assert (
        pipeline.main(
            ["--shapes", str(logs / "shapes.json"), "--logs", str(other), "--stage", "shapes"]
        )
        == 0
    )
    assert json.loads((other / "shapes.json").read_text()) == shapes


def test_actual_cli_compatibility(tmp_path, config):
    from benchmark.kernel_tuning import linear_smallm, linear_smallm_bxq_sweep, linear_smallm_table

    args = SimpleNamespace(
        logs=tmp_path, hf_config=Path("config.json"), shapes=None, tp=1, m="7,42", stage="all"
    )
    commands = pipeline.command_list(args, pipeline.derive_shapes(config))
    assert linear_smallm_bxq_sweep.parse_args(commands[1][1][2:]).shapes
    for _, argv, _ in commands[3:6]:
        parsed = linear_smallm.parse_args(argv[2:])
        # GB10 rows qualify only with >= 60 pairs (linear_smallm_table.GB10_MIN_PAIRS).
        assert parsed.accept and parsed.model == "all"
        assert parsed.pairs == linear_smallm_table.GB10_MIN_PAIRS
    # Parse table flags in report mode, without mutating a dispatch table.
    assert linear_smallm_table.main(commands[-1][1][2:]) == 0


def write_run(path, *, run_id="run", completion_id="run", complete=True):
    rows = [
        {"kind": "accept_begin", "run_id": run_id},
        {"kind": "accept_completion", "run_id": completion_id, "complete": complete},
    ]
    path.write_text("".join("RESULT " + json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    "failure", ["candidate", "accept", "rewrite", "report", "incomplete", "missing", None]
)
def test_pipeline_restores_dispatch_on_failure(monkeypatch, tmp_path, failure):
    monkeypatch.chdir(tmp_path)
    dispatch = Path("python/ctikernels/linear/_dispatch.py")
    dispatch.parent.mkdir(parents=True)
    original = b"# original table\r\n"
    dispatch.write_bytes(original)
    source = tmp_path / "shapes.json"
    source.write_text(json.dumps({"models": {"small": [{"N": 16, "K": 64}]}}))
    seen = []

    def run(argv, *, stdout=None, check):
        stage = (
            "candidate"
            if "--candidate" in argv
            else "accept"
            if "--l2" in argv
            else "report"
            if "--markdown" in argv
            else "rewrite"
            if "--rewrite" in argv
            else "other"
        )
        seen.append(stage)
        if stage in ("candidate", "rewrite"):
            dispatch.write_bytes(stage.encode())
        if stage == "accept":
            stdout.write('RESULT {"kind":"accept_begin","run_id":"run"}\n')
            if failure != "incomplete":
                stdout.write('RESULT {"kind":"accept_completion","run_id":"run","complete":true}\n')
            if failure == "missing":
                Path(stdout.name).unlink()
        return SimpleNamespace(returncode=int(stage == failure))

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    result = pipeline.main(["--shapes", str(source), "--logs", str(tmp_path / "logs")])
    # A finished acceptance run exits nonzero on per-row FAILs; its rows are still evidence.
    succeeded = failure in (None, "accept")
    assert (result == 0) == succeeded
    assert dispatch.read_bytes() == (b"rewrite" if succeeded else original)
    if failure in ("incomplete", "missing"):
        assert "rewrite" not in seen


@pytest.mark.parametrize(
    "bad", ["missing", "incomplete", "mismatch", "no-begin", "trailing-run", "malformed"]
)
@pytest.mark.parametrize("regime", ["cold", "engine", "hot"])
def test_rewrite_requires_all_completed_logs(monkeypatch, tmp_path, bad, regime):
    monkeypatch.chdir(tmp_path)
    dispatch = Path("python/ctikernels/linear/_dispatch.py")
    dispatch.parent.mkdir(parents=True)
    dispatch.write_bytes(b"unchanged")
    source = tmp_path / "shapes.json"
    source.write_text(json.dumps({"models": {"small": [{"N": 16, "K": 64}]}}))
    for name in ("cold", "engine", "hot"):
        write_run(tmp_path / f"accept-{name}.log")
    path = tmp_path / f"accept-{regime}.log"
    if bad == "missing":
        path.unlink()
    elif bad == "incomplete":
        write_run(path, complete=False)
    elif bad == "mismatch":
        write_run(path, completion_id="other")
    elif bad == "no-begin":
        path.write_text('RESULT {"kind":"accept_completion","run_id":"run","complete":true}\n')
    elif bad == "trailing-run":
        with path.open("a") as stream:
            stream.write('RESULT {"kind":"accept_begin","run_id":"unfinished"}\n')
    else:
        path.write_text("RESULT {\n")

    def forbidden(*args, **kwargs):
        pytest.fail("rewrite ran without completed logs")

    monkeypatch.setattr(pipeline.subprocess, "run", forbidden)
    assert (
        pipeline.main(["--shapes", str(source), "--logs", str(tmp_path), "--stage", "rewrite"]) != 0
    )
    assert dispatch.read_bytes() == b"unchanged"


@pytest.mark.parametrize("candidate", [False, True])
def test_only_measured_preserves_other_devices_and_shapes(monkeypatch, tmp_path, candidate):
    from benchmark.kernel_tuning import linear_smallm_table as table
    from ctikernels.linear import _dispatch

    source = tmp_path / "dispatch.py"
    untouched_shape = "        (32, 64): {(1, 2): Config(16, 64, 2)},  # keep shape café\n"
    untouched_device = '    "gb10": { (16, 64): { (1, 2): Config(16, 64, 2) } },  # keep device\n'
    source.write_text(
        table.BEGIN + '\nTUNED = {\n    "rtx5090": {\n'
        "        (16, 64): {(1, 1): Config(16, 64, 2)},\n"
        + untouched_shape
        + "    },\n"
        + untouched_device
        + "}\n"
        + table.END
        + "\n"
    )
    cfg = _dispatch.Config(16, 64, 2)
    monkeypatch.setattr(
        _dispatch,
        "TUNED",
        {
            "rtx5090": {(16, 64): {(1, 1): cfg}, (32, 64): {(1, 2): cfg}},
            "gb10": {(16, 64): {(1, 2): cfg}},
        },
    )
    log = tmp_path / "run.log"
    # A measured fallback removes the old CuTe entry; unrelated rows survive.
    rows = [
        {"kind": "accept_begin", "run_id": "run"},
        {
            "kind": "smallm_accept",
            "run_id": "run",
            "device_class": "rtx5090",
            "N": 16,
            "K": 64,
            "M": 1,
            "provider": "cublas",
            "verdict": "PASS",
        },
        {"kind": "accept_completion", "run_id": "run", "complete": True, "rows": 1},
    ]
    if candidate:
        rows = [{"kind": "smallm_probe", "device_class": "rtx5090", "N": 16, "K": 64}]
        monkeypatch.setattr(
            table,
            "select_configs",
            lambda *args, **kwargs: {"rtx5090": {(16, 64): {(1, 1): _dispatch.Config()}}},
        )
    log.write_text("".join("RESULT " + json.dumps(row) + "\n" for row in rows))
    flags = (
        ["--candidate", "--probe", str(log), "--bucketize"] if candidate else ["--accept", str(log)]
    )
    assert table.main([*flags, "--only-measured", "--rewrite", str(source)]) == 0
    result = source.read_text()
    assert untouched_shape in result and untouched_device in result
    namespace = {"Config": _dispatch.Config}
    exec(result, namespace)
    actual = namespace["TUNED"]
    assert actual["gb10"] == _dispatch.TUNED["gb10"]
    assert actual["rtx5090"][32, 64] == _dispatch.TUNED["rtx5090"][32, 64]
    if candidate:
        assert actual["rtx5090"][16, 64] == {(1, 1): _dispatch.Config()}
    else:
        assert (16, 64) not in actual["rtx5090"]


def test_completion_rejects_wrong_row_count_and_orphan_rows(tmp_path):
    path = tmp_path / "accept.log"
    write_run(path)
    with path.open("a") as stream:
        stream.write('RESULT {"kind":"smallm_accept","run_id":"run"}\n')
    assert not pipeline._finished_acceptance(path)
    path.write_text(
        'RESULT {"kind":"accept_begin","run_id":"run"}\n'
        'RESULT {"kind":"accept_completion","run_id":"run","complete":true,"rows":1}\n'
    )
    assert not pipeline._finished_acceptance(path)


@pytest.mark.parametrize("kind", ["rtx5090", "gb10"])
def test_only_measured_can_add_shape_or_device(tmp_path, kind):
    from benchmark.kernel_tuning import linear_smallm_table as table
    from ctikernels.linear._dispatch import Config

    path = tmp_path / "dispatch.py"
    original = (
        (
            table.BEGIN + '\nTUNED = {"rtx5090": {\n'
            "    (32, 64): {(1, 1): Config()},  # café\n"
            "}}\n" + table.END + "\n"
        )
        .encode()
        .replace(b"\n", b"\r\n")
    )
    path.write_bytes(original)
    table.rewrite(path, {kind: {(16, 64): {(1, 1): Config()}}}, only_measured={(kind, 16, 64)})
    result = path.read_bytes()
    assert "    (32, 64): {(1, 1): Config()},  # café\r\n".encode() in result
    namespace = {"Config": Config}
    exec(result, namespace)
    assert namespace["TUNED"][kind][16, 64] == {(1, 1): Config()}
    assert namespace["TUNED"]["rtx5090"][32, 64] == {(1, 1): Config()}


def test_only_measured_drops_shapes_without_qualifying_buckets(tmp_path):
    from benchmark.kernel_tuning import linear_smallm_table as table
    from ctikernels.linear._dispatch import Config

    path = tmp_path / "dispatch.py"
    original = (
        table.BEGIN + '\nTUNED = {"rtx5090": {\n'
        "    (32, 64): {(1, 1): Config()},\n"
        "    (48, 64): {(2, 2): Config()},\n"
        "}}\n" + table.END + "\n"
    )
    path.write_text(original)
    measured = {("rtx5090", 32, 64), ("rtx5090", 16, 64), ("gb10", 16, 64)}
    table.rewrite(path, {}, only_measured=measured)
    namespace = {"Config": Config}
    exec(path.read_text(), namespace)
    assert namespace["TUNED"] == {"rtx5090": {(48, 64): {(2, 2): Config()}}}
    assert "    (48, 64): {(2, 2): Config()},\n" in path.read_text()
