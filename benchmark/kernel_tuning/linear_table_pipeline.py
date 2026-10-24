# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Model config -> shapes -> probe -> paired acceptance -> qualified Linear table.

Run from the repository root. Dry-run uses only the standard library. Candidate
rewrites are provisional until all three acceptance regimes have been re-gated.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

TOOLS = Path("benchmark/kernel_tuning")


def derive_shapes(config, tp=1):
    """Decode projections, with column/row parallel dimensions divided by TP."""
    if type(tp) is not int or tp < 1:
        raise ValueError("tp must be a positive integer")
    config = config.get("text_config", config)

    def positive(name, default=None):
        value = config.get(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def shard(value):
        if value % tp:
            raise ValueError("sharded dimensions must be divisible by tp")
        return value // tp

    h = positive("hidden_size")
    layers = positive("num_hidden_layers")
    heads = positive("num_attention_heads")
    dim = positive("head_dim", h // heads)
    kv = positive("num_key_value_heads", heads)
    gated = bool(config.get("attn_output_gate", config.get("attention_output_gate", False)))
    gdn = "linear_num_key_heads" in config
    if "layer_types" in config:
        types = config["layer_types"]
        # Sliding-window layers use the same projections as full attention.
        if len(types) != layers or any(
            t not in ("full_attention", "sliding_attention", "linear_attention") for t in types
        ):
            raise ValueError("layer_types must describe each full/sliding/linear attention layer")
        attn = layers - types.count("linear_attention")
    else:
        attn = layers // positive("full_attention_interval", 4) if gdn else layers
    rows = []

    def add(name, n, k, count, **metadata):
        if count:
            rows.append(dict(name=name, N=n, K=k, count_per_step=count, **metadata))

    if "dflash_config" in config:
        # DFlash draft: fc projects the concatenated target hidden states (context
        # rows), context_kv projects them to K/V per layer; block rows (M = B*Q) run
        # the fused qkv, o and MLP. The draft shares the target's lm_head.
        if tp != 1:
            raise ValueError("DFlash draft shapes are derived at tp=1")
        if gdn or gated:
            raise ValueError("DFlash drafts are plain (ungated) attention models")
        targets = config["dflash_config"].get("target_layer_ids")
        if not isinstance(targets, list) or not targets:
            raise ValueError("dflash_config.target_layer_ids must be a non-empty list")
        q, v = heads * dim, kv * dim
        add("fc", h, h * len(targets), 1, row_domain="context")
        add("context_kv", 2 * v, h, layers, row_domain="context")
        add("attention.qkv", q + 2 * v, h, layers, row_domain="block")
        add("attention.o", h, q, layers, row_domain="block")
        ffn = positive("intermediate_size")
        add("mlp.gate_up", 2 * ffn, h, layers, row_domain="block")
        add("mlp.down", h, ffn, layers, row_domain="block")
        return {"models": {"model": rows}}

    # Fewer KV heads than ranks are replicated: each rank keeps one.
    if kv < tp and tp % kv:
        raise ValueError("tp above the KV heads must be divisible by them")
    q, v = shard(heads * dim), (shard(kv * dim) if kv >= tp else dim)
    add(
        "attention.q_gate" if gated else "attention.q",
        q * (2 if gated else 1),
        h,
        attn,
        layout="split",
    )
    add("attention.k", v, h, attn, layout="split")
    add("attention.v", v, h, attn, layout="split")
    add("attention.qkv", q * (2 if gated else 1) + 2 * v, h, attn, layout="fused")
    add("attention.o", h, q, attn)
    if "num_experts" in config:
        # MoE blocks (HF rule): routed experts are fused MoE kernels, not Linear; the
        # replicated router and an optional shared expert are. Other layers are dense.
        experts = positive("num_experts")
        step = positive("decoder_sparse_step", 1)
        dense_only = set(config.get("mlp_only_layers") or ())
        sparse = sum(i not in dense_only and (i + 1) % step == 0 for i in range(layers))
        add("router.gate", experts, h, sparse)
        shared = config.get("shared_expert_intermediate_size")
        if shared:
            shared = positive("shared_expert_intermediate_size")
            add("shared_expert.gate_up", 2 * shard(shared), h, sparse)
            add("shared_expert.down", h, shard(shared), sparse)
            add("shared_expert.gate", 1, h, sparse)
        if layers - sparse:
            ffn = positive("intermediate_size")
            add("mlp.gate_up", 2 * shard(ffn), h, layers - sparse)
            add("mlp.down", h, shard(ffn), layers - sparse)
    else:
        ffn = positive("intermediate_size", config.get("shared_expert_intermediate_size"))
        add("mlp.gate_up", 2 * shard(ffn), h, layers)
        add("mlp.down", h, shard(ffn), layers)
    add("lm_head", shard(positive("vocab_size")), h, 1)
    if gdn:
        key = positive("linear_num_key_heads") * positive("linear_key_head_dim")
        values = positive("linear_num_value_heads")
        value = values * positive("linear_value_head_dim")
        qkvz, ba = 2 * shard(key) + 2 * shard(value), 2 * shard(values)
        add("gdn.in_proj_qkvz", qkvz, h, layers - attn, layout="split")
        add("gdn.in_proj_ba", ba, h, layers - attn, layout="split")
        add("gdn.in_proj_qkvzba", qkvz + ba, h, layers - attn, layout="fused")
        add("gdn.out_proj", h, shard(value), layers - attn)
    return {"models": {"model": rows}}


def measured_ms(value):
    """Engine M values plus both edges of every bucket containing one (all edges for bxq)."""
    if value == "accept":
        return value
    if value == "bxq":
        ms = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 32, 36, 48, 64}
    else:
        ms = set(map(int, value.split(",")))
    if not ms or min(ms) < 1 or max(ms) > 64:
        raise ValueError("M must be in 1..64, bxq or accept")
    buckets = ((1, 1), (2, 2), (3, 4), (5, 8), *((lo, lo + 7) for lo in range(9, 65, 8)))
    edges = {
        edge
        for bucket in buckets
        if any(bucket[0] <= m <= bucket[1] for m in ms)
        for edge in bucket
    }
    if value == "bxq":
        edges = {edge for bucket in buckets for edge in bucket}
    return ",".join(map(str, sorted(edges | ms)))


def command_list(args, shapes):
    """(stage, argv, stdout path) in execution order, using the tools' actual CLIs."""

    def tool(name, *flags):
        return [sys.executable, str(TOOLS / name), *map(str, flags)]

    logs = args.logs
    shape_path, probe = logs / "shapes.json", logs / "probe.log"
    source = ["--hf-config", args.hf_config] if args.hf_config else ["--shapes", args.shapes]
    commands = [
        (
            "shapes",
            tool(
                "linear_table_pipeline.py",
                *source,
                "--tp",
                args.tp,
                "--logs",
                logs,
                "--stage",
                "shapes",
            ),
            None,
        )
    ]
    unique = sorted({(r["N"], r["K"]) for rows in shapes["models"].values() for r in rows})
    if not unique or any(n <= 0 or k <= 0 or k % 8 for n, k in unique):
        raise ValueError("positive N/K and K divisible by eight required")
    ms = measured_ms(args.m)
    commands.append(
        (
            "probe",
            tool(
                "linear_smallm_bxq_sweep.py",
                "--shapes",
                ",".join(f"{n}:{k}" for n, k in unique),
                "--m",
                ms,
            ),
            probe,
        )
    )
    commands.append(
        (
            "candidate",
            tool(
                "linear_smallm_table.py",
                "--candidate",
                "--probe",
                probe,
                "--bucketize",
                "--only-measured",
                "--rewrite",
            ),
            logs / "candidate.log",
        )
    )
    accept_logs = [logs / f"accept-{regime}.log" for regime in ("cold", "engine", "hot")]
    for regime, log in zip(("cold", "engine", "hot"), accept_logs, strict=True):
        commands.append(
            (
                "accept",
                tool(
                    "linear_smallm.py",
                    "--shapes",
                    shape_path,
                    "--model",
                    "all",
                    "--m",
                    ms,
                    "--accept",
                    "--l2",
                    regime,
                    "--pairs",
                    getattr(args, "pairs", 60),
                ),
                log,
            )
        )
    commands.append(
        (
            "rewrite",
            tool(
                "linear_smallm_table.py", "--accept", *accept_logs, "--only-measured", "--rewrite"
            ),
            logs / "rewrite.log",
        )
    )
    commands.append(
        (
            "rewrite",
            tool("linear_smallm_table.py", "--accept", *accept_logs, "--markdown"),
            logs / "table.md",
        )
    )
    if getattr(args, "out_dtype", "bf16") == "fp32":
        for stage, argv, _ in commands:
            if stage in ("shapes", "probe", "accept"):
                argv.extend(["--out-dtype", "fp32"])
    return [c for c in commands if args.stage in ("all", c[0])]


def _finished_acceptance(path):
    begun, completed = set(), set()
    counts = {}
    for line in path.read_text().splitlines():
        if line.startswith("RESULT "):
            record = json.loads(line[7:])
            if not isinstance(record, dict):
                return False
            kind = record.get("kind")
            if kind not in ("accept_begin", "accept_completion", "smallm_accept"):
                continue
            run = record.get("run_id")
            if not isinstance(run, str) or not run:
                return False
            if kind == "accept_begin":
                begun.add(run)
                completed.discard(run)
                counts[run] = 0
            elif kind == "accept_completion":
                if (
                    run not in begun
                    or record.get("complete") is not True
                    or record.get("rows", counts[run]) != counts[run]
                ):
                    return False
                completed.add(run)
            else:
                if run not in begun or run in completed:
                    return False
                counts[run] += 1
    return bool(begun) and begun == completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-config", type=Path)
    source.add_argument("--shapes", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--out-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--m", default="accept")
    parser.add_argument(
        "--pairs",
        type=int,
        default=60,
        help="acceptance pairs per record (>= 20; GB10 rows qualify only with >= 60)",
    )
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("shapes", "probe", "candidate", "accept", "rewrite", "all"),
        default="all",
    )
    args = parser.parse_args(argv)
    if args.pairs < 20:
        parser.error("acceptance requires at least 20 pairs")
    try:
        if args.hf_config:
            shapes = derive_shapes(json.loads(args.hf_config.read_text()), args.tp)
        else:
            shapes = json.loads(args.shapes.read_text())
        commands = command_list(args, shapes)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for _, command, log in commands:
            print(shlex.join(command) + (" > " + shlex.quote(str(log)) if log else ""))
        return 0
    args.logs.mkdir(parents=True, exist_ok=True)
    if args.stage == "shapes":
        (args.logs / "shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
        return 0
    dispatch = Path("python/ctikernels/linear/_dispatch.py")
    snapshot = None
    succeeded = False
    try:
        for stage, command, log in commands:
            if snapshot is None and stage in ("candidate", "accept", "rewrite"):
                snapshot = dispatch.read_bytes()
            if stage == "rewrite":
                for regime in ("cold", "engine", "hot"):
                    if not _finished_acceptance(args.logs / f"accept-{regime}.log"):
                        raise ValueError(f"incomplete {regime} acceptance log")
            if log is None:
                result = subprocess.run(command, check=False)
            else:
                with log.open("w") as output:
                    result = subprocess.run(command, stdout=output, check=False)
            if stage == "accept":
                # Per-row FAILs exit nonzero but are evidence; only an unfinished run fails.
                if not _finished_acceptance(log):
                    raise ValueError("incomplete acceptance log")
            elif result.returncode:
                return result.returncode
        succeeded = True
    except (OSError, ValueError, TypeError) as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if snapshot is not None and not succeeded:
            dispatch.write_bytes(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
