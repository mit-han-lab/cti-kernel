# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Sweep policy generation remains usable without CUDA or Triton installed."""

import ast
import json

import pytest
import torch

from benchmark.kernel_tuning.block_fp8_moe_triton_sweep import candidates, emit_table, make_cases
from ctikernels import _moe_block_fp8_triton_table as triton_policy

RTX5090 = (12, 0)


# The measured devices (RTX 5090, RTX PRO 6000, GB10); others carry placeholder rows.
@pytest.mark.parametrize("device", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
@pytest.mark.parametrize("shape", [(128, 2048, 512, 8), (64, 2048, 768, 8)])
def test_ep2_exact_measured_dispatch_rows(device, shape):
    rows = triton_policy.MEASURED.get(device, {}).get(shape, ())
    assert rows, (
        f"EP2 requires exact measured key {shape} for device {device}; "
        "placeholder/heuristic rows do not qualify"
    )
    assert triton_policy.TABLE[device][shape] == triton_policy._measured_rows(rows)
    for m, fc1, fc2 in triton_policy.TABLE[device][shape]:
        expected = triton_policy._route_config(fc1, fc2, m * shape[-1])
        assert triton_policy.selection(device, m, *shape) == expected


def test_emit_table_merges_devices_and_sorts_tokens(tmp_path):
    paths = []
    for m in (64, 1):
        records = []
        for device in ((12, 0, 170), (12, 1, 48)):
            for e in (128, 256):
                config = candidates(m, e, capability=device[:2])[0]
                for gemm in ("fc1", "fc2"):
                    records.append(
                        {
                            "device_class": device,
                            "M": m,
                            "E_local": e,
                            "gemm": gemm,
                            "best": {"config": config, "us": 1.0},
                        }
                    )
        records.append({"TABLE": []})
        path = tmp_path / f"{m}.jsonl"
        path.write_text("\n".join(map(json.dumps, records)))
        paths.append(path)
    source = emit_table(paths)
    table = ast.literal_eval(source.removeprefix("MEASURED = "))
    for device in ((12, 0, 170), (12, 1, 48)):
        for e in (128, 256):
            assert [r[0] for r in table[device][(e, 2048, 512, 8)]] == [1, 64]
            for _, fc1, fc2 in table[device][(e, 2048, 512, 8)]:
                assert fc1[0] == fc2[0]


@pytest.mark.parametrize("m", [1, 2, 4, 64, 1024, 4096])
def test_candidate_budget_and_constraints(m):
    grid = candidates(m, 128, capability=RTX5090)
    assert len(grid) <= (13 if m <= 4 or m >= 1024 else 25)
    assert all(c["BLOCK_SIZE_K"] == 128 for c in grid)
    assert all(c["BLOCK_SIZE_M"] <= max(16, m * 8) for c in grid)
    if m <= 4:
        assert all(c["GROUP_SIZE_M"] == 1 for c in grid)
    assert {c["num_stages"] for c in candidates(m, 128, True, capability=RTX5090)} == {2, 3, 4, 5}


def test_emit_rejects_incompatible_row_tiles(tmp_path):
    rows = []
    for gemm, bm in (("fc1", 16), ("fc2", 32)):
        config = candidates(4, 256, capability=RTX5090)[0] | {"BLOCK_SIZE_M": bm}
        rows.append(
            {
                "device_class": [12, 0, 170],
                "M": 4,
                "E_local": 256,
                "gemm": gemm,
                "best": {"config": config},
            }
        )
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="share BLOCK_SIZE_M"):
        emit_table([path])


@pytest.mark.parametrize(
    "shape,global_e,e",
    [("qwen3.5", 256, 256), ("qwen3.5", 256, 128), ("qwen3", 128, 128), ("qwen3", 128, 64)],
)
@pytest.mark.parametrize("m", [1, 4, 32, 64])
def test_eight_distinct_global_routings(m, shape, global_e, e):
    cases = make_cases(m, e, device="cpu", shape=shape)
    for index, (x, weights, ids) in enumerate(cases):
        assert x.shape == (m, 2048)
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert int((ids < e).sum()) == m * 8 * e // global_e
        torch.testing.assert_close(weights.sum(-1), torch.ones(m))
        assert all(not torch.equal(ids, previous[2]) for previous in cases[:index])


@pytest.mark.parametrize("shape,e", [("qwen3", 128), ("qwen3", 64)])
@pytest.mark.parametrize("m", [1, 4, 64, 4096])
def test_qwen3_candidates(shape, e, m):
    from ctikernels.moe_impl import block_fp8_moe_config

    grid = candidates(m, e, shape=shape, capability=RTX5090)
    assert grid[0] == block_fp8_moe_config(m, e, 1536, 2048, 8, capability=RTX5090)
    assert all(c["BLOCK_SIZE_K"] == 128 for c in grid)


def test_emit_separates_shapes(tmp_path):
    records = []
    for shape, i in (("qwen3.5", 512), ("qwen3", 768)):
        for gemm in ("fc1", "fc2"):
            records.append(
                {
                    "device_class": [12, 0, 170],
                    "M": 1,
                    "E_local": 128,
                    "shape": shape,
                    "H": 2048,
                    "I": i,
                    "top_k": 8,
                    "gemm": gemm,
                    "best": {"config": candidates(1, 128, shape=shape, capability=RTX5090)[0]},
                }
            )
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    merged = ast.literal_eval(emit_table([path]).removeprefix("MEASURED = "))
    assert set(merged[12, 0, 170]) == {(128, 2048, 512, 8), (128, 2048, 768, 8)}
    selected = ast.literal_eval(emit_table([path], "qwen3").removeprefix("MEASURED = "))
    assert set(selected[12, 0, 170]) == {(128, 2048, 768, 8)}


def test_emit_does_not_pair_different_shapes(tmp_path):
    records = [
        {
            "device_class": [12, 0, 170],
            "M": 1,
            "E_local": 128,
            "shape": shape,
            "gemm": gemm,
            "best": {"config": candidates(1, 128, capability=RTX5090)[0]},
        }
        for shape, gemm in (("qwen3.5", "fc1"), ("qwen3", "fc2"))
    ]
    path = tmp_path / "incomplete.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    with pytest.raises(ValueError, match="both FC1 and FC2"):
        emit_table([path])
