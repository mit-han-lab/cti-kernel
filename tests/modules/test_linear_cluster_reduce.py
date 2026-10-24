# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU model of the single-launch (cluster) split-K reduction and its contracts.

No CUDA/DSL dependency: the slot/owner arithmetic mirrors Kernel.push and
Kernel.cluster_reduce; the summation-order check mirrors the reduce kernels.
"""

from dataclasses import asdict, replace
from itertools import product

import pytest
import torch

from benchmark.kernel_tuning import linear_smallm_table as table
from ctikernels.linear._dispatch import DSMEM, L2, NONE, Config, parse_config


def cluster_configs():
    for rows, splits, variant in product((16, 32, 64, 128), (2, 4, 8, 16), ("smem", "reg", "wide")):
        warps = rows // 16 if variant == "smem" else max(1, rows // 32)
        try:
            yield Config(rows, 128, 2, splits, min(warps, 8), variant, 4, DSMEM)
        except ValueError:
            pass


def test_cluster_field_validation_and_parse():
    assert Config().cluster == NONE and Config().scratch_splits == 0 and Config().kernels == 1
    assert Config(splits=4).scratch_splits == 4 and Config(splits=4).kernels == 2
    assert Config(splits=4, cluster=DSMEM).scratch_splits == 0
    assert Config(splits=4, cluster=DSMEM).kernels == 1
    assert (
        Config(splits=4, cluster=L2).scratch_splits == 4
        and Config(splits=4, cluster=L2).kernels == 1
    )
    # Records written with booleans load as DSMEM / NONE.
    assert Config(splits=4, cluster=True) == Config(splits=4, cluster=DSMEM)
    assert type(Config(splits=4, cluster=True).cluster) is int
    for bad in (
        {"splits": 1, "cluster": DSMEM},
        {"splits": 1, "cluster": L2},
        {"splits": 32, "cluster": DSMEM},
        {"rows": 16, "warps": 1, "splits": 3, "cluster": L2},
        {"splits": 2, "cluster": 3},
    ):
        with pytest.raises(ValueError):
            Config(**bad)
    assert parse_config("16,128,2,8,1,smem,4,1") == Config(16, 128, 2, 8, 1, "smem", 4, DSMEM)
    assert parse_config("16,128,2,8,1,smem,4,2") == Config(16, 128, 2, 8, 1, "smem", 4, L2)
    assert parse_config("16,128,2,8,1,smem,4,0") == Config(16, 128, 2, 8, 1, "smem", 4)
    assert parse_config("64,128,2,1") == Config(64, 128, 2, 1)
    with pytest.raises(ValueError):
        parse_config("16,128,2,8,1,smem,4,3")


@pytest.mark.parametrize("mtiles", (1, 2, 8))
def test_cluster_shared_bytes_add_the_partial_tile(mtiles):
    plain = Config(64, 128, 2, 4, 4, "smem", 4)
    fused = replace(plain, cluster=DSMEM)
    assert replace(plain, cluster=L2).shared_bytes_for(mtiles) == plain.shared_bytes_for(mtiles)
    # Receive and staging tiles (staging padded by four floats per owner).
    tiles = (2 * 8 * mtiles * 64 + 4 * 4) * 4 + 16
    assert fused.shared_bytes_for(mtiles) == plain.shared_bytes_for(mtiles) + tiles
    reg = Config(16, 128, 2, 8, 1, "reg", 4, DSMEM)
    tiles = (2 * 8 * mtiles * 16 + 4 * 8) * 4 + 16
    assert reg.shared_bytes_for(mtiles, True) == 16 * mtiles * 136 * 2 + tiles


@pytest.mark.parametrize("cfg", list(cluster_configs()), ids=str)
@pytest.mark.parametrize("mtiles", (1, 3, 8))
def test_stage_copy_and_reduce_deliver_every_partial_once(cfg, mtiles):
    xrows, share, threads = 8 * mtiles, cfg.rows // cfg.splits, cfg.warps * 32
    run, vectors = xrows * share + 4, xrows * share // 4  # Kernel.owner_run, 16-B vectors.
    for m in sorted({1, 3, xrows // 2 or 1, xrows}):
        # push(): CTA `split` stages (mr < m, row) at [owner][mr][row % share].
        stage = [[None] * (cfg.splits * run) for _ in range(cfg.splits)]
        for split, mr, row in product(range(cfg.splits), range(m), range(cfg.rows)):
            slot = row // share * run + mr * share + row % share
            assert stage[split][slot] is None
            stage[split][slot] = (split, mr, row)
        # Copy loop: vector (owner, i) of CTA `split` lands at [split][i*4:(i+1)*4] of the
        # owner's receive tile; tail vectors carry unread rows >= m.
        receive = [[None] * (xrows * cfg.rows) for _ in range(cfg.splits)]
        for split in range(cfg.splits):
            for j in range(-(-cfg.splits * vectors // threads)):
                for tid in range(threads):
                    q = j * threads + tid
                    owner, i = q // vectors, q % vectors
                    if owner < cfg.splits and i * 4 < m * share:
                        for v in range(4):
                            assert i * 4 + v < xrows * share  # Inside the owner's run.
                            target = split * xrows * share + i * 4 + v
                            assert receive[owner][target] is None
                            receive[owner][target] = stage[split][owner * run + i * 4 + v]
        # cluster_reduce(): owner r sums (s, mr, row) over s in split order.
        seen = set()
        for owner in range(cfg.splits):
            for j in range(-(-xrows * share // threads)):
                for tid in range(threads):
                    e = j * threads + tid
                    mr, row = e // share, owner * share + e % share
                    if mr < m:
                        assert (mr, row) not in seen
                        seen.add((mr, row))
                        for s in range(cfg.splits):
                            got = receive[owner][(s * xrows + mr) * share + e % share]
                            assert got == (s, mr, row)
        assert seen == set(product(range(m), range(cfg.rows)))


def test_split_order_sum_matches_the_scratch_reduction():
    gen = torch.Generator().manual_seed(0)
    partial = torch.randn((8, 4, 64), generator=gen) * torch.logspace(-3, 3, 64)
    # Reduce kernels: value = 0; value += partial[s] for s = 0..S-1 (FP32).
    expected = torch.zeros((4, 64))
    for s in range(8):
        expected = expected + partial[s]
    value = torch.zeros((4, 64))
    for s in range(8):
        value += partial[s]
    assert torch.equal(value, expected)


def test_pre_cluster_records_identify_the_scratch_reduction():
    cfg = Config(16, 128, 2, 8, 1, "reg", 4)
    legacy = {k: v for k, v in asdict(cfg).items() if k != "cluster"}
    row = {"M": 8, "launch_config": {"config": legacy, "mtiles": 1, "cta_threads": 32}}
    assert table.launch_identity(row) == (cfg, 1)
    for mode in (DSMEM, L2):
        fused = dict(asdict(cfg), cluster=mode)
        row = {"M": 8, "launch_config": {"config": fused, "mtiles": 1, "cta_threads": 32}}
        assert table.launch_identity(row) == (replace(cfg, cluster=mode), 1)
    missing = {k: v for k, v in legacy.items() if k != "prefetch"}
    row = {"M": 8, "launch_config": {"config": missing, "mtiles": 1, "cta_threads": 32}}
    assert table.launch_identity(row) is None


def test_table_literal_keeps_seven_fields_without_cluster():
    chosen = {
        "rtx5090": {
            (128, 2048): {
                (1, 1): Config(16, 128, 2, 8, 1, "smem", 4, DSMEM),
                (2, 2): Config(16, 128, 2, 8, 1, "smem", 4, L2),
                (2, 8): Config(64, 128, 2, 8, 4, "smem", 4),
            }
        }
    }
    text = table.python_literal(chosen)
    assert 'Config(16, 128, 2, 8, 1, "smem", 4, 1)' in text.replace("'", '"')
    assert 'Config(16, 128, 2, 8, 1, "smem", 4, 2)' in text.replace("'", '"')
    assert 'Config(64, 128, 2, 8, 4, "smem", 4)' in text.replace("'", '"')
    namespace = {"Config": Config}
    exec("result = " + text, namespace)
    assert namespace["result"] == chosen


def test_cluster_admission_is_capability_aware():
    from ctikernels.linear import inference

    big = Config(16, 128, 2, 16, 1, "smem", 4, DSMEM)
    small = Config(16, 128, 2, 8, 1, "smem", 4, DSMEM)
    assert inference._admits(big, 1, 0, (12, 0))
    assert not inference._admits(big, 1, 0, (12, 1))
    assert inference._admits(small, 1, 0, (12, 1))
    assert inference._admits(Config(16, 128, 2, 16, 1, "smem", 4), 1, 0, (12, 1))
