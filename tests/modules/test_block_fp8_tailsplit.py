# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU proofs of the hybrid schedule, progress, and register scratch addressing."""

import random
from collections import defaultdict
from dataclasses import replace
from graphlib import TopologicalSorter
from itertools import product
from math import ceil

import pytest

from ctikernels.linear.cute._block_fp8_provider import (
    CONFIGS,
    Config,
    schedule_for,
    tail_contributors,
    tail_work,
    tail_workspace,
)


@pytest.mark.parametrize(
    "tiles,slots,kb,expected,w1_expected",
    [
        (256, 188, 32, (0, 256, 188), (188, 68, 188)),
        (272, 188, 40, (0, 272, 188), (188, 84, 188)),
        (192, 188, 16, (0, 192, 188), (188, 4, 16)),
        (64, 48, 32, (0, 64, 48), (48, 16, 48)),
        (256, 48, 64, (192, 64, 48), (240, 16, 48)),
        (96, 48, 32, (96, 0, 0), (96, 0, 0)),
        (8, 188, 40, (0, 8, 80), (0, 8, 80)),
    ],
)
@pytest.mark.parametrize("sk_waves", (1, 2))
def test_tailsplit_expected_partition(tiles, slots, kb, expected, w1_expected, sk_waves):
    config = Config(scheduler="tailsplit", sk_waves=sk_waves)
    schedule = schedule_for(tiles, slots, kb, config)
    assert (schedule.dp_tiles, schedule.sk_tiles, schedule.sk_ctas) == (
        w1_expected if sk_waves == 1 else expected
    )
    _check_schedule(tiles, slots, kb, config)


@pytest.mark.parametrize("sk_order", ("helper_first", "dp_first"))
def test_tailsplit_one_wave_middle_fragments(sk_order):
    config = Config(scheduler="tailsplit", sk_waves=1, sk_order=sk_order)
    schedule = schedule_for(49, 48, 32, config)
    assert (schedule.dp_tiles, schedule.sk_tiles, schedule.sk_ctas) == (48, 1, 8)
    # Six CTAs own only middle pieces, each publishing its own scratch slot.
    for bid in range(8):
        sk = [w for w in tail_work(schedule, bid) if w.role != "dp"]
        assert len(sk) == 1
        assert (sk[0].tile, sk[0].k_begin, sk[0].k_end) == (48, 4 * bid, 4 * (bid + 1))
        assert sk[0].role == ("owner" if bid == 7 else "helper")
    assert tail_contributors(schedule, 48) == tuple(range(7))
    _check_schedule(49, 48, 32, config)


def _check_schedule(tiles, sms, kb, config):
    schedule = schedule_for(tiles, sms, kb, config)
    slots = sms * config.ctas_per_sm
    work = [tail_work(schedule, bid) for bid in range(slots)]
    by_tile = defaultdict(list)
    ranks = {"helper": 0, "dp": 1, "unsplit": 2, "owner": 3}
    if config.sk_order == "dp_first":
        ranks.update(dp=0, helper=1)
    # Nodes are completed work items; edges include per-CTA sequencing and
    # each owner's acquire waiting for all helper publications on its tile.
    dependencies = {}
    first_sk = {}
    partial_words, counters = tail_workspace(config, sms)
    helper_slots = set()
    for bid, items in enumerate(work):
        assert [ranks[w.role] for w in items] == sorted(ranks[w.role] for w in items)
        # A helper is the interval's right boundary; a split owner is its left
        # boundary. The two may coincide, but then there is only one fragment.
        assert sum(w.role == "helper" for w in items) <= 1
        assert sum(w.role == "owner" for w in items) <= 1
        for i, w in enumerate(items):
            dependencies[bid, i] = {(bid, i - 1)} if i else set()
            if w.role != "dp":
                first_sk.setdefault(bid, i)
            assert 0 <= w.tile < tiles and 0 <= w.k_begin < w.k_end <= kb
            by_tile[w.tile].append((w.k_begin, w.k_end, bid, w.role))
            if w.role == "helper":
                assert i == first_sk[bid] and w.k_end < kb
                assert i == (
                    sum(w.role == "dp" for w in items) if config.sk_order == "dp_first" else 0
                )
                assert bid not in helper_slots
                helper_slots.add(bid)
                assert (bid + 1) * config.bm * config.bn <= partial_words
            elif w.role == "owner":
                assert i == len(items) - 1 and w.k_end == kb and w.k_begin > 0
            else:
                assert (w.k_begin, w.k_end) == (0, kb)
            if w.role == "dp":
                assert w.tile % slots == bid and w.tile < schedule.dp_tiles
            else:
                counter = partial_words + w.tile - schedule.dp_tiles
                assert partial_words <= counter < partial_words + counters

        # Independently check that reordering preserves the assigned flat range.
        sk = sorted(
            (w.tile - schedule.dp_tiles) * kb + k
            for w in items
            if w.role != "dp"
            for k in range(w.k_begin, w.k_end)
        )
        if bid < schedule.sk_ctas:
            iterations = schedule.sk_tiles * kb
            assert sk == list(
                range(
                    bid * iterations // schedule.sk_ctas, (bid + 1) * iterations // schedule.sk_ctas
                )
            )
        else:
            assert not sk
    assert set(by_tile) == set(range(tiles))
    for tile, fragments in by_tile.items():
        fragments.sort()
        # Adjoining intervals prove every iteration is covered exactly once.
        assert fragments[0][0] == 0 and fragments[-1][1] == kb
        assert all(a[1] == b[0] for a, b in zip(fragments, fragments[1:], strict=False))
        helpers = tuple(bid for _, _, bid, role in fragments if role == "helper")
        assert helpers == tuple(sorted(helpers)) == tail_contributors(schedule, tile)
        if len(fragments) > 1:
            assert fragments[-1][3] == "owner"
            owner = fragments[-1][2]
            assert helpers == tuple(range(helpers[0], owner))
            # Every wait targets FIRST SK work, which has no wait of its own.
            for h in helpers:
                helper = work[h][first_sk[h]]
                assert helper.role == "helper" and helper.tile == tile
                dependencies[owner, len(work[owner]) - 1].add((h, first_sk[h]))
        else:
            assert fragments[0][3] in ("dp", "unsplit")
    assert len(tuple(TopologicalSorter(dependencies).static_order())) == len(dependencies)
    assert schedule.sk_tiles < config.sk_waves * slots
    assert partial_words == slots * config.bm * config.bn
    assert counters == 2 * slots
    if schedule.sk_tiles:
        assert schedule.dp_tiles % slots == 0
        assert {sum(w.role == "dp" for w in items) for items in work} == {
            schedule.dp_tiles // slots
        }
        if config.sk_waves == 1 or tiles < slots:
            iterations = schedule.sk_tiles * kb
            assert schedule.sk_ctas == min(slots, max(1, iterations // config.min_sk_iters))
            # Only the unavoidable single-CTA case can fall below the floor.
            assert iterations // schedule.sk_ctas >= min(config.min_sk_iters, iterations)
    else:
        assert schedule.dp_tiles == tiles


@pytest.mark.parametrize("sk_order", ("helper_first", "dp_first"))
@pytest.mark.parametrize("sk_waves", (1, 2))
def test_tailsplit_partition_coverage_roles_order_and_progress(sk_order, sk_waves):
    config = Config(scheduler="tailsplit", sk_order=sk_order, sk_waves=sk_waves)
    # Exhaustive small partitions cover same-tile ranges and aligned boundaries.
    for slots, kb in product(range(1, 13), range(1, 10)):
        for tiles in range(4 * slots + 1):
            _check_schedule(tiles, slots, kb, config)
    # Physical GPU sizes, threshold boundaries, large DP regions, and min_iters
    # large enough that one CTA owns several unsplit SK tiles.
    rng = random.Random(9182)
    for _ in range(400):
        sms = rng.choice((24, 48, 170, 188))
        ctas = rng.choice((1, 2))
        cfg = Config(
            bm=64,
            bn=64,
            warps=4,
            scheduler="tailsplit",
            sk_order=sk_order,
            sk_waves=sk_waves,
            ctas_per_sm=ctas,
            min_sk_iters=rng.choice((1, 4, 7, 256)),
            tail_split_threshold=rng.choice((0.1, 0.5, 1.0)),
        )
        _check_schedule(rng.randrange(4 * sms * ctas), sms, rng.choice((1, 2, 16, 32, 40, 64)), cfg)


@pytest.mark.parametrize("ctas", (1, 2))
@pytest.mark.parametrize("threshold", (0.1, 0.5, 1.0))
@pytest.mark.parametrize("sk_order", ("helper_first", "dp_first"))
@pytest.mark.parametrize("sk_waves", (1, 2))
def test_tailsplit_fallback_matches_persistent(ctas, threshold, sk_order, sk_waves):
    config = Config(
        bm=64,
        bn=64,
        warps=4,
        scheduler="tailsplit",
        ctas_per_sm=ctas,
        tail_split_threshold=threshold,
        sk_order=sk_order,
        sk_waves=sk_waves,
    )
    slots = 48 * ctas
    for tiles, kb in product(range(3 * slots), (1, 2, 32)):
        split = kb > 1 and 0 < tiles % slots < ceil(threshold * slots)
        schedule = schedule_for(tiles, 48, kb, config)
        assert bool(schedule.sk_tiles) == split
        if not split:
            persistent = schedule_for(tiles, 48, kb, replace(config, scheduler="persistent"))
            assert schedule == persistent
            assert all(tail_work(schedule, b) == tail_work(persistent, b) for b in range(slots))


def test_tailsplit_register_workspace_bijection():
    for config in set(CONFIGS.values()):
        if config.scheduler != "tailsplit":
            continue
        threads, elements = config.warps * 32, config.bm * config.bn
        chunks = elements // (threads * 4)
        assert chunks * threads * 4 == elements
        # This is the kernel's register->workspace map, independent of normal/
        # swapped MMA coordinates and output epilogue type. Each vector is aligned.
        offsets = [(chunk * threads + tid) * 4 for chunk in range(chunks) for tid in range(threads)]
        assert offsets == list(range(0, elements, 4))
        words, counters = tail_workspace(config, 188)
        assert words % 4 == 0 and counters == 2 * 188 * config.ctas_per_sm
        assert words == elements * 188 * config.ctas_per_sm
