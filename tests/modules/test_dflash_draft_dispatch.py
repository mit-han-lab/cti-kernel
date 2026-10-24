# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
import csv
from collections import defaultdict
from pathlib import Path

import pytest

from ctikernels.attention._draft import dispatch


@pytest.fixture(autouse=True)
def legacy_tuning(monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_INLINE", "0")


def _row(**overrides):
    row = {
        "batch_size": 3,
        "query_length": 9,
        "max_kv_tokens": 4105,
        "page_size": 16,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "causal": False,
        "window": None,
        "device": "cuda:0",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "causal,window,name",
    [
        (False, None, "full"),
        (True, None, "causal_full"),
        (True, 4096, "causal_sliding"),
        (False, 2048, "noncausal_sliding"),
    ],
)
def test_mask_names(causal, window, name):
    assert dispatch.mask_name(causal, window) == name


@pytest.mark.parametrize(
    "hardware,cute_queries",
    [
        # B=64 (the last entry) is past every rule's B<=4 band.
        ((12, 0, 170), (*({12, 16, 32},) * 4, set())),
        ((12, 0, 188), (set(), {16, 32}, {12, 16, 32}, {12, 16, 32}, set())),
        (
            (12, 1, 48),
            (
                {9, 11, 12, 16, 32},
                {9, 11, 12, 16, 32},
                *({2, 4, 8, 9, 11, 12, 16, 32},) * 2,
                set(),
            ),
        ),
        ((12, 0, 169), (set(),) * 5),
        ((12, 1, 170), (set(),) * 5),
        ((11, 0, 170), (set(),) * 5),
    ],
)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    "causal,window", [(True, None), (False, None), (True, 4096), (False, 2048)]
)
def test_hd256_default_provider_rows(monkeypatch, hardware, cute_queries, head_dim, causal, window):
    from ctikernels.attention._draft import cute, flashinfer
    from ctikernels.attention.draft import _provider

    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(dispatch, "hardware", lambda _: hardware)
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    for batch, queries in zip((1, 2, 3, 4, 64), cute_queries, strict=True):
        for query in (2, 4, 8, 9, 11, 12, 16, 32):
            use_cute = head_dim == 128 or (causal and window is None and query in queries)
            # Neither capacity nor local head count changes the hd256 rule.
            for capacity, heads in ((32768, 4), (4096, 2)):
                row = _row(
                    batch_size=batch,
                    query_length=query,
                    head_dim=head_dim,
                    causal=causal,
                    window=window,
                    max_kv_tokens=capacity,
                    num_qo_heads=heads * 4,
                    num_kv_heads=heads,
                )
                assert dispatch.prefers_flashinfer(**row) == (not use_cute)
                assert _provider(None, row) is (cute if use_cute else flashinfer)


def test_rules_match_every_field_and_the_union_bound(monkeypatch):
    monkeypatch.setattr(dispatch, "hardware", lambda device: (12, 1, 48))
    monkeypatch.setattr(
        dispatch,
        "FLASHINFER_ROWS",
        {
            (12, 1, 48): (
                {
                    "masks": {"full", "noncausal_sliding"},
                    "batch": {3},
                    "query": (8, 12),
                    "group": {4},
                    "union": (4000, 5000),
                },
            )
        },
    )
    assert dispatch.prefers_flashinfer(**_row())
    assert not dispatch.prefers_flashinfer(**_row(batch_size=2))
    assert not dispatch.prefers_flashinfer(**_row(query_length=16, max_kv_tokens=4112))
    assert not dispatch.prefers_flashinfer(**_row(num_qo_heads=32, num_kv_heads=4))
    assert not dispatch.prefers_flashinfer(**_row(causal=True, window=4096))
    assert not dispatch.prefers_flashinfer(**_row(causal=True))
    # Sliding rows key on min(C, W + Q - 1), not the capacity.
    assert not dispatch.prefers_flashinfer(**_row(window=2048, max_kv_tokens=65536))
    assert not dispatch.prefers_flashinfer(**_row(max_kv_tokens=65536))
    monkeypatch.setattr(dispatch, "hardware", lambda device: (12, 0, 170))
    assert not dispatch.prefers_flashinfer(**_row())


@pytest.mark.parametrize("b,expected", [(1, 6), (2, 3), (3, 2), (4, 2)])
@pytest.mark.parametrize("capacity", [4100, 32768])
def test_gb10_chunk_cap_avoids_forced_extra_splits(monkeypatch, b, expected, capacity):
    from types import SimpleNamespace

    import torch

    from ctikernels.attention._draft import cute

    for name in ("SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=48, major=12, minor=1),
    )
    geometry = {
        "batch_size": b,
        "query_length": 4,
        "max_kv_tokens": capacity,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "window": None,
        "device": torch.device("cuda"),
    }
    # The floor stops at MID_UNION, even when the capacity grows to 32768.
    cap = cute.chunk_cap(geometry["device"])
    assert cap == 4096
    occupancy = min((capacity + 63) // 64, max(1, 48 // (b * 8)))
    floor = (min(capacity, cute.MID_UNION) + cap - 1) // cap
    assert cute.configuration(**geometry)[-1] == max(occupancy, floor) == expected
    if capacity == 32768 and b <= 3:
        assert expected == occupancy
        assert expected < (capacity + cap - 1) // cap


@pytest.mark.parametrize(
    "capability,batch,rows,base,expected",
    [
        ((12, 0), 1, 48, 8, (1, 32)),
        ((12, 0), 3, 32, 24, (2, 32)),
        ((12, 0), 4, 64, 32, (2, 32)),
        ((12, 0), 2, 32, 16, (1, 32)),
        ((12, 1), 1, 48, 8, (2, 32)),
        ((12, 1), 1, 32, 8, (1, 64)),
        ((12, 1), 2, 48, 16, (1, 64)),
        ((12, 1), 1, 64, 4, (1, 64)),
    ],
)
def test_cute_row_tuning_table(capability, batch, rows, base, expected):
    from ctikernels.attention._draft import cute

    assert cute.row_tuning(capability, batch, rows, base) == expected


@pytest.mark.parametrize(
    "minor,query,window,capacity,expected",
    [
        (1, 4, None, 4099, 3),
        (1, 4, None, 4101, 3),
        (1, 4, None, 4096, 4),
        (1, 5, None, 4101, 4),
        (1, 4, 4096, 4101, 4),
        # sm_120: the 1024-key chunk cap floors this union at five splits.
        (0, 4, None, 4101, 5),
    ],
)
def test_gb10_split_rules_cover_only_the_measured_rows(
    monkeypatch, minor, query, window, capacity, expected
):
    from types import SimpleNamespace

    import torch

    from ctikernels.attention._draft import cute

    for name in ("SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    # 48 SMs on both capabilities isolates the GB10 rule from the SM count.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=48, major=12, minor=minor),
    )
    geometry = {
        "batch_size": 3,
        "query_length": query,
        "max_kv_tokens": capacity,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "window": window,
        "device": torch.device("cuda"),
    }
    assert cute.configuration(**geometry)[-1] == expected


def test_causal_full_dispatch_uses_capacity_and_distinct_mask(monkeypatch):
    monkeypatch.setattr(dispatch, "hardware", lambda device: (12, 1, 48))
    monkeypatch.setattr(
        dispatch,
        "FLASHINFER_ROWS",
        {(12, 1, 48): ({"masks": {"causal_full"}, "union": (8192, 8192)},)},
    )
    assert dispatch.prefers_flashinfer(**_row(causal=True, max_kv_tokens=8192))
    assert not dispatch.prefers_flashinfer(**_row(causal=True, max_kv_tokens=8191))
    assert not dispatch.prefers_flashinfer(**_row(causal=True, window=4096, max_kv_tokens=8192))
    assert not dispatch.prefers_flashinfer(**_row(max_kv_tokens=8192))


RECEIPT = Path(__file__).resolve().parents[2] / (
    "benchmark/kernel_tuning/results/dflash_dispatch_df2fix.csv"
)
HARDWARE = {"RTX 5090": (12, 0, 170), "RTX PRO 6000": (12, 0, 188), "GB10": (12, 1, 48)}
SHAPES = {"draft_noncausal": (32, 8, False), "q3_target": (32, 8, True)}


def test_rows_follow_the_measured_receipt(monkeypatch):
    """Pin FLASHINFER_ROWS to the per-layer receipt at engine capacity C=32768.

    The provider is fixed at construction, so a row routes to FlashInfer only when
    it wins at ctx 100 by more than 2% and loses by at most 2% at ctx 400/1000.
    """
    ratios = defaultdict(dict)
    with RECEIPT.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["device"], row["shape"], int(row["Q"]), int(row["B"]))
            ratios[key][int(row["ctx"])] = float(row["flashinfer_over_cute"])
    assert len(ratios) == 3 * 2 * 7 * 4
    for (device, shape, query, batch), by_ctx in ratios.items():
        assert sorted(by_ctx) == [100, 400, 1000]
        expected = by_ctx[100] < 0.98 and max(by_ctx.values()) <= 1.02
        hq, hk, causal = SHAPES[shape]
        monkeypatch.setattr(dispatch, "hardware", lambda _, hw=HARDWARE[device]: hw)
        assert expected == dispatch.prefers_flashinfer(
            batch_size=batch,
            query_length=query,
            max_kv_tokens=32768,
            num_qo_heads=hq,
            num_kv_heads=hk,
            causal=causal,
            window=None,
            device="cuda:0",
        ), (device, shape, query, batch, by_ctx)


def test_gb10_full_b4_q4_engine_capacity_stays_on_cute(monkeypatch):
    # Review example: FlashInfer wins this row by 2% at ctx 100 but loses 15-31%
    # at ctx 400/1000 under the same construction-time choice.
    monkeypatch.setattr(dispatch, "hardware", lambda device: (12, 1, 48))
    assert not dispatch.prefers_flashinfer(
        **_row(batch_size=4, query_length=4, max_kv_tokens=32768, num_qo_heads=32, num_kv_heads=8)
    )


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("heads", [(2, 2), (8, 2), (64, 1)])
@pytest.mark.parametrize("query", [2, 8, 16, 32])
def test_auto_covers_every_mask_and_gqa(monkeypatch, capability, head_dim, heads, query):
    import torch

    from ctikernels.attention._draft import cute, flashinfer
    from ctikernels.attention.draft import DraftAttention, _provider

    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    # Exercise real geometry/device admission without requiring the optional compiler.
    import sys
    from types import ModuleType, SimpleNamespace

    package = ModuleType("cutlass")
    package.cute = ModuleType("cutlass.cute")
    monkeypatch.setitem(sys.modules, "cutlass", package)
    monkeypatch.setitem(sys.modules, "cutlass.cute", package.cute)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(dispatch, "hardware", lambda _: (*capability, 170))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=capability[0], minor=capability[1], multi_processor_count=170
        ),
    )
    for causal in (False, True):
        for window in (1, 2, 7, 64, 2048, 4096, 8191, None):
            row = _row(
                query_length=query,
                max_kv_tokens=8256,
                head_dim=head_dim,
                num_qo_heads=heads[0],
                num_kv_heads=heads[1],
                causal=causal,
                window=window,
            )
            narrow = not causal and window is not None and window < query
            expected = cute if narrow or not dispatch.prefers_flashinfer(**row) else flashinfer
            assert _provider(None, row) is expected
            assert _provider("cute", row) is cute
            assert DraftAttention.requires_page_indices(**row, dtype=torch.bfloat16) == (
                expected is flashinfer
            )


def test_auto_falls_back_when_cute_rejects(monkeypatch):
    from ctikernels.attention._draft import cute, flashinfer
    from ctikernels.attention.draft import _provider

    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)

    def reject(**kwargs):
        raise NotImplementedError("unsupported device")

    monkeypatch.setattr(cute, "check_support", reject)
    assert _provider(None, _row(head_dim=128)) is flashinfer


@pytest.mark.parametrize(
    "hardware,hq,hk,batch,query,flashinfer",
    [
        ((12, 0, 170), 16, 2, 1, 16, False),
        ((12, 0, 170), 16, 2, 2, 16, True),
        ((12, 0, 170), 16, 2, 4, 16, True),
        ((12, 0, 170), 16, 2, 1, 8, True),
        ((12, 0, 170), 8, 1, 3, 16, False),
        ((12, 0, 170), 8, 1, 1, 16, True),
        ((12, 0, 188), 16, 2, 1, 8, True),
        ((12, 0, 188), 16, 2, 2, 8, False),
        ((12, 0, 188), 16, 2, 1, 16, False),
        ((12, 0, 188), 16, 2, 2, 4, True),
        ((12, 0, 188), 8, 1, 1, 16, True),
        # No GQA-8 table on GB10: the 16/4 rows apply (B=1 Q>=9 is CuTe).
        ((12, 1, 48), 16, 2, 1, 16, False),
        ((12, 1, 48), 16, 2, 1, 8, True),
    ],
)
def test_hd256_gqa8_routed_rows(monkeypatch, hardware, hq, hk, batch, query, flashinfer):
    monkeypatch.setattr(dispatch, "hardware", lambda device: hardware)
    assert (
        dispatch.prefers_flashinfer(
            batch_size=batch,
            query_length=query,
            max_kv_tokens=32768,
            num_qo_heads=hq,
            num_kv_heads=hk,
            causal=True,
            window=None,
            device="cuda",
            head_dim=256,
        )
        is flashinfer
    )


def _hd256_flashinfer(batch, hq, hk, query=16):
    return dispatch.prefers_flashinfer(
        batch_size=batch,
        query_length=query,
        max_kv_tokens=32768,
        num_qo_heads=hq,
        num_kv_heads=hk,
        causal=True,
        window=None,
        device="cuda",
        head_dim=256,
    )


@pytest.mark.parametrize(
    "hardware",
    sorted({*dispatch.HD256_CUTE_ROWS, *dispatch.HD256_GQA8_CUTE_ROWS}),
)
@pytest.mark.parametrize("hq,hk", [(16, 4), (16, 2), (8, 1), (64, 4)])
def test_hd256_cute_rules_stop_at_b4(monkeypatch, hardware, hq, hk):
    # Every hd256 CuTe rule, including the GQA-8 table that replaces the capped one,
    # stops at B=4: B=5 and above use FlashInfer on every hardware key and query.
    monkeypatch.setattr(dispatch, "hardware", lambda device: hardware)
    for query in (2, 8, 9, 12, 16, 32):
        for batch in (5, 8, 32, 64):
            assert _hd256_flashinfer(batch, hq, hk, query)


@pytest.mark.parametrize(
    "hardware,hq,hk,query",
    [
        # Rows whose B=4 is CuTe, so the B=4/B=5 step is the cap and not a rule gap.
        ((12, 0, 170), 16, 4, 16),
        ((12, 0, 188), 16, 4, 16),
        ((12, 1, 48), 16, 4, 16),
        ((12, 1, 48), 16, 2, 16),
        ((12, 0, 188), 16, 2, 8),
        ((12, 0, 188), 16, 2, 16),
    ],
)
def test_hd256_b4_b5_boundary(monkeypatch, hardware, hq, hk, query):
    monkeypatch.setattr(dispatch, "hardware", lambda device: hardware)
    assert not _hd256_flashinfer(4, hq, hk, query)
    assert _hd256_flashinfer(5, hq, hk, query)


@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48), (12, 0, 169)])
def test_hd64_defaults_to_cute(monkeypatch, hardware):
    # D=64 has no measured FlashInfer exception: CuTe on every hardware key, batch and mask.
    monkeypatch.setattr(dispatch, "hardware", lambda device: hardware)
    assert not dispatch.FLASHINFER_HD64_ROWS.get(hardware)
    for batch in (1, 4, 5, 16, 64):
        for query in (1, 2, 16, 32):
            for causal, window in ((False, None), (True, None), (True, 4096), (False, 2048)):
                assert not dispatch.prefers_flashinfer(
                    batch_size=batch,
                    query_length=query,
                    max_kv_tokens=32768,
                    num_qo_heads=16,
                    num_kv_heads=4,
                    causal=causal,
                    window=window,
                    device="cuda",
                    head_dim=64,
                )


@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48), (12, 0, 169)])
def test_long_query_cute_rows(monkeypatch, hardware):
    # Q>32: CuTe only for B=1 D=128 full/causal rows on the measured sm_120 keys;
    # B>=2, other head dims, sliding windows and other hardware stay on FA2.
    monkeypatch.setattr(dispatch, "hardware", lambda device: hardware)
    measured = hardware in dispatch.CUTE_LONG_QUERY_ROWS

    def flashinfer(**overrides):
        row = {
            "batch_size": 1,
            "query_length": 50,
            "max_kv_tokens": 131122,
            "num_qo_heads": 16,
            "num_kv_heads": 4,
            "causal": False,
            "window": None,
            "device": "cuda",
            "head_dim": 128,
        }
        return dispatch.prefers_flashinfer(**(row | overrides))

    for query in (33, 50, 147, 1024):
        assert flashinfer(query_length=query) is not measured
        assert flashinfer(query_length=query, causal=True) is not measured
        assert flashinfer(query_length=query, batch_size=2)
        assert flashinfer(query_length=query, head_dim=64)
        assert flashinfer(query_length=query, head_dim=256)
        assert flashinfer(query_length=query, causal=True, window=4096)
        assert flashinfer(query_length=query, window=query)
        # Symmetric noncausal W<Q has no FA2 right bound: CuTe on every key.
        assert not flashinfer(query_length=query, window=7)
