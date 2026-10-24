# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Provider precedence and construction-time split/workspace arithmetic, without CUDA."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute, dispatch, flashinfer
from ctikernels.attention.draft import DraftAttention, _provider


@pytest.fixture(autouse=True)
def legacy_tuning(monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_INLINE", "0")


def geometry(b=1, q=8, window=None):
    return {
        "batch_size": b,
        "query_length": q,
        "max_kv_tokens": 4160,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 16,
        "dtype": torch.bfloat16,
        "causal": window == 4096,
        "window": window,
        "device": torch.device("cuda"),
    }


def test_provider_precedence_and_auto_fallback(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    # An sm_120 tuple: its dispatch table has no FlashInfer rows.
    monkeypatch.setattr(dispatch, "hardware", lambda device: (12, 0, 170))
    assert _provider(None, geometry()) is cute
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "flashinfer")
    assert _provider(None, geometry()) is flashinfer
    assert _provider("cute", geometry()) is cute
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "cute")
    assert _provider("flashinfer", geometry()) is flashinfer

    def unsupported(**kwargs):
        raise NotImplementedError("unsupported")

    monkeypatch.setattr(cute, "check_support", unsupported)
    with pytest.raises(NotImplementedError, match="unsupported"):
        _provider(None, geometry())
    monkeypatch.delenv("CTI_DRAFT_PROVIDER")
    assert _provider(None, geometry()) is flashinfer
    with pytest.raises(NotImplementedError, match="unsupported"):
        _provider("cute", geometry())
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "invalid")
    with pytest.raises(ValueError, match="provider"):
        _provider(None, geometry())


@pytest.mark.parametrize(
    "hardware",
    [(12, 0, 170), (12, 0, 188), (12, 1, 48), (12, 0, 169), (12, 1, 170), (11, 0, 170)],
)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 4), (64, 1), (8, 4)])
@pytest.mark.parametrize(
    "causal,window", [(False, None), (True, None), (True, 4096), (False, 2048)]
)
def test_legacy_merge_rows(monkeypatch, hardware, head_dim, hq, hk, causal, window):
    monkeypatch.setattr(dispatch, "hardware", lambda _: hardware)
    for batch in (1, 4, 64):
        for query in range(1, 33):
            pair = (query, causal, window) in (
                (16, True, 4096),
                (8, False, 2048),
            )
            expected = (
                pair
                and head_dim == 128
                and hq // hk == 4
                and hardware in ((12, 0, 170), (12, 0, 188), (12, 1, 48))
            )
            row = {
                "batch_size": batch,
                "query_length": query,
                "num_qo_heads": hq,
                "num_kv_heads": hk,
                "head_dim": head_dim,
                "causal": causal,
                "window": window,
                "device": "cuda",
            }
            assert dispatch.prefers_legacy_merge(**row) is expected
            assert cute.default_merge_mode(**row) == ("legacy" if expected else "distributed")


@pytest.mark.parametrize("minor,sms", [(0, 170), (0, 188), (1, 48)])
@pytest.mark.parametrize(
    "query,causal,window,default",
    [
        (16, False, None, "distributed"),
        (16, True, 4096, "legacy"),
        (8, False, 2048, "legacy"),
        (16, True, None, "distributed"),
        (9, False, None, "distributed"),
    ],
)
@pytest.mark.parametrize(
    "override", [None, ("MERGE", "distributed"), ("INLINE", "1"), ("INLINE", "0")]
)
def test_provider_merge_defaults_and_overrides(
    monkeypatch, minor, sms, query, causal, window, default, override
):
    for name in ("MERGE", "INLINE", "SHORT", "ROWS", "KEY_TILE", "KEY_GROUPS", "STAGES", "PIECES"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    # One requested split distinguishes inline from the legacy capacity floor.
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "1")
    monkeypatch.setenv("CTI_DRAFT_MIN_CHUNK", "64")
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", "1024")
    monkeypatch.setenv("CTI_DRAFT_COOP_PDL", "0")
    mode = default
    if override is not None:
        name, value = override
        monkeypatch.setenv(f"CTI_DRAFT_{name}", value)
        mode = "legacy" if value == "0" else "distributed"
    inline = mode == "distributed"
    short = (288 if sms == 170 else 416) if inline else 0
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=minor, multi_processor_count=sms),
    )
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    monkeypatch.setattr(cute._cooperative, "supported", lambda _: True)
    monkeypatch.setattr(cute._cooperative, "blocks_per_sm", lambda *args: 1)
    args = geometry(4, query, window) | {"causal": causal}
    _, rows, rt, splits = cute.configuration(**args)
    union = cute.tiling_union(4160, query, window)
    assert splits == (1 if inline else -(-min(union, cute.MID_UNION) // 1024))
    size, integer_size = DraftAttention.workspace_size(**args, provider="cute")
    compiled = []

    def compile_stub(*constants, head_dim, write_pages=True, **options):
        assert head_dim == 128
        assert constants[6] == splits
        assert constants[17:19] == (short, inline)
        compiled.append(constants)
        return None, None, None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(4, dtype=torch.int32)
    provider = cute.Provider(
        torch.empty(size, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, 4160),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=query,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        dtype=torch.bfloat16,
        causal=causal,
        window_left=-1 if window is None else window - 1,
    )
    assert len(compiled) == len(cute.capacity_tiers(4160, window is not None))
    assert provider.merge_mode == mode
    assert provider.inline_merge is inline
    assert provider.short_union == short
    assert provider.splits == splits
    payload = (4 * 8 * rt * provider.splits * rows * 129 + 4) * 4
    assert size == (max(256, -(-payload // 256) * 256) if splits > 1 else 256)
    assert integer_size == 256


@pytest.mark.parametrize("b", [1, 2, 3, 4])
@pytest.mark.parametrize("q", [2, 4, 5, 8, 9, 12, 13, 16, 17, 32])
@pytest.mark.parametrize("window", [None, 2048, 4096])
@pytest.mark.parametrize("splits", [1, 2, 7, None])
def test_split_workspace_agrees_with_public_sizing(monkeypatch, b, q, window, splits):
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    monkeypatch.delenv("CTI_DRAFT_MIN_CHUNK", raising=False)
    monkeypatch.delenv("CTI_DRAFT_TARGET_CTAS", raising=False)
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", str(cute.CHUNK_CAP))
    if splits is not None:
        monkeypatch.setenv("CTI_DRAFT_SPLITS", str(splits))
    args = geometry(b, q, window)
    group, rows, rt, selected = cute.configuration(**args)
    assert group == 4
    # No mocked capability (GB10 rule): B<=2 outside the 513..MID_UNION unions.
    union = cute.tiling_union(4160, q, window)
    favors_32 = union <= cute.SHORT_UNION or union > cute.MID_UNION
    legacy = min(64, max(32, (q * 4 + 15) // 16 * 16))
    assert rows == (32 if favors_32 and b <= 2 else legacy)
    assert rt == (q * 4 + rows - 1) // rows
    union = cute.tiling_union(4160, q, window)
    expected = (
        splits if splits is not None else min((union + 63) // 64, max(1, 170 // (b * 8 * rt)))
    )
    floor = (min(union, cute.MID_UNION) + cute.CHUNK_CAP - 1) // cute.CHUNK_CAP
    assert selected == max(expected, floor)
    floating, integer = DraftAttention.workspace_size(**args, provider="cute")
    sizes = [256]
    for tier in cute.capacity_tiers(4160, window is not None):
        _, tier_rows, tier_rt, tier_splits = cute.configuration(**args, union=tier)
        payload = (b * 8 * tier_rt * tier_splits * tier_rows * 129 + b) * 4
        if tier_splits > 1:
            sizes.append((payload + 255) // 256 * 256)
    assert floating == max(sizes)
    assert integer == 256


@pytest.mark.parametrize(
    "capability,b,capacity,rows",
    [
        ((12, 0), 1, 4160, 32),
        ((12, 0), 4, 4128, 64),
        ((12, 0), 4, 4129, 32),
        ((12, 0), 4, 512, 32),
        ((12, 0), 4, 513, 64),
        ((12, 1), 1, 4128, 64),
        ((12, 1), 1, 4160, 32),
        ((12, 1), 2, 512, 32),
        ((12, 1), 2, 4128, 64),
        ((12, 1), 2, 32768, 32),
        ((12, 1), 3, 32768, 64),
        ((12, 1), 3, 512, 64),
    ],
)
def test_row_tile_defaults(monkeypatch, capability, b, capacity, rows):
    monkeypatch.delenv("CTI_DRAFT_ROWS", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            multi_processor_count=170, major=capability[0], minor=capability[1]
        ),
    )
    args = {**geometry(b, 16), "max_kv_tokens": capacity}
    assert cute.configuration(**args)[1] == rows
    monkeypatch.setenv("CTI_DRAFT_ROWS", "48")
    assert cute.configuration(**args)[1] == 48


@pytest.mark.parametrize("capacity,splits", [(144, 2), (512, 2), (1024, 10)])
def test_short_unions_size_splits_for_one_cta_per_sm(monkeypatch, capacity, splits):
    for name in ("ROWS", "SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170, major=12, minor=0),
    )
    # B=4, Q=16, 32/8 heads: base = 4 * 8 * row tiles; one CTA per SM up to 512 keys.
    _, rows, rt, selected = cute.configuration(**{**geometry(4, 16), "max_kv_tokens": capacity})
    assert selected == min(-(-capacity // 64), (1 if capacity <= 512 else 2) * 170 // (32 * rt))
    assert selected == splits


@pytest.mark.parametrize("name", ["SPLITS", "MIN_CHUNK", "TARGET_CTAS"])
@pytest.mark.parametrize("override", ["0", "-1", "1.5", "invalid"])
def test_invalid_split_override_rejected(monkeypatch, name, override):
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    monkeypatch.setenv(f"CTI_DRAFT_{name}", override)
    with pytest.raises(ValueError, match="positive integer"):
        cute.configuration(**geometry())


@pytest.mark.parametrize("minimum,expected", [(128, 33), (256, 17), (512, 9)])
def test_minimum_chunk_limits_grid_and_workspace(monkeypatch, minimum, expected):
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    monkeypatch.setenv("CTI_DRAFT_MIN_CHUNK", str(minimum))
    monkeypatch.setenv("CTI_DRAFT_TARGET_CTAS", "510")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    assert cute.configuration(**geometry())[-1] == expected
    # The CTA target, rather than the context cap, limits this larger batch.
    monkeypatch.setenv("CTI_DRAFT_TARGET_CTAS", "170")
    # floor(170 / (4 * 8)): the CTAs of one launch fit a single wave.
    assert cute.configuration(**geometry(b=4))[-1] == 5
    assert cute.configuration(**(geometry() | {"max_kv_tokens": 16}))[-1] == 1


@pytest.mark.parametrize("entry", ["run", "prepare_and_run"])
@pytest.mark.parametrize("which", ["q", "k", "v", "out"])
def test_operation_rejects_grad_before_any_launch(which, entry):
    # The provider launches whatever DraftAttention bound and checked; per-call
    # gradient checks live in DraftAttention's operand check.
    from ctikernels.attention.draft import DraftAttention

    op = DraftAttention.__new__(DraftAttention)
    op._device, op._dtype = torch.device("cpu"), torch.float32
    op._b, op._q, op._p, op._hq, op._hk, op._d = 1, 1, 1, 1, 1, 4
    op._query_row_stride = 4
    op._metadata = op._readonly = op._writable = ()
    op._bind()

    def unexpected_launch(**kwargs):
        pytest.fail("launched before rejecting gradients")

    op._implementation = SimpleNamespace(run=unexpected_launch, prepare_and_run=unexpected_launch)
    tensors = {
        name: torch.zeros((1, 1, 4) if name in ("q", "out") else (1, 1, 1, 4))
        for name in ("q", "k", "v", "out")
    }
    tensors[which].requires_grad_()
    with pytest.raises(ValueError, match="gradients"):
        getattr(op, entry)(
            q=tensors["q"], paged_kv_cache=(tensors["k"], tensors["v"]), out=tensors["out"]
        )


@pytest.mark.parametrize("which", range(8))
def test_provider_prepare_rejects_grad_before_any_launch(which):
    provider = cute.Provider.__new__(cute.Provider)
    tensors = [torch.zeros(1, requires_grad=i == which) for i in range(8)]
    provider._prepare_inputs, provider._work = tuple(tensors[:7]), tensors[7]

    def unexpected_launch(*args):
        pytest.fail("launched before rejecting gradients")

    provider._prepare = unexpected_launch
    with pytest.raises(ValueError, match="gradients"):
        provider.prepare()


@pytest.mark.parametrize("changes", [{"head_dim": 32}, {"head_dim": 192}])
def test_unsupported_geometry_rejected_before_cuda(changes):
    with pytest.raises(NotImplementedError, match=r"D in \{64,128,256\}"):
        cute.check_support(**(geometry() | changes))


@pytest.mark.parametrize("b", [1, 3])
@pytest.mark.parametrize("q", [5, 16])
def test_large_full_capacity_uses_occupancy_splits(monkeypatch, b, q):
    for name in ("SPLITS", "MIN_CHUNK", "TARGET_CTAS", "ROWS"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", str(cute.CHUNK_CAP))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    capacity = 32768
    args = geometry(b, q) | {"max_kv_tokens": capacity}
    _, _, rt, splits = cute.configuration(**args)
    expected = min((capacity + 63) // 64, max(1, 170 // (b * 8 * rt)))
    floor = (min(capacity, cute.MID_UNION) + cute.CHUNK_CAP - 1) // cute.CHUNK_CAP
    assert splits == max(expected, floor) == expected
    assert splits < (capacity + cute.CHUNK_CAP - 1) // cute.CHUNK_CAP


@pytest.mark.parametrize(
    "capacity", [0, 127, 1024, 1025, 4096, 4128, 4129, 32768, 44033, 2**31 - 1]
)
@pytest.mark.parametrize("cap", [64, 128, 1024, 4096])
@pytest.mark.parametrize("minimum", [64, 128, 255, 256, 2048])
@pytest.mark.parametrize("override", [None, 1, 7, 43])
def test_split_capacity_covers_every_runtime_length(monkeypatch, capacity, cap, minimum, override):
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", str(cap))
    monkeypatch.setenv("CTI_DRAFT_MIN_CHUNK", str(minimum))
    monkeypatch.setenv("CTI_DRAFT_TARGET_CTAS", "1")
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    if override is not None:
        monkeypatch.setenv("CTI_DRAFT_SPLITS", str(override))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    splits = cute.configuration(**(geometry() | {"max_kv_tokens": capacity}))[-1]
    # A target of one CTA selects one occupancy split before applying the floor.
    expected = override if override is not None else 1
    assert splits == max(expected, (min(capacity, cute.MID_UNION) + cap - 1) // cap)
    # Retain the existing arithmetic mirror; sample the int32 endpoint without
    # iterating through two billion lengths. Rounded chunks need not fit int32.
    unions = (
        range(capacity + 1) if capacity < 2**31 - 1 else (0, 1, 2055, 4103, capacity - 1, capacity)
    )
    for union in unions:
        for kt in (32, 64):
            chunk = max(((union + splits - 1) // splits + kt - 1) // kt * kt, minimum)
            count = (union + chunk - 1) // chunk
            assert count <= splits
            assert count * chunk >= union
            assert chunk >= minimum


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    "hardware,hd128,hd256",
    [
        ((12, 0, 170), (192, 288, 288, 288, 288), (160,) * 5),
        ((12, 0, 188), (288, 288, 416, 416, 416), (288,) * 5),
        ((12, 1, 48), (416, 416, 416, 416, 416), (288,) * 5),
        ((12, 0, 169), (288,) * 5, (288,) * 5),
        ((12, 1, 170), (288,) * 5, (288,) * 5),
        ((11, 0, 170), (288,) * 5, (288,) * 5),
    ],
)
def test_short_defaults_by_hardware_and_batch(monkeypatch, head_dim, hardware, hd128, hd256):
    expected = hd128 if head_dim == 128 else hd256
    monkeypatch.setattr(dispatch, "hardware", lambda _: hardware)
    monkeypatch.delenv("CTI_DRAFT_SHORT", raising=False)
    monkeypatch.setenv("CTI_DRAFT_MERGE", "distributed")
    for batch, short in zip((1, 2, 3, 4, 64), expected, strict=True):
        default = cute.short_default("cuda", batch, head_dim)
        assert default == short
        assert cute.merge_tuning(default) == (True, short)
        for override in (0, 128, 416):
            with monkeypatch.context() as patch:
                patch.setenv("CTI_DRAFT_SHORT", str(override))
                assert cute.merge_tuning(default) == (True, override)
    # The head dimensions retain independent tuning tables.
    assert cute.SHORT_ROWS[128] is not cute.SHORT_ROWS[256]


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("inline", [0, 1])
@pytest.mark.parametrize("short", [None, 0, 416])
@pytest.mark.parametrize("capacity", [16, 4128, 32768])
@pytest.mark.parametrize("minor,sms", [(0, 170), (0, 188), (1, 48)])
def test_provider_initializes_arrivals_once(
    monkeypatch, inline, short, capacity, minor, sms, batch, head_dim
):
    for name in ("ROWS", "SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP", "KEY_TILE"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", str(inline))
    if short is None:
        monkeypatch.delenv("CTI_DRAFT_SHORT", raising=False)
        defaults = {170: (192, 288, 288, 288), 188: (288, 288, 416, 416), 48: (416,) * 4}
        if head_dim == 256:
            defaults = {170: (160,) * 4, 188: (288,) * 4, 48: (288,) * 4}
        short = defaults[sms][batch - 1] if inline else 0
    else:
        monkeypatch.setenv("CTI_DRAFT_SHORT", str(short))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=sms, major=12, minor=minor),
    )
    args = geometry(b=batch, q=4) | {"max_kv_tokens": capacity, "head_dim": head_dim}
    size, _ = cute.workspace_size(**args)
    work = torch.full((size,), 255, dtype=torch.uint8)
    calls = []

    expected_dim = head_dim

    def compile_stub(*constants, head_dim=128, write_pages=True, **options):
        assert head_dim == expected_dim
        assert constants[-2:] == (short, bool(inline))
        assert constants[11] == (64 if head_dim == 128 and (inline or minor == 1) else 32)
        assert constants[10] == 64  # MIN_CHUNK stays unchanged.
        return (lambda *args: None), (lambda *args: calls.append(args)), None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(batch, dtype=torch.int32)
    provider = cute.Provider(
        work,
        integer,
        prepare_inputs=(integer, integer, integer, integer, capacity),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=4,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
        page_size=16,
        dtype=torch.bfloat16,
        causal=False,
        window_left=-1,
    )
    _, rows, rt, splits = cute.configuration(**args)
    assert provider.inline_merge == bool(inline)
    assert provider.short_union == short
    assert provider.chunk_cap >= short
    assert provider.chunk_cap % provider.key_tile == 0
    if capacity <= 4128 and (splits > 1 or capacity <= cute.chunk_cap(args["device"])):
        # Reserving the short single split must not turn on the piece loop.
        assert not provider.pieces
    if inline and capacity == 16:
        assert provider.chunk_cap == max(64, -(-short // provider.key_tile) * provider.key_tile)
    assert provider._counters.dtype == torch.int32
    assert provider._counters.numel() == batch * 8 * rt * sum(provider.merge_levels)
    if provider.spread:
        # A spread grid row of S >= B * s CTAs sizes the tree.
        assert provider.splits >= batch * splits
        splits = provider.splits
    expected_levels = cute.counter_levels(splits, provider.merge_fan_in) if inline else ()
    assert provider.merge_levels == expected_levels
    assert torch.count_nonzero(provider._counters) == 0
    if provider._counters.numel():
        assert provider._counters.untyped_storage().data_ptr() != work.untyped_storage().data_ptr()
    assert (work == 255).all()
    work.zero_()
    assert torch.count_nonzero(provider._counters) == 0
    # Runtime merely forwards the stable view; no implicit memset per call.
    provider._counters.fill_(17)
    provider.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
    assert calls[0][8] is provider._counters
    assert (provider._counters == 17).all()


@pytest.mark.parametrize(
    "name,value", [("SHORT", "-1"), ("SHORT", "1.5"), ("INLINE", "2"), ("INLINE", "-1")]
)
def test_invalid_merge_tuning(monkeypatch, name, value):
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    monkeypatch.setenv(f"CTI_DRAFT_{name}", value)
    with pytest.raises(ValueError, match=f"CTI_DRAFT_{name}"):
        cute.configuration(**geometry())


@pytest.mark.parametrize("short", [-1, 1025, 1.5])
def test_short_union_constructor_validation(short):
    cutlass = pytest.importorskip("cutlass")
    from ctikernels.attention._draft.cute._kernel import Attention

    with pytest.raises(ValueError, match="short_union <= chunk_cap"):
        Attention(4, 4, 32, 8, False, False, 7, cutlass.BFloat16, short_union=short)


@pytest.mark.parametrize("sms,minor", [(48, 1), (170, 0), (188, 0)])
@pytest.mark.parametrize("batch", [1, 2, 4, 64])
@pytest.mark.parametrize("query", [4, 9, 16, 32])
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("capacity", [100, 4100, 32768])
def test_inline_occupancy_target(monkeypatch, sms, minor, batch, query, tile, capacity):
    for name in ("ROWS", "SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.setenv("CTI_DRAFT_KEY_TILE", str(tile))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=sms, major=12, minor=minor),
    )
    args = geometry(batch, query) | {"max_kv_tokens": capacity}
    _, rows, rt, splits = cute.configuration(**args)
    base = batch * 8 * rt
    factor, _ = cute.row_tuning((12, minor), batch, rows, base)
    if capacity <= cute.SHORT_UNION:
        factor = 1
    _, arena = cute.merge_geometry(rows, tile, cute.key_groups(rows, (12, minor)), 2, True)
    cap = cute.chunk_cap(args["device"])
    threads = rows // 16 * 32 * cute.key_groups(rows, (12, minor))
    if 2 * (arena + cap * 4 + 1024) > 100 * 1024 or 2 * 255 * threads > 65536:
        factor = 1
    assert splits == min(-(-capacity // 64), max(1, factor * sms // base))
    # Tuning may exceed a resident wave: hierarchy correctness does not depend on it.
    monkeypatch.setenv("CTI_DRAFT_TARGET_CTAS", str(sms * 3))
    assert cute.configuration(**args)[-1] == min(-(-capacity // 64), max(1, sms * 3 // base))


@pytest.mark.parametrize("batch,override", [(1, "24"), (32, None)])
def test_provider_keeps_inline_when_grid_cannot_reside(monkeypatch, batch, override):
    for name in ("ROWS", "SPLITS", "MIN_CHUNK", "TARGET_CTAS", "CHUNK_CAP", "KEY_TILE"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    if override:
        monkeypatch.setenv("CTI_DRAFT_SPLITS", override)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=188, major=12, minor=0),
    )
    args = geometry(batch, 4) | {"max_kv_tokens": 32768}
    size, _ = cute.workspace_size(**args)
    launches = []

    def compile_stub(*constants, head_dim=128, write_pages=True, **options):
        assert head_dim == 128
        assert constants[-1] is True
        merge = None
        return lambda *args: None, lambda *args: launches.append("attention"), merge

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(batch, dtype=torch.int32)
    provider = cute.Provider(
        torch.zeros(size, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, 32768),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=4,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        dtype=torch.bfloat16,
        causal=False,
        window_left=-1,
    )
    assert provider.inline_merge
    assert provider.splits * batch * 8 > 188
    provider.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
    assert launches == ["attention"]


def test_provider_rejects_short_above_index_capacity(monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.setenv("CTI_DRAFT_SHORT", "416")
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", "384")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170, major=12, minor=0),
    )
    integer = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="SHORT <= CHUNK_CAP"):
        cute.Provider(
            torch.empty(256, dtype=torch.uint8),
            integer,
            prepare_inputs=(integer, integer, integer, integer, 128),
            query_indptr=integer,
            kv_indptr=integer,
            kv_indices=integer,
            last_page_lengths=integer,
            query_length=4,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            dtype=torch.bfloat16,
            causal=False,
            window_left=-1,
        )


@pytest.mark.parametrize("batch,query,heads,page", [(1, 9, 4, 16), (3, 4, 8, 16)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("minor,sms", [(0, 170), (0, 188), (1, 48)])
def test_default_short_adapts_to_access_check_chunk_cap(
    monkeypatch, batch, query, heads, page, head_dim, minor, sms
):
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.delenv("CTI_DRAFT_SHORT", raising=False)
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", "256")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=sms, major=12, minor=minor),
    )
    monkeypatch.setattr(cute, "compile_kernels", lambda *args, **kwargs: (None, None, None))
    integer = torch.zeros(batch, dtype=torch.int32)
    work = torch.empty(1 << 24, dtype=torch.uint8)

    def construct():
        return cute.Provider(
            work,
            integer,
            prepare_inputs=(integer, integer, integer, integer, 8192),
            query_indptr=integer,
            kv_indptr=integer,
            kv_indices=integer,
            last_page_lengths=integer,
            query_length=query,
            num_qo_heads=heads * 4,
            num_kv_heads=heads,
            head_dim=head_dim,
            page_size=page,
            dtype=torch.bfloat16,
            causal=False,
            window_left=-1,
        )

    provider = construct()
    expected_short = 192 if sms == 170 and batch == 1 else 256
    if head_dim == 256:
        expected_short = 160 if sms == 170 else 256
    assert provider.short_union == expected_short
    assert provider.chunk_cap == 256
    assert provider.pieces
    assert provider._counters.numel() > 0
    provider._counters.fill_(19)
    other = construct()
    assert other._counters.data_ptr() != provider._counters.data_ptr()
    assert torch.count_nonzero(other._counters) == 0
    work.fill_(255)
    assert (provider._counters == 19).all()
    assert torch.count_nonzero(other._counters) == 0


@pytest.mark.parametrize(
    "fan_in,expected",
    [
        (2, {1: (), 2: (1,), 3: (2, 1), 4: (2, 1), 5: (3, 2, 1), 65: (33, 17, 9, 5, 3, 2, 1)}),
        (4, {1: (), 2: (1,), 4: (1,), 5: (2, 1), 16: (4, 1), 17: (5, 2, 1), 65: (17, 5, 2, 1)}),
    ],
)
def test_hierarchy_counter_sizes(fan_in, expected):
    for splits, levels in expected.items():
        assert cute.counter_levels(splits, fan_in) == levels
        # Exactly one counter per group; include singleton tails and root.
        for count in range(2, splits + 1):
            runtime = cute.counter_levels(count, fan_in)
            assert len(runtime) <= len(levels)
            assert all(n <= cap for n, cap in zip(runtime, levels, strict=False))


@pytest.mark.parametrize("rows,tile,expected", [(32, 32, 2), (32, 64, 4), (64, 32, 2), (64, 64, 2)])
def test_merge_arena_holds_whole_fan_in(rows, tile, expected):
    fan_in, arena = cute.merge_geometry(rows, tile, 2, 2, True)
    assert fan_in == expected
    assert arena >= fan_in * rows * 129 * 4 + 4
    assert arena + 4096 + 1024 <= 99 * 1024


def test_compile_key_has_only_windowed():
    import inspect

    parameters = inspect.signature(cute.compile_kernels).parameters
    assert "window" not in parameters
    assert "windowed" in parameters


@pytest.mark.parametrize("window", [1, 2, 7, 64, 2048, 3000, 4096, 8191, 2**31 - 1, None])
@pytest.mark.parametrize("capacity", [32, 8256, 2**31 - 1])
def test_tiling_union_tiers(window, capacity):
    query = 8
    union = cute.tiling_union(capacity, query, window)
    actual = capacity if window is None else min(capacity, window + query - 1)
    assert actual <= union <= capacity
    assert union in cute.capacity_tiers(capacity, window is not None)
    if window is None:
        assert union == actual
    if window == 3000:
        assert union == cute.tiling_union(capacity, query, 4096)
    if window == 8191:
        assert union == cute.tiling_union(capacity, query, 8192)


@pytest.mark.parametrize("minor,sms", [(0, 170), (1, 48)])
@pytest.mark.parametrize("mode", ["legacy", "tree"])
@pytest.mark.parametrize("cap", [64, 1024])
def test_window_width_reuses_provider_compile(monkeypatch, minor, sms, mode, cap):
    from functools import lru_cache

    for name in ("ROWS", "SPLITS", "MIN_CHUNK", "TARGET_CTAS", "KEY_TILE", "PIECES"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setenv("CTI_DRAFT_SHORT", "0")
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", str(cap))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=minor, multi_processor_count=sms),
    )
    launches = []

    @lru_cache(maxsize=None)
    def compile_stub(*constants, **kwargs):
        assert type(constants[5]) is bool
        return None, lambda *args: launches.append(args), None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(4, dtype=torch.int32)
    work = torch.empty(1 << 20, dtype=torch.uint8)
    for window in (4096, 3000, 1, 7, 64, 8191, 2**31 - 1, 2**40, None):
        args = geometry(b=4, window=window) | {"max_kv_tokens": 8256}
        provider = cute.Provider(
            work,
            integer,
            prepare_inputs=(integer, integer, integer, integer, 8256),
            query_indptr=integer,
            kv_indptr=integer,
            kv_indices=integer,
            last_page_lengths=integer,
            query_length=8,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            dtype=torch.bfloat16,
            causal=False,
            window_left=-1 if window is None else window - 1,
        )
        if window == 3000:
            assert compile_stub.cache_info().misses == 2
            other = args | {"window": 4096}
            assert cute.configuration(**args) == cute.configuration(**other)
            assert cute.workspace_size(**args) == cute.workspace_size(**other)
        provider.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
        assert launches[-1][-2:] == (0, min(window or 0, 2**31 - 1))
        for length in (8, 100, 3000, 8256):
            union = length if window is None else min(length, window + 7)
            need = -(-union // provider.splits)
            chunk = max(
                -(-need // provider.key_tile) * provider.key_tile,
                provider.min_chunk,
            )
            assert -(-union // chunk) <= provider.splits
            assert provider.pieces or chunk <= provider.chunk_cap


@pytest.mark.parametrize(
    "batch,splits,row_ctas,target,limit,expected",
    [
        (3, 2, 24, 170, 3888, 7),  # RTX 5090 B3 Q18 32/8: fill to target // row CTAs
        (3, 1, 16, 48, 3888, 3),  # GB10 B3 at 16 row CTAs: S = B, no room
        (4, 8, 2, 170, 8, 32),  # short capacity: no request takes more than limit
        (4, 5, 8, None, 64, 20),  # CTI_DRAFT_SPLITS override: B * s
        (33, 1, 8, 170, 256, 54),
        (64, 1, 8, 188, 256, 87),
        (84, 1, 8, 188, 256, 107),
        (256, 1, 8, 48, 256, 262),
        (33, 1, 8, 1, 256, 34),  # At least one slack CTA.
        (64, 2, 8, 188, 2, 128),  # Capacity leaves no room.
        (84, 2, 8, None, 256, 168),
        (84, 1, 8, 2**40, 2**31, (2**31 - 1) // 8),
        (84, 1, 1, 2**40, 2**31, 2**31 - 1),
    ],
)
def test_spread_grid_rows(monkeypatch, batch, splits, row_ctas, target, limit, expected):
    monkeypatch.delenv("CTI_DRAFT_SPREAD_SLACK", raising=False)
    assert cute.spread_splits(batch, splits, row_ctas, target, limit) == expected
    monkeypatch.setenv("CTI_DRAFT_SPREAD_SLACK", "1")
    slack = batch - 1 if batch <= 32 else 0
    assert cute.spread_splits(batch, splits, row_ctas, target, limit) == expected + slack


@pytest.mark.parametrize("environment", [None, "1"])
@pytest.mark.parametrize("splits,target,limit", [(1, 188, 1), (2, 188, 2), (2, None, 256)])
def test_spread_wide_without_room(monkeypatch, environment, splits, target, limit):
    monkeypatch.delenv("CTI_DRAFT_SPREAD", raising=False)
    if environment is not None:
        monkeypatch.setenv("CTI_DRAFT_SPREAD", environment)
    assert cute.spread_grid(84, splits, 8, target, limit, True, False) == (0, splits)


@pytest.mark.parametrize(
    "batch,inline,table,environment,expected",
    [
        (1, True, False, None, 0),
        (2, True, False, None, 2),
        (32, True, False, None, 32),
        (33, True, False, "1", 33),
        (33, True, False, None, 33),
        (64, True, False, None, 64),
        (84, True, False, None, 84),
        (256, True, False, None, 256),
        (84, True, False, "0", 0),
        (84, False, False, "1", 0),
        (84, True, "row", "1", 0),
        (4, False, False, None, 0),
        (4, True, "row", None, 0),
        (4, True, False, "0", 0),
    ],
)
def test_spread_batch(monkeypatch, batch, inline, table, environment, expected):
    monkeypatch.delenv("CTI_DRAFT_SPREAD", raising=False)
    if environment is not None:
        monkeypatch.setenv("CTI_DRAFT_SPREAD", environment)
    assert cute.spread_batch(batch, inline, table) == expected


@pytest.mark.parametrize(
    "b,q,capacity",
    [(2, 18, 4160), (3, 18, 241957), (4, 18, 32768), (8, 8, 32768), (84, 18, 32768)],
)
def test_spread_workspace_holds_the_grid(monkeypatch, b, q, capacity):
    # Inline merge (the spread path), sm_120 with 170 SMs: the borrowed workspace must
    # hold Hkv * row_tiles * S partial slots plus the per-request counts.
    pytest.importorskip("cutlass.cute")  # workspace_size(provider="cute") needs the DSL
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.delenv("CTI_DRAFT_SPREAD", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170),
    )
    args = {**geometry(b, q), "max_kv_tokens": capacity, "causal": False}
    _, rows, rt, splits, target, limit, inline = cute._configuration(**args)
    spread, grid = cute.spread_grid(b, splits, 8 * rt, target, limit, inline, False)
    assert inline and spread == b and grid >= b * splits
    floating, _ = DraftAttention.workspace_size(**args, provider="cute")
    assert floating >= (8 * rt * grid * rows * 129 + b) * 4


@pytest.mark.parametrize(
    "batch,sms,minor", [(33, 170, 0), (64, 188, 0), (84, 48, 1), (256, 170, 0)]
)
@pytest.mark.parametrize("merge", [None, "tree", "distributed"])
def test_spread_wide_provider_merge(monkeypatch, batch, sms, minor, merge):
    for name in ("MERGE", "SPLITS", "TARGET_CTAS", "SPREAD", "ROWS", "SPREAD_SLACK"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.setenv("CTI_DRAFT_COOP_PDL", "0")
    if merge is not None:
        monkeypatch.setenv("CTI_DRAFT_MERGE", merge)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=minor, multi_processor_count=sms),
    )
    cooperative = []
    monkeypatch.setattr(cute._cooperative, "supported", lambda _: cooperative.append(True) or True)
    monkeypatch.setattr(cute._cooperative, "blocks_per_sm", lambda *args: 1000)
    compiled = []

    def compile_stub(*constants, **options):
        compiled.append((constants, options))
        return None, None, None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    args = geometry(batch, 8) | {"max_kv_tokens": 16384}
    _, rows, rt, splits, target, limit, inline = cute._configuration(**args)
    spread, grid = cute.spread_grid(batch, splits, 8 * rt, target, limit, inline, False)
    size, _ = cute.workspace_size(**args)
    assert spread == batch and grid > batch * splits
    assert size >= (8 * rt * grid * rows * 129 + batch) * 4
    integer = torch.zeros(batch, dtype=torch.int32)
    provider = cute.Provider(
        torch.empty(size, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, 16384),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=8,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        dtype=torch.bfloat16,
        causal=False,
        window_left=-1,
    )
    assert provider.spread == batch
    assert provider.splits == grid
    assert provider.spread_budget == batch * splits
    assert provider.merge == (merge or "tree")
    assert provider.fallbacks == ()
    assert cooperative == ([True] if merge == "distributed" else [])
    assert len(compiled) == 1
    constants, options = compiled[0]
    assert constants[6] == grid
    assert options["spread"] == batch
    assert options["spread_budget"] == batch * splits
    if merge != "distributed":
        levels = cute.counter_levels(grid, provider.merge_fan_in)
        assert provider.merge_levels == levels
        assert provider._counters.numel() == batch * 8 * rt * sum(levels)
