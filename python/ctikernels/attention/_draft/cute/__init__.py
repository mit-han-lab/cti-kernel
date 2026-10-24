# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Construction-time CuTe compilation; runtime only borrows buffers and launches."""

import os
import threading
import warnings
from functools import lru_cache

import torch

from ...._storage import reject_grad
from .. import dispatch
from . import _cooperative
from ._geometry import counter_levels, merge_geometry

_compiling = threading.Lock()
# The index buffer bounds each CTA independently of the runtime CSR lengths.
CHUNK_CAP = 1024
KEY_TILE = 32
STAGES = 2
MIN_CHUNK = 64
# DF3 per-layer graph chains, C=32768, Hq/Hkv=32/8, D=128: inline TILE=64
# wins on sm_120 and sm_121 (GB10). At ctx=100: 0.83-0.88x vLLM on RTX 5090,
# 0.94-0.996x on RTX PRO 6000.
INLINE_KEY_TILE = 64
# Receipt: cooperative per-layer chains, C=32768, D128, 32/8 heads, Q4/8/16, ctx100-400; loss count then geomean vs min(vLLM, SGLang FlashInfer 0.7.1).
# D -> (major, minor, SMs) -> ordered (inclusive maximum B, SHORT); None is unbounded.
# D256: benchmark/kernel_tuning/results/dflash_df3_hd256_dispatch.csv.
SHORT_ROWS = {
    128: {
        (12, 0, 170): ((1, 192), (None, 288)),
        (12, 0, 188): ((2, 288), (None, 416)),
        (12, 1, 48): ((None, 416),),
    },
    256: {
        (12, 0, 170): ((None, 160),),
        (12, 0, 188): ((None, 288),),
        (12, 1, 48): ((None, 288),),
    },
}
DEFAULT_SHORT = 288
# Tuning breakpoint from the DF2/DF3 paired sweeps, independent of window width.
MID_UNION = 4128
# (major, minor, SMs) -> smallest B that takes 32-row tiles over every key union.
# RTX PRO 6000 paired per-layer chains, Q9/12/16, B3/B4, ctx 1000/2000/4000 at exact
# capacity: 0.88-0.997x SGLang FlashInfer 0.7.1 against 0.96-1.05x with 48/64 rows
# (benchmark/kernel_tuning/results/dflash_df4_receipts.md).
ROWS32_MIN_BATCH = {(12, 0, 188): 3}
# Key unions up to this bound take 32-row tiles at any batch size.
SHORT_UNION = 512
# GB10 (sm_121) per-row split counts from three-process paired sweeps (DF2 round 5):
# full attention, B=3, Q=4, 16/4 heads just past 4096 keys measured 52-59 us with
# three splits against 58-79 us with the occupancy default of four.
GB10_SPLITS = (
    {
        "full": True,
        "batch": 3,
        "query": 4,
        "kv_heads": 4,
        "group": 4,
        "union": (4097, MID_UNION),
        "splits": 3,
    },
)


# Largest B scheduled in one warp group: one lane per request.
SPREAD_MAX_BATCH = 32


def spread_batch(batch_size, inline, table):
    """B when one grid row of B * splits CTAs follows the runtime lengths, else 0.

    Per-request grids give every request the same construction-time split count, so a
    ragged batch's longest request is walked by few CTAs (32/8 Q18 B3 at a 199k
    context in a random-shape evaluation row: 0.39x SGLang). Uniform batches keep the same
    chunks under either grid.
    """
    value = os.environ.get("CTI_DRAFT_SPREAD")
    enabled = batch_size >= 2
    if value is not None:
        enabled = nonnegative_env("CTI_DRAFT_SPREAD", 0) == 1
    if not enabled or not inline or table or batch_size < 2:
        return 0
    return batch_size


def spread_splits(batch_size, splits, row_ctas, target, limit):
    """CTAs per row tile of a spread grid: B * s, filled up to the occupancy target.

    Uniform batches use B * s of them with the per-request chunks. Ragged batches
    spread the rest over their long requests; with s = 1 (S = B) there is no room
    unless the target leaves some. CTI_DRAFT_SPREAD_SLACK=1 adds B - 1 more CTAs,
    past the target (measured: cooperative residency then selects the tree merge).
    """
    grid = batch_size * splits
    if batch_size > SPREAD_MAX_BATCH:
        if target is not None:
            grid += max(1, target // row_ctas)
        return min(grid, batch_size * limit, (2**31 - 1) // row_ctas, 2**31 - 1)
    if target is not None:
        # No request takes more than limit = ceil(union / MIN_CHUNK) splits.
        grid = max(grid, min(target // row_ctas, batch_size * limit))
    if nonnegative_env("CTI_DRAFT_SPREAD_SLACK", 0):
        grid += batch_size - 1
    return grid


def spread_grid(batch_size, splits, row_ctas, target, limit, inline, table):
    """(spread B or 0, CTAs per row tile) for one capacity tier."""
    spread = spread_batch(batch_size, inline, table)
    if spread:
        grid = spread_splits(spread, splits, row_ctas, target, limit)
        if spread > SPREAD_MAX_BATCH and grid <= batch_size * splits:
            return 0, splits
        if grid > spread or "CTI_DRAFT_SPREAD" in os.environ:
            return spread, grid
    # S = B leaves every request one CTA either way (GB10 B=3 at 16 row CTAs: no
    # change on ragged rows, 0.94x at ctx 100); keep the per-request grid.
    return 0, splits


def capacity_tiers(capacity, windowed):
    return tuple(sorted({min(capacity, MID_UNION), capacity})) if windowed else (capacity,)


def tiling_union(max_kv_tokens, query_length, window):
    actual = max_kv_tokens if window is None else min(max_kv_tokens, window + query_length - 1)
    return next(
        tier for tier in capacity_tiers(max_kv_tokens, window is not None) if tier >= actual
    )


def gb10_splits(*, batch_size, query_length, num_kv_heads, group, windowed, union):
    for rule in GB10_SPLITS:
        if (
            (not windowed) == rule["full"]
            and batch_size == rule["batch"]
            and query_length == rule["query"]
            and num_kv_heads == rule["kv_heads"]
            and group == rule["group"]
            and rule["union"][0] <= union <= rule["union"][1]
        ):
            return rule["splits"]
    return None


def key_groups(rows, capability=(12, 0)):
    # Device sweeps: two key groups pay off through three row warps on sm_120;
    # GB10 (sm_121) prefers two at every row count.
    return 2 if rows <= 48 or capability == (12, 1) else 1


def positive_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def requested_merge_mode(default="distributed"):
    mode = os.environ.get("CTI_DRAFT_MERGE")
    if mode is None:
        mode = default
        if "CTI_DRAFT_INLINE" in os.environ:
            mode = "distributed" if nonnegative_env("CTI_DRAFT_INLINE", 1) else "legacy"
    if mode not in ("distributed", "tree", "legacy"):
        raise ValueError("CTI_DRAFT_MERGE must be distributed, tree or legacy")
    return mode


def cooperative_pdl():
    value = nonnegative_env("CTI_DRAFT_COOP_PDL", 1)
    if value not in (0, 1):
        raise ValueError("CTI_DRAFT_COOP_PDL must be 0 or 1")
    return bool(value)


def short_default(device, batch_size, head_dim):
    for maximum, short in SHORT_ROWS.get(head_dim, {}).get(dispatch.hardware(device), ()):
        if maximum is None or batch_size <= maximum:
            return short
    return DEFAULT_SHORT


def default_merge_mode(**row):
    return "legacy" if dispatch.prefers_legacy_merge(**row) else "distributed"


def merge_tuning(default_short=DEFAULT_SHORT, default_mode="distributed"):
    inline = nonnegative_env("CTI_DRAFT_INLINE", 1)
    if inline not in (0, 1):
        raise ValueError("CTI_DRAFT_INLINE must be 0 or 1")
    inline = requested_merge_mode(default_mode) != "legacy"
    return inline, nonnegative_env("CTI_DRAFT_SHORT", default_short if inline else 0)


def _capability(device):
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "major", 0), getattr(properties, "minor", 0)


def chunk_cap(device):
    # GB10 sweeps: larger index chunks avoid forced extra splits (partial waves)
    # on DRAM-bound B>=2 rows near L=4096; the 16 KiB index buffer still fits in shared memory.
    return 4096 if _capability(device) == (12, 1) else CHUNK_CAP


def row_tuning(capability, batch_size, rows, base):
    """Per-row defaults (CTA target in SMs, key tile) from paired sweeps per device.

    sm_120: with B*Hkv*row-tiles >= 24 CTAs per split, two CTAs per SM (twice the
    splits) beat one (RTX PRO 6000 B3/B4 at L=4096: 0.80-0.82x vs 0.90-1.00x the
    stock graph). sm_121 (GB10): B=1 rows packing 33-48 query rows run two 32-key-
    tile CTAs per SM instead of one 64-key-tile CTA.
    """
    if capability == (12, 0):
        return (2 if base >= 24 else 1), KEY_TILE
    if capability == (12, 1):
        if batch_size == 1 and rows == 48:
            return 2, 32
        return 1, 64
    return 1, KEY_TILE


def check_support(device, *, query_length, num_qo_heads, num_kv_heads, head_dim, **kwargs):
    if head_dim not in (64, 128, 256):
        raise NotImplementedError("DRAFT CuTe requires D in {64,128,256}")
    if query_length < 1 or num_kv_heads < 1 or num_qo_heads < 1 or num_qo_heads % num_kv_heads:
        raise ValueError("DRAFT CuTe requires Q >= 1 and dividing GQA heads")
    if num_kv_heads * (-(-query_length * (num_qo_heads // num_kv_heads) // 64)) > 65535:
        raise NotImplementedError("DRAFT CuTe grid-y exceeds 65535")
    try:
        import cutlass.cute  # noqa: F401
    except ImportError as error:
        raise NotImplementedError("DRAFT CuTe requires nvidia-cutlass-dsl") from error
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise NotImplementedError("DRAFT CuTe requires capability (12,0) or (12,1)")
    # The actual row tile can be 16/32/48/64, including environment overrides.
    # Check every compiled capacity tier here so auto can fall back before sizing.
    if "max_kv_tokens" in kwargs:
        for tier in capacity_tiers(kwargs["max_kv_tokens"], kwargs.get("window") is not None):
            configuration(
                device=device,
                query_length=query_length,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                union=tier,
                **kwargs,
            )


def configuration(**kwargs):
    """(group, rows, row tiles, per-request splits) for one capacity tier."""
    return _configuration(**kwargs)[:4]


def _configuration(
    *,
    batch_size,
    query_length,
    max_kv_tokens,
    num_qo_heads,
    num_kv_heads,
    window,
    device,
    head_dim=128,
    causal=False,
    union=None,
    table=False,
    **kwargs,
):
    mode = (
        "distributed"
        if table
        else default_merge_mode(
            batch_size=batch_size,
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            window=1 if window is not None else None,
            device=device,
        )
    )
    inline, _ = merge_tuning(default_mode=mode)
    group = num_qo_heads // num_kv_heads
    # At least two row warps keep the default two key groups at four or more warps.
    rows = min(64, max(32, (query_length * group + 15) // 16 * 16))
    if union is None:
        union = tiling_union(max_kv_tokens, query_length, window)
    favors_32 = union <= SHORT_UNION or union > MID_UNION
    if _capability(device) == (12, 0):
        # DF4 measured the B>=3 32-row rule with the inline merge; legacy-merge
        # rows (DW1's frozen draft pairs) keep their 0.1.12 tiles.
        minimum = ROWS32_MIN_BATCH.get(dispatch.hardware(device)) if inline else None
        use_32 = batch_size == 1 or favors_32 or (minimum is not None and batch_size >= minimum)
    else:
        # GB10 keeps its tuned B=1 tiles over 513-4128 keys (32 rows there cost up
        # to 1.3x on hot-L2 replays) and the longer tiles at B>=3 (DRAM re-reads).
        use_32 = batch_size <= 2 and favors_32
    if use_32:
        # 32-row tiles double the CTAs per KV head without a merge (paired sweeps:
        # up to 1.3x at Q=9-16), also under large capacities whose runtime lengths
        # vary (per-layer sweeps at C=32768). At B>=2 over unions of 513-4128 keys
        # the second row tile's K/V re-read costs more than it gains (up to 1.2x on
        # hot-L2 replays).
        rows = 32
    if head_dim == 256:
        # Opt-in VERIFY: 64 KiB KV ring, two row warps per key group. Q16/G4
        # takes two row tiles; preserve the measured D128 tuning above.
        rows = 32
    if "CTI_DRAFT_ROWS" in os.environ:
        rows = positive_env("CTI_DRAFT_ROWS", rows)
        if rows not in (16, 32, 48, 64):
            raise ValueError("CTI_DRAFT_ROWS must be 16, 32, 48 or 64")
    rt = (query_length * group + rows - 1) // rows
    if num_kv_heads * rt > 65535 or batch_size > 65535:
        raise NotImplementedError("DRAFT CuTe grid-y/grid-z exceeds 65535")
    minimum = positive_env("CTI_DRAFT_MIN_CHUNK", MIN_CHUNK)
    limit = max(1, (union + minimum - 1) // minimum)
    cap = positive_env("CTI_DRAFT_CHUNK_CAP", chunk_cap(device))
    override = os.environ.get("CTI_DRAFT_SPLITS")
    target = None
    if override is not None:
        splits = positive_env("CTI_DRAFT_SPLITS", 1)
    else:
        # Round down to the occupancy target: a partial additional wave costs
        # a whole tile walk. This is a performance choice, not a residency rule.
        base = batch_size * num_kv_heads * rt
        factor, _ = row_tuning(_capability(device), batch_size, rows, base)
        if union <= SHORT_UNION:
            # Short unions: one CTA per SM. A second resident CTA's extra splits
            # only add merge work (RTX PRO 6000, B=4 L=128: 10.3 -> 8.3 us).
            factor = 1
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        target = positive_env("CTI_DRAFT_TARGET_CTAS", factor * sms)
        if inline:
            # Preserve DF2's two-CTA target only when both shared memory and a
            # conservative register budget fit. No correctness residency bound.
            tile = positive_env(
                "CTI_DRAFT_KEY_TILE", KEY_TILE if head_dim == 256 else INLINE_KEY_TILE
            )
            groups = positive_env("CTI_DRAFT_KEY_GROUPS", key_groups(rows, _capability(device)))
            stages = positive_env("CTI_DRAFT_STAGES", STAGES)
            _, arena = merge_geometry(rows, tile, groups, stages, True, head_dim)
            threads = rows // 16 * 32 * groups
            if 2 * (arena + cap * 4 + 1024) > 100 * 1024 or 2 * 255 * threads > 65536:
                factor = 1
            target = positive_env("CTI_DRAFT_TARGET_CTAS", factor * sms)
        splits = min(limit, max(1, target // base))
        if (
            not inline
            and _capability(device) == (12, 1)
            and "CTI_DRAFT_TARGET_CTAS" not in os.environ
        ):
            tuned = gb10_splits(
                batch_size=batch_size,
                query_length=query_length,
                num_kv_heads=num_kv_heads,
                group=group,
                windowed=window is not None,
                union=union,
            )
            splits = tuned or splits
    # A split walks any number of keys in chunk_cap pieces, so capacity no longer
    # forces splits. Retain the measured legacy minimum at the tuning breakpoint.
    if not inline:
        floor = (min(union, MID_UNION) + cap - 1) // cap
        if (
            _capability(device) == (12, 1)
            and window is not None
            and not causal
            and query_length == 8
            and group == 4
            and head_dim == 128
            and "CTI_DRAFT_CHUNK_CAP" not in os.environ
        ):
            # The DF2 noncausal Q8 pair keeps one split at B4 on GB10.
            floor = 1
        splits = max(splits, floor)
    merge_grid = -(-batch_size * query_length * num_qo_heads // 4)
    if splits > 2**31 - 1 or (not inline and splits > 1 and merge_grid > 2**31 - 1):
        raise NotImplementedError("DRAFT CuTe grid-x exceeds 2**31-1")
    return group, rows, rt, splits, target, limit, inline


def workspace_size(**kwargs):
    # Every window at this capacity borrows the same maximum over the fixed tiers.
    sizes = [256]
    batch, heads = kwargs["batch_size"], kwargs["num_kv_heads"]
    for tier in capacity_tiers(kwargs["max_kv_tokens"], kwargs["window"] is not None):
        _, rows, rt, splits, target, limit, inline = _configuration(**kwargs, union=tier)
        # A spread grid row of S CTAs holds the B per-request rows of s CTAs, or more.
        spread, grid = spread_grid(batch, splits, heads * rt, target, limit, inline, False)
        slots = grid if spread else batch * splits
        size = (heads * rt * slots * rows * (kwargs.get("head_dim", 128) + 1) + batch) * 4
        if (grid if spread else splits) > (batch if spread else 1):
            sizes.append((size + 255) // 256 * 256)
    return max(sizes), 256


@lru_cache(maxsize=None)
def compile_kernels(
    query,
    group,
    rows,
    heads,
    causal,
    windowed,
    splits,
    dtype,
    page,
    arch,
    min_chunk=128,
    key_tile=KEY_TILE,
    key_groups=1,
    stages=STAGES,
    chunk_cap=CHUNK_CAP,
    pieces=False,
    sparse=False,
    short_union=0,
    inline_merge=False,
    distributed_merge=False,
    coop_pdl=True,
    early_trigger=False,
    head_dim=128,
    write_pages=True,
    iket=False,
    q_row_stride=None,
    table=False,
    spread=0,
    spread_budget=0,
):
    """Fake arguments keep compilation independent of any CUDA allocation or launch."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ctikernels._cute_prepare import Prepare

    from ._kernel import Attention, Merge

    element = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    fake = cute.runtime.make_fake_tensor

    def tensor(kind, alignment=4):
        return fake(kind, (cute.sym_int(),), (1,), assumed_align=alignment)

    i = [tensor(cutlass.Int32) for _ in range(7)]
    q = fake(
        element,
        (cute.sym_int(), heads * group, head_dim),
        (heads * group * head_dim if q_row_stride is None else q_row_stride, head_dim, 1),
        assumed_align=16,
    )
    out = fake(
        element,
        (cute.sym_int(), heads * group, head_dim),
        (heads * group * head_dim, head_dim, 1),
        assumed_align=16,
    )
    kv = fake(
        element,
        (cute.sym_int(), page, heads, head_dim),
        (page * heads * head_dim, heads * head_dim, head_dim, 1),
        assumed_align=16,
    )
    data = [q, kv, kv, out]
    work = tensor(cutlass.Float32)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--enable-tvm-ffi --gpu-arch {arch}"
    if iket:
        options += " iket"
    constants = (
        query,
        group,
        rows,
        heads,
        causal,
        windowed,
        splits,
        element,
        min_chunk,
        key_tile,
        key_groups,
        stages,
        chunk_cap,
    )
    prepare = (
        None
        if table
        else cute.compile(
            Prepare(query, page, write_pages=write_pages), *i, stream, options=options
        )
    )
    operation = Attention(
        *constants,
        page=page,
        pieces=pieces,
        short_union=short_union,
        inline_merge=inline_merge,
        distributed_merge=distributed_merge,
        coop_pdl=coop_pdl,
        head_dim=head_dim,
        write_pages=write_pages,
        iket=iket,
        q_row_stride=q_row_stride,
        table=table,
        early_trigger=early_trigger,
        spread=spread,
        spread_budget=spread_budget,
    )
    if table:
        arguments = (
            i[0],
            tensor(cutlass.Int64, 8),
            tensor(cutlass.Uint8, 1),
            i[2],
            *data,
            work,
            tensor(cutlass.Int32),
            cutlass.Int64(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
    else:
        arguments = (
            *i[:3],
            *data,
            work,
            tensor(cutlass.Int32),
            *i[3:],
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
    attention = cute.compile(
        operation.table_call if table else operation, *arguments, options=options
    )
    merge = None
    if splits > 1 and not inline_merge:
        merge = cute.compile(
            Merge(*constants, sparse=sparse, head_dim=head_dim, iket=iket),
            *i[:2],
            work,
            data[-1],
            stream,
            options=options,
        )
    return prepare, attention, merge


class Provider:
    """One attention launch with inline merge (legacy merge remains selectable).

    Owns a dedicated zeroed int32 counter tensor, separate from borrowed scratch.
    Each complete call resets arrivals; distributed generation flags advance.
    Calls on this instance must be ordered on one stream/graph; independent
    instances never share counters.
    """

    def __init__(
        self,
        float_workspace,
        int_workspace,
        *,
        prepare_inputs,
        query_indptr,
        kv_indptr,
        kv_indices,
        last_page_lengths,
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        window_left,
        query_row_stride=None,
        table_inputs=None,
        table_mode="request",
    ):
        self._table_inputs = table_inputs
        table = table_inputs is not None
        slots, ptr, idx, _, capacity = prepare_inputs
        window = None if window_left == -1 else window_left + 1
        # Larger windows are equivalent: every legal key distance is < C < 2**31.
        self._window = min(window or 0, 2**31 - 1)
        selected = tiling_union(capacity, query_length, window)
        for tier in capacity_tiers(capacity, window is not None):
            self._configure_tier(
                slots,
                capacity,
                query_length,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                dtype,
                causal,
                window is not None,
                tier,
                kv_indices is not None,
                query_row_stride,
                table_mode if table else False,
            )
            if tier == selected:
                state = self.__dict__.copy()
        self.__dict__.update(state)
        self._tier = selected
        rows, rt = self.rows, self.rt
        self._page_dummy = None
        if kv_indices is None:
            self._page_dummy = torch.empty(1, dtype=torch.int32, device=slots.device)
            kv_indices = self._page_dummy
        self._inputs = slots, ptr, idx
        self._prepare_inputs = (
            slots,
            ptr,
            idx,
            query_indptr,
            kv_indptr,
            kv_indices,
            last_page_lengths,
        )
        self._work = float_workspace.view(torch.float32)
        self.merge_fan_in, _ = merge_geometry(
            rows, self.key_tile, self.key_groups, self.stages, self.inline_merge, head_dim
        )
        self.merge_levels = (
            counter_levels(self.splits, self.merge_fan_in) if self.merge_mode == "tree" else ()
        )
        entries = (
            2 if self.merge_mode == "distributed" and self.splits > 1 else sum(self.merge_levels)
        )
        count = slots.numel() * num_kv_heads * rt * entries
        self._counters = torch.zeros(count, dtype=torch.int32, device=slots.device)
        self.counter_bytes = count * 4
        self._arrivals = self._counters[::2] if self.merge_mode == "distributed" else self._counters
        # Every launch argument except q/k/v/out is bound here, once.
        if table:
            done, stride, guard, maximum = table_inputs
            slots, frontier, indices = self._inputs
            self._leading = (slots, frontier, done, indices)
            tail = (self._work, self._counters, stride, guard, maximum, self._window)
            self._trailing = (tail, tail)
        else:
            self._leading = self._inputs
            self._trailing = tuple(
                (self._work, self._counters, *self._prepare_inputs[3:], fused, self._window)
                for fused in (0, 1)
            )

    def _configure_tier(
        self,
        slots,
        capacity,
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        windowed,
        union,
        write_pages,
        query_row_stride,
        table_mode=False,
    ):
        table = bool(table_mode)
        window = 1 if windowed else None
        group, rows, rt, self.splits, target, limit, inline = _configuration(
            batch_size=slots.numel(),
            query_length=query_length,
            max_kv_tokens=capacity,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            window=window,
            device=slots.device,
            head_dim=head_dim,
            causal=causal,
            union=union,
            table=table,
        )
        # Table attention merges inline; DW1's legacy rows apply to the CSR path.
        mode = (
            "distributed"
            if table
            else default_merge_mode(
                batch_size=slots.numel(),
                query_length=query_length,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                causal=causal,
                window=window,
                device=slots.device,
            )
        )
        self.rows, self.rt = rows, rt
        self._fallbacks = ()
        major, minor = torch.cuda.get_device_capability(slots.device)
        _, tile = row_tuning(
            (major, minor),
            slots.numel(),
            rows,
            slots.numel() * num_kv_heads * ((query_length * group + rows - 1) // rows),
        )
        # Recorded for benchmark reports.
        self.inline_merge, self.short_union = merge_tuning(
            short_default(slots.device, slots.numel(), head_dim), mode
        )
        self.merge_mode = requested_merge_mode(mode)
        if table and self.splits > 1 and not self.inline_merge:
            raise ValueError("table attention requires an inline merge")
        self.coop_pdl = cooperative_pdl()
        self.cooperative_blocks_per_sm = None
        self.min_chunk = positive_env("CTI_DRAFT_MIN_CHUNK", MIN_CHUNK)
        self.key_tile = positive_env(
            "CTI_DRAFT_KEY_TILE",
            KEY_TILE if head_dim == 256 else (INLINE_KEY_TILE if self.inline_merge else tile),
        )
        self.key_groups = positive_env("CTI_DRAFT_KEY_GROUPS", key_groups(rows, (major, minor)))
        self.stages = positive_env("CTI_DRAFT_STAGES", STAGES)
        # Size the shared index buffer for bounded split chunks and the short path,
        # rather than always allocating the cap: on GB10 the 16 KiB
        # cap buffer alone kept deeper rings and 64-key tiles at one CTA per SM.
        window_union = union
        # Actual union <= tier union: rounded chunks fit this bound, or pieces
        # walk them in cap-sized steps. chunk >= ceil(actual/splits), so count <= splits.
        need = -(-max(-(-window_union // self.splits), self.min_chunk) // self.key_tile)
        cap = positive_env("CTI_DRAFT_CHUNK_CAP", chunk_cap(slots.device))
        if (major, minor) == (12, 1) and union <= MID_UNION and not self.inline_merge:
            if "CTI_DRAFT_CHUNK_CAP" not in os.environ:
                # Let the middle tier fit in one piece, preserving the Q8/B4 kernel.
                cap = max(cap, -(-union // self.key_tile) * self.key_tile)
        if "CTI_DRAFT_SHORT" not in os.environ:
            self.short_union = min(self.short_union, cap)
        short_need = -(-self.short_union // self.key_tile) * self.key_tile
        if self.short_union > cap:
            raise ValueError("DRAFT CuTe requires SHORT <= CHUNK_CAP")
        self.chunk_cap = min(cap, max(need * self.key_tile, short_need))
        assert self.short_union <= self.chunk_cap
        # The piece loop is compiled only where a chunk can exceed one index buffer.
        self.pieces = need * self.key_tile > cap or os.environ.get("CTI_DRAFT_PIECES") == "1"
        # Spread grids keep the split budget (B * splits CTAs per row tile) and chunk bound.
        self.spread, grid = spread_grid(
            slots.numel(), self.splits, num_kv_heads * rt, target, limit, self.inline_merge, table
        )
        self.spread_budget = self.spread * self.splits
        self.splits = grid
        if self.spread > SPREAD_MAX_BATCH and "CTI_DRAFT_MERGE" not in os.environ:
            self.merge_mode = "tree"
        if self.merge_mode == "distributed" and not _cooperative.supported(slots.device):
            self._fallbacks += ("distributed -> tree: cooperative launch unsupported",)
            self.merge_mode = "tree"
        constants = (
            query_length,
            group,
            rows,
            num_kv_heads,
            causal,
            window is not None,
            self.splits,
            dtype,
            page_size,
            f"sm_{major}{minor}a",
            self.min_chunk,
            self.key_tile,
            self.key_groups,
            self.stages,
            self.chunk_cap,
            self.pieces,
            window_union > MID_UNION,
            self.short_union,
            self.inline_merge,
        )
        self._attention_no_pdl = None
        # Paired per-layer chains (RTX 5090, ctx 1000, B 1-4): 1-4% faster with the
        # dependent launch triggered right after the PDL wait; neutral at ctx 100.
        self.early_trigger = nonnegative_env("CTI_DRAFT_EARLY_TRIGGER", 1) == 1
        compile_options = {
            "head_dim": head_dim,
            "write_pages": write_pages,
            "early_trigger": self.early_trigger,
        }
        if os.environ.get("CTI_DRAFT_IKET") == "1":
            compile_options["iket"] = True
        if query_row_stride is not None:
            compile_options["q_row_stride"] = query_row_stride
        if table:
            compile_options["table"] = table_mode
        if self.spread:
            compile_options["spread"] = self.spread
            compile_options["spread_budget"] = self.spread_budget
        with _compiling:
            if self.merge_mode == "distributed":
                try:
                    kernels = compile_kernels(*constants, True, self.coop_pdl, **compile_options)
                except Exception as error:
                    if not _cooperative.launch_option_error(error):
                        raise
                    # Preserve cooperative semantics if CuTe rejects PDL with it.
                    try:
                        kernels = compile_kernels(*constants, True, False, **compile_options)
                        self.coop_pdl = False
                        self._fallbacks += ("PDL disabled: cooperative compile rejected PDL",)
                    except Exception as retry_error:
                        if not _cooperative.launch_option_error(retry_error):
                            raise
                        self._fallbacks += ("distributed -> tree: cooperative compile rejected",)
                        self.merge_mode = "tree"
                if self.merge_mode == "distributed":
                    _, arena = merge_geometry(
                        rows,
                        self.key_tile,
                        self.key_groups,
                        self.stages,
                        False,
                        head_dim,
                        distributed=True,
                    )
                    self.cooperative_blocks_per_sm = _cooperative.blocks_per_sm(
                        kernels[1],
                        slots.device,
                        rows // 16 * 32 * self.key_groups,
                        arena + self.chunk_cap * 4 + 1024,
                    )
                    sms = torch.cuda.get_device_properties(slots.device).multi_processor_count
                    grid = num_kv_heads * rt * self.splits * (1 if self.spread else slots.numel())
                    if grid > sms * self.cooperative_blocks_per_sm:
                        self._fallbacks += ("distributed -> tree: grid exceeds residency",)
                        self.merge_mode = "tree"
                    elif self.coop_pdl:
                        # Construction-only compilation permits an eager launch retry.
                        self._attention_no_pdl = compile_kernels(
                            *constants, True, False, **compile_options
                        )[1]
                        self.cooperative_blocks_per_sm = min(
                            self.cooperative_blocks_per_sm,
                            _cooperative.blocks_per_sm(
                                self._attention_no_pdl,
                                slots.device,
                                rows // 16 * 32 * self.key_groups,
                                arena + self.chunk_cap * 4 + 1024,
                            ),
                        )
                        if grid > sms * self.cooperative_blocks_per_sm:
                            self._fallbacks += (
                                "distributed -> tree: no-PDL grid exceeds residency",
                            )
                            self.merge_mode = "tree"
                            self._attention_no_pdl = None
            if self.merge_mode != "distributed":
                kernels = compile_kernels(*constants, **compile_options)
            self._prepare, self._attention, self._merge = kernels

    @property
    def tier(self):
        return self._tier

    @property
    def pdl(self):
        return self.merge_mode != "distributed" or self.coop_pdl

    @property
    def merge(self):
        return "none" if self.splits == 1 else self.merge_mode

    @property
    def fallbacks(self):
        return self._fallbacks

    def prepare(self):
        reject_grad(*self._prepare_inputs, self._work)
        self._prepare(*self._prepare_inputs)

    def _launch(self, q, paged_kv_cache, out, fused):
        # The caller (DraftAttention) has checked q/k/v/out; bound inputs are checked
        # at construction and by prepare().
        k, v = paged_kv_cache
        arguments = (*self._leading, q, k, v, out, *self._trailing[fused])
        try:
            self._attention(*arguments)
        except Exception as error:
            if self._attention_no_pdl is None or not _cooperative.launch_option_error(error):
                raise
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cooperative PDL rejected during capture; reconstruct with CTI_DRAFT_COOP_PDL=0"
                ) from error
            self._attention = self._attention_no_pdl
            self._attention_no_pdl = None
            self.coop_pdl = False
            self._fallbacks += ("PDL disabled: eager cooperative launch rejected PDL",)
            warnings.warn(
                "DRAFT CuTe attention PDL disabled after cooperative launch rejection; "
                "retrying the cooperative no-PDL kernel",
                RuntimeWarning,
                stacklevel=2,
            )
            self._attention(*arguments)
        if self._merge is not None:
            self._merge(*self._inputs[:2], self._work, out)
        return out

    def run(self, *, q, paged_kv_cache, out):
        return self._launch(q, paged_kv_cache, out, 0)

    def prepare_and_run(self, *, q, paged_kv_cache, out):
        # Attention never reads the CSR that prepare writes, so the attention CTAs
        # publish it after their PDL wait: one launch fewer than prepare(); run().
        return self._launch(q, paged_kv_cache, out, 1)


class TableProvider(Provider):
    """Direct table reads with the identical CSR split/tile schedule."""

    def __init__(
        self,
        float_workspace,
        int_workspace,
        *,
        request_indices,
        frontier_table,
        done_table,
        cache_index_table,
        guard_start,
        max_seq_len,
        metadata_index="request",
        **kwargs,
    ):
        flat = cache_index_table.as_strided(
            (
                (cache_index_table.shape[0] - 1) * cache_index_table.stride(0)
                + cache_index_table.shape[1],
            ),
            (1,),
        )
        done = done_table.view(torch.uint8)
        super().__init__(
            float_workspace,
            int_workspace,
            prepare_inputs=(
                request_indices,
                frontier_table,
                flat,
                done,
                max_seq_len + kwargs["query_length"],
            ),
            query_indptr=done,
            kv_indptr=done,
            kv_indices=done,
            last_page_lengths=done,
            table_inputs=(done, cache_index_table.stride(0), guard_start, max_seq_len),
            table_mode=metadata_index,
            **kwargs,
        )
