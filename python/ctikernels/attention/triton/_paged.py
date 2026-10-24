# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from vLLM 0.31.0 db9527a46873454610df6dbedf79a36d6bf1a7f6
# (`vllm/v1/attention/ops/triton_unified_attention.py`); see THIRD_PARTY_NOTICES.

"""Paged attention over a token CSR: one kernel for decode, verify, extend and DRAFT rows.

A program attends one query block of one sequence (``BLOCK_Q`` queries times the ``GROUP``
query heads of one KV head, packed into ``BLOCK_M`` rows) over one of ``SEGMENTS`` key
segments. With one segment it writes the output; otherwise it writes an FP32 partial that
``_reduce`` combines in segment order. Segment bounds depend only on the sequence's own
lengths and the launch's segment count. Launches borrow
their tensors, run on the current stream and read no device value on the host.
"""

import torch
import triton
import triton.language as tl

_MAX_SEGMENTS = 64


@triton.jit
def _sequence(Q_INDPTR, block, num_seqs, BLOCK_Q: tl.constexpr):
    """The sequence owning query block ``block``: sequence s owns blocks starting at
    ``q_indptr[s] // BLOCK_Q + s`` (a bound that needs no prefix scan)."""
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(Q_INDPTR + mid) // BLOCK_Q + mid
        if start <= block:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@triton.jit
def _attend(
    Q,
    K,
    V,
    OUT,
    PART_O,
    PART_ML,
    Q_INDPTR,
    KV_INDPTR,
    KV_INDICES,
    SLOTS,
    DONE,
    table_stride,
    guard,
    maximum,
    scale_log2,
    v_scale,
    window_left,
    window_right,
    num_seqs,
    num_q_heads,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_PAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEGMENTS: tl.constexpr,
    FIXED_Q: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
    TABLE: tl.constexpr,
    ROW_KEYED: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        # Wait for the producers of Q and the cache; let the reduction launch and wait.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    block = tl.program_id(0)
    kv_head = tl.program_id(1)
    segment = tl.program_id(2)

    if FIXED_Q > 0:
        blocks_per_seq: tl.constexpr = (FIXED_Q + BLOCK_Q - 1) // BLOCK_Q
        seq = block // blocks_per_seq
        q_block = block % blocks_per_seq
        q_start = seq * FIXED_Q
        q_len = FIXED_Q
    else:
        seq = _sequence(Q_INDPTR, block, num_seqs, BLOCK_Q)
        q_start = tl.load(Q_INDPTR + seq)
        q_len = tl.load(Q_INDPTR + seq + 1) - q_start
        q_block = block - (q_start // BLOCK_Q + seq)
        if q_block * BLOCK_Q >= q_len:
            return
    active = seq >= 0
    if HAS_SLOTS:
        # Inactive rows read no Q or key CSR and produce zeros.
        active = tl.load(SLOTS + seq) != -1
    if TABLE:
        # Row seq of request r: its frontier f (from KV_INDPTR) admits table[r, :f + Q],
        # capped at ``maximum`` and continued by its guard rows; otherwise only Q guard keys.
        request = tl.load(SLOTS + seq)
        keyed = seq if ROW_KEYED else request
        frontier = tl.load(KV_INDPTR + keyed)
        valid = (tl.load(DONE + keyed) == 0) & (frontier >= 0) & (frontier < maximum)
        kv_len = tl.where(valid, frontier + FIXED_Q, FIXED_Q).to(tl.int32)
        prefix = tl.where(valid, tl.minimum(kv_len, maximum), 0)
        kv_begin = request.to(tl.int64) * table_stride
    else:
        kv_begin = tl.load(KV_INDPTR + seq, mask=active, other=0).to(tl.int64)
        kv_len = (tl.load(KV_INDPTR + seq + 1, mask=active, other=0) - kv_begin).to(tl.int32)

    rows = tl.arange(0, BLOCK_M)
    query = q_block * BLOCK_Q + rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    row_ok = (rows < BLOCK_Q * GROUP) & (query < q_len)
    dims = tl.arange(0, D_PAD)
    dim_ok = dims < HEAD_DIM
    token = (q_start + query).to(tl.int64)

    # Bottom-right aligned query positions; keys run 0..kv_len-1. A key j is visible to
    # position p when p - j <= window_left (LEFT) and j - p <= window_right (RIGHT).
    position = kv_len - q_len + query
    first = kv_len - q_len + q_block * BLOCK_Q
    last = kv_len - q_len + tl.minimum(q_len, (q_block + 1) * BLOCK_Q) - 1
    key_lo = 0
    key_hi = kv_len
    if LEFT:
        key_lo = tl.maximum(first - window_left, 0)
    if RIGHT:
        key_hi = tl.minimum(key_hi, last + window_right + 1)
    key_hi = tl.maximum(key_hi, key_lo)
    tile_lo = key_lo // BLOCK_N
    tiles = tl.cdiv(key_hi, BLOCK_N) - tile_lo
    per_segment = tl.cdiv(tiles, SEGMENTS)
    seg_lo = tile_lo + segment * per_segment
    seg_hi = tl.minimum(seg_lo + per_segment, tile_lo + tiles)

    q = tl.load(
        Q + token[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :],
        mask=(row_ok & active)[:, None] & dim_ok[None, :],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D_PAD], tl.float32)
    k_base = K + kv_head * stride_kh
    v_base = V + kv_head * stride_vh
    for tile in range(seg_lo, seg_hi):
        keys = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        key_ok = (keys >= key_lo) & (keys < key_hi)
        if TABLE:
            listed = keys < prefix
            slot = tl.load(KV_INDICES + kv_begin + keys, mask=key_ok & listed, other=0)
            spill = tl.where(prefix != 0, tl.minimum(keys - maximum, FIXED_Q - 1), keys)
            slot = tl.where(listed, slot, guard + seq * FIXED_Q + spill).to(tl.int64)
        else:
            slot = tl.load(KV_INDICES + kv_begin + keys, mask=key_ok, other=0).to(tl.int64)
        k = tl.load(
            k_base + slot[None, :] * stride_kt + dims[:, None],
            mask=key_ok[None, :] & dim_ok[:, None],
            other=0.0,
        )
        # FP8 caches widen to the query type; their scales are folded into scale_log2
        # and the output.
        s = tl.dot(q, k.to(q.dtype)) * scale_log2
        allowed = row_ok[:, None] & key_ok[None, :]
        if LEFT:
            allowed &= position[:, None] - keys[None, :] <= window_left
        if RIGHT:
            allowed &= keys[None, :] - position[:, None] <= window_right
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            v_base + slot[:, None] * stride_vt + dims[None, :],
            mask=key_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
        m_i = m_new

    if SEGMENTS == 1:
        out = acc * (v_scale / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
        tl.store(
            OUT + token[:, None] * stride_ot + head[:, None] * stride_oh + dims[None, :],
            out.to(OUT.dtype.element_ty),
            mask=row_ok[:, None] & dim_ok[None, :],
        )
    else:
        part = (token * num_q_heads + head) * SEGMENTS + segment
        tl.store(PART_ML + part * 2, m_i, mask=row_ok)
        tl.store(PART_ML + part * 2 + 1, l_i, mask=row_ok)
        # An empty segment writes m = -inf and a zero partial, so the reduction loads
        # every operand at once.
        tl.store(
            PART_O + part[:, None] * D_PAD + dims[None, :],
            acc,
            mask=row_ok[:, None],
        )


@triton.jit
def _reduce(
    OUT,
    PART_O,
    PART_ML,
    v_scale,
    num_q_heads,
    stride_ot,
    stride_oh,
    HEAD_DIM: tl.constexpr,
    D_PAD: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SEG_PAD: tl.constexpr,
    D_CHUNK: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    segments = tl.arange(0, SEG_PAD)
    dims = tl.program_id(2) * D_CHUNK + tl.arange(0, D_CHUNK)
    part = (token * num_q_heads + head) * SEGMENTS + segments
    m = tl.load(PART_ML + part * 2, mask=segments < SEGMENTS, other=float("-inf"))
    l_seg = tl.load(PART_ML + part * 2 + 1, mask=segments < SEGMENTS, other=0.0)
    live = m > float("-inf")
    m_max = tl.max(m, 0)
    weight = tl.where(live, tl.exp2(m - tl.where(m_max == float("-inf"), 0.0, m_max)), 0.0)
    partial = tl.load(
        PART_O + part[:, None] * D_PAD + dims[None, :],
        mask=(segments < SEGMENTS)[:, None],
        other=0.0,
    )
    total = tl.sum(weight * l_seg, 0)
    out = tl.sum(partial * weight[:, None], 0) * (v_scale / tl.where(total == 0.0, 1.0, total))
    tl.store(
        OUT + token * stride_ot + head * stride_oh + dims,
        out.to(OUT.dtype.element_ty),
        mask=dims < HEAD_DIM,
    )


def device_traits(device):
    """``(SM count, PDL)`` of the device, PDL on sm_90+; raises NotImplementedError below
    compute capability 8.9."""
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) < (8, 9):
        raise NotImplementedError("Triton paged attention needs CUDA compute capability 8.9+")
    return properties.multi_processor_count, properties.major >= 9


def tiles(group, head_dim, fixed_q, kv_bytes=2):
    """``(BLOCK_Q, BLOCK_M, BLOCK_N, stages, warps)``: query rows times the GQA group pack
    into M.
    Decode streams 32-byte-per-channel key tiles unpipelined (16 BF16 or 32 FP8 keys; many
    short programs keep the loads in flight)."""
    block_n = 32 // kv_bytes if fixed_q == 1 else (64 if head_dim <= 128 else 32)
    if fixed_q:
        block_m = min(max(16, triton.next_power_of_2(fixed_q * group)), 64)
    else:
        block_m = 64 if head_dim <= 128 else 32
    block_m = max(block_m, triton.next_power_of_2(group))
    return max(1, block_m // group), block_m, block_n, 1 if fixed_q == 1 else 2, 4


def query_blocks(rows, num_seqs, block_q, fixed_q):
    """Query-block programs per KV head (an upper bound for variable query lengths)."""
    if fixed_q:
        return num_seqs * triton.cdiv(fixed_q, block_q)
    return rows // block_q + num_seqs


def split_count(*, sms, num_qo_heads, num_kv_heads, head_dim, num_seqs, rows, fixed_q):
    """Key segments per row: about 2.5 programs per SM over the launch's query blocks
    (measured best for 27B decode at B=1-4), at most 64."""
    block_q = tiles(num_qo_heads // num_kv_heads, head_dim, fixed_q)[0]
    programs = num_kv_heads * query_blocks(rows, num_seqs, block_q, fixed_q)
    return max(1, min(triton.cdiv(5 * sms, 2 * programs), _MAX_SEGMENTS))


def invariant_split_count(*, sms, num_qo_heads, num_kv_heads, head_dim, max_rows):
    """One sequence's split count, halved until ``max_rows`` rows of partials fit 64 MiB."""
    segments = split_count(
        sms=sms,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_seqs=1,
        rows=1,
        fixed_q=1,
    )
    while segments > 1 and max_rows * segments * partial_bytes(num_qo_heads, head_dim) > (64 << 20):
        segments //= 2
    return segments


def partial_bytes(num_qo_heads, head_dim):
    """FP32 partial bytes of one query row and one segment."""
    return num_qo_heads * (triton.next_power_of_2(head_dim) + 2) * 4


def attend(
    q,
    k,
    v,
    out,
    *,
    q_indptr,
    kv_indptr,
    kv_indices,
    num_seqs,
    fixed_q,
    segments,
    window_left,
    window_right,
    scale,
    k_scale=1.0,
    v_scale=1.0,
    slots=None,
    table=None,
    scratch=None,
    pdl=False,
):
    """Attend ``q`` [rows, Hq, D] over flat caches ``k``/``v`` [slots, Hkv, D] into ``out``.

    Sequence s owns query rows ``q_indptr[s]:q_indptr[s + 1]`` (rows ``s * fixed_q`` on when
    ``fixed_q`` > 0, q_indptr unread) and keys ``kv_indices[kv_indptr[s]:kv_indptr[s + 1]]``;
    queries are bottom-right aligned. Negative window bounds are unbounded. A sequence with
    no keys, or with ``slots[s] == -1``, writes zeros. With several segments the FP32
    partials use ``scratch`` (``rows * segments * partial_bytes`` bytes) or a new tensor.
    ``table`` = ``(frontier, done, stride, guard, maximum, row_keyed)`` replaces the key CSR
    for fixed-query rows of requests ``slots`` (see ``_attend``): kv_indptr is unread and
    kv_indices is the cache-index table.
    ``pdl`` launches with programmatic dependent launch (sm_90+).
    """
    rows, num_qo_heads, head_dim = q.shape
    if rows == 0:
        return out
    num_kv_heads = k.shape[1]
    group = num_qo_heads // num_kv_heads
    block_q, block_m, block_n, stages, warps = tiles(group, head_dim, fixed_q, k.element_size())
    d_pad = triton.next_power_of_2(head_dim)
    blocks = query_blocks(rows, num_seqs, block_q, fixed_q)
    if segments > 1:
        count = rows * num_qo_heads * segments
        if scratch is None:
            scratch = torch.empty(count * (d_pad + 2), dtype=torch.float32, device=q.device)
        else:
            scratch = scratch.view(torch.float32)
        part_ml = scratch[: count * 2]
        part_o = scratch[count * 2 : count * (d_pad + 2)]
    else:
        part_o = part_ml = out
    frontier, done, stride, guard, maximum, row_keyed = table or (None, None, 0, 0, 0, False)
    _attend[(blocks, num_kv_heads, segments)](
        q,
        k,
        v,
        out,
        part_o,
        part_ml,
        q_indptr,
        frontier if table else kv_indptr,
        kv_indices,
        q_indptr if slots is None else slots,
        q_indptr if table is None else done,
        stride,
        guard,
        maximum,
        scale * k_scale * 1.4426950408889634,
        v_scale,
        window_left,
        window_right,
        num_seqs,
        num_qo_heads,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP=group,
        HEAD_DIM=head_dim,
        D_PAD=d_pad,
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SEGMENTS=segments,
        FIXED_Q=fixed_q,
        LEFT=window_left >= 0,
        RIGHT=window_right >= 0,
        HAS_SLOTS=slots is not None and table is None,
        TABLE=table is not None,
        ROW_KEYED=row_keyed,
        PDL=pdl,
        num_warps=warps,
        num_stages=stages,
        launch_pdl=pdl,
    )
    if segments > 1:
        # Narrow reduction programs: one warp per 16 channels of one head.
        d_chunk = min(d_pad, 16)
        _reduce[(rows, num_qo_heads, d_pad // d_chunk)](
            out,
            part_o,
            part_ml,
            v_scale,
            num_qo_heads,
            out.stride(0),
            out.stride(1),
            HEAD_DIM=head_dim,
            D_PAD=d_pad,
            SEGMENTS=segments,
            SEG_PAD=triton.next_power_of_2(segments),
            D_CHUNK=d_chunk,
            PDL=pdl,
            num_warps=1,
            launch_pdl=pdl,
        )
    return out
