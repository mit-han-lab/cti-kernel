# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own Qwen3.5-MoE decode-block Triton tree (98e1e538); CuTe rewrite pending.
"""Persistent MoE queue: router, top-k, shared and routed SwiGLU, fixed-order FP32 combine.

Producers release counters after stores; consumers acquire before reading scratch.
Down tiles may preload weights before awaiting activations. State is private to the op.
The final CTA restores counters, plan lines and gamma sentinels for the next call;
each completing arrival restores its own tile counter. Static first-wave items
precede the atomic queue, and dependencies point to earlier items. The last arrival
combines slots 0..7 then shared, independent of scheduling. Sparse routed groups use
whole-row GEMVs; larger groups use MMA strips with evict-first weight-load hints.
"""

# Triton pointer annotations and the conventional intermediate-width symbol.
# ruff: noqa: E741, F722
# fmt: off


import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...triton.rounding import INTERPRET, bf16_round
from ...triton.topk_softmax import topk_softmax

# ctl slots, 1 KB apart so that polled flags, the queue head and the counters do not share
# L2 lines; per-expert gate/up completion counters are one 128-byte line apart
C_HEAD = tl.constexpr(0)
C_ROUTER = tl.constexpr(256)
C_TOPK = tl.constexpr(512)
C_PLAN = tl.constexpr(768)
C_SH1 = tl.constexpr(1024)
C_EXIT = tl.constexpr(1280)
C_PLANA = tl.constexpr(1536)
CTL_WORDS = tl.constexpr(2048)
G1D_STRIDE = tl.constexpr(32)
# one 128-byte line per touched expert: word 0 = MARK | local slot | token count << 16,
# words 1..16 = MARK | its first 16 pair indices
TL = tl.constexpr(32)
MARK = tl.constexpr(1 << 30)


@triton.jit
def _div_rn(a, b):
    # IEEE division without FTZ: Triton links libdevice with FTZ on, so libdevice.div_rn would
    # flush FP32 subnormals (BF16 gates, tiny sigmoids) where SGLang keeps them.
    if INTERPRET:
        return a / b
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _nap(SLEEP: tl.constexpr):
    if SLEEP > 0:
        # back off so hundreds of waiting CTAs do not crowd a counter's L2 slice
        tl.inline_asm_elementwise(f"nanosleep.u32 {SLEEP};\n\tmov.u32 $0, 0;", "=r",
                                  [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _poll(ptr):
    # one lane reads the counter (not every warp), the others get it through the reduction
    lane = tl.arange(0, 128)
    v = tl.load(ptr + lane * 0, mask=lane == 0, other=-2147483647, volatile=True)
    return tl.max(v, axis=0)


@triton.jit
def _fence_acquire():
    # after a relaxed read of a value written by a release: the acquire pattern (PTX memory
    # model), without another round trip; the caller's bar.sync extends it to the CTA
    if not INTERPRET:
        tl.inline_asm_elementwise("fence.acq_rel.gpu;\n\tmov.u32 $0, 0;", "=r", [],
                                  dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _wait_geq(ptr, target, SLEEP: tl.constexpr):
    # poll (one lane, backing off), then fence: acquire semantics
    v = _poll(ptr)
    while v < target:
        _nap(SLEEP)
        v = _poll(ptr)
    _fence_acquire()
    tl.debug_barrier()


@triton.jit
def _signal(ptr):
    tl.debug_barrier()
    tl.atomic_add(ptr, 1, sem="release", scope="gpu")


@triton.jit
def _publish(ctl_ptr, U):
    # the plan header, twice: C_PLAN = U + 1 at once (relaxed: the routed gate/up items only
    # read their marked lines after it), and C_PLANA = U + 1 released after every store of
    # the plan and of top-k, for the readers of top-k's other outputs
    lane = tl.arange(0, 128)
    tl.store(ctl_ptr + C_PLAN + lane * 0, U + 1 + lane * 0, mask=lane == 0)
    tl.debug_barrier()
    tl.atomic_xchg(ctl_ptr + C_PLANA, U + 1, sem="release", scope="gpu")


@triton.jit
def _plan_u(ctl_ptr, SLEEP: tl.constexpr):
    """The touched-expert count, once the plan header is published (relaxed polls: the
    routed gate/up items only need the marked words of their line)."""
    v = _poll(ctl_ptr + C_PLAN)
    while v == 0:
        _nap(SLEEP)
        v = _poll(ctl_ptr + C_PLAN)
    return v - 1


@triton.jit
def _grab(ctl_ptr):
    # LA: the next queue index, requested now (one lane's atomic) and read later by _grabbed,
    # so the round trip overlaps the item's work instead of following it
    lane = tl.arange(0, 128)
    return tl.atomic_add(ctl_ptr + C_HEAD + lane * 0, (lane == 0).to(tl.int32),
                         mask=lane == 0, sem="relaxed", scope="gpu")


@triton.jit
def _grabbed(g):
    lane = tl.arange(0, 128)
    return tl.max(tl.where(lane == 0, g, -2147483647), axis=0)


@triton.jit
def _line(touched_ptr, u, BM: tl.constexpr, SLEEP: tl.constexpr):
    """A touched expert's local slot, token count and first BM pair indices, polled until
    every word it needs carries MARK (the plan's stores need not land in order)."""
    i = tl.arange(0, BM)
    w0 = tl.load(touched_ptr + u * TL, volatile=True)
    first = tl.load(touched_ptr + u * TL + 1 + i, volatile=True)
    n = (w0 >> 16) & 0x3FFF
    todo = ((w0 & MARK) == 0) | (tl.max(((i < n) & ((first & MARK) == 0)).to(tl.int32), 0) > 0)
    while todo:
        _nap(SLEEP)
        w0 = tl.load(touched_ptr + u * TL, volatile=True)
        first = tl.load(touched_ptr + u * TL + 1 + i, volatile=True)
        n = (w0 >> 16) & 0x3FFF
        todo = ((w0 & MARK) == 0) | (tl.max(((i < n) & ((first & MARK) == 0)).to(tl.int32),
                                            0) > 0)
    return w0 & 0xFFFF, n, first & (MARK - 1)


@triton.jit
def _ldw(ptr, EVF: tl.constexpr):
    # weights are read once per call: with EVF their lines are the first L2 evicts, so the
    # kernel's code, x and workspace stay (plain loads only: Triton's pipelined cp.async
    # drops the hint)
    if EVF:
        v = tl.load(ptr, eviction_policy="evict_first")
    else:
        v = tl.load(ptr)
    return v


@triton.jit
def _ldwm(ptr, mask, EVF: tl.constexpr):
    if EVF:
        v = tl.load(ptr, mask=mask, other=0.0, eviction_policy="evict_first")
    else:
        v = tl.load(ptr, mask=mask, other=0.0)
    return v


@triton.jit
def _mm(a, b, acc, F32: tl.constexpr):
    # F32 is for the CPU interpreter only, whose dot does not handle BF16
    if F32:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _bf16(v):
    return v.to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _silu_mul(g, u, SHARED: tl.constexpr = False):
    if SHARED:  # vLLM's shared expert is its dense MLP: a BF16 gate/up GEMM output
        g, u = bf16_round(g), bf16_round(u)
    return _div_rn(g, 1.0 + libdevice.exp(-g)) * u


@triton.jit
def _router_item(r, x_ptr, rw_ptr, sgw_ptr, plog_ptr, valid, T,
                 H: tl.constexpr, E: tl.constexpr, NEB: tl.constexpr, KS: tl.constexpr,
                 BT_R: tl.constexpr, BE_R: tl.constexpr, F32: tl.constexpr,
                 EVF: tl.constexpr = False):
    """One FP32 partial product of BT_R rows x 16 router rows over one H / KS slice of the
    hidden axis, all loads issued at once (the router is on the critical path; they do not
    wait for the valid count). Block NEB is the shared-gate row (column E). Partials of the
    valid rows go to plog[ks, row, :] and are summed in the fixed order ks = 0..KS-1 by
    top-k."""
    BK: tl.constexpr = H // KS
    tb = r // ((NEB + 1) * KS)
    rem = r % ((NEB + 1) * KS)
    eb = rem // KS
    ks = rem % KS
    rows = tb * BT_R + tl.arange(0, BT_R)
    kk = ks * BK + tl.arange(0, BK)
    xa = tl.load(x_ptr + rows[:, None] * H + kk[None, :], mask=(rows < T)[:, None], other=0.0)
    cols = tl.arange(0, BE_R)
    if eb < NEB:
        wb = _ldw(rw_ptr + (eb * BE_R + cols)[None, :] * H + kk[:, None], EVF)
    else:
        wb = tl.load(sgw_ptr + kk[:, None] + 0 * cols[None, :], mask=(cols == 0)[None, :],
                     other=0.0)
    acc = _mm(xa, wb, tl.zeros((BT_R, BE_R), dtype=tl.float32), F32)
    smask = (rows < valid)[:, None] & ((eb < NEB) | (cols == 0))[None, :]
    tl.store(plog_ptr + (ks * T + rows)[:, None] * (E + BE_R) + (eb * BE_R + cols)[None, :],
             acc, mask=smask)


@triton.jit
def _logits(plog_ptr, rows, rmask, T, E: tl.constexpr, KS: tl.constexpr, BE_R: tl.constexpr):
    """The FP32 router products of rows and their shared-gate products, the KS partials summed
    in the fixed order 0..KS-1 (complete: the caller acquired the router counter)."""
    ids = tl.arange(0, E)
    base = rows[:, None] * (E + BE_R) + ids[None, :]
    gbase = rows * (E + BE_R) + E
    stride = T * (E + BE_R)
    z = tl.load(plog_ptr + base, mask=rmask[:, None], other=0.0, cache_modifier=".cg")
    s = tl.load(plog_ptr + gbase, mask=rmask, other=0.0, cache_modifier=".cg")
    for ks in tl.static_range(1, KS):
        z += tl.load(plog_ptr + ks * stride + base, mask=rmask[:, None], other=0.0,
                     cache_modifier=".cg")
        s += tl.load(plog_ptr + ks * stride + gbase, mask=rmask, other=0.0, cache_modifier=".cg")
    return z, s


@triton.jit
def _trace(tr_ptr, k, item, ph, t_f, t_r, TRACE: tl.constexpr):
    # development: one record [item, phase, fetched, ready, done] per item of this CTA
    if TRACE > 0:
        if k < TRACE:
            f = tl.arange(0, 8)
            t_d = tl.extra.cuda.globaltimer()
            z = tl.zeros((8,), dtype=tl.int64)
            v = tl.where(f == 0, z + item, tl.where(f == 1, z + ph, tl.where(
                f == 2, z + t_f, tl.where(f == 3, z + t_r, z + t_d))))
            tl.store(tr_ptr + (tl.program_id(0) * TRACE + k) * 5 + f, v, mask=f < 5)


@triton.jit
def _stamp(ev_ptr, slot, EV: tl.constexpr):
    if EV > 0:
        tl.store(ev_ptr + slot, tl.extra.cuda.globaltimer())


@triton.jit
def _select(plog_ptr, gam_ptr, mapv, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr, r0, valid,
            T, ps, pe, H: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
            KS: tl.constexpr, BE_R: tl.constexpr, SB: tl.constexpr, ev_ptr, EV: tl.constexpr,
            RS: tl.constexpr):
    """Top-k of SB rows: the record, each pair's local slot (ploc, -1 off the rank) and
    weight, each row's arrival count and shared gate, and zeros for rows nobody else writes.
    Returns the local slots [SB, TOPK] (-1 where none). Loops stay rolled: this code runs
    once per call on an SM, so its size (instruction fetches) costs more than its
    arithmetic. The caller acquired the router counter: every partial has landed."""
    rows = r0 + tl.arange(0, SB)
    rmask = rows < valid
    ks = tl.arange(0, TOPK)
    loc = tl.full((SB, TOPK), -1, dtype=tl.int32)
    need = tl.zeros((SB,), dtype=tl.int32)
    if r0 < valid:
        zf, s = _logits(plog_ptr, rows, rmask, T, E, KS, BE_R)
        _stamp(ev_ptr, 11, EV)
        # vLLM's shared gate: sigmoid of the BF16 logit, rounded to BF16 (torch's sigmoid)
        sb16 = bf16_round(s)
        tl.store(gam_ptr + rows, bf16_round(_div_rn(1.0, 1.0 + libdevice.exp(-sb16))),
                 mask=rmask)
        # contract logits: the FP32 dot products rounded once to BF16; vLLM's softmax top-k
        sel, wts = topk_softmax(bf16_round(zf), SB, E, TOPK)
        _stamp(ev_ptr, 12, EV)
        pmask = rmask[:, None] & (ks[None, :] >= 0)
        loc = tl.where(pmask, tl.reshape(tl.gather(mapv, tl.reshape(sel, (SB * TOPK,)), 0),
                                         (SB, TOPK)), -1)
        p = rows[:, None] * TOPK + ks[None, :]
        tl.store(rec_ptr + p, sel.to(tl.uint8), mask=pmask)
        tl.store(ploc_ptr + p, loc, mask=pmask)
        tl.store(pw_ptr + p, wts, mask=pmask)
        # arrivals each (token, hidden tile) waits for: its local slots and the shared term
        inpart = rmask & (rows >= ps) & (rows < pe)
        # (RS shared down tiles per combine tile)
        need = tl.sum((loc >= 0).to(tl.int32), axis=1) + inpart.to(tl.int32) * RS
        tl.store(need_ptr + rows, need, mask=rmask)
        _stamp(ev_ptr, 13, EV)
    # padded rows' weights read as zero (MoEBlock.routing_weights), as the composed route's
    pad = (rows < T) & (rows >= valid)
    tl.store(pw_ptr + rows[:, None] * TOPK + ks[None, :], tl.zeros((SB, TOPK), dtype=tl.float32),
             mask=pad[:, None] & (ks[None, :] >= 0))
    # rows nobody will write: masked rows and valid rows with nothing on this rank
    zero = (rows < T) & (need == 0)
    if tl.max(zero.to(tl.int32), axis=0) > 0:
        hz = tl.arange(0, 512)
        for h0 in range(0, H, 512):
            tl.store(out_ptr + rows[:, None] * H + h0 + hz[None, :],
                     tl.zeros((SB, 512), dtype=tl.bfloat16), mask=zero[:, None])
    return loc


@triton.jit
def _topk_item(kb, plog_ptr, gam_ptr, map_ptr, rec_ptr, cnt_ptr, lists_ptr, ploc_ptr, pw_ptr,
               need_ptr, out_ptr, valid, T, ps, pe,
               H: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, BT_K: tl.constexpr,
               KS: tl.constexpr, BE_R: tl.constexpr, SB: tl.constexpr, RS: tl.constexpr):
    """Top-k of BT_K rows in SB-row sub-blocks (few registers per round), and their pairs
    appended to per-expert lists through atomic counters."""
    mapv = tl.load(map_ptr + tl.arange(0, E))
    ks = tl.arange(0, TOPK)
    for sb in range(BT_K // SB):
        r0 = kb * BT_K + sb * SB
        rows = r0 + tl.arange(0, SB)
        loc = _select(plog_ptr, gam_ptr, mapv, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr, r0,
                      valid, T, ps, pe, H, E, TOPK, KS, BE_R, SB, plog_ptr, 0, RS)
        lmask = loc >= 0
        pos = tl.atomic_add(cnt_ptr + loc, 1, mask=lmask, sem="relaxed", scope="gpu")
        tl.store(lists_ptr + loc * T + pos, rows[:, None] * TOPK + ks[None, :], mask=lmask)


@triton.jit
def _plan(cnt_ptr, lists_ptr, touched_ptr, ctl_ptr, T, E_LOC: tl.constexpr, BM: tl.constexpr,
          ev_ptr, EV: tl.constexpr):
    """The compacted touched-expert list from the per-expert counts (the caller took the
    last top-k arrival with acquire semantics: counts and lists are complete); each touched
    expert's line also carries its first BM pair indices. Then the plan header."""
    tl.debug_barrier()
    e = tl.arange(0, E_LOC)
    c = tl.load(cnt_ptr + e, cache_modifier=".cg")
    nz = (c > 0).to(tl.int32)
    u = tl.cumsum(nz, axis=0) - nz
    i = tl.arange(0, BM)
    # loaded with the counts, not after them (entries past a count are never stored)
    first = tl.load(lists_ptr + e[:, None] * T + i[None, :], mask=i[None, :] < T, other=0,
                    cache_modifier=".cg")
    tl.store(touched_ptr + (u * TL + 1)[:, None] + i[None, :], first | MARK,
             mask=(nz > 0)[:, None] & (i[None, :] < c[:, None]))
    tl.store(touched_ptr + u * TL, e | (c << 16) | MARK, mask=nz > 0)
    _stamp(ev_ptr, 14, EV)
    _publish(ctl_ptr, tl.sum(nz, axis=0))
    _stamp(ev_ptr, 15, EV)


@triton.jit
def _route_one_block(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr,
                     touched_ptr, ctl_ptr, valid, T, ps, pe, H: tl.constexpr, E: tl.constexpr,
                     E_LOC: tl.constexpr, TOPK: tl.constexpr, KS: tl.constexpr,
                     BE_R: tl.constexpr, BT_K: tl.constexpr, SB: tl.constexpr, ev_ptr,
                     EV: tl.constexpr, RS: tl.constexpr):
    """Top-k and plan of a chunk of at most BT_K rows in one CTA (top-k in SB-row sub-blocks,
    a rolled loop), then the plan from the local slots it stored: each touched expert's
    line gets its count and all its pair indices (an expert has at most BT_K <= 16 pairs
    here), so there are no atomics or token lists."""
    P: tl.constexpr = BT_K * TOPK
    mapv = tl.load(map_ptr + tl.arange(0, E))
    _stamp(ev_ptr, 10, EV)
    for sb in range(BT_K // SB):
        _select(plog_ptr, gam_ptr, mapv, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr, sb * SB,
                valid, T, ps, pe, H, E, TOPK, KS, BE_R, SB, ev_ptr, EV, RS)
    # this CTA's own stores: visible to it after the barrier
    tl.debug_barrier()
    pi = tl.arange(0, P)  # pair index = row * TOPK + slot
    flat = tl.load(ploc_ptr + pi, mask=(pi // TOPK) < valid, other=-1)
    use = flat >= 0
    slot = tl.maximum(flat, 0)
    counts = tl.histogram(slot, E_LOC, mask=use)
    nz = (counts > 0).to(tl.int32)
    u_e = tl.cumsum(nz, axis=0) - nz
    e = tl.arange(0, E_LOC)
    tl.store(touched_ptr + u_e * TL, e | (counts << 16) | MARK, mask=nz > 0)
    # each pair's rank among the pairs of its expert, in pair order
    before = (flat[:, None] == flat[None, :]) & (pi[None, :] < pi[:, None])
    rank = tl.sum(before.to(tl.int32), axis=1)
    u_p = tl.gather(u_e, slot, 0)
    tl.store(touched_ptr + u_p * TL + 1 + rank, pi | MARK, mask=use)
    _stamp(ev_ptr, 14, EV)
    _publish(ctl_ptr, tl.sum(nz, axis=0))
    _stamp(ev_ptr, 15, EV)


@triton.jit
def _gate_up_tile(x_ptr, w_ptr, act_ptr, rows_tok, rows_act, mmask, j,
                  H: tl.constexpr, IW: tl.constexpr, IA: tl.constexpr,
                  BM: tl.constexpr, BI: tl.constexpr, BK1: tl.constexpr, F32: tl.constexpr,
                  SHARED: tl.constexpr = False):
    # columns interleave gate row i and up row i: column 2c -> gate j*BI+c, 2c+1 -> up
    cols = tl.arange(0, 2 * BI)
    wrow = (cols % 2) * IW + j * BI + cols // 2
    acc = tl.zeros((BM, 2 * BI), dtype=tl.float32)
    for k0 in range(0, H, BK1):
        kk = k0 + tl.arange(0, BK1)
        xa = tl.load(x_ptr + rows_tok[:, None] * H + kk[None, :], mask=mmask[:, None], other=0.0)
        wb = tl.load(w_ptr + wrow[None, :] * H + kk[:, None])
        acc = _mm(xa, wb, acc, F32)
    g, u = tl.split(tl.reshape(acc, (BM, BI, 2)))
    a = _silu_mul(g, u, SHARED)
    ci = j * BI + tl.arange(0, BI)
    tl.store(act_ptr + rows_act[:, None] * IA + ci[None, :], _bf16(a), mask=mmask[:, None])


@triton.jit
def _gv_pick(v, q: tl.constexpr, BM: tl.constexpr):
    # element q of a [BM] vector, as a scalar
    return tl.sum(tl.where(tl.arange(0, BM) == q, v, 0), 0)


@triton.jit
def _gv_out(w, xq, act_ptr, arow, ok, ci, IA: tl.constexpr, GR: tl.constexpr,
            SHARED: tl.constexpr):
    # one token's gate/up products of the step's GR row pairs (rows interleave gate and up)
    s = tl.sum(w.to(tl.float32) * xq.to(tl.float32)[None, :], axis=1)
    g, u = tl.split(tl.reshape(s, (GR, 2)))
    tl.store(act_ptr + arow * IA + ci, _bf16(_silu_mul(g, u, SHARED)), mask=(ci >= 0) & ok)


@triton.jit
def _gate_up_gemv(x_ptr, w_ptr, act_ptr, rows_tok, rows_act, mmask, j,
                  H: tl.constexpr, IW: tl.constexpr, IA: tl.constexpr, BM: tl.constexpr,
                  BI: tl.constexpr, GR: tl.constexpr, GT: tl.constexpr,
                  EVF: tl.constexpr = False, SHARED: tl.constexpr = False):
    """The gate/up tile j for at most GT rows (the first GT of rows_tok) on the CUDA cores:
    each step reads GR whole gate rows and GR whole up rows (GR x 4 KB contiguous each), the
    token rows stay in registers, FP32 products summed over the hidden axis. On a GPU whose
    DRAM favours long contiguous reads (GB10 LPDDR5X) this streams faster than the MMA tile's
    [2 BI x BK] strips; the MMA needs 16 rows, so at a few tokens it is no cheaper either."""
    k = tl.arange(0, H)
    mi = mmask.to(tl.int32)
    a0 = _gv_pick(rows_act, 0, BM)
    m0 = _gv_pick(mi, 0, BM) > 0
    x0 = tl.load(x_ptr + _gv_pick(rows_tok, 0, BM) * H + k, mask=(k < H) & m0, other=0.0)
    if GT > 1:
        a1 = _gv_pick(rows_act, 1, BM)
        m1 = _gv_pick(mi, 1, BM) > 0
        x1 = tl.load(x_ptr + _gv_pick(rows_tok, 1, BM) * H + k, mask=(k < H) & m1, other=0.0)
    if GT > 2:
        a2 = _gv_pick(rows_act, 2, BM)
        m2 = _gv_pick(mi, 2, BM) > 0
        x2 = tl.load(x_ptr + _gv_pick(rows_tok, 2, BM) * H + k, mask=(k < H) & m2, other=0.0)
        a3 = _gv_pick(rows_act, 3, BM)
        m3 = _gv_pick(mi, 3, BM) > 0
        x3 = tl.load(x_ptr + _gv_pick(rows_tok, 3, BM) * H + k, mask=(k < H) & m3, other=0.0)
    cols = tl.arange(0, 2 * GR)
    for r0 in range(0, BI, GR):
        wrow = (cols % 2) * IW + j * BI + r0 + cols // 2
        w = _ldw(w_ptr + wrow[:, None] * H + k[None, :], EVF)
        ci = j * BI + r0 + tl.arange(0, GR)
        _gv_out(w, x0, act_ptr, a0, m0, ci, IA, GR, SHARED)
        if GT > 1:
            _gv_out(w, x1, act_ptr, a1, m1, ci, IA, GR, SHARED)
        if GT > 2:
            _gv_out(w, x2, act_ptr, a2, m2, ci, IA, GR, SHARED)
            _gv_out(w, x3, act_ptr, a3, m3, ci, IA, GR, SHARED)


@triton.jit
def _gate_up_gemv_loop(x_ptr, w_ptr, act_ptr, line_ptr, n, j, H: tl.constexpr,
                       IW: tl.constexpr, IA: tl.constexpr, BI: tl.constexpr, GR: tl.constexpr,
                       TOPK: tl.constexpr, EVF: tl.constexpr):
    """The gate/up tile j of an expert with n pairs (all in its plan line at line_ptr, which
    _line polled complete) on the CUDA cores, with a run-time loop over the pairs in groups of
    4 (GVL): each step reads GR whole gate rows and GR whole up rows (as _gate_up_gemv), then
    for each group loads its pair indices (L2) and token rows (L1/L2) together, so no token row
    is held in registers across steps and the work follows the actual pair count. Per pair the
    arithmetic is _gate_up_gemv's (_gv_out)."""
    k = tl.arange(0, H)
    cols = tl.arange(0, 2 * GR)
    for r0 in range(0, BI, GR):
        wrow = (cols % 2) * IW + j * BI + r0 + cols // 2
        w = _ldw(w_ptr + wrow[:, None] * H + k[None, :], EVF)
        ci = j * BI + r0 + tl.arange(0, GR)
        for t0 in range(0, n, 4):
            ok1 = t0 + 1 < n
            ok2 = t0 + 2 < n
            ok3 = t0 + 3 < n
            p0 = tl.load(line_ptr + 1 + t0, cache_modifier=".cg") & (MARK - 1)
            p1 = tl.load(line_ptr + 2 + t0, mask=ok1, other=0, cache_modifier=".cg") & (MARK - 1)
            p2 = tl.load(line_ptr + 3 + t0, mask=ok2, other=0, cache_modifier=".cg") & (MARK - 1)
            p3 = tl.load(line_ptr + 4 + t0, mask=ok3, other=0, cache_modifier=".cg") & (MARK - 1)
            x0 = tl.load(x_ptr + (p0 // TOPK) * H + k)
            x1 = tl.load(x_ptr + (p1 // TOPK) * H + k, mask=(k < H) & ok1, other=0.0)
            x2 = tl.load(x_ptr + (p2 // TOPK) * H + k, mask=(k < H) & ok2, other=0.0)
            x3 = tl.load(x_ptr + (p3 // TOPK) * H + k, mask=(k < H) & ok3, other=0.0)
            _gv_out(w, x0, act_ptr, p0, True, ci, IA, GR, False)
            _gv_out(w, x1, act_ptr, p1, ok1, ci, IA, GR, False)
            _gv_out(w, x2, act_ptr, p2, ok2, ci, IA, GR, False)
            _gv_out(w, x3, act_ptr, p3, ok3, ci, IA, GR, False)


@triton.jit
def _down_tile(act_ptr, w_ptr, rows_act, mmask, h, IW: tl.constexpr, IA: tl.constexpr,
               BM: tl.constexpr, BH: tl.constexpr, BK2: tl.constexpr, F32: tl.constexpr):
    hh = h * BH + tl.arange(0, BH)
    acc = tl.zeros((BM, BH), dtype=tl.float32)
    for k0 in range(0, IW, BK2):
        kk = k0 + tl.arange(0, BK2)
        a = tl.load(act_ptr + rows_act[:, None] * IA + kk[None, :], mask=mmask[:, None],
                    other=0.0, cache_modifier=".cg")
        wb = tl.load(w_ptr + hh[None, :] * IW + kk[:, None])
        acc = _mm(a, wb, acc, F32)
    return acc, hh


@triton.jit
def _arrive(tok, mmask, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe,
            shared_rows, H: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr,
            BM: tl.constexpr, BH: tl.constexpr):
    """Count this tile's arrival for each row's token; the arrival that completes a token's
    tile writes out[token, tile] as vLLM's finalize and block: acc = fma(w_k, y_k, acc) over
    the unweighted BF16 slots 0..7 in order, rounded once, then + the gated shared row in
    BF16."""
    need = tl.load(need_ptr + tok, mask=mmask, other=0, cache_modifier=".cg")
    kk = tl.arange(0, TOPK)
    locs = tl.load(ploc_ptr + tok[:, None] * TOPK + kk[None, :], mask=mmask[:, None], other=-1,
                   cache_modifier=".cg")
    tl.debug_barrier()
    old = tl.atomic_add(red_ptr + tok * NT2 + h, 1, mask=mmask, sem="acq_rel", scope="gpu")
    fin = mmask & (old + 1 == need)
    if tl.max(fin.to(tl.int32), axis=0) > 0:
        tl.debug_barrier()
        hh = h * BH + tl.arange(0, BH)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            pk = tok * TOPK + k
            loc = tl.sum(tl.where(kk[None, :] == k, locs, 0), axis=1)
            use = fin & (loc >= 0)
            yk = tl.load(y_ptr + pk[:, None] * H + hh[None, :], mask=use[:, None], other=0.0,
                         cache_modifier=".cg")
            wk = tl.load(pw_ptr + pk, mask=fin, other=0.0, cache_modifier=".cg")
            acc = tl.fma(wk[:, None], yk, acc)
        inpart = fin & (tok >= ps) & (tok < pe)
        ys = tl.load(y_ptr + (shared_rows + tok)[:, None] * H + hh[None, :],
                     mask=inpart[:, None], other=0.0, cache_modifier=".cg")
        out = _bf16(bf16_round(acc) + ys)
        tl.store(out_ptr + tok[:, None] * H + hh[None, :], out, mask=fin[:, None])
        # no other arrival comes for this tile in this call: reset it for the next one
        tl.store(red_ptr + tok * NT2 + h, tl.zeros((BM,), dtype=tl.int32), mask=fin)


@triton.jit
def _down_pre(act_ptr, wpre, pw_ptr, y_ptr, tok, p, mm, h, red_ptr, need_ptr, ploc_ptr,
                out_ptr, ps, pe, shared_rows, H: tl.constexpr, I: tl.constexpr,
                IA: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr, BM: tl.constexpr,
                BH: tl.constexpr, F32: tl.constexpr):
    """A routed down tile from its preloaded weights wpre [I, BH]: y = BF16(act @ W^T),
    unweighted (one K = I product: the same chained MMAs as the K-looped tile), then the
    arrival."""
    kk = tl.arange(0, I)
    a = tl.load(act_ptr + p[:, None] * IA + kk[None, :], mask=mm[:, None], other=0.0,
                cache_modifier=".cg")
    acc = _mm(a, wpre, tl.zeros((BM, BH), dtype=tl.float32), F32)
    hh = h * BH + tl.arange(0, BH)
    tl.store(y_ptr + p[:, None] * H + hh[None, :], bf16_round(acc), mask=mm[:, None])
    _arrive(tok, mm, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe, shared_rows,
            H, TOPK, NT2, BM, BH)


@triton.jit
def _down_arrive(act_ptr, wbase, pw_ptr, y_ptr, tok, p, mm, h, red_ptr, need_ptr, ploc_ptr,
                 out_ptr, ps, pe, shared_rows, H: tl.constexpr, I: tl.constexpr,
                 IA: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr, BM: tl.constexpr,
                 BH: tl.constexpr, BK2: tl.constexpr, F32: tl.constexpr):
    """One routed down tile for the pairs p (tokens tok): y = BF16(act @ W_down^T),
    unweighted, then the tile's arrival."""
    acc, hh = _down_tile(act_ptr, wbase, p, mm, h, I, IA, BM, BH, BK2, F32)
    tl.store(y_ptr + p[:, None] * H + hh[None, :], bf16_round(acc), mask=mm[:, None])
    _arrive(tok, mm, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe, shared_rows,
            H, TOPK, NT2, BM, BH)


@triton.jit
def moe_block_kernel(
    # typed pointers, given as addresses (the host wrapper: every tensor goes in as an int)
    x_ptr: "*bf16", rw_ptr: "*bf16", sgw_ptr: "*bf16", wgu_ptr: "*bf16", wd_ptr: "*bf16",
    sgu_ptr: "*bf16", sd_ptr: "*bf16", part_ptr: "*i32", map_ptr: "*i32", valid_ptr: "*i32",
    out_ptr: "*bf16", rec_ptr: "*u8",
    ctl_ptr: "*i32", cnt_ptr: "*i32", g1d_ptr: "*i32", touched_ptr: "*i32", lists_ptr: "*i32",
    plog_ptr: "*fp32", gam_ptr: "*fp32", ploc_ptr: "*i32", pw_ptr: "*fp32", need_ptr: "*i32",
    red_ptr: "*i32", act_ptr: "*bf16", y_ptr: "*fp32", ev_ptr: "*i64",
    T, row0, T_ALL, NUM_CTAS,
    H: tl.constexpr, E: tl.constexpr, E_LOC: tl.constexpr, TOPK: tl.constexpr,
    I: tl.constexpr, SI: tl.constexpr, IA: tl.constexpr,
    BT_R: tl.constexpr, BE_R: tl.constexpr, KS: tl.constexpr, BT_K: tl.constexpr,
    BM: tl.constexpr, BI: tl.constexpr, BK1: tl.constexpr, BH: tl.constexpr,
    BK2: tl.constexpr, BMS: tl.constexpr,
    PRE2: tl.constexpr = False, SPRE: tl.constexpr = False, BHS: tl.constexpr = 0,
    F32: tl.constexpr = False, SLEEP: tl.constexpr = 100, EV: tl.constexpr = 0,
    EV_RING: tl.constexpr = 16, TRACE: tl.constexpr = 0,
    GV: tl.constexpr = 0, GR: tl.constexpr = 2, EVF: tl.constexpr = False,
    BMG: tl.constexpr = 0, BMSG: tl.constexpr = 0, SGV: tl.constexpr = -1,
    GVD: tl.constexpr = 0, GVS: tl.constexpr = False,
    SBK: tl.constexpr = 0, LA: tl.constexpr = 0, GVL: tl.constexpr = 0,
):
    if (EV > 0) | (TRACE > 0):
        t_start = tl.extra.cuda.globaltimer()
    NEB: tl.constexpr = E // BE_R
    NR = (NEB + 1) * KS * ((T + BT_R - 1) // BT_R)
    v_raw = tl.load(valid_ptr)
    item = tl.program_id(0)
    if item < NR:
        if LA == 1:
            g0 = _grab(ctl_ptr)
        # the first wave's router items run before anything waits on a load: their weight
        # loads are the head of the routing chain
        vr = tl.minimum(tl.maximum(tl.minimum(tl.maximum(v_raw, 0), T_ALL) - row0, 0), T)
        _router_item(item, x_ptr, rw_ptr, sgw_ptr, plog_ptr, vr, T,
                     H, E, NEB, KS, BT_R, BE_R, F32, EVF)
        _signal(ctl_ptr + C_ROUTER)
        if TRACE > 0:
            _trace(ev_ptr, 0, item, 0, t_start, t_start, TRACE)
        if EV > 0:
            tl.atomic_max(ev_ptr + 9 + (tl.load(ev_ptr + EV_RING * 32, volatile=True)
                                        % EV_RING) * 32,
                          tl.extra.cuda.globaltimer(), sem="relaxed")
        if LA == 1:
            item = NUM_CTAS + _grabbed(g0)
        else:
            item = NUM_CTAS + tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
    # rows of this chunk that are valid, and the chunk's rows of the rank's shared part
    valid_all = tl.minimum(tl.maximum(v_raw, 0), T_ALL)
    valid = tl.minimum(tl.maximum(valid_all - row0, 0), T)
    pidx = tl.load(part_ptr)
    pcnt = tl.maximum(tl.load(part_ptr + 1), 1)
    q = valid_all // pcnt
    m = valid_all - q * pcnt
    pstart = pidx * q + tl.minimum(pidx, m)
    pend = pstart + q + (pidx < m).to(tl.int32)
    ps = tl.minimum(tl.maximum(pstart - row0, 0), valid)
    pe = tl.minimum(tl.maximum(pend - row0, 0), valid)

    NT1: tl.constexpr = I // BI
    NT2: tl.constexpr = H // BH
    # rows per MMA block of the shared gate/up tiles
    BMSG_: tl.constexpr = BMS if BMSG == 0 else BMSG
    # the shared gate/up tiles as GEMVs too (SGV < 0: as the routed ones)
    SGV_: tl.constexpr = GV if SGV < 0 else SGV
    NS1: tl.constexpr = SI // BI
    # shared down tiles: BHS_ rows, RS of them per combine tile of BH rows
    BHS_: tl.constexpr = BH if BHS == 0 else BHS
    RS: tl.constexpr = BH // BHS_
    NS2: tl.constexpr = H // BHS_
    NK = (T + BT_K - 1) // BT_K
    # rows per top-k round of a single top-k block (SBK: fewer than its BT_K rows)
    SBK_: tl.constexpr = BT_K if SBK == 0 else SBK
    s_topk = NR
    s_sh1 = s_topk + NK
    s_sh2 = s_sh1 + NS1
    s_dyn = s_sh2 + NS2
    shared_rows = T * TOPK  # act/y rows of the shared expert start here

    U = -1  # touched experts, once this CTA has seen the plan header
    acq = 0  # whether this CTA has acquired the plan header (top-k's data visible)
    if EV > 0:
        # development events (int64 globaltimer): 0 first CTA start, 2 last top-k done,
        # 3 first routed g1 ready, 4 last sh-g1 done, 5 last g1 done, 6 first g2 ready,
        # 7 last CTA end, 8 last sh-g2 done, 9 last router item done, 10-15 top-k steps.
        # A ring of EV_RING slots of 32 words, one per call (the call count after them).
        ev_seq = tl.load(ev_ptr + EV_RING * 32, volatile=True)
        ev_ptr = ev_ptr + (ev_seq % EV_RING) * 32
        tl.atomic_min(ev_ptr + 0, t_start, sem="relaxed")
    running = 1
    tk = (tl.program_id(0) < NR).to(tl.int32)  # TRACE: records written by this CTA
    while running != 0:
        # an item's completion counter is released after the next item is grabbed, so the
        # release's wait for this item's stores overlaps the grab. LA 1: the next item is
        # requested as this one starts (a CTA then holds its current item and its next one;
        # the lowest unfinished item is still always some CTA's current item, since a held
        # next item has a higher index than its CTA's current one, so progress holds)
        if LA == 1:
            gnx = _grab(ctl_ptr)
        sig = 0
        sig_ptr = ctl_ptr + C_EXIT
        ph = -1
        t_f = 0
        t_r = 0
        if TRACE > 0:
            t_f = tl.extra.cuda.globaltimer()
            t_r = t_f
        if item < s_topk:
            ph = 0
            _router_item(item, x_ptr, rw_ptr, sgw_ptr, plog_ptr, valid, T,
                         H, E, NEB, KS, BT_R, BE_R, F32, EVF)
            # top-k waits on this: released now, not after the next grab
            _signal(ctl_ptr + C_ROUTER)
            if EV > 0:
                tl.atomic_max(ev_ptr + 9, tl.extra.cuda.globaltimer(), sem="relaxed")
        elif item < s_sh1:
            _wait_geq(ctl_ptr + C_ROUTER, NR, SLEEP)
            ph = 2
            if TRACE > 0:
                t_r = tl.extra.cuda.globaltimer()
            if NK == 1:
                _route_one_block(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr,
                                 need_ptr, out_ptr, touched_ptr, ctl_ptr, valid, T, ps, pe,
                                 H, E, E_LOC, TOPK, KS, BE_R, BT_K, SBK_, ev_ptr, EV, RS)
            else:
                if EV > 0:
                    # multi-block steps: 10 first block past its wait, 11 last block's top-k and
                    # list appends done, 12 last arrival, 14 plan lines stored, 15 published
                    tl.atomic_min(ev_ptr + 10, tl.extra.cuda.globaltimer(), sem="relaxed")
                _topk_item(item - s_topk, plog_ptr, gam_ptr, map_ptr, rec_ptr, cnt_ptr, lists_ptr,
                           ploc_ptr, pw_ptr, need_ptr, out_ptr, valid, T, ps, pe, H, E, TOPK,
                           BT_K, KS, BE_R, BT_K, RS)
                if EV > 0:
                    tl.atomic_max(ev_ptr + 11, tl.extra.cuda.globaltimer(), sem="relaxed")
                tl.debug_barrier()
                old = tl.atomic_add(ctl_ptr + C_TOPK, 1, sem="acq_rel", scope="gpu")
                # the last top-k block to finish runs the plan
                if old == NK - 1:
                    if EV > 0:
                        tl.store(ev_ptr + 12, tl.extra.cuda.globaltimer())
                    _plan(cnt_ptr, lists_ptr, touched_ptr, ctl_ptr, T, E_LOC, BM, ev_ptr, EV)
            if EV > 0:
                tl.atomic_max(ev_ptr + 2, tl.extra.cuda.globaltimer(), sem="relaxed")
        elif item < s_sh2:
            # shared gate/up tile j over the part rows
            ph = 1
            j = item - s_sh1
            if SGV_ > 0:
                # (the rank's part has at most GV rows here)
                rr = ps + tl.arange(0, BMS)
                _gate_up_gemv(x_ptr, sgu_ptr, act_ptr, rr, shared_rows + rr, rr < pe, j,
                              H, SI, IA, BMS, BI, GR, GV, EVF, SHARED=True)
            else:
                for m0 in range(ps, pe, BMSG_):
                    rr = m0 + tl.arange(0, BMSG_)
                    mm = rr < pe
                    _gate_up_tile(x_ptr, sgu_ptr, act_ptr, rr, shared_rows + rr, mm, j,
                                  H, SI, IA, BMSG_, BI, BK1, F32, SHARED=True)
            if EV > 0:
                tl.atomic_max(ev_ptr + 4, tl.extra.cuda.globaltimer(), sem="relaxed")
            sig = 1
            sig_ptr = ctl_ptr + C_SH1
        elif item < s_dyn:
            # shared down tile h (BHS_ hidden rows), gated by sigmoid(s)
            h = item - s_sh2
            if PRE2 | SPRE:
                # the shared down tile streams during the routing chain, not after the gate/up
                # tiles
                ks2 = tl.arange(0, SI)
                hs = h * BHS_ + tl.arange(0, BHS_)
                wspre = _ldwm(sd_ptr + hs[None, :] * SI + ks2[:, None], pe > ps, EVF)
            _wait_geq(ctl_ptr + C_SH1, NS1, SLEEP)
            ph = 3
            if TRACE > 0:
                t_r = tl.extra.cuda.globaltimer()
            for m0 in range(ps, pe, BMS):
                rr = m0 + tl.arange(0, BMS)
                mm = rr < pe
                if PRE2 | SPRE:
                    a = tl.load(act_ptr + (shared_rows + rr)[:, None] * IA + ks2[None, :],
                                mask=mm[:, None], other=0.0, cache_modifier=".cg")
                    acc = _mm(a, wspre, tl.zeros((BMS, BHS_), dtype=tl.float32), F32)
                    hh = hs
                else:
                    acc, hh = _down_tile(act_ptr, sd_ptr, shared_rows + rr, mm, h, SI, IA,
                                         BMS, BH, BK2, F32)
                # gamma = BF16(sigmoid(BF16(w_sg . x))), stored by top-k (the sentinel until then);
                # the row is BF16(gamma * BF16(shared down)), vLLM's gated shared expert
                gam = tl.load(gam_ptr + rr, mask=mm, other=0.0, volatile=True)
                while tl.max(((gam.to(tl.int32, bitcast=True) == -1) & mm).to(tl.int32),
                             0) > 0:
                    _nap(SLEEP)
                    gam = tl.load(gam_ptr + rr, mask=mm, other=0.0, volatile=True)
                tl.store(y_ptr + (shared_rows + rr)[:, None] * H + hh[None, :],
                         bf16_round(gam[:, None] * bf16_round(acc)),
                         mask=mm[:, None])
            if pe > ps:
                # the arrival counts come from top-k, visible once the plan is acquired
                if U < 0:
                    U = _plan_u(ctl_ptr, SLEEP)
                if acq == 0:
                    _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                    acq = 1
                for m0 in range(ps, pe, BMS):
                    rr = m0 + tl.arange(0, BMS)
                    _arrive(rr, rr < pe, h // RS, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps,
                            pe, shared_rows, H, TOPK, NT2, BMS, BH)
            if EV > 0:
                tl.atomic_max(ev_ptr + 8, tl.extra.cuda.globaltimer(), sem="relaxed")
        else:
            if U < 0:
                U = _plan_u(ctl_ptr, SLEEP)
            d = item - s_dyn
            if d < U * NT1:
                # routed gate/up tile j of touched expert u
                u = d // NT1
                j = d % NT1
                e, n, first = _line(touched_ptr, u, BM, SLEEP)
                ph = 4
                if TRACE > 0:
                    t_r = tl.extra.cuda.globaltimer()
                if EV > 0:
                    tl.atomic_min(ev_ptr + 3, tl.extra.cuda.globaltimer(), sem="relaxed")
                wbase = wgu_ptr + e.to(tl.int64) * (2 * I * H)
                if n > BM:
                    # the rest of the token list was written by other top-k CTAs
                    if acq == 0:
                        _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                        acq = 1
                if GV > 0:
                    # (an expert has at most GV pairs here: one per token)
                    _gate_up_gemv(x_ptr, wbase, act_ptr, first // TOPK, first,
                                  tl.arange(0, BM) < n, j, H, I, IA, BM, BI, GR, GV, EVF)
                elif BMG > BM:
                    if NK > 1:
                        # BMG pairs per MMA block (many pairs per expert: each block
                        # re-reads the item's weights from L2), all from the token lists
                        if acq == 0:
                            _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                            acq = 1
                        for m0 in range(0, n, BMG):
                            mi = m0 + tl.arange(0, BMG)
                            mm = mi < n
                            p = tl.load(lists_ptr + e * T + mi, mask=mm, other=0,
                                        cache_modifier=".cg")
                            _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j,
                                          H, I, IA, BMG, BI, BK1, F32)
                    else:
                        # one top-k block wrote no token lists: its at most BT_K pairs are
                        # all in the plan line
                        _gate_up_tile(x_ptr, wbase, act_ptr, first // TOPK, first,
                                      tl.arange(0, BM) < n, j, H, I, IA, BM, BI, BK1, F32)
                else:
                    # GVD: an expert with at most GVD pairs (its whole list is in the plan
                    # line) takes the GEMV over whole rows, a larger one the MMA strips
                    gv_ok = 0
                    if GVD > 0:
                        gv_ok = (n <= GVD).to(tl.int32)
                    if gv_ok != 0:
                        pm = tl.arange(0, BM) < n
                        if GVS:
                            # the GEMV as wide as the pair count (1, 2, then GVD token rows)
                            if n <= 1:
                                _gate_up_gemv(x_ptr, wbase, act_ptr, first // TOPK, first, pm,
                                              j, H, I, IA, BM, BI, GR, 1, EVF)
                            elif n <= 2:
                                _gate_up_gemv(x_ptr, wbase, act_ptr, first // TOPK, first, pm,
                                              j, H, I, IA, BM, BI, GR, 2, EVF)
                            else:
                                _gate_up_gemv(x_ptr, wbase, act_ptr, first // TOPK, first, pm,
                                              j, H, I, IA, BM, BI, GR, GVD, EVF)
                        else:
                            _gate_up_gemv(x_ptr, wbase, act_ptr, first // TOPK, first, pm,
                                          j, H, I, IA, BM, BI, GR, GVD, EVF)
                    else:
                        # GVL: an expert with more than GVD pairs and at most GVL takes the
                        # run-time pair-loop GEMV, a larger one the MMA strips
                        gvl = 0
                        if GVL > 0:
                            tl.static_assert(GVL <= BM)
                            gvl = (n <= GVL).to(tl.int32)
                        if gvl != 0:
                            _gate_up_gemv_loop(x_ptr, wbase, act_ptr, touched_ptr + u * TL, n,
                                               j, H, I, IA, BI, GR, TOPK, EVF)
                        else:
                            for m0 in range(0, n, BM):
                                mi = m0 + tl.arange(0, BM)
                                mm = mi < n
                                if m0 == 0:
                                    p = first
                                else:
                                    p = tl.load(lists_ptr + e * T + mi, mask=mm, other=0,
                                                cache_modifier=".cg")
                                _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j,
                                              H, I, IA, BM, BI, BK1, F32)
                if EV > 0:
                    tl.atomic_max(ev_ptr + 5, tl.extra.cuda.globaltimer(), sem="relaxed")
                sig = 1
                sig_ptr = g1d_ptr + u * G1D_STRIDE
            elif d < U * (NT1 + NT2):
                # routed down tile h of touched expert u, once its gate/up tiles are done
                d2 = d - U * NT1
                u = d2 // NT2
                h = d2 % NT2
                e, n, first = _line(touched_ptr, u, BM, SLEEP)
                wbase = wd_ptr + e.to(tl.int64) * (H * I)
                if PRE2:
                    # the down tile's weights are issued before the wait on the expert's
                    # gate/up tiles, so they stream while those finish
                    kk2 = tl.arange(0, I)
                    wpre = _ldw(wbase + (h * BH + tl.arange(0, BH))[None, :] * I + kk2[:, None],
                                EVF)
                if acq == 0:
                    _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                    acq = 1
                _wait_geq(g1d_ptr + u * G1D_STRIDE, NT1, SLEEP)
                ph = 5
                if TRACE > 0:
                    t_r = tl.extra.cuda.globaltimer()
                if EV > 0:
                    tl.atomic_min(ev_ptr + 6, tl.extra.cuda.globaltimer(), sem="relaxed")
                for m0 in range(0, n, BM):
                    mi = m0 + tl.arange(0, BM)
                    mm = mi < n
                    if m0 == 0:
                        p = first
                    else:
                        p = tl.load(lists_ptr + e * T + mi, mask=mm, other=0, cache_modifier=".cg")
                    if PRE2:
                        _down_pre(act_ptr, wpre, pw_ptr, y_ptr, p // TOPK, p, mm, h, red_ptr,
                                  need_ptr, ploc_ptr, out_ptr, ps, pe, shared_rows, H, I, IA,
                                  TOPK, NT2, BM, BH, F32)
                    else:
                        _down_arrive(act_ptr, wbase, pw_ptr, y_ptr, p // TOPK, p, mm, h, red_ptr,
                                     need_ptr, ploc_ptr, out_ptr, ps, pe, shared_rows, H, I, IA,
                                     TOPK, NT2, BM, BH, BK2, F32)
            else:
                running = 0
        if TRACE > 0:
            if ph >= 0:
                _trace(ev_ptr, tk, item, ph, t_f, t_r, TRACE)
                tk += 1
        if running != 0:
            if LA == 1:
                nxt = NUM_CTAS + _grabbed(gnx)
            else:
                nxt = NUM_CTAS + tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
            if sig != 0:
                _signal(sig_ptr)
            item = nxt

    if EV > 0:
        tl.atomic_max(ev_ptr + 7, tl.extra.cuda.globaltimer(), sem="relaxed")
    # the last CTA out resets the plan and the counters for the next call
    old = tl.atomic_add(ctl_ptr + C_EXIT, 1, sem="acq_rel", scope="gpu")
    if old == NUM_CTAS - 1:
        tl.debug_barrier()
        if EV > 0:
            # the next call's event slot, then the call count
            ev_next = ev_ptr - (ev_seq % EV_RING) * 32 + ((ev_seq + 1) % EV_RING) * 32
            w32 = tl.arange(0, 32)
            big = tl.full((32,), 1 << 62, dtype=tl.int64)
            init = tl.where((w32 == 0) | (w32 == 3) | (w32 == 6) | ((w32 == 10) & (NK > 1)), big,
                            tl.where((w32 >= 10) | (w32 == 1) | (w32 == 2), -1, 0).to(tl.int64))
            tl.store(ev_next + w32, init)
            tl.store(ev_ptr - (ev_seq % EV_RING) * 32 + EV_RING * 32, ev_seq + 1)
        # the touched-expert count this CTA already read, else the header (one round trip)
        uf = U
        if uf < 0:
            uf = tl.load(ctl_ptr + C_PLAN, cache_modifier=".cg") - 1
        # a marked word left from this call must never be read by the next one
        wl = tl.arange(0, 8 * TL)
        for l0 in range(0, uf * TL, 8 * TL):
            tl.store(touched_ptr + l0 + wl, tl.zeros((8 * TL,), dtype=tl.int32),
                     mask=l0 + wl < uf * TL)
        rt = tl.arange(0, 256)
        for t0 in range(0, T, 256):
            tl.store(gam_ptr + t0 + rt,
                     tl.full((256,), -1, dtype=tl.int32).to(tl.float32, bitcast=True),
                     mask=t0 + rt < T)
        ee = tl.arange(0, E_LOC)
        tl.store(cnt_ptr + ee, tl.zeros((E_LOC,), dtype=tl.int32))
        tl.store(g1d_ptr + ee * G1D_STRIDE, tl.zeros((E_LOC,), dtype=tl.int32))
        cc = tl.arange(0, 256)
        for c0 in tl.static_range(0, CTL_WORDS, 256):
            tl.store(ctl_ptr + c0 + cc, tl.zeros((256,), dtype=tl.int32))
