# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own Qwen3.5-MoE decode-block Triton tree (548eb81d); CuTe rewrite pending.
"""Persistent MoE queue: router, top-k, shared and routed SwiGLU, fixed-order FP32 combine.

Producers release counters after stores; consumers acquire before reading scratch.
Down tiles may preload weights before awaiting activations. State is private to the op.
The router clears counted state and sentinels before its PDL-dependent block runs.
Static first-wave items precede the atomic queue; dependencies point to earlier items.
The last arrival combines slots 0..7 then the shared term, independent of scheduling.
Marked plan lines publish local slots/counts and the first token pairs to consumers.
"""

# Triton pointer annotations and the conventional intermediate-width symbol.
# ruff: noqa: E741, F722
# fmt: off


import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...triton.rounding import INTERPRET, bf16_round
from ...triton.topk_softmax import topk_softmax

# ctl slots, one 128-byte line apart so that polled flags, the queue head and the counters
# do not share L2 lines; per-expert gate/up completion counters are one line apart too
C_HEAD = tl.constexpr(0)
C_TOPK = tl.constexpr(64)
C_PLAN = tl.constexpr(96)
C_SH1 = tl.constexpr(128)
C_PLANA = tl.constexpr(160)
CTL_WORDS = tl.constexpr(192)
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
def _ld_slot(ptr, m, VOL: tl.constexpr):
    # (DAF) a slot tile: .cg (L2) on the first read, volatile on a re-read
    if VOL:
        v = tl.load(ptr, mask=m, other=0.0, volatile=True)
    else:
        v = tl.load(ptr, mask=m, other=0.0, cache_modifier=".cg")
    return v


@triton.jit
def _isent(v, m):
    # (DAF) the elements under m still holding the sentinel (-1 bits: no FP32 result has them), elementwise
    return (v.to(tl.int32, bitcast=True) == -1) & m


@triton.jit
def _resent(v, ptr, m):
    # (DAF) the elements under m still holding the sentinel, read again
    return tl.where(m & (v.to(tl.int32, bitcast=True) == -1), _ld_slot(ptr, m, True), v)


def _await9_asm():
    # (DAF 9 / 10) one wait per thread over its 36 slot elements (9 slot tiles x 4 elements, pack = 4): outputs
    # $0..$35 (slot k's j-th element at 4k + j), inputs $36..$71 (the same order; masked-out elements are 0, never
    # the sentinel), $72..$75 the elements' slot-0 addresses (slot k at + k * 8192 bytes: H = 2048 FP32 words), $76..$79
    # the shared slot's. Not taken unless a value still holds the sentinel: one predicate over the 36 values and one
    # branch, no exchange across threads.
    out = ["{", ".reg .pred p, q;"]
    out += [f"mov.b32 ${o}, ${36 + o};" for o in range(36)]
    test = ["setp.eq.b32 p, $0, -1;"] + [f"setp.eq.or.b32 p, ${o}, -1, p;" for o in range(1, 36)]
    out += test + ["@!p bra LB${:uid};"]
    out.append("LA${:uid}:")
    out.append("nanosleep.u32 100;")
    for k in range(9):
        for j in range(4):
            o = 4 * k + j
            addr = f"[${72 + j}+{k * 8192}]" if k < 8 else f"[${76 + j}]"
            out += [f"setp.eq.b32 q, ${o}, -1;", f"@q ld.volatile.global.b32 ${o}, {addr};"]
    out += test + ["@p bra LA${:uid};", "LB${:uid}:", "}"]
    return " ".join(out)


_A9 = tl.constexpr(_await9_asm())
_A9C = tl.constexpr(",".join(["=r"] * 36 + ["r"] * 36 + ["l"] * 8))


@triton.jit
def _sent(v):
    # (DAF 9 / 10) the elements still holding the sentinel
    return v.to(tl.int32, bitcast=True) == -1


@triton.jit
def _await9(w0, w1, w2, w3, w4, w5, w6, w7, ws, sb, shb, H: tl.constexpr):
    # (DAF 9 / 10) the masked slot values (0 where unused), each element still holding the sentinel read again by its
    # own thread until it does not; in the interpreter (no PTX) the same as a CTA-wide loop
    tl.static_assert(H == 2048, "_await9's slot stride")
    if INTERPRET:
        nb = tl.max((_sent(w0) | _sent(w1) | _sent(w2) | _sent(w3) | _sent(w4) | _sent(w5) | _sent(w6) | _sent(w7)
                     | _sent(ws)).to(tl.int32))
        while nb > 0:
            w0 = _resent(w0, sb + 0 * H, _sent(w0))
            w1 = _resent(w1, sb + 1 * H, _sent(w1))
            w2 = _resent(w2, sb + 2 * H, _sent(w2))
            w3 = _resent(w3, sb + 3 * H, _sent(w3))
            w4 = _resent(w4, sb + 4 * H, _sent(w4))
            w5 = _resent(w5, sb + 5 * H, _sent(w5))
            w6 = _resent(w6, sb + 6 * H, _sent(w6))
            w7 = _resent(w7, sb + 7 * H, _sent(w7))
            ws = _resent(ws, shb, _sent(ws))
            nb = tl.max((_sent(w0) | _sent(w1) | _sent(w2) | _sent(w3) | _sent(w4) | _sent(w5) | _sent(w6)
                         | _sent(w7) | _sent(ws)).to(tl.int32))
        r0, r1, r2, r3, r4, r5, r6, r7, rs = w0, w1, w2, w3, w4, w5, w6, w7, ws
    else:
        q0, q1, q2, q3, q4, q5, q6, q7, qs = tl.inline_asm_elementwise(
            _A9, _A9C, [w0.to(tl.int32, bitcast=True), w1.to(tl.int32, bitcast=True), w2.to(tl.int32, bitcast=True),
                        w3.to(tl.int32, bitcast=True), w4.to(tl.int32, bitcast=True), w5.to(tl.int32, bitcast=True),
                        w6.to(tl.int32, bitcast=True), w7.to(tl.int32, bitcast=True), ws.to(tl.int32, bitcast=True),
                        sb, shb],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
            is_pure=False, pack=4)
        r0 = q0.to(tl.float32, bitcast=True)
        r1 = q1.to(tl.float32, bitcast=True)
        r2 = q2.to(tl.float32, bitcast=True)
        r3 = q3.to(tl.float32, bitcast=True)
        r4 = q4.to(tl.float32, bitcast=True)
        r5 = q5.to(tl.float32, bitcast=True)
        r6 = q6.to(tl.float32, bitcast=True)
        r7 = q7.to(tl.float32, bitcast=True)
        rs = qs.to(tl.float32, bitcast=True)
    return r0, r1, r2, r3, r4, r5, r6, r7, rs


@triton.jit
def _opaque_zero():
    # (LRZ) 0 that the compiler cannot see through or move out of a loop: index math built on
    # it is computed where it is used, not hoisted to the kernel's start
    if INTERPRET:
        return 0
    else:
        return tl.inline_asm_elementwise("mov.u32 $0, 0;", "=r", [], dtype=tl.int32,
                                         is_pure=False, pack=1)


@triton.jit
def _vmix_q(valid, VBIN_MIN: tl.constexpr, VBIN_MAX: tl.constexpr, VMIX: tl.constexpr,
            VMIX_MIN: tl.constexpr, VMIX_MAX: tl.constexpr):
    # (VMIX) the narrow experts' share in quarters of the touched experts, by the valid count
    return tl.where((valid >= VBIN_MIN) & (valid <= VBIN_MAX), 4,
                    tl.where((valid >= VMIX_MIN) & (valid <= VMIX_MAX), VMIX, 0))


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
def _grid_wait():
    # griddepcontrol.wait: the preceding grid (router_kernel) complete, its stores visible
    # (the CPU interpreter runs grids one after another: nothing to wait for)
    if not INTERPRET:
        tl.extra.cuda.gdc_wait()


@triton.jit
def _launch_dependents():
    # griddepcontrol.launch_dependents: the dependent grid may launch now
    if not INTERPRET:
        tl.extra.cuda.gdc_launch_dependents()


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
                 VLATE: tl.constexpr = False, valid_ptr=None, row0=0, T_ALL=0):
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
        wb = tl.load(rw_ptr + (eb * BE_R + cols)[None, :] * H + kk[:, None])
    else:
        wb = tl.load(sgw_ptr + kk[:, None] + 0 * cols[None, :], mask=(cols == 0)[None, :],
                     other=0.0)
    acc = _mm(xa, wb, tl.zeros((BT_R, BE_R), dtype=tl.float32), F32)
    if VLATE:
        # (router_kernel's SKIP, the first token block) the valid count read only here, after
        # the item's loads: the store mask is its only use
        v_raw = tl.load(valid_ptr)
        valid = tl.minimum(tl.maximum(tl.minimum(tl.maximum(v_raw, 0), T_ALL) - row0, 0), T)
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


@triton.jit(noinline=True)
def _route_one_block_ni(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr,
                        touched_ptr, ctl_ptr, valid, T, ps, pe, H: tl.constexpr, E: tl.constexpr,
                        E_LOC: tl.constexpr, TOPK: tl.constexpr, KS: tl.constexpr,
                        BE_R: tl.constexpr, BT_K: tl.constexpr, SB: tl.constexpr, ev_ptr,
                        EV: tl.constexpr, RS: tl.constexpr):
    """_route_one_block compiled apart from the kernel (ONI): its registers are not the item
    loop's, so the branch that calls it adds no stack frame to the kernel."""
    _route_one_block(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr, need_ptr, out_ptr,
                     touched_ptr, ctl_ptr, valid, T, ps, pe, H, E, E_LOC, TOPK, KS, BE_R, BT_K,
                     SB, ev_ptr, EV, RS)


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
def _routed_items(U, NUM_CTAS, NT1: tl.constexpr, NS1: tl.constexpr, NS2: tl.constexpr,
                  I: tl.constexpr, BIN: tl.constexpr, BIN_SLACK: tl.constexpr):
    """Routed gate/up items per touched expert. With BIN, the narrower items (I // BIN per
    expert) are used when every one of them has a CTA of the resident grid by the time the
    plan is published: the grid's CTAs minus the shared expert's items (the top-k CTAs free
    up with the plan), plus BIN_SLACK. Otherwise some of them would start only once shared
    items finish, which ends the gate/up phase later instead of earlier. Every CTA computes
    it from the same plan header, so all agree; the output bits do not depend on it.
    (Called only with BIN > 0.)"""
    NT1N: tl.constexpr = I // BIN
    fits = NS1 + NS2 + U * NT1N <= NUM_CTAS + BIN_SLACK
    return tl.where(fits, NT1N, NT1)


@triton.jit
def _gate_up_tile(x_ptr, w_ptr, act_ptr, rows_tok, rows_act, mmask, j,
                  H: tl.constexpr, IW: tl.constexpr, IA: tl.constexpr,
                  BM: tl.constexpr, BI: tl.constexpr, BK1: tl.constexpr, F32: tl.constexpr,
                  GCG: tl.constexpr = 0, hend=0, STG: tl.constexpr = 0,
                  LRZ: tl.constexpr = False, XB: tl.constexpr = False,
                  SHARED: tl.constexpr = False):
    # columns interleave gate row i and up row i: column 2c -> gate j*BI+c, 2c+1 -> up
    if LRZ:
        # (LRZ) the tile's thread-dependent index vectors built on an opaque zero, so they are
        # computed in the item rather than hoisted out of the item loop and kept live (spilled)
        # across it; scaled by the tile's own sizes, so the offsets keep their alignment and the
        # loads stay 16-byte vectors
        oz = _opaque_zero()
    else:
        oz = 0
    cols = tl.arange(0, 2 * BI) + oz * (2 * BI)
    wrow = (cols % 2) * IW + j * BI + cols // 2
    acc = tl.zeros((BM, 2 * BI), dtype=tl.float32)
    if XB:
        # (XB: at most one token in the tile, the RTX 5090's ep4 one-row configuration) the
        # x operand is that token's row broadcast over the tile's rows, unmasked: the other
        # rows compute copies of it and are not stored (mmask), and the token's row has the
        # same operands, K steps and FP32 accumulation as with the masked tile load. The
        # operand stays a broadcast of the 1-D row in registers, so the loop stages only the
        # weights (no 16-row x staging, no per-step conversion through shared memory)
        tok0 = tl.max(tl.where(mmask, rows_tok, 0), axis=0)
    if STG > 0:
        # (STG) this loop's own pipeline depth, not the kernel's
        for k0 in tl.range(0, H, BK1, num_stages=STG):
            kk = k0 + oz * BK1 + tl.arange(0, BK1)
            xa = tl.load(x_ptr + rows_tok[:, None] * H + kk[None, :], mask=mmask[:, None],
                         other=0.0)
            wb = tl.load(w_ptr + wrow[None, :] * H + kk[:, None])
            acc = _mm(xa, wb, acc, F32)
    elif GCG:
        # (GCG) the same pipelined K loop (the kernel's stages) with a trip count the compiler
        # cannot fold (hend is H at run time): its body compiled once, not unrolled per K step
        for k0 in range(0, hend, BK1):
            kk = k0 + oz * BK1 + tl.arange(0, BK1)
            xa = tl.load(x_ptr + rows_tok[:, None] * H + kk[None, :], mask=mmask[:, None],
                         other=0.0)
            wb = tl.load(w_ptr + wrow[None, :] * H + kk[:, None])
            acc = _mm(xa, wb, acc, F32)
    else:
        for k0 in range(0, H, BK1):
            kk = k0 + oz * BK1 + tl.arange(0, BK1)
            if XB:
                xr = tl.load(x_ptr + tok0 * H + kk)
                xa = tl.broadcast_to(xr[None, :], (BM, BK1))
            else:
                xa = tl.load(x_ptr + rows_tok[:, None] * H + kk[None, :], mask=mmask[:, None],
                             other=0.0)
            wb = tl.load(w_ptr + wrow[None, :] * H + kk[:, None])
            acc = _mm(xa, wb, acc, F32)
    g, u = tl.split(tl.reshape(acc, (BM, BI, 2)))
    a = _silu_mul(g, u, SHARED)
    ci = j * BI + tl.arange(0, BI)
    tl.store(act_ptr + rows_act[:, None] * IA + ci[None, :], _bf16(a), mask=mmask[:, None])


@triton.jit
def _routed_gate_up(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr, ctl_ptr, T, acq,
                    H: tl.constexpr, I: tl.constexpr, IA: tl.constexpr, TOPK: tl.constexpr,
                    BM: tl.constexpr, BIW: tl.constexpr, BK1: tl.constexpr, F32: tl.constexpr,
                    GCGW: tl.constexpr, hend, STG: tl.constexpr, SLEEP: tl.constexpr,
                    LRZ: tl.constexpr = False):
    """(VBIN) Routed gate/up item d at the compile-time width BIW: its expert and tile from a
    constant divisor. Instantiated once per width, so the two widths share no item code (a
    shared body, decoded by the run-time item count, cost every item of the wide width)."""
    NTW: tl.constexpr = I // BIW
    u = d // NTW
    j = d % NTW
    e, n, first = _line(touched_ptr, u, BM, SLEEP)
    wbase = wgu_ptr + e.to(tl.int64) * (2 * I * H)
    if n > BM:
        # the rest of the token list was written by other top-k CTAs
        if acq == 0:
            _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
            acq = 1
    for m0 in range(0, n, BM):
        mi = m0 + tl.arange(0, BM)
        mm = mi < n
        if m0 == 0:
            p = first
        else:
            p = tl.load(lists_ptr + e * T + mi, mask=mm, other=0, cache_modifier=".cg")
        _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j, H, I, IA, BM, BIW, BK1, F32,
                      GCGW, hend, STG, LRZ)
    return u, acq


@triton.jit(noinline=True)
def _routed_gate_up_ni(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr, ctl_ptr, T, acq,
                       H: tl.constexpr, I: tl.constexpr, IA: tl.constexpr, TOPK: tl.constexpr,
                       BM: tl.constexpr, BIW: tl.constexpr, BK1: tl.constexpr, F32: tl.constexpr,
                       GCGW: tl.constexpr, hend, STG: tl.constexpr, SLEEP: tl.constexpr,
                       LRZ: tl.constexpr = 0):
    """(NIG) _routed_gate_up as a separate function (not inlined): its registers are allocated
    apart from the item dispatch around it, so the dispatch's mixed widths (VMIX) leave the
    tile's code as it is."""
    return _routed_gate_up(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr, ctl_ptr, T, acq,
                           H, I, IA, TOPK, BM, BIW, BK1, F32, GCGW, hend, STG, SLEEP, LRZ)


@triton.jit
def _down_tile(act_ptr, w_ptr, rows_act, mmask, h, IW: tl.constexpr, IA: tl.constexpr,
               BM: tl.constexpr, BH: tl.constexpr, BK2: tl.constexpr, F32: tl.constexpr,
               DST: tl.constexpr = 0):
    acc = tl.zeros((BM, BH), dtype=tl.float32)
    if DST > 0:
        # (DST) this loop's own pipeline depth, not the kernel's; its index math on the opaque
        # zero (as LRZ), so it is not hoisted out of the item loop
        oz = _opaque_zero()
        hh = h * BH + oz * BH + tl.arange(0, BH)
        for k0 in tl.range(0, IW, BK2, num_stages=DST):
            kk = k0 + oz * BK2 + tl.arange(0, BK2)
            a = tl.load(act_ptr + rows_act[:, None] * IA + kk[None, :], mask=mmask[:, None],
                        other=0.0, cache_modifier=".cg")
            wb = tl.load(w_ptr + hh[None, :] * IW + kk[:, None])
            acc = _mm(a, wb, acc, F32)
    else:
        hh = h * BH + tl.arange(0, BH)
        for k0 in range(0, IW, BK2):
            kk = k0 + tl.arange(0, BK2)
            a = tl.load(act_ptr + rows_act[:, None] * IA + kk[None, :], mask=mmask[:, None],
                        other=0.0, cache_modifier=".cg")
            wb = tl.load(w_ptr + hh[None, :] * IW + kk[:, None])
            acc = _mm(a, wb, acc, F32)
    return acc, hh


@triton.jit
def _wk(pw_ptr, tok, k, fin, TOPK: tl.constexpr):
    # slot k's FP32 routing weight per row, as a column (the finalize's row_scale)
    return tl.load(pw_ptr + tok * TOPK + k, mask=fin, other=0.0, cache_modifier=".cg")[:, None]


@triton.jit
def _join(acc, shared):
    # the routed sum rounded once, then + the BF16 gated shared row in BF16 (vLLM's block)
    return _bf16(bf16_round(acc) + shared)


@triton.jit
def _arrive(tok, mmask, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe, shared_rows,
            H: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr, BM: tl.constexpr,
            BH: tl.constexpr, CB: tl.constexpr = False, CBR: tl.constexpr = False,
            DAF: tl.constexpr = 0, own=None, kself=None, HAS_OWN: tl.constexpr = False, pbits=None):
    """Combine as vLLM's finalize: acc = fma(w_k, y_k, acc) over the unweighted BF16 slots
    0..7 in order, rounded once, then the gated shared term added in BF16. DAF uses -1 sentinel
    bits as data-ready flags: after the relaxed arrival count, the completer polls
    slots through L2 until their stores arrive. HAS_OWN can use its register value;
    DAF 9/10 instead reads its own stored slot after a CTA barrier. DAF 10 reuses
    slot-presence bits formed before the down projection's K loop.
    """
    need = tl.load(need_ptr + tok, mask=mmask, other=0, cache_modifier=".cg")
    kk = tl.arange(0, TOPK)
    locs = tl.load(ploc_ptr + tok[:, None] * TOPK + kk[None, :], mask=mmask[:, None], other=-1,
                   cache_modifier=".cg")
    hh = h * BH + tl.arange(0, BH)
    if DAF >= 9 and HAS_OWN:
        tok2 = tok[:, None] + 0 * hh[None, :]
        mm2 = mmask[:, None] & (hh[None, :] >= 0)
        sb = y_ptr + (tok2 * TOPK) * H + hh[None, :]
        shb = y_ptr + (shared_rows + tok2) * H + hh[None, :]
        tl.debug_barrier()
        if pbits is not None:
            # (DAF 10) only the rows' used slots
            v0 = _ld_slot(sb + 0 * H, mm2 & ((pbits & 1) != 0), False)
            v1 = _ld_slot(sb + 1 * H, mm2 & (((pbits >> 1) & 1) != 0), False)
            v2 = _ld_slot(sb + 2 * H, mm2 & (((pbits >> 2) & 1) != 0), False)
            v3 = _ld_slot(sb + 3 * H, mm2 & (((pbits >> 3) & 1) != 0), False)
            v4 = _ld_slot(sb + 4 * H, mm2 & (((pbits >> 4) & 1) != 0), False)
            v5 = _ld_slot(sb + 5 * H, mm2 & (((pbits >> 5) & 1) != 0), False)
            v6 = _ld_slot(sb + 6 * H, mm2 & (((pbits >> 6) & 1) != 0), False)
            v7 = _ld_slot(sb + 7 * H, mm2 & (((pbits >> 7) & 1) != 0), False)
        else:
            # (DAF 9) every slot: the unused ones read the fill's sentinel and are masked out on the completer
            v0 = _ld_slot(sb + 0 * H, mm2, False)
            v1 = _ld_slot(sb + 1 * H, mm2, False)
            v2 = _ld_slot(sb + 2 * H, mm2, False)
            v3 = _ld_slot(sb + 3 * H, mm2, False)
            v4 = _ld_slot(sb + 4 * H, mm2, False)
            v5 = _ld_slot(sb + 5 * H, mm2, False)
            v6 = _ld_slot(sb + 6 * H, mm2, False)
            v7 = _ld_slot(sb + 7 * H, mm2, False)
        vs = _ld_slot(shb, mm2 & (tok2 >= ps) & (tok2 < pe), False)
        old = tl.atomic_add(red_ptr + tok * NT2 + h, 1, mask=mmask, sem="relaxed", scope="gpu")
        fin = mmask & (old + 1 == need)
        if tl.max(fin.to(tl.int32), axis=0) > 0:
            tl.debug_barrier()
            if pbits is not None:
                bits2 = pbits
            else:
                bits = tl.sum(tl.where(locs >= 0, 1 << kk[None, :], 0), axis=1)
                bits2 = bits[:, None] + 0 * hh[None, :]
            f2 = fin[:, None] & (hh[None, :] >= 0)
            m0 = f2 & ((bits2 & 1) != 0)
            m1 = f2 & (((bits2 >> 1) & 1) != 0)
            m2 = f2 & (((bits2 >> 2) & 1) != 0)
            m3 = f2 & (((bits2 >> 3) & 1) != 0)
            m4 = f2 & (((bits2 >> 4) & 1) != 0)
            m5 = f2 & (((bits2 >> 5) & 1) != 0)
            m6 = f2 & (((bits2 >> 6) & 1) != 0)
            m7 = f2 & (((bits2 >> 7) & 1) != 0)
            ms = f2 & (tok2 >= ps) & (tok2 < pe)
            y0, y1, y2, y3, y4, y5, y6, y7, ys = _await9(
                tl.where(m0, v0, 0.0), tl.where(m1, v1, 0.0), tl.where(m2, v2, 0.0), tl.where(m3, v3, 0.0),
                tl.where(m4, v4, 0.0), tl.where(m5, v5, 0.0), tl.where(m6, v6, 0.0), tl.where(m7, v7, 0.0),
                tl.where(ms, vs, 0.0), sb, shb, H)
            acc = tl.zeros((BM, BH), dtype=tl.float32)
            acc = tl.fma(_wk(pw_ptr, tok, 0, fin, TOPK), y0, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 1, fin, TOPK), y1, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 2, fin, TOPK), y2, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 3, fin, TOPK), y3, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 4, fin, TOPK), y4, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 5, fin, TOPK), y5, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 6, fin, TOPK), y6, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 7, fin, TOPK), y7, acc)
            tl.store(out_ptr + tok[:, None] * H + hh[None, :], _join(acc, ys), mask=fin[:, None])
    elif DAF > 0:
        tok2 = tok[:, None] + 0 * hh[None, :]
        mm2 = mmask[:, None] & (hh[None, :] >= 0)
        sb = y_ptr + (tok2 * TOPK) * H + hh[None, :]
        shb = y_ptr + (shared_rows + tok2) * H + hh[None, :]
        if HAS_OWN:
            ks2 = kself[:, None] + 0 * hh[None, :]
        else:
            ks2 = tok2 * 0 - 1
        if HAS_OWN:
            # the other slots read now, without the slot bits - unused slots, never written this
            # call, read the fill's sentinel and are masked out on the completer
            v0 = _ld_slot(sb + 0 * H, mm2 & (ks2 != 0), False)
            v1 = _ld_slot(sb + 1 * H, mm2 & (ks2 != 1), False)
            v2 = _ld_slot(sb + 2 * H, mm2 & (ks2 != 2), False)
            v3 = _ld_slot(sb + 3 * H, mm2 & (ks2 != 3), False)
            v4 = _ld_slot(sb + 4 * H, mm2 & (ks2 != 4), False)
            v5 = _ld_slot(sb + 5 * H, mm2 & (ks2 != 5), False)
            v6 = _ld_slot(sb + 6 * H, mm2 & (ks2 != 6), False)
            v7 = _ld_slot(sb + 7 * H, mm2 & (ks2 != 7), False)
            vs = _ld_slot(shb, mm2 & (tok2 >= ps) & (tok2 < pe), False)
        tl.debug_barrier()
        old = tl.atomic_add(red_ptr + tok * NT2 + h, 1, mask=mmask, sem="relaxed", scope="gpu")
        fin = mmask & (old + 1 == need)
        if tl.max(fin.to(tl.int32), axis=0) > 0:
            tl.debug_barrier()
            bits = tl.sum(tl.where(locs >= 0, 1 << kk[None, :], 0), axis=1)
            bits2 = bits[:, None] + 0 * hh[None, :]
            f2 = fin[:, None] & (hh[None, :] >= 0)
            m0 = f2 & ((bits2 & 1) != 0) & (ks2 != 0)
            m1 = f2 & (((bits2 >> 1) & 1) != 0) & (ks2 != 1)
            m2 = f2 & (((bits2 >> 2) & 1) != 0) & (ks2 != 2)
            m3 = f2 & (((bits2 >> 3) & 1) != 0) & (ks2 != 3)
            m4 = f2 & (((bits2 >> 4) & 1) != 0) & (ks2 != 4)
            m5 = f2 & (((bits2 >> 5) & 1) != 0) & (ks2 != 5)
            m6 = f2 & (((bits2 >> 6) & 1) != 0) & (ks2 != 6)
            m7 = f2 & (((bits2 >> 7) & 1) != 0) & (ks2 != 7)
            ms = f2 & (tok2 >= ps) & (tok2 < pe)
            if not HAS_OWN:
                v0 = _ld_slot(sb + 0 * H, m0, False)
                v1 = _ld_slot(sb + 1 * H, m1, False)
                v2 = _ld_slot(sb + 2 * H, m2, False)
                v3 = _ld_slot(sb + 3 * H, m3, False)
                v4 = _ld_slot(sb + 4 * H, m4, False)
                v5 = _ld_slot(sb + 5 * H, m5, False)
                v6 = _ld_slot(sb + 6 * H, m6, False)
                v7 = _ld_slot(sb + 7 * H, m7, False)
                vs = _ld_slot(shb, ms, False)
            # one elementwise test of the nine slots, then one reduction
            nb = tl.max((_isent(v0, m0) | _isent(v1, m1) | _isent(v2, m2) | _isent(v3, m3) | _isent(v4, m4)
                         | _isent(v5, m5) | _isent(v6, m6) | _isent(v7, m7) | _isent(vs, ms)).to(tl.int32))
            while nb > 0:
                # a contributor's store not yet visible: read those elements again
                _nap(100)
                v0 = _resent(v0, sb + 0 * H, m0)
                v1 = _resent(v1, sb + 1 * H, m1)
                v2 = _resent(v2, sb + 2 * H, m2)
                v3 = _resent(v3, sb + 3 * H, m3)
                v4 = _resent(v4, sb + 4 * H, m4)
                v5 = _resent(v5, sb + 5 * H, m5)
                v6 = _resent(v6, sb + 6 * H, m6)
                v7 = _resent(v7, sb + 7 * H, m7)
                vs = _resent(vs, shb, ms)
                nb = tl.max((_isent(v0, m0) | _isent(v1, m1) | _isent(v2, m2) | _isent(v3, m3) | _isent(v4, m4)
                             | _isent(v5, m5) | _isent(v6, m6) | _isent(v7, m7) | _isent(vs, ms)).to(tl.int32))
            if HAS_OWN:
                # (own in the slots' layout once)
                own = tl.where(f2, own, 0.0)
                y0 = tl.where(f2 & (ks2 == 0), own, tl.where(m0, v0, 0.0))
                y1 = tl.where(f2 & (ks2 == 1), own, tl.where(m1, v1, 0.0))
                y2 = tl.where(f2 & (ks2 == 2), own, tl.where(m2, v2, 0.0))
                y3 = tl.where(f2 & (ks2 == 3), own, tl.where(m3, v3, 0.0))
                y4 = tl.where(f2 & (ks2 == 4), own, tl.where(m4, v4, 0.0))
                y5 = tl.where(f2 & (ks2 == 5), own, tl.where(m5, v5, 0.0))
                y6 = tl.where(f2 & (ks2 == 6), own, tl.where(m6, v6, 0.0))
                y7 = tl.where(f2 & (ks2 == 7), own, tl.where(m7, v7, 0.0))
            else:
                y0 = tl.where(m0, v0, 0.0)
                y1 = tl.where(m1, v1, 0.0)
                y2 = tl.where(m2, v2, 0.0)
                y3 = tl.where(m3, v3, 0.0)
                y4 = tl.where(m4, v4, 0.0)
                y5 = tl.where(m5, v5, 0.0)
                y6 = tl.where(m6, v6, 0.0)
                y7 = tl.where(m7, v7, 0.0)
            acc = tl.zeros((BM, BH), dtype=tl.float32)
            acc = tl.fma(_wk(pw_ptr, tok, 0, fin, TOPK), y0, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 1, fin, TOPK), y1, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 2, fin, TOPK), y2, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 3, fin, TOPK), y3, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 4, fin, TOPK), y4, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 5, fin, TOPK), y5, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 6, fin, TOPK), y6, acc)
            acc = tl.fma(_wk(pw_ptr, tok, 7, fin, TOPK), y7, acc)
            tl.store(out_ptr + tok[:, None] * H + hh[None, :], _join(acc, tl.where(ms, vs, 0.0)),
                     mask=fin[:, None])
    else:
        tl.debug_barrier()
        old = tl.atomic_add(red_ptr + tok * NT2 + h, 1, mask=mmask, sem="acq_rel", scope="gpu")
        fin = mmask & (old + 1 == need)
        if tl.max(fin.to(tl.int32), axis=0) > 0:
            tl.debug_barrier()
            acc = tl.zeros((BM, BH), dtype=tl.float32)
            if CB:
                # the row's local slots as bits in the tile's layout: one reduction and one layout
                # change, then the eight slot loads back to back (per slot, each load waited on a
                # reduction and a layout change of its own: ~30 barriers between the first and the
                # ninth load)
                bits = tl.sum(tl.where(locs >= 0, 1 << kk[None, :], 0), axis=1)
                bits2 = bits[:, None] + 0 * hh[None, :]
                fin2 = fin[:, None] & (hh[None, :] >= 0)
                if CBR:
                    # the rows' token index in the tile's layout once, as bits2: each slot's address
                    # (and the shared term's below) is then formed in that layout, without a layout
                    # change - shared memory and two barriers - before each load
                    tok2 = tok[:, None] + 0 * hh[None, :]
                    for k in tl.static_range(TOPK):
                        use = fin2 & (((bits2 >> k) & 1) != 0)
                        yk = tl.load(y_ptr + (tok2 * TOPK + k) * H + hh[None, :], mask=use,
                                     other=0.0, cache_modifier=".cg")
                        acc = tl.fma(_wk(pw_ptr, tok, k, fin, TOPK), yk, acc)
                else:
                    for k in tl.static_range(TOPK):
                        pk = tok * TOPK + k
                        use = fin2 & (((bits2 >> k) & 1) != 0)
                        yk = tl.load(y_ptr + pk[:, None] * H + hh[None, :], mask=use, other=0.0,
                                     cache_modifier=".cg")
                        acc = tl.fma(_wk(pw_ptr, tok, k, fin, TOPK), yk, acc)
            else:
                for k in tl.static_range(TOPK):
                    pk = tok * TOPK + k
                    loc = tl.sum(tl.where(kk[None, :] == k, locs, 0), axis=1)
                    use = fin & (loc >= 0)
                    yk = tl.load(y_ptr + pk[:, None] * H + hh[None, :], mask=use[:, None],
                                 other=0.0, cache_modifier=".cg")
                    acc = tl.fma(_wk(pw_ptr, tok, k, fin, TOPK), yk, acc)
            if CB and CBR:
                inpart2 = fin2 & (tok2 >= ps) & (tok2 < pe)
                ys = tl.load(y_ptr + (shared_rows + tok2) * H + hh[None, :], mask=inpart2,
                             other=0.0, cache_modifier=".cg")
            else:
                inpart = fin & (tok >= ps) & (tok < pe)
                ys = tl.load(y_ptr + (shared_rows + tok)[:, None] * H + hh[None, :],
                             mask=inpart[:, None], other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + tok[:, None] * H + hh[None, :], _join(acc, ys), mask=fin[:, None])
            # no other arrival comes for this tile in this call: reset it for the next one


@triton.jit
def _down_pre(act_ptr, wpre, pw_ptr, y_ptr, tok, p, mm, h, red_ptr, need_ptr, ploc_ptr,
                out_ptr, ps, pe, shared_rows, H: tl.constexpr, I: tl.constexpr,
                IA: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr, BM: tl.constexpr,
                BH: tl.constexpr, F32: tl.constexpr, CB: tl.constexpr = False, DAF: tl.constexpr = 0):
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
            H, TOPK, NT2, BM, BH, CB, DAF=DAF)


@triton.jit
def _down_arrive(act_ptr, wbase, pw_ptr, y_ptr, tok, p, mm, h, red_ptr, need_ptr, ploc_ptr,
                 out_ptr, ps, pe, shared_rows, H: tl.constexpr, I: tl.constexpr,
                 IA: tl.constexpr, TOPK: tl.constexpr, NT2: tl.constexpr, BM: tl.constexpr,
                 BH: tl.constexpr, BK2: tl.constexpr, F32: tl.constexpr, CB: tl.constexpr = False,
                 DST: tl.constexpr = 0, CBR: tl.constexpr = False,
                 DAF: tl.constexpr = 0):
    """One routed down tile for the pairs p (tokens tok): y = BF16(act @ W_down^T),
    unweighted, then the tile's arrival."""
    if DAF == 10:
        # (DAF 10) the rows' slot bits in the tile's layout before the tile: their
        # reduction and layout change leave the completer's path, and the arrival reads only the used slots
        kk = tl.arange(0, TOPK)
        locs = tl.load(ploc_ptr + tok[:, None] * TOPK + kk[None, :], mask=mm[:, None], other=-1,
                       cache_modifier=".cg")
        pbits = tl.sum(tl.where(locs >= 0, 1 << kk[None, :], 0), axis=1)
        pbits = pbits[:, None] + 0 * (h * BH + tl.arange(0, BH))[None, :]
    else:
        pbits = None
    acc, hh = _down_tile(act_ptr, wbase, p, mm, h, I, IA, BM, BH, BK2, F32, DST)
    if DAF > 0:
        yv = bf16_round(acc)
        tl.store(y_ptr + p[:, None] * H + hh[None, :], yv, mask=mm[:, None])
        _arrive(tok, mm, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe, shared_rows,
                H, TOPK, NT2, BM, BH, CB, CBR, DAF=DAF, own=yv, kself=p % TOPK, HAS_OWN=True, pbits=pbits)
    else:
        tl.store(y_ptr + p[:, None] * H + hh[None, :], bf16_round(acc), mask=mm[:, None])
        _arrive(tok, mm, h, red_ptr, need_ptr, ploc_ptr, pw_ptr, y_ptr, out_ptr, ps, pe, shared_rows,
                H, TOPK, NT2, BM, BH, CB, CBR)


@triton.jit
def _ev_advance(ev_base, ev_ptr, ev_seq, EV_RING: tl.constexpr):
    # (development events) the next call's ring slot, then the call count: every CTA read
    # the count before it counted itself started
    ev_next = ev_base + ((ev_seq + 1) % EV_RING) * 32
    w32 = tl.arange(0, 32)
    big = tl.full((32,), 1 << 62, dtype=tl.int64)
    init = tl.where((w32 == 0) | (w32 == 3) | (w32 == 6), big,
                    tl.where((w32 >= 10) | (w32 == 1) | (w32 == 2), -1, 0).to(tl.int64))
    tl.store(ev_next + w32, init)
    tl.store(ev_base + EV_RING * 32, ev_seq + 1)
    tl.atomic_max(ev_ptr + 20, tl.extra.cuda.globaltimer(), sem="relaxed")


# Avoid a separate alignment-specialized binary for row counts divisible by 16:
# token-list and shared-row strides must use the same scheduling rules.
@triton.jit(do_not_specialize_on_alignment=["T", "T_ALL"])
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
    EV_RING: tl.constexpr = 16, PEEK: tl.constexpr = False, CB: tl.constexpr = False,
    BIN: tl.constexpr = 0, BIN_SLACK: tl.constexpr = 0, SHWAIT: tl.constexpr = False,
    SHDW: tl.constexpr = True, SIGFIRST: tl.constexpr = False, GCG: tl.constexpr = False,
    VSPRE: tl.constexpr = 0, VSHW: tl.constexpr = 0, ONEB: tl.constexpr = False,
    VBIN: tl.constexpr = 0, VBIN_MIN: tl.constexpr = 0, VBIN_MAX: tl.constexpr = 0,
    VBIN_ST: tl.constexpr = 4, LRZ: tl.constexpr = False,
    VMIX_: tl.constexpr = 0, VMIX_MIN: tl.constexpr = 0, VMIX_MAX: tl.constexpr = 0,
    NIG: tl.constexpr = False, DPFT: tl.constexpr = 0, DPFT_N: tl.constexpr = 1,
    ONI: tl.constexpr = False, PSW: tl.constexpr = False, DPFT_PSW: tl.constexpr = False,
    PSW_V: tl.constexpr = 0,
    DST: tl.constexpr = 0,  # the routed down tile loop's own pipeline depth (0: the kernel's)
    CBR: tl.constexpr = False,  # the routed combine's slot addresses formed in the tile's layout
    SDPF: tl.constexpr = False,  # SPRE's early shared down tile issued as an L2 prefetch
    SDPFW: tl.constexpr = False,  # with SHDW: the shared down tile into L2 before its wait
    PSW_NS: tl.constexpr = False,  # PSW's static path without its swap (see the item loop)
    XB1: tl.constexpr = False,  # routed gate/up items: x as the one token's row, broadcast
    DAF: tl.constexpr = 0,  # the combine's slots as data and flag (see _arrive; router_kernel's YN)
):
    VMIX: tl.constexpr = VMIX_ if (VBIN > 0 and BIN == 0 and 2 * VBIN == BI) else 0
    # (DPFT) a routed down tile's weights into L2 once DPFT_N of its expert's gate/up items are
    # done, in chunks of at most DPFT valid rows (the device's count); see the down item
    DPFTE: tl.constexpr = DPFT > 0 and VMIX == 0 and not PRE2 and not INTERPRET
    # (PSW) at the plan, when the routed gate/up items overflow the first wave by ov items, 0 < ov
    # <= NS2, the first ov static shared down CTAs take the overflowing gate/up items and the first
    # ov claims take their shared down items (see the item loop); with DPFT_PSW, DPFT also applies
    # then; with PSW_V > 0, only in chunks of at most PSW_V valid rows (the device's count)
    PSWE: tl.constexpr = PSW and BIN == 0 and VBIN == 0
    if EV > 0:
        t_start = tl.extra.cuda.globaltimer()
        # development events: a ring of EV_RING slots of 32 words, one per call (the call
        # count after them), read first
        ev_seq = tl.load(ev_ptr + EV_RING * 32, volatile=True)
        ev_base = ev_ptr
        ev_ptr = ev_ptr + (ev_seq % EV_RING) * 32
    v_raw = tl.load(valid_ptr)
    item = tl.program_id(0)
    lane = tl.arange(0, 128)
    # Every CTA waits for router_kernel (griddepcontrol.wait: that grid complete, its partial
    # logits and its clearing of this call's counted and polled state visible) before its
    # first access to that state; weight loads that do not depend on it go first.
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

    # routed gate/up items: BI columns, or BIN (narrower, more items) when the plan shows that
    # every one of them still gets a CTA of the resident grid (see _routed_items)
    NT1: tl.constexpr = I // BI
    NT2: tl.constexpr = H // BH
    NS1: tl.constexpr = SI // BI
    # shared down tiles: BHS_ rows, RS of them per combine tile of BH rows
    BHS_: tl.constexpr = BH if BHS == 0 else BHS
    RS: tl.constexpr = BH // BHS_
    NS2: tl.constexpr = H // BHS_
    NK = (T + BT_K - 1) // BT_K
    s_sh1 = NK
    s_sh2 = s_sh1 + NS1
    s_dyn = s_sh2 + NS2
    shared_rows = T * TOPK  # act/y rows of the shared expert start here

    U = -1  # touched experts, once this CTA has seen the plan header
    acq = 0  # whether this CTA has acquired the plan header (top-k's data visible)
    if EV > 0:
        # development events (int64 globaltimer): 0 first CTA start, 2 last top-k done,
        # 3 first routed g1 ready, 4 last sh-g1 done, 5 last g1 done, 6 first g2 ready,
        # 7 last CTA end, 8 last sh-g2 done, 9 last router item done, 10-15 top-k steps,
        # 16 last g2 ready, 17 last g2 done, 18 last combine, 19 last g2 fetched,
        # 20 other set cleared
        tl.atomic_min(ev_ptr + 0, t_start, sem="relaxed")
    running = 1
    stat = 1  # (PSW) the static item not yet looked at
    while running != 0:
        # an item's completion counter is released after the next item is grabbed, so the
        # release's wait for this item's stores overlaps the grab
        sig = 0
        sig_ptr = ctl_ptr + C_SH1  # (signalled only where set below)
        # (PEEK) the queue head as read by one lane while a down item runs (-1: not read);
        # reduced across the CTA only after the item, so every warp decides alike
        headv = tl.full((128,), -1, dtype=tl.int32)
        if PSWE:
            if stat != 0:
                stat = 0
                if (item >= s_sh2) & (item < s_dyn) & ((PSW_V == 0) | (valid <= PSW_V)):
                    # (PSW) a static shared down item: its tile into L2 now (whoever runs the
                    # item loads it from there; a prefetch returns nothing), then the plan
                    # decides who runs it. ov = the routed gate/up items beyond the first wave
                    # (0 unless every shared item is in the static wave, NUM_CTAS >= s_dyn):
                    # if 0 < ov <= NS2 this CTA, the h-th, h < ov, takes routed item NUM_CTAS
                    # + h, the h-th claim's; the first ov claims take the shared down items
                    # (below). The same items, permuted: the overflowing gate/up items start at
                    # the plan instead of as CTAs free up, and the shared down items, which
                    # wait for the shared gate/up items anyway, go to the earliest claimers
                    hsw = item - s_sh2
                    if pe > ps:
                        dlw = tl.arange(0, BHS_ * SI * 2 // 128)
                        tl.inline_asm_elementwise(
                            "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l",
                            [sd_ptr + hsw * (BHS_ * SI) + dlw * 64], dtype=tl.int32,
                            is_pure=False, pack=1)
                    _grid_wait()
                    if U < 0:
                        U = _plan_u(ctl_ptr, SLEEP)
                    ovs = tl.where(NUM_CTAS >= s_dyn, s_dyn + U * NT1 - NUM_CTAS, 0)
                    if PSW_NS:
                        # (PSW_NS) the static path's prefetch and waits, never the swap
                        ovs = ovs * 0
                    if (ovs > 0) & (ovs <= NS2) & (hsw < ovs):
                        item = NUM_CTAS + hsw
        if item < s_sh1:
            # top-k block: the router grid complete (partial logits, cleared state visible)
            _grid_wait()
            if ONEB:
                # (ONEB) every valid row in block 0 (the device's valid count): block 0 routes
                # them alone with the plan in registers; the other blocks (masked rows only)
                # store their zeros and arrive, and without block 0 the count never reaches
                # the plan's turn
                one_blk = ((valid <= BT_K) & (item == 0)) | (NK == 1)
            else:
                one_blk = NK == 1
            if one_blk:
                if ONI:
                    _route_one_block_ni(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr,
                                        need_ptr, out_ptr, touched_ptr, ctl_ptr, valid, T, ps, pe,
                                        H, E, E_LOC, TOPK, KS, BE_R, BT_K, BT_K, ev_ptr, EV, RS)
                else:
                    _route_one_block(plog_ptr, gam_ptr, map_ptr, rec_ptr, ploc_ptr, pw_ptr,
                                     need_ptr, out_ptr, touched_ptr, ctl_ptr, valid, T, ps, pe,
                                     H, E, E_LOC, TOPK, KS, BE_R, BT_K, BT_K, ev_ptr, EV, RS)
            else:
                if EV > 0:
                    # multi-block steps: 10 first block past its wait, 11 last block's top-k and
                    # list appends done, 12 last arrival, 14 plan lines stored, 15 published
                    tl.atomic_min(ev_ptr + 10, tl.extra.cuda.globaltimer(), sem="relaxed")
                _topk_item(item, plog_ptr, gam_ptr, map_ptr, rec_ptr, cnt_ptr, lists_ptr,
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
            j = item - s_sh1
            if SHWAIT:
                # its loads only once the router grid is done (see _config)
                _grid_wait()
            elif VSHW > 0:
                # (VSHW) the same when the chunk has at most VSHW valid rows (the device's count)
                if valid <= VSHW:
                    _grid_wait()
            for m0 in range(ps, pe, BMS):
                rr = m0 + tl.arange(0, BMS)
                mm = rr < pe
                _gate_up_tile(x_ptr, sgu_ptr, act_ptr, rr, shared_rows + rr, mm, j,
                              H, SI, IA, BMS, BI, BK1, F32,
                              GCG, H + (NUM_CTAS >> 30) * BK1, SHARED=True)
            if EV > 0:
                tl.atomic_max(ev_ptr + 4, tl.extra.cuda.globaltimer(), sem="relaxed")
            sig = 1
            sig_ptr = ctl_ptr + C_SH1
        elif item < s_dyn:
            # shared down tile h (BHS_ hidden rows), gated by sigmoid(s)
            h = item - s_sh2
            if SDPFW and (SHWAIT or VSPRE > 0) and SHDW and not INTERPRET:
                # (SDPFW) the tile into L2 now, while the router's grid runs and DRAM carries
                # little else; its load below still waits for that grid
                if pe > ps:
                    dlw2 = tl.arange(0, BHS_ * SI * 2 // 128)
                    tl.inline_asm_elementwise(
                        "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l",
                        [sd_ptr + h * (BHS_ * SI) + dlw2 * 64], dtype=tl.int32,
                        is_pure=False, pack=1)
            if (SHWAIT or VSPRE > 0) and SHDW:
                # (SHDW: this item's loads too; without it, with SPRE, its tile is issued at
                # once and only its use waits)
                _grid_wait()
            if PRE2 | SPRE:
                # the shared down tile streams during the routing chain, not after the gate/up
                # tiles
                ks2 = tl.arange(0, SI)
                hs = h * BHS_ + tl.arange(0, BHS_)
                if VSPRE > 0:
                    # (VSPRE) only when the chunk has at most VSPRE valid rows (the device's
                    # count): with more, the same tile is loaded after the waits below
                    wspre = tl.load(sd_ptr + hs[None, :] * SI + ks2[:, None],
                                    mask=(pe > ps) & (valid <= VSPRE), other=0.0)
                elif SDPF:
                    # (SDPF) the tile into L2 now and into registers after the router grid: a
                    # prefetch holds no registers or load slots on this SM, so the router's
                    # reads beside it finish sooner
                    if (pe > ps) and not INTERPRET:
                        dlf = tl.arange(0, BHS_ * SI * 2 // 128)
                        tl.inline_asm_elementwise(
                            "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l",
                            [sd_ptr + h * (BHS_ * SI) + dlf * 64], dtype=tl.int32,
                            is_pure=False, pack=1)
                else:
                    wspre = tl.load(sd_ptr + hs[None, :] * SI + ks2[:, None], mask=(pe > ps),
                                    other=0.0)
            _grid_wait()
            if SDPF and (PRE2 | SPRE) and (VSPRE == 0):
                wspre = tl.load(sd_ptr + hs[None, :] * SI + ks2[:, None], mask=(pe > ps),
                                other=0.0)
            _wait_geq(ctl_ptr + C_SH1, NS1, SLEEP)
            if (VSPRE > 0) and (PRE2 | SPRE):
                if valid > VSPRE:
                    wspre = tl.load(sd_ptr + hs[None, :] * SI + ks2[:, None], mask=(pe > ps),
                                    other=0.0)
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
                            pe, shared_rows, H, TOPK, NT2, BMS, BH, CB, DAF=DAF)
            if EV > 0:
                tl.atomic_max(ev_ptr + 8, tl.extra.cuda.globaltimer(), sem="relaxed")
        else:
            if U < 0:
                _grid_wait()
                U = _plan_u(ctl_ptr, SLEEP)
            if BIN > 0:
                nt1 = _routed_items(U, NUM_CTAS, NT1, NS1, NS2, I, BIN, BIN_SLACK)
            elif VMIX > 0:
                # (VMIX) in a chunk of VMIX_MIN..VMIX_MAX valid rows (the device's count), the first un
                # touched experts' gate/up items are VBIN (= BI / 2) columns wide (VBIN_ST
                # stages), the others BI: the narrow experts finish first and their down tiles
                # stream while the wide ones finish (with VBIN's range: every expert narrow)
                # (one loop-invariant value, the narrow experts in quarters of U: 4 in VBIN's
                # range, VMIX in VMIX's, else 0)
                un = (U * _vmix_q(valid, VBIN_MIN, VBIN_MAX, VMIX, VMIX_MIN, VMIX_MAX)) // 4
                nt1 = NT1
            elif VBIN > 0:
                # (VBIN) VBIN-column items in a chunk of VBIN_MIN..VBIN_MAX valid rows (the
                # device's count: every CTA reads the same count, so all agree)
                nt1 = tl.where((valid >= VBIN_MIN) & (valid <= VBIN_MAX), I // VBIN, NT1)
            else:
                nt1 = NT1
            if VMIX > 0:
                gend = (U + un) * NT1
            else:
                gend = 0  # (unused)
            d = item - s_dyn
            if VBIN > 0 and d < (gend if VMIX > 0 else U * nt1):
                # (VBIN) one item body per width; the narrow items keep VBIN_ST pipeline stages
                if VMIX > 0:
                    # (VMIX) the first un experts' items narrow, the others wide
                    if d < 2 * un * NT1:
                        if NIG:
                            u, acq = _routed_gate_up_ni(d, x_ptr, wgu_ptr, act_ptr, lists_ptr,
                                                     touched_ptr, ctl_ptr, T, acq, H, I, IA, TOPK,
                                                     BM, VBIN, BK1, F32, False, H, VBIN_ST, SLEEP,
                                                     LRZ)
                        else:
                            u, acq = _routed_gate_up(d, x_ptr, wgu_ptr, act_ptr, lists_ptr,
                                                     touched_ptr, ctl_ptr, T, acq, H, I, IA, TOPK,
                                                     BM, VBIN, BK1, F32, False, H, VBIN_ST, SLEEP,
                                                     LRZ)
                    else:
                        if NIG:
                            u, acq = _routed_gate_up_ni(d - un * NT1, x_ptr,
                                                     wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                     ctl_ptr, T, acq, H, I, IA, TOPK, BM, BI, BK1,
                                                     F32, GCG, H + (NUM_CTAS >> 30) * BK1, 0, SLEEP,
                                                     LRZ)
                        else:
                            u, acq = _routed_gate_up(d - un * NT1, x_ptr,
                                                     wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                     ctl_ptr, T, acq, H, I, IA, TOPK, BM, BI, BK1,
                                                     F32, GCG, H + (NUM_CTAS >> 30) * BK1, 0, SLEEP,
                                                     LRZ)
                elif nt1 != NT1:
                    if NIG:
                        u, acq = _routed_gate_up_ni(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                 ctl_ptr, T, acq, H, I, IA, TOPK, BM, VBIN, BK1, F32,
                                                 False, H, VBIN_ST, SLEEP, LRZ)
                    else:
                        u, acq = _routed_gate_up(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                 ctl_ptr, T, acq, H, I, IA, TOPK, BM, VBIN, BK1, F32,
                                                 False, H, VBIN_ST, SLEEP, LRZ)
                else:
                    if NIG:
                        u, acq = _routed_gate_up_ni(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                 ctl_ptr, T, acq, H, I, IA, TOPK, BM, BI, BK1, F32,
                                                 GCG, H + (NUM_CTAS >> 30) * BK1, 0, SLEEP, LRZ)
                    else:
                        u, acq = _routed_gate_up(d, x_ptr, wgu_ptr, act_ptr, lists_ptr, touched_ptr,
                                                 ctl_ptr, T, acq, H, I, IA, TOPK, BM, BI, BK1, F32,
                                                 GCG, H + (NUM_CTAS >> 30) * BK1, 0, SLEEP, LRZ)
                sig = 1
                sig_ptr = g1d_ptr + u * G1D_STRIDE
            elif VMIX == 0 and d < U * nt1:
                # routed gate/up tile j of touched expert u (VMIX: every gate/up item is above, a
                # range the compiler cannot match with this one, so this body is cut statically)
                u = d // nt1
                j = d % nt1
                e, n, first = _line(touched_ptr, u, BM, SLEEP)
                if EV > 0:
                    tl.atomic_min(ev_ptr + 3, tl.extra.cuda.globaltimer(), sem="relaxed")
                wbase = wgu_ptr + e.to(tl.int64) * (2 * I * H)
                if n > BM:
                    # the rest of the token list was written by other top-k CTAs
                    if acq == 0:
                        _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                        acq = 1
                for m0 in range(0, n, BM):
                    mi = m0 + tl.arange(0, BM)
                    mm = mi < n
                    if m0 == 0:
                        p = first
                    else:
                        p = tl.load(lists_ptr + e * T + mi, mask=mm, other=0, cache_modifier=".cg")
                    if BIN > 0:
                        if nt1 != NT1:
                            _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j,
                                          H, I, IA, BM, BIN, BK1, F32,
                              GCG, H + (NUM_CTAS >> 30) * BK1)
                        else:
                            _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j,
                                          H, I, IA, BM, BI, BK1, F32,
                              GCG, H + (NUM_CTAS >> 30) * BK1)
                    else:
                        _gate_up_tile(x_ptr, wbase, act_ptr, p // TOPK, p, mm, j,
                                      H, I, IA, BM, BI, BK1, F32,
                              GCG, H + (NUM_CTAS >> 30) * BK1, XB=XB1)
                if EV > 0:
                    tl.atomic_max(ev_ptr + 5, tl.extra.cuda.globaltimer(), sem="relaxed")
                sig = 1
                sig_ptr = g1d_ptr + u * G1D_STRIDE
            elif d < (gend + U * NT2 if VMIX > 0 else U * (nt1 + NT2)):
                # routed down tile h of touched expert u, once its gate/up tiles are done
                d2 = d - (gend if VMIX > 0 else U * nt1)
                u = d2 // NT2
                h = d2 % NT2
                if VMIX > 0:
                    nt1 = NT1 + (u < un).to(tl.int32) * NT1
                if EV > 0:
                    tl.atomic_max(ev_ptr + 19, tl.extra.cuda.globaltimer(), sem="relaxed")
                e, n, first = _line(touched_ptr, u, BM, SLEEP)
                wbase = wd_ptr + e.to(tl.int64) * (H * I)
                if PRE2:
                    # the first reads of the plan header and of the expert's gate/up count go
                    # out before the tile's loads and are looked at after them: the tile is
                    # moved into shared memory (its operand layout) as soon as it lands, before
                    # anything after it runs, so an item claimed once its expert is done gets
                    # both answers within the tile's own latency instead of after it. Lanes
                    # other than 0 read nothing and get the smallest value, so the reductions
                    # below return lane 0's read (as in _poll)
                    pa_first = tl.load(ctl_ptr + C_PLANA + lane * 0, mask=(lane == 0) & (acq == 0),
                                       other=-2147483647, volatile=True)
                    g1_first = tl.load(g1d_ptr + u * G1D_STRIDE + lane * 0, mask=lane == 0,
                                       other=-2147483647, volatile=True)
                    # the down tile's weights are issued before the wait on the expert's
                    # gate/up tiles, so they stream while those finish
                    kk2 = tl.arange(0, I)
                    wpre = tl.load(wbase + (h * BH + tl.arange(0, BH))[None, :] * I + kk2[:, None])
                    if acq == 0:
                        if tl.max(pa_first, axis=0) < 1:
                            _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                        acq = 1
                    if tl.max(g1_first, axis=0) < nt1:
                        _wait_geq(g1d_ptr + u * G1D_STRIDE, nt1, SLEEP)
                    else:
                        # (both counts seen by relaxed reads: one fence gives them acquire
                        # semantics, the barrier extends it to the CTA)
                        _fence_acquire()
                        tl.debug_barrier()
                else:
                    if acq == 0:
                        _wait_geq(ctl_ptr + C_PLANA, 1, SLEEP)
                        acq = 1
                    if DPFTE:
                        # (DPFT) the tile's weights (BH contiguous rows) into L2 once DPFT_N of the
                        # expert's gate/up items are done (relaxed polls: the wait below still
                        # acquires all of them), so they stream while its last items finish
                        # instead of after them (with every gate/up item in the first wave the
                        # experts all finish together and the bandwidth idles until the down
                        # tiles' loads land); a prefetch returns nothing, so it observes no
                        # store. Only when every routed gate/up item had a CTA of the first wave
                        # (the plan's count): otherwise the prefetches would compete with the
                        # gate/up items that start late, which every down item waits for
                        if DPFT_PSW:
                            # (DPFT_PSW) with PSW, an overflow of at most NS2 also gives every
                            # gate/up item a CTA from the plan on
                            dfit = s_dyn + U * nt1 <= NUM_CTAS + NS2
                        if not DPFT_PSW:
                            dfit = s_dyn + U * nt1 <= NUM_CTAS
                        if (valid <= DPFT) & dfit:
                            gv = _poll(g1d_ptr + u * G1D_STRIDE)
                            while gv < tl.minimum(nt1, DPFT_N):
                                _nap(SLEEP)
                                gv = _poll(g1d_ptr + u * G1D_STRIDE)
                            dl = tl.arange(0, BH * I * 2 // 128)
                            dq = wbase + h * (BH * I) + dl * 64
                            tl.inline_asm_elementwise(
                                "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l", [dq],
                                dtype=tl.int32, is_pure=False, pack=1)
                    _wait_geq(g1d_ptr + u * G1D_STRIDE, nt1, SLEEP)
                if PEEK:
                    # the queue head, read (not claimed) while this item runs and looked at
                    # only after it: once every item is claimed (the head only grows) this
                    # CTA leaves without the claim's round trip, which would otherwise end the
                    # kernel's last CTA
                    headv = tl.load(ctl_ptr + C_HEAD + lane * 0, mask=lane == 0, other=-1,
                                    volatile=True)
                if EV > 0:
                    tl.atomic_min(ev_ptr + 6, tl.extra.cuda.globaltimer(), sem="relaxed")
                    tl.atomic_max(ev_ptr + 16, tl.extra.cuda.globaltimer(), sem="relaxed")
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
                                  TOPK, NT2, BM, BH, F32, CB, DAF=DAF)
                    else:
                        _down_arrive(act_ptr, wbase, pw_ptr, y_ptr, p // TOPK, p, mm, h, red_ptr,
                                     need_ptr, ploc_ptr, out_ptr, ps, pe, shared_rows, H, I, IA,
                                     TOPK, NT2, BM, BH, BK2, F32, CB, DST=DST,
                                     CBR=CBR, DAF=DAF)
                if EV > 0:
                    tl.atomic_max(ev_ptr + 17, tl.extra.cuda.globaltimer(), sem="relaxed")
            else:
                running = 0
        # (the claim and the signal below touch the counted state: router_kernel's clearing
        # must be visible; after the first time this returns at once)
        _grid_wait()
        if PEEK:
            head = tl.max(headv, axis=0)
            if head >= 0:
                if BIN > 0:
                    nt1p = _routed_items(U, NUM_CTAS, NT1, NS1, NS2, I, BIN, BIN_SLACK)
                elif VBIN > 0:
                    nt1p = tl.where((valid >= VBIN_MIN) & (valid <= VBIN_MAX), I // VBIN, NT1)
                else:
                    nt1p = NT1
                if VMIX > 0:
                    unp = (U * _vmix_q(valid, VBIN_MIN, VBIN_MAX, VMIX, VMIX_MIN, VMIX_MAX)) // 4
                    endp = (U + unp) * NT1 + U * NT2
                    done_all = NUM_CTAS + head >= s_dyn + endp
                else:
                    done_all = NUM_CTAS + head >= s_dyn + U * (nt1p + NT2)
                if done_all:
                    running = 0
        if running != 0:
            if SIGFIRST:
                # the completion released before the claim: the release's fence would
                # otherwise also wait for the claim, which queues behind every other CTA's
                # claim when many items end together
                if sig != 0:
                    _signal(sig_ptr)
                if PSWE:
                    cj = tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
                    item = NUM_CTAS + cj
                    if (PSW_V == 0) | (valid <= PSW_V):
                        # (PSW) the first ov claims: the swapped shared down items
                        if U < 0:
                            U = _plan_u(ctl_ptr, SLEEP)
                        ovc = tl.where(NUM_CTAS >= s_dyn, s_dyn + U * NT1 - NUM_CTAS, 0)
                        if PSW_NS:
                            # (PSW_NS) the static path's prefetch and waits, never the swap
                            ovc = ovc * 0
                        item = tl.where((ovc > 0) & (ovc <= NS2) & (cj < ovc), s_sh2 + cj, item)
                else:
                    item = NUM_CTAS + tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
            elif PSWE:
                cj = tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
                if sig != 0:
                    _signal(sig_ptr)
                item = NUM_CTAS + cj
                if (PSW_V == 0) | (valid <= PSW_V):
                    # (PSW) the first ov claims: the swapped shared down items
                    if U < 0:
                        U = _plan_u(ctl_ptr, SLEEP)
                    ovc = tl.where(NUM_CTAS >= s_dyn, s_dyn + U * NT1 - NUM_CTAS, 0)
                    if PSW_NS:
                        # (PSW_NS) the static path's prefetch and waits, never the swap
                        ovc = ovc * 0
                    item = tl.where((ovc > 0) & (ovc <= NS2) & (cj < ovc), s_sh2 + cj, item)
            else:
                nxt = NUM_CTAS + tl.atomic_add(ctl_ptr + C_HEAD, 1, sem="relaxed", scope="gpu")
                if sig != 0:
                    _signal(sig_ptr)
                item = nxt

    if EV > 0:
        tl.atomic_max(ev_ptr + 7, tl.extra.cuda.globaltimer(), sem="relaxed")
        # (development events) CTA 0, the last to leave in practice, moves the ring on
        if tl.program_id(0) == 0:
            _ev_advance(ev_base, ev_ptr, ev_seq, EV_RING)


@triton.jit
def router_kernel(x_ptr: "*bf16", rw_ptr: "*bf16", sgw_ptr: "*bf16", valid_ptr: "*i32",
                  plog_ptr: "*fp32", zero_ptr: "*i32", n_zero, sent_ptr: "*i32", n_sent,
                  T, row0, T_ALL, lists_ptr: "*i32", n_local,
                  H: tl.constexpr, E: tl.constexpr, KS: tl.constexpr, BT_R: tl.constexpr,
                  BE_R: tl.constexpr, F32: tl.constexpr, PDL_IN: tl.constexpr = False,
                  PF: tl.constexpr = False, SKIP: tl.constexpr = False, LZ: tl.constexpr = 0,
                  YN: tl.constexpr = 0, YOFF: tl.constexpr = 0):
    """The router items of a chunk (token block x expert block x hidden slice: FP32 partial
    logits, plus the shared expert's gate logit), one per CTA, and the clearing of the call's
    counted and polled state before the main kernel's launch: n_zero words of zeros (control
    words, per-expert counts, gate/up completion counts, plan lines, arrival counters) and
    n_sent gamma sentinels (-1 bits). The main kernel is launched as this grid's
    programmatic dependent: it may start at once and waits for this grid (complete, stores
    visible) before its first access to that state (moe_block_kernel). With PDL_IN this grid
    too is a dependent of the stream's previous kernel and waits for it (complete, its stores
    visible) before anything, and only then lets the main kernel launch: the main kernel
    reads the call's inputs (x, the valid count, the part) without a wait of its own, so it
    must not start before the kernel that wrote them is done.

    With PF, the first token block's items first bring their router rows into L2 (one prefetch
    per 128-byte line), before that wait: a prefetch returns nothing, so it cannot observe a
    store of the kernel waited for (L2 is the point of coherence), and the rows' DRAM round
    trip overlaps that kernel's completion; the loads themselves stay after the wait.

    With SKIP (chunks of several token blocks), a token block past the first computes only if
    it holds a valid row (the device's count, read first by those CTAs; volatile, so it is not
    merged with the first block's read); the first block's copy of the item reads the count
    only after its loads, so they still go out at once. Every CTA still waits for the previous
    kernel, lets the main kernel launch and clears its share of the counted state."""
    NEB: tl.constexpr = E // BE_R
    if PF and not INTERPRET:
        pf_item = tl.program_id(0)
        if pf_item < (NEB + 1) * KS:
            BK: tl.constexpr = H // KS
            LPR: tl.constexpr = BK * 2 // 128  # 128-byte lines per row slice
            eb = pf_item // KS
            ks = pf_item % KS
            ln = tl.arange(0, BE_R * LPR)
            c = ks * BK + (ln % LPR) * (128 // 2)
            if eb < NEB:
                pp = rw_ptr + (eb * BE_R + ln // LPR) * H + c
            else:
                pp = sgw_ptr + c
            tl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l",
                                      [pp], dtype=tl.int32, is_pure=False, pack=1)
    if PDL_IN:
        _grid_wait()
    _launch_dependents()
    item = tl.program_id(0)
    if SKIP:
        if item < (NEB + 1) * KS:
            _router_item(item, x_ptr, rw_ptr, sgw_ptr, plog_ptr, 0, T, H, E, NEB, KS, BT_R,
                         BE_R, F32, True, valid_ptr, row0, T_ALL)
        else:
            v_raw = tl.load(valid_ptr, volatile=True)
            vr = tl.minimum(tl.maximum(tl.minimum(tl.maximum(v_raw, 0), T_ALL) - row0, 0), T)
            if (item // ((NEB + 1) * KS)) * BT_R < vr:
                _router_item(item, x_ptr, rw_ptr, sgw_ptr, plog_ptr, vr, T, H, E, NEB, KS,
                             BT_R, BE_R, F32)
    else:
        v_raw = tl.load(valid_ptr)
        vr = tl.minimum(tl.maximum(tl.minimum(tl.maximum(v_raw, 0), T_ALL) - row0, 0), T)
        _router_item(item, x_ptr, rw_ptr, sgw_ptr, plog_ptr, vr, T, H, E, NEB, KS, BT_R, BE_R,
                     F32)
    if YN > 0:
        # DAF slot sentinels are established before the dependent grid can read them.
        oy2 = tl.arange(0, 512)
        for i in range(item * 512, YN, tl.num_programs(0) * 512):
            tl.store(sent_ptr + YOFF + i + oy2, tl.full((512,), -1, dtype=tl.int32), mask=i + oy2 < YN)
    o = tl.arange(0, 512)
    for i in range(item * 512, n_zero, tl.num_programs(0) * 512):
        tl.store(zero_ptr + i + o, tl.zeros((512,), dtype=tl.int32), mask=i + o < n_zero)
    for i in range(item * 512, n_sent, tl.num_programs(0) * 512):
        tl.store(sent_ptr + i + o, tl.full((512,), -1, dtype=tl.int32), mask=i + o < n_sent)
    if LZ > 0:
        # (LZ, a chunk of more than one top-k block) the first min(T, LZ) entries of every local
        # expert's token list cleared: _plan loads them with the counts, before it knows the
        # counts (the entries past a count are masked out of what it stores), so without this
        # that load reads list entries this call never wrote. LZ is a power of two: entry q is
        # expert q // LZ, position q % LZ
        n_l = n_local * LZ
        for i in range(item * 512, n_l, tl.num_programs(0) * 512):
            q = i + o
            tl.store(lists_ptr + (q // LZ) * T + q % LZ, tl.zeros((512,), dtype=tl.int32),
                     mask=(q < n_l) & (q % LZ < T))
