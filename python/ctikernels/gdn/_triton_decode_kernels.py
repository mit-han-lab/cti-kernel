# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own gdn-decode Gluon tree d9ce6c3f (``gdn_decode.py``). Its design learned
# from SGLang, vLLM and FlashInfer (see the docstring); no code is copied from them.

"""The fused Gated DeltaNet decode step: one Gluon kernel per call (``_triton_decode`` picks
its configuration and launches it).

Everything between ``in_proj`` and ``out_proj`` for one token of each of B sequences: the
causal conv update (width 4) with SiLU over q | k | v, the q/k L2 norms, the gates, the delta
rule on the FP32 state pool (updated in place) and the gated RMSNorm. Inputs follow the
contract of :mod:`ctikernels.gdn.inference` (contiguous pools and weights, unit column stride
of ``projected``), so every stride but ``projected``'s row stride derives from the geometry.

**Work split.** One state CTA per (sequence n, value head h, tile of BV of the head's 128
state rows), NV = 128 / BV tiles per head, program id ``(n * Hv + h) * NV + tile``. A state
CTA reads its slot index, then its state rows and conv history once, and writes them once.
Padding (slot -1) masks every pool access, writes y = 0 and takes no part in the cross-CTA
exchange. The state tile is ``BlockedLayout([1, 4], [rows, 32 / rows], [W, 1])``: a row's
dot products are 32 products per thread and two shuffle levels. A3 splits a head in three
(rows 0-63 as a 64-row tile, 64-95 and 96-127 as 32-row tiles, program id
``(n * Hv + h) * 3 + j``).

**Arithmetic** (FP32, no FP64, no flush to zero on the data path): the conv's four tap
products are BF16 x BF16, exact in FP32, summed error-free (TwoSum) and rounded once; SiLU
and sigmoid from e^-|x|; softplus as ``max(x, 0) + log1p(e^-|x|)``; q and k stay
unnormalised until the recurrence (their norms and q.k come from one reduction and are
applied as scalars); the recurrence takes one pass over the state tile: ``sk = S k``,
``sq = S q``, ``d = beta (v - alpha sk)``, ``o = alpha sq + d (k.q)``,
``S' = alpha S + d k^T`` (algebraically ``o = S'^T q``).

**Exchange.** A head split over NV CTAs needs the sum of o^2 over its 128 rows for the
RMSNorm, and the key head's q | k history may be shifted only once every CTA of the key head
(a group of R * NV CTAs, R = Hv / Hk) has read it. Two forms, chosen per configuration:

- *Finishers* (FIN): one extra CTA per (sequence, key head), after every state CTA in program
  order. State CTAs publish their rows' o (and, with WSUM, each warp's sum of o^2) to the o
  scratch and store their state rows without waiting; the finisher polls the key head's words,
  computes y of its R value heads and shifts the q | k history. Used while the whole grid is
  resident at once (``_triton_decode`` guarantees it), so no CTA waits for one not running.
- *Flags*: every CTA publishes its tile's partial sum of o^2 and polls its value head's NV
  words; the key head's last CTA in program order polls all R * NV words and then shifts the
  q | k history. A CTA only waits for CTAs of its own group, which are adjacent program ids
  dispatched in order, so a resident prefix of the grid always completes.

Scratch words are 64-bit ``[tag : 32 | value : 32]``, polled with relaxed GPU-scope loads;
data and tag are one aligned word, so no fence is needed. Two tagging protocols:

- *Epochs* (SC off): each set of words carries a counter; a call reads it as c, tags its words
  c + 1 and the set's last reader stores c + 1 into the counter. Every word of a set always
  carries its counter's value before a call, so the tag a call waits for is never already
  there, however the counter wraps.
- *Self-cleaning* (SC): the tag is the constant 1, each word has exactly one reader, which
  stores 0 once its wait is over; the scratch is all-zero before and after every call. Flag
  scratch holds one word per (source tile, destination tile) of a value head. ROWIDX indexes a
  graph-private scratch by batch row instead of pool slot.

Scratch is indexed by pool slot, so calls that cannot touch the same pool rows (the contract's
unique slots) never share a word.

**Launch.** On compute capability 9.0 and up (PDL), programmatic dependent launch:
``griddepcontrol.wait`` is the first statement, so no global access precedes the previous
kernel's completion; below 9.0 an ordinary launch.

The design learned from SGLang's and vLLM's packed decode recurrence (V tiles of the state
per program) and FlashInfer's pretranspose decode (the [V, K] pool layout); the fusion, the
exchange, the exact conv and the one-pass recurrence are cti's own.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra import libdevice

K = V = 128
# Words per (slot, value head) of the o scratch: 128 o words, then 16 warp sums of o^2.
O_WORDS = K + 16


def flag_words(ratio: int) -> int:
    """Epoch flag words per (slot, key head): one set (its counter, then ratio * NV words) per
    NV = 1, 2, 4."""
    return 7 * ratio + 3


def counter_words(ratio: int) -> int:
    """Epoch counter words per (slot, key head) of finisher calls: the o set's counter at
    ``ratio * 4``."""
    return 4 * ratio + 1


def sc_flag_words(ratio: int) -> int:
    """Self-cleaning flag words per (slot, key head): one per (value head, source tile,
    destination tile), ``ratio * NV^2`` of them used."""
    return 16 * ratio


@gluon.jit
def _gdc_wait():
    gl.inline_asm_elementwise(
        "griddepcontrol.wait; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _gdc_launch_dependents():
    gl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _sigmoid(x):
    # e^-|x| never overflows: 1 / (1 + e^-x) for x >= 0, e^x / (1 + e^x) below.
    e = gl.exp(-gl.abs(x))
    r = 1.0 / (1.0 + e)
    return gl.where(x >= 0, r, e * r)


@gluon.jit
def _bf16_bits_to_f32(bits):
    return ((bits & 0xFFFF) << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@gluon.jit
def _add3(a0, a1, a2, b0, b1, b2):
    return a0 + b0, a1 + b1, a2 + b2


@gluon.jit
def _word(tag, value):
    """The scratch word ``[tag : 32 | bits of value : 32]`` (``tag`` already shifted)."""
    return tag | (value.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF)


@gluon.jit
def _low(words):
    return (words & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True)


@gluon.jit
def _high(words):
    return (words >> 32) & 0xFFFFFFFF


@gluon.jit
def _poll(ptrs, mask):
    # Relaxed GPU-scope loads of scratch words; masked lanes read 0.
    return gl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; mov.b64 $0, 0; "
        "@p ld.relaxed.gpu.global.b64 $0, [$1]; }",
        "=l,l,r",
        [ptrs, mask.to(gl.int32)],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _mask64(mask):
    return gl.inline_asm_elementwise(
        "cvt.u64.u32 $0, $1;", "=l,r", [mask.to(gl.int32)], dtype=gl.int64, is_pure=False, pack=1
    )


@gluon.jit
def _poll_z(ptrs, mask, z):
    # _poll with the masked lanes' 0 taken from z (made ahead of earlier stores, PWZ).
    return gl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; mov.b64 $0, $3; "
        "@p ld.relaxed.gpu.global.b64 $0, [$1]; }",
        "=l,l,r,l",
        [ptrs, mask.to(gl.int32), z],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


def _wait_asm(pack: int, sums: bool = False, butterfly: bool = False) -> str:
    """PTX of the finisher's block-cooperative wait for ``pack`` o words per thread: load,
    compare each word's tag with the thread's expected tag, OR the CTA's pending predicates with
    one ``bar.red`` (named barrier 1; Triton's barriers use 0), loop while any is pending.

    With ``sums`` each thread also polls ``pack`` warp-sum words, then gathers its value head's
    16 warp sums of o^2 from lanes 0..15 and adds them in a fixed order (every thread gets the
    same bits); each output element is ``[sum : 32 | o : 32]``. pack 2: two value heads, the
    elements are the heads; pack 4: one warp per value head, only the first element carries
    sum pointers (the others read 0). ``butterfly`` (pack 2) adds the 16 sums by a xor
    butterfly over 1..16 with the heads interleaved instead of 16 sequential gathers."""
    o = lambda i: f"${pack + i}"
    om = lambda i: f"${2 * pack + i}"
    if not sums:
        tag = lambda i: f"${3 * pack + i}"
        lines = ["{ .reg .pred p, q, r, " + ", ".join(f"m{i}" for i in range(pack)) + ";"]
        lines.append(".reg .b32 lo, hi;")
        for i in range(pack):
            lines.append(f"setp.ne.u32 m{i}, {om(i)}, 0; mov.b64 ${i}, 0;")
        lines.append("L${:uid}:")
        for i in range(pack):
            lines.append(f"@m{i} ld.relaxed.gpu.global.b64 ${i}, [{o(i)}];")
        for i in range(pack):
            test = "q" if i == 0 else "r"
            lines.append(f"mov.b64 {{lo, hi}}, ${i}; setp.ne.and.u32 {test}, hi, {tag(i)}, m{i};")
            if i:
                lines.append("or.pred q, q, r;")
        lines.append("bar.red.or.pred p, 1, q; @p bra L${:uid}; }")
        return " ".join(lines)
    pp = lambda i: f"${3 * pack + i}"
    pm = lambda i: f"${4 * pack + i}"
    tag = lambda i: f"${5 * pack + i}"
    regs = ", ".join(f"m{i}, n{i}" for i in range(pack))
    words = ", ".join(f"w{i}, v{i}" for i in range(pack))
    extra = ", t0, t1" if butterfly else ""
    lines = [
        "{ .reg .pred p, q, r, " + regs + "; .reg .b32 lo, hi, t, x, s0, s1, s2, s3" + extra + ";",
        ".reg .b64 " + words + ";",
    ]
    for i in range(pack):
        lines.append(
            f"setp.ne.u32 m{i}, {om(i)}, 0; setp.ne.u32 n{i}, {pm(i)}, 0; "
            f"mov.b64 w{i}, 0; mov.b64 v{i}, 0;"
        )
    lines.append("L${:uid}:")
    for i in range(pack):
        lines.append(
            f"@m{i} ld.relaxed.gpu.global.b64 w{i}, [{o(i)}]; "
            f"@n{i} ld.relaxed.gpu.global.b64 v{i}, [{pp(i)}];"
        )
    for i in range(pack):
        if i == 0:
            lines.append(f"mov.b64 {{lo, hi}}, w0; setp.ne.and.u32 q, hi, {tag(0)}, m0;")
        else:
            lines.append(
                f"mov.b64 {{lo, hi}}, w{i}; setp.ne.and.u32 r, hi, {tag(i)}, m{i}; or.pred q, q, r;"
            )
        lines.append(
            f"mov.b64 {{lo, hi}}, v{i}; setp.ne.and.u32 r, hi, {tag(i)}, n{i}; or.pred q, q, r;"
        )
    lines.append("bar.red.or.pred p, 1, q; @p bra L${:uid};")
    heads = range(pack) if pack == 2 else [0]
    if pack != 2:
        lines.append("mov.b64 {lo, hi}, v0; mov.b32 x, lo;")
        for i in range(1, pack):
            lines.append(f"mov.b64 {{lo, hi}}, v{i}; add.f32 x, x, lo;")
    if butterfly:
        for h in heads:
            lines.append(f"mov.b64 {{lo, hi}}, v{h}; mov.b32 s{h}, lo;")
        for d in (1, 2, 4, 8, 16):
            for h in heads:
                lines.append(
                    f"shfl.sync.bfly.b32 t{h}, s{h}, {d}, 31, -1; add.f32 s{h}, s{h}, t{h};"
                )
    else:
        for h in heads:
            if pack == 2:
                lines.append(f"mov.b64 {{lo, hi}}, v{h}; mov.b32 x, lo;")
            lines.append(f"shfl.sync.idx.b32 s{h}, x, 0, 31, -1;")
            for t in range(1, 16):
                lines.append(f"shfl.sync.idx.b32 t, x, {t}, 31, -1; add.f32 s{h}, s{h}, t;")
    for i in range(pack):
        lines.append(f"mov.b64 {{lo, hi}}, w{i}; mov.b64 ${i}, {{lo, s{i if pack == 2 else 0}}};")
    lines.append("}")
    return " ".join(lines)


def _wait_constraints(pack: int, sums: bool = False) -> str:
    inputs = ["l", "r", "l", "r", "r"] if sums else ["l", "r", "r"]
    return ",".join(["=l"] * pack + [c for c in inputs for _ in range(pack)])


_WAIT1 = gl.constexpr(_wait_asm(1))
_WAIT2 = gl.constexpr(_wait_asm(2))
_WAIT4 = gl.constexpr(_wait_asm(4))
_WAIT1_C = gl.constexpr(_wait_constraints(1))
_WAIT2_C = gl.constexpr(_wait_constraints(2))
_WAIT4_C = gl.constexpr(_wait_constraints(4))
_SUMS2 = gl.constexpr(_wait_asm(2, sums=True))
_SUMS2_BFLY = gl.constexpr(_wait_asm(2, sums=True, butterfly=True))
_SUMS4 = gl.constexpr(_wait_asm(4, sums=True))
_SUMS2_C = gl.constexpr(_wait_constraints(2, sums=True))
_SUMS4_C = gl.constexpr(_wait_constraints(4, sums=True))


@gluon.jit
def _wait_words(ptrs, mask, tag, RP: gl.constexpr):
    if RP == 1:
        return gl.inline_asm_elementwise(
            _WAIT1, _WAIT1_C, [ptrs, mask.to(gl.int32), tag], dtype=gl.int64, is_pure=False, pack=1
        )
    elif RP == 2:
        return gl.inline_asm_elementwise(
            _WAIT2, _WAIT2_C, [ptrs, mask.to(gl.int32), tag], dtype=gl.int64, is_pure=False, pack=2
        )
    else:
        return gl.inline_asm_elementwise(
            _WAIT4, _WAIT4_C, [ptrs, mask.to(gl.int32), tag], dtype=gl.int64, is_pure=False, pack=4
        )


@gluon.jit
def _wait_words_sums(optrs, omask, pptrs, pmask, tag, RP: gl.constexpr, BFLY: gl.constexpr):
    args = [optrs, omask.to(gl.int32), pptrs, pmask.to(gl.int32), tag]
    if RP == 2 and BFLY:
        return gl.inline_asm_elementwise(
            _SUMS2_BFLY, _SUMS2_C, args, dtype=gl.int64, is_pure=False, pack=2
        )
    elif RP == 2:
        return gl.inline_asm_elementwise(
            _SUMS2, _SUMS2_C, args, dtype=gl.int64, is_pure=False, pack=2
        )
    else:
        return gl.inline_asm_elementwise(
            _SUMS4, _SUMS4_C, args, dtype=gl.int64, is_pure=False, pack=4
        )


@gluon.jit
def _warp_rows_sumsq(x):
    # Sum of x^2 over a warp's 8 rows of a [8, 4]-thread row slice (rows differ in lane bits
    # 2-4): a butterfly over xor 4, 8, 16, the same bits on every lane.
    return gl.inline_asm_elementwise(
        "{ .reg .f32 s, t; mul.f32 s, $1, $1; shfl.sync.bfly.b32 t, s, 4, 31, -1; "
        "add.f32 s, s, t; shfl.sync.bfly.b32 t, s, 8, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 16, 31, -1; add.f32 $0, s, t; }",
        "=f,f",
        [x],
        dtype=gl.float32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _prefetch2(a, b):
    # Thread 0 asks L2 for the lines of a and b (hints: no register, no fault).
    gl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, 0; "
        "@p prefetch.global.L2 [$1]; @p prefetch.global.L2 [$2]; mov.u32 $0, 0; }",
        "=r,l,l",
        [a, b],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


def _publish_sumsq_asm() -> str:
    """OSQ: a 64-row 4-warp flag tile's sum of o^2 and its publish (2 rows per thread).

    Per thread ``fma(o0, o0, o1 * o1)``, a butterfly over lane xor 16, 8, 4; lane 0 of each warp
    stores its sum to shared memory, thread 0 a zero word next to them; named barrier 2; warp 0
    adds the four sums as (w0 + w2) + (w1 + w3) (Triton's cross-warp order, the same bits) and
    its lane 0 stores the word. Every warp then waits at named barrier 3, so no state store goes
    out ahead of the publish, and reads the zero word back (volatile): alpha and d * rk come
    back as their bits plus that zero, so the state arithmetic follows the publish."""
    o, drk, alpha = ["$4", "$5"], ["$6", "$7"], ["$8", "$9"]
    tag, address, valid = "$10", "$12", "$14"
    lines = [
        "{ .reg .pred w0, l0, t0, pv; .reg .f32 s, t, x, y, z, u;",
        ".reg .u32 tid, lane, warp, sb, off, zz, b; .reg .b64 word, lo;",
        ".shared .align 16 .f32 osqbuf[8];",
        f"mul.rn.f32 s, {o[1]}, {o[1]}; fma.rn.f32 s, {o[0]}, {o[0]}, s;",
    ]
    for d in (16, 8, 4):
        lines.append(f"shfl.sync.bfly.b32 t, s, {d}, 31, -1; add.rn.f32 s, s, t;")
    lines += [
        "mov.u32 tid, %tid.x; and.b32 lane, tid, 31; shr.u32 warp, tid, 5; setp.eq.u32 l0, lane, 0;",
        "setp.eq.u32 w0, warp, 0; setp.eq.u32 t0, tid, 0; mov.u32 sb, osqbuf; shl.b32 off, warp, 2;",
        "add.u32 off, sb, off; @l0 st.shared.f32 [off], s; @t0 st.shared.u32 [sb+16], 0;",
        "bar.sync 2, 128;",
        "@w0 ld.shared.v4.f32 {x, y, z, u}, [sb]; add.rn.f32 x, x, z; add.rn.f32 y, y, u;",
        "add.rn.f32 s, x, y;",
        f"setp.ne.u32 pv, {valid}, 0; and.pred pv, pv, l0; and.pred pv, pv, w0;",
        "mov.b32 off, s; cvt.u64.u32 lo, off;",
        f"or.b64 word, {tag}, lo; @pv st.global.b64 [{address}], word;",
        "bar.sync 3, 128; ld.volatile.shared.u32 zz, [sb+16];",
    ]
    for i, source in enumerate([alpha[0], alpha[1], drk[0], drk[1]]):
        lines.append(f"mov.b32 b, {source}; add.u32 b, b, zz; mov.b32 ${i}, b;")
    lines.append("}")
    return " ".join(lines)


_OSQ = gl.constexpr(_publish_sumsq_asm())
_OSQ_C = gl.constexpr(",".join(["=f"] * 4 + ["f"] * 6 + ["l"] * 2 + ["l"] * 2 + ["r"] * 2))


@gluon.jit
def _publish_sumsq(o, drk, alpha, tag, address, valid):
    return gl.inline_asm_elementwise(
        _OSQ,
        _OSQ_C,
        [o, drk, alpha, tag, address.to(gl.int64), valid.to(gl.int32)],
        dtype=(gl.float32, gl.float32),
        is_pure=False,
        pack=2,
    )


@gluon.constexpr_function
def _pow2(n):
    return triton.next_power_of_2(n)


@gluon.jit
def _zero_words(ptr, N: gl.constexpr, valid):
    # SC: zeroes the N words from ptr, two contiguous words (16 bytes) per thread.
    NZ: gl.constexpr = _pow2(N)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [gl.num_warps()], [0])
    offs = gl.arange(0, NZ, layout=layout)
    gl.store(ptr + offs, gl.full([NZ], 0, gl.int64, layout=layout), mask=(offs < N) & valid)


@gluon.jit
def _taps(convw_ptr, ch, PACKED: gl.constexpr):
    """The four conv taps of channels ``ch`` as FP32: one 8-byte word per channel when PACKED
    (``convw_ptr`` is then an int64 view of the [conv_dim, 4] BF16 taps), else four loads."""
    if PACKED:
        w = gl.load(convw_ptr + ch)
        lo = w.to(gl.int32)
        hi = (w >> 32).to(gl.int32)
        return (
            _bf16_bits_to_f32(lo),
            _bf16_bits_to_f32(lo >> 16),
            _bf16_bits_to_f32(hi),
            _bf16_bits_to_f32(hi >> 16),
        )
    else:
        base = convw_ptr + ch * 4
        return (
            gl.load(base).to(gl.float32),
            gl.load(base + 1).to(gl.float32),
            gl.load(base + 2).to(gl.float32),
            gl.load(base + 3).to(gl.float32),
        )


@gluon.jit
def _conv_silu(x, h0, h1, h2, w0, w1, w2, w3):
    """SiLU of one channel's causal conv: taps w0..w3 over history rows h0..h2 and this row x.
    The products are exact in FP32; their sum is error-free and rounded once."""
    p0 = w0 * h0.to(gl.float32)
    p1 = w1 * h1.to(gl.float32)
    p2 = w2 * h2.to(gl.float32)
    p3 = w3 * x.to(gl.float32)
    s, e1 = _two_sum(p0, p1)
    s, e2 = _two_sum(s, p2)
    s, e3 = _two_sum(s, p3)
    c = s + ((e1 + e2) + e3)
    return c * _sigmoid(c)


@gluon.jit
def gdn_decode_kernel(
    proj_ptr,
    convw_ptr,
    alog_ptr,
    dtb_ptr,
    normw_ptr,
    slot_ptr,
    state_ptr,
    cstate_ptr,
    y_ptr,
    flag_ptr,
    ow_ptr,
    n_main,
    SP0: gl.constexpr,
    HK: gl.constexpr,
    HV: gl.constexpr,
    BV: gl.constexpr,
    NV: gl.constexpr,
    G: gl.constexpr,
    PACKED: gl.constexpr,
    TILE: gl.constexpr,
    VEC: gl.constexpr,
    GRP: gl.constexpr,
    FIN: gl.constexpr,
    RP: gl.constexpr,
    FTILE: gl.constexpr,
    SPL: gl.constexpr,
    WSUM: gl.constexpr,
    STILE: gl.constexpr,
    VSH: gl.constexpr,
    A3: gl.constexpr,
    NCOL: gl.constexpr,
    BFLY: gl.constexpr,
    OSQ: gl.constexpr,
    SC: gl.constexpr,
    ROWIDX: gl.constexpr,
    PWZ: gl.constexpr,
    OCC: gl.constexpr,
    PDL: gl.constexpr,
):
    """State CTAs ``[0, n_main)``, then (FIN) one finisher per (sequence, key head)."""
    gl.static_assert(not (SC and OSQ), "the self-cleaning scratch has no OSQ publish")
    if PDL:
        _gdc_wait()
        # The next PDL kernel may start its prologue now; its own wait still covers our writes.
        _gdc_launch_dependents()
    pid = gl.program_id(0)
    if FIN:
        if pid >= n_main:
            _finisher_cta(
                pid - n_main, proj_ptr, normw_ptr, slot_ptr, cstate_ptr, y_ptr, flag_ptr, ow_ptr,
                SP0, HK, HV, RP, FTILE, WSUM, STILE, VSH, BFLY and RP == 2 and not A3, SC, ROWIDX,
            )  # fmt: skip
        elif A3:
            # Rows 0-63 as tile 0 of a head in 64-row tiles, rows 64-95 and 96-127 as tiles 2
            # and 3 of a head in 32-row tiles (BV = 32 here): each an unchanged state CTA (same
            # rows, warp-sum words and finisher) under its own program id.
            t = pid // 3
            j = pid % 3
            if j == 0:
                _state_cta(
                    t * 2, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr,
                    state_ptr, cstate_ptr, y_ptr, flag_ptr, ow_ptr, SP0, HK, HV, 2 * BV, 2, G,
                    PACKED, TILE, VEC, GRP, True, 0, WSUM, False, False, SC, ROWIDX, False,
                )  # fmt: skip
            else:
                _state_cta(
                    t * 4 + j + 1, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr,
                    state_ptr, cstate_ptr, y_ptr, flag_ptr, ow_ptr, SP0, HK, HV, BV, 4, G,
                    PACKED, TILE, VEC, GRP, True, 0, WSUM, False, False, SC, ROWIDX, False,
                )  # fmt: skip
        else:
            _state_cta(
                pid, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr, state_ptr,
                cstate_ptr, y_ptr, flag_ptr, ow_ptr, SP0, HK, HV, BV, NV, G, PACKED, TILE, VEC,
                GRP, True, 0, WSUM, False, False, SC, ROWIDX, False,
            )  # fmt: skip
    elif OCC > 0:
        # OCC = B: the tile index in this form (the key-head groups' interleave, selected away
        # by a predicate on n_main that is always false, so the tile is pid) compiles the 64/4
        # flag tile to 166 registers, three CTAs per SM, against 184-186 for pid itself.
        GS: gl.constexpr = (HV // HK) * NV
        g = pid // GS
        inter = ((g % OCC) * HV + (g // OCC) * (HV // HK)) * NV + pid % GS
        pick = (n_main < 0).to(gl.int32)
        tile = pid + pick * (inter - pid)
        _state_cta(
            tile, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr, state_ptr,
            cstate_ptr, y_ptr, flag_ptr, ow_ptr, SP0, HK, HV, BV, NV, G, PACKED, TILE, VEC, GRP,
            False, SPL, False, NCOL, OSQ, SC, ROWIDX, PWZ,
        )  # fmt: skip
    else:
        _state_cta(
            pid, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr, state_ptr,
            cstate_ptr, y_ptr, flag_ptr, ow_ptr, SP0, HK, HV, BV, NV, G, PACKED, TILE, VEC, GRP,
            False, SPL, False, NCOL, OSQ, SC, ROWIDX, PWZ,
        )  # fmt: skip


@gluon.jit
def _finisher_cta(
    fid, proj_ptr, normw_ptr, slot_ptr, cstate_ptr, y_ptr, flag_ptr, ow_ptr,
    SP0: gl.constexpr, HK: gl.constexpr, HV: gl.constexpr, RP: gl.constexpr,
    FTILE: gl.constexpr, WSUM: gl.constexpr, STILE: gl.constexpr, VSH: gl.constexpr,
    BFLY: gl.constexpr, SC: gl.constexpr, ROWIDX: gl.constexpr,
):  # fmt: skip
    """Finisher of (sequence, key head) ``fid``: y of the key head's R value heads from the o
    words its state CTAs publish, then the q | k history shift and (epochs) the counter; with
    SC it zeroes the words it read instead."""
    K: gl.constexpr = 128
    OWS: gl.constexpr = K + 16  # o words per value head (O_WORDS)
    R: gl.constexpr = HV // HK
    KW: gl.constexpr = HK * K
    DC: gl.constexpr = 2 * KW + HV * K
    VW: gl.constexpr = HV * K
    OCG: gl.constexpr = R * 4 + 1
    FR: gl.constexpr = gl.SliceLayout(1, FTILE)
    FC: gl.constexpr = gl.SliceLayout(0, FTILE)
    n = fid // HK
    kh = fid % HK
    n64 = n.to(gl.int64)
    proj_row = proj_ptr + n64 * SP0
    r2 = gl.arange(0, RP, layout=FR)  # value head within the key head (RP = R rounded up)
    c2 = gl.arange(0, K, layout=FC)  # row of the value head
    rmask = gl.expand_dims(r2 < R, 1) & gl.expand_dims(c2 >= 0, 0)
    hv2 = kh * R + r2
    slot = gl.load(slot_ptr + n64).to(gl.int64)
    z = gl.load(
        proj_row + gl.expand_dims(DC + hv2 * K, 1) + gl.expand_dims(c2, 0), mask=rmask, other=0.0
    ).to(gl.float32)
    nw = gl.expand_dims(gl.load(normw_ptr + c2).to(gl.float32), 0)
    q_ch = kh * K + c2
    k_ch = KW + kh * K + c2
    if not VSH:
        xq = gl.load(proj_row + q_ch)
        xk = gl.load(proj_row + k_ch)
    valid = slot >= 0
    base = gl.maximum(slot, 0)
    if ROWIDX:
        sbase = n64
    else:
        sbase = base
    hist_row = cstate_ptr + base * (3 * DC)
    vmask = (c2 >= 0) & valid
    if not VSH:
        q1 = gl.load(hist_row + DC + q_ch, mask=vmask, other=0.0)
        q2 = gl.load(hist_row + 2 * DC + q_ch, mask=vmask, other=0.0)
        k1 = gl.load(hist_row + DC + k_ch, mask=vmask, other=0.0)
        k2 = gl.load(hist_row + 2 * DC + k_ch, mask=vmask, other=0.0)
    else:
        # The q | k history shift as one [8, 128] tile: row j < 6 is (q, k)[j // 3]'s new row
        # j % 3 (old rows 1, 2, then this token's input). Loaded now, stored after the wait.
        sj = gl.arange(0, 8, layout=gl.SliceLayout(1, STILE))
        sc = gl.arange(0, K, layout=gl.SliceLayout(0, STILE))
        s_ch = gl.expand_dims(gl.where(sj >= 3, KW, 0) + kh * K, 1) + gl.expand_dims(sc, 0)
        s_r = gl.expand_dims(sj % 3, 1)
        s_live = gl.expand_dims(sj < 6, 1) & gl.expand_dims(sc >= 0, 0) & valid
        s_new = s_live & (s_r == 2)
        s_old = s_live & (s_r < 2)
        shifted = gl.where(
            s_new,
            gl.load(proj_row + s_ch, mask=s_new, other=0.0),
            gl.load(hist_row + (s_r + 1) * DC + s_ch, mask=s_old, other=0.0),
        )
    if not SC:
        ctr_ptr = flag_ptr + (base * HK + kh) * OCG + R * 4
        epoch = (gl.load(ctr_ptr, mask=valid, other=0) + 1) & 0xFFFFFFFF
    gate = nw * z * _sigmoid(z)
    owords = ow_ptr + (sbase * HV + kh * R) * OWS
    wptr = owords + gl.expand_dims(r2 * OWS, 1) + gl.expand_dims(c2, 0)
    need = rmask & valid
    if SC:
        tag = gl.full([RP, K], 1, gl.int32, layout=FTILE)
    else:
        tag = gl.full([RP, K], 0, gl.int32, layout=FTILE) + epoch.to(gl.int32)
    if WSUM:
        # The state CTAs' warp sums of o^2 (RP = 2 or 4), polled by lanes 0..15 with the o
        # words and gathered within the warp.
        lane = c2 % 32
        pmask = need & gl.expand_dims(lane < 16, 0)
        if RP == 4:
            pmask = pmask & gl.expand_dims(c2 < 32, 0)
        pptr = owords + gl.expand_dims(r2 * OWS, 1) + K + gl.expand_dims(lane, 0)
        words = _wait_words_sums(wptr, need, pptr, pmask, tag, RP, BFLY)
    elif RP <= 4:
        words = _wait_words(wptr, need, tag, RP)
    else:
        words = _poll(wptr, need)
        while gl.sum(gl.sum(gl.where(need & (_high(words) != tag), 1, 0), axis=1), axis=0) > 0:
            words = _poll(wptr, need)
        if SC:
            gl.barrier()  # the generic wait's exit is per warp
    if SC:
        # Every thread's last poll has returned (the wait ends CTA-wide): zero the words.
        _zero_words(owords, R * OWS, valid)
    if VSH:
        gl.store(hist_row + s_r * DC + s_ch, shifted, mask=s_live)
        if not SC:
            gl.store(ctr_ptr, epoch, mask=valid)
    o = _low(words)
    if WSUM:
        rho = gl.rsqrt(_low(words >> 32) * (1.0 / 128.0) + 1e-6)
        y = gate * (o * rho)
    else:
        rho = gl.rsqrt(gl.sum(o * o, axis=1) * (1.0 / 128.0) + 1e-6)
        y = gate * (o * gl.expand_dims(rho, 1))
    y = gl.where(need, y, 0.0)
    y_off = gl.expand_dims(hv2 * K, 1) + gl.expand_dims(c2, 0)
    gl.store(y_ptr + n64 * VW + y_off, y.to(gl.bfloat16), mask=rmask)
    if not VSH:
        # Every state CTA of the key head has read the q | k history (it published after).
        gl.store(hist_row + q_ch, q1, mask=vmask)
        gl.store(hist_row + DC + q_ch, q2, mask=vmask)
        gl.store(hist_row + 2 * DC + q_ch, xq, mask=vmask)
        gl.store(hist_row + k_ch, k1, mask=vmask)
        gl.store(hist_row + DC + k_ch, k2, mask=vmask)
        gl.store(hist_row + 2 * DC + k_ch, xk, mask=vmask)
        if not SC:
            gl.store(ctr_ptr, epoch, mask=valid)


@gluon.jit
def _state_cta(
    pid, proj_ptr, convw_ptr, alog_ptr, dtb_ptr, normw_ptr, slot_ptr, state_ptr, cstate_ptr,
    y_ptr, flag_ptr, ow_ptr,
    SP0: gl.constexpr, HK: gl.constexpr, HV: gl.constexpr, BV: gl.constexpr, NV: gl.constexpr,
    G: gl.constexpr, PACKED: gl.constexpr, TILE: gl.constexpr, VEC: gl.constexpr,
    GRP: gl.constexpr, FIN: gl.constexpr, SPL: gl.constexpr, WSUM: gl.constexpr,
    NCOL: gl.constexpr, OSQ: gl.constexpr, SC: gl.constexpr, ROWIDX: gl.constexpr,
    PWZ: gl.constexpr,
):  # fmt: skip
    """One state CTA: (sequence, value head, tile of BV rows)."""
    K: gl.constexpr = 128
    OWS: gl.constexpr = K + 16  # o words per value head (O_WORDS)
    R: gl.constexpr = HV // HK
    KW: gl.constexpr = HK * K
    DC: gl.constexpr = 2 * KW + HV * K
    VW: gl.constexpr = HV * K
    # Epoch flag words per (slot, key head): a set per NV = 1, 2, 4 (counter, then R * NV words).
    WPG: gl.constexpr = R * 7 + 3
    FB: gl.constexpr = 0 if NV == 1 else (R + 1 if NV == 2 else 3 * R + 2)
    OCG: gl.constexpr = R * 4 + 1
    WSC: gl.constexpr = R * 16
    TAG: gl.constexpr = 1 << 32
    ROW: gl.constexpr = gl.SliceLayout(1, TILE)
    COL: gl.constexpr = gl.SliceLayout(0, TILE)

    iv = pid % NV
    t = pid // NV
    hv = t % HV
    n = t // HV
    kh = hv // R
    rows = iv * BV + gl.arange(0, BV, layout=ROW)
    cols = gl.arange(0, K, layout=COL)
    offs = gl.arange(0, K, layout=VEC)
    n64 = n.to(gl.int64)
    proj_row = proj_ptr + n64 * SP0
    y_row = y_ptr + n64 * VW + hv * K

    # Everything that does not depend on the slot first.
    slot = gl.load(slot_ptr + n64).to(gl.int64)
    q_ch = kh * K + offs
    k_ch = KW + kh * K + offs
    v_ch = 2 * KW + hv * K + rows
    xq = gl.load(proj_row + q_ch)
    xk = gl.load(proj_row + k_ch)
    xv = gl.load(proj_row + v_ch)
    wq0, wq1, wq2, wq3 = _taps(convw_ptr, q_ch, PACKED)
    wk0, wk1, wk2, wk3 = _taps(convw_ptr, k_ch, PACKED)
    wv0, wv1, wv2, wv3 = _taps(convw_ptr, v_ch, PACKED)
    z = gl.load(proj_row + DC + hv * K + rows).to(gl.float32)
    nw = gl.load(normw_ptr + rows).to(gl.float32)
    b_raw = gl.load(proj_row + DC + VW + hv).to(gl.float32)
    a_raw = gl.load(proj_row + DC + VW + HV + hv).to(gl.float32)
    A_log = gl.load(alog_ptr + hv).to(gl.float32)
    dt_bias = gl.load(dtb_ptr + hv).to(gl.float32)

    # The slot's pool rows and scratch, every access masked on padding; int64 offsets.
    valid = slot >= 0
    base = gl.maximum(slot, 0)
    if ROWIDX:
        sbase = n64
    else:
        sbase = base
    hist_row = cstate_ptr + base * (3 * DC)
    state_tile = (
        state_ptr
        + (base * HV + hv) * (K * K)
        + gl.expand_dims(rows, 1) * K
        + gl.expand_dims(cols, 0)
    )
    if SC:
        grp_ptr = flag_ptr + (sbase * HK + kh) * WSC
    else:
        grp_ptr = flag_ptr + (base * HK + kh) * WPG + FB + 1
    if NV <= 2 and not FIN:
        # The group's scratch lines to L2 now (a cold pool's scratch is cold too).
        if SC:
            _prefetch2(grp_ptr, grp_ptr + (R * NV * NV - 1))
        else:
            _prefetch2(grp_ptr - 1, grp_ptr + (R * NV - 1))
    if not SC:
        if FIN:
            counter = gl.load(flag_ptr + (base * HK + kh) * OCG + R * 4, mask=valid, other=0)
        else:
            counter = gl.load(grp_ptr - 1, mask=valid, other=0)
    tile_mask = gl.full([BV, K], 1, gl.int1, layout=TILE) & valid
    vec_mask = gl.full([K], 1, gl.int1, layout=VEC) & valid
    row_mask = gl.full([BV], 1, gl.int1, layout=ROW) & valid
    # The conv history first (small, needed first), then the state rows.
    q0 = gl.load(hist_row + q_ch, mask=vec_mask, other=0.0)
    q1 = gl.load(hist_row + DC + q_ch, mask=vec_mask, other=0.0)
    q2 = gl.load(hist_row + 2 * DC + q_ch, mask=vec_mask, other=0.0)
    k0 = gl.load(hist_row + k_ch, mask=vec_mask, other=0.0)
    k1 = gl.load(hist_row + DC + k_ch, mask=vec_mask, other=0.0)
    k2 = gl.load(hist_row + 2 * DC + k_ch, mask=vec_mask, other=0.0)
    v0 = gl.load(hist_row + v_ch, mask=row_mask, other=0.0)
    v1 = gl.load(hist_row + DC + v_ch, mask=row_mask, other=0.0)
    v2 = gl.load(hist_row + 2 * DC + v_ch, mask=row_mask, other=0.0)
    S = gl.load(state_tile, mask=tile_mask, other=0.0)

    # Gates of the value head.
    xg = a_raw + dt_bias
    softplus = gl.maximum(xg, 0.0) + libdevice.log1p(gl.exp(-gl.abs(xg)))
    alpha = gl.exp(-gl.exp(A_log) * softplus)
    beta = _sigmoid(b_raw)

    # Conv + SiLU; q and k stay unnormalised, so the layout change does not wait for norms.
    q = _conv_silu(xq, q0, q1, q2, wq0, wq1, wq2, wq3)
    k = _conv_silu(xk, k0, k1, k2, wk0, wk1, wk2, wk3)
    v = _conv_silu(xv, v0, v1, v2, wv0, wv1, wv2, wv3)
    if NCOL:
        # q.q, k.k and q.k from the column slice each warp holds whole after the layout
        # change: in-warp shuffles, no cross-warp pass ahead of the norms.
        kcol = gl.convert_layout(k, COL)
        qcol = gl.convert_layout(q, COL)
        kc = gl.expand_dims(kcol, 0)
        qc = gl.expand_dims(qcol, 0)
        qq, kk, qk = gl.reduce((qcol * qcol, kcol * kcol, qcol * kcol), 0, _add3)
    else:
        kc = gl.expand_dims(gl.convert_layout(k, COL), 0)
        qc = gl.expand_dims(gl.convert_layout(q, COL), 0)
        qq, kk, qk = gl.reduce((q * q, k * k, q * k), 0, _add3)
    rq = gl.rsqrt(qq + 1e-6) * 0.08838834764831845  # K^-1/2
    rk = gl.rsqrt(kk + 1e-6)
    kq = qk * rq * rk
    # This tile's v channels are its own: shift their history now.
    gl.store(hist_row + v_ch, v1, mask=row_mask)
    gl.store(hist_row + DC + v_ch, v2, mask=row_mask)
    gl.store(hist_row + 2 * DC + v_ch, xv, mask=row_mask)

    # The delta rule from one pass over the tile (k = rk k_raw, q = rq q_raw).
    sk = gl.sum(S * kc, axis=1) * rk
    sq = gl.sum(S * qc, axis=1) * rq
    d = beta * (v - alpha * sk)
    o = alpha * sq + d * kq

    if SC:
        hi = TAG
    else:
        epoch = (counter + 1) & 0xFFFFFFFF
        hi = epoch << 32
    if FIN:
        # Publish this tile's o for the key head's finisher, then store the new state rows.
        ow = ow_ptr + (sbase * HV + hv) * OWS
        gl.store(ow + rows, _word(hi, o), mask=row_mask)
        if WSUM:
            # Each warp's sum of o^2 over its 8 rows, word K + 4 iv + warp of the value head,
            # published by the warp's row-0 lanes ahead of the state stores.
            wsum = _warp_rows_sumsq(o)
            gl.store(
                ow + K + iv * 4 + (rows - iv * BV) // 8,
                _word(hi, wsum),
                mask=row_mask & ((rows % 8) == 0),
            )
        gl.store(
            state_tile,
            alpha * S + gl.expand_dims(d * rk, 1) * kc,
            mask=tile_mask,
            cache_modifier=".cs",
        )
    else:
        first = (hv % R) * NV
        if OSQ:
            # The sum and its publish (_publish_sumsq_asm), then the new state rows from the
            # alpha and d * rk it hands back once the word is out.
            al_r, drk_r = _publish_sumsq(o, d * rk, alpha, hi, grp_ptr + first + iv, valid)
            s_new = gl.expand_dims(al_r, 1) * S + gl.expand_dims(drk_r, 1) * kc
        else:
            part = gl.sum(o * o, axis=0)
            if SC:
                # One word per destination tile at (first + iv) * NV + destination: the copies
                # this tile's peers read, and at destination iv the token the key head's writer
                # reads (none in the writer's own value head, whose copies it reads).
                dst = gl.arange(0, G, layout=GRP)
                gl.store(
                    grp_ptr + (first + iv) * NV + dst,
                    _word(hi, part),
                    mask=(dst < NV) & ((dst != iv) | (hv % R < R - 1)) & valid,
                )
            else:
                gl.store(grp_ptr + first + iv, _word(hi, part), mask=valid)
            s_new = alpha * S + gl.expand_dims(d * rk, 1) * kc
        if SPL == 0:
            gl.store(state_tile, s_new, mask=tile_mask, cache_modifier=".cs")

        # Wait for the head's other tiles (and, as the key head's last CTA, the whole group).
        writer = (hv % R == R - 1) & (iv == NV - 1)
        idx = gl.arange(0, G, layout=GRP)
        mine = (idx >= first) & (idx < first + NV)
        if SC:
            group = grp_ptr + idx * NV + gl.where(mine, iv, idx % NV)
            ready = 1
        else:
            group = grp_ptr + idx
            ready = epoch
        if NV == 1:
            total = part
            if writer:
                if SC:
                    need = (idx < R - 1) & valid
                else:
                    need = (idx < R * NV) & valid
                words = _poll(group, need)
                while gl.sum(gl.where(need & (_high(words) != ready), 1, 0), axis=0) > 0:
                    words = _poll(group, need)
                if SC:
                    gl.barrier()  # every warp has left its wait before the words are zeroed
                    gl.store(group, 0, mask=need)
        else:
            need = (mine | (writer & (idx < R * NV))) & valid
            if SC:
                need = need & (idx != first + iv)
            if SPL > 0:
                # The first poll goes out between the tile's first SPL columns and the rest: a
                # poll load waits behind the CTA's own stores.
                if PWZ:
                    zero = _mask64(need)
                    gl.store(
                        state_tile,
                        s_new,
                        mask=tile_mask & gl.expand_dims(cols < SPL, 0),
                        cache_modifier=".cs",
                    )
                    words = _poll_z(group, need, zero)
                else:
                    gl.store(
                        state_tile,
                        s_new,
                        mask=tile_mask & gl.expand_dims(cols < SPL, 0),
                        cache_modifier=".cs",
                    )
                    words = _poll(group, need)
                gl.store(
                    state_tile,
                    s_new,
                    mask=tile_mask & gl.expand_dims(cols >= SPL, 0),
                    cache_modifier=".cs",
                )
            else:
                words = _poll(group, need)
            while gl.sum(gl.where(need & (_high(words) != ready), 1, 0), axis=0) > 0:
                words = _poll(group, need)
            if SC:
                gl.barrier()
                gl.store(group, 0, mask=need)
                # This tile's own partial from registers: the same values in the same order.
                parts = gl.where(idx == first + iv, part, _low(words))
            else:
                parts = _low(words)
            total = gl.sum(gl.where(mine, parts, 0.0), axis=0)

        # Gated RMSNorm of this tile.
        rho = gl.rsqrt(total * (1.0 / 128.0) + 1e-6)
        y = (nw * z * _sigmoid(z)) * (o * rho)
        y = gl.where(row_mask, y, 0.0)
        gl.store(y_row + rows, y.to(gl.bfloat16))

        # Every CTA of the key head has consumed the q | k history: shift it, then advance
        # the set's counter (every CTA of the group read it before publishing).
        if writer:
            gl.store(hist_row + q_ch, q1, mask=vec_mask)
            gl.store(hist_row + DC + q_ch, q2, mask=vec_mask)
            gl.store(hist_row + 2 * DC + q_ch, xq, mask=vec_mask)
            gl.store(hist_row + k_ch, k1, mask=vec_mask)
            gl.store(hist_row + DC + k_ch, k2, mask=vec_mask)
            gl.store(hist_row + 2 * DC + k_ch, xk, mask=vec_mask)
            if not SC:
                gl.store(grp_ptr - 1, epoch, mask=valid)


@triton.jit(do_not_specialize=["n_words"])
def zero_scratch_kernel(
    scratch_ptr, slot_ptr, n_words, PER_SLOT: tl.constexpr, BLOCK: tl.constexpr
):
    """Zeroes scratch: all ``n_words`` words, or (PER_SLOT; program id = batch row) the
    ``n_words`` words of each slot the call names, padding skipped."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if PER_SLOT:
        slot = tl.load(slot_ptr + pid).to(tl.int64)
        tl.store(scratch_ptr + slot * n_words + offs, 0, mask=(offs < n_words) & (slot >= 0))
    else:
        at = pid.to(tl.int64) * BLOCK + offs
        tl.store(scratch_ptr + at, 0, mask=at < n_words)
