# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Torch CPU execution emulation of the K1 FP32 arithmetic order.

CDF ABI r2.1 uses 4096-token chunks of four 1024-token tiles. Each tile has
256 logical lanes of four tokens; local prefixes are sequential, lane/warp
totals use Hillis-Steele scans. Chunk totals start at zero, row totals sum
chunks sequentially. Candidate prefixes retain their sequential order.
Every arithmetic operation below rounds to FP32, including each scan step.
"""

import torch

from .rng import ACCEPT, BONUS, CORRECTION, ORDINARY_NAMESPACE, PROPOSAL, uniform

CHUNK = 4096
TILE = 1024
VEC = 4


def _fp32(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _exp(value):
    """Correctly rounded FP32 exp (FP64 then one rounding), identical on every CPU.

    The contract leaves exp provider-specific (FP32 p is reproducible within a
    provider/device); SIMD FP32 exp kernels differ across CPU ISAs.
    """
    return torch.exp(value.double()).float()


def _sequential_prefix(values):
    """Scan the last axis, rounding every add (torch.cumsum uses FP64 on CPU)."""
    result = torch.empty_like(values)
    if values.shape[-1]:
        result[..., 0] = values[..., 0]
        for i in range(1, values.shape[-1]):
            result[..., i] = result[..., i - 1] + values[..., i]
    return result


def _uint64_to_float32(value):
    """One integer-to-binary32 RNE rounding, without an intermediate double."""
    if not 0 <= value < 2**64:
        raise ValueError("value must fit uint64")
    if value == 0:
        return _fp32(0)
    exponent = value.bit_length() - 1
    shift = max(0, exponent - 23)
    significand, remainder = divmod(value, 1 << shift)
    if shift and (
        remainder > 1 << (shift - 1) or (remainder == 1 << (shift - 1) and significand & 1)
    ):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    significand <<= max(0, 23 - exponent)
    bits = ((exponent + 127) << 23) | (significand & 0x7FFFFF)
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32)


def request_seed(seeds, slot):
    seeds = torch.as_tensor(seeds)
    return int(seeds[slot, 0]) | (int(seeds[slot, 1]) << 32)


def cdf_parts(weights, *, with_chunks=False):
    """Token prefixes and row total; optionally include chunk-selection prefixes."""
    w = _fp32(weights)
    if not len(w):
        raise ValueError("CDF requires nonempty support")
    padded = torch.zeros(((len(w) + TILE - 1) // TILE) * TILE, dtype=torch.float32)
    padded[: len(w)] = w
    local = _sequential_prefix(padded.reshape(-1, 8, 32, VEC))
    lanes = local[..., -1].clone()
    for offset in (1, 2, 4, 8, 16):
        lanes[..., offset:] = lanes[..., :-offset] + lanes[..., offset:]
    warps = lanes[..., -1].clone()
    for offset in (1, 2, 4):
        warps[:, offset:] = warps[:, :-offset] + warps[:, offset:]
    before_lane = torch.zeros_like(lanes)
    before_lane[..., 1:] = lanes[..., :-1]
    before_warp = torch.zeros_like(warps)
    before_warp[:, 1:] = warps[:, :-1]
    preceding = before_warp[..., None] + before_lane
    tile_totals = warps[:, -1]
    chunk_totals = torch.stack(
        [
            _sequential_prefix(tile_totals[i : i + CHUNK // TILE])[-1]
            for i in range(0, len(tile_totals), CHUNK // TILE)
        ]
    )
    chunk_prefixes = _sequential_prefix(chunk_totals)
    before_tile = torch.empty_like(tile_totals)
    for c in range(len(chunk_totals)):
        carry = chunk_prefixes[c - 1] if c else _fp32(0)
        for t in range(c * (CHUNK // TILE), min((c + 1) * (CHUNK // TILE), len(tile_totals))):
            before_tile[t] = carry
            carry = carry + tile_totals[t]
    prefixes = before_tile[:, None, None, None] + (preceding[..., None] + local)
    parts = (prefixes.flatten()[: len(w)], chunk_prefixes[-1])
    return (*parts, chunk_prefixes) if with_chunks else parts


def candidate_cdf_parts(weights):
    """The small-support candidate ABI is sequential, in ascending ID order."""
    prefixes = _sequential_prefix(_fp32(weights).flatten())
    return prefixes, prefixes[-1]


def categorical(weights, u, *, candidates=False):
    start = 0
    if candidates:
        prefixes, total = candidate_cdf_parts(weights)
    else:
        prefixes, total, chunks = cdf_parts(weights, with_chunks=True)
        eligible = torch.nonzero(chunks > _fp32(u) * total).flatten()
        if eligible.numel():
            start = int(eligible[0]) * CHUNK
            prefixes = prefixes[start : start + CHUNK]
        else:
            prefixes = prefixes[:0]
    hits = torch.nonzero(prefixes > _fp32(u) * total).flatten()
    if hits.numel():
        return start + int(hits[0])
    positive = torch.nonzero(_fp32(weights) > 0).flatten()
    if not positive.numel():
        raise ValueError("categorical requires positive mass")
    return int(positive[-1])


def residual(p, q):
    p, q = _fp32(p), _fp32(q)
    difference = p - q
    # NumPy maximum chooses the second operand at equality, including -0.
    zero = _fp32(0)
    r = torch.where(difference <= zero, zero, difference)
    return p if cdf_parts(r)[1] == 0 else r


def sparse_dense(ids, q, vocab):
    result = torch.zeros(vocab, dtype=torch.float32)
    for token, mass in zip(ids, _fp32(q), strict=True):
        result[int(token)] = result[int(token)] + mass
    return result


def fixed_mass(value, *, ceiling=False):
    """Exact floor/ceil of a nonnegative FP32 value times 2^62, from its bits."""
    bits = int(_fp32(value).view(torch.int32))
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent:
        mantissa |= 1 << 23
    shift = exponent - 88 if exponent else -87
    if shift >= 0:
        return mantissa << shift
    divisor = 1 << -shift
    return (mantissa + (divisor - 1 if ceiling else 0)) // divisor


def probability_chunk_sum(values):
    """4096 tokens: two four-token iterations per lane, then butterfly tree."""
    padded = torch.zeros(CHUNK, dtype=torch.float32)
    padded[: len(values)] = _fp32(values)
    lanes = torch.zeros(512, dtype=torch.float32)
    for tile in padded.reshape(-1, 512, 4):
        for j in range(4):
            lanes = lanes + tile[:, j]
    warps = lanes.reshape(16, 32)
    indices = torch.arange(32)
    for offset in (1, 2, 4, 8, 16):
        warps = warps + warps[:, indices ^ offset]
    totals = torch.zeros(32, dtype=torch.float32)
    totals[:16] = warps[:, 0]
    for offset in (1, 2, 4, 8, 16):
        totals = totals + totals[indices ^ offset]
    return totals[0]


def probability_sum(values):
    """Chunk tree partials combined sequentially in ascending chunk order."""
    total = _fp32(0)
    for start in range(0, len(values), CHUNK):
        total = total + probability_chunk_sum(values[start : start + CHUNK])
    return total


def min_p_row(p_row, min_p):
    """Mask a final FP32 probability row, then renormalize with the frozen tree.

    Admission guarantees finite min_p in [0,1]. Zero preserves every input bit;
    the provider bypasses this helper for greedy and padding rows.
    """
    if min_p <= 0:
        return p_row
    pivot = _fp32(min_p) * torch.max(p_row)
    kept = p_row >= pivot
    weights = torch.where(kept, p_row, 0.0)
    return torch.where(kept, p_row / probability_sum(weights), 0.0)


def shifted(logits, maximum, t):
    """Softmax exponent in FP32: subtract first, scale first only where that overflows.

    d = l - max is exact for nearby logits, so d/T is used whenever d is finite. Only
    when |l - max| > FLT_MAX (d = -inf) does it use l/T - max/T, which then stays
    finite for large T; for small T both orders give -inf, i.e. probability 0. A masked
    logit (-inf) is not an overflow: it keeps d/T = -inf, i.e. probability 0.
    """
    d = logits - maximum
    overflow = torch.isneginf(d) & torch.isfinite(logits)
    return torch.where(overflow, logits / t - maximum / t, d / t)


def probability_row(logits, temperature, k, p):
    """Stable ID ties and exact integer top-p preceding masses."""
    logits = _fp32(logits)
    out = torch.zeros_like(logits)
    if temperature == 0:
        out[torch.argmax(logits)] = 1
        return out
    # Rank raw logits, then scale with the overflow-safe exponent (shifted).
    t = _fp32(temperature)
    order = torch.argsort(logits, descending=True, stable=True)
    kept = order[:k]
    maximum = logits[order[0]]
    weights = torch.zeros_like(logits)
    weights[kept] = _exp(shifted(logits[kept], maximum, t))
    if k < len(logits):
        tau = logits[order[k - 1]]
        above = logits > tau
        partials = torch.where(above, weights, _fp32(0))
        ties = k - int(torch.count_nonzero(above))
        tie_e = _exp(shifted(tau, maximum, t))
        total = probability_sum(partials) + _fp32(ties) * tie_e
    else:
        total = probability_sum(weights)
    probs = weights / total
    if p == 1:
        return probs
    masses = [fixed_mass(value) for value in probs]
    threshold = fixed_mass(p, ceiling=True)
    if sum(masses) < threshold:
        selected, preceding = torch.arange(len(probs)), sum(masses)
    else:
        order = torch.argsort(probs, descending=True, stable=True)
        preceding = 0
        selected = []
        for token in order.tolist():
            if preceding >= threshold:
                break
            selected.append(token)
            preceding += masses[token]
    scale = _uint64_to_float32(preceding) * _fp32(2.0**-62)
    out[selected] = probs[selected] / scale
    return out


class Provider:
    def run(self, mode, tensors, position=0):
        getattr(self, mode)(*tensors, position=position)

    def probabilities(self, logits, temperatures, top_k, top_p, slots, out, workspace, *, position):
        for b in range(len(slots)):
            out[b].zero_()
            if int(slots[b]) == -1:
                continue
            for i in range(logits.shape[1]):
                row = probability_row(
                    logits[b, i].float(),
                    temperatures[b],
                    int(top_k[b]),
                    top_p[b],
                )
                out[b, i].copy_(row)

    def cursors(self, anchors, prompts, slots, out, *, position):
        for b in range(len(slots)):
            slot = int(slots[b])
            out[b] = 0 if slot == -1 else int(anchors[b]) - int(prompts[slot]) + 1

    def candidates(
        self,
        scores,
        ids,
        temperatures,
        slots,
        seeds,
        cursors,
        work,
        out,
        oq,
        *position_tensor,
        position,
    ):
        for b in range(len(slots)):
            out[b], oq[b] = -1, 0
            slot = int(slots[b])
            if slot == -1:
                continue
            sorted_ids = ids[b]
            s = scores[b]
            t = float(temperatures[b])
            if t == 0:
                out[b] = sorted_ids[int(torch.argmax(s))]
            else:
                draw_position = int(position_tensor[0]) if position is None else position
                w = _exp(shifted(s, torch.max(s), _fp32(t)))
                q = w / candidate_cdf_parts(w)[1]
                oq[b].copy_(q)
                u = uniform(request_seed(seeds, slot), int(cursors[b]), PROPOSAL, draw_position)
                out[b] = sorted_ids[categorical(q, u, candidates=True)]

    def probabilities_min_p(self, *tensors, position):
        self.probabilities(*tensors, position=position)

    def min_p_renormalize(self, out, temperatures, min_p, slots, workspace, *, position):
        for b in range(len(slots)):
            if int(slots[b]) == -1 or temperatures[b] <= 0 or min_p[b] <= 0:
                continue
            for i in range(out.shape[1]):
                out[b, i].copy_(min_p_row(out[b, i], min_p[b]))

    def ordinary_min_p(self, *tensors, position):
        *args, min_p, _maxima = tensors
        self.sample(*args, position=position, domain=ORDINARY_NAMESPACE, min_p=min_p)

    def ordinary(self, *tensors, position):
        self.sample(*tensors, position=position, domain=ORDINARY_NAMESPACE)

    def sample(
        self,
        q,
        temperatures,
        slots,
        seeds,
        cursors,
        work,
        out,
        *,
        position,
        domain=PROPOSAL,
        min_p=None,
    ):
        for b in range(len(slots)):
            out[b].fill_(-1)
            slot = int(slots[b])
            if slot == -1:
                continue
            for i in range(q.shape[1]):
                w = q[b, i]
                if temperatures[b] == 0:
                    token = int(torch.argmax(w))
                else:
                    if min_p is not None and min_p[b] > 0:
                        # Both operands are FP32: exactly one rounding at the pivot.
                        pivot = min_p[b] * torch.max(w)
                        w = torch.where(w >= pivot, w, 0.0)
                    token = categorical(
                        w, uniform(request_seed(seeds, slot), int(cursors[b]), domain, i)
                    )
                out[b, i] = token

    def dense(self, p, x, q, temperatures, slots, seeds, cursors, work, out, counts, *, position):
        self._verify(p, x, q, None, temperatures, slots, seeds, cursors, out, counts)

    def point(self, p, x, temperatures, slots, seeds, cursors, work, out, counts, *, position):
        """Bitwise dense-one-hot emulation, materializing only a visited q row."""
        self._verify(p, x, None, None, temperatures, slots, seeds, cursors, out, counts)

    def sparse(
        self, p, x, q, ids, temperatures, slots, seeds, cursors, work, out, counts, *, position
    ):
        self._verify(p, x, q, ids, temperatures, slots, seeds, cursors, out, counts)

    def _verify(self, p, x, q, ids, temperatures, slots, seeds, cursors, out, counts):
        gamma, vocab = x.shape[1], p.shape[2]

        def proposal_row(b, a):
            if q is None:
                row = torch.zeros(vocab, dtype=torch.float32)
                row[int(x[b, a])] = 1
                return row
            if ids is None:
                return q[b, a]
            return sparse_dense(ids[b, a], q[b, a], vocab)

        for b in range(len(slots)):
            out[b].fill_(-1)
            counts[b] = 0
            slot = int(slots[b])
            if slot == -1:
                continue
            greedy = temperatures[b] == 0
            a = 0
            while a < gamma:
                token = int(x[b, a])
                if greedy:
                    accept = token == int(torch.argmax(p[b, a]))
                else:
                    qi = proposal_row(b, a)
                    u = uniform(request_seed(seeds, slot), int(cursors[b]), ACCEPT, a)
                    accept = _fp32(u) * qi[token] < p[b, a, token]
                if not accept:
                    break
                out[b, a] = token
                a += 1
            counts[b] = a
            if greedy:
                token = int(torch.argmax(p[b, a]))
            else:
                w = p[b, a]
                if a < gamma:
                    qi = proposal_row(b, a)
                    w = residual(w, qi)
                u = uniform(
                    request_seed(seeds, slot),
                    int(cursors[b]),
                    BONUS if a == gamma else CORRECTION,
                    0,
                )
                token = categorical(w, u)
            out[b, a] = token
