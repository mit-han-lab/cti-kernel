# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen pre-refactor NumPy implementation for differential tests.

CDF ABI r2.1 uses fixed 4096-token chunks of four 1024-token tiles.
Each tile has 256 logical lanes of four tokens; local prefixes are sequential,
lane/warp totals use Hillis-Steele scans. Chunk totals start at zero, row totals
sum chunks sequentially. Select the first chunk prefix > FP32(u * total), then
resume the tile scan from that chunk's exclusive row prefix. Rounding spill
uses the last positive token in the entire row. All operations round to FP32.
Candidate-only softmax/CDF retains its sequential 16-entry order.
Sparse collisions are summed in original column order before one subtraction.
"""

import pytest
import torch

from ctikernels.sampling._dflash import reference as actual
from ctikernels.sampling._dflash.reference import ACCEPT, BONUS, CORRECTION, PROPOSAL, uniform
from ctikernels.sampling._dflash.reference.ops import _uint64_to_float32

np = pytest.importorskip("numpy")


def _exp(value):
    """The reference's provider-specific exp: correctly rounded via FP64."""
    return np.exp(np.asarray(value, np.float64)).astype(np.float32)


CHUNK = 4096
TILE = 1024
VEC = 4


def request_seed(seeds, slot):
    return int(seeds[slot, 0]) | (int(seeds[slot, 1]) << 32)


def cdf_parts(weights, *, with_chunks=False):
    """Token prefixes and row total; optionally include the chunk-selection prefixes."""
    w = np.asarray(weights, dtype=np.float32)
    if not len(w):
        raise ValueError("CDF requires nonempty support")
    padded = np.zeros(((len(w) + TILE - 1) // TILE) * TILE, dtype=np.float32)
    padded[: len(w)] = w
    local = np.cumsum(padded.reshape(-1, 8, 32, VEC), axis=-1, dtype=np.float32)
    lanes = local[..., -1].copy()
    for offset in (1, 2, 4, 8, 16):
        lanes[..., offset:] = lanes[..., :-offset] + lanes[..., offset:]
    warps = lanes[..., -1].copy()
    for offset in (1, 2, 4):
        warps[:, offset:] = warps[:, :-offset] + warps[:, offset:]
    before_lane = np.zeros_like(lanes)
    before_lane[..., 1:] = lanes[..., :-1]
    before_warp = np.zeros_like(warps)
    before_warp[:, 1:] = warps[:, :-1]
    preceding = before_warp[..., None] + before_lane
    # Sum tile totals from zero separately in each chunk. A partial final chunk
    # has only its real tiles (zero-padding its final tile does not change sums).
    tile_totals = warps[:, -1]
    chunk_totals = np.array(
        [
            np.cumsum(tile_totals[i : i + CHUNK // TILE], dtype=np.float32)[-1]
            for i in range(0, len(tile_totals), CHUNK // TILE)
        ],
        dtype=np.float32,
    )
    chunk_prefixes = np.cumsum(chunk_totals, dtype=np.float32)
    before_tile = np.empty_like(tile_totals)
    for c in range(len(chunk_totals)):
        carry = chunk_prefixes[c - 1] if c else np.float32(0)
        for t in range(c * (CHUNK // TILE), min((c + 1) * (CHUNK // TILE), len(tile_totals))):
            before_tile[t] = carry
            carry = np.float32(carry + tile_totals[t])
    prefixes = before_tile[:, None, None, None] + (preceding[..., None] + local)
    parts = (prefixes.ravel()[: len(w)], chunk_prefixes[-1])
    return (*parts, chunk_prefixes) if with_chunks else parts


def candidate_cdf_parts(weights):
    """The existing small-support candidate ABI is sequential, in ascending ID order."""
    prefixes = np.cumsum(np.asarray(weights, dtype=np.float32), dtype=np.float32)
    return prefixes, prefixes[-1]


def categorical(weights, u, *, candidates=False):
    start = 0
    if candidates:
        prefixes, total = candidate_cdf_parts(weights)
    else:
        prefixes, total, chunks = cdf_parts(weights, with_chunks=True)
        eligible = np.flatnonzero(chunks > np.float32(np.float32(u) * total))
        if eligible.size:
            start = int(eligible[0]) * CHUNK
            prefixes = prefixes[start : start + CHUNK]
        else:
            prefixes = prefixes[:0]
    hits = np.flatnonzero(prefixes > np.float32(np.float32(u) * total))
    if hits.size:
        return start + int(hits[0])
    positive = np.flatnonzero(np.asarray(weights) > 0)
    if not positive.size:
        raise ValueError("categorical requires positive mass")
    return int(positive[-1])


def residual(p, q):
    r = np.maximum(np.asarray(p, np.float32) - np.asarray(q, np.float32), np.float32(0))
    return np.asarray(p, np.float32) if cdf_parts(r)[1] == 0 else r


def sparse_dense(ids, q, vocab):
    result = np.zeros(vocab, np.float32)
    for token, mass in zip(ids, q, strict=True):
        result[token] = np.float32(result[token] + mass)
    return result


def fixed_mass(value, *, ceiling=False):
    """Exact floor/ceil of a nonnegative FP32 value times 2^62, from its bits."""
    bits = int(np.float32(value).view(np.uint32))
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
    padded = np.zeros(CHUNK, dtype=np.float32)
    padded[: len(values)] = values
    lanes = np.zeros(512, dtype=np.float32)
    for tile in padded.reshape(-1, 512, 4):
        for j in range(4):
            lanes = np.float32(lanes + tile[:, j])
    warps = lanes.reshape(16, 32)
    for offset in (1, 2, 4, 8, 16):
        warps = np.float32(warps + warps[:, np.arange(32) ^ offset])
    totals = np.zeros(32, dtype=np.float32)
    totals[:16] = warps[:, 0]
    for offset in (1, 2, 4, 8, 16):
        totals = np.float32(totals + totals[np.arange(32) ^ offset])
    return totals[0]


def probability_sum(values):
    """Chunk tree partials combined sequentially in ascending chunk order."""
    total = np.float32(0)
    for start in range(0, len(values), CHUNK):
        total = np.float32(total + probability_chunk_sum(values[start : start + CHUNK]))
    return total


def shifted(logits, maximum, t):
    with np.errstate(over="ignore", invalid="ignore"):
        d = np.float32(logits - maximum)
        scaled = np.float32(np.float32(logits / t) - np.float32(maximum / t))
        overflow = np.isneginf(d) & np.isfinite(logits)
        return np.where(overflow, scaled, np.float32(d / t)).astype(np.float32)


def probability_row(logits, temperature, k, p):
    """FP32 numpy exp; stable ID ties and exact integer top-p preceding masses."""
    logits = np.asarray(logits, dtype=np.float32)
    out = np.zeros_like(logits)
    if temperature == 0:
        out[np.argmax(logits)] = 1
        return out
    t = np.float32(temperature)
    order = np.argsort(-logits, kind="stable")
    kept = order[:k]
    maximum = logits[order[0]]
    weights = np.zeros_like(logits)
    weights[kept] = _exp(shifted(logits[kept], maximum, t))
    if k < len(logits):
        tau = logits[order[k - 1]]
        above = logits > tau
        partials = np.where(above, weights, np.float32(0))
        ties = k - np.count_nonzero(above)
        tie_e = _exp(shifted(tau, maximum, t))
        total = np.float32(probability_sum(partials) + np.float32(np.float32(ties) * tie_e))
    else:
        total = probability_sum(weights)
    probs = np.float32(weights / total)
    if p == 1:
        return probs
    masses = [fixed_mass(value) for value in probs]
    threshold = fixed_mass(p, ceiling=True)
    # For integer preceding mass, preceding < real P iff preceding < ceil(P).
    if sum(masses) < threshold:
        selected, preceding = np.arange(len(probs)), sum(masses)
    else:
        order = np.argsort(-probs, kind="stable")
        preceding = 0
        selected = []
        for token in order:
            if preceding >= threshold:
                break
            selected.append(token)
            preceding += masses[token]
    scale = np.float32(np.uint64(preceding)) * np.float32(2.0**-62)
    out[selected] = np.float32(probs[selected] / scale)
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
                    logits[b, i].float().numpy(),
                    np.float32(temperatures[b]),
                    int(top_k[b]),
                    np.float32(top_p[b]),
                )
                out[b, i].copy_(torch.from_numpy(row))

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
            s = scores[b].numpy()
            t = float(temperatures[b])
            if t == 0:
                out[b] = sorted_ids[int(np.argmax(s))]
            else:
                draw_position = int(position_tensor[0]) if position is None else position
                w = _exp(shifted(s, np.max(s), np.float32(t)))
                q = w / candidate_cdf_parts(w)[1]
                oq[b].copy_(torch.from_numpy(q))
                u = uniform(request_seed(seeds, slot), int(cursors[b]), PROPOSAL, draw_position)
                out[b] = sorted_ids[categorical(q, u, candidates=True)]

    def sample(self, q, temperatures, slots, seeds, cursors, work, out, *, position):
        for b in range(len(slots)):
            out[b].fill_(-1)
            slot = int(slots[b])
            if slot == -1:
                continue
            for i in range(q.shape[1]):
                w = q[b, i].numpy()
                if temperatures[b] == 0:
                    token = int(np.argmax(w))
                else:
                    token = categorical(
                        w, uniform(request_seed(seeds, slot), int(cursors[b]), PROPOSAL, i)
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
                row = np.zeros(vocab, np.float32)
                row[int(x[b, a])] = 1
                return row
            if ids is None:
                return q[b, a].numpy()
            return sparse_dense(ids[b, a].numpy(), q[b, a].numpy(), vocab)

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
                    accept = np.float32(np.float32(u) * qi[token]) < p[b, a, token].numpy()
                if not accept:
                    break
                out[b, a] = token
                a += 1
            counts[b] = a
            if greedy:
                token = int(torch.argmax(p[b, a]))
            else:
                w = p[b, a].numpy()
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


# Keep the implementation above frozen: these tests compare its original FP32
# rounding points against the shipped torch provider, never against a tolerance.


def assert_bits(result, expected):
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(
        result.numpy().view(np.uint32), np.asarray(expected).view(np.uint32)
    )


@pytest.mark.parametrize("vocab", [1, 16, 4095, 4097, 8195, 50000])
@pytest.mark.parametrize("kind", ["random", "subnormal", "ties", "chunk_rounding"])
def test_numpy_reductions_and_sampling(vocab, kind):
    rng = np.random.default_rng(997 + vocab)
    if kind == "random":
        weights = np.ldexp(rng.random(vocab, dtype=np.float32), rng.integers(-24, 1, vocab))
    elif kind == "subnormal":
        weights = rng.integers(0, 256, vocab, dtype=np.uint32).view(np.float32)
    elif kind == "ties":
        weights = np.full(vocab, np.float32(1 / vocab))
    else:
        weights = np.zeros(vocab, np.float32)
        weights[0] = 1
        weights[1024::1024] = 2**-25
    inputs = torch.from_numpy(weights.copy())
    for got, expected in zip(
        actual.cdf_parts(inputs, with_chunks=True),
        cdf_parts(weights, with_chunks=True),
        strict=True,
    ):
        assert_bits(got, expected)
    assert_bits(actual.probability_sum(inputs), probability_sum(weights))
    for start in range(0, vocab, CHUNK):
        assert_bits(
            actual.probability_chunk_sum(inputs[start : start + CHUNK]),
            probability_chunk_sum(weights[start : start + CHUNK]),
        )
    small = weights[:16]
    for got, expected in zip(
        actual.candidate_cdf_parts(small.tolist()), candidate_cdf_parts(small), strict=True
    ):
        assert_bits(got, expected)
    for u in [0, 0.5, np.nextafter(np.float32(1), np.float32(0)), 1]:
        assert actual.categorical(inputs, float(u)) == categorical(weights, u)
        if small.any():
            assert actual.categorical(small.tolist(), float(u), candidates=True) == categorical(
                small, u, candidates=True
            )
    other = rng.random(vocab, dtype=np.float32)
    assert_bits(actual.residual(inputs, other.tolist()), residual(weights, other))
    assert_bits(actual.residual(inputs, inputs), residual(weights, weights))


def test_numpy_collision_order_and_fixed_masses():
    ids = [3, 0, 3, 3, 1, 1, 3]
    weights = np.array([1, 2**-149, 2**-24, 2**-24, 0.125, 0.125, 2**-24], np.float32)
    assert_bits(actual.sparse_dense(ids, weights.tolist(), 5), sparse_dense(ids, weights, 5))
    rng = np.random.default_rng(775)
    values = rng.integers(0, 0x3F800001, 1024, dtype=np.uint32).view(np.float32)
    for value in [*values, np.float32(2**-149), np.float32(1)]:
        for ceiling in [False, True]:
            assert actual.fixed_mass(float(value), ceiling=ceiling) == fixed_mass(
                value, ceiling=ceiling
            )


def test_numpy_single_round_uint64_conversion():
    # Values immediately either side of a binary32 midpoint lose the low bit
    # when first converted to binary64. Include both even/odd significands.
    values = [0, 1, 2**64 - 1]
    for exponent in range(24, 64):
        for significand in [2**23, 2**23 + 1, 2**24 - 1]:
            midpoint = (significand << (exponent - 23)) + (1 << (exponent - 24))
            values.extend(midpoint + delta for delta in [-1, 0, 1])
    rng = np.random.default_rng(123)
    values.extend(int(v) for v in rng.integers(0, 2**64, 1024, dtype=np.uint64))
    for value in values:
        assert_bits(_uint64_to_float32(value), np.float32(np.uint64(value)))


@pytest.mark.parametrize("vocab", [1, 16, 4095, 4097, 8195, 50000])
@pytest.mark.parametrize("kind", ["random", "ties", "tiny"])
def test_numpy_probability_filters(vocab, kind):
    rng = np.random.default_rng(123 + vocab)
    if kind == "random":
        logits = rng.normal(size=vocab).astype(np.float32)
    elif kind == "ties":
        logits = rng.integers(-2, 3, size=vocab).astype(np.float32)
    else:
        logits = rng.uniform(-104, -87, size=vocab).astype(np.float32)
        logits[0] = 0
    for temperature, k, p in [
        (0, 1, 0.5),
        (1, vocab, 1),
        (0.7, min(17, vocab), 1),
        (1, vocab, 0.5),
        (1, min(1025, vocab), np.nextafter(np.float32(1), np.float32(0))),
        (1, vocab, 2**-149),
    ]:
        assert_bits(
            actual.probability_row(logits.tolist(), temperature, k, p),
            probability_row(logits, temperature, k, p),
        )


def test_numpy_first_max_and_stable_ties():
    for values in [[-0.0, 0.0, -0.0], [3.0, 3.0, 1.0], [-np.inf, 1.0, 1.0], [1.0, np.nan, np.nan]]:
        assert int(torch.argmax(torch.tensor(values))) == int(
            np.argmax(np.asarray(values, np.float32))
        )
    values = np.array([2.0, 1.0, 2.0, -0.0, 0.0, -np.inf], np.float32)
    assert (
        torch.argsort(torch.from_numpy(values), descending=True, stable=True).tolist()
        == np.argsort(-values, kind="stable").tolist()
    )


def test_numpy_candidate_provider():
    generator = torch.Generator().manual_seed(43)
    scores = torch.randn(4, 16, generator=generator)
    ids = torch.arange(16, dtype=torch.int32).repeat(4, 1)
    temperatures = torch.tensor([1.0, 0.7, 0.0, float("nan")])
    slots = torch.tensor([1, 0, 2, -1], dtype=torch.int32)
    seeds = torch.tensor([[0, 1], [0xFFFFFFFF, 0xFFFFFFFF], [7, 0]], dtype=torch.uint32)
    cursors = torch.tensor([2**32 + 1, 9, 17, -1])
    work = torch.empty(0, dtype=torch.uint8)
    out, oq = torch.empty(4, dtype=torch.int32), torch.empty_like(scores)
    expected, expected_q = torch.empty_like(out), torch.empty_like(oq)
    args = scores, ids, temperatures, slots, seeds, cursors, work
    actual.Provider().candidates(*args, out, oq, position=2)
    Provider().candidates(*args, expected, expected_q, position=2)
    assert torch.equal(out, expected)
    assert_bits(oq, expected_q.numpy())


@pytest.mark.parametrize("vocab", [1, 4097, 8195])
@pytest.mark.parametrize("mode", ["dense", "point", "sparse"])
def test_numpy_dispatch_acceptance_and_tokens(vocab, mode):
    generator = torch.Generator().manual_seed(vocab)
    p = torch.rand(4, 4, vocab, generator=generator)
    p /= p.sum(-1, keepdim=True)
    q = torch.rand(4, 3, vocab, generator=generator)
    q /= q.sum(-1, keepdim=True)
    x = torch.randint(vocab, (4, 3), generator=generator, dtype=torch.int32)
    ids = torch.randint(vocab, (4, 3, 16), generator=generator, dtype=torch.int32)
    sparse_q = torch.rand(4, 3, 16, generator=generator)
    sparse_q /= sparse_q.sum(-1, keepdim=True)
    if mode == "sparse":
        x = ids[..., 0].clone()
    temperatures = torch.tensor([1.0, 0.0, 1.0, float("nan")])
    slots = torch.tensor([1, 0, 2, -1], dtype=torch.int32)
    seeds = torch.tensor([[0, 1], [0xFFFFFFFF, 0xFFFFFFFF], [7, 0]], dtype=torch.uint32)
    cursors = torch.tensor([2**32 + 1, 9, 17, -1])
    out, expected = torch.empty(4, 4, dtype=torch.int32), torch.empty(4, 4, dtype=torch.int32)
    counts, expected_counts = torch.empty(4, dtype=torch.int32), torch.empty(4, dtype=torch.int32)
    work = torch.empty(0, dtype=torch.uint8)
    head = {"dense": (p, x, q), "point": (p, x), "sparse": (p, x, sparse_q, ids)}[mode]
    tail = temperatures, slots, seeds, cursors, work
    getattr(actual.Provider(), mode)(*head, *tail, out, counts, position=0)
    getattr(Provider(), mode)(*head, *tail, expected, expected_counts, position=0)
    assert torch.equal(out, expected)
    assert torch.equal(counts, expected_counts)
    actual.Provider().sample(q, *tail, out[:, :3], position=0)
    Provider().sample(q, *tail, expected[:, :3], position=0)
    assert torch.equal(out[:, :3], expected[:, :3])
    assert actual.request_seed(seeds, 1) == request_seed(seeds.numpy(), 1) == 2**64 - 1


@pytest.mark.parametrize("vocab", [1, 4095, 4097, 8195, 50000])
def test_numpy_exact_exponential_filter_boundaries(vocab):
    # exp(0)=1 and exp(-inf)=0 isolate all filter/reduction rounding points
    # from the different transcendental approximations in NumPy and torch.
    logits = np.zeros(vocab, np.float32)
    if vocab > 1:
        logits[1::7] = -np.inf
    for k in sorted({1, min(1024, vocab), min(1025, vocab), vocab}):
        for p in [
            2**-149,
            np.nextafter(np.float32(0.5), np.float32(0)),
            0.5,
            np.nextafter(np.float32(0.5), np.float32(1)),
            0.95,
            1,
        ]:
            assert_bits(
                actual.probability_row(logits.tolist(), 1, k, p), probability_row(logits, 1, k, p)
            )


def test_numpy_signed_zero_residual():
    for p, q in [([-0.0, 1.0], [0.0, 0.0]), ([0.0, 1.0], [-0.0, 0.0]), ([-0.0, 1.0], [0.0, 1.0])]:
        assert_bits(actual.residual(p, q), residual(p, q))
