# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU arithmetic, admission and exact gamma=2/V=3 probability enumeration."""

from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from math import prod

import pytest
import torch

from ctikernels.sampling._dflash import RequestLocalChainSampling, validate_filters
from ctikernels.sampling._dflash.reference import (
    categorical,
    cdf_parts,
    residual,
    sparse_dense,
)

from .dflash_oracle import acceptance_residual_fp64, point_mass_acceptance_residual_fp64


def fp32(value):
    return torch.as_tensor(value, dtype=torch.float32)


def seed_words(values):
    """Encode fixture keys in the engine/K1 uint32[S,2] ABI."""
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return torch.tensor([[s & 0xFFFFFFFF, s >> 32] for s in values], dtype=torch.uint32).reshape(
        -1, 2
    )


def buffers(b=4, gamma=2, vocab=3, device="cpu"):
    op = RequestLocalChainSampling(device, gamma + 1, vocab)
    generator = torch.Generator().manual_seed(123)
    p = torch.rand(b, gamma + 1, vocab, generator=generator)
    q = torch.rand(b, gamma, vocab, generator=generator)
    p /= p.sum(-1, keepdim=True)
    q /= q.sum(-1, keepdim=True)
    data = {
        "p": p,
        "q": q,
        "drafts": torch.zeros((b, gamma), dtype=torch.int32),
        "temperatures": torch.ones(b),
        "request_indices": torch.arange(b, dtype=torch.int32),
        "seeds": seed_words(torch.arange(b, dtype=torch.int64) * 0x123456789 + 42),
        "cursors": torch.arange(b, dtype=torch.int64) + 2**32 + 7,
        "workspace": torch.empty(op.workspace_size(b), dtype=torch.uint8),
        "out_tokens": torch.empty((b, gamma + 1), dtype=torch.int32),
        "accepted_counts": torch.empty(b, dtype=torch.int32),
    }
    return op, {k: t.to(device) for k, t in data.items()}


def verify(op, data, sparse=False, *, point=False):
    kw = {
        k: data[k]
        for k in (
            "request_indices",
            "seeds",
            "cursors",
            "workspace",
            "out_tokens",
            "accepted_counts",
        )
    }
    if point:
        op.verify_point_mass(data["p"], data["drafts"], data["temperatures"], **kw)
    elif sparse:
        op.verify_sparse(
            data["p"], data["drafts"], data["cand_ids"], data["cand_q"], data["temperatures"], **kw
        )
    else:
        op.verify_dense(data["p"], data["drafts"], data["q"], data["temperatures"], **kw)


def sparsify(data):
    b, g, v = data["q"].shape
    assert v <= 8
    ids = torch.zeros((b, g, 16), dtype=torch.int32, device=data["q"].device)
    q = torch.zeros((b, g, 16), device=data["q"].device)
    for k in range(v):
        ids[:, :, 2 * k : 2 * k + 2] = k
        q[:, :, 2 * k : 2 * k + 2] = data["q"][:, :, k : k + 1] / 2
    data["cand_ids"], data["cand_q"] = ids, q


@pytest.mark.parametrize("k,p", [(None, None), (0, 0), (-1, -1), (30, 1), (100, 1)])
def test_identity_filters(k, p):
    assert validate_filters(k, p, 30) == (30, 1)


@pytest.mark.parametrize("k,p", [(1, 1), (29, 1), (0, 0.9), (2, 0.7)])
def test_filter_policy_before_greedy(k, p):
    # Identical policy applies to T=0 and T>0; no temperature bypass is exposed.
    assert validate_filters(k, p, 30) == (30 if k == 0 else k, p)


@pytest.mark.parametrize("p", [float("nan"), float("inf"), 1.01])
def test_invalid_filters(p):
    with pytest.raises(ValueError):
        validate_filters(None, p, 30)


def test_cdf_edges_and_subnormal_residual():
    assert categorical([0, 0.5, 0.5, 0], 0) == 1
    assert categorical([0, 0.5, 0.5, 0], 0.5) == 2
    assert categorical([0, 0.5, 0.5, 0], 1) == 2  # rounding spill fallback
    tiny = torch.nextafter(fp32(0), fp32(1))
    r = residual([tiny, 1], [0, 1])
    assert torch.equal(r, torch.tensor([tiny, 0]))
    assert categorical(r, 0.5) == 0
    assert torch.equal(residual([0.5, 0.5], [0.5, 0.5]), torch.tensor([0.5, 0.5]))


def scalar_tiled_cdf(weights):
    """Independent scalar oracle, including every shuffle-stage rounding."""

    def scan(values):
        result = list(values)
        offset = 1
        while offset < len(result):
            old = result.copy()
            for lane in range(offset, len(result)):
                result[lane] = fp32(old[lane - offset] + old[lane])
            offset *= 2
        return result

    zero = fp32(0)
    tile_totals, tile_prefixes = [], []
    for start in range(0, len(weights), 1024):
        prefixes = []
        tile = list(weights[start : start + 1024])
        tile += [zero] * (1024 - len(tile))
        local = []
        for lane in range(256):
            running, lane_prefixes = zero, []
            for value in tile[lane * 4 : lane * 4 + 4]:
                running = fp32(running + value)
                lane_prefixes.append(running)
            local.append(lane_prefixes)
        lanes = [scan([local[w * 32 + lane][-1] for lane in range(32)]) for w in range(8)]
        warps = scan([w[-1] for w in lanes])
        for lane in range(256):
            warp, lane_in_warp = divmod(lane, 32)
            before = fp32(
                (warps[warp - 1] if warp else zero)
                + (lanes[warp][lane_in_warp - 1] if lane_in_warp else zero)
            )
            for value in local[lane]:
                prefixes.append(fp32(before + value))
        tile_totals.append(warps[-1])
        tile_prefixes.append(prefixes)
    chunks = []
    for start in range(0, len(tile_totals), 4):
        total = zero
        for value in tile_totals[start : start + 4]:
            total = fp32(total + value)
        chunks.append(total)
    prefixes, chunk_prefixes, total = [], [], zero
    for c, value in enumerate(chunks):
        carry = total
        total = fp32(total + value)
        chunk_prefixes.append(total)
        for t in range(c * 4, min(c * 4 + 4, len(tile_totals))):
            prefixes.extend(fp32(carry + value) for value in tile_prefixes[t])
            carry = fp32(carry + tile_totals[t])
    return (
        torch.tensor(prefixes[: len(weights)], dtype=torch.float32),
        total,
        torch.tensor(chunk_prefixes),
    )


@pytest.mark.parametrize(
    "vocab", [1, 3, 4, 5, 127, 128, 129, 1023, 1024, 1025, 2049, 4095, 4096, 4097, 8193, 248320]
)
def test_tiled_cdf_fp32_order(vocab):
    rng = torch.Generator().manual_seed(919)
    # Wide dynamic range distinguishes reassociation and tile boundary carries.
    w = torch.ldexp(
        torch.rand(vocab, generator=rng), torch.randint(-24, 1, (vocab,), generator=rng)
    )
    expected, expected_total, chunks = scalar_tiled_cdf(w)
    prefix, total, actual_chunks = cdf_parts(w, with_chunks=True)
    assert torch.equal(actual_chunks, chunks)
    assert torch.equal(prefix, expected)
    assert total == expected_total
    for u in (0, 0.5, torch.nextafter(fp32(1), fp32(0))):
        threshold = fp32(fp32(u) * expected_total)
        selected = torch.nonzero(chunks > threshold).flatten()
        start = int(selected[0]) * 4096 if len(selected) else len(w)
        hits = torch.nonzero(expected[start : start + 4096] > threshold).flatten()
        winner = start + int(hits[0]) if len(hits) else int(torch.nonzero(w > 0).flatten()[-1])
        assert categorical(w, u) == winner


def test_candidate_cdf_keeps_sequential_order():
    from ctikernels.sampling._dflash.reference import candidate_cdf_parts

    weights = torch.tensor([1] + [2**-24] * 15, dtype=torch.float32)
    prefixes, total = candidate_cdf_parts(weights)
    assert total == 1 and torch.equal(prefixes, torch.ones(16, dtype=torch.float32))
    assert cdf_parts(weights)[1] > total  # deliberately different dense ABI
    assert categorical(weights, 1, candidates=True) == 15


def test_tiled_cdf_boundary_zero_and_subnormal():
    tiny = torch.nextafter(fp32(0), fp32(1))
    for token in (3, 4, 127, 128, 1023, 1024, 2048, 4095, 4096, 8192):
        w = torch.zeros(8195, dtype=torch.float32)
        w[token] = tiny
        prefix, total = cdf_parts(w)
        assert total == tiny
        assert (prefix[:token] == 0).all() and (prefix[token:] == tiny).all()
        assert categorical(w, 0) == categorical(w, 1) == token


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_every_acceptance_length_and_padding(sparse, greedy):
    g = 7
    op, data = buffers(g + 2, g, 3)
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    data["p"].zero_()
    data["p"][:, :, 0] = 1
    for a in range(g):
        data["p"][a, a, 0], data["p"][a, a, 1] = 0, 1
    data["temperatures"].fill_(0 if greedy else 1)
    data["request_indices"][-1] = -1
    data["p"][-1].fill_(float("nan"))
    data["q"][-1].fill_(float("nan"))
    data["drafts"][-1].fill_(-999)
    data["temperatures"][-1] = float("nan")
    data["cursors"][-1] = -1
    if sparse:
        sparsify(data)
    verify(op, data, sparse)
    assert data["accepted_counts"].tolist() == list(range(g + 1)) + [0]
    for a in range(g + 1):
        assert data["out_tokens"][a, :a].tolist() == [0] * a
        assert data["out_tokens"][a, a] == (1 if a < g else 0)
        assert (data["out_tokens"][a, a + 1 :] == -1).all()
    assert (data["out_tokens"][-1] == -1).all()


def test_all_padding_no_live_state():
    op, data = buffers()
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.empty((0, 2), dtype=torch.uint32)
    data["p"].fill_(float("nan"))
    verify(op, data)
    assert (data["out_tokens"] == -1).all()
    assert (data["accepted_counts"] == 0).all()


def test_dense_sparse_and_batch_invariance():
    op, data = buffers()
    op.sample_dense_proposals(
        data["q"],
        data["temperatures"],
        **{k: data[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
        out_ids=data["drafts"],
    )
    verify(op, data)
    expected, counts = data["out_tokens"].clone(), data["accepted_counts"].clone()
    sparsify(data)
    verify(op, data, True)
    assert torch.equal(expected, data["out_tokens"])
    assert torch.equal(counts, data["accepted_counts"])
    for row in range(4):
        one_op, one = buffers(1)
        for name in ("p", "q", "drafts", "temperatures", "cursors"):
            one[name].copy_(data[name][row : row + 1])
        one["seeds"][0] = data["seeds"][row]
        verify(one_op, one)
        assert torch.equal(one["out_tokens"][0], expected[row])


def test_candidates_order_greedy_padding():
    op = RequestLocalChainSampling("cpu", 8, 32)
    scores = torch.zeros(3, 16)
    ids = torch.arange(16, dtype=torch.int32).repeat(3, 1)
    oi, oq = (
        torch.empty(3, dtype=torch.int32),
        torch.empty(3, 16),
    )
    op.sample_candidates(
        scores,
        ids,
        torch.tensor([0.0, 1.0, float("nan")]),
        6,
        request_indices=torch.tensor([1, 0, -1], dtype=torch.int32),
        seeds=seed_words([7, 9]),
        cursors=torch.tensor([2**32, 3, -1]),
        workspace=torch.empty(0, dtype=torch.uint8),
        out_ids=oi,
        out_q=oq,
    )
    assert op.provider == "reference"
    assert oi[0] == 0 and oi[2] == -1
    assert (oq[0] == 0).all() and (oq[1] == 1 / 16).all() and (oq[2] == 0).all()


def test_candidate_device_position_and_full_width_keys():
    from ctikernels.sampling import RequestLocalChainSampling as PublicSampling
    from ctikernels.sampling._dflash.reference import PROPOSAL, uniform

    assert PublicSampling is RequestLocalChainSampling
    op = PublicSampling("cpu", 8, 32)
    scores = torch.zeros(1, 16)
    ids = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    out, q = torch.empty(1, dtype=torch.int32), torch.empty_like(scores)
    kwargs = {
        "request_indices": torch.zeros(1, dtype=torch.int32),
        "seeds": seed_words([0xFFFFFFFFFFFFFFFF]),
        "cursors": torch.tensor([2**32 + 23]),
        "workspace": torch.empty(0, dtype=torch.uint8),
        "out_ids": out,
        "out_q": q,
    }
    position = torch.tensor(0, dtype=torch.int32)
    for i in (0, 6):
        position.fill_(i)
        op.sample_candidates(scores, ids, torch.ones(1), position, **kwargs)
        expected = categorical([1 / 16] * 16, uniform(0xFFFFFFFFFFFFFFFF, 2**32 + 23, PROPOSAL, i))
        assert out.item() == expected
    for invalid in (-1, 7, 2**32, True):
        with pytest.raises(ValueError, match="position_i"):
            op.sample_candidates(scores, ids, torch.ones(1), invalid, **kwargs)
    with pytest.raises(TypeError, match="position_i"):
        op.sample_candidates(scores, ids, torch.ones(1), position.long(), **kwargs)
    with pytest.raises(ValueError, match="position_i"):
        op.sample_candidates(scores, ids, torch.ones(1), position.reshape(1), **kwargs)


@pytest.mark.parametrize("name", ["scores", "temperatures", "out_q"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_candidate_grad_rejected_before_provider(name, grad_enabled):
    op = RequestLocalChainSampling("cpu", 8, 32)
    args = {
        "scores": torch.zeros(1, 16),
        "ids": torch.arange(16, dtype=torch.int32).reshape(1, 16),
        "temperatures": torch.ones(1),
        "position_i": 0,
        "request_indices": torch.zeros(1, dtype=torch.int32),
        "seeds": seed_words([1]),
        "cursors": torch.zeros(1, dtype=torch.int64),
        "workspace": torch.empty(0, dtype=torch.uint8),
        "out_ids": torch.empty(1, dtype=torch.int32),
        "out_q": torch.empty(1, 16),
    }
    args[name].requires_grad_(True)
    op._provider.run = lambda *a, **kw: pytest.fail("provider reached with grad tensor")
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="requires grad"):
        op.sample_candidates(**args)


def test_workspace_layout():
    op = RequestLocalChainSampling("cpu", 8, 17)
    layout = op.workspace_layout(3)
    assert layout["total_bytes"] == op.workspace_size(3) == 256
    assert layout["base_alignment"] == 4
    assert layout["regions"] == (
        {
            "name": "chunk_partials",
            "payload_bytes": 192,
            "alignment": 4,
            "offset": 0,
            "aligned_extent": 256,
        },
    )
    assert op.workspace_layout(3, "candidates")["regions"] == ()


@pytest.mark.parametrize("block", [pytest.param(1, id="q1"), 2, 8, 16, 32])
@pytest.mark.parametrize("vocab", [1, 4095, 4096, 4097, 248320])
@pytest.mark.parametrize("batch", [0, 1, 3, 8])
def test_chunk_workspace_capacity(block, vocab, batch):
    op = RequestLocalChainSampling("cpu", block, vocab)
    payload = batch * max(2, block) * ((vocab + 4095) // 4096) * 8
    for representation in ("dense", "sparse"):
        layout = op.workspace_layout(batch, representation)
        assert layout["total_bytes"] == (payload + 255) // 256 * 256
        assert sum(r["payload_bytes"] for r in layout["regions"]) == payload
    assert op.launch_counts["verify_dense"] == op.launch_counts["verify_sparse"] == 2
    assert op.launch_counts["sample_dense_proposals"] == (0 if block == 1 else 2)


def test_chunk_selection_and_whole_row_rounding_fallback():
    w = torch.zeros(3 * 4096 + 3, dtype=torch.float32)
    w[7], w[4096 + 5], w[8192 + 1] = 0.25, 0.25, 0.5
    assert categorical(w, 0) == 7
    assert categorical(w, 0.25) == 4101  # strict chunk boundary
    assert categorical(w, 0.5) == 8193
    assert categorical(w, 1) == 8193  # trailing zero chunks do not supply fallback


def test_chunk_gating_precedes_reassociated_token_prefix():
    # A token prefix can exceed its chunk total due to reassociation. At equality
    # the first chunk must be skipped even when it contains a token prefix hit.
    rng = torch.Generator().manual_seed(17)
    for _ in range(100):
        first = torch.ldexp(
            torch.rand(4096, generator=rng), torch.randint(-24, 1, (4096,), generator=rng)
        )
        prefix, total = cdf_parts(first)
        if prefix[-1] > total:
            break
    else:
        pytest.fail("fixture must expose a reassociated prefix exceeding chunk total")
    w = torch.cat((first, torch.tensor([total], dtype=torch.float32)))
    assert categorical(w, 0.5) == 4096


def test_chunk_partials_reset_fp32_accumulation():
    # Sequential global tiles would lose every 2^-25 tile after the leading 1.
    # Summing four tiles inside each later chunk preserves their combined mass.
    w = torch.zeros(4096 * 3, dtype=torch.float32)
    w[0] = 1
    w[4096::1024] = 2**-25
    _, total, chunks = cdf_parts(w, with_chunks=True)
    assert torch.equal(chunks, torch.tensor([1, 1 + 2**-23, 1 + 2**-22], dtype=torch.float32))
    assert total == fp32(1 + 2**-22)


def test_selected_chunk_without_token_hit_uses_global_last_positive():
    w = torch.zeros(8192 + 8, dtype=torch.float32)
    w[0] = 1
    w[4096:8192:1024] = 2**-25
    w[-1] = 2**-26
    prefix, total, chunks = cdf_parts(w, with_chunks=True)
    u = fp32(1 - 2**-23)
    threshold = fp32(u * total)
    assert threshold == 1 and chunks[1] > threshold
    assert (prefix[4096:8192] <= threshold).all()
    # Adding the four small tile totals to the exclusive prefix loses them,
    # although their chunk partial was big enough to select this chunk.
    assert categorical(w, u) == len(w) - 1


@pytest.mark.parametrize(
    "problem", ["grad", "dtype", "shape", "layout", "workspace", "alias", "device"]
)
def test_validation_before_provider(problem):
    op, data = buffers()

    def fail(*args, **kwargs):
        pytest.fail("provider reached with invalid input")

    op._provider.run = fail
    if problem == "grad":
        data["p"].requires_grad_(True)
    elif problem == "dtype":
        data["drafts"] = data["drafts"].long()
    elif problem == "shape":
        data["cursors"] = data["cursors"][:1]
    elif problem == "layout":
        data["p"] = data["p"].transpose(1, 2)
    elif problem == "workspace":
        data["workspace"] = data["workspace"][:1]
    elif problem == "alias":
        data["accepted_counts"] = data["request_indices"]
    elif problem == "device":
        data["p"] = data["p"].to("meta")
    with torch.no_grad(), pytest.raises((TypeError, ValueError)):
        verify(op, data)


@pytest.mark.parametrize("sparse", [False, True])
def test_exact_rational_chain_law(sparse):
    p = [[F(1, 2), F(1, 3), F(1, 6)], [F(1, 4), F(1, 4), F(1, 2)], [F(1, 3)] * 3]

    def proposal(previous):
        return (
            [F(1, 4), F(1, 2), F(1, 4)] if sparse and previous == 1 else [F(1, 2), F(1, 4), F(1, 4)]
        )

    observed = defaultdict(F)
    for draft in product(range(3), repeat=2):
        qs = [proposal(-1), proposal(draft[0])]
        draft_mass = prod(qs[i][draft[i]] for i in range(2))
        alive = F(1)
        for a in range(3):
            if a < 2:
                accept = min(F(1), p[a][draft[a]] / qs[a][draft[a]])
                stop = alive * (1 - accept)
                r = [max(p[a][t] - qs[a][t], F(0)) for t in range(3)]
                mass = sum(r)
                correction = [v / mass for v in r] if mass else p[a]
                alive *= accept
                pa, ra = acceptance_residual_fp64(
                    torch.tensor([float(v) for v in p[a]], dtype=torch.float64),
                    torch.tensor([float(v) for v in qs[a]], dtype=torch.float64),
                )
                assert abs(float(pa) - float(sum(min(p[a][t], qs[a][t]) for t in range(3)))) < 1e-15
                assert torch.allclose(
                    ra,
                    torch.tensor([float(v) for v in correction], dtype=torch.float64),
                    atol=1e-15,
                    rtol=0,
                )
                if sparse:
                    # Duplicate columns emulate the same conditional proposal law.
                    ids = torch.tensor([0, 1, 2, 0])
                    vals = torch.tensor(
                        [
                            float(qs[a][0] / 2),
                            float(qs[a][1]),
                            float(qs[a][2]),
                            float(qs[a][0] / 2),
                        ],
                        dtype=torch.float32,
                    )
                    assert torch.equal(
                        sparse_dense(ids, vals, 3),
                        torch.tensor([float(v) for v in qs[a]], dtype=torch.float32),
                    )
            else:
                stop, correction = alive, p[2]
            for suffix in product(range(3), repeat=3 - a):
                probability = draft_mass * stop * correction[suffix[0]]
                probability *= prod(p[a + j][suffix[j]] for j in range(1, len(suffix)))
                observed[draft[:a] + suffix] += probability
    assert len(observed) == 27
    for tokens, probability in observed.items():
        assert probability == prod(p[i][tokens[i]] for i in range(3))


def test_cursor_lifecycle_and_slot_reuse():
    op = RequestLocalChainSampling("cpu", 8, 3)
    slots = torch.tensor([1, -1, 0], dtype=torch.int32)
    prompts = torch.tensor([17, 2**32 + 100], dtype=torch.int64)
    anchors = torch.tensor([2**32 + 100, -999, 18], dtype=torch.int64)
    cursors = torch.empty(3, dtype=torch.int64)
    workspace = torch.empty(0, dtype=torch.uint8)
    op.prepare_cursors(
        anchors, prompts, request_indices=slots, workspace=workspace, out_cursors=cursors
    )
    assert cursors.tolist() == [1, 0, 2]
    anchors += torch.tensor([4, 0, 8])  # committed m=a+1, not predicted block width
    op.prepare_cursors(
        anchors, prompts, request_indices=slots, workspace=workspace, out_cursors=cursors
    )
    assert cursors.tolist() == [5, 0, 10]
    prompts[1], anchors[0] = 9, 9  # new request in a reused slot
    op.prepare_cursors(
        anchors, prompts, request_indices=slots, workspace=workspace, out_cursors=cursors
    )
    assert cursors.tolist() == [1, 0, 10]


def test_temperature_only_controls_greedy(monkeypatch):
    from ctikernels.sampling._dflash.reference import ops as reference

    op, data = buffers(1)
    data["q"].zero_()
    data["q"][:, :, 1] = 1  # caller's top-k=1 output (seeded sampling still draws)
    calls = []

    def draw(*args):
        calls.append(args)
        return 0.25

    monkeypatch.setattr(reference, "uniform", draw)
    common = {k: data[k] for k in ("request_indices", "seeds", "cursors", "workspace")}
    op.sample_dense_proposals(data["q"], data["temperatures"], **common, out_ids=data["drafts"])
    assert len(calls) == 2
    calls.clear()
    data["temperatures"].zero_()
    op.sample_dense_proposals(data["q"], data["temperatures"], **common, out_ids=data["drafts"])
    assert calls == []


@pytest.mark.parametrize("block_size", [2, 8, 16])
@pytest.mark.parametrize("greedy", [False, True])
def test_point_mass_every_prefix(block_size, greedy):
    g = block_size - 1
    op, data = buffers(g + 2, g, 3)
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    data["p"].zero_()
    data["p"][:, :, 0] = 1
    for a in range(g):
        data["p"][a, a, 0], data["p"][a, a, 1] = 0, 1
    data["temperatures"].fill_(0 if greedy else 1)
    data["request_indices"][-1] = -1
    for name in ("p", "q", "temperatures"):
        data[name][-1].fill_(float("nan"))
    data["drafts"][-1].fill_(-999)
    data["cursors"][-1] = -1
    verify(op, data)
    expected = data["out_tokens"].clone()
    verify(op, data, point=True)
    assert torch.equal(data["out_tokens"], expected)
    assert data["accepted_counts"].tolist() == list(range(g + 1)) + [0]
    for a in range(g + 1):
        assert data["out_tokens"][a, :a].tolist() == [0] * a
        assert data["out_tokens"][a, a] == (1 if a < g else 0)
        assert (data["out_tokens"][a, a + 1 :] == -1).all()
    assert (data["out_tokens"][-1] == -1).all()


@pytest.mark.parametrize("block_size", [2, 8, 16])
@pytest.mark.parametrize("vocab", [1, 3, 4097, 8193])
def test_point_mass_random_dense_equivalence(block_size, vocab):
    op, data = buffers(8, block_size - 1, vocab)
    data["drafts"].copy_(data["q"].argmax(-1))
    data["q"].zero_().scatter_(-1, data["drafts"].long().unsqueeze(-1), 1)
    # Mixed greedy/random temperatures, including smallest-ID ties.
    data["temperatures"][:2] = 0
    data["p"][0].fill_(1 / vocab)
    verify(op, data)
    expected, counts = data["out_tokens"].clone(), data["accepted_counts"].clone()
    verify(op, data, point=True)
    assert torch.equal(data["out_tokens"], expected)
    assert torch.equal(data["accepted_counts"], counts)
    assert op.launch_counts["verify_point_mass"] == 2


@pytest.mark.parametrize("draft", [0, 1, 2])
@pytest.mark.parametrize("row", [[0.25, 0.5, 0.25], [0, 1, 0], [2**-149, 1, 0]])
def test_point_mass_exact_oracle(row, draft):
    p = torch.tensor(row, dtype=torch.float32)
    acceptance, r = point_mass_acceptance_residual_fp64(p, torch.tensor(draft))
    exact = [F(float(value)) for value in p]
    assert acceptance.item() == float(exact[draft])
    residual_exact = [mass if t != draft else F(0) for t, mass in enumerate(exact)]
    total = sum(residual_exact)
    expected = [mass / total for mass in residual_exact] if total else exact
    assert torch.equal(r, torch.tensor([float(value) for value in expected], dtype=torch.float64))
    q = torch.zeros_like(p)
    q[draft] = 1
    dense_acceptance, dense_r = acceptance_residual_fp64(p, q)
    assert torch.equal(acceptance, dense_acceptance)
    assert torch.equal(r, dense_r)
    raw = residual(p, q)
    assert torch.equal(
        raw, torch.tensor([float(v) for v in residual_exact] if total else row, dtype=torch.float32)
    )


def test_point_mass_zero_residual_fallback(monkeypatch):
    # p(d)=1 cannot reject for an ABI uniform in [0,1). Inject the upper
    # endpoint to exercise the otherwise unreachable defensive fallback.
    import ctikernels.sampling._dflash.reference.ops as reference

    monkeypatch.setattr(reference, "uniform", lambda *args: fp32(1))
    op, data = buffers(1, 1, 3)
    data["p"].zero_()
    data["p"][:, :, 0] = 1
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    verify(op, data)
    expected = data["out_tokens"].clone()
    verify(op, data, point=True)
    assert torch.equal(data["out_tokens"], expected)
    assert data["out_tokens"].tolist() == [[0, -1]]
    assert data["accepted_counts"].tolist() == [0]


@pytest.mark.parametrize("name", ["p", "temperatures"])
def test_point_mass_requires_grad(name):
    op, data = buffers()
    data[name].requires_grad_(True)
    with pytest.raises(ValueError, match="requires grad"):
        verify(op, data, point=True)


@pytest.mark.parametrize("mass", [0, 0.5, 1])
@pytest.mark.parametrize("coin", [0, 0.5, 1 - 2**-24])
def test_point_mass_acceptance_boundary(monkeypatch, mass, coin):
    import ctikernels.sampling._dflash.reference.ops as reference
    from ctikernels.sampling._dflash.reference import ACCEPT

    monkeypatch.setattr(
        reference,
        "uniform",
        lambda seed, cursor, domain, position: fp32(coin if domain == ACCEPT else 0.5),
    )
    op, data = buffers(1, 1, 3)
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    data["p"][0, 0] = torch.tensor([mass, 1 - mass, 0])
    verify(op, data)
    expected = data["out_tokens"].clone()
    verify(op, data, point=True)
    assert torch.equal(data["out_tokens"], expected)
    assert data["accepted_counts"].item() == int(coin < mass)


def test_residual_rounds_negative_zero_to_positive_zero():
    result = residual([-0.0, 1], [0.0, 0])
    assert result.view(torch.int32).tolist() == [0, 0x3F800000]
    # The zero-mass fallback returns p unchanged, including its signed zeros.
    fallback = residual([-0.0, 1], [0.0, 1])
    assert fallback.view(torch.int32).tolist() == [-0x80000000, 0x3F800000]


def overflow_candidate_data():
    """Candidate scores whose s/T overflows FP32; exact q from (s - max)/T."""
    temperatures = torch.tensor([2.0**-149, 2.0**-140, 2.0**-126, 2.0**-149])
    scores = torch.full((4, 16), -3e38)
    expected = torch.zeros(4, 16)
    scores[0], expected[0] = 1, 1 / 16
    scores[1].zero_()
    scores[1, 5], expected[1, 5] = 1, 1
    scores[2, [3, 9]], expected[2, [3, 9]] = 3e38, 0.5
    scores[3, 12], expected[3, 12] = -1e38, 1
    b = len(temperatures)
    data = {
        "scores": scores,
        "ids": torch.arange(16, dtype=torch.int32).repeat(b, 1) * 7,
        "temperatures": temperatures,
        "request_indices": torch.arange(b, dtype=torch.int32),
        "seeds": seed_words(torch.arange(b, dtype=torch.int64) + 0x9E3779B9),
        "cursors": torch.arange(b, dtype=torch.int64) + 2**33,
        "workspace": torch.empty(0, dtype=torch.uint8),
        "out_ids": torch.empty(b, dtype=torch.int32),
        "out_q": torch.empty(b, 16),
    }
    return data, expected


def test_candidates_overflowing_scaled_scores():
    data, expected = overflow_candidate_data()
    RequestLocalChainSampling("cpu", 8, 32).sample_candidates(position_i=3, **data)
    assert torch.equal(data["out_q"].view(torch.int32), expected.view(torch.int32))
    support = expected > 0
    for b in range(len(expected)):
        assert int(data["out_ids"][b]) in data["ids"][b][support[b]].tolist()


def test_max_vocab_is_a_construction_limit():
    from ctikernels.sampling._dflash import MAX_VOCAB

    assert MAX_VOCAB == 2**21
    RequestLocalChainSampling("cpu", 2, MAX_VOCAB)
    for vocab in (MAX_VOCAB + 1, 2**26, 2**31 - 1):
        with pytest.raises(ValueError, match="vocab_size"):
            RequestLocalChainSampling("cpu", 2, vocab)


def spread_candidate_data():
    """Candidate score spreads above FLT_MAX (scale-first fallback) and nearby large scores."""
    temperatures = torch.tensor(
        [2.0**127, 2.0**64, 1.0, 3e38, 3.0, 1.0, 0.5, 2.0**-149, 2.0**-126, 1.0, 2.0**127]
    )
    scores = torch.full((11, 16), -(2.0**127))
    scores[:4, 3] = 2.0**127
    scores[3, 7] = -3e38
    scores[4:7, 3], scores[4:7, 5] = 1e7, 9999999.0
    # Masked scores (-inf) keep q = 0 under every T, also next to FLT_MAX spreads.
    scores[7:, 3], scores[7:, 5], scores[7:, 9] = -1.0, -torch.inf, -torch.inf
    scores[10, 1] = 2.0**127
    b = len(temperatures)
    data = {
        "scores": scores,
        "ids": torch.arange(16, dtype=torch.int32).repeat(b, 1) * 7,
        "temperatures": temperatures,
        "request_indices": torch.arange(b, dtype=torch.int32),
        "seeds": seed_words(torch.arange(b, dtype=torch.int64) + 0x9E3779B9),
        "cursors": torch.arange(b, dtype=torch.int64) + 2**33,
        "workspace": torch.empty(0, dtype=torch.uint8),
        "out_ids": torch.empty(b, dtype=torch.int32),
        "out_q": torch.empty(b, 16),
    }
    expected = torch.softmax(scores.double() / temperatures.double().unsqueeze(1), -1)
    return data, expected


def test_candidates_spread_scores_match_fp64():
    data, expected = spread_candidate_data()
    RequestLocalChainSampling("cpu", 8, 32).sample_candidates(position_i=3, **data)
    assert not torch.isnan(data["out_q"]).any()
    torch.testing.assert_close(data["out_q"].double(), expected, atol=2e-7, rtol=2e-6)


def long_chain_data(block, vocab, mode, greedy, rejection, *, coin=False):
    # Live, greedy/stochastic and poisoned padding rows share the same geometry.
    op, data = buffers(2, block - 1, vocab)
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    data["p"].zero_()
    data["p"][:, :, 0] = 1
    data["p"][0, -1, 0] = 0
    data["p"][0, -1, 1] = 1
    if rejection < block - 1:
        data["p"][0, rejection, 0] = 0.5 if coin else 0
        data["p"][0, rejection, 1] = 0.5 if coin else 1
    if mode == "sparse":
        sparsify(data)
    data["temperatures"].fill_(0 if greedy else 1)
    data["request_indices"][-1] = -1
    for name in ("p", "q", "temperatures"):
        data[name][-1].fill_(float("nan"))
    data["drafts"][-1].fill_(-999)
    data["cursors"][-1] = -1
    return op, data


@pytest.mark.parametrize("block", [33, 64, 65, 128, 513])
@pytest.mark.parametrize("mode", ["dense", "sparse", "point"])
@pytest.mark.parametrize("greedy", [False, True])
def test_long_chain_reference_boundaries(block, mode, greedy):
    for rejection in sorted({0, 31, 32, min(63, block - 1), min(64, block - 1), block - 1}):
        op, data = long_chain_data(block, 3, mode, greedy, rejection)
        verify(op, data, mode == "sparse", point=mode == "point")
        assert data["accepted_counts"].tolist() == [rejection, 0]
        assert data["out_tokens"][0].tolist() == [0] * rejection + [1] + [-1] * (
            block - rejection - 1
        )
        assert (data["out_tokens"][1] == -1).all()


@pytest.mark.parametrize("block", [0, 2**31])
def test_chain_position_bounds(block):
    with pytest.raises(ValueError, match="int32 positions/counts"):
        RequestLocalChainSampling("cpu", block, 3)


def test_chain_flattened_grid_bound():
    op = RequestLocalChainSampling("cpu", 65536, 3)
    with pytest.raises(ValueError, match=r"B\*Q must be <= 2147483647"):
        op.workspace_size(32768)
    assert op.workspace_size(1) > 0


@pytest.mark.parametrize("mode", ["dense", "sparse", "point"])
@pytest.mark.parametrize("greedy", [False, True])
def test_q1_bonus_matches_all_accepted(mode, greedy):
    op, data = buffers(4, 0, 5)
    larger, expected = buffers(4, 2, 5)
    expected["q"].zero_()
    expected["q"][..., 0] = 1
    expected["p"].zero_()
    expected["p"][:, :2, 0] = 1
    data["p"][0].fill_(0.2)  # Smaller-ID greedy tie.
    expected["p"][:, -1].copy_(data["p"][:, 0])
    for d in (data, expected):
        d["temperatures"].fill_(0 if greedy else 1)
        d["request_indices"][-1] = -1
        d["p"][-1].fill_(float("nan"))
        d["temperatures"][-1] = float("nan")
        d["cursors"][-1] = -1
        if mode == "sparse":
            sparsify(d)
    verify(larger, expected, mode == "sparse", point=mode == "point")
    verify(op, data, mode == "sparse", point=mode == "point")
    assert torch.equal(data["out_tokens"][:, 0], expected["out_tokens"][:, -1])
    assert data["accepted_counts"].tolist() == [0] * 4
    if greedy:
        assert data["out_tokens"][0, 0] == 0


@pytest.mark.parametrize("device_position", [False, True])
def test_q1_proposals_are_noops(monkeypatch, device_position):
    op, data = buffers(3, 0, 16)

    def fail(*args, **kwargs):
        pytest.fail("zero-draft proposal path dispatched")

    monkeypatch.setattr(op._provider, "run", fail)
    common = {k: data[k] for k in ("request_indices", "seeds", "cursors", "workspace")}
    op.sample_dense_proposals(data["q"], data["temperatures"], **common, out_ids=data["drafts"])
    out = torch.full((3,), -99, dtype=torch.int32)
    oq = torch.full((3, 16), -99.0)
    op.sample_candidates(
        torch.full((3, 16), float("nan")),
        torch.full((3, 16), -999, dtype=torch.int32),
        data["temperatures"],
        torch.tensor(0, dtype=torch.int32) if device_position else 0,
        **common,
        out_ids=out,
        out_q=oq,
    )
    assert (out == -99).all() and (oq == -99).all()


def test_q1_padding_never_reads_rng(monkeypatch):
    from ctikernels.sampling._dflash.reference import ops

    op, data = buffers(3, 0, 5)
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32)
    data["temperatures"].fill_(float("nan"))
    data["p"].fill_(float("nan"))

    def fail(*args):
        pytest.fail("padding consumed RNG")

    monkeypatch.setattr(ops, "uniform", fail)
    verify(op, data)
    assert data["accepted_counts"].tolist() == [0] * 3
    assert data["out_tokens"].tolist() == [[-1]] * 3


def test_q1_rng_domain_and_greedy(monkeypatch):
    from ctikernels.sampling._dflash.reference import BONUS, ops, request_seed

    op, data = buffers(3, 0, 5)
    data["temperatures"][1] = 0
    data["request_indices"][2] = -1
    calls = []

    def draw(*args):
        calls.append(args)
        return 0.25

    monkeypatch.setattr(ops, "uniform", draw)
    verify(op, data)
    assert calls == [(request_seed(data["seeds"], 0), int(data["cursors"][0]), BONUS, 0)]
