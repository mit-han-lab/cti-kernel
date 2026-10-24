# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Independent distribution, prefix and graph contracts for linear sampling."""

import math

import pytest
import torch

from ctikernels.sampling import ChainSampling


def _rng(seed):
    return torch.tensor([seed], dtype=torch.int64, device="cuda"), torch.zeros(
        1, dtype=torch.int64, device="cuda"
    )


def _frequencies(values, expected):
    counts = torch.bincount(values.long(), minlength=len(expected)).cpu()
    assert len(counts) == len(expected)
    expected = torch.tensor(expected, dtype=torch.float64)
    assert not counts[expected == 0].any()
    # Hoeffding union bound; <= 1e-9 false rejection probability per check.
    tolerance = math.sqrt(math.log(2 * len(expected) / 1e-9) / (2 * values.numel()))
    torch.testing.assert_close(counts.double() / values.numel(), expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("block_size", [2, 7, 16, 32, 33, 64, 65, 128])
@pytest.mark.parametrize("captured", [False, True])
def test_exact_first_rejection_and_bonus(block_size, captured):
    batch_size, vocab = block_size, 257
    q = torch.zeros(batch_size, block_size - 1, vocab, device="cuda")
    q[..., 0] = 1
    p = torch.zeros(batch_size, block_size, vocab, device="cuda")
    p[..., 0] = 1
    rejection = torch.arange(batch_size, device="cuda")
    p[rejection, rejection, 0] = 0
    p[rejection, rejection, 1] = 1
    proposals = torch.zeros(batch_size, block_size - 1, dtype=torch.int32, device="cuda")
    counts = torch.full((batch_size,), 100, dtype=torch.int32, device="cuda")
    initial = [value.clone() for value in (q, p, proposals)]
    seed, offset = _rng(13)
    operation = ChainSampling()
    result = torch.empty(batch_size, block_size, dtype=torch.int32, device="cuda")

    def run():
        result.copy_(
            operation.verify(q, proposals, p, accepted_counts=counts, seed=seed, offset=offset)
        )

    run()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        run = graph.replay
    for _ in range(3):
        counts.fill_(100)  # Must not accumulate the provider's diagnostic count.
        run()
    assert counts.tolist() == list(range(block_size))
    for row in range(batch_size):
        assert result[row].tolist() == [0] * row + [1] + [-1] * (block_size - row - 1)
    for actual, before in zip((q, p, proposals), initial, strict=True):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "target", [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.5, 0.5]]
)
def test_rejection_sampling_preserves_target_law_and_prefix_length(target):
    batch_size, block_size = 32768, 4
    draft = [0.1, 0.2, 0.3, 0.4]
    q = torch.tensor(draft, device="cuda").repeat(batch_size, block_size - 1, 1)
    p = torch.tensor(target, device="cuda").repeat(batch_size, block_size, 1)
    original_q, original_p = q.clone(), p.clone()
    operation = ChainSampling()
    draft_seed, draft_offset = _rng(15)
    seed, offset = _rng(982)
    proposals = operation.sample(q.flatten(0, 1), seed=draft_seed, offset=draft_offset).view(
        batch_size, -1
    )
    counts = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    result = operation.verify(q, proposals, p, accepted_counts=counts, seed=seed, offset=offset)
    _frequencies(result[:, 0], target)
    acceptance = sum(min(a, b) for a, b in zip(draft, target, strict=True))
    _frequencies(counts, [(acceptance**i) * (1 - acceptance) for i in range(3)] + [acceptance**3])
    if acceptance < 1:
        residual = [max(p - q, 0) / (1 - acceptance) for q, p in zip(draft, target, strict=True)]
        _frequencies(result[counts == 0, 0], residual)
    if acceptance > 0:
        _frequencies(result[counts == block_size - 1, -1], target)
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    torch.testing.assert_close(p, original_p, rtol=0, atol=0)


def test_filtered_probabilities_match_independent_fp32_reference():
    # Distinct logits avoid unspecified tie-breaking at the top-k boundary.
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]], device="cuda")
    temperatures = torch.tensor([0.7, 1.2], device="cuda")
    ks = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    ps = torch.tensor([0.8, 0.9], device="cuda")
    expected = (logits.cpu() / temperatures.cpu()[:, None]).softmax(-1)
    for row, (k, cutoff) in enumerate(zip(ks.tolist(), ps.tolist(), strict=True)):
        order = expected[row].argsort(descending=True)
        expected[row, order[k:]] = 0
        expected[row] /= expected[row].sum()
        ordered = expected[row, order]
        expected[row, order[ordered.cumsum(0) - ordered >= cutoff]] = 0
        expected[row] /= expected[row].sum()
    actual = ChainSampling().probabilities(logits, temperatures, ks, ps)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-6)


def test_separate_graphs_advance_device_rng_and_reset_reproduces():
    batch_size, block_size, vocab = 32, 7, 257
    operation = ChainSampling()
    q = torch.full((batch_size, block_size - 1, vocab), 1 / vocab, device="cuda")
    p = torch.full((batch_size, block_size, vocab), 1 / vocab, device="cuda")
    proposals = torch.empty(batch_size, block_size - 1, dtype=torch.int32, device="cuda")
    result = torch.empty(batch_size, block_size, dtype=torch.int32, device="cuda")
    counts = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    ds, do = _rng(17)
    vs, vo = _rng(299)

    def draft():
        proposals.copy_(operation.sample(q.flatten(0, 1), seed=ds, offset=do).view_as(proposals))

    def verify():
        result.copy_(operation.verify(q, proposals, p, accepted_counts=counts, seed=vs, offset=vo))

    draft()
    verify()
    dg, vg = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(dg):
        draft()
    with torch.cuda.graph(vg, pool=dg.pool()):
        verify()
    do.zero_()
    vo.zero_()
    outputs = []
    for _ in range(3):
        dg.replay()
        vg.replay()
        outputs.append(result.clone())
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    assert counts.tolist() == [block_size - 1] * batch_size
    assert do.item() == 12 and vo.item() == 24
    do.zero_()
    vo.zero_()
    dg.replay()
    vg.replay()
    torch.testing.assert_close(result, outputs[0], atol=0, rtol=0)
    # Same captured addresses, changed probabilities: graph must consume new p.
    p.zero_()
    p[..., 7] = 1
    dg.replay()
    vg.replay()
    assert result[:, 0].tolist() == [7] * batch_size


def test_reject_per_row_rng_arrays_instead_of_silently_ignoring_them():
    probs = torch.full((8, 4), 0.25, device="cuda")
    seeds = torch.arange(8, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="single-element"):
        ChainSampling().sample(probs, seed=seeds, offset=torch.zeros_like(seeds))


def test_chain_q1_bonus_matches_all_accepted_capture():
    batch, vocab = 32, 257
    op = ChainSampling()
    q = torch.empty(batch, 0, vocab, device="cuda")
    drafts = torch.empty(batch, 0, dtype=torch.int32, device="cuda")
    p = torch.rand(batch, 1, vocab, device="cuda")
    p /= p.sum(-1, keepdim=True)
    counts = torch.full((batch,), 99, dtype=torch.int32, device="cuda")
    seed, offset = _rng(982)
    result = torch.empty(batch, 1, dtype=torch.int32, device="cuda")

    def run():
        result.copy_(op.verify(q, drafts, p, accepted_counts=counts, seed=seed, offset=offset))

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    larger_q = torch.zeros(batch, 1, vocab, device="cuda")
    larger_q[..., 0] = 1
    larger_p = torch.cat((larger_q, p), dim=1)
    larger_drafts = torch.zeros(batch, 1, dtype=torch.int32, device="cuda")
    larger_counts = torch.empty_like(counts)
    for start in (0, 4, 8):
        # The shared stream consumes one acceptance uniform before its bonus.
        offset.fill_(start + 1)
        counts.fill_(99)
        graph.replay()
        larger_offset = torch.tensor([start], dtype=torch.int64, device="cuda")
        expected = op.verify(
            larger_q,
            larger_drafts,
            larger_p,
            accepted_counts=larger_counts,
            seed=seed,
            offset=larger_offset,
        )
        assert torch.equal(result[:, 0], expected[:, -1])
        assert counts.cpu().tolist() == [0] * batch
        assert offset.item() == start + 5


def test_empty_batch_sample_and_probabilities_draw_nothing():
    seed, offset = _rng(3)
    operation = ChainSampling()
    empty = torch.empty(0, device="cuda")
    probabilities = operation.probabilities(
        torch.empty(0, 257, device="cuda"), empty, empty.int(), empty
    )
    tokens = operation.sample(probabilities, seed=seed, offset=offset)
    assert probabilities.shape == (0, 257) and probabilities.dtype == torch.float32
    assert tokens.shape == (0,) and tokens.dtype == torch.int32
    assert offset.item() == 0


@pytest.mark.parametrize("block_size", [1, 4])
def test_empty_batch_verify_draws_nothing(block_size):
    seed, offset = _rng(3)
    tokens = ChainSampling().verify(
        torch.empty(0, block_size - 1, 257, device="cuda"),
        torch.empty(0, block_size - 1, dtype=torch.int32, device="cuda"),
        torch.empty(0, block_size, 257, device="cuda"),
        accepted_counts=torch.empty(0, dtype=torch.int32, device="cuda"),
        seed=seed,
        offset=offset,
    )
    assert tokens.shape == (0, block_size) and tokens.dtype == torch.int32
    assert offset.item() == 0
