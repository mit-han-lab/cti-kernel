# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen GPU qualification; never run on CPU-only workers.

N=200,000 distinct request keys per statistical experiment, no retries. The contract's
3,900 hypotheses remain reserved across three devices, two representations and
five engine filtering modes. Candidate sampling adds 3*16=48 binomial hypotheses.
Freeze M=3,948, family alpha=0.01, per-test alpha=0.01/3948 BEFORE execution.
Use equal-tailed Clopper–Pearson intervals with the frozen Bonferroni alpha.
Zero-probability observations always fail. Missing devices never reduce M.
Modes here exercise prefiltered assembled p/q; they do not qualify the required
probability-construction operation.
These tests do not qualify TP collectives or model-level score construction.
"""

import pytest
import torch

from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.cute import Provider

from ..modules._binomial import clopper_pearson
from ..modules.test_dflash_rng_golden import GOLDEN, KATS, SEED
from ..modules.test_dflash_sampling_reference import buffers, seed_words, sparsify, verify
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU qualification only")
N = 200_000
M = 3948
ALPHA = 0.01 / M


# min_p per active row cycles through 0.05, 1, an exact tie at the pivot and 0
# (row 1 is greedy and the last row is padding, see min_p_data). Two covering
# sweeps instead of the full product: every B x L of the
# DN5 grid at V=4097, and every V x dtype x hint at B=5, L=2.
DN5_MINS = (0.05, "tie", 1, "tie", 0)
DN5_CORE = [
    (0, 1, 4097, torch.float32, "auto"),
    (1, 1, 4097, torch.float32, "auto"),
    (4, 16, 4097, torch.float32, "auto"),
    (8, 5, 4097, torch.float32, "auto"),
    (48, 64, 4097, torch.float32, "auto"),
    (5, 2, 16, torch.float32, "none"),
    (5, 2, 4097, torch.float16, "top_k"),
    (5, 2, 151936, torch.bfloat16, "top_p"),
    (5, 2, 248320, torch.float32, "both"),
]


def _dn5_cases():
    shapes = ci_cases(
        [0, 1, 2, 3, 4, 8, 16, 32, 48],
        [1, 2, 4, 5, 8, 9, 12, 16, 33, 64],
        [4097],
        [torch.float32],
        ["auto"],
        core=[case for case in DN5_CORE if case[2] == 4097 and case[4] == "auto"],
    )
    vocabs = ci_cases(
        [5],
        [2],
        [16, 4097, 151936, 248320],
        [torch.bfloat16, torch.float16, torch.float32],
        ["none", "top_k", "top_p", "both", "auto"],
        core=[case for case in DN5_CORE if case[:2] == (5, 2)],
    )
    return shapes + vocabs


@pytest.mark.parametrize("batch,length,vocab,dtype,hint", _dn5_cases())
def test_dn5_min_p_grid(batch, length, vocab, dtype, hint):
    from ctikernels.sampling._dflash.reference.ops import min_p_row

    from ..modules.test_dflash_min_p import assert_bits, min_p_data, tie_min_p

    if batch * length * vocab > 4_000_000:
        pytest.skip("DN5 case capped at four million probability elements")
    data = min_p_data(batch, length, vocab, dtype, hint)
    gpu = {name: value.cuda() for name, value in data.items()}
    op = RequestLocalChainSampling("cuda", length, vocab)
    op.probabilities(**gpu, filters=hint)
    expected = gpu["out_probs"].cpu()
    mins = torch.zeros(batch)
    for b in range(batch):
        if data["request_indices"][b] == -1 or data["temperatures"][b] == 0:
            mins[b] = float("nan")  # Ignored before any maxima access.
            continue
        minimum = DN5_MINS[b % len(DN5_MINS)]
        mins[b] = tie_min_p(expected[b, 0]) if minimum == "tie" else minimum
        for i in range(length):
            expected[b, i] = min_p_row(expected[b, i], mins[b])
    gpu["workspace"].fill_(165)
    op.probabilities(**gpu, min_p=mins.cuda(), filters=hint)
    assert_bits(gpu["out_probs"].cpu(), expected)


@pytest.mark.parametrize("vocab", [16, 4097])
@pytest.mark.parametrize("hint", ["none", "top_k", "top_p", "both", "auto"])
def test_dn5_min_p_full_reference(vocab, hint):
    from ..modules.test_dflash_min_p import assert_bits, min_p_data

    data = min_p_data(3, 2, vocab, torch.float32, hint)
    # Exact exp(0)/exp(-inf) isolates the frozen arithmetic from exp providers.
    data["logits"].fill_(-torch.inf)
    data["logits"][..., :8] = 0
    data["min_p"] = torch.tensor([0.5, float("nan"), float("nan")])
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=hint)
    gpu = {name: value.cuda() for name, value in data.items()}
    RequestLocalChainSampling("cuda", 2, vocab).probabilities(**gpu, filters=hint)
    assert_bits(gpu["out_probs"].cpu(), data["out_probs"])


def test_dn5_min_p_point_mass():
    from ..modules.test_dflash_min_p import point_min_p_data
    from ..modules.test_dflash_probabilities import probability_data

    op, data = point_min_p_data()
    verify(op, data, point=True)
    gpu = {name: value.cuda() for name, value in data.items()}
    device_op = RequestLocalChainSampling("cuda", 3, 4)
    probability = probability_data(
        torch.tensor([0.0, -4.0, -8.0, -12.0]).expand(2, 3, 4).contiguous()
    )
    probability = {name: value.cuda() for name, value in probability.items()}
    probability["out_probs"] = gpu["p"]
    device_op.probabilities(**probability, min_p=torch.full((2,), 0.5, device="cuda"))
    verify(device_op, gpu, point=True)
    assert torch.equal(gpu["out_tokens"].cpu(), data["out_tokens"])
    assert torch.equal(gpu["accepted_counts"].cpu(), data["accepted_counts"])


def test_dn5_min_p_warm_up_capture(monkeypatch):
    from ..modules.test_dflash_min_p import HINTS, assert_bits, min_p_data

    op = RequestLocalChainSampling("cuda", 4, 16)
    op.warm_up(batch_capacity=3, dtypes=(torch.float32,), modes=())

    def no_compile(*args, **kwargs):
        raise AssertionError("warm-up missed a captured variant")

    monkeypatch.setattr(op._provider, "_compile", no_compile)
    for hint in HINTS:
        data = {k: v.cuda() for k, v in min_p_data(3, 4, 16, torch.float32, hint).items()}
        minimum = torch.tensor([0.5, 1, float("nan")], device="cuda")
        for use in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.probabilities(**data, min_p=minimum if use else None, filters=hint)
            graph.replay()
            captured = data["out_probs"].clone()
            op.probabilities(**data, min_p=minimum if use else None, filters=hint)
            assert_bits(data["out_probs"], captured)
        # Runtime zero must preserve base bits even with poisoned partials.
        minimum.zero_()
        data["workspace"].fill_(165)
        graph.replay()
        captured = data["out_probs"].clone()
        op.probabilities(**data, filters=hint)
        assert_bits(data["out_probs"], captured)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before capture"), torch.cuda.graph(graph):
        op.warm_up(batch_capacity=3, modes=())


def test_dn5_ordinary_min_p_gamma_three():
    from ctikernels.sampling._dflash.reference import Provider as ReferenceProvider

    batch, gamma, vocab = 4, 3, 4097
    q = torch.zeros(batch, gamma, vocab)
    for b in range(batch):
        for i in range(gamma):
            weights = (
                torch.tensor([1.0, 0, 0, 0]),
                torch.tensor([0.5, 0.25, 0.125, 0.125]),
                torch.tensor([0.25, 0.25, 0.25, 0.25]),
            )[(b + i) % 3]
            q[b, i, [0, 17, 4095, 4096]] = weights
    maxima = torch.stack(
        [q[..., start : start + 4096].amax(-1) for start in range(0, vocab, 4096)], dim=-1
    ).reshape(-1)
    tensors = (
        q,
        torch.tensor([1.0, 1.0, 0.0, float("nan")]),
        torch.tensor([0, 1, 2, -1], dtype=torch.int32),
        seed_words([11, 22, 33]),
        torch.tensor([7, 8, 9, 10], dtype=torch.int64),
        torch.empty(batch * gamma * 2 * 8, dtype=torch.uint8),
        torch.empty(batch, gamma, dtype=torch.int32),
        torch.tensor([0.6, 1.0, float("nan"), float("nan")]),
        maxima,
    )
    ReferenceProvider().run("ordinary_min_p", tensors)
    gpu = tuple(t.cuda() for t in tensors)
    Provider().run("ordinary_min_p", gpu)
    assert torch.equal(gpu[6].cpu(), tensors[6])


def test_device_philox_golden():
    rows = [
        ((c & 0xFFFFFFFF, c >> 32, d, i), (SEED & 0xFFFFFFFF, SEED >> 32), expected)
        for c, d, i, expected in GOLDEN
    ] + KATS
    counter = torch.tensor([r[0] for r in rows], dtype=torch.int64, device="cuda")
    key = torch.tensor([r[1] for r in rows], dtype=torch.int64, device="cuda")
    out = torch.empty_like(counter)
    Provider().run("rng", (counter, key, out))
    assert out.cpu().tolist() == [list(r[2]) for r in rows]


# CI samples dense/sparse/point and tiled/compact/large probability paths.
# Each parametrization retains its selected core; remaining rows are exhaustive.
# Statistical core retains both representations without changing N or alpha.
@pytest.mark.parametrize(
    "sparse,gamma",
    ci_cases([False, True], [1, 7, 15, 31], core=[(False, 1), (True, 15)]),
)
def test_all_prefixes(sparse, gamma):
    cpu_op, data = buffers(gamma + 2, gamma, 3)
    data["q"].zero_()
    data["q"][:, :, 0] = 1
    data["p"].zero_()
    data["p"][:, :, 0] = 1
    for a in range(gamma):
        data["p"][a, a, 0], data["p"][a, a, 1] = 0, 1
    data["request_indices"][-1] = -1
    data["p"][-1].fill_(float("nan"))
    data["q"][-1].fill_(float("nan"))
    data["drafts"][-1].fill_(-999)
    if sparse:
        sparsify(data)
    gpu_op = RequestLocalChainSampling("cuda", gamma + 1, 3)
    for greedy in (False, True):
        data["temperatures"].fill_(0 if greedy else 1)
        verify(cpu_op, data, sparse)
        gpu = {k: t.cuda() for k, t in data.items()}
        verify(gpu_op, gpu, sparse)
        assert torch.equal(gpu["out_tokens"].cpu(), data["out_tokens"])
        assert torch.equal(gpu["accepted_counts"].cpu(), data["accepted_counts"])
        assert gpu["accepted_counts"].cpu().tolist() == list(range(gamma + 1)) + [0]


@pytest.mark.parametrize(
    "sparse,b",
    ci_cases([False, True], [1, 2, 3, 4, 8], core=[(False, 1), (True, 4)]),
)
def test_replay_partition_permutation_padding(sparse, b):
    op, cpu = buffers(b, 7, 5)
    op.sample_dense_proposals(
        cpu["q"],
        cpu["temperatures"],
        **{k: cpu[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
        out_ids=cpu["drafts"],
    )
    if sparse:
        sparsify(cpu)
    verify(op, cpu, sparse)
    gpu_op = RequestLocalChainSampling("cuda", 8, 5)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    proposal = torch.empty_like(gpu["drafts"])

    def sample():
        gpu_op.sample_dense_proposals(
            gpu["q"],
            gpu["temperatures"],
            **{k: gpu[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
            out_ids=proposal,
        )

    sample()
    assert torch.equal(proposal.cpu(), cpu["drafts"])
    verify(gpu_op, gpu, sparse)
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
    assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sample()
        verify(gpu_op, gpu, sparse)
    for _ in range(100):
        graph.replay()
    assert torch.equal(proposal.cpu(), cpu["drafts"])
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
    # Device cursor mutation feeds the same captured graph, without host staging.
    gpu["cursors"].add_(100003)
    cpu["cursors"].add_(100003)
    verify(op, cpu, sparse)
    graph.replay()
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
    for row in range(b):
        one_op, one = buffers(1, 7, 5, "cuda")
        for name in ("p", "q", "drafts", "temperatures", "cursors"):
            one[name].copy_(gpu[name][row : row + 1])
        one["seeds"][0] = gpu["seeds"][row]
        if sparse:
            one["cand_ids"] = gpu["cand_ids"][row : row + 1].clone()
            one["cand_q"] = gpu["cand_q"][row : row + 1].clone()
        verify(one_op, one, sparse)
        assert torch.equal(one["out_tokens"].cpu()[0], cpu["out_tokens"][row])
    # Reverse live rows, insert poisoned padding at both ends, retain slot pool.
    pad_op, padded = buffers(b + 2, 7, 5, "cuda")
    permutation = torch.arange(b - 1, -1, -1, device="cuda")
    for name in ("p", "q", "drafts", "temperatures", "cursors", "request_indices"):
        padded[name][1:-1].copy_(gpu[name][permutation])
    padded["seeds"] = gpu["seeds"]
    padded["request_indices"][[0, -1]] = -1
    padded["p"][[0, -1]] = float("nan")
    padded["drafts"][[0, -1]] = -999
    if sparse:
        sparsify(padded)
    verify(pad_op, padded, sparse)
    assert torch.equal(padded["out_tokens"][1:-1].cpu(), cpu["out_tokens"].flip(0))
    assert (padded["out_tokens"][[0, -1]] == -1).all()
    padded["request_indices"].fill_(-1)
    padded["seeds"] = torch.empty((0, 2), dtype=torch.uint32, device="cuda")
    verify(pad_op, padded, sparse)
    assert (padded["out_tokens"] == -1).all() and (padded["accepted_counts"] == 0).all()


def candidate_data(b, device):
    return {
        "scores": torch.arange(15, -1, -1, dtype=torch.float32, device=device).div(8).repeat(b, 1),
        "ids": torch.arange(16, dtype=torch.int32, device=device).repeat(b, 1),
        "temperatures": torch.ones(b, device=device),
        "request_indices": torch.arange(b, dtype=torch.int32, device=device),
        "seeds": seed_words(torch.arange(b, dtype=torch.int64) + 0x123456789).to(device),
        "cursors": torch.arange(b, dtype=torch.int64, device=device) + 2**32,
        "workspace": torch.empty(0, dtype=torch.uint8, device=device),
        "out_ids": torch.empty(b, dtype=torch.int32, device=device),
        "out_q": torch.empty((b, 16), device=device),
    }


@pytest.mark.parametrize("batch", [0, 3])
def test_empty_pools_and_batch_graph(batch):
    op, data = buffers(batch, 2, 3, "cuda")
    sparsify(data)
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32, device="cuda")
    for name in ("p", "q", "cand_q", "temperatures"):
        data[name].fill_(float("nan"))
    data["cursors"].fill_(-999)
    candidate_op = RequestLocalChainSampling("cuda", 3, 32)
    candidates = candidate_data(batch, "cuda")
    candidates["request_indices"].fill_(-1)
    candidates["seeds"] = data["seeds"]
    candidates["scores"].fill_(float("nan"))
    position = torch.tensor(1, dtype=torch.int32, device="cuda")
    prompts = torch.empty(0, dtype=torch.int64, device="cuda")
    anchors = torch.full((batch,), -999, dtype=torch.int64, device="cuda")
    out_cursors = torch.empty_like(anchors)

    def call():
        op.prepare_cursors(
            anchors,
            prompts,
            request_indices=data["request_indices"],
            workspace=data["workspace"],
            out_cursors=out_cursors,
        )
        op.sample_dense_proposals(
            data["q"],
            data["temperatures"],
            **{k: data[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
            out_ids=data["drafts"],
        )
        verify(op, data)
        verify(op, data, True)
        candidate_op.sample_candidates(position_i=position, **candidates)

    call()
    cache_size = len(op._provider.compiled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        data["drafts"].fill_(99)
        data["out_tokens"].fill_(99)
        data["accepted_counts"].fill_(99)
        candidates["out_ids"].fill_(99)
        candidates["out_q"].fill_(99)
        out_cursors.fill_(99)
        graph.replay()
        assert (data["drafts"] == -1).all()
        assert (data["out_tokens"] == -1).all()
        assert (data["accepted_counts"] == 0).all()
        assert (candidates["out_ids"] == -1).all()
        assert (candidates["out_q"] == 0).all()
        assert (out_cursors == 0).all()
    assert len(op._provider.compiled) == cache_size


@pytest.mark.parametrize(
    "batch",
    ci_cases([1, 2, 3, 4], core=[(4,)]),
)
def test_point_mass_all_inactive_empty_pool_graph(batch):
    op, data = buffers(batch, 2, 4097, "cuda")
    data["request_indices"].fill_(-1)
    data["seeds"] = torch.empty(0, 2, dtype=torch.uint32, device="cuda")
    for name in ("p", "temperatures"):
        data[name].fill_(float("nan"))
    data["drafts"].fill_(-999)
    data["cursors"].fill_(-999)
    data["workspace"].fill_(165)
    data["out_tokens"].fill_(99)
    data["accepted_counts"].fill_(99)
    verify(op, data, point=True)  # dirty first use; warm before capture
    assert (data["out_tokens"] == -1).all()
    assert (data["accepted_counts"] == 0).all()
    cache_size = len(op._provider.compiled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify(op, data, point=True)
    for garbage in (255, 90, 165):
        data["workspace"].fill_(garbage)
        data["out_tokens"].fill_(99)
        data["accepted_counts"].fill_(99)
        graph.replay()
        assert (data["out_tokens"] == -1).all()
        assert (data["accepted_counts"] == 0).all()
    assert len(op._provider.compiled) == cache_size


def test_candidate_reference_graph_and_distribution():
    cpu = candidate_data(8, "cpu")
    cpu["request_indices"][-1] = -1
    cpu["temperatures"][0] = 0
    cpu["scores"][0].zero_()
    cpu["scores"][-1].fill_(float("nan"))
    op = RequestLocalChainSampling("cpu", 8, 32)
    op.sample_candidates(position_i=6, **cpu)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    gpu_op = RequestLocalChainSampling("cuda", 8, 32)
    gpu_op.sample_candidates(position_i=6, **gpu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gpu_op.sample_candidates(position_i=6, **gpu)
    for _ in range(100):
        graph.replay()
    assert torch.equal(cpu["out_ids"], gpu["out_ids"].cpu())
    torch.testing.assert_close(cpu["out_q"], gpu["out_q"].cpu(), atol=2e-6, rtol=2e-5)
    data = candidate_data(N, "cuda")
    gpu_op.sample_candidates(position_i=6, **data)
    expected = torch.softmax(data["scores"][0].double(), -1).cpu()
    torch.testing.assert_close(data["out_q"][0].double().cpu(), expected, atol=2e-6, rtol=2e-5)
    check_bins(data["out_ids"].cpu(), expected)


def test_candidate_device_position_replay():
    cpu = candidate_data(2, "cpu")
    cpu["seeds"] = seed_words([0xFFFFFFFFFFFFFFFF, 0x8000000000000000])
    gpu = {name: tensor.cuda() for name, tensor in cpu.items()}
    reference = RequestLocalChainSampling("cpu", 8, 32)
    op = RequestLocalChainSampling("cuda", 8, 32)
    position = torch.tensor(0, dtype=torch.int32, device="cuda")
    op.sample_candidates(position_i=position, **gpu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.sample_candidates(position_i=position, **gpu)
    for i in (0, 6):
        position.fill_(i)
        reference.sample_candidates(position_i=i, **cpu)
        graph.replay()
        assert torch.equal(gpu["out_ids"].cpu(), cpu["out_ids"])


def check_binomial(counts, expected):
    # Evaluate all bins, including exact-zero bins,
    # with the same family correction; hardware omissions never shrink M.
    counts = torch.as_tensor(counts, dtype=torch.int64).cpu()
    expected = torch.as_tensor(expected, dtype=torch.float64).cpu()
    bounds = torch.tensor(
        [clopper_pearson(int(k), N, ALPHA) for k in counts.flatten()], dtype=torch.float64
    )
    bounds = bounds.reshape(*counts.shape, 2)
    lower, upper = bounds[..., 0], bounds[..., 1]
    assert ((expected != 0) | (counts == 0)).all()
    assert ((lower <= expected) & (expected <= upper)).all(), (
        f"Clopper–Pearson intervals: lower={lower}, upper={upper}, oracle={expected}"
    )


def check_bins(samples, expected):
    counts = torch.bincount(samples.long(), minlength=len(expected))
    assert len(samples) == N
    check_binomial(counts, expected)


@pytest.mark.parametrize(
    "sparse,mode",
    ci_cases(
        [False, True],
        ["temperature", "k", "p", "kp", "seeded_temperature"],
        core=[(False, "temperature"), (True, "kp")],
    ),
)
def test_frozen_chain_distributions(sparse, mode):
    gamma, v = (7 if sparse else 15), 32
    op = RequestLocalChainSampling("cuda", gamma + 1, v)
    # Freeze independent request keys; no batch row enters RNG.
    data = {
        "p": torch.zeros(N, gamma + 1, v, device="cuda"),
        "q": torch.zeros(N, gamma, v, device="cuda"),
        "drafts": torch.zeros(N, gamma, dtype=torch.int32, device="cuda"),
        "temperatures": torch.ones(N, device="cuda"),
        "request_indices": torch.arange(N, dtype=torch.int32, device="cuda"),
        "seeds": seed_words(torch.arange(N, dtype=torch.int64) + 42).cuda(),
        "cursors": torch.arange(N, dtype=torch.int64, device="cuda") + 2**32,
        "workspace": torch.empty(op.workspace_size(N), dtype=torch.uint8, device="cuda"),
        "out_tokens": torch.empty(N, gamma + 1, dtype=torch.int32, device="cuda"),
        "accepted_counts": torch.empty(N, dtype=torch.int32, device="cuda"),
    }
    data["q"][:, :, 0] = 1
    if sparse:
        data["cand_ids"] = torch.zeros(N, gamma, 16, dtype=torch.int32, device="cuda")
        data["cand_q"] = torch.zeros(N, gamma, 16, device="cuda")
        data["cand_q"][:, :, 0:2] = 0.5  # deliberate duplicate IDs
    # First-position and all-accepted binomial fixtures.
    for probability in (0.5, 0.95):
        data["p"].zero_()
        data["p"][:, :, 0] = probability
        data["p"][:, :, 1] = 1 - probability
        verify(op, data, sparse)
        counts = data["accepted_counts"].cpu()
        if probability == 0.5:
            observed, expected = (counts > 0).sum(), 0.5
        else:
            observed, expected = (
                (counts == gamma).sum(),
                float(data["p"][0, 0, 0].cpu()) ** gamma,
            )
        check_binomial(observed, expected)
    # Two direct residual and two direct bonus fixtures, N draws each.
    for fixture in (0, 1):
        weights = torch.arange(1, v + 1, dtype=torch.float64)
        if fixture:
            weights = weights.flip(0).square()
        weights[0] = 0
        # Engine-side truncation fixture; K1 receives final normalized probabilities.
        if mode in ("k", "kp"):
            weights[:16] = 0
        if mode in ("p", "kp"):
            weights[weights < weights.max() * 0.5] = 0
        expected = weights / weights.sum()
        probabilities = expected.float().cuda()
        for bonus in (False, True):
            data["p"].zero_()
            if bonus:
                data["p"][:, :gamma, 0] = 1
                data["p"][:, gamma] = probabilities
            else:
                data["p"][:] = probabilities  # q(x)=1, p(x)=0 -> forced first rejection
            verify(op, data, sparse)
            a = gamma if bonus else 0
            assert (data["accepted_counts"] == a).all()
            check_bins(data["out_tokens"][:, a].cpu(), expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_large_vocab_pad_free(sparse):
    v, gamma = 248320, 7 if sparse else 15
    op, cpu = buffers(1, gamma, v)
    cpu["q"].zero_()
    cpu["q"][:, :, 0] = 1
    cpu["p"].zero_()
    cpu["p"][:, :, v - 1] = 1
    if sparse:
        cpu["cand_ids"] = torch.zeros(1, gamma, 16, dtype=torch.int32)
        cpu["cand_q"] = torch.zeros(1, gamma, 16)
        cpu["cand_q"][:, :, 0] = 1
    verify(op, cpu, sparse)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    verify(RequestLocalChainSampling("cuda", gamma + 1, v), gpu, sparse)
    assert gpu["accepted_counts"].item() == 0
    assert gpu["out_tokens"][0, 0].item() == v - 1
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])


def test_subnormal_correction():
    op, cpu = buffers(1, 2, 3)
    cpu["q"].zero_()
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    cpu["q"][:, :, 1] = tiny
    cpu["q"][:, :, 2] = 1
    cpu["drafts"].fill_(1)
    cpu["p"].zero_()
    cpu["p"][:, :, 0] = tiny
    cpu["p"][:, :, 2] = 1
    verify(op, cpu)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    verify(RequestLocalChainSampling("cuda", 3, 3), gpu)
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
    assert gpu["out_tokens"][0, 0].item() == 0


def test_device_cursor_lifecycle():
    op = RequestLocalChainSampling("cuda", 8, 3)
    anchors = torch.tensor([2**32 + 17, -999, 2**63 - 2], device="cuda")
    prompts = torch.tensor([17, 2**32 + 15], device="cuda")
    slots = torch.tensor([1, -1, 0], dtype=torch.int32, device="cuda")
    cursors = torch.empty(3, dtype=torch.int64, device="cuda")
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")

    def call():
        op.prepare_cursors(
            anchors, prompts, request_indices=slots, workspace=workspace, out_cursors=cursors
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    assert cursors.cpu().tolist() == [3, 0, 2**63 - 18]
    anchors[0].add_(8)
    graph.replay()
    assert cursors[0].item() == 11


@pytest.mark.parametrize("sparse", [False, True])
def test_residual_draw_and_input_immutability(sparse):
    op, data = buffers(4, 2, 5, "cuda")
    data["drafts"].zero_()
    data["q"][:] = torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125], device="cuda")
    data["p"][:] = torch.tensor([0.0, 0.4, 0.1, 0.2, 0.3], device="cuda")
    if sparse:
        sparsify(data)
    names = ["p", "q", "drafts", "seeds", "cursors", "request_indices", "temperatures"]
    if sparse:
        names += ["cand_ids", "cand_q"]
    saved = {name: data[name].clone() for name in names}
    verify(op, data, sparse)
    for name in names:
        assert torch.equal(data[name], saved[name])
    # Workspace contents are private. The fused draw need not materialize r.
    cpu_op = RequestLocalChainSampling("cpu", 3, 5)
    cpu = {name: value.cpu() for name, value in data.items()}
    verify(cpu_op, cpu, sparse)
    assert torch.equal(data["out_tokens"].cpu(), cpu["out_tokens"])
    assert torch.equal(data["accepted_counts"].cpu(), cpu["accepted_counts"])


def test_candidate_batch_permutation_and_inactivity():
    op = RequestLocalChainSampling("cuda", 8, 32)
    data = candidate_data(8, "cuda")
    op.sample_candidates(position_i=1, **data)
    expected = data["out_ids"].clone()
    for row in range(8):
        one = candidate_data(1, "cuda")
        one["cursors"].copy_(data["cursors"][row : row + 1])
        one["seeds"].copy_(data["seeds"][row : row + 1])
        op.sample_candidates(position_i=1, **one)
        assert one["out_ids"][0] == expected[row]
    padded = candidate_data(10, "cuda")
    for name in ("scores", "ids", "temperatures", "request_indices", "cursors"):
        padded[name][1:-1].copy_(data[name].flip(0))
    padded["seeds"] = data["seeds"]
    padded["request_indices"][[0, -1]] = -1
    padded["scores"][[0, -1]] = float("nan")
    op.sample_candidates(position_i=1, **padded)
    assert torch.equal(padded["out_ids"][1:-1], expected.flip(0))
    assert (padded["out_ids"][[0, -1]] == -1).all()
    padded["request_indices"].fill_(-1)
    padded["seeds"] = torch.empty((0, 2), dtype=torch.uint32, device="cuda")
    op.sample_candidates(position_i=1, **padded)
    assert (padded["out_ids"] == -1).all() and (padded["out_q"] == 0).all()


@pytest.mark.parametrize(
    "vocab,sparse",
    ci_cases(
        [1023, 1024, 1025, 2049, 4095, 4096, 4097, 8193, 248320],
        [False, True],
        core=[(1024, False), (4096, True), (4097, False), (4097, True)],
    ),
)
def test_tiled_verify_alignment_and_greedy(vocab, sparse):
    from ctikernels.sampling._dflash.reference import sparse_dense

    gamma = 7 if sparse else 15
    cpu_op, cpu = buffers(4, gamma, vocab)
    candidate_ids = torch.tensor(
        [
            0,
            0,
            3,
            4,
            31,
            32,
            127,
            128,
            255,
            256,
            511,
            512,
            vocab - 3,
            vocab - 2,
            vocab - 1,
            vocab - 1,
        ],
        dtype=torch.int32,
    )
    candidate_q = torch.arange(1, 17, dtype=torch.float32)
    candidate_q /= candidate_q.sum()
    cpu["q"][:] = sparse_dense(candidate_ids, candidate_q, vocab)
    if sparse:
        cpu["cand_ids"] = candidate_ids.repeat(4, gamma, 1)
        cpu["cand_q"] = candidate_q.repeat(4, gamma, 1)
    cpu["drafts"].zero_()
    for b, a in enumerate((0, gamma // 2, gamma)):
        cpu["p"][b, :a].zero_()
        cpu["p"][b, :a, 0] = 1
        if a < gamma:
            cpu["p"][b, a, 0] = 0
            cpu["p"][b, a] /= cpu["p"][b, a].sum()
            # Later rows must not affect verification, even with invalid values.
            cpu["p"][b, a + 1 :].fill_(float("nan"))
    cpu["request_indices"][-1] = -1
    cpu["p"][-1].fill_(float("nan"))
    cpu["q"][-1].fill_(float("nan"))
    gpu_op = RequestLocalChainSampling("cuda", gamma + 1, vocab)
    for greedy in (False, True):
        cpu["temperatures"].fill_(0 if greedy else 1)
        verify(cpu_op, cpu, sparse)
        assert cpu["accepted_counts"].tolist() == [0, gamma // 2, gamma, 0]
        # Same cached specialization must handle actual addresses, including rows
        # whose alignment changes with vocab stride, in eager and captured calls.
        for offset in (0, 1):
            gpu = {k: t.cuda() for k, t in cpu.items()}
            for name in ("p", "q"):
                value = gpu[name]
                backing = torch.empty(value.numel() + offset, device="cuda", dtype=value.dtype)
                gpu[name] = backing[offset:].view(value.shape)
                gpu[name].copy_(value)
            verify(gpu_op, gpu, sparse)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                verify(gpu_op, gpu, sparse)
            graph.replay()
            assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
            assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])


@pytest.mark.parametrize(
    "vocab",
    # One/full, two/tail, three and 61 chunks preserve every split-count family.
    ci_cases([1025, 4095, 4096, 4097, 8193, 248320], core=[(4096,), (4097,), (8193,), (248320,)]),
)
def test_tiled_dense_proposal_alignment(vocab):
    op, cpu = buffers(2, 15, vocab)
    common = ("request_indices", "seeds", "cursors", "workspace")
    op.sample_dense_proposals(
        cpu["q"], cpu["temperatures"], **{k: cpu[k] for k in common}, out_ids=cpu["drafts"]
    )
    gpu_op = RequestLocalChainSampling("cuda", 16, vocab)
    for offset in (0, 1):
        gpu = {k: t.cuda() for k, t in cpu.items()}
        backing = torch.empty(gpu["q"].numel() + offset, device="cuda")
        q = backing[offset:].view(gpu["q"].shape)
        q.copy_(gpu["q"])
        gpu_op.sample_dense_proposals(
            q, gpu["temperatures"], **{k: gpu[k] for k in common}, out_ids=gpu["drafts"]
        )
        assert torch.equal(gpu["drafts"].cpu(), cpu["drafts"])


@pytest.mark.parametrize("sparse", [False, True])
def test_tiled_zero_fallback_subnormal_and_argmax_ties(sparse):
    vocab = 8195
    cpu_op, cpu = buffers(3, 1, vocab)
    cpu["p"].zero_()
    cpu["q"].zero_()
    cpu["drafts"].zero_()
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    cpu["q"][:, :, 0] = tiny  # proposals still have strictly positive q(x)
    cpu["q"][:, :, 127] = 1
    cpu["p"][:, :, 127] = 1
    cpu["p"][0, :, 0] = tiny
    # For this frozen key/cursor, acceptance U>0.5: FP32(U*tiny)==tiny.
    # Thus p=q can reject through subnormal rounding, exercising zero fallback
    # without violating the positive-q(x) admission precondition.
    cpu["seeds"][0] = seed_words([5])[0]
    # Exact-zero residual must fall back to p; positive subnormal must not.
    cpu["p"][1, 0, 8192] = tiny
    # Ties across warp/tile/tail boundaries choose the smaller global ID.
    cpu["p"][2, 0, [128, 4096, 8194]] = 1
    cpu["temperatures"][2] = 0
    if sparse:
        cpu["cand_ids"] = torch.full((3, 1, 16), 127, dtype=torch.int32)
        cpu["cand_ids"][:, :, 0] = 0
        cpu["cand_q"] = torch.zeros((3, 1, 16))
        cpu["cand_q"][:, :, 0] = tiny
        cpu["cand_q"][:, :, 1:3] = 0.5  # original-column duplicate aggregation
    verify(cpu_op, cpu, sparse)
    assert cpu["out_tokens"][:, 0].tolist() == [127, 8192, 127]
    assert cpu["accepted_counts"].tolist() == [0, 0, 0]
    gpu_op = RequestLocalChainSampling("cuda", 2, vocab)
    for offset in (0, 1):
        gpu = {k: t.cuda() for k, t in cpu.items()}
        for name in ("p", "q"):
            value = gpu[name]
            backing = torch.empty(value.numel() + offset, device="cuda")
            gpu[name] = backing[offset:].view(value.shape)
            gpu[name].copy_(value)
        verify(gpu_op, gpu, sparse)
        assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
        assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])


# Probability construction is a deterministic qualification, separate from the
# frozen statistical family above. These tests add no hypotheses to M=3948.
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=pytest.mark.exhaustive),
        pytest.param(torch.float16, marks=pytest.mark.exhaustive),
        torch.float32,
    ],
)
def test_probabilities_exact_goldens(dtype):
    from ..modules.test_dflash_probabilities import golden_probability_data

    cpu, expected = golden_probability_data(dtype)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", 2, 6)
    op.probabilities(**gpu)
    assert torch.equal(gpu["out_probs"].cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "dtype,vocab",
    ci_cases(
        [torch.bfloat16, torch.float16, torch.float32],
        [32, 4097, 50000, 248320],
        core=[(torch.bfloat16, 32)],
    ),
)
def test_probabilities_fp64_oracle(vocab, dtype):
    from ..modules.test_dflash_probabilities import (
        assert_probability_tolerance,
        margin_probability_data,
        oracle,
    )

    cpu = margin_probability_data(vocab, dtype)
    expected = oracle(cpu)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    # Offset by one native element: exercise scalar fallbacks and row alignment.
    backing = torch.empty(gpu["logits"].numel() + 1, dtype=dtype, device="cuda")
    logits = backing[1:].view(gpu["logits"].shape)
    logits.copy_(gpu["logits"])
    gpu["logits"] = logits
    RequestLocalChainSampling("cuda", 2, vocab).probabilities(**gpu)
    assert_probability_tolerance(gpu["out_probs"].cpu(), expected)


def test_probabilities_replay_permutation_padding_and_mixed_rows():
    from ..modules.test_dflash_probabilities import margin_probability_data, probability_data

    vocab, length = 4097, 3
    cpu = margin_probability_data(vocab)
    logits = torch.cat((cpu["logits"], cpu["logits"][:2]), 0).repeat(1, length, 1)
    mixed = probability_data(
        logits,
        torch.cat((cpu["temperatures"], torch.tensor([0.0, float("nan")]))),
        torch.cat((cpu["top_k"], torch.tensor([1, -999], dtype=torch.int32))),
        torch.cat((cpu["top_p"], torch.tensor([0.01, float("nan")]))),
        torch.tensor([7, 3, 10, 0, 8, -1], dtype=torch.int32),
    )
    mixed["logits"][-1].fill_(float("nan"))
    gpu = {name: value.cuda() for name, value in mixed.items()}
    op = RequestLocalChainSampling("cuda", length, vocab)
    op.probabilities(**gpu)  # warm before capture
    expected = gpu["out_probs"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.probabilities(**gpu)
    for _ in range(100):
        graph.replay()
        assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
    assert (expected[-1] == 0).all()
    assert (expected[-2].sum(-1) == 1).all()
    for b in range(6):
        one = {
            name: value[b : b + 1] if name != "workspace" else value for name, value in gpu.items()
        }
        one["out_probs"] = torch.empty_like(one["out_probs"])
        op.probabilities(**one)
        assert torch.equal(one["out_probs"], expected[b : b + 1])
    for permutation in (torch.arange(5, -1, -1), torch.arange(6).roll(2)):
        permuted = {
            name: value[permutation].contiguous() if name != "workspace" else value
            for name, value in gpu.items()
        }
        op.probabilities(**permuted)
        assert torch.equal(permuted["out_probs"], expected[permutation])
    padded = {}
    for name, value in gpu.items():
        if name == "workspace":
            padded[name] = value
        else:
            padded[name] = torch.cat((value[-1:], value, value[-1:]), 0)
    padded["workspace"] = torch.empty(
        op.workspace_size(8, "probabilities", length=length), dtype=torch.uint8, device="cuda"
    )
    op.probabilities(**padded)
    assert torch.equal(padded["out_probs"][1:-1], expected)
    assert (padded["out_probs"][[0, -1]] == 0).all()
    # Captured kernels consume changed device filters without recompilation.
    gpu["temperatures"][:-1].zero_()
    graph.replay()
    captured = gpu["out_probs"].clone()
    op.probabilities(**gpu)
    assert torch.equal(captured, gpu["out_probs"])


# Valid FP32 temperatures outside the fast-division window: subnormal and FLT_MIN
# multiples (rcp.rn(T) can be inf) and very large T. The contract leaves FP32 exp
# provider-specific, so a row is bitwise against the reference wherever every
# exponential is exact ((l - max)/T is 0 or underflows exp to 0); other rows must
# keep the same support and match within one FP32 exp rounding. Large rows reach
# +-3e38, where l/T overflows for tiny T: (l - max)/T must never form inf - inf.
EXTREME_TEMPERATURES = (
    2.0**-149,
    2.0**-140,
    2.0**-127,
    2.0**-126,
    3 * 2.0**-126,
    2.0**-31,
    1e30,
    2.0**127,
    3.4e38,
)


def exact_exponentials(logits, temperatures):
    """Rows whose every scaled exponent (l - max)/T is 0 or far below FP32 exp range."""
    rows = logits.float()
    z = (rows - rows.amax(-1, keepdim=True)) / temperatures.view(-1, 1, 1)
    return ((z == 0) | (z <= -110)).all(-1)


@pytest.mark.parametrize("filters", ["none", "top_k", "top_p", "both"])
@pytest.mark.parametrize("top_k", [7, 2050])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_probabilities_extreme_temperatures_match_reference(filters, top_k, dtype):
    from ..modules.test_dflash_probabilities import probability_data

    if filters in ("none", "top_p") and top_k != 7:
        pytest.skip("top_k only applies to top-k hints")
    vocab, length = 4100, 2
    temps = torch.tensor(EXTREME_TEMPERATURES, dtype=torch.float32)
    n = temps.numel()
    gen = torch.Generator().manual_seed(149)
    logits = torch.zeros(3 * n, length, vocab)
    steps = torch.randint(-8, 9, (2, n, length, vocab), generator=gen).float()
    # Small multiples of 2^-133 and large multiples of 2^120 (both exact in BF16),
    # plus +-3e38 entries; every temperature sees the same three rows.
    logits[n : 2 * n] = steps[0] * 2.0**-133
    logits[2 * n :] = steps[1] * 2.0**120
    logits[2 * n :, :, 1::97] = 3e38
    logits[2 * n :, :, 2::89] = -3e38
    logits = logits.to(dtype)
    data = probability_data(
        logits,
        temps.repeat(3),
        torch.full((3 * n,), top_k if filters in ("top_k", "both") else vocab, dtype=torch.int32),
        torch.full((3 * n,), 0.5 if filters in ("top_p", "both") else 1.0),
    )
    expected = {name: value.clone() for name, value in data.items()}
    RequestLocalChainSampling("cpu", length, vocab).probabilities(**expected, filters=filters)
    gpu = {name: value.cuda() for name, value in data.items()}
    RequestLocalChainSampling("cuda", length, vocab).probabilities(**gpu, filters=filters)
    actual = gpu["out_probs"].cpu()
    reference = expected["out_probs"]
    assert not torch.isnan(reference).any() and not torch.isnan(actual).any()
    assert ((reference.double().sum(-1) - 1).abs() <= 2e-6).all()
    exact = exact_exponentials(logits, temps.repeat(3))
    inexact = ~exact
    assert torch.equal(actual[exact].view(torch.int32), reference[exact].view(torch.int32))
    assert torch.equal(actual[inexact] > 0, reference[inexact] > 0)
    torch.testing.assert_close(actual[inexact], reference[inexact], rtol=4e-7, atol=0)
    if filters == "none":
        # Identity probabilities match the multi-phase path bitwise on device.
        auto = {name: value.clone() for name, value in gpu.items()}
        RequestLocalChainSampling("cuda", length, vocab).probabilities(**auto, filters="auto")
        assert torch.equal(auto["out_probs"].view(torch.int32), gpu["out_probs"].view(torch.int32))
        # All-zero logits are uniform at every valid temperature.
        assert (actual[:n] == actual[0, 0, 0]).all() and actual[0, 0, 0] > 0


def test_probabilities_split_ties_across_cta_tiles():
    from ..modules.test_dflash_probabilities import probability_data

    data = probability_data(torch.zeros(1, 2, 4097))
    data["top_k"].fill_(1024)
    data["top_p"].fill_(0.75)
    gpu = {name: value.cuda() for name, value in data.items()}
    op = RequestLocalChainSampling("cuda", 2, 4097)
    op.probabilities(**gpu)
    expected = torch.zeros_like(data["out_probs"])
    expected[:, :, :768] = 0.0013020833721384406
    assert torch.equal(gpu["out_probs"].cpu(), expected)


def test_probabilities_to_dense_chain():
    from ..modules.test_dflash_probabilities import probability_data

    batch, gamma, vocab = 6, 2, 32
    cpu_op, cpu = buffers(batch, gamma, vocab)
    cpu["temperatures"] = torch.tensor([0.0, 0.7, 1.0, 1.2, 0.9, float("nan")])
    cpu["request_indices"][-1] = -1
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", gamma + 1, vocab)
    gen = torch.Generator().manual_seed(7721)
    for name, length in (("p", gamma + 1), ("q", gamma)):
        data = probability_data(
            torch.randn(batch, length, vocab, generator=gen),
            cpu["temperatures"],
            torch.tensor([1, vocab, 8, vocab, 8, -999], dtype=torch.int32),
            torch.tensor([0.1, 1, 1, 0.8, 0.8, float("nan")]),
            cpu["request_indices"],
        )
        data["logits"][-1].fill_(float("nan"))
        data = {key: value.cuda() for key, value in data.items()}
        data["out_probs"] = gpu[name]
        op.probabilities(**data)
        cpu[name] = gpu[name].cpu()  # identical GPU-produced distributions for both samplers
    common = ("request_indices", "seeds", "cursors", "workspace")
    for operation, tensors in ((op, gpu), (cpu_op, cpu)):
        operation.sample_dense_proposals(
            tensors["q"],
            tensors["temperatures"],
            **{name: tensors[name] for name in common},
            out_ids=tensors["drafts"],
        )
        verify(operation, tensors)
    for name in ("drafts", "out_tokens", "accepted_counts"):
        assert torch.equal(gpu[name].cpu(), cpu[name])


@pytest.mark.parametrize("name", ["logits", "temperatures", "top_p", "out_probs"])
def test_probabilities_requires_grad(name):
    from ..modules.test_dflash_probabilities import probability_data

    data = {key: value.cuda() for key, value in probability_data(torch.zeros(1, 1, 4)).items()}
    data[name].requires_grad_(True)
    with torch.no_grad(), pytest.raises(ValueError, match="requires grad"):
        RequestLocalChainSampling("cuda", 2, 4).probabilities(**data)


@pytest.mark.parametrize(
    "vocab",
    ci_cases([4097, 50000, 248320], core=[(50000,)]),
)
def test_probabilities_ties_in_multiple_chunks(vocab):
    from ..modules.test_dflash_probabilities import chunk_tie_probability_data

    data, expected = chunk_tie_probability_data(vocab)
    gpu = {name: value.cuda() for name, value in data.items()}
    op = RequestLocalChainSampling("cuda", 2, vocab)
    # A dirty first use is valid. No persistent zero state is required.
    gpu["workspace"].fill_(255)
    op.probabilities(**gpu)
    assert torch.equal(gpu["out_probs"].cpu().view(torch.int32), expected.view(torch.int32))
    # L-independent geometry: the same position as a length-one call.
    one = dict(gpu)
    one["logits"] = gpu["logits"][:, :1].contiguous()
    one["out_probs"] = torch.empty_like(one["logits"], dtype=torch.float32)
    op.probabilities(**one)
    assert torch.equal(one["out_probs"].cpu(), expected[:, :1])


def test_probabilities_row_state_arrivals_and_dirty_replay():
    from ..modules.test_dflash_probabilities import probability_data

    # Three chunks, including a one-element tail. For each position, two chunks
    # have no selected finite keys; they must still arrive at every used level.
    batch, length, vocab = 6, 2, 8193
    logits = torch.full((batch, length, vocab), float("-inf"))
    logits[:, 0, :4096] = 0
    logits[:, 1, 4096:8192] = 0
    cpu = probability_data(
        logits,
        torch.tensor([1, 1, 1, 0, float("nan"), 1]),
        torch.tensor([vocab, 2048, 2, 1, -999, vocab], dtype=torch.int32),
        torch.tensor([0.5, 0.5, 0.5, 0.5, float("nan"), 1]),
        torch.tensor([0, 1, 2, 3, -1, 5], dtype=torch.int32),
    )
    cpu["logits"][4].fill_(float("nan"))
    expected = torch.zeros_like(logits)
    for position, start in enumerate((0, 4096)):
        for row, kept in ((0, 2048), (1, 1024), (2, 1), (3, 1), (5, 4096)):
            expected[row, position, start : start + kept] = 1 / kept
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", length, vocab)
    region = next(
        r
        for r in op.workspace_layout(batch, "probabilities")["regions"]
        if r["name"] == "row_states"
    )
    state_bytes = gpu["workspace"][region["offset"] : region["offset"] + region["payload_bytes"]]
    states = state_bytes.view(torch.int64).view(batch, length, 6, 6)
    arrivals = torch.tensor(
        [
            [0, 0, 0, 3, 3, 3],  # large top-p only
            [3, 3, 3, 3, 3, 3],  # large top-k and top-p
            [3, 3, 3, 0, 0, 0],  # compact top-k and top-p
            [0, 0, 0, 0, 0, 0],  # greedy
            [0, 0, 0, 0, 0, 0],  # padding
            [0, 0, 0, 0, 0, 0],  # identity device filters
        ],
        dtype=torch.int64,
    )[:, None, :].expand(batch, length, 6)
    gpu["workspace"].fill_(255)
    op.probabilities(**gpu)
    assert torch.equal(gpu["out_probs"].cpu(), expected)
    assert torch.equal(states[..., 0].cpu(), arrivals)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.probabilities(**gpu)
    for dirty in (90, 255):
        gpu["workspace"].fill_(dirty)
        graph.replay()
        assert torch.equal(gpu["out_probs"].cpu(), expected)
        assert torch.equal(states[..., 0].cpu(), arrivals)
    # The three-launch identity specialization does not touch any row state.
    gpu["top_k"].fill_(vocab)
    gpu["top_p"].fill_(1)
    state_bytes.fill_(90)
    op.probabilities(**gpu, filters="none")
    assert torch.all(state_bytes == 90)


@pytest.mark.parametrize(
    "hint",
    [
        pytest.param("none", marks=pytest.mark.exhaustive),
        pytest.param("top_k", marks=pytest.mark.exhaustive),
        "top_p",
        pytest.param("both", marks=pytest.mark.exhaustive),
        "auto",
    ],
)
def test_probabilities_hint_reuse_and_graph(hint):
    from ..modules.test_dflash_probabilities import margin_probability_data

    vocab = 50000
    cpu = margin_probability_data(vocab, torch.bfloat16)
    if hint in ("none", "top_p"):
        cpu["top_k"].fill_(vocab)
    if hint in ("none", "top_k"):
        cpu["top_p"].fill_(1)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", 2, vocab)
    gpu["workspace"].fill_(165)
    op.probabilities(**gpu)
    expected = gpu["out_probs"].clone()
    op.probabilities(**gpu, filters=hint)
    assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.probabilities(**gpu, filters=hint)
    for _ in range(100):
        graph.replay()
        assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
    # Reuse across both -> padding -> greedy -> identity -> both. The captured
    # hint then sees the original data, after all scratch has changed meaning.
    for mode in ("both", "padding", "greedy", "identity", "both"):
        gpu["request_indices"].copy_(cpu["request_indices"])
        gpu["temperatures"].fill_(1)
        gpu["top_k"].fill_(8)
        gpu["top_p"].fill_(0.7)
        if mode == "padding":
            gpu["request_indices"].fill_(-1)
        elif mode == "greedy":
            gpu["temperatures"].zero_()
        elif mode == "identity":
            gpu["top_k"].fill_(vocab)
            gpu["top_p"].fill_(1)
        op.probabilities(**gpu)
        reused = gpu["out_probs"].clone()
        fresh = {**gpu, "workspace": torch.empty_like(gpu["workspace"])}
        op.probabilities(**fresh)
        assert torch.equal(gpu["out_probs"], reused)
    for name in ("request_indices", "temperatures", "top_k", "top_p"):
        gpu[name].copy_(cpu[name])
    graph.replay()
    assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=pytest.mark.exhaustive),
        pytest.param(torch.float16, marks=pytest.mark.exhaustive),
        torch.float32,
    ],
)
def test_probabilities_compact_capacity_and_cross_chunk_ties(dtype):
    from ..modules.test_dflash_probabilities import compact_boundary_probability_data

    cpu, expected = compact_boundary_probability_data(dtype)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", 2, 9001)
    op.probabilities(**gpu, filters="both")
    assert torch.equal(gpu["out_probs"].cpu().view(torch.int32), expected.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.probabilities(**gpu, filters="both")
    for _ in range(20):
        gpu["workspace"].fill_(165)
        gpu["out_probs"].fill_(float("nan"))
        graph.replay()
        assert torch.equal(gpu["out_probs"].cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "dtype,offset",
    ci_cases(
        [torch.bfloat16, torch.float16, torch.float32],
        [0, 1],
        core=[(torch.float32, 1)],
    ),
)
def test_probabilities_compact_matches_large_and_graph_mode_changes(dtype, offset):
    from ctikernels.sampling._dflash.cute import Provider
    from ctikernels.sampling._dflash.cute.kernels import Probabilities
    from ctikernels.sampling._dflash.reference import fixed_mass

    from ..modules.test_dflash_probabilities import probability_data

    batch, length, vocab = 6, 2, 8193
    generator = torch.Generator().manual_seed(821)
    backing = torch.randn(batch * length * vocab + offset, generator=generator).to(dtype).cuda()
    logits = backing[offset:].view(batch, length, vocab)
    cpu = probability_data(
        torch.empty(batch, length, vocab, dtype=dtype),
        torch.tensor([0.6, 0.7, 1, 0.6, 0.7, 1]),
        torch.tensor([20, 1024, 1025, vocab, 20, 1024], dtype=torch.int32),
        torch.tensor([0.95, 0.8, 0.9, 0.95, 1, 0.9999999403953552]),
    )
    gpu = {name: value.cuda() for name, value in cpu.items()}
    gpu["logits"] = logits
    # A private compile-time override exercises the full-vocabulary algorithm
    # on exactly the same k and logits. It retains the original Z/tie formula.
    full = Probabilities(length, vocab, "both", batch)
    full.compact_cap = 0
    expected = torch.empty_like(gpu["out_probs"])
    baseline_args = (
        logits,
        gpu["temperatures"],
        gpu["top_k"],
        torch.ones_like(gpu["top_p"]),
        gpu["request_indices"],
        expected,
        torch.empty_like(gpu["workspace"]),
    )
    baseline = Provider._compile(full, baseline_args)
    op = RequestLocalChainSampling("cuda", 2, vocab)
    op.probabilities(**gpu, filters="both")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.probabilities(**gpu, filters="both")
    for ks in ([20, 1024, 1025, vocab, 20, 1024], [1025, 20, 1024, 20, vocab, 1025]):
        gpu["top_k"].copy_(torch.tensor(ks, dtype=torch.int32))
        baseline(*baseline_args)
        # Freeze the old P5 write path's actual device p bits, then independently
        # sort/sum exact masses on CPU. This also checks every recompute pass
        # against stored p without depending on CPU exp matching device exp.
        plain = expected.cpu()
        wanted = torch.zeros_like(plain)
        for row in range(batch):
            threshold = fixed_mass(cpu["top_p"][row].item(), ceiling=True)
            for position in range(length):
                values = plain[row, position]
                if cpu["top_p"][row] == 1:
                    wanted[row, position] = values
                else:
                    preceding, selected = 0, []
                    for token in torch.argsort(values, descending=True, stable=True).tolist():
                        if preceding < threshold:
                            selected.append(token)
                            preceding += fixed_mass(values[token].item())
                    scale = torch.tensor(preceding, dtype=torch.int64).to(torch.float32) * (
                        2.0**-62
                    )
                    wanted[row, position, selected] = values[selected] / scale
        expected.copy_(wanted)
        for _ in range(5):
            gpu["workspace"].fill_(90)
            gpu["out_probs"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "block_size,vocab",
    ci_cases([2, 8, 16], [3, 4097], core=[(2, 3), (16, 4097)]),
)
def test_point_mass_dense_reference_replay(block_size, vocab):
    # Deterministic equivalence gates; no additional statistical hypotheses.
    g = block_size - 1
    b = g + 4
    cpu_op, cpu = buffers(b, g, vocab)
    cpu["drafts"].copy_(cpu["q"].argmax(-1))
    cpu["q"].zero_().scatter_(-1, cpu["drafts"].long().unsqueeze(-1), 1)
    # Force every prefix and leave two random rows plus poisoned padding.
    cpu["p"][: g + 1, :g] = cpu["q"][: g + 1]
    for a in range(g):
        draft = int(cpu["drafts"][a, a])
        cpu["p"][a, a].zero_()
        cpu["p"][a, a, (draft + 1) % vocab] = 1
    cpu["request_indices"][-1] = -1
    cpu["drafts"][-1].fill_(-999)
    cpu["p"][-1].fill_(float("nan"))
    cpu["q"][-1].fill_(float("nan"))
    cpu["cursors"][-1] = -1
    op = RequestLocalChainSampling("cuda", block_size, vocab)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    for greedy in (False, True):
        cpu["temperatures"].fill_(0 if greedy else 1)
        cpu["temperatures"][-1] = float("nan")
        gpu["temperatures"].copy_(cpu["temperatures"])
        verify(cpu_op, cpu, point=True)
        verify(op, gpu)
        assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
        assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])
        verify(op, gpu, point=True)  # warm before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify(op, gpu, point=True)
        for _ in range(3):
            graph.replay()
        assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
        assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])
        assert cpu["accepted_counts"][: g + 1].tolist() == list(range(g + 1))
        # Reorder all row inputs in place, including padding, and mutate cursors.
        order = torch.arange(b - 1, -1, -1)
        for key in ("p", "q", "drafts", "temperatures", "request_indices", "cursors"):
            cpu[key].copy_(cpu[key][order])
            gpu[key].copy_(cpu[key])
        cpu["cursors"].add_(100003)
        gpu["cursors"].copy_(cpu["cursors"])
        verify(cpu_op, cpu, point=True)
        graph.replay()
        expected, counts = gpu["out_tokens"].clone(), gpu["accepted_counts"].clone()
        assert torch.equal(expected.cpu(), cpu["out_tokens"])
        assert torch.equal(counts.cpu(), cpu["accepted_counts"])
        verify(op, gpu)
        assert torch.equal(gpu["out_tokens"], expected)
        assert torch.equal(gpu["accepted_counts"], counts)
        for row in range(b):
            one_op, one = buffers(1, g, vocab, device="cuda")
            for key in ("p", "q", "drafts", "temperatures", "request_indices", "cursors"):
                one[key].copy_(gpu[key][row : row + 1])
            one["seeds"] = gpu["seeds"]
            verify(one_op, one, point=True)
            assert torch.equal(one["out_tokens"][0], expected[row])
            assert one["accepted_counts"][0] == counts[row]
        # Restore row order for the next temperature mode.
        for key in ("p", "q", "drafts", "temperatures", "request_indices", "cursors"):
            cpu[key].copy_(cpu[key][order])
            gpu[key].copy_(cpu[key])


@pytest.mark.parametrize(
    "batch,hint",
    ci_cases([1, 2, 3, 4], ["top_k", "top_p", "both"], core=[(4, "top_p")]),
)
def test_probabilities_to_point_mass_chain_graph(batch, hint):
    from ..modules.test_dflash_probabilities import probability_data

    gamma, vocab = 2, 4097
    cpu_op, cpu = buffers(batch, gamma, vocab)
    op = RequestLocalChainSampling("cuda", gamma + 1, vocab)
    # Four equal finite logits span multiple chunks. Filtering must remove
    # finite-mass alternatives, including the draft in the excluded case.
    logits = torch.full((batch, gamma + 1, vocab), float("-inf"))
    logits[:, :, [0, 128, 2048, 4096]] = 0
    for support in ("excludes_draft", "only_draft", "includes_draft"):
        singleton = support == "only_draft"
        cpu["drafts"].fill_(4096 if support == "excludes_draft" else 0)
        cpu["q"].zero_().scatter_(-1, cpu["drafts"].long().unsqueeze(-1), 1)
        k = vocab if hint == "top_p" else (1 if singleton else 2)
        p = 1.0 if hint == "top_k" else (0.25 if singleton else 0.5)
        # With both filters, k=3 followed by p=0.5 retains two entries.
        if hint == "both":
            k = 3
        data = probability_data(
            logits,
            cpu["temperatures"],
            torch.full((batch,), k, dtype=torch.int32),
            torch.full((batch,), p),
            cpu["request_indices"],
        )
        gpu = {name: value.cuda() for name, value in cpu.items()}
        probabilities = {name: value.cuda() for name, value in data.items()}
        probabilities["out_probs"] = gpu["p"]

        def call(probabilities=probabilities, gpu=gpu):
            op.probabilities(**probabilities, filters=hint)
            verify(op, gpu, point=True)

        gpu["workspace"].fill_(165)
        probabilities["workspace"].fill_(165)
        call()  # direct eager chain, also warms both operations before capture
        cpu["p"] = gpu["p"].cpu()  # same FP32 p bits for the CPU dense one-hot oracle
        assert (torch.count_nonzero(cpu["p"], dim=-1) == (1 if singleton else 2)).all()
        draft_mass = cpu["p"][:, :gamma].gather(-1, cpu["drafts"].long().unsqueeze(-1))
        expected_mass = 0 if support == "excludes_draft" else (1 if singleton else 0.5)
        assert (draft_mass == expected_mass).all()
        verify(cpu_op, cpu)
        if support != "includes_draft":
            assert (cpu["accepted_counts"] == (gamma if singleton else 0)).all()
        for name in ("out_tokens", "accepted_counts"):
            assert torch.equal(gpu[name].cpu(), cpu[name])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for garbage in (255, 90, 165):
            gpu["workspace"].fill_(garbage)
            probabilities["workspace"].fill_(garbage)
            gpu["p"].fill_(float("nan"))
            gpu["out_tokens"].fill_(99)
            gpu["accepted_counts"].fill_(99)
            graph.replay()
            assert torch.equal(gpu["p"].cpu().view(torch.int32), cpu["p"].view(torch.int32))
            for name in ("out_tokens", "accepted_counts"):
                assert torch.equal(gpu[name].cpu(), cpu[name])


@pytest.mark.parametrize("name", ["p", "temperatures"])
def test_point_mass_requires_grad(name):
    op, data = buffers(device="cuda")
    data[name].requires_grad_(True)
    with pytest.raises(ValueError, match="requires grad"):
        verify(op, data, point=True)


@pytest.mark.parametrize(
    "mode,b,length",
    ci_cases(
        ["point", "dense", "sparse"],
        [1, 2, 3, 4],
        [4, 5, 8, 9, 12, 13, 16, 17],
        core=[("point", 1, 4), ("dense", 4, 5), ("sparse", 4, 16)],
    ),
)
def test_draft_length_sweep(length, b, mode):
    # Q counts the anchor: 3/4/7/8/11/12/15/16 drafts. Deterministic gates only.
    g = length - 1
    vocab = 5 if mode == "sparse" else 4097
    cpu_op, base = buffers(b, g, vocab)
    if mode == "point":
        base["drafts"].copy_(base["q"].argmax(-1))
        base["q"].zero_().scatter_(-1, base["drafts"].long().unsqueeze(-1), 1)
    op = RequestLocalChainSampling("cuda", length, vocab)
    gpu = {k: t.cuda() for k, t in base.items()}
    graph = None
    for start in range(0, g + 1, b):
        cpu = {k: t.clone() for k, t in base.items()}
        forced = [(start + r) % (g + 1) for r in range(b)]
        for r, a in enumerate(forced):
            # Accept exactly a drafts at T>0: p equals q on the prefix and puts
            # no mass on the next draft; a=g leaves a random bonus row.
            cpu["p"][r, :a, :] = 0
            cpu["p"][r, :a, : cpu["q"].shape[-1]] = cpu["q"][r, :a]
            if a < g:
                draft = int(cpu["drafts"][r, a])
                cpu["p"][r, a].zero_()
                cpu["p"][r, a, (draft + 1) % vocab] = 1
        if mode == "sparse":
            sparsify(cpu)
        for temperature in (1.0, 0.0):
            cpu["temperatures"].fill_(temperature)
            for key, value in cpu.items():
                if key in gpu:
                    gpu[key].copy_(value)
                else:
                    gpu[key] = value.cuda()
            verify(cpu_op, cpu, mode == "sparse", point=mode == "point")
            if temperature > 0 or mode == "point":
                assert cpu["accepted_counts"].tolist() == forced
            verify(op, gpu, mode == "sparse", point=mode == "point")
            assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
            assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    verify(op, gpu, mode == "sparse", point=mode == "point")
            gpu["out_tokens"].fill_(99)
            gpu["accepted_counts"].fill_(99)
            graph.replay()
            assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
            assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])


@pytest.mark.parametrize(
    "batch,length,vocab,dtype",
    [
        (5, 2, 4097, torch.float16),
        (5, 3, 151936, torch.bfloat16),
        (5, 2, 248320, torch.float32),
        (5, 2, 262145, torch.bfloat16),
    ],
)
def test_probabilities_identity_groups(batch, length, vocab, dtype):
    from ctikernels.sampling._dflash.cute import kernels
    from ctikernels.sampling._dflash.cute.provider import Provider

    from ..modules.test_dflash_probabilities import probability_data

    gen = torch.Generator().manual_seed(vocab)
    logits = torch.randn(batch, length, vocab, generator=gen).to(dtype)
    # Avoid inf-inf at tiny T; BF16/FP32 retain these small multiples exactly.
    logits[3] = (torch.randint(-8, 9, (length, vocab), generator=gen).float() * 2.0**-133).to(dtype)
    logits[-1].fill_(float("nan"))
    cpu = probability_data(
        logits,
        torch.tensor([0.7, 1.0, 0.0, 2.0**-149, float("nan")]),
        slots=torch.tensor([7, 3, 0, 2, -1], dtype=torch.int32),
    )
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", length, vocab)
    tensors = tuple(
        gpu[name]
        for name in (
            "logits",
            "temperatures",
            "top_k",
            "top_p",
            "request_indices",
            "out_probs",
            "workspace",
        )
    )
    # Bypass the provider cache: its key does not contain the identity group.
    variants = [
        Provider._compile(kernels.Probabilities(length, vocab, "none", batch, group), tensors)
        for group in (1, 2)
    ]
    op.probabilities(**gpu, filters="auto")
    expected = gpu["out_probs"].clone()
    assert not expected.isnan().any().item()
    for compiled in variants:
        gpu["out_probs"].fill_(-99)
        compiled(*tensors)
        assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))

    # exp(0)/exp(-inf) are exact across providers; irregular masks exercise
    # chunk sums, partial chunks and first-ID greedy ties without exp ULP noise.
    cpu["logits"].fill_(-float("inf"))
    cpu["logits"][:, :, 3::7] = 0
    cpu["logits"][1, 0].zero_()
    cpu["logits"][3].zero_()  # Uniform p at the smallest positive FP32 T.
    cpu["logits"][-1].fill_(float("nan"))
    RequestLocalChainSampling("cpu", length, vocab).probabilities(**cpu, filters="none")
    gpu["logits"].copy_(cpu["logits"])
    expected = cpu["out_probs"].cuda()
    op.probabilities(**gpu, filters="auto")
    assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
    for compiled in variants:
        gpu["out_probs"].fill_(-99)
        compiled(*tensors)
        assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(*tensors)
        for _ in range(20):
            gpu["out_probs"].fill_(-99)
            graph.replay()
            assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))
    assert (expected[-1] == 0).all().item()
    assert (expected[3] == expected[3, 0, 0]).all().item()
    # Also qualify the provider-selected identity specialization and launch count.
    op.probabilities(**gpu, filters="none")
    from benchmark.dflash_sampling.bench import count_launches

    call = lambda: op.probabilities(**gpu, filters="none")
    assert count_launches(call)["kernel_launches"] == 3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(20):
        gpu["out_probs"].fill_(-99)
        graph.replay()
        assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("filters", ["none", "top_k", "top_p", "both"])
@pytest.mark.parametrize("vocab", [6, 4100])
def test_probabilities_overflowing_scaled_logits_match_reference(filters, vocab):
    from ..modules.test_dflash_probabilities import overflow_probability_data

    data, expected = overflow_probability_data(filters, vocab)
    gpu = {name: value.cuda() for name, value in data.items()}
    RequestLocalChainSampling("cuda", 2, vocab).probabilities(**gpu, filters=filters)
    actual = gpu["out_probs"].cpu()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_candidates_overflowing_scaled_scores_match_reference():
    from ..modules.test_dflash_sampling_reference import overflow_candidate_data

    cpu, expected_q = overflow_candidate_data()
    RequestLocalChainSampling("cpu", 8, 32).sample_candidates(position_i=3, **cpu)
    gpu = {k: t.cuda() for k, t in overflow_candidate_data()[0].items()}
    RequestLocalChainSampling("cuda", 8, 32).sample_candidates(position_i=3, **gpu)
    assert torch.equal(cpu["out_q"].view(torch.int32), expected_q.view(torch.int32))
    assert torch.equal(gpu["out_q"].cpu().view(torch.int32), expected_q.view(torch.int32))
    assert torch.equal(gpu["out_ids"].cpu(), cpu["out_ids"])


@pytest.mark.parametrize("point", [False, True])
def test_max_vocab_verify_boundary(point):
    # B=1, Q=2 at the contract's MAX_VOCAB: shared-memory staging fits and
    # every chunk, including the last, is read; one past the limit raises.
    from ctikernels.sampling._dflash import MAX_VOCAB

    v, gamma = MAX_VOCAB, 1
    op, cpu = buffers(1, gamma, v)
    cpu["drafts"].fill_(3)
    verify(op, cpu, point=point)
    gpu = {k: t.cuda() for k, t in cpu.items()}
    verify(RequestLocalChainSampling("cuda", gamma + 1, v), gpu, point=point)
    assert torch.equal(gpu["out_tokens"].cpu(), cpu["out_tokens"])
    assert torch.equal(gpu["accepted_counts"].cpu(), cpu["accepted_counts"])
    gpu["temperatures"].zero_()
    gpu["p"][0, 0].zero_()
    gpu["p"][0, 0, v - 1] = 1  # greedy winner in the last chunk
    verify(RequestLocalChainSampling("cuda", gamma + 1, v), gpu, point=point)
    assert gpu["out_tokens"][0, 0].item() == v - 1
    assert gpu["accepted_counts"].item() == 0
    with pytest.raises(ValueError, match="vocab_size"):
        RequestLocalChainSampling("cuda", 2, v + 1)


@pytest.mark.parametrize("filters", ["none", "auto"])
def test_max_vocab_probabilities_boundary(filters):
    from ctikernels.sampling._dflash import MAX_VOCAB

    from ..modules.test_dflash_probabilities import (
        assert_probability_tolerance,
        margin_probability_data,
        oracle,
    )

    cpu = margin_probability_data(MAX_VOCAB)
    if filters == "none":
        cpu["top_k"].fill_(MAX_VOCAB)
        cpu["top_p"].fill_(1)
    expected = oracle(cpu)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    RequestLocalChainSampling("cuda", 2, MAX_VOCAB).probabilities(**gpu, filters=filters)
    assert_probability_tolerance(gpu["out_probs"].cpu(), expected)


@pytest.mark.parametrize("filters", ["none", "top_k", "top_p", "both", "auto"])
@pytest.mark.parametrize("vocab", [2, 6, 4100])
def test_probabilities_extreme_spreads_match_fp64_oracle(filters, vocab):
    from ..modules.test_dflash_probabilities import (
        assert_extreme_probabilities,
        assert_masked_probabilities,
        extreme_probability_data,
        oracle,
    )

    data = extreme_probability_data(filters, vocab)
    expected = oracle(data)
    gpu = {name: value.cuda() for name, value in data.items()}
    RequestLocalChainSampling("cuda", 2, vocab).probabilities(**gpu, filters=filters)
    actual = gpu["out_probs"].cpu()
    assert_extreme_probabilities(actual, expected)
    assert_masked_probabilities(data, actual)
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=filters)
    torch.testing.assert_close(actual, data["out_probs"], rtol=4e-7, atol=1e-30)


@pytest.mark.parametrize("filters", ["none", "top_k", "top_p", "both", "auto"])
@pytest.mark.parametrize("vocab", [2, 6, 4100])
def test_probabilities_nearby_top_p_boundary_matches_fp64_oracle(filters, vocab):
    from ..modules.test_dflash_probabilities import (
        assert_extreme_probabilities,
        nearby_top_p_boundary_data,
        oracle,
    )

    data = nearby_top_p_boundary_data(vocab)
    if filters in ("none", "top_k"):
        data["top_p"].fill_(1)
    expected = oracle(data)
    gpu = {name: value.cuda() for name, value in data.items()}
    RequestLocalChainSampling("cuda", 2, vocab).probabilities(**gpu, filters=filters)
    actual = gpu["out_probs"].cpu()
    assert_extreme_probabilities(actual, expected)
    assert (actual[0, 0, 1] == 0) == (filters not in ("none", "top_k"))
    RequestLocalChainSampling("cpu", 2, vocab).probabilities(**data, filters=filters)
    torch.testing.assert_close(actual, data["out_probs"], rtol=4e-7, atol=1e-30)


def test_candidates_spread_scores_match_fp64():
    from ..modules.test_dflash_sampling_reference import spread_candidate_data

    cpu, expected = spread_candidate_data()
    RequestLocalChainSampling("cpu", 8, 32).sample_candidates(position_i=3, **cpu)
    gpu = {k: t.cuda() for k, t in spread_candidate_data()[0].items()}
    RequestLocalChainSampling("cuda", 8, 32).sample_candidates(position_i=3, **gpu)
    actual = gpu["out_q"].cpu()
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(actual, cpu["out_q"], rtol=4e-7, atol=1e-30)
    assert torch.equal(gpu["out_ids"].cpu(), cpu["out_ids"])


@pytest.mark.parametrize(
    "block,vocab,mode",
    [
        (q, 3 if mode == "sparse" else 257, mode)
        for q in (33, 64, 65, 128)
        for mode in ("dense", "sparse", "point")
    ]
    + [(64, 4097, "dense"), (64, 151936, "point"), (513, 3, "point")],
)
@pytest.mark.parametrize("greedy", [False, True])
def test_long_chain_reference_graph(block, vocab, mode, greedy):
    from ..modules.test_dflash_sampling_reference import long_chain_data

    op = RequestLocalChainSampling("cuda", block, vocab)
    for rejection in sorted({0, 31, 32, min(63, block - 1), min(64, block - 1), block - 1}):
        cpu_op, cpu = long_chain_data(block, vocab, mode, greedy, rejection)
        # The large-vocabulary case qualifies exactly one request row.
        if vocab == 151936:
            for name in (
                "p",
                "q",
                "drafts",
                "temperatures",
                "request_indices",
                "cursors",
                "out_tokens",
                "accepted_counts",
            ):
                cpu[name] = cpu[name][:1]
        gpu = {name: value.cuda() for name, value in cpu.items()}
        verify(cpu_op, cpu, mode == "sparse", point=mode == "point")
        verify(op, gpu, mode == "sparse", point=mode == "point")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify(op, gpu, mode == "sparse", point=mode == "point")
        for _ in range(2):
            gpu["out_tokens"].fill_(99)
            gpu["accepted_counts"].fill_(99)
            graph.replay()
            for name in ("out_tokens", "accepted_counts"):
                assert torch.equal(gpu[name].cpu(), cpu[name])
        if not greedy and rejection < block - 1:
            # Late acceptance coins must use the full position in Philox's i word.
            cpu["p"][0, rejection, :2] = 0.5
            gpu["p"].copy_(cpu["p"])
            verify(cpu_op, cpu, mode == "sparse", point=mode == "point")
            graph.replay()
            for name in ("out_tokens", "accepted_counts"):
                assert torch.equal(gpu[name].cpu(), cpu[name])


@pytest.mark.parametrize("block", [33, 64, 65, 128])
@pytest.mark.parametrize("filters", ["none", "both"])
def test_long_probabilities_and_proposals(block, filters):
    from ..modules.test_dflash_probabilities import long_probability_data

    cpu = long_probability_data(block, filters)
    cpu_op = RequestLocalChainSampling("cpu", block, 4)
    cpu_op.probabilities(**cpu, filters=filters)
    gpu = {name: value.cuda() for name, value in cpu.items()}
    op = RequestLocalChainSampling("cuda", block, 4)
    op.probabilities(**gpu, filters=filters)
    assert torch.equal(gpu["out_probs"].cpu().view(torch.int32), cpu["out_probs"].view(torch.int32))
    _, data = buffers(2, block - 1, 4)
    data["q"].copy_(cpu["out_probs"][:, :-1])
    data["request_indices"].copy_(cpu["request_indices"])
    data["out_tokens"] = torch.empty_like(data["drafts"])
    names = ("request_indices", "seeds", "cursors", "workspace")
    cpu_op.sample_dense_proposals(
        data["q"], data["temperatures"], out_ids=data["out_tokens"], **{n: data[n] for n in names}
    )
    staged = {n: value.cuda() for n, value in data.items()}

    def call():
        op.probabilities(**gpu, filters=filters)
        op.sample_dense_proposals(
            staged["q"],
            staged["temperatures"],
            out_ids=staged["out_tokens"],
            **{n: staged[n] for n in names},
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        graph.replay()
        assert torch.equal(staged["out_tokens"].cpu(), data["out_tokens"])
        assert torch.equal(
            gpu["out_probs"].cpu().view(torch.int32), cpu["out_probs"].view(torch.int32)
        )


@pytest.mark.parametrize("mode", ["dense", "sparse", "point"])
@pytest.mark.parametrize("vocab", [1, 5, 4097])
@pytest.mark.parametrize("padding_only", [False, True])
def test_q1_bonus_capture_replay(mode, vocab, padding_only):
    cpu_op, data = buffers(4, 0, vocab)
    data["temperatures"][1] = 0
    data["p"][1].fill_(1 / vocab)
    data["request_indices"][-1] = -1
    if padding_only:
        data["request_indices"].fill_(-1)
        data["seeds"] = torch.empty(0, 2, dtype=torch.uint32)
    inactive = data["request_indices"] < 0
    data["p"][inactive] = float("nan")
    data["temperatures"][inactive] = float("nan")
    data["cursors"][inactive] = -1
    data["cand_ids"] = torch.empty(4, 0, 16, dtype=torch.int32)
    data["cand_q"] = torch.empty(4, 0, 16)
    verify(cpu_op, data, mode == "sparse", point=mode == "point")
    gpu = {k: t.cuda() for k, t in data.items()}
    op = RequestLocalChainSampling("cuda", 1, vocab)
    # A guard catches writes beyond the exact reported workspace extent.
    size = op.workspace_size(4)
    backing = torch.full((size + 256,), 0xA5, dtype=torch.uint8, device="cuda")
    gpu["workspace"] = backing[:size]

    def run():
        op.sample_dense_proposals(
            gpu["q"],
            gpu["temperatures"],
            **{k: gpu[k] for k in ("request_indices", "seeds", "cursors", "workspace")},
            out_ids=gpu["drafts"],
        )
        verify(op, gpu, mode == "sparse", point=mode == "point")

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        gpu["out_tokens"].fill_(-99)
        gpu["accepted_counts"].fill_(99)
        graph.replay()
        assert torch.equal(gpu["out_tokens"].cpu(), data["out_tokens"])
        assert gpu["accepted_counts"].cpu().tolist() == [0] * 4
    gpu["cursors"].add_(17)
    data["cursors"].add_(17)
    verify(cpu_op, data, mode == "sparse", point=mode == "point")
    graph.replay()
    assert torch.equal(gpu["out_tokens"].cpu(), data["out_tokens"])
    assert (backing[size:] == 0xA5).all().item()


def test_q1_probabilities_cursors_capture():
    from ..modules.test_dflash_probabilities import golden_probability_data

    data, expected = golden_probability_data()
    gpu = {k: t.cuda() for k, t in data.items()}
    op = RequestLocalChainSampling("cuda", 1, 6)
    batch = len(gpu["request_indices"])
    anchors = torch.full((batch,), 23, dtype=torch.int64, device="cuda")
    prompts = torch.tensor([20], dtype=torch.int64, device="cuda")
    cursors = torch.empty_like(anchors)

    def run():
        op.probabilities(**gpu)
        op.prepare_cursors(
            anchors,
            prompts,
            request_indices=gpu["request_indices"],
            workspace=gpu["workspace"],
            out_cursors=cursors,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert torch.equal(gpu["out_probs"].cpu(), expected)
    assert torch.equal(cursors.cpu(), torch.where(data["request_indices"] < 0, 0, 4))


# SYM-B is a covering sweep, not a Cartesian product. Every requested B/L/V
# value occurs; large batches use V=16 and large vocabularies use B<=4, L<=2.
SYMB_SHAPES = [
    (0, 1, 16),
    (1, 2, 4095),
    (2, 4, 4097),
    (3, 5, 16),
    (4, 8, 16),
    (5, 9, 16),
    (8, 12, 16),
    (16, 16, 16),
    (32, 33, 16),
    (48, 64, 16),
    (191, 1, 16),
    (256, 2, 16),
    (1024, 1, 16),
    (1, 1, 151936),
    (2, 2, 248320),
    (1, 1, 2**21),
]
SYMB_HINTS = ("none", "top_k", "top_p", "both", "auto")
SYMB_CASES = [
    pytest.param(
        b,
        length,
        vocab,
        (torch.bfloat16, torch.float16, torch.float32)[i % 3],
        SYMB_HINTS[i % 5],
        bool(i % 2),
        marks=() if i in (0, 1, 2, 5) else pytest.mark.exhaustive,
    )
    for i, (b, length, vocab) in enumerate(SYMB_SHAPES)
]
SYMB_BATCHES = (0, 1, 2, 3, 4, 5, 8, 16, 32, 48, 191, 256, 1024)


def symb_probability_data(batch, length, vocab, dtype, hint, use_min_p):
    from ..modules.test_dflash_probabilities import probability_data

    # exp(0)/exp(-inf) makes the CPU/device comparison bitwise, including
    # compact/large reductions, exact tie selection and min-p renormalization.
    logits = torch.full((batch, length, vocab), -torch.inf, dtype=dtype)
    logits[..., :8] = 0
    data = probability_data(logits)
    if hint in ("top_k", "both", "auto"):
        data["top_k"].fill_(4)
    if hint in ("top_p", "both", "auto"):
        data["top_p"].fill_(0.75)
    data["temperatures"][1::3] = 0
    data["request_indices"][2::4] = -1
    if use_min_p:
        data["min_p"] = torch.full((batch,), 0.5)
    return data


@pytest.mark.parametrize("batch,length,vocab,dtype,hint,use_min_p", SYMB_CASES)
def test_symb_strided_probability_grid(batch, length, vocab, dtype, hint, use_min_p):
    from ..modules.test_dflash_symb import strided_copy

    data = symb_probability_data(batch, length, vocab, dtype, hint, use_min_p)
    RequestLocalChainSampling("cpu", length, vocab).probabilities(**data, filters=hint)
    gpu = {name: value.cuda() for name, value in data.items()}
    gpu["logits"] = strided_copy(gpu["logits"], 3)
    op = RequestLocalChainSampling("cuda", length, vocab)
    gpu["out_probs"].fill_(-17)
    op.probabilities(**gpu, filters=hint)
    assert torch.equal(
        gpu["out_probs"].cpu().view(torch.int32), data["out_probs"].view(torch.int32)
    )
    expected = gpu["out_probs"].clone()
    gpu["logits"] = gpu["logits"].contiguous()
    op.probabilities(**gpu, filters=hint)
    assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("hint", SYMB_HINTS)
@pytest.mark.parametrize("use_min_p", [False, True])
def test_symb_strided_probability_exp(hint, use_min_p):
    from ..modules.test_dflash_symb import strided_copy

    data = symb_probability_data(5, 2, 4097, torch.float32, hint, use_min_p)
    # Nontrivial exponents with well separated top-k/top-p boundaries.
    data["logits"][..., :8] = torch.arange(8).float() / 4
    RequestLocalChainSampling("cpu", 2, 4097).probabilities(**data, filters=hint)
    gpu = {name: value.cuda() for name, value in data.items()}
    gpu["logits"] = strided_copy(gpu["logits"], 1)
    op = RequestLocalChainSampling("cuda", 2, 4097)
    op.probabilities(**gpu, filters=hint)
    torch.testing.assert_close(gpu["out_probs"].cpu(), data["out_probs"], atol=2e-6, rtol=2e-5)
    expected = gpu["out_probs"].clone()
    gpu["logits"] = gpu["logits"].contiguous()
    op.probabilities(**gpu, filters=hint)
    assert torch.equal(gpu["out_probs"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "batches,lengths",
    [((0, 1, 5), (1, 4)), pytest.param(SYMB_BATCHES, (1, 9, 64), marks=pytest.mark.exhaustive)],
)
@pytest.mark.parametrize("vocab", [16, pytest.param(4097, marks=pytest.mark.exhaustive)])
def test_symb_probability_capacity_capture(batches, lengths, vocab, monkeypatch):
    from ..modules.test_dflash_symb import strided_copy

    op = RequestLocalChainSampling("cuda", max(lengths), vocab)
    monkeypatch.setattr(op._provider, "compiled", {})
    keys = op.warm_up(batch_capacity=1024, modes=())
    count = len(op._provider.compiled)
    assert count == len(keys)
    for index, b in enumerate(batches):
        for length in (lengths[0] if b > 48 else lengths[index % len(lengths)],):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for hint in SYMB_HINTS:
                    for use in (False, True):
                        cpu = symb_probability_data(b, length, vocab, dtype, hint, use)
                        RequestLocalChainSampling("cpu", max(lengths), vocab).probabilities(
                            **cpu, filters=hint
                        )
                        data = {name: value.cuda() for name, value in cpu.items()}
                        data["logits"] = strided_copy(data["logits"], 1 + b % 4)
                        graph = torch.cuda.CUDAGraph()
                        # No eager call at this shape before capture.
                        with torch.cuda.graph(graph):
                            op.probabilities(**data, filters=hint)
                        graph.replay()
                        assert len(op._provider.compiled) == count
                        assert torch.equal(
                            data["out_probs"].cpu().view(torch.int32),
                            cpu["out_probs"].view(torch.int32),
                        )


@pytest.mark.parametrize(
    "batches",
    [(0, 1, 5), pytest.param(SYMB_BATCHES, marks=pytest.mark.exhaustive)],
)
def test_symb_chain_capacity_capture(batches, monkeypatch):
    op = RequestLocalChainSampling("cuda", 4, 16)
    monkeypatch.setattr(op._provider, "compiled", {})
    op.warm_up(batch_capacity=1024, dtypes=(), filters=())
    count = len(op._provider.compiled)
    for b in batches:
        cpu_op, cpu = buffers(b, 3, 16)
        cpu["temperatures"][1::3] = 0
        cpu["request_indices"][2::4] = -1
        for pool_rows in (0, max(1, b), b + 7):
            data = {name: value.clone() for name, value in cpu.items()}
            data["seeds"] = seed_words(torch.arange(pool_rows, dtype=torch.int64) + 42)
            if not pool_rows:
                data["request_indices"].fill_(-1)
            data["cand_ids"] = torch.arange(16, dtype=torch.int32).expand(b, 3, 16).contiguous()
            data["cand_q"] = data["q"].clone()
            gpu = {name: value.cuda() for name, value in data.items()}
            for mode in ("point", "dense", "sparse"):
                verify(cpu_op, data, sparse=mode == "sparse", point=mode == "point")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    verify(op, gpu, sparse=mode == "sparse", point=mode == "point")
                graph.replay()
                assert torch.equal(gpu["out_tokens"].cpu(), data["out_tokens"])
                assert torch.equal(gpu["accepted_counts"].cpu(), data["accepted_counts"])
            common = {
                name: gpu[name] for name in ("request_indices", "seeds", "cursors", "workspace")
            }
            cpu_common = {name: data[name] for name in common}
            expected = torch.empty(b, 3, dtype=torch.int32)
            cpu_op.sample_dense_proposals(
                data["q"], data["temperatures"], **cpu_common, out_ids=expected
            )
            out = torch.empty_like(expected, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.sample_dense_proposals(gpu["q"], gpu["temperatures"], **common, out_ids=out)
            graph.replay()
            assert torch.equal(out.cpu(), expected)
            scores, ids = torch.zeros(b, 16), data["cand_ids"][:, 0].contiguous()
            gscores, gids = scores.cuda(), ids.cuda()
            expected_id, expected_q = torch.empty(b, dtype=torch.int32), torch.empty(b, 16)
            out_id, out_q = (
                torch.empty_like(expected_id, device="cuda"),
                torch.empty_like(expected_q, device="cuda"),
            )
            for position in (0, 2, torch.tensor(1, dtype=torch.int32)):
                cpu_op.sample_candidates(
                    scores,
                    ids,
                    data["temperatures"],
                    position,
                    **cpu_common,
                    out_ids=expected_id,
                    out_q=expected_q,
                )
                gposition = position.cuda() if isinstance(position, torch.Tensor) else position
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    op.sample_candidates(
                        gscores,
                        gids,
                        gpu["temperatures"],
                        gposition,
                        **common,
                        out_ids=out_id,
                        out_q=out_q,
                    )
                graph.replay()
                assert torch.equal(out_id.cpu(), expected_id)
                torch.testing.assert_close(out_q.cpu(), expected_q, atol=2e-6, rtol=2e-5)
            anchors, prompts = data["cursors"], torch.zeros(pool_rows, dtype=torch.int64)
            expected_cursors, out_cursors = (
                torch.empty(b, dtype=torch.int64),
                torch.empty(b, dtype=torch.int64, device="cuda"),
            )
            cpu_op.prepare_cursors(
                anchors,
                prompts,
                request_indices=data["request_indices"],
                workspace=data["workspace"],
                out_cursors=expected_cursors,
            )
            ganchors, gprompts = anchors.cuda(), prompts.cuda()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.prepare_cursors(
                    ganchors,
                    gprompts,
                    request_indices=gpu["request_indices"],
                    workspace=gpu["workspace"],
                    out_cursors=out_cursors,
                )
            graph.replay()
            assert torch.equal(out_cursors.cpu(), expected_cursors)
            assert len(op._provider.compiled) == count
