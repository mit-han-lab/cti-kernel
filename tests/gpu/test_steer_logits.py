# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Sparse steering on the device: bitwise against the CPU torch provider, equal to the engine's
dense form except for the sign of untouched -0.0, and graph replay equal to eager."""

import pytest
import torch

from ctikernels.sampling import steer_logits_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

V = 248320  # Qwen3.5
NAN = torch.tensor([0x7FC00123], dtype=torch.int32).view(torch.float32).item()
SPECIALS = (-0.0, float("-inf"), float("inf"), NAN, 1e-45, -1e-40)


def steering(n, seed, *, max_bias=1024, max_stop=128):
    """Random CSR steering; some stop ids are also biased, some lists are empty."""
    g = torch.Generator().manual_seed(seed)
    bias, stops = [], []
    for _ in range(n):
        ids = torch.randperm(V, generator=g)[: int(torch.randint(0, max_bias + 1, (), generator=g))]
        bias.append(ids)
        stop = torch.randint(
            0, V, (int(torch.randint(0, max_stop + 1, (), generator=g)),), generator=g
        )
        stops.append(torch.cat([stop, ids[:3]]))

    def csr(lists):
        counts = torch.tensor([len(x) for x in lists])
        indptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        return indptr.int(), torch.cat(lists).int()

    bias_indptr, bias_ids = csr(bias)
    stop_indptr, stop_ids = csr(stops)
    return {
        "min_tokens": torch.randint(0, 6, (n,), generator=g),
        "stop_indptr": stop_indptr,
        "stop_ids": stop_ids,
        "bias_indptr": bias_indptr,
        "bias_ids": bias_ids,
        "bias_values": (torch.randn(len(bias_ids), generator=g) * 30).clamp(-100, 100),
    }


def random_logits(r, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(r, V, generator=g) * 4
    spots = torch.rand(r, V, generator=g) < 0.05
    specials = torch.tensor(SPECIALS)
    logits[spots] = specials[torch.randint(0, len(SPECIALS), (int(spots.sum()),), generator=g)]
    return logits


def write_set(shape, rows, generated, s):
    """The elements the op writes, from its inputs: (biased, stop-masked) bool [R, V] on the CPU."""
    biased = torch.zeros(shape, dtype=torch.bool)
    masked = torch.zeros(shape, dtype=torch.bool)
    for r, n in enumerate(rows.tolist()):
        lo, hi = int(s["bias_indptr"][n]), int(s["bias_indptr"][n + 1])
        biased[r, s["bias_ids"][lo:hi].long().cpu()] = True
        if int(generated[r]) < int(s["min_tokens"][n]):
            lo, hi = int(s["stop_indptr"][n]), int(s["stop_indptr"][n + 1])
            masked[r, s["stop_ids"][lo:hi].long().cpu()] = True
    return biased, masked


def assert_same(actual, expected, before, rows, generated, s):
    """Against the op's write set: unwritten elements keep their input bits, NaN payloads
    included; stop-masked ones are -inf; biased ones are bitwise, except that a NaN produced by
    an add only has to be a NaN (the device returns its canonical NaN)."""
    actual, expected, before = (x.cpu().contiguous() for x in (actual, expected, before))
    a, e, b = (x.view(torch.int32) for x in (actual, expected, before))
    biased, masked = write_set(actual.shape, rows, generated, s)
    untouched = ~(biased | masked)
    assert torch.equal(a[untouched], b[untouched])
    inf = torch.tensor(float("-inf")).view(torch.int32)
    assert bool((a[masked] == inf).all()) and bool((e[masked] == inf).all())
    added = biased & ~masked
    nan = added & expected.isnan()
    assert torch.equal(actual[added].isnan(), nan[added])
    assert torch.equal(a[added & ~nan], e[added & ~nan])


def case(r, seed):
    n = max(1, r // 2)
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, n, (r,), generator=g, dtype=torch.int32)
    return random_logits(r, seed), rows, torch.randint(0, 7, (r,), generator=g), steering(n, seed)


def on(device, s):
    return {k: v.to(device) for k, v in s.items()}


@pytest.mark.parametrize("r", [1, 16, 64])
def test_bitwise_against_torch_provider(r):
    for seed in range(3):
        logits, rows, generated, s = case(r, 10 * r + seed)
        expected = logits.clone()
        steer_logits_(expected, rows, generated, **s)
        actual = logits.cuda()
        steer_logits_(actual, rows.cuda(), generated.cuda(), **on("cuda", s))
        assert_same(actual, expected, logits, rows, generated, s)


def test_empty_lists_and_no_rows():
    """Empty id lists reach the kernel as zero-size (null) tensors; R = 0 launches nothing."""
    logits, rows, generated, _ = case(16, 3)
    for max_bias, max_stop in ((0, 0), (0, 8)):
        s = steering(8, 3, max_bias=max_bias, max_stop=max_stop)
        expected = logits.clone()
        steer_logits_(expected, rows, generated, **s)
        actual = logits.cuda()
        steer_logits_(actual, rows.cuda(), generated.cuda(), **on("cuda", s))
        assert_same(actual, expected, logits, rows, generated, s)
    empty = torch.empty(0, V, device="cuda")
    none = torch.empty(0, dtype=torch.int32, device="cuda")
    steer_logits_(empty, none, none, **on("cuda", s))


@pytest.mark.parametrize("r", [1, 16, 64])
def test_matches_engine_dense_form(r):
    """The engine's dense form adds +0.0 to every untouched element: -0.0 becomes +0.0 there
    (and an untouched NaN becomes the device's canonical NaN); everything else is equal."""
    logits, rows, generated, s = case(r, 1000 + r)
    logits, rows, generated, s = logits.cuda(), rows.cuda(), generated.cuda(), on("cuda", s)
    bias = torch.zeros(s["min_tokens"].shape[0], V, device="cuda")
    stops = torch.zeros(bias.shape, dtype=torch.bool, device="cuda")
    for n in range(bias.shape[0]):
        lo, hi = s["bias_indptr"][n : n + 2].tolist()
        bias[n, s["bias_ids"][lo:hi].long()] = s["bias_values"][lo:hi]
        lo, hi = s["stop_indptr"][n : n + 2].tolist()
        stops[n, s["stop_ids"][lo:hi].long()] = True
    masked = stops[rows.long()] & (generated < s["min_tokens"][rows.long()])[:, None]
    dense = (logits.float() + bias[rows.long()]).masked_fill(masked, float("-inf"))
    sparse = logits.clone()
    steer_logits_(sparse, rows, generated, **s)
    assert torch.equal(sparse.isnan(), dense.isnan())
    finite = ~dense.isnan()
    assert torch.equal(sparse[finite], dense[finite])  # -0.0 == +0.0
    signs = sparse.view(torch.int32) != dense.view(torch.int32)
    signs &= finite
    assert torch.equal(signs, finite & (logits.view(torch.int32) == -(2**31)) & (dense == 0))


def test_graph_replay_equals_eager():
    logits, rows, generated, s = case(16, 5)
    logits, rows, generated, s = logits.cuda(), rows.cuda(), generated.cuda(), on("cuda", s)
    static = logits.clone()
    steer_logits_(static, rows, generated, **s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        steer_logits_(static, rows, generated, **s)
    fresh = case(16, 6)
    static.copy_(fresh[0])
    generated.copy_(fresh[2])
    rows.copy_(fresh[1])
    graph.replay()
    eager = fresh[0].cuda()
    steer_logits_(eager, rows, generated, **s)
    torch.cuda.synchronize()
    assert torch.equal(static.view(torch.int32), eager.view(torch.int32))
