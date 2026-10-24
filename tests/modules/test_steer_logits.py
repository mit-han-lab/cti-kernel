# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Sparse steering: the torch provider and the Triton kernel (CPU interpreter) bitwise against a
per-element oracle."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ctikernels.sampling import steer_logits_

NAN = torch.tensor([0x7FC00123], dtype=torch.int32).view(torch.float32).item()  # with a payload
SPECIALS = (-0.0, float("-inf"), float("inf"), NAN, 1e-45)


def steering(n, v, seed, *, max_bias=8, max_stop=4, max_min_tokens=5, int64=False):
    """Random CSR steering for ``n`` requests over ``v`` tokens; stop ids overlap bias ids."""
    g = torch.Generator().manual_seed(seed)
    bias_lists, stop_lists = [], []
    for _ in range(n):
        ids = torch.randperm(v, generator=g)[: int(torch.randint(0, max_bias + 1, (), generator=g))]
        bias_lists.append(ids)
        stops = torch.randint(
            0, v, (int(torch.randint(0, max_stop + 1, (), generator=g)),), generator=g
        )
        if len(ids) and torch.rand((), generator=g) < 0.5:
            stops = torch.cat([stops, ids[:1]])  # a biased stop token
        stop_lists.append(stops)
    index = torch.int64 if int64 else torch.int32

    def csr(lists):
        counts = torch.tensor([len(x) for x in lists], dtype=torch.int64)
        indptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(index)
        return indptr, torch.cat([torch.empty(0, dtype=torch.int64), *lists]).int()

    bias_indptr, bias_ids = csr(bias_lists)
    stop_indptr, stop_ids = csr(stop_lists)
    return {
        "min_tokens": torch.randint(0, max_min_tokens + 1, (n,), generator=g, dtype=index),
        "stop_indptr": stop_indptr,
        "stop_ids": stop_ids,
        "bias_indptr": bias_indptr,
        "bias_ids": bias_ids,
        "bias_values": (torch.randn(len(bias_ids), generator=g) * 30).clamp(-100, 100),
    }


def random_logits(r, v, seed, pad=0):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(r, v + pad, generator=g) * 4)[:, :v]
    specials = torch.tensor(SPECIALS)
    spots = torch.rand(r, v, generator=g) < 0.1
    logits[spots] = specials[torch.randint(0, len(SPECIALS), (int(spots.sum()),), generator=g)]
    return logits


def oracle(logits, rows, generated, s):
    """Element by element, in vLLM's order: FP32 add each bias, then -inf each stop token."""
    out = logits.clone()
    for r, n in enumerate(rows.tolist()):
        for k in range(int(s["bias_indptr"][n]), int(s["bias_indptr"][n + 1])):
            t = int(s["bias_ids"][k])
            out[r, t] = out[r, t] + s["bias_values"][k]
        if int(generated[r]) < int(s["min_tokens"][n]):
            for k in range(int(s["stop_indptr"][n]), int(s["stop_indptr"][n + 1])):
                out[r, int(s["stop_ids"][k])] = float("-inf")
    return out


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


def _cases(seed):
    """(logits, rows, generated, steering) over a few shapes; rows repeat requests."""
    for index, (r, n, v, pad, int64) in enumerate(
        [
            (1, 1, 64, 0, False),
            (7, 3, 300, 5, False),
            (16, 16, 1000, 0, True),
            (12, 2, 129, 3, True),
        ]
    ):
        g = torch.Generator().manual_seed(seed + index)
        s = steering(n, v, seed + index, int64=int64)
        rows = torch.randint(0, n, (r,), generator=g, dtype=torch.int32)
        generated = torch.randint(
            0, 7, (r,), generator=g, dtype=torch.int64 if int64 else torch.int32
        )
        yield random_logits(r, v, seed + index, pad), rows, generated, s


def _check(steer):
    for seed in range(6):
        for logits, rows, generated, s in _cases(100 * seed):
            before = logits.clone()
            steer(logits, rows, generated, s)
            assert_same(logits, oracle(before, rows, generated, s), before, rows, generated, s)


def _public(logits, rows, generated, s):
    steer_logits_(logits, rows, generated, **s)


def test_torch_provider_matches_oracle():
    _check(_public)


def _exercise():
    from ctikernels.sampling.triton._steer import launch

    def interpreted(logits, rows, generated, s):
        launch(logits, rows, generated, *(s[k] for k in s))

    _check(interpreted)
    # Larger lists than one block: 300 biases and 200 stops on one request.
    g = torch.Generator().manual_seed(1)
    s = {
        "min_tokens": torch.tensor([1]),
        "stop_indptr": torch.tensor([0, 200]),
        "stop_ids": torch.randint(0, 4096, (200,), generator=g, dtype=torch.int32),
        "bias_indptr": torch.tensor([0, 300]),
        "bias_ids": torch.randperm(4096, generator=g)[:300].int(),
        "bias_values": torch.randn(300, generator=g),
    }
    logits, rows, generated = (
        random_logits(2, 4096, 3),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([0, 1]),
    )
    before = logits.clone()
    launch(logits, rows, generated, *(s[k] for k in s))
    assert_same(logits, oracle(before, rows, generated, s), before, rows, generated, s)


def test_triton_kernel_on_cpu_interpreter():
    pytest.importorskip("triton")
    command = f"import importlib; importlib.import_module({__spec__.name!r})._exercise()"
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mask_wins_and_min_tokens_crosses_within_a_request():
    # One request, three rows at generated 1, 2, 3 with min_tokens 3: rows 0-1 are masked.
    s = {
        "min_tokens": torch.tensor([3]),
        "stop_indptr": torch.tensor([0, 2], dtype=torch.int32),
        "stop_ids": torch.tensor([5, 2], dtype=torch.int32),
        "bias_indptr": torch.tensor([0, 2], dtype=torch.int32),
        "bias_ids": torch.tensor([5, 1], dtype=torch.int32),
        "bias_values": torch.tensor([100.0, -0.5]),
    }
    logits = torch.zeros(3, 8)
    logits[:, 0] = -0.0
    steer_logits_(logits, torch.zeros(3, dtype=torch.int32), torch.tensor([1, 2, 3]), **s)
    inf = float("inf")
    expected = torch.tensor(
        [[-0.0, -0.5, -inf, 0, 0, -inf, 0, 0]] * 2 + [[-0.0, -0.5, 0, 0, 0, 100, 0, 0]]
    )
    assert torch.equal(logits.view(torch.int32), expected.view(torch.int32))


def test_nan_payload_biased_and_untouched():
    """A biased NaN stays a NaN (its payload is the provider's: the device canonicalises it), an
    unwritten NaN keeps its payload, a masked one becomes -inf; assert_same holds exactly that."""
    payload = torch.tensor([0x7FC00123], dtype=torch.int32).view(torch.float32)
    canonical = torch.tensor([0x7FFFFFFF], dtype=torch.int32).view(torch.float32)
    s = {
        "min_tokens": torch.tensor([1]),
        "stop_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "stop_ids": torch.tensor([3], dtype=torch.int32),
        "bias_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "bias_ids": torch.tensor([1], dtype=torch.int32),
        "bias_values": torch.tensor([2.5]),
    }
    rows, generated = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    before = payload.expand(1, 4).clone()
    logits = before.clone()
    steer_logits_(logits, rows, generated, **s)
    bits = logits.view(torch.int32)[0]
    assert logits[0, 1].isnan() and logits[0, 3] == float("-inf")
    assert bits[0] == bits[2] == 0x7FC00123
    expected = oracle(before, rows, generated, s)
    device = logits.clone()
    device[0, 1] = canonical  # what CUDA returns for NaN + bias
    assert_same(device, expected, before, rows, generated, s)
    for column, value in ((0, canonical), (1, torch.tensor([1.0])), (3, payload)):
        wrong = logits.clone()
        wrong[0, column] = value
        with pytest.raises(AssertionError):
            assert_same(wrong, expected, before, rows, generated, s)


def test_empty_lists_and_no_rows():
    s = steering(4, 32, 0, max_bias=0, max_stop=0)
    logits = random_logits(4, 32, 0)
    before = logits.clone()
    steer_logits_(
        logits, torch.arange(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32), **s
    )
    assert torch.equal(logits.view(torch.int32), before.view(torch.int32))
    empty = torch.empty(0, 32)
    steer_logits_(empty, torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32), **s)


def test_view_of_rows_and_positions():
    # A [N, Q, V] slice viewed as [N * Q, V] rows of a wider buffer.
    s = steering(2, 50, 3)
    view = random_logits(2 * 3, 64, 3).view(2, 3, 64)[:, :, :50].flatten(0, 1)
    assert view.stride() == (64, 1)
    before = view.clone()
    rows = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    generated = torch.tensor([0, 1, 2, 4, 5, 6])
    steer_logits_(view, rows, generated, **s)
    assert_same(view, oracle(before, rows, generated, s), before, rows, generated, s)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda a: a.update(logits=a["logits"].double()), "FP32"),
        (lambda a: a.update(logits=a["logits"].t()), "unit column stride"),
        (lambda a: a.update(logits=a["logits"][:, :1].expand(4, 8)), "unit column stride"),
        (lambda a: a.update(logits=torch.zeros(16).as_strided((4, 8), (2, 1))), "overlap"),
        (lambda a: a.update(rows=a["rows"].long()), "rows"),
        (lambda a: a.update(generated=a["generated"][:3]), "generated"),
        (lambda a: a.update(stop_indptr=a["stop_indptr"][:2]), "stop_indptr"),
        (lambda a: a.update(bias_values=a["bias_values"][:0]), "bias_values"),
        (lambda a: a.update(bias_ids=a["bias_ids"].long()), "bias_ids"),
        (lambda a: a.update(stop_ids=torch.zeros(4, dtype=torch.int32)[::2]), "contiguous"),
    ],
)
def test_validation(change, message):
    s = steering(2, 8, 1, max_bias=3)
    s["bias_indptr"] = torch.tensor([0, 1, 1], dtype=torch.int32)
    s["bias_ids"], s["bias_values"] = torch.tensor([3], dtype=torch.int32), torch.tensor([1.0])
    args = {"logits": torch.zeros(4, 8), "rows": torch.zeros(4, dtype=torch.int32),
            "generated": torch.zeros(4, dtype=torch.int32), **s}  # fmt: skip
    change(args)
    with pytest.raises(ValueError, match=message):
        steer_logits_(args.pop("logits"), args.pop("rows"), args.pop("generated"), **args)
