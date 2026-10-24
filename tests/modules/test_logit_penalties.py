# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU DN4 qualification and composition with S1/K1/DN3."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.sampling import (
    LogitPenalties,
    RequestLocalChainSampling,
    RequestLocalSampling,
    TopLogprobs,
)
from ctikernels.sampling._penalties_reference import (
    ERROR_COEFFICIENT,
    UNIT_ROUNDOFF,
    reference_penalties,
)


def penalty_rows(n=6):
    return SimpleNamespace(
        rows=torch.arange(n - 1, -1, -1, dtype=torch.int32),
        prompt_lengths=torch.tensor(([0, 0, 7, 13, 19, 4] * n)[:n], dtype=torch.int64),
        lengths=torch.tensor(([0, 13, 7, 31, 31, 31] * n)[:n], dtype=torch.int32),
        presence=torch.tensor(([0, -2, 2, 0.5, -2, 2] * n)[:n], dtype=torch.float32),
        frequency=torch.tensor(([0, 2, -2, 0.25, 2, -2] * n)[:n], dtype=torch.float32),
        repetition=torch.tensor(([1, 1e-30, 1e30, 2, 1, 0.3] * n)[:n], dtype=torch.float32),
    )


def assert_bits(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)
    )


def test_engine_example():
    table = torch.tensor([[1, 2, 2, 3, 0, 0], [3, 0, 0, 0, 0, 0]], dtype=torch.int32)
    rows = SimpleNamespace(
        rows=torch.tensor([0, 1]),
        prompt_lengths=torch.tensor([2, 1]),
        lengths=torch.tensor([4, 1]),
        presence=torch.tensor([0.5, 0]),
        frequency=torch.tensor([0.25, 0]),
        repetition=torch.tensor([2.0, 2]),
    )
    logits = torch.tensor([[1, 2, -2, 4, 1], [1, 1, 1, -1, 1]], dtype=torch.float32)
    out = torch.empty_like(logits)
    op = LogitPenalties(table, 5)
    assert op.provider == "torch"
    assert op.apply(logits, rows, out=out) is None
    assert_bits(out, torch.tensor([[1, 1, -4.75, 1.25, 1], [1, 1, 1, -2, 1]]))


@pytest.mark.parametrize(
    "dtype,layout",
    [
        (dtype, layout)
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
        for layout in ("dense", "strided", "inplace")
        if layout != "inplace" or dtype == torch.float32
    ],
)
@pytest.mark.parametrize("q", [1, 2, 16])
def test_randomized_bitwise(dtype, q, layout):
    torch.manual_seed(1701 + q)
    n, v = 6, 37
    table = torch.randint(-3, v + 3, (n, 47), dtype=torch.int64)
    table[:, :4] = torch.tensor([-1, v, 0, v - 1])
    rows = penalty_rows()
    shape = (n, q, v + 9 if layout == "strided" else v)
    logits = torch.randn(shape).to(dtype)[..., :v]
    logits[..., :3] = torch.tensor([-torch.inf, -0.0, 0.0], dtype=dtype)
    drafts = torch.randint(-2, v + 2, (n, q - 1), dtype=torch.int32)
    if q == 1:
        logits = logits[:, 0]
    expected = reference_penalties(logits, table, rows, drafts)
    out = logits if layout == "inplace" else torch.empty((*logits.shape[:-1], v + 11))[..., :v]
    original = logits.float().clone()
    LogitPenalties(table, v).apply(logits, rows, drafts, out=out)
    assert_bits(out, expected)
    assert_bits(out[0], original[0])


def test_drafts_equal_extending_context():
    v, q = 11, 16
    table = torch.tensor([[1, 2, 2, 3] + [0] * 20], dtype=torch.int32)
    rows = SimpleNamespace(
        rows=torch.tensor([0]),
        prompt_lengths=torch.tensor([2]),
        lengths=torch.tensor([4]),
        presence=torch.tensor([0.5]),
        frequency=torch.tensor([-2.0]),
        repetition=torch.tensor([1.2]),
    )
    drafts = torch.tensor([[2, 2, -1, 11, 3, 7, 3, 0, 1, 2, 3, 4, 5, 6, 7]])
    logits = torch.linspace(-5, 5, v).expand(1, q, v).clone()
    actual = torch.empty_like(logits)
    LogitPenalties(table, v).apply(logits, rows, drafts, out=actual)
    for j in range(q):
        extended = table.clone()
        extended[0, 4 : 4 + j] = drafts[0, :j]
        rows.lengths.fill_(4 + j)
        expected = torch.empty((1, v))
        LogitPenalties(extended, v).apply(logits[:, j], rows, out=expected)
        assert_bits(actual[:, j], expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_identity_extremes(dtype):
    limit = torch.finfo(dtype).max
    x = torch.tensor([[-torch.inf, -0.0, 0.0, limit, -limit, 1.0]], dtype=dtype)
    rows = penalty_rows(1)
    rows.presence.fill_(-0.0)
    rows.frequency.fill_(-0.0)
    table = torch.zeros((1, 0), dtype=torch.int32)
    out = torch.empty_like(x, dtype=torch.float32)
    LogitPenalties(table, 6).apply(x, rows, out=out)
    assert_bits(out, x.float())


def test_tiny_budget_chunks_requests_columns_and_drafts(monkeypatch):
    v, n, q = 7, 6, 16
    table = torch.arange(n * 67, dtype=torch.int32).reshape(n, 67) % (v + 2) - 1
    rows = penalty_rows()
    x = torch.randn((n, q, v))
    drafts = torch.arange(n * (q - 1)).reshape(n, q - 1) % (v + 2) - 1
    expected = reference_penalties(x, table, rows, drafts)
    op = LogitPenalties(table, v, element_budget=3 * v + 56)
    assert op.row_chunk == 1 and op.column_chunk == 2
    original = torch.Tensor.index_select
    gathered = []

    def checked(src, dim, index):
        gathered.append((src.shape[1], index.numel()))
        assert src.shape[1] <= 2 and index.numel() == 1
        return original(src, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", checked)
    out = torch.empty_like(x)
    op.apply(x, rows, drafts, out=out)
    assert len(gathered) == n * 34
    assert_bits(out, expected)


def test_nonstandard_leading_strides_and_empty_batch():
    table = torch.zeros((6, 31), dtype=torch.int32)
    rows = penalty_rows()
    x = torch.randn((3, 6, 9)).transpose(0, 1)
    out = torch.empty((3, 6, 12)).transpose(0, 1)[..., :9]
    drafts = torch.zeros((6, 2), dtype=torch.int64)
    LogitPenalties(table, 9).apply(x, rows, drafts, out=out)
    assert_bits(out, reference_penalties(x, table, rows, drafts))
    empty = SimpleNamespace(**{name: tensor[:0] for name, tensor in vars(rows).items()})
    LogitPenalties(table, 9).apply(x[:0], empty, drafts[:0], out=out[:0])


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "inner_stride",
        "out_dtype",
        "out_shape",
        "overlap",
        "bf16_inplace",
        "missing",
        "lookahead_shape",
        "lookahead_dtype",
        "grad",
        "coefficient_grad",
        "coefficient_dtype",
        "row_shape",
        "out_stride",
        "out_overlap",
        "device",
    ],
)
def test_validation(fault):
    v = 7
    table = torch.zeros((1, 31), dtype=torch.int32)
    rows = penalty_rows(1)
    x, out, drafts = (
        torch.zeros((1, 2, v)),
        torch.empty((1, 2, v)),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    if fault == "dtype":
        x = x.double()
    elif fault == "inner_stride":
        x = torch.zeros((1, 2, 2 * v))[..., ::2]
    elif fault == "out_dtype":
        out = out.half()
    elif fault == "out_shape":
        out = out[:, :1]
    elif fault == "overlap":
        buffer = torch.zeros((1, 2, v + 1))
        x, out = buffer[..., :v], buffer[..., 1:]
    elif fault == "bf16_inplace":
        x = x.bfloat16()
        out = x
    elif fault == "missing":
        drafts = None
    elif fault == "lookahead_shape":
        drafts = torch.zeros((1, 2), dtype=torch.int32)
    elif fault == "lookahead_dtype":
        drafts = drafts.float()
    elif fault == "grad":
        x.requires_grad_()
    elif fault == "coefficient_grad":
        rows.presence.requires_grad_()
    elif fault == "coefficient_dtype":
        rows.presence = rows.presence.double()
    elif fault == "row_shape":
        rows.rows = rows.rows[:, None]
    elif fault == "out_stride":
        out = torch.empty((1, 2, v * 2))[..., ::2]
    elif fault == "out_overlap":
        out = torch.empty(v + 1).as_strided((1, 2, v), (1, 1, 1))
    elif fault == "device":
        out = torch.empty_like(out, device="meta")
    with pytest.raises((TypeError, ValueError)):
        LogitPenalties(table, v).apply(x, rows, drafts, out=out)


@pytest.mark.parametrize("v,budget", [(0, 100), (True, 100), (2**31, 100), (7, 1), (7, True)])
def test_constructor_validation(v, budget):
    with pytest.raises(ValueError):
        LogitPenalties(torch.zeros((1, 2), dtype=torch.int32), v, element_budget=budget)


def test_apply_never_reads_device_values_on_host(monkeypatch):
    n, v, q = 6, 17, 3
    rows = penalty_rows()
    table = torch.zeros((n, 33), dtype=torch.int32)
    x, out = torch.zeros((n, q, v)), torch.empty((n, q, v))
    drafts = torch.zeros((n, q - 1), dtype=torch.int64)

    def forbidden(*args, **kwargs):
        raise AssertionError("device value readback")

    original = torch.Tensor.__getitem__

    def checked(tensor, key):
        keys = key if isinstance(key, tuple) else (key,)
        assert not any(isinstance(k, torch.Tensor) and k.dtype == torch.bool for k in keys)
        return original(tensor, key)

    for name in (
        "item",
        "tolist",
        "cpu",
        "numpy",
        "nonzero",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
    ):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "__getitem__", checked)
    LogitPenalties(table, v, element_budget=3 * v + 68).apply(x, rows, drafts, out=out)


def test_fp64_bound():
    torch.manual_seed(42)
    n, v, q = 6, 127, 16
    rows = penalty_rows()
    rows.repetition[:] = torch.tensor([1, 1e-30, 1e30, 1.2, 0.7, 3.0])
    table = torch.randint(-1, v + 1, (n, 257), dtype=torch.int32)
    rows.lengths[3:] = 257
    x = torch.randn((n, q, v)) * 4096
    drafts = torch.randint(-1, v + 1, (n, q - 1))
    out = torch.empty_like(x)
    LogitPenalties(table, v).apply(x, rows, drafts, out=out)
    expected, magnitude = reference_penalties(x, table, rows, drafts, fp64=True)
    error = (out.double() - expected).abs()
    assert torch.isfinite(out).all()
    assert (error <= ERROR_COEFFICIENT * UNIT_ROUNDOFF * magnitude).all()


@pytest.mark.parametrize("q", [1, 2, 16])
def test_sampling_and_processed_logprobs_composition(q):
    torch.manual_seed(17)
    n, v = 2, 37
    rows = penalty_rows(n)
    rows.repetition.fill_(1.2)
    rows.prompt_lengths[:] = torch.tensor([2, 0])
    rows.lengths[:] = torch.tensor([17, 31])
    table = torch.randint(0, v, (n, 31), dtype=torch.int32)
    drafts = torch.randint(0, v, (n, q - 1))
    logits = torch.randn((n, q, v)).bfloat16()
    actual = torch.empty_like(logits, dtype=torch.float32)
    LogitPenalties(table, v).apply(logits, rows, drafts, out=actual)
    expected = reference_penalties(logits, table, rows, drafts)
    assert_bits(actual, expected)
    temperatures, k, p = (
        torch.tensor([0.7, 1.2]),
        torch.tensor([19, v], dtype=torch.int32),
        torch.tensor([0.8, 1.0]),
    )
    slots = torch.arange(n, dtype=torch.int32)
    chain = RequestLocalChainSampling("cpu", q, v)
    workspace = torch.empty(chain.workspace_size(n, "probabilities"), dtype=torch.uint8)
    probs, ref_probs = torch.empty_like(actual), torch.empty_like(expected)
    for source, dest in ((actual, probs), (expected, ref_probs)):
        chain.probabilities(
            source, temperatures, k, p, request_indices=slots, workspace=workspace, out_probs=dest
        )
    assert_bits(probs, ref_probs)
    if q == 1:
        sampler = RequestLocalSampling("cpu", v)
        work = torch.empty(sampler.workspace_size(n), dtype=torch.uint8)
        tokens, ref_tokens = torch.empty(n, dtype=torch.int32), torch.empty(n, dtype=torch.int32)
        for source, dest in ((actual[:, 0], tokens), (expected[:, 0], ref_tokens)):
            sampler.sample(
                source,
                temperatures,
                k,
                p,
                request_indices=slots,
                seeds=torch.tensor([[7, 13], [29, 31]], dtype=torch.uint32),
                positions=torch.tensor([0, 17]),
                workspace=work,
                out_tokens=dest,
            )
        assert torch.equal(tokens, ref_tokens)
    r = n * q
    top = TopLogprobs("cpu", v, 5)
    results = []
    for supplied in (None, probs.reshape(r, v)):
        args = {
            "row_indices": torch.arange(r, dtype=torch.int32),
            "num_rows": torch.tensor([r], dtype=torch.int32),
            "selected_ids": probs.reshape(r, v).argmax(-1).int(),
            "top_k": torch.full((r,), 5, dtype=torch.int32),
            "mode": "processed",
            "temperatures": temperatures.repeat_interleave(q),
            "filter_top_k": k.repeat_interleave(q),
            "top_p": p.repeat_interleave(q),
            "probabilities": supplied,
            "out_selected_logprobs": torch.empty(r),
            "out_ranks": torch.empty(r, dtype=torch.int32),
            "out_top_ids": torch.empty((r, 5), dtype=torch.int32),
            "out_top_logprobs": torch.empty((r, 5)),
        }
        top.run(actual.reshape(r, v), **args)
        results.append(args)
    for name in ("out_ranks", "out_top_ids"):
        assert torch.equal(results[0][name], results[1][name])
    for name in ("out_selected_logprobs", "out_top_logprobs"):
        a, b = results[0][name], results[1][name]
        assert torch.equal(torch.isneginf(a), torch.isneginf(b))
        finite = torch.isfinite(a)
        assert ((a[finite] - b[finite]).abs() <= 8e-6 + 5e-7 * b[finite].abs()).all()


def test_constructor_table_metadata_and_empty_q():
    for table in (None, torch.zeros(2, dtype=torch.int32), torch.zeros((1, 2))):
        with pytest.raises((TypeError, ValueError)):
            LogitPenalties(table, 7)
    with pytest.raises(ValueError, match="Q must"):
        LogitPenalties(torch.zeros((1, 31), dtype=torch.int32), 7).apply(
            torch.empty((1, 0, 7)), penalty_rows(1), out=torch.empty((1, 0, 7))
        )


def test_exact_alias_view_and_other_input_alias_rejection():
    x = torch.randn((6, 17))
    table = torch.zeros((6, 31), dtype=torch.int32)
    rows = penalty_rows()
    expected = reference_penalties(x, table, rows)
    op = LogitPenalties(table, 17)
    op.apply(x, rows, out=x.view_as(x))
    assert_bits(x, expected)
    # A coefficient backed by output storage must be rejected before writing.
    rows.presence = x[:, 0]
    with pytest.raises(ValueError, match="overlap"):
        op.apply(x.clone(), rows, out=x)


def test_wide_table_scan_and_ignored_tail():
    v = 19
    table = torch.full((1, 262145), -1, dtype=torch.int32)
    table[0, :5] = torch.tensor([0, 1, 1, 18, 19])
    table[0, 5:] = 2  # Must not count this out-of-context tail.
    rows = penalty_rows(1)
    rows.prompt_lengths.fill_(1)
    rows.lengths.fill_(5)
    rows.presence.fill_(0.5)
    rows.frequency.fill_(0.25)
    rows.repetition.fill_(2)
    logits = torch.linspace(-2, 2, v)[None, :]
    out = torch.empty_like(logits)
    op = LogitPenalties(table, v, element_budget=2**20)
    op.apply(logits, rows, out=out)
    assert_bits(out, reference_penalties(logits, table, rows))
    assert out[0, 2] == logits[0, 2]


def test_reciprocal_rounding_and_overflow_are_part_of_contract():
    rows = penalty_rows(1)
    rows.prompt_lengths.fill_(3)
    rows.lengths.fill_(3)
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    op = LogitPenalties(table, 3)
    x = torch.tensor([[5.0, 0.3, -0.0]])
    rows.repetition.fill_(3.0)
    out = torch.empty_like(x)
    op.apply(x, rows, out=out)
    assert_bits(out, reference_penalties(x, table, rows))
    assert out[0, 0] != (x / rows.repetition[:, None])[0, 0]
    # Reciprocal overflow is allowed even though true division is finite.
    rows.repetition.fill_(2.0**-149)
    x.fill_(2.0**-149)
    op.apply(x, rows, out=out)
    assert torch.isposinf(out).all()
    fp64, _ = reference_penalties(x, table, rows, fp64=True)
    assert torch.equal(fp64, torch.ones_like(fp64))


def test_invalid_int64_ids_never_count_at_vocabulary_edges():
    table = torch.tensor([[-(2**63), 2**63 - 1, -1, 5]], dtype=torch.int64)
    rows = penalty_rows(1)
    rows.prompt_lengths.fill_(2)
    rows.lengths.fill_(4)
    rows.presence.fill_(2)
    rows.frequency.fill_(2)
    rows.repetition.fill_(2)
    x = torch.ones((1, 2, 5))
    out = torch.empty_like(x)
    LogitPenalties(table, 5).apply(x, rows, torch.tensor([[2**63 - 1]]), out=out)
    assert_bits(out, x)


def test_interleaved_positive_leading_strides():
    table = torch.zeros((3, 31), dtype=torch.int32)
    rows = penalty_rows(3)
    x = torch.randn((3, 2, 2))
    out = torch.empty(16).as_strided((3, 2, 2), (4, 6, 1))
    drafts = torch.zeros((3, 1), dtype=torch.int32)
    LogitPenalties(table, 2).apply(x, rows, drafts, out=out)
    assert_bits(out, reference_penalties(x, table, rows, drafts))


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("all_equal", [False, True])
@pytest.mark.parametrize("budget", [2**24, 4096])
def test_q64_duplicate_and_invalid_draft_fixups(inplace, all_equal, budget):
    torch.manual_seed(64)
    n, q, v = 6, 64, 137
    table = torch.randint(-2, v + 2, (n, 47), dtype=torch.int64)
    rows = penalty_rows()
    rows.presence[0] = rows.frequency[0] = -0.0
    x = torch.randn((n, q, v)).to(torch.float32 if inplace else torch.bfloat16)
    x[..., :3] = torch.tensor([-0.0, 0.0, -torch.inf])
    drafts = torch.full((n, q - 1), 0, dtype=torch.int64)
    if not all_equal:
        pattern = torch.tensor([0, 0, -1, v, v - 1, v - 1, 2, 2, -(2**63), 2**63 - 1])
        drafts[:] = pattern.repeat(7)[: q - 1]
    expected = reference_penalties(x, table, rows, drafts)
    out = x if inplace else torch.empty_like(x, dtype=torch.float32)
    LogitPenalties(table, v, element_budget=budget).apply(x, rows, drafts, out=out)
    assert_bits(out, expected)


@pytest.mark.parametrize("v", [1, 7, 17])
def test_minimum_budget_inplace_draft_fixups(v):
    q = 16
    rows = penalty_rows(1)
    rows.presence.fill_(0.5)
    rows.frequency.fill_(-0.25)
    rows.repetition.fill_(3.0)
    table = torch.zeros((1, 0), dtype=torch.int32)
    x = torch.randn((1, q, v))
    drafts = torch.arange(q - 1)[None] % (v + 2) - 1
    expected = reference_penalties(x, table, rows, drafts)
    LogitPenalties(table, v, element_budget=3 * v + 56).apply(x, rows, drafts, out=x)
    assert_bits(x, expected)
    with pytest.raises(ValueError, match=r"3\*vocab_size\+56"):
        LogitPenalties(table, v, element_budget=3 * v + 55)
