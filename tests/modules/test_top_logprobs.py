# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contract and independent FP64/S1 boundary checks for DN3."""

import pytest
import torch

from ctikernels.sampling import TopLogprobs
from ctikernels.sampling._dflash.reference.ops import probability_row
from ctikernels.sampling._logprobs_reference import ERROR_ATOL, ERROR_RTOL, reference_row


def arguments(n, capacity, device="cpu"):
    def full(value, dtype=torch.float32):
        return torch.full((n,), value, dtype=dtype, device=device)

    return {
        "row_indices": torch.arange(n, dtype=torch.int32, device=device),
        "num_rows": torch.tensor([n], dtype=torch.int32, device=device),
        "selected_ids": full(0, torch.int32),
        "top_k": full(capacity, torch.int32),
        "out_selected_logprobs": full(float("nan")),
        "out_ranks": full(-77, torch.int32),
        "out_top_ids": torch.full((n, capacity), -77, dtype=torch.int32, device=device),
        "out_top_logprobs": torch.full((n, capacity), float("nan"), device=device),
    }


def processing(n, vocab):
    return {
        "temperatures": torch.ones(n),
        "filter_top_k": torch.full((n,), vocab, dtype=torch.int32),
        "top_p": torch.ones(n),
    }


def assert_precision(actual, expected):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    error = (actual[finite] - expected[finite]).abs()
    assert torch.all(error <= ERROR_ATOL + ERROR_RTOL * expected[finite].abs()), error.max()
    assert not torch.isnan(actual).any()
    return float(error.max()) if error.numel() else 0.0


def test_padding_stride_chunks_and_rank_ties():
    backing = torch.full((8, 1207), -torch.inf)
    x = backing[::2, :1103]
    x[:, :33] = 4
    x[:, 3] = 5
    op = TopLogprobs("cpu", 1103, 20, row_chunk=2)
    args = arguments(5, 20)
    args["row_indices"][:] = torch.tensor([3, 0, 999, 1, -999])
    args["selected_ids"][:] = torch.tensor([2, 3, -1, 1000, 999999])
    args["top_k"][:] = torch.tensor([5, 0, 20, 20, 20])
    args["num_rows"].fill_(4)
    assert op.run(x, **args) is None
    assert args["out_ranks"].tolist() == [33, 1, 0, 1103, 0]
    assert args["out_top_ids"][0, :5].tolist() == [3, 0, 1, 2, 4]
    for i in (2, 4):
        assert args["out_selected_logprobs"][i] == 0
        assert (args["out_top_ids"][i] == -1).all()
        assert (args["out_top_logprobs"][i] == 0).all()
    assert (args["out_top_ids"][0, 5:] == -1).all()
    assert (args["out_top_logprobs"][0, 5:] == 0).all()
    assert (args["out_top_ids"][1] == -1).all()


@pytest.mark.parametrize("row_chunk", [1, 3, 4])
@pytest.mark.parametrize("mode", ["raw", "processed_a"])
def test_flattened_ties_across_rows_chunks_and_nan_padding(row_chunk, mode, monkeypatch):
    x = torch.zeros((7, 17), dtype=torch.float32)
    x[0, :9] = 4  # Many ties followed by a flat prefix-count tail.
    x[1].fill_(torch.nan)  # Skipped in the middle of a chunk.
    x[2, 10:] = 3  # Every boundary tie occurs late in this row.
    x[2, 1] = 9
    x[3, [0, 2, 4, 6, 16]] = 5
    x[4].fill_(torch.inf)  # Another skipped row, including a chunk start.
    x[5, 11:] = torch.tensor([3, 4, 5, 6, 7, 8])  # Only one boundary tie needed.
    x[6].fill_(torch.nan)  # Final padding row with no boundary ties.
    args = arguments(7, 5)
    args["selected_ids"][:] = torch.tensor([8, -1, 10, 16, -1, 11, 2**31 - 1])
    args["row_indices"][-1] = 2**31 - 1
    args["num_rows"].fill_(6)
    args["top_k"][3] = 3
    op = TopLogprobs("cpu", 17, 5, row_chunk=row_chunk)
    options = {"mode": "processed", "probabilities": x} if mode == "processed_a" else {}
    if mode == "processed_a":
        options.update(processing(7, 17))

    def forbidden(*args, **kwargs):
        raise AssertionError("top-list selection must reuse cached ranges")

    ordered_ids = op._ordered_ids

    def checked_ordered_ids(values):
        # Both processed sources now construct the S1 distribution first.
        # The flattened selection scan itself still uses cached ranges.
        with monkeypatch.context() as patch:
            patch.setattr(torch, "arange", forbidden)
            return ordered_ids(values)

    monkeypatch.setattr(op, "_ordered_ids", checked_ordered_ids)
    op.run(x, **args, **options)
    for row in (0, 2, 3, 5):
        lp, ids, values = reference_row(
            x[row],
            mode="processed" if mode == "processed_a" else "raw",
            probabilities=x[row] if mode == "processed_a" else None,
        )
        k, sid = int(args["top_k"][row]), int(args["selected_ids"][row])
        assert torch.equal(args["out_top_ids"][row, :k].long(), ids[:k])
        assert_precision(args["out_top_logprobs"][row, :k], lp[ids[:k]])
        assert_precision(args["out_selected_logprobs"][row : row + 1], lp[sid : sid + 1])
        assert args["out_ranks"][row] == (values >= values[sid]).sum()
        assert (args["out_top_ids"][row, k:] == -1).all()
        assert (args["out_top_logprobs"][row, k:] == 0).all()
    for row in (1, 4, 6):
        assert args["out_selected_logprobs"][row] == 0
        assert args["out_ranks"][row] == 0
        assert (args["out_top_ids"][row] == -1).all()
        assert (args["out_top_logprobs"][row] == 0).all()


@pytest.mark.parametrize("vocab,dtype", [(2**30 - 1, torch.int32), (2**30, torch.int64)])
def test_flattened_scan_count_width(vocab, dtype):
    # Construction does not allocate vocabulary-sized storage.
    op = TopLogprobs("cpu", vocab, 5, row_chunk=2)
    assert op._scan_dtype == dtype
    assert op._targets.dtype == dtype  # Avoid searchsorted promoting the full scan.


@pytest.mark.parametrize("mode", ["raw", "processed_a", "processed_b"])
def test_full_vocab_and_small_path_match(mode):
    x = ((torch.arange(1105) * 17) % 11).float()[None, :].repeat(2, 1)
    x[:, 700:] = -torch.inf
    options = {}
    if mode == "processed_a":
        options = {
            "mode": "processed",
            "probabilities": x.softmax(-1),
            "min_p": torch.tensor([0.0, 0.5]),
            **processing(2, 1105),
        }
    elif mode == "processed_b":
        options = {
            "mode": "processed",
            "temperatures": torch.tensor([1.0, 0.0]),
            "filter_top_k": torch.tensor([33, 11], dtype=torch.int32),
            "top_p": torch.ones(2),
            "min_p": torch.tensor([0.0, 0.5]),
        }
    small, full = arguments(2, 64), arguments(2, 1105)
    TopLogprobs("cpu", 1105, 64).run(x, **small, **options)
    TopLogprobs("cpu", 1105, 1105).run(x, **full, **options)
    for name in ("out_top_ids", "out_top_logprobs"):
        torch.testing.assert_close(small[name], full[name][:, :64], rtol=0, atol=0)
    assert torch.equal(
        full["out_top_ids"].sort(-1).values, torch.arange(1105, dtype=torch.int32).expand(2, -1)
    )
    expected, ids, _ = reference_row(
        x[0],
        mode="raw" if mode == "raw" else "processed",
        probabilities=options.get("probabilities", [None])[0],
        filter_top_k=33 if mode == "processed_b" else None,
    )
    assert torch.equal(full["out_top_ids"][0].long(), ids)
    assert_precision(full["out_top_logprobs"][0], expected[ids])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capacity", [1, 5, 20, 1024])
@pytest.mark.parametrize("boundary", [0.0, -torch.inf])
def test_tie_repair_ignores_topk_boundary_choice(dtype, capacity, boundary, monkeypatch):
    values = torch.full((2, 1105), boundary, dtype=dtype)
    values[0, -3:] = torch.tensor([4.0, 5.0, 4.0], dtype=dtype)
    calls = []

    def highest_id_topk(x, k, dim, sorted):
        assert not sorted
        calls.append(x.dtype)
        # A valid topk that deliberately returns the highest IDs among ties.
        ids = x.shape[1] - 1 - x.flip(-1).argsort(dim=dim, descending=True, stable=True)[:, :k]
        # Return the selected set in reverse order to exercise sorted=False.
        ids = ids.flip(-1)
        return x.gather(1, ids), ids

    monkeypatch.setattr(torch, "topk", highest_id_topk)
    actual = TopLogprobs("cpu", 1105, capacity)._ordered_ids(values)
    expected = values.argsort(dim=-1, descending=True, stable=True)[:, :capacity]
    assert torch.equal(actual, expected)
    assert calls == [dtype]


@pytest.mark.parametrize(
    "vocab,dtype",
    [
        (151936, torch.float32),
        (151936, torch.bfloat16),
        (248320, torch.float32),
        (248320, torch.bfloat16),
        (4096, torch.float16),
    ],
)
def test_raw_adversarial_precision(vocab, dtype, record_property):
    x = torch.linspace(-10000, 10000, vocab).to(dtype)[None, :].repeat(2, 1)
    x[0, ::7] = -torch.inf
    x[1].fill_(9984)
    x[1, 0] = 9983
    args = arguments(2, 20)
    args["selected_ids"][:] = torch.tensor([1, 0])
    TopLogprobs("cpu", vocab, 20).run(x, **args)
    errors = []
    for row in range(2):
        lp, ids, values = reference_row(x[row])
        sid = int(args["selected_ids"][row])
        assert torch.equal(args["out_top_ids"][row].long(), ids[:20])
        assert args["out_ranks"][row] == (values >= values[sid]).sum()
        errors.append(
            assert_precision(args["out_selected_logprobs"][row : row + 1], lp[sid : sid + 1])
        )
        errors.append(assert_precision(args["out_top_logprobs"][row], lp[ids[:20]]))
    record_property("max_abs_error", max(errors))


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0, 2.5])
def test_processed_b_precision_and_filters(temperature):
    x = torch.tensor([[10000.0, 10000.0, 9999.0, 9998.0, -10000.0, -torch.inf]])
    args = arguments(1, 6)
    args["selected_ids"].fill_(4)
    opts = {
        "mode": "processed",
        "temperatures": torch.tensor([temperature]),
        "filter_top_k": torch.tensor([5], dtype=torch.int32),
        "top_p": torch.ones(1),
    }
    TopLogprobs("cpu", 6, 6).run(x, **args, **opts)
    lp, order, values = reference_row(
        x[0], mode="processed", temperature=temperature, filter_top_k=5
    )
    assert torch.equal(args["out_top_ids"][0].long(), order)
    assert_precision(args["out_top_logprobs"][0], lp[order])
    assert_precision(args["out_selected_logprobs"], lp[4:5])
    assert args["out_ranks"][0] == (values >= values[4]).sum()


@pytest.mark.parametrize(
    "p", [0.5, float(torch.nextafter(torch.tensor(0.5), torch.tensor(1.0))), 1.0]
)
@pytest.mark.parametrize("min_p", [0.0, 0.5, 1.0])
def test_s1_exact_boundaries(p, min_p):
    x = torch.zeros((1, 8))
    args = arguments(1, 8)
    args["selected_ids"].fill_(7)
    TopLogprobs("cpu", 8, 8).run(
        x,
        **args,
        mode="processed",
        temperatures=torch.ones(1),
        filter_top_k=torch.tensor([4], dtype=torch.int32),
        top_p=torch.tensor([p]),
        min_p=torch.tensor([min_p]),
    )
    probs = probability_row(x[0], 1.0, 4, p)
    probs = torch.where(probs >= torch.tensor(min_p) * probs.max(), probs, 0)
    expected = probs.double().log() - probs.double().sum().log()
    ids = args["out_top_ids"][0].long()
    assert_precision(args["out_top_logprobs"][0], expected[ids])
    assert ids.tolist() == list(range(8))
    assert args["out_ranks"].item() == 8


def test_probability_source_stride_min_p_and_indexing():
    probs = torch.zeros((4, 12))[:, :5]
    probs[:] = torch.tensor([0.5, 0.25, 0.125, 0.125, 0.0])
    probs[2] = torch.tensor([0.0, 0.5, 0.25, 0.25, 0.0])
    x = probs.log()
    x[2, 3] += 0.25  # Equal supplied masses use the indexed logits to break ties.
    args = arguments(3, 5)
    args["row_indices"][:] = torch.tensor([2, 1, 123456])
    args["selected_ids"][:] = torch.tensor([3, 2, -1])
    args["top_k"][:] = torch.tensor([3, 5, 0])
    TopLogprobs("cpu", 5, 5, row_chunk=2).run(
        x,
        **args,
        mode="processed",
        probabilities=probs,
        **processing(3, 5),
        min_p=torch.tensor([0.5, 0.5, float("nan")]),
    )
    assert args["out_top_ids"][0].tolist() == [1, 3, 2, -1, -1]
    assert args["out_ranks"].tolist() == [2, 5, 0]
    for row, source in ((0, 2), (1, 1)):
        lp, _, _ = reference_row(
            x[source], mode="processed", probabilities=probs[source], min_p=0.5
        )
        assert_precision(
            args["out_selected_logprobs"][row : row + 1], lp[args["selected_ids"][row]].reshape(1)
        )


@pytest.mark.parametrize("vocab", [151936, 248320])
@pytest.mark.parametrize("min_p", [None, 0.5])
def test_large_probability_source_precision(vocab, min_p, record_property):
    probs = torch.linspace(-103, 0, vocab).exp()[None, :].repeat(2, 1)
    probs[0, ::17] = 0
    probs[1].fill_(0.25)
    args = arguments(2, 20)
    args["selected_ids"].fill_(1)
    TopLogprobs("cpu", vocab, 20).run(
        torch.zeros((2, vocab)),
        **args,
        mode="processed",
        probabilities=probs,
        **processing(2, vocab),
        min_p=None if min_p is None else torch.full((2,), min_p),
    )
    errors = []
    for row in range(2):
        lp, ids, values = reference_row(
            torch.zeros(vocab),
            mode="processed",
            probabilities=probs[row],
            min_p=0 if min_p is None else min_p,
        )
        assert torch.equal(args["out_top_ids"][row].long(), ids[:20])
        assert args["out_ranks"][row] == (values >= values[1]).sum()
        errors.append(assert_precision(args["out_top_logprobs"][row], lp[ids[:20]]))
        errors.append(assert_precision(args["out_selected_logprobs"][row : row + 1], lp[1:2]))
    record_property("max_abs_error", max(errors))


def test_raw_order_uses_logits_not_rounded_logprobs():
    x = torch.tensor([[0.0, 1e-9, 0.0, -torch.inf]])
    args = arguments(1, 4)
    TopLogprobs("cpu", 4, 4).run(x, **args)
    assert args["out_top_logprobs"][0, 0] == args["out_top_logprobs"][0, 1]
    assert args["out_top_ids"].tolist() == [[1, 0, 2, 3]]
    assert args["out_ranks"].tolist() == [3]


def test_zero_capacities_and_warmup(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero capacity must skip top-list selection")

    monkeypatch.setattr(TopLogprobs, "_ordered_ids", forbidden)
    for n in (0, 3):
        op = TopLogprobs("cpu", 13, 0)
        assert op.provider == "torch" and op.workspace_size(n) == 0
        op.run(
            torch.zeros((max(n, 1), 13)),
            **arguments(n, 0),
            workspace=torch.empty(0, dtype=torch.uint8),
        )
        op.warm_up([n], dtypes=(torch.float32,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"vocab_size": True},
        {"vocab_size": 2**31},
        {"max_top_k": -1},
        {"max_top_k": 11},
        {"row_chunk": 0},
        {"row_chunk": True},
    ],
)
def test_constructor_validation(kwargs):
    values = {"vocab_size": 10, "max_top_k": 3}
    values.update(kwargs)
    with pytest.raises(ValueError):
        TopLogprobs("cpu", **values)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("num_rows", torch.tensor([1]), TypeError),
        ("selected_ids", torch.tensor([0.0]), TypeError),
        ("row_indices", [0], TypeError),
        ("top_k", torch.zeros(2, dtype=torch.int32), ValueError),
        ("out_top_ids", torch.zeros((3, 1), dtype=torch.int32), ValueError),
        ("out_top_logprobs", torch.empty((1, 6))[:, ::2], ValueError),
        ("workspace", torch.empty(0), TypeError),
        ("min_p", torch.ones(1, dtype=torch.float64), TypeError),
        ("mode", "unknown", ValueError),
    ],
)
def test_argument_validation(field, value, error):
    args = arguments(1, 3)
    args[field] = value
    with pytest.raises(error):
        TopLogprobs("cpu", 10, 3).run(torch.zeros((1, 10)), **args)


def test_layout_grad_missing_params_and_alias_validation():
    op, args = TopLogprobs("cpu", 10, 3), arguments(1, 3)
    for x in (
        torch.zeros((1, 20))[:, ::2],
        torch.zeros((1, 10), requires_grad=True),
        torch.zeros((2, 10)).as_strided((1, 10), (0, 1)),
    ):
        with pytest.raises(ValueError):
            op.run(x, **args)
    with pytest.raises(ValueError, match="temperatures"):
        op.run(torch.zeros((1, 10)), **args, mode="processed")
    with pytest.raises(ValueError, match="stride"):
        op.run(
            torch.zeros((1, 10)),
            **args,
            mode="processed",
            probabilities=torch.zeros((1, 20))[:, ::2],
            **processing(1, 10),
        )
    args["out_ranks"] = args["selected_ids"]
    with pytest.raises(ValueError, match="alias"):
        op.run(torch.zeros((1, 10)), **args)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_processed_precision(dtype, record_property):
    vocab = 248320
    row = torch.linspace(-10000, 10000, vocab).to(dtype)
    row[::17] = -torch.inf
    x = row[None, :].repeat(4, 1)
    args = arguments(4, vocab)
    args["selected_ids"].fill_(1)
    temperatures = [0.0, 0.3, 1.0, 2.5]
    TopLogprobs("cpu", vocab, vocab, row_chunk=2).run(
        x,
        **args,
        mode="processed",
        temperatures=torch.tensor(temperatures),
        filter_top_k=torch.full((4,), vocab, dtype=torch.int32),
        top_p=torch.ones(4),
    )
    errors = []
    for i, temperature in enumerate(temperatures):
        lp, ids, _ = reference_row(row, mode="processed", temperature=temperature)
        got = args["out_top_ids"][i].long()
        # Provider-specific exponentials can move subnormal masses by a
        # quantum. Exact orders are checked separately on controlled ties.
        normal = lp[ids].exp() >= torch.finfo(torch.float32).tiny
        assert torch.equal(got[normal], ids[normal])
        assert torch.equal(got[~normal].sort().values, ids[~normal].sort().values)
        p = lp[got[~normal]].exp()
        assert (p[1:] <= p[:-1] + 2**-148).all()
        errors.append(assert_precision(args["out_top_logprobs"][i], lp[got]))
    record_property("max_abs_error", max(errors))


@pytest.mark.parametrize("supplied", [False, True])
def test_processed_orders_equal_probabilities_by_lower_id(supplied):
    # S1 rounds all three finite entries to the same FP32 probability.
    x = torch.tensor([[0.0, 1e-9, 0.0, -torch.inf]])
    args = arguments(1, 4)
    args["selected_ids"].fill_(1)
    params = processing(1, 4)
    if supplied:
        params["probabilities"] = probability_row(x[0], 1.0, 4, 1.0)[None, :]
    TopLogprobs("cpu", 4, 4).run(x, **args, mode="processed", **params)
    assert args["out_top_ids"].tolist() == [[0, 1, 2, 3]]
    assert args["out_ranks"].tolist() == [3]
    _, ids, values = reference_row(
        x[0], mode="processed", probabilities=params.get("probabilities", [None])[0]
    )
    assert ids.tolist() == [0, 1, 2, 3]
    assert (values >= values[1]).sum() == 3


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
@pytest.mark.parametrize(
    "logits,k,expected_ids,expected_ranks",
    [
        ([-120.0, -110.0, 0.0, -torch.inf], 4, [2, 1, 0, 3], [3, 2, 1, 4]),
        (
            [-120.0, 0.0, -110.0, 0.0, -120.0, -torch.inf, -115.0, -1.0, -130.0, -140.0],
            8,
            [1, 3, 7, 2, 6, 0, 4, 8, 5, 9],
            [7, 2, 4, 2, 7, 10, 5, 3, 8, 10],
        ),
    ],
    ids=["reviewer_underflow", "mixed_ties_and_filtered"],
)
def test_processed_total_order_and_ranks(
    supplied, full, values, logits, k, expected_ids, expected_ranks
):
    x = torch.tensor([logits])
    vocab = len(logits)
    capacity = vocab if full else 3
    probs = probability_row(x[0], 1.0, k, 1.0)
    assert probs[0] == probs[2 if vocab == 10 else 1] == 0
    args = arguments(vocab, capacity)
    args["row_indices"].zero_()
    args["selected_ids"][:] = torch.arange(vocab, dtype=torch.int32)
    opts = processing(vocab, vocab)
    opts["filter_top_k"].fill_(k)
    if supplied:
        opts["probabilities"] = probs[None, :]
    TopLogprobs("cpu", vocab, capacity, row_chunk=3).run(
        x, **args, mode="processed", values=values, **opts
    )
    assert args["out_top_ids"].tolist() == [expected_ids[:capacity]] * vocab
    assert args["out_ranks"].tolist() == expected_ranks
    lp, ids, values = reference_row(
        x[0],
        mode="processed",
        values=values,
        probabilities=probs if supplied else None,
        filter_top_k=k,
    )
    assert ids.tolist() == expected_ids
    assert [(values >= value).sum().item() for value in values] == expected_ranks
    assert_precision(args["out_selected_logprobs"], lp)
    assert_precision(args["out_top_logprobs"], lp[ids[:capacity]].expand(vocab, -1))
    # Supplied zero masses stay -inf even though their finite logits break ties.
    assert torch.isneginf(args["out_selected_logprobs"][0]).item() == supplied


def test_extreme_finite_logits_shift_overflow():
    limit = torch.finfo(torch.float32).max
    x = torch.tensor([[limit, -limit, -torch.inf, 0.0]])
    args = arguments(1, 4)
    TopLogprobs("cpu", 4, 4).run(
        x,
        **args,
        mode="processed",
        temperatures=torch.tensor([4.0]),
        filter_top_k=torch.tensor([4], dtype=torch.int32),
        top_p=torch.ones(1),
    )
    lp, ids, _ = reference_row(x[0], mode="processed", temperature=4.0)
    assert torch.equal(args["out_top_ids"][0].long(), ids)
    assert_precision(args["out_top_logprobs"][0], lp[ids])


def test_top_p_probability_tie_can_remove_raw_argmax():
    x = torch.tensor([[0.0, 1e-9, -torch.inf]])
    args = arguments(1, 3)
    TopLogprobs("cpu", 3, 3).run(
        x,
        **args,
        mode="processed",
        temperatures=torch.ones(1),
        filter_top_k=torch.tensor([3], dtype=torch.int32),
        top_p=torch.tensor([0.5]),
    )
    assert args["out_selected_logprobs"].item() == 0
    assert args["out_top_ids"].tolist() == [[0, 1, 2]]
    assert torch.isneginf(args["out_top_logprobs"][0, 1:]).all()


@pytest.mark.parametrize(
    "temperature,k,p,mp", [(0.3, 31, 0.8, 0.5), (1.0, 73, 0.95, 0.1), (2.5, 127, 0.7, 0.4)]
)
def test_nonuniform_s1_support(temperature, k, p, mp):
    x = ((torch.arange(127) * 17) % 41).float()[None, :] / 7
    args = arguments(1, 127)
    args["selected_ids"].fill_(4)
    TopLogprobs("cpu", 127, 127).run(
        x,
        **args,
        mode="processed",
        temperatures=torch.tensor([temperature]),
        filter_top_k=torch.tensor([k], dtype=torch.int32),
        top_p=torch.tensor([p]),
        min_p=torch.tensor([mp]),
    )
    lp, ids, values = reference_row(
        x[0], mode="processed", temperature=temperature, filter_top_k=k, top_p=p, min_p=mp
    )
    assert torch.equal(args["out_top_ids"][0].long(), ids)
    assert_precision(args["out_top_logprobs"][0], lp[ids])
    assert args["out_ranks"].item() == int((values >= values[4]).sum())


@pytest.mark.parametrize("source", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
@pytest.mark.parametrize("floor", [None, -9999.0])
def test_run_never_reads_device_values_on_host(source, values, floor, monkeypatch):
    args = arguments(3, 5)
    args["num_rows"].fill_(2)
    args["row_indices"][-1] = 999
    options = {}
    if source == "processed_a":
        options = {
            "mode": "processed",
            "probabilities": torch.full((3, 13), 1 / 13),
            **processing(3, 13),
        }
    elif source == "processed_b":
        options = {
            "mode": "processed",
            "temperatures": torch.ones(3),
            "filter_top_k": torch.full((3,), 13, dtype=torch.int32),
            "top_p": torch.ones(3),
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("device value readback")

    x = torch.zeros((3, 13))
    x[0].fill_(-torch.inf)
    if source == "processed_a":
        options["probabilities"][0].zero_()
    for method in ("item", "tolist", "nonzero", "__bool__", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    TopLogprobs("cpu", 13, 5, row_chunk=2).run(x, **args, **options, values=values, floor=floor)


@pytest.mark.parametrize("mode", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("capacity", [0, 5, 1105])
@pytest.mark.parametrize("garbage", [float("nan"), torch.inf, -torch.inf])
def test_padding_can_gather_an_unused_storage_row(mode, capacity, garbage):
    x = torch.full((2, 1105), -torch.inf)
    x[0].fill_(garbage)
    x[1, 0] = 0
    args = arguments(3, capacity)
    args["row_indices"][:] = torch.tensor([1, 999, -999])
    args["selected_ids"][:] = torch.tensor([0, -1, 999999])
    args["num_rows"].fill_(2)
    options = {}
    if mode == "processed_a":
        options = {"mode": "processed", "probabilities": x.exp(), **processing(3, 1105)}
    elif mode == "processed_b":
        options = {
            "mode": "processed",
            "temperatures": torch.tensor([1.0, float("nan"), float("nan")]),
            "filter_top_k": torch.tensor([1105, -999, -999], dtype=torch.int32),
            "top_p": torch.tensor([1.0, float("nan"), float("nan")]),
        }
    TopLogprobs("cpu", 1105, capacity, row_chunk=2).run(x, **args, **options)
    assert args["out_ranks"].tolist() == [1, 0, 0]
    assert args["out_selected_logprobs"].tolist() == [0.0, 0.0, 0.0]
    assert (args["out_top_ids"][1:] == -1).all()
    assert (args["out_top_logprobs"][1:] == 0).all()


def test_top_p_uses_integer_prefix_not_rounded_float_cumsum():
    # Three FP32 sevenths round UP to this FP32 threshold. Their exact fixed
    # mass is still below it, so S1 must retain a fourth token.
    seventh = torch.tensor(1 / 7, dtype=torch.float32)
    threshold = seventh * 3
    assert int(threshold.double() * 2**62) > 3 * int(seventh.double() * 2**62)
    args = arguments(1, 7)
    TopLogprobs("cpu", 7, 7).run(
        torch.zeros((1, 7)),
        **args,
        mode="processed",
        temperatures=torch.ones(1),
        filter_top_k=torch.tensor([7], dtype=torch.int32),
        top_p=threshold.reshape(1),
    )
    assert torch.isfinite(args["out_top_logprobs"][0, :4]).all()
    assert torch.isneginf(args["out_top_logprobs"][0, 4:]).all()
    assert_precision(
        args["out_top_logprobs"][0, :4],
        torch.full((4,), -torch.log(torch.tensor(4.0, dtype=torch.float64))),
    )


def test_probability_source_requires_processing_and_logit_storage_rows():
    args = arguments(1, 2)
    op = TopLogprobs("cpu", 3, 2)
    opts = {"mode": "processed", "probabilities": torch.tensor([[0.5, 0.25, 0.25]])}
    with pytest.raises(ValueError, match="temperatures"):
        op.run(torch.zeros((1, 3)), **args, **opts)
    with pytest.raises(ValueError, match="storage row"):
        op.run(torch.empty((0, 3)), **args, **opts, **processing(1, 3))


@pytest.mark.parametrize("source", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0, 2.5])
def test_so1r_value_modes(source, values, dtype, temperature):
    x = torch.tensor(
        [
            [8, 8, 7, 6, 0, -120, -130, -torch.inf],
            [0, 1e-9, 0, -torch.inf, -110, -120, -130, -140],
            [10000, 10000, 9999, 9998, 0, -10000, -torch.inf, -torch.inf],
        ],
        dtype=dtype,
    )
    x[1, 0] = -0.0
    # Request every token as selected, across repeated storage rows and chunks.
    n, vocab = x.numel(), x.shape[1]
    args = arguments(n, vocab)
    args["row_indices"][:] = torch.arange(3, dtype=torch.int32).repeat_interleave(vocab)
    args["selected_ids"][:] = torch.arange(vocab, dtype=torch.int32).repeat(3)
    opts = {}
    ref_opts = {}
    if source != "raw":
        opts = {"mode": "processed", **processing(n, vocab)}
        opts["temperatures"].fill_(temperature)
        opts["filter_top_k"].fill_(6)
        opts["top_p"].fill_(0.75)
        opts["min_p"] = torch.full((n,), 0.1)
        ref_opts = {
            "mode": "processed",
            "temperature": temperature,
            "filter_top_k": 6,
            "top_p": 0.75,
            "min_p": 0.1,
        }
        if source == "processed_a":
            # Intentionally differs from the derived support/order, including
            # equal masses whose logits-derived tiebreak is significant.
            opts["probabilities"] = torch.tensor([[0.0, 0.25, 0.25, 0.5, 0, 0, 0, 0]]).repeat(3, 1)
    op = TopLogprobs("cpu", vocab, vocab, row_chunk=7)
    baseline = arguments(n, vocab)
    baseline["row_indices"].copy_(args["row_indices"])
    baseline["selected_ids"].copy_(args["selected_ids"])
    op.run(x, **baseline, **opts)
    op.run(x, **args, **opts, values=values)
    assert torch.equal(args["out_top_ids"], baseline["out_top_ids"])
    assert torch.equal(args["out_ranks"], baseline["out_ranks"])
    for row in range(3):
        probabilities = opts["probabilities"][row] if source == "processed_a" else None
        expected, ids, scores = reference_row(
            x[row], **ref_opts, probabilities=probabilities, values=values
        )
        sl = slice(row * vocab, (row + 1) * vocab)
        assert torch.equal(args["out_top_ids"][sl].long(), ids.expand(vocab, -1))
        assert_precision(args["out_selected_logprobs"][sl], expected)
        assert_precision(args["out_top_logprobs"][sl], expected[ids].expand(vocab, -1))
        assert torch.equal(args["out_ranks"][sl], (scores[None, :] >= scores[:, None]).sum(-1))
    if values == "logits":
        expected = x.float().reshape(-1)
        if source != "raw":
            expected = expected / torch.tensor(temperature if temperature else 1.0)
            expected = torch.where(
                torch.isfinite(baseline["out_selected_logprobs"]), expected, -torch.inf
            )
        assert torch.equal(
            args["out_selected_logprobs"].view(torch.int32), expected.view(torch.int32)
        )
        gathered = (
            x.float().index_select(0, args["row_indices"]).gather(1, args["out_top_ids"].long())
        )
        if source != "raw":
            gathered = gathered / torch.tensor(temperature if temperature else 1.0)
            gathered = torch.where(
                torch.isfinite(baseline["out_top_logprobs"]), gathered, -torch.inf
            )
        assert torch.equal(args["out_top_logprobs"].view(torch.int32), gathered.view(torch.int32))


@pytest.mark.parametrize("source", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
@pytest.mark.parametrize("capacity", [0, 5, 1105])
@pytest.mark.parametrize("floor", [None, -9999.0, torch.finfo(torch.float32).min, 7.0])
def test_so1r_empty_rows_floor_and_padding(source, values, capacity, floor):
    vocab = 1105
    x = torch.full((5, vocab), -torch.inf)
    x[1, :3] = torch.tensor([3.0, 2.0, 1.0])
    if source == "processed_a":
        x[2] = torch.arange(vocab).float()  # All-zero supplied mass with nonflat logits.
    x[3].fill_(torch.nan)  # Skipped selected ID.
    x[4].fill_(torch.nan)  # Beyond num_rows.
    args = arguments(5, capacity)
    args["num_rows"].fill_(4)
    args["selected_ids"][:] = torch.tensor([vocab - 1, vocab - 1, 0, -1, 0])
    opts = {}
    if source != "raw":
        opts = {"mode": "processed", **processing(5, vocab)}
        opts["temperatures"][:] = torch.tensor([0.0, 0.3, 2.5, 1.0, 1.0])
        opts["filter_top_k"].fill_(2)
        opts["top_p"].fill_(0.5)
        opts["min_p"] = torch.full((5,), 0.5)
        if source == "processed_a":
            opts["probabilities"] = torch.zeros_like(x)
            opts["probabilities"][1, :3] = torch.tensor([0.5, 0.25, 0.125])
    op = TopLogprobs("cpu", vocab, capacity, row_chunk=2)
    op.run(x, **args, **opts, values=values)
    baseline = {name: value.clone() for name, value in args.items() if name.startswith("out_")}
    op.run(x, **args, **opts, values=values, floor=floor)
    for name in ("out_top_ids", "out_ranks"):
        assert torch.equal(args[name], baseline[name])
    for name in ("out_selected_logprobs", "out_top_logprobs"):
        expected = baseline[name]
        if floor is not None:
            expected = torch.where(torch.isneginf(expected), floor, expected)
        assert torch.equal(args[name], expected)
        assert not torch.isnan(args[name]).any()
        assert (args[name][3:] == 0).all()
    for row in [0, 2] if source == "processed_a" else [0]:
        assert args["out_ranks"][row] == vocab
        assert torch.equal(args["out_top_ids"][row], torch.arange(capacity, dtype=torch.int32))
        expected = -torch.inf if floor is None else floor
        assert args["out_selected_logprobs"][row] == expected
        assert (args["out_top_logprobs"][row] == expected).all()
        ref, ids, scores = reference_row(
            x[row],
            mode="raw" if source == "raw" else "processed",
            values=values,
            floor=floor,
            probabilities=opts.get("probabilities", [None] * 5)[row],
            temperature=0.0 if row == 0 else 2.5,
        )
        assert (ref == expected).all()
        assert torch.equal(ids, torch.arange(vocab))
        assert (scores >= scores[0]).sum() == vocab
    assert (args["out_top_ids"][3:] == -1).all()
    assert (args["out_ranks"][3:] == 0).all()


@pytest.mark.parametrize("values", ["unknown", "raw_logits", None, 1, ["logits"], torch.tensor(0)])
def test_so1r_bad_values(values):
    with pytest.raises(ValueError, match="values"):
        TopLogprobs("cpu", 8, 5).run(torch.zeros((1, 8)), **arguments(1, 5), values=values)


@pytest.mark.parametrize(
    "floor", [float("nan"), float("inf"), -float("inf"), 1, True, "-9999", torch.tensor(-9999.0)]
)
def test_so1r_bad_floor(floor):
    with pytest.raises(ValueError, match="floor"):
        TopLogprobs("cpu", 8, 5).run(torch.zeros((1, 8)), **arguments(1, 5), floor=floor)


def test_so1r_warmup_exercises_value_modes_and_floor(monkeypatch):
    calls = set()
    original = TopLogprobs.run

    def record(self, *args, **kwargs):
        calls.add(
            (
                kwargs["mode"],
                kwargs["values"],
                kwargs["floor"],
                kwargs.get("probabilities") is not None,
            )
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TopLogprobs, "run", record)
    TopLogprobs("cpu", 8, 5).warm_up([0, 1], dtypes=(torch.float32,))
    assert calls == {
        (mode, values, floor, supplied)
        for mode in ("raw", "processed")
        for values in ("logprobs", "logits")
        for floor in (None, -9999.0)
        for supplied in ((False, True) if mode == "processed" else (False,))
    }
