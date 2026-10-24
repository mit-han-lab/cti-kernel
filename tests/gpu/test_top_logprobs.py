# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Bounded DN3 device qualification; cleanly skipped on CPU-only workers."""

import pytest
import torch

from ctikernels.sampling import TopLogprobs
from ctikernels.sampling._logprobs_reference import ERROR_ATOL, ERROR_RTOL, reference_row

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


def assert_precision(actual, expected):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    error = (actual[finite] - expected[finite]).abs()
    assert torch.all(error <= ERROR_ATOL + ERROR_RTOL * expected[finite].abs()), error.max()
    assert not torch.isnan(actual).any()
    return float(error.max()) if error.numel() else 0.0


@pytest.mark.parametrize("vocab", [151936, 248320])
@pytest.mark.parametrize("n", [1, 4, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["raw", "processed_a"])
def test_many_boundary_ties_choose_lowest_ids(vocab, n, dtype, mode):
    # Spread more than kcap ties across the vocabulary to exercise topk's
    # single- and multi-block paths, with a tied strictly-greater part too.
    capacity = 20
    base = torch.zeros(vocab, dtype=dtype)
    tie_ids = torch.arange(0, vocab - 4, 97)
    base[tie_ids] = 2
    base[-1], base[-2], base[-4] = 5, 4, 4
    x = base.cuda()[None, :].repeat(n, 1)
    args = arguments(n, capacity, "cuda")
    args["selected_ids"].fill_(97)
    options = {"mode": "processed", "probabilities": x.float()} if mode == "processed_a" else {}
    if mode == "processed_a":
        options.update(
            temperatures=torch.ones(n, device="cuda"),
            filter_top_k=torch.full((n,), vocab, dtype=torch.int32, device="cuda"),
            top_p=torch.ones(n, device="cuda"),
        )
    TopLogprobs("cuda", vocab, capacity).run(x, **args, **options)
    expected = torch.cat((torch.tensor([vocab - 1, vocab - 4, vocab - 2]), tie_ids[: capacity - 3]))
    assert torch.equal(args["out_top_ids"].cpu().long(), expected.expand(n, -1))
    assert (args["out_ranks"] == tie_ids.numel() + 3).all()
    lp = base.double().log() if mode == "processed_a" else base.double()
    lp = lp.log_softmax(-1)
    assert_precision(args["out_top_logprobs"], lp[expected].expand(n, -1))
    assert_precision(args["out_selected_logprobs"], lp[97].expand(n))


@pytest.mark.parametrize("vocab", [151936, 248320])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 16, 32, 48, 64])
def test_raw_decode_and_flat_verify(vocab, dtype, n, record_property):
    # Strided rows, exact ties, masks and wide finite range in the same matrix.
    base = ((torch.arange(vocab) * 31) % 1001).float() * 20 - 10000
    base[::13] = -torch.inf
    base = base.to(dtype)
    backing = torch.empty((n + 1, vocab + 8), dtype=dtype, device="cuda")
    x = backing[:, :vocab]
    x.copy_(base.cuda().expand_as(x))
    args = arguments(n + 1, 64, "cuda")
    args["row_indices"].copy_(torch.arange(n, -1, -1, dtype=torch.int32, device="cuda"))
    args["num_rows"].fill_(n)
    args["row_indices"][-1] = -1000
    args["selected_ids"].fill_(1)
    args["top_k"].copy_(
        torch.tensor(
            ([0, 1, 5, 20, 64] * ((n + 5) // 5))[: n + 1], dtype=torch.int32, device="cuda"
        )
    )
    TopLogprobs("cuda", vocab, 64, row_chunk=7).run(x, **args)
    lp, ids, values = reference_row(base)
    max_error = assert_precision(args["out_selected_logprobs"][:n], lp[1].expand(n))
    assert (args["out_ranks"][:n].cpu() == (values >= values[1]).sum()).all()
    lengths = args["top_k"].cpu()
    output_ids = args["out_top_ids"].cpu()
    output_lp = args["out_top_logprobs"].cpu()
    for row in range(n):
        k = int(lengths[row])
        assert torch.equal(output_ids[row, :k].long(), ids[:k])
        max_error = max(max_error, assert_precision(output_lp[row, :k], lp[ids[:k]]))
        assert (output_ids[row, k:] == -1).all()
        assert (output_lp[row, k:] == 0).all()
    assert args["out_ranks"][-1] == 0
    assert args["out_selected_logprobs"][-1] == 0
    assert (args["out_top_ids"][-1] == -1).all()
    assert (args["out_top_logprobs"][-1] == 0).all()
    record_property("max_abs_error", max_error)


@pytest.mark.parametrize("vocab", [151936, 248320])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_full_vocab_and_processed_precision(vocab, dtype, record_property):
    cpu = torch.linspace(-10000, 10000, vocab).to(dtype)
    cpu[::11] = -torch.inf
    x = cpu.cuda()[None, :].repeat(2, 1)
    small, full = arguments(2, 64, "cuda"), arguments(2, vocab, "cuda")
    small["selected_ids"].fill_(1)
    full["selected_ids"].fill_(1)
    TopLogprobs("cuda", vocab, 64).run(x, **small)
    TopLogprobs("cuda", vocab, vocab).run(x, **full)
    assert torch.equal(small["out_top_ids"], full["out_top_ids"][:, :64])
    torch.testing.assert_close(
        small["out_top_logprobs"], full["out_top_logprobs"][:, :64], rtol=0, atol=0
    )
    lp, ids, _ = reference_row(cpu)
    assert torch.equal(full["out_top_ids"][0].cpu().long(), ids)
    errors = [assert_precision(full["out_top_logprobs"][0], lp[ids])]
    for t in (0.0, 0.3, 1.0, 2.5):
        opts = {
            "mode": "processed",
            "temperatures": torch.full((2,), t, device="cuda"),
            "filter_top_k": torch.full((2,), vocab, dtype=torch.int32, device="cuda"),
            "top_p": torch.ones(2, device="cuda"),
        }
        TopLogprobs("cuda", vocab, 64).run(x, **small, **opts)
        lp, ids, _ = reference_row(cpu, mode="processed", temperature=t)
        assert torch.equal(small["out_top_ids"][0].cpu().long(), ids[:64])
        errors.append(assert_precision(small["out_top_logprobs"][0], lp[ids[:64]]))
        errors.append(assert_precision(small["out_selected_logprobs"][0:1], lp[1:2]))
    record_property("max_abs_error", max(errors))


@pytest.mark.parametrize("vocab", [151936, 248320])
def test_processed_sources_boundary_ties(vocab):
    x = torch.full((4, vocab + 3), -torch.inf, device="cuda")[:, :vocab]
    x[:, :8] = 0
    args = arguments(4, 20, "cuda")
    args["selected_ids"].fill_(7)
    params = {
        "mode": "processed",
        "temperatures": torch.tensor([0.0, 0.3, 1.0, 2.5], device="cuda"),
        "filter_top_k": torch.full((4,), 4, dtype=torch.int32, device="cuda"),
        "top_p": torch.full((4,), 0.5, device="cuda"),
        "min_p": torch.tensor([0.0, 0.5, 1.0, 0.0], device="cuda"),
    }
    TopLogprobs("cuda", vocab, 20).run(x, **args, **params)
    assert torch.equal(args["out_top_ids"].cpu(), torch.arange(20, dtype=torch.int32).expand(4, -1))
    assert (args["out_ranks"] == vocab).all()
    assert args["out_top_logprobs"][0, 0] == 0
    assert torch.isneginf(args["out_top_logprobs"][0, 1:]).all()
    assert_precision(
        args["out_top_logprobs"][1:, :2],
        torch.full((3, 2), -torch.log(torch.tensor(2.0, dtype=torch.float64))),
    )
    assert torch.isneginf(args["out_top_logprobs"][1:, 2:]).all()
    probs = torch.zeros((4, vocab + 7), device="cuda")[:, :vocab]
    probs[:, :3] = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    TopLogprobs("cuda", vocab, 20).run(
        x,
        **args,
        mode="processed",
        probabilities=probs,
        temperatures=torch.ones(4, device="cuda"),
        filter_top_k=torch.full((4,), vocab, dtype=torch.int32, device="cuda"),
        top_p=torch.ones(4, device="cuda"),
        min_p=torch.full((4,), 0.5, device="cuda"),
    )
    assert_precision(
        args["out_top_logprobs"][:, :3],
        torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64).log().expand(4, -1),
    )
    # Zero supplied masses tie-break on the logits-derived logprob: IDs 3-7
    # share log(1/8) with selected ID 7, so the three masses plus five ties rank it 8.
    assert (args["out_ranks"] == 8).all()


@pytest.mark.parametrize("mode", ["raw", "processed_a", "processed_b"])
def test_graph_mutation_replay(mode):
    vocab, n = 151936, 5
    x = torch.zeros((n + 2, vocab + 8), device="cuda")[:, :vocab]
    x[:, :17] = 3
    storage_rows = torch.arange(n + 2, device="cuda")
    x[storage_rows, storage_rows + 23] = storage_rows.float() + 5
    op = TopLogprobs("cuda", vocab, 20, row_chunk=3)
    op.warm_up([n], dtypes=(torch.float32,), modes=("raw", "processed"))
    args = arguments(n, 20, "cuda")
    params = {}
    if mode != "raw":
        params = {
            "mode": "processed",
            "min_p": torch.zeros(n, device="cuda"),
            "temperatures": torch.ones(n, device="cuda"),
            "filter_top_k": torch.full((n,), vocab, dtype=torch.int32, device="cuda"),
            "top_p": torch.ones(n, device="cuda"),
        }
        if mode == "processed_a":
            params["probabilities"] = x.softmax(-1)
        else:
            params.update(
                temperatures=torch.ones(n, device="cuda"),
                filter_top_k=torch.full((n,), 17, dtype=torch.int32, device="cuda"),
                top_p=torch.ones(n, device="cuda"),
            )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(x, **args, **params)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="capture"):
            op.warm_up([n])
        op.run(x, **args, **params)
    for live in (3, 5, 0, 4):
        x.add_(0.25)
        x[:, 23].add_(4)
        args["row_indices"].copy_(torch.tensor([4, 2, 0, 3, 1], dtype=torch.int32, device="cuda"))
        args["num_rows"].fill_(live)
        args["selected_ids"].copy_(
            torch.tensor([23, -1, 2, 0, 14], dtype=torch.int32, device="cuda")
        )
        args["top_k"].copy_(torch.tensor([5, 20, 0, 1, 20], dtype=torch.int32, device="cuda"))
        if params:
            params["min_p"].fill_(0.5 if live == 4 else 0.0)
            if mode == "processed_a":
                params["probabilities"].copy_(x.softmax(-1))
            else:
                params["temperatures"].copy_(torch.tensor([0.3, 0.0, 1.0, 2.5, 0.0], device="cuda"))
                params["filter_top_k"].fill_(11 if live == 4 else 17)
                params["top_p"].fill_(0.5 if live == 4 else 1.0)
        for name, value in args.items():
            if name.startswith("out_"):
                value.fill_(-77)
        graph.replay()
        captured = {name: value.clone() for name, value in args.items() if name.startswith("out_")}
        op.run(x, **args, **params)
        for name, value in captured.items():
            torch.testing.assert_close(value, args[name], rtol=0, atol=0)
        # Inspect replayed results, rather than capture-time output memory.
        assert (captured["out_ranks"][live:] == 0).all()
        assert (captured["out_top_ids"][live:] == -1).all()
        assert (captured["out_top_logprobs"][live:] == 0).all()
        assert (captured["out_selected_logprobs"][live:] == 0).all()
        assert captured["out_ranks"][1] == 0
        assert (captured["out_top_ids"][1] == -1).all()


def so1r_grid():
    # Rotate two output lengths over each row count, then add full-vocabulary
    # cases only for small batches. Every axis is covered without a product
    # over all k; large-R prompt coverage lives in the separate chunked test.
    cases = []
    for vocab in (16, 151936, 248320):
        lengths = [k for k in (0, 1, 5, 20, 1024) if k <= vocab]
        for index, n in enumerate((0, 1, 2, 3, 4, 64)):
            chosen = {lengths[index % len(lengths)], lengths[(index + 1) % len(lengths)]}
            if n in (1, 4) or (vocab == 16 and n == 64):
                chosen.add(vocab)
            cases.extend((n, vocab, k) for k in sorted(chosen))
    return cases


@pytest.mark.parametrize("n,vocab,capacity", so1r_grid())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["raw", "processed"])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
def test_so1r_range_grid(n, vocab, capacity, dtype, mode, values):
    # Discrete levels keep exact ties abundant and avoid provider-dependent
    # subnormal ordering. Three reusable rows bound CPU oracle work per case.
    base = ((torch.arange(vocab) * 17) % 7).float() / 2
    base[::13] = -torch.inf
    templates = torch.stack((base, base.flip(0), torch.zeros_like(base))).to(dtype)
    template_ids = torch.arange(max(n, 1)) % 3
    x = templates[template_ids].cuda()
    if n == 64:
        x = x.reshape(4, 16, vocab).flatten(0, 1)  # Flattened verify rows.
    x[0].fill_(-torch.inf)
    args = arguments(n, capacity, "cuda")
    live = n - 1 if n == 64 else n
    args["num_rows"].fill_(live)
    args["selected_ids"].fill_(1)
    opts = {}
    if mode == "processed":
        opts = {
            "temperatures": torch.full((n,), 2.5, device="cuda"),
            "filter_top_k": torch.full((n,), vocab, dtype=torch.int32, device="cuda"),
            "top_p": torch.ones(n, device="cuda"),
            "min_p": torch.full((n,), 0.5, device="cuda"),
        }
    TopLogprobs("cuda", vocab, capacity).run(x, **args, mode=mode, values=values, **opts)
    outputs = {name: value.cpu() for name, value in args.items() if name.startswith("out_")}
    assert not torch.isnan(outputs["out_selected_logprobs"]).any()
    assert not torch.isnan(outputs["out_top_logprobs"]).any()
    cache = {}
    for row in range(live):
        template = -1 if row == 0 else int(template_ids[row])
        if template not in cache:
            cpu = torch.full((vocab,), -torch.inf) if template == -1 else templates[template]
            cache[template] = reference_row(
                cpu, mode=mode, values=values, temperature=2.5, min_p=0.5
            )
        expected, ids, scores = cache[template]
        assert torch.equal(outputs["out_top_ids"][row].long(), ids[:capacity])
        assert_precision(outputs["out_top_logprobs"][row], expected[ids[:capacity]])
        assert_precision(outputs["out_selected_logprobs"][row : row + 1], expected[1:2])
        assert outputs["out_ranks"][row] == (scores >= scores[1]).sum()
    if live:
        assert outputs["out_ranks"][0] == vocab
        assert torch.equal(outputs["out_top_ids"][0], torch.arange(capacity, dtype=torch.int32))
        assert torch.isneginf(outputs["out_selected_logprobs"][0])
        assert torch.isneginf(outputs["out_top_logprobs"][0]).all()
    assert (outputs["out_selected_logprobs"][live:] == 0).all()
    assert (outputs["out_ranks"][live:] == 0).all()
    assert (outputs["out_top_ids"][live:] == -1).all()
    assert (outputs["out_top_logprobs"][live:] == 0).all()


@pytest.mark.parametrize(
    "n,vocab,capacity",
    [(8192, 248320, k) for k in (0, 1, 5, 20)] + [(16384, 16, k) for k in (0, 1, 5, 16)],
)
def test_so1r_prompt_rows_chunked_fp64(n, vocab, capacity):
    # The largest BF16 input is about 4 GB; outputs are under 2 MB. Generate
    # and validate in bounded chunks, never materializing an R-by-V FP64 copy.
    chunk = 64
    x = torch.empty((n, vocab), dtype=torch.bfloat16, device="cuda")
    base = ((torch.arange(vocab, device="cuda") * 31) % 101).float() / 8
    base[::13] = -torch.inf
    for start in range(0, n, chunk):
        offsets = (torch.arange(start, min(start + chunk, n), device="cuda") % 8).float() / 8
        x[start : start + chunk].copy_(base[None, :] + offsets[:, None])
    x[0].fill_(-torch.inf)
    args = arguments(n, capacity, "cuda")
    args["selected_ids"].copy_((torch.arange(n, device="cuda") % vocab).int())
    op = TopLogprobs("cuda", vocab, capacity)  # Exercise default row_chunk.
    op.run(x, **args)
    assert not torch.isnan(args["out_selected_logprobs"]).any()
    assert not torch.isnan(args["out_top_logprobs"]).any()
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        double = x[sl].double()
        empty = torch.isneginf(double.amax(-1, keepdim=True))
        lp = torch.where(empty, 0, double).log_softmax(-1)
        lp = torch.where(empty, -torch.inf, lp)
        selected = lp.gather(1, args["selected_ids"][sl].long()[:, None])
        assert_precision(args["out_selected_logprobs"][sl], selected[:, 0])
        ranks = (lp >= selected).sum(-1, dtype=torch.int32)
        assert torch.equal(args["out_ranks"][sl], ranks)
    # Eight rows span starts, ends and chunk boundaries, with all eight offset
    # classes represented. Fully check selected values/ranks and every top ID.
    samples = (0, 1, 62, 63, 66, n // 2 + 3, n - 3, n - 4)
    for row in samples:
        expected, ids, scores = reference_row(x[row].cpu())
        sid = row % vocab
        assert_precision(args["out_selected_logprobs"][row : row + 1], expected[sid : sid + 1])
        assert args["out_ranks"][row].cpu() == (scores >= scores[sid]).sum()
        assert torch.equal(args["out_top_ids"][row].cpu().long(), ids[:capacity])
        assert_precision(args["out_top_logprobs"][row], expected[ids[:capacity]])


@pytest.mark.parametrize("source", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("values", ["logprobs", "logits"])
def test_so1r_floor_empty_and_undrawable(source, values):
    x = torch.tensor(
        [[-torch.inf] * 16, [4, 4, 2, 1] + [-torch.inf] * 12, list(range(16))],
        device="cuda",
    )
    args = arguments(3, 16, "cuda")
    args["selected_ids"].fill_(15)
    opts = {}
    if source != "raw":
        opts = {
            "mode": "processed",
            "temperatures": torch.tensor([0.0, 0.3, 2.5], device="cuda"),
            "filter_top_k": torch.full((3,), 2, dtype=torch.int32, device="cuda"),
            "top_p": torch.full((3,), 0.5, device="cuda"),
        }
        if source == "processed_a":
            opts["probabilities"] = torch.zeros_like(x)
            opts["probabilities"][1, :2] = 0.5
    op = TopLogprobs("cuda", 16, 16)
    op.run(x, **args, **opts, values=values)
    baseline = {name: tensor.clone() for name, tensor in args.items() if name.startswith("out_")}
    op.run(x, **args, **opts, values=values, floor=-9999.0)
    for name in ("out_top_ids", "out_ranks"):
        assert torch.equal(args[name], baseline[name])
    for name in ("out_selected_logprobs", "out_top_logprobs"):
        assert not torch.isnan(args[name]).any()
        assert torch.equal(
            args[name], torch.where(torch.isneginf(baseline[name]), -9999.0, baseline[name])
        )
    for row in range(3):
        expected, ids, scores = reference_row(
            x[row].cpu(),
            mode="raw" if source == "raw" else "processed",
            values=values,
            floor=-9999.0,
            temperature=(0.0, 0.3, 2.5)[row],
            filter_top_k=2,
            top_p=0.5,
            probabilities=opts["probabilities"][row].cpu() if source == "processed_a" else None,
        )
        assert torch.equal(args["out_top_ids"][row].cpu().long(), ids)
        assert_precision(args["out_top_logprobs"][row], expected[ids])
        assert_precision(args["out_selected_logprobs"][row : row + 1], expected[15:16])
        assert args["out_ranks"][row].cpu() == (scores >= scores[15]).sum()


@pytest.mark.parametrize("source", ["raw", "processed_a", "processed_b"])
def test_so1r_logits_floor_graph_replay(source):
    vocab, n = 16, 4
    x = torch.zeros((n, vocab), device="cuda")
    args = arguments(n, vocab, "cuda")
    opts = {"values": "logits", "floor": -9999.0}
    if source != "raw":
        opts.update(
            mode="processed",
            temperatures=torch.ones(n, device="cuda"),
            filter_top_k=torch.full((n,), 2, dtype=torch.int32, device="cuda"),
            top_p=torch.ones(n, device="cuda"),
        )
        if source == "processed_a":
            opts["probabilities"] = torch.full_like(x, 1 / vocab)
    op = TopLogprobs("cuda", vocab, vocab, row_chunk=3)
    op.warm_up([n], dtypes=(torch.float32,))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(x, **args, **opts)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.run(x, **args, **opts)
    for live in (3, 4, 0, 2):
        x.copy_(torch.arange(vocab, device="cuda").expand_as(x))
        x[0].fill_(-torch.inf)
        args["num_rows"].fill_(live)
        if source != "raw":
            opts["temperatures"].copy_(torch.tensor([0.0, 0.3, 1.0, 2.5], device="cuda"))
        if source == "processed_a":
            opts["probabilities"].zero_()
            opts["probabilities"][1:, -2:] = 0.5
        graph.replay()
        captured = {
            name: tensor.clone() for name, tensor in args.items() if name.startswith("out_")
        }
        op.run(x, **args, **opts)
        for name, tensor in captured.items():
            assert torch.equal(tensor, args[name])
        assert not torch.isnan(captured["out_selected_logprobs"]).any()
        assert not torch.isnan(captured["out_top_logprobs"]).any()
        if live:
            assert (captured["out_top_logprobs"][0] == -9999).all()
            assert captured["out_selected_logprobs"][0] == -9999
            assert captured["out_ranks"][0] == vocab
            assert torch.equal(
                captured["out_top_ids"][0], torch.arange(vocab, dtype=torch.int32, device="cuda")
            )
        assert (captured["out_selected_logprobs"][live:] == 0).all()
        assert (captured["out_ranks"][live:] == 0).all()
        assert (captured["out_top_ids"][live:] == -1).all()
        assert (captured["out_top_logprobs"][live:] == 0).all()


def _mode_options(mode, n, vocab, x):
    if mode == "raw":
        return {}
    options = {
        "mode": "processed",
        "temperatures": torch.linspace(0, 2, n, device="cuda"),
        "filter_top_k": torch.full((n,), 50, dtype=torch.int32, device="cuda"),
        "top_p": torch.full((n,), 0.9, device="cuda"),
        "min_p": torch.full((n,), 0.01, device="cuda"),
    }
    if mode == "processed_a":
        options["probabilities"] = torch.softmax(x.float(), -1)
    return options


def _run(x, mode, capacity, rows):
    n, vocab = rows.numel(), x.shape[1]
    args = arguments(n, capacity, "cuda")
    args["row_indices"].copy_(rows)
    args["selected_ids"].copy_((rows * 7919) % vocab)
    options = _mode_options(mode, x.shape[0], vocab, x)
    options = {
        key: value if key == "probabilities" else value[rows.long()]
        for key, value in options.items()
        if isinstance(value, torch.Tensor)
    } | {key: value for key, value in options.items() if not isinstance(value, torch.Tensor)}
    TopLogprobs("cuda", vocab, capacity).run(x, **args, **options)
    return {key: value for key, value in args.items() if key.startswith("out_")}


def _inputs(rows, vocab, values, seed):
    # "ties": few distinct values, so thousands of keys tie at the selection boundary.
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if values == "ties":
        return torch.randint(-2, 3, (rows, vocab), device="cuda", generator=generator).float()
    return torch.randn((rows, vocab), device="cuda", generator=generator) * 4


@pytest.mark.parametrize("values", ["random", "ties"])
@pytest.mark.parametrize("mode", ["raw", "processed_a", "processed_b"])
@pytest.mark.parametrize("capacity", [20, 2048])
def test_rows_are_batch_invariant(mode, capacity, values):
    # Each row's outputs are bitwise those of the same row computed alone.
    x = _inputs(6, 151936, values, 5).to(torch.bfloat16)
    batch = _run(x, mode, capacity, torch.arange(6, dtype=torch.int32, device="cuda"))
    for row in range(6):
        alone = _run(x, mode, capacity, torch.tensor([row], dtype=torch.int32, device="cuda"))
        for key, value in alone.items():
            assert torch.equal(batch[key][row : row + 1], value), (row, key)


@pytest.mark.parametrize("values", ["random", "ties"])
@pytest.mark.parametrize("mode", ["raw", "processed_a", "processed_b"])
def test_runs_are_deterministic(mode, values):
    x = _inputs(4, 248320, values, 6)
    rows = torch.arange(4, dtype=torch.int32, device="cuda")
    first = _run(x, mode, 64, rows)
    for _ in range(3):
        again = _run(x, mode, 64, rows)
        for key, value in first.items():
            assert torch.equal(again[key], value), key
