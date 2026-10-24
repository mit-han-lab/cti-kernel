# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""mHC device qualification, including strict floors and mutable graph replay."""

import pytest
import torch

from ctikernels.mhc import MhcExit, exact_reference, measured_floor, reference, support
from ctikernels.mhc._reference import MhcConfig, make_state, make_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 128, 129, 256, 1000)


def same_bits(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.fixture(
    scope="module",
    params=[(h, variant) for h in (4096, 1024, 4104) for variant in ("cluster", "grid")],
    ids=lambda case: f"H{case[0]}-{case[1]}",
)
def prepared(request):
    hidden, variant = request.param
    reason = support("cuda", hidden=hidden)
    if reason is not None:
        pytest.skip(reason)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CTI_MHC_DECODE", variant)
        op = MhcExit("cuda", hidden=hidden, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
    op.prepare(COUNTS)
    yield op, MhcConfig(hidden=hidden)
    op.close()
    torch.backends.cuda.matmul.allow_tf32 = previous


def invoke(op, mode, state, weights, buffers=None):
    outputs = (
        {}
        if buffers is None
        else dict(
            zip(
                {
                    "exit": ("residual_out", "post_out", "comb_out", "out"),
                    "pre": ("post_out", "comb_out", "out"),
                    "post": ("residual_out",),
                }[mode],
                buffers,
                strict=True,
            )
        )
    )
    if mode == "pre":
        return op.pre(state["R"], **weights, **outputs)
    args = (state["x"], state["R"], state["post"], state["comb"])
    if mode == "post":
        return (op.post(*args, **outputs),)
    return op(*args, **weights, **outputs)


def check(actual, mode, state, weights):
    args = (state["x"], state["R"], state["post"], state["comb"])
    market = reference(*args, **weights, mode=mode)
    # Oracle stays on CPU; market runs full-T GEMM on the device under test.
    exact = exact_reference(
        *(v.cpu() for v in args), **{k: v.cpu() for k, v in weights.items()}, mode=mode
    )
    assert all(torch.isfinite(v).all().item() for v in actual)
    # Pre reads exact BF16 inputs, so the market's error is pure FP32 rounding; ours runs the same
    # FP32 Sinkhorn chain in another order (compensated projection, no FP64). Measured on GB300
    # at 1c08243f: post/comb max and RMS 1.0-1.5x market. Gate: 2x on those FP32 outputs, at
    # least 4 FP32 ULPs of the output (T=1 has 4 post values: worst 2.4 ULP = 2.2x market).
    slack = {"post_out": (2.0, 4), "comb_out": (2.0, 4)} if mode == "pre" else None
    failures, stats = measured_floor(actual, market, exact, slack=slack)
    assert not failures, (failures, stats)


@pytest.mark.parametrize("tokens", COUNTS, ids=lambda t: f"T{t}")
@pytest.mark.parametrize("mode", ["exit", "pre", "post"])
def test_floor_determinism_graph(prepared, tokens, mode):
    op, cfg = prepared
    state, weights = make_state(cfg, tokens, device="cuda"), make_weights(cfg, device="cuda")
    first = invoke(op, mode, state, weights)
    check(first, mode, state, weights)
    saved = tuple(v.clone() for v in first)
    second = invoke(op, mode, state, weights, first)
    assert all(same_bits(a, b) for a, b in zip(saved, second, strict=True))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Captured calls must neither compile nor allocate any tensor storage.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            invoke(op, mode, state, weights, first)
    for replay in range(3):
        changed = make_state(cfg, tokens, seed=2000 + replay, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for key in ("x", "R", "post", "comb"):
                state[key].copy_(changed[key])
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        check(first, mode, state, weights)
        eager = invoke(op, mode, state, weights)
        assert all(same_bits(a, b) for a, b in zip(first, eager, strict=True))


@pytest.mark.parametrize("tokens", [1, 3, 5, 8, 17, 33, 128, 256])
def test_pdl_chain(prepared, tokens):
    op, cfg = prepared
    weights = make_weights(cfg, device="cuda")
    state = make_state(cfg, tokens, device="cuda")
    # pre -> exit -> exit -> post. All activation edges depend on the preceding
    # launch; immutable weights remain fixed and ready before the chain.
    pre = invoke(op, "pre", state, weights)
    s1 = {"x": pre[2], "R": state["R"], "post": pre[0], "comb": pre[1]}
    a = invoke(op, "exit", s1, weights)
    s2 = {"x": a[3], "R": a[0], "post": a[1], "comb": a[2]}
    b = invoke(op, "exit", s2, weights)
    s3 = {"x": b[3], "R": b[0], "post": b[1], "comb": b[2]}
    final = invoke(op, "post", s3, weights)
    torch.cuda.synchronize()
    expected = tuple(v.clone() for v in (*pre, *a, *b, *final))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        invoke(op, "pre", state, weights, pre)
        invoke(op, "exit", s1, weights, a)
        invoke(op, "exit", s2, weights, b)
        invoke(op, "post", s3, weights, final)
    for replay in range(3):
        changed = make_state(cfg, tokens, seed=3000 + replay, device="cuda")
        for key in ("x", "R", "post", "comb"):
            state[key].copy_(changed[key])
        invoke(op, "pre", state, weights, pre)
        invoke(op, "exit", s1, weights, a)
        invoke(op, "exit", s2, weights, b)
        invoke(op, "post", s3, weights, final)
        expected = tuple(v.clone() for v in (*pre, *a, *b, *final))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        actual = (*pre, *a, *b, *final)
        assert all(same_bits(x, y) for x, y in zip(actual, expected, strict=True))
    check(final, "post", s3, weights)


def test_residual_midpoint_rounding(prepared):
    op, cfg = prepared
    state = make_state(cfg, 1, device="cuda")
    state["x"].fill_(1)
    state["R"].fill_(1)
    midpoint, tiny = 1 + 2**-8, 2**-26
    state["post"].copy_(torch.tensor([[midpoint, midpoint, -midpoint, -midpoint]], device="cuda"))
    state["comb"].zero_()
    state["comb"][0, 0].copy_(torch.tensor([tiny, -tiny, tiny, -tiny], device="cuda"))
    from ctikernels.mhc._reference import round_bf16

    exact = round_bf16(
        exact_reference(
            state["x"].cpu(),
            state["R"].cpu(),
            state["post"].cpu(),
            state["comb"].cpu(),
            mode="post",
        )[0]
    )
    weights = make_weights(cfg, device="cuda")
    for mode in ("post", "exit"):
        actual = invoke(op, mode, state, weights)[0]
        assert torch.equal(actual.cpu(), exact)


def test_graph_switches_decode_grid(prepared):
    op, cfg = prepared
    weights = make_weights(cfg, device="cuda")
    calls = []
    for mode, tokens in (
        ("exit", 17),
        ("pre", 1),
        ("exit", 7),
        ("pre", 31),
        ("exit", 4),
        ("exit", 33),
        ("exit", 65),
        ("exit", 129),
        ("exit", 256),
        ("exit", 64),
    ):
        state = make_state(cfg, tokens, device="cuda")
        result = invoke(op, mode, state, weights)
        calls.append((mode, state, result, tuple(v.clone() for v in result)))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for mode, state, result, _ in calls:
            invoke(op, mode, state, weights, result)
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        for _, _, result, saved in calls:
            assert all(same_bits(a, b) for a, b in zip(result, saved, strict=True))


@pytest.mark.parametrize("tokens", [1, 3, 4, 8, 33, 128, 256])
@pytest.mark.parametrize("fastdiv", ["0", "1"])
def test_chain_and_trigger_switches(monkeypatch, tokens, fastdiv):
    """Scheduling switches on identical inputs, including graph replay: the trigger
    placement never changes bits; the serial and 16-lane chains sum in different
    orders, so each must pass the floor on its own."""
    monkeypatch.setenv("CTI_MHC_FASTDIV", fastdiv)
    hidden = 4096
    reason = support("cuda", hidden=hidden)
    if reason is not None:
        pytest.skip(reason)
    cfg = MhcConfig(hidden=hidden)
    state, weights = make_state(cfg, tokens, device="cuda"), make_weights(cfg, device="cuda")
    expected = {}
    for chain, early in (("serial", "1"), ("lane16", "1"), ("lane16", "0")):
        monkeypatch.setenv("CTI_MHC_CHAIN", chain)
        monkeypatch.setenv("CTI_MHC_EARLY_TRIGGER", early)
        op = MhcExit("cuda", hidden=hidden, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
        try:
            op.prepare([tokens])
            buffers = invoke(op, "exit", state, weights)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    invoke(op, "exit", state, weights, buffers)
            graph.replay()
            torch.cuda.synchronize()
            actual = tuple(v.clone() for v in buffers)
            check(actual, "exit", state, weights)
            if chain not in expected:
                expected[chain] = actual
            else:
                assert all(same_bits(a, b) for a, b in zip(actual, expected[chain], strict=True))
        finally:
            op.close()


# One token count per decode capacity bucket (4, 16, 32) plus the first prefill count; each case
# compiles its own variants. publish "early" is the default (unset).
@pytest.mark.parametrize(
    "hidden,tokens",
    [(4096, t) for t in (1, 3, 5, 32, 33)] + [(4104, t) for t in (1, 3)],
)
@pytest.mark.parametrize("mode", ["exit", "pre"])
@pytest.mark.parametrize("chain", ["lane16", "serial"])
@pytest.mark.parametrize("publish", ["early", "late"])
def test_last_arriver_matches_all_and_mutable_graph(
    monkeypatch, hidden, tokens, mode, chain, publish
):
    reason = support("cuda", hidden=hidden)
    if reason is not None:
        pytest.skip(reason)
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    cfg = MhcConfig(hidden=hidden)
    state, weights = make_state(cfg, tokens, device="cuda"), make_weights(cfg, device="cuda")
    monkeypatch.setenv("CTI_MHC_CHAIN", chain)
    monkeypatch.delenv("CTI_MHC_PUBLISH", raising=False)
    monkeypatch.setenv("CTI_MHC_EXCHANGE", "all")
    all_op = MhcExit("cuda", hidden=hidden, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
    try:
        all_op.prepare([tokens])
        expected = invoke(all_op, mode, state, weights)
    finally:
        all_op.close()
    monkeypatch.setenv("CTI_MHC_EXCHANGE", "last")
    monkeypatch.setenv("CTI_MHC_PUBLISH", publish)
    last_op = MhcExit("cuda", hidden=hidden, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
    try:
        last_op.prepare([tokens])
        actual = invoke(last_op, mode, state, weights)
        assert all(same_bits(a, b) for a, b in zip(actual, expected, strict=True))
        if publish != "early":  # the mutable-graph check runs on the default
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            invoke(last_op, mode, state, weights, actual)
        for replay in range(3):
            changed = make_state(cfg, tokens, seed=4000 + replay, device="cuda")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for key in ("x", "R", "post", "comb"):
                    state[key].copy_(changed[key])
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            eager = invoke(last_op, mode, state, weights)
            assert all(same_bits(a, b) for a, b in zip(actual, eager, strict=True))
    finally:
        last_op.close()


@pytest.mark.parametrize("hidden", [4096, 4104])
# One token count per decode capacity bucket (4, 16, 32), plus 3 for the graph check below.
@pytest.mark.parametrize("tokens", [1, 3, 5, 32])
@pytest.mark.parametrize("mode", ["exit", "pre"])
@pytest.mark.parametrize("exchange", ["last", "all"])
def test_decode_projection_and_fence_bits(monkeypatch, hidden, tokens, mode, exchange):
    reason = support("cuda", hidden=hidden)
    if reason is not None:
        pytest.skip(reason)
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    monkeypatch.setenv("CTI_MHC_EXCHANGE", exchange)
    for name in ("CTI_MHC_CHAIN", "CTI_MHC_PUBLISH", "CTI_MHC_FASTDIV", "CTI_MHC_EARLY_TRIGGER"):
        monkeypatch.delenv(name, raising=False)
    cfg = MhcConfig(hidden=hidden)
    state, weights = make_state(cfg, tokens, device="cuda"), make_weights(cfg, device="cuda")
    expected = None
    for proj, fence in (("token", "all"), ("token", "single"), ("split", "all"), (None, None)):
        for name, value in (("CTI_MHC_PROJ", proj), ("CTI_MHC_FENCE", fence)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        op = MhcExit("cuda", hidden=hidden, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
        try:
            op.prepare([tokens])
            actual = invoke(op, mode, state, weights)
            if expected is None:
                expected = tuple(v.clone() for v in actual)
            else:
                assert all(same_bits(a, b) for a, b in zip(actual, expected, strict=True)), (
                    proj,
                    fence,
                )
            if proj is None and exchange == "last" and hidden == 4096 and tokens == 3:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    invoke(op, mode, state, weights, actual)
                changed = make_state(cfg, tokens, seed=5000, device="cuda")
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for key in ("x", "R", "post", "comb"):
                        state[key].copy_(changed[key])
                    graph.replay()
                torch.cuda.current_stream().wait_stream(stream)
                eager = invoke(op, mode, state, weights)
                assert all(same_bits(a, b) for a, b in zip(actual, eager, strict=True))
                assert any(not same_bits(a, b) for a, b in zip(actual, expected, strict=True))
        finally:
            op.close()
