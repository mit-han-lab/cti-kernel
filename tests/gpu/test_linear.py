# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU qualification: cuBLAS-relative FP64 error, tails, determinism and capture.

Condition-aware error uses sum(abs(x*w)), avoiding cancellation denominators.
The ULP floor scales with sqrt(K/1024); RMS permits 5% cuBLAS slack. FP64 is computed
in N chunks to keep the LM head within a 24 GB device's memory budget.
"""

import importlib.metadata
import json

import pytest
import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.linear_smallm import Instance, accuracy, reference_and_condition
from ctikernels.linear._dispatch import DSMEM, L2, QWEN_9B, Config, _hardware, tuned_config
from ctikernels.linear.inference import Linear

from ._exhaustive import ci_cases


@pytest.fixture(scope="module", autouse=True)
def dependencies():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("nvidia-cutlass-dsl absent")
    assert version == "4.8.0", "install the pinned cute extra"
    import cutlass  # noqa: F401 -- broken installs must fail, not skip


@pytest.fixture(autouse=True)
def force(monkeypatch):
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "cute")
    monkeypatch.delenv("CTI_LINEAR_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)


def assert_accuracy(actual, baseline, ref, condition, k):
    assert actual.dtype == torch.bfloat16 and actual.shape == ref.shape
    metrics = accuracy(actual, baseline, ref, condition, k)
    assert metrics["correct"], json.dumps(metrics, sort_keys=True)


@pytest.fixture(scope="module", params=[*QWEN_9B, (1024, 4096), (67, 264), (1, 8), (129, 520)])
def operands(request):
    n, k = request.param
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn((8, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    return x, w, *reference_and_condition(x, w)


@pytest.mark.parametrize("m", range(1, 9))
def test_all_shapes(operands, m):
    x, w, ref, condition = operands
    op = Linear()
    actual = op(x[:m], w)
    assert op.provider == "cute"
    assert_accuracy(actual, F.linear(x[:m], w), ref[:m], condition[:m], x.shape[1])
    assert torch.equal(op(x[:m], w), actual)


@pytest.mark.parametrize(
    "config",
    [
        Config(64, 64, 2, 1),
        Config(128, 128, 2, 4),
        Config(16, 64, 3, 16),
        Config(128, 128, 2, 1, 8),
        Config(128, 128, 2, 2, 8),
        Config(128, 64, 3, 1, 4),
        Config(128, 64, 4, 1, 8),
        Config(64, 128, 4, 1),
        Config(64, 128, 4, 2),
        Config(64, 256, 2, 1),
        Config(16, 64, 2, 1, 1, "reg", 2),
        Config(32, 128, 2, 3, 1, "reg", 4),
        Config(64, 256, 2, 2, 2, "reg", 8),
        Config(96, 128, 2, 4, 3, "reg", 3),
        Config(256, 256, 2, 16, 8, "reg", 8),
        Config(16, 64, 2, 1, 1, "wide", 2),
        Config(64, 512, 2, 8, 4, "wide", 4),
        Config(64, 256, 2, 16, 4, "wide", 2),
        Config(96, 128, 2, 4, 3, "wide", 2),
    ],
)
def test_graphs_and_tail_configs(config):
    x = torch.randn(5, 264, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    op = Linear(config=config)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    out = torch.empty((5, 67), device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert op(x, w, out=out) is out
        other = op(x, w)  # two calls must have independent capture scratch
    assert op.provider == "cute"
    assert op.kernels_per_call == (2 if config.splits > 1 else 1)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert_accuracy(out, F.linear(x, w), *reference_and_condition(x, w), x.shape[1])
        assert torch.equal(out, other)
        assert torch.equal(out, op(x, w))


def test_capture_first_call_raises_then_eager_warm_up_captures():
    x = torch.randn(3, 536, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(71, 536, device="cuda", dtype=torch.bfloat16)
    op = Linear()
    # A never-compiled bucket must not silently run cuBLAS inside the graph.
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(x, w)
    expected = op(x, w)
    graph, outputs = capture_prefixes(op, x, w, (3,))
    graph.replay()
    torch.cuda.synchronize()
    assert op.provider == "cute"
    assert_bits(outputs[3], expected)


def test_validation_and_fallback():
    op = Linear()
    x = torch.randn(65, 272, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 272, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="frozen"):
        op(x, w.requires_grad_())
    w.requires_grad_(False)
    for xx, ww in [
        (x, w),
        (x[:3, ::2], w[:, ::2]),
        (x[:3].float(), w.float()),
        (x[:3, 1:265], w[:, 1:265]),
    ]:
        actual = op(xx, ww)
        assert op.provider == "cublas"
        torch.testing.assert_close(actual, F.linear(xx, ww), rtol=0, atol=0)


def test_alignment_and_k_fallback():
    op = Linear()
    x = torch.randn(3, 272, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 272, device="cuda", dtype=torch.bfloat16)
    shifted = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
    shifted.copy_(x)
    y = op(shifted, w)
    assert op.provider == "cublas"
    assert torch.equal(y, F.linear(x, w))
    out = torch.empty(3 * 67 + 1, device="cuda", dtype=x.dtype)[1:].view(3, 67)
    assert op(x, w, out) is out
    assert op.provider == "cublas"
    assert torch.equal(out, F.linear(x, w))
    xx, ww = x[:, :263].contiguous(), w[:, :263].contiguous()
    assert torch.equal(op(xx, ww), F.linear(xx, ww))
    assert op.provider == "cublas"


def test_independent_graph_pools_and_streams():
    op = Linear(config=Config(16, 64, 2, 8))
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    xs = [torch.randn(4, 264, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in xs]
    graphs, outputs = [], []
    for x, stream in zip(xs, streams, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(x, w)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(op(x, w))
        graphs.append(graph)
    for _ in range(3):
        for x in xs:
            x.copy_(torch.randn_like(x))
        for stream, graph, x in zip(streams, graphs, xs, strict=True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
                op(x, w)  # eager scratch must not race the other graph
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for x, output in zip(xs, outputs, strict=True):
            assert_accuracy(output, F.linear(x, w), *reference_and_condition(x, w), x.shape[1])


@pytest.mark.parametrize("config", [Config(16, 128, 2, 32), Config(128, 128, 2, 4)])
def test_probe_graph_sequence(config):
    """Original probe crash: 64x4096, M=8 after destroying M=1..7 graphs."""
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    ref, condition = reference_and_condition(x, w)
    baseline = {m: Instance("cublas", F.linear, x[:m], w) for m in range(1, 9)}
    op = Linear(config=config)
    op(x, w)
    torch.cuda.synchronize()
    for m in range(1, 9):
        eager = op(x[:m], w)
        torch.cuda.synchronize()
        assert_accuracy(eager, baseline[m].output, ref[:m], condition[:m], x.shape[1])
        ours = Instance("ours", op, x[:m], w)
        for _ in range(3):
            ours.graph.replay()
            torch.cuda.synchronize()
            assert_accuracy(ours.output, baseline[m].output, ref[:m], condition[:m], x.shape[1])
        del ours


@pytest.mark.parametrize(
    "config",
    [
        Config(16, 64, 3, 16),
        Config(128, 128, 2, 4),
        Config(32, 128, 2, 3, 1, "reg", 4),
        Config(32, 128, 2, 3, 1, "wide", 2),
    ],
)
def test_fragment_basis_and_zero_padding(config):
    # Distinct one-hot K positions expose exchanged halves/registers exactly;
    # random inputs alone can obscure small mapping errors. Include the K tail.
    x = torch.zeros(8, 264, device="cuda", dtype=torch.bfloat16)
    columns = torch.tensor([0, 7, 8, 15, 16, 127, 128, 263], device="cuda")
    x[torch.arange(8, device="cuda"), columns] = 1
    w = (torch.arange(67 * 264, device="cuda").reshape(67, 264) % 127).bfloat16()
    op = Linear(config=config)
    for m in range(1, 9):
        actual = op(x[:m], w)
        torch.cuda.synchronize()
        assert op.provider == "cute"
        torch.testing.assert_close(actual, w[:, columns[:m]].t(), rtol=0, atol=0)


@pytest.mark.parametrize("m", range(1, 9))
def test_shipped_dispatch_graphs(operands, m, monkeypatch):
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    source, w, ref, condition = operands
    x = source[:m].clone()
    op = Linear()
    graph = Instance("dispatched", op, x, w)
    kind, _, _ = _hardware(x.device)
    cfg = tuned_config(kind, w.shape[0], w.shape[1], m)
    assert op.provider == ("cute" if cfg is not None else "cublas")
    assert op.kernels_per_call == (cfg.kernels if cfg is not None else 1)
    assert_accuracy(graph.output, F.linear(x, w), ref[:m], condition[:m], x.shape[1])
    x.neg_()
    graph.graph.replay()
    torch.cuda.synchronize()
    assert_accuracy(graph.output, F.linear(x, w), -ref[:m], condition[:m], x.shape[1])


@pytest.mark.parametrize("k", [4096, 17408])
@pytest.mark.parametrize("splits", [1, 3])
def test_register_large_k(k, splits):
    gen = torch.Generator(device="cuda").manual_seed(k)
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(67, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    ref, condition = reference_and_condition(x, w)
    op = Linear(config=Config(32, 256, 2, splits, 1, "reg", 8))
    for m in (1, 8):
        actual = op(x[:m], w)
        assert op.provider == "cute"
        assert_accuracy(actual, F.linear(x[:m], w), ref[:m], condition[:m], k)


@pytest.mark.parametrize(
    "splits,variant,n,k,m",
    ci_cases(
        [1, 3],
        ["reg", "smem", "wide"],
        [(67, 264), (129, 520), (64, 4096)],
        [9, 16, 17, 24, 27, 33, 48, 49, 56, 64],
        core=[
            (3, "reg", 67, 264, 16),
            (3, "reg", 67, 264, 9),
            (1, "reg", 64, 4096, 9),
            (3, "reg", 129, 520, 9),
            (1, "reg", 67, 264, 9),
            (1, "reg", 129, 520, 9),
            (3, "reg", 64, 4096, 9),
            (3, "reg", 67, 264, 17),
            (1, "reg", 64, 4096, 17),
            (1, "reg", 67, 264, 17),
        ],
    ),
)
def test_bxq_float64_accuracy(m, n, k, variant, splits):
    gen = torch.Generator(device="cuda").manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    op = Linear(config=Config(32, 128, 2, splits, 1, variant, 4), provider="cute")
    actual = op(x, w)
    assert op.provider == "cute"
    assert op.kernels_per_call == (2 if splits > 1 else 1)
    assert op.launch_config["mtiles"] == (m + 7) // 8
    assert_accuracy(actual, F.linear(x, w), *reference_and_condition(x, w), k)


@pytest.mark.parametrize(
    "splits,variant,m",
    ci_cases(
        [1, 3],
        ["reg", "smem", "wide"],
        [9, 17, 24, 33, 49, 56, 64],
        core=[(3, "reg", 56), (1, "reg", 9), (3, "reg", 9)],
    ),
)
def test_bxq_bitwise_determinism(m, variant, splits):
    # Repeated eager calls and graph replays give identical bits for each
    # configuration (split-K reduces in a fixed order, no atomics). Rows are
    # also invariant to the runtime M inside one bucket. Different split counts
    # change the FP32 summation order, so they are compared by accuracy only.
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(129, 4096, device="cuda", dtype=torch.bfloat16, generator=gen)
    op = Linear(config=Config(32, 128, 2, splits, 1, variant, 4), provider="cute")
    first = op(x, w).clone()
    assert op.provider == "cute" and op.launch_config["mtiles"] == (m + 7) // 8
    for _ in range(3):
        torch.testing.assert_close(op(x, w), first, rtol=0, atol=0)
    graph = Instance("ours", op, x, w)
    for _ in range(3):
        graph.graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph.output, first, rtol=0, atol=0)
    low = 8 * ((m - 1) // 8) + 1
    prefix = Linear(config=op.config, provider="cute")(x[:low].clone(), w)
    torch.testing.assert_close(prefix, first[:low], rtol=0, atol=0)


@pytest.mark.parametrize("m", [16, 64])
@pytest.mark.parametrize("variant", ["reg", "smem", "wide"])
@pytest.mark.parametrize("splits", [1, 3])
def test_bxq_graph_replay(m, variant, splits):
    x = torch.randn(m, 264, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    op = Linear(config=Config(32, 128, 2, splits, 1, variant, 4), provider="cute")
    graph = Instance("ours", op, x, w)
    assert op.provider == "cute"
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.graph.replay()
        torch.cuda.synchronize()
        assert_accuracy(graph.output, F.linear(x, w), *reference_and_condition(x, w), 264)


def shipped_shapes(limit=3):
    from ctikernels.linear._dispatch import TUNED, device_class

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    kind = device_class((props.major, props.minor), props.multi_processor_count)
    shapes = [
        shape
        for shape, ranges in TUNED.get(kind, {}).items()
        if ranges and shape[0] * shape[1] <= 64 * 1024**2
    ]
    if not shapes:
        pytest.skip("no bounded shipped-table shapes for this device")
    # Smallest three keep compilation and graph checks bounded on GB10 too.
    return kind, sorted(shapes, key=lambda shape: (shape[0] * shape[1], shape))[:limit]


def assert_bits(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def capture_prefixes(op, x, weight, ms, workspace=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launches = {}
    with torch.cuda.stream(stream):
        for m in ms:
            op(x[:m], weight, workspace=workspace)
            launches[m] = (op.provider, op.launch_config)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = {}
        for m in ms:
            outputs[m] = op(x[:m], weight, workspace=workspace)
            assert (op.provider, op.launch_config) == launches[m]
    return graph, outputs


@pytest.mark.parametrize("n,k", QWEN_9B)
def test_qwen35_decode_graph_matches_eager_route(monkeypatch, n, k):
    # Decode B=1-4: same provider, configuration and bits in eager and graph.
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    op = Linear()
    expected = {m: op(x[:m], w) for m in (1, 2, 3, 4)}
    graph, outputs = capture_prefixes(op, x, w, tuple(expected))
    graph.replay()
    torch.cuda.synchronize()
    for m, value in expected.items():
        assert_bits(outputs[m], value)


def test_memory_flat_across_all_m(monkeypatch):
    from ctikernels.linear import inference

    _, shapes = shipped_shapes()
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    # Isolate this test's stream scratch from preceding GPU tests.
    monkeypatch.setattr(inference, "_SCRATCH", {})
    cases = [(Linear(), n, k) for n, k in shapes]
    cases.append((Linear(provider="cute", config=Config(32, 128, 2, 3, 1, "reg", 4)), 67, 264))
    bound = max(op.workspace_bytes(n, k, 64) for op, n, k in cases)
    cases.sort(key=lambda case: case[0].workspace_bytes(case[1], case[2], 64), reverse=True)
    operands = [
        (
            op,
            torch.randn(64, k, device="cuda", dtype=torch.bfloat16),
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
        )
        for op, n, k in cases
    ]
    for op, x, w in operands:
        for m in range(1, 65):
            op(x[:m], w)
    torch.cuda.synchronize()
    allocated, scratch = torch.cuda.memory_allocated(), inference.scratch_bytes()
    assert 0 < scratch <= bound
    for order in (range(64, 0, -1), [*range(1, 65, 2), *range(2, 65, 2)]):
        for op, x, w in operands:
            for m in order:
                op(x[:m], w)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert inference.scratch_bytes() == scratch


def test_engine_workspace_eager_and_graph():
    n, k, m = 67, 264, 16
    op = Linear(provider="cute", config=Config(32, 128, 2, 3, 1, "reg", 4))
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(op.workspace_bytes(n, k, 64) // 4, device="cuda", dtype=torch.float32)
    expected = op(x, w)
    assert_bits(op(x, w, workspace=workspace), expected)
    with pytest.raises(ValueError, match="workspace"):
        op(x, w, workspace=workspace[: 3 * m * n - 1])
    graph, outputs = capture_prefixes(op, x, w, [m], workspace)
    for _ in range(2):
        x.normal_()
        expected = op(x, w)
        graph.replay()
        torch.cuda.synchronize()
        assert_bits(outputs[m], expected)
        assert_bits(op(x, w, workspace=workspace), expected)


def test_shipped_bucket_rows_match_padded_edge_eager_and_graph():
    from ctikernels.linear._dispatch import M_BUCKETS

    kind, shapes = shipped_shapes()
    for n, k in shapes:
        op = Linear(provider="auto")
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        buckets = [(lo, hi) for lo, hi in M_BUCKETS if tuned_config(kind, n, k, hi) is not None]
        ms = sorted({m for lo, hi in buckets for m in (lo, (lo + hi) // 2, hi)})
        expected = {}
        for lo, hi in buckets:
            padded = op(x[:hi], w)
            launch = op.launch_config
            assert op.provider == "cute" and launch is not None
            for m in sorted({lo, (lo + hi) // 2, hi}):
                expected[m] = op(x[:m], w)
                assert op.provider == "cute" and op.launch_config == launch
                assert_bits(expected[m], padded[:m])
        graph, outputs = capture_prefixes(op, x, w, ms)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            for lo, hi in buckets:
                launch = op._plans[x.get_device(), n, k, hi].launch
                for m in {lo, (lo + hi) // 2, hi}:
                    assert_bits(outputs[m], outputs[hi][:m])
                    assert_bits(outputs[m], expected[m])
                    op(x[:m], w)
                    assert op.provider == "cute" and op.launch_config == launch


@pytest.mark.parametrize("variant", ["reg", "smem", "wide"])
@pytest.mark.parametrize("splits", [1, 3])
def test_forced_every_m_matches_bucket_edge(variant, splits):
    op = Linear(provider="cute", config=Config(32, 128, 2, splits, 1, variant, 4))
    x = torch.randn(64, 264, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    for lo, hi in ((5, 8), (9, 16), (57, 64)):
        padded = op(x[:hi], w)
        launch = op.launch_config
        for m in range(lo, hi + 1):
            assert_bits(op(x[:m], w), padded[:m])
            assert op.provider == "cute" and op.launch_config == launch


GATE_CONFIGS = [
    Config(32, tile, stages, splits, 1, variant, 4)
    for variant, tile, stages in (("smem", 128, 2), ("smem", 64, 3), ("reg", 128, 2))
    for splits in (1, 3)
]


GATE_OPERAND_SHAPES = [(4096, 4096, 10240), (5120, 3072, 7168), (96, 256, 264)]


@pytest.fixture(scope="module", params=GATE_OPERAND_SHAPES)
def gate_operands(request):
    n, k, stride = request.param
    gen = torch.Generator(device="cuda").manual_seed(n + k + stride)
    x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    gate = (torch.randn(64, stride, device="cuda", generator=gen) * 3).bfloat16()[:, :k]
    gate[:, 0] = -100
    gate[:, 1] = 100
    from ctikernels.linear import gated_input

    a = gated_input(x, gate)
    return x, w, gate, a, *reference_and_condition(a, w)


# Keep fixture/configuration combinations explicit while sharing module-scoped operands.
@pytest.mark.parametrize(
    "gate_operands,config,m",
    [
        pytest.param(
            shape,
            config,
            m,
            marks=()
            if m == 1 and config_index in (2, 3, 4, 5) and (shape_index, config_index) != (1, 5)
            else pytest.mark.exhaustive,
            id=f"gate_operands{shape_index}-config{config_index}-{m}",
        )
        for shape_index, shape in enumerate(GATE_OPERAND_SHAPES)
        for config_index, config in enumerate(GATE_CONFIGS)
        for m in [1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64]
    ],
    indirect=["gate_operands"],
    scope="module",
)
def test_gate_accuracy_and_bucket_invariance(gate_operands, m, config):
    from ctikernels.linear._dispatch import m_bucket

    x, w, gate, a, ref, condition = gate_operands
    op = Linear(provider="cute", config=config)
    assert op.gate_fused(w.shape[0], x.shape[1], gate.stride(0), m, x.device)
    actual = op(x[:m], w, gate=gate[:m])
    # The fused prologue gates bitwise as gated_input does: only the matmul remains.
    assert_bits(actual, Linear(provider="cute", config=config)(a[:m], w))
    assert_accuracy(actual, F.linear(a[:m], w), ref[:m], condition[:m], x.shape[1])
    assert op.provider == "cute-gated"
    assert op.kernels_per_call == (2 if config.splits > 1 else 1)
    assert op.launch_config["gate_stride"] == gate.stride(0)
    launch = op.launch_config
    edge = m_bucket(m)[1]
    padded = op(x[:edge], w, gate=gate[:edge])
    assert_bits(actual, padded[:m])
    assert op.launch_config == launch


@pytest.mark.parametrize("config", GATE_CONFIGS)
def test_gate_subnormal_sigmoid_matches_torch(config):
    # sigmoid(g) is subnormal for g < -87.3 and 0 in FP32 torch below about -88.7.
    n, k, stride = 96, 256, 264
    values = [-87.0, -87.5, -88.0, -88.5, -89.0, -100.0, 88.0, 0.0]
    x = torch.zeros(1, k, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(1, stride, device="cuda", dtype=torch.bfloat16)[:, :k]
    w = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    for i, g in enumerate(values):
        x[0, i], gate[0, i], w[i, i] = 2.0**126, g, 1.0
    expected = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    actual = Linear(provider="cute", config=config)(x, w, gate=gate)
    torch.testing.assert_close(
        actual[0, : len(values)], expected[0, : len(values)], rtol=2**-7, atol=0
    )


@pytest.mark.parametrize("config", GATE_CONFIGS)
@pytest.mark.parametrize("m", [1, 9, 64])
def test_gate_graph_replay(config, m):
    n, k, stride = 96, 256, 264
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(m, stride, device="cuda") * 3).bfloat16()[:, :k]
    op = Linear(provider="cute", config=config)
    workspace = torch.empty(
        op.workspace_bytes(n, k, m, gate_stride=stride), device="cuda", dtype=torch.uint8
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = op(x, w, gate=gate, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert op(x, w, out, gate=gate, workspace=workspace) is out
        other = op(x, w, gate=gate)  # split scratch from the graph pool
    assert op.provider == "cute-gated"
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert_bits(out, eager)
        assert_bits(other, eager)
        x.normal_()
        gate.normal_().mul_(3)
        eager = op(x, w, gate=gate, workspace=workspace)


@pytest.mark.parametrize("stride", [264, 261])  # Aligned (no gated row) and unaligned.
@pytest.mark.parametrize("plain_cute", [False, True])
def test_gate_auto_unfused_dispatch(monkeypatch, plain_cute, stride):
    from ctikernels.linear import _dispatch as dispatch

    kind, _, _ = _hardware(torch.device("cuda", torch.cuda.current_device()))
    if kind is None:
        pytest.skip("unmeasured hardware tuple")
    n, k, m = 96, 256, 5
    monkeypatch.setitem(dispatch.TUNED_GATED, kind, {})
    monkeypatch.setitem(
        dispatch.TUNED, kind, {(n, k): {(5, 8): Config(32, 128, 2, 3)}} if plain_cute else {}
    )
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(m, stride, device="cuda") * 3).bfloat16()[:, :k]
    a = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    plain = Linear(provider="auto")
    expected = plain(a, w)
    op = Linear(provider="auto")
    assert not op.gate_fused(n, k, stride, m)
    # Exactly the advertised bytes cover the unfused fallback's split-K scratch.
    workspace = torch.empty(
        op.workspace_bytes(n, k, m, gate_stride=stride), device="cuda", dtype=torch.uint8
    )
    assert workspace.numel() == (3 * m * n * 4 if plain_cute else 0)
    actual = op(x, w, gate=gate, workspace=workspace)
    assert_bits(actual, expected)
    assert op.provider == "unfused+" + plain.provider
    if not plain_cute:
        assert_bits(actual, F.linear(a, w))
    assert_accuracy(actual, F.linear(a, w), *reference_and_condition(a, w), k)


def test_gate_cuda_validation():
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 256, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 264, device="cuda", dtype=torch.bfloat16)[:, :256]
    op = Linear(provider="cute", config=Config(rows=32))
    for bad, error in [
        (gate[:, ::2], "shape"),
        (gate[:2], "shape"),
        (torch.empty(3, 512, device="cuda", dtype=x.dtype)[:, ::2], "stride"),
        (gate.detach().requires_grad_(), "gradients"),
    ]:
        with pytest.raises(ValueError, match=error):
            op(x, w, gate=bad)
    with pytest.raises(ValueError, match="out must not share gate"):
        op(x, w, gate=gate, out=gate._base.view(-1)[: 3 * 96].view(3, 96))
    with pytest.raises(ValueError, match="workspace must not share gate"):
        op(x, w, gate=gate, workspace=gate._base)


@pytest.fixture(scope="module", params=[(256, 4096), (512, 4096), (2048, 4096)])
def fp32_operands(request):
    n, k = request.param
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn(64, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    return x, w, *reference_and_condition(x, w)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 9, 16, 33, 64])
@pytest.mark.parametrize("splits", [4, 8, 16, 32])
@pytest.mark.parametrize("variant", ["smem", "reg", "wide"])
def test_fp32_forced_accuracy_repeat_out_and_graph(fp32_operands, m, splits, variant):
    xall, w, ref, condition = fp32_operands
    x = xall[:m]
    op = Linear(provider="cute", config=Config(32, 128, 2, splits, 2, variant))
    actual = op(x, w, out_dtype=torch.float32)
    assert op.provider == "cute" and actual.dtype == torch.float32
    assert op.kernels_per_call == (2 if splits > 1 else 1)
    baseline = torch.mm(x, w.t(), out_dtype=torch.float32)
    metrics = accuracy(actual, baseline, ref[:m], condition[:m], x.shape[1], out_dtype="fp32")
    # FP32 rows ship split-K only (every TUNED_F32 split is in this set): one
    # 4096-long FP32 chain (split 1) exceeds the 4 eps32 floor at M >= 33.
    assert metrics["correct"], json.dumps(metrics, sort_keys=True)
    assert torch.equal(actual, op(x, w, out_dtype=torch.float32))
    out = torch.empty_like(actual)
    assert op(x, w, out=out) is out
    assert torch.equal(out, actual)
    assert op(x, w, out=out, out_dtype=torch.float32) is out
    workspace = torch.empty(
        op.workspace_bytes(w.shape[0], w.shape[1], m, out_dtype=torch.float32),
        dtype=torch.uint8,
        device=x.device,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, w, out=out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, w, out=out, workspace=workspace)
        independent = op(x, w, out_dtype=torch.float32)
    for _ in range(3):
        graph.replay()
        assert torch.equal(out, actual)
        assert torch.equal(independent, actual)


@pytest.mark.parametrize("provider", ["auto", "cublas"])
@pytest.mark.parametrize("m", [1, 3, 16, 64])
def test_fp32_cublas_market_bitwise(monkeypatch, provider, m):
    from ctikernels.linear import _dispatch

    monkeypatch.setattr(_dispatch, "TUNED_F32", {})
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    baseline = torch.mm(x, w.t(), out_dtype=torch.float32)
    op = Linear(provider=provider)
    assert torch.equal(op(x, w, out_dtype=torch.float32), baseline)
    out = torch.empty_like(baseline)
    assert op(x, w, out=out) is out
    expected_out = torch.empty_like(baseline)
    torch.mm(x, w.t(), out=expected_out, out_dtype=torch.float32)
    assert torch.equal(out, expected_out)
    assert op.provider == "cublas" and op.launch_config is None


def strided_outputs(m, widths, pad, dtype=torch.bfloat16):
    """Column slices of one [M, sum(widths) + pad] NaN-filled buffer."""
    buffer = torch.full((m, sum(widths) + pad), float("nan"), device="cuda", dtype=dtype)
    starts = [sum(widths[:i]) for i in range(len(widths))]
    return buffer, [buffer[:, s : s + w] for s, w in zip(starts, widths, strict=True)]


def assert_padding(buffer, used):
    assert torch.isnan(buffer[:, used:]).all()


@pytest.mark.parametrize("m", [1, 2, 9, 64])
@pytest.mark.parametrize("variant", ["reg", "smem", "wide"])
@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_row_strided_out_matches_contiguous_eager_and_graph(m, variant, splits, out_dtype):
    # Two projections fill one [M, N1+N2] buffer (N2 odd: scalar reduce path).
    k, widths, pad = 264, (96, 67), 5
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for n in widths]
    op = Linear(config=Config(32, 128, 2, splits, 1, variant, 4), provider="cute")
    expected = [op(x, w, out_dtype=out_dtype) for w in weights]
    buffer, outs = strided_outputs(m, widths, pad, out_dtype)
    for w, out, want in zip(weights, outs, expected, strict=True):
        assert op(x, w, out, out_dtype=out_dtype) is out
        assert op.provider == "cute"
        assert torch.equal(out, want)
        if m > 1:
            assert op.launch_config["out_stride"] == sum(widths) + pad
    assert_padding(buffer, sum(widths))
    buffer.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w, out in zip(weights, outs, strict=True):
            op(x, w, out, out_dtype=out_dtype)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for out, want in zip(outs, expected, strict=True):
            assert torch.equal(out, want)
        assert_padding(buffer, sum(widths))


@pytest.mark.parametrize("provider", ["cute", "auto", "cublas"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_row_strided_out_unaligned_stride_writes_in_place_with_cublas(provider, out_dtype):
    # Row stride 163 elements is not a multiple of 16 bytes: cuBLAS, in place.
    m, k, widths = 9, 264, (96, 67)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(widths[1], k, device="cuda", dtype=torch.bfloat16)
    buffer, outs = strided_outputs(m, widths, 0, out_dtype)
    op = Linear(config=Config(32, 128, 2, 3), provider=provider)
    assert op(x, w, outs[1], out_dtype=out_dtype) is outs[1]
    assert op.provider == "cublas"
    if out_dtype == torch.float32:
        expected = torch.mm(x, w.t(), out_dtype=torch.float32)
    else:
        expected = F.linear(x, w)
    assert torch.equal(outs[1], expected)
    assert torch.isnan(buffer[:, : widths[0]]).all()


def test_row_strided_out_gated_and_capture_miss():
    m, n, k, stride = 9, 96, 256, 264
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(m, stride, device="cuda") * 3).bfloat16()[:, :k]
    op = Linear(config=Config(32, 128, 2, 3), provider="cute")
    expected = op(x, w, gate=gate)
    buffer, outs = strided_outputs(m, (n, 8), 0)
    assert op(x, w, outs[0], gate=gate) is outs[0]
    assert op.provider == "cute-gated"
    assert torch.equal(outs[0], expected)
    assert torch.isnan(buffer[:, n:]).all()
    # A row stride first seen during capture raises instead of switching provider.
    op(x, w)
    _, (other,) = strided_outputs(m, (n,), 16)
    with pytest.raises(RuntimeError, match="out row stride 112"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(x, w, other)


def test_row_strided_out_validation():
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 256, device="cuda", dtype=torch.bfloat16)
    op = Linear(config=Config(32, 128, 2, 1), provider="cute")
    with pytest.raises(ValueError, match="row-strided"):
        op(x, w, torch.empty(96, 3, device="cuda", dtype=x.dtype).t())
    with pytest.raises(ValueError, match="row-strided"):
        op(x, w, torch.empty(3 * 96, device="cuda", dtype=x.dtype).as_strided((3, 96), (64, 1)))
    # One row: any row stride (it is never used).
    one = torch.empty(1, 512, device="cuda", dtype=x.dtype)[:, 8:104]
    assert op(x[:1], w, one) is one and op.provider == "cute"
    assert torch.equal(one, op(x[:1], w))


@pytest.mark.parametrize("m", [1, 2, 8, 9, 33, 64])
@pytest.mark.parametrize("n,k", [(67, 4096), (128, 2048)])
@pytest.mark.parametrize("variant", ["reg", "smem", "wide"])
@pytest.mark.parametrize("splits", [2, 8, 16])
@pytest.mark.parametrize("mode", [DSMEM, L2])
def test_cluster_reduce_matches_two_launch_eager_and_graph(m, n, k, variant, splits, mode):
    # One launch: the split CTAs of an N tile reduce in split order (through
    # distributed shared memory, or the L2 scratch), bitwise equal to the reduce kernel.
    gen = torch.Generator(device="cuda").manual_seed(m + n + splits)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    twin = Config(32, 128, 2, splits, 1, variant, 4)
    op = Linear(config=Config(32, 128, 2, splits, 1, variant, 4, mode), provider="cute")
    if splits > 8 and torch.cuda.get_device_capability() == (12, 1):
        # sm_121 rejects non-portable (> 8 CTA) clusters: refused before compiling.
        assert op.decision(n, k, m)["provider"] == "cublas"
        with pytest.raises(ValueError, match="cluster"):
            op(x, w)
        return
    actual = op(x, w)
    assert op.provider == "cute" and op.kernels_per_call == 1
    assert op.launch_config["config"]["cluster"] == mode
    assert op.workspace_bytes(n, k) == (0 if mode == DSMEM else splits * 64 * n * 4)
    assert_bits(actual, Linear(config=twin, provider="cute")(x, w))
    assert_accuracy(actual, F.linear(x, w), *reference_and_condition(x, w), k)
    graph = Instance("ours", op, x, w)
    for _ in range(2):
        graph.graph.replay()
        torch.cuda.synchronize()
        assert_bits(graph.output, actual)


@pytest.mark.parametrize("m", [1, 9])
@pytest.mark.parametrize("variant", ["reg", "smem", "wide"])
@pytest.mark.parametrize("mode", [DSMEM, L2])
def test_cluster_reduce_strided_and_fp32_out(m, variant, mode):
    k, widths, pad = 4096, (96, 64), 8
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for n in widths]
    op = Linear(config=Config(32, 128, 2, 4, 1, variant, 4, mode), provider="cute")
    twin = Linear(config=Config(32, 128, 2, 4, 1, variant, 4), provider="cute")
    for out_dtype in (torch.bfloat16, torch.float32):
        buffer, outs = strided_outputs(m, widths, pad, out_dtype)
        for w, out in zip(weights, outs, strict=True):
            assert op(x, w, out, out_dtype=out_dtype) is out
            assert op.kernels_per_call == 1
            assert torch.equal(out, twin(x, w, out_dtype=out_dtype))
        assert_padding(buffer, sum(widths))


@pytest.mark.parametrize("m", [1, 9, 64])
@pytest.mark.parametrize("variant", ["reg", "smem"])
@pytest.mark.parametrize("mode", [DSMEM, L2])
def test_cluster_reduce_gated(gate_operands, m, variant, mode):
    x, w, gate, a, ref, condition = gate_operands
    op = Linear(config=Config(32, 64, 2, 4, 1, variant, 4, mode), provider="cute")
    actual = op(x[:m], w, gate=gate[:m])
    assert op.provider == "cute-gated" and op.kernels_per_call == 1
    twin = Linear(config=Config(32, 64, 2, 4, 1, variant, 4), provider="cute")
    assert_bits(actual, twin(x[:m], w, gate=gate[:m]))
    assert_accuracy(actual, F.linear(a[:m], w), ref[:m], condition[:m], w.shape[1])


@pytest.mark.parametrize("lead", [(), (3,), (2, 3)])
def test_sglang_range_leading_dims_and_bias(monkeypatch, lead):
    # F.linear's range (SGLang's dense call): [..., K] inputs and an [N] bias.
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    k, n = 4096, 4096
    x = torch.randn(*lead, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    op = Linear()
    flat = op(x.reshape(-1, k), w)
    assert torch.equal(op(x, w), flat.view(*lead, n))
    assert torch.equal(op(x, w, bias=b), F.linear(x, w, b))
    assert op.provider == "cublas"


def test_warm_up_then_capture_never_raises(monkeypatch):
    # KW4: after warm_up, capture of every M up to max_m finds a compiled route.
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    kind, shapes = shipped_shapes(limit=2)
    for n, k in shapes:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        op = Linear()
        routes = op.warm_up(w, 64)
        assert all(route["source"] == "table" for route in routes)
        F.linear(x, w)  # cuBLAS handle for the table holes.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = {m: op(x[:m], w) for m in range(1, 65)}
        graph.replay()
        torch.cuda.synchronize()
        for m in (1, 9, 64):
            assert_accuracy(outs[m], F.linear(x[:m], w), *reference_and_condition(x[:m], w), k)


def test_graph_workspace_holds_eager_split_k_scratch():
    from ctikernels import GraphWorkspace
    from ctikernels.linear import inference

    ops = [Linear(config=Config(32, 128, 2, 3, 1, "reg", 4), provider="cute") for _ in range(2)]
    x = torch.randn(33, 264, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    inference.release_scratch()
    with GraphWorkspace(max_rows=32) as group:
        eager = [op(x[:m], w).clone() for op, m in zip(ops, (32, 16), strict=True)]
        assert inference.scratch_bytes() == 0 and group.nbytes >= 4 * 3 * 32 * 67
        graphs, outputs = [], []
        for op, m in zip(ops, (32, 16), strict=True):
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1]):
                outputs.append(op(x[:m], w))  # captures stay call-owned (graph pool)
        with pytest.raises(RuntimeError, match="33 rows exceed max_rows=32"):
            ops[0](x, w)
    for graph, out, expected in zip(graphs, outputs, eager, strict=True):
        graph.replay()
        assert_bits(out, expected)
    group.close()


def test_graph_workspace_largest_first_ladder_reserves_split_k(monkeypatch):
    """The engine's ladder in one GraphWorkspace, no caller workspace: largest bucket first
    (cuBLAS, no split), then smaller split-K buckets whose eager calls come after another
    consumer's take froze the workspace at the first capture. Replays equal eager bitwise."""
    from ctikernels import GraphWorkspace
    from ctikernels.linear import inference

    split = Config(32, 128, 2, 3, 1, "reg", 4)
    monkeypatch.setattr(
        inference, "tuned_config", lambda kind, n, k, edge, **_: split if edge <= 16 else None
    )
    op = Linear(provider="auto", autotune="0")
    rows = (32, 16, 4)
    x = torch.randn(max(rows), 264, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(67, 264, device="cuda", dtype=torch.bfloat16)
    other = ("other", x.device, torch.float32)
    graphs, outputs, eager = [], [], []
    with GraphWorkspace(max_rows=max(rows)) as group:
        group.reserve(
            other, (), lambda _: (1,), lambda counts: (torch.empty(counts[0], device="cuda"),)
        )
        for m in rows:
            eager.append(op(x[:m], w).clone())
            assert (op.provider == "cublas") == (m > 16)
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1]):
                group.take(other, m, (1,))
                outputs.append(op(x[:m], w))
    for graph, out, expected in zip(graphs, outputs, eager, strict=True):
        graph.replay()
        assert_bits(out, expected)
    del graphs
    group.close()


def _torch_gate(x, gate):
    """PyTorch's FP32 gating (exact sigmoid), the reference within one BF16 ULP."""
    return torch.sigmoid(gate.float()).mul_(x).to(x.dtype)


def assert_within_one_ulp(got, reference):
    """BF16 ``got`` equals ``reference`` or is its neighbour (same NaNs)."""
    nan = torch.isnan(reference)
    assert torch.equal(torch.isnan(got), nan)
    a, b = got[~nan].view(torch.int16).int(), reference[~nan].view(torch.int16).int()
    # Signed-magnitude bits: neighbours differ by one in the same sign (or are +-0).
    distance = torch.where((a < 0) == (b < 0), (a - b).abs(), (a & 0x7FFF) + (b & 0x7FFF))
    assert int(distance.max()) <= 1 if distance.numel() else True


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 64, 70000])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_unfused_gate_is_one_launch_within_one_ulp_of_pytorch(m, layout):
    """Qwen3.5-9B's gated o_proj (N = K = 4096) on the unfused route: the Triton gate is within
    one BF16 ULP of PyTorch's exact FP32 gating, extreme gates included, on row-strided x and
    gate, eager and in a CUDA graph; the call is the gate launch plus the plain matmul."""
    from ctikernels.linear.triton._gate import gate_multiply

    n = k = 4096
    pad = 24 if layout == "strided" else 0
    generator = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, k + pad, device="cuda", generator=generator).bfloat16()[:, :k]
    gate = (torch.randn(m, k + pad, device="cuda", generator=generator) * 6).bfloat16()[:, :k]
    gate[:, :6] = torch.tensor([-100, 100, -88.5, 88.5, float("inf"), -float("inf")])
    expected_a = gate_multiply(x, gate)
    assert_within_one_ulp(expected_a, _torch_gate(x, gate))
    if m > 64:
        return
    w = torch.randn(n, k, device="cuda", generator=generator).bfloat16()
    op = Linear(provider="cublas")
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    assert op(x, w, out, gate=gate) is out
    assert_bits(out, Linear(provider="cublas")(expected_a, w))
    assert op.provider == "unfused+cublas" and op.kernels_per_call == 2
    graph = torch.cuda.CUDAGraph()
    replayed = torch.empty_like(out)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        op(x, w, replayed, gate=gate)
    new_x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    x.copy_(new_x)
    graph.replay()
    assert_bits(replayed, op(x, w, gate=gate))


@pytest.mark.parametrize("m", [1, 4, 64])
def test_public_gated_input_matches_the_fallback_and_the_fused_path(m):
    """ctikernels.linear.gated_input: within one BF16 ULP of PyTorch on [M, K] and [B, T, K]
    views, one launch replayed in a CUDA graph bitwise; where the fused CuTe gate is admitted,
    the fused call equals plain(gated_input) bitwise (the same gating, the same matmul)."""
    from ctikernels.linear import gated_input

    n, k, stride = 96, 256, 264
    generator = torch.Generator(device="cuda").manual_seed(2500 + m)
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    gate = (torch.randn(m, stride, device="cuda", generator=generator) * 6).bfloat16()[:, :k]
    expected = gated_input(x, gate)
    assert_within_one_ulp(expected, _torch_gate(x, gate))
    assert_bits(gated_input(x.view(1, m, k), gate.unsqueeze(0)), expected.view(1, m, k))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        replayed = gated_input(x, gate)
    x.copy_(torch.randn(m, k, device="cuda", generator=generator).bfloat16())
    graph.replay()
    assert_bits(replayed, gated_input(x, gate))
    w = torch.randn(n, k, device="cuda", generator=generator).bfloat16()
    op = Linear(provider="auto")
    if not op.gate_fused(n, k, stride, m):
        return
    fused = op(x, w, gate=gate)
    config = op.launch_config["config"]
    assert_bits(fused, Linear(provider="cute", config=Config(**config))(gated_input(x, gate), w))


@pytest.mark.parametrize("config", GATE_CONFIGS[:2])
def test_every_bf16_gate_value_gates_bitwise_on_the_fused_and_unfused_paths(config):
    """All 65536 BF16 gate values times random x: the unfused Triton gate is within one BF16 ULP
    of PyTorch's FP32 ops (NaN included), and the fused CuTe prologue equals it bitwise for
    every non-NaN gate (infinities included), read back through a one-hot matmul exactly."""
    from ctikernels.linear import gated_input

    m, k, n = 8, 8192, 128
    gate = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    gate = gate.view(torch.bfloat16).view(m, k)
    x = torch.randn(m, k, device="cuda").bfloat16()
    assert_within_one_ulp(gated_input(x, gate), _torch_gate(x, gate))
    gate = torch.where(torch.isnan(gate), torch.zeros_like(gate), gate)
    # The matmul's FP32 sum carries no sign of zero: a gated -0 reads back as +0.
    expected = gated_input(x, gate) + 0.0
    rows = torch.arange(n, device="cuda")
    for start in range(0, k, n):
        w = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
        w[rows, start + rows] = 1.0
        fused = Linear(provider="cute", config=config)(x, w, gate=gate)
        assert_bits(fused, expected[:, start : start + n])
