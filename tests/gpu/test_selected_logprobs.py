# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch
import torch.nn.functional as F

from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(difference) / denominator)


@pytest.mark.parametrize(
    ("dtype", "use_bias", "logit_scale"),
    (
        (torch.bfloat16, True, 1.0),
        (torch.bfloat16, False, 32.0),
        (torch.float16, True, 1.0),
        (torch.float16, False, 8.0),
        (torch.float32, True, 1.0),
        (torch.float32, False, 32.0),
    ),
)
def test_selected_logprobs_match_same_projection_oracle(
    dtype: torch.dtype,
    use_bias: bool,
    logit_scale: float,
) -> None:
    """Separate operation correctness from low-precision projection error."""

    torch.manual_seed(20260909)
    hidden_data = torch.randn((11, 32), device="cuda", dtype=dtype)
    weight = torch.randn((97, 32), device="cuda", dtype=dtype) * logit_scale
    bias = torch.randn((97,), device="cuda", dtype=dtype) if use_bias else None
    target_ids = torch.randint(0, 97, (11,), device="cuda", dtype=torch.int64)
    upstream = torch.randn((11,), device="cuda", dtype=torch.float32)
    original_upstream = upstream.clone()
    temperature = 0.7

    reference_hidden = hidden_data.clone().requires_grad_()
    reference_logits = F.linear(reference_hidden, weight, bias).float() / temperature
    reference = F.log_softmax(reference_logits, dim=-1).gather(1, target_ids[:, None])[:, 0]
    reference.backward(upstream)

    hidden_states = hidden_data.clone().requires_grad_()
    actual = ChunkedLinearSelectedTokenLogprobs(chunk_size=4).forward(
        hidden_states,
        weight,
        target_ids,
        bias=bias,
        temperature=temperature,
    )
    actual.backward(upstream)

    assert actual.dtype == torch.float32 and actual.shape == target_ids.shape
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    assert hidden_states.grad is not None and reference_hidden.grad is not None
    if dtype == torch.float32:
        torch.testing.assert_close(hidden_states.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
    else:
        assert _relative_l2(hidden_states.grad, reference_hidden.grad) <= 4e-3
    torch.testing.assert_close(hidden_states, hidden_data, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


@pytest.mark.parametrize("temperature", (0.7, 1.0))
@pytest.mark.parametrize("logit_scale", (1.0, 32.0), ids=("ordinary", "extreme"))
def test_selected_logprobs_match_fp32_hidden_gradient(
    temperature: float,
    logit_scale: float,
) -> None:
    """Bound BF16 projection error separately from the same-projection gate."""

    torch.manual_seed(20260722)
    hidden_data = torch.randn((11, 32), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((97, 32), device="cuda", dtype=torch.bfloat16) * logit_scale
    bias = torch.randn((97,), device="cuda", dtype=torch.bfloat16) * logit_scale
    target_ids = torch.randint(0, 97, (11,), device="cuda", dtype=torch.int64)
    upstream = torch.randn((11,), device="cuda", dtype=torch.float32)

    reference_hidden = hidden_data.float().detach().requires_grad_()
    reference_logits = F.linear(reference_hidden, weight.float(), bias.float()) / temperature
    reference = (
        torch.log_softmax(reference_logits, dim=-1)
        .gather(
            1,
            target_ids[:, None],
        )
        .squeeze(1)
    )
    (reference * upstream).sum().backward()

    hidden_states = hidden_data.detach().clone().requires_grad_()
    actual = ChunkedLinearSelectedTokenLogprobs(chunk_size=4).forward(
        hidden_states,
        weight,
        target_ids,
        bias=bias,
        temperature=temperature,
    )
    (actual * upstream).sum().backward()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all())
    assert hidden_states.grad is not None and reference_hidden.grad is not None
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=1.0)
    assert _relative_l2(hidden_states.grad, reference_hidden.grad) <= 3e-2
    torch.testing.assert_close(hidden_states, hidden_data, rtol=0, atol=0)


def test_selected_logprobs_reject_trainable_vocabulary_head() -> None:
    hidden_states = torch.randn((2, 4), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        (8, 4),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target_ids = torch.tensor([1, 2], device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="frozen vocabulary head"):
        ChunkedLinearSelectedTokenLogprobs().forward(hidden_states, weight, target_ids)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_selected_logprobs_accept_empty_token_batch(dtype: torch.dtype) -> None:
    hidden_states = torch.empty((0, 4), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((8, 4), device="cuda", dtype=dtype)
    target_ids = torch.empty((0,), device="cuda", dtype=torch.int64)

    actual = ChunkedLinearSelectedTokenLogprobs().forward(hidden_states, weight, target_ids)
    actual.sum().backward()

    assert actual.shape == (0,) and actual.dtype == torch.float32
    assert hidden_states.grad is not None and hidden_states.grad.shape == (0, 4)


def _same_projection_fp64(hidden, weight, targets, bias, temperature, chunk_size=64):
    results = []
    for start in range(0, hidden.shape[0], chunk_size):
        logits = F.linear(hidden[start : start + chunk_size], weight, bias)
        # Scaling is explicitly FP32; FP64 isolates the normalization error.
        scaled = (logits.float() / temperature).double()
        peak = scaled.amax(-1)
        total = (scaled - peak[:, None]).exp().sum(-1)
        selected = scaled.gather(1, targets[start : start + chunk_size, None])[:, 0]
        results.append((selected - peak - total.log()).float())
    return torch.cat(results)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("vocab", (97, 4099, 151936, 248320))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 8, 64, 65))
@pytest.mark.parametrize("temperature", (0.7, 1.0, 1e-3, 1e4))
def test_fused_selected_fp64_matrix(dtype, vocab, rows, temperature):
    torch.manual_seed(517)
    hidden = torch.randn(rows, 16, device="cuda", dtype=dtype) * 0.1
    weight = torch.randn(vocab, 16, device="cuda", dtype=dtype)
    bias = torch.randn(vocab, device="cuda", dtype=dtype) * 0.1
    op = ChunkedLinearSelectedTokenLogprobs()
    # Include both endpoints even for the one-row case, and strided target ids.
    target_storage = torch.zeros(rows * 2, device="cuda", dtype=torch.int64)
    targets = target_storage[::2]
    for endpoint in (0, vocab - 1):
        targets.fill_(endpoint)
        actual = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
        assert op.provider == "cute"
        expected = _same_projection_fp64(hidden, weight, targets, bias, temperature)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("vocab", (1, 97, 4099, 151936, 248320))
def test_fused_masks_and_nonfinite_logits(dtype, vocab):
    from ctikernels.logprob.cute import Provider

    # Offset base and nontrivial row stride exercise scalar edges + aligned core.
    storage = torch.randn(8, vocab + 7, device="cuda", dtype=dtype)
    logits = storage[:, 1 : vocab + 1]
    logits[0, ::2] = -float("inf")
    logits[1].fill_(-float("inf"))
    logits[2, -1] = float("inf")
    logits[3, 0] = float("nan")
    logits[4].fill_(-float("inf"))
    logits[4, -1] = 2
    logits[5].fill_(0)
    logits[6].fill_(100)
    logits[7].fill_(-100)
    targets = torch.tensor([0, vocab - 1, 0, 0, vocab - 1, 0, vocab - 1, 0], device="cuda")
    provider = Provider(logits.device)
    output = torch.empty(8, device="cuda", dtype=torch.float32)
    for temperature in (0.7, 1.0, 1e-3, 1e4):
        provider(logits, targets, output, temperature)
        scaled = (logits.float() / temperature).double()
        reference = scaled.gather(1, targets[:, None])[:, 0] - scaled.logsumexp(-1)
        torch.testing.assert_close(output, reference.float(), rtol=2e-6, atol=1e-5, equal_nan=True)
    assert len(provider._compiled) == 1  # Temperature is a runtime scalar.


@pytest.mark.parametrize("vocab", (2, 4099, 151936))
@pytest.mark.parametrize("temperature", (1e-3, 0.7, 3.0, 1e-30, 1e30))
def test_fused_scaling_matches_cuda_scalar_division(vocab, temperature):
    """Near-equal large logits at small T expose 1/T multiply vs true division."""
    from ctikernels.logprob.cute import Provider

    generator = torch.Generator(device="cuda").manual_seed(vocab)
    logits = 100 + torch.randn(16, vocab, device="cuda", generator=generator) * 1e-5
    logits[0, :2] = torch.tensor([100.00016021728516, 100.00016784667969], device="cuda")
    logits[1] = 1e4 + torch.arange(vocab, device="cuda") * 2**-10
    logits[2] = -3e4 - torch.arange(vocab, device="cuda") * 2**-9
    targets = torch.randint(0, vocab, (16,), device="cuda", generator=generator)
    targets[:3] = 0
    output = torch.empty(16, device="cuda", dtype=torch.float32)
    Provider(logits.device)(logits, targets, output, temperature)
    # Normalize PyTorch CUDA's own scaled logits in FP64, shifted by the row max
    # first: at |x/T| ~ 1e34, x - logsumexp(x) would absorb log(2) even in FP64.
    scaled = (logits / temperature).double()
    scaled = scaled - scaled.amax(-1, keepdim=True)
    reference = scaled.gather(1, targets[:, None])[:, 0] - scaled.logsumexp(-1)
    torch.testing.assert_close(output, reference.float(), rtol=2e-6, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("rows", (1, 65))
def test_fused_graph_replays_reset_counters(dtype, rows):
    torch.manual_seed(518)
    hidden = torch.randn(rows, 16, device="cuda", dtype=dtype) * 0.1
    weight = torch.randn(248320, 16, device="cuda", dtype=dtype)
    targets = torch.arange(rows, device="cuda", dtype=torch.int64)
    op = ChunkedLinearSelectedTokenLogprobs()
    eager = op.forward(hidden, weight, targets, temperature=0.7)
    assert op.provider == "cute"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op.forward(hidden, weight, targets, temperature=0.7)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    # Changing graph inputs must not pick up stale partials or counters.
    hidden.mul_(0.5)
    expected = op.forward(hidden, weight, targets, temperature=0.7)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    provider, _ = op._providers[hidden.device]
    assert torch.count_nonzero(provider._workspace[1]).item() == 0


@pytest.mark.parametrize("ignore_index", (-100, -1, 5))
def test_fused_invalid_targets_have_safe_addresses(ignore_index):
    from ctikernels.logprob.cute import Provider

    logits = torch.randn(2, 4099, device="cuda")
    targets = torch.tensor([-(2**63), 2**63 - 1], device="cuda", dtype=torch.int64)
    output = torch.empty(2, device="cuda")
    Provider(logits.device)(logits, targets, output, 1.0, ignore_index)
    safe = torch.tensor([0, 4098], device="cuda")
    reference = logits.double().log_softmax(-1).gather(1, safe[:, None])[:, 0].float()
    torch.testing.assert_close(output, reference, rtol=2e-6, atol=1e-5)


def test_selected_torch_fallback_on_cuda(monkeypatch):
    import ctikernels.logprob.cute as provider

    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    monkeypatch.setattr(provider, "unavailable_reason", lambda: "forced torch test")
    hidden = torch.randn(65, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(97, 16, device="cuda", dtype=torch.float16)
    targets = torch.zeros(65, device="cuda", dtype=torch.int64)
    op = ChunkedLinearSelectedTokenLogprobs()
    actual = op.forward(hidden, weight, targets)
    assert op.provider == "torch"
    reference = _same_projection_fp64(hidden, weight, targets, None, 1.0)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-5)


def test_fused_graph_survives_workspace_growth():
    from ctikernels.logprob.cute import Provider

    provider = Provider(torch.device("cuda", torch.cuda.current_device()))
    logits = torch.randn(1, 4099, device="cuda", dtype=torch.bfloat16)
    targets = torch.zeros(1, device="cuda", dtype=torch.int64)
    output = torch.empty(1, device="cuda")
    provider(logits, targets, output, 0.7)
    expected = output.clone()
    old_workspace = provider._workspace
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider(logits, targets, output, 0.7)
    provider(
        torch.randn(64, 248320, device="cuda", dtype=torch.bfloat16),
        torch.zeros(64, device="cuda", dtype=torch.int64),
        torch.empty(64, device="cuda"),
        1.0,
    )
    assert provider._workspace is not old_workspace
    assert any(workspace is old_workspace for workspace in provider._retired_workspaces)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_fused_capture_at_new_temperature_with_cuda_default_device():
    """An uncached temperature must not sync the device while capturing."""
    from ctikernels.logprob.cute import _provider

    hidden = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(4099, 16, device="cuda", dtype=torch.bfloat16)
    targets = torch.arange(3, device="cuda", dtype=torch.int64)
    op = ChunkedLinearSelectedTokenLogprobs()
    op.forward(hidden, weight, targets, temperature=0.7)  # Warm the shape.
    assert op.provider == "cute"
    _provider.inverse_temperature.cache_clear()
    torch.set_default_device("cuda")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op.forward(hidden, weight, targets, temperature=0.4321)
    finally:
        torch.set_default_device(None)
    graph.replay()
    expected = op.forward(hidden, weight, targets, temperature=0.4321)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("provider_name", ("cute", "torch"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("rows", (1, 2, 65))
@pytest.mark.parametrize("vocab", (151936, 151937))
@pytest.mark.parametrize("stride", (1, 3))
def test_row_temperature_forward_and_fp64_backward(
    monkeypatch, provider_name, dtype, rows, vocab, stride
):
    import ctikernels.logprob.cute as cute_provider

    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    if provider_name == "torch":
        monkeypatch.setattr(cute_provider, "unavailable_reason", lambda: "forced torch test")
    torch.manual_seed(812)
    hidden = (torch.randn(rows, 16, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    weight = torch.randn(vocab, 16, device="cuda", dtype=dtype)
    bias = torch.randn(vocab, device="cuda", dtype=dtype) * 0.1
    targets = torch.arange(rows * 2, device="cuda", dtype=torch.int64)[::2] % vocab
    targets[-1] = vocab - 1
    if rows > 1:
        targets[1::3] = -7
    temperature = torch.linspace(0.3, 1.7, rows * stride, device="cuda")[::stride]
    upstream = torch.randn(rows * 2, device="cuda")[::2]
    valid = targets != -7
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=-7)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
        assert op.provider == provider_name
        reference = []
        for start in range(0, rows, 64):
            end = min(start + 64, rows)
            chunk = hidden[start:end]
            if end - start == 1 and provider_name == "cute":
                chunk = chunk.expand(2, -1).contiguous()
            logits = F.linear(chunk, weight, bias)[: end - start].float()
            scaled = logits / temperature[start:end, None]
            selected = scaled.gather(1, targets[start:end].clamp_min(0)[:, None])[:, 0]
            selected = selected - scaled.logsumexp(-1)
            reference.append(torch.where(valid[start:end], selected, 0.0))
        torch.testing.assert_close(actual, torch.cat(reference), rtol=2e-6, atol=1e-5)
        oracle_hidden = hidden.detach().double().requires_grad_()
        oracle_logits = F.linear(oracle_hidden, weight.double(), bias.double())
        oracle_selected = (
            (oracle_logits / temperature.double()[:, None])
            .log_softmax(-1)
            .gather(1, targets.clamp_min(0)[:, None])[:, 0]
        )
        oracle = torch.where(valid, oracle_selected, 0.0)
        dx = torch.autograd.grad(actual, hidden, upstream)[0]
        oracle_dx = torch.autograd.grad(oracle, oracle_hidden, upstream.double())[0]
        if dtype == torch.float32:
            # FP32 gradient rule: max abs error within 1e-5 of the largest reference
            # magnitude (a K=V FP32 reduction cannot hold elementwise rtol near zero).
            error = float((dx.double() - oracle_dx).abs().max())
            assert error <= 1e-5 * float(oracle_dx.abs().max())
        else:
            scalar_output = torch.empty_like(actual)
            scalar_dx = torch.empty_like(dx)
            # Keep the full batch so scalar and row temperatures use the same GEMMs.
            for value in temperature.unique().tolist():
                scalar = op.forward(hidden, weight, targets, bias=bias, temperature=float(value))
                gradient = torch.autograd.grad(scalar, hidden, upstream)[0]
                matching = temperature == value
                scalar_output[matching] = scalar.detach()[matching]
                scalar_dx[matching] = gradient[matching]
            magnitude = scalar_output.abs()
            ulp = torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude
            assert bool(((actual - scalar_output).abs() <= 2 * ulp).all())
            # Sharp softmaxes amplify low-precision projection error in both paths.
            error = (dx.double() - oracle_dx).abs().max()
            scalar_error = (scalar_dx.double() - oracle_dx).abs().max()
            assert error <= 1.1 * scalar_error
            assert _relative_l2(dx, oracle_dx) <= 1.1 * _relative_l2(scalar_dx, oracle_dx)
        assert torch.count_nonzero(actual[~valid]) == 0
        assert torch.count_nonzero(dx[~valid]) == 0
        targets.fill_(-7)
        ignored = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
        assert torch.count_nonzero(ignored) == 0
        assert torch.count_nonzero(torch.autograd.grad(ignored.sum(), hidden)[0]) == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("provider_name", ("cute", "torch"))
@pytest.mark.parametrize("rows", (1, 2, 65))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_row_temperature_graph_replays(monkeypatch, provider_name, rows, dtype):
    import ctikernels.logprob.cute as cute_provider

    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0" if provider_name == "torch" else "1")
    if provider_name == "torch":
        monkeypatch.setattr(cute_provider, "unavailable_reason", lambda: "forced torch test")
    torch.manual_seed(813)
    hidden = torch.randn(rows, 16, device="cuda", dtype=dtype) * 0.1
    weight = torch.randn(151937, 16, device="cuda", dtype=dtype)
    targets = torch.arange(rows, device="cuda", dtype=torch.int64)
    targets[1::3] = -100
    temperature = torch.linspace(0.3, 1.7, rows * 2, device="cuda")[::2]
    op = ChunkedLinearSelectedTokenLogprobs()

    expected = op.forward(hidden, weight, targets, temperature=temperature)
    assert op.provider == provider_name
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op.forward(hidden, weight, targets, temperature=temperature)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    temperature.mul_(1.3)
    expected = op.forward(hidden, weight, targets, temperature=temperature)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_cute_row_temperature_division_is_bitwise_true_division():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.cute.runtime import make_ptr

    from ctikernels.logprob.cute._kernel import divide

    @cute.kernel
    def division(values: cute.Pointer, temperatures: cute.Pointer, output: cute.Pointer):
        index, _, _ = cute.arch.thread_idx()
        output[index] = divide(Float32(values[index]), Float32(temperatures[index]))

    @cute.jit
    def launch(
        values: cute.Pointer,
        temperatures: cute.Pointer,
        output: cute.Pointer,
        stream: cuda.CUstream,
    ):
        division(values, temperatures, output).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    torch.manual_seed(814)
    values = torch.randn(128, device="cuda") * 100
    temperatures = torch.linspace(0.001, 1.7, 128, device="cuda")
    output = torch.empty_like(values)
    pointers = [
        make_ptr(Float32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        for tensor in (values, temperatures, output)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    major, minor = torch.cuda.get_device_capability()
    compiled = cute.compile(
        launch,
        *pointers,
        stream,
        options=f"--gpu-arch sm_{major}{minor} --ptxas-options=--fmad=false",
    )
    compiled(*pointers, stream)
    expected = values / temperatures
    assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
    assert not torch.equal(expected, values * temperatures.reciprocal())


@pytest.mark.parametrize("ascending", (False, True))
def test_fused_graph_workspace_sizes_once(ascending):
    from ctikernels import GraphWorkspace
    from ctikernels.logprob.cute import Provider

    provider = Provider(torch.device("cuda", torch.cuda.current_device()))
    rows = (1, 7, 65) if ascending else (65, 7, 1)
    logits = torch.randn(65, 4099, device="cuda", dtype=torch.bfloat16)
    targets = torch.arange(65, device="cuda", dtype=torch.int64)
    outputs = {m: torch.empty(m, device="cuda") for m in rows}
    graphs, eager = {}, {}
    with GraphWorkspace(max_rows=65) as group:
        for m in rows:
            provider(logits[:m], targets[:m], outputs[m], 0.7)
            eager[m] = outputs[m].clone()
        for m in rows:
            graphs[m] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[m]):
                provider(logits[:m], targets[:m], outputs[m], 0.7)
        with pytest.raises(RuntimeError, match="66 rows exceed max_rows=65"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                provider(
                    torch.cat([logits, logits[:1]]),
                    torch.cat([targets, targets[:1]]),
                    torch.empty(66, device="cuda"),
                    0.7,
                )
    assert provider._workspace is None and not provider._retired_workspaces
    for _ in range(2):
        for m in rows:
            outputs[m].zero_()
            graphs[m].replay()
            torch.testing.assert_close(outputs[m], eager[m], rtol=0, atol=0)
    group.close()
