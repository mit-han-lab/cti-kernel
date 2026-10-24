# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""K-Q2: Qwen3-30B-A3B routed experts, independent routing and BF16 floor gates."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from ctikernels.moe import Experts, route  # noqa: E402
from ctikernels.moe._block import hardware  # noqa: E402
from ctikernels.moe._configs import TUNED, lookup  # noqa: E402
from ctikernels.moe._routing_check import classify_topk  # noqa: E402

from ..modules import _moe_reference as ref  # noqa: E402
from ._moe_helpers import _check_align  # noqa: E402

H, E, INTER, K = 2048, 128, 768, 8


def _mapping(ep, rank, device="cuda"):
    local = E // ep
    result = torch.full((E,), -1, device=device, dtype=torch.int32)
    result[rank * local : (rank + 1) * local] = torch.arange(
        local, device=device, dtype=torch.int32
    )
    return result


@pytest.fixture(scope="module")
def weights():
    for module in ("cutlass", "triton"):
        pytest.importorskip(module)
    gen = torch.Generator(device="cuda").manual_seed(23)
    opts = {"device": "cuda", "dtype": torch.bfloat16, "generator": gen}
    result = tuple(
        torch.randn(shape, **opts).mul_(0.02)
        for shape in ((E, H), (E, 2 * INTER, H), (E, H, INTER))
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 5, 8, 9, 64, 512, 513, 4096])
@pytest.mark.parametrize("ep", [1, 2, 4])
def test_qwen3_alignment(rows, ep):
    pytest.importorskip("cutlass")
    gen = torch.Generator(device="cuda").manual_seed(41 + rows)
    ids = torch.randint(E, (rows, K), generator=gen, device="cuda", dtype=torch.int32)
    if rows:
        ids[-1] = -1
    if rows > 1:
        ids[0, 0] = -1
    for rank in range(ep):
        _check_align(ids, _mapping(ep, rank), E // ep)
    # Permute both ownership and local slots; the last global ID may be local,
    # so accidentally indexing mapping[-1] cannot hide a padding error.
    order = torch.cat(
        (torch.tensor([E - 1], device="cuda"), torch.randperm(E - 1, generator=gen, device="cuda"))
    )
    mapping = torch.full((E,), -1, device="cuda", dtype=torch.int32)
    mapping[order[: E // ep]] = torch.arange(E // ep, device="cuda", dtype=torch.int32)
    _check_align(ids, mapping, E // ep)


def _route(x, router, norm, record_property):
    scales, ids = route(x, router, K, renormalize=norm)
    ties, mismatches = classify_topk(x, router, ids)
    record_property("route_ties", ties)
    record_property("route_mismatches", mismatches)
    assert mismatches == 0, (ties, mismatches)
    expected = F.linear(x, router).softmax(-1, dtype=torch.float32).gather(1, ids.long())
    if norm:
        expected /= expected.sum(-1, keepdim=True)
    assert scales.dtype == torch.float32 and ids.dtype == torch.int32
    # route's logits round once to BF16 like F.linear's, but may sum in another order.
    torch.testing.assert_close(scales, expected, rtol=2e-2, atol=1e-4)
    return scales, ids


def _experts_reference(x, up, down, scales, ids, mapping, bf16=False):
    """Independent expert loop, FP32 oracle or HF eager rounding points.

    transformers 5.16.1, models/qwen3_moe/modeling_qwen3_moe.py,
    Qwen3MoeExperts.forward: ascending hit experts, BF16 gate/up linear,
    BF16 SiLU/product, BF16 down linear, BF16 route multiply and index_add_.
    https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
    """
    dtype = torch.bfloat16 if bf16 else torch.float32
    out = torch.zeros_like(x, dtype=dtype)
    route_cpu, map_cpu = ids.cpu(), mapping.cpu()
    for expert in route_cpu.unique(sorted=True).tolist():
        if expert < 0 or int(map_cpu[expert]) < 0:
            continue
        local = int(map_cpu[expert])
        # HF's expert mask is [expert, slot, token], hence slot-major gather.
        slots, rows = torch.where(route_cpu.T == expert)
        rows, slots = rows.to(x.device), slots.to(x.device)
        gate, value = F.linear(x[rows].to(dtype), up[local].to(dtype)).chunk(2, -1)
        projected = F.linear(F.silu(gate) * value, down[local].to(dtype))
        out.index_add_(0, rows, projected * scales[rows, slots, None].to(dtype))
    return out


def _hf_scales(x, router, ids, norm):
    # Qwen3MoeTopKRouter in the same HF source: full-batch BF16 linear,
    # FP32 softmax/top-k, optional normalization, then cast to logits dtype.
    # Condition on the independently verified selection to allow boundary ties.
    logits = F.linear(x, router)
    scales = logits.softmax(-1, dtype=torch.float32).gather(1, ids.long())
    if norm:
        scales /= scales.sum(-1, keepdim=True)
    return scales.to(logits.dtype)


def _vllm_scales(x, router, ids, norm):
    top = F.linear(x, router).float().softmax(-1).gather(1, ids.long())
    return top / top.sum(-1, keepdim=True) if norm else top


def _assert_floor(errors):
    # The block floor gate's rule (test_moe_block._assert_floor): floor = max(|HF|, |vLLM|) errors.
    actual = torch.cat([a.flatten() for a, _ in errors])
    floor = torch.cat([f.flatten() for _, f in errors])
    assert torch.isfinite(actual).all() and torch.isfinite(floor).all()
    assert actual.max() <= 1.25 * floor.max(), ("max", actual.max(), floor.max())
    assert actual.square().mean() <= floor.square().mean(), (
        "RMS",
        actual.square().mean().sqrt(),
        floor.square().mean().sqrt(),
    )


@pytest.mark.parametrize("rows", [1, 4, 64, 513])
@pytest.mark.parametrize("ep", [1, 2, 4])
@pytest.mark.parametrize("norm", [True, False])
def test_qwen3_accuracy_and_ep_sum(weights, rows, ep, norm, record_property):
    _accuracy_and_ep_sum(weights, rows, ep, norm, record_property, seeds=4)


# The tuned prefill tiles (M=1024-16384), capped at two seeds per row.
@pytest.mark.parametrize("rows", [1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("ep", [1, 2, 4])
@pytest.mark.parametrize("norm", [True, False])
def test_qwen3_prefill_accuracy_and_ep_sum(weights, rows, ep, norm, record_property):
    _accuracy_and_ep_sum(weights, rows, ep, norm, record_property, seeds=2)


def _floor(hf, vllm, oracle):
    return torch.maximum((hf.float() - oracle).abs(), (vllm.float() - oracle).abs())


def _accuracy_and_ep_sum(weights, rows, ep, norm, record_property, seeds):
    router, up, down = weights
    errors = [[] for _ in range(ep)]
    full_errors = []
    for seed in range(seeds):
        gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
        x = torch.randn((rows, H), device="cuda", dtype=torch.bfloat16, generator=gen)
        scales, ids = _route(x, router, norm, record_property)
        hf_scales = _hf_scales(x, router, ids, norm)
        whole = Experts()(x, up, down, scales, ids)
        full_map = _mapping(1, 0)
        oracle = _experts_reference(x, up, down, scales, ids, full_map)
        hf = _experts_reference(x, up, down, hf_scales, ids, full_map, bf16=True)
        vllm_scales = _vllm_scales(x, router, ids, norm)
        vllm = ref.vllm_infer_reference(x, up, down, vllm_scales, ids, expert_map=full_map)
        full_errors.append(((whole.float() - oracle).abs(), _floor(hf, vllm, oracle)))
        total = torch.zeros_like(x, dtype=torch.float32)
        scale = float(whole.float().abs().max())
        for rank in range(ep):
            local = E // ep
            sl = slice(rank * local, (rank + 1) * local)
            mapping = _mapping(ep, rank)
            actual = Experts()(x, up[sl], down[sl], scales, ids, expert_map=mapping)
            oracle = _experts_reference(x, up[sl], down[sl], scales, ids, mapping)
            hf = _experts_reference(x, up[sl], down[sl], hf_scales, ids, mapping, bf16=True)
            vllm = ref.vllm_infer_reference(
                x, up[sl], down[sl], vllm_scales, ids, expert_map=mapping
            )
            errors[rank].append(((actual.float() - oracle).abs(), _floor(hf, vllm, oracle)))
            total += actual.float()
            scale = max(scale, float(actual.float().abs().max()))
        # BF16 partials and EP1 each round once; allow half an ulp per term.
        assert (total - whole.float()).abs().max() <= (ep + 1) * 2.0**-8 * scale
    _assert_floor(full_errors)
    for samples in errors:
        _assert_floor(samples)


@pytest.mark.parametrize("norm", [True, False])
def test_qwen3_padding_remote_and_valid_tokens(weights, norm):
    _, up, down = weights
    errors = []
    for seed in range(4):
        gen = torch.Generator(device="cuda").manual_seed(3000 + seed)
        x = torch.randn((5, H), device="cuda", dtype=torch.bfloat16, generator=gen)
        ids = torch.tensor(
            [
                [-1] * K,
                list(range(64, 72)),
                [0, 65, -1, 3, 70, 5, 6, 7],
                list(range(K)),
                list(range(K)),
            ],
            device="cuda",
            dtype=torch.int32,
        )
        scales = (
            torch.randn((5, E), device="cuda", generator=gen).softmax(-1).topk(K, dim=-1).values
        )
        if norm:
            scales /= scales.sum(-1, keepdim=True)
        mapping = _mapping(4, 0)
        op = Experts()
        actual = op(x, up[:32], down[:32], scales, ids, expert_map=mapping)
        assert torch.count_nonzero(actual[:2].view(torch.int16)) == 0
        oracle = _experts_reference(x, up[:32], down[:32], scales, ids, mapping)
        hf = _experts_reference(x, up[:32], down[:32], scales, ids, mapping, bf16=True)
        errors.append(((actual.float() - oracle).abs(), (hf.float() - oracle).abs()))
        valid = torch.full((1,), 3, device="cuda", dtype=torch.int32)
        padded = op(x, up[:32], down[:32], scales, ids, expert_map=mapping, valid_tokens=valid)
        assert torch.equal(padded[:3], actual[:3])
        assert torch.count_nonzero(padded[3:].view(torch.int16)) == 0
        valid.zero_()
        assert (
            torch.count_nonzero(
                op(x, up[:32], down[:32], scales, ids, expert_map=mapping, valid_tokens=valid).view(
                    torch.int16
                )
            )
            == 0
        )
        assert op(x[:0], up[:32], down[:32], scales[:0], ids[:0], expert_map=mapping).shape == (
            0,
            H,
        )
    _assert_floor(errors)


@pytest.mark.parametrize("rows", [1, 4, 64, 512])
@pytest.mark.parametrize("ep", [1, 2, 4])
def test_qwen3_graph_replays_new_inputs(weights, rows, ep, record_property):
    router, up, down = weights
    x = torch.randn((rows, H), device="cuda", dtype=torch.bfloat16)
    scales, ids = _route(x, router, True, record_property)
    local, rank = E // ep, ep - 1
    sl = slice(rank * local, (rank + 1) * local)
    mapping = _mapping(ep, rank)
    op = Experts()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(x, up[sl], down[sl], scales, ids, expert_map=mapping)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = op(x, up[sl], down[sl], scales, ids, expert_map=mapping)
    for norm in (False, True):
        new_x = torch.randn_like(x)
        new_scales, new_ids = _route(new_x, router, norm, record_property)
        x.copy_(new_x)
        scales.copy_(new_scales)
        ids.copy_(new_ids)
        if rows > 1:
            ids[-1] = -1
        graph.replay()
        saved = captured.clone()
        eager = op(x, up[sl], down[sl], scales, ids, expert_map=mapping)
        assert torch.equal(saved, eager)
        if rows > 1:
            assert torch.count_nonzero(saved[-1].view(torch.int16)) == 0


@pytest.mark.parametrize("local", [128, 64, 32])
def test_qwen3_launch_tables(weights, local):
    router, up, down = weights
    kind = hardware(up.device)
    if (local, INTER) not in TUNED.get(kind, {}):
        pytest.skip("Qwen3 tables have not been installed for this device/shape")
    op = Experts()
    for rows in (1, 4, 64, 2048):
        x = torch.randn((rows, H), device="cuda", dtype=torch.bfloat16)
        scales, ids = route(x, router, K)
        op(x, up[:local], down[:local], scales, ids, expert_map=_mapping(E // local, 0))
        assert op.launch_config == {
            phase: lookup(kind, local, INTER, rows, phase) for phase in ("up", "down")
        }
