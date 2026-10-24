# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from itertools import accumulate, pairwise

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# The varlen training attention runs on flash-attn, an optional dependency.
pytest.importorskip("flash_attn.flash_attn_interface")

from ctikernels.attention.varlen import CausalVarlenAttention, NoncausalVarlenAttention

from ._exhaustive import ci_cases

# Packed layouts: one long image, mixed lengths off the tile boundaries, empty
# sequences in the middle and at the end, and one long plus one short image.
_NONCAUSAL_LAYOUTS = (
    pytest.param((0, 1024), id="single"),
    pytest.param((0, 1, 18, 81, 337, 1106), id="mixed"),
    pytest.param((0, 64, 64, 193, 193), id="empty"),
    pytest.param((0, 2304, 2560), id="long"),
)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    boundaries: tuple[int, ...],
    *,
    causal: bool,
) -> torch.Tensor:
    """FP32 math-backend SDPA per sequence; empty sequences contribute no rows."""
    group_size = q.shape[1] // k.shape[1]
    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for start, end in pairwise(boundaries):
            if start == end:
                continue
            query = q[start:end].float().transpose(0, 1)
            key = k[start:end].float().transpose(0, 1).repeat_interleave(group_size, dim=0)
            value = v[start:end].float().transpose(0, 1).repeat_interleave(group_size, dim=0)
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
            outputs.append(attended.transpose(0, 1))
    return torch.cat(outputs)


def test_forward_backward_matches_fp32_reference() -> None:
    assert torch.cuda.is_available(), "GPU suite requires CUDA"

    torch.manual_seed(20260815)
    operation = CausalVarlenAttention()
    stream = torch.cuda.Stream()
    boundaries = (0, 1, 18, 81, 1106)
    with torch.cuda.stream(stream):
        q = torch.randn(
            (1106, 16, 128),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            (1106, 8, 128),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn(
            (1106, 8, 128),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        output = operation.forward(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=1025,
        )
        output_gradient = torch.randn_like(output, dtype=torch.float32)
        output.float().backward(output_gradient)
        changed_v = v.detach().clone()
        changed_v[1:18].add_(1.0)
        changed_output = operation.forward(
            q.detach(),
            k.detach(),
            changed_v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=1025,
        )
        completion = torch.cuda.Event()
        completion.record(stream)

    completion.synchronize()
    reference_q = q.detach().float().requires_grad_()
    reference_k = k.detach().float().requires_grad_()
    reference_v = v.detach().float().requires_grad_()
    reference_output = _reference_attention(
        reference_q,
        reference_k,
        reference_v,
        boundaries,
        causal=True,
    )
    reference_output.backward(output_gradient)

    assert output.shape == q.shape
    torch.testing.assert_close(output.float(), reference_output, rtol=0.02, atol=0.02)
    for tensor, reference in (
        (q, reference_q),
        (k, reference_k),
        (v, reference_v),
    ):
        assert tensor.grad is not None and reference.grad is not None
        torch.testing.assert_close(tensor.grad.float(), reference.grad, rtol=0.03, atol=0.04)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if index == 1:
            assert not torch.equal(changed_output[start:end], output[start:end])
        else:
            torch.testing.assert_close(
                changed_output[start:end],
                output[start:end],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("boundaries", _NONCAUSAL_LAYOUTS)
@pytest.mark.parametrize("head_dim", (64, 72, 128))
def test_noncausal_forward_matches_fp32_sdpa_on_current_stream(
    head_dim: int,
    boundaries: tuple[int, ...],
) -> None:
    torch.manual_seed(20260930)
    operation = NoncausalVarlenAttention()
    num_tokens = boundaries[-1]
    max_sequence_length = max(end - start for start, end in pairwise(boundaries))
    changed = next(index for index, (start, end) in enumerate(pairwise(boundaries)) if end > start)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Vision blocks slice q, k and v from one merged projection without copies.
        qkv = torch.randn((num_tokens, 3, 16, head_dim), device="cuda", dtype=torch.bfloat16)
        original = qkv.clone()
        q, k, v = qkv.unbind(1)
        cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        output = operation.forward(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        changed_v = v.clone()
        changed_v[boundaries[changed] : boundaries[changed + 1]].add_(1.0)
        changed_output = operation.forward(
            q,
            k,
            changed_v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        completion = torch.cuda.Event()
        completion.record(stream)

    completion.synchronize()
    assert output.shape == q.shape and output.dtype == torch.bfloat16
    assert torch.equal(qkv, original)
    torch.testing.assert_close(
        output.float(),
        _reference_attention(q, k, v, boundaries, causal=False),
        rtol=0.02,
        atol=0.02,
    )
    for index, (start, end) in enumerate(pairwise(boundaries)):
        if index == changed:
            assert not torch.equal(changed_output[start:end], output[start:end])
        else:
            assert torch.equal(changed_output[start:end], output[start:end])


def test_noncausal_grouped_query_forward_backward_matches_fp32_sdpa() -> None:
    torch.manual_seed(20261001)
    operation = NoncausalVarlenAttention()
    boundaries = (0, 1, 18, 18, 81, 1106)
    q = torch.randn((1106, 16, 72), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1106, 4, 72), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((1106, 4, 72), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = operation.forward(
        q,
        k,
        v,
        cu_seqlens=torch.tensor(boundaries, device="cuda", dtype=torch.int32),
        max_sequence_length=1025,
    )
    output_gradient = torch.randn_like(output, dtype=torch.float32)
    output.float().backward(output_gradient)

    reference_q = q.detach().float().requires_grad_()
    reference_k = k.detach().float().requires_grad_()
    reference_v = v.detach().float().requires_grad_()
    reference_output = _reference_attention(
        reference_q, reference_k, reference_v, boundaries, causal=False
    )
    reference_output.backward(output_gradient)

    torch.testing.assert_close(output.float(), reference_output, rtol=0.02, atol=0.02)
    for tensor, reference in ((q, reference_q), (k, reference_k), (v, reference_v)):
        assert tensor.grad is not None and reference.grad is not None
        torch.testing.assert_close(tensor.grad.float(), reference.grad, rtol=0.03, atol=0.04)


@pytest.mark.parametrize(
    ("operation_type", "causal"),
    ((CausalVarlenAttention, True), (NoncausalVarlenAttention, False)),
    ids=("causal", "noncausal"),
)
def test_graph_replays_changed_inputs_and_boundaries(operation_type: type, causal: bool) -> None:
    torch.manual_seed(20261002)
    operation = operation_type()
    q, k, v = torch.randn((1106, 3, 16, 72), device="cuda", dtype=torch.bfloat16).unbind(1)
    cu_seqlens = torch.tensor((0, 1, 18, 81, 1106), device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        operation.forward(q, k, v, cu_seqlens=cu_seqlens, max_sequence_length=1025)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = operation.forward(q, k, v, cu_seqlens=cu_seqlens, max_sequence_length=1025)

    # The sequence count and total stay fixed; lengths move but never exceed 1025.
    for boundaries in ((0, 1025, 1050, 1105, 1106), (0, 300, 300, 806, 1106)):
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor))
        cu_seqlens.copy_(torch.tensor(boundaries, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            output.float(),
            _reference_attention(q, k, v, boundaries, causal=causal),
            rtol=0.02,
            atol=0.02,
        )


@pytest.mark.parametrize(
    "query_tokens,kv_tokens,dtype,causal,scale",
    ci_cases(
        [(16, 1024), (1, 512), (32, 17)],
        [torch.float16, torch.bfloat16],
        [False, True],
        [None, 0.07],
        core=[
            (16, 1024, dtype, causal, 0.07)
            for dtype in (torch.float16, torch.bfloat16)
            for causal in (False, True)
        ],
    ),
)
def test_asymmetric_varlen_forward_backward(query_tokens, kv_tokens, dtype, causal, scale):
    torch.manual_seed(20261016)
    q_lengths = (1, query_tokens - 1) if query_tokens > 1 else (1,)
    k_lengths = (kv_tokens // 2, kv_tokens - kv_tokens // 2) if query_tokens > 1 else (kv_tokens,)
    q = torch.randn(query_tokens, 8, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(kv_tokens, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    operation = CausalVarlenAttention() if causal else NoncausalVarlenAttention()
    cq = torch.tensor([0, *accumulate(q_lengths)], dtype=torch.int32, device="cuda")
    ck = torch.tensor([0, *accumulate(k_lengths)], dtype=torch.int32, device="cuda")
    output = operation.forward(
        q,
        k,
        v,
        cu_seqlens=cq,
        max_sequence_length=max(q_lengths),
        cu_seqlens_k=ck,
        max_sequence_length_k=max(k_lengths),
        softmax_scale=scale,
    )
    grad = torch.randn_like(output, dtype=torch.float32)
    output.float().backward(grad)
    rq, rk, rv = [t.detach().float().requires_grad_() for t in (q, k, v)]
    pieces = []
    q_start = k_start = 0
    with sdpa_kernel(SDPBackend.MATH):
        for nq, nk in zip(q_lengths, k_lengths, strict=True):
            query = rq[q_start : q_start + nq].transpose(0, 1)
            key = rk[k_start : k_start + nk].transpose(0, 1).repeat_interleave(4, dim=0)
            value = rv[k_start : k_start + nk].transpose(0, 1).repeat_interleave(4, dim=0)
            mask = None
            if causal:
                pos = torch.arange(nq, device="cuda")[:, None] + nk - nq
                mask = torch.arange(nk, device="cuda")[None, :] <= pos
            pieces.append(
                F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, scale=scale
                ).transpose(0, 1)
            )
            q_start += nq
            k_start += nk
    expected = torch.cat(pieces)
    expected.backward(grad)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)
    for actual, reference in zip((q, k, v), (rq, rk, rv), strict=True):
        torch.testing.assert_close(actual.grad.float(), reference.grad, rtol=0.03, atol=0.04)
