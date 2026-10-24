# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Support-range fixtures with exact declared workspaces and independent FP32 math."""

import torch

from ctikernels.attention.inference import FlashInferPagedAttention

from ._attention import paged_metadata, reference_attention


def paged_support_case(
    *,
    mode,
    graph,
    heads,
    batch,
    page,
    layout="ragged",
    fp8=False,
    options=None,
    custom=False,
    fused=False,
    scales=None,
):
    options = options or {}
    qh, kh, dim = heads
    lengths = (
        (1,) * batch
        if mode == "decode"
        else tuple(
            (512 if layout == "long" else 64) if i == 0 else (1, 7, 31)[i - 1] for i in range(batch)
        )
    )
    capacity = sum(lengths)
    generator = torch.Generator().manual_seed(20261016)
    query_cpu = torch.randn(capacity, qh, dim, dtype=torch.bfloat16, generator=generator)
    if options.get("logits_soft_cap"):
        query_cpu.mul_(8)
    max_kv = 4096 if layout == "long" else 1024
    kv_dtype = torch.float8_e4m3fn if fp8 else query_cpu.dtype
    shape = (2 * batch * ((max_kv + page - 1) // page) + 1, page, kh, dim)
    cache_cpu = tuple(torch.randn(shape, generator=generator).to(kv_dtype) for _ in range(2))
    if scales is None:
        scales = {"k_scale": 0.75, "v_scale": 1.25} if fp8 else {}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        query = query_cpu.cuda()
        output = torch.empty_like(query)
        if fused:
            # Fused-QKV view: query rows strided by (Hq + 2 Hkv) * D.
            qkv = torch.randn(capacity, qh + 2 * kh, dim, dtype=query.dtype, device="cuda")
            qkv[:, :qh].copy_(query)
            query = qkv[:, :qh]
        cache = tuple(value.cuda() for value in cache_cpu)
        sizes = FlashInferPagedAttention.workspace_size(
            mode=mode,
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=graph,
            max_batch_size=batch,
            max_query_tokens=capacity,
            num_qo_heads=qh,
            num_kv_heads=kh,
            head_dim=dim,
            device=query.device,
        )
        workspace = (
            torch.zeros(sizes[0], dtype=torch.uint8, device="cuda"),
            torch.empty(sizes[1], dtype=torch.uint8, device="cuda"),
        )
        buffers = {}
        if graph:
            buffers = {
                "paged_kv_indptr_buffer": torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
                "paged_kv_indices_buffer": torch.empty(
                    batch * ((max_kv + page - 1) // page), dtype=torch.int32, device="cuda"
                ),
                "paged_kv_last_page_len_buffer": torch.empty(
                    batch, dtype=torch.int32, device="cuda"
                ),
            }
            if mode == "prefill":
                buffers["qo_indptr_buffer"] = torch.empty(
                    batch + 1, dtype=torch.int32, device="cuda"
                )
                if custom:
                    buffers["custom_mask_buffer"] = torch.empty(
                        (capacity * max_kv + 7) // 8 + batch, dtype=torch.uint8, device="cuda"
                    )
                    buffers["mask_indptr_buffer"] = torch.empty(
                        batch + 1, dtype=torch.int32, device="cuda"
                    )
        operation = FlashInferPagedAttention(
            *workspace,
            mode=mode,
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=graph,
            **buffers,
        )
        eager_sizes = FlashInferPagedAttention.workspace_size(
            mode=mode,
            backend="fa2",
            use_tensor_cores=True,
            max_batch_size=batch,
            max_query_tokens=capacity,
            num_qo_heads=qh,
            num_kv_heads=kh,
            head_dim=dim,
            device=query.device,
            dispatch="provider",
        )
        eager_workspace = (
            torch.zeros(eager_sizes[0], dtype=torch.uint8, device="cuda"),
            torch.empty(eager_sizes[1], dtype=torch.uint8, device="cuda"),
        )
        eager = FlashInferPagedAttention(
            *eager_workspace, mode=mode, backend="fa2", use_tensor_cores=True, dispatch="provider"
        )
    captured = None
    pointers = [
        value.data_ptr() for value in (query, *cache, output, *workspace, *buffers.values())
    ]
    for turn in range(3 if graph else 1):
        q_lengths = lengths if turn != 1 else tuple(max(1, n // 2) for n in reversed(lengths))
        kv_lengths = tuple(max(q_lengths[i], max_kv - turn * 137 - i * 17) for i in range(batch))
        metadata = paged_metadata(q_lengths, kv_lengths, page)
        mask = None
        if custom:
            masks = []
            for qlen, kvlen in zip(q_lengths, kv_lengths, strict=True):
                pos = torch.arange(qlen)[:, None] + kvlen - qlen
                keys = torch.arange(kvlen)[None, :]
                masks.append(((keys <= pos) & ((keys % 3 != 1) | (keys == pos))).flatten())
            mask = torch.cat(masks)
        expected = reference_attention(
            query_cpu[: sum(q_lengths)],
            cache_cpu,
            q_lengths,
            kv_lengths,
            metadata["indices"],
            custom_mask=mask,
            **options,
            **scales,
        )
        with torch.cuda.stream(stream), torch.inference_mode():
            device_indices = metadata["indices"].cuda()
            plan_options = dict(
                **(metadata | {"indices": device_indices}),
                num_qo_heads=qh,
                num_kv_heads=kh,
                head_dim=dim,
                page_size=page,
                pos_encoding_mode="NONE",
                dtype=query.dtype,
                kv_dtype=kv_dtype,
                custom_mask=mask.cuda() if mask is not None else None,
                **options,
            )
            operation.plan(**plan_options)
            eager.plan(**plan_options)
            eager_output = eager.run(q=query[: sum(q_lengths)], paged_kv_cache=cache, **scales)
            if captured is None:
                operation.run(q=query, paged_kv_cache=cache, out=output, **scales)
                stream.synchronize()
                if graph:
                    captured = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured, stream=stream):
                        operation.run(q=query, paged_kv_cache=cache, out=output, **scales)
            if graph:
                output.fill_(torch.nan)
                captured.replay()
        stream.synchronize()
        actual = output[: sum(q_lengths)].cpu().float()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(actual, eager_output.cpu().float(), rtol=0.01, atol=0.01)
        assert pointers == [
            value.data_ptr() for value in (query, *cache, output, *workspace, *buffers.values())
        ]
