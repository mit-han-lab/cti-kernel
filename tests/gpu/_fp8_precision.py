# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shared FP64 precision gate for native decode and fixed-query FP16 staging."""

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

from ctikernels.attention.inference import FlashInferFixedQueryAttention, FlashInferPagedAttention

from ._attention import paged_metadata, reference_attention


def staged_precision_case(
    *,
    heads,
    page,
    kv_dtype,
    query_length=1,
    decode_graph=None,
    use_tensor_cores=True,
    range_case=None,
    range_scale=None,
):
    batch, length = (3, 1727) if range_case is None else (1, 2)
    hq, hk, dim = heads
    fixed = decode_graph is None
    host = paged_metadata((query_length,) * batch, (length,) * batch, page)
    generator = torch.Generator().manual_seed(20261017)
    cpu_q = torch.randn(batch * query_length, hq, dim, dtype=torch.bfloat16, generator=generator)
    cpu_kv = tuple(
        torch.randn(2 * host["indices"].numel() + 1, page, hk, dim, generator=generator).to(
            kv_dtype
        )
        for _ in range(2)
    )
    if range_case is not None:
        assert (hq, hk, dim) == (1, 1, 64) and kv_dtype == torch.float8_e4m3fn
        cpu_q.zero_()
        cpu_q[..., 0] = 8  # Default 1/sqrt(64) scale makes QK logits exactly [0, -20].
        key = torch.zeros(cpu_kv[0].shape)
        value = torch.zeros_like(key)
        if range_case == "underflow":
            key[:, 1, :, 0] = -20
            value[:, 1] = 1
        elif range_case == "threshold":
            key[:, 1, :, 0] = -2
            value.fill_(1)
            value[:, 1] = 3
        else:
            assert range_case == "overflow"
            value.fill_(448)
        cpu_kv = (key.to(kv_dtype), value.to(kv_dtype))
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        # Includes the requested [B*Q,48,128][:,:32,:] fused-QKV view.
        fused = torch.empty(
            batch * query_length, hq + 2 * hk, dim, device="cuda", dtype=torch.bfloat16
        )
        q = fused[:, :hq]
        q.copy_(cpu_q)
        kv = tuple(value.cuda() for value in cpu_kv)
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        device_meta = {name: value.cuda() for name, value in host.items()}
        graph_buffers = {
            "paged_kv_indptr_buffer": device_meta["cu_seqlens_k"],
            "paged_kv_indices_buffer": device_meta["indices"],
            "paged_kv_last_page_len_buffer": device_meta["last_page_len"],
        }
        geometry = {"num_qo_heads": hq, "num_kv_heads": hk, "head_dim": dim}
        if fixed:
            sizes = FlashInferFixedQueryAttention.workspace_size(
                batch_size=batch, query_length=query_length, device=q.device, **geometry
            )
        else:
            sizes = FlashInferPagedAttention.workspace_size(
                mode="decode",
                backend="fa2",
                use_tensor_cores=use_tensor_cores,
                use_cuda_graph=decode_graph,
                max_batch_size=batch,
                device=q.device,
                dispatch="provider",
                **geometry,
            )
        workspace = tuple(torch.zeros(n, dtype=torch.uint8, device=q.device) for n in sizes)
        if fixed:
            operation = FlashInferFixedQueryAttention(
                *workspace,
                query_indptr=device_meta["cu_seqlens_q"],
                kv_indptr=device_meta["cu_seqlens_k"],
                kv_indices=device_meta["indices"],
                last_page_lengths=device_meta["last_page_len"],
                query_length=query_length,
                page_size=page,
                dtype=q.dtype,
                kv_dtype=kv_dtype,
                causal=True,
                **geometry,
            )
            # Construction initializes dummy metadata. Replace it before prepare.
            for name, value in host.items():
                device_meta[name].copy_(value)
            operation.prepare()
        else:
            operation = FlashInferPagedAttention(
                *workspace,
                mode="decode",
                backend="fa2",
                use_tensor_cores=use_tensor_cores,
                use_cuda_graph=decode_graph,
                dispatch="provider",
                **(graph_buffers if decode_graph else {}),
            )
            operation.plan(
                **(host | {"indices": device_meta["indices"]}),
                **geometry,
                page_size=page,
                pos_encoding_mode="NONE",
                dtype=q.dtype,
                kv_dtype=kv_dtype,
            )
            assert operation.decode_route == ("fa2" if use_tensor_cores else "cuda_cores")
        stage = operation._fp8_output.stage
        assert stage.dtype == torch.float16 and stage.shape == q.shape
        stage_pointer = stage.data_ptr()

        baseline_work = tuple(torch.zeros_like(value) for value in workspace)
        # A plain provider with its own BF16 output and post-kernel multiply.
        if fixed:
            baseline_meta = {name: value.clone() for name, value in device_meta.items()}
            baseline = BatchPrefillWithPagedKVCacheWrapper(
                baseline_work[0],
                backend="fa2",
                use_cuda_graph=True,
                qo_indptr_buf=baseline_meta["cu_seqlens_q"],
                paged_kv_indptr_buf=baseline_meta["cu_seqlens_k"],
                paged_kv_indices_buf=baseline_meta["indices"],
                paged_kv_last_page_len_buf=baseline_meta["last_page_len"],
            )
            baseline.reset_workspace_buffer(*baseline_work)
            baseline.plan(
                qo_indptr=host["cu_seqlens_q"],
                paged_kv_indptr=host["cu_seqlens_k"],
                paged_kv_indices=device_meta["indices"],
                paged_kv_last_page_len=host["last_page_len"],
                num_qo_heads=hq,
                num_kv_heads=hk,
                head_dim_qk=dim,
                page_size=page,
                causal=True,
                q_data_type=q.dtype,
                kv_data_type=kv_dtype,
            )
        else:
            baseline = BatchDecodeWithPagedKVCacheWrapper(
                baseline_work[0],
                backend="fa2",
                use_tensor_cores=use_tensor_cores,
                use_cuda_graph=decode_graph,
                **(graph_buffers if decode_graph else {}),
            )
            baseline.reset_workspace_buffer(*baseline_work)
            baseline.plan(
                indptr=host["cu_seqlens_k"],
                indices=device_meta["indices"],
                last_page_len=host["last_page_len"],
                **geometry,
                page_size=page,
                q_data_type=q.dtype,
                kv_data_type=kv_dtype,
            )
        range_scales = {"underflow": 1e8, "overflow": 1e3, "threshold": range_scale}
        scales_to_test = (
            ((1.0, range_scales[range_case]),)
            if range_case is not None
            else ((0.59, 0.31), (1.0, 1.0), (0.59, 0.0), (0.59, -0.31), (1.0, None))
        )
        for k_scale, v_scale in scales_to_test:
            scales = {"k_scale": k_scale, "v_scale": v_scale}
            output_dtype = torch.float16 if v_scale is None or abs(v_scale) <= 1 else torch.bfloat16
            if fixed:
                operation.prepare()  # first run uses the non-PDL path
            actual = operation.run(q=q, paged_kv_cache=kv, out=out, **scales)
            assert actual is out
            assert operation._fp8_output.last_output_dtype == output_dtype
            first = actual.clone()
            allocated = operation.run(q=q, paged_kv_cache=kv, **scales)
            assert operation._fp8_output.last_output_dtype == output_dtype
            plain = baseline.run(q=q, paged_kv_cache=kv, **scales)
            torch.testing.assert_close(allocated, first, atol=0, rtol=0)
            # Capture run alone (PDL on), and prepare + run (PDL off).
            for prepare in (False, True) if fixed else (False,):
                graph = torch.cuda.CUDAGraph()
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    if prepare:
                        operation.prepare()
                    operation.run(q=q, paged_kv_cache=kv, out=out, **scales)
                assert operation._fp8_output.last_output_dtype == output_dtype
                for _ in range(2):
                    out.fill_(torch.nan)
                    graph.replay()
                    torch.testing.assert_close(out, first, atol=0, rtol=0)
                assert operation._fp8_output.stage.data_ptr() == stage_pointer
            stream.synchronize()
            if v_scale == 0.0:
                assert torch.count_nonzero(out).item() == 0
                continue
            oracle = reference_attention(
                cpu_q,
                cpu_kv,
                (query_length,) * batch,
                (length,) * batch,
                host["indices"],
                k_scale=k_scale,
                v_scale=1.0 if v_scale is None else v_scale,
                compute_dtype=torch.float64,
            )
            rms = (out.cpu().double() - oracle).square().mean().sqrt()
            baseline_rms = (plain.cpu().double() - oracle).square().mean().sqrt()
            if range_case is not None:
                assert torch.isfinite(out).all()
                torch.testing.assert_close(out.cpu().double(), oracle, atol=0, rtol=0.008)
            elif use_tensor_cores:
                # Unit V scale gains only from FP16 vs BF16 provider rounding;
                # non-unit V scale also removes the BF16 post-scale rounding.
                ratio = 0.9 if v_scale is None or v_scale == 1.0 else 0.75
                assert rms <= ratio * baseline_rms, (rms.item(), baseline_rms.item(), scales)
            else:
                # The measured improvement gate is specific to FA2. CUDA-core
                # decode still exercises staging/scaling and graph correctness.
                torch.testing.assert_close(out.cpu().double(), oracle, atol=0.01, rtol=0.01)
