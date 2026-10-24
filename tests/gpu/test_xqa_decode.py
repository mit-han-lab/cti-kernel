# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FP8 XQA correctness, graph replanning and exact scratch capacity."""

import pytest
import torch

from ctikernels.attention import _split_tables
from ctikernels.attention._flashinfer import xqa_flashinfer_version
from ctikernels.attention.inference import FlashInferPagedAttention

from ._attention import paged_metadata, reference_attention
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def xqa_device():
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("XQA route requires SM120/121")
    pytest.importorskip("flashinfer")
    if not xqa_flashinfer_version():
        pytest.skip("XQA route requires FlashInfer 0.7.0.post1 or 0.7.1")


def install_row(monkeypatch, heads, batch):
    # These tests exercise the XQA route itself; rows the device table sends to the
    # CuTe FP8 decode are covered by test_decode_fp8_cute and test_inference_attention.
    monkeypatch.setattr(_split_tables, "FP8_DECODE_CUTE", {})
    key = _split_tables.device_key(torch.device("cuda"))
    if key is None:
        # Exercise qualification on other SM120/121 devices without adding policy.
        key = "test-sm12"
        monkeypatch.setattr(_split_tables, "device_key", lambda _: key)
    row_key = (key, *heads)
    row = _split_tables.XQA_DECODE_SUBSEQ.get(row_key, {})
    monkeypatch.setitem(_split_tables.XQA_DECODE_SUBSEQ, row_key, {batch: 8} | row)


def make_operation(heads, batch, graph, capacity, *, sentinel=False):
    options = {
        "mode": "decode",
        "backend": "fa2",
        "use_tensor_cores": True,
        "use_cuda_graph": graph,
    }
    sizes = FlashInferPagedAttention.workspace_size(
        **options,
        max_batch_size=batch,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=heads[2],
        device=torch.device("cuda"),
    )
    backing = torch.full(
        (sizes[0] + (4096 if sentinel else 0),), 0xA5, dtype=torch.uint8, device="cuda"
    )
    workspace = backing[: sizes[0]]
    workspace.zero_()
    integer = torch.empty(sizes[1], dtype=torch.uint8, device="cuda")
    buffers = {}
    if graph:
        buffers = {
            "paged_kv_indptr_buffer": torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
            "paged_kv_indices_buffer": torch.empty(capacity, dtype=torch.int32, device="cuda"),
            "paged_kv_last_page_len_buffer": torch.empty(batch, dtype=torch.int32, device="cuda"),
        }
    op = FlashInferPagedAttention(workspace, integer, **options, **buffers)
    return op, backing


def plan(op, metadata, heads, page, dtype, window):
    op.plan(
        **metadata,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=heads[2],
        page_size=page,
        pos_encoding_mode="NONE",
        dtype=dtype,
        kv_dtype=torch.float8_e4m3fn,
        window_left=window,
    )
    assert op._decode_route[0] == "xqa"
    assert op._wrapper is None


CASES = ci_cases(
    ((32, 8, 128), (16, 4, 256), (24, 4, 256), (16, 4, 128), (8, 2, 256), (12, 2, 256)),
    (1, 2, 3, 4),
    (1, 17, 512, 4096),
    (16, 64),
    (False, True),
    (torch.bfloat16, torch.float16),
    core={
        (32, 8, 128, 1, 1, 16, False, torch.bfloat16),
        (32, 8, 128, 2, 17, 64, True, torch.bfloat16),
        (32, 8, 128, 3, 512, 16, True, torch.float16),
        (32, 8, 128, 4, 4096, 64, True, torch.bfloat16),
        (16, 4, 256, 1, 4096, 16, True, torch.bfloat16),
        (16, 4, 256, 2, 512, 64, False, torch.float16),
        (16, 4, 256, 3, 17, 16, True, torch.bfloat16),
        (16, 4, 256, 4, 1, 64, False, torch.bfloat16),
        (24, 4, 256, 2, 4096, 16, True, torch.bfloat16),
        (24, 4, 256, 4, 512, 64, False, torch.bfloat16),
        (16, 4, 128, 3, 4096, 16, True, torch.bfloat16),
        (8, 2, 256, 4, 4096, 16, True, torch.bfloat16),
        (12, 2, 256, 2, 512, 16, True, torch.float16),
    },
)


@pytest.mark.parametrize("qh,kh,dim,batch,length,page,graph,dtype", CASES)
def test_xqa_matches_fp32(monkeypatch, qh, kh, dim, batch, length, page, graph, dtype):
    check_reference(monkeypatch, (qh, kh, dim), batch, length, page, graph, dtype)


def check_reference(
    monkeypatch, heads, batch, length, page, graph, dtype, window=-1, fused=False, v_scale=0.25
):
    install_row(monkeypatch, heads, batch)
    generator = torch.Generator().manual_seed(173 + length + batch)
    cpu_q = torch.randn(batch, heads[0], heads[2], dtype=dtype, generator=generator)
    target = paged_metadata((1,) * batch, (length,) * batch, page)
    capacity = target["indices"].numel()
    shape = (2 * capacity + 1, page, heads[1], heads[2])
    cpu_cache = tuple(
        torch.randn(shape, dtype=dtype, generator=generator).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        q = cpu_q.cuda()
        out = torch.empty_like(q)
        if fused:
            # Fused-QKV view: query rows strided by (Hq + 2 Hkv) * D.
            qkv = torch.randn(batch, heads[0] + 2 * heads[1], heads[2], dtype=dtype, device="cuda")
            qkv[:, : heads[0]].copy_(q)
            q = qkv[:, : heads[0]]
        cache = tuple(value.cuda() for value in cpu_cache)
        op, backing = make_operation(heads, batch, graph, capacity)
        captured = None
        addresses = None
        # The second plan changes every length (including the length=1 case),
        # and the last changes page counts and CSR offsets for mixed requests.
        for lengths in (
            (2,) * batch,
            (length,) * batch,
            tuple(max(1, length - i * page - 1) for i in range(batch)),
        ):
            metadata = paged_metadata((1,) * batch, lengths, page)
            cpu_indices = metadata["indices"]
            metadata["indices"] = cpu_indices.cuda()
            plan(op, metadata, heads, page, dtype, window)
            if graph and captured is None:
                op.run(q=q, paged_kv_cache=cache, out=out, k_scale=0.5, v_scale=v_scale)
                stream.synchronize()
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    op.run(q=q, paged_kv_cache=cache, out=out, k_scale=0.5, v_scale=v_scale)
                addresses = (
                    op._xqa.page_table.data_ptr(),
                    op._xqa.seq_lens.data_ptr(),
                    op._xqa.semaphores.data_ptr(),
                )
            if graph:
                assert addresses == (
                    op._xqa.page_table.data_ptr(),
                    op._xqa.seq_lens.data_ptr(),
                    op._xqa.semaphores.data_ptr(),
                )
                # Indices are a plan-time snapshot; run/replay must not reread them.
                metadata["indices"].zero_()
                captured.replay()
            else:
                op.run(q=q, paged_kv_cache=cache, out=out, k_scale=0.5, v_scale=v_scale)
            stream.synchronize()  # also finishes pinned staging before replan
            expected = reference_attention(
                cpu_q,
                cpu_cache,
                (1,) * batch,
                lengths,
                cpu_indices,
                k_scale=0.5,
                v_scale=v_scale,
                window_left=window,
            )
            torch.testing.assert_close(out.cpu().float(), expected, atol=1e-2, rtol=3e-2)
            assert torch.count_nonzero(op._xqa.semaphores.to(torch.int64)).item() == 0
        del captured, op, backing


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("window", [0, 31, 128])
def test_xqa_window(monkeypatch, graph, window):
    check_reference(monkeypatch, (32, 8, 128), 2, 512, 16, graph, torch.bfloat16, window)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 4, 256)])
def test_xqa_fused_qkv_views(monkeypatch, heads, graph):
    check_reference(monkeypatch, heads, 2, 512, 16, graph, torch.bfloat16, fused=True)


@pytest.mark.parametrize("graph", [False, True])
def test_xqa_zero_v_scale(monkeypatch, graph):
    # FA2 multiplies the output by v_scale; zero gives zeros, not an error.
    check_reference(monkeypatch, (32, 8, 128), 2, 512, 16, graph, torch.bfloat16, v_scale=0.0)


SENTINELS = ci_cases(
    ((32, 8, 128), (16, 4, 256), (1, 1, 16), (2, 1, 80), (8, 1, 144), (16, 1, 192), (32, 1, 256)),
    (False, True),
    core={(32, 8, 128, True), (16, 4, 256, False), (1, 1, 16, False)},
)


@pytest.mark.parametrize("qh,kh,dim,graph", SENTINELS)
def test_xqa_scratch_sentinel(monkeypatch, qh, kh, dim, graph):
    heads, batch, page = (qh, kh, dim), 1, 16
    install_row(monkeypatch, heads, batch)
    subseq = _split_tables.xqa_decode_subseq(torch.device("cuda"), *heads, batch)
    # Repeated valid page zero makes a long sequence inexpensive in cache memory.
    # W*page >= S*256 forces the full split count, including alignment padding.
    length = subseq * 256
    metadata = paged_metadata((1,), (length,), page)
    metadata["indices"] = torch.zeros(length // page, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        # Ensure the device indices produced on the default stream are ready.
        stream.wait_stream(torch.cuda.default_stream())
        op, backing = make_operation(heads, batch, graph, length // page, sentinel=True)
        q = torch.zeros(batch, qh, dim, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, page, kh, dim, device="cuda").to(torch.float8_e4m3fn)
        v = torch.ones(1, page, kh, dim, device="cuda").to(torch.float8_e4m3fn)
        plan(op, metadata, heads, page, q.dtype, -1)
        bound = op._xqa.scratch_bytes
        backing[bound:].fill_(0xA5)
        out = torch.empty_like(q)
        op.run(q=q, paged_kv_cache=(k, v), out=out, k_scale=0.5, v_scale=0.25)
        stream.synchronize()
        captured = None
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                op.run(q=q, paged_kv_cache=(k, v), out=out, k_scale=0.5, v_scale=0.25)
        for _ in range(3):
            if graph:
                captured.replay()
            else:
                op.run(q=q, paged_kv_cache=(k, v), out=out, k_scale=0.5, v_scale=0.25)
        stream.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, 0.25), atol=0, rtol=0)
        assert torch.all(backing[bound:] == 0xA5).item()
        assert torch.count_nonzero(op._xqa.semaphores.to(torch.int64)).item() == 0
        del captured, op


@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 4, 256)])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [513, 4097])
@pytest.mark.parametrize("seed", [173, 20261017])
@torch.inference_mode()
def test_xqa_bitwise_matches_table_split(monkeypatch, heads, batch, length, seed):
    check_bitwise_split(monkeypatch, heads, batch, length, seed)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [513, 4097])
@pytest.mark.parametrize("seed", [173, 20261017])
@torch.inference_mode()
def test_xqa_rtx5090_bitwise_matches_vllm_split(monkeypatch, batch, length, seed):
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms != 170 or _split_tables.device_key(device) != "rtx5090":
        pytest.skip("vLLM split equivalence requires the 170-SM RTX 5090 row")
    assert _split_tables.xqa_decode_subseq(device, 32, 8, 128, batch) == sms // (batch * 8)
    check_bitwise_split(monkeypatch, (32, 8, 128), batch, length, seed, sm_count=sms)


def check_bitwise_split(monkeypatch, heads, batch, length, seed, *, sm_count=None):
    from flashinfer.xqa import xqa

    install_row(monkeypatch, heads, batch)
    page, context = 16, 32768
    width = context // page
    hq, hk, dim = heads
    lengths = tuple(length - i * 17 for i in range(batch))
    metadata = paged_metadata((1,) * batch, lengths, page)
    # Build vLLM's context-wide block table independently of the route's table.
    table = torch.zeros(batch, width, dtype=torch.int32)
    offsets = metadata["cu_seqlens_k"].tolist()
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        table[row, : end - start] = metadata["indices"][start:end]
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, hq, dim, dtype=torch.bfloat16, generator=generator).cuda()
    cache = tuple(
        torch.randn(2 * metadata["indices"].numel() + 1, page, hk, dim, generator=generator)
        .to(torch.float8_e4m3fn)
        .cuda()
        for _ in range(2)
    )
    metadata["indices"] = metadata["indices"].cuda()
    op, backing = make_operation(heads, batch, True, width)
    plan(op, metadata, heads, page, q.dtype, -1)
    assert op._xqa.page_table.shape == (batch, width)
    expected = torch.empty_like(q)
    out = torch.empty_like(q)
    # Independent provider scratch/semaphores, using the table split by default.
    # The RTX 5090 comparison instead forwards the actual SM count as vLLM does.
    if sm_count is None:
        subseq = _split_tables.xqa_decode_subseq(q.device, *heads, batch)
        sm_count = subseq * batch * hk
    scratch = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    semaphores = torch.zeros(batch * hk, dtype=torch.uint32, device="cuda")
    xqa(
        q=q.unsqueeze(1),
        k_cache=cache[0],
        v_cache=cache[1],
        page_table=table.cuda(),
        seq_lens=metadata["seq_lens"].reshape(batch, 1).cuda(),
        output=expected.unsqueeze(1),
        workspace_buffer=scratch,
        semaphores=semaphores,
        num_kv_heads=hk,
        page_size=page,
        q_scale=dim**-0.5 * 0.5 / 0.25 * dim**0.5,
        kv_scale=0.25,
        sliding_win_size=0,
        kv_layout="NHD",
        sm_count=sm_count,
        enable_pdl=None,
    )
    op.run(q=q, paged_kv_cache=cache, out=out, k_scale=0.5, v_scale=0.25)
    torch.testing.assert_close(out.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(q=q, paged_kv_cache=cache, out=out, k_scale=0.5, v_scale=0.25)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(
            out.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
    del graph, op, backing
