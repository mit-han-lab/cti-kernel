# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FP8 paged decode against FP32, including runtime CSR changes under capture."""

import pytest
import torch

from ctikernels.attention._decode_fp8 import FP8PagedDecode, _split_geometry, workspace_size

from ._exhaustive import pairwise_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def device_and_config(monkeypatch):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("FP8 CuTe decode requires sm_120/121")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cutlass.runtime", reason="CuTe DSL 4.8 required")
    for key in ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE"):
        monkeypatch.delenv(f"CTI_FP8DEC_{key}", raising=False)
    # Bounded test workspace, and predictable one/split route transitions.
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "4")


def metadata(lengths, page, num_pages, *, rotated=0):
    counts = [(length + page - 1) // page for length in lengths]
    ptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(0).to(torch.int32)
    indices = (torch.arange(sum(counts), dtype=torch.int32) * 7 + rotated) % num_pages
    last = torch.tensor(
        [(length - 1) % page + 1 if length else 0 for length in lengths], dtype=torch.int32
    )
    return ptr, indices, last


def reference(q, k, v, host, lengths, k_scale, v_scale, sm_scale):
    # Explicit dequantization and FP32 math; unrelated to kernel fragments/splits.
    ptr, indices, _ = host
    q, k, v = q.float(), k.float() * k_scale, v.float() * v_scale
    expected = torch.zeros_like(q)
    for b, length in enumerate(lengths):
        if length:
            pages = indices[ptr[b] : ptr[b + 1]].long().to(q.device)
            keys = k[pages].flatten(0, 1)[:length]
            values = v[pages].flatten(0, 1)[:length]
            mapping = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[2])
            keys, values = keys[:, mapping], values[:, mapping]
            scores = torch.einsum("hd,jhd->hj", q[b], keys) * sm_scale
            expected[b] = torch.einsum("hj,jhd->hd", scores.softmax(-1), values)
    return expected


def exercise(
    batch,
    hq,
    hk,
    dim,
    page,
    lengths,
    dtype,
    k_scale=1.0,
    v_scale=1.0,
    *,
    strided=False,
    graph=False,
    sm_scale=None,
    wrap_generation=False,
):
    device = torch.device("cuda")
    geometry = {"batch_size": batch, "num_qo_heads": hq, "num_kv_heads": hk, "head_dim": dim}
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        torch.manual_seed(1701)
        # A bounded, permuted pool keeps page_size=1 and long sweeps inexpensive.
        # Repeated page ids are legal; varied rows still test both fragment maps.
        k = torch.randn(37, page, hk, dim, device=device).to(torch.float8_e4m3fn)
        v = torch.randn_like(k, dtype=torch.float32).to(torch.float8_e4m3fn)
        q = torch.randn(batch, hq * (3 if strided else 1), dim, device=device, dtype=dtype)[:, :hq]
        out = torch.empty(batch, hq, dim, device=device, dtype=dtype)
        size = workspace_size(device, **geometry)
        backing = torch.full((size + 256,), 0xA5, dtype=torch.uint8, device=device)
        op = FP8PagedDecode(
            backing[:size],
            **geometry,
            page_size=page,
            q_dtype=dtype,
            device=device,
            sm_scale=sm_scale,
        )
        generations = [0] * batch
        if wrap_generation and op.config["merge"] == "distributed":
            op._counters[1::2].fill_(-1)
            generations = [2**32 - 1] * batch
        scale = dim**-0.5 if sm_scale is None else sm_scale
        # Same pointers and launch geometry for every state, with enough CSR
        # capacity for the longest state captured in this test.
        states = [lengths]
        if graph:
            states = [
                (1,) * batch,
                lengths,
                (0,) * batch,
                tuple(17 + b for b in range(batch)),
                lengths,
            ]
        capacity = max(sum((n + page - 1) // page for n in state) for state in states)
        ptr = torch.empty(batch + 1, device=device, dtype=torch.int32)
        indices = torch.empty(capacity, device=device, dtype=torch.int32)
        last = torch.empty(batch, device=device, dtype=torch.int32)
        captured = None
        for state_id, state in enumerate(states):
            host = metadata(state, page, 37, rotated=state_id * 3)
            ptr.copy_(host[0])
            indices[: host[1].numel()].copy_(host[1])
            last.copy_(host[2])
            if graph and captured is None:
                op.run(q, k, v, ptr, indices, last, out, k_scale=k_scale, v_scale=v_scale)
                stream.synchronize()
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    op.run(q, k, v, ptr, indices, last, out, k_scale=k_scale, v_scale=v_scale)
            for _ in range(3):
                out.fill_(float("nan"))
                if graph:
                    captured.replay()
                else:
                    op.run(q, k, v, ptr, indices, last, out, k_scale=k_scale, v_scale=v_scale)
                expected = reference(q, k, v, host, state, k_scale, v_scale, scale)
                torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=3e-2)
                if v_scale == 0:
                    assert torch.count_nonzero(out).item() == 0
                for b, length in enumerate(state):
                    if length == 0:
                        assert torch.count_nonzero(out[b]).item() == 0
                assert torch.count_nonzero(op._arrivals).item() == 0
                if op.config["merge"] == "distributed":
                    for b, length in enumerate(state):
                        _, splits = _split_geometry(
                            length, op.config["min_chunk"], op.config["splits"]
                        )
                        generations[b] = (generations[b] + int(splits > 1)) % 2**32
                    expected_flags = torch.tensor(
                        [value if value < 2**31 else value - 2**32 for value in generations],
                        dtype=torch.int32,
                        device=device,
                    ).repeat_interleave(hk)
                    torch.testing.assert_close(op._counters[1::2], expected_flags)
                assert torch.all(backing[size:] == 0xA5).item()
        stream.synchronize()


# Pairwise sweep of every axis value, reusing kernels across lengths and scales.
CASES = pairwise_cases(
    (1, 2, 3, 4, 7),
    ((32, 8), (16, 4), (24, 4), (8, 2), (12, 2), (4, 4), (64, 4)),
    (64, 128, 256),
    (1, 16, 64),
    (0, 1, 15, 16, 17, 255, 256, 257, 1000, 4096, 8192, 32768),
    (torch.bfloat16, torch.float16),
    ((1.0, 1.0), (0.5, 0.5), (0.5, 0.0)),
    seed=1701,
    compile_axes=(0, 1, 2, 3, 5),  # batch, heads, dim, page, dtype
    core={
        (1, 32, 8, 128, 16, 1, torch.bfloat16, 1.0, 1.0),
        (2, 24, 4, 64, 1, 257, torch.float16, 0.5, 0.5),
        (3, 64, 4, 256, 64, 1000, torch.bfloat16, 0.5, 0.5),
        (4, 4, 4, 64, 16, 0, torch.float16, 1.0, 1.0),
        (7, 12, 2, 128, 16, 17, torch.bfloat16, 0.5, 0.0),
        (1, 8, 2, 128, 1, 32768, torch.float16, 1.0, 1.0),
    },
)


@pytest.mark.parametrize("batch,hq,hk,dim,page,length,dtype,k_scale,v_scale", CASES)
def test_fp8_decode(batch, hq, hk, dim, page, length, dtype, k_scale, v_scale):
    exercise(batch, hq, hk, dim, page, (length,) * batch, dtype, k_scale, v_scale)


@pytest.mark.parametrize(
    "dim,dtype", [(64, torch.float16), (128, torch.bfloat16), (256, torch.float16)]
)
@pytest.mark.parametrize("graph", [False, True])
def test_mixed_strided_and_graph(dim, dtype, graph):
    exercise(
        4,
        24,
        4,
        dim,
        16,
        (0, 17, 257, 4096),
        dtype,
        0.5,
        0.5,
        strided=True,
        graph=graph,
        sm_scale=0.07,
    )


@pytest.mark.parametrize("keys,warps,stages", [(32, 2, 2), (16, 1, 4), (16, 8, 2)])
def test_tuning_variants(monkeypatch, keys, warps, stages):
    for key, value in {"KEYS": keys, "WARPS": warps, "STAGES": stages}.items():
        monkeypatch.setenv(f"CTI_FP8DEC_{key}", str(value))
    exercise(2, 12, 2, 64, 7, (15, 1000), torch.bfloat16, graph=True)


@pytest.mark.parametrize("group", [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15])
@pytest.mark.exhaustive
def test_remaining_gqa_groups(group):
    exercise(2, group * 2, 2, 128, 7, (256, 257), torch.float16)


@pytest.mark.parametrize("merge", ["distributed", "last"])
@pytest.mark.parametrize("split", [1, 7])
def test_split_limits(monkeypatch, split, merge):
    monkeypatch.setenv("CTI_FP8DEC_MERGE", merge)
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", str(split))
    exercise(3, 16, 4, 128, 16, (1, 257, 8192), torch.bfloat16, v_scale=0.0, graph=True)


@pytest.mark.parametrize("dim,hk", [(128, 8), (256, 4), (256, 2)])
@pytest.mark.parametrize("merge", ["distributed", "last"])
def test_many_splits_and_replay(monkeypatch, dim, hk, merge):
    monkeypatch.setenv("CTI_FP8DEC_MERGE", merge)
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "170")
    # Nonzero scale exercises the merge sum, tail vectors, and refreshed LSEs.
    exercise(1, hk * 4, hk, dim, 16, (8193,), torch.bfloat16, graph=True)


def test_small_min_chunk(monkeypatch):
    monkeypatch.setenv("CTI_FP8DEC_MIN_CHUNK", "1")
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "7")
    exercise(2, 6, 2, 128, 7, (5, 17), torch.float16, graph=True, wrap_generation=True)


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("length,splits", [(8192, 21), (2048, 10), (33, 7)])
def test_warp_aligned_split_tails(monkeypatch, dim, length, splits):
    monkeypatch.setenv("CTI_FP8DEC_MIN_CHUNK", "1")
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", str(splits))
    exercise(1, 32, 8, dim, 16, (length,), torch.bfloat16, graph=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("merge", ["last", "distributed"])
@torch.inference_mode()
def test_runtime_strides_and_bf16_range(monkeypatch, dtype, merge):
    monkeypatch.setenv("CTI_FP8DEC_MIN_CHUNK", "64")
    monkeypatch.setenv("CTI_FP8DEC_MERGE", merge)
    geometry = {"batch_size": 2, "num_qo_heads": 8, "num_kv_heads": 2, "head_dim": 128}
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    op = FP8PagedDecode(workspace, **geometry, page_size=16, q_dtype=dtype, device="cuda")
    lengths = (17, 193)
    host = metadata(lengths, 16, 19)
    gpu = [x.cuda() for x in host]
    torch.manual_seed(42)
    packed = torch.randn(19, 2, 16, 2, 128, device="cuda").to(torch.float8_e4m3fn)
    other = torch.randn(19, 3, 16, 2, 128, device="cuda").to(torch.float8_e4m3fn)
    storage = torch.randn(2, 24, 128, dtype=dtype, device="cuda")
    out = torch.empty(2, 8, 128, dtype=dtype, device="cuda")
    for q in (storage[:, :8], storage[:1, :8].expand(2, -1, -1)):
        for k, v in ((packed[:, 0], packed[:, 1]), (packed[:, 0], other[:, 1])):
            # Neither ABI needs a warm-up launch to compile during capture.
            dense_k, dense_v = k.contiguous(), v.contiguous()
            dense_out = torch.empty_like(out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.run(q, k, v, *gpu, out)
                op.run(q, dense_k, dense_v, *gpu, dense_out)
            graph.replay()
            assert torch.equal(out.view(torch.int16), dense_out.view(torch.int16))
            expected = reference(q, k, v, host, lengths, 1, 1, 128**-0.5)
            torch.testing.assert_close(out.float(), expected, atol=0.01, rtol=0.03)
    if dtype == torch.bfloat16:
        packed[:, 0].zero_()
        packed[:, 1].fill_(2)
        for magnitude in (65536, 1e30):
            storage.fill_(magnitude)
            storage[:, :, ::2].neg_()
            op.run(storage[:, :8], packed[:, 0], packed[:, 1], *gpu, out)
            torch.testing.assert_close(out, torch.full_like(out, 2), atol=0, rtol=0)


@pytest.mark.parametrize("merge", ["last", "distributed"])
@pytest.mark.parametrize("group,dim", [(4, 128), (16, 256)])
@pytest.mark.parametrize("page", [1, 16])
@torch.inference_mode()
def test_bf16_qk_path_selection(monkeypatch, merge, group, dim, page):
    monkeypatch.setenv("CTI_FP8DEC_MIN_CHUNK", "64")
    # Page-one splits cross piece boundaries in both QK paths.
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "3")
    monkeypatch.setenv("CTI_FP8DEC_MERGE", merge)
    geometry = {"batch_size": 1, "num_qo_heads": group * 2, "num_kv_heads": 2, "head_dim": dim}
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    op = FP8PagedDecode(
        workspace, **geometry, page_size=page, q_dtype=torch.bfloat16, device="cuda"
    )
    torch.manual_seed(713)
    q = torch.randn(1, group * 2, dim, dtype=torch.bfloat16, device="cuda")
    pages = (257 + page - 1) // page + 2
    k = torch.randn(pages, page, 2, dim, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn_like(k, dtype=torch.float32).to(torch.float8_e4m3fn)
    lengths = (257,)
    host = metadata(lengths, page, pages)
    gpu = [x.cuda() for x in host]
    out = torch.empty_like(q)
    # Ordinary BF16 queries are FP16-exact. Compare the fast path to FP32
    # attention (including the existing FP16 PV rounding tolerance).
    torch.testing.assert_close(q.half().bfloat16(), q, atol=0, rtol=0)
    op.run(q, k, v, *gpu, out)
    torch.testing.assert_close(
        out.float(), reference(q, k, v, host, lengths, 1, 1, dim**-0.5), atol=0.01, rtol=0.03
    )
    # Replay one graph while changing the Q-dependent branch. Only the last
    # element of one head needs BF16; the other KV head's CTA stays FP16.
    for magnitude in (65536, 1e30, 2**-30):
        op.sm_scale = 1 / magnitude
        # Scales are captured host arguments, so recapture for each magnitude.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.run(q, k, v, *gpu, out)
        for sign in (1, -1):
            q.zero_()
            q[0, group - 1, -1] = sign * magnitude
            graph.replay()
            expected = reference(q, k, v, host, lengths, 1, 1, op.sm_scale)
            torch.testing.assert_close(out.float(), expected, atol=0.01, rtol=0.03)
        q.zero_()
        graph.replay()
        torch.testing.assert_close(
            out.float(), reference(q, k, v, host, lengths, 1, 1, op.sm_scale), atol=0.01, rtol=0.03
        )


def test_bf16_fp16_representability_all_codes():
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ctikernels.attention._decode_fp8._kernel import bf16_fp16_exact

    @cute.kernel
    def check(source: cute.Tensor, target: cute.Tensor):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        index = block * 256 + tid
        target[index] = bf16_fp16_exact(source[index])

    @cute.jit
    def launch(source: cute.Tensor, target: cute.Tensor):
        check(source, target).launch(grid=(256, 1, 1), block=(256, 1, 1))

    # All BF16 bit patterns, including +/-0, subnormals, infinities and NaNs.
    source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    expected = torch.isfinite(source) & (source.float() == source.half().float())
    source = source.cuda()
    target = torch.empty(65536, dtype=torch.int32, device="cuda")
    compiled = cute.compile(launch, from_dlpack(source), from_dlpack(target))
    compiled(from_dlpack(source), from_dlpack(target))
    torch.testing.assert_close(target.cpu().bool(), expected, atol=0, rtol=0)


def test_e4m3_to_bf16_exact():
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ctikernels.attention._decode_fp8._kernel import fp8_bf16_pair

    @cute.kernel
    def convert(source: cute.Tensor, target: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        target[tid] = fp8_bf16_pair(source[tid])

    @cute.jit
    def launch(source: cute.Tensor, target: cute.Tensor):
        convert(source, target).launch(grid=(1, 1, 1), block=(256, 1, 1))

    # Every code occupies both halves, paired with a different code/sign.
    codes = torch.arange(256, dtype=torch.int32)
    pairs = torch.stack((codes, codes ^ 0xA5), dim=1).to(torch.uint8).contiguous()
    source = (pairs[:, 0].int() | (pairs[:, 1].int() << 8)).cuda()
    target = torch.empty(256, dtype=torch.int32, device="cuda")
    compiled = cute.compile(launch, from_dlpack(source), from_dlpack(target))
    compiled(from_dlpack(source), from_dlpack(target))
    expected = pairs.view(torch.float8_e4m3fn).to(torch.bfloat16)
    actual = target.cpu().view(torch.bfloat16).view(256, 2)
    finite = torch.isfinite(expected)
    # Compare bits so negative zero is checked too.
    torch.testing.assert_close(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], atol=0, rtol=0
    )


@pytest.mark.parametrize("layout", ["heads", "padded", "mixed", "packed"])
@pytest.mark.parametrize("dim,page", [(128, 7), (256, 16)])
@torch.inference_mode()
def test_aligned_kv_views(monkeypatch, layout, dim, page):
    from ._attention import kv_views

    monkeypatch.setenv("CTI_FP8DEC_MIN_CHUNK", "64")
    geometry = {"batch_size": 2, "num_qo_heads": 8, "num_kv_heads": 2, "head_dim": dim}
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    op = FP8PagedDecode(
        workspace, **geometry, page_size=page, q_dtype=torch.bfloat16, device="cuda"
    )
    torch.manual_seed(42)
    q = torch.randn(2, 8, dim, dtype=torch.bfloat16, device="cuda")
    dense = tuple(
        torch.randn(19, page, 2, dim, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
    )
    cache = kv_views(dense, layout)
    lengths = (17, 193)
    host = metadata(lengths, page, 19)
    gpu = [x.cuda() for x in host]
    out, dense_out = torch.empty_like(q), torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(q, *cache, *gpu, out)
        op.run(q, *dense, *gpu, dense_out)
    graph.replay()
    assert torch.equal(out.view(torch.int16), dense_out.view(torch.int16))
    expected = reference(q, *dense, host, lengths, 1, 1, dim**-0.5)
    torch.testing.assert_close(out.float(), expected, atol=0.01, rtol=0.03)
