# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Plan and execute boundary/legacy-overflow shapes using only declared scratch."""

from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from ctikernels.attention.inference import FlashInferPagedAttention

from ._attention import paged_metadata
from ._exhaustive import ci_cases


@pytest.mark.parametrize("page", [1, 16])
@pytest.mark.parametrize(
    "name",
    "extend split32 split27 tc-eager tc-graph tc-max cuda-eager cuda-graph cuda-full prefill mixed".split(),
)
def test_workspace_bound(name, page):
    _workspace_bound(name, page)


def _workspace_bound(name, page, kv_dtype=torch.bfloat16):
    prop = torch.cuda.get_device_properties(0)
    sms = prop.multi_processor_count
    qh, kh, dim = (12, 2, 256) if name == "split27" else (64, 8, 128)
    batch, query, kv = 1, 16, 2 * sms // kh * 128
    mode, tc, graph = "prefill", False, False
    if name == "extend":
        query, kv = 128, 2176
    elif name.startswith(("tc", "cuda")):
        mode, query, tc = "decode", 1, name.startswith("tc")
        batch, graph = 512, name.endswith(("graph", "full", "max"))
        if name == "tc-max":
            qh, kh, dim = 64, 1, 256  # largest graph scratch in the declared sweep
        elif name == "tc-eager":
            batch = 1  # fills the split grid with long KV
        elif name == "cuda-graph":
            batch = 4
        elif name == "cuda-full":
            batch = -(-(sms * (prop.max_threads_per_multi_processor // 128)) // kh)
        kv = kv if name == "tc-eager" else 512
    elif name in ("prefill", "fa3"):
        batch, query, kv, dim = 512, 128, 128, 256  # 65536 query tokens
    if name.startswith("prefill-graph"):
        mode, graph, batch, query, kv = "prefill", True, 4, 512, 4096
        if name.endswith("small"):
            query, dim = 1, 256
        elif name.endswith("padding"):
            batch, query, kv = 128, 16, 1024
        elif name.endswith("fewer"):
            # Sized for B64 x one token; a B1 graph within those bounds needs a larger tile.
            batch, query, kv = 1, 64, 4096
        elif name.endswith("sparse"):
            # Sized for B64 x 64 tokens; one token over 64 requests wraps to tile 128.
            batch, query, kv = 64, 1, 1024
    lengths = (query,) * batch
    tokens = batch * query
    if name.endswith("sparse"):
        lengths, tokens = (1,) + (0,) * (batch - 1), 64
    if name == "mixed":
        batch, tokens = 256, 576
        lengths = (288,) + (1,) * 255
        kv = 288  # Unsplit: query tiles already exceed the split occupancy grid.
    sized_batch = 64 if name.endswith(("fewer", "sparse")) else batch
    options = {
        "mode": mode,
        "backend": "fa3" if name == "fa3" else "fa2",
        "use_tensor_cores": tc,
        "use_cuda_graph": graph,
    }
    sizes = FlashInferPagedAttention.workspace_size(
        **options,
        max_batch_size=sized_batch,
        max_query_tokens=tokens,
        num_qo_heads=qh,
        num_kv_heads=kh,
        head_dim=dim,
        device=torch.device("cuda"),
    )
    if name == "cuda-full":
        assert sizes[0] == 0
    metadata = paged_metadata(lengths, (kv,) * batch, page)
    metadata["indices"] = metadata["indices"].cuda()
    buffers = {}
    if graph:
        buffers = {
            "paged_kv_indptr_buffer": torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
            "paged_kv_indices_buffer": torch.empty_like(metadata["indices"]),
            "paged_kv_last_page_len_buffer": torch.empty(batch, dtype=torch.int32, device="cuda"),
        }

    if graph and mode == "prefill":
        buffers["qo_indptr_buffer"] = torch.empty(batch + 1, dtype=torch.int32, device="cuda")

    def plan(float_bytes, int_bytes):
        operation = FlashInferPagedAttention(
            torch.zeros(float_bytes, dtype=torch.uint8, device="cuda"),
            torch.empty(int_bytes, dtype=torch.uint8, device="cuda"),
            **options,
            **buffers,
        )
        operation.plan(
            **metadata,
            num_qo_heads=qh,
            num_kv_heads=kh,
            head_dim=dim,
            page_size=page,
            pos_encoding_mode="NONE",
            dtype=torch.bfloat16,
            kv_dtype=kv_dtype,
        )
        torch.cuda.synchronize()
        return operation

    operation = plan(*sizes)
    q = torch.zeros(sum(lengths), qh, dim, dtype=torch.bfloat16, device="cuda")
    shape = (2 * metadata["indices"].numel() + 1, page, kh, dim)
    k = torch.zeros(shape, dtype=q.dtype, device="cuda")
    values = torch.arange(kh, dtype=q.dtype, device="cuda")
    v = values[None, None, :, None].expand(shape).contiguous()
    k, v = k.to(kv_dtype), v.to(kv_dtype)
    output = operation.run(q=q, paged_kv_cache=(k, v))
    expected = values.repeat_interleave(qh // kh)[None, :, None].expand_as(q)
    torch.testing.assert_close(output, expected, rtol=0, atol=0.01)
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            operation.run(q=q, paged_kv_cache=(k, v), out=output)
        captured.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0.01)
    torch.cuda.synchronize()
    del operation, q, k, v, output
    # The old fixed 128 MiB buffer still fails on the reported overflowing shapes before
    # FlashInfer 0.7.1, whose per-query-row Split-KV scratch (#5177) fits it.
    legacy = Version(version("flashinfer-python")) < Version("0.7.1")
    overflow = legacy and {
        "extend": sms >= 128,
        "split32": sms >= 128,
        "split27": sms >= 171,
        "tc-graph": True,
    }.get(name, False)
    if overflow:
        with pytest.raises(RuntimeError, match="Buffer overflow"):
            plan(128 * 1024**2, 8 * 1024**2)


@pytest.mark.parametrize(
    "name,page,kv_dtype",
    ci_cases(
        [
            "prefill-graph",
            "prefill-graph-small",
            "prefill-graph-padding",
            "prefill-graph-fewer",
            "prefill-graph-sparse",
        ],
        [1, 16],
        [torch.bfloat16, torch.float8_e4m3fn],
        core=[
            ("prefill-graph", 16, torch.bfloat16),
            ("prefill-graph-small", 1, torch.float8_e4m3fn),
            ("prefill-graph-padding", 16, torch.float8_e4m3fn),
            ("prefill-graph-fewer", 16, torch.bfloat16),
            ("prefill-graph-sparse", 16, torch.bfloat16),
        ],
    ),
)
def test_graph_prefill_workspace_bound(name, page, kv_dtype):
    _workspace_bound(name, page, kv_dtype)
