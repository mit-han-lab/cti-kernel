# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from ctikernels.attention.inference import (
    FlashInferPagedAttention,
    check_flash_attention_available,
    check_flashinfer_available,
    check_trtllm_available,
    flash_attention_with_kv_cache,
    select_flashinfer_prefill_implementation,
    trtllm_context_attention,
    trtllm_decode_attention,
)


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: object) -> None:
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _assert_arguments(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, (torch.Tensor, tuple)):
            assert actual[key] is value
        else:
            assert actual[key] == value


@pytest.fixture
def available_attention_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_execution(*args, **kwargs):
        pytest.fail("Availability checks must not execute native attention")

    # Only the unified decode wrapper is required, including for CUDA graphs.
    _install_module(
        monkeypatch,
        "flashinfer",
        BatchDecodeWithPagedKVCacheWrapper=unexpected_execution,
        BatchPrefillWithPagedKVCacheWrapper=unexpected_execution,
    )
    _install_module(
        monkeypatch,
        "flashinfer.utils",
        PosEncodingMode=SimpleNamespace(NONE=SimpleNamespace(value=0)),
        determine_attention_backend=unexpected_execution,
    )
    _install_module(
        monkeypatch,
        "flashinfer.decode",
        trtllm_batch_decode_with_kv_cache=unexpected_execution,
    )
    _install_module(
        monkeypatch,
        "flashinfer.prefill",
        trtllm_batch_context_with_kv_cache=unexpected_execution,
    )
    _install_module(monkeypatch, "sgl_kernel")
    _install_module(
        monkeypatch, "sgl_kernel.flash_attn", flash_attn_with_kvcache=unexpected_execution
    )


@pytest.mark.usefixtures("available_attention_providers")
@pytest.mark.parametrize(
    "check_available",
    [check_flashinfer_available, check_flash_attention_available, check_trtllm_available],
)
def test_attention_availability_accepts_required_native_imports(check_available):
    assert check_available() is None


@pytest.mark.usefixtures("available_attention_providers")
@pytest.mark.parametrize(
    ("check_available", "module_name", "symbol"),
    [
        (check_flashinfer_available, "flashinfer", "BatchDecodeWithPagedKVCacheWrapper"),
        (check_flashinfer_available, "flashinfer", "BatchPrefillWithPagedKVCacheWrapper"),
        (check_flashinfer_available, "flashinfer.utils", "PosEncodingMode"),
        (check_flashinfer_available, "flashinfer.utils", "determine_attention_backend"),
        (check_flash_attention_available, "sgl_kernel.flash_attn", "flash_attn_with_kvcache"),
        (check_trtllm_available, "flashinfer.decode", "trtllm_batch_decode_with_kv_cache"),
        (check_trtllm_available, "flashinfer.prefill", "trtllm_batch_context_with_kv_cache"),
    ],
)
def test_attention_availability_requires_each_native_symbol(
    monkeypatch: pytest.MonkeyPatch, check_available, module_name: str, symbol: str
):
    monkeypatch.delattr(sys.modules[module_name], symbol)

    with pytest.raises(ImportError, match=symbol):
        check_available()


@pytest.mark.usefixtures("available_attention_providers")
@pytest.mark.parametrize(
    ("check_available", "module_name"),
    [
        (check_flashinfer_available, "flashinfer"),
        (check_flashinfer_available, "flashinfer.utils"),
        (check_flash_attention_available, "sgl_kernel.flash_attn"),
        (check_trtllm_available, "flashinfer.decode"),
        (check_trtllm_available, "flashinfer.prefill"),
    ],
)
@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_attention_availability_propagates_native_load_failures(
    monkeypatch: pytest.MonkeyPatch, check_available, module_name: str, error_type
):
    error = error_type(f"cannot load {module_name}")
    real_import = builtins.__import__

    def import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)

    with pytest.raises(error_type) as caught:
        check_available()

    assert caught.value is error


@pytest.mark.parametrize("implementation", ["fa2", "fa3", "unknown"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flashinfer_prefill_selection_routes_non_quantized_arguments(
    monkeypatch: pytest.MonkeyPatch, implementation: str, dtype: torch.dtype
):
    calls = []
    device = torch.device("cpu")
    no_position_encoding = 73

    def determine_attention_backend(*args):
        calls.append(args)
        return implementation

    _install_module(monkeypatch, "flashinfer")
    _install_module(
        monkeypatch,
        "flashinfer.utils",
        PosEncodingMode=SimpleNamespace(NONE=SimpleNamespace(value=no_position_encoding)),
        determine_attention_backend=determine_attention_backend,
    )

    assert select_flashinfer_prefill_implementation(device, dtype) == implementation
    assert calls == [(device, no_position_encoding, False, False, dtype, dtype)]


@pytest.mark.parametrize(
    ("mode", "graph", "plan_options"),
    [
        ("prefill", False, {}),
        ("prefill", False, {"causal": False}),
        ("prefill", False, {"disable_split_kv": True}),
        ("prefill", False, {"causal": False, "disable_split_kv": True}),
        ("decode", False, {}),
        ("decode", True, {}),
    ],
)
def test_flashinfer_owns_native_plan_and_run_without_copying(
    monkeypatch: pytest.MonkeyPatch, mode: str, graph: bool, plan_options: dict
) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []
    result = torch.empty(3, 4, 8)
    # Model caller-zeroed storage already mutated by earlier native work. A new
    # wrapper must preserve reused scratch, as must subsequent plans and runs.
    float_workspace = torch.full((8,), 23, dtype=torch.uint8)
    int_workspace = torch.empty(8, dtype=torch.uint8)
    expected_workspace = float_workspace.clone()

    def use_native_workspace(workspace: torch.Tensor) -> None:
        torch.testing.assert_close(workspace, expected_workspace, rtol=0, atol=0)
        workspace.add_(1)
        expected_workspace.add_(1)

    monkeypatch.setattr("ctikernels.attention.inference.lookup", lambda *args: None)

    class NativeWrapper:
        _plan_info = [8, 1, 2, 3]

        def __init__(self, workspace, **kwargs):
            self.workspace = workspace
            use_native_workspace(workspace)
            calls.append(("construct", workspace, kwargs))

        def reset_workspace_buffer(self, float_workspace, int_workspace):
            use_native_workspace(float_workspace)
            calls.append(("reset", float_workspace, {"int_workspace": int_workspace}))

        def plan(self, **kwargs):
            use_native_workspace(self.workspace)
            calls.append(("plan", None, kwargs))

        def run(self, **kwargs):
            use_native_workspace(self.workspace)
            calls.append(("run", None, kwargs))
            return kwargs.get("out", result)

    class PrefillWrapper(NativeWrapper):
        def __init__(self, *args, **kwargs):
            assert mode == "prefill"
            super().__init__(*args, **kwargs)

    class DecodeWrapper(NativeWrapper):
        def __init__(self, *args, **kwargs):
            assert mode == "decode"
            super().__init__(*args, **kwargs)

    _install_module(
        monkeypatch,
        "flashinfer",
        BatchPrefillWithPagedKVCacheWrapper=PrefillWrapper,
        BatchDecodeWithPagedKVCacheWrapper=DecodeWrapper,
    )
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 6], dtype=torch.int32)
    indices = torch.arange(6, dtype=torch.int32)
    last_page_len = torch.ones(2, dtype=torch.int32)
    seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    graph_buffers = (
        {
            "paged_kv_indptr_buffer": cu_k,
            "paged_kv_indices_buffer": indices,
            "paged_kv_last_page_len_buffer": last_page_len,
        }
        if graph
        else {}
    )
    operation = FlashInferPagedAttention(
        float_workspace,
        int_workspace,
        mode=mode,
        backend="fa3" if mode == "prefill" else "fa2",
        use_tensor_cores=True,
        use_cuda_graph=graph,
        dispatch="provider",
        **graph_buffers,
    )
    expected_constructor = {"kv_layout": "NHD", "backend": "fa3" if mode == "prefill" else "fa2"}
    if mode == "decode":
        expected_constructor["use_tensor_cores"] = True
    if graph:
        expected_constructor.update(use_cuda_graph=True, **graph_buffers)
    assert [call[0] for call in calls] == ["construct", "reset"]
    assert calls[0][1] is float_workspace
    _assert_arguments(calls[0][2], expected_constructor)
    assert calls[1][1] is float_workspace
    assert calls[1][2]["int_workspace"] is int_workspace

    operation.plan(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        page_size=1,
        pos_encoding_mode="NONE",
        seq_lens=seq_lens,
        dtype=torch.bfloat16,
        **plan_options,
    )
    expected_plan = {
        "num_qo_heads": 4,
        "num_kv_heads": 2,
        "page_size": 1,
        "pos_encoding_mode": "NONE",
        "seq_lens": seq_lens,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "non_blocking": True,
        "sm_scale": None,
        "window_left": -1,
        "logits_soft_cap": None,
        "rope_scale": None,
        "rope_theta": None,
    }
    if mode == "decode":
        expected_plan.update(
            indptr=cu_k,
            indices=indices,
            last_page_len=last_page_len,
            head_dim=8,
            data_type=torch.bfloat16,
        )
    else:
        expected_plan.update(
            qo_indptr=cu_q,
            paged_kv_indptr=cu_k,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last_page_len,
            head_dim_qk=8,
            o_data_type=None,
            custom_mask=None,
            packed_custom_mask=None,
            causal=plan_options.get("causal", True),
            disable_split_kv=plan_options.get("disable_split_kv", False),
        )
    _assert_arguments(calls[-1][2], expected_plan)

    q = torch.empty_like(result)
    cache = (torch.empty(6, 1, 2, 8), torch.empty(6, 1, 2, 8))
    assert operation.run(q=q, paged_kv_cache=cache) is result
    _assert_arguments(calls[-1][2], {"q": q, "paged_kv_cache": cache})
    out = torch.empty_like(q)
    assert operation.run(q=q, paged_kv_cache=cache, out=out) is out
    _assert_arguments(calls[-1][2], {"q": q, "paged_kv_cache": cache, "out": out})
    assert [call[0] for call in calls] == ["construct", "reset", "plan", "run", "run"]
    torch.testing.assert_close(float_workspace, expected_workspace, rtol=0, atol=0)


@pytest.mark.usefixtures("available_attention_providers")
def test_flashinfer_prefill_graph_requires_buffers() -> None:
    with pytest.raises(ValueError, match="graph prefill requires"):
        FlashInferPagedAttention(
            torch.zeros(8, dtype=torch.uint8),
            torch.empty(8, dtype=torch.uint8),
            mode="prefill",
            backend="fa2",
            use_cuda_graph=True,
        )


def test_flash_attention_translates_only_provider_keyword_names(monkeypatch: pytest.MonkeyPatch):
    received = []
    output = torch.empty(3, 4, 8)

    def provider(**kwargs):
        received.append(kwargs)
        return output

    _install_module(monkeypatch, "sgl_kernel")
    _install_module(monkeypatch, "sgl_kernel.flash_attn", flash_attn_with_kvcache=provider)
    tensors = {
        "q": torch.empty_like(output),
        "k_cache": torch.empty(2, 4, 2, 8),
        "v_cache": torch.empty(2, 4, 2, 8),
        "page_table": torch.tensor([[0], [1]], dtype=torch.int32),
        "cache_seqlens": torch.tensor([2, 4], dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, 1, 3], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 2, 6], dtype=torch.int32),
    }
    assert (
        flash_attention_with_kv_cache(**tensors, max_seqlen_q=2, softmax_scale=0.25, version=4)
        is output
    )
    expected = dict(tensors)
    expected["cu_seqlens_k_new"] = expected.pop("cu_seqlens_k")
    expected.update(
        max_seqlen_q=2,
        softmax_scale=0.25,
        sm_margin=0,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=0,
        pack_gqa=None,
        causal=True,
        ver=4,
    )
    # Tuple options are values, unlike the borrowed K/V tuple above.
    assert received[0].pop("window_size") == expected.pop("window_size")
    _assert_arguments(received[0], expected)


@pytest.mark.parametrize("context", [True, False])
def test_trtllm_translates_metadata_without_copying(monkeypatch: pytest.MonkeyPatch, context: bool):
    received = []
    output = torch.empty(3, 4, 8, dtype=torch.bfloat16)

    def provider(**kwargs):
        received.append(kwargs)
        return output

    _install_module(monkeypatch, "flashinfer")
    _install_module(monkeypatch, "flashinfer.prefill", trtllm_batch_context_with_kv_cache=provider)
    _install_module(monkeypatch, "flashinfer.decode", trtllm_batch_decode_with_kv_cache=provider)
    q = torch.empty_like(output)
    cache = (torch.empty(2, 4, 2, 8), torch.empty(2, 4, 2, 8))
    workspace = torch.empty(8, dtype=torch.uint8)
    page_table = torch.tensor([[0], [1]], dtype=torch.int32)
    seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 6], dtype=torch.int32)
    arguments = {
        "workspace_buffer": workspace,
        "page_table": page_table,
        "cache_seqlens": seq_lens,
        "max_seqlen_k": 4,
        "softmax_scale": 0.25,
    }
    if context:
        result = trtllm_context_attention(
            q,
            cache,
            **arguments,
            max_seqlen_q=2,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            batch_size=2,
        )
    else:
        result = trtllm_decode_attention(q, cache, **arguments)
    assert result is output
    expected = {
        "query": q,
        "kv_cache": cache,
        "workspace_buffer": workspace,
        "block_tables": page_table,
        "seq_lens": seq_lens,
        "bmm1_scale": 0.25,
        "bmm2_scale": 1.0,
        "kv_layout": "NHD",
        "out_dtype": q.dtype,
    }
    if context:
        expected.update(
            max_q_len=2, max_kv_len=4, cum_seq_lens_q=cu_q, cum_seq_lens_kv=cu_k, batch_size=2
        )
    else:
        expected["max_seq_len"] = 4
    _assert_arguments(received[0], expected)


@pytest.fixture
def paged_native(monkeypatch):
    from unittest.mock import Mock

    native = Mock()
    constructor = Mock(return_value=native)
    _install_module(
        monkeypatch,
        "flashinfer",
        BatchDecodeWithPagedKVCacheWrapper=constructor,
        BatchPrefillWithPagedKVCacheWrapper=constructor,
    )
    monkeypatch.setattr("ctikernels.attention.inference.qualified_flashinfer_version", lambda: True)
    monkeypatch.setattr(
        "ctikernels.attention.inference.staged_output_flashinfer_version", lambda: True
    )
    monkeypatch.setattr(
        "ctikernels.attention._scale_cast.FP16StagedOutput",
        Mock(side_effect=lambda shape, device: Mock(device=device, stage=Mock(shape=shape))),
    )
    return constructor, native


def _paged_plan(operation, **options):
    operation.plan(
        **(
            {
                "cu_seqlens_q": torch.tensor([0, 16, 32], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 512, 1024], dtype=torch.int32),
                "indices": torch.arange(1024, dtype=torch.int32),
                "last_page_len": torch.ones(2, dtype=torch.int32),
                "seq_lens": torch.full((2,), 512, dtype=torch.int32),
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "page_size": 1,
                "pos_encoding_mode": "NONE",
                "dtype": torch.bfloat16,
            }
            | options
        )
    )


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_fp16_staging_is_confined_to_qualified_bf16_decode(
    monkeypatch, paged_native, qualified, dtype, mode
):
    from ctikernels.attention import _scale_cast

    monkeypatch.setattr(
        "ctikernels.attention.inference.staged_output_flashinfer_version", lambda: qualified
    )
    op = FlashInferPagedAttention(None, None, mode=mode, backend="fa2", dispatch="provider")
    _paged_plan(op, dtype=dtype, kv_dtype=torch.float8_e4m3fn)
    native = paged_native[1]
    active = qualified and dtype == torch.bfloat16 and mode == "decode"
    assert op._use_fp8_output == active
    if active:
        _scale_cast.FP16StagedOutput.assert_called_once_with((2, 32, 128), torch.device("cpu"))
        assert native.plan.call_args.kwargs["o_data_type"] == torch.float16
        # A second eager plan reuses the same owner and reserves at plan time.
        owner = op._fp8_output
        owner.prepare.assert_called_once_with(
            native, head_dim=128, idx_dtype=torch.int32, decode=True
        )
        _paged_plan(op, dtype=dtype, kv_dtype=torch.float8_e4m3fn)
        assert op._fp8_output is owner
        owner.reserve.assert_called_once_with((2, 32, 128))
        _paged_plan(op, dtype=dtype, kv_dtype=dtype)
        assert not op._use_fp8_output
        assert "o_data_type" not in native.plan.call_args.kwargs
    else:
        _scale_cast.FP16StagedOutput.assert_not_called()
        assert native.plan.call_args.kwargs.get("o_data_type") is None


def _graph_buffers():
    return {
        "qo_indptr_buffer": torch.empty(3, dtype=torch.int32),
        "paged_kv_indptr_buffer": torch.empty(3, dtype=torch.int32),
        "paged_kv_indices_buffer": torch.empty(1024, dtype=torch.int32),
        "paged_kv_last_page_len_buffer": torch.empty(2, dtype=torch.int32),
        "custom_mask_buffer": torch.empty(4096, dtype=torch.uint8),
        "mask_indptr_buffer": torch.empty(3, dtype=torch.int32),
    }


def test_graph_fp16_stage_keeps_shape_and_storage_owner(paged_native):
    op = FlashInferPagedAttention(
        None,
        None,
        mode="decode",
        backend="fa2",
        dispatch="provider",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(3, dtype=torch.int32),
        paged_kv_indices_buffer=torch.empty(1024, dtype=torch.int32),
        paged_kv_last_page_len_buffer=torch.empty(2, dtype=torch.int32),
    )
    _paged_plan(op, kv_dtype=torch.float8_e4m3fn)
    owner = op._fp8_output
    _paged_plan(op, kv_dtype=torch.float8_e4m3fn)
    assert op._fp8_output is owner
    owner.reserve.assert_called_once_with((2, 32, 128))
    for changes in ({"head_dim": 256}, {"dtype": torch.float16}, {"kv_dtype": torch.bfloat16}):
        with pytest.raises(RuntimeError, match="first plan"):
            _paged_plan(op, **({"kv_dtype": torch.float8_e4m3fn} | changes))


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "qo_indptr_buffer",
        "paged_kv_indptr_buffer",
        "paged_kv_indices_buffer",
        "paged_kv_last_page_len_buffer",
    ],
)
def test_graph_prefill_borrows_native_buffers(paged_native, missing):
    constructor, native = paged_native
    buffers = _graph_buffers()
    if missing is not None:
        buffers.pop(missing)
        with pytest.raises(ValueError, match=missing):
            FlashInferPagedAttention(
                None, None, mode="prefill", backend="fa2", use_cuda_graph=True, **buffers
            )
        constructor.assert_not_called()
        return
    operation = FlashInferPagedAttention(
        None,
        None,
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        dispatch="provider",
        **buffers,
    )
    for name, value in buffers.items():
        assert constructor.call_args.kwargs[name.removesuffix("buffer") + "buf"] is value
    assert constructor.call_args.kwargs["use_cuda_graph"] is True
    _paged_plan(operation)
    assert native.plan.call_args.kwargs["qo_indptr"].tolist() == [0, 16, 32]


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_paged_options_dtypes_and_scales_forwarded(paged_native, mode, kv_dtype):
    _, native = paged_native
    operation = FlashInferPagedAttention(None, None, mode=mode, backend="fa2", dispatch="provider")
    options = {
        "kv_dtype": kv_dtype,
        "sm_scale": 0.07,
        "window_left": 128,
        "logits_soft_cap": 30.0,
        "rope_scale": 2.0,
        "rope_theta": 20000.0,
    }
    if mode == "prefill":
        options.update(
            o_dtype=torch.float16,
            custom_mask=torch.ones(32, dtype=torch.bool),
            packed_custom_mask=torch.ones(4, dtype=torch.uint8),
        )
    _paged_plan(operation, **options)
    actual = native.plan.call_args.kwargs
    assert actual["q_data_type"] == torch.bfloat16
    assert actual["kv_data_type"] == kv_dtype
    if mode == "decode":
        assert actual["data_type"] == kv_dtype
    for key, value in options.items():
        translated = {"kv_dtype": "kv_data_type", "o_dtype": "o_data_type"}.get(key, key)
        assert actual[translated] is value or actual[translated] == value
    operation.run(q=None, paged_kv_cache=None, k_scale=0.5, v_scale=1.5)
    if mode == "decode" and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert actual["o_data_type"] == torch.float16
        operation._fp8_output.run.assert_called_once_with(
            native, q=None, paged_kv_cache=None, out=None, k_scale=0.5, v_scale=1.5
        )
        native.run.assert_not_called()
    else:
        native.run.assert_called_once_with(q=None, paged_kv_cache=None, k_scale=0.5, v_scale=1.5)


@pytest.mark.parametrize("mask", ["custom_mask", "packed_custom_mask"])
def test_decode_rejects_prefill_masks(paged_native, mask):
    operation = FlashInferPagedAttention(None, None, mode="decode", backend="fa2")
    with pytest.raises(ValueError, match="prefill only"):
        _paged_plan(operation, **{mask: torch.ones(1)})
    paged_native[1].plan.assert_not_called()


@pytest.mark.parametrize("mode,graph", [("decode", False), ("decode", True), ("prefill", False)])
@pytest.mark.parametrize(
    "options",
    [
        {"kv_dtype": torch.float8_e4m3fn},
        {"kv_dtype": torch.float16},
        {"dtype": torch.float32},
        {"window_left": 128},
        {"logits_soft_cap": 30.0},
        {"rope_scale": 2.0},
        {"rope_theta": 20000.0},
    ],
)
def test_off_domain_plans_use_provider(monkeypatch, paged_native, mode, graph, options):
    from unittest.mock import Mock

    forbidden = Mock(side_effect=AssertionError("off-domain dispatch"))
    for name in ("decode_uses_cuda_cores", "lookup", "prefill_query_split"):
        monkeypatch.setattr("ctikernels.attention.inference." + name, forbidden)
    operation = FlashInferPagedAttention(
        None,
        None,
        mode=mode,
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=graph,
        **(_graph_buffers() if graph else {}),
    )
    _paged_plan(operation, **options)
    assert "fixed_split_size" not in paged_native[1].plan.call_args.kwargs
    if mode == "decode":
        assert paged_native[0].call_args.kwargs["use_tensor_cores"] is True


def test_graph_decode_replan_cannot_switch_captured_route(monkeypatch, paged_native):
    monkeypatch.setattr("ctikernels.attention.inference.decode_uses_cuda_cores", lambda *a: True)
    monkeypatch.setattr(FlashInferPagedAttention, "_workspace_fits", lambda *a, **k: True)
    operation = FlashInferPagedAttention(
        torch.empty(0),
        None,
        mode="decode",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        **_graph_buffers(),
    )
    _paged_plan(operation)
    assert operation._use_tensor_cores is False
    with pytest.raises(RuntimeError, match="first plan"):
        _paged_plan(operation, kv_dtype=torch.float8_e4m3fn)
    assert paged_native[0].call_count == 1


def test_graph_decode_off_domain_first_plan_keeps_provider_route(monkeypatch, paged_native):
    monkeypatch.setattr("ctikernels.attention.inference.decode_uses_cuda_cores", lambda *a: True)
    monkeypatch.setattr(FlashInferPagedAttention, "_workspace_fits", lambda *a, **k: True)
    operation = FlashInferPagedAttention(
        torch.empty(0),
        None,
        mode="decode",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        **_graph_buffers(),
    )
    _paged_plan(operation, kv_dtype=torch.float8_e4m3fn)
    assert operation._use_tensor_cores is True


@pytest.mark.parametrize(
    "options",
    [
        {"custom_mask": torch.ones(32, dtype=torch.bool)},
        {"packed_custom_mask": torch.ones(4, dtype=torch.uint8)},
        {"o_dtype": torch.float16},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"disable_split_kv": True},
    ],
)
def test_prefill_options_disable_query_splitting(monkeypatch, paged_native, options):
    from unittest.mock import Mock

    split = Mock(side_effect=AssertionError("off-domain query split"))
    monkeypatch.setattr("ctikernels.attention.inference.prefill_query_split", split)
    operation = FlashInferPagedAttention(torch.empty(0), None, mode="prefill", backend="fa2")
    _paged_plan(operation, **options)
    split.assert_not_called()


def test_custom_sm_scale_preserves_qualified_query_dispatch(monkeypatch, paged_native):
    from unittest.mock import Mock

    split = Mock(return_value=None)
    monkeypatch.setattr("ctikernels.attention.inference.prefill_query_split", split)
    monkeypatch.setattr("ctikernels.attention.inference.device_key", lambda _: "rtx5090")
    operation = FlashInferPagedAttention(torch.empty(0), None, mode="prefill", backend="fa2")
    _paged_plan(operation, sm_scale=0.07)
    split.assert_called_once()
    assert paged_native[1].plan.call_args.kwargs["sm_scale"] == 0.07


@pytest.mark.parametrize("installed,moved", [("0.6.11.post3", True), ("0.7.0.post1", False)])
def test_custom_mask_metadata_moves_only_for_old_providers(monkeypatch, installed, moved):
    from unittest.mock import Mock

    from ctikernels.attention import _flashinfer

    monkeypatch.setattr(_flashinfer, "version", lambda _: installed)
    mask = Mock(is_cuda=True, device="cuda:0")
    values = [Mock() for _ in range(3)]
    result = _flashinfer.custom_mask_metadata(mask, *values)
    if moved:
        assert result == tuple(value.to.return_value for value in values)
        for value in values:
            value.to.assert_called_once_with("cuda:0")
    else:
        assert result == tuple(values)


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_empty_batch_plan_raises_value_error(paged_native, mode):
    operation = FlashInferPagedAttention(None, None, mode=mode, backend="fa2")
    empty = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="at least one request"):
        _paged_plan(
            operation,
            cu_seqlens_q=empty,
            cu_seqlens_k=empty,
            indices=torch.empty(0, dtype=torch.int32),
            last_page_len=torch.empty(0, dtype=torch.int32),
            seq_lens=torch.empty(0, dtype=torch.int32),
        )
    paged_native[1].plan.assert_not_called()
