# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only policy, lazy-import and capture-key checks for the CuTe prefill adapter."""

import csv
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels._runtime.tables import DeviceClass
from ctikernels.gdn import _cute_prefill, _triton_prefill
from ctikernels.gdn.inference import _CaptureGuard

from ..support.child import run_child


def _resolve_for_count(tune, hv, count, sms):
    return _cute_prefill.resolve_tune(tune, hv, count * hv * (128 // tune["bv"]) <= sms)


@pytest.mark.parametrize(
    ("cc", "tokens", "count", "mode", "expected"),
    [
        ((12, 0), 1000, 1, None, True),
        ((12, 0), 1024, 1, None, True),
        ((12, 0), 2048, 1, None, True),
        ((12, 0), 4096, 2, None, True),
        ((12, 1), 2048, 1, None, True),
        ((12, 1), 64, 1, "1", True),
        ((12, 0), 1664, 3, "1", True),
        ((12, 0), 2048, 1, "0", False),
        ((9, 0), 2048, 1, "1", False),
    ],
)
def test_policy_uses_only_host_shape(cc, tokens, count, mode, expected):
    assert _cute_prefill.select(cc, tokens, count, mode) is expected


@pytest.mark.parametrize("installed", [False, True])
def test_package_import_does_not_import_cutlass_or_initialize_cuda(installed):
    code = """
import importlib.abc
import importlib.metadata
import sys
import torch
class NoCutlass(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('cutlass', 'nvidia_cutlass_dsl'):
            raise AssertionError('cutlass imported at package import time')
sys.meta_path.insert(0, NoCutlass())
def no_cuda(*args, **kwargs):
    raise AssertionError('CUDA initialization during import')
torch.cuda._lazy_init = no_cuda
old = importlib.metadata.version
def version(name):
    if name == 'nvidia-cutlass-dsl':
        if INSTALLED:
            return '4.8.0'
        raise importlib.metadata.PackageNotFoundError(name)
    return old(name)
importlib.metadata.version = version
import ctikernels.gdn
from ctikernels.gdn import _cute_prefill
assert _cute_prefill.available() == INSTALLED
assert 'cutlass' not in sys.modules
""".replace("INSTALLED", repr(installed))
    result = run_child(
        [sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=600
    )
    assert result.returncode == 0, result.stderr


def test_policy_is_bound_per_operation_and_capture_key_contains_implementation(monkeypatch):
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")
    monkeypatch.setattr(_cute_prefill, "load", lambda device, hv=48: object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=170),
    )
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    cute = _triton_prefill.TritonPrefill(_triton_prefill.RTX_5090, torch.device("cuda:0"))
    assert cute.provider == "cute"
    for tokens, count in ((16, 1), (64, 1), (128, 2), (511, 1), (512, 1), (2048, 4)):
        assert cute.selected_variant(tokens, count) == "cute"
    x = torch.empty(64, 12352, dtype=torch.bfloat16)
    # variant receives four weights, cumulative lengths, slots, flags, state and history.
    tensors = (torch.empty(1),) * 4 + (torch.empty(2), torch.empty(1)) + (torch.empty(1),) * 3
    cv, tv = cute.variant(x, *tensors), ("other", x.stride(0))
    assert cv[0] == "cute"
    guard = _CaptureGuard("GatedDeltaNetPrefill", torch.device("cuda:0"))
    guard.ran((64, 1), tv)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="once eagerly"):
        guard.capturing(SimpleNamespace(is_cuda=True), (64, 1), cv)
    guard.ran((64, 1), cv)
    assert guard.capturing(SimpleNamespace(is_cuda=True), (64, 1), cv)


@pytest.mark.parametrize(
    "value",
    [
        "missing",
        "unknown=1",
        "bv=8",
        "split_h=2",
        "bv=32,bv=64",
        "mma_m=16,mma_n=16",
        "recurrence_warps=8,mma_n=16",
    ],
)
def test_invalid_tune_fails_on_cpu(value):
    with pytest.raises(ValueError):
        _cute_prefill.read_tune(value)


def test_tune_bound_at_construction_and_capture_variant(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            multi_processor_count=170,
            shared_memory_per_multiprocessor=102400,
            max_threads_per_multi_processor=1536,
        ),
    )
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "SPLIT_H=1,bv=64,form=wy")
    launcher = _cute_prefill.Launcher(torch.device("cuda"), object())
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "form=tk")
    other = _cute_prefill.Launcher(torch.device("cuda"), object())
    assert launcher.tune["split_h"] == 1 and launcher.tune["bv"] == 64
    assert launcher.tune["form"] == "wy" and other.tune["form"] == "tk"
    assert launcher.tune_key != other.tune_key
    operation = object.__new__(_triton_prefill.TritonPrefill)
    operation._cute = launcher
    operation._port = None
    operation._capability, operation._cute_mode = (12, 0), "1"
    operation._hk, operation._hv, operation.kind = 16, 48, _triton_prefill.RTX_5090
    operation._triton, operation._device_class = None, DeviceClass(12, 0, 170)
    x = torch.empty(64, 12352, dtype=torch.bfloat16)
    tensors = (torch.empty(1),) * 9
    first = operation.variant(x, *tensors)
    operation._cute = other
    assert operation.variant(x, *tensors) != first


@pytest.mark.parametrize(
    "form,t_store,v_store,u_store,w_split,k_split",
    [
        ("wy", "split", "bf16", "fp32", 0, 0),
        ("wy", "split", "bf16", "bf16", 1, 1),
        ("tk", "split", "bf16", "fp32", 1, 1),
        ("tk", "fp32", "fp32", "bf16", 0, 0),
        ("tk", "bf16", "bf16", "fp32", 0, 1),
    ],
)
@pytest.mark.parametrize("bv", [16, 32, 64])
@pytest.mark.parametrize("o_bf16", [0, 1])
@pytest.mark.parametrize("merge_prepare", [0, 1])
@pytest.mark.parametrize("aq_qk", [0, 1])
@pytest.mark.parametrize("aq_out", [0, 1])
def test_scratch_contract_without_device_execution(
    form,
    t_store,
    v_store,
    u_store,
    w_split,
    k_split,
    bv,
    o_bf16,
    merge_prepare,
    aq_qk,
    aq_out,
    o_form="legacy",
    qp_split=1,
    o_split=0,
    bvo=64,
):
    """Exercise the real host allocation/signature path with a recording executable."""
    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(16, 48)
    tokens, count = 67, 3
    chunks = (tokens + 63) // 64 + count - 1
    width = (2 * 16 + 2 * 48) * 128 + 2 * 48
    x = torch.empty(tokens, width, dtype=torch.bfloat16)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune(
        f"form={form},t_store={t_store},v_store={v_store},bv={bv},o_bf16={o_bf16},"
        f"u_store={u_store},w_split={w_split},k_split={k_split},merge_prepare={merge_prepare},"
        f"aq_qk={aq_qk},aq_out={aq_out},o_form={o_form},qp_split={qp_split},"
        f"o_split={o_split},bvo={bvo}"
    )
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    calls = []
    launcher.compiled = {
        (16, 48, "serial", False, width, launcher.tune_key, 0): lambda *args: calls.append(args)
    }
    output = launcher(
        geometry,
        x,
        torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(48),
        torch.empty(48),
        torch.empty(128),
        torch.tensor([0, 1, 3, 67], dtype=torch.int32),
        torch.full((count,), -1, dtype=torch.int32),
        torch.zeros(count, dtype=torch.bool),
        torch.empty(0, 48, 128, 128),
        torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
    )
    assert output.shape == (tokens, 48 * 128)
    assert len(calls) == 1
    q, k, w, u, aq, g, history, out, partial, y, kk, qk, kl, ready, qp, op = calls[0][10:26]
    hbuf, vbuf = calls[0][28:30]
    assert hbuf.dtype == torch.bfloat16 and vbuf.dtype == torch.float32
    assert hbuf.numel() == (chunks * 48 * 128 * 128 if o_split else 0)
    assert vbuf.numel() == (chunks * 48 * 64 * 128 if o_split else 0)
    pre = o_form == "pre"
    qk_split, aq_split = pre or aq_qk, pre or aq_out
    assert qp.dtype == torch.bfloat16 and op.dtype == torch.float32
    assert qp.numel() == (chunks * 48 * 64 * 128 * (1 + qp_split) if pre else 0)
    assert op.numel() == (chunks * 48 * 64 * 128 if pre else 0)
    heads = 48 if merge_prepare else 16
    assert ready.dtype == torch.int32
    assert ready.numel() == 0
    assert all(t.dtype == torch.bfloat16 for t in (q, k, kl, aq, history, y))
    assert k.numel() == chunks * heads * 64 * 128
    assert q.numel() == k.numel() * (1 + qk_split)
    assert u.numel() == chunks * 48 * 64 * 128
    tsize = 64 * (2 if t_store == "split" else 1)
    assert w.numel() == chunks * 48 * 64 * (tsize if form == "tk" else 128 * (1 + w_split))
    assert w.dtype == (torch.float32 if form == "tk" and t_store == "fp32" else torch.bfloat16)
    assert u.dtype == (
        torch.float32 if (v_store if form == "tk" else u_store) == "fp32" else torch.bfloat16
    )
    assert kl.numel() == (k.numel() if k_split or qk_split else 0)
    assert aq.numel() == chunks * 48 * 64 * 64 * (1 + aq_split)
    assert g.dtype == torch.float32 and g.numel() == chunks * 48 * (128 if form == "tk" else 64)
    assert out.numel() == tokens * 48 * 128
    assert partial.numel() == tokens * 48 * max(4, 128 // bv)
    assert out.dtype == (torch.bfloat16 if o_bf16 else torch.float32)
    assert partial.dtype == torch.float32
    assert kk.dtype == qk.dtype == torch.float32
    assert kk.numel() == qk.numel() == chunks * heads * 64 * 64


@pytest.mark.parametrize("stop", ["conv", "norm", "gram", "solve", "wy", "all"])
@pytest.mark.parametrize("form", ["tk", "wy"])
def test_phase_launches_and_prepare_occupancy(stop, form):
    tune = _cute_prefill.read_tune(
        f"stop_after={stop},form={form},overlap=0,merge_prepare=0,prep_mode=0,rec_mode=0"
    )
    info = _cute_prefill.launch_info(16, 48, 1024, 1, tune)
    expected = 1 if stop in ("conv", "norm", "gram") else 2
    if stop == "all":
        expected += 2
    assert len(info) == expected
    assert all(item["dynamic_smem_bytes"] + 1024 <= 48 * 1024 for item in info[:2])
    if stop in ("solve", "wy", "all"):
        assert info[1]["ctas"] >= 4 * 170
    assert not any(item["kernel"] == "recurrence" for item in info) or stop == "all"


@pytest.mark.parametrize(
    "value",
    [
        "stop_after=bad",
        "o_bf16=2",
        "cluster=0",
        "form=bad",
        "t_store=bad",
        "k_split=2",
        "r_split=2",
        "v_store=bad",
        "update_split=2",
        "aq_precise=2",
        "aq_qk=2",
        "aq_out=2",
        "solve_mma=3",
        "u_store=bad",
        "ut_split=2",
        "w_split=2",
        "overlap=2",
        "overlap_norm=2",
        "merge_prepare=2",
        "rec_mode=3",
        "prep_mode=2",
        "prefetch_depth=3",
        "overlap_norm=1",
    ],
)
def test_invalid_new_knobs(value):
    with pytest.raises(ValueError):
        _cute_prefill.read_tune(value)


@pytest.mark.parametrize("bv", [16, 32, 64])
@pytest.mark.parametrize("warps", [2, 4, 8])
def test_recurrence_knob_combinations(bv, warps):
    tune = _cute_prefill.read_tune(f"bv={bv},recurrence_warps={warps}")
    assert tune["bv"] == bv and tune["recurrence_warps"] == warps


def test_one_executable_serves_multiple_token_counts():
    """Grid/scratch sizes vary while the compile cache retains the same entry."""
    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(16, 32)
    width = (2 * 16 + 2 * 32) * 128 + 2 * 32
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    calls = []
    launcher.compiled = {
        (16, 32, variant, False, width, launcher.tune_key, 0): lambda *a: calls.append(a)
        for variant in ("serial", 0)
    }
    for tokens in [64, 1000, 4096]:
        y = launcher(
            geometry,
            torch.empty(tokens, width, dtype=torch.bfloat16),
            torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
            torch.empty(32),
            torch.empty(32),
            torch.empty(128),
            torch.tensor([0, tokens], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([False]),
            torch.empty(0, 32, 128, 128),
            torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
        )
        assert y.shape == (tokens, 32 * 128)
        assert calls[-1][10].numel() == ((tokens + 63) // 64) * 16 * 64 * 128 * 2
    assert len(calls) == 3 and len(launcher.compiled) == 2


def test_wy_defaults():
    tune = _cute_prefill.read_tune("")
    assert tune["form"] == "wy" and tune["u_store"] == "fp32"
    assert all(
        tune[key] == 0 for key in ("split_h", "k_split", "update_split", "ut_split", "w_split")
    )


def test_two_recurrence_ctas_fit_sm12x_resource_budget():
    resources = _cute_prefill.resource_limits(
        {"threads": 128, "registers": 238, "smem_bytes": 46592}
    )
    assert resources == {
        "register_ctas_per_sm": 2,
        "shared_ctas_per_sm_100k": 2,
        "ctas_per_sm_100k": 2,
    }
    assert (
        _cute_prefill.resource_limits({"threads": 256, "registers": 200, "smem_bytes": 46592})[
            "ctas_per_sm_100k"
        ]
        == 1
    )


def test_removed_norm_overlap_is_rejected_before_launch():
    with pytest.raises(ValueError, match="overlap_norm"):
        _cute_prefill.read_tune("overlap_norm=1")


def test_overlap_selection_does_not_depend_on_compilation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(
            multi_processor_count=170,
            shared_memory_per_multiprocessor=102400,
            max_threads_per_multi_processor=1536,
        ),
    )
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "bv=32,overlap=1")
    launcher = _cute_prefill.Launcher(torch.device("cuda:0"), object())
    key = (16, 48, 0, False, 16480, launcher.tune_key, 0)
    other_stride = (16, 48, 0, 1, 16481, launcher.tune_key, 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a: pytest.fail("capture query")
    )
    assert launcher._effective_tune(48, key[3], key)["overlap"] == 1
    assert launcher._effective_tune(48, other_stride[3], other_stride)["overlap"] == 1


def test_blocking_launch_mode_disables_device_waits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(
            multi_processor_count=170,
            shared_memory_per_multiprocessor=102400,
            max_threads_per_multi_processor=1536,
        ),
    )
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    launcher = _cute_prefill.Launcher(torch.device("cuda:0"), object())
    assert launcher._effective_tune(32, True)["overlap"] == 0


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("norm", [0])
@pytest.mark.parametrize("merged", [0, 1])
def test_launch_geometry_is_head_fast(overlap, norm, merged):
    tune = _cute_prefill.read_tune(
        f"overlap={overlap},overlap_norm={norm},merge_prepare={merged},prep_mode=0"
    )
    records = {r["kernel"]: r for r in _cute_prefill.launch_info(16, 48, 2048, 1, tune)}
    assert ("prepare_key" in records) != bool(merged)
    assert records["prepare_value"]["grid"] == [48 * 32, 1, 1]
    assert "commit_history" not in records
    norm_name = "norm_side" if overlap and norm else "norm"
    assert records[norm_name]["ctas"] == (48 if overlap and norm else 256 * 48)


@pytest.mark.parametrize("overlap_key", [0, 1])
@pytest.mark.parametrize("captures", [0, 1])
def test_pipeline_uses_only_current_stream_and_call_local_flags(monkeypatch, overlap_key, captures):
    from ctikernels.gdn._contract import Geometry

    log, workspaces, recorded = [], [], []
    main = object()

    def forbidden(*args, **kwargs):
        pytest.fail("pipeline used a side stream, event, or record_stream")

    for name in ("Stream", "Event", "stream"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "record_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d: main)

    def reset(flags, stream):
        assert stream is main
        log.append("reset")
        flags.zero_()
        workspaces.append(flags)

    monkeypatch.setattr(_cute_prefill, "reset_flags", reset)
    geometry = Geometry(1, 2)
    width = (2 + 4) * 128 + 4
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.device = torch.device("cpu")
    launcher.tune = _cute_prefill.read_tune(f"overlap_min_tokens=0,overlap_key={overlap_key}")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    rows = torch.zeros(2, dtype=torch.int32)
    points = []

    def stage(name):
        def execute(*args):
            log.append(name)
            assert args[23] is workspaces[-1]
            if captures:
                # p = floor(L / 64) * 64 per sequence, computed on the device.
                assert args[26].tolist() == points[-1]
                assert args[27].data_ptr() == rows.data_ptr()
            if name == "pipeline":
                recorded[:] = args
                assert torch.count_nonzero(args[23]) == 0
                args[23].fill_(1)
            else:
                assert all(a is b for a, b in zip(recorded, args, strict=True))
                assert torch.all(args[23] == 1)

        return execute

    launcher.compiled = {
        (1, 2, 0, False, width, launcher.tune_key, captures): {
            name: stage(name) for name in ("pipeline", "norm")
        }
    }
    for lengths in ([1, 66], [64, 64], [65, 2]):
        log.clear()
        points.append([length // 64 * 64 for length in lengths])
        tokens = sum(lengths)
        launcher(
            geometry,
            torch.empty(tokens, width, dtype=torch.bfloat16),
            torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
            torch.empty(2),
            torch.empty(2),
            torch.empty(128),
            torch.tensor([0, lengths[0], tokens], dtype=torch.int32),
            torch.tensor([-1, -1], dtype=torch.int32),
            torch.tensor([False, False]),
            torch.empty(0, 2, 128, 128),
            torch.empty(0, 3, geometry.conv_dim),
            capture_rows=rows if captures else None,
        )
        assert log == ["reset", "pipeline", "norm"]
    assert len({id(flags) for flags in workspaces}) == 3
    assert len(launcher.compiled) == 1


def test_flag_reset_uses_async_memset_on_the_callers_stream(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType

    calls = []
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuMemsetD32Async=lambda *args: (calls.append(args) or 0,),
    )
    package = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    bindings.driver = driver
    monkeypatch.setitem(sys.modules, "cuda", package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    flags = torch.empty(17, dtype=torch.int32)
    _cute_prefill.reset_flags(flags, SimpleNamespace(cuda_stream=123))
    assert calls == [(flags.data_ptr(), 0, 17, 123)]
    driver.cuMemsetD32Async = lambda *args: (SimpleNamespace(name="CUDA_ERROR_INVALID_VALUE"),)
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_VALUE"):
        _cute_prefill.reset_flags(flags, SimpleNamespace(cuda_stream=123))


@pytest.mark.parametrize(
    "cc,hv,bv,overlap,threshold",
    [
        ((12, 0), 32, 32, 1, 2048),
        ((12, 0), 48, 32, 0, 512),
        ((12, 1), 32, 32, 1, 512),
        ((12, 1), 48, 64, 1, 2048),
    ],
)
def test_geometry_defaults_and_overrides(cc, hv, bv, overlap, threshold, monkeypatch):
    tune = _cute_prefill.read_tune("", cc, (8 if hv in (16, 24) else 16, hv))
    assert tune["bv"] == bv
    assert (tune["overlap"], tune["overlap_min_tokens"]) == (overlap, threshold)
    assert tune["overlap_key"] == 1
    assert tune["merge_prepare"] == tune["overlap_norm"] == 0
    assert tune["prepare_warps"] == 8
    assert (
        _cute_prefill.read_tune("bv=16,overlap_key=0", cc, (8 if hv in (16, 24) else 16, hv))["bv"]
        == 16
    )
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "bv=16,overlap_key=0")
    overridden = _cute_prefill.read_tune(capability=cc, geometry=(8 if hv in (16, 24) else 16, hv))
    assert overridden["bv"] == 16
    assert overridden["overlap_key"] == 0


@pytest.mark.parametrize("name,cc", _triton_prefill.DEVICE_TYPES)
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("tokens", [1023, 1024, 2047, 2048])
@pytest.mark.parametrize("count", [1, 2])
def test_explicit_product_dispatch_table(name, cc, hv, tokens, count):
    kind = _triton_prefill.DEVICE_TYPES[name, cc]
    expected = cc in ((12, 0), (12, 1))
    assert _cute_prefill.select(cc, tokens, count, None, hv, kind) == expected
    assert _cute_prefill.select(cc, tokens, count, "1", hv, kind)
    assert not _cute_prefill.select(cc, tokens, count, "0", hv, kind)


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("missing", [None, "prepare_key", "prepare_value"])
def test_offline_resources_cover_default_prepare_kernels(monkeypatch, tmp_path, hv, missing):
    import shutil

    tune = _cute_prefill.read_tune("", (12, 0), (8 if hv in (16, 24) else 16, hv))
    registers = {"prepare_value": 128, "recurrence": 168, "norm": 40}
    if hv == 48:
        registers["prepare_key"] = 106
    binary = tmp_path / "prepare.cubin"
    binary.write_bytes(b"cubin")
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/cuobjdump")

    def resource_usage(command, **kwargs):
        assert command == ["/fake/cuobjdump", "--dump-resource-usage", str(binary)]
        return SimpleNamespace(
            stdout="".join(
                f"Function kernel_cutlass_{name}_test:\n REG:{regs} SHARED:1024 STACK:0 LOCAL:0\n"
                for name, regs in registers.items()
                if name != missing
            )
        )

    monkeypatch.setattr(subprocess, "run", resource_usage)
    if missing in registers:
        with pytest.raises(RuntimeError, match="missing CuTe cubin resource data"):
            _cute_prefill.compile_info(tmp_path, 16, hv, 2048, 1, tune)
    else:
        records = _cute_prefill.compile_info(tmp_path, 16, hv, 2048, 1, tune)
        assert {r["kernel"] for r in records} == set(registers)
        assert all(
            r["registers"] == registers[r["kernel"]]
            and r["smem_bytes"] == r["dynamic_smem_bytes"] + 2048
            for r in records
        )


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("prep", [-1, 0, 1])
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("overlap_key", [0, 1])
@pytest.mark.parametrize("captures", [0, 1])
def test_compile_pipeline_or_serial_without_cuda(
    monkeypatch, blocking, small, prep, hv, overlap_key, captures, key_overrides=None
):
    """Exercise real phase compilation/control flow with a recording fake DSL."""
    from types import ModuleType

    cutlass = ModuleType("cutlass")
    cute = ModuleType("cutlass.cute")
    runtime = ModuleType("cutlass.runtime")
    cutlass.cute = cute
    for name, width in (
        ("BFloat16", 16),
        ("Float32", 32),
        ("Int8", 8),
        ("Int32", 32),
        ("Int64", 64),
    ):
        setattr(cutlass, name, SimpleNamespace(width=width))
    cute.runtime = SimpleNamespace(make_fake_tensor=lambda dtype, *a, **kw: dtype)
    cute.sym_int = lambda: 0
    runtime.make_fake_stream = lambda **kw: None
    calls = []

    def compile_kernel(kernel, *args, **kwargs):
        assert "--keep-cubin" not in kwargs["options"]
        assert kernel.captures == captures
        assert len(args) == 31  # 30 tensors and the caller stream
        assert args[-5] is (cutlass.Int64 if captures else cutlass.Int32)
        assert args[-4] is cutlass.Int32
        assert args[-3] is cutlass.BFloat16
        assert args[-2] is cutlass.Float32
        assert args[13] is (
            cutlass.Float32 if kernel.tune["u_store"] == "fp32" else cutlass.BFloat16
        )
        assert args[17] is (cutlass.BFloat16 if kernel.tune["o_bf16"] else cutlass.Float32)
        calls.append(kernel)
        return kernel

    cute.compile = compile_kernel
    for name, module in (
        ("cutlass", cutlass),
        ("cutlass.cute", cute),
        ("cutlass.runtime", runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune(f"bv=32,overlap_key={overlap_key},prep_mode={prep}")
    launcher.kernels = SimpleNamespace(
        Prefill=lambda hk, hv, stride, tune, phase, captures: SimpleNamespace(
            tune=tune.copy(), phase=phase, captures=captures
        )
    )
    launcher.options = ""
    launcher.device = torch.device("cuda:0")
    launcher.sms = 170
    launcher.launch_blocking = blocking
    key = (
        16,
        hv,
        "serial" if small else 0,
        hv * 4 <= launcher.sms,
        16480,
        tuple(sorted((launcher.tune | (key_overrides or {})).items())),
        captures,
    )
    compiled = launcher._compile(key)
    overlap = not (blocking or small)
    assert [c.phase for c in calls] == (["pipeline", "norm"] if overlap else ["all"])
    effective = compiled["pipeline"].tune if overlap else compiled.tune
    assert effective["overlap"] == int(overlap)
    assert effective == launcher._effective_tune(hv, key[3], key)
    if key_overrides:
        assert all(effective[k] == v for k, v in key_overrides.items())
    assert effective["prep_mode"] == (int(overlap and hv == 32) if prep == -1 else prep)


@pytest.mark.parametrize("sms", [48, 170, 188])
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("count", [1, 4])
def test_prefetch_budget_selected_from_recurrence_grid(sms, hv, count):
    base = _cute_prefill.read_tune(
        "", (12, 1) if sms == 48 else (12, 0), (8 if hv in (16, 24) else 16, hv)
    )
    tune = _resolve_for_count(base, hv, count, sms)
    assert tune["rec_mode"] == 2 and tune["prep_mode"] == int(hv == 32)
    expected_depth = 4 if count * hv * (128 // tune["bv"]) <= sms and tune["bv"] <= 32 else 1
    assert tune["prefetch_depth"] == expected_depth
    info = {r["kernel"]: r for r in _cute_prefill.launch_info(16, hv, 2048, count, tune)}
    assert ("prepare_key" in info) == (hv == 48)
    assert "norm_side" not in info
    assert info["prepare_value"]["grid"] == [(16 if hv == 32 else hv) * (32 + count - 1), 1, 1]
    if hv == 48:
        assert info["prepare_key"]["grid"] == [16 * (32 + count - 1), 1, 1]
    if tune["bv"] <= 32:
        assert (
            info["recurrence"]["dynamic_smem_bytes"] <= (96 if expected_depth == 4 else 48) * 1024
        )


def test_explicit_prefetch_and_precision_fallbacks():
    full = _resolve_for_count(_cute_prefill.read_tune("bv=32,prefetch_depth=4"), 48, 4, 48)
    assert full["prefetch_depth"] == 4
    with pytest.raises(ValueError, match="96 KiB"):
        _resolve_for_count(_cute_prefill.read_tune("bv=64,prefetch_depth=4"), 48, 1, 188)
    for extra, mode in (("form=tk", 0), ("k_split=1", 1), ("w_split=1", 1)):
        assert _resolve_for_count(_cute_prefill.read_tune(extra), 32, 1, 188)["rec_mode"] == mode


@pytest.mark.parametrize("sms", [48, 170, 188])
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("count", [1, 4])
def test_default_compiler_footprints_match_launch_geometry(sms, hv, count):
    arch = 121 if sms == 48 else 120
    tune = _resolve_for_count(
        _cute_prefill.read_tune(
            "", (12, 1) if sms == 48 else (12, 0), (8 if hv in (16, 24) else 16, hv)
        ),
        hv,
        count,
        sms,
    )
    snapshot = Path(__file__).with_name("gdn_cute_prefill_resources.csv")
    with snapshot.open() as source:
        rows = [
            r
            for r in csv.DictReader(source)
            if tuple(int(r[k]) for k in ("arch", "hv", "sequences", "overlap", "aq_qk", "aq_out"))
            == (arch, hv, count, tune["overlap"], tune["aq_qk"], tune["aq_out"])
            and r["o_form"] == tune["o_form"]
            and int(r["qp_split"]) == tune["qp_split"]
        ]
    by_name = {r["kernel"]: r for r in rows}
    records = _cute_prefill.launch_info(16, hv, 2048, count, tune)
    assert len(rows) == len(by_name) == len(records)
    assert set(by_name) == {r["kernel"] for r in records}
    resources = []
    for record in records:
        measured = by_name[record["kernel"]]
        assert int(measured["bv"]) == tune["bv"]
        assert int(measured["prefetch_depth"]) == tune["prefetch_depth"]
        assert record["dynamic_smem_bytes"] == int(measured["dynamic_smem_bytes"])
        resources.append(
            dict(
                record,
                registers=int(measured["registers"]),
                smem_bytes=record["dynamic_smem_bytes"] + int(measured["static_smem_bytes"]) + 1024,
            )
        )
    assert all(_cute_prefill.resource_limits(r)["ctas_per_sm_100k"] >= 1 for r in resources)


@pytest.mark.parametrize("tokens", [16, 64, 65, 511])
@pytest.mark.parametrize("preallocated", [False, True])
def test_small_pipeline_has_no_events_or_flag_storage(tokens, preallocated, monkeypatch):
    from ctikernels.gdn._contract import Geometry

    def forbidden(*args, **kwargs):
        pytest.fail("small serial path used CUDA streams/events/reset")

    for name in ("current_stream", "Stream", "Event", "stream", "is_current_stream_capturing"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(_cute_prefill, "reset_flags", forbidden)
    monkeypatch.setattr(torch.Tensor, "record_stream", forbidden)
    geometry = Geometry(1, 2)
    width = 6 * 128 + 4
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    key = (1, 2, "serial", False, width, launcher.tune_key, 0)
    calls = []
    launcher.compiled = {key: lambda *args: calls.append(args)}
    out = torch.empty(tokens, geometry.value_width, dtype=torch.bfloat16) if preallocated else None
    result = launcher(
        geometry,
        torch.empty(tokens, width, dtype=torch.bfloat16),
        torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(2),
        torch.empty(2),
        torch.empty(128),
        torch.tensor([0, 0, 1, 1, tokens], dtype=torch.int32),
        torch.full((4,), -1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.bool),
        torch.empty(0, 2, 128, 128),
        torch.empty(0, 3, geometry.conv_dim),
        out=out,
    )
    assert len(calls) == 1 and calls[0][23].numel() == 0
    assert calls[0][19].data_ptr() == result.data_ptr()
    if preallocated:
        assert result is out
    tune = launcher._effective_tune(2, key[3], key)
    assert tune["overlap"] == 0
    info = _cute_prefill.launch_info(1, 2, tokens, 4, tune)
    assert [r["kernel"] for r in info] == ["prepare_key", "prepare_value", "recurrence", "norm"]
    assert info[0]["grid"] == [_cute_prefill.chunk_capacity(tokens, 4), 1, 1]


@pytest.mark.parametrize("threshold", [0, 64, 512, 1024])
def test_overlap_threshold_is_an_exact_host_boundary(threshold):
    tune = _cute_prefill.read_tune(f"overlap_min_tokens={threshold}")
    assert _cute_prefill.pipeline_variant(threshold, tune) == 0
    assert _cute_prefill.pipeline_variant(threshold + 1, tune) == 0
    if threshold:
        assert _cute_prefill.pipeline_variant(threshold - 1, tune) == "serial"


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("stop", ["conv", "norm", "gram", "solve", "wy", "all"])
def test_pipeline_variant_requires_overlap_and_full_pipeline(overlap, stop):
    tune = _cute_prefill.read_tune(f"overlap={overlap},stop_after={stop}")
    assert _cute_prefill.pipeline_variant(16384, tune) == (
        0 if overlap and stop == "all" else "serial"
    )


def test_chunk_capacity_is_tight_for_ragged_and_empty_rows():
    # Enumerate all two-row compositions, including empty rows and chunk edges.
    for tokens in range(2, 260):
        actual = [((a + 63) // 64 + (tokens - a + 63) // 64) for a in range(tokens + 1)]
        assert _cute_prefill.chunk_capacity(tokens, 2) == max(actual)
    for count in (1, 4, 64, 129):
        for tokens in (count, count + 63, count + 64, count + 129):
            lengths = [1] * (count - 1) + [tokens - count + 1]
            assert _cute_prefill.chunk_capacity(tokens, count) == sum(
                (n + 63) // 64 for n in lengths
            )


@pytest.mark.parametrize(
    "override",
    [
        "overlap_min_tokens=-1",
        "rec_ws=2",
        "rec_ws=1,bv=64",
        "rec_ws=1,bv=32,split_h=1",
        "rec_ws=1,bv=32,w_split=1",
        "rec_ws=1,bv=32,form=tk",
        "rec_ws=1,bv=32,rec_mode=1",
    ],
)
def test_invalid_latency_knobs_fail_before_compile(override):
    with pytest.raises(ValueError):
        _cute_prefill.read_tune(override)


@pytest.mark.parametrize("bv", [16, 32])
def test_specialized_recurrence_geometry_and_resources(bv):
    tune = _resolve_for_count(_cute_prefill.read_tune(f"rec_ws=1,bv={bv}"), 32, 1, 188)
    assert tune["recurrence_warps"] == 8
    assert tune["prefetch_depth"] == 1
    records = _cute_prefill.launch_info(16, 32, 2048, 1, tune)
    rec = next(r for r in records if r["kernel"] == "recurrence")
    assert rec["threads"] == 256 and rec["ctas"] == 32 * 128 // bv
    assert rec["dynamic_smem_bytes"] == 57344 + 768 * bv + 256
    assert len(records) == 3


@pytest.mark.parametrize("prep", [-1, 0, 1])
def test_capture_key_tracks_host_selected_prepare(prep):
    from ctikernels.gdn._contract import Geometry

    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune(f"prep_mode={prep}")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = 170
    geometry = Geometry(16, 32)
    x = torch.empty(512, (2 * 16 + 2 * 32) * 128 + 2 * 32)
    pipeline = launcher.capture_variant(geometry, x, 1)
    assert pipeline == launcher.compile_key(geometry, x, 1)
    tune = launcher._effective_tune(32, pipeline[3], pipeline)
    assert (tune["prep_mode"], tune["overlap"]) == (1 if prep == -1 else prep, 1)
    guard = _CaptureGuard("prefill", torch.device("cuda:0"))
    guard.ran(tuple(x.shape), pipeline)
    assert (tuple(x.shape), launcher.capture_variant(geometry, x, 1)) in guard._eager
    serial = launcher.capture_variant(geometry, x[:128], 1)
    assert serial[2] == "serial" and serial[:2] + serial[3:] == pipeline[:2] + pipeline[3:]
    launcher.launch_blocking = True
    tune = launcher._effective_tune(32, pipeline[3], pipeline)
    assert (tune["prep_mode"], tune["overlap"]) == (0 if prep == -1 else prep, 0)


@pytest.mark.parametrize("blocking", [False, True])
def test_product_warmup_records_host_selected_prepare(monkeypatch, blocking):
    from ctikernels.gdn import inference
    from ctikernels.gdn._contract import Geometry

    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = 170
    launcher.launch_blocking = blocking
    x = torch.empty(512, 1)

    class Kernel:
        _cute = launcher

        def variant(self, *args):
            return ("cute",)

        def __call__(self, geometry, projected, *args):
            return projected

    operation = object.__new__(inference.GatedDeltaNetPrefill)
    operation._geometry = Geometry(16, 32)
    operation._device = torch.device("cpu")
    operation._kernel = Kernel()
    operation._guard = _CaptureGuard("prefill", operation._device)
    monkeypatch.setattr(inference, "check_prefill", lambda *args: (512, 1))
    args = (x, *(torch.empty(1) for _ in range(9)))
    operation(*args)
    expected = ("cute", launcher.capture_variant(operation._geometry, x, 1))
    assert operation._guard._eager == {((), expected)}

    # Simulate the capture lookup without CUDA; it must use the host-selected choice.
    def capturing(projected, shape, variant):
        assert (shape, variant) in operation._guard._eager
        return True

    monkeypatch.setattr(operation._guard, "capturing", capturing)
    operation(*args)


@pytest.mark.parametrize("bv", [16, 32, 64])
@pytest.mark.parametrize("valid", [1, 37, 64])
@pytest.mark.parametrize("aq_qk,aq_out", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_aq_independent_rounding_against_fp64(bv, valid, aq_qk, aq_out):
    """CPU arithmetic model: isolate Q/K, A storage and unweighted v_new errors.

    This checks the approximation, not GPU execution or the product precision gate.
    Includes masked tail rows, causal diagonals and both correction terms.
    """

    def split(value):
        hi = value.bfloat16().float()
        return hi, (value - hi).bfloat16().float()

    coarse_errors, precise_errors, qk_errors, coarse_qk_errors = [], [], [], []
    for seed in range(5):
        generator = torch.Generator().manual_seed(1701 + seed)
        q, k = [torch.randn(64, 128, generator=generator) for _ in range(2)]
        q, k = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
        q[valid:], k[valid:] = 0, 0
        vn = torch.randn(64, bv, generator=generator) * 3
        vn[valid:] = 0
        g = -torch.rand(64, generator=generator).cumsum(0) * 0.1
        decay = torch.exp(g[:, None] - g[None, :]).tril()
        qh, ql = split(q)
        kh, kl = split(k)
        vh, vl = split(vn)
        product = qh @ kh.T
        if aq_qk:
            product = product + ql @ kh.T + qh @ kl.T
        aq = product * decay
        ah, al = split(aq)
        precise = ah @ vh
        if aq_out:
            precise = precise + ah @ vl + al @ vh
        else:
            # A single BF16 plane is the complete recurrence operand.
            assert torch.equal(precise, aq.bfloat16().float() @ vh)
        exact_qk = q.double() @ k.double().T
        qk_errors.append((product - exact_qk).abs().max().item())
        coarse_qk_errors.append((qh @ kh.T - exact_qk).abs().max().item())
        coarse = ((qh @ kh.T) * decay).bfloat16().float() @ vh
        exact = ((q.double() @ k.double().T) * decay.double()) @ vn.double()
        # A's two planes retain exact causal zeros, including padded rows/columns.
        assert torch.count_nonzero(ah.triu(1)) == torch.count_nonzero(al.triu(1)) == 0
        assert torch.count_nonzero(precise[valid:]) == 0
        coarse_errors.append((coarse - exact).abs().max().item())
        precise_errors.append((precise - exact).abs().max().item())
    if aq_qk:
        assert sum(qk_errors) < 0.02 * sum(coarse_qk_errors)
    if aq_qk and aq_out:
        assert sum(precise_errors) < 0.02 * sum(coarse_errors)
    elif not aq_qk and not aq_out:
        assert precise_errors == coarse_errors


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("bv", [16, 32])
def test_aq_precise_preserves_shared_budget_and_mode(depth, bv):
    records = []
    for precise in (0, 1):
        tune = _resolve_for_count(
            _cute_prefill.read_tune(
                f"o_form=legacy,aq_precise={precise},prefetch_depth={depth},bv={bv}"
            ),
            48,
            1,
            170,
        )
        assert tune["rec_mode"] == 2
        records.append(_cute_prefill.launch_info(16, 48, 512, 1, tune))
    assert records[0] == records[1]
    rec = next(r for r in records[1] if r["kernel"] == "recurrence")
    assert rec["dynamic_smem_bytes"] + 2048 <= (96 if depth == 4 else 48) * 1024
    assert _cute_prefill.read_tune("")["aq_qk"] == 1
    assert _cute_prefill.read_tune("")["aq_out"] == 0


def test_precise_gb10_default_fits_compact_budget_and_keeps_explicit_bv():
    assert _cute_prefill.read_tune("", (12, 1), (16, 48))["bv"] == 64
    assert _cute_prefill.read_tune("aq_out=1", (12, 1), (16, 48))["bv"] == 32
    assert _cute_prefill.read_tune("o_form=legacy,aq_precise=0", (12, 1), (16, 48))["bv"] == 64
    assert _cute_prefill.read_tune("bv=64", (12, 1), (16, 48))["bv"] == 64


def test_offline_precision_matrix_covers_both_knobs_and_shared_budgets():
    snapshot = Path(__file__).with_name("gdn_cute_prefill_resources.csv")
    with snapshot.open() as source:
        rows = list(csv.DictReader(source))
    recs = [r for r in rows if r["kernel"] == "recurrence" and r["o_form"] == "legacy"]
    keys = {
        tuple(int(r[k]) for k in ("arch", "hv", "sequences", "overlap", "aq_qk", "aq_out"))
        for r in recs
    }
    assert len(recs) == len(keys) == 64
    assert keys == {
        (arch, hv, n, overlap, qk, out)
        for arch in (120, 121)
        for hv in (32, 48)
        for n in (1, 4)
        for overlap in (0, 1)
        for qk in (0, 1)
        for out in (0, 1)
    }
    for r in recs:
        if int(r["aq_out"]):
            full = int(r["prefetch_depth"]) == 4
            record = {
                "threads": 128,
                "registers": int(r["registers"]),
                "smem_bytes": int(r["dynamic_smem_bytes"]) + int(r["static_smem_bytes"]) + 1024,
            }
            assert record["smem_bytes"] <= (96 if full else 48) * 1024
            assert _cute_prefill.resource_limits(record)["ctas_per_sm_100k"] == (1 if full else 2)
            assert int(r["local_bytes"]) == int(r["stack_bytes"]) == 0


@pytest.mark.parametrize("qp_split", [0, 1])
@pytest.mark.parametrize("aq_qk", [0, 1])
@pytest.mark.parametrize("aq_out", [0, 1])
@pytest.mark.parametrize("bv", [16, 32, 64])
@pytest.mark.parametrize("merge_prepare", [0, 1])
@pytest.mark.parametrize("u_store", ["fp32", "bf16"])
def test_pre_scratch_contract(qp_split, aq_qk, aq_out, bv, merge_prepare, u_store):
    test_scratch_contract_without_device_execution(
        "wy",
        "split",
        "bf16",
        u_store,
        0,
        0,
        bv,
        0,
        merge_prepare,
        aq_qk,
        aq_out,
        o_form="pre",
        qp_split=qp_split,
    )


@pytest.mark.parametrize(
    "extra", ["form=tk", "rec_mode=0", "rec_mode=1", "rec_ws=1,bv=32", "w_split=1", "k_split=1"]
)
def test_pre_requires_supported_schedule_and_preserves_legacy_ablations(extra):
    assert _cute_prefill.read_tune(extra)["o_form"] == "legacy"
    with pytest.raises(ValueError, match="o_form=pre requires"):
        _cute_prefill.read_tune("o_form=pre," + extra)


@pytest.mark.parametrize("knob", ["o_form=unknown", "qp_split=2"])
def test_output_knob_validation(knob):
    with pytest.raises(ValueError):
        _cute_prefill.read_tune(knob)


def test_optional_pre_matrix_fits_shared_and_register_budgets():
    assert _cute_prefill.read_tune("")["o_form"] == "legacy"
    assert _cute_prefill.read_tune("")["qp_split"] == 1
    snapshot = Path(__file__).with_name("gdn_cute_prefill_resources.csv")
    with snapshot.open() as source:
        rows = [
            r
            for r in csv.DictReader(source)
            if r["o_form"] == "pre" and r["kernel"] == "recurrence"
        ]
    assert len(rows) == 32
    assert {
        tuple(int(r[k]) for k in ("arch", "hv", "sequences", "overlap", "qp_split")) for r in rows
    } == {
        (arch, hv, n, overlap, split)
        for arch in (120, 121)
        for hv in (32, 48)
        for n in (1, 4)
        for overlap in (0, 1)
        for split in (0, 1)
    }
    for row in rows:
        default = _cute_prefill.read_tune(
            "",
            (12, int(row["arch"][-1])),
            (8 if int(row["hv"]) in (16, 24) else 16, int(row["hv"])),
        )
        assert (int(row["aq_qk"]), int(row["aq_out"])) == (
            default["aq_qk"],
            default["aq_out"],
        )
        full = int(row["prefetch_depth"]) == 4
        record = {
            "threads": 128,
            "registers": int(row["registers"]),
            "smem_bytes": int(row["dynamic_smem_bytes"]) + int(row["static_smem_bytes"]) + 1024,
        }
        assert record["smem_bytes"] <= (96 if full else 47) * 1024
        assert _cute_prefill.resource_limits(record)["ctas_per_sm_100k"] == (1 if full else 2)
        assert int(row["stack_bytes"]) == int(row["local_bytes"]) == 0


@pytest.mark.parametrize("precise", [0, 1])
def test_aq_alias_is_canonical_and_explicit_knobs_win(precise):
    assert _cute_prefill.read_tune(f"aq_precise={precise}") == _cute_prefill.read_tune(
        f"aq_qk={precise},aq_out={precise}"
    )
    for key in ("aq_qk", "aq_out"):
        explicit = f"{key}={1 - precise}"
        first = _cute_prefill.read_tune(f"aq_precise={precise},{explicit}")
        last = _cute_prefill.read_tune(f"{explicit},aq_precise={precise}")
        assert first == last
        assert first[key] == 1 - precise
        assert first["aq_out" if key == "aq_qk" else "aq_qk"] == precise


def test_qk_only_recurrence_matches_coarse_compiler_footprint():
    snapshot = Path(__file__).with_name("gdn_cute_prefill_resources.csv")
    with snapshot.open() as source:
        recs = [
            r
            for r in csv.DictReader(source)
            if r["o_form"] == "legacy" and r["aq_out"] == "0" and r["kernel"] == "recurrence"
        ]
    rows = {tuple(r[k] for k in ("arch", "hv", "sequences", "overlap", "aq_qk")): r for r in recs}
    assert len(recs) == len(rows) == 32
    for key, row in rows.items():
        if key[-1] == "0":
            continue
        other = rows[(*key[:-1], "0")]
        for field in (
            "bv",
            "prefetch_depth",
            "dynamic_smem_bytes",
            "static_smem_bytes",
            "registers",
            "stack_bytes",
            "local_bytes",
        ):
            assert row[field] == other[field], (key, field)


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("overlap_key", [0, 1])
@pytest.mark.parametrize("prep_mode,merged", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("captures", [0, 1])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 17])
def test_prefill_pdl_launch_chain_and_kernel_boundaries(
    overlap, overlap_key, prep_mode, merged, captures, count, o_split=0, tokens=16384
):
    """Execute the actual host launch method with fake kernels, without importing CuTe."""
    import ast
    import copy

    path = Path(_cute_prefill.__file__).with_name("_cute_prefill_kernels.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prefill")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    host = copy.deepcopy(cls)
    host.body = [m for m in host.body if m.name in ("__init__", "__call__")]
    for method in host.body:
        method.decorator_list = []
        for arg in method.args.args:
            arg.annotation = None
    namespace = {"const_expr": bool, "Int32": int, "cutlass": SimpleNamespace(max=max, min=min)}
    exec(compile(ast.Module(body=[host], type_ignores=[]), str(path), "exec"), namespace)
    tune = _resolve_for_count(
        _cute_prefill.read_tune(
            f"overlap={overlap},overlap_key={overlap_key},prep_mode={prep_mode},"
            f"merge_prepare={merged},o_split={o_split}"
        ),
        48,
        8,
        170,
    )
    phases = ["pipeline", "norm"] if overlap else ["all"]
    launches = []
    stream = object()
    tensor = SimpleNamespace(iterator=object(), shape=(tokens,))
    slots = SimpleNamespace(iterator=object(), shape=(count,))
    offsets = SimpleNamespace(iterator=object())
    rows = SimpleNamespace(iterator=object())
    hbuf = SimpleNamespace(iterator=object())
    vbuf = SimpleNamespace(iterator=object())
    for phase in phases:
        kernel = namespace["Prefill"](16, 48, 16480, tune, phase, captures)
        for name in ("prepare_key", "prepare_value", "recurrence", "output", "norm"):

            def record(*args, name=name):
                if name.startswith("prepare_"):
                    assert args[-1] == count
                if name == "norm":
                    assert args[6] == count
                if name == "output":
                    assert args[-1] == count
                    assert args[4] is hbuf.iterator and args[5] is vbuf.iterator
                if name == "recurrence":
                    assert args[-3] is hbuf.iterator and args[-2] is vbuf.iterator
                    assert args[-1] == tokens
                if name in ("recurrence", "norm"):
                    offset_index, row_index = (-5, -4) if name == "recurrence" else (-2, -1)
                    assert args[offset_index] is (offsets.iterator if captures else None)
                    assert args[row_index] is (rows.iterator if captures else None)
                if name == "norm":
                    assert args[-3] is tensor.iterator

                def launch(**kwargs):
                    assert kwargs["stream"] is stream
                    if name.startswith("prepare_"):
                        heads = 16 if name == "prepare_key" or prep_mode else 48
                        assert kwargs["grid"] == (
                            heads * _cute_prefill.chunk_capacity(tokens, count),
                            1,
                            1,
                        )
                    if name == "recurrence":
                        assert kwargs["grid"] == ((128 // tune["bv"]) * 48 * count, 1, 1)
                    if name == "output":
                        assert kwargs["grid"] == (
                            _cute_prefill.chunk_capacity(tokens, count),
                            48,
                            128 // tune["bvo"],
                        )
                        assert kwargs["block"] == (32 * tune["output_warps"], 1, 1)
                    launches.append((name, kwargs["use_pdl"]))

                return SimpleNamespace(launch=launch)

            setattr(kernel, name, record)
        args = [tensor] * 26
        args[6] = slots
        kernel(*args, offsets, rows, hbuf, vbuf, stream)
    has_key = not (prep_mode or merged)
    assert launches == [
        *([("prepare_key", False)] if has_key else []),
        ("prepare_value", bool(overlap and has_key)),
        ("recurrence", bool(overlap)),
        *([("output", False)] if o_split else []),
        ("norm", False),
    ]

    # Entry/exit snippets must be outside every data-dependent branch, including
    # empty chunks and slot=-1. Execute those exact statements for each variant.
    signals = []
    arch = SimpleNamespace(
        griddepcontrol_launch_dependents=lambda: signals.append("trigger"),
        griddepcontrol_wait=lambda: signals.append("wait"),
    )
    scope = {
        "self": kernel,
        "cute": SimpleNamespace(arch=arch),
        "const_expr": bool,
        "tokens": 16384,
    }
    for name, statement, expected in (
        ("prepare_key", methods["prepare_key"].body[0], ["trigger"]),
        (
            "prepare_value",
            methods["prepare_value"].body[0],
            ["trigger", "wait"] if has_key else ["trigger"],
        ),
        ("recurrence", methods["recurrence"].body[-1], ["wait"]),
    ):
        signals.clear()
        exec(compile(ast.Module(body=[statement], type_ignores=[]), str(path), "exec"), scope)
        assert signals == (expected if overlap else []), name
    waits = [
        n
        for n in ast.walk(methods["recurrence"])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "griddepcontrol_wait"
    ]
    assert len(waits) == 1
    assert waits[0] in list(ast.walk(methods["recurrence"].body[-1]))


def test_capture_twin_compiles_with_key_and_takes_argument_pair(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(1, 2)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("overlap=0")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.device = torch.device("cpu")
    launcher.compiled = {}
    keys, arguments = [], []

    def compile_key(key):
        keys.append(key)
        return lambda *args: arguments.append(args)

    launcher._compile = compile_key
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    for captures in (0, 1, 0, 1):
        rows = torch.zeros(1, dtype=torch.int32)
        launcher(
            geometry,
            torch.empty(130, geometry.proj_width, dtype=torch.bfloat16),
            torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
            torch.empty(2),
            torch.empty(2),
            torch.empty(128),
            torch.tensor([0, 130], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([False]),
            torch.empty(3, 2, 128, 128),
            torch.empty(3, 3, geometry.conv_dim, dtype=torch.bfloat16),
            capture_rows=rows if captures else None,
        )
        if captures:
            assert arguments[-1][26].tolist() == [128]  # floor(130 / 64) * 64, on the device
            assert arguments[-1][27].data_ptr() == rows.data_ptr()
    # The first call compiles both executables of its variant.
    assert [key[-1] for key in keys] == [0, 1]
    assert all(key[:-1] == keys[0][:-1] for key in keys)


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("count", [1, 4])
def test_capture_supports_default_schedules(cc, hv, overlap, count):
    tune = _resolve_for_count(
        _cute_prefill.read_tune(f"overlap={overlap}", cc, (8 if hv in (16, 24) else 16, hv)),
        hv,
        count,
        170 if cc == (12, 0) else 48,
    )
    _cute_prefill.validate_captures(tune, 1)


@pytest.mark.parametrize(
    "value", ["rec_mode=0", "rec_ws=1,bv=32", "k_split=1", "form=tk", "o_form=pre"]
)
def test_capture_rejects_unsupported_experiments(value):
    tune = _cute_prefill.read_tune(value)
    _cute_prefill.validate_captures(tune, 0)
    with pytest.raises(ValueError, match="capture requires"):
        _cute_prefill.validate_captures(tune, 1)


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("tokens", [64, 512])
@pytest.mark.parametrize("captures", [0, 1])
@pytest.mark.parametrize("sms,one_wave", [(0, False), (100000, True)])
def test_sequence_counts_share_compile_and_capture_variant(hv, tokens, captures, sms, one_wave):
    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(16, hv)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("bv=32")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = sms
    x = torch.empty(tokens, geometry.proj_width, dtype=torch.bfloat16)
    keys = {launcher.compile_key(geometry, x, n, captures) for n in (1, 2, 3, 5, 17)}
    assert len(keys) == 1
    key = keys.pop()
    assert key == (
        16,
        hv,
        "serial" if tokens < 512 else 0,
        one_wave,
        x.stride(0),
        launcher.tune_key,
        captures,
    )
    for n in (1, 2, 3, 5, 17):
        assert launcher.capture_variant(geometry, x, n) == (*key[:-1], 0)
    assert launcher._effective_tune(hv, key[3], key)["prefetch_depth"] == (4 if one_wave else 1)


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("captures", [0, 1])
def test_sequence_counts_reuse_launcher_cache(monkeypatch, overlap, captures):
    from contextlib import nullcontext

    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(1, 2)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune(f"overlap={overlap}")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.device = torch.device("cpu")
    launcher.sms = 0
    launcher.compiled = {}
    calls = []
    launcher._compile = lambda key: lambda *args: calls.append(args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    for count in (1, 2, 3, 5, 17):
        tokens = max(512 if overlap else 0, 32 * count)
        launcher(
            geometry,
            torch.empty(tokens, geometry.proj_width, dtype=torch.bfloat16),
            torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
            torch.empty(2),
            torch.empty(2),
            torch.empty(128),
            torch.empty(count + 1, dtype=torch.int32),
            torch.full((count,), -1, dtype=torch.int32),
            torch.zeros(count, dtype=torch.bool),
            torch.empty(0, 2, 128, 128),
            torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
            capture_rows=torch.zeros(count, dtype=torch.int32) if captures else None,
        )
        assert calls[-1][6].numel() == count
        assert calls[-1][16].numel() == count * 3 * geometry.conv_dim
        assert len(launcher.compiled) == 2  # One variant: with and without captures.
    assert len(calls) == 5


@pytest.mark.parametrize("tokens", [64, 512])
def test_product_capture_warmup_is_per_compiled_variant(monkeypatch, tokens):
    from ctikernels.gdn import inference
    from ctikernels.gdn._contract import Geometry

    geometry = Geometry(1, 2)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune("bv=32")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = 16  # N=2 fits one wave; N>=3 needs a new variant.

    class Kernel:
        _cute = launcher
        provider = "cute"

        def variant(self, *args):
            return ("cute",)

        def __call__(self, geometry, projected, *args, **kwargs):
            return projected

    operation = object.__new__(inference.GatedDeltaNetPrefill)
    operation._geometry = geometry
    operation._device = torch.device("cpu")
    operation._kernel = Kernel()
    operation._guard = _CaptureGuard("prefill", operation._device)
    monkeypatch.setattr(inference, "check_prefill", lambda *a: (a[2].shape[0], a[8].numel()))

    def arguments(n):
        return (
            torch.empty(tokens + n, geometry.proj_width),
            *(torch.empty(1) for _ in range(4)),
            torch.empty(n + 1, dtype=torch.int32),
            torch.empty(n, dtype=torch.int32),
            torch.empty(n, dtype=torch.bool),
            torch.empty(1),
            torch.empty(1),
        )

    operation(*arguments(2))
    real_guard = operation._guard.capturing
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        operation._guard,
        "capturing",
        lambda x, shape, variant: real_guard(SimpleNamespace(is_cuda=True), shape, variant),
    )
    operation(*arguments(1))  # Different T and N, same one-wave executable.
    with pytest.raises(RuntimeError, match="once eagerly"):
        operation(*arguments(3))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    operation(*arguments(3))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for n in (5, 17):
        operation(*arguments(n))
    # Captures share the variant's guard entry (both executables compile together).
    operation(*arguments(2), capture_rows=torch.full((2,), -1, dtype=torch.int32))


@pytest.mark.parametrize("value", ["o_split=2", "bvo=16", "bvo=256", "output_warps=2"])
def test_output_split_invalid_knobs(value):
    with pytest.raises(ValueError):
        _cute_prefill.read_tune(value)


@pytest.mark.parametrize(
    "other",
    [
        "rec_ws=1,bv=32",
        "form=tk",
        "rec_mode=0",
        "rec_mode=1",
        "o_form=pre",
        "split_h=1",
        "w_split=1",
        "k_split=1",
    ],
)
def test_output_split_rejects_incompatible_schedules(other):
    with pytest.raises(ValueError, match="o_split=1"):
        _cute_prefill.read_tune(f"o_split=1,{other}")
    tune = _cute_prefill.read_tune(other) | {"o_split": 1}
    with pytest.raises(ValueError, match="o_split=1"):
        _cute_prefill.resolve_tune(tune, 16)


@pytest.mark.parametrize("bv", [16, 32, 64])
@pytest.mark.parametrize("bvo", [32, 64, 128])
@pytest.mark.parametrize("aq_out", [0, 1])
def test_output_split_scratch_contract(bv, bvo, aq_out):
    test_scratch_contract_without_device_execution(
        "wy",
        "split",
        "bf16",
        "fp32",
        0,
        0,
        bv,
        0,
        0,
        1,
        aq_out,
        o_split=1,
        bvo=bvo,
    )


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("bv,bvo", [(16, 64), (32, 128), (64, 32)])
@pytest.mark.parametrize("depth", [1, 4])
def test_output_split_launch_info(overlap, bv, bvo, depth):
    tune = _cute_prefill.read_tune(
        f"o_split=1,bv={bv},bvo={bvo},overlap={overlap},prefetch_depth={depth}"
    )
    if depth == 4 and bv == 64:
        for o_split in (0, 1):
            with pytest.raises(ValueError, match="bv<=32"):
                _cute_prefill.launch_info(8, 16, 2048, 3, tune | {"o_split": o_split})
        return
    fused = _cute_prefill.launch_info(8, 16, 2048, 3, tune | {"o_split": 0})
    split = _cute_prefill.launch_info(8, 16, 2048, 3, tune)
    assert len(split) == len(fused) + 1
    assert [r["kernel"] for r in split][-3:] == ["recurrence", "output", "norm"]
    assert split[-2]["grid"] == [_cute_prefill.chunk_capacity(2048, 3), 16, 128 // bvo]
    assert split[-2]["dynamic_smem_bytes"] == 24576 + 384 * bvo + 256
    assert split[-3]["dynamic_smem_bytes"] == fused[-2]["dynamic_smem_bytes"] - (
        32768 if depth == 4 else 0
    )


@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("captures", [0, 1])
def test_output_split_ordinary_launch_after_recurrence(overlap, captures):
    test_prefill_pdl_launch_chain_and_kernel_boundaries(overlap, 1, 0, 0, captures, 3, o_split=1)


def test_output_split_defaults_remain_disabled():
    assert _cute_prefill.DEFAULT_TUNE["o_split"] == 0
    assert _cute_prefill.DEFAULT_TUNE["bvo"] == 64
    assert _cute_prefill.DEFAULT_TUNE["output_warps"] == 4
    assert all(t["o_split"] == 0 for t in _cute_prefill.GEOMETRY_TUNES.values())


def _row_launcher(monkeypatch, hv=16, cc=(12, 0), sms=170):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cc)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )
    return _cute_prefill.Launcher(torch.device("cpu"), object(), (8 if hv in (16, 24) else 16, hv))


@pytest.mark.parametrize(
    "tokens,count,bvo",
    [
        (1, 1, 32),
        (99, 9, 32),
        (100, 1, 128),
        (200, 2, 128),
        (150, 3, 64),
        (201, 2, 64),
        (300, 2, 64),
        (301, 2, 64),
        (400, 99, 128),
        (10000, 99, 128),
    ],
)
def test_row_tune_bands_and_first_match(monkeypatch, tokens, count, bvo):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        (
            (100, 200, 2, {"o_split": 1, "bvo": 128}),
            (100, 300, None, {"o_split": 1, "bvo": 64}),
            (None, 99, None, {"o_split": 1, "bvo": 32}),
            (400, None, None, {"o_split": 1, "bvo": 128}),
        ),
    )
    launcher = _row_launcher(monkeypatch)
    tune, key = launcher._row_tune(tokens, count)
    assert tune["bvo"] == bvo
    assert tune["o_split"] == (tokens != 301)
    assert key == tuple(sorted(tune.items()))
    assert launcher.tune == _cute_prefill.read_tune("", (12, 0), (8, 16))
    assert tune["bv"] == launcher.tune["bv"]  # Inherit the geometry tune.
    if tokens == 301:
        assert tune is launcher.tune and key is launcher.tune_key


@pytest.mark.parametrize("value", [None, "", "o_split=0", "bvo=32"])
def test_row_tunes_bound_at_construction_and_explicit_bypass(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    else:
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", value)
    # Independent of the shipped measured table.
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        ((512, None, None, {"o_split": 1, "bvo": 128}),),
    )
    launcher = _row_launcher(monkeypatch)
    expected = _cute_prefill.read_tune(value or "", (12, 0), (8, 16))
    if not value:
        expected |= {"o_split": 1, "bvo": 128}
    # Neither subsequent environment nor table changes affect a constructed launcher.
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "o_split=1,bvo=32")
    monkeypatch.setitem(_cute_prefill.ROW_TUNES, ((12, 0), (8, 16)), ())
    tune, key = launcher._row_tune(600, 4)
    assert tune == expected and key == tuple(sorted(expected.items()))
    assert launcher._row_tune(511, 1) == (launcher.tune, launcher.tune_key)


@pytest.mark.parametrize(
    "overrides",
    [
        {"o_split": 2},
        {"unknown": 1},
        {"bvo": 256},
        {"o_split": 1, "form": "tk"},
        {"o_split": 1, "split_h": 1},
        {"o_split": 1, "mma_m": 16, "output_warps": 8, "prepare_warps": 4},
        {"rec_ws": 1, "bv": 64},
    ],
)
def test_bad_row_tune_rejected_at_construction(monkeypatch, overrides):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    # Validate even a shadowed entry that cannot be selected.
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        ((None, None, None, {}), (512, None, None, overrides)),
    )
    with pytest.raises(ValueError):
        _row_launcher(monkeypatch)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "o_split=0")
    launcher = _row_launcher(monkeypatch)
    assert launcher._row_tune(600, 1)[0]["o_split"] == 0


def test_row_tune_compile_key_wave_and_symbolic_count(monkeypatch):
    from ctikernels.gdn._contract import Geometry

    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        ((512, None, 3, {"o_split": 1, "bvo": 128, "bv": 16, "overlap_min_tokens": 512}),),
    )
    launcher = _row_launcher(monkeypatch, sms=64)
    geometry = Geometry(8, 16)
    x = torch.empty(600, geometry.proj_width)
    base = launcher.compile_key(geometry, x[:511], 1)
    selected = launcher.compile_key(geometry, x, 1)
    assert base[5] == launcher.tune_key
    assert selected[5] == launcher._row_tune(600, 1)[1] != base[5]
    assert base[3] is True and selected[3] is False  # Selected BV changes the wave class.
    assert base[2] == "serial" and selected[2] == 0
    assert launcher._effective_tune(16, selected[3], selected)["o_split"] == 1
    for n in (1, 2, 3):
        assert launcher.compile_key(geometry, x, n) == selected
        assert launcher.capture_variant(geometry, x, n) == selected
    outside = launcher.compile_key(geometry, x, 4)
    assert outside[5] == launcher.tune_key
    assert outside == launcher.compile_key(geometry, x, 17)


@pytest.mark.parametrize("tokens,count", [(511, 1), (512, 1), (600, 3), (600, 4)])
def test_row_tune_scratch_follows_selected_schedule(monkeypatch, tokens, count):
    from ctikernels.gdn._contract import Geometry

    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        (
            (
                512,
                None,
                3,
                {
                    "o_split": 1,
                    "bvo": 128,
                    "bv": 16,
                    "o_bf16": 1,
                    "aq_qk": 1,
                    "overlap_min_tokens": 512,
                },
            ),
        ),
    )
    launcher = _row_launcher(monkeypatch)
    geometry = Geometry(8, 16)
    x = torch.empty(tokens, geometry.proj_width, dtype=torch.bfloat16)
    calls = []
    key = launcher.compile_key(geometry, x, count)
    launcher.compiled[key] = lambda *args: calls.append(args)
    launcher(
        geometry,
        x,
        torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(16),
        torch.empty(16),
        torch.empty(128),
        torch.empty(count + 1, dtype=torch.int32),
        torch.empty(count, dtype=torch.int32),
        torch.zeros(count, dtype=torch.bool),
        torch.empty(0, 16, 128, 128),
        torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
    )
    split = tokens >= 512 and count <= 3
    chunks = _cute_prefill.chunk_capacity(tokens, count)
    (args,) = calls
    assert args[28].numel() == (chunks * 16 * 128 * 128 if split else 0)
    assert args[29].numel() == (chunks * 16 * 64 * 128 if split else 0)
    assert args[28].dtype == torch.bfloat16 and args[29].dtype == torch.float32
    assert args[17].dtype == (torch.bfloat16 if split else torch.float32)
    assert args[18].numel() == tokens * 16 * (8 if split else 4)
    assert args[10].numel() == chunks * 8 * 64 * 128 * (2 if split else 1)
    assert args[22].numel() == (chunks * 8 * 64 * 128 if split else 0)
    assert args[23].numel() == (chunks * 16 if split else 0)


@pytest.mark.parametrize("small", [False, True])
def test_row_tune_compilation_uses_key_schedule_and_dtypes(monkeypatch, small):
    test_compile_pipeline_or_serial_without_cuda(
        monkeypatch,
        False,
        small,
        -1,
        32,
        1,
        1,
        key_overrides={"o_split": 1, "bvo": 128, "o_bf16": 1, "u_store": "bf16"},
    )


def test_row_tune_capture_validation_uses_selected_schedule(monkeypatch):
    from ctikernels.gdn._contract import Geometry

    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        ((512, None, None, {"merge_prepare": 1}),),
    )
    launcher = _row_launcher(monkeypatch)
    geometry = Geometry(8, 16)
    with pytest.raises(ValueError, match="capture requires"):
        launcher(
            geometry,
            torch.empty(600, geometry.proj_width),
            *(torch.empty(1) for _ in range(9)),
            capture_rows=torch.empty(1, dtype=torch.int32),
        )
    # The fast path's plan has no capturing twin for such a tune; a capture raises the same.
    plan = launcher.plan(geometry, 600, 1, geometry.proj_width)
    assert plan.key is not None and plan.capture_key is None
    with pytest.raises(ValueError, match="capture requires"):
        launcher.run(plan, geometry, *(None for _ in range(11)), torch.empty(1), False)


def test_row_tune_product_warmup_tracks_selected_variant(monkeypatch):
    from ctikernels.gdn import inference
    from ctikernels.gdn._contract import Geometry

    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setitem(
        _cute_prefill.ROW_TUNES,
        ((12, 0), (8, 16)),
        ((512, None, None, {"o_split": 1, "bvo": 128}),),
    )
    launcher = _row_launcher(monkeypatch, sms=0)

    class Kernel(_triton_prefill.TritonPrefill):
        def __init__(self):
            self._cute = launcher
            self._port = None
            self._capability, self._cute_mode = (12, 0), "1"
            self._hk, self._hv, self.kind = 8, 16, _triton_prefill.RTX_5090
            self._triton, self._device_class = None, DeviceClass(12, 0, 170)

        def __call__(self, geometry, projected, *args):
            return projected

    operation = object.__new__(inference.GatedDeltaNetPrefill)
    operation._geometry = Geometry(8, 16)
    operation._device = torch.device("cpu")
    operation._kernel = Kernel()
    operation._guard = _CaptureGuard("prefill", operation._device)
    monkeypatch.setattr(inference, "check_prefill", lambda *a: (a[2].shape[0], a[8].numel()))

    def arguments(tokens, count=1):
        return (
            torch.empty(tokens, operation._geometry.proj_width),
            *(torch.empty(1) for _ in range(4)),
            torch.empty(count + 1, dtype=torch.int32),
            torch.empty(count, dtype=torch.int32),
            torch.empty(count, dtype=torch.bool),
            torch.empty(1),
            torch.empty(1),
        )

    operation(*arguments(511))
    real_guard = operation._guard.capturing
    monkeypatch.setattr(
        operation._guard,
        "capturing",
        lambda x, shape, variant: real_guard(SimpleNamespace(is_cuda=True), shape, variant),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    operation(*arguments(128))
    with pytest.raises(RuntimeError, match="once eagerly"):
        operation(*arguments(600))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    operation(*arguments(600))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    operation(*arguments(601, 2))
    assert {variant[-1][5] for _, variant in operation._guard._eager} == {
        launcher.tune_key,
        launcher._row_tune(600, 1)[1],
    }


@pytest.mark.parametrize("cc,sms", [((12, 0), 188), ((12, 0), 170), ((12, 1), 48)])
@pytest.mark.parametrize("heads", [(8, 16), (8, 24), (16, 32)])
def test_warm_up_primes_every_variant_with_monotone_cu_seqlens(monkeypatch, cc, sms, heads):
    """The engine warms tokens 1, 2, 4, ... with max_sequences = min(tokens, R); variant_keys
    pairs token and count representatives independently (count may exceed tokens + 1, e.g.
    T = 1, N = 3 at TP2 8x16 on sm120), and every priming call must be a valid batch."""
    from ctikernels.gdn._contract import Geometry
    from ctikernels.gdn.inference import _PrimeScratch

    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    launcher = _row_launcher(monkeypatch, heads[1], cc, sms)
    geometry = Geometry(*heads)
    seen = set()
    for running in (4, 16, 256):
        for tokens in (1, 2, 4, 8, 16, 64, 256, 2048):
            keys = launcher.variant_keys(
                geometry, tokens, min(tokens, running), geometry.proj_width
            )
            assert {key[-1] for key in keys} == {0, 1}  # Both executables of each variant.
            seen.update(keys.values())
    assert any(n > t + 1 for t, n in seen) == (heads == (8, 16) and cc == (12, 0))
    scratch = _PrimeScratch(geometry, torch.device("cpu"), 2048, geometry.proj_width)
    for t, n in seen:
        (_, _, _, _, _, cu, slots, initial, *_), keywords = scratch.args(t, n, 1)
        assert keywords["capture_rows"].tolist() == [-1] * n
        assert cu.shape == (n + 1,) and int(cu[0]) == 0 and int(cu[-1]) == t
        assert bool((cu[1:] >= cu[:-1]).all()), (t, n, cu.tolist())


@pytest.mark.parametrize(
    "values,rows,ok",
    [([0, 1, 1, 1], 1, True), ([0, 1, 2, 1], 1, False), ([0, 2], 1, False), ([1, 1], 1, False)],
)
def test_engine_checks_read_cu_seqlens_back(monkeypatch, values, rows, ok):
    from ctikernels.gdn._contract import check_cu_seqlens

    cu = torch.tensor(values, dtype=torch.int32)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    assert check_cu_seqlens(torch.device("cpu"), cu, rows, allow_empty=True) == len(values) - 1
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    if ok:
        check_cu_seqlens(torch.device("cpu"), cu, rows, allow_empty=True)
    else:
        with pytest.raises(ValueError, match="must rise from 0"):
            check_cu_seqlens(torch.device("cpu"), cu, rows, allow_empty=True)
