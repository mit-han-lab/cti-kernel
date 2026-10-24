# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only v2 policy, actual host launch methods, scratch ABI and arena checks."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _cute_prefill as host
from ctikernels.gdn._contract import Geometry

GEOMETRIES = [(16, 32), (16, 48), (8, 16), (8, 24)]


@pytest.mark.parametrize("hv", [16, 24, 32, 48])
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_default_compile_keys_unchanged(hv, cc):
    baseline = host.DEFAULT_TUNE | host.GEOMETRY_TUNES[(cc, (8 if hv in (16, 24) else 16, hv))]
    assert host.read_tune("", cc, (8 if hv in (16, 24) else 16, hv)) == baseline
    assert host.read_tune("v2=0", cc, (8 if hv in (16, 24) else 16, hv)) == baseline
    assert host.read_tune("v2=0,vsplit=4", cc, (8 if hv in (16, 24) else 16, hv)) == baseline
    assert host.read_tune("v2=0,v2_debug=0", cc, (8 if hv in (16, 24) else 16, hv)) == baseline
    assert host.read_tune("v2=0,v2_ldt=0", cc, (8 if hv in (16, 24) else 16, hv)) == baseline
    assert "v2" not in baseline and "vsplit" not in baseline


@pytest.mark.parametrize(
    "value",
    [
        "v2=2",
        "v2=1,v2_debug=2",
        "v2_debug=1",
        "v2=0,v2_debug=1",
        "v2=1,v2_debug=1,v2_debug=0",
        "v2=1,v2_ldt=2",
        "v2_ldt=1",
        "v2=0,v2_ldt=1",
        "v2=1,v2_ldt=1,v2_ldt=0",
        "vsplit=0",
        "vsplit=3",
        "v2=1,v2=1",
        "v2=1,overlap=1",
        "v2=1,o_split=1",
        "v2=1,form=tk",
        "v2=1,o_form=pre",
        "v2=1,solve_mma=1",
        "v2=1,split_h=1",
        "v2=1,update_split=1",
        "v2=1,aq_out=1",
        "v2=1,w_split=1",
        "v2=1,k_split=1",
        "v2=1,ut_split=1",
        "v2=1,u_store=bf16",
        "v2=1,o_bf16=1",
        "v2=1,stop_after=conv",
        "v2=1,aq_qk=0",
    ],
)
def test_rejected_tunes(value):
    with pytest.raises(ValueError):
        host.read_tune(value, (12, 0), (16, 48))


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("vsplit", [1, 2, 4])
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_tune_and_budget(hk, hv, vsplit, cc):
    tune = host.read_tune(f"v2=1,vsplit={vsplit}", cc, (hk, hv))
    assert tune["overlap"] == 0 and tune["bv"] == 32
    assert tune["aq_qk"] == int(hv // hk == 3)
    assert host.read_tune(f"v2=1,vsplit={vsplit},v2_debug=0", cc, (hk, hv)) == tune
    assert host.read_tune(f"v2=1,vsplit={vsplit},v2_ldt=0", cc, (hk, hv)) == tune
    assert host.read_tune(f"v2=1,vsplit={vsplit},v2_ldt=1", cc, (hk, hv)) == tune | {"v2_ldt": 1}
    layout = host.v2_smem_layout(hk, hv, vsplit)
    assert layout["bytes"] == max(layout["steps"].values())
    assert layout["bytes"] <= 98816 < 99 * 1024
    # Leave room even for the 1 KiB static reservation seen in v1 cubins.
    assert layout["bytes"] + 1024 <= 99 * 1024
    for captures in (1, 2):
        host.validate_captures(tune, captures)


def _launcher(hv, vsplit=1, snorm=0):
    launcher = object.__new__(host.Launcher)
    launcher.tune = host.read_tune(
        f"v2=1,vsplit={vsplit},v2_snorm={snorm}", (12, 0), (8 if hv in (16, 24) else 16, hv)
    )
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = 170
    return launcher


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_symbolic_n_and_t_compile_key(hk, hv, vsplit):
    launcher = _launcher(hv, vsplit)
    geometry = Geometry(hk, hv)
    width = (2 * hk + 2 * hv) * 128 + 2 * hv
    keys = {
        launcher.compile_key(geometry, torch.empty(t, width, device="meta"), n)
        for t, n in ((16, 1), (64, 4), (2048, 17), (8192, 257))
    }
    assert len(keys) == 1
    assert next(iter(keys))[2:4] == ("serial", False)
    baseline = next(iter(keys))
    capture_keys = {
        launcher.compile_key(geometry, torch.empty(t, width, device="meta"), n, captures)
        for t, n in ((16, 1), (2048, 17), (8192, 257))
        for captures in (0, 1, 2)
    }
    assert capture_keys == {baseline[:-1] + (c,) for c in (0, 1, 2)}


def _host_classes():
    """Strip device code/decorators; execute the real Python launch methods."""
    namespace = {"const_expr": bool, "Int32": int, "v2_smem_layout": host.v2_smem_layout}
    for filename, name, methods in (
        ("_cute_prefill_kernels.py", "Prefill", ("__init__",)),
        ("_cute_prefill_v2_kernels.py", "FusedPrefill", ("__init__", "__call__", "conv_rows_v2")),
    ):
        path = Path(host.__file__).with_name(filename)
        tree = ast.parse(path.read_text())
        cls = copy.deepcopy(
            next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        )
        cls.body = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in methods]
        for method in cls.body:
            method.decorator_list = []
            for arg in method.args.args:
                arg.annotation = None
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FusedPrefill"]


@pytest.mark.parametrize("kt", [0, 1])
def test_update_load_specializes_compile_key_and_kernel(kt):
    launcher = _launcher(32)
    baseline = launcher.compile_key(Geometry(16, 32), torch.empty(16, 12352, device="meta"), 1)
    launcher.tune = host.read_tune(f"v2=1,v2_ldt={kt}", (12, 0), (16, 32))
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    key = launcher.compile_key(Geometry(16, 32), torch.empty(16, 12352, device="meta"), 1)
    assert (key != baseline) == bool(kt)
    kernel = _host_classes()(16, 32, 12352, host.resolve_tune(launcher.tune, 32))
    assert kernel.kt == (not kt)  # kt: scalar update load unless v2_ldt=1


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("vsplit", [1, 2, 4])
@pytest.mark.parametrize("count", [1, 4, 257])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("captures", [0, 1, 2])
@pytest.mark.parametrize("snorm", [0, 1])
def test_actual_two_kernel_launch_plan(hk, hv, vsplit, count, debug, captures, snorm, tokens=8192):
    tune = host.resolve_tune(_launcher(hv, vsplit, snorm).tune, hv)
    if debug:
        tune["v2_debug"] = 1
    kernel = _host_classes()(hk, hv, 12352, tune, captures)
    assert kernel.overlap  # Only the inherited norm history-commit branch.
    assert kernel.captures == captures and kernel.bv == 32
    assert kernel.commit.captures == 0 and kernel.commit.overlap
    tensor = SimpleNamespace(iterator=object(), shape=(tokens,))
    slots = SimpleNamespace(iterator=object(), shape=(count,))
    args = [tensor] * 30
    args[6] = slots
    stream = object()
    launches = []
    second = "norm" if snorm and tokens else "commit"
    assert kernel.fnorm == (not snorm)
    assert kernel.cluster == (1 if snorm else vsplit)
    for name in ("fused_v2", second):

        def record(*pointers, name=name):
            if name == "fused_v2":
                assert pointers[-1] is (tensor.iterator if debug else None)
                assert pointers[-3] is (tensor.iterator if captures else None)
                assert pointers[-2] is (tensor.iterator if captures else None)
            elif name == "norm":
                assert pointers[-3:] == (tensor.iterator, None, None)
            else:
                assert len(pointers) == 5 and None not in pointers

            def launch(**kwargs):
                assert kwargs["stream"] is stream and not kwargs["use_pdl"]
                cluster = kwargs.get("cluster", (1, 1, 1))
                launches.append((name, kwargs["grid"], kwargs["block"][0], cluster))

            return SimpleNamespace(launch=launch)

        if name == "norm":
            setattr(kernel.commit, name, record)
        else:
            setattr(kernel, "commit_v2" if name == "commit" else name, record)
    kernel(*args, stream)
    info = host.launch_info(hk, hv, tokens, count, tune)
    assert launches == [
        (r["kernel"], tuple(r["grid"]), r["threads"], tuple(r.get("cluster", (1, 1, 1))))
        for r in info
    ]
    assert [r[0] for r in launches] == ["fused_v2", second]


@pytest.mark.parametrize("vsplit", [1, 2, 4])
@pytest.mark.parametrize("misaligned", [False, True])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("snorm", [0, 1])
def test_scratch_and_preallocated_output(vsplit, misaligned, debug, snorm):
    hk, hv, tokens, count = 8, 24, 67, 3
    geometry = Geometry(hk, hv)
    width = (2 * hk + 2 * hv) * 128 + 2 * hv
    x = torch.empty(tokens, width, dtype=torch.bfloat16)
    launcher = _launcher(hv, vsplit, snorm)
    if debug:
        launcher.tune["v2_debug"] = 1
        launcher.tune_key = tuple(sorted(launcher.tune.items()))
    calls = []

    def execute(*args):
        calls.append(args)
        args[19].fill_(1)

    launcher.compiled = {launcher.compile_key(geometry, x, count): execute}
    out = torch.empty(tokens * hv * 128 + int(misaligned), dtype=torch.bfloat16)
    out = out[int(misaligned) :].view(tokens, hv * 128)
    result = launcher(
        geometry,
        x,
        torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(hv),
        torch.empty(hv),
        torch.empty(128),
        torch.tensor([0, 1, 3, 67], dtype=torch.int32),
        torch.full((count,), -1, dtype=torch.int32),
        torch.zeros(count, dtype=torch.bool),
        torch.empty(0, hv, 128, 128),
        torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
        out=out,
    )
    assert result is out and torch.all(out == 1)
    assert len(calls) == 1
    args = calls[0]
    for i in (*range(10, 16), *range(20, 25), 28, 29):
        assert args[i].numel() == 0
    if debug:
        assert launcher.last_debug.shape == (count * hv * vsplit, 14, 64, 128)
        assert launcher.last_debug.dtype == torch.float32
        assert args[25].data_ptr() == launcher.last_debug.data_ptr()
        assert torch.count_nonzero(launcher.last_debug) == 0
    else:
        assert launcher.last_debug is None and args[25].numel() == 0
    assert args[16].numel() == count * 3 * geometry.conv_dim
    # The fused norm writes y directly: no FP32 o or norm partials in DRAM.
    assert args[17].numel() == tokens * hv * 128 * snorm and args[17].dtype == torch.float32
    assert args[18].numel() == tokens * hv * 4 * snorm and args[18].dtype == torch.float32


def test_spill_instruction_parser():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/compile_gdn_cute_prefill_v2.py"
    spec = importlib.util.spec_from_file_location("v2budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.local_instructions("""
        Function : kernel_cutlass_fused_v2_test
        /*0000*/ LDL R3, [R1];
        /*0010*/ @P0 STL.64 [R1], R4;
        /*0020*/ LDL.128 R4, [R1];
        Function : kernel_cutlass_norm_test
        /*0000*/ STL [R1], R4;
    """)
    assert records["kernel_cutlass_fused_v2_test"]["spill_loads"] == 2
    assert records["kernel_cutlass_fused_v2_test"]["spill_stores"] == 1


@pytest.mark.parametrize("retain", [False, True])
def test_v1_scratch_retention_is_opt_in(retain):
    geometry = Geometry(8, 16)
    x = torch.empty(16, geometry.conv_dim + 16 * 128 + 32, dtype=torch.bfloat16)
    launcher = object.__new__(host.Launcher)
    launcher.tune = host.read_tune("v2=0,overlap=0", (12, 0), (8, 16))
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    if retain:
        launcher.retain_scratch = True
    calls = []
    launcher.compiled = {launcher.compile_key(geometry, x, 1): lambda *args: calls.append(args)}
    launcher(
        geometry,
        x,
        torch.empty(geometry.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(16),
        torch.empty(16),
        torch.empty(128),
        torch.tensor([0, 16], dtype=torch.int32),
        torch.tensor([-1], dtype=torch.int32),
        torch.tensor([False]),
        torch.empty(0, 16, 128, 128),
        torch.empty(0, 3, geometry.conv_dim, dtype=torch.bfloat16),
    )
    if retain:
        assert tuple(launcher.last_scratch) == ("q", "k", "w", "u", "aq", "g")
        for actual, argument in zip(launcher.last_scratch.values(), calls[0][10:16], strict=True):
            assert actual is argument
        assert launcher.last_scratch["u"].dtype == torch.float32
    else:
        assert not hasattr(launcher, "last_scratch")


@pytest.mark.parametrize("captures", [0, 1, 2])
def test_compile_selects_only_fused_wrapper_without_cuda(monkeypatch, captures):
    """Exercise Launcher._compile with fake DSL entry points; never compile IR."""
    import sys
    from types import ModuleType

    cutlass, cute = ModuleType("cutlass"), ModuleType("cutlass.cute")
    runtime = ModuleType("cutlass.runtime")
    v2_module = ModuleType("ctikernels.gdn._cute_prefill_v2_kernels")
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
    v2_module.FusedPrefill = lambda hk, hv, stride, tune, captures: SimpleNamespace(
        hk=hk, hv=hv, stride=stride, tune=tune, captures=captures
    )
    calls = []

    def compile_kernel(kernel, *args, **kwargs):
        assert len(args) == 31 and args[-1] is None
        assert kernel.tune["v2"] == 1 and kernel.tune["overlap"] == 0
        assert args[17] is args[18] is cutlass.Float32
        assert kernel.captures == captures
        assert args[26] is (cutlass.Int64 if captures else cutlass.Int32)
        assert args[27] is cutlass.Int32
        calls.append(kernel)
        return kernel

    cute.compile = compile_kernel
    for name, module in (
        ("cutlass", cutlass),
        ("cutlass.cute", cute),
        ("cutlass.runtime", runtime),
        ("ctikernels.gdn._cute_prefill_v2_kernels", v2_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    launcher = _launcher(48)
    launcher.options = ""
    key = launcher.compile_key(Geometry(16, 48), torch.empty(64, 16480, device="meta"), 1, captures)
    compiled = launcher._compile(key)
    assert calls == [compiled] and not isinstance(compiled, dict)


@pytest.mark.parametrize("stride", [12352, 12353, 12360])
@pytest.mark.parametrize("x_address,history_address", [(16, 32), (18, 32), (16, 34), (16, 0)])
def test_v2_conv_alignment_dispatch(stride, x_address, history_address):
    launcher = _launcher(32)
    tune = host.resolve_tune(launcher.tune, 32)
    kernel = _host_classes()(16, 32, stride, tune)
    assert kernel.aligned_rows == (stride % 8 == 0)
    calls = []
    kernel.conv_rows = lambda *args, **kwargs: calls.append("fallback")
    kernel.conv_aligned_v2 = lambda *args, **kwargs: calls.append("vector")
    kernel.conv_rows_v2(
        SimpleNamespace(toint=lambda: x_address),
        None,
        SimpleNamespace(toint=lambda: history_address),
        None,
        None,
        0,
        0,
        64,
        0,
        0,
        1,
    )
    expected = (
        "vector"
        if stride % 8 == 0 and x_address % 16 == 0 and history_address % 16 == 0
        else "fallback"
    )
    assert calls == [expected]
    projected = torch.empty_strided((16, 12352), (stride, 1), device="meta")
    assert launcher.compile_key(Geometry(16, 32), projected, 1)[4] == stride


@pytest.mark.parametrize("snorm,vsplit", [(0, 1), (0, 2), (1, 1), (1, 2)])
def test_zero_tokens_launch_fused_and_commit(snorm, vsplit):
    test_actual_two_kernel_launch_plan(8, 16, vsplit, 3, False, 1, snorm, tokens=0)
