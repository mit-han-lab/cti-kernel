# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU contracts for source proofs, process handoff and identical benchmark pools."""

import ast
import json
import math
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.norm_act_market import (
    RemoteCandidate,
    decode_message,
    encode_message,
    sample_us,
)
from benchmark.kernel_tuning.norm_act_pools import build_norm_pools, pool_count, pool_spec
from benchmark.kernel_tuning.norm_act_vllm import (
    RuntimeMetadata,
    equality_receipt,
    extract_kernels,
    geometry_receipt,
    normalize_kernel,
)


def toy_kernel(name="triton_example", suffix=0, expression="+ 1", metadata=1):
    # Synthetic test fixture, not vendor code.
    return f"""@triton_heuristics.pointwise(size_hints={{'x': 64}}, triton_meta={{'device': {metadata}}}, inductor_meta={{'test': {metadata}}})
@triton.jit
def {name}(in_ptr{suffix}, out_ptr{suffix}, XBLOCK: tl.constexpr):
    tmp{suffix} = tl.load(in_ptr{suffix})
    tl.store(out_ptr{suffix}, tmp{suffix} {expression})
"""


def test_source_proof_ignores_only_names_and_host_metadata():
    assert normalize_kernel(toy_kernel()) == normalize_kernel(
        toy_kernel("triton_other", 27, metadata=99)
    )
    assert normalize_kernel(toy_kernel()) != normalize_kernel(toy_kernel(expression="* 1"))
    assert normalize_kernel(toy_kernel()) != normalize_kernel(
        toy_kernel().replace("tl.load(in_ptr0)", "tl.load(in_ptr0).to(tl.bfloat16)")
    )
    assert normalize_kernel(toy_kernel()) != normalize_kernel(
        toy_kernel().replace("'x': 64", "'x': 128")
    )
    assert normalize_kernel(toy_kernel()) != normalize_kernel(
        toy_kernel().replace("tl.store(out_ptr0, tmp0 + 1)", "pass")
    )


def test_embedded_inductor_source_extraction_and_bounded_diff(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text(toy_kernel())
    wrapper = "kernel = async_compile.triton('name', " + repr(toy_kernel()) + ")"
    assert len(extract_kernels(wrapper)) == 1
    assert equality_receipt("a", [wrapper], path)["equal"]
    mismatch = equality_receipt("a", [toy_kernel(expression="* 2")], path)
    assert not mismatch["equal"] and 0 < len(mismatch["diff"]) <= 40
    assert not equality_receipt("a", [wrapper, wrapper], path)["equal"]
    with pytest.raises(ValueError, match="one JIT"):
        normalize_kernel("x = 1")


def test_runtime_device_patch_leaves_jit_body_untouched():
    source = toy_kernel().replace(
        "{'device': 1}", "{'device': DeviceProperties(type='cuda', index=7)}"
    )
    tree = ast.fix_missing_locations(RuntimeMetadata().visit(ast.parse(source)))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    original = ast.parse(source).body[0]
    assert ast.dump(ast.Module(body=fn.body, type_ignores=[])) == ast.dump(
        ast.Module(body=original.body, type_ignores=[])
    )
    assert "_current_device" in ast.unparse(tree)


def test_protocol_roundtrip_handles_newlines_and_nonfinite_error_metrics():
    message = {
        "version": 1,
        "id": 4,
        "command": "check",
        "nested": ["one\ntwo", {"rms": math.inf, "max_ulp": -math.inf}],
    }
    encoded = encode_message(message)
    assert encoded.count("\n") == 1
    assert decode_message(encoded) == message
    assert "Infinity" not in encoded
    assert json.loads(encoded)["nested"][0] == "one\ntwo"
    with pytest.raises(ValueError):
        decode_message("[]")
    with pytest.raises(ValueError):
        decode_message('{"$float":"garbage"}')


@pytest.mark.parametrize("regime", ("hot", "engine", "cold"))
@pytest.mark.parametrize("op", ("add_norm", "gemma_add_norm", "add_norm_sum"))
def test_worker_and_parent_pools_are_identical_and_restore_all_hot_inputs(regime, op):
    args = SimpleNamespace(op=op, width=16, layers=4, dtype="bf16", static_weight=0)
    spec = pool_spec(args, 3, regime, 2048)
    parent = build_norm_pools(spec, device="cpu")
    torch.randn(17)  # unrelated process RNG consumption must not affect the pools
    worker = build_norm_pools(decode_message(encode_message(spec)), device="cpu")
    for name in ("x", "residual", "weight", "h1"):
        assert torch.equal(getattr(parent, name), getattr(worker, name))
    assert parent.count == pool_count(spec) and parent.count % args.layers == 0
    assert len(parent.tensors) == (1 if regime == "hot" else parent.count)
    if regime != "hot":
        assert parent.input_bytes > 2 * spec["l2_bytes"]
    for name in ("x", "residual", "h1"):
        getattr(parent, name).fill_(42)
    parent.restore()
    for name in ("x", "residual", "h1"):
        assert torch.equal(getattr(parent, name), getattr(worker, name))
    if op.endswith("_sum"):
        assert parent.tensors[0][3].stride() == (48, 1)
    h1_pointers = {parent.previous_hidden(i).data_ptr() for i in range(parent.count)}
    assert len(h1_pointers) == (1 if regime == "hot" else parent.count)


def test_sample_receipt_math_weights_every_captured_call():
    assert sample_us([0.036, 0.072], 36) == pytest.approx(1.5)
    for times, layers in (([], 1), ([1], 0), ([math.inf], 1), ([-1], 1)):
        with pytest.raises(ValueError):
            sample_us(times, layers)


def test_geometry_5090_matches_trace_and_other_devices_only_record():
    kernels = [{"name": "test", "grid": [16, 1, 1], "block": [512, 1, 1]}]
    assert geometry_receipt("a", 16, kernels, "NVIDIA GeForce RTX 5090")["valid"]
    kernels[0]["block"] = [128, 1, 1]
    assert not geometry_receipt("a", 16, kernels, "NVIDIA GeForce RTX 5090")["valid"]
    other = geometry_receipt("a", 16, kernels, "GB10")
    assert other["valid"] and not other["checked"] and other["warning"]
    kernels[0]["grid"] = [192, 1, 1]
    assert geometry_receipt("silu", 16, kernels, "RTX 5090")["valid"]
    assert not geometry_receipt("silu", 64, kernels, "RTX 5090")["valid"]


def test_remote_candidate_rejects_nonpositive_samples():
    class Worker:
        def request(self, command, **payload):
            assert command == "sample" and payload == {"candidate": "sglang"}
            return {"us": 0}

    with pytest.raises(ValueError, match="timing"):
        RemoteCandidate(Worker(), "sglang").sample()


def test_worker_can_omit_unused_h1_without_changing_its_benchmark_inputs():
    args = SimpleNamespace(op="gemma_add_norm", width=16, layers=4, dtype="bf16", static_weight=1)
    spec = pool_spec(args, 3, "cold", 2048)
    parent = build_norm_pools(spec, device="cpu")
    worker = build_norm_pools(spec, device="cpu", need_h1=False)
    assert worker.h1.numel() == worker.saved_h1.numel() == 0
    for name in ("x", "residual", "weight"):
        assert torch.equal(getattr(parent, name), getattr(worker, name))
    worker.restore()


def test_sglang_d2_copies_updated_residual_only_on_odd_layers():
    from benchmark.kernel_tuning.norm_act_market import SglangNormCall

    def norm(x, residual, weight, eps, *, enable_pdl):
        assert eps == 1e-6 and enable_pdl is None
        residual.fill_(23)  # Distinguishes post-norm capture from a pre-norm add.
        x.fill_(-5)

    call = SglangNormCall(norm)
    buffer = torch.full((4, 48), 17.0)
    aux = buffer[:, 16:32]
    for index in (1, 0, 3, 2):
        buffer.fill_(17)
        tensors = (torch.ones(4, 16), torch.ones(4, 16), torch.ones(16), aux)
        call.call_index(tensors, None, index)
        assert (aux == (23 if index % 2 else 17)).all()
        assert (buffer[:, :16] == 17).all() and (buffer[:, 32:] == 17).all()


@pytest.mark.parametrize("fused", (False, True))
def test_local_d2_aux_capture_uses_layer_index_not_invocation_count(fused):
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall

    class Norm:
        zero_centered = False

        def __call__(self, tensors):
            x, residual, _ = tensors
            residual.add_(x)
            x.fill_(-5)
            return x, residual

        def op(self, x, residual, weight, eps, *, zero_centered, sum_out):
            torch.add(x, residual, out=sum_out)
            return self((x, residual, weight))

    call = SumNormCall(Norm(), fused=fused)
    buffer = torch.full((4, 48), 17.0)
    aux = buffer[:, 16:32]
    for index in (1, 0, 3, 2):
        buffer.fill_(17)
        tensors = (torch.ones(4, 16), torch.ones(4, 16), torch.ones(16), aux)
        call.call_index(tensors, None, index)
        assert (aux == (2 if index % 2 else 17)).all()
        assert (buffer[:, :16] == 17).all() and (buffer[:, 32:] == 17).all()


def test_vllm_d2_alternates_a_and_b_aux_with_distinct_outputs():
    from benchmark.kernel_tuning.norm_act_norm import VllmNormCall

    pools = build_norm_pools(
        pool_spec(
            SimpleNamespace(op="add_norm_sum", width=16, layers=4, dtype="bf16", static_weight=1),
            3,
            "engine",
            512,
        ),
        device="cpu",
    )
    calls = []

    class Site:
        compiled = None

        def __init__(self, name):
            self.name = name

        def __call__(self, *tensors, **kwargs):
            calls.append((self.name, tensors, kwargs))

    buffers = (torch.empty_like(pools.residual), torch.empty_like(pools.residual))
    call = VllmNormCall({name: Site(name) for name in ("a", "d2")}, pools, buffers=buffers)
    for index in (1, 0, 3, 2):
        tensors = pools.inputs[index]
        call.call_index(tensors, None, index)
        name, inputs, outputs = calls[-1]
        assert name == ("d2" if index % 2 else "a")
        assert inputs[0] is tensors[0]
        assert outputs["output"].data_ptr() != outputs["residual_output"].data_ptr()
        if index % 2:
            assert inputs[1].data_ptr() == pools.previous_hidden(index).data_ptr()
            assert inputs[2] is tensors[1] and inputs[3] is tensors[2]
            assert outputs["aux_output"] is tensors[3]
            assert outputs["aux_output"].stride() == (48, 1)
        else:
            assert "aux_output" not in outputs


@pytest.mark.parametrize("mode", ("shipped", "compiled", "auto"))
def test_d2_shipped_dispatch_and_compiled_equality_attempt(monkeypatch, tmp_path, mode):
    from benchmark.kernel_tuning import norm_act_probe as probe
    from benchmark.kernel_tuning import norm_act_vllm as vllm

    path = tmp_path / "kernel.py"
    path.write_text(
        toy_kernel().replace(
            "tl.store(out_ptr0, tmp0 + 1)",
            "tl.store(out_ptr3 + (r0_1 + 4096 * x0), tmp0 + 1)",
        )
    )
    (tmp_path / "layer_graph.wrapper.txt").write_text("synthetic wrapper")
    (tmp_path / "vllm-ir").mkdir()
    (tmp_path / "vllm-ir/layernorm.py").write_text("synthetic IR")
    monkeypatch.setattr(vllm, "site_file", lambda root, site, zero, width: path)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=123))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "RTX 5090")
    specs = []
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda spec: specs.append(spec) or {"kernels": [{"grid": [2, 1, 1], "block": [512, 1, 1]}]},
    )
    launches = []
    config = SimpleNamespace(kwargs={"XBLOCK": 1, "R0_BLOCK": 4096}, num_warps=16, num_stages=1)

    class Kernel:
        launchers = [SimpleNamespace(config=config)]

        def run(self, *args, stream):
            assert stream == 123
            launches.append(args)

    monkeypatch.setattr(vllm, "load_shipped", lambda _, **kwargs: (Kernel(), None))
    proof_calls = []
    if mode != "shipped":
        from torch._inductor import utils

        fragment = object()
        monkeypatch.setattr(vllm, "fragments", lambda *args: {"d2": fragment})
        monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *args: None)

        def compile_fn(native, **kwargs):
            assert native is fragment and kwargs["options"] == vllm.COMPILE_OPTIONS
            return native

        monkeypatch.setattr(torch, "compile", compile_fn)
        monkeypatch.setattr(utils, "run_and_get_code", lambda *args: (None, [toy_kernel()]))

        def proof(site, sources, expected):
            proof_calls.append(site)
            assert expected == path
            return {"equal": False, "diff": ["synthetic mismatch"]}

        monkeypatch.setattr(vllm, "equality_receipt", proof)

    args = SimpleNamespace(market_root=tmp_path, vllm_mode=mode, allow_vllm_mismatch=False)
    tensors = (
        *(torch.ones(2, 4096, dtype=torch.bfloat16) for _ in range(3)),
        torch.ones(4096, dtype=torch.bfloat16),
    )
    if mode == "compiled":
        with pytest.raises(vllm.MarketError, match="not proven equal"):
            vllm.VllmSite(args, "d2", tensors)
        assert proof_calls == ["d2"] and not launches
        return
    site = vllm.VllmSite(args, "d2", tensors)
    assert site.metadata["mode"] == "shipped" and site.metadata["proof"]["equal"]
    expected = [{"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 16, "num_stages": 1}]
    assert specs[0]["configs"] == specs[1]["configs"] == specs[1]["aux_configs"] == expected
    assert specs[0]["op"] == "add_norm" and specs[1]["op"] == "add_norm_sum"
    assert specs[1]["aux"] is True
    if mode == "auto":
        assert proof_calls == ["d2"] and not site.metadata["compiled_proof"]["equal"]
    out, residual, aux = (torch.empty_like(tensors[0]) for _ in range(3))
    result = site(*tensors, output=out, residual_output=residual, aux_output=aux)
    assert result[0] is out and result[1] is residual and result[2] is aux
    launch = launches[-1]
    assert all(launch[i] is value for i, value in enumerate((*tensors, residual, out, aux)))
    assert launch[-2:] == (2, 4096)
    assert site.metadata["aux_layout"]["slice_row_stride"] == 12288
    assert not site.metadata["aux_layout"]["original_jit_equal"]
    packed = torch.empty(2, 12288, dtype=torch.bfloat16)
    aux_slice = packed[:, 4096:8192]
    result = site(*tensors, aux_output=aux_slice)
    assert result[2] is aux_slice and launches[-1][6] is aux_slice


def test_d2_aux_stride_specialization_changes_only_store_address():
    from pathlib import Path

    from benchmark.kernel_tuning.norm_act_vllm import MarketError, site_file, specialize_aux_stride

    root = Path("/tmp/na1/market")
    if not root.exists():
        pytest.skip("external orchestrator market snapshot not installed")
    for zero in (False, True):
        source = site_file(root, "d2", zero).read_text()
        adapted = specialize_aux_stride(source, 12288)
        assert adapted.count("12288") == 1
        assert normalize_kernel(adapted.replace("12288", "4096")) == normalize_kernel(source)
        aux_store = next(
            node
            for node in ast.walk(ast.parse(adapted))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "tl.store"
            and ast.unparse(node.args[0]).startswith("out_ptr3 +")
        )
        # Evaluate the actual adapted address for several rows. A contiguous
        # aux address would corrupt both guards even though M=1 looks correct.
        offsets = eval(
            compile(ast.Expression(aux_store.args[0]), "<aux address>", "eval"),
            {"out_ptr3": 4096, "r0_1": torch.arange(4096)[None, :], "x0": torch.arange(4)[:, None]},
        )
        buffer = torch.full((4, 12288), 17.0)
        buffer.view(-1)[offsets] = 23
        assert (buffer[:, 4096:8192] == 23).all()
        assert (buffer[:, :4096] == 17).all() and (buffer[:, 8192:] == 17).all()
        assert (
            len(
                [
                    node
                    for node in ast.walk(ast.parse(adapted))
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.store"
                ]
            )
            == 3
        )
        with pytest.raises(MarketError, match="aux address"):
            specialize_aux_stride(adapted, 12288)
    with pytest.raises(MarketError, match="exactly one aux store"):
        specialize_aux_stride(toy_kernel(), 12288)


def test_d2_warmup_preserves_aux_stride_and_warms_both_parities():
    from benchmark.kernel_tuning.norm_act_pools import warm_norm_call

    aux = torch.full((4, 48), 17.0)[:, 16:32]
    tensors = (torch.ones(4, 16), torch.ones(4, 16), torch.ones(16), aux)
    seen = []

    class Call:
        def call_index(self, private, unused, index):
            seen.append(index)
            assert private[3].stride() == (48, 1)
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(private, tensors, strict=True))
            private[3].fill_(23)

    warm_norm_call(Call(), tensors)
    assert seen == [0, 1] and (aux == 17).all()


def test_shipped_aux_loader_exposes_specialized_source_to_inspect_only_in_memory(
    monkeypatch, tmp_path
):
    import inspect
    import linecache
    import sys

    from torch._inductor.runtime import hints

    from benchmark.kernel_tuning.norm_act_vllm import load_shipped

    # Invented kernel tests the loader without importing or compiling Triton.
    source = """import triton
@triton.jit
def triton_synthetic(out_ptr3, r0_1, x0):
    tl.store(out_ptr3 + (r0_1 + 4096 * x0), 23)
"""
    path = tmp_path / "synthetic.py"
    path.write_text(source)
    path.with_suffix(".best_config").write_text("{}")
    monkeypatch.setitem(
        sys.modules,
        "triton",
        SimpleNamespace(jit=lambda fn: SimpleNamespace(source=inspect.getsource(fn))),
    )
    monkeypatch.setattr(hints, "DeviceProperties", SimpleNamespace(create=lambda device: None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU loader test")
    monkeypatch.setattr(linecache, "cache", {})
    original, _ = load_shipped(path)
    adapted, _ = load_shipped(path, aux_stride=12288)
    assert "4096" in original.source and "12288" not in original.source
    assert "12288" in adapted.source and "4096" not in adapted.source
    assert path.read_text() == source
    assert {p.name for p in tmp_path.iterdir()} == {"synthetic.py", "synthetic.best_config"}


@pytest.mark.parametrize("summed", (False, True))
@pytest.mark.parametrize("check_enabled", (False, True))
def test_parent_row_connects_worker_checks_profiles_and_candidate_banks(
    monkeypatch, summed, check_enabled
):
    from benchmark.kernel_tuning import norm_act_market as market
    from benchmark.kernel_tuning import norm_act_norm as norm
    from benchmark.kernel_tuning import norm_act_paired as bench
    from benchmark.kernel_tuning import norm_act_pools as pools_module
    from benchmark.kernel_tuning import norm_act_vllm as vllm
    from benchmark.kernel_tuning.norm_act_reference import norm_reference

    args = SimpleNamespace(
        op="add_norm_sum" if summed else "add_norm",
        width=16,
        layers=2,
        dtype="bf16",
        static_weight=1,
        check=check_enabled,
        rounds=30,
        fi07=None,
        fi07_child=False,
        sweep=False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="cpu mock", L2_cache_size=256, major=12, minor=0, multi_processor_count=170
        ),
    )
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: "shared pool")
    monkeypatch.setattr(torch.cuda, "Stream", lambda: "shared stream")
    original_builder = pools_module.build_norm_pools
    monkeypatch.setattr(
        pools_module,
        "build_norm_pools",
        lambda spec, **kw: original_builder(spec, device="cpu", **kw),
    )

    class Local:
        provider = "test"
        zero_centered = False

        def __call__(self, triple, unused=None):
            exact, residual = norm_reference(*triple)
            triple[0].copy_(exact)
            triple[1].copy_(residual)
            return triple[:2]

        def op(self, x, residual, weight, eps, *, zero_centered, sum_out):
            sum_out.copy_(x + residual)
            return self((x, residual, weight))

    monkeypatch.setattr(
        norm,
        "norm_candidates",
        lambda *a, **kw: ({name: Local() for name in ("ours", "aa", "cute", "flashinfer")}, {}),
    )

    class Site:
        compiled = None

        def __init__(self, args, site, example, zero):
            self.site = site
            self.metadata = {"site": site}

        def __call__(self, *tensors, **kwargs):
            h, residual, weight = tensors[0], tensors[-2], tensors[-1]
            if self.site in ("b", "d2"):
                residual = (tensors[1].float() + residual.float()).to(h.dtype)
            exact, residual_out = norm_reference(h, residual, weight)
            pair = (exact.to(h.dtype), residual_out)
            if self.site == "d2":
                aux = kwargs.get("aux_output", torch.empty_like(residual_out))
                aux.copy_(residual_out)
                return *pair, aux
            return pair[0] if self.site == "a" else pair

    monkeypatch.setattr(vllm, "VllmSite", Site)
    requests = []

    class Worker:
        def __init__(self, spec):
            self.pools = original_builder(spec, device="cpu", need_h1=False)

        def request(self, command, **payload):
            requests.append(command)
            if command == "check":
                inputs = [t[:3] for t in self.pools.tensors[payload["start"] : payload["stop"]]]
                result = norm.check_norm_candidates(
                    {name: Local() for name in ("sglang", "fi071_direct", "cute_remote")},
                    inputs,
                    False,
                    record_inputs=True,
                )
                result["sum_out"] = dict.fromkeys(("sglang", "fi071_direct", "cute_remote"), True)
                return result
            if command == "profile":
                return {
                    "kernels": [
                        {"name": payload["candidate"], "grid": [1, 1, 1], "block": [128, 1, 1]}
                    ]
                }
            return {}

        def close(self):
            requests.append("exit")

    def start(args, pools, *, cute_config):
        worker = Worker(pools.spec)
        worker.pools = pools
        return worker, {
            "count": worker.pools.count,
            "distinct_weights": len(worker.pools.tensors),
            "metadata": {"version": "0.7.1"},
            "shared_buffers": True,
        }

    monkeypatch.setattr(market, "start_market", start)
    captured = []

    def capture(fn, inputs, outputs, layers, **kwargs):
        captured.append(fn)
        for i, triple in enumerate(inputs):
            if hasattr(fn, "call_index"):
                fn.call_index(triple, None, i)
            else:
                fn(triple)
        return [], []

    monkeypatch.setattr(bench, "capture_bank", capture)

    def measure(banks, rounds, layers, before):
        assert set(banks) >= {
            "ours",
            "aa",
            "cute",
            "cute_remote",
            "vllm",
            "sglang",
            "fi071_direct",
            "flashinfer",
        }
        before()
        return {name: [1.0 if name in ("ours", "aa") else 2.0] * rounds for name in banks}

    monkeypatch.setattr(bench, "measure", measure)
    row = norm.run_norm_row(args, 4, "engine")
    assert row["verdict"] == "PASS"
    assert row["profiler_clean"] is True
    assert row["us"]["fi071_direct"] == 2.0
    assert requests.count("profile") == 3 and requests[-1] == "exit"
    assert ("check" in requests) is check_enabled
    if check_enabled:
        assert row["check"]["candidate_passed"]["ours"]
        assert set(row["check"]["remote_metrics"]) == {"sglang", "fi071_direct", "cute_remote"}
    if summed:
        assert row["unfused_ratio"] == 2.0
    else:
        assert {"vllm_a", "vllm_b"} <= row["us"].keys()
