# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU contract tests; vLLM and CUDA operators are replaced only in these fixtures."""

import builtins
import runpy
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import nvfp4_parity as parity
from benchmark.kernel_tuning import nvfp4_vllm as comparator


def test_import_is_lazy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"vllm", "flashinfer", "torch"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = runpy.run_path(comparator.__file__)
    assert module["_LOADED"] is None


@pytest.mark.parametrize("name", ["sglang", "vllm", "vllm-eager"])
def test_cli_and_comparator_selection(name):
    assert parity._parser().parse_args(["--comparator", name]).comparator == name
    linear, gated = parity._comparator_classes(name)
    if name.startswith("vllm"):
        assert linear is comparator.VLLMLinear
        assert gated.func is comparator.VLLMGated
        assert gated.keywords == {"eager": name == "vllm-eager"}
    else:
        assert linear.__name__ == "SGLangLinear"
        assert gated.__name__ == "SGLangGated"


def test_cli_default_and_unknown():
    assert parity._parser().parse_args([]).comparator == "sglang"
    with pytest.raises(SystemExit):
        parity._parser().parse_args(["--comparator", "unknown"])


@pytest.fixture
def fake_api(monkeypatch):
    events = []

    class FlashInferCutlassNvFp4LinearKernel:
        pass

    class Method:
        def create_weights(self, layer, k, widths, input_size, output_size, dtype):
            events.append(("create", k, widths, input_size, output_size, dtype))
            self.kernel = FlashInferCutlassNvFp4LinearKernel()
            layer.weight = torch.empty(sum(widths), k // 2, dtype=torch.uint8)
            layer.weight_scale = torch.empty(sum(widths), k // 16)
            layer.weight_scale_2 = torch.empty(len(widths))
            layer.input_scale = torch.empty(len(widths))

        def process_weights_after_loading(self, layer):
            events.append("process")
            layer.input_global_scale_inv = layer.input_scale.max().reciprocal()

        def apply(self, layer, x, bias=None):
            assert bias is None
            events.append("apply")
            return torch.ones(x.shape[0], layer.weight.shape[0], dtype=torch.bfloat16)

    def build(config, algorithm, prefix):
        events.append(("build", config, algorithm, prefix))
        return Method()

    def allocate(m, n, device, swizzled):
        assert swizzled
        events.append("allocate")
        return torch.zeros(m, n // 2, dtype=torch.uint8), torch.zeros(
            128, n // 64, dtype=torch.int32
        )

    def fused(packed, scales, gate_up, inverse):
        events.append("fused")
        assert gate_up.shape[1] == packed.shape[1] * 4
        torch.testing.assert_close(inverse, torch.tensor([4.0]))

    def activation(out, gate_up):
        events.append("activation")
        out.fill_(2)

    def quantize(x, inverse, swizzled, *, backend):
        events.append("quantize")
        assert swizzled and backend == "flashinfer-cutlass"
        torch.testing.assert_close(inverse, torch.tensor([4.0]))
        assert (x == 2).all()
        packed, scales = allocate(*x.shape, x.device, True)
        return packed, scales.view(torch.float8_e4m3fn)

    api = SimpleNamespace(
        torch=torch,
        config=lambda **kwargs: kwargs,
        build=build,
        kernel=FlashInferCutlassNvFp4LinearKernel,
        ops=SimpleNamespace(create_fp4_output_tensors=allocate, scaled_fp4_quant=quantize),
    )
    monkeypatch.setattr(comparator, "_LOADED", api)
    monkeypatch.setattr(
        comparator.md,
        "version",
        lambda name: {
            "vllm": "0.31.0",
            "flashinfer-python": "0.7.0.post1",
        }[name],
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul_nvfp4_quant", fused, raising=False)
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", activation, raising=False)
    return api, events


def weight(global_scale=0.5):
    return SimpleNamespace(
        data=torch.full((128, 32), 17, dtype=torch.uint8),
        scale=torch.ones(128, 4),
        global_scale=torch.tensor([global_scale]),
    )


def test_real_method_lifecycle_and_version_recording(fake_api):
    _, events = fake_api
    w = weight()
    op = comparator.VLLMLinear((w,), torch.tensor([0.125]))
    assert events[0] == (
        "build",
        {
            "quant_method": "NVFP4",
            "is_checkpoint_nvfp4_serialized": True,
            "group_size": 16,
        },
        "NVFP4",
        "model.layers.0.mlp.down_proj",
    )
    assert events[1] == ("create", 64, [128], 64, 128, torch.bfloat16)
    assert events[2] == "process"
    torch.testing.assert_close(op.layer.weight, w.data)
    torch.testing.assert_close(op.layer.weight_scale, w.scale)
    torch.testing.assert_close(op.layer.weight_scale_2, w.global_scale)
    with pytest.raises(AssertionError, match="Probe every M"):
        op.metadata(8)
    op.probe(torch.ones(8, 64, dtype=torch.bfloat16))
    metadata = op.metadata(8)
    assert metadata["vllm_version"] == "0.31.0"
    assert metadata["flashinfer_version"] == "0.7.0.post1"
    assert metadata["kernel_class"] == "FlashInferCutlassNvFp4LinearKernel"
    events.clear()
    op(torch.ones(8, 64, dtype=torch.bfloat16))
    assert events == ["apply"]  # no preparation in the timed boundary


@pytest.mark.parametrize("eager", [False, True])
def test_gate_boundary_and_output_layout(fake_api, eager):
    _, events = fake_api
    op = comparator.VLLMGated(
        weight(), weight(), torch.tensor([0.125]), torch.tensor([0.25]), eager=eager, shared=True
    )
    assert events[0][-1].endswith("gate_up_proj")
    events.clear()
    packed, scales = op.probe(torch.ones(8, 64, dtype=torch.bfloat16))
    assert packed.shape == (8, 64)
    assert scales.shape == (128, 8) and scales.dtype == torch.float8_e4m3fn
    assert events == (
        ["apply", "activation", "quantize", "allocate"] if eager else ["apply", "allocate", "fused"]
    )
    assert op.metadata(8)["activation_path"] == (
        "eager-separate" if eager else "compiled-serving-fused-op"
    )


def test_rejects_unequal_globals_and_wrong_kernel(fake_api):
    api, _ = fake_api
    with pytest.raises(AssertionError, match="equal gate/up"):
        comparator.VLLMGated(weight(), weight(1), torch.ones(1), torch.ones(1))
    api.kernel = type("UnexpectedKernel", (), {})
    with pytest.raises(AssertionError, match="Expected FlashInferCutlass"):
        comparator.VLLMLinear((weight(),), torch.ones(1))


@pytest.mark.parametrize("name", ["sglang", "vllm", "vllm-eager"])
def test_noise_uses_comparator_self_pairs(monkeypatch, name):
    graphs = [("ours", 1), ("reference", 1), ("control", 1)]
    calls = []

    def pairs(selected, count, replays, deadline):
        calls.append(selected)
        return ([10, 10], [20, 20]) if len(calls) == 1 else ([20, 20], [20, 20])

    monkeypatch.setattr(parity, "_pairs", pairs)
    result = parity._measure(graphs, 2, 5, comparator=name)
    assert calls == [graphs[:2], graphs[1:]]
    assert result["eps"] == 0 and result["speedup"] == 2
    assert result[f"{name}_us"] == 20
    assert result["samples_us"][name] == [20, 20]
    if name != "sglang":
        assert "sglang_us" not in result


def test_load_rejects_wrong_version_before_framework_import(monkeypatch):
    monkeypatch.setattr(comparator, "_LOADED", None)
    monkeypatch.setattr(comparator.md, "version", lambda _: "wrong")
    with pytest.raises(RuntimeError, match="vllm==0.31.0"):
        comparator.load()


def test_load_uses_installed_methods_lazily(monkeypatch):
    config, build, kernel = object(), object(), object()
    ops = object()
    imports = []
    original = builtins.__import__
    modules = {
        "vllm": SimpleNamespace(_custom_ops=ops),
        "vllm.model_executor.kernels.linear.nvfp4.flashinfer": SimpleNamespace(
            FlashInferCutlassNvFp4LinearKernel=kernel
        ),
        "vllm.model_executor.layers.quantization.modelopt": SimpleNamespace(
            ModelOptNvFp4Config=config, build_linear_method=build
        ),
    }

    def importing(name, *args, **kwargs):
        if name in modules:
            imports.append(name)
            return modules[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(comparator, "_LOADED", None)
    monkeypatch.setattr(
        comparator.md,
        "version",
        lambda name: {
            "vllm": "0.31.0",
            "flashinfer-python": "0.7.0.post1",
        }[name],
    )
    groups = []
    monkeypatch.setattr(comparator, "_single_process_groups", lambda: groups.append(1))
    monkeypatch.setattr(builtins, "__import__", importing)
    api = comparator.load()
    assert (api.config, api.build, api.kernel, api.ops) == (config, build, kernel, ops)
    assert comparator.load() is api
    assert len(imports) == 3 and groups == [1]


@pytest.mark.parametrize("name", ["vllm", "vllm-eager"])
def test_main_records_selected_comparator_and_versions(monkeypatch, capsys, name):
    import json
    import sys

    calls = []
    monkeypatch.setattr(sys, "argv", ["parity", "--comparator", name, "--rows", "linear/9b/1"])
    monkeypatch.setattr(comparator, "load", lambda: calls.append("load"))
    monkeypatch.setattr(
        parity.importlib.metadata,
        "version",
        lambda package: {
            "vllm": "0.31.0",
            "flashinfer-python": "0.7.0.post1",
        }[package],
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(L2_cache_size=1024)
    )
    monkeypatch.setattr(parity, "_device_name", lambda: "fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def operations(kind, shape, rotation, selected, provider):
        assert provider == "dispatch"
        calls.append(("operations", selected))
        return [], [], 2048

    def row(*args, comparator):
        calls.append(("row", comparator))
        return {"kernel_class": "FlashInferCutlassNvFp4LinearKernel"}

    monkeypatch.setattr(parity, "_operations", operations)
    monkeypatch.setattr(parity, "_row", row)
    parity.main()
    record = next(
        json.loads(line.removeprefix("PARITY "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("PARITY ")
    )
    assert calls == ["load", ("operations", name), ("row", name)]
    assert record["comparator"] == name
    assert record["provider_mode"] == "dispatch"
    assert record["vllm_version"] == "0.31.0"
    assert record["flashinfer_version"] == "0.7.0.post1"
    assert record["kernel_class"] == "FlashInferCutlassNvFp4LinearKernel"
