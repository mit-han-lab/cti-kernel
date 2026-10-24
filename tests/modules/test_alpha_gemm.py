# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch

from ctikernels._alpha_gemm import alpha_gemm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalar_shape", [(), (1,)])
def test_cpu_alpha_gemm_strided_reference(dtype, scalar_shape):
    torch.manual_seed(814)
    a = torch.randn(37, 31).to(dtype)[:, 1:24]
    b = torch.randn(37, 19).to(dtype)[:, :17]
    out = torch.empty(23, 17, dtype=dtype)
    alpha = torch.full(scalar_shape, 0.375)
    expected = (a.float().t() @ b.float() * alpha).to(dtype)
    assert alpha_gemm(a, b, alpha, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["shape", "dtype", "alpha_dtype", "alpha_shape", "stride", "out"])
def test_alpha_gemm_validation(bad):
    a, b, alpha, out = torch.ones(7, 3), torch.ones(7, 5), torch.tensor(0.375), torch.empty(3, 5)
    if bad == "shape":
        b = b[:6]
    elif bad == "dtype":
        b = b.bfloat16()
    elif bad == "alpha_dtype":
        alpha = alpha.double()
    elif bad == "alpha_shape":
        alpha = torch.ones(1, 1)
    elif bad == "stride":
        a = torch.ones(3, 7).t()
    elif bad == "out":
        out = torch.empty(5, 3).t()
    with pytest.raises((TypeError, ValueError), match="alpha_gemm"):
        alpha_gemm(a, b, alpha, out=out)


def test_alpha_gemm_finds_wheel_soname_and_propagates_build_failure(monkeypatch, tmp_path):
    import ctikernels._alpha_gemm as module

    # CUDA 13 wheels use nvidia/cu13/lib; CUDA 12 wheels use nvidia/cublas/lib.
    library_dir = tmp_path / "cu13" / "lib"
    library_dir.mkdir(parents=True)
    (library_dir / "libcublasLt.so.13").touch()

    class Spec:
        submodule_search_locations = [str(tmp_path)]

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: Spec())
    # Hide torch's own nvidia wheel directory so only the fake wheel is found.
    monkeypatch.setattr(module.torch, "__file__", str(tmp_path / "site" / "torch" / "__init__.py"))

    def build(*args, **kwargs):
        assert args == ("alpha_gemm",)
        assert kwargs["cuda_files"] == ["alpha_gemm.cu"]
        assert kwargs["extra_ldflags"] == [
            f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}",
            "-l:libcublasLt.so.13",
        ]
        raise RuntimeError("test build failure")

    monkeypatch.setattr(module, "load_aot", build)
    module._load_module.cache_clear()
    with pytest.raises(RuntimeError, match="test build failure"):
        module._load_module()
    module._load_module.cache_clear()
