# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Our Triton GDN prefill kernels compile ahead of time for every admitted architecture,
including those without a device to run on (sm89, sm100): no GPU is used."""

import pytest

triton = pytest.importorskip("triton")

from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from ctikernels.gdn.triton import _features, _intra, _output, _state  # noqa: E402

HK, HV, D = 16, 32, 128
WIDTH = (2 * HK + HV) * D
STATE = {
    **dict.fromkeys(("k", "w", "u", "hbuf", "projected", "conv_state"), "*bf16"),
    **dict.fromkeys(("gamma", "state"), "*fp32"),
    **dict.fromkeys(("cu_seqlens", "slot_indices", "capture_rows"), "*i32"),
    "has_initial": "*i1",
    "capturing": "i32",
    "p_stride": "i64",
}
KERNELS = {
    "features": (
        _features._features_kernel,
        {
            **dict.fromkeys(
                ("projected", "conv_weight", "conv_state", "q", "k", "v", "a"), "*bf16"
            ),
            **dict.fromkeys(("cu_seqlens", "slot_indices"), "*i32"),
            "has_initial": "*i1",
            "beta": "*fp32",
            "p_stride": "i64",
            **dict.fromkeys(("rows", "count"), "i32"),
        },
        {
            "HK": HK,
            "HV": HV,
            "D": D,
            "BH": 32,
            "CONV": 4,
            "HISTORY": 3,
            "EPS": 1e-6,
            "BT": 32,
            "BN": 128,
        },
        {},
    ),
    "intra": (
        _intra._intra_kernel,
        {
            **dict.fromkeys(("k", "v", "a", "w", "u"), "*bf16"),
            **dict.fromkeys(("beta", "A_log", "dt_bias", "gamma"), "*fp32"),
            "cu_seqlens": "*i32",
            "count": "i32",
        },
        {
            "HK": HK,
            "HV": HV,
            "D": D,
            "CHUNK": 64,
            "BC": 16,
            "BN": 128,
            "RCP_LN2": 1.4426950408889634,
            "PRECISION": "tf32",
        },
        {"num_warps": 2, "num_stages": 1},
    ),
    "output": (
        _output._output_kernel,
        {
            **dict.fromkeys(("q", "k", "hbuf", "vnew", "projected", "y"), "*bf16"),
            **dict.fromkeys(("gamma", "weight"), "*fp32"),
            "cu_seqlens": "*i32",
            "p_stride": "i64",
            "count": "i32",
            "scale": "fp32",
        },
        {"HK": HK, "HV": HV, "D": D, "CHUNK": 64, "BN": 128, "Z": WIDTH, "EPS": 1e-6},
        {"num_warps": 8, "num_stages": 1},
    ),
    "state": (
        _state._state_kernel,
        STATE,
        {
            "HK": HK,
            "HV": HV,
            "D": D,
            "BV": 64,
            "CHUNK": 64,
            "HISTORY": 3,
            "WIDTH": WIDTH,
            "BW": 128,
            "BN": 128,
        },
        {"num_stages": 2},
    ),
}


@pytest.mark.parametrize("capability", [89, 100, 103, 120, 121])
@pytest.mark.parametrize("name", list(KERNELS))
def test_kernels_compile_ahead_of_time(name, capability):
    function, signature, constants, options = KERNELS[name]
    index = {arg: (function.arg_names.index(arg),) for arg in (*signature, *constants)}
    source = ASTSource(
        function,
        {**signature, **dict.fromkeys(constants, "constexpr")},
        constexprs={index[k]: v for k, v in constants.items()},
        attrs={index[k]: [["tt.divisibility", 16]] for k, t in signature.items() if t[0] == "*"},
    )
    compiled = triton.compile(
        source, target=GPUTarget("cuda", capability, 32), options={"num_warps": 4, **options}
    )
    # Shared memory per block: 227 KiB on sm10x, 99 KiB on sm89 and sm12x.
    limit = 227 if capability in (100, 103) else 99
    assert compiled.metadata.shared <= limit * 1024, compiled.metadata.shared
