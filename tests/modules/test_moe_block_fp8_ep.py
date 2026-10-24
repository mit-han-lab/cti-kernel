# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU metadata checks for checkpoint packing and global expert maps."""

import pytest
import torch

from ctikernels import _moe_block_fp8_triton_table as table
from ctikernels.moe_block_fp8 import (
    BlockFP8MoE,
    _check_weights,
    _select_provider,
    pack_block_fp8_experts,
)
from ctikernels.moe_impl import _check_expert_map, block_fp8_moe_config, fused_moe_kernel_triton


def checkpoint(dtype=torch.bfloat16):
    gate = torch.ones(2, 128, 256).to(torch.float8_e4m3fn)
    up = (torch.ones_like(gate, dtype=torch.float32) * 2).to(gate.dtype)
    down = torch.zeros(2, 256, 128).to(gate.dtype)
    return (
        gate,
        torch.full((2, 1, 2), 0.25, dtype=dtype),
        up,
        torch.full((2, 1, 2), 0.5, dtype=dtype),
        down,
        torch.full((2, 2, 1), 0.125, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pack(dtype):
    args = checkpoint(dtype)
    w13, s13, w2, s2 = pack_block_fp8_experts(*args)
    assert _check_weights(w13, s13, w2, s2, 8) == (2, 256, 128)
    assert w13.dtype == w2.dtype == torch.float8_e4m3fn
    assert s13.dtype == s2.dtype == dtype
    assert torch.equal(w13[:, :128].float(), args[0].float())
    assert torch.equal(w13[:, 128:].float(), args[2].float())
    assert torch.equal(s13[:, :1], args[1]) and torch.equal(s13[:, 1:], args[3])
    assert w2 is args[4] and s2 is args[5]


@pytest.mark.parametrize(
    "slot,value,error",
    [
        (0, torch.empty(2, 128), TypeError),
        (1, torch.ones(2, 1, 1), ValueError),
        (2, torch.empty(2, 256, 256).to(torch.float8_e4m3fn), ValueError),
        (3, torch.ones(2, 1, 2), TypeError),
        (5, torch.ones(2, 2, 1, dtype=torch.float16), TypeError),
    ],
)
def test_bad_pack(slot, value, error):
    args = list(checkpoint())
    args[slot] = value
    with pytest.raises(error):
        pack_block_fp8_experts(*args)


@pytest.mark.parametrize(
    "mapping,error,match",
    [
        (torch.zeros(4, dtype=torch.int64), TypeError, "int32"),
        (torch.zeros(2, 2, dtype=torch.int32), ValueError, "1-D"),
        (torch.zeros(8, dtype=torch.int32)[::2], ValueError, "contiguous"),
        (torch.zeros(1, dtype=torch.int32), ValueError, "cover"),
        (torch.empty(4, dtype=torch.int32, device="meta"), ValueError, "device"),
    ],
)
def test_bad_map(mapping, error, match):
    with pytest.raises(error, match=match):
        _check_expert_map(mapping, torch.device("cpu"), 2)
    with pytest.raises(error, match=match):
        BlockFP8MoE(*pack_block_fp8_experts(*checkpoint()), expert_map=mapping)
    # The launcher checks metadata before importing Triton, for BF16 as well.
    with pytest.raises(error, match=match):
        fused_moe_kernel_triton(
            torch.empty(1, 128),
            torch.empty(2, 128, 128),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            8,
            {},
            torch.bfloat16,
            expert_map=mapping,
        )


def test_global_extent_and_cute_rejection():
    mapping = torch.tensor([-1, 1, -1, 0], dtype=torch.int32)
    assert _check_expert_map(mapping, mapping.device, 2, 4) == 4
    with pytest.raises(ValueError, match="cover"):
        _check_expert_map(mapping, mapping.device, 2, 5)
    with pytest.raises(ValueError, match="requires expert_map"):
        _check_expert_map(None, mapping.device, 2, 4)
    params = pack_block_fp8_experts(*checkpoint())
    with pytest.raises(ValueError, match="equal"):
        BlockFP8MoE(*params, expert_map=mapping, num_global_experts=3)
    with pytest.raises(ValueError, match="does not support expert_map"):
        BlockFP8MoE(*params, expert_map=mapping, provider="cute-fp8")


def test_default_provider_and_configs():
    for device, experts in table.TABLE.items():
        assert _select_provider(None, device, 4, True, True) == "triton-fp8"
        for shape, rows in experts.items():
            e, h, i, k = shape
            assert [row[0] for row in rows] == sorted({row[0] for row in rows})
            for limit, fc1, fc2 in rows:
                assert fc1["BLOCK_SIZE_M"] == fc2["BLOCK_SIZE_M"]
                assert fc1["BLOCK_SIZE_K"] == fc2["BLOCK_SIZE_K"] == 128
                selected = table.selection(device, limit, e, h, i, k)
                default = limit * k <= 64 and fc1["BLOCK_SIZE_M"] == 16
                assert selected == tuple(
                    dict(c, route_blocks=c.get("route_blocks", default)) for c in (fc1, fc2)
                )
            if shape in table.MEASURED.get(device, {}):
                continue
            for m in (0, 1, e // 2, e // 2 + 1, 4096, 4097):
                for device_class in (device, (99, 0, 1)):
                    expected = block_fp8_moe_config(m, e, 2 * i, h, k, capability=device_class[:2])
                    expected["route_blocks"] = m * k <= 64 and expected["BLOCK_SIZE_M"] == 16
                    assert table.selection(device_class, m, e, h, i, k) == (expected, expected)
    fc1, fc2 = table.selection((12, 0, 170), 1, 256)
    fc1["BLOCK_SIZE_M"] = 999
    assert fc2["BLOCK_SIZE_M"] == 16
    assert table.selection((12, 0, 170), 1, 256)[0]["BLOCK_SIZE_M"] == 16


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zero_tokens_never_launches(dtype):
    op = object.__new__(BlockFP8MoE)
    op.w13, op.s13, op.w2, op.s2 = pack_block_fp8_experts(*checkpoint())
    op.top_k, op.h, op.device = 8, 256, torch.device("cpu")
    op.device_class, op.provider = (12, 0, 170), "triton-fp8"
    op._available, op._enabled = True, True
    assert op(
        torch.empty(0, 256, dtype=dtype),
        torch.empty(0, 8),
        torch.empty(0, 8, dtype=torch.int32),
    ).shape == (0, 256)


@pytest.mark.parametrize(
    "fp8,mapped,scale",
    [
        (False, False, "fp32"),
        (False, True, "fp32"),
        (True, False, "bf16"),
        (True, False, "fp32"),
        (True, True, "bf16"),
        (True, True, "fp32"),
    ],
)
def test_offline_expert_map_compile(fp8, mapped, scale, tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from ctikernels.triton.fused_moe import fused_moe_kernel

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    constants = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "BLOCK_FP8": fp8,
        "USE_GDC": True,
        "GDC_EARLY": True,
        "MUL_ROUTED_WEIGHT": True,
        "top_k": 8,
        "compute_type": tl.bfloat16,
        "even_Ks": True,
        # The fused-activation epilogue's operands and switches stay off.
        "SWIGLU_QUANT": False,
        "ACT_OUT": False,
        "LORA_R": 0,
        **dict.fromkeys(
            (
                "c_scale_ptr",
                "act_ptr",
                "lora_low_ptr",
                "lora_gate_ptr",
                "lora_up_ptr",
                "lora_scale_ptr",
            ),
        ),  # fmt: skip
    }
    if not mapped:
        constants["expert_map_ptr"] = None
    signature = {name: "i32" for name in fused_moe_kernel.arg_names if name not in constants}
    signature.update(
        {
            "a_ptr": "*fp8e4nv" if fp8 else "*bf16",
            "b_ptr": "*fp8e4nv" if fp8 else "*bf16",
            "a_scale_ptr": "*fp32",
            "b_scale_ptr": "*" + scale,
            "c_ptr": "*bf16",
            "topk_weights_ptr": "*fp32",
            "sorted_token_ids_ptr": "*i32",
            "expert_ids_ptr": "*i32",
            "num_tokens_post_padded_ptr": "*i32",
        }
    )
    if mapped:
        signature["expert_map_ptr"] = "*i32"
    kernel = triton.compile(
        ASTSource(fused_moe_kernel, signature, constants),
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": 4, "num_stages": 3},
    )
    assert kernel.metadata.shared <= 99 * 1024
    # The optional pointer is appended, preserving the RC's argument order.
    import re

    index = sum(
        name not in constants
        for name in fused_moe_kernel.arg_names[: fused_moe_kernel.arg_names.index("expert_map_ptr")]
    )
    loaded = re.search(
        rf"ld\.param\.[bu]64\s+[^;]*\[fused_moe_kernel_param_{index}\]", kernel.asm["ptx"]
    )
    assert bool(loaded) == mapped
    assert "griddepcontrol.wait" in kernel.asm["ptx"]
