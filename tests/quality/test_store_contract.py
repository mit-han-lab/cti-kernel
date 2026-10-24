# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "python/ctikernels"


def test_negative_store_indices_guard_destination_addresses_and_writes() -> None:
    source = (SOURCE_ROOT / "csrc/jit/store.cu").read_text(encoding="utf-8")
    helper = re.split(r"void\s+store_pair_if\(", source, maxsplit=1)[1].split("template <", 1)[0]
    guard = helper.index("if (pos >= 0) {")
    for destination in (
        "pointer::offset(k_cache",
        "pointer::offset(v_cache",
        "*dst_k =",
        "*dst_v =",
    ):
        offsets = [match.start() for match in re.finditer(re.escape(destination), helper)]
        assert offsets and all(guard < offset for offset in offsets)
    single = re.split(r"void\s+store_if\(", source, maxsplit=1)[1].split("template <", 1)[0]
    guard = single.index("if (pos >= 0)")
    assert guard < single.index("pointer::offset(cache, pos * stride")
    kernel = source.split("store_kv_cache(", 1)[1].split("inline auto device_attribute", 1)[0]
    wait = kernel.index("PDL::wait<kUsePDL>()")
    trigger_text = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory")'
    k_then_v_start = kernel.index("if constexpr (kKThenV) {")
    default_start = kernel.index("\n  }\n", k_then_v_start)
    k_then_v = kernel[k_then_v_start:default_start]
    # Default path: K and V held together, both stores after the trigger.
    trigger = kernel.index(trigger_text, default_start)
    assert kernel.count("indices)[row_id]") == 2
    # PDL lets the K/V producer overlap this kernel: no read-only-cache loads.
    assert "__ldg" not in kernel and "ld.global.nc" not in kernel
    for load in ("indices)[row_id]", "key = src_k[", "value = src_v["):
        assert wait < kernel.index(load, default_start) < trigger
    assert trigger < kernel.index("store_pair_if(pos,")
    assert "if constexpr (kUsePDL)" not in kernel[wait:trigger]
    # Mid-sized batches: K stored before V is loaded, every store guarded.
    order = [
        k_then_v.index(text)
        for text in (
            "indices)[row_id]",
            "key = src_k[",
            trigger_text,
            "store_if(pos, kv_cache_stride, byte_offset, k_cache, key)",
            "value = src_v[",
            "store_if(pos, kv_cache_stride, byte_offset, v_cache, value)",
        )
    ]
    assert wait < k_then_v_start and order == sorted(order)
    assert "pos *" not in kernel


def test_store_defaults_enable_pdl_and_adaptive_eight_warp_blocks() -> None:
    from ctikernels.store import DEFAULT_INDEX_KERNEL_CONFIG

    assert DEFAULT_INDEX_KERNEL_CONFIG.use_pdl is True
    assert DEFAULT_INDEX_KERNEL_CONFIG.num_threads == 256
    source = (SOURCE_ROOT / "csrc/jit/store.cu").read_text(encoding="utf-8")
    assert "blockIdx.x * blockDim.y + threadIdx.y" in source
    assert "length * row_splits(element_size)" in source
    assert "total_warps > sms * block_warps" in source
    assert "block_warps *= 2" in source
    assert "LaunchKernel(num_blocks, {32, block_warps}, device)" in source
    # Size bands, in 512-byte row pieces: one row per block for small batches,
    # the K-then-V copy for mid-sized ones.
    assert "kSmallBatchWarps = 1024;" in source
    assert "kKThenVMinWarps = 8192;" in source
    assert "kKThenVMaxWarps = 32768;" in source
    assert "total_warps <= kSmallBatchWarps && element_size >= 2048" in source
    assert "total_warps >= kKThenVMinWarps" in source
    assert "total_warps <= kKThenVMaxWarps" in source
    assert ".with_attr(use_pdl && pdl_supported(device.device_id))(kernel, params)" in source


def test_store_griddepcontrol_is_guarded_below_sm90() -> None:
    source = (SOURCE_ROOT / "csrc/jit/store.cu").read_text(encoding="utf-8")
    kernel = source.split("store_kv_cache(", 1)[1].split("inline auto device_attribute", 1)[0]
    guard = "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"
    for instruction in ("PDL::wait<kUsePDL>()", "griddepcontrol.launch_dependents"):
        sites = [match.start() for match in re.finditer(re.escape(instruction), kernel)]
        assert sites
        for at in sites:
            opened = kernel.rfind(guard, 0, at)
            assert opened >= 0 and "#endif" not in kernel[opened:at]
    assert 'asm volatile("" ::: "memory")' in kernel
    assert "cudaDevAttrComputeCapabilityMajor" in source


def test_store_cache_docstring_documents_negative_indices() -> None:
    module = ast.parse((SOURCE_ROOT / "store.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "store_cache"
    )
    docstring = " ".join((ast.get_docstring(function) or "").lower().split())
    assert "negative index" in docstring
    assert "skip" in docstring
    assert "cache rows untouched" in docstring
