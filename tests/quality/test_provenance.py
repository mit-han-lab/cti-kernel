# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import tomllib

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PACKAGE_ROOT / "python"

MINISGL_REVISION = "9a91cfafe754aa85daee49998176275667eb58f2"
LIGER_REVISION = "c4b16d43f9d8f69068e6a15bd879dfc6a63b2449"
FLA_REVISION = "bce39a4efe9b2c70f546133ef51bb875e3e00071"
FLA_GDN_REVISION = "9c8e42e762fce087c27b673af4922795d9edb85e"  # v0.5.2, the gdn extra's pin
UNSLOTH_REVISION = "4c06c1dcc771f3d4f2aecc8d3ab77f2e01aa9107"
VLLM_MOE_REVISION = "a6221a144af772fd1a68fe7e627935dc53e81738"
VLLM_REVISION = "0decac0d96c42b49572498019f0a0e3600f50398"
VLLM_ATTENTION_REVISION = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
MINISGL_NCCL_INTRO_REVISION = "80e7cfd74b8aed13e92fa98e125e5c5061cf664e"
NCCL_REVISION = "f1308997d0420148b1be1c24d63f19d902ae589b"

FLA_GDN_SOURCES = {
    "python/ctikernels/gdn/_fla_fixed.py": (
        "fla/ops/gated_delta_rule/chunk.py",
        "fla/ops/gated_delta_rule/chunk_fwd.py",
        "fla/ops/gated_delta_rule/gate.py",
        "fla/ops/common/chunk_o.py",
        "fla/ops/utils/solve_tril.py",
        "fla/ops/gated_delta_rule/wy_fast.py",
        "fla/ops/common/chunk_scaled_dot_kkt.py",
        "fla/ops/common/chunk_delta_h.py",
        "fla/ops/utils/cumsum.py",
        "fla/modules/fused_norm_gate.py",
    ),
    "python/ctikernels/gdn/_gated_norm.py": ("fla/modules/fused_norm_gate.py",),
    "python/ctikernels/gdn/_l2norm.py": ("fla/modules/l2norm.py",),
    "python/ctikernels/gdn/_fla.py": ("fla/ops/gated_delta_rule/chunk.py",),
    "python/ctikernels/gdn/_fla_training.py": (
        "fla/modules/conv/triton/ops.py",
        "fla/modules/conv/triton/kernels.py",
        "fla/ops/gated_delta_rule/gate.py",
    ),
    "python/ctikernels/gdn/triton/_train_conv.py": (
        "fla/modules/conv/triton/kernels.py",
        "fla/modules/l2norm.py",
    ),
    "tests/modules/_gdn_training_reference.py": (
        "fla/modules/conv/triton/ops.py",
        "fla/modules/conv/triton/kernels.py",
        "fla/ops/gated_delta_rule/gate.py",
    ),
    "python/ctikernels/gdn/_cute.py": ("fla/ops/gated_delta_rule/chunk.py",),
    "python/ctikernels/gdn/_cute_bwd.py": ("fla/ops/common/chunk_o.py",),
    "tests/gpu/test_gdn_cute.py": ("fla/ops/gated_delta_rule/chunk.py",),
    "python/ctikernels/gdn/triton/_train_intra.py": (
        "fla/ops/gated_delta_rule/chunk_fwd.py",
        "fla/ops/gated_delta_rule/wy_fast.py",
        "fla/ops/gated_delta_rule/gate.py",
    ),
    "python/ctikernels/gdn/triton/_train_output.py": ("fla/ops/common/chunk_o.py",),
    "python/ctikernels/gdn/triton/_train_state.py": ("fla/ops/common/chunk_delta_h.py",),
}

FLASHMLA_SOURCES = frozenset(
    {
        "python/ctikernels/csrc/jit/flashmla/defines.h",
        "python/ctikernels/csrc/jit/flashmla/flashmla_utils.h",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/common/common.h",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/common.h",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/device.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm100/gemm.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm100/helpers.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm100/intrinsics.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm100/tma_cta_group2_nosplit.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm80/helpers.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm80/intrinsics.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm90/helpers.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/device/sm90/intrinsics.cuh",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/host/host.h",
        "python/ctikernels/csrc/jit/flashmla/kerutils/include/kerutils/kerutils.cuh",
        "python/ctikernels/csrc/jit/flashmla/params.h",
        "python/ctikernels/csrc/jit/flashmla/sm100/helpers.h",
        "python/ctikernels/csrc/jit/flashmla/sm100/prefill/sparse/common_subroutine.h",
        "python/ctikernels/csrc/jit/flashmla/sm100/prefill/sparse/fwd/head64/config.h",
        "python/ctikernels/csrc/jit/flashmla/sm100/prefill/sparse/fwd/head64/phase1.cuh",
        "python/ctikernels/csrc/jit/flashmla/sm100/prefill/sparse/fwd/head64/phase1.h",
        "python/ctikernels/csrc/jit/flashmla/utils.h",
    }
)

FLASHINFER_GDN_SOURCES = {
    "python/ctikernels/gdn/chunk_ws/_gates.py": ("alpha.py",),
    "python/ctikernels/gdn/chunk_ws/_inverse.py": ("collective_inverse_hmma.py",),
    "python/ctikernels/gdn/chunk_ws/_o_store.py": ("collective_store_tma.py",),
    "python/ctikernels/gdn/chunk_ws/_mma.py": ("helpers.py",),
    "python/ctikernels/gdn/chunk_ws/_kernel.py": ("delta_rule_sm120.py", "schedule.py"),
    "python/ctikernels/gdn/chunk_ws/_launch.py": ("delta_rule_sm120.py",),
    "python/ctikernels/gdn/chunk_ws/_fused.py": ("delta_rule_cp_sm120.py", "varlen_helper.py"),
    "python/ctikernels/gdn/chunk_ws/prefill.py": ("delta_rule_cp_sm120.py", "varlen_helper.py"),
    "python/ctikernels/gdn/chunk_ws/_cp.py": ("delta_rule_cp_sm120.py", "varlen_helper.py"),
    "python/ctikernels/gdn/chunk_ws/_cp_fixup.py": ("delta_rule_cp_sm120.py", "varlen_helper.py"),
}

ROOT_MIT_LICENSE_SHA256 = "bc633b0dd653b4ca9b2f949472ae3e094ffa1f813ea66aec38108b060e0f50c2"
APACHE_LICENSE_SHA256 = "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"
NCCL_PAYLOAD_SHA256 = "4d4956afe21b4208c071a681b54c13343ea71129b17a019e949d418c8fa3f423"
NOTICE_SECTION_SHA256 = {
    "TRITON MIT LICENSE": "92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946",
    "MINI-SGLANG MIT LICENSE": "de93fca1c2ab406b177d405d0b05e0598ed22f082ce1b5acb5ee7991007f096c",
    "LIGER KERNEL BSD 2-CLAUSE LICENSE": (
        "ce44b62686a1c1d23328eda871e2ad3a3dc21d7a6d291045dc125e779e0cbc36"
    ),
    "EFFICIENT CROSS ENTROPY MIT LICENSE": (
        "9c6d722bc4b703548d8ebad4c6513a092ebb02cd88221c6f451c4550a3d491ea"
    ),
    "FLASH LINEAR ATTENTION MIT LICENSE": (
        "41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7"
    ),
    "FLASHMLA MIT LICENSE": "6c73a30c14a2938abaa6afe98ebb19e6d81dac5bd4ace0cb05c61c20afa37b85",
    "NVIDIA NCCL BSD 3-CLAUSE LICENSE": (
        "48dd135e274e9b039484a4a451fe3f5c108d098b2682931da95582a0811f2d36"
    ),
}
MINISGL_DERIVED_SOURCES = frozenset(
    {
        "python/ctikernels/__init__.py",
        "python/ctikernels/__main__.py",
        "python/ctikernels/attention/inference.py",
        "python/ctikernels/csrc/include/ctiengine/tensor.h",
        "python/ctikernels/csrc/include/ctiengine/utils.cuh",
        "python/ctikernels/csrc/include/ctiengine/utils.h",
        "python/ctikernels/csrc/include/ctiengine/warp.cuh",
        "python/ctikernels/csrc/jit/index.cu",
        "python/ctikernels/csrc/jit/store.cu",
        "python/ctikernels/csrc/src/pynccl.cu",
        "python/ctikernels/csrc/src/radix.cpp",
        "python/ctikernels/csrc/src/tensor.cpp",
        "python/ctikernels/index.py",
        "python/ctikernels/moe_impl.py",
        "python/ctikernels/pynccl.py",
        "python/ctikernels/radix.py",
        "python/ctikernels/sampling/_categorical.py",
        "python/ctikernels/store.py",
        "python/ctikernels/tensor.py",
        "python/ctikernels/triton/fused_moe.py",
        "python/ctikernels/utils.py",
    }
)
# Mini-SGLang files whose fused MoE lineage reaches SGLang/vLLM keep Apache-2.0.
MINISGL_APACHE_LINEAGE = frozenset(
    {"python/ctikernels/moe_impl.py", "python/ctikernels/triton/fused_moe.py"}
)
W4A8_FLASHINFER_SOURCES = frozenset(
    f"python/ctikernels/moe/cute/_w4a8{name}.py"
    for name in ("", "_epilogue", "_fc1", "_fc2", "_gemm", "_layout", "_schedule")
)
EXPECTED_SOURCE_LICENSES = {
    "python/ctikernels/gdn/replay/triton/_handoff.py": ("MIT",),
    "python/ctikernels/gdn/triton/_train_chunks.py": ("MIT",),
    "python/ctikernels/embedding/__init__.py": ("MIT",),
    "python/ctikernels/embedding/inference.py": ("MIT",),
    "python/ctikernels/embedding/triton/__init__.py": ("MIT",),
    "python/ctikernels/embedding/triton/_bilinear.py": ("MIT",),
    "python/ctikernels/kvcache/inference.py": ("MIT",),
    "python/ctikernels/kvcache/torch/__init__.py": ("MIT",),
    "python/ctikernels/kvcache/triton/__init__.py": ("MIT",),
    "python/ctikernels/kvcache/triton/_kernels.py": ("MIT",),
    "python/ctikernels/gdn/replay/_host.py": ("MIT",),
    "benchmark/kernel_tuning/gdn_replay_host_time.py": ("MIT",),
    "tests/modules/test_gdn_replay_host_fast.py": ("MIT",),
    "tests/gpu/test_gdn_replay_host_fast.py": ("MIT",),
    "python/ctikernels/sampling/_penalties.py": ("MIT",),
    "python/ctikernels/sampling/triton/__init__.py": ("MIT",),
    "python/ctikernels/sampling/triton/_penalties.py": ("MIT",),
    "python/ctikernels/sampling/triton/_top_logprobs.py": ("MIT",),
    "python/ctikernels/sampling/triton/_top_logprobs_processed.py": ("MIT",),
    "python/ctikernels/sampling/triton/_topk_by_id.py": ("MIT",),
    "python/ctikernels/sampling/_penalties_reference.py": ("MIT",),
    "python/ctikernels/_alpha_gemm.py": ("MIT",),
    "python/ctikernels/csrc/src/alpha_gemm.cu": ("MIT",),
    "python/ctikernels/attention/_draft/flashinfer_mask/__init__.py": ("MIT",),
    "python/ctikernels/attention/_draft/flashinfer_mask/prepare.py": ("MIT",),
    "python/ctikernels/logprob/cute/__init__.py": ("MIT",),
    "python/ctikernels/logprob/cute/_provider.py": ("MIT",),
    "python/ctikernels/logprob/cute/_kernel.py": ("MIT",),
    "tests/modules/test_gdn_replay_status_slot.py": ("MIT",),
    "tests/gpu/test_gdn_replay_status_slot.py": ("MIT",),
    "python/ctikernels/gdn/replay/_layout.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_dx.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_dx_gemm.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_reduce.py": ("MIT",),
    "python/ctikernels/moe_block_fp8.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_cute.py": ("Apache-2.0",),
    "python/ctikernels/_moe_block_fp8_cute_act.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_cute_decode.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_cute_table.py": ("MIT",),
    "python/ctikernels/_moe_block_fp8_triton_table.py": ("MIT",),
    "python/ctikernels/collective/_two_node/__init__.py": ("MIT",),
    "python/ctikernels/collective/_two_node/cute/__init__.py": ("MIT",),
    "python/ctikernels/collective/_two_node/cute/kernels.py": ("MIT",),
    "python/ctikernels/collective/_two_node/discovery.py": ("MIT",),
    "python/ctikernels/collective/_two_node/proxy.py": ("MIT",),
    "python/ctikernels/collective/two_node.py": ("MIT",),
    "python/ctikernels/csrc/src/two_node_proxy.cpp": ("MIT",),
    **dict.fromkeys(FLASHINFER_GDN_SOURCES, ("Apache-2.0",)),
    "python/ctikernels/gdn/chunk_ws/__init__.py": ("MIT",),
    "python/ctikernels/gdn/chunk_ws/_watchdog.py": ("MIT",),
    "python/ctikernels/linear/_fp8_workspace.py": ("MIT",),
    "python/ctikernels/collective/__init__.py": ("MIT",),
    "python/ctikernels/collective/_common.py": ("MIT",),
    "python/ctikernels/collective/triton/__init__.py": ("MIT",),
    "python/ctikernels/collective/triton/exchange.py": ("MIT",),
    "python/ctikernels/collective/_host_exchange/__init__.py": ("MIT",),
    "python/ctikernels/collective/_host_exchange/cute/__init__.py": ("MIT",),
    "python/ctikernels/collective/_host_exchange/cute/exchange.py": ("MIT",),
    "python/ctikernels/collective/_host_exchange/cute/kernels.py": ("MIT",),
    "python/ctikernels/collective/inference.py": ("MIT",),
    "python/ctikernels/moe/_configs.py": ("Apache-2.0",),
    "python/ctikernels/moe/_dispatch.py": ("MIT",),
    "python/ctikernels/moe/_routing_check.py": ("MIT",),
    "python/ctikernels/moe/_route.py": ("MIT",),
    "python/ctikernels/moe/_route_kernels.py": ("MIT",),
    "python/ctikernels/moe/_block.py": ("MIT",),
    "python/ctikernels/moe/triton_block/__init__.py": ("MIT",),
    "python/ctikernels/moe/triton_block/kernels.py": ("MIT",),
    "python/ctikernels/moe/triton_block_gb10/__init__.py": ("MIT",),
    "python/ctikernels/moe/triton_block_gb10/kernels.py": ("MIT",),
    "python/ctikernels/moe/__init__.py": ("MIT",),
    "python/ctikernels/moe/_align.py": ("MIT",),
    "python/ctikernels/moe/inference.py": ("MIT",),
    "python/ctikernels/moe/_training_backend.py": ("MIT",),
    "python/ctikernels/moe/cute/__init__.py": ("MIT",),
    "python/ctikernels/moe/cute/_gemm.py": ("MIT",),
    "python/ctikernels/moe/cute/_provider.py": ("MIT",),
    "python/ctikernels/moe/cute/_routing.py": ("MIT",),
    "python/ctikernels/moe/cute/_routing_kernels.py": ("MIT",),
    "python/ctikernels/moe/cute/_token.py": ("MIT",),
    "python/ctikernels/moe/cute/_token_kernel.py": ("MIT",),
    "python/ctikernels/moe/training.py": ("MIT",),
    "python/ctikernels/moe/triton/__init__.py": ("MIT",),
    "python/ctikernels/moe/triton/_kernels.py": ("MIT",),
    "python/ctikernels/moe/triton/_lora_backward.py": ("MIT",),
    "python/ctikernels/moe/triton/_lora_kernels.py": ("MIT",),
    "python/ctikernels/moe/triton/_lora_ops.py": ("MIT",),
    "python/ctikernels/moe/triton/_lora_train.py": ("MIT",),
    "python/ctikernels/moe/triton/_lora_sorted.py": ("MIT",),
    "python/ctikernels/moe/triton/_layout.py": ("MIT",),
    "python/ctikernels/moe/triton/_layout_kernels.py": ("MIT",),
    "python/ctikernels/moe/triton/_provider.py": ("MIT",),
    "python/ctikernels/moe/_common.py": ("MIT",),
    "python/ctikernels/moe/cute/_quant.py": ("MIT",),
    "python/ctikernels/moe/cute/_runtime.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_debug.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8_epilogue.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_fc1.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_fc2.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_finalize.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8_gemm.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_layout.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_lora.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8_model.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8_prepare.py": ("MIT",),
    "python/ctikernels/moe/cute/_w4a8_schedule.py": ("Apache-2.0",),
    "python/ctikernels/moe/cute/_w4a8_watchdog.py": ("MIT",),
    "python/ctikernels/moe/torch/_w4a8_tiled_reference.py": ("MIT",),
    "python/ctikernels/moe/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/moe/flashinfer/_w4a8.py": ("MIT",),
    "python/ctikernels/moe/torch/_w4a8_reference.py": ("MIT",),
    "python/ctikernels/moe/triton/_w4a8.py": ("MIT",),
    "python/ctikernels/moe/triton/_w4a8_kernels.py": ("MIT",),
    "python/ctikernels/moe/_lora_backward.py": ("MIT",),
    "python/ctikernels/moe/_lora_weights.py": ("MIT",),
    "python/ctikernels/moe/_lora_workspace.py": ("MIT",),
    "python/ctikernels/moe/lora.py": ("MIT",),
    "python/ctikernels/moe/cute/_lora_block.py": ("MIT",),
    "python/ctikernels/moe/_lora_common.py": ("MIT",),
    "python/ctikernels/moe/cute/_lora_dense.py": ("MIT",),
    "python/ctikernels/moe/cute/_lora_pair.py": ("MIT",),
    "python/ctikernels/moe/cute/_lora_provider.py": ("MIT",),
    "python/ctikernels/moe/cute/_lora_shared_table.py": ("MIT",),
    "python/ctikernels/moe/torch/__init__.py": ("MIT",),
    "python/ctikernels/moe/torch/_lora_reference.py": ("MIT",),
    "python/ctikernels/moe/triton/_slot_layout.py": ("MIT",),
    "python/ctikernels/speculative/__init__.py": ("MIT",),
    "python/ctikernels/speculative/_round/__init__.py": ("MIT",),
    "python/ctikernels/speculative/_round/cute/__init__.py": ("MIT",),
    "python/ctikernels/speculative/_round/cute/kernels.py": ("MIT",),
    "python/ctikernels/speculative/_round/op.py": ("MIT",),
    "python/ctikernels/speculative/_round/reference/__init__.py": ("MIT",),
    "python/ctikernels/speculative/_round/triton/__init__.py": ("MIT",),
    "python/ctikernels/speculative/_round/triton/kernels.py": ("MIT",),
    "python/ctikernels/_flashinfer_version.py": ("MIT",),
    "python/ctikernels/_reroute.py": ("MIT",),
    "python/ctikernels/_norm_act_reroute.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/reference/ops.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/cute/provider.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/op.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/ordinary.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/_warmup.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/argmax.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/cute/argmax.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/cute/_pdl.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/reference/argmax.py": ("MIT",),
    **dict.fromkeys(MINISGL_DERIVED_SOURCES - MINISGL_APACHE_LINEAGE, ("MIT",)),
    **dict.fromkeys(MINISGL_APACHE_LINEAGE, ("Apache-2.0",)),
    "python/ctikernels/activation/_silu_and_mul.py": ("MIT",),
    "python/ctikernels/_cute_prepare.py": ("MIT",),
    "python/ctikernels/activation/inference.py": ("MIT",),
    "python/ctikernels/activation/_dispatch.py": ("MIT",),
    "python/ctikernels/activation/cute/__init__.py": ("MIT",),
    "python/ctikernels/activation/cute/_provider.py": ("MIT",),
    "python/ctikernels/activation/cute/_kernel.py": ("MIT",),
    "python/ctikernels/activation/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/norm/inference.py": ("MIT",),
    "python/ctikernels/norm/_dispatch.py": ("MIT",),
    "python/ctikernels/norm/cute/__init__.py": ("MIT",),
    "python/ctikernels/norm/cute/_kernel.py": ("MIT",),
    "python/ctikernels/norm/cute/_provider.py": ("MIT",),
    "python/ctikernels/norm/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/norm/triton/__init__.py": ("MIT",),
    "python/ctikernels/norm/triton/_layer_norm.py": ("MIT",),
    "python/ctikernels/_storage.py": ("MIT",),
    "python/ctikernels/attention/draft.py": ("MIT",),
    "python/ctikernels/attention/_draft/__init__.py": ("MIT",),
    "python/ctikernels/attention/_draft/cute/__init__.py": ("MIT",),
    "python/ctikernels/attention/_draft/cute/_kernel.py": ("MIT",),
    "python/ctikernels/attention/_draft/cute/_geometry.py": ("MIT",),
    "python/ctikernels/attention/_draft/cute/_cooperative.py": ("MIT",),
    "python/ctikernels/attention/_decode_fp8/__init__.py": ("MIT",),
    "python/ctikernels/attention/_decode_fp8/_kernel.py": ("MIT",),
    "python/ctikernels/attention/_prefill_cute/__init__.py": ("MIT",),
    "python/ctikernels/attention/_prefill_cute/_kernel.py": ("MIT",),
    "python/ctikernels/attention/_scale_cast.py": ("MIT",),
    "python/ctikernels/attention/_scale_cast_kernel.py": ("MIT",),
    "python/ctikernels/attention/_draft/dispatch.py": ("MIT",),
    "python/ctikernels/attention/_draft/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/attention/_draft/flashinfer/_table_csr.py": ("MIT",),
    "python/ctikernels/attention/_flashinfer.py": ("Apache-2.0",),
    "python/ctikernels/attention/_flashinfer_schedule.py": ("Apache-2.0",),
    "python/ctikernels/attention/_split_tables.py": ("MIT",),
    "python/ctikernels/attention/_prologue.py": ("MIT",),
    "python/ctikernels/attention/_prologue_cute.py": ("MIT",),
    "python/ctikernels/cross_entropy.py": ("BSD-2-Clause",),
    "python/ctikernels/vocab_parallel.py": ("MIT",),
    "python/ctikernels/csrc/include/ctiengine/nccl227.h": ("BSD-3-Clause",),
    "python/ctikernels/csrc/jit/block_fp8_gemm_sm12x.cu": ("MIT",),
    "python/ctikernels/csrc/jit/request_local_sampling.cu": ("Apache-2.0",),
    "python/ctikernels/sampling/_dflash/__init__.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/cute/__init__.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/cute/kernels.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/reference/__init__.py": ("MIT",),
    "python/ctikernels/sampling/_dflash/reference/rng.py": ("MIT",),
    **dict.fromkeys(FLA_GDN_SOURCES, ("MIT",)),
    "python/ctikernels/gdn/_cute_fwd.py": ("MIT",),
    "python/ctikernels/gdn/replay/__init__.py": ("MIT",),
    "python/ctikernels/gdn/replay/_api.py": ("MIT",),
    "python/ctikernels/gdn/replay/_storage.py": ("MIT",),
    "python/ctikernels/gdn/replay/_status.py": ("MIT",),
    "python/ctikernels/gdn/replay/torch_reference/_status.py": ("MIT",),
    "python/ctikernels/gdn/replay/triton/_status.py": ("MIT",),
    "python/ctikernels/gdn/replay/torch_reference/__init__.py": ("MIT",),
    "python/ctikernels/gdn/replay/triton/__init__.py": ("MIT",),
    "python/ctikernels/gdn/replay/triton/_kernels.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/__init__.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_config.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_kernel.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_phase_launch.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_handoff.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_handoff_launch.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_index.py": ("MIT",),
    "python/ctikernels/gdn/replay/cute/_phases.py": ("MIT",),
    "python/ctikernels/gdn/_triton_decode.py": ("MIT",),
    "python/ctikernels/gdn/_triton_decode_kernels.py": ("MIT",),
    "python/ctikernels/gdn/_triton_prefill.py": ("MIT",),
    "python/ctikernels/gdn/_cute_prefill.py": ("MIT",),
    "python/ctikernels/gdn/_cute_prefill_kernels.py": ("MIT",),
    "python/ctikernels/gdn/_cute_prefill_v2_kernels.py": ("MIT",),
    "python/ctikernels/gdn/_fla_glue.py": ("MIT",),
    "python/ctikernels/gdn/triton/__init__.py": ("MIT",),
    "python/ctikernels/gdn/triton/_features.py": ("MIT",),
    "python/ctikernels/gdn/torch/__init__.py": ("MIT",),
    "python/ctikernels/gdn/torch/training.py": ("MIT",),
    "python/ctikernels/gdn/triton/_train_configs.py": ("MIT",),
    "python/ctikernels/gdn/triton/_train_features.py": ("MIT",),
    "python/ctikernels/gdn/triton/training.py": ("MIT",),
    "python/ctikernels/gdn/triton/_intra.py": ("MIT",),
    "python/ctikernels/gdn/triton/_launch.py": ("MIT",),
    "python/ctikernels/gdn/triton/_output.py": ("MIT",),
    "python/ctikernels/gdn/triton/_norm.py": ("MIT",),
    "python/ctikernels/gdn/triton/_state.py": ("MIT",),
    "python/ctikernels/gdn/triton/prefill.py": ("MIT",),
    "python/ctikernels/linear/_lora_inference.py": ("Apache-2.0",),
    "python/ctikernels/linear/_lora_policy.py": ("MIT",),
    "python/ctikernels/linear/_lora_segmented.py": ("MIT",),
    "python/ctikernels/triton/rounding.py": ("MIT",),
    "python/ctikernels/triton/silu.py": ("MIT",),
    "python/ctikernels/triton/topk_softmax.py": ("MIT",),
    "python/ctikernels/_runtime/__init__.py": ("MIT",),
    "python/ctikernels/_runtime/launch.py": ("MIT",),
    "python/ctikernels/_runtime/tables/__init__.py": ("MIT",),
    "python/ctikernels/_runtime/workspace.py": ("MIT",),
    "python/ctikernels/_runtime/prewarm.py": ("MIT",),
    "python/ctikernels/linear/inference.py": ("MIT",),
    "python/ctikernels/linear/_dispatch.py": ("MIT",),
    "python/ctikernels/linear/__init__.py": ("MIT",),
    "python/ctikernels/linear/_autotune.py": ("MIT",),
    "python/ctikernels/linear/cute/__init__.py": ("MIT",),
    "python/ctikernels/linear/cute/_provider.py": ("MIT",),
    "python/ctikernels/linear/cute/_kernel.py": ("MIT",),
    "python/ctikernels/linear/cute/variants.py": ("MIT",),
    "python/ctikernels/linear/cublas/__init__.py": ("MIT",),
    "python/ctikernels/linear/cute/_block_fp8_provider.py": ("Apache-2.0",),
    "python/ctikernels/linear/cute/_block_fp8_gemm.py": ("Apache-2.0",),
    "python/ctikernels/linear/cute/_block_fp8_act_quant.py": ("Apache-2.0",),
    "python/ctikernels/linear/cute/_block_fp8_fused.py": ("MIT",),
    "python/ctikernels/linear/cute/_block_fp8_fused_provider.py": ("MIT",),
    "python/ctikernels/linear/cute/_block_fp8_quant_math.py": ("Apache-2.0",),
    "python/ctikernels/linear/_block_fp8_dispatch.py": ("MIT",),
    "python/ctikernels/linear/flashinfer/_block_fp8.py": ("MIT",),
    "python/ctikernels/linear/cuda/_block_fp8_sm12x.py": ("MIT",),
    "python/ctikernels/linear/triton/_block_fp8_provider.py": ("MIT",),
    "python/ctikernels/linear/triton/_block_fp8_tuning.py": ("MIT",),
    "python/ctikernels/linear/triton/_block_fp8.py": ("Apache-2.0",),
    "python/ctikernels/linear/triton/__init__.py": ("MIT",),
    "python/ctikernels/linear/triton/_lora_segmented.py": ("MIT",),
    "python/ctikernels/linear/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/linear/cuda/__init__.py": ("MIT",),
    "python/ctikernels/linear/block_fp8.py": ("MIT",),
    "python/ctikernels/linear/fp8_static.py": ("MIT",),
    "python/ctikernels/linear/_fp8_static_dispatch.py": ("MIT",),
    "python/ctikernels/linear/cute/_fp8_static_quant.py": ("MIT",),
    "python/ctikernels/linear/triton/_fp8_static_quant.py": ("MIT",),
    "python/ctikernels/linear/_gate_ptx.py": ("MIT",),
    "python/ctikernels/linear/triton/_gate.py": ("MIT",),
    "python/ctikernels/linear/torch/__init__.py": ("MIT",),
    "python/ctikernels/mlp/torch/__init__.py": ("MIT",),
    "python/ctikernels/mlp/torch/_swiglu.py": ("MIT",),
    "python/ctikernels/mlp/triton/__init__.py": ("MIT",),
    "python/ctikernels/mlp/triton/_swiglu.py": ("MIT",),
    "python/ctikernels/linear/torch/_fp8_static.py": ("MIT",),
    "python/ctikernels/linear/flashinfer/_fp8_static.py": ("MIT",),
    "python/ctikernels/linear/block_fp8_gated.py": ("MIT",),
    "python/ctikernels/linear/_block_fp8_gated_dispatch.py": ("MIT",),
    "python/ctikernels/linear/_block_fp8_gated_table.py": ("MIT",),
    "python/ctikernels/linear/flashinfer/_block_fp8_gated.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_quant/__init__.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_quant/kernel.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_quant/launcher.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_small/__init__.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_recipe.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_cute_small/kernel.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_dispatch.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_quack.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_quack_epilogues.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_recipe.py": ("MIT",),
    "python/ctikernels/linear/_nvfp4_triton.py": ("MIT",),
    "python/ctikernels/linear/nvfp4.py": ("MIT",),
    "python/ctikernels/linear/training.py": ("MIT",),
    "python/ctikernels/linear/cuda/_marlin_nvfp4.py": ("Apache-2.0",),
    "python/ctikernels/linear/triton/_nvfp4_w4a16.py": ("MIT",),
    "python/ctikernels/csrc/jit/marlin_nvfp4.cu": ("Apache-2.0",),
    "python/ctikernels/linear/mxfp8.py": ("MIT",),
    "python/ctikernels/mhc/__init__.py": ("MIT",),
    "python/ctikernels/mhc/_config.py": ("MIT",),
    "python/ctikernels/mhc/_cute/__init__.py": ("MIT",),
    "python/ctikernels/mhc/_cute/cluster.py": ("MIT",),
    "python/ctikernels/mhc/_cute/kernels.py": ("MIT",),
    "python/ctikernels/mhc/_reference.py": ("MIT",),
    "python/ctikernels/mhc/inference.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/__init__.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/inference.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/triton/__init__.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/triton/_quant.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/triton/_compressor.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/triton/_indexer.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/dsv4_indexer/flashinfer/_logits.py": ("MIT",),
    "python/ctikernels/linear/flashinfer/_mxfp8.py": ("MIT",),
    "python/ctikernels/linear/triton/_mxfp8.py": ("MIT",),
    "python/ctikernels/norm/_frozen_rms_norm.py": ("Apache-2.0",),
    "python/ctikernels/norm/_zero_centered_rms_norm.py": ("MIT",),
    "python/ctikernels/rotary/_key_only.py": ("MIT",),
    "python/ctikernels/rotary/_vision.py": ("MIT",),
    "python/ctikernels/speculative/inference.py": ("MIT",),
    "python/ctikernels/speculative/triton/_block_conv.py": ("Apache-2.0",),
    "python/ctikernels/sampling/triton/_steer.py": ("Apache-2.0",),
    "python/ctikernels/sampling/_chain.py": ("MIT",),
    "python/ctikernels/sampling/_request_local.py": ("MIT",),
    "python/ctikernels/sampling/_logprobs.py": ("MIT",),
    "python/ctikernels/sampling/_logprobs_reference.py": ("MIT",),
    "python/ctikernels/attention/cuda/__init__.py": ("MIT",),
    "python/ctikernels/attention/cuda/_dsv4_flashmla.py": ("MIT",),
    "python/ctikernels/attention/flashinfer/__init__.py": ("MIT",),
    "python/ctikernels/attention/flashinfer/_dsv4.py": ("MIT",),
    "python/ctikernels/attention/triton/__init__.py": ("MIT",),
    "python/ctikernels/attention/triton/_draft.py": ("MIT",),
    "python/ctikernels/attention/triton/_dsv4.py": ("MIT",),
    "python/ctikernels/attention/triton/_kv_metadata.py": ("MIT",),
    "python/ctikernels/attention/triton/_paged.py": ("Apache-2.0",),
    "python/ctikernels/csrc/jit/dsv4_flashmla.cu": ("MIT",),
    "python/ctikernels/csrc/jit/dsv4_topk.cu": ("MIT",),
    "python/ctikernels/attention/cuda/_dsv4_glue.py": ("MIT",),
    "python/ctikernels/csrc/jit/dsv4_rope.cu": ("MIT",),
    "python/ctikernels/activation/__init__.py": ("MIT",),
    "python/ctikernels/activation/training.py": ("MIT",),
    "python/ctikernels/attention/__init__.py": ("MIT",),
    "python/ctikernels/attention/varlen.py": ("MIT",),
    "python/ctikernels/gdn/__init__.py": ("MIT",),
    "python/ctikernels/gdn/_contract.py": ("MIT",),
    "python/ctikernels/gdn/_rows.py": ("MIT",),
    "python/ctikernels/gdn/inference.py": ("MIT",),
    "python/ctikernels/gdn/training.py": ("MIT",),
    "python/ctikernels/kvcache/__init__.py": ("MIT",),
    "python/ctikernels/linear/lora.py": ("MIT",),
    "python/ctikernels/logprob/__init__.py": ("MIT",),
    "python/ctikernels/logprob/linear.py": ("MIT",),
    "python/ctikernels/mlp/__init__.py": ("MIT",),
    "python/ctikernels/mlp/training.py": ("MIT",),
    "python/ctikernels/norm/__init__.py": ("MIT",),
    "python/ctikernels/norm/training.py": ("MIT",),
    "python/ctikernels/rotary/__init__.py": ("MIT",),
    "python/ctikernels/rotary/inference.py": ("MIT",),
    "python/ctikernels/rotary/training.py": ("MIT",),
    "python/ctikernels/rotary/vision.py": ("MIT",),
    "python/ctikernels/sampling/__init__.py": ("MIT",),
    "python/ctikernels/sampling/inference.py": ("MIT",),
    "python/ctikernels/speculative/triton/__init__.py": ("MIT",),
    "python/ctikernels/triton/__init__.py": ("MIT",),
    **dict.fromkeys(FLASHMLA_SOURCES, ("MIT",)),
    "tests/gpu/test_bilinear_pos_embed.py": ("Apache-2.0",),
}
THIRD_PARTY_DERIVED_SOURCES = MINISGL_DERIVED_SOURCES | {
    *FLASHINFER_GDN_SOURCES,
    *W4A8_FLASHINFER_SOURCES,
    "python/ctikernels/moe/_configs.py",
    "python/ctikernels/_moe_block_fp8_cute.py",
    "python/ctikernels/attention/_flashinfer.py",
    "python/ctikernels/attention/_flashinfer_schedule.py",
    "python/ctikernels/attention/triton/_paged.py",
    "python/ctikernels/sampling/triton/_steer.py",
    "python/ctikernels/speculative/triton/_block_conv.py",
    "python/ctikernels/activation/_silu_and_mul.py",
    "python/ctikernels/cross_entropy.py",
    "python/ctikernels/csrc/include/ctiengine/nccl227.h",
    "python/ctikernels/csrc/jit/request_local_sampling.cu",
    *FLA_GDN_SOURCES,
    "python/ctikernels/linear/triton/_block_fp8.py",
    "python/ctikernels/linear/_lora_inference.py",
    "python/ctikernels/norm/_frozen_rms_norm.py",
    "python/ctikernels/csrc/jit/marlin_nvfp4.cu",
    "python/ctikernels/linear/cuda/_marlin_nvfp4.py",
    *FLASHMLA_SOURCES,
    "tests/gpu/test_bilinear_pos_embed.py",
}

_RELEASE_SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".h", ".py"})
_SPDX_PATTERN = re.compile(r"SPDX-License-Identifier:\s*([^\r\n]+)")
# Every tracked source under these roots carries an SPDX header; MIT unless listed above.
_HEADER_ROOTS = ("python/", "tests/", "benchmark/", "scripts/")
_HEADER_SUFFIXES = _RELEASE_SOURCE_SUFFIXES | {".sh"}


def _normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"


def _notice_section(notice: str, name: str) -> str:
    begin = f"----- BEGIN {name} (VERBATIM) -----\n"
    end = f"----- END {name} (VERBATIM) -----"
    body = notice.split(begin, maxsplit=1)[1].split(end, maxsplit=1)[0]
    return _normalized_text(body)


def _source_licenses() -> dict[str, tuple[str, ...]]:
    licenses: dict[str, tuple[str, ...]] = {}
    paths = list((PYTHON_ROOT / "ctikernels").rglob("*"))
    paths.append(PACKAGE_ROOT / "tests/modules/_gdn_training_reference.py")
    paths.append(PACKAGE_ROOT / "tests/gpu/test_gdn_cute.py")
    paths.append(PACKAGE_ROOT / "tests/modules/test_gdn_replay_status_slot.py")
    paths.append(PACKAGE_ROOT / "tests/gpu/test_gdn_replay_status_slot.py")
    paths.append(PACKAGE_ROOT / "tests/modules/test_gdn_replay_host_fast.py")
    paths.append(PACKAGE_ROOT / "tests/gpu/test_gdn_replay_host_fast.py")
    paths.append(PACKAGE_ROOT / "tests/gpu/test_bilinear_pos_embed.py")
    paths.append(PACKAGE_ROOT / "benchmark/kernel_tuning/gdn_replay_host_time.py")
    for path in paths:
        if not path.is_file() or path.suffix not in _RELEASE_SOURCE_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        # A file without an identifier maps to () and fails the exact comparison.
        identifiers = tuple(_SPDX_PATTERN.findall(path.read_text()))
        licenses[path.relative_to(PACKAGE_ROOT).as_posix()] = identifiers
    return licenses


def test_distribution_preserves_license_and_notice_payloads() -> None:
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "cti-kernels"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES"]
    license_text = _normalized_text((PACKAGE_ROOT / "LICENSE").read_text())
    assert hashlib.sha256(license_text.encode()).hexdigest() == ROOT_MIT_LICENSE_SHA256

    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "cti-kernels is licensed under the MIT License (see LICENSE)." in notice
    assert "2026 cti-engine contributors covers original code and local modifications" in notice
    # Apache-2.0 entries share one verbatim copy of the license at the end of the notice.
    assert notice.count("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION") == 1
    apache_text = _notice_section(notice, "APACHE LICENSE 2.0")
    assert hashlib.sha256(apache_text.encode()).hexdigest() == APACHE_LICENSE_SHA256
    apache_reference = (
        "Licensed under the Apache License, Version 2.0; full text at the end of this file."
    )
    assert notice.count(apache_reference) == 8
    sections = re.split(r"\n[^\n]+\n={10,}\n", notice)[1:]
    for upstream in (
        "flashinfer-ai/flashinfer",
        "huggingface/transformers",
        "sgl-project/sglang",
        "unslothai/unsloth",
        "vllm-project/vllm",
    ):
        section = next(
            s for s in sections if s.startswith(f"Project: https://github.com/{upstream}\n")
        )
        assert apache_reference in section
    assert "Copyright (c) 2026 sgl-project" in notice
    assert "upstream attribution and applicable license terms are preserved below" in notice
    for name, expected_hash in NOTICE_SECTION_SHA256.items():
        section = _notice_section(notice, name)
        assert hashlib.sha256(section.encode()).hexdigest() == expected_hash


def test_sm12x_original_sources_keep_design_references_out_of_notices() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    for local_path in (
        "python/ctikernels/csrc/jit/block_fp8_gemm_sm12x.cu",
        "benchmark/kernel_tuning/sm12x_roofline.cu",
    ):
        source = (PACKAGE_ROOT / local_path).read_text()
        header = source.split("\n\n", maxsplit=1)[0]
        assert _SPDX_PATTERN.findall(header) == ["MIT"]
        references = [line for line in header.splitlines() if "Reference:" in line]
        assert len(references) == 1
        for credit in ("CUTLASS/CuTe", "warp-specialized TMA", "BSD-3-Clause", "public PTX ISA"):
            assert credit in references[0]
        assert "no adapted code" in header
        assert "Adapted from" not in source
        assert local_path not in notice


def test_sources_track_every_local_spdx_identifier() -> None:
    assert _source_licenses() == EXPECTED_SOURCE_LICENSES


def test_every_tracked_source_has_an_spdx_header() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--", *_HEADER_ROOTS],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    missing, unlisted = [], []
    for local_path in tracked:
        path = PACKAGE_ROOT / local_path
        if path.suffix not in _HEADER_SUFFIXES or not path.is_file():
            continue
        header = "\n".join(path.read_text().splitlines()[:12])
        identifiers = _SPDX_PATTERN.findall(header)
        if not identifiers:
            missing.append(local_path)
        elif identifiers != ["MIT"] and local_path not in EXPECTED_SOURCE_LICENSES:
            unlisted.append(local_path)
        elif local_path in EXPECTED_SOURCE_LICENSES:
            assert tuple(identifiers) == EXPECTED_SOURCE_LICENSES[local_path], local_path
        if identifiers == ["MIT"] and "SPDX-FileCopyrightText:" not in header:
            missing.append(local_path)
    assert missing == [] and unlisted == []


def test_distribution_credits_local_modifications_without_replacing_upstream_credit() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text()
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "Original code and modifications are copyright 2026 cti-engine contributors" in readme
    for project in (
        "Mini-SGLang",
        "SGLang",
        "vLLM",
        "Triton",
        "Liger Kernel",
        "Unsloth",
        "Flash Linear Attention",
        "FlashInfer",
        "NVIDIA NCCL",
    ):
        assert project in readme
    assert "credits Efficient Cross Entropy for the chunked cross-entropy idea" in readme
    assert "which keeps BSD-3-Clause" in readme
    assert "2026 cti-engine contributors covers original code and local modifications" in notice
    assert "upstream attribution and applicable license terms are preserved below" in notice
    for local_path in THIRD_PARTY_DERIVED_SOURCES:
        source = (PACKAGE_ROOT / local_path).read_text()
        assert "Copyright (c) 2026 cti-engine contributors" in source


def test_distribution_preserves_minisglang_lineage() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "https://github.com/sgl-project/mini-sglang" in notice
    assert MINISGL_REVISION in notice
    assert "https://github.com/sgl-project/sglang" in notice
    assert "SGLang MoE launch/reduction and kernel lineage inherited through Mini-SGLang" in notice

    for local_path in MINISGL_DERIVED_SOURCES:
        source = (PACKAGE_ROOT / local_path).read_text()
        spdx = f"SPDX-License-Identifier: {EXPECTED_SOURCE_LICENSES[local_path][0]}\n"
        assert source.startswith(f"# {spdx}") or source.startswith(f"// {spdx}")
        assert "Copyright (c) 2026 sgl-project" in source
        assert "Copyright (c) 2026 cti-engine contributors" in source
        assert MINISGL_REVISION in source
        assert "`python/minisgl/" in source
        assert "THIRD_PARTY_NOTICES" in source


def test_distribution_preserves_liger_license_provenance() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "Project: https://github.com/linkedin/Liger-Kernel" in notice
    assert LIGER_REVISION in notice
    for upstream_source in (
        "LICENSE",
        "NOTICE",
        "src/liger_kernel/ops/cross_entropy.py",
        "src/liger_kernel/ops/fused_linear_cross_entropy.py",
    ):
        assert upstream_source in notice

    source = (PYTHON_ROOT / "ctikernels/cross_entropy.py").read_text()
    assert source.startswith("# SPDX-License-Identifier: BSD-2-Clause\n")
    assert "Copyright 2024 LinkedIn Corporation" in source
    assert LIGER_REVISION in source


def test_distribution_preserves_fla_license_provenance() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "Project: https://github.com/fla-org/flash-linear-attention" in notice
    assert FLA_REVISION in notice
    assert "fla/modules/activations.py" in notice
    assert "License source: LICENSE" in notice

    source = (PYTHON_ROOT / "ctikernels/activation/_silu_and_mul.py").read_text()
    assert source.startswith("# SPDX-License-Identifier: MIT\n")
    for holder in ("Songlin Yang", "Yu Zhang", "Zhiyuan Li"):
        assert f"SPDX-FileCopyrightText: Copyright (c) 2023-2026 {holder}\n" in source
    assert FLA_REVISION in source
    assert "fla/modules/activations.py" in source

    assert FLA_GDN_REVISION in notice
    for local_path, upstream_paths in FLA_GDN_SOURCES.items():
        source = (PACKAGE_ROOT / local_path).read_text()
        for holder in ("Songlin Yang", "Yu Zhang", "Zhiyuan Li"):
            assert f"SPDX-FileCopyrightText: Copyright (c) 2023-2026 {holder}\n" in source
        assert "SPDX-License-Identifier: MIT" in source
        assert FLA_GDN_REVISION in source
        assert "THIRD_PARTY_NOTICES" in source
        header = source.split("\n\n", maxsplit=1)[0]
        assert tuple(re.findall(r"`([^`]+)`", header)) == upstream_paths
        assert all(path in notice for path in upstream_paths)
        assert "no fla-core source is copied" not in source
    assert "without copying its source" not in notice


def test_distribution_preserves_nccl_header_and_license_provenance() -> None:
    header = (PYTHON_ROOT / "ctikernels/csrc/include/ctiengine/nccl227.h").read_text()
    inherited_payload = header[header.index("#ifndef") :]
    assert hashlib.sha256(inherited_payload.encode()).hexdigest() == NCCL_PAYLOAD_SHA256
    assert "SPDX-License-Identifier: BSD-3-Clause" in header
    assert NCCL_REVISION in header
    assert "src/nccl.h.in" in header

    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert MINISGL_NCCL_INTRO_REVISION in notice
    assert MINISGL_REVISION in notice
    assert "Project: https://github.com/NVIDIA/nccl" in notice
    assert NCCL_REVISION in notice
    assert "License source: LICENSE.txt" in notice


def test_distribution_preserves_apache_source_provenance() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert UNSLOTH_REVISION in notice
    assert VLLM_REVISION in notice
    assert (
        "   Copyright [2024-] [Unsloth AI. Inc team, Daniel Han-Chen & Michael Han-Chen]\n"
    ) in notice
    assert "   Files under unsloth/*, tests/*, scripts/* are Apache 2.0 licensed.\n" in notice

    for local_path, copyright_text, revision, upstream_paths in (
        (
            "ctikernels/norm/_frozen_rms_norm.py",
            "Copyright 2023-present Daniel Han-Chen. All rights reserved.",
            UNSLOTH_REVISION,
            ("unsloth/kernels/rms_layernorm.py",),
        ),
        (
            "ctikernels/linear/_lora_inference.py",
            "Copyright contributors to the vLLM project",
            VLLM_REVISION,
            (
                "vllm/lora/ops/triton_ops/lora_shrink_op.py",
                "vllm/lora/ops/triton_ops/lora_expand_op.py",
                "vllm/lora/ops/triton_ops/kernel_utils.py",
                "vllm/lora/ops/triton_ops/utils.py",
            ),
        ),
        (
            "ctikernels/attention/triton/_paged.py",
            "Copyright contributors to the vLLM project",
            VLLM_ATTENTION_REVISION,
            ("vllm/v1/attention/ops/triton_unified_attention.py",),
        ),
        (
            "ctikernels/speculative/triton/_block_conv.py",
            "Copyright contributors to the vLLM project",
            VLLM_ATTENTION_REVISION,
            ("vllm/model_executor/models/qwen3_dflash2.py",),
        ),
        (
            "ctikernels/sampling/triton/_steer.py",
            "Copyright contributors to the vLLM project",
            VLLM_ATTENTION_REVISION,
            ("vllm/v1/worker/gpu/sample/logit_bias.py",),
        ),
    ):
        source = (PYTHON_ROOT / local_path).read_text()
        assert source.startswith("# SPDX-License-Identifier: Apache-2.0\n")
        assert "SPDX-License-Identifier: Apache-2.0" in source
        assert copyright_text in source
        assert revision in source
        assert "THIRD_PARTY_NOTICES" in source
        assert all(path in source for path in upstream_paths)


def test_third_party_headers_are_concise_and_complete() -> None:
    for local_path in EXPECTED_SOURCE_LICENSES:
        source = (PACKAGE_ROOT / local_path).read_text()
        header = source.split("\n\n", maxsplit=1)[0]
        # Ten lines allow separate holders and wrapped upstream paths without file exceptions.
        assert len(header.splitlines()) <= 10, local_path
        # The own-tree port contract requires this exact single-line provenance.
        assert all(
            len(line) <= 100
            or line.startswith("# Port of cti's own Qwen3.5-MoE decode-block Triton tree (")
            for line in header.splitlines()
        ), local_path
        copyrights = [line for line in header.splitlines() if "SPDX-FileCopyrightText:" in line]
        assert all(line.count("Copyright") == 2 for line in copyrights), local_path
        # NVIDIA's corporate attribution is one holder, including its affiliates.
        assert all(
            not re.search(
                r";| & |, (?:Yu Zhang|Zhiyuan Li)",
                line.replace("NVIDIA CORPORATION & AFFILIATES", "NVIDIA CORPORATION"),
            )
            for line in copyrights
        )
        assert "SPDX-License-Identifier:" in header
        if local_path not in THIRD_PARTY_DERIVED_SOURCES:
            continue
        assert "SPDX-FileCopyrightText:" in header
        assert "Copyright (c) 2026 cti-engine contributors" in header
        assert "Adapted from" in header and "THIRD_PARTY_NOTICES" in header
        if local_path == "python/ctikernels/_moe_block_fp8_cute.py":
            assert "FlashInfer 0.7.1" in header and "CUTLASS" in header
        else:
            assert re.search(r"[0-9a-f]{40}", header)


def test_moe_transitive_lineage_and_flashinfer_notice() -> None:
    for path in ("moe_impl.py", "triton/fused_moe.py"):
        header = (PYTHON_ROOT / "ctikernels" / path).read_text().split("\n\n")[0]
        assert "SPDX-License-Identifier: Apache-2.0" in header
        assert "Adapted from Mini-SGLang" in header
        # One line names the lineage; holders and license texts live in the notice.
        assert "builds on SGLang/vLLM's fused MoE" in header
    notice_text = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    vllm = notice_text.split("\nvLLM\n", maxsplit=1)[1].split("\nNVIDIA NCCL\n")[0]
    assert f"{VLLM_REVISION} (LoRA)" in vllm
    assert f"{VLLM_MOE_REVISION} (MoE, 2024-11-15)" in vllm
    assert "vllm/model_executor/layers/fused_moe/fused_moe.py (MoE revision)" in vllm
    assert (
        "FlashInfer’s LICENSE also lists bundled third-party components; "
        "none of the upstream paths above\n"
        "is among them, and CTI does not redistribute them."
    ) in notice_text
    notice = notice_text.encode()
    flashinfer = notice.split(b"----- BEGIN UPSTREAM NOTICE (VERBATIM) -----\n")[1].split(
        b"----- END UPSTREAM NOTICE -----"
    )[0]
    # SHA256 of NOTICE downloaded at fa381e5162d9b6ab5479b77e5b6d287b732d108e.
    assert (
        hashlib.sha256(flashinfer).hexdigest()
        == "90bb9e1dec06f26a34f8f8ce98c53ded10b4fd8e1a9e91e2360e6be354e81db3"
    )
    triton = notice.split(b"----- BEGIN TRITON MIT LICENSE (VERBATIM) -----\n")[1].split(
        b"----- END TRITON MIT LICENSE (VERBATIM) -----"
    )[0]
    # SHA256 of LICENSE downloaded at f797708c0626e5f9840ca5b0a98790e2c7cb09ad.
    assert (
        hashlib.sha256(triton).hexdigest()
        == "92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946"
    )
    for path, copyright_year in (
        ("attention/_flashinfer.py", 2023),
        ("attention/_flashinfer_schedule.py", 2023),
        ("csrc/jit/request_local_sampling.cu", 2024),
    ):
        header = (PYTHON_ROOT / "ctikernels" / path).read_text().split("\n\n")[0]
        assert "fa381e5162d9b6ab5479b77e5b6d287b732d108e" in header
        assert f"SPDX-FileCopyrightText: Copyright (c) {copyright_year} FlashInfer team" in header


def test_notice_maps_exact_local_files_per_upstream() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    expected = {
        "Mini-SGLang": MINISGL_DERIVED_SOURCES,
        "FlashInfer (GDN chunk prefill)": set(FLASHINFER_GDN_SOURCES),
        "FlashInfer (W4A8 MoE)": set(W4A8_FLASHINFER_SOURCES),
        "FlashInfer": {
            "python/ctikernels/attention/_flashinfer.py",
            "python/ctikernels/attention/_flashinfer_schedule.py",
            "python/ctikernels/csrc/jit/request_local_sampling.cu",
        },
        "SGLang": {
            "python/ctikernels/linear/triton/_block_fp8.py",
            "python/ctikernels/moe_impl.py",
            "python/ctikernels/triton/fused_moe.py",
        },
        "Triton tutorials": {"python/ctikernels/triton/fused_moe.py"},
        "Liger Kernel": {"python/ctikernels/cross_entropy.py"},
        "Flash Linear Attention": {
            "python/ctikernels/activation/_silu_and_mul.py",
            *FLA_GDN_SOURCES,
        },
        "Efficient Cross Entropy (attribution carried through Liger)": {
            "python/ctikernels/cross_entropy.py"
        },
        "Unsloth": {"python/ctikernels/norm/_frozen_rms_norm.py"},
        "Unsloth Zoo (MoE training numerics)": {
            "python/ctikernels/moe/_training_backend.py",
            "python/ctikernels/moe/cute/_gemm.py",
            "python/ctikernels/moe/triton/_kernels.py",
        },
        "vLLM": {
            "python/ctikernels/attention/triton/_paged.py",
            "python/ctikernels/linear/_lora_inference.py",
            "python/ctikernels/moe_impl.py",
            "python/ctikernels/sampling/triton/_steer.py",
            "python/ctikernels/speculative/triton/_block_conv.py",
            "python/ctikernels/triton/fused_moe.py",
        },
        "NVIDIA NCCL": {"python/ctikernels/csrc/include/ctiengine/nccl227.h"},
        "FlashMLA": set(FLASHMLA_SOURCES),
        "Transformers": {"tests/gpu/test_bilinear_pos_embed.py"},
    }
    for upstream, paths in expected.items():
        section = notice.split("\n" + upstream + "\n", maxsplit=1)[1]
        local_files = section.split("Local files:\n", maxsplit=1)[1].split(
            "Upstream paths:", maxsplit=1
        )[0]
        assert {line.strip() for line in local_files.splitlines() if line.strip()} == paths
        assert all((PACKAGE_ROOT / path).is_file() for path in paths)
    assert "python/ctikernels/activation/_silu_and_mul.py" in notice


def test_ece_paths_and_notice_source_paths_are_unambiguous() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    ece = notice.split("\nEfficient Cross Entropy (attribution carried through Liger)\n")[1]
    ece = ece.split("\nUnsloth\n")[0]
    assert "Project: https://github.com/mgmalek/efficient_cross_entropy" in ece
    assert "Source revision: 049d44460051a82f58f7ff49a2ad0653ecf026d8" in ece
    assert "Upstream paths:\n  none (the code follows Liger Kernel" in ece
    assert "License source: LICENSE\n" in ece
    for section in re.split(r"\n[^\n]+\n={10,}\n", notice)[1:]:
        paths = section.split("Upstream paths:\n", maxsplit=1)[1].split("\n\n")[0]
        paths = [line.strip().split(" (", maxsplit=1)[0] for line in paths.splitlines()]
        assert len(paths) == len(set(paths))
        assert all(not path.startswith("https://") for path in paths)
        assert "/blob/" not in section


def test_flashinfer_gdn_wheel_provenance() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    section = notice.split("\nFlashInfer (GDN chunk prefill)\n", maxsplit=1)[1]
    section = section.split("\nMini-SGLang\n", maxsplit=1)[0]
    assert "Source: flashinfer-python 0.7.0.post1 wheel" in section
    for local, upstream in FLASHINFER_GDN_SOURCES.items():
        header = (PACKAGE_ROOT / local).read_text().split("\n\n", maxsplit=1)[0]
        assert "Adapted from flashinfer 0.7.0.post1" in header
        assert "Copyright (c) 2025-2026 FlashInfer team" in header
        assert "Copyright (c) 2026 cti-engine contributors" in header
        for name in upstream:
            path = f"flashinfer/gdn_kernels/delta_rule_dsl/{name}"
            assert path in header and path in section


def test_own_moe_block_ports_record_provenance_and_lazy_layout_constants() -> None:
    import ast

    for provider, digest in (("triton_block", "548eb81d"), ("triton_block_gb10", "98e1e538")):
        directory = PYTHON_ROOT / "ctikernels/moe" / provider
        for filename in ("__init__.py", "kernels.py"):
            source = (directory / filename).read_text()
            assert (
                f"# Port of cti's own Qwen3.5-MoE decode-block Triton tree ({digest}); "
                "CuTe rewrite pending."
            ) in source
        # The lazy host wrapper repeats only these layout constants, so it can
        # answer admission/config questions without importing Triton on the CPU.
        values = []
        for filename in ("__init__.py", "kernels.py"):
            constants = {}
            for node in ast.parse((directory / filename).read_text()).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ("CTL_WORDS", "G1D_STRIDE", "TL"):
                        value = (
                            node.value.args[0] if isinstance(node.value, ast.Call) else node.value
                        )
                        constants[name] = ast.literal_eval(value)
            values.append(constants)
        assert values[0] == values[1]


def test_block_fp8_cute_preserves_design_references() -> None:
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    kernel = (PYTHON_ROOT / "ctikernels/linear/cute/_block_fp8_gemm.py").read_text()
    provider = (PYTHON_ROOT / "ctikernels/linear/cute/_block_fp8_provider.py").read_text()
    for reference in (
        "FlashInfer 0.7.1",
        "blackwell_sm12x/kernel_moe_fp8_gemm.py",
        "blackwell_geforce/dense_gemm.py",
        "QuACK 0.6.5",
        "gemm_sm120.py",
    ):
        assert reference in kernel and reference in notice
    assert "SGLang 72730d6" in provider and "SGLang 72730d6" in notice
    assert "revision 950aa9e" in notice


def test_gated_cute_activation_keeps_inherited_attribution():
    source = (PYTHON_ROOT / "ctikernels/linear/cute/_block_fp8_act_quant.py").read_text()
    notice = (PACKAGE_ROOT / "THIRD_PARTY_NOTICES").read_text()
    for reference in (
        "2428d0f531602b882892f9f29521cd07ddd3c959",
        "f4b8d43a681a908ce22b63ece3a86ab5b198e760",
        "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        "vllm/model_executor/layers/fusion/fused_act_quant.py",
        "SGLang 72730d6",
    ):
        assert reference in source and reference in notice
