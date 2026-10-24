# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FlashMLA's sm100 sparse prefill for DeepSeek-V4 (``csrc/jit/flashmla``, MIT; built on first
use through tvm-ffi): 64 BF16 query heads of 512 channels over a BF16 ``[N, 512]`` row set,
one FP32 sink per head. Index widths are multiples of 64; -1 entries are skipped."""

from __future__ import annotations

HEADS = 64
ALIGN = 64


def load(capability: tuple[int, int], cutlass_includes: tuple[str, ...]):
    """Build (first use) or load the entry for an sm100-family ``capability``."""
    from ...utils import KERNEL_PATH, load_jit

    flashmla = KERNEL_PATH / "jit" / "flashmla"
    return load_jit(
        "dsv4_flashmla",
        f"{capability[0]}{capability[1]}a",
        cuda_files=["dsv4_flashmla.cu"],
        cuda_wrappers=[("prefill", "Dsv4FlashMla::prefill")],
        extra_include_paths=[
            str(flashmla),
            str(flashmla / "kerutils" / "include"),
            *cutlass_includes,
        ],
        extra_cuda_cflags=["-DNDEBUG", "--use_fast_math", "-Xptxas=--warn-on-spills"],
        cuda_arch=f"{capability[0]}.{capability[1]}a",
    )


def prefill(module, q, kv, indices, topk_length, sinks, scale, out, sms, *, stats):
    """``out [S, 64, 512]`` BF16 over BF16 rows ``kv [N, 512]``; ``stats [2, S, 64]`` FP32
    receives the max-logits and LSE."""
    module.prefill(q, kv, indices, topk_length, sinks, out, stats[0], stats[1], float(scale), sms)
