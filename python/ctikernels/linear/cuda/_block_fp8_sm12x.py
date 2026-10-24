# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The ``cuda-sm12x`` provider: the local CUDA block-FP8 GEMM for compute capability 12.x
(``csrc/jit/block_fp8_gemm_sm12x.cu``, built on first use through tvm-ffi). E4M3 tensor-core
MMAs with TMA-fed, warp-specialized persistent CTAs; BF16/FP16 activations are quantized
inside the GEMM with the standalone quantizer's recipe, bit for bit, so ``op(x)`` and
``op.quantized(*quantize_per_token_group_fp8(x))`` return identical tensors."""

from __future__ import annotations

import functools

import torch

GROUP = 128
NAME = "cuda-sm12x"


def _launch_config(n: int) -> tuple[int, int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, WARPS_M, WARPS_N, STAGES, GROUP_M), from GB10 sweeps: 64 x 256 tiles
    halve how often each activation tile is quantized; 128 x 128 when N is not a multiple of
    256."""
    return (64, 256, 2, 4, 2, 8) if n % 256 == 0 else (128, 128, 2, 4, 3, 8)


@functools.cache
def _module(config: tuple[int, ...], capability: tuple[int, int]):
    from ...utils import load_jit, make_cpp_args

    args = make_cpp_args(*config)
    arch = f"{capability[0]}{capability[1]}a"
    return load_jit(
        "block_fp8_gemm_sm12x",
        arch,
        *args,
        cuda_files=["block_fp8_gemm_sm12x.cu"],
        cuda_wrappers=[
            ("run", f"BlockFP8GemmSm12x<{args}>::run"),
            ("run_fused", f"BlockFP8GemmSm12x<{args}>::run_fused"),
        ],
        extra_cuda_cflags=[f"-gencode=arch=compute_{arch},code=sm_{arch}"],
    )


class CudaSm12xGemm:
    name = NAME

    def __init__(self, weight, scale, *, capability, sms, operation="BlockFP8Linear") -> None:
        if capability[0] != 12:
            raise RuntimeError(f"{operation}: {NAME} needs compute capability 12.x")
        n, k = weight.shape
        if n % GROUP or k % GROUP:
            raise ValueError(f"{operation}: {NAME} needs N and K multiples of 128")
        if weight.stride(1) != 1 or weight.stride(0) % 16 or weight.data_ptr() % 16:
            raise ValueError(f"{operation}: {NAME} needs 16-byte aligned weight rows")
        try:
            import tvm_ffi  # noqa: F401  (the JIT builder)
        except ImportError as error:
            raise RuntimeError(f"{operation}: {NAME} needs apache-tvm-ffi") from error
        self._capability = tuple(capability)
        self._config = _launch_config(n)
        self._sms = sms
        self._weight = weight.view(torch.uint8)  # strided shard views are read in place
        # The GEMM reads FP32 scales: a lossless copy of the small grid when it is BF16.
        self._scale = scale.to(torch.float32)

    def run(
        self, x: torch.Tensor, x_scale: torch.Tensor | None, out: torch.Tensor, *, buffers=None
    ) -> None:
        module = _module(self._config, self._capability)
        # The CUDA entry reads LDO; its paired 16-bit stores need 4-byte alignment.
        target = out
        if out.stride(0) % 2 or out.data_ptr() % 4:
            target = (
                torch.empty(out.shape, dtype=out.dtype, device=out.device)
                if buffers is None
                else buffers.get("staging", out.dtype)
            )
        # TMA and the quantizer warps read 16-byte aligned rows: copy other activations.
        row_bytes = x.stride(0) * x.element_size()
        if row_bytes % 16 or x.data_ptr() % 16:
            x = (
                x.clone(memory_format=torch.contiguous_format)
                if buffers is None
                else buffers.get("x" if x_scale is None else "q", x.dtype).copy_(x)
            )
        if x_scale is not None:
            if buffers is not None and not x_scale.is_contiguous():
                x_scale = buffers.get("scale").copy_(x_scale)
            module.run(x.view(torch.uint8), x_scale, self._weight, self._scale, target, self._sms)
        else:
            module.run_fused(x, self._weight, self._scale, target, self._sms)
        if target is not out:
            out.copy_(target)
