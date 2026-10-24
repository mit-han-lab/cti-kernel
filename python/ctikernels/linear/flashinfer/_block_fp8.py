# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: FlashInfer's gemm_fp8_nt_groupwise is the called library.

"""FlashInfer CUTLASS provider for block-FP8 GEMM (no imported library until construction)."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

import torch

GROUP = 128
# The core dependency pins 0.7.0.post1; the GEMM is also qualified on the neighbouring releases
# (the same set as the attention providers).
FLASHINFER_VERSION = "0.7.0.post1"
QUALIFIED_VERSIONS = ("0.6.11.post3", "0.7.0.post1", "0.7.1")


def _flashinfer_gemm(name: str, capability: tuple[int, int], shape: Sequence[int]):
    if capability[0] not in (10, 12):
        raise RuntimeError(f"{name}: flashinfer-cutlass needs compute capability 10.x or 12.x")
    if shape[0] % GROUP or shape[1] % GROUP:
        raise ValueError(f"{name}: flashinfer-cutlass needs N and K multiples of 128")
    accepted = " or ==".join(QUALIFIED_VERSIONS)
    hint = f"pip install 'flashinfer-python=={FLASHINFER_VERSION}' (qualified: =={accepted})"
    try:
        version = importlib.metadata.version("flashinfer-python")
        from flashinfer.gemm import gemm_fp8_nt_groupwise
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(f"{name}: flashinfer-cutlass needs FlashInfer: {hint}") from error
    if version not in QUALIFIED_VERSIONS:
        raise RuntimeError(f"{name}: flashinfer-cutlass is qualified on {hint}, found {version}")
    return gemm_fp8_nt_groupwise


class FlashInferCutlassGemm:
    name = "flashinfer-cutlass"

    def __init__(self, weight, scale, *, capability, operation="BlockFP8Linear") -> None:
        # The pinned GEMM derives the weight's strides from its shape: a row-parallel shard view
        # (row stride > K) would be misread, so only packed weights are accepted (the stride-aware
        # Triton GEMMs take views).
        if not weight.is_contiguous():
            raise ValueError(
                f"{operation}: flashinfer-cutlass needs a packed (contiguous) weight, not strides "
                f"{weight.stride()}; strided shard views run on the Triton providers"
            )
        self._gemm = _flashinfer_gemm(operation, capability, weight.shape)
        self._capability = capability
        self._weight = weight
        # FlashInfer reads FP32 K-major scales: a lossless copy of the small grid.
        self._scale = scale.to(torch.float32).contiguous()

    @property
    def workspace_bytes(self):
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

        return DEFAULT_WORKSPACE_SIZE

    @staticmethod
    def _buffer(rows, columns, *, dtype, device):
        # Per-call staging from the caching allocator (the graph's pool under capture); the
        # instance keeps none. Engines share staging across ops through a GraphWorkspace.
        return torch.empty((rows, columns), dtype=dtype, device=device)

    def rows(self, m):
        """Rows the library computes for ``m``. On sm100 the pinned groupwise GEMM is wrong and
        not run-to-run deterministic for M % 4 != 0 and for every M in 18..32 (GB300, all DSV4
        shapes; M <= 17 padded to 4 rows and M >= 33 are exact): M is padded to a multiple of
        4, and 17..32 to 64. Rows past ``m`` are scratch."""
        if self._capability[0] != 10:
            return m
        return 64 if 16 < m <= 32 else m + (-m) % 4

    def row_class(self, m):
        """The library's M classes: none on sm12x (one 128x128x128 CUTLASS kernel, M a runtime
        argument, the fixed 40 MiB workspace: FlashInfer 0.7.0.post1 gemm_groupwise_sm120.cuh:74,
        gemm_base.py:147/8595); on sm100 its low-latency kernel up to 32 rows, the cooperative
        one above (gemm_groupwise_sm100.cu:120, one module), and whether ``rows`` pads."""
        if self._capability[0] != 10:
            return None
        rows = self.rows(m)
        return rows <= 32, rows != m

    def _staging(self, rows, *, dtype, device):
        """A per-call ``[rows (padded), N]`` output staging tensor."""
        return self._buffer(self.rows(rows), self._weight.shape[0], dtype=dtype, device=device)

    def _packed(self, kind, tensor, buffers=None):
        # Even a contiguous one-row view can have an unaligned base.
        if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
            return tensor
        target = (
            self._buffer(*tensor.shape, dtype=tensor.dtype, device=tensor.device)
            if buffers is None
            else buffers.get(kind, tensor.dtype)
        )
        target.copy_(tensor)
        return target

    def run(self, q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, *, buffers=None) -> None:
        m = out.shape[0]
        rows = self.rows(m)
        if rows != m:  # padded operands in staging; rows past m are scratch
            if buffers is None:
                target = self._staging(m, dtype=out.dtype, device=out.device)
                padded_q = self._buffer(rows, q.shape[1], dtype=q.dtype, device=q.device)
                padded_scale = self._buffer(
                    rows, scale.shape[1], dtype=scale.dtype, device=scale.device
                )
            else:
                target = buffers.get("staging", out.dtype)[:rows]
                padded_q, padded_scale = buffers.get("q")[:rows], buffers.get("scale")[:rows]
            for padded, operand in ((padded_q, q), (padded_scale, scale)):
                if padded.data_ptr() != operand.data_ptr():
                    padded[:m].copy_(operand[:m])
            q, scale = padded_q, padded_scale
        # The library's row-stride/unaligned-store support is not guaranteed, even
        # when a single-row column slice reports itself contiguous.
        elif out.is_contiguous() and out.data_ptr() % 16 == 0:
            target = out
        elif buffers is None:
            target = self._staging(m, dtype=out.dtype, device=out.device)
        else:
            target = buffers.get("staging", out.dtype)
        if buffers is not None:
            from flashinfer.gemm import gemm_base

            # The public wrapper owns a global cache buffer. Call its same CUTLASS
            # entry with engine storage; do not replace process-wide library state.
            module = (
                gemm_base.get_gemm_sm120_module()
                if self._capability[0] == 12
                else gemm_base.get_gemm_sm100_module()
            )
            args = () if self._capability[0] == 12 else (1,)
            module.gemm_fp8_nt_groupwise(
                buffers.get("library"),
                self._packed("q", q, buffers),
                self._weight,
                self._packed("scale", scale, buffers),
                self._scale,
                target,
                1,
                128,
                128,
                "K",
                *args,
            )
            if target is not out:
                out.copy_(target[:m])
            return
        self._gemm(
            self._packed("q", q),
            self._weight,
            self._packed("scale", scale),
            self._scale,
            scale_major_mode="K",
            out_dtype=out.dtype,
            out=target,
            backend="cutlass",
        )
        if target is not out:
            out.copy_(target[:m])


class FlashInferGroupedGemm:
    """FlashInfer's grouped groupwise CUTLASS GEMM (``group_gemm_fp8_nt_groupwise``) for
    ``groups`` equal-size segments: ``out[g] = q[g] . weight[g]^T`` on compute capability 10.x
    (the pinned library rejects more than one group on 12.x)."""

    name = "flashinfer-cutlass"

    def __init__(self, weight, scale, *, groups, capability, operation) -> None:
        if capability[0] != 10:
            raise RuntimeError(f"{operation}: flashinfer-cutlass needs compute capability 10.x")
        _flashinfer_gemm(operation, capability, weight.shape)
        from flashinfer.gemm import group_gemm_fp8_nt_groupwise

        self._gemm = group_gemm_fp8_nt_groupwise
        rows, k = weight.shape
        # The library reads [G, N, K] weights and FP32 K-major [G, N / 128, K / 128] scales.
        self._weight = weight.view(groups, rows // groups, k)
        self._scale = scale.to(torch.float32).view(groups, -1, scale.shape[-1]).contiguous()
        self._indptr: dict[tuple[int, torch.device], torch.Tensor] = {}

    def indptr(self, rows_per_group: int, device) -> torch.Tensor:
        key = (rows_per_group, device)
        table = self._indptr.get(key)
        if table is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GroupedBlockFP8Linear: warm_up this row count before graph capture"
                )
            groups = self._weight.shape[0]
            table = torch.arange(groups + 1, dtype=torch.int32, device=device) * rows_per_group
            self._indptr[key] = table
        return table

    def run(self, q, scale, out, rows_per_group: int) -> None:
        """``q`` ``[G * R, K]`` E4M3 and ``scale`` ``[G * R, K / 128]`` FP32 (R a multiple of
        4) into ``out`` ``[G * R, N]``."""
        self._gemm(
            q,
            self._weight,
            scale,
            self._scale,
            self.indptr(rows_per_group, q.device),
            scale_major_mode="K",
            out=out,
            out_dtype=out.dtype,
        )
