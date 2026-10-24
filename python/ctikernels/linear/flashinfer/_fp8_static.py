# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: FlashInfer flashinfer/gemm/gemm_base.py, bmm_fp8 signature/layout.

"""Library per-tensor FP8 BMM; optional imports happen only at construction."""

import importlib.metadata

import torch

from ..torch._fp8_static import _equal_scales, aligned, available, direct_output
from ._block_fp8 import FLASHINFER_VERSION, QUALIFIED_VERSIONS


def _flashinfer_bmm(capability):
    if capability[0] not in (10, 12):
        raise RuntimeError("flashinfer-bmm needs compute capability 10.x or 12.x")
    hint = f"flashinfer-python=={FLASHINFER_VERSION}; qualified: {QUALIFIED_VERSIONS}"
    try:
        version = importlib.metadata.version("flashinfer-python")
        from flashinfer import bmm_fp8
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(f"flashinfer-bmm needs {hint}") from error
    if version not in QUALIFIED_VERSIONS:
        raise RuntimeError(f"flashinfer-bmm needs {hint}, found {version}")
    return bmm_fp8


def _backend_selector():
    from flashinfer.gemm import gemm_base
    from flashinfer.utils import backend_requirement

    checks = {
        name: getattr(gemm_base, f"_{name.replace('-', '_')}_bmm_fp8_requirement")
        for name in ("cudnn", "cublas", "cutlass", "cute-dsl")
        if hasattr(gemm_base, f"_{name.replace('-', '_')}_bmm_fp8_requirement")
    }

    @backend_requirement(
        checks,
        common_check=gemm_base._check_bmm_fp8_problem_size,
        heuristic_func=gemm_base._heuristic_func_bmm_fp8,
    )
    def select(A, B, A_scale, B_scale, dtype, out=None, backend="auto"):
        return select.suitable_auto_backends

    return select


class FlashInferBMM:
    def __init__(self, weight, weight_scale, input_scale, slot_rows, *, capability, backend):
        self._bmm = _flashinfer_bmm(capability)
        rows = (weight.shape[0],) if _equal_scales(weight_scale) else slot_rows
        if not available(weight.shape[1], rows):
            raise ValueError("flashinfer-bmm needs K and every GEMM's N divisible by 16")
        self._backend = backend
        self._capability = capability
        self._select_backends = None
        self._input_scale = input_scale.reshape(1)
        weight = aligned(weight)
        scales = weight_scale.reshape(-1)
        self._slots = []
        start = 0
        for i, count in enumerate(rows):
            stop = start + count
            self._slots.append(
                (slice(start, stop), weight[start:stop].t().unsqueeze(0), scales[i : i + 1])
            )
            start = stop

    @property
    def workspace_bytes(self):
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

        return DEFAULT_WORKSPACE_SIZE

    def add_workspace(self, layout, m):
        # cuBLASLt scratch needs 256-byte alignment, even though FP8 operands
        # and their leading dimensions only need 16-byte alignment.
        layout.add("library", (self.workspace_bytes,), torch.uint8, alignment=256)
        if self._capability[0] != 12 or self._backend == "cublas":
            return
        k = self._slots[0][1].shape[1]
        n = max(w.shape[2] for _, w, _ in self._slots)
        kp, np = (k + 127) // 128 * 128, (n + 127) // 128 * 128
        layout.add("library_scale_a", (m, kp // 128))
        layout.add("library_scale_b", (np // 128, kp // 128))
        if kp != k:
            layout.add("library_q", (1, m, kp), torch.float8_e4m3fn)
        if kp != k or any(w.shape[2] % 128 for _, w, _ in self._slots):
            layout.add("library_weight", (np, kp), torch.float8_e4m3fn)
        if any(w.shape[2] % 128 for _, w, _ in self._slots):
            layout.add("library_out", (m, np), torch.bfloat16)

    def _cutlass_workspace(self, q, weight, scale, out, buffers):
        from flashinfer.gemm import gemm_base

        m, k, n = q.shape[1], q.shape[2], weight.shape[2]
        kp, np = (k + 127) // 128 * 128, (n + 127) // 128 * 128
        if kp != k:
            padded = buffers.get("library_q").zero_()
            padded[:, :, :k].copy_(q)
            q = padded
        if kp != k or np != n:
            padded = buffers.get("library_weight").view(-1)[: np * kp].view(1, np, kp).zero_()
            padded[:, :n, :k].copy_(weight.transpose(1, 2))
            weight = padded.transpose(1, 2)
        target = out
        if np != n:
            target = buffers.get("library_out", out.dtype).view(-1)[: m * np].view(1, m, np)
        sa = buffers.get("library_scale_a").copy_(self._input_scale)
        sb = (
            buffers.get("library_scale_b")
            .view(-1)[: np // 128 * (kp // 128)]
            .view(np // 128, kp // 128)
        )
        sb.copy_(scale)
        gemm_base.get_gemm_sm120_module().gemm_fp8_nt_groupwise(
            buffers.get("library"), q, weight, sa, sb, target, 1, 128, 128, "MN"
        )
        if target is not out:
            out.copy_(target[:, :, :n])

    def _run_workspace(self, q, weight, scale, out, buffers, backends):
        from flashinfer.gemm import gemm_base

        factories = {
            "cutlass_sm10x": lambda: (
                gemm_base.get_gemm_sm100_module_cutlass_fp8().cutlass_fp8_gemm_runner()
            ),
            "cutlass_sm12x": lambda: (
                gemm_base.get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner()
            ),
            "cublas": lambda: gemm_base.get_gemm_module().cublas_fp8_gemm_runner(),
            "cudnn": lambda: gemm_base._cudnn_gemm_fp8_runner(),
        }
        # Preserve FlashInfer's runner objects, tuning key and tactic choice.
        # Only its SM12x runner's tensor preparation needs caller-owned views.
        names = [name for name in factories if name in backends]
        runners = [factories[name]() for name in names]
        inputs = [q, weight, self._input_scale, scale, out, buffers.get("library")]
        runner, tactic = gemm_base.AutoTuner.get().choose_one(
            "fp8_gemm", runners, gemm_base._FP8_GEMM_SM100_TUNING_CONFIG, inputs
        )
        if names[runners.index(runner)] == "cutlass_sm12x":
            self._cutlass_workspace(q, weight, scale, out, buffers)
        else:
            runner(inputs=inputs, tactic=tactic)

    def run(self, q, out, *, buffers=None):
        if buffers is None:
            q = aligned(q)
        elif not direct_output(q) or q.stride(0) % 16:
            q = buffers.get("q").copy_(q)
        q = q.unsqueeze(0)
        for columns, weight, scale in self._slots:
            direct = len(self._slots) == 1 and direct_output(out)
            target = (
                out
                if direct
                else buffers.get("staging", out.dtype)
                .view(-1)[: out.shape[0] * weight.shape[2]]
                .view(out.shape[0], weight.shape[2])
                if buffers is not None
                else torch.empty(
                    (out.shape[0], weight.shape[2]), device=out.device, dtype=out.dtype
                )
            )
            if buffers is not None:
                from flashinfer.gemm import gemm_base

                backend = self._backend
                if backend == "auto":
                    if self._select_backends is None:
                        self._select_backends = _backend_selector()
                    backends = self._select_backends(
                        q,
                        weight,
                        self._input_scale,
                        scale,
                        out.dtype,
                        out=target.unsqueeze(0),
                        backend=backend,
                    )
                elif backend == "cutlass":
                    backends = gemm_base._heuristic_func_bmm_fp8(
                        [backend],
                        q,
                        weight,
                        self._input_scale,
                        scale,
                        out.dtype,
                        target.unsqueeze(0),
                        backend,
                    )
                else:
                    backends = [backend]
                self._run_workspace(q, weight, scale, target.unsqueeze(0), buffers, backends)
            else:
                self._bmm(
                    q,
                    weight,
                    self._input_scale,
                    scale,
                    out.dtype,
                    out=target.unsqueeze(0),
                    backend=self._backend,
                )
            if not direct:
                out[:, columns].copy_(target)
