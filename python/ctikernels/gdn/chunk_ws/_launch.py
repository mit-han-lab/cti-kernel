# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""Symbolic launch adapter for the SM12x chunk kernel.

Device metadata contract: positive sequence lengths partition [0, T]; each slot is
-1 or in [0, S), and nonnegative slots are unique within a call. Metadata values
are deliberately not read on the host. Each call allocates its own per-SM
descriptor workspace on the current stream.
"""

import math
import threading

import torch

_compiling = threading.Lock()


def _validate(
    device,
    hk,
    hv,
    q,
    k,
    v,
    g,
    beta,
    o,
    pool,
    slots,
    initial,
    cu_seqlens,
    scale,
):
    """Only metadata checks; no CUDA calls or device-value inspection."""
    specs = (
        ("q", q, torch.bfloat16, 3),
        ("k", k, torch.bfloat16, 3),
        ("v", v, torch.bfloat16, 3),
        ("g", g, torch.float32, 2),
        ("beta", beta, torch.float32, 2),
        ("o", o, torch.bfloat16, 3),
        ("pool", pool, torch.float32, 4),
        ("slots", slots, torch.int32, 1),
        ("initial", initial, torch.bool, 1),
        ("cu_seqlens", cu_seqlens, torch.int32, 1),
    )
    for name, tensor, dtype, ndim in specs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        # The TMA descriptor requires a 16-byte base alignment. Scalar metadata
        # and gates need only element alignment; sliced contiguous inputs are legal.
        if name in ("q", "k", "v", "o", "pool") and tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    t, n = q.shape[0], slots.shape[0]
    shapes = (
        ("q", q, (t, hk, 128)),
        ("k", k, (t, hk, 128)),
        ("v", v, (t, hv, 128)),
        ("o", o, (t, hv, 128)),
        ("g", g, (t, hv)),
        ("beta", beta, (t, hv)),
        ("pool", pool, (pool.shape[0], hv, 128, 128)),
        ("initial", initial, (n,)),
        ("cu_seqlens", cu_seqlens, (n + 1,)),
    )
    for name, tensor, shape in shapes:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if n < 1 or t < n or pool.shape[0] < 1:
        raise ValueError("require T >= N > 0 and S > 0")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite host number")


def compile_kernel(hk, hv, options):
    """Compile with symbolic T, N, S and workspace size; never query a device."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int8, Int32, Uint8
    from cutlass.runtime import make_fake_stream

    from ._kernel import ChunkKernelSm120

    fake = cute.runtime.make_fake_tensor

    def tokens(heads, dtype=BFloat16):
        return fake(dtype, (cute.sym_int(), heads, 128), (heads * 128, 128, 1), assumed_align=16)

    def gate():
        return fake(Float32, (cute.sym_int(), hv), (hv, 1), assumed_align=4)

    def vector(dtype, alignment):
        return fake(dtype, (cute.sym_int(),), (1,), assumed_align=alignment)

    pool = fake(
        Float32,
        (cute.sym_int(), hv, 128, 128),
        (hv * 128 * 128, 128 * 128, 128, 1),
        assumed_align=16,
    )
    return cute.compile(
        ChunkKernelSm120(hk, hv),
        tokens(hk),
        tokens(hk),
        tokens(hv),
        gate(),
        gate(),
        tokens(hv),
        pool,
        vector(Int32, 4),
        vector(Int8, 1),
        vector(Int32, 4),
        Float32(1.0),
        vector(Uint8, 128),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=options,
    )


class ChunkPrefillSm120:
    """BF16 D=128 GVA prefill, with FP32 slot-indexed state updated in place."""

    def __init__(self, device, num_k_heads, num_v_heads):
        if (
            not isinstance(num_k_heads, int)
            or not isinstance(num_v_heads, int)
            or num_k_heads <= 0
            or num_v_heads < num_k_heads
            or num_v_heads % num_k_heads
        ):
            raise ValueError("GVA requires positive Hk and Hv a multiple of Hk")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("ChunkPrefillSm120 requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.hk, self.hv = num_k_heads, num_v_heads
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("construct ChunkPrefillSm120 before CUDA graph capture")
            cc = torch.cuda.get_device_capability(device)
            if cc not in ((12, 0), (12, 1)):
                raise ValueError("ChunkPrefillSm120 requires sm120 or sm121")
            self.options = f"--enable-tvm-ffi --gpu-arch sm_{cc[0]}{cc[1]}a"
            import cutlass.cute as cute
            from cutlass import Int32
            from cutlass.runtime import make_fake_stream

            from ._kernel import read_nsmid

            result = torch.empty(1, dtype=torch.int32, device=device)
            with _compiling:
                reader = cute.compile(
                    read_nsmid,
                    cute.runtime.make_fake_tensor(Int32, (1,), (1,), assumed_align=4),
                    make_fake_stream(use_tvm_ffi_env_stream=True),
                    options=self.options,
                )
            reader(result)
            # The only device read/synchronization: construction, never __call__.
            self.nsmid = result.item()
        self.compiled = None

    def __call__(self, q, k, v, g, beta, o, pool, slots, initial, cu_seqlens, scale):
        _validate(
            self.device,
            self.hk,
            self.hv,
            q,
            k,
            v,
            g,
            beta,
            o,
            pool,
            slots,
            initial,
            cu_seqlens,
            scale,
        )
        with torch.cuda.device(self.device):
            if self.compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "ChunkPrefillSm120 must run eagerly before CUDA graph capture"
                    )
                with _compiling:
                    if self.compiled is None:
                        self.compiled = compile_kernel(self.hk, self.hv, self.options)
            self.compiled(
                q,
                k,
                v,
                g,
                beta,
                o,
                pool,
                slots,
                initial.view(torch.int8),
                cu_seqlens,
                float(scale),
                torch.empty(self.nsmid * 128, dtype=torch.uint8, device=self.device),
            )
