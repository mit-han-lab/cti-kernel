# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shape-only token backward dispatch, with stream-independent compiled kernels.

CTI_MOE_CUTE_TOKEN=0 on the backend retains Triton's token launch. Warm each
specialization before capture. No tensor values are read by this dispatcher.
"""

import threading

import torch

_LOCK = threading.RLock()
_COMPILED = {}


def token_splits(tokens, h):
    # Four token warps per CTA. Small batches split H into 256-column strips;
    # large batches reuse each warp's routing derivatives across the entire H.
    # A single token necessarily has limited parallelism (H/256 CTAs at most).
    return min((h + 255) // 256, max(1, (512 + tokens - 1) // tokens))


def compile_token(h, e, k, nt, shared, masked, wide_sel, splits, capability):
    """Compile symbolic token/slot counts offline; metadata can be offset views."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass.runtime import make_fake_stream

    from ._token_kernel import TokenBackward

    if h <= 0 or h % 64 or not 1 <= k <= min(e, 8) or e > 256 or nt <= 0:
        raise ValueError("CuTe token backward requires H % 64 == 0, 1 <= k <= 8, E <= 256")
    if capability not in ((12, 0), (12, 1)) or not 1 <= splits <= (h + 255) // 256:
        raise ValueError("Unsupported CuTe token target or hidden-dimension split")
    t, rows = cute.sym_int(), cute.sym_int()
    kp1 = k + shared
    fake = cute.runtime.make_fake_tensor
    args = (
        fake(BFloat16, (rows, h), (h, 1), assumed_align=16),
        fake(Int32, (t * kp1,), (1,), assumed_align=4),
        fake(Float32, (rows, nt), (nt, 1), assumed_align=4),
        fake(Float32, (t * kp1,), (1,), assumed_align=4),
        fake(Int64 if wide_sel else Int32, (t, k), (k, 1), assumed_align=8 if wide_sel else 4),
        fake(BFloat16, (e, h), (h, 1), assumed_align=16),
        fake(BFloat16, (1 if shared else e, h), (h, 1), assumed_align=16),
        fake(BFloat16, (t, h), (h, 1), assumed_align=16),
        fake(Int32, (1,), (1,), assumed_align=4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(
        TokenBackward(h, k, nt, shared, masked, splits),
        *args,
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )


def token_backward(dxp, inv, dwpart, wslot, sel, r, sg, dx, valid, *, capability):
    t, h = dx.shape
    if t == 0:
        return
    key = (
        h,
        r.shape[0],
        sel.shape[1],
        dwpart.shape[1],
        sg is not None,
        valid is not None,
        sel.dtype == torch.int64,
        token_splits(t, h),
        capability,
    )
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up this MoE CuTe token shape before CUDA graph capture")
        with _LOCK:
            fn = _COMPILED.get(key)
            if fn is None:
                fn = compile_token(*key)
                _COMPILED[key] = fn
    fn(
        dxp,
        inv,
        dwpart,
        wslot,
        sel,
        r,
        r if sg is None else sg,
        dx,
        inv[:1] if valid is None else valid.reshape(1),
    )
