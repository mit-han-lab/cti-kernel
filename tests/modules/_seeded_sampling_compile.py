# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline CuTe 4.8.0 compilation; run with CUDA_VISIBLE_DEVICES='' and CUTE_DSL_ARCH.

No CUDA initialization or launches. Cover all input dtypes/filter specializations,
ordinary-domain draw and compact row scratch, including large and tail vocabularies.
"""

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass.runtime import make_fake_stream

from ctikernels.sampling import ORDINARY, RequestLocalSampling
from ctikernels.sampling._dflash.cute.kernels import Chain, Probabilities


def fake(dtype, shape):
    strides, size = [], 1
    for extent in reversed(shape):
        strides.insert(0, size)
        size *= extent
    return cute.runtime.make_fake_tensor(
        dtype, shape, tuple(strides), assumed_align=dtype.width // 8
    )


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("CUTE_DSL_ARCH")
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    f32, i32, i64, u32, u8 = (
        cutlass.Float32,
        cutlass.Int32,
        cutlass.Int64,
        cutlass.Uint32,
        cutlass.Uint8,
    )
    count = 0
    for v in (1, 4097, 151936, 248320):
        b = 3
        q, out = fake(f32, (b, 1, v)), fake(i32, (b, 1))
        args = (
            q,
            out,
            q,
            out,
            fake(f32, (b,)),
            fake(i32, (b,)),
            fake(u32, (b, 2)),
            fake(i64, (b,)),
            fake(u8, (b * ((v + 4095) // 4096) * 8,)),
            out,
            out,
        )
        cute.compile(
            Chain(1, v, "sample", domain=ORDINARY, scratch_positions=1),
            *args,
            stream,
            options="--enable-tvm-ffi",
        )
        count += 1
        min_p_args = list(args)
        min_p_args[1] = fake(f32, (b,))
        min_p_args[3] = fake(f32, (b * ((v + 4095) // 4096),))
        cute.compile(
            Chain(1, v, "sample", domain=ORDINARY, scratch_positions=1, min_p=True),
            *min_p_args,
            stream,
            options="--enable-tvm-ffi",
        )
        count += 1
        print("ordinary and min_p", v, "OK", flush=True)
    for dtype in (cutlass.BFloat16, cutlass.Float16, f32):
        for hint in ("none", "top_k", "top_p", "both", "auto"):
            b, v = 3, 4097
            op = RequestLocalSampling("cpu", v)
            layout = op.workspace_layout(b)
            size = next(r["offset"] for r in layout["regions"] if r["name"] == "probabilities")
            args = (
                fake(dtype, (b, 1, v)),
                fake(f32, (b,)),
                fake(i32, (b,)),
                fake(f32, (b,)),
                fake(i32, (b,)),
                fake(f32, (b, 1, v)),
                fake(u8, (size,)),
            )
            cute.compile(Probabilities(1, v, hint, b), *args, stream, options="--enable-tvm-ffi")
            count += 1
            cute.compile(
                Probabilities(1, v, hint, b, group=4, output_maxima=True),
                *args,
                stream,
                options="--enable-tvm-ffi",
            )
            count += 1
            print(dtype, hint, "with/without min_p OK", flush=True)
    assert not torch.cuda.is_initialized()
    print(count, "offline ordinary sampling configurations passed", flush=True)


if __name__ == "__main__":
    main()
