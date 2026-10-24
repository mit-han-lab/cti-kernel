# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Compile the production symbolic ABI without CUDA initialization or launches.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONPATH=python. --vocab and --block-size
select fixed op parameters; --match restricts modes for a focused compile check.
The default covers every hint, dtype, min-p presence, group and public mode.
"""

import argparse
import os

import cutlass.cute as cute
import torch
from cutlass.runtime import make_fake_stream

from ctikernels.sampling import ORDINARY, RequestLocalSampling
from ctikernels.sampling._dflash import RequestLocalChainSampling, _warmup
from ctikernels.sampling._dflash.cute.provider import Provider, build_fake_args, compile_key


def variants(vocab, block_size, dtypes=_warmup.DTYPES):
    provider = Provider()
    chain = RequestLocalChainSampling("cpu", block_size, vocab)
    ordinary = RequestLocalSampling("cpu", vocab)
    descriptions = [
        *_warmup.probability_variants(chain, 1024, dtypes, _warmup.FILTERS, (False, True)),
        *_warmup.chain_variants(chain, _warmup.MODES),
        *_warmup.ordinary_variants(ordinary, 1024, dtypes, _warmup.FILTERS, True, 48),
    ]
    seen = set()
    for mode, inputs, hint in descriptions:
        prepared = provider._prepare(mode, inputs, hint, 48)
        if prepared is None:
            continue
        kernel, tensors, position, _ = prepared
        key = compile_key(mode, position, kernel, tensors, "cuda:0")
        if key not in seen:
            seen.add(key)
            yield key, kernel, tensors
    # Internal fallback draw and RNG ABIs also use the same symbolic builder.
    t = _warmup.tensor
    for use in (False, True):
        kernel = provider.kernels.Chain(
            1, vocab, "sample", domain=ORDINARY, scratch_positions=1, min_p=use
        )
        p, out = t(torch.float32, 1, 1, vocab), t(torch.int32, 1, 1)
        tensors = (
            p,
            t(torch.float32, 1) if use else out,
            p,
            t(torch.float32, 1) if use else out,
            t(torch.float32, 1),
            t(torch.int32, 1),
            t(torch.uint32, 1, 2),
            t(torch.int64, 1),
            t(torch.uint8, 8),
            out,
            out,
        )
        yield ("ordinary_min_p" if use else "ordinary",), kernel, tensors
    yield (
        ("rng",),
        provider.kernels.Rng(),
        (t(torch.uint32, 1, 4), t(torch.uint32, 1, 2), t(torch.int64, 1, 4)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=4097)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--match", default="")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    count = 0
    dtypes = _warmup.DTYPES
    if args.dtype:
        dtypes = (
            {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
        )
    for key, kernel, tensors in variants(args.vocab, args.block_size, dtypes):
        if args.match and key[0] not in args.match.split(","):
            continue
        cute.compile(
            kernel,
            *build_fake_args(kernel, tensors),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --gpu-arch=sm_120 --opt-level=3",
        )
        count += 1
        print(count, key[:3], "OK", flush=True)
    assert not torch.cuda.is_initialized()
    print(count, "symbolic sampling variants compiled", flush=True)


if __name__ == "__main__":
    main()
