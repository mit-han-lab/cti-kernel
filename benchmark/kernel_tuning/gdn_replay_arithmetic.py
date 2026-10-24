# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline product/CuTe arithmetic audit. Does not discover or launch a GPU.

Write compiler artifacts outside the repository; PTX debug records can contain
compiler-local paths. Inspect SASS only with NVIDIA's disassembler.
"""

import argparse
import json
from pathlib import Path


def save_cute(fn, out, name):
    for ext in ("ptx", "cubin"):
        data = getattr(fn, f"__{ext}__")
        # DSL versions expose either bytes/text or an artifact path.
        if isinstance(data, str) and not data.lstrip().startswith(("//", ".version")):
            data = Path(data.strip()).read_bytes()
        path = out / f"{name}.{ext}"
        if isinstance(data, bytes):
            path.write_bytes(data)
        elif isinstance(data, str):
            path.write_text(data)
        else:
            raise RuntimeError("set CUTE_DSL_KEEP=ptx,cubin before compiling")


def product(out, arch, sms, hv, batch):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    from ctikernels.gdn._contract import Geometry
    from ctikernels.gdn._triton_decode import decode_plan
    from ctikernels.gdn._triton_decode_kernels import gdn_decode_kernel

    stride = (32 + 2 * hv) * 128 + 2 * hv
    plan = decode_plan(Geometry(16, hv), batch, stride, True, sms=sms, pdl=arch >= 90)
    arguments = dict(plan.arguments)
    options = {
        k: arguments.pop(k) for k in ("num_warps", "launch_pdl", "maxnreg") if k in arguments
    }
    arguments["ROWIDX"] = False
    signature = dict(
        zip(
            gdn_decode_kernel.arg_names[:12],
            (
                "*bf16",
                "*i64",
                "*fp32",
                "*fp32",
                "*fp32",
                "*i32",
                "*fp32",
                "*bf16",
                "*bf16",
                "*i64",
                "*i64",
                "i32",
            ),
            strict=True,
        )
    )
    fn = triton.compile(
        GluonASTSource(
            gdn_decode_kernel,
            signature,
            constexprs=arguments,
            attrs={(i,): [["tt.divisibility", 16]] for i in range(11)},
        ),
        target=GPUTarget("cuda", arch, 32),
        options=options,
    )
    name = f"product-{arch}-{sms}-{hv}-{batch}"
    for ext in ("ptx", "llir", "ttgir"):
        (out / f"{name}.{ext}").write_text(fn.asm[ext])
    (out / f"{name}.cubin").write_bytes(fn.asm["cubin"])
    return {
        "name": name,
        "grid": plan.grid,
        "options": options,
        "arguments": {k: str(v) for k, v in arguments.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--cute", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for arch, sms in ((120, 170), (120, 188), (121, 48)):
        for hv in (32, 48):
            for batch in args.batches:
                row = product(args.out, arch, sms, hv, batch)
                if args.cute:
                    from ctikernels.gdn.replay.cute import Config, compile_step

                    fn = compile_step(
                        16,
                        hv,
                        8,
                        batch,
                        (32 + 2 * hv) * 128 + 2 * hv,
                        Config(),
                        str(arch),
                        batch,
                        True,
                        True,
                        sms,
                    )
                    save_cute(fn, args.out, f"cute-{arch}-{sms}-{hv}-{batch}")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
