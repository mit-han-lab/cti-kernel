# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline compile and public PTX checks; does not initialize CUDA.

Run in a fresh process with CuTe 4.8 installed:
  python -m tests.modules._scale_cast_compile --arch sm_120 --output out/scale-cast
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUTE_DSL_ARCH=args.arch,
        CUTE_DSL_KEEP="ptx,cubin",
        CUTE_DSL_DUMP_DIR=str(args.output),
        CUTE_DSL_DISABLE_FILE_CACHING="1",
    )
    import torch

    from ctikernels.attention._scale_cast import _compile

    compiled = _compile(torch.float16, torch.bfloat16, args.arch)
    assert _compile(torch.float16, torch.bfloat16, args.arch) is compiled
    assert _compile.cache_info().misses == 1
    assert not torch.cuda.is_initialized()
    ptx = "\n".join(path.read_text() for path in args.output.glob("*.ptx"))
    assert ptx, "compiler did not emit PTX"
    assert "ld.global.v4.b32" in ptx and "st.global.v4.b32" in ptx
    assert "mul.f32" in ptx and "cvt.rn.bf16" in ptx
    if int(args.arch.removeprefix("sm_").rstrip("a")) >= 90:
        assert ptx.index("griddepcontrol.wait") < ptx.index("ld.global")
        assert "griddepcontrol.launch_dependents" in ptx
    else:
        assert "griddepcontrol" not in ptx
    print(f"{args.arch}: symbolic sizes, vector IO, FP32 scale, BF16 RN, PDL checks passed")


if __name__ == "__main__":
    main()
