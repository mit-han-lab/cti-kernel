# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline VERIFY compilation; optional cubin resource checks without a driver.

Set CUDA_VISIBLE_DEVICES='', CUTE_DSL_ARCH=sm_120a or sm_121a.
To inspect resources, also set CUTE_DSL_KEEP=ptx,cubin and CUTE_DSL_DUMP_DIR;
set CUTE_DSL_NO_CACHE=1 and put cuobjdump on PATH. Each compile is inspected before artifacts are reused.
"""

import os
import re
import subprocess
from itertools import product
from pathlib import Path

import cutlass
import torch

from ctikernels.attention._draft.cute import compile_kernels
from ctikernels.attention._draft.cute._kernel import Attention


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    arch = os.environ["CUTE_DSL_ARCH"]
    assert arch in ("sm_120a", "sm_121a")
    dump = os.environ.get("CUTE_DSL_DUMP_DIR")
    if dump:
        assert os.environ.get("CUTE_DSL_NO_CACHE") == "1", "resource checks require fresh cubins"
        Path(dump).mkdir(parents=True, exist_ok=True)
    count = 0
    cases = [
        (rows, mode, causal, heads, 2, 8 if heads == 2 else 16)
        for rows, mode, causal, heads in product(
            (32, 64), ("legacy", "tree", "distributed"), (False, True), (2, 4)
        )
    ]
    # The default is two key groups; include a smaller row tile and the one-group override.
    cases += [
        (rows, mode, True, heads, groups, query)
        for mode in ("tree", "distributed")
        for rows, heads, groups, query in ((16, 4, 2, 4), (64, 2, 1, 8))
    ]
    for rows, mode, causal, heads, groups, query in cases:
        inline = mode != "legacy"
        constants = {
            "min_chunk": 64,
            "key_tile": 32,
            "key_groups": groups,
            "stages": 2,
            "chunk_cap": 4096 if arch == "sm_121a" else 1024,
            "pieces": True,
            "sparse": True,
            "short_union": 416,
            "inline_merge": inline,
            "distributed_merge": mode == "distributed",
            "head_dim": 256,
        }
        _, _, merge = compile_kernels(
            query, 4, rows, heads, causal, False, 23, torch.bfloat16, 16, arch, **constants
        )
        assert (merge is None) == inline
        kernel = Attention(query, 4, rows, heads, causal, False, 23, cutlass.BFloat16, **constants)
        report = {
            "arch": arch,
            "rows": rows,
            "groups": groups,
            "query": query,
            "heads": heads,
            "causal": causal,
            "mode": mode,
            "smem": kernel.smem_bytes + 1024,
        }
        if dump:
            for kind in ("Attention",) if inline else ("Attention", "Merge"):
                artifacts = list(Path(dump).glob(f"*{kind}_*.{arch}.cubin"))
                assert len(artifacts) >= 1, f"missing {kind} cubin"
                # Fake tensor geometry is part of the filename; select this compile's newest.
                artifact = max(artifacts, key=lambda path: path.stat().st_mtime_ns)
                result = subprocess.run(
                    ["cuobjdump", "--dump-resource-usage", str(artifact)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                usage = {
                    key: int(value)
                    for key, value in re.findall(r"(REG|STACK|LOCAL):([0-9]+)", result.stdout)
                }
                assert usage["LOCAL"] == 0, usage
                # The corrected operand loads expose a small spill frame in the
                # optional single-group override. Default two-group CTAs stay
                # spill-free; keep a tight bound on the override as well.
                stack_limit = 64 if kind == "Attention" and groups == 1 else 0
                assert usage["STACK"] <= stack_limit, usage
                assert usage["REG"] <= 255, usage
                if kind == "Attention":
                    # Register allocation is in 256-register units per warp.
                    per_warp = (usage["REG"] * 32 + 255) // 256 * 256
                    assert per_warp * (kernel.threads // 32) <= 65536
                report[kind] = usage
        print(report, flush=True)
        count += 1
    from ._dflash_draft_window_compile import extended_shapes

    count += extended_shapes(arch, dims=(256,))
    assert not torch.cuda.is_initialized()
    print(f"{count} offline hd256 configurations passed for {arch}", flush=True)


if __name__ == "__main__":
    main()
