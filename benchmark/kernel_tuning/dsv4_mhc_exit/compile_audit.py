# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline sm_121a audit. CUDA driver entry points are deliberately blocked."""

import argparse
import os
import re
import subprocess
from pathlib import Path


def audit_sass(
    path, ptx, *, cluster, early_publish=False, single_fence=False, split_proj=False, fp64=False
):
    """Save disassembly/resources and reject spills in the new exchange paths."""
    sass = subprocess.check_output(["nvdisasm", str(path)], text=True)
    resources = subprocess.check_output(["cuobjdump", "-res-usage", str(path)], text=True)
    path.with_suffix(".sass").write_text(sass)
    path.with_suffix(".resources.txt").write_text(resources)
    assert not re.search(r"\b(?:LDL|STL)(?:\.|\s)", sass), f"local memory instructions: {path.name}"
    assert "STACK:0" in resources and "LOCAL:0" in resources, path.name
    if single_fence:
        assert "fence.acq_rel.gpu" not in ptx, path.name
        # ptxas lowers a single-thread acq_rel RMW to MEMBAR + ATOMG.
        # Its release must follow the CTA barrier in the thread-0 branch;
        # an all-thread fence before that barrier is a regression.
        arrival = re.search(r"\bATOMG?\.", sass)
        assert arrival is not None, path.name
        barrier = sass.rfind("BAR.SYNC", 0, arrival.start())
        assert barrier >= 0 and "MEMBAR" not in sass[:barrier], path.name
    if split_proj:
        assert "ld.volatile.shared" not in ptx, path.name
        assert "LDS.128" in sass, path.name
    print("RESOURCES", path.stem, " ".join(resources.split()), flush=True)
    if early_publish:
        # In FP32 lane16 decode the first two exponentials are pre/post
        # sigmoid; the third is the Sinkhorn softmax in the last-CTA branch.
        ptx_exp = list(re.finditer(r"ex2\.approx\.ftz\.f32", ptx))
        sass_exp = list(re.finditer(r"MUFU\.EX2", sass))
        # ptxas may duplicate a softmax exponential in an out-of-line continuation.
        assert len(ptx_exp) == 3 and len(sass_exp) >= 3, path.name
        release = sass.index("STG.E.64.STRONG.GPU")
        assert ptx_exp[1].start() < ptx.index("st.release.gpu.global.b64") < ptx_exp[2].start()
        assert sass_exp[1].start() < release < sass_exp[2].start()
        # The last CTA also writes its BF16 output before entering Sinkhorn.
        assert "STG.E.128" in sass[release : sass_exp[2].start()]
        assert "SHFL.BFLY" in sass[release : sass_exp[2].start()]
        assert "MUFU.RCP" in sass[sass_exp[2].end() :]
    if cluster:
        assert ptx.index("cp.async.bulk.prefetch.L2.global") < ptx.index("griddepcontrol.wait")
        assert sass.index("UBLKPF.L2") < sass.index("ACQBULK")
        assert "mapa.shared::cluster" in ptx and "ld.shared::cluster" in ptx
        assert ptx.count("barrier.cluster.arrive;") == 2
        assert ptx.count("barrier.cluster.wait;") == 2
        assert "UCGABAR_ARV" in sass and "UCGABAR_WAIT" in sass
        assert "LD.E " in sass
    # Accurate div/sqrt retain guarded exceptional slow paths. Do not mistake
    # their static CALL sites for unconditionally executed hot-path calls.
    calls = [line for line in sass.splitlines() if re.search(r"\bCALL\.", line)]
    assert all(
        "_slowpath" in line
        or (fp64 and re.search(r"__cuda_sm20_(?:div_rn_f64_full|dsqrt_rn_f64_mediumpath_v1)", line))
        for line in calls
    ), path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", nargs="+", type=int, default=[1024, 4096, 4104])
    parser.add_argument(
        "--sass", action="store_true", help="audit new variants with nvdisasm/cuobjdump"
    )
    parser.add_argument("--fastdiv", type=int, choices=(0, 1), default=0)
    parser.add_argument("--exchange", choices=("last", "all"), default="last")
    parser.add_argument("--publish", choices=("early", "late"), default="early")
    parser.add_argument("--fence", choices=("single", "all"), default="single")
    parser.add_argument("--proj", choices=("split", "token"), default="split")
    parser.add_argument("--chain", choices=("lane16", "serial"), default="lane16")
    parser.add_argument(
        "--caps", nargs="+", type=int, default=[4, 16, 32, 64], help="exchange-only capacities"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("exit", "pre"),
        default=["exit", "pre"],
        help="exchange-only modes",
    )
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--controls-only", action="store_true")
    subset.add_argument("--slices-only", action="store_true")
    subset.add_argument("--new-only", action="store_true")
    subset.add_argument("--exchange-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.out)
    os.environ["CTI_MHC_FASTDIV"] = str(args.fastdiv)
    os.environ["CTI_MHC_EXCHANGE"] = args.exchange
    os.environ["CTI_MHC_PUBLISH"] = args.publish
    os.environ["CTI_MHC_FENCE"] = args.fence
    os.environ["CTI_MHC_PROJ"] = args.proj
    import cuda.bindings.driver as driver

    def forbidden(*args, **kwargs):
        raise RuntimeError("offline audit forbids CUDA device access")

    for name in (
        "cuInit",
        "cuDeviceGetCount",
        "cuDeviceGet",
        "cuCtxGetCurrent",
        "cuCtxCreate",
        "cuModuleLoadData",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
    ):
        setattr(driver, name, forbidden)

    from iket import Context

    from ctikernels.mhc._cute.kernels import compile_kernel
    from ctikernels.mhc._reference import MhcConfig
    from ctikernels.mhc.inference import _cluster_plan, _policy

    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for h in (
        []
        if args.controls_only or args.slices_only or args.new_only or args.exchange_only
        else args.hidden
    ):
        for trace in (False, True):
            for mode in ("exit", "pre", "post"):
                buckets = (
                    [(False, 32, 16)]
                    if mode == "post"
                    else (
                        [(True, cap, 16) for cap in (4, 16, 32)]
                        + [(False, 32, tile) for tile in (1, 2, 4, 8, 16, 32)]
                    )
                )
                for decode, cap, tile in buckets:
                    cases.append((h, trace, mode, decode, cap, tile, "lane16", True, None))
    # Exercise both compile-time comparisons on decode and prefill, including
    # FP64 serial and IKET. The main matrix covers every shape/capacity/tile.
    for trace in () if args.slices_only or args.new_only or args.exchange_only else (False, True):
        for mode in ("exit", "pre", "post"):
            for decode in (True, False) if mode != "post" else (False,):
                for chain, early in (("serial", True), ("lane16", False), ("serial", False)):
                    cases.append((4096, trace, mode, decode, 4, 1, chain, early, None))
    # Explicit S=128 overrides for the small capacities also change layouts.
    for h in [] if args.controls_only or args.new_only or args.exchange_only else args.hidden:
        for trace in (False, True):
            for mode in ("exit", "pre"):
                for cap in (4, 16):
                    cases.append((h, trace, mode, True, cap, 16, "lane16", True, 128))
    if not args.controls_only and not args.slices_only and not args.exchange_only:
        for h in args.hidden:
            for trace in (False, True):
                for cap in (1001, 1002, 1004, 1008, 64, 128, 256):
                    cases.append((h, trace, "exit", True, cap, 16, "lane16", True, None))
                for cap in (64, 128, 256):
                    if _policy(h, 48, 99 * 1024, cap)[0] != 128:
                        cases.append((h, trace, "exit", True, cap, 16, "lane16", True, 128))
        for trace in (False, True):
            for chain, early in (("serial", True), ("lane16", False), ("serial", False)):
                for cap in (1008, 256):
                    cases.append((4096, trace, "exit", True, cap, 16, chain, early, None))
    if args.exchange_only:
        for h in args.hidden:
            for trace in (False, True):
                for mode in args.modes:
                    for cap in args.caps:
                        cases.append((h, trace, mode, True, cap, 16, args.chain, True, None))
    for h, trace, mode, decode, cap, tile, chain, early, size in cases:
        name = (
            f"h{h}-{mode}-d{int(decode)}-c{cap}-tt{tile}-{chain}-early{int(early)}-iket{int(trace)}"
        )
        name += f"-exchange-{args.exchange}"
        name += f"-publish-{args.publish}-fence-{args.fence}-proj-{args.proj}"
        if size is not None:
            name += f"-s{size}"
        if args.fastdiv:
            name += "-fastdiv1"
        os.environ["CTI_MHC_IKET"] = str(int(trace))
        os.environ["CTI_MHC_EARLY_TRIGGER"] = str(int(early))
        os.environ["CTI_MHC_CHAIN"] = chain
        print("START", name, flush=True)
        compiled = compile_kernel(
            MhcConfig(hidden=h),
            _cluster_plan(h, cap - 1000)
            if cap > 1000
            else _policy(h, 48, 99 * 1024, cap, slice_size=size),
            mode,
            decode,
            token_tile=tile,
            cluster=cap > 1000,
            options="--gpu-arch sm_121a --keep-ptx --keep-cubin" + (" --iket" if trace else ""),
        )
        (args.out / (name + ".ptx")).write_text(compiled.__ptx__)
        (args.out / (name + ".cubin")).write_bytes(compiled.__cubin__)
        context = Context(compiled.__cubin__)
        assert context.is_instrumented() == (trace and mode != "post")
        if args.sass and (cap > 32 or args.exchange_only):
            audit_sass(
                args.out / (name + ".cubin"),
                compiled.__ptx__,
                cluster=cap > 1000,
                fp64=mode == "pre",
                single_fence=decode and cap < 1000 and args.fence == "single",
                split_proj=decode and cap <= 32 and args.proj == "split",
                early_publish=(
                    args.exchange == "last"
                    and args.publish == "early"
                    and decode
                    and cap <= 4
                    and mode == "exit"
                    and chain == "lane16"
                ),
            )
        print("PASS", name, flush=True)


if __name__ == "__main__":
    main()
