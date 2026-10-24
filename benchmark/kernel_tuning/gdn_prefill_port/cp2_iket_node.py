#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU-side IKET driver: run-iket over cp2_iket.py, then summarize, in one call.

Run on the GPU machine as cp2_iket_node.py [cp2_iket.py args]. The trace stays
local; only the summary (and the decoder's file list) is printed.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    workload = HERE / "cp2_iket.py"
    if not workload.is_file():
        workload = Path.cwd() / "benchmark/kernel_tuning/gdn_prefill_port/cp2_iket.py"
    args = sys.argv[1:]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = Path(tempfile.mkdtemp(prefix="cp2-iket-")) / "trace"
    launcher = (
        "import sys; sys.path[:0] = %r; from iket.cli.main import entrypoint; "
        "sys.argv = ['run-iket'] + sys.argv[1:]; sys.exit(entrypoint())"
    ) % [p for p in sys.path if p]
    command = [
        sys.executable,
        "-c",
        launcher,
        "--output-dir",
        str(out),
        "profile",
        "--postprocess",
        "all",
        "--keep",
        "--enabled-cluster",
        "0,0,0",
        "--max-ts-cnt-per-warp",
        "8192",
        "--",
        sys.executable,
        str(workload),
        *args,
    ]
    rc = subprocess.call(command, env=env)
    print(f"CP2_IKET_RC {rc}", flush=True)
    traces = sorted(out.rglob("*.trace.json")) + sorted(out.rglob("*decoded*.json"))
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"CP2_IKET_FILE {path.relative_to(out)} {path.stat().st_size}", flush=True)
    for trace in traces:
        print(f"CP2_IKET_SUMMARY {trace.name}", flush=True)
        subprocess.call([sys.executable, str(workload), *args, "--summarize", str(trace)], env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
