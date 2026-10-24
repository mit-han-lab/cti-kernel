# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Run IKET and print DRAFT summaries; keep raw and decoded traces on the runner.

Pass draft_iket.py workload arguments unchanged. No run-iket console script is
needed: the launcher imports iket.cli.main using the runner's Python paths.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    for root in (Path(__file__).resolve().parents[2], Path.cwd()):
        workload = root / "benchmark/kernel_tuning/draft_iket.py"
        if workload.is_file() and (root / "python/ctikernels").is_dir():
            break
    else:
        raise FileNotFoundError("DRAFT workload and shipped ctikernels source not found")
    paths = [str(root), str(root / "python"), *[p for p in sys.path if p]]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["CTI_DRAFT_IKET"] = "1"
    # Keep outputs under the shipped worktree; each invocation owns a fresh directory.
    output = Path(tempfile.mkdtemp(prefix="draft-iket-", dir=root)) / "trace"
    launcher = (
        f"import sys; sys.path[:0] = {paths!r}; "
        "from iket.cli.main import entrypoint; "
        "sys.argv = ['run-iket', *sys.argv[1:]]; sys.exit(entrypoint())"
    )
    command = [
        sys.executable,
        "-c",
        launcher,
        "--output-dir",
        str(output),
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
        *sys.argv[1:],
    ]
    rc = subprocess.call(command, env=env, cwd=root)
    print(f"DRAFT_IKET_RC {rc}", flush=True)
    traces = sorted(set(output.rglob("*.trace.json")) | set(output.rglob("*decoded*.json")))
    summary_rc = 0
    for trace in traces:
        print(f"DRAFT_IKET_SUMMARY {trace}", flush=True)
        result = subprocess.call(
            [sys.executable, str(workload), "--summarize", str(trace)], env=env, cwd=root
        )
        summary_rc = summary_rc or result
    if not traces:
        print(f"DRAFT_IKET_ERROR no decoded traces under {output}", flush=True)
        summary_rc = 1
    return rc or summary_rc


if __name__ == "__main__":
    sys.exit(main())
