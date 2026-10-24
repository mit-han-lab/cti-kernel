# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Eager DRAFT attention access probe; R5 bookkeeping has its own sanitizer script."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from tests.gpu._dflash import Fixture

from .dflash_capture_trace import MASKS, diagnostic_stdout


def access_check():
    cases = []
    for q, causal, window in MASKS:
        f = Fixture(4, q, 1, 32, 8, causal, window, c=128)
        for slots in ([0, -1, 2, -1], [-1] * 4, [3, 2, 1, 0]):
            f.stage([1, 17, 64, 100], slots)
            f.run()
            torch.cuda.synchronize()
            f.check_metadata()
            torch.testing.assert_close(f.out.float(), f.oracle(), atol=2e-2, rtol=2e-2)
            cases.append({"q": q, "causal": causal, "window": window, "slots": slots})
    return {
        "type": "access_check",
        "cases": cases,
        "caching_disabled": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1",
        "ok": True,
    }


def find_sanitizer():
    found = shutil.which("compute-sanitizer")
    if found:
        return Path(found)
    roots = [Path("/usr/local/cuda")]
    if os.environ.get("CUDA_HOME"):
        roots.insert(0, Path(os.environ["CUDA_HOME"]))
    for root in roots:
        candidate = root / "bin" / "compute-sanitizer"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("compute-sanitizer is unavailable")


def launch_sanitizer():
    binary = find_sanitizer()
    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"[Vv]ersion\s+([\d.]+)", version.stdout + version.stderr)
    env = os.environ.copy()
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    command = [
        str(binary),
        "--tool",
        "memcheck",
        "--padding",
        "4096",
        "--print-limit",
        "50",
        "--error-exitcode",
        "1",
        sys.executable,
        "-m",
        "benchmark.kernel_tuning.dflash_access_check",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    # Forward only our structured receipt; package/sanitizer output can contain paths.
    child_ok = False
    for line in result.stdout.splitlines():
        try:
            receipt = json.loads(line)
        except ValueError:
            continue
        if isinstance(receipt, dict) and receipt.get("type") == "access_check":
            print(json.dumps(receipt), flush=True)
            child_ok = receipt.get("ok") is True and receipt.get("caching_disabled") is True
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors?", result.stdout + result.stderr)
    errors = max(map(int, summaries)) if summaries else None
    ok = result.returncode == 0 and errors == 0 and child_ok and match is not None
    print(
        json.dumps(
            {
                "type": "sanitizer",
                "sanitizer": binary.name,
                "version": match.group(1) if match else None,
                "error_summary": f"ERROR SUMMARY: {errors} errors" if errors is not None else None,
                "rc": result.returncode,
                "ok": ok,
            }
        ),
        flush=True,
    )
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-sanitizer", action="store_true")
    args = parser.parse_args()
    if args.launch_sanitizer:
        return launch_sanitizer()
    with diagnostic_stdout():
        result = access_check()
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
