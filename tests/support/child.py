# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Child Python processes for tests: bounded, reaped, and never blocked on pipes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile

import pytest

REAP_SECONDS = 30
# Its own process group, for one kill of the child and its descendants. Not its own session: GPU
# runners treat a GPU process outside the call's session as another user's and stop the call.
_OWN_GROUP = {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}


def run_child(args, *, timeout, env=None, cwd=None, input=None, check=False):
    """``subprocess.run(args, capture_output=True, text=True, ...)`` for a child that may use the GPU.

    The child runs in its own process group with its output in files, so no descendant can hold a
    pipe open. On timeout its whole process group is killed and reaped for at most REAP_SECONDS
    seconds; a child that still cannot be reaped (stuck in the driver) ends the test session with
    its state instead of hanging it."""
    with (
        tempfile.TemporaryFile("w+") as stdin,
        tempfile.TemporaryFile("w+") as out,
        tempfile.TemporaryFile("w+") as err,
    ):
        if input is not None:
            stdin.write(input)
            stdin.seek(0)
        process = subprocess.Popen(
            args, env=env, cwd=cwd, stdin=stdin, stdout=out, stderr=err, text=True, **_OWN_GROUP
        )  # fmt: skip
        try:
            returncode = process.wait(timeout)
        except subprocess.TimeoutExpired:
            _kill(process)
            raise subprocess.TimeoutExpired(args, timeout, _read(out), _read(err)) from None
        stdout, stderr = _read(out), _read(err)
    if check and returncode:
        sys.stderr.write(stderr[-4000:])
        raise subprocess.CalledProcessError(returncode, args, stdout, stderr)
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _read(file):
    file.seek(0)
    return file.read()


def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(REAP_SECONDS)
    except subprocess.TimeoutExpired:
        pytest.exit(
            f"child {process.pid} survived SIGKILL for {REAP_SECONDS} s ({_state(process.pid)}):"
            " a GPU/driver hang; stopping the run",
            returncode=3,
        )


def _state(pid):
    try:
        with open(f"/proc/{pid}/status") as status:
            state = next(
                line.split(":", 1)[1].strip() for line in status if line.startswith("State")
            )
        with open(f"/proc/{pid}/wchan") as wchan:
            return f"state {state}, wchan {wchan.read() or '-'}"
    except (OSError, StopIteration):
        return "state unknown"
