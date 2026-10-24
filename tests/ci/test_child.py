# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""tests/support/child.run_child: output, bounds, and the whole process group on timeout."""

import os
import subprocess
import sys
import time

import pytest

from ..support import child
from ..support.child import run_child


def test_output_input_and_check():
    code = "import sys; print(sys.stdin.read().upper()); print('err', file=sys.stderr); sys.exit(3)"
    result = run_child([sys.executable, "-c", code], input="hi", timeout=60)
    assert (result.returncode, result.stdout, result.stderr) == (3, "HI\n", "err\n")
    with pytest.raises(subprocess.CalledProcessError):
        run_child([sys.executable, "-c", code], timeout=60, check=True)


def test_timeout_kills_the_group_even_when_a_grandchild_holds_stdout(tmp_path):
    marker = tmp_path / "grandchild"
    code = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
        f"open({str(marker)!r}, 'w').write(str(p.pid)); print('started', flush=True); "
        "time.sleep(600)"
    )
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_child([sys.executable, "-c", code], timeout=5)
    assert time.monotonic() - start < 5 + child.REAP_SECONDS
    assert "started" in error.value.output
    grandchild = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild outlived the timeout")


def test_an_unreapable_child_ends_the_run(monkeypatch):
    class Stuck:
        pid = os.getpid()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("child", timeout)

    monkeypatch.setattr(child.os, "killpg", lambda pid, signal: None)
    with pytest.raises(pytest.exit.Exception, match="survived SIGKILL.*state"):
        child._kill(Stuck())
