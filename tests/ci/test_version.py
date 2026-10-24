# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""A release tag must name the version in pyproject.toml."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import tomllib
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]


def _release_tags() -> list[str]:
    """``v*`` tags naming this commit: the pushed ref in CI, else ``git tag --points-at``."""
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return [os.environ["GITHUB_REF_NAME"]]
    try:
        out = subprocess.run(
            ("git", "tag", "--points-at", "HEAD"),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []  # an sdist or a copy without .git: nothing is tagged
    return [tag for tag in out.split() if tag.startswith("v")]


def test_release_tag_equals_pyproject_version() -> None:
    with open(REPO / "pyproject.toml", "rb") as file:
        version = Version(tomllib.load(file)["project"]["version"])
    wrong = [tag for tag in _release_tags() if Version(tag[1:]) != version]
    assert not wrong, f"tags {wrong} do not name pyproject.toml version {version}"
