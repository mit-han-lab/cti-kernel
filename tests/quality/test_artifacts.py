# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import os
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns
from tarfile import open as open_tar
from zipfile import ZipFile

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h"}


def test_sdist_wheel_contains_standalone_sources_and_imports_without_engine(tmp_path: Path) -> None:
    """Build from only the kernel project, then install and import its actual wheel."""
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())["project"]
    expected_version = Version(project["version"])
    expected_license_files = {"LICENSE", "THIRD_PARTY_NOTICES"}
    expected_requirements = {
        requirement
        for dependency in project["dependencies"]
        if (requirement := Requirement(dependency)).marker is None
        or requirement.marker.evaluate({"extra": ""})
    }
    source = tmp_path / "source"
    copytree(
        PACKAGE_ROOT / "python/ctikernels",
        source / "python/ctikernels",
        ignore=ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES"):
        copy2(PACKAGE_ROOT / name, source / name)
    stale = source / "build/lib/ctikernels/stale_deleted_module.py"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        "raise AssertionError('stale build tree entered the wheel')\n", encoding="utf-8"
    )
    artifacts = tmp_path / "artifacts"
    completed = subprocess.run(
        (sys.executable, "-m", "build", "--no-isolation", "--outdir", str(artifacts)),
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Building wheel from sdist" in completed.stdout + completed.stderr
    wheel_path = next(artifacts.glob("*.whl"))
    sdist_path = next(artifacts.glob("*.tar.gz"))
    expected_sources = {
        path.relative_to(PACKAGE_ROOT / "python").as_posix(): path.read_bytes()
        for path in (PACKAGE_ROOT / "python/ctikernels").rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    }
    assert expected_sources
    with ZipFile(wheel_path) as wheel:
        members = wheel.namelist()
        assert "ctikernels/stale_deleted_module.py" not in members
        assert not any(name.startswith(("ctiengine/", "tests/")) for name in members)
        for name, content in expected_sources.items():
            assert wheel.read(name) == content, name
        metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        assert metadata["Name"] == "cti-kernels"
        assert Version(metadata["Version"]) == expected_version
        assert set(metadata.get_all("License-File", failobj=[])) == expected_license_files
        assert metadata["License-Expression"] == "MIT"
        requirements = {
            Requirement(dependency) for dependency in metadata.get_all("Requires-Dist", failobj=[])
        }
        assert not any(
            canonicalize_name(requirement.name) in {"cti-engine", "torch-memory-saver"}
            for requirement in requirements
        )
        runtime_requirements = {
            requirement
            for requirement in requirements
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        }
        assert runtime_requirements == expected_requirements
        for name in ("LICENSE", "THIRD_PARTY_NOTICES"):
            member = next(item for item in members if item.endswith(f".dist-info/licenses/{name}"))
            assert wheel.read(member) == (PACKAGE_ROOT / name).read_bytes()

    with open_tar(sdist_path, "r:gz") as sdist:
        members = sdist.getnames()
        root = members[0].split("/")[0]
        package_info = sdist.extractfile(f"{root}/PKG-INFO")
        assert package_info is not None
        with package_info:
            metadata = Parser().parsestr(package_info.read().decode())
        assert Version(metadata["Version"]) == expected_version
        assert set(metadata.get_all("License-File", failobj=[])) == expected_license_files
        assert metadata["License-Expression"] == project["license"]
        assert not any("/tests/" in name or "/python/ctiengine/" in name for name in members)
        for name, content in expected_sources.items():
            member = sdist.extractfile(f"{root}/python/{name}")
            assert member is not None
            with member:
                assert member.read() == content, name
        for name in ("LICENSE", "THIRD_PARTY_NOTICES"):
            member = sdist.extractfile(f"{root}/{name}")
            assert member is not None
            with member:
                assert member.read() == (PACKAGE_ROOT / name).read_bytes()

    installed = tmp_path / "installed"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(installed),
            str(wheel_path),
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    code = """
import importlib
import importlib.abc
import importlib.metadata
import sys
from pathlib import Path

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ctiengine" or fullname.startswith("ctiengine."):
            raise AssertionError("installed kernel wheel imported engine: " + fullname)
sys.meta_path.insert(0, NoEngine())
for name in (
    "ctikernels", "ctikernels.activation.inference",
    "ctikernels.attention.inference", "ctikernels.attention.varlen",
    "ctikernels.norm.inference", "ctikernels.gdn",
    "ctikernels.linear.lora", "ctikernels.moe.lora", "ctikernels.linear.block_fp8",
    "ctikernels.cross_entropy",
    "ctikernels.linear.flashinfer._block_fp8", "ctikernels.linear.triton._block_fp8_provider",
    "ctikernels.linear._block_fp8_dispatch", "ctikernels.linear.cuda._block_fp8_sm12x",
    "ctikernels.logprob",
    "ctikernels.sampling", "ctikernels.utils",
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), module.__file__
assert importlib.metadata.version("cti-kernels") == sys.argv[2]
assert not any(name == "ctiengine" or name.startswith("ctiengine.") for name in sys.modules)
utils = importlib.import_module("ctikernels.utils")
assert utils.KERNEL_PATH.is_relative_to(installed)
assert (utils.KERNEL_PATH / "include/ctiengine/nccl227.h").is_file()
assert (utils.KERNEL_PATH / "src/pynccl.cu").is_file()
"""
    subprocess.run(
        (sys.executable, "-I", "-c", code, str(installed), str(expected_version)),
        cwd=tmp_path,
        env=environment,
        check=True,
    )
